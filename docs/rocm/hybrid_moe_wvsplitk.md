# Hybrid MoE: Fused wvSplitK Decode + Triton Prefill (ROCm)

## Overview

For MoE models during decode, per-expert token count is typically 1-4. The Triton `fused_moe` kernel uses `BLOCK_SIZE_M=16` minimum, wasting 93% of compute on padding when M=1. This implementation adds a fused HIP `wvSplitK` kernel path for small M decode on ROCm, falling back to Triton for larger M (prefill/batched decode).

## Key Idea: Transpose Trick

The existing `wvSplitK` kernel handles skinny GEMMs where N (input rows) is 1-4. For MoE decode:
- Per-expert token count (1-4 during decode) → becomes wvSplitK's **N dimension**
- Expert weight rows → becomes wvSplitK's **M dimension**

```
wvSplitK(in_a=weight(M_weight, K), in_b=input(N_tokens, K), cu_count)
  → output(N_tokens, M_weight)

GEMM1: wvSplitK(w1[expert], tokens, cu_count)  → (n_tok, 2*N_inter)
GEMM2: wvSplitK(w2[expert], act_out, cu_count) → (n_tok, K)
```

## Kernel Fusion: From 32 Launches to 3

### Before (Python per-expert loop)
For M=1, topk=8, ~8 active experts:
- 8× `wvSplitK` for GEMM1
- 8× activation
- 8× `wvSplitK` for GEMM2
- 8× weight multiply + scatter-add
- **= 32 kernel launches**, dominated by host→GPU dispatch overhead

### After (fused blockIdx.y dispatch)
- 1× `moe_wvSplitK` for GEMM1 — `grid(CuCount, P)` covers all slots
- 1× activation (contiguous P×N buffer)
- 1× `moe_wvSplitK` for GEMM2
- 1× vectorized weight+reduce
- **= 3-4 kernel launches total**

The key insight from the int4 path (ROCm/vllm commit `0b992ff`): instead of looping over experts on the host, push expert selection into the kernel grid. Each GPU block independently looks up which expert it belongs to via `expert_ids[blockIdx.y]` and offsets the weight pointer. No host coordination needed.

## Expert Dispatch via blockIdx.y

```cpp
// Grid: dim3(CuCount, num_expert_blocks)
const int expert_block = blockIdx.y;
const int expert_id = expert_ids[expert_block];
if (expert_id < 0) return;  // skip padding

// Offset weight pointer to this expert
const scalar_t* B = B_base + expert_id * expert_stride_b;

// Gather activation row for this slot
const int slot_start = expert_block * block_size_m;
int src_row = sorted_token_ids ? sorted_token_ids[slot] / top_k : slot;
// Load activation into LDS...
```

- `blockIdx.y` = slot index (0 to P-1, where P = M×topk)
- `expert_ids[blockIdx.y]` = which expert's weights to use
- `blockIdx.x` = which chunk of weight rows (M dimension) this CU handles

## HIP Block Scope vs Triton Block Scope

### Triton fused_moe
```
Preprocessing: sort all P slots by expert → sorted_token_ids, expert_offsets
Grid: (ceildiv(tokens_per_expert, BLOCK_SIZE_M) × E, ceildiv(2N, BLOCK_SIZE_N))

One block computes:
  BLOCK_SIZE_M tokens × BLOCK_SIZE_N weight cols, full K reduction
  e.g. 16 tokens × 64 weight cols × 2048 K

Problem at M=1: BLOCK_SIZE_M=16 minimum, 1 token padded to 16 = 93% wasted.
```

### HIP wvSplitK
```
No sorting needed. expert_ids maps each slot to its expert directly.
Grid: (CuCount, P)        -- e.g. (48, 8) for M=1 topk=8

One block computes:
  1 token × YTILE weight rows, full K reduction via LDS
  e.g. 1 token × 2 weight rows × 2048 K

All weight rows covered by CuCount × WvPrGrp × YTILE strided across M_weight.
```

| Aspect | Triton | HIP wvSplitK |
|--------|--------|--------------|
| Block scope | BLOCK_M tokens × BLOCK_N cols | 1 token × YTILE rows |
| Expert dispatch | Presorted, grid tiles over groups | `expert_ids[blockIdx.y]` direct |
| Min tokens | BLOCK_SIZE_M=16 (padded) | 1 (no padding) |
| K reduction | Tiled loops within block | Full K in one pass via LDS |
| CU utilization M=1 | Few blocks, mostly padding | grid.y=8 × grid.x=48 = 384 blocks |

### YTILE

YTILE is the loop unrolling factor over the weight row (output) dimension. Each warp group computes YTILE consecutive weight rows simultaneously, reusing the activation data already in LDS:

```cpp
float sum[N][YTILE] = {};
for (int y = 0; y < YTILE; y++)
    bigB[y][k2].h8 = loadnt(&B[(y + m) * Kbp]);  // load YTILE weight rows
```

Selected automatically based on rows-per-CU: YTILE=1 (few rows) to YTILE=4 (many rows).

## Dispatch Logic

In `fused_experts_impl()`, M≤5 on ROCm routes to HIP, otherwise Triton:

```python
if (
    current_platform.is_rocm()
    and M <= 5
    and not use_fp8_w8a8 and not use_int8_w8a8
    and not use_int8_w8a16 and not use_int4_w4a16
    and expert_map is None
    and hidden_states.dtype in (torch.float16, torch.bfloat16)
    and hidden_states.shape[1] % 8 == 0
    and w2.shape[2] % 8 == 0
):
    return _hip_skinny_moe_gemm(...)
```

## Why Further Fusion Is Hard

Fusing GEMM1 + activation + GEMM2 into a single kernel would eliminate 2 launch boundaries, but requires a **global sync** between stages:

- GEMM1 distributes weight rows across CUs via `grid.x`. One slot's full output is spread across many CUs.
- Activation (silu_and_mul) needs the **complete** GEMM1 output for each slot.
- No efficient global barrier exists across all CUs — only `__syncthreads()` within a block.

Kernel launch boundaries serve as implicit global syncs. This same constraint applies to Triton, which also separates GEMM1 → activation → GEMM2 stages.

## Files Modified

| File | Change |
|------|--------|
| `csrc/rocm/skinny_gemms.cu` | `moe_wvSplitK_hf_sml_` GPU kernel + `moe_wvSplitK` host function + launch macros |
| `csrc/rocm/ops.h` | Declaration for `moe_wvSplitK` |
| `csrc/rocm/torch_bindings.cpp` | Torch op registration |
| `vllm/_custom_ops.py` | Python wrapper `moe_wvSplitK()` |
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | `_hip_skinny_moe_gemm()` + dispatch in `fused_experts_impl()` |

## Benchmark Results (W7900, gfx1100)

### Kernel-level: HIP wvSplitK vs Triton (Qwen3-30B shapes E=128, topk=8)

| M | P (slots) | HIP | Triton | Speedup |
|---|-----------|-----|--------|---------|
| 1 | 8 | 0.067ms | 0.154ms | **2.29x** |
| 2 | 16 | 0.084ms | 0.155ms | **1.84x** |
| 3 | 24 | 0.108ms | 0.155ms | **1.43x** |
| 4 | 32 | 0.132ms | 0.156ms | **1.18x** |
| 6 | 48 | 0.185ms | 0.174ms | 0.94x |
| 8 | 64 | 0.239ms | 0.232ms | 0.97x |
| 16 | 128 | 0.431ms | 0.344ms | 0.80x |
| 64 | 512 | 1.670ms | 0.495ms | 0.30x |

Crossover at M≈5-6. HIP scales linearly with P (each slot is its own grid block). Triton stays flat because BLOCK_SIZE_M tiling absorbs extra tokens cheaply.

### End-to-end: Qwen3-30B-A3B (TP=4, ShareGPT)

| Config | req/s | total tok/s |
|--------|-------|-------------|
| Default Triton | 10.30 | 4,319 |
| Tuned Triton | ~11.7 | ~4,900 |
| Hybrid + tuned (1000 prompts) | 9.78 | 4,099 |
| Hybrid (max_num_seqs=1, 30 prompts) | 0.13 | 55.35 |
| Triton (max_num_seqs=1, 30 prompts) | 0.13 | 55.17 |

No measurable e2e difference because:
1. **Batched serving**: M stays above 5, hybrid path rarely triggers
2. **TP=4 allreduce**: 40% of GPU time masks kernel gains
3. **Single-request**: Allreduce latency dominates at low batch

### Practical Impact

The kernel-level gain is real (2.3x at M=1) but unlocking it e2e requires:
- **TP=1** (no allreduce overhead) with a small MoE model that fits single GPU
- **Low-concurrency serving** where M naturally stays ≤5

## References

- ROCm/vllm commit `0b992ff` — hybrid W4A16 MoE with HIP decode + Triton prefill (int4 path, same blockIdx.y pattern)
- vllm-project/vllm PR #40977 — hybrid W4A16 linear kernel for dense layers
