# MoE Int4 Triton Kernel: Interleave Branch vs ROCm/gfx11 Branch

Comparison of two approaches to optimizing the MoE int4 Triton kernel on ROCm
using `tl.interleave` for weight unpacking.

- **Interleave branch** (`moe-triton-int4-interleave`): standalone patch against upstream `main`
- **gfx11 branch** (`ROCm/vllm:gfx11`): Matthias Gehre's commit `0b992ff237`

Both target the same bottleneck: the `fused_moe_kernel_gptq_awq` Triton kernel
loads each packed int4 byte twice and uses per-element variable shifts for
unpacking, wasting memory bandwidth and ALU.

## Weight Layout After Repacking

| | Interleave branch | gfx11 branch |
|---|---|---|
| Weights | `[E, K, N//8]` int32 | `[E, N, K//8]` int32 |
| Packing dim | **N-packed** (8 int4 along N) | **K-packed** (8 int4 along K) |
| Packing order | GPTQ sequential: shifts `[0, 4, 8, ..., 28]` | ExLlama shuffle: nibble order `[0,2,4,6,1,3,5,7]` |
| Scales | `[E, K//G, N]` (transposed) | `[E, N, K//G]` (original) |
| Zero points | `[E, K//G, N//8]` int32 (repacked) | N/A (symmetric only) |

## Repacking at Load Time (`moe_wna16.py`)

### Interleave branch

Added `process_weights_after_loading()` to `MoeWNA16Method`. On ROCm + int4:

1. Unpack `[E, N, K//2]` uint8 → `[E, N, K]` uint8 (split lo/hi nibbles)
2. Transpose to `[E, K, N]`
3. Repack groups of 8 along N into int32 with sequential shifts → `[E, K, N//8]` int32
4. Transpose scales `[E, N, K//G]` → `[E, K//G, N]`
5. Repack zero points (if `has_zp`) to `[E, K//G, N//8]` int32

### gfx11 branch

Added `_process_weights_hybrid_w4a16()` to `CompressedTensorsWNA16MoEMethod`
(gated by `VLLM_MOE_HYBRID_W4A16` env var). On ROCm + int4:

1. Unpack `[E, K//8, N]` int32 (GPTQ format) → `[E, K, N]` uint8
2. Transpose to `[E, N, K]`
3. Repack with `pack_int4_exllama_shuffle()` → `[E, N, K//8]` int32

Scales stay as `[E, N, K//G]` (already in the right orientation for the kernel).

### Key difference

The interleave branch patches **`MoeWNA16Method`** (AWQ/GPTQ native models),
while gfx11 patches **`CompressedTensorsWNA16MoEMethod`** only. AWQ models
(e.g. Qwen3-30B-A3B-AWQ) use `MoeWNA16Method`, so gfx11's shuffle path
never activates for them without an additional patch.

## Triton Kernel

### Interleave branch — new kernel `fused_moe_kernel_gptq_awq_interleave`

Separate kernel function, dispatched from within `invoke_fused_moe_wna16_triton_kernel`
when `current_platform.is_rocm() and use_int4_w4a16 and B.dtype == torch.int32`.

**Weight loading:**
```python
# B: [E, K, N//8] int32 — load [BLOCK_K, BLOCK_N//8]
offs_bn_packed = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
b_ptrs = b_ptr + off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn_packed[None, :] * stride_bn

b_packed = tl.load(b_ptrs)                    # [BLOCK_K, BLOCK_N//8] int32
b = tl.interleave(b_packed, b_packed)         # 3x interleave
b = tl.interleave(b, b)
b = tl.interleave(b, b)                       # → [BLOCK_K, BLOCK_N]
b = (b >> shifts) & 0xF                       # shifts = [0,4,8,...,28] tiled
```

Unpacked tile is `[BLOCK_K, BLOCK_N]` — ready for `tl.dot`, no transpose needed.

**Scale loading:**
```python
# Scales: [E, K//G, N] — load [BLOCK_K, BLOCK_N] with per-element group index
g_idx = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + g_idx * stride_bsk + offs_sn[None, :] * stride_bsn
b_scale = tl.load(b_scale_ptrs)               # [BLOCK_K, BLOCK_N]
```

**Zero-point loading (when `has_zp=True`):**
```python
# Zeros: [E, K//G, N//8] int32 — same interleave pattern
b_zp_packed = tl.load(b_zp_ptrs)
b_zp = tl.interleave(b_zp_packed, b_zp_packed)  # 3x interleave
b_zp = tl.interleave(b_zp, b_zp)
b_zp = tl.interleave(b_zp, b_zp)
b_zp = (b_zp >> shifts) & 0xF
```

### gfx11 branch — constexpr flag on existing kernel

Adds `use_shuffle_w4a16: tl.constexpr = False` to the existing
`fused_moe_kernel_gptq_awq`. Dispatched via a separate function
`invoke_fused_moe_kernel_hybrid_triton`.

**Weight loading:**
```python
# B: [E, N, K//8] int32 — load [BLOCK_N, BLOCK_K//8]
offs_k8 = tl.arange(0, BLOCK_SIZE_K // 8)
b_packed_ptrs = b_ptr + off_experts * stride_be + offs_bn[:, None] * stride_bn + offs_k8[None, :] * stride_bk

b_packed = tl.load(b_packed_ptrs)              # [BLOCK_N, BLOCK_K//8] int32
b_exp = tl.interleave(b_packed, b_packed)      # 3x interleave
b_exp = tl.interleave(b_exp, b_exp)
b_exp = tl.interleave(b_exp, b_exp)            # → [BLOCK_N, BLOCK_K]
b_nk = (b_exp >> exl_shifts) & 0xF            # ExLlama unshuffle shifts
b = tl.trans(b_nk)                             # → [BLOCK_K, BLOCK_N]
```

Requires `tl.trans()` because the packed dimension is K, so unpacking produces
`[BLOCK_N, BLOCK_K]` which must be transposed for `tl.dot(a, b)`.

**Scale loading:**
```python
# Scales: [E, N, K//G] — load 1D [BLOCK_N], broadcast
g_idx = (k * BLOCK_SIZE_K) // group_size       # scalar (assumes BLOCK_K <= group_size)
b_scale = tl.load(b_scale_ptr + offs_bn * stride_bsn + g_idx * stride_bsk)  # [BLOCK_N]
b = ((b.to(tl.float32) - 8) * b_scale[None, :]).to(compute_type)            # broadcast
```

**Zero-point loading:** Not supported. Hardcodes `b_zp_num = 8` (symmetric only).

## Dispatch

### Interleave branch

Dispatch stays inside `invoke_fused_moe_wna16_triton_kernel`. Detection:
```python
use_interleave = current_platform.is_rocm() and use_int4_w4a16 and B.dtype == torch.int32
```
Calls `fused_moe_kernel_gptq_awq_interleave` (separate kernel), passing
`has_zp=B_zp is not None` to support both symmetric and asymmetric quantization.

### gfx11 branch

Dispatch is a separate intercept in `dispatch_fused_moe_kernel`:
```python
if use_int4_w4a16 and B.dtype == torch.int32:
    invoke_fused_moe_kernel_hybrid_triton(...)
    return
```
Calls the same `fused_moe_kernel_gptq_awq` kernel with `use_shuffle_w4a16=True`
and `has_zp=False` (hardcoded symmetric).

## Feature Comparison

| Feature | Interleave branch | gfx11 branch |
|---|---|---|
| Quantization method support | `MoeWNA16Method` (AWQ/GPTQ) | `CompressedTensorsWNA16MoEMethod` only |
| Asymmetric quantization (zero points) | Yes | No (symmetric only, `bias=8`) |
| Kernel approach | Separate kernel function | Constexpr flag on existing kernel |
| Transpose in inner loop | None | `tl.trans()` per tile |
| Scale loading | 2D `[BLOCK_K, BLOCK_N]` per-element | 1D `[BLOCK_N]` broadcast (requires `BLOCK_K <= group_size`) |
| Env var gating | None (auto-detect via dtype) | `VLLM_MOE_HYBRID_W4A16` |
| HIP wvSplitK decode path | No | Yes (via `HybridW4A16MoEExperts`) |
| Modular kernel integration | No | Yes (`FusedMoEKernel` + `HybridW4A16MoEExperts`) |

## Performance

Both achieve similar kernel-level speedup (~1.55-1.6x) on Qwen3-30B-A3B-AWQ
shapes (E=128, N=384, K=2048) on gfx1100 (AMD Radeon Pro W7900).

### ATT (Advanced Thread Trace) per-instruction analysis

Collected with `rocprofv3 --att` on gfx1100, decoded with `att-tool`.
Shape: Qwen3-30B-A3B-AWQ MoE layer (E=128, N=1536, K=2048, group_size=128, top_k=8).
Latency is summed per opcode category, normalized by max hitcount to give per-wave cycles.

```
                      ExLlama M=128   Interleave M=128   ExLlama M=4096   Interleave M=4096
                      Lat/wave    %    Lat/wave    %      Lat/wave    %    Lat/wave    %
  ─────────────────────────────────────────────────────────────────────────────────────────
  WAITCNT             47,204  67.4%    10,596  81.7%       1,343  48.9%     1,002  28.2%
  BARRIER             22,506  32.1%     1,142   8.8%         365  13.3%       634  17.9%
  BPERMUTE               100   0.1%         0   0.0%         542  19.7%         0   0.0%
  LDS                     28   0.0%       990   7.6%         160   5.8%     1,690  47.6%
  VALU                   105   0.1%       187   1.4%         306  11.1%       197   5.5%
  WMMA                     2   0.0%         2   0.0%           3   0.1%         2   0.1%
  ─────────────────────────────────────────────────────────────────────────────────────────
  TOTAL               70,059   100%    12,964   100%       2,746   100%     3,548   100%

  Per-wave ratio:     ExLlama/Interleave = 5.4x              ExLlama/Interleave = 0.77x
  Wall-clock:         2.195ms vs 1.627ms (IL wins)            6.836ms vs 7.811ms (EX wins)
```

**Key observations:**
- WMMA (compute) is <0.1% at all M — both kernels are memory/sync-bound.
- At M=128 (small prefill/decode), ExLlama spends 32% in `s_barrier` (from `tl.trans()` LDS transpose requiring workgroup sync). Interleave avoids barriers for unpacking entirely.
- At M=4096 (large prefill), Interleave spends 48% in LDS ops (`tl.interleave` compiles to 286 DS read/write ops). ExLlama's `ds_bpermute` (cross-lane shuffle) rises to 20%.
- `s_waitcnt` = single-wave stall waiting on its own VMEM loads. `s_barrier` = workgroup sync (__syncthreads equivalent).

**ATT collection commands:**
```bash
SKIP_OCCUPANCY=1 rocprofv3 --att \
  python3 /tmp/att_interleave_runner.py 128   # or att_exllama_runner.py

# Decode binary ATT trace:
att-tool -i <att_dir>/<trace>.att -o output.csv --csv
```

### End-to-end benchmark: Interleave vs ExLlama

Model: Qwen3-30B-A3B-AWQ, dataset: ShareGPT (500 prompts), gfx1100 (W7900).

```bash
# Server
vllm serve /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --max-model-len 4096 --gpu-memory-utilization 0.95 --port 8000

# Benchmark
vllm bench serve \
  --model /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --dataset-name sharegpt \
  --dataset-path /mnt/nas_share/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500
```

| Metric | Interleave (GPTQ) | ExLlama Shuffle | Winner |
|---|---|---|---|
| Duration (s) | 94.88 | 102.44 | Interleave (+8%) |
| Request throughput (req/s) | 5.27 | 4.88 | Interleave (+8%) |
| Output token throughput (tok/s) | 1158.13 | 1071.57 | Interleave (+8%) |
| Total token throughput (tok/s) | 2248.28 | 2081.29 | Interleave (+8%) |
| Mean TTFT (ms) | 22726 | 15875 | ExLlama (-30%) |
| Median TTFT (ms) | 18972 | 11722 | ExLlama (-38%) |
| Mean TPOT (ms) | 183.97 | 188.03 | Interleave (-2%) |
| Median TPOT (ms) | 149.36 | 158.54 | Interleave (-6%) |
| Mean ITL (ms) | 129.24 | 141.49 | Interleave (-9%) |
| Median ITL (ms) | 110.95 | 126.62 | Interleave (-12%) |

Interleave wins overall throughput (+8%) and decode latency (TPOT/ITL).
ExLlama wins TTFT (prefill) by 30-38%, consistent with the ATT M=4096 result
where ExLlama has lower per-wave latency. The decode advantage of Interleave
accumulates over longer generations, yielding higher total throughput.
