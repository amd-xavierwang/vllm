# MoE Int4 Triton Kernel: Interleave Branch vs ROCm/gfx11 Branch

Comparison of two approaches to optimizing the MoE int4 Triton kernel on ROCm
using `tl.interleave` for weight unpacking.

- **Interleave branch** (`moe-triton-int4-interleave`): standalone patch against upstream `main`
- **gfx11 branch** (`ROCm/vllm:gfx11`): Matthias Gehre's commit `0b992ff237`

Both target the same bottleneck: the `fused_moe_kernel_gptq_awq` Triton kernel
uses per-element variable shifts for int4 unpacking, causing high register
pressure (256 VGPRs with scratch spilling on FP16/gfx1100) and ALU divergence.

## Weight Layout After Repacking

| | Interleave branch | gfx11 branch |
| --- | --- | --- |
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
| --- | --- | --- |
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

```text
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
| --- | --- | --- | --- |
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

### End-to-end benchmark: Interleave vs Main (compressed-tensors)

Model: RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (CompressedTensorsWNA16MoEMethod),
dataset: ShareGPT, gfx1100 (W7900). 3×500 prompts per configuration, results
averaged over warm runs (runs 2 & 3; run 1 excluded due to Triton cache warmup).

```bash
# Server
vllm serve /mnt/nas_share/models/RedHatAI/Qwen3-30B-A3B-quantized.w4a16 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000

# Benchmark (repeated 3 times)
vllm bench serve \
  --model /mnt/nas_share/models/RedHatAI/Qwen3-30B-A3B-quantized.w4a16 \
  --dataset-name sharegpt \
  --dataset-path /mnt/nas_share/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500
```

**Interleave branch (warm avg):**

| Metric | Run 2 | Run 3 | Avg |
| --- | --- | --- | --- |
| Output tok/s | 1227.62 | 1229.30 | 1228.46 |
| Total tok/s | 2383.18 | 2386.44 | 2384.81 |
| Median TPOT (ms) | 131.15 | 132.25 | 131.70 |
| Median ITL (ms) | 121.73 | 122.57 | 122.15 |
| Median TTFT (ms) | 1550.05 | 1696.83 | 1623.44 |

**Main branch (warm avg):**

| Metric | Run 2 | Run 3 | Avg |
| --- | --- | --- | --- |
| Output tok/s | 1110.69 | 1076.49 | 1093.59 |
| Total tok/s | 2156.18 | 2089.79 | 2122.99 |
| Median TPOT (ms) | 139.70 | 138.39 | 139.05 |
| Median ITL (ms) | 128.27 | 128.25 | 128.26 |
| Median TTFT (ms) | 1630.08 | 1667.57 | 1648.83 |

**Comparison (interleave vs main):**

| Metric | Interleave | Main | Delta |
| --- | --- | --- | --- |
| Output tok/s | 1228 | 1094 | **+12.3%** |
| Total tok/s | 2385 | 2123 | **+12.3%** |
| Median TPOT (ms) | 131.7 | 139.0 | **-5.3%** |
| Median ITL (ms) | 122.2 | 128.3 | **-4.8%** |
| Median TTFT (ms) | 1623 | 1649 | -1.5% (within noise) |

The interleave branch delivers **+12% throughput** and **~5% decode latency
reduction** on the compressed-tensors model. TTFT is equivalent.

### End-to-end decode latency: Interleave vs Main (AWQ, single-batch)

Model: Qwen3-30B-A3B-AWQ (MoeWNA16Method), dataset: ShareGPT (100 prompts),
gfx1100 (W7900). `max-num-seqs=1` isolates pure decode latency
(memory-bandwidth-bound), removing batching/scheduling noise.

```bash
# Server
vllm serve /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --max-num-seqs 1

# Benchmark
vllm bench serve \
  --model /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --dataset-name sharegpt \
  --dataset-path /mnt/nas_share/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100
```

| Metric | Interleave | Main | Delta |
| --- | --- | --- | --- |
| Mean TPOT (ms) | 17.87 | 23.63 | **-24.4%** |
| Median TPOT (ms) | 17.95 | 24.14 | **-25.6%** |
| Mean ITL (ms) | 17.87 | 23.58 | **-24.2%** |
| Median ITL (ms) | 17.89 | 24.09 | **-25.7%** |
| Output tok/s | 54.22 | 40.19 | **+34.9%** |
| Mean TTFT (s) | 195.1 | 261.2 | **-25.3%** |
| Median TTFT (s) | 194.3 | 256.2 | **-24.2%** |

TTFT is dominated by queuing delay (`max-num-seqs=1` serializes all 100
requests), so absolute values reflect total queue drain time rather than
single-request prefill latency. The ~25% reduction mirrors the decode
speedup — faster per-token generation drains the queue sooner.

The large decode latency improvement confirms the interleave kernel genuinely
reduces global memory traffic — not just a Triton compilation artifact. The
fp16 path benefits disproportionately because the original kernel compiles to
256 VGPRs with register spilling (Scratch=184-192 bytes on gfx1100), while
the interleave kernel avoids this.

### End-to-end decode latency: Interleave vs Main (compressed-tensors, single-batch)

Model: RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (CompressedTensorsWNA16MoEMethod,
bf16), dataset: ShareGPT (100 prompts), gfx1100 (W7900). Same `max-num-seqs=1`
setup as above.

| Metric | Interleave | Main | Delta |
| --- | --- | --- | --- |
| Mean TPOT (ms) | 18.06 | 17.94 | +0.7% |
| Median TPOT (ms) | 18.13 | 18.02 | +0.6% |
| Mean ITL (ms) | 18.05 | 17.93 | +0.7% |
| Median ITL (ms) | 18.06 | 17.97 | +0.5% |
| Output tok/s | 53.63 | 53.83 | -0.4% |
| Mean TTFT (s) | 195.5 | 204.4 | -4.4% |
| Median TTFT (s) | 188.1 | 198.1 | -5.0% |

Decode latency is identical (within noise). This is expected: the CT model
auto-detects bf16, so the original kernel already compiles without register
spilling (240 VGPRs, Scratch=0). Without the spilling penalty, the pure
algorithmic bandwidth gain from interleave unpacking is too small to measure
at batch=1. The interleave advantage materializes at higher batch sizes
(+12% throughput in the 500-prompt batched benchmark above) where reduced
load instruction count becomes the bottleneck.

### Accuracy (lm_eval gsm8k, 5-shot, 200 samples)

Hardware: gfx1100 (AMD Radeon Pro W7900).

| Model | Branch | strict-match | flexible-extract |
| --- | --- | --- | --- |
| Qwen3-30B-A3B-AWQ (MoeWNA16Method) | main | 0.920 ± 0.019 | 0.890 ± 0.022 |
| Qwen3-30B-A3B-AWQ (MoeWNA16Method) | interleave | 0.915 ± 0.020 | 0.885 ± 0.023 |
| RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (CompressedTensorsWNA16) | main | 0.900 ± 0.021 | 0.910 ± 0.020 |
| RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (CompressedTensorsWNA16) | interleave | 0.895 ± 0.022 | 0.890 ± 0.022 |

No accuracy regression — all deltas are within stderr.
