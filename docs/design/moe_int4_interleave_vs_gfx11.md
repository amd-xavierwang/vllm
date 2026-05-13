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

End-to-end throughput benchmark on the interleave branch showed 2.95x improvement
(2.16 → 6.38 req/s) on Qwen3-30B-A3B-AWQ with 1000 ShareGPT prompts. The gfx11
branch could not be benchmarked end-to-end due to build/startup issues unrelated
to the kernel.
