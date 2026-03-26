# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the HybridW4A16LinearKernel across decode and prefill shapes.

This benchmarks _hybrid_w4a16_apply_impl which dispatches:
  M <= 4:  HIP wvSplitK_int4_g skinny GEMM (decode)
  M > 4:   Triton W4A16 fused dequant GEMM (prefill)

Also includes:
  - torch fp16 baseline for comparison
  - hybrid-unified: reads ONLY from skinny format [N, K//8] for both paths
    (triton kernel modified to read transposed ExLlama-packed weights)

Weight layouts:
  Triton:  qweight [K, N//8] int32 (GPTQ sequential), scales [K//G, N] fp16
  Skinny:  qweight [N, K//8] int8 (ExLlama shuffle),  scales [N, K//G] fp16

Usage:
    python benchmark_int4_gemm.py
    python benchmark_int4_gemm.py --models Qwen/Qwen3-4B
    python benchmark_int4_gemm.py --group-size 128
"""

import argparse
import copy
import itertools
import os

import torch

from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# Weight shapes: [K, N], TP_SPLIT_DIM
# ---------------------------------------------------------------------------
WEIGHT_SHAPES = {
    "Qwen/Qwen3-4B": [
        ([2560, 3840], 1),  # qkv_proj
        ([2560, 2560], 0),  # o_proj
        ([2560, 19456], 1),  # gate_up_proj
        ([9728, 2560], 0),  # down_proj
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        ([3584, 4608], 1),
        ([3584, 3584], 0),
        ([3584, 37888], 1),
        ([18944, 3584], 0),
    ],
}


# ---------------------------------------------------------------------------
# Weight packing (mirrors hybrid_w4a16.py process_weights_after_loading)
# ---------------------------------------------------------------------------
def prepare_hybrid_weights(K, N, group_size, device="cuda"):
    """Create random int4 weights in both triton and skinny formats.

    Returns (w_q_skinny, w_s_skinny, w_q_triton, w_s_triton, w_fp16).
    """
    num_groups = K // group_size

    # Random uint4 values [N, K] (skinny's native layout: output_dim=0)
    w_int_NK = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    # Random scales [N, K//G] (skinny's native layout)
    scales_NK = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.01

    # ---- Triton weights: [K, N//8] int32 (GPTQ sequential) ----
    w_KN = w_int_NK.t().contiguous()  # [K, N]
    N8 = N // 8
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    w_q_triton = torch.sum(
        (w_KN.view(K, N8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()

    # Triton scales: [K//G, N]
    w_s_triton = scales_NK.t().contiguous()

    # ---- Skinny weights: [N, K//8] int8 (ExLlama shuffle for fp16) ----
    unsigned = w_int_NK.to(torch.uint8)
    g = unsigned.view(N, K // 8, 8).to(torch.int32)
    shuffled = (
        g[:, :, 0]
        | (g[:, :, 2] << 4)
        | (g[:, :, 4] << 8)
        | (g[:, :, 6] << 12)
        | (g[:, :, 1] << 16)
        | (g[:, :, 3] << 20)
        | (g[:, :, 5] << 24)
        | (g[:, :, 7] << 28)
    )
    w_q_skinny = shuffled.contiguous().view(torch.int8).contiguous()

    # Also store the skinny weights as int32 for the triton kernel to read
    w_q_skinny_i32 = shuffled.contiguous()

    # Skinny scales: [N, K//G]
    w_s_skinny = scales_NK.contiguous()

    # ---- FP16 baseline: dequantize to [N, K] for F.linear ----
    w_fp = w_int_NK.to(torch.float16) - 8.0
    w_fp = w_fp.view(N, num_groups, group_size)
    w_fp = w_fp * scales_NK.unsqueeze(-1)
    w_fp16 = w_fp.view(N, K).contiguous()

    return w_q_skinny, w_s_skinny, w_q_triton, w_s_triton, w_fp16, w_q_skinny_i32


# ---------------------------------------------------------------------------
# Triton kernel that reads from skinny format [N, K//8] with ExLlama shuffle
# ---------------------------------------------------------------------------
@triton.jit
def triton_w4a16_gemm_skinny_fmt_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [N, K//8]  int32 packed (ExLlama shuffle, K is packed dim)
    scales_ptr,  # [N, K//G]  fp16/bf16 scales (skinny layout)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    # Quantization parameters
    group_size,
    ZP_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM reading weights from skinny format [N, K//8].

    B is stored as [N, K//8] int32 using ExLlama shuffle packing:
      each int32 packs 8 K-values with interleave [0,2,4,6,1,3,5,7]:
        packed = val[0] | (val[2]<<4) | (val[4]<<8) | (val[6]<<12)
               | (val[1]<<16) | (val[3]<<20) | (val[5]<<24) | (val[7]<<28)

    Scales are [N, K//G] (skinny layout, NOT transposed).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ExLlama unshuffle shifts: to extract val[j] from the packed int32,
    # we need shift[j] = (j//2)*4 + (j%2)*16
    # For 8 values: [0, 16, 4, 20, 8, 24, 12, 28]
    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    # Tile across BLOCK_K: repeat the 8-element pattern BLOCK_K//8 times
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    # Broadcast to [BLOCK_N, BLOCK_K]
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    # b_ptr layout: [N, K//8] int32, row-major
    # stride along N = K//8, stride along packed-K = 1
    K8 = K // 8
    num_groups = K // group_size

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load packed weights B: [BLOCK_N, BLOCK_K//8] int32 ----
        # From [N, K//8] layout
        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]
        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)  # [BLOCK_N, BLOCK_K//8]

        # ---- Unpack int4 weights with ExLlama unshuffle ----
        # Interleave to [BLOCK_N, BLOCK_K]
        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        # Apply ExLlama unshuffle shifts
        b = (b >> shifts_full) & 0xF  # [BLOCK_N, BLOCK_K]

        # ---- Load scales: [BLOCK_N] from [N, K//G] layout ----
        g_idx = (k_start * BLOCK_K) // group_size
        scale_ptrs = scales_ptr + offs_n * num_groups + g_idx
        scale_mask = offs_n < N
        scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)  # [BLOCK_N]

        # ---- Dequantize: (w - zp_bias) * scale → [BLOCK_N, BLOCK_K] ----
        b_fp = (b - ZP_BIAS).to(scales.dtype) * scales[:, None]

        # ---- Transpose to [BLOCK_K, BLOCK_N] for matmul ----
        b_fp_t = tl.trans(b_fp)

        # ---- Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] ----
        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)

    # ---- Store output C: [BLOCK_M, BLOCK_N] ----
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_w4a16_gemm_skinny_fmt(
    a: torch.Tensor,  # [M, K] fp16
    b_q: torch.Tensor,  # [N, K//8] int32 (ExLlama shuffle packed)
    scales: torch.Tensor,  # [N, K//G] fp16
    group_size: int,
    zp_bias: int = 8,
) -> torch.Tensor:
    """Triton W4A16 GEMM reading from skinny weight format."""
    M, K = a.shape
    N = b_q.shape[0]

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    if M <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
    elif M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    triton_w4a16_gemm_skinny_fmt_kernel[grid](
        a,
        b_q,
        scales,
        c,
        M,
        N,
        K,
        group_size=group_size,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


# ---------------------------------------------------------------------------
# Unified hybrid dispatch: both paths read from skinny format only
# ---------------------------------------------------------------------------
def _hybrid_unified_apply(
    x_2d: torch.Tensor,
    w_q_skinny: torch.Tensor,  # [N, K//8] int8 for skinny, or int32 for triton
    w_s_skinny: torch.Tensor,  # [N, K//G] fp16
    w_q_skinny_i32: torch.Tensor,  # [N, K//8] int32 for triton kernel
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and triton, both reading skinny format."""
    import vllm._custom_ops as ops

    SKINNY_THRESHOLD = 4
    LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2

    N = x_2d.shape[0]
    K = x_2d.shape[1]

    if N <= SKINNY_THRESHOLD and K * N <= LDS_CAPACITY_ELEMENTS:
        if group_size > 0:
            return ops.wvSplitK_int4_g(
                w_q_skinny, x_2d, w_s_skinny, cu_count, group_size, None
            )
        else:
            return ops.wvSplitK_int4(w_q_skinny, x_2d, w_s_skinny, cu_count, None)

    return triton_w4a16_gemm_skinny_fmt(
        a=x_2d,
        b_q=w_q_skinny_i32,
        scales=w_s_skinny,
        group_size=group_size,
        zp_bias=8,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
PROVIDERS = ["torch-fp16", "hybrid-w4a16", "hybrid-unified"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        ylabel="TFLOP/s (larger is better)",
        plot_name="FP16 vs Hybrid W4A16",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, group_size, weights):
    M = batch_size
    device = "cuda"
    dtype = torch.float16
    a = torch.randn((M, K), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-fp16":
        w_fp16 = weights["w_fp16"]
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, w_fp16),
            quantiles=quantiles,
        )
    elif provider == "hybrid-w4a16":
        from vllm.model_executor.kernels.linear.mixed_precision.hybrid_w4a16 import (
            _hybrid_w4a16_apply_impl,
        )
        from vllm.utils.platform_utils import num_compute_units

        w = weights
        cu_count = num_compute_units()

        def run():
            return _hybrid_w4a16_apply_impl(
                a,
                w["w_q_skinny"],
                w["w_s_skinny"],
                w["w_q_triton"],
                w["w_s_triton"],
                None,  # w_zp_triton
                None,  # bias
                cu_count,
                group_size,
                8,  # zp_bias
            )

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            run,
            quantiles=quantiles,
        )
    elif provider == "hybrid-unified":
        from vllm.utils.platform_utils import num_compute_units

        w = weights
        cu_count = num_compute_units()

        def run():
            return _hybrid_unified_apply(
                a,
                w["w_q_skinny"],
                w["w_s_skinny"],
                w["w_q_skinny_i32"],
                cu_count,
                group_size,
            )

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            run,
            quantiles=quantiles,
        )
    else:
        return 0.0, 0.0, 0.0

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


def prepare_shapes(args):
    KN_model_names = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            KN_model_names.append(KN)
    return KN_model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark HybridW4A16LinearKernel")
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["Qwen/Qwen3-4B"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    for K, N, model in prepare_shapes(args):
        group_size = args.group_size
        print(f"\n{'=' * 70}")
        print(f"{model}, N={N} K={K}, group_size={group_size}")
        print(f"{'=' * 70}")

        w_q_skinny, w_s_skinny, w_q_triton, w_s_triton, w_fp16, w_q_skinny_i32 = (
            prepare_hybrid_weights(K, N, group_size)
        )

        weights = {
            "w_q_skinny": w_q_skinny,
            "w_s_skinny": w_s_skinny,
            "w_q_triton": w_q_triton,
            "w_s_triton": w_s_triton,
            "w_fp16": w_fp16,
            "w_q_skinny_i32": w_q_skinny_i32,
        }

        save_path = args.save_path or f"bench_int4_res_n{N}_k{K}"
        os.makedirs(save_path, exist_ok=True)
        benchmark.run(
            print_data=True,
            show_plots=False,
            save_path=save_path,
            N=N,
            K=K,
            group_size=group_size,
            weights=weights,
        )

    print("\nBenchmark finished!")
