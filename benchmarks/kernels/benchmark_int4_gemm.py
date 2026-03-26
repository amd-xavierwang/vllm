# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the HybridW4A16LinearKernel across decode and prefill shapes.

This benchmarks _hybrid_w4a16_apply_impl which dispatches:
  M <= 4:  HIP wvSplitK_int4_g skinny GEMM (decode)
  M > 4:   Triton W4A16 fused dequant GEMM (prefill, skinny format)

Weights are stored ONCE in skinny layout [N, K//8] int32 (ExLlama shuffle).
Both the HIP skinny kernel and the triton kernel read from this single copy.

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

from vllm.triton_utils import triton

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
    """Create random int4 weights in skinny format + fp16 baseline.

    Returns (w_q_skinny, w_s_skinny, w_fp16, w_q_skinny_i32).
    """
    num_groups = K // group_size

    # Random uint4 values [N, K] (skinny's native layout: output_dim=0)
    w_int_NK = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    # Random scales [N, K//G] (skinny's native layout)
    scales_NK = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.01

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

    return w_q_skinny, w_s_skinny, w_fp16, w_q_skinny_i32


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
PROVIDERS = ["torch-fp16", "hybrid-w4a16"]


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
                w["w_q_skinny_i32"],
                None,  # bias
                cu_count,
                group_size,
                8,  # zp_bias
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

        w_q_skinny, w_s_skinny, w_fp16, w_q_skinny_i32 = prepare_hybrid_weights(
            K, N, group_size
        )

        weights = {
            "w_q_skinny": w_q_skinny,
            "w_s_skinny": w_s_skinny,
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
