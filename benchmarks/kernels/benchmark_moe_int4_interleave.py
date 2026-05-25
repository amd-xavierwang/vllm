"""Kernel-level benchmark for MoE int4 interleave vs ExLlama shuffle.

Profiles the fused_moe kernel directly with model-specific shapes.
Supports Qwen3.6-35B-A3B-AWQ and Qwen3-30B-A3B-AWQ.
"""
import argparse

import torch
import torch.cuda

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config

MODEL_CONFIGS = {
    "qwen3.6-35b": {
        "name": "Qwen3.6-35B-A3B-AWQ",
        "E": 256, "hidden": 2048, "inter": 512, "top_k": 8, "group_size": 128,
    },
    "qwen3-30b": {
        "name": "Qwen3-30B-A3B-AWQ",
        "E": 128, "hidden": 2048, "inter": 768, "top_k": 8, "group_size": 128,
    },
}


def make_exllama_weights(E, N, K, group_size, has_zp, device):
    """Create ExLlama shuffle K-packed weights [E, N, K//8] int32."""
    K8 = K // 8
    num_groups = K // group_size
    w_int4 = torch.randint(0, 16, (E, N, K), dtype=torch.int32, device=device)
    g = w_int4.view(E, N, K8, 8)
    w_packed = (
        g[..., 0] | (g[..., 2] << 4) | (g[..., 4] << 8) | (g[..., 6] << 12)
        | (g[..., 1] << 16) | (g[..., 3] << 20) | (g[..., 5] << 24) | (g[..., 7] << 28)
    ).contiguous()
    scales = torch.rand(E, N, num_groups, dtype=torch.float16, device=device) * 0.1
    if has_zp:
        N8 = N // 8
        zp_unpacked = torch.randint(0, 16, (E, num_groups, N), dtype=torch.int32, device=device)
        shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
        zp_packed = (zp_unpacked.view(E, num_groups, N8, 8) << shifts).sum(
            dim=-1, dtype=torch.int32)
    else:
        zp_packed = None
    return w_packed, scales, zp_packed


def make_interleave_weights(E, N, K, group_size, has_zp, device):
    """Create GPTQ sequential N-packed weights [E, K, N//8] int32."""
    N8 = N // 8
    num_groups = K // group_size
    w_int4 = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    g = w_int4.view(E, K, N8, 8)
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    w_packed = (g << shifts).sum(dim=-1, dtype=torch.int32).contiguous()
    scales = torch.rand(E, num_groups, N, dtype=torch.float16, device=device) * 0.1
    if has_zp:
        zp_unpacked = torch.randint(0, 16, (E, num_groups, N), dtype=torch.int32, device=device)
        zp_packed = (zp_unpacked.view(E, num_groups, N8, 8) << shifts).sum(
            dim=-1, dtype=torch.int32)
    else:
        zp_packed = None
    return w_packed, scales, zp_packed


def bench_fused_experts(label, E, N_out, K_in, group_size, M_values, top_k,
                        has_zp, layout, warmup_iters, bench_iters, device):
    """Benchmark fused_experts (w1+SiLU+w2) for a given shape."""
    print(f"\n  {label}: E={E}, N_out={N_out}, K_in={K_in}")
    print(f"  {'M':>6} {'Time(ms)':>10} {'TFLOPS':>8} {'GB/s':>8}")
    print(f"  {'-'*38}")

    results = []
    for M in M_values:
        torch.manual_seed(42)

        make_fn = make_exllama_weights if layout == "exllama" else make_interleave_weights

        w1, s1, zp1 = make_fn(E, N_out, K_in, group_size, has_zp, device)
        w2_N_out = K_in
        w2_K_in = N_out // 2
        w2, s2, zp2 = make_fn(E, w2_N_out, w2_K_in, group_size, has_zp, device)

        x = torch.randn(M, K_in, dtype=torch.float16, device=device)
        topk_weights = torch.ones(M, top_k, dtype=torch.float16, device=device) / top_k
        topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)

        quant_config = int4_w4a16_moe_quant_config(
            w1_scale=s1, w2_scale=s2,
            w1_zp=zp1, w2_zp=zp2,
            block_shape=[0, group_size],
        )

        for _ in range(warmup_iters):
            out = fused_experts(x, w1, w2, topk_weights=topk_weights,
                                topk_ids=topk_ids, quant_config=quant_config)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(bench_iters):
            out = fused_experts(x, w1, w2, topk_weights=topk_weights,
                                topk_ids=topk_ids, quant_config=quant_config)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / bench_iters

        flops = M * top_k * 2 * (N_out * K_in + K_in * (N_out // 2))
        tflops = flops / (elapsed_ms * 1e-3) / 1e12

        w1_bytes = N_out * K_in // 2
        w2_bytes = K_in * (N_out // 2) // 2
        act_bytes = M * (K_in + N_out + K_in) * 2
        total_bytes = (w1_bytes + w2_bytes) * min(M * top_k, E) + act_bytes
        gbps = total_bytes / (elapsed_ms * 1e-3) / 1e9

        print(f"  {M:>6} {elapsed_ms:>10.3f} {tflops:>8.3f} {gbps:>8.1f}")
        results.append((M, elapsed_ms, tflops, gbps))

        del w1, w2, s1, s2, zp1, zp2, x, out
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["interleave", "exllama"], required=True)
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen3-30b")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-zp", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    has_zp = not args.no_zp
    cfg = MODEL_CONFIGS[args.model]

    E = cfg["E"]
    hidden = cfg["hidden"]
    inter = cfg["inter"]
    group_size = cfg["group_size"]
    top_k = cfg["top_k"]

    M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    print(f"=== Kernel Benchmark: {args.layout} layout ===")
    print(f"Model: {cfg['name']} (E={E}, hidden={hidden}, inter={inter}, top_k={top_k})")
    print(f"has_zp={has_zp}, warmup={args.warmup}, iters={args.iters}")

    bench_fused_experts(
        label="fused_experts (w1+SiLU+w2)",
        E=E, N_out=2*inter, K_in=hidden,
        group_size=group_size, M_values=M_values, top_k=top_k,
        has_zp=has_zp, layout=args.layout,
        warmup_iters=args.warmup, bench_iters=args.iters, device=device,
    )


if __name__ == "__main__":
    main()
