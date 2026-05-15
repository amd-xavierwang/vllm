"""Standalone test for ExLlama shuffle MoE kernel with realistic shapes."""
import torch
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config

def test_exllama_kernel(E, N, K, group_size, M, top_k, has_zp=True):
    print(f"\n--- E={E}, N={N}, K={K}, G={group_size}, M={M}, top_k={top_k}, has_zp={has_zp} ---")
    torch.manual_seed(42)
    device = "cuda"

    # Create ExLlama shuffle packed weights [E, N, K//8] int32
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
        zp_packed = (zp_unpacked.view(E, num_groups, N8, 8) << shifts).sum(dim=-1, dtype=torch.int32)
    else:
        zp_packed = None

    x = torch.randn(M, K, dtype=torch.float16, device=device)
    topk_weights = torch.ones(M, top_k, dtype=torch.float16, device=device) / top_k
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)

    # w1 = gate+up [E, 2*N, K//8], w2 = down [E, N, K//8]
    # For w2, the input K dimension = N (intermediate size)
    w1 = w_packed.repeat(1, 2, 1)
    w2_int4 = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    g2 = w2_int4.view(E, K, N // 8, 8)
    w2 = (
        g2[..., 0] | (g2[..., 2] << 4) | (g2[..., 4] << 8) | (g2[..., 6] << 12)
        | (g2[..., 1] << 16) | (g2[..., 3] << 20) | (g2[..., 5] << 24) | (g2[..., 7] << 28)
    ).contiguous()  # [E, K, N//8] — wait, w2 input dim is N not K

    # Actually for w2: input_dim = intermediate_size = N, output_dim = hidden_size = K
    # So w2 shape should be [E, K, N//8] where N is the packed input dim
    # But the model stores w2 as [E, hidden, inter//2] uint8 originally
    # After ExLlama repacking: [E, hidden, inter//8] int32 = [E, K, N//8]
    # Hmm but our repacking keeps the original [E, N_out, K_in//8] orientation
    # For w2: N_out = K (hidden), K_in = N (inter)
    # So w2 = [E, K, N//8] int32

    # Let me just match what the repacking produces
    # w2_qweight original: [E, hidden, inter//2] uint8 = [E, K, N//2]
    # After _repack_int4_to_int32: [E, K, N//8] int32
    w2_raw = torch.randint(0, 16, (E, K, N), dtype=torch.int32, device=device)
    g2 = w2_raw.view(E, K, N // 8, 8)
    w2 = (
        g2[..., 0] | (g2[..., 2] << 4) | (g2[..., 4] << 8) | (g2[..., 6] << 12)
        | (g2[..., 1] << 16) | (g2[..., 3] << 20) | (g2[..., 5] << 24) | (g2[..., 7] << 28)
    ).contiguous()  # [E, K, N//8]

    s1 = scales.repeat(1, 2, 1)
    # s2: [E, K, N//group_size] — w2 output=K rows, input groups of N
    s2_groups = N // group_size
    s2 = torch.rand(E, K, s2_groups, dtype=torch.float16, device=device) * 0.1

    if has_zp:
        zp1 = zp_packed.repeat(1, 1, 2)
        # zp2: [E, s2_groups, K//8] int32
        K8_zp = K // 8
        zp2_unpacked = torch.randint(0, 16, (E, s2_groups, K), dtype=torch.int32, device=device)
        zp2 = (zp2_unpacked.view(E, s2_groups, K8_zp, 8) << shifts).sum(dim=-1, dtype=torch.int32)
    else:
        zp1 = None
        zp2 = None

    print(f"w1: {w1.shape}, w2: {w2.shape}")
    print(f"s1: {s1.shape}, s2: {s2.shape}")
    if has_zp:
        print(f"zp1: {zp1.shape}, zp2: {zp2.shape}")

    quant_config = int4_w4a16_moe_quant_config(
        w1_scale=s1, w2_scale=s2,
        w1_zp=zp1, w2_zp=zp2,
        block_shape=[0, group_size],
    )

    try:
        out = fused_experts(
            x, w1, w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_config=quant_config,
        )
        torch.cuda.synchronize()
        print(f"SUCCESS! Output shape: {out.shape}, sample: {out[0, :3]}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

# Test with small shapes first
test_exllama_kernel(E=8, N=512, K=2048, group_size=128, M=4, top_k=2)

# Test with Qwen3.6 shapes
test_exllama_kernel(E=256, N=512, K=2048, group_size=128, M=4, top_k=2)

# Test with larger batch
test_exllama_kernel(E=256, N=512, K=2048, group_size=128, M=64, top_k=2)
