"""Microbenchmark: aiter pure-Triton mla_decode_fwd vs vLLM in-tree
_fwd_grouped_kernel_stage1 (decode_attention_fwd), on GLM MLA decode shapes.

Runs on gfx1100 (RDNA3). aiter's compiled/ASM path has no RDNA binary; this
compares the *newer* pure-Triton aiter kernel (arch-agnostic `else` branch)
against the kernel vLLM currently ships.
"""

import importlib
import os
import sys
import types

import torch

# --- Bypass aiter's heavy top-level __init__ (which JIT-builds the C core).
# We only need the pure-Triton kernel module. Seed a stub package with the
# right __path__ so real submodules load without running aiter/__init__.py.
AITER_ROOT = "/home/user/workspace/aiter"
_stub = types.ModuleType("aiter")
_stub.__path__ = [os.path.join(AITER_ROOT, "aiter")]
sys.modules["aiter"] = _stub
if AITER_ROOT not in sys.path:
    sys.path.insert(0, AITER_ROOT)

aiter_mla = importlib.import_module("aiter.ops.triton.attention.mla")
print(f"[aiter] loaded {aiter_mla.__file__}")
print(f"[aiter] DEVICE_ARCH={aiter_mla.DEVICE_ARCH} "
      f"IS_GFX12={aiter_mla.IS_DEVICE_ARCH_GFX12} WARP_SIZE={aiter_mla.WARP_SIZE}")

from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

DEV = "cuda"
DTYPE = torch.bfloat16
KV_LORA = 512          # GLM kv_lora_rank -> Lv
ROPE = 64              # qk_rope_head_dim
QK = KV_LORA + ROPE    # 576 = Lk


def build_inputs(B, H, seq_len, page_size, seed=0):
    """Build identical paged KV data usable by both kernels."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    npages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = B * npages_per_seq + 8
    # Shared KV cache: [num_pages, page_size, num_kv_heads=1, QK]
    kv = torch.randn(total_pages, page_size, 1, QK, dtype=DTYPE, device=DEV,
                     generator=g)
    q = torch.randn(B, H, QK, dtype=DTYPE, device=DEV, generator=g)
    # Distinct physical pages per seq (random permutation, no overlap).
    perm = torch.randperm(total_pages, device=DEV, generator=g)
    block_table = perm[:B * npages_per_seq].view(B, npages_per_seq).contiguous()
    block_table = block_table.to(torch.int32)
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=DEV)
    return kv, q, block_table, seq_lens, npages_per_seq


def run_intree(kv, q, block_table, seq_lens, num_kv_splits, scale, page_size):
    B, H, _ = q.shape
    o = torch.empty(B, H, KV_LORA, dtype=DTYPE, device=DEV)
    lse = torch.empty(B, H, dtype=DTYPE, device=DEV)
    v = kv[..., :KV_LORA]
    attn_logits = torch.empty(B, H, num_kv_splits, KV_LORA + 1,
                              dtype=torch.float32, device=DEV)
    decode_attention_fwd(q, kv, v, o, lse, block_table, seq_lens, attn_logits,
                         num_kv_splits, scale, page_size, is_mla=True)
    return o


def run_aiter(kv, q, block_table, seq_lens, scale, page_size):
    B, H, _ = q.shape
    out = torch.empty(B, H, KV_LORA, dtype=DTYPE, device=DEV)
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=DEV)  # 1 tok/seq
    max_seqlen_kv = int(seq_lens.max().item())
    aiter_mla.mla_decode_fwd(
        q, kv, out,
        cu_seqlens_q, seq_lens, max_seqlen_kv, block_table,
        scale, KV_LORA, ROPE, True,
        None, None,  # q_descale, kv_descale (bf16 -> unused)
    )
    return out


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def num_kv_splits_for(seq_len, sm_count):
    import triton
    ideal = triton.next_power_of_2(max(1, seq_len // 512))
    return min(ideal, sm_count * 2)


def main():
    from vllm.platforms import current_platform
    sm = current_platform.num_compute_units()
    scale = 1.0 / (QK ** 0.5)
    print(f"sm_count(CUs)={sm}\n")

    # --- correctness ---
    kv, q, bt, sl, _ = build_inputs(B=4, H=16, seq_len=1027, page_size=16, seed=1)
    nks = num_kv_splits_for(1027, sm)
    o_in = run_intree(kv, q, bt, sl, nks, scale, 16)
    o_ai = run_aiter(kv, q, bt, sl, scale, 16)
    diff = (o_in.float() - o_ai.float()).abs()
    print(f"[correctness B=4 H=16 seq=1027] max_abs={diff.max():.4f} "
          f"mean_abs={diff.mean():.5f} "
          f"rel={ (diff.max()/o_in.float().abs().max()):.4f}\n")

    # --- perf sweep ---
    print(f"{'B':>4} {'H':>4} {'seq':>6} {'page':>4} "
          f"{'intree_ms':>10} {'aiter_ms':>10} {'speedup':>8}")
    for H in (5, 20):  # GLM-4.7-Flash: 20 heads total; ~5/rank on TP4
        for seq in (2048, 8192, 16384):
            for B in (1, 8, 32, 64, 128, 256):
                page = 16
                kv, q, bt, sl, _ = build_inputs(B, H, seq, page, seed=B + seq)
                nks = num_kv_splits_for(seq, sm)
                t_in = bench(lambda: run_intree(kv, q, bt, sl, nks, scale, page))
                t_ai = bench(lambda: run_aiter(kv, q, bt, sl, scale, page))
                spd = t_in / t_ai
                print(f"{B:>4} {H:>4} {seq:>6} {page:>4} "
                      f"{t_in:>10.4f} {t_ai:>10.4f} {spd:>7.2f}x")


if __name__ == "__main__":
    main()
