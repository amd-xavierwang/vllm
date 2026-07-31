# MLA Decode: aiter backends on RDNA (gfx1100) — Findings

Investigation into whether any of aiter's MLA-decode kernels can beat vLLM's
in-tree Triton MLA decode (`TRITON_MLA`) on AMD RDNA3 (gfx1100, W7900).
Target model: GLM-4.7-Flash (20 attn heads, `kv_lora_rank=512`,
`qk_rope_head_dim=64` → latent decode `Lk=576`, `Lv=512`, `num_kv_heads=1`).

## TL;DR

- vLLM's MLA decode on ROCm always uses the in-tree `TRITON_MLA` backend.
  All `ROCM_AITER_*` MLA paths are arch-gated to CDNA MI3xx (gfx942/gfx950) and
  never engage on RDNA.
- aiter ships **three** MLA-decode implementations. Only the newest pure-Triton
  one is a real contender on RDNA.
- Benchmarked against in-tree: **correctness-identical** (rel ~0.3%), aiter
  **loses at low batch** (≤8), **wins 1.1–1.5×** at B≥32, and **up to ~3.4×**
  at B≥128 / long sequences.
- Root cause of the in-tree cliff: vLLM's `num_kv_splits` heuristic ignores
  batch·heads, so at high load it over-splits and the stage-2 reduce becomes
  bandwidth-bound (≈336 MB partial-logits scratch at B=256).

## Context vs vLLM (very brief)

vLLM's in-tree MLA decode (`vllm/v1/attention/ops/triton_decode_attention.py`,
selected by `vllm/v1/attention/backends/mla/triton_mla.py`) is a split-KV /
flash-decoding kernel: stage 1 attends over KV segments in parallel, stage 2
merges the partials via log-sum-exp. It is itself SGLang-derived. On ROCm the
AITER MLA backends (`ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA`) are gated to
MI3xx and fall back to `TRITON_MLA` on RDNA.

## aiter's three MLA-decode backends

| # | Location | Kind | Arches | RDNA? |
|---|---|---|---|---|
| 1 | `aiter/mla.py` → `mla_decode_stage1_asm_fwd` | Precompiled ASM (HSACO `.co`) | gfx942, gfx950, gfx1250 | **No** — no RDNA binary |
| 2 | `aiter/ops/triton/attention/mla_decode.py` | Pure Triton | any (JIT) | Yes, but stale |
| 3 | `aiter/ops/triton/attention/mla.py` | Pure Triton (newer) | any (JIT) | **Yes — tested** |

- **#1 (precompiled, MI):** hand-written assembly, binaries only for CDNA MI3xx
  (+gfx1250). This is what `ROCM_AITER_MLA` calls; dead on RDNA. `MI = CDNA =
  gfx942/gfx950`; note `gfx1250 ≠ MI` (a separate newer arch).
- **#2 (stale fork of vLLM):** header cites vLLM as its source — it is a
  frozen snapshot of vLLM's own in-tree kernel (added 2026-04-17, only a ruff
  formatting commit since). ROCm-path handling has since diverged from
  live vLLM in both directions; **no unique optimization worth harvesting.**
- **#3 (new, tested):** adapted from `triton_unified_attention.py`. Dispatch:
  ```python
  if IS_DEVICE_ARCH_GFX12:   # DEVICE_ARCH == "gfx1250" ONLY
      gluon_mla_decode_fwd_kernel[...]        # Gluon, gfx1250-specialized
  else:                       # gfx1100 / gfx1201 / gfx942 / gfx950
      triton_mla_decode_fwd_kernel[...]       # pure Triton — extends to RDNA
  _reduce_kernel = triton_mla_decode_fwd_reduce_kernel   # pure Triton, both
  ```
  Neither branch is precompiled; both are JIT. The `else` branch (+ the
  unconditional reduce kernel) is what runs on RDNA. `IS_DEVICE_ARCH_GFX12`
  means gfx1250 *only* — RDNA4 (gfx1200/1201) also takes the `else` branch.
  Same two-stage split-KV structure as vLLM (`_mla_decode_fwd_kernel` ≈
  stage 1, `_mla_decode_fwd_reduce_kernel` ≈ stage 2, `NUM_SEGMENTS_PER_SEQ` ≈
  `num_kv_splits`).

## Benchmark

Microbenchmark: aiter #3 `mla_decode_fwd` vs in-tree `decode_attention_fwd`,
identical paged KV cache `[num_pages, page_size, 1, 576]`, bf16, CUDA-event
timing (warmup 10, iters 50). Script: `bench_mla_aiter_vs_intree.py`.
aiter's heavy `__init__` (JIT C-core build) is bypassed by stubbing
`sys.modules["aiter"]` and importing only the pure-Triton submodule — no
`setup.py develop`, so the container's Triton is untouched.

**Correctness (B=4, H=16, seq=1027):** `max_abs ≈ 0.001`, rel ≈ 0.3% — outputs
match; the two kernels compute the same thing.

**Performance (speedup = in-tree_ms / aiter_ms; >1 means aiter faster):**

| Regime | Batch | speedup |
|---|---|---|
| Low batch | B ≤ 8 | 0.5–0.9× (aiter slower) |
| Mid batch | B = 32–64 | 1.1–1.5× |
| High load | B ≥ 128, seq 8k–16k | 3.0–3.4× |

Swept H∈{5,20}, seq∈{2048,8192,16384}, B∈{1,8,32,64,128,256}, page=16.

## Root cause of the high-batch gap

vLLM's split count is chosen from sequence length and CU count only:

```python
# vllm/v1/attention/backends/mla/triton_mla.py
def _compute_num_kv_splits(max_seq_len, sm_count):
    ideal = triton.next_power_of_2(max(1, max_seq_len // 512))
    return min(ideal, sm_count * 2)
```

It ignores `B` and `H`. At high batch every sequence still requests many splits,
so the stage-1 partial-logits buffer `(B, H, num_kv_splits, Lv+1)` explodes
(≈336 MB at B=256) and stage 2 becomes bandwidth-bound. aiter chooses
`NUM_SEGMENTS_PER_SEQ` in a way that scales better under load, which is where
its high-batch win comes from. At low batch there is already enough parallelism,
so aiter's extra machinery just adds overhead and it loses.

## Takeaways / possible next steps (not yet pursued)

1. **Cheap in-tree fix:** make `_compute_num_kv_splits` occupancy-aware (gate on
   `B·H` so splits back off once the machine is already full). Likely captures
   much of aiter's high-batch win with no new dependency or integration.
2. **Wire aiter #3 behind a batch gate:** dispatch to aiter's Triton
   `mla_decode_fwd` only when B is large, keeping in-tree for low batch. More
   integration cost; only worth it if (1) leaves gains on the table.

Backends #1 and #2 are dead ends on RDNA (precompiled MI-only; stale vLLM fork).
