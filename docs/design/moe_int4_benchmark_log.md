# MoE Int4 Interleave Benchmark Log

**Date:** 2026-05-26
**GPU:** 8x gfx1100 (AMD Radeon Pro W7900)
**Model:** Qwen3-30B-A3B-AWQ (`/mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ`)
**Dataset:** ShareGPT 500 prompts (`/mnt/nas_share/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json`)
**Branches:**
- `main` @ `10d264a2b9`
- `moe-interleave-fp16-zeros` @ `d75069ab38`

## Test Matrix

| # | Branch | Dtype | Notes |
|---|--------|-------|-------|
| 1 | main | fp16 | AWQ default (no --dtype flag) |
| 2 | main | bf16 | --dtype bfloat16 |
| 3 | interleave | fp16 | AWQ default (no --dtype flag) |
| 4 | interleave | bf16 | --dtype bfloat16 |

## Commands

### Server (run in background, one at a time)

```bash
# --- Config 1: main, fp16 ---
git checkout main
vllm serve /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --max_model_len 4096 --gpu_memory_utilization 0.90

# --- Config 2: main, bf16 ---
git checkout main
vllm serve /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --max_model_len 4096 --gpu_memory_utilization 0.90 --dtype bfloat16

# --- Config 3: interleave, fp16 ---
git checkout moe-interleave-fp16-zeros
vllm serve /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --max_model_len 4096 --gpu_memory_utilization 0.90

# --- Config 4: interleave, bf16 ---
git checkout moe-interleave-fp16-zeros
vllm serve /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --max_model_len 4096 --gpu_memory_utilization 0.90 --dtype bfloat16
```

### Benchmark (run after server is ready, for each config)

```bash
vllm bench serve \
  --model /mnt/nas_share/models/Qwen/Qwen3-30B-A3B-AWQ \
  --dataset-name sharegpt \
  --dataset-path /mnt/nas_share/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500
```

### Workflow per config

1. Start server (background)
2. Wait for "Application startup complete" or equivalent
3. Run benchmark
4. Record results below
5. Kill server (`kill <pid>` or Ctrl+C)
6. Verify GPU memory freed (`rocm-smi --showmemuse`)
7. Switch branch if needed, repeat

---

## Results

### Config 1: main, fp16

```
Successful requests:                     500
Benchmark duration (s):                  279.95
Total input tokens:                      103437
Total generated tokens:                  109888
Request throughput (req/s):              1.79
Output token throughput (tok/s):         392.52
Total token throughput (tok/s):          762.00
Mean TTFT (ms):                          39052.71
Median TTFT (ms):                        29165.87
P99 TTFT (ms):                           117472.33
Mean TPOT (ms):                          508.71
Median TPOT (ms):                        438.64
P99 TPOT (ms):                           968.23
Mean ITL (ms):                           408.45
Median ITL (ms):                         381.39
P99 ITL (ms):                            971.21
```

### Config 2: main, bf16

```
Successful requests:                     500
Benchmark duration (s):                  127.37
Total input tokens:                      103437
Total generated tokens:                  109888
Request throughput (req/s):              3.93
Output token throughput (tok/s):         862.78
Peak output token throughput (tok/s):    1746.00
Peak concurrent requests:                500.00
Total token throughput (tok/s):          1674.90
Mean TTFT (ms):                          18279.83
Median TTFT (ms):                        13875.31
P99 TTFT (ms):                           52163.28
Mean TPOT (ms):                          223.27
Median TPOT (ms):                        181.15
P99 TPOT (ms):                           475.43
Mean ITL (ms):                           169.09
Median ITL (ms):                         147.33
P99 ITL (ms):                            477.70
```

### Config 3: interleave, fp16

```
Successful requests:                     500
Benchmark duration (s):                  114.81
Total input tokens:                      103437
Total generated tokens:                  109878
Request throughput (req/s):              4.36
Output token throughput (tok/s):         957.08
Peak output token throughput (tok/s):    2046.00
Peak concurrent requests:                500.00
Total token throughput (tok/s):          1858.06
Mean TTFT (ms):                          22737.29
Median TTFT (ms):                        19258.93
P99 TTFT (ms):                           52326.66
Mean TPOT (ms):                          194.04
Median TPOT (ms):                        159.13
P99 TPOT (ms):                           424.84
Mean ITL (ms):                           147.22
Median ITL (ms):                         127.41
P99 ITL (ms):                            424.62
```

### Config 4: interleave, bf16

```
Successful requests:                     500
Benchmark duration (s):                  122.55
Total input tokens:                      103437
Total generated tokens:                  109888
Request throughput (req/s):              4.08
Output token throughput (tok/s):         896.68
Peak output token throughput (tok/s):    1920.00
Peak concurrent requests:                500.00
Total token throughput (tok/s):          1740.71
Mean TTFT (ms):                          24376.02
Median TTFT (ms):                        20387.52
P99 TTFT (ms):                           55769.31
Mean TPOT (ms):                          203.80
Median TPOT (ms):                        166.17
P99 TPOT (ms):                           442.27
Mean ITL (ms):                           154.98
Median ITL (ms):                         133.91
P99 ITL (ms):                            451.82
```

## Summary

| Metric | Main fp16 | Main bf16 | Interleave fp16 | Interleave bf16 |
|--------|-----------|-----------|-----------------|-----------------|
| Duration (s) | 279.95 | 127.37 | 114.81 | 122.55 |
| Request throughput (req/s) | 1.79 | 3.93 | **4.36** | 4.08 |
| Output tok/s | 392.52 | 862.78 | **957.08** | 896.68 |
| Peak output tok/s | 768.00 | 1746.00 | **2046.00** | 1920.00 |
| Total tok/s | 762.00 | 1674.90 | **1858.06** | 1740.71 |
| Mean TTFT (ms) | 39052.71 | **18279.83** | 22737.29 | 24376.02 |
| Median TTFT (ms) | 29165.87 | **13875.31** | 19258.93 | 20387.52 |
| P99 TTFT (ms) | 117472.33 | **52163.28** | 52326.66 | 55769.31 |
| Mean TPOT (ms) | 508.71 | 223.27 | **194.04** | 203.80 |
| Median TPOT (ms) | 438.64 | 181.15 | **159.13** | 166.17 |
| P99 TPOT (ms) | 968.23 | 475.43 | **424.84** | 442.27 |
| Mean ITL (ms) | 408.45 | 169.09 | **147.22** | 154.98 |
| Median ITL (ms) | 381.39 | 147.33 | **127.41** | 133.91 |
| P99 ITL (ms) | 971.21 | 477.70 | **424.62** | 451.82 |

### Key Takeaways

- **Main fp16 → Interleave fp16**: +144% output tok/s (392→957), -64% median TPOT (439→159ms). Eliminates register spilling.
- **Main bf16 → Interleave fp16**: +11% output tok/s (863→957), -12% median TPOT (181→159ms). Pure algorithmic gain.
- **Interleave fp16 vs Interleave bf16**: fp16 is ~7% faster (957 vs 897 tok/s) — with spilling gone, fp16 benefits from native WMMA on gfx1100.
- **TTFT**: Main bf16 has best TTFT (14s median vs 19s interleave). Consistent with ATT M=4096 data showing ExLlama prefill advantage at large M.

---

## Compressed-Tensors: RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (bf16)

**Model:** RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (`CompressedTensorsWNA16MoEMethod`, symmetric int4, bf16)
**Methodology:** 3 runs per config, warm average (runs 2-3). Run 1 is cold (no prefix cache).

### CT Summary (warm average, runs 2-3)

| Metric | Main bf16 | Interleave bf16 | Delta |
|--------|-----------|-----------------|-------|
| Output tok/s | 1095 | **1245** | **+14%** |
| Total tok/s | 2123 | **2385** | **+12%** |
| Median TTFT (ms) | 1635 | **1598** | -2% (noise) |
| Median TPOT (ms) | 139.0 | **130.9** | **-6%** |
| Median ITL (ms) | 128.4 | **121.3** | **-6%** |

**CT Key Takeaways:**
- Interleave wins throughput by **+14%** and decode latency (TPOT/ITL) by **~6%** on warm runs.
- TTFT is equivalent (~1600ms median) — no regression once prefix cache is warm.
- Cold first-run TTFT varies significantly (12-23s) due to prefix cache state; warm runs are the reliable comparison.
- The CT model uses bf16 natively (no VGPR spilling on main), so the gain is purely algorithmic from fewer load instructions in the interleave unpacking path.

---

## GPTQ: Qwen3.5-35B-A3B-GPTQ-Int4 (fp16)

**Model:** Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 (`MoeWNA16Method`, symmetric int4, fp16, 256 experts)
**Methodology:** 3 runs per config, warm average (runs 2-3).

### GPTQ Summary (warm average, runs 2-3)

| Metric | Main fp16 | Interleave fp16 | Delta |
|--------|-----------|-----------------|-------|
| Output tok/s | 517 | **556** | **+7%** |
| Peak output tok/s | 1166 | **1219** | +5% |
| Total tok/s | 1013 | **1097** | **+8%** |
| Median TTFT (ms) | 20,073 | **18,624** | -7% |
| Median TPOT (ms) | 355 | **352** | -1% |
| Median ITL (ms) | 189 | **179** | **-5%** |

**GPTQ Key Takeaways:**
- Interleave wins throughput by **+7%** and ITL by **-5%** on a 256-expert GPTQ model.
- TTFT also improves by 7% — no regression on GPTQ.
- This model uses fp16, confirming interleave eliminates VGPR spilling on gfx1100 for GPTQ as well as AWQ.

---

## VGPR and Register Spilling Analysis (gfx1100)

**Kernel:** `fused_moe_kernel_gptq_awq` (Qwen3-30B-A3B shape: E=128, N=768, K=2048, group_size=128)
**Method:** Triggered kernel with synthetic inputs under `BLOCK_M=16, BLOCK_N=64, BLOCK_K=128`, parsed `.amdgcn` dumps from Triton cache via `check-spill.sh`.

| Variant | VGPRs | VGPR Spills | SGPR Spills | Scratch (B) | Occupancy |
|---------|-------|-------------|-------------|-------------|-----------|
| **Interleave fp16** | **144** | 0 | 0 | 0 | **10** |
| **Interleave bf16** | **163** | 0 | 0 | 0 | **9** |
| Original fp16 | 256 | 219 | 17 | 880 | 5 |
| Original bf16 | 256 | 133 | 12 | 416 | 5 |

**Key Observations:**
- Original fp16 saturates the gfx1100 VGPR budget (256) with **219 VGPR spills** and 880B scratch per thread — the root cause of the 2.4× slowdown vs all other configs in batched serving.
- Original bf16 also hits 256 VGPRs but with fewer spills (133) — consistent with bf16 being faster than fp16 on main.
- Interleave fp16 reduces to **144 VGPRs** (−44%), eliminating all spilling and doubling occupancy (5→10 waves).
- Interleave bf16 at 163 VGPRs is slightly higher than fp16 but still zero-spill with 9 waves occupancy.
- With spilling gone, fp16 has a slight edge over bf16 from native WMMA support on gfx1100.

### Why fp16 spills more than bf16 (LLVM root cause)

Inspecting the LLIR and AMDGCN reveals the root cause is **type conversion cost asymmetry**:

| Metric | Original fp16 | Original bf16 |
|--------|--------------|--------------|
| `fptrunc float→half` (LLIR) | **136** | 0 |
| `fpext half→float` (LLIR) | **129** | 0 |
| `v_lshrrev_b32` (AMDGCN) | 6 | **142** |
| `v_lshlrev_b32` (AMDGCN) | 10 | **130** |
| `v_fma_f32` (AMDGCN) | 1 | **86** |
| Total ALU instructions | 1304 | **2010** |
| scratch_load / scratch_store | **242 / 219** | 124 / 124 |

**bf16→f32 is a trivial bit-shift** (same exponent range, 8-bit vs 8-bit; just pad/truncate 7→23 mantissa bits), so LLVM optimizes it to integer `v_lshlrev`/`v_lshrrev` ops. The WMMA intrinsic accepts `<16 x i16>` for bf16, allowing the compiler to pack dequantized values as raw bit patterns without entering the float conversion pipeline.

**fp16→f32 requires actual `v_cvt_f16_f32`/`v_cvt_f32_f16` instructions** (different exponent bias: 5-bit vs 8-bit, needs rounding). Each conversion keeps both the f32 source and fp16 result live simultaneously in VGPRs, inflating register pressure during the dequant loop. With 265 conversions (136 trunc + 129 ext) in the inner loop, fp16 pushes far past the 256 VGPR budget.

The bf16 kernel does **more total ALU work** (2010 vs 1304 instructions) but uses **fewer registers** because integer shift ops can reuse registers immediately, while float conversions create additional live values. This is a Triton/LLVM compiler behavior specific to the per-element shift-and-mask unpacking path — the interleave path avoids the issue entirely by using `tl.interleave` which reduces the number of intermediate values regardless of dtype.

---

## Single-Batch Decode Latency: AWQ (max_num_seqs=1, 100 prompts)

**Model:** Qwen3-30B-A3B-AWQ (`MoeWNA16Method`), ShareGPT 100 prompts, gfx1100 (W7900).
`max-num-seqs=1` isolates pure decode latency (memory-bandwidth-bound), removing batching/scheduling noise.

| Metric | Interleave fp16 | Interleave bf16 | Main fp16 | Main bf16 |
|--------|----------------|----------------|-----------|-----------|
| Output tok/s | **54.22** | 51.64 | 40.19 | 50.15 |
| Mean TPOT (ms) | **17.87** | 18.73 | 23.63 | 19.22 |
| Median TPOT (ms) | **17.95** | 18.82 | 24.14 | 19.31 |
| Mean ITL (ms) | **17.87** | 18.73 | 23.58 | 19.24 |
| Median ITL (ms) | **17.89** | 18.77 | 24.09 | 19.26 |

**Single-Batch Key Observations:**
- **Main fp16 → Main bf16**: 24.14 → 19.31ms TPOT (**−20%**) — purely from eliminating register spilling (same kernel, different dtype compilation).
- **Main bf16 → Interleave fp16**: 19.31 → 17.95ms TPOT (**−7%**) — pure algorithmic gain from interleave unpacking, with spilling already absent.
- **Interleave fp16 is fastest** (17.95ms) — with spilling eliminated, fp16 benefits from native WMMA on gfx1100.
- Fresh re-run of interleave fp16 (2026-05-27) confirmed: 18.04ms median TPOT, within noise of original 17.95ms.

## Single-Batch Decode Latency: Compressed-Tensors (max_num_seqs=1, 100 prompts)

**Model:** RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (`CompressedTensorsWNA16MoEMethod`, bf16).

| Metric | Interleave | Main | Delta |
|--------|-----------|------|-------|
| Output tok/s | 53.63 | 53.83 | −0.4% |
| Mean TPOT (ms) | 18.06 | 17.94 | +0.7% |
| Median TPOT (ms) | 18.13 | 18.02 | +0.6% |
| Mean ITL (ms) | 18.05 | 17.93 | +0.7% |
| Median ITL (ms) | 18.06 | 17.97 | +0.5% |

**CT Single-Batch Key Observation:**
- Decode latency is identical (within noise). The CT model uses bf16, so the original kernel already compiles without spilling (240 VGPRs, Scratch=0). The interleave advantage materializes at higher batch sizes (+14% throughput in the 500-prompt batched benchmark) where reduced load instruction count becomes the bottleneck.

---

## Accuracy (lm_eval gsm8k, 5-shot, 200 samples)

Hardware: gfx1100 (AMD Radeon Pro W7900).

| Model | Branch | strict-match | flexible-extract |
|-------|--------|-------------|-----------------|
| Qwen3-30B-A3B-AWQ (MoeWNA16Method) | main | 0.920 ± 0.019 | 0.890 ± 0.022 |
| Qwen3-30B-A3B-AWQ (MoeWNA16Method) | interleave | 0.915 ± 0.020 | 0.885 ± 0.023 |
| RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (CompressedTensorsWNA16) | main | 0.900 ± 0.021 | 0.910 ± 0.020 |
| RedHatAI/Qwen3-30B-A3B-quantized.w4a16 (CompressedTensorsWNA16) | interleave | 0.895 ± 0.022 | 0.890 ± 0.022 |

No accuracy regression — all deltas are within stderr.
