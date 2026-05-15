# MoE int4 Kernel Benchmark: Interleave vs ExLlama Shuffle

Comparison of two int4 weight unpacking strategies for the fused MoE Triton kernel on ROCm.

- **Interleave (N-packed)**: Weights `[E, K, N//8]` int32, 3x `tl.interleave` unpacking, GPTQ sequential shifts `[0,4,...,28]`
- **ExLlama (K-packed)**: Weights `[E, N, K//8]` int32, nibble order `[0,2,4,6,1,3,5,7]`, requires `tl.trans()` in kernel

**GPU**: AMD Radeon Pro W7900 Dual Slot (gfx1100)

## End-to-End Results

`vllm bench throughput`, ShareGPT 1000 prompts, `--max-model-len 4096`, `--gpu-memory-utilization 0.8`:

### Qwen3.6-35B-A3B-AWQ (E=256, hidden=2048, inter=512, top_k=8)

| Branch | req/s | Speedup vs Main |
|--------|------:|:---------------:|
| main (upstream, no int4 Triton) | 1.20 | 1.00x |
| **interleave (N-packed)** | **2.58** | **2.15x** |
| exllama (K-packed) | 2.25 | 1.88x |

Interleave is **1.15x faster** than ExLlama end-to-end.

### Qwen3-30B-A3B-AWQ (E=128, hidden=2048, inter=768, top_k=8)

| Branch | req/s | tok/s | Speedup vs Main |
|--------|------:|------:|:---------------:|
| main (upstream, no int4 Triton) | 2.14 | 897.83 | 1.00x |
| **interleave (N-packed)** | **6.11** | **2562.38** | **2.86x** |
| exllama (K-packed) | 5.43 | 2275.94 | 2.54x |

Interleave is **1.13x faster** than ExLlama end-to-end.

### Qwen3-30B-A3B-AWQ — Single-Sequence Decode (max_num_seqs=1)

`vllm serve` + `vllm bench serve`, ShareGPT 80 prompts:

| Metric | Interleave | ExLlama | Ratio |
|--------|----------:|--------:|:-----:|
| req/s | 0.26 | 0.23 | 1.13x |
| Output tok/s | 53.69 | 47.31 | 1.13x |
| Mean TPOT (ms) | **17.40** | 19.97 | 0.87x |
| Mean ITL (ms) | **17.93** | 20.59 | 0.87x |
| P99 ITL (ms) | **18.99** | 21.44 | 0.89x |

Interleave is **13% faster** in single-sequence decode (pure M=1 regime).

## Kernel-Level Results

### Qwen3.6-35B-A3B-AWQ (E=256, hidden=2048, inter=512, top_k=8)

`fused_experts` (w1+SiLU+w2), w1=[E,1024,2048], w2=[E,2048,512]:

| M | Interleave (ms) | ExLlama (ms) | Ratio (Ex/Int) | Winner |
|----:|----------------:|--------------:|:--------------:|:------:|
| 1 | 0.193 | 0.193 | 1.00x | tie |
| 4 | 0.196 | 0.380 | 1.94x | Interleave |
| 16 | 0.501 | 1.441 | 2.88x | Interleave |
| 64 | 1.892 | 2.214 | 1.17x | Interleave |
| 256 | 2.203 | 2.586 | 1.17x | Interleave |
| 512 | 2.248 | 2.610 | 1.16x | Interleave |
| 1024 | 2.390 | 3.074 | 1.29x | Interleave |
| 2048 | 3.546 | 3.289 | 0.93x | ExLlama |
| 4096 | 5.928 | 5.216 | 0.88x | ExLlama |

### Qwen3-30B-A3B-AWQ (E=128, hidden=2048, inter=768, top_k=8)

`fused_experts` (w1+SiLU+w2), w1=[E,1536,2048], w2=[E,2048,768]:

| M | Interleave (ms) | ExLlama (ms) | Ratio (Ex/Int) | Winner |
|----:|----------------:|--------------:|:--------------:|:------:|
| 1 | 0.190 | 0.193 | 1.02x | tie |
| 2 | 0.190 | 0.258 | 1.36x | Interleave |
| 4 | 0.203 | 0.480 | 2.36x | Interleave |
| 8 | 0.376 | 0.970 | 2.58x | Interleave |
| 16 | 0.626 | 1.609 | 2.57x | Interleave |
| 32 | 0.951 | 1.645 | 1.73x | Interleave |
| 64 | 1.432 | 1.983 | 1.38x | Interleave |
| 128 | 1.627 | 2.195 | 1.35x | Interleave |
| 256 | 1.690 | 2.336 | 1.38x | Interleave |
| 512 | 1.760 | 2.359 | 1.34x | Interleave |
| 1024 | 2.578 | 2.483 | 0.96x | ExLlama |
| 2048 | 4.319 | 3.830 | 0.89x | ExLlama |
| 4096 | 7.811 | 6.836 | 0.88x | ExLlama |

## Analysis

- **Interleave wins at small M** (decode / small-batch, memory-bound): up to 2.9x faster on Qwen3.6, 2.6x on Qwen3-30B
- **ExLlama wins at large M >= 1024-2048** (prefill / large-batch, compute-bound): ~12% faster
- Crossover at M~2048 for Qwen3.6 (smaller inter=512), M~1024 for Qwen3-30B (larger inter=768)
- E2E with ShareGPT workload (decode-heavy), interleave is the clear winner

## Reproducing

```bash
# Interleave (on moe-triton-int4-interleave branch)
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/kernels/benchmark_moe_int4_interleave.py \
    --layout interleave --model qwen3-30b --warmup 10 --iters 100

# ExLlama (on exllama-shuffle-benchmark branch)
CUDA_VISIBLE_DEVICES=4 python3 benchmarks/kernels/benchmark_moe_int4_interleave.py \
    --layout exllama --model qwen3-30b --warmup 10 --iters 100
```
