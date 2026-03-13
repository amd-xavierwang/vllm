# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


class HipW4A16SkinnyLinearKernel(MPLinearKernel):
    """W4A16 skinny GEMM for ROCm (gfx11) using packed int4 weights.

    Supports both per-channel (group_size=-1) and per-group (group_size=32/128)
    symmetric quantization. Uses wvSplitK_int4/wvSplitK_int4_g for small batch
    sizes where activations fit in LDS.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_int4"
            ):
                return False, "wvSplitK_int4 op not available in this build"
        except Exception:
            return False, "ROCm ops not available"

        if c.weight_type.size_bits != 4:
            return False, "requires 4-bit weights"

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.group_size not in (-1, 32, 128):
            return False, f"group_size must be -1, 32, or 128 (got {c.group_size})"

        if c.zero_points:
            return False, "does not support zero points (asymmetric)"

        if c.has_g_idx:
            return False, "does not support g_idx reordering"

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        if c.group_size > 0 and K % c.group_size != 0:
            return (
                False,
                f"K={K} must be divisible by group_size={c.group_size}",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        def transform_w_q(x: BasevLLMParameter) -> torch.Tensor:
            unpacked = unpack_quantized_values_into_int32(
                x.data, c.weight_type, packed_dim=x.packed_dim
            )
            if c.act_type == torch.float16:
                # ExLlama shuffle: keep unsigned (0..15), reorder nibbles
                # within each group of 8 for fast bitwise fp16 dequant.
                # Layout per uint32: [k7 k5 k3 k1 | k6 k4 k2 k0] nibbles
                unsigned = unpacked.to(torch.uint8)
                M, K = unsigned.shape
                g = unsigned.view(M, K // 8, 8).to(torch.int32)
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
                return shuffled.contiguous().view(torch.int8).contiguous()
            else:
                bias_val = c.weight_type.bias
                signed = (unpacked - bias_val).to(torch.int8)
                M, K = signed.shape
                low = signed[:, 0::2] & 0xF
                high = signed[:, 1::2] & 0xF
                packed = (low | (high << 4)).to(torch.uint8)
                return packed.view(torch.int8).contiguous()

        def transform_w_s(x: BasevLLMParameter) -> torch.Tensor:
            if c.group_size == -1:
                return x.data.squeeze(-1).contiguous()
            return x.data.contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import vllm._custom_ops as ops
        from vllm.utils.platform_utils import num_compute_units

        w_q, w_s, _, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        N = x_2d.shape[0]
        K = x_2d.shape[1]
        M = w_q.shape[0]
        out_shape = x.shape[:-1] + (M,)

        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"hip_w4a16_skinny {N}x{M}x{K}")
        )
        with ctx:
            if N <= 4 and K * N <= LDS_CAPACITY_ELEMENTS:
                cu_count = num_compute_units()
                if self.config.group_size > 0:
                    output = ops.wvSplitK_int4_g(
                        w_q, x_2d, w_s, cu_count, self.config.group_size, bias
                    )
                else:
                    output = ops.wvSplitK_int4(w_q, x_2d, w_s, cu_count, bias)
            else:
                # Fall back to dequant + torch.linear for large batches
                K_logical = K
                if x.dtype == torch.float16:
                    # ExLlama shuffled format: unpack from uint32
                    u32 = w_q.contiguous().view(torch.int32)  # [M, K/8]
                    w_dequant = torch.empty(
                        M, K_logical, dtype=x.dtype, device=x.device
                    )
                    w_dequant[:, 0::8] = ((u32 >> 0) & 0xF).to(x.dtype) - 8
                    w_dequant[:, 2::8] = ((u32 >> 4) & 0xF).to(x.dtype) - 8
                    w_dequant[:, 4::8] = ((u32 >> 8) & 0xF).to(x.dtype) - 8
                    w_dequant[:, 6::8] = ((u32 >> 12) & 0xF).to(x.dtype) - 8
                    w_dequant[:, 1::8] = ((u32 >> 16) & 0xF).to(x.dtype) - 8
                    w_dequant[:, 3::8] = ((u32 >> 20) & 0xF).to(x.dtype) - 8
                    w_dequant[:, 5::8] = ((u32 >> 24) & 0xF).to(x.dtype) - 8
                    w_dequant[:, 7::8] = ((u32 >> 28) & 0xF).to(x.dtype) - 8
                else:
                    packed = w_q.view(torch.uint8)
                    low = ((packed & 0xF).to(torch.int8) << 4 >> 4).to(x.dtype)
                    high = (packed.to(torch.int8) >> 4).to(x.dtype)
                    w_dequant = torch.empty(
                        M, K_logical, dtype=x.dtype, device=x.device
                    )
                    w_dequant[:, 0::2] = low
                    w_dequant[:, 1::2] = high
                if self.config.group_size > 0:
                    num_groups = K_logical // self.config.group_size
                    w_dequant = w_dequant.view(M, num_groups, self.config.group_size)
                    w_dequant = (w_dequant * w_s.unsqueeze(-1)).view(M, K_logical)
                else:
                    w_dequant = w_dequant * w_s.unsqueeze(1)
                output = torch.nn.functional.linear(x_2d, w_dequant, bias)

        return output.reshape(out_shape)
