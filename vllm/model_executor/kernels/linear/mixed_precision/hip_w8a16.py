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


class HipW8A16LinearKernel(MPLinearKernel):
    """W8A16 per-channel int8 skinny GEMM for ROCm (gfx11).

    Uses the wvSplitK_int8 kernel for small batch sizes where activations
    fit in LDS. Falls back to dequant + torch.linear for larger batches.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_int8"
            ):
                return False, "wvSplitK_int8 op not available in this build"
        except Exception:
            return False, "ROCm ops not available"

        if c.weight_type.is_floating_point() or c.weight_type.size_bits != 8:
            return False, "requires 8-bit integer weights"

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.group_size != -1:
            return False, "requires per-channel quantization (group_size=-1)"

        if c.zero_points:
            return False, "does not support zero points (asymmetric)"

        if c.has_g_idx:
            return False, "does not support g_idx reordering"

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        def transform_w_q(x: BasevLLMParameter) -> torch.Tensor:
            unpacked = unpack_quantized_values_into_int32(
                x.data, c.weight_type, packed_dim=x.packed_dim
            )
            bias_val = c.weight_type.bias
            return (unpacked - bias_val).to(torch.int8).contiguous()

        def transform_w_s(x: BasevLLMParameter) -> torch.Tensor:
            return x.data.squeeze(-1).contiguous()

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
            else torch.profiler.record_function(f"hip_w8a16 {N}x{M}x{K}")
        )
        with ctx:
            if K * N <= LDS_CAPACITY_ELEMENTS:
                cu_count = num_compute_units()
                output = ops.wvSplitK_int8(w_q, x_2d, w_s, cu_count, bias)
            else:
                raise AssertionError("hip_w8a16 does not support large batch sizes")
                w_dequant = w_q.to(x.dtype) * w_s.unsqueeze(1)
                output = torch.nn.functional.linear(x_2d, w_dequant, bias)

        return output.reshape(out_shape)
