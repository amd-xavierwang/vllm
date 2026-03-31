# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    permute_param_layout_,
)

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements

SKINNY_GEMM_MAX_N = 5


def _w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    w_dequant: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM kernel and dequant fallback.

    Registered as a custom op so torch.compile treats it as opaque,
    avoiding issues with the data-dependent N<=4 branch.
    """
    import vllm._custom_ops as ops

    N = x_2d.shape[0]
    K = x_2d.shape[1]

    if N <= SKINNY_GEMM_MAX_N and K * N <= LDS_CAPACITY_ELEMENTS:
        if w_zp is not None and group_size > 0:
            return ops.wvSplitK_int4_g_zp(
                w_q, x_2d, w_s, w_zp, cu_count, group_size, bias
            )
        elif group_size > 0:
            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, bias)
        else:
            return ops.wvSplitK_int4(w_q, x_2d, w_s, cu_count, bias)

    assert w_dequant is not None, (
        "w_dequant must be pre-computed for large-batch fallback"
    )
    return torch.nn.functional.linear(x_2d, w_dequant, bias)


def _w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    w_dequant: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    N = x_2d.size(0)
    M = w_q.size(0)
    return torch.empty((N, M), dtype=x_2d.dtype, device=x_2d.device)


def _register_w4a16_op():
    lib = torch.library.Library("_rocm_skinny", "DEF")
    lib.define(
        "w4a16_apply(Tensor x_2d, Tensor w_q, Tensor w_s, Tensor? w_zp,"
        " Tensor? w_dequant, Tensor? bias, int cu_count, int group_size)"
        " -> Tensor"
    )
    lib.impl("w4a16_apply", _w4a16_apply_impl, "CUDA")
    lib.impl("w4a16_apply", _w4a16_apply_fake, "Meta")
    return lib


_W4A16_LIB = _register_w4a16_op()


class HipW4A16SkinnyLinearKernel(MPLinearKernel):
    """W4A16 skinny GEMM for ROCm (gfx11) using packed int4 weights.

    Supports both per-channel (group_size=-1) and per-group (group_size=32/128)
    quantization, including symmetric (uint4b8) and asymmetric (uint4 with zero
    points). Uses wvSplitK_int4/wvSplitK_int4_g/wvSplitK_int4_g_zp for small
    batch sizes where activations fit in LDS, with dequant+linear fallback for
    larger batches. Wrapped as a custom op to avoid torch.compile issues with
    the data-dependent N<=4 branch.
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

        if c.zero_points and c.group_size == -1:
            return False, "zero points require per-group quantization"

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
        K = c.partition_weight_shape[0]
        M = c.partition_weight_shape[1]  # output features

        # Dequantize from the raw unpacked representation *before* repacking
        # into the kernel-specific layout. This is simpler than reversing the
        # ExLlama shuffle / packed-byte format after transform_param.
        w_q_raw, w_s_raw, _, _ = self._get_weight_params(layer)
        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # Normalize to (M, K) regardless of source layout.
        # AWQ-converted weights arrive as (K, N) with output_dim=1,
        # compressed-tensors arrive as (N, K) with output_dim=0.
        if getattr(w_q_raw, "output_dim", 0) != 0:
            unpacked = unpacked.t().contiguous()

        # Normalize scales to (M, num_groups) or (M, 1).
        w_s_raw_data = w_s_raw.data
        if getattr(w_s_raw, "output_dim", 0) != 0:
            w_s_raw_data = w_s_raw_data.t().contiguous()
        w_s_clean = w_s_raw_data
        if c.group_size == -1:
            w_s_clean = w_s_clean.squeeze(-1)
        w_s_clean = w_s_clean.contiguous()

        # Process zero points for asymmetric quantization
        self._w_zp = None
        if c.zero_points:
            assert self.w_zp_name is not None
            w_zp_raw = getattr(layer, self.w_zp_name)
            # Normalize zp layout to (M, num_groups) via permute_param_layout_
            permute_param_layout_(w_zp_raw, input_dim=1, output_dim=0, packed_dim=0)
            zp_unpacked = unpack_quantized_values_into_int32(
                w_zp_raw.data, c.weight_type, packed_dim=0
            )
            # zp_unpacked: [M, num_groups]
            num_groups = K // c.group_size

            # The kernel dequant always produces (nibble - 8). To get
            # (nibble - zp_raw), we subtract (zp_raw - 8) after dequant.
            self._w_zp = (zp_unpacked - 8).to(c.act_type).contiguous()

            # Build dequant fallback with zero-point subtraction
            zp_expanded = zp_unpacked.to(c.act_type)
            zp_expanded = zp_expanded.unsqueeze(-1).expand(-1, num_groups, c.group_size)
            w_fp = unpacked.to(c.act_type).view(M, num_groups, c.group_size)
            w_fp = (w_fp - zp_expanded) * w_s_clean.unsqueeze(-1)
            w_fp = w_fp.view(M, K)
        else:
            bias_val = c.weight_type.bias
            w_fp = (unpacked - bias_val).to(c.act_type)

            if c.group_size > 0:
                num_groups = K // c.group_size
                w_fp = w_fp.view(M, num_groups, c.group_size)
                w_fp = (w_fp * w_s_clean.unsqueeze(-1)).view(M, K)
            else:
                w_fp = w_fp * w_s_clean.unsqueeze(1)

        self._w_dequant = w_fp.contiguous()

        # Now repack w_q / w_s into the format the skinny GEMM kernel expects.
        def transform_w_q(x: BasevLLMParameter) -> torch.Tensor:
            unpacked = unpack_quantized_values_into_int32(
                x.data, c.weight_type, packed_dim=x.packed_dim
            )
            # Normalize to (M, K)
            if getattr(x, "output_dim", 0) != 0:
                unpacked = unpacked.t().contiguous()
            # Both fp16 and bf16 kernels use ExLlama shuffle format:
            # 8 unsigned int4 values packed per uint32 as
            # [v0,v2,v4,v6] in low 16 bits, [v1,v3,v5,v7] in high.
            # The kernel subtracts 8 from each nibble during dequant.
            unsigned = unpacked.to(torch.uint8)
            M, K_dim = unsigned.shape
            g = unsigned.view(M, K_dim // 8, 8).to(torch.int32)
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

        def transform_w_s(x: BasevLLMParameter) -> torch.Tensor:
            data = x.data
            # Normalize to (M, num_groups)
            if getattr(x, "output_dim", 0) != 0:
                data = data.t().contiguous()
            if c.group_size == -1:
                return data.squeeze(-1).contiguous()
            return data.contiguous()

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            cu_count = num_compute_units()
            output = torch.ops._rocm_skinny.w4a16_apply(
                x_2d,
                w_q,
                w_s,
                self._w_zp,
                self._w_dequant,
                bias,
                cu_count,
                self.config.group_size,
            )
        return output.reshape(out_shape)
