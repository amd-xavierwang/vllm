# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid W4A16 kernel: Triton for prefill, HIP skinny for decode.

Routes based on batch size M:
  M <= SKINNY_THRESHOLD: HIP skinny GEMM (wvSplitK_int4/int4_g)
  M > SKINNY_THRESHOLD:  Triton W4A16 fused dequant GEMM

Stores weights ONCE in skinny layout [N, K//8] int32 (ExLlama shuffle).
Both the HIP skinny kernel and the triton kernel read from this single
weight copy. The triton kernel transposes tiles in-register.
"""

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from .triton_w4a16 import (
    TRITON_W4A16_SUPPORTED_GROUP_SIZES,
    triton_w4a16_skinny_fmt_gemm,
)

# Match the skinny kernel's threshold
SKINNY_THRESHOLD = 4
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


def _hybrid_w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_q_i32: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and Triton based on batch size M.

    Both paths read from the same skinny-format weights:
      w_q:     [N, K//8] int8 (ExLlama shuffle, for skinny kernel)
      w_q_i32: [N, K//8] int32 (same data viewed as int32, for triton kernel)
      w_s:     [N, K//G] fp16 (skinny-layout scales)

    Registered as a custom op so torch.compile treats it as opaque.
    """
    import vllm._custom_ops as ops

    N = x_2d.shape[0]
    K = x_2d.shape[1]

    if N <= SKINNY_THRESHOLD and K * N <= LDS_CAPACITY_ELEMENTS:
        if group_size > 0:
            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, bias)
        else:
            return ops.wvSplitK_int4(w_q, x_2d, w_s, cu_count, bias)

    output = triton_w4a16_skinny_fmt_gemm(
        a=x_2d,
        b_q=w_q_i32,
        scales=w_s,
        group_size=group_size if group_size > 0 else K,
        zp_bias=zp_bias,
    )
    if bias is not None:
        output.add_(bias)
    return output


def _hybrid_w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_q_i32: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    N = x_2d.size(0)
    M = w_q.size(0)
    return torch.empty((N, M), dtype=x_2d.dtype, device=x_2d.device)


def _register_hybrid_w4a16_op():
    lib = torch.library.Library("_rocm_hybrid", "DEF")
    lib.define(
        "w4a16_apply(Tensor x_2d, Tensor w_q, Tensor w_s,"
        " Tensor w_q_i32, Tensor? bias,"
        " int cu_count, int group_size, int zp_bias) -> Tensor"
    )
    lib.impl("w4a16_apply", _hybrid_w4a16_apply_impl, "CUDA")
    lib.impl("w4a16_apply", _hybrid_w4a16_apply_fake, "Meta")
    return lib


_HYBRID_W4A16_LIB = _register_hybrid_w4a16_op()


class HybridW4A16LinearKernel(MPLinearKernel):
    """Hybrid W4A16 kernel: HIP skinny for decode, Triton for prefill.

    Stores weights once in skinny layout [N, K//8] (ExLlama shuffle packed).
    Both the HIP skinny kernel and the triton kernel read from this single
    weight copy, eliminating the memory overhead of dual weight storage.
    """

    SUPPORTED_QUANT_TYPES = [
        scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
    ]

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "HybridW4A16LinearKernel only targets ROCm"

        # Check HIP skinny op availability
        try:
            if not hasattr(torch.ops, "_rocm_C") or not hasattr(
                torch.ops._rocm_C, "wvSplitK_int4"
            ):
                return False, "wvSplitK_int4 op not available in this build"
        except Exception:
            return False, "ROCm ops not available"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "requires float16 or bfloat16 activations"

        if c.zero_points:
            return False, "does not support zero points (asymmetric)"

        if c.has_g_idx:
            return False, "does not support g_idx reordering"

        gs = c.group_size
        if (
            gs not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W4A16_SUPPORTED_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        if K % 16 != 0:
            return False, f"K={K} must be divisible by 16"

        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return False, f"K={K} not divisible by group_size={eff_gs}"

        if c.group_size > 0 and K % c.group_size != 0:
            return (
                False,
                f"K={K} must be divisible by group_size={c.group_size}",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        w_q_raw = getattr(layer, self.w_q_name)
        w_s_raw = getattr(layer, self.w_s_name)

        # Unpack raw weights to [N, K] int32
        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )

        # ---- Pack into skinny format: [N, K//8] ExLlama shuffle ----
        unsigned = unpacked.to(torch.uint8)  # [N, K]
        N_dim, K_dim = unsigned.shape
        g = unsigned.view(N_dim, K_dim // 8, 8).to(torch.int32)
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

        # Store as int8 for skinny kernel, keep int32 view for triton kernel
        w_q_skinny = shuffled.contiguous().view(torch.int8).contiguous()
        w_q_skinny_i32 = shuffled.contiguous()

        # ---- Prepare skinny scales: [N, K//G] ----
        w_s_skinny = w_s_raw.data
        if c.group_size == -1:
            w_s_skinny = w_s_skinny.squeeze(-1)
        w_s_skinny = w_s_skinny.contiguous()

        # ---- Store on layer ----
        # Replace w_q with skinny int8 (primary weights for skinny kernel)
        self._transform_param(layer, self.w_q_name, lambda x: w_q_skinny)
        # Replace w_s with skinny scales
        self._transform_param(layer, self.w_s_name, lambda x: w_s_skinny)

        # Store int32 view for triton kernel
        layer.register_parameter(
            "_hybrid_w_q_i32",
            torch.nn.Parameter(w_q_skinny_i32, requires_grad=False),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)
        w_q_i32 = layer._hybrid_w_q_i32

        x_2d = x.reshape(-1, x.shape[-1])
        N = x_2d.shape[0]
        K = x_2d.shape[1]
        M_out = w_q.shape[0]
        out_shape = x.shape[:-1] + (M_out,)

        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"hybrid_w4a16 {N}x{M_out}x{K}")
        )
        with ctx:
            cu_count = num_compute_units()
            output = torch.ops._rocm_hybrid.w4a16_apply(
                x_2d,
                w_q,
                w_s,
                w_q_i32,
                bias,
                cu_count,
                c.group_size,
                zp_bias,
            )
        return output.reshape(out_shape)
