# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid W4A16 kernel: Triton for prefill, HIP skinny for decode.

Routes based on batch size M:
  M <= SKINNY_THRESHOLD: HIP skinny GEMM (wvSplitK_int4/int4_g)
  M > SKINNY_THRESHOLD:  Triton W4A16 fused dequant GEMM

Stores weights in Triton layout [K, N//8] as primary (no dequantized copy),
plus a second packed copy in skinny layout [N, K//8] for the decode path.
Net memory: ~2x int4 weights (~3.6GB for Qwen3-4B) vs ~8.8GB for dequantized
fp16 + int4 in the skinny-only kernel.
"""

from contextlib import nullcontext

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig
from .triton_w4a16 import (
    TRITON_W4A16_SUPPORTED_GROUP_SIZES,
    triton_w4a16_gemm,
)

# Match the skinny kernel's threshold
SKINNY_THRESHOLD = 4
LDS_CAPACITY_ELEMENTS = 64 * 1024 // 2  # 32768 fp16 elements


def _hybrid_w4a16_apply_impl(
    x_2d: torch.Tensor,
    w_q_skinny: torch.Tensor,
    w_s_skinny: torch.Tensor,
    w_q_triton: torch.Tensor,
    w_s_triton: torch.Tensor,
    w_zp_triton: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    """Dispatch between skinny GEMM and Triton based on batch size M.

    Registered as a custom op so torch.compile treats it as opaque,
    avoiding issues with the data-dependent branch.
    """
    import vllm._custom_ops as ops

    N = x_2d.shape[0]
    K = x_2d.shape[1]

    if N <= SKINNY_THRESHOLD and K * N <= LDS_CAPACITY_ELEMENTS:
        if group_size > 0:
            return ops.wvSplitK_int4_g(
                w_q_skinny, x_2d, w_s_skinny, cu_count, group_size, bias
            )
        else:
            return ops.wvSplitK_int4(w_q_skinny, x_2d, w_s_skinny, cu_count, bias)

    output = triton_w4a16_gemm(
        a=x_2d,
        b_q=w_q_triton,
        scales=w_s_triton,
        qzeros=w_zp_triton,
        group_size=group_size if group_size > 0 else K,
        zp_bias=zp_bias,
    )
    if bias is not None:
        output.add_(bias)
    return output


def _hybrid_w4a16_apply_fake(
    x_2d: torch.Tensor,
    w_q_skinny: torch.Tensor,
    w_s_skinny: torch.Tensor,
    w_q_triton: torch.Tensor,
    w_s_triton: torch.Tensor,
    w_zp_triton: torch.Tensor | None,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
    zp_bias: int,
) -> torch.Tensor:
    N = x_2d.size(0)
    M = w_q_skinny.size(0)
    return torch.empty((N, M), dtype=x_2d.dtype, device=x_2d.device)


def _register_hybrid_w4a16_op():
    lib = torch.library.Library("_rocm_hybrid", "DEF")
    lib.define(
        "w4a16_apply(Tensor x_2d, Tensor w_q_skinny, Tensor w_s_skinny,"
        " Tensor w_q_triton, Tensor w_s_triton, Tensor? w_zp_triton,"
        " Tensor? bias, int cu_count, int group_size, int zp_bias) -> Tensor"
    )
    lib.impl("w4a16_apply", _hybrid_w4a16_apply_impl, "CUDA")
    lib.impl("w4a16_apply", _hybrid_w4a16_apply_fake, "Meta")
    return lib


_HYBRID_W4A16_LIB = _register_hybrid_w4a16_op()


class HybridW4A16LinearKernel(MPLinearKernel):
    """Hybrid W4A16 kernel: HIP skinny for decode, Triton for prefill.

    Stores weights in both Triton layout [K, N//8] and skinny layout [N, K//8]
    (both int4 packed, ~2x int4 memory) to avoid the large dequantized fp16
    copy that the skinny-only kernel requires.
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

        # Get raw weight parameters before any transformation
        w_q_raw = getattr(layer, self.w_q_name)
        w_s_raw = getattr(layer, self.w_s_name)

        # ---- Unpack raw weights to [N, K] int32 ----
        unpacked = unpack_quantized_values_into_int32(
            w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
        )
        # At this point, unpacked has input_dim and output_dim from the
        # checkpoint. We need to get it to [N, K] layout.
        # The checkpoint has weight_packed with input_dim=1, output_dim=0,
        # packed_dim=1. After unpacking, shape is [N, K].

        # ---- Prepare Triton weights: [K, N//8] int32 ----
        # Transpose to [K, N], then repack N into N//8
        w_KN = unpacked.t().contiguous()  # [K, N]
        N_dim = w_KN.shape[1]
        K_dim = w_KN.shape[0]
        shifts = torch.arange(8, device=w_KN.device, dtype=torch.int32) * 4
        N8 = N_dim // 8
        w_q_triton = torch.sum(
            (w_KN.view(K_dim, N8, 8) & 0xF) << shifts,
            dim=2,
            dtype=torch.int32,
        ).contiguous()

        # ---- Prepare Triton scales: [K//G, N] ----
        # Checkpoint scales are [N, K//G] (output_dim=0, input_dim=1)
        w_s_triton = w_s_raw.data.t().contiguous()

        # ---- Prepare skinny weights: [N, K//8] repacked ----
        if c.act_type == torch.float16:
            # ExLlama shuffle: group 8 values, interleave even/odd
            unsigned = unpacked.to(torch.uint8)  # [N, K]
            M_dim, K_dim2 = unsigned.shape
            g = unsigned.view(M_dim, K_dim2 // 8, 8).to(torch.int32)
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
            w_q_skinny = shuffled.contiguous().view(torch.int8).contiguous()
        else:
            # bf16: simple nibble packing
            bias_val = c.weight_type.bias
            signed = (unpacked - bias_val).to(torch.int8)  # [N, K]
            low = signed[:, 0::2] & 0xF
            high = signed[:, 1::2] & 0xF
            w_q_skinny = (low | (high << 4)).to(torch.uint8)
            w_q_skinny = w_q_skinny.view(torch.int8).contiguous()

        # ---- Prepare skinny scales ----
        w_s_skinny = w_s_raw.data
        if c.group_size == -1:
            w_s_skinny = w_s_skinny.squeeze(-1)
        w_s_skinny = w_s_skinny.contiguous()

        # ---- Store Triton weights on the layer ----
        # Replace w_q with Triton layout (primary weights, saves memory)
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(w_q_triton, requires_grad=False),
        )
        # Replace w_s with Triton layout
        replace_parameter(
            layer,
            self.w_s_name,
            torch.nn.Parameter(w_s_triton, requires_grad=False),
        )

        # ---- Store skinny weights as extra parameters ----
        layer.register_parameter(
            "_hybrid_w_q_skinny",
            torch.nn.Parameter(w_q_skinny, requires_grad=False),
        )
        layer.register_parameter(
            "_hybrid_w_s_skinny",
            torch.nn.Parameter(w_s_skinny, requires_grad=False),
        )

        # Handle Triton zero points (for symmetric, we don't store any)
        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
                )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        c = self.config
        w_q_triton, w_s_triton, w_zp_triton, _ = self._get_weight_params(layer)
        w_q_skinny = layer._hybrid_w_q_skinny
        w_s_skinny = layer._hybrid_w_s_skinny

        x_2d = x.reshape(-1, x.shape[-1])
        N = x_2d.shape[0]
        K = x_2d.shape[1]
        M_out = w_q_skinny.shape[0]
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
                w_q_skinny,
                w_s_skinny,
                w_q_triton,
                w_s_triton,
                w_zp_triton,
                bias,
                cu_count,
                c.group_size,
                zp_bias,
            )
        return output.reshape(out_shape)
