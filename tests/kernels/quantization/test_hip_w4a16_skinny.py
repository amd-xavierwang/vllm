# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for HipW4A16SkinnyLinearKernel (symmetric and asymmetric paths)."""

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.kernels.linear.mixed_precision.hip_w4a16_skinny import (  # noqa: E501
    HipW4A16SkinnyLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E501
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType, scalar_types
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not is_quant_method_supported("awq"),
    reason="HipW4A16SkinnyLinearKernel requires ROCm",
)


def _ensure_single_process_model_parallel() -> None:
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )

    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file:///tmp/vllm_test_dist_skinny",
            backend="gloo",
        )
    if not model_parallel_is_initialized():
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)


def _reference_output_asymmetric(
    w_q_packed: torch.Tensor,
    w_zp_packed: torch.Tensor,
    w_s_data: torch.Tensor,
    x: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    weight_type: ScalarType,
) -> torch.Tensor:
    """Reference dequant: (w - zp) * scale, then matmul."""
    weights = unpack_quantized_values_into_int32(
        w_q_packed, weight_type, packed_dim=1
    ).to(torch.float32)
    # weights: [n, k]

    zeros = unpack_quantized_values_into_int32(
        w_zp_packed, weight_type, packed_dim=0
    ).to(torch.float32)
    # zeros: [n, num_groups] after unpacking packed_dim=0

    scales = w_s_data.to(torch.float32)
    # scales: [n, num_groups]

    zeros_exp = zeros.repeat_interleave(group_size, dim=1)  # [n, k]
    scales_exp = scales.repeat_interleave(group_size, dim=1)  # [n, k]
    dequant = (weights - zeros_exp) * scales_exp  # [n, k]
    return x.to(torch.float32) @ dequant.t()  # [batch, n]


def _reference_output_symmetric(
    w_q_packed: torch.Tensor,
    w_s_data: torch.Tensor,
    x: torch.Tensor,
    n: int,
    k: int,
    group_size: int,
    weight_type: ScalarType,
) -> torch.Tensor:
    """Reference dequant: (w - bias) * scale, then matmul."""
    weights = unpack_quantized_values_into_int32(
        w_q_packed, weight_type, packed_dim=1
    ).to(torch.float32)

    bias_val = weight_type.bias
    w_fp = weights - bias_val

    scales = w_s_data.to(torch.float32)
    if group_size > 0:
        num_groups = k // group_size
        w_fp = w_fp.view(n, num_groups, group_size)
        dequant = (w_fp * scales.unsqueeze(-1)).view(n, k)
    else:
        dequant = w_fp * scales

    return x.to(torch.float32) @ dequant.t()


# ============================================================
# Asymmetric (zero_points=True) tests
# ============================================================


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("group_size", "n", "k"),
    [
        (32, 256, 32 * 4),
        (32, 2048, 32 * 16),
        (128, 256, 128 * 2),
        (128, 2048, 128 * 7),
        (128, 2048, 128 * 16),
    ],
)
@pytest.mark.parametrize("act_dtype", [torch.float16])
def test_skinny_asymmetric_correctness(
    group_size: int,
    n: int,
    k: int,
    batch_size: int,
    act_dtype: torch.dtype,
    random_seed: int,
) -> None:
    _ensure_single_process_model_parallel()
    pack_factor = 8
    set_random_seed(random_seed)

    config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4,
        act_type=act_dtype,
        group_size=group_size,
        zero_points=True,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16SkinnyLinearKernel.can_implement(config)
    assert ok, err

    w_q_packed = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (n, k // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    w_zp_packed = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (n // pack_factor, k // group_size),
        dtype=torch.int32,
        device="cuda",
    )
    w_s_data = torch.rand((n, k // group_size), dtype=act_dtype, device="cuda")
    x = torch.rand((batch_size, k), dtype=act_dtype, device="cuda")

    layer = torch.nn.Module()
    weight_loader = lambda *_args, **_kwargs: None

    w_q = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_q_packed,
    )
    w_s = GroupQuantScaleParameter(
        output_dim=0,
        input_dim=1,
        weight_loader=weight_loader,
        data=w_s_data,
    )
    w_zp = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=0,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_zp_packed,
    )

    layer.register_parameter("weight_packed", w_q)
    layer.register_parameter("weight_scale", w_s)
    layer.register_parameter("weight_zero_point", w_zp)

    kernel = HipW4A16SkinnyLinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point",
    )
    kernel.process_weights_after_loading(layer)

    y = kernel.apply_weights(layer, x)

    y_ref = _reference_output_asymmetric(
        w_q_packed=w_q_packed,
        w_zp_packed=w_zp_packed,
        w_s_data=w_s_data,
        x=x,
        n=n,
        k=k,
        group_size=group_size,
        weight_type=scalar_types.uint4,
    ).to(y.dtype)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=2e-1)


# ============================================================
# Symmetric (zero_points=False) tests
# ============================================================


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize(
    ("group_size", "n", "k"),
    [
        (32, 256, 32 * 4),
        (32, 2048, 32 * 16),
        (128, 256, 128 * 2),
        (128, 2048, 128 * 7),
    ],
)
@pytest.mark.parametrize("act_dtype", [torch.float16])
def test_skinny_symmetric_correctness(
    group_size: int,
    n: int,
    k: int,
    batch_size: int,
    act_dtype: torch.dtype,
    random_seed: int,
) -> None:
    _ensure_single_process_model_parallel()
    pack_factor = 8
    set_random_seed(random_seed)

    config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4b8,
        act_type=act_dtype,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
        out_type=None,
    )
    ok, err = HipW4A16SkinnyLinearKernel.can_implement(config)
    assert ok, err

    w_q_packed = torch.randint(
        0,
        torch.iinfo(torch.int32).max,
        (n, k // pack_factor),
        dtype=torch.int32,
        device="cuda",
    )
    if group_size > 0:
        w_s_data = torch.rand((n, k // group_size), dtype=act_dtype, device="cuda")
    else:
        w_s_data = torch.rand((n, 1), dtype=act_dtype, device="cuda")
    x = torch.rand((batch_size, k), dtype=act_dtype, device="cuda")

    layer = torch.nn.Module()
    weight_loader = lambda *_args, **_kwargs: None

    w_q = PackedvLLMParameter(
        input_dim=1,
        output_dim=0,
        packed_dim=1,
        packed_factor=pack_factor,
        weight_loader=weight_loader,
        data=w_q_packed,
    )
    w_s = GroupQuantScaleParameter(
        output_dim=0,
        input_dim=1,
        weight_loader=weight_loader,
        data=w_s_data,
    )

    layer.register_parameter("weight_packed", w_q)
    layer.register_parameter("weight_scale", w_s)

    kernel = HipW4A16SkinnyLinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
    )
    kernel.process_weights_after_loading(layer)

    y = kernel.apply_weights(layer, x)

    y_ref = _reference_output_symmetric(
        w_q_packed=w_q_packed,
        w_s_data=w_s_data,
        x=x,
        n=n,
        k=k,
        group_size=group_size,
        weight_type=scalar_types.uint4b8,
    ).to(y.dtype)

    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=2e-1)
