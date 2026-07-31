# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid MLA decode dispatch (PoC, RDNA3/gfx1100).

Routes MLA decode to aiter's pure-Triton ``mla_decode_fwd`` at large batch and
to vLLM's in-tree kernel otherwise. The batch-size branch lives *inside* a
custom op so that torch.compile/Dynamo treats it as an opaque runtime call and
does not fold it into a static branch at trace/capture time (cf.
vllm-project/vllm#37494 — Dynamo evaluates Python-level conditions once at
warmup and bakes the result into the cached graph).

This is a proof-of-concept to measure the high-batch speedup end to end; it is
not upstream-ready (hard aiter dependency, dev-tree import fallback, no LSE from
the aiter branch so it must not be used with decode context parallelism).
"""

import importlib
import os
import sys
import types

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)

# Batch at/above which the aiter kernel is used. gfx1100 microbenchmark: aiter
# wins ~1.1-1.5x from B>=32 and ~3x from B>=128; keep the conservative default.
VLLM_MLA_AITER_MIN_BATCH = int(os.getenv("VLLM_MLA_AITER_MIN_BATCH", "128"))

_AITER_MLA_DECODE_FWD = None
_AITER_LOAD_ATTEMPTED = False


def _load_aiter_mla_decode_fwd():
    """Lazily import aiter's pure-Triton ``mla_decode_fwd`` (returns None if
    unavailable). Tries a normal import first; falls back to a stub package
    pointing at a source checkout so only the Triton submodule loads without
    running aiter's heavy ``__init__`` (which JIT-builds its C core)."""
    global _AITER_MLA_DECODE_FWD, _AITER_LOAD_ATTEMPTED
    if _AITER_LOAD_ATTEMPTED:
        return _AITER_MLA_DECODE_FWD
    _AITER_LOAD_ATTEMPTED = True
    try:
        try:
            from aiter.ops.triton.attention.mla import mla_decode_fwd
        except Exception:
            aiter_root = os.getenv("VLLM_AITER_ROOT", "/home/user/workspace/aiter")
            if "aiter" not in sys.modules:
                stub = types.ModuleType("aiter")
                stub.__path__ = [os.path.join(aiter_root, "aiter")]
                sys.modules["aiter"] = stub
                if aiter_root not in sys.path:
                    sys.path.insert(0, aiter_root)
            mod = importlib.import_module("aiter.ops.triton.attention.mla")
            mla_decode_fwd = mod.mla_decode_fwd
        _AITER_MLA_DECODE_FWD = mla_decode_fwd
        logger.info("Hybrid MLA: loaded aiter mla_decode_fwd (min_batch=%d)",
                    VLLM_MLA_AITER_MIN_BATCH)
    except Exception as e:
        logger.warning("Hybrid MLA: aiter unavailable, in-tree only (%s)", e)
        _AITER_MLA_DECODE_FWD = None
    return _AITER_MLA_DECODE_FWD


def _mla_decode_hybrid(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_kv_splits: int,
    scale: float,
    page_size: int,
    kv_lora_rank: int,
    max_seq_len: int,
    aiter_min_batch: int,
) -> None:
    B = q.shape[0]
    aiter_fn = _load_aiter_mla_decode_fwd()
    use_aiter = (
        aiter_fn is not None
        and B >= aiter_min_batch
        and kv_buffer.dtype in (torch.bfloat16, torch.float16)
    )
    if use_aiter:
        qk_rope_head_dim = q.shape[-1] - kv_lora_rank
        # One query token per sequence on the decode path.
        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=q.device)
        aiter_fn(
            q,
            kv_buffer,
            o,
            cu_seqlens_q,
            seq_lens,
            max_seq_len,
            block_table,
            scale,
            kv_lora_rank,
            qk_rope_head_dim,
            True,  # causal
            None,  # q_descale
            None,  # kv_descale
        )
        return

    kv_c_cache = kv_buffer[..., :kv_lora_rank]
    decode_attention_fwd(
        q,
        kv_buffer,
        kv_c_cache,
        o,
        lse,
        block_table,
        seq_lens,
        attn_logits,
        num_kv_splits,
        scale,
        page_size,
        k_scale=k_scale,
        v_scale=v_scale,
        is_mla=True,
    )


def _mla_decode_hybrid_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_kv_splits: int,
    scale: float,
    page_size: int,
    kv_lora_rank: int,
    max_seq_len: int,
    aiter_min_batch: int,
) -> None:
    return None


direct_register_custom_op(
    op_name="mla_decode_hybrid",
    op_func=_mla_decode_hybrid,
    mutates_args=["o", "lse", "attn_logits"],
    fake_impl=_mla_decode_hybrid_fake,
)


def mla_decode_hybrid(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    attn_logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_kv_splits: int,
    scale: float,
    page_size: int,
    kv_lora_rank: int,
    max_seq_len: int,
    aiter_min_batch: int = VLLM_MLA_AITER_MIN_BATCH,
) -> None:
    """Runtime-branched MLA decode; see module docstring. Writes into ``o``
    (and ``lse``/``attn_logits`` on the in-tree branch)."""
    torch.ops.vllm.mla_decode_hybrid(
        q,
        kv_buffer,
        o,
        lse,
        attn_logits,
        block_table,
        seq_lens,
        k_scale,
        v_scale,
        num_kv_splits,
        scale,
        page_size,
        kv_lora_rank,
        max_seq_len,
        aiter_min_batch,
    )
