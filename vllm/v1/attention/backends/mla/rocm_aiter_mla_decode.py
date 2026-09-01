# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER Triton MLA decode backend for RDNA (gfx11/gfx12).

This backend reuses the in-tree :class:`TritonMLA` metadata builder and prefill
path unchanged, and only swaps the decode MQA kernel for aiter's
unified-attention-derived Triton kernel
(``aiter.ops.triton.attention.mla.mla_decode_fwd``), which is tuned for RDNA.

The aiter kernel does not return the log-sum-exp of the decode attention, so it
cannot feed decode context-parallel partial-output merging. Backend selection
(:func:`vllm.platforms.rocm._use_aiter_mla_decode`) therefore only offers this
backend when decode context parallelism is disabled, and
:attr:`can_return_lse_for_decode` is set ``False`` to make that explicit to the
MLA framework as well.
"""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLABackend,
    TritonMLAImpl,
)

logger = init_logger(__name__)


class AiterMLADecodeBackend(TritonMLABackend):
    # aiter's decode kernel is only wired up here for BF16/FP16 KV. FP8 KV would
    # need q/kv descales that this backend does not plumb through, so keep the
    # supported set narrower than the parent TritonMLA backend.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA_DECODE"

    @staticmethod
    def get_impl_cls() -> type["AiterMLADecodeImpl"]:
        return AiterMLADecodeImpl


class AiterMLADecodeImpl(TritonMLAImpl):
    # aiter's mla_decode_fwd drops the LSE, so it cannot participate in decode
    # context-parallel merging. Selection gates this backend off when DCP is on;
    # this flag communicates the same limitation to the MLA framework.
    can_return_lse_for_decode: bool = False

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from aiter.ops.triton.attention.mla import mla_decode_fwd

        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)

        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

        # aiter consumes the paged cache with an explicit (single) KV head dim,
        # and derives the rope split from the query's trailing dim.
        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)
        qk_rope_head_dim = q.shape[-1] - self.kv_lora_rank
        # Decode issues one query token per sequence.
        cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device=q.device)

        mla_decode_fwd(
            q,
            kv_buffer,
            o,
            cu_seqlens_q,
            attn_metadata.decode.seq_lens,
            attn_metadata.max_seq_len,
            attn_metadata.decode.block_table,
            self.scale,
            self.kv_lora_rank,
            qk_rope_head_dim,
            True,  # causal
            None,  # q_descale (BF16/FP16 KV only)
            None,  # kv_descale
        )

        # LSE is intentionally not produced (see class docstring); the DCP gate
        # in selection guarantees no caller needs it here.
        return o, None
