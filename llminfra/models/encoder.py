"""Composable encoder-only Transformer for representation learning tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..layers.normalization import LayerNorm, RMSNorm
from ..module_registry import build_positional_encoding
from ..positional.classic import (
    LearnedAbsolutePositionEmbedding,
    SinusoidalPositionEmbedding,
)
from .encoder_decoder import EncoderBlock
from .heads import (
    EmbeddingHead,
    RewardModelHead,
    SequenceClassificationHead,
    TokenClassificationHead,
    pool_hidden_state,
)


@dataclass
class EncoderOutput:
    """Structured output from :class:`EncoderOnlyModel`."""

    last_hidden_state: torch.Tensor
    pooled_output: torch.Tensor
    head_output: torch.Tensor | None = None
    language_model_logits: torch.Tensor | None = None


class EncoderOnlyModel(nn.Module):
    """Bidirectional Transformer encoder with optional task head.

    Args:
        vocab_size: Token vocabulary size.
        hidden_size: Model hidden dimension.
        num_layers: Number of bidirectional encoder blocks.
        num_heads: Attention heads per block.
        intermediate_size: SwiGLU intermediate dimension.
        max_seq_len: Maximum supported sequence length.
        positional: Input-side positional encoding. ``"learned"`` matches the
            classic BERT-style choice; sinusoidal, RoPE scaling variants, and
            ``"nope"`` are also accepted. Score-bias modules such as ALiBi and
            T5 relative bias require attention-level integration and are
            rejected here.
        positional_kwargs: Extra positional module arguments.
        type_vocab_size: Optional token-type vocabulary; zero disables it.
        embedding_dropout: Dropout after embedding composition.
        norm_type: Final normalization, ``"layernorm"`` or ``"rmsnorm"``.
        output_head: Optional classification, token, reward, embedding, or
            custom tensor-to-tensor head.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 512,
        positional: str = "learned",
        positional_kwargs: dict[str, Any] | None = None,
        type_vocab_size: int = 0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        norm_type: str = "layernorm",
        output_head: nn.Module | None = None,
        tie_word_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_size, num_layers, num_heads, intermediate_size) < 1:
            raise ValueError("model dimensions must be >= 1")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if max_seq_len < 1 or type_vocab_size < 0:
            raise ValueError("max_seq_len must be >= 1 and type_vocab_size >= 0")
        if norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError("norm_type must be 'layernorm' or 'rmsnorm'")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_seq_len = int(max_seq_len)
        self.type_vocab_size = int(type_vocab_size)
        self.tie_word_embeddings = bool(tie_word_embeddings)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embeddings = (
            nn.Embedding(type_vocab_size, hidden_size) if type_vocab_size else None
        )
        positional_kwargs = dict(positional_kwargs or {})
        if positional in {"alibi", "t5_bias"}:
            raise ValueError(
                f"{positional} is an attention-score bias and cannot be applied "
                "as an encoder input embedding"
            )
        self.positional = build_positional_encoding(
            positional,
            dim=hidden_size,
            max_seq_len=max_seq_len,
            **positional_kwargs,
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                norm_eps=norm_eps,
                dropout=embedding_dropout,
            )
            for _ in range(num_layers)
        )
        self.norm = (
            LayerNorm(hidden_size, eps=norm_eps)
            if norm_type == "layernorm"
            else RMSNorm(hidden_size, eps=norm_eps)
        )
        self.output_head = output_head

        self.lm_head: nn.Linear | None = None
        if tie_word_embeddings:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Encode tokens bidirectionally and optionally execute a task head."""
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        if attention_mask is None:
            # All positions are valid: skip the padding-mask machinery
            # (all-ones masks multiply to identity and mask nothing).
            mask = None
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            if not mask.any(dim=1).all():
                raise ValueError("every batch row must contain at least one token")

        hidden_state = self.embed_tokens(input_ids)
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            if token_type_ids.shape != input_ids.shape:
                raise ValueError("token_type_ids must have the same shape as input_ids")
            if token_type_ids.numel() and (
                token_type_ids.min() < 0 or token_type_ids.max() >= self.type_vocab_size
            ):
                raise ValueError("token_type_ids contain an out-of-range id")
            hidden_state = hidden_state + self.token_type_embeddings(token_type_ids)
        elif token_type_ids is not None:
            raise ValueError("token_type_ids require type_vocab_size > 0")

        if isinstance(
            self.positional,
            LearnedAbsolutePositionEmbedding | SinusoidalPositionEmbedding,
        ):
            hidden_state = self.positional(hidden_state, position_ids)
        else:
            if position_ids is not None:
                raise ValueError(
                    "explicit position_ids require learned or sinusoidal positions"
                )
            hidden_state = self.positional(hidden_state)
        hidden_state = self.embedding_dropout(hidden_state)

        query_mask: torch.Tensor | None = None
        key_mask: torch.Tensor | None = None
        if mask is not None:
            query_mask = mask[:, :, None].to(hidden_state.dtype)
            hidden_state = hidden_state * query_mask
            key_mask = mask[:, None, None, :]
        for block in self.blocks:
            hidden_state = block(hidden_state, attention_mask=key_mask)
            if query_mask is not None:
                hidden_state = hidden_state * query_mask
        hidden_state = self.norm(hidden_state)
        if query_mask is not None:
            hidden_state = hidden_state * query_mask

        pooled = pool_hidden_state(hidden_state, mask, pooling="first")
        head_output = self._run_head(hidden_state, mask)
        return EncoderOutput(
            last_hidden_state=hidden_state,
            pooled_output=pooled,
            head_output=head_output,
            language_model_logits=(
                self.lm_head(hidden_state) if self.lm_head is not None else None
            ),
        )

    def _run_head(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.output_head is None:
            return None
        if isinstance(
            self.output_head,
            SequenceClassificationHead | RewardModelHead | EmbeddingHead,
        ):
            output: torch.Tensor = self.output_head(hidden_state, attention_mask)
            return output
        if isinstance(self.output_head, TokenClassificationHead):
            output = self.output_head(hidden_state)
            return output
        output = self.output_head(hidden_state)
        return output

    def extra_repr(self) -> str:
        """Report the architecture hyperparameters shown in the module ``repr``."""
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"type_vocab_size={self.type_vocab_size}"
        )


__all__ = ["EncoderOnlyModel", "EncoderOutput"]
