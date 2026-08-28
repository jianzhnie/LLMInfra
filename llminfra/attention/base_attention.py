"""Base attention module providing common functionality for all attention mechanisms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn


class BaseAttention(nn.Module, ABC):
    """Abstract base class for attention mechanisms.

    This class provides common functionality and interface for all attention
    implementations, ensuring consistency across different attention types.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        logit_softcap: float | None = None,
        attention_sink: bool = False,
    ) -> None:
        """Initialize base attention parameters.

        Args:
            hidden_size: Dimensionality of input and output features
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            bias: Whether to use bias in linear projections
            qk_norm: Whether to RMS-normalize queries and keys over the
                head dimension after :meth:`split_head` and before the
                attention scores are computed (parameter-free, Qwen3-style).
                Subclasses opt in by calling :meth:`_apply_qk_norm`.
            logit_softcap: Optional Gemma 2 style logit soft-capping value.
                Scores become ``softcap * tanh(scores / softcap)`` before
                masking and softmax.
            attention_sink: Whether to append one learnable sink logit per
                head to the softmax denominator (GPT-OSS / StreamingLLM
                style). The sink absorbs probability mass without contributing
                to the output, so attention weights over real keys may sum to
                less than 1.

        Raises:
            ValueError: If hidden_size is not divisible by num_heads, or if
                ``logit_softcap`` is not positive.

        """
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible "
                f"by num_heads ({num_heads})"
            )
        if logit_softcap is not None and logit_softcap <= 0:
            raise ValueError(f"logit_softcap must be > 0, got {logit_softcap}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_prob = dropout
        self.bias = bias
        self.qk_norm = qk_norm
        self.logit_softcap = logit_softcap

        # Pre-compute scaling factor for efficiency
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        self.sink_logits: nn.Parameter | None
        if attention_sink:
            self.sink_logits = nn.Parameter(torch.zeros(num_heads))
        else:
            self.sink_logits = None

    @abstractmethod
    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for attention mechanism.

        Args:
            hidden_state: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask, broadcastable against the
                (batch_size, num_heads, seq_len, seq_len) score tensor.
                1/True marks positions to attend to, 0/False masks them out.
                A 3D mask is interpreted as (batch_size, seq_len, seq_len) and
                broadcast over all heads.
            return_attention_weights: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights

        """

    def split_head(self, x: torch.Tensor, num_heads: int | None = None) -> torch.Tensor:
        """Split input tensor into attention heads.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads * head_dim)
            num_heads: Number of heads to split into. Defaults to the module's
                ``num_heads``; pass a smaller number (e.g. 1) for the shared
                key/value projections of MQA-style variants.

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)

        """
        num_heads = num_heads or self.num_heads
        batch_size, seq_len, features = x.size()
        return x.view(batch_size, seq_len, num_heads, features // num_heads).transpose(
            1, 2
        )

    def _apply_qk_norm(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RMS-normalize queries and keys over the head dimension.

        Parameter-free normalization (``F.rms_norm`` without a learnable
        weight) applied to head-split tensors of shape
        ``(batch_size, num_heads, seq_len, head_dim)``, as in Qwen3. Callers
        should apply it after :meth:`split_head` and before computing
        attention scores. It is a no-op unless the module was constructed
        with ``qk_norm=True``.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len,
                head_dim)
            key: Key tensor of shape (batch_size, num_kv_heads, kv_len,
                head_dim)

        Returns:
            The (possibly normalized) query and key tensors.

        """
        if not self.qk_norm:
            return query, key
        return F.rms_norm(query, (self.head_dim,)), F.rms_norm(key, (self.head_dim,))

    def combine_head(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multiple attention heads into single tensor.

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)

        """
        batch_size, _, seq_len, _ = x.size()
        return (
            x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        )

    def apply_attention_mask(
        self, attention_scores: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply attention mask to scores.

        Args:
            attention_scores: Attention scores tensor
            attention_mask: Optional attention mask

        Returns:
            Masked attention scores

        """
        if attention_mask is not None:
            # A 3D mask is (batch, q_len, kv_len); promote it to
            # (batch, 1, q_len, kv_len) so it broadcasts over heads instead
            # of misaligning its batch dim with the head dim.
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_scores = torch.masked_fill(
                attention_scores, attention_mask == 0, float("-inf")
            )
        return attention_scores

    def compute_attention_weights(
        self,
        attention_scores: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention weights from scores.

        Args:
            attention_scores: Raw attention scores
            attention_mask: Optional attention mask

        Returns:
            Normalized attention weights. When ``attention_mask`` is given,
            rows whose keys are all masked are defined to be all-zero
            (softmax over all ``-inf`` would be NaN). With
            ``attention_sink`` enabled the sink logit keeps every row finite,
            so masked rows simply put all their mass on the sink and the
            returned weights over real keys are still all-zero.

        """
        if self.logit_softcap is not None:
            # Gemma 2 order: soft-cap the raw scores first, then mask.
            cap = self.logit_softcap
            attention_scores = cap * torch.tanh(attention_scores / cap)
        attention_scores = self.apply_attention_mask(attention_scores, attention_mask)
        attention_weights: torch.Tensor
        if self.sink_logits is not None:
            # Append one learnable sink logit per head as an extra key column;
            # it takes part in the softmax but is sliced off afterwards, so
            # it never contributes to the weighted value sum.
            sink = self.sink_logits.to(attention_scores.dtype)
            sink = sink.view(1, -1, 1, 1).expand(*attention_scores.shape[:-1], 1)
            scores_with_sink = torch.cat([attention_scores, sink], dim=-1)
            attention_weights = torch.softmax(scores_with_sink, dim=-1)[..., :-1]
        else:
            attention_weights = torch.softmax(attention_scores, dim=-1)
            if attention_mask is not None:
                # Fully masked rows produce NaN in the softmax; define their
                # weights (and therefore their output) as zero instead.
                # Without a mask no row can be fully masked, so skip the
                # extra full pass.
                attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = self.dropout(attention_weights)
        return attention_weights

    @staticmethod
    def _init_projections(*modules: nn.Linear) -> None:
        """Initialize projections with Xavier uniform weights and zero biases."""
        for module in modules:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, bias={self.bias}"
        )


def validate_attention_inputs(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor | None, num_heads: int
) -> tuple[int, int]:
    """Validate attention input tensors.

    Args:
        hidden_state: Input hidden state tensor
        attention_mask: Optional attention mask
        num_heads: Number of attention heads

    Returns:
        Tuple of (batch_size, seq_len)

    Raises:
        ValueError: If input tensors have invalid shapes

    """
    if hidden_state.dim() != 3:
        raise ValueError(f"hidden_state must be 3D, got {hidden_state.dim()}D")

    batch_size, seq_len, _hidden_size = hidden_state.size()

    if attention_mask is not None:
        if attention_mask.dim() not in [3, 4]:
            raise ValueError(
                f"attention_mask must be 3D or 4D, got {attention_mask.dim()}D"
            )

        if attention_mask.size(0) != batch_size:
            raise ValueError(
                f"attention_mask batch size {attention_mask.size(0)} "
                f"must match hidden_state batch size {batch_size}"
            )

        if attention_mask.size(-1) != seq_len:
            raise ValueError(
                f"attention_mask sequence length {attention_mask.size(-1)} "
                f"must match hidden_state sequence length {seq_len}"
            )

        if attention_mask.is_floating_point():
            raise ValueError(
                "attention_mask must use the 1/0 (or bool) convention where "
                "1 marks positions to attend to; additive float masks "
                "(0/-inf) are not supported"
            )

    return batch_size, seq_len
