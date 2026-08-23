"""T5-style encoder-decoder architecture built from LLMInfra modules.

This module provides teaching-grade building blocks for sequence-to-sequence
models: a bidirectional ``EncoderBlock``, a ``DecoderBlock`` with masked
self-attention plus cross-attention, and an ``EncoderDecoderModel`` skeleton
with a shared vocabulary embedding and an LM head.

Simplifications (documented for teaching purposes):

- No positional encoding is applied. T5 uses relative position biases inside
  its attention scores; here we omit position information entirely to keep
  the data flow easy to follow.
- The decoder cross-attention is a small standalone module
  (``CrossAttention``) because ``MultiHeadAttention`` only supports
  self-attention (query length == key/value length).
"""

from __future__ import annotations

import torch
from torch import nn

from ..attention.base_attention import BaseAttention
from ..attention.multi_head_attention import MultiHeadAttention
from ..layers.feed_forward import SwiGLUFFN
from ..layers.normalization import RMSNorm

#: Maximum number of cached ``(seq_len, device)`` causal masks. Caching makes
#: mask construction O(1) on repeated sequence lengths, but each entry holds
#: an O(seq^2) bool tensor on its device, so an unbounded cache could
#: accumulate device memory across many distinct lengths. When the cap is
#: exceeded the cache is simply rebuilt from scratch: the trade-off favors
#: bounding memory over keeping a perfect hit rate for workloads that cycle
#: through many sequence lengths.
_MAX_CACHED_CAUSAL_MASKS = 8

_CAUSAL_MASK_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def cached_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a shared lower-triangular keep mask of shape ``(1, 1, seq, seq)``.

    Building the ``(seq, seq)`` triangle costs O(seq^2) host work on every
    forward; caching it per ``(seq_len, device)`` makes mask construction
    O(1). The cached tensor is treated as read-only: callers combine it with
    padding or prefix masks through logical ops that allocate fresh tensors,
    and attention only reads it. At most ``_MAX_CACHED_CAUSAL_MASKS`` entries
    are kept; beyond that the cache is cleared and rebuilt.
    """
    key = (seq_len, device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        if len(_CAUSAL_MASK_CACHE) >= _MAX_CACHED_CAUSAL_MASKS:
            _CAUSAL_MASK_CACHE.clear()
        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )[None, None]
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


class CrossAttention(BaseAttention):
    """Multi-head cross-attention where queries and keys/values differ.

    Queries are projected from ``query_state`` while keys and values are
    projected from ``key_value_state``, so the two inputs may have different
    sequence lengths. The implementation mirrors ``MultiHeadAttention`` and
    reuses ``split_head``/``combine_head``/``compute_attention_weights`` from
    ``BaseAttention``.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads. Must divide ``hidden_size``.
        dropout: Dropout probability for attention weights. Defaults to 0.0
            so the module is deterministic in eval-style teaching examples.
        bias: Whether to use bias in the linear projections.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    # Cross-attention deliberately widens the base signature: queries come
    # from query_state while keys/values come from key_value_state, so the
    # single-hidden-state BaseAttention.forward contract cannot apply.
    def forward(  # type: ignore[override]
        self,
        query_state: torch.Tensor,
        key_value_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend from ``query_state`` positions to ``key_value_state``.

        Args:
            query_state: Tensor of shape ``(batch, q_len, hidden_size)``.
            key_value_state: Tensor of shape ``(batch, kv_len, hidden_size)``;
                ``kv_len`` may differ from ``q_len``.
            attention_mask: Optional mask broadcastable against the
                ``(batch, num_heads, q_len, kv_len)`` score tensor, e.g. a
                ``(batch, 1, 1, kv_len)`` padding mask. 1/True marks visible
                key positions, 0/False masks them out.
            return_attention_weights: Also return the attention weights.

        Returns:
            Output tensor of shape ``(batch, q_len, hidden_size)``, and
            optionally the weights of shape ``(batch, num_heads, q_len,
            kv_len)``.

        """
        if query_state.dim() != 3 or key_value_state.dim() != 3:
            raise ValueError("query_state and key_value_state must be 3D")
        if query_state.size(0) != key_value_state.size(0):
            raise ValueError("query_state and key_value_state batch sizes must match")

        query = self.split_head(self.q_proj(query_state))
        key = self.split_head(self.k_proj(key_value_state))
        value = self.split_head(self.v_proj(key_value_state))

        # (batch, num_heads, q_len, head_dim)
        # * (batch, num_heads, head_dim, kv_len)
        # -> (batch, num_heads, q_len, kv_len)
        attention_scores = (
            torch.matmul(query, key.transpose(-1, -2)) * self.scale_factor
        )
        attention_weights = self.compute_attention_weights(
            attention_scores, attention_mask
        )

        output: torch.Tensor = torch.matmul(attention_weights, value)
        output = self.o_proj(self.combine_head(output))

        if return_attention_weights:
            return output, attention_weights
        return output


class EncoderBlock(nn.Module):
    """Pre-norm encoder block with bidirectional (non-causal) self-attention.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        norm_eps: RMSNorm epsilon.
        dropout: Dropout probability for the attention weights.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout=dropout)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one bidirectional encoder block.

        Args:
            hidden_state: Tensor of shape ``(batch, src_len, hidden_size)``.
            attention_mask: Optional padding mask broadcastable against the
                attention scores, e.g. ``(batch, 1, 1, src_len)``. No causal
                structure is added here: every visible position attends to
                every other visible position.

        Returns:
            Tensor of shape ``(batch, src_len, hidden_size)``.

        """
        hidden_state = hidden_state + self.attention(
            self.norm1(hidden_state), attention_mask=attention_mask
        )
        output: torch.Tensor = hidden_state + self.ffn(self.norm2(hidden_state))
        return output

    def extra_repr(self) -> str:
        """Report the block dimensions shown in the module ``repr``."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"intermediate_size={self.intermediate_size}"
        )


class DecoderBlock(nn.Module):
    """Pre-norm decoder block with masked self-attention and cross-attention.

    The sublayer order follows the original transformer: causal self-attention
    over the target sequence, then cross-attention over the encoder output,
    then a feed-forward network. Each sublayer uses a pre-norm residual.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        norm_eps: RMSNorm epsilon.
        dropout: Dropout probability for the attention weights.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, num_heads, dropout=dropout
        )
        self.cross_attention = CrossAttention(hidden_size, num_heads, dropout=dropout)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm3 = RMSNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_state: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        cross_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decoder block.

        Args:
            hidden_state: Target-side tensor of shape
                ``(batch, tgt_len, hidden_size)``.
            encoder_output: Source-side tensor of shape
                ``(batch, src_len, hidden_size)``.
            self_attention_mask: Mask for the causal self-attention,
                broadcastable against ``(batch, num_heads, tgt_len, tgt_len)``.
            cross_attention_mask: Mask for the cross-attention, broadcastable
                against ``(batch, num_heads, tgt_len, src_len)``, e.g. a
                ``(batch, 1, 1, src_len)`` source padding mask.

        Returns:
            Tensor of shape ``(batch, tgt_len, hidden_size)``.

        """
        hidden_state = hidden_state + self.self_attention(
            self.norm1(hidden_state), attention_mask=self_attention_mask
        )
        hidden_state = hidden_state + self.cross_attention(
            self.norm2(hidden_state),
            encoder_output,
            attention_mask=cross_attention_mask,
        )
        output: torch.Tensor = hidden_state + self.ffn(self.norm3(hidden_state))
        return output

    def extra_repr(self) -> str:
        """Report the block dimensions shown in the module ``repr``."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"intermediate_size={self.intermediate_size}"
        )


class EncoderDecoderModel(nn.Module):
    """T5-style encoder-decoder skeleton with a shared vocabulary embedding.

    The model composes a token embedding, a stack of ``EncoderBlock`` modules,
    a stack of ``DecoderBlock`` modules, a final RMSNorm and an LM head. It is
    intended for architecture experiments, not for reproducing T5 exactly
    (see the module docstring for simplifications).

    Args:
        vocab_size: Vocabulary size for the shared embedding and LM head.
        hidden_size: Model hidden dimension.
        num_encoder_layers: Number of encoder blocks.
        num_decoder_layers: Number of decoder blocks.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        max_seq_len: Maximum supported source/target sequence length.
        norm_eps: RMSNorm epsilon.
        dropout: Dropout probability for the attention weights.
        tie_word_embeddings: Share the LM head weight with the token
            embedding (as T5 does). Defaults to True.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 512,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or hidden_size < 1:
            raise ValueError("vocab_size and hidden_size must be >= 1")
        if num_encoder_layers < 1 or num_decoder_layers < 1:
            raise ValueError("num_encoder_layers and num_decoder_layers must be >= 1")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_seq_len = int(max_seq_len)
        self.tie_word_embeddings = bool(tie_word_embeddings)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.ModuleList(
            EncoderBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                norm_eps=norm_eps,
                dropout=dropout,
            )
            for _ in range(num_encoder_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                norm_eps=norm_eps,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        )
        self.encoder_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.decoder_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``src_ids`` and decode ``tgt_ids`` into next-token logits.

        Args:
            src_ids: Long tensor of shape ``(batch, src_len)``.
            tgt_ids: Long tensor of shape ``(batch, tgt_len)``.
            src_mask: Optional source padding mask of shape
                ``(batch, src_len)``; 1/True marks valid tokens.
            tgt_mask: Optional target padding mask of shape
                ``(batch, tgt_len)``; 1/True marks valid tokens.

        Returns:
            Logits of shape ``(batch, tgt_len, vocab_size)``.

        """
        if src_ids.dim() != 2 or tgt_ids.dim() != 2:
            raise ValueError("src_ids and tgt_ids must have shape (batch, seq_len)")
        batch_size, src_len = src_ids.size()
        tgt_len = tgt_ids.size(1)
        if tgt_ids.size(0) != batch_size:
            raise ValueError("src_ids and tgt_ids batch sizes must match")
        if src_len > self.max_seq_len or tgt_len > self.max_seq_len:
            raise ValueError(f"sequence length exceeds max_seq_len {self.max_seq_len}")

        # (batch, 1, 1, src_len) key padding mask for the (bidirectional)
        # encoder self-attention and for the decoder cross-attention.
        src_padding = None
        if src_mask is not None:
            if src_mask.shape != (batch_size, src_len):
                raise ValueError("src_mask must have shape (batch, src_len)")
            src_padding = src_mask[:, None, None, :].bool()

        encoder_output = self.embed_tokens(src_ids)
        for block in self.encoder:
            encoder_output = block(encoder_output, attention_mask=src_padding)
        encoder_output = self.encoder_norm(encoder_output)

        # Causal mask over the target sequence combined with target padding.
        causal = cached_causal_mask(tgt_len, tgt_ids.device)
        if tgt_mask is not None:
            if tgt_mask.shape != (batch_size, tgt_len):
                raise ValueError("tgt_mask must have shape (batch, tgt_len)")
            causal = causal & tgt_mask[:, None, None, :].bool()
        causal = causal.expand(batch_size, 1, tgt_len, tgt_len)

        hidden_state = self.embed_tokens(tgt_ids)
        for block in self.decoder:
            hidden_state = block(
                hidden_state,
                encoder_output,
                self_attention_mask=causal,
                cross_attention_mask=src_padding,
            )
        hidden_state = self.decoder_norm(hidden_state)
        logits: torch.Tensor = self.lm_head(hidden_state)
        return logits

    def extra_repr(self) -> str:
        """Report the architecture hyperparameters shown in the module ``repr``."""
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"num_encoder_layers={self.num_encoder_layers}, "
            f"num_decoder_layers={self.num_decoder_layers}, "
            f"tie_word_embeddings={self.tie_word_embeddings}"
        )
