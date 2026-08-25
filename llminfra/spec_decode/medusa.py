"""Medusa-style parallel draft heads for speculative decoding.

This module implements the trainable model component: several residual heads
predict future tokens from one backbone hidden state and expose Top-k draft
candidates.  It does not implement the tree-attention verifier or KV-cache
scheduler required by a production Medusa runtime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MedusaPredictionHead(nn.Module):
    """One residual Medusa branch followed by a vocabulary projection."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.residual = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.residual.weight)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.residual.bias is not None:
            nn.init.zeros_(self.residual.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        refined = hidden_state + F.silu(self.residual(hidden_state))
        logits: torch.Tensor = self.projection(refined)
        return logits


class MedusaHead(nn.Module):
    """Predict several future tokens in parallel from shared hidden states.

    Args:
        hidden_size: Backbone hidden-state dimension.
        vocab_size: Output vocabulary size.
        num_heads: Number of parallel future-token branches.
        bias: Whether each residual transform uses a bias.

    The forward result has shape ``(batch, seq, num_heads, vocab_size)``.
    Head ``h`` is trained against the token ``h + 1`` positions ahead.  This
    reference convention makes the component usable without a base LM head;
    a serving runtime may reserve head zero for the base model and shift the
    auxiliary offsets accordingly.

    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(hidden_size, vocab_size, num_heads) < 1:
            raise ValueError("hidden_size, vocab_size, and num_heads must be >= 1")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.num_heads = int(num_heads)
        self.heads = nn.ModuleList(
            MedusaPredictionHead(hidden_size, vocab_size, bias=bias)
            for _ in range(num_heads)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return parallel future-token logits."""
        if hidden_state.dim() != 3 or hidden_state.size(-1) != self.hidden_size:
            raise ValueError(
                "hidden_state must have shape (batch, seq_len, hidden_size)"
            )
        return torch.stack([head(hidden_state) for head in self.heads], dim=2)

    @torch.no_grad()
    def generate_candidates(
        self,
        hidden_state: torch.Tensor,
        top_k: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Top-k token ids and scores for the final sequence position.

        Returns:
            A pair ``(token_ids, scores)``, both shaped
            ``(batch, num_heads, top_k)``.

        """
        if not 1 <= top_k <= self.vocab_size:
            raise ValueError(f"top_k must be in [1, {self.vocab_size}]")
        # Only the final sequence position is scored; slice before the heads
        # so the per-head vocab projections do not scale with seq_len.
        final_logits = self(hidden_state[:, -1:])[:, -1]
        result = torch.topk(final_logits, k=top_k, dim=-1)
        return result.indices, result.values

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"num_heads={self.num_heads}"
        )


def medusa_loss(
    head: MedusaHead,
    hidden_state: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight_decay: float = 1.0,
) -> torch.Tensor:
    """Compute an aligned, weighted loss over all Medusa branches.

    Head ``h`` predicts ``labels[:, h + 1:]`` from hidden states ending
    ``h + 1`` positions earlier. Targets equal to ``-100`` are ignored.
    Branches without any valid target contribute no weight.
    """
    if labels.shape != hidden_state.shape[:2]:
        raise ValueError("labels must match hidden_state batch and sequence axes")
    if labels.size(1) <= head.num_heads:
        raise ValueError("sequence length must be greater than num_heads")
    if weight_decay <= 0:
        raise ValueError("weight_decay must be > 0")

    logits = head(hidden_state)
    total = hidden_state.sum() * 0.0
    weight_sum = 0.0
    for index in range(head.num_heads):
        offset = index + 1
        step_logits = logits[:, :-offset, index]
        targets = labels[:, offset:]
        if not targets.ne(-100).any():
            continue
        weight = weight_decay**index
        total = total + weight * F.cross_entropy(
            step_logits.reshape(-1, head.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        weight_sum += weight
    if weight_sum == 0:
        return total
    return total / weight_sum


__all__ = ["MedusaHead", "medusa_loss"]
