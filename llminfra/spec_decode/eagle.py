"""EAGLE-1, EAGLE-2 and EAGLE-3 speculative decoding interfaces."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class EagleSpeculator(nn.Module):
    """Eagle-style speculative decoder using hidden states for drafting.

    ``draft_head`` maps hidden states to next-token logits; ``target_model``
    maps token ids to logits for verification. This is an interface-level
    simulation, not a trained Eagle model; like `SpeculativeDecoder`, draft
    tokens are read off the last positions of the draft logits instead of an
    autoregressive rollout.

    Args:
        draft_head: Callable mapping hidden states to logits.
        target_model: Callable mapping ``(batch, seq)`` ids to logits.
        num_speculative_tokens: Number of draft tokens generated per block.
        append_bonus_token: When True, take one extra "bonus" token (argmax
            of the final target logits) after a fully accepted draft block.
            Defaults to False to preserve the original output contract.

    """

    def __init__(
        self,
        draft_head: Callable[[torch.Tensor], torch.Tensor],
        target_model: Callable[[torch.Tensor], torch.Tensor],
        num_speculative_tokens: int = 4,
        append_bonus_token: bool = False,
    ) -> None:
        super().__init__()
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        self.draft_head = draft_head
        self.target_model = target_model
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.append_bonus_token = bool(append_bonus_token)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Draft from hidden states and verify with the target model.

        Simplification: verification stops at the first mismatched draft
        position across the whole batch (any row), unlike the per-row
        verification of ``SpeculativeDecoder``. Emitted tokens are still
        exact target-model argmax outputs; rows that would have accepted
        more tokens simply receive a shorter continuation.
        """
        if input_ids.size(1) < self.num_speculative_tokens:
            raise ValueError(
                "input sequence must be at least num_speculative_tokens long"
            )
        if hidden_states.dim() != 3 or hidden_states.size(0) != input_ids.size(0):
            raise ValueError(
                "hidden_states must have shape (batch, seq_len, hidden_size) "
                "matching the input_ids batch size"
            )
        if hidden_states.size(1) < self.num_speculative_tokens:
            raise ValueError(
                "hidden_states must be at least num_speculative_tokens long"
            )
        draft_logits = self.draft_head(hidden_states)
        draft_tokens = torch.argmax(
            draft_logits[:, -self.num_speculative_tokens :], dim=-1
        )
        target_input = torch.cat([input_ids, draft_tokens], dim=-1)
        target_logits = self.target_model(target_input)
        accepted: list[torch.Tensor] = []
        start = input_ids.size(1)
        for step in range(self.num_speculative_tokens):
            target_next = torch.argmax(
                target_logits[:, start + step - 1 : start + step], dim=-1
            )
            accepted.append(target_next)
            if (draft_tokens[:, step] != target_next[:, 0]).any():
                break
        else:
            if self.append_bonus_token:
                accepted.append(torch.argmax(target_logits[:, -1:], dim=-1))
        return torch.cat([input_ids, *accepted], dim=-1)

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"num_speculative_tokens={self.num_speculative_tokens}, "
            f"append_bonus_token={self.append_bonus_token}"
        )


class Eagle1Speculator(EagleSpeculator):
    """EAGLE-1 interface backed by the shared hidden-state speculator."""


class Eagle2Speculator(EagleSpeculator):
    """EAGLE-2 interface backed by the shared hidden-state speculator."""


class Eagle3Speculator(EagleSpeculator):
    """EAGLE-3 interface backed by the shared hidden-state speculator."""


__all__ = [
    "Eagle1Speculator",
    "Eagle2Speculator",
    "Eagle3Speculator",
    "EagleSpeculator",
]
