"""Dynamic speculative block scheduling inspired by DSpark-style systems."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from .base import SpeculativeDecoder


class DSparkScheduler:
    """Markov-style controller for speculative block length.

    The state space is a sorted list of allowed draft lengths. High observed
    acceptance moves one state upward, low acceptance moves one state
    downward, and intermediate acceptance stays in the current state. This is
    a portable scheduling interface, not a reproduction of an unpublished
    serving runtime.
    """

    def __init__(
        self,
        block_lengths: Sequence[int] = (1, 2, 4, 8),
        grow_threshold: float = 0.8,
        shrink_threshold: float = 0.5,
        initial_length: int | None = None,
    ) -> None:
        lengths = tuple(sorted({int(length) for length in block_lengths}))
        if not lengths or lengths[0] < 1:
            raise ValueError("block_lengths must contain positive integers")
        if not 0.0 <= shrink_threshold <= grow_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= shrink <= grow <= 1")
        if initial_length is None:
            initial_length = lengths[0]
        if initial_length not in lengths:
            raise ValueError("initial_length must be present in block_lengths")
        self.block_lengths = lengths
        self.grow_threshold = float(grow_threshold)
        self.shrink_threshold = float(shrink_threshold)
        self.state_index = lengths.index(initial_length)

    @property
    def draft_length(self) -> int:
        """Current number of tokens to draft."""
        return self.block_lengths[self.state_index]

    def update(self, num_accepted: int, num_drafted: int) -> int:
        """Update the scheduler state and return the next draft length."""
        if num_drafted < 1 or not 0 <= num_accepted <= num_drafted:
            raise ValueError("require 0 <= num_accepted <= num_drafted and drafted > 0")
        acceptance_rate = num_accepted / num_drafted
        if acceptance_rate >= self.grow_threshold:
            self.state_index = min(self.state_index + 1, len(self.block_lengths) - 1)
        elif acceptance_rate <= self.shrink_threshold:
            self.state_index = max(self.state_index - 1, 0)
        return self.draft_length

    def reset(self, length: int | None = None) -> None:
        """Reset to the smallest block or an explicitly allowed length."""
        if length is None:
            self.state_index = 0
        elif length in self.block_lengths:
            self.state_index = self.block_lengths.index(length)
        else:
            raise ValueError("reset length must be present in block_lengths")


class DSparkDecoder(nn.Module):
    """Speculative decoder whose draft block is chosen by a scheduler."""

    def __init__(
        self,
        draft_model: Callable[[torch.Tensor], torch.Tensor],
        target_model: Callable[[torch.Tensor], torch.Tensor],
        scheduler: DSparkScheduler | None = None,
        *,
        temperature: float = 0.0,
        append_bonus_token: bool = True,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.scheduler = scheduler or DSparkScheduler()
        self.temperature = float(temperature)
        self.append_bonus_token = bool(append_bonus_token)
        self.pad_token_id = int(pad_token_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate one dynamically sized block and update scheduler state.

        Note: every call advances ``self.scheduler`` based on the observed
        acceptance rate, even under ``eval()``/``no_grad()``. Repeated calls
        with the same input may therefore use different draft lengths. This
        stateful behavior is intentional for teaching purposes.
        """
        decoder = SpeculativeDecoder(
            self.draft_model,
            self.target_model,
            num_speculative_tokens=self.scheduler.draft_length,
            temperature=self.temperature,
            append_bonus_token=self.append_bonus_token,
            pad_token_id=self.pad_token_id,
        )
        output: torch.Tensor = decoder(input_ids)
        self.scheduler.update(
            decoder.last_num_accepted,
            decoder.last_num_drafted,
        )
        return output

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"draft_length={self.scheduler.draft_length}, "
            f"temperature={self.temperature}, "
            f"append_bonus_token={self.append_bonus_token}"
        )
