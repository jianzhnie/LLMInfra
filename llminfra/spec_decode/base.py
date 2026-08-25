"""Educational speculative decoding interface.

This module simulates draft-target verification without requiring trained
draft weights. It is an interface-level implementation, not a production
sampler.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class SpeculativeDecoder(nn.Module):
    """Draft-then-verify speculative decoding loop.

    Args:
        draft_model: Callable mapping ``(batch, seq)`` ids to logits.
        target_model: Callable mapping ``(batch, seq)`` ids to logits.
        num_speculative_tokens: Number of draft tokens generated per block.
        temperature: Sampling temperature; 0 selects argmax. With
            ``temperature > 0`` a rejection sampler verifies
            drafts: a draft token is accepted with probability
            ``min(1, p_target / p_draft)`` and otherwise replaced by a fresh
            sample from ``norm(max(0, p_target - p_draft))``.
        append_bonus_token: When True, sample one extra "bonus" token from
            the target logits after a fully accepted draft block (the
            standard speculative decoding behavior). Defaults to False to
            preserve the original block-size-only output contract.

    """

    def __init__(
        self,
        draft_model: Callable[[torch.Tensor], torch.Tensor],
        target_model: Callable[[torch.Tensor], torch.Tensor],
        num_speculative_tokens: int = 4,
        temperature: float = 0.0,
        append_bonus_token: bool = False,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.draft_model = draft_model
        self.target_model = target_model
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.temperature = float(temperature)
        self.append_bonus_token = bool(append_bonus_token)
        self.pad_token_id = int(pad_token_id)
        self.last_num_drafted = 0
        self.last_num_accepted = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate one speculative block.

        Draft tokens are rolled out autoregressively for the whole batch in
        parallel (one ``draft_model`` call per speculative step), while
        every batch row is verified independently. Because rows may accept
        different numbers of tokens, shorter results are right-padded with
        ``pad_token_id``.

        Returns:
            Input ids concatenated with accepted tokens (plus one bonus
            token when ``append_bonus_token`` is enabled and every draft
            was accepted).

        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq)")
        if input_ids.size(1) < 1:
            raise ValueError("input sequence must contain at least one token")
        drafted, draft_step_logits = self._draft(input_ids)
        target_input = torch.cat([input_ids, drafted], dim=-1)
        target_logits = self.target_model(target_input)
        rows: list[torch.Tensor] = []
        total_accepted = 0
        for row_index in range(input_ids.size(0)):
            decoded, num_accepted = self._verify_row(
                input_ids[row_index : row_index + 1],
                drafted[row_index : row_index + 1],
                [logits[row_index : row_index + 1] for logits in draft_step_logits],
                target_logits[row_index : row_index + 1],
            )
            rows.append(decoded[0])
            total_accepted += num_accepted
        max_length = max(row.numel() for row in rows)
        output = input_ids.new_full(
            (input_ids.size(0), max_length),
            self.pad_token_id,
        )
        for row_index, row in enumerate(rows):
            output[row_index, : row.numel()] = row
        self.last_num_drafted = input_ids.size(0) * self.num_speculative_tokens
        self.last_num_accepted = total_accepted
        return output

    def _draft(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Autoregressively draft tokens for all batch rows in parallel."""
        draft_tokens: list[torch.Tensor] = []
        draft_step_logits: list[torch.Tensor] = []
        draft_input = input_ids
        for _ in range(self.num_speculative_tokens):
            logits = self.draft_model(draft_input)[:, -1]
            token = self._sample(logits)
            draft_step_logits.append(logits)
            draft_tokens.append(token[:, None])
            draft_input = torch.cat([draft_input, token[:, None]], dim=-1)
        return torch.cat(draft_tokens, dim=-1), draft_step_logits

    def _verify_row(
        self,
        input_ids: torch.Tensor,
        drafted: torch.Tensor,
        draft_step_logits: list[torch.Tensor],
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Independently verify one batch row against its draft tokens."""
        prompt_length = input_ids.size(1)
        accepted: list[torch.Tensor] = []
        num_accepted = 0
        for step in range(self.num_speculative_tokens):
            target_step_logits = target_logits[:, prompt_length + step - 1]
            draft_token = drafted[:, step : step + 1]
            if self.temperature == 0.0:
                target_token = self._sample(target_step_logits)[:, None]
                if torch.equal(draft_token, target_token):
                    accepted.append(draft_token)
                    num_accepted += 1
                    continue
                accepted.append(target_token)
                break

            accepted_mask = self._accept_draft(
                draft_step_logits[step], target_step_logits, draft_token
            )
            if bool(accepted_mask.item()):
                accepted.append(draft_token)
                num_accepted += 1
            else:
                accepted.append(
                    self._sample_residual(draft_step_logits[step], target_step_logits)
                )
                break
        else:
            if self.append_bonus_token:
                accepted.append(self._sample(target_logits[:, -1])[:, None])
        return torch.cat([input_ids, *accepted], dim=-1), num_accepted

    def _accept_draft(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_token: torch.Tensor,
    ) -> torch.Tensor:
        """Return one rejection-sampling decision per batch row.

        Accepts the draft token with probability ``min(1, p_target/p_draft)``
        per row, where both distributions are temperature-scaled softmaxes
        evaluated at the draft token.
        """
        p_draft = torch.softmax(draft_logits / self.temperature, dim=-1)
        p_target = torch.softmax(target_logits / self.temperature, dim=-1)
        p_draft_token = p_draft.gather(-1, draft_token)
        p_target_token = p_target.gather(-1, draft_token)
        ratio = p_target_token / p_draft_token.clamp_min(1e-12)
        accept_prob = torch.clamp(ratio, max=1.0)
        return (torch.rand_like(accept_prob) < accept_prob).squeeze(-1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        flat = probabilities.reshape(-1, probabilities.size(-1))
        sampled = torch.multinomial(flat, 1)
        return sampled.reshape(probabilities.shape[:-1])

    def _sample_residual(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the correction distribution after rejecting a draft.

        Standard speculative decoding samples from the normalized positive
        difference ``max(0, p_target - p_draft)``. Numerical or identical-
        distribution edge cases can leave zero total mass; those rows safely
        fall back to the target distribution.
        """
        p_draft = torch.softmax(draft_logits / self.temperature, dim=-1)
        p_target = torch.softmax(target_logits / self.temperature, dim=-1)
        residual = (p_target - p_draft).clamp_min(0.0)
        total = residual.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            total > 1e-12,
            residual / total.clamp_min(1e-12),
            p_target,
        )
        return torch.multinomial(probabilities, 1)

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"num_speculative_tokens={self.num_speculative_tokens}, "
            f"temperature={self.temperature}, "
            f"append_bonus_token={self.append_bonus_token}, "
            f"pad_token_id={self.pad_token_id}"
        )
