"""Multi-Token Prediction (MTP) head and speculative decoder."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


class MTPDecoder(nn.Module):
    """Verify greedily predicted MTP tokens with a target model.

    Simplification: verification stops at the first mismatched draft
    position across the whole batch (any row), unlike the per-row
    verification of ``SpeculativeDecoder``. Emitted tokens are still exact
    target-model argmax outputs; rows that would have accepted more tokens
    simply receive a shorter continuation.

    Args:
        mtp_head: Head producing one logit tensor for each future position.
        target_model: Callable mapping token ids to target-model logits.

    """

    def __init__(
        self,
        mtp_head: MultiTokenPredictionHead,
        target_model: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.mtp_head = mtp_head
        self.target_model = target_model

    def forward(
        self, input_ids: torch.Tensor, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Draft tokens with the MTP head and verify them with the target model.

        Each head of ``mtp_head`` contributes one greedy draft token: the
        argmax of that head's logits at the final position of
        ``hidden_state``. The draft tokens are appended to ``input_ids`` and
        scored by the target model, which verifies the draft
        position-by-position: a draft token is accepted while the target's
        argmax at the corresponding position agrees with it. Verification
        stops at the first mismatch (batch-wide, any row), where the rejected
        draft token is replaced by the correct target-model token, so every
        emitted token is an exact target-model argmax output.

        Args:
            input_ids: Prompt token ids of shape ``(batch, seq)``.
            hidden_state: Backbone hidden states of shape
                ``(batch, seq, hidden_size)`` with the same batch size as
                ``input_ids``; only the final position feeds the draft heads.

        Returns:
            ``input_ids`` concatenated with the emitted tokens, of shape
            ``(batch, seq + k)`` where ``k`` ranges from 1 (immediate
            mismatch, only the correcting target token) to
            ``mtp_head.num_predictions`` (all draft tokens accepted).

        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq)")
        if hidden_state.dim() != 3 or hidden_state.size(0) != input_ids.size(0):
            raise ValueError(
                "hidden_state must have shape (batch, seq, hidden_size) "
                "matching the input_ids batch size"
            )
        # Only the final position feeds the draft heads (see the docstring);
        # slice before the head so the (batch, seq, vocab) projections shrink
        # to a single position instead of scaling with the full sequence.
        logits = self.mtp_head(hidden_state[:, -1:])
        draft = torch.stack(
            [torch.argmax(item[:, -1], dim=-1) for item in logits], dim=-1
        )
        target_logits = self.target_model(torch.cat((input_ids, draft), dim=-1))
        accepted = []
        for index, token in enumerate(draft.unbind(dim=1)):
            target = torch.argmax(
                target_logits[:, input_ids.size(1) - 1 + index], dim=-1
            )
            accepted.append(target)
            if not torch.equal(target, token):
                break
        return torch.cat((input_ids, *[item[:, None] for item in accepted]), dim=-1)


class MultiTokenPredictionHead(nn.Module):
    """Predict the next ``num_predictions`` tokens from one hidden state.

    Args:
        hidden_size: Hidden state dimension.
        vocab_size: Vocabulary size.
        num_predictions: Number of future tokens predicted per step.

    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predictions: int = 2,
    ) -> None:
        super().__init__()
        if num_predictions < 1:
            raise ValueError("num_predictions must be >= 1")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.num_predictions = int(num_predictions)
        self.heads = nn.ModuleList(
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_predictions)
        )

    def forward(self, hidden_state: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of logit tensors, one per prediction head."""
        return [head(hidden_state) for head in self.heads]

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"num_predictions={self.num_predictions}"
        )


def mtp_loss(
    head: MultiTokenPredictionHead,
    hidden_state: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_future: int | None = None,
    weight_decay: float = 1.0,
    logits_list: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Teaching-grade multi-token prediction training loss.

    Shape convention: ``hidden_state`` is ``(batch, seq, hidden_size)`` and
    ``labels`` is ``(batch, seq)`` token ids. Head ``k`` (0-based) predicts
    the token ``k + 1`` steps ahead, so its logits at positions
    ``[:, : seq - k - 1]`` are scored against ``labels[:, k + 1 :]``. Each
    step contributes a cross-entropy over all such aligned positions, and
    the returned loss is the weighted mean over steps.

    Simplification: all prediction steps share the same ``hidden_state``
    (no sequential re-embedding of predicted tokens as in DeepSeek-V3's
    chained MTP modules).

    Args:
        head: The MTP head providing one logit tensor per prediction step.
        hidden_state: Backbone hidden states, ``(batch, seq, hidden_size)``.
        labels: Target token ids, ``(batch, seq)``; ``seq`` must be greater
            than ``num_future`` so every used step has targets.
        num_future: How many of the head's prediction steps to include;
            defaults to ``head.num_predictions``.
        weight_decay: Geometric per-step discount; step ``k`` is weighted by
            ``weight_decay ** k``. 1.0 weights all steps equally.
        logits_list: Optional pre-computed ``head(hidden_state)`` result, so
            callers that already ran the head (e.g. to return the logits) can
            avoid a second forward pass.

    Returns:
        Scalar loss tensor (weighted mean of per-step cross-entropies).

    """
    if num_future is None:
        num_future = head.num_predictions
    if not 1 <= num_future <= head.num_predictions:
        raise ValueError(
            f"num_future must be in [1, {head.num_predictions}], got {num_future}"
        )
    if weight_decay <= 0:
        raise ValueError("weight_decay must be > 0")
    if labels.shape != hidden_state.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match hidden_state "
            f"(batch, seq) {tuple(hidden_state.shape[:2])}"
        )
    if labels.size(1) <= num_future:
        raise ValueError("sequence length must be greater than num_future")

    if logits_list is None:
        logits_list = head(hidden_state)
    total = torch.zeros((), device=hidden_state.device)
    weight_sum = 0.0
    for step in range(num_future):
        logits = logits_list[step][:, : labels.size(1) - step - 1]
        targets = labels[:, step + 1 :]
        valid = targets.reshape(-1).ne(-100)
        if not valid.any():
            continue
        step_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        weight = weight_decay**step
        total = total + weight * step_loss
        weight_sum += weight
    if weight_sum == 0:
        return hidden_state.sum() * 0.0
    return total / weight_sum


__all__ = ["MTPDecoder", "MultiTokenPredictionHead", "mtp_loss"]
