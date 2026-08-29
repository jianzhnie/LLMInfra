"""Sampling logits processors and beam search for autoregressive decoding.

The logits processors follow the semantics of Hugging Face
``transformers.generation.logits_process``: each processor is a composable
transform on ``(batch, vocab)`` logits, optionally conditioned on the token
ids generated so far. :func:`beam_search` is a teaching-level implementation
of standard beam search with GNMT length normalization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn.functional as F


class LogitsProcessor(ABC):
    """Base class for composable transforms on next-token logits.

    A processor maps ``(batch, vocab)`` logits to logits of the same shape.
    ``input_ids`` carries the full sequences generated so far, prompt
    included, shaped ``(batch, seq_len)``; processors that need no context
    ignore it.
    """

    @abstractmethod
    def __call__(
        self, logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform ``logits``, optionally conditioned on ``input_ids``."""
        ...


class TopKLogitsProcessor(LogitsProcessor):
    """Keep only the ``k`` highest logits; all others become ``-inf``.

    Reference: Fan et al., "Hierarchical Neural Story Generation",
    arXiv:1805.04833.

    Args:
        k: Number of highest-scoring tokens to keep per row.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mask every logit below the ``k``-th largest one with ``-inf``."""
        if self.k >= logits.size(-1):
            return logits
        kth_value = torch.topk(logits, self.k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < kth_value, float("-inf"))


class TopPLogitsProcessor(LogitsProcessor):
    """Nucleus filtering: drop the tail beyond cumulative probability ``p``.

    The vocabulary is sorted by probability and the smallest prefix whose
    cumulative probability strictly exceeds ``p`` is kept; the token that
    crosses the threshold is kept as well (it is only masked from the next
    position on), matching ``transformers`` semantics. At least the argmax
    token always survives.

    Reference: Holtzman et al., "The Curious Case of Neural Text
    Degeneration", arXiv:1904.09751.

    Args:
        p: Cumulative probability threshold in ``(0, 1]``.
    """

    def __init__(self, p: float) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError("p must be in (0, 1]")
        self.p = float(p)

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mask the lowest-probability tokens whose cumulative mass is past ``p``."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs > self.p
        # Shift the mask right so the token that first crosses ``p`` is kept.
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(
            -1, sorted_indices, sorted_remove
        )
        return logits.masked_fill(remove, float("-inf"))


class MinPLogitsProcessor(LogitsProcessor):
    """Drop tokens whose probability is below ``min_p`` times the top one.

    The threshold scales with the row's maximum probability, so confident
    distributions are truncated aggressively while flat ones keep most of
    the vocabulary.

    Reference: "Turning Up the Heat: Min-p Sampling for Creative and
    Coherent LLM Outputs", arXiv:2407.01082.

    Args:
        min_p: Relative probability threshold in ``(0, 1]``.
    """

    def __init__(self, min_p: float) -> None:
        if not 0.0 < min_p <= 1.0:
            raise ValueError("min_p must be in (0, 1]")
        self.min_p = float(min_p)

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mask tokens with ``prob < min_p * max_prob`` with ``-inf``."""
        probs = F.softmax(logits, dim=-1)
        threshold = self.min_p * probs.amax(dim=-1, keepdim=True)
        return logits.masked_fill(probs < threshold, float("-inf"))


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """Penalize tokens that already appeared in the sequence.

    The logit of an already-seen token is divided by ``penalty`` when it is
    positive and multiplied by it when negative, so the penalty always moves
    the probability of repeated tokens down (CTRL-style, matching
    ``transformers``). The prompt counts as "already seen".

    Reference: Keskar et al., "CTRL: A Conditional Transformer Language
    Model for Controllable Generation", arXiv:1909.05858.

    Args:
        penalty: Multiplicative penalty; ``1.0`` is a no-op.
    """

    def __init__(self, penalty: float) -> None:
        if penalty <= 0.0:
            raise ValueError("penalty must be > 0")
        self.penalty = float(penalty)

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the penalty to the logits of every token in ``input_ids``."""
        if input_ids is None:
            raise ValueError("RepetitionPenaltyLogitsProcessor requires input_ids")
        selected = logits.gather(-1, input_ids)
        penalized = torch.where(
            selected > 0, selected / self.penalty, selected * self.penalty
        )
        return logits.scatter(-1, input_ids, penalized)


class LogitsProcessorList(list[LogitsProcessor]):
    """Ordered composition of logits processors.

    Processors apply in list order, each receiving the previous one's
    output, mirroring ``transformers.LogitsProcessorList``.
    """

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply every processor in order and return the final logits."""
        for processor in self:
            logits = processor(logits, input_ids)
        return logits


def beam_search(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    num_beams: int,
    max_new_tokens: int,
    length_penalty_alpha: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    processors: LogitsProcessorList | None = None,
) -> torch.Tensor:
    """Run standard beam search over a step-wise logits function.

    This is a teaching implementation: ``logits_fn`` is re-evaluated on the
    full sequence at every step (no KV cache) and each batch row is searched
    independently. Candidate scores are cumulative log-probabilities.
    Hypotheses that emit ``eos_token_id`` are moved to a per-row heap and
    stop expanding; a row is finished once its heap holds ``num_beams``
    hypotheses. At the end, every row picks the best hypothesis among its
    heap and its still-live beams, scored by cumulative log-probability
    divided by the GNMT length penalty ``((5 + len) / 6) ** alpha``
    (Wu et al., 2016, arXiv:1609.08144), where ``len`` counts the generated
    tokens including a terminating eos. ``length_penalty_alpha = 0``
    disables normalization (pure cumulative log-probability).

    Args:
        logits_fn: Maps ``(batch * num_beams, seq_len)`` token ids to
            ``(batch * num_beams, vocab)`` next-token logits.
        input_ids: Prompt token ids of shape ``(batch, prompt_len)``.
        num_beams: Beam width, i.e. live hypotheses kept per batch row.
        max_new_tokens: Maximum number of tokens generated per row.
        length_penalty_alpha: Exponent of the GNMT length penalty.
        eos_token_id: Optional stop token. ``None`` means every hypothesis
            runs for exactly ``max_new_tokens`` steps.
        pad_token_id: Token appended to finished rows during the loop and
            used to right-pad the returned batch to a common length.
        processors: Optional logits processors applied to the raw logits
            before scoring (e.g. top-k filtering of beam candidates).

    Returns:
        Long tensor of shape ``(batch, prompt_len + num_generated)`` with
        the best hypothesis per row, right-padded with ``pad_token_id``.

    Raises:
        ValueError: If the shapes or hyper-parameters are invalid.

    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch, seq_len)")
    if num_beams < 1:
        raise ValueError("num_beams must be >= 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if length_penalty_alpha < 0.0:
        raise ValueError("length_penalty_alpha must be >= 0")

    def length_penalty(length: int) -> float:
        return float(((5.0 + length) / 6.0) ** length_penalty_alpha)

    batch_size, prompt_len = input_ids.shape
    device = input_ids.device
    sequences = (
        input_ids[:, None, :]
        .expand(batch_size, num_beams, prompt_len)
        .reshape(batch_size * num_beams, prompt_len)
        .contiguous()
    )
    # Beam 0 starts as the only live hypothesis; the rest enter after the
    # first expansion through their -inf initial score.
    beam_scores = torch.full((batch_size, num_beams), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_lengths = torch.zeros(batch_size, num_beams, dtype=torch.long, device=device)
    row_done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    # Per-row heaps of (normalized_score, length, sequence) for hypotheses
    # that ended with eos.
    heaps: list[list[tuple[float, int, torch.Tensor]]] = [[] for _ in range(batch_size)]

    for _ in range(max_new_tokens):
        logits = logits_fn(sequences)
        vocab_size = logits.size(-1)
        if processors is not None and len(processors) > 0:
            logits = processors(logits, sequences)
        log_probs = F.log_softmax(logits.float(), dim=-1).view(
            batch_size, num_beams, vocab_size
        )
        candidate_scores = beam_scores[:, :, None] + log_probs
        cur_len = sequences.size(1)

        next_scores = torch.zeros_like(beam_scores)
        next_lengths = torch.zeros_like(beam_lengths)
        next_tokens: list[torch.Tensor] = []
        for row in range(batch_size):
            row_seqs = sequences.view(batch_size, num_beams, cur_len)[row]
            if row_done[row]:
                # Frozen row: keep the beams as-is and pad to stay in shape.
                pad_col = torch.full_like(row_seqs[:, :1], pad_token_id)
                next_tokens.append(torch.cat([row_seqs, pad_col], dim=-1))
                next_scores[row] = beam_scores[row]
                next_lengths[row] = beam_lengths[row]
                continue
            flat_scores = candidate_scores[row].reshape(-1)
            top_scores, top_indices = flat_scores.topk(
                min(2 * num_beams, flat_scores.numel())
            )
            # Split the top 2*num_beams candidates into eos continuations
            # (heap) and live continuations (next beams).
            chosen: list[tuple[float, int, int]] = []
            for score, flat_index in zip(
                top_scores.tolist(), top_indices.tolist(), strict=True
            ):
                beam_index, token = divmod(flat_index, vocab_size)
                if eos_token_id is not None and token == eos_token_id:
                    length = int(beam_lengths[row, beam_index]) + 1
                    finished_seq = torch.cat(
                        [row_seqs[beam_index], row_seqs.new_tensor([token])]
                    )
                    heaps[row].append(
                        (score / length_penalty(length), length, finished_seq)
                    )
                    continue
                chosen.append((score, beam_index, token))
                if len(chosen) == num_beams:
                    break
            if not chosen:
                # Every candidate ended in eos: nothing left to expand.
                row_done[row] = True
                pad_col = torch.full_like(row_seqs[:, :1], pad_token_id)
                next_tokens.append(torch.cat([row_seqs, pad_col], dim=-1))
                next_scores[row] = beam_scores[row]
                next_lengths[row] = beam_lengths[row]
                continue
            # Fewer than num_beams live candidates survived (the rest were
            # eos): repeat the worst one to keep the beam count constant.
            while len(chosen) < num_beams:
                chosen.append(chosen[-1])
            next_tokens.append(
                torch.stack(
                    [
                        torch.cat([row_seqs[beam_index], row_seqs.new_tensor([token])])
                        for _, beam_index, token in chosen
                    ]
                )
            )
            next_scores[row] = torch.tensor(
                [score for score, _, _ in chosen], device=device
            )
            next_lengths[row] = torch.tensor(
                [int(beam_lengths[row, beam_index]) + 1 for _, beam_index, _ in chosen],
                device=device,
            )
            if len(heaps[row]) >= num_beams:
                row_done[row] = True

        sequences = torch.cat(next_tokens, dim=0)
        beam_scores = next_scores
        beam_lengths = next_lengths
        if bool(row_done.all()):
            break

    results: list[torch.Tensor] = []
    final_len = sequences.size(1)
    for row in range(batch_size):
        candidates: list[tuple[float, torch.Tensor]] = [
            (norm_score, seq) for norm_score, _, seq in heaps[row]
        ]
        row_seqs = sequences.view(batch_size, num_beams, final_len)[row]
        for beam in range(num_beams):
            candidates.append(
                (
                    float(beam_scores[row, beam])
                    / length_penalty(int(beam_lengths[row, beam])),
                    row_seqs[beam],
                )
            )
        # Ties prefer finished (heap) hypotheses over still-live beams.
        results.append(max(candidates, key=lambda candidate: candidate[0])[1])

    out_len = max(seq.size(0) for seq in results)
    output = torch.full(
        (batch_size, out_len), pad_token_id, dtype=input_ids.dtype, device=device
    )
    for row, seq in enumerate(results):
        output[row, : seq.size(0)] = seq
    return output


__all__ = [
    "LogitsProcessor",
    "LogitsProcessorList",
    "MinPLogitsProcessor",
    "RepetitionPenaltyLogitsProcessor",
    "TopKLogitsProcessor",
    "TopPLogitsProcessor",
    "beam_search",
]
