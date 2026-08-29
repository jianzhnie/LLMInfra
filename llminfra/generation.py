"""Sampling logits processors and simple generate loops for decoding.

The logits processors follow the semantics of Hugging Face
``transformers.generation.logits_process``: each processor is a composable
transform on ``(batch, vocab)`` logits, optionally conditioned on the token
ids generated so far. :func:`generate` is the naive greedy/sampling teaching
loop (full recompute per step); :func:`generate_with_cache` is its KV-cache
counterpart, driving a stateful step function (prefill once, then one new
token per step).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

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


def generate(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    processors: LogitsProcessorList | None = None,
) -> torch.Tensor:
    """Generate tokens greedily or by sampling.

    Every step keeps exactly one continuation per batch row. With the
    default ``temperature = 0`` each step takes the argmax (greedy decoding);
    otherwise the next token is drawn from the softmax of the processed
    logits divided by ``temperature``.

    This is a teaching implementation: ``logits_fn`` is re-evaluated on the
    full sequence at every step (no KV cache).

    Args:
        logits_fn: Maps ``(batch, seq_len)`` token ids to ``(batch, vocab)``
            next-token logits.
        input_ids: Prompt token ids of shape ``(batch, prompt_len)``.
        max_new_tokens: Maximum number of tokens generated per row.
        temperature: Sampling temperature; the default 0 selects greedy
            argmax decoding.
        eos_token_id: Optional stop token. A row stops right after emitting
            it; all remaining positions in that row are filled with
            ``pad_token_id``.
        pad_token_id: Token used to fill row positions after
            ``eos_token_id``.
        processors: Optional logits processors applied to the raw logits
            before sampling (e.g. top-k / top-p filtering).

    Returns:
        Long tensor of shape ``(batch, prompt_len + num_generated)`` with
        ``num_generated <= max_new_tokens``, right-padded with
        ``pad_token_id``.

    Raises:
        ValueError: If the shapes or hyper-parameters are invalid.

    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch, seq_len)")
    if input_ids.size(1) < 1:
        raise ValueError("input_ids must contain at least one token")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")

    sequences = input_ids
    finished = torch.zeros(sequences.size(0), dtype=torch.bool, device=sequences.device)
    for _ in range(max_new_tokens):
        logits = logits_fn(sequences)
        if processors is not None and len(processors) > 0:
            logits = processors(logits, sequences)
        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, 1).squeeze(-1)
        # Finished rows keep emitting pad so every row stays the same length.
        next_token = torch.where(
            finished, next_token.new_full((), pad_token_id), next_token
        )
        sequences = torch.cat([sequences, next_token[:, None]], dim=-1)
        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if bool(finished.all()):
                break
    return sequences


# Opaque per-call cache state threaded through ``generate_with_cache``; the
# caller (e.g. a model's ``forward_with_cache``) decides what it holds.
CacheState = TypeVar("CacheState")


def generate_with_cache(
    step_fn: Callable[
        [torch.Tensor, CacheState | None], tuple[torch.Tensor, CacheState]
    ],
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    processors: LogitsProcessorList | None = None,
) -> torch.Tensor:
    """Generate tokens greedily or by sampling, reusing a KV cache.

    This is the cached counterpart of :func:`generate`: instead of a
    ``logits_fn`` that re-reads the whole sequence every step, it drives a
    stateful ``step_fn``. The first call is the prefill pass — it receives
    the full prompt with ``past=None`` — and every later call receives only
    the newly generated ``(batch, 1)`` tokens plus the cache state returned
    by the previous call. Sampling, logits processing and eos/pad handling
    are identical to :func:`generate`, so a faithful ``step_fn`` produces
    the same tokens in O(1) sequence-length work per step.

    For :class:`llminfra.models.CausalLMModel`, the step function is simply
    ``model._forward_with_cache`` with the last-position logits sliced off;
    ``model.generate(use_cache=True)`` wraps exactly that.

    Args:
        step_fn: Maps ``(tokens, past)`` to ``(logits, new_past)`` where
            ``tokens`` is the full prompt on the first call and ``(batch, 1)``
            afterwards, ``logits`` holds the next-token logits of shape
            ``(batch, vocab)`` for the last fed position, and ``past`` is an
            opaque cache state threaded between calls.
        input_ids: Prompt token ids of shape ``(batch, prompt_len)``.
        max_new_tokens: Maximum number of tokens generated per row.
        temperature: Sampling temperature; the default 0 selects greedy
            argmax decoding.
        eos_token_id: Optional stop token. A row stops right after emitting
            it; all remaining positions in that row are filled with
            ``pad_token_id``.
        pad_token_id: Token used to fill row positions after
            ``eos_token_id``.
        processors: Optional logits processors applied to the raw logits
            before sampling (e.g. top-k / top-p filtering).

    Returns:
        Long tensor of shape ``(batch, prompt_len + num_generated)`` with
        ``num_generated <= max_new_tokens``, right-padded with
        ``pad_token_id``.

    Raises:
        ValueError: If the shapes or hyper-parameters are invalid.

    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch, seq_len)")
    if input_ids.size(1) < 1:
        raise ValueError("input_ids must contain at least one token")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")

    sequences = input_ids
    finished = torch.zeros(sequences.size(0), dtype=torch.bool, device=sequences.device)
    past_key_values: CacheState | None = None
    tokens = input_ids  # prefill consumes the full prompt
    for _ in range(max_new_tokens):
        logits, past_key_values = step_fn(tokens, past_key_values)
        if processors is not None and len(processors) > 0:
            logits = processors(logits, sequences)
        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, 1).squeeze(-1)
        # Finished rows keep emitting pad so every row stays the same length.
        next_token = torch.where(
            finished, next_token.new_full((), pad_token_id), next_token
        )
        sequences = torch.cat([sequences, next_token[:, None]], dim=-1)
        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if bool(finished.all()):
                break
        tokens = next_token[:, None]  # decode steps feed only the new token
    return sequences


__all__ = [
    "LogitsProcessor",
    "LogitsProcessorList",
    "MinPLogitsProcessor",
    "RepetitionPenaltyLogitsProcessor",
    "TopKLogitsProcessor",
    "TopPLogitsProcessor",
    "generate",
    "generate_with_cache",
]
