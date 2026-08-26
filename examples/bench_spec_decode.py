"""CPU benchmark for the speculative decoding interfaces.

Covers the draft-then-verify decoders (``SpeculativeDecoder``,
``EagleSpeculator``, ``MTPDecoder``, ``NGramSpeculator``) plus the
trainable draft heads (``MedusaHead`` / ``medusa_loss``,
``MultiTokenPredictionHead`` / ``mtp_loss``) across a shape matrix of
realistic batch / sequence / hidden sizes.

The draft and target models are tiny embedding+MLP networks so the timed
region includes realistic model compute next to the decoding harness code
being optimized.

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/bench_spec_decode.py
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
from torch import nn

# The package is not pip-installed in this repo; make the script runnable
# directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llminfra.spec_decode import (
    EagleSpeculator,
    MedusaHead,
    MTPDecoder,
    MultiTokenPredictionHead,
    NGramSpeculator,
    SpeculativeDecoder,
    medusa_loss,
    mtp_loss,
)

VOCAB_SIZE = 512
NUM_SPECULATIVE_TOKENS = 4


class TinyLM(nn.Module):
    """Deterministic stand-in for a draft/target causal LM."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.mlp(self.embedding(input_ids)))


def _make_ids(batch: int, seq: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (batch, seq), generator=generator)


def _make_hidden(batch: int, seq: int, hidden_size: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, hidden_size, generator=generator)


def _median_time(fn: Callable[[], None], budget_seconds: float = 0.5) -> float:
    """Median wall time of ``fn``; repeat count adapts to one timed warmup."""
    start = time.perf_counter()
    fn()
    warmup = time.perf_counter() - start
    repeats = max(1, min(7, int(budget_seconds / max(warmup, 1e-6))))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def bench_decoders() -> None:
    """Wall time of one speculative block for each decoder interface."""
    # (name, batch, seq, hidden)
    shapes = [
        ("b2-s128", 2, 128, 256),
        ("b4-s512", 4, 512, 256),
        ("b8-s512", 8, 512, 256),
        ("b2-s2048", 2, 2048, 256),
        ("b4-s1024-h512", 4, 1024, 512),
    ]
    header = f"{'shape':<16}" + "".join(
        f"{name:>14}"
        for name in ("spec-greedy", "spec-temp1.0", "eagle", "mtp", "ngram")
    )
    print(header)
    print("-" * len(header))
    for shape_name, batch, seq, hidden in shapes:
        torch.manual_seed(0)
        draft_model = TinyLM(VOCAB_SIZE, hidden).eval()
        target_model = TinyLM(VOCAB_SIZE, hidden).eval()
        draft_head = nn.Linear(hidden, VOCAB_SIZE).eval()
        mtp_head = MultiTokenPredictionHead(hidden, VOCAB_SIZE, NUM_SPECULATIVE_TOKENS)
        ids = _make_ids(batch, seq)
        hidden_state = _make_hidden(batch, seq, hidden)

        spec_greedy = SpeculativeDecoder(
            draft_model, target_model, NUM_SPECULATIVE_TOKENS
        )
        spec_temp = SpeculativeDecoder(
            draft_model, target_model, NUM_SPECULATIVE_TOKENS, temperature=1.0
        )
        eagle = EagleSpeculator(draft_head, target_model, NUM_SPECULATIVE_TOKENS)
        mtp = MTPDecoder(mtp_head, target_model)
        ngram = NGramSpeculator(target_model, ngram_size=2, num_speculative_tokens=4)

        cases: list[tuple[str, Callable[[], None]]] = [
            ("spec-greedy", partial(spec_greedy, ids)),
            ("spec-temp1.0", partial(spec_temp, ids)),
            ("eagle", partial(eagle, ids, hidden_state)),
            ("mtp", partial(mtp, ids, hidden_state)),
            ("ngram", partial(ngram, ids)),
        ]
        row = f"{shape_name:<16}"
        for _name, fn in cases:
            row += f"{_median_time(fn) * 1e3:>13.1f} "
        print(row, flush=True)


def bench_heads() -> None:
    """Wall time of the trainable heads and their losses (fwd and fwd+bwd)."""
    # (name, batch, seq, hidden)
    shapes = [
        ("b2-s128-h256", 2, 128, 256),
        ("b8-s512-h256", 8, 512, 256),
        ("b4-s1024-h512", 4, 1024, 512),
        ("b2-s2048-h1024", 2, 2048, 1024),
    ]
    header = f"{'shape':<16}" + "".join(
        f"{name:>12}"
        for name in ("medusa-fwd", "medusa+bwd", "candidates", "mtp-fwd", "mtp+bwd")
    )
    print(header)
    print("-" * len(header))
    for shape_name, batch, seq, hidden in shapes:
        torch.manual_seed(0)
        medusa = MedusaHead(hidden, VOCAB_SIZE, num_heads=4)
        mtp = MultiTokenPredictionHead(hidden, VOCAB_SIZE, num_predictions=4)
        hidden_state = _make_hidden(batch, seq, hidden)
        labels = _make_ids(batch, seq)

        def medusa_fwd(
            head: MedusaHead = medusa, state: torch.Tensor = hidden_state
        ) -> None:
            with torch.no_grad():
                head(state)

        def medusa_fwd_bwd(
            head: MedusaHead = medusa,
            state: torch.Tensor = hidden_state,
            target: torch.Tensor = labels,
        ) -> None:
            medusa_loss(head, state.requires_grad_(), target)

        def candidates(
            head: MedusaHead = medusa, state: torch.Tensor = hidden_state
        ) -> None:
            head.generate_candidates(state, top_k=4)

        def mtp_fwd(
            head: MultiTokenPredictionHead = mtp, state: torch.Tensor = hidden_state
        ) -> None:
            with torch.no_grad():
                head(state)

        def mtp_fwd_bwd(
            head: MultiTokenPredictionHead = mtp,
            state: torch.Tensor = hidden_state,
            target: torch.Tensor = labels,
        ) -> None:
            mtp_loss(head, state.requires_grad_(), target)

        row = f"{shape_name:<16}"
        for fn in (medusa_fwd, medusa_fwd_bwd, candidates, mtp_fwd, mtp_fwd_bwd):
            row += f"{_median_time(fn) * 1e3:>11.1f} "
        print(row, flush=True)


def main() -> None:
    bench_decoders()
    print()
    bench_heads()


if __name__ == "__main__":
    main()
