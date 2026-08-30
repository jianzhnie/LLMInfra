"""CPU benchmark for the positional encoding modules.

Measures forward wall time for every positional module across realistic
shapes (batch 2-4, seq 512-2048, hidden 512). Each module is called
``LAYERS`` times per timed iteration to mimic a multi-layer model reusing
one module instance per layer stack.

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/bench_positional.py
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

# The package is not pip-installed in this repo; make the script runnable
# directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llminfra.positional import (
    ALiBiBias,
    DynamicNTKRotaryEmbedding,
    LearnedAbsolutePositionEmbedding,
    LongRoPEScaledRotaryEmbedding,
    MultiModalRotaryPositionEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    RotaryPositionEmbedding,
    SinusoidalPositionEmbedding,
    T5RelativePositionBias,
    TwoDimensionalPositionEmbedding,
    YaRNParameters,
    YaRNScaledRotaryEmbedding,
)
from llminfra.positional.rotary import apply_rotary_pos_emb

# Calls per timed iteration, emulating a module shared across layers.
LAYERS = 8


def _median_time(fn: Callable[[], None], budget_seconds: float = 0.6) -> float:
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
    return statistics.median(samples) / LAYERS


def _rope_family() -> list[tuple[str, Callable[[], object], torch.Tensor]]:
    """(name, factory, input) rows for the RoPE variants on (B, S, hidden)."""
    dim, max_seq = 512, 2048
    x = torch.randn(4, 512, dim)
    x_long = torch.randn(2, 2048, dim)
    yarn_params = YaRNParameters(factor=4.0, original_max_position_embeddings=1024)
    factors = [1.0] * (dim // 2)
    rows: list[tuple[str, Callable[[], object], torch.Tensor]] = [
        ("rope", lambda: RotaryPositionEmbedding(dim, max_seq_len=max_seq), x),
        (
            "rope-long",
            lambda: RotaryPositionEmbedding(dim, max_seq_len=max_seq),
            x_long,
        ),
        ("yarn", lambda: YaRNScaledRotaryEmbedding(dim, max_seq, yarn_params), x),
        (
            "ntk",
            lambda: DynamicNTKRotaryEmbedding(dim, 1024, max_seq_len=max_seq),
            x_long,
        ),
        (
            "partial-rope",
            lambda: PartialRotaryPositionEmbedding(dim, 0.25, max_seq_len=max_seq),
            x,
        ),
        (
            "interpolation",
            lambda: PositionInterpolation(dim, 1024, max_seq_len=max_seq),
            x,
        ),
        (
            "longrope",
            lambda: LongRoPEScaledRotaryEmbedding(dim, 1024, max_seq, factors, factors),
            x,
        ),
    ]
    return rows


def _run(module: object, x: torch.Tensor, *args: torch.Tensor) -> None:
    for _ in range(LAYERS):
        module(x, *args)  # type: ignore[operator]


def main() -> None:
    torch.manual_seed(0)
    print(f"median forward time per call over {LAYERS} layered calls (ms)")
    header = f"{'case':<16}{'ms/call':>10}"
    print(header)
    print("-" * len(header))

    for name, factory, x in _rope_family():
        module = factory()
        ms = _median_time(partial(_run, module, x))
        print(f"{name:<16}{ms * 1e3:>9.2f} ")

    # mRoPE on (batch, heads, seq, head_dim) with and without position ids.
    dim = 64
    section = (11, 11, 10)
    mrope = MultiModalRotaryPositionEmbedding(dim, section, max_seq_len=2048)
    x4 = torch.randn(2, 8, 512, dim)
    seq = x4.size(-2)
    position_ids = torch.stack(
        [
            torch.arange(seq).expand(2, -1),
            torch.arange(seq).expand(2, -1) // 4,
            torch.arange(seq).expand(2, -1) % 7,
        ]
    )
    ms = _median_time(partial(_run, mrope, x4))
    print(f"{'mrope-text':<16}{ms * 1e3:>9.2f} ")
    ms = _median_time(partial(_run, mrope, x4, position_ids))
    print(f"{'mrope-ids':<16}{ms * 1e3:>9.2f} ")

    # Standalone rotary kernel on a 4-D attention-shaped input.
    x_attn = torch.randn(2, 8, 1024, 64)
    positions = torch.arange(x_attn.size(-2), dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
    freqs = torch.outer(positions, inv_freq)
    cos_half, sin_half = freqs.cos(), freqs.sin()
    cos_full = cos_half.repeat_interleave(2, dim=-1)
    sin_full = sin_half.repeat_interleave(2, dim=-1)

    def _kernel(cos: torch.Tensor, sin: torch.Tensor) -> None:
        for _ in range(LAYERS):
            apply_rotary_pos_emb(x_attn, cos, sin)

    ms = _median_time(partial(_kernel, cos_half, sin_half))
    print(f"{'kernel-half-cos':<16}{ms * 1e3:>9.2f} ")
    ms = _median_time(partial(_kernel, cos_full, sin_full))
    print(f"{'kernel-full-cos':<16}{ms * 1e3:>9.2f} ")

    # Bias producers.
    alibi = ALiBiBias(num_heads=8, max_seq_len=2048)
    x_attn_small = torch.randn(2, 8, 512, 64)
    ms = _median_time(partial(_run, alibi, x_attn_small))
    print(f"{'alibi':<16}{ms * 1e3:>9.2f} ")

    t5 = T5RelativePositionBias(num_heads=8, max_seq_len=2048)
    ms = _median_time(partial(_run, t5, (512, 512)))
    print(f"{'t5-bias':<16}{ms * 1e3:>9.2f} ")

    # Additive embeddings on (batch, seq, hidden).
    hidden = 512
    x_h = torch.randn(4, 512, hidden)
    sinusoidal = SinusoidalPositionEmbedding(hidden, max_seq_len=2048)
    ms = _median_time(partial(_run, sinusoidal, x_h))
    print(f"{'sinusoidal':<16}{ms * 1e3:>9.2f} ")
    learned = LearnedAbsolutePositionEmbedding(hidden, max_seq_len=2048)
    ms = _median_time(partial(_run, learned, x_h))
    print(f"{'learned-abs':<16}{ms * 1e3:>9.2f} ")

    two_d = TwoDimensionalPositionEmbedding(hidden, 64, 32)
    ms = _median_time(partial(_run, two_d, x_h))
    print(f"{'two-d':<16}{ms * 1e3:>9.2f} ")


if __name__ == "__main__":
    main()
