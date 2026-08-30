"""CPU benchmark for the educational building blocks in ``llminfra.layers``.

Measures forward wall time for the normalization layers, FFN variants, the
Mamba2 selective SSM (recurrent and chunked scans), TransformerBlock, the
hybrid SSM/attention stacks and the mHC residual mixer across a shape matrix
covering short prefill, long prefill and decode-like single-step calls.

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/benchmark/bench_layers.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llminfra.layers.feed_forward import FeedForward, SwiGLUFFN
from llminfra.layers.gated_feed_forward import ClampedSwiGLUFFN, GeGLUFFN
from llminfra.layers.hybrid_layers import HybridLayerStack, HybridSSMBlock
from llminfra.layers.hyper_connection import ManifoldConstrainedHyperConnection
from llminfra.layers.mamba2 import Mamba2Layer
from llminfra.layers.normalization import LayerNorm, RMSNorm
from llminfra.layers.transformer_block import TransformerBlock

HIDDEN = 512
INTERMEDIATE = 1024
HEADS = 8
D_STATE = 16

# (name, batch, seq_len) covering short/long prefill and decode steps.
SHAPES = [
    ("b2-s128", 2, 128),
    ("b4-s512", 4, 512),
    ("b2-s2048", 2, 2048),
    ("b8-s1", 8, 1),
]


def _make_hidden(batch: int, seq_len: int, hidden: int = HIDDEN) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, seq_len, hidden, generator=generator)


def _median_time(fn: Callable[[], None], budget_seconds: float = 0.4) -> float:
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


def main() -> None:
    torch.manual_seed(0)
    modules: list[tuple[str, torch.nn.Module, Callable[[torch.Tensor], object]]] = [
        ("RMSNorm", RMSNorm(HIDDEN), lambda m, x: m(x)),
        ("LayerNorm", LayerNorm(HIDDEN), lambda m, x: m(x)),
        ("FeedForward", FeedForward(HIDDEN, INTERMEDIATE), lambda m, x: m(x)),
        ("SwiGLUFFN", SwiGLUFFN(HIDDEN, INTERMEDIATE), lambda m, x: m(x)),
        ("GeGLUFFN", GeGLUFFN(HIDDEN, INTERMEDIATE), lambda m, x: m(x)),
        ("ClampedSwiGLUFFN", ClampedSwiGLUFFN(HIDDEN, INTERMEDIATE), lambda m, x: m(x)),
        ("Mamba2-rec", Mamba2Layer(HIDDEN, d_state=D_STATE), lambda m, x: m(x)),
        (
            "Mamba2-chunk",
            Mamba2Layer(HIDDEN, d_state=D_STATE),
            lambda m, x: m(x, scan="chunked", chunk_size=64),
        ),
        (
            "TransformerBlock",
            TransformerBlock(HIDDEN, HEADS, INTERMEDIATE),
            lambda m, x: m(x),
        ),
        (
            "HybridSSMBlock",
            HybridSSMBlock(HIDDEN, num_heads=HEADS, pattern="ssm:ssm:attn"),
            lambda m, x: m(x),
        ),
        (
            "HybridLayerStack",
            HybridLayerStack(
                HIDDEN,
                num_heads=HEADS,
                intermediate_size=INTERMEDIATE,
                layer_map="linear:ssm:full",
            ),
            lambda m, x: m(x),
        ),
        (
            "mHC",
            ManifoldConstrainedHyperConnection(HIDDEN),
            lambda m, x: m(x, x),
        ),
    ]
    for module_name, module, call in modules:
        module = module.eval()
        row = f"{module_name:<18}"
        for _, batch, seq_len in SHAPES:
            x = _make_hidden(batch, seq_len)
            with torch.no_grad():
                ms = _median_time(partial(call, module, x))
            row += f"{ms * 1e3:>11.2f} "
        print(row, flush=True)


if __name__ == "__main__":
    print(f"{'module':<18}" + "".join(f"{name:>11} " for name, _, _ in SHAPES))
    print("-" * (18 + 12 * len(SHAPES)))
    main()
