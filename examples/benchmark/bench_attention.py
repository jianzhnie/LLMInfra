"""CPU benchmark for the ``llminfra.attention`` modules.

Measures forward wall time for the core attention classes across a shape
matrix covering typical prefill/decode sizes (batch 1-4, seq 128-2048,
hidden 256-512). Timings are medians over an adaptively chosen number of
repeats. Run from the repository root:

    python examples/benchmark/bench_attention.py
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

from llminfra.attention import (
    ALiBiAttention,
    BlockSparseAttention,
    CompressedSparseAttention,
    DynamicSparseAttention,
    GatedDeltaNet,
    GroupedQueryAttention,
    KimiDeltaAttention,
    LightningAttention,
    LinearAttention,
    MultiHeadAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
    RingAttention,
    SlidingWindowAttention,
)

# (name, batch, seq_len, hidden, heads)
SHAPES = [
    ("small", 4, 128, 256, 4),
    ("medium", 2, 512, 512, 8),
    ("large", 2, 1024, 512, 8),
    ("long", 1, 2048, 512, 8),
]


def _impls(hidden: int, heads: int) -> dict[str, torch.nn.Module]:
    """Build one instance of every attention module for the given shape."""
    kv_groups = max(1, heads // 4)
    return {
        "mha": MultiHeadAttention(hidden, heads, dropout=0.0),
        "gqa": GroupedQueryAttention(hidden, heads, kv_groups, dropout=0.0),
        "mqa": MultiQueryAttention(hidden, heads, dropout=0.0),
        "mla": MultiHeadLatentAttention(hidden, heads, hidden // 2, hidden // 4),
        "alibi": ALiBiAttention(hidden, heads, num_kv_groups=kv_groups),
        "swa": SlidingWindowAttention(hidden, heads, 64, kv_groups, dropout=0.0),
        "linear": LinearAttention(hidden, heads),
        "lightning": LightningAttention(hidden, heads),
        "gdn": GatedDeltaNet(hidden, heads),
        "kda": KimiDeltaAttention(hidden, heads),
        "ring": RingAttention(hidden, heads, num_chunks=4),
        "bsa": BlockSparseAttention(hidden, heads, 64, kv_groups, top_k=4),
        "csa": CompressedSparseAttention(hidden, heads, 4, kv_groups, top_k=4),
        "dsa": DynamicSparseAttention(
            hidden, heads, block_size=64, top_k=4, num_kv_groups=kv_groups
        ),
    }


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
    return statistics.median(samples)


def main() -> None:
    torch.manual_seed(0)
    names: list[str] | None = None
    for shape_name, batch, seq_len, hidden, heads in SHAPES:
        generator = torch.Generator().manual_seed(0)
        hidden_state = torch.randn(batch, seq_len, hidden, generator=generator)
        padding_mask = torch.rand(batch, 1, 1, seq_len, generator=generator) > 0.2
        padding_mask[..., 0] = True
        impls = {name: m.eval() for name, m in _impls(hidden, heads).items()}
        if names is None:
            names = list(impls)
            header = f"{'shape':<9}{'variant':<9}" + "".join(
                f"{name:>11}" for name in names
            )
            print(header)
            print("-" * len(header))
        for variant, mask in (("plain", None), ("padding", padding_mask)):
            row = f"{shape_name:<9}{variant:<9}"
            for name in names:
                if name == "ring" and variant == "padding":
                    row += f"{'n/a':>11}"
                    continue
                module = impls[name]
                timed = partial(
                    lambda m, h, am: m(h, attention_mask=am),
                    module,
                    hidden_state,
                    mask,
                )
                with torch.no_grad():
                    ms = _median_time(timed)
                row += f"{ms * 1e3:>10.1f} "
            print(row, flush=True)
        print()


if __name__ == "__main__":
    main()
