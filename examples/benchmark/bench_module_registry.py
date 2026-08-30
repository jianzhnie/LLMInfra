"""CPU micro-benchmark for ``llminfra.module_registry`` dispatch overhead.

The registry functions are called only at model-construction time, so this
benchmark exists to demonstrate (rather than exploit) where the time goes:
the dict lookup and name validation are nanoseconds; essentially all of the
per-call cost lives in the concrete module constructors, which belong to the
attention/positional subpackages and are out of scope here.

Run from the repository root:

    python examples/benchmark/bench_module_registry.py
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

# The package is not pip-installed in this repo; make the script runnable
# directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llminfra.module_registry import (
    ATTENTION_REGISTRY,
    build_attention,
    build_positional_encoding,
    list_attentions,
)


def _bench_us(fn: Callable[[], object], n: int = 2000) -> float:
    """Median per-call wall time in microseconds over several batches."""
    fn()
    fn()
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(n):
            fn()
        samples.append((time.perf_counter() - start) / n * 1e6)
    return statistics.median(samples)


def main() -> None:
    rows = [
        ("registry dict lookup", lambda: ATTENTION_REGISTRY["mha"]),
        ("list_attentions()", list_attentions),
        (
            "build_attention('mha')",
            lambda: build_attention("mha", hidden_size=256, num_heads=8),
        ),
        (
            "build_positional_encoding('rope')",
            lambda: build_positional_encoding("rope", dim=64),
        ),
    ]
    print(f"{'call':<38}{'us/call':>10}")
    print("-" * 48)
    for name, fn in rows:
        print(f"{name:<38}{_bench_us(fn):>9.3f} ")
    print(
        "\nNote: build_* time is constructor time in the attention/positional\n"
        "subpackages; registry dispatch itself is the first two rows."
    )


if __name__ == "__main__":
    main()
