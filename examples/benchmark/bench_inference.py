"""CPU benchmark for the educational inference components.

Measures wall time for the hot paths of `llminfra.inference`:

- ``PagedAttentionCache.append`` (prefill chunks and single-token decode)
- ``PagedAttentionCache.get`` (dense gather of a cached sequence)
- ``paged_attention`` (prefill and decode query shapes)
- ``BlockSparseIndexer.forward`` (batched top-k block selection)
- ``TieredKVCache.put/get`` (bulk put with evictions, CPU-tier promotion)

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/benchmark/bench_inference.py
"""

from __future__ import annotations

import itertools
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch

# The package is not pip-installed in this repo; make the script runnable
# directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llminfra.inference import (
    BlockSparseIndexer,
    PagedAttentionCache,
    TieredKVCache,
    paged_attention,
)

NUM_HEADS = 8
HEAD_DIM = 64
BLOCK_SIZE = 16


def _make_cache(num_tokens_capacity: int) -> PagedAttentionCache:
    num_blocks = 2 * (num_tokens_capacity // BLOCK_SIZE + 1)
    return PagedAttentionCache(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )


def _bench_paged_append(seq_len: int, chunk: int) -> float:
    """Append ``seq_len`` tokens in chunks of ``chunk`` (chunk=1 is decode)."""
    cache = _make_cache(seq_len)
    key = torch.randn(seq_len, NUM_HEADS, HEAD_DIM)
    value = torch.randn(seq_len, NUM_HEADS, HEAD_DIM)

    def run() -> None:
        cache.reset(0)
        for start in range(0, seq_len, chunk):
            cache.append(0, key[start : start + chunk], value[start : start + chunk])

    return _median_time(run)


def _bench_paged_get(seq_len: int) -> float:
    cache = _make_cache(seq_len)
    cache.append(
        0,
        torch.randn(seq_len, NUM_HEADS, HEAD_DIM),
        torch.randn(seq_len, NUM_HEADS, HEAD_DIM),
    )
    return _median_time(lambda: cache.get(0))


def _bench_paged_attention(q_len: int, kv_len: int) -> float:
    cache = _make_cache(kv_len)
    cache.append(
        0,
        torch.randn(kv_len, NUM_HEADS, HEAD_DIM),
        torch.randn(kv_len, NUM_HEADS, HEAD_DIM),
    )
    query = torch.randn(q_len, NUM_HEADS, HEAD_DIM)
    timed = partial(
        paged_attention,
        query,
        cache.key_cache,
        cache.value_cache,
        cache.block_tables[0],
        kv_len,
        BLOCK_SIZE,
    )
    return _median_time(timed)


def _bench_indexer(batch: int, seq_len: int, hidden: int, block_size: int) -> float:
    indexer = BlockSparseIndexer(
        hidden,
        NUM_HEADS,
        block_size=block_size,
        top_k=8,
        max_seq_len=seq_len,
        causal=True,
    )
    hidden_state = torch.randn(batch, seq_len, hidden)

    def run() -> None:
        with torch.no_grad():
            indexer(hidden_state)

    return _median_time(run)


def _bench_tiered(num_entries: int, seq_len: int) -> tuple[float, float]:
    with tempfile.TemporaryDirectory(prefix="llminfra_bench_kv_") as tmpdir:
        cache = TieredKVCache(
            tmpdir,
            max_hbm_entries=num_entries // 2,
            max_cpu_entries=num_entries,
            hbm_device="cpu",
        )
        tensors = [
            (
                torch.randn(seq_len, NUM_HEADS, HEAD_DIM),
                torch.randn(seq_len, NUM_HEADS, HEAD_DIM),
            )
            for _ in range(num_entries)
        ]

        def put_all() -> None:
            for seq_id, (key, value) in enumerate(tensors):
                cache.put(seq_id, key, value)

        put_all()
        put_ms = _median_time(put_all)
        # Cycle through every entry: HBM holds only half of them, so each
        # timed get forces a CPU-tier promotion instead of an HBM hit.
        get_order = itertools.cycle(range(num_entries))
        get_ms = _median_time(lambda: cache.get(next(get_order)))
        return put_ms, get_ms


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


def main() -> None:
    torch.manual_seed(0)
    rows: list[tuple[str, float]] = []

    def add(name: str, ms: float) -> None:
        rows.append((name, ms))

    for seq_len in (128, 512, 2048):
        add(
            f"append prefill seq={seq_len} (one shot)",
            _bench_paged_append(seq_len, seq_len),
        )
        add(
            f"append decode seq={seq_len} (token-at-a-time)",
            _bench_paged_append(seq_len, 1),
        )
        add(f"get seq={seq_len}", _bench_paged_get(seq_len))

    attention_shapes = ((128, 128), (512, 512), (1, 512), (1, 2048), (128, 2048))
    for q_len, kv_len in attention_shapes:
        add(
            f"paged_attention q={q_len} kv={kv_len}",
            _bench_paged_attention(q_len, kv_len),
        )

    indexer_shapes = (
        (2, 512, 256, 32),
        (8, 512, 256, 32),
        (2, 2048, 512, 32),
        (4, 1024, 1024, 64),
    )
    for batch, seq_len, hidden, block_size in indexer_shapes:
        add(
            f"indexer b={batch} s={seq_len} h={hidden} bs={block_size}",
            _bench_indexer(batch, seq_len, hidden, block_size),
        )

    put_ms, get_ms = _bench_tiered(num_entries=16, seq_len=512)
    add("tiered put x16 seq=512 (incl. evictions)", put_ms)
    add("tiered get (promotion)", get_ms)

    header = f"{'benchmark':<48}{'median ms':>12}"
    print(header)
    print("-" * len(header))
    for name, ms in rows:
        print(f"{name:<48}{ms * 1e3:>11.2f} ")


if __name__ == "__main__":
    main()
