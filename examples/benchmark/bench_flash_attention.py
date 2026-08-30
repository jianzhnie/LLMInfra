"""CPU benchmark for the educational FlashAttention versions (fa1-fa4).

Measures forward and forward+backward wall time for each tiled version and
the dense `reference_attention` baseline across a shape matrix covering
square prefill, single-query decode, long sequences, several head dims,
causal / non-causal and padding-mask variants.

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/bench_flash_attention.py
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

from llminfra.flash_attention import ATTENTION_FN_REGISTRY, list_versions
from llminfra.flash_attention.common import FlashAttentionConfig, reference_attention

# Realistic tile sizes so every version actually loops over multiple tiles.
# keep_debug_state=False measures the hot path without per-tile host reads.
CONFIG = FlashAttentionConfig(block_size_q=64, block_size_kv=64, keep_debug_state=False)

# (name, batch, heads, q_len, kv_len, head_dim)
SHAPES = [
    ("prefill-128", 2, 8, 128, 128, 64),
    ("prefill-512", 2, 8, 512, 512, 64),
    ("prefill-1024", 1, 8, 1024, 1024, 64),
    ("decode-512", 2, 8, 1, 512, 64),
    ("decode-2048", 2, 8, 1, 2048, 64),
    ("long-2048", 1, 8, 2048, 2048, 64),
    ("hd32-512", 2, 8, 512, 512, 32),
    ("hd128-512", 1, 8, 512, 512, 128),
]

# (name, causal, use_padding_mask)
VARIANTS = [
    ("plain", False, False),
    ("causal", True, False),
    ("padding", False, True),
]

IMPLS = [*list_versions(), "reference"]


def _make_inputs(
    batch: int, heads: int, q_len: int, kv_len: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(batch, heads, q_len, head_dim, generator=generator)
    k = torch.randn(batch, heads, kv_len, head_dim, generator=generator)
    v = torch.randn(batch, heads, kv_len, head_dim, generator=generator)
    mask = torch.rand(batch, kv_len, generator=generator) > 0.3
    mask[:, 0] = True  # keep at least one valid key per batch row
    return q, k, v, mask


def _run_once(
    impl: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    mask: torch.Tensor | None,
    *,
    backward: bool,
) -> None:
    q_ref, k_ref, v_ref = (t.detach().requires_grad_(backward) for t in (q, k, v))
    if impl == "reference":
        out = reference_attention(
            q_ref, k_ref, v_ref, causal=causal, key_padding_mask=mask
        )
    else:
        out = ATTENTION_FN_REGISTRY[impl](
            q_ref, k_ref, v_ref, causal=causal, key_padding_mask=mask, config=CONFIG
        )
    if backward:
        out.sum().backward()


def _median_time(fn: Callable[[], None], budget_seconds: float = 1.0) -> float:
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
    header = f"{'shape':<14}{'variant':<9}{'mode':<9}" + "".join(
        f"{impl:>12}" for impl in IMPLS
    )
    print(header)
    print("-" * len(header))
    for shape_name, batch, heads, q_len, kv_len, head_dim in SHAPES:
        q, k, v, padding_mask = _make_inputs(batch, heads, q_len, kv_len, head_dim)
        for variant_name, causal, use_mask in VARIANTS:
            mask = padding_mask if use_mask else None
            for backward in (False, True):
                mode = "fwd+bwd" if backward else "fwd"
                row = f"{shape_name:<14}{variant_name:<9}{mode:<9}"
                for impl in IMPLS:
                    timed = partial(
                        _run_once, impl, q, k, v, causal, mask, backward=backward
                    )
                    ms = _median_time(timed)
                    row += f"{ms * 1e3:>11.1f} "
                print(row, flush=True)
            print()


if __name__ == "__main__":
    main()
