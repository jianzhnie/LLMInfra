"""CPU benchmark for the educational MoE modules.

Measures forward and forward+backward wall time for MixtureOfExperts,
DeepSeekMoE, LatentMoE, ExpertParallelMoE (local-owner mode), the routers
and the load-balancing loss across a realistic shape matrix.

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/bench_moe.py
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

from llminfra import (
    DeepSeekMoE,
    ExpertParallelMoE,
    LatentMoE,
    MixtureOfExperts,
    TopKRouter,
    load_balance_loss,
)
from llminfra.moe.mixture_of_experts import ExpertChoiceRouter

# (name, batch, seq, hidden)
SHAPES = [
    ("small-2x128", 2, 128, 256),
    ("mid-4x512", 4, 512, 512),
    ("large-2x2048", 2, 2048, 512),
    ("wide-8x256", 8, 256, 1024),
]

NUM_EXPERTS = 8
TOP_K = 2


def _make_modules(hidden: int) -> dict[str, nn.Module]:
    """Build one instance of every module under test for a hidden size."""
    torch.manual_seed(0)
    return {
        "moe": MixtureOfExperts(
            hidden_size=hidden,
            num_experts=NUM_EXPERTS,
            intermediate_size=hidden * 2,
            top_k=TOP_K,
        ),
        "deepseek": DeepSeekMoE(
            hidden_size=hidden,
            num_routed_experts=NUM_EXPERTS,
            num_shared_experts=2,
            intermediate_size=hidden * 2,
            top_k=TOP_K,
        ),
        "latent": LatentMoE(
            hidden_size=hidden,
            latent_size=hidden // 2,
            num_experts=NUM_EXPERTS,
            intermediate_size=hidden,
            top_k=TOP_K,
        ),
        "ep_local": ExpertParallelMoE(
            hidden_size=hidden,
            num_experts=NUM_EXPERTS,
            intermediate_size=hidden * 2,
            top_k=TOP_K,
            world_size=2,
            rank=0,
        ),
        "topk_router": TopKRouter(hidden, num_experts=NUM_EXPERTS, top_k=TOP_K),
        "ec_router": ExpertChoiceRouter(hidden, num_experts=NUM_EXPERTS, top_tokens=64),
    }


def _run_once(
    name: str,
    modules: dict[str, nn.Module],
    x: torch.Tensor,
    *,
    backward: bool,
) -> None:
    if name == "aux_loss":
        router = modules["topk_router"]
        logits = router.routing_logits(x.reshape(-1, x.size(-1)))
        _, indices = router(x.reshape(-1, x.size(-1)))
        loss = load_balance_loss(logits, indices, num_experts=NUM_EXPERTS)
        if backward:
            loss.backward()
        return
    x_in = x.detach().requires_grad_(backward)
    out = modules[name](x_in)
    if backward:
        # Routers return (weights, indices); only the weights carry grad.
        (out[0] if isinstance(out, tuple) else out).sum().backward()


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
    bench_names = [
        "moe",
        "deepseek",
        "latent",
        "ep_local",
        "topk_router",
        "ec_router",
        "aux_loss",
    ]
    header = f"{'shape':<14}{'mode':<9}" + "".join(f"{n:>13}" for n in bench_names)
    print(header)
    print("-" * len(header))
    for shape_name, batch, seq, hidden in SHAPES:
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(batch, seq, hidden, generator=generator)
        modules = _make_modules(hidden)
        for backward in (False, True):
            mode = "fwd+bwd" if backward else "fwd"
            row = f"{shape_name:<14}{mode:<9}"
            for bench_name in bench_names:
                inp = x.reshape(-1, hidden) if bench_name == "ec_router" else x
                timed = partial(_run_once, bench_name, modules, inp, backward=backward)
                ms = _median_time(timed)
                row += f"{ms * 1e3:>12.1f} "
            print(row, flush=True)
        print()


if __name__ == "__main__":
    main()
