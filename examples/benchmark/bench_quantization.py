"""CPU benchmark for the QAT utilities in ``llminfra.quantization``.

Measures forward and forward+backward wall time for ``FakeQuantizer``
(int4 / int8 / fp8_e4m3, per-tensor and per-channel) over realistic
activation and weight shapes, plus an end-to-end ``QATWrapper`` around a
small transformer-style block.

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/benchmark/bench_quantization.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llminfra import FakeQuantizer, QuantizationConfig, build_quantized

# (name, shape) — activations follow (batch, seq, hidden); the weight shape
# is a typical per-channel projection matrix.
ACTIVATION_SHAPES = [
    ("act-2x128x256", (2, 128, 256)),
    ("act-4x512x512", (4, 512, 512)),
    ("act-2x1024x1024", (2, 1024, 1024)),
    ("act-8x2048x256", (8, 2048, 256)),
    ("weight-1024x1024", (1024, 1024)),
]

MODES = ["int4", "int8", "fp8_e4m3"]


def _median_time(fn: Callable[[], None], budget_seconds: float = 0.5) -> float:
    """Median wall time of ``fn``; repeat count adapts to one timed warmup."""
    start = time.perf_counter()
    fn()
    warmup = time.perf_counter() - start
    repeats = max(1, min(9, int(budget_seconds / max(warmup, 1e-6))))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _run_quantizer(
    quantizer: FakeQuantizer, x: torch.Tensor, *, backward: bool
) -> None:
    x_ref = x.detach().requires_grad_(backward)
    out = quantizer(x_ref)
    if backward:
        out.sum().backward()


def bench_fake_quantizer() -> None:
    header = f"{'shape':<18}{'config':<22}{'mode':<9}{'median ms':>10}"
    print(header)
    print("-" * len(header))
    for shape_name, shape in ACTIVATION_SHAPES:
        x = torch.randn(*shape)
        configs = [
            ("per-tensor", {}),
            # per-channel only makes sense along a retained axis; use the
            # leading axis so every shape exercises the reduction path.
            ("per-channel", {"per_channel": True, "channel_axis": 0}),
        ]
        for config_name, overrides in configs:
            for mode in MODES:
                quantizer = FakeQuantizer(QuantizationConfig(mode=mode, **overrides))  # type: ignore[arg-type]
                for backward in (False, True):
                    timed = partial(_run_quantizer, quantizer, x, backward=backward)
                    ms = _median_time(timed)
                    mode_name = "fwd+bwd" if backward else "fwd"
                    print(
                        f"{shape_name:<18}{config_name:<22}{mode + '/' + mode_name:<9}"
                        f"{ms * 1e3:>9.2f} ",
                        flush=True,
                    )
        print()


def _make_block(hidden: int = 512, layers: int = 2) -> nn.Module:
    """Small transformer-style block: stacked MLPs in Linear form."""
    mods: list[nn.Module] = []
    for _ in range(layers):
        mods += [
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        ]
    return nn.Sequential(*mods)


def bench_qat_wrapper() -> None:
    header = f"{'input':<18}{'flags':<26}{'mode':<9}{'median ms':>10}"
    print(header)
    print("-" * len(header))
    module = _make_block()
    x = torch.randn(4, 512, 512)
    flag_sets = [
        ("all", {}),
        ("weights-only", {"quantize_inputs": False, "quantize_outputs": False}),
        ("activations-only", {"quantize_weights": False}),
    ]
    for flags_name, overrides in flag_sets:
        config = QuantizationConfig(mode="int8", **overrides)  # type: ignore[arg-type]
        wrapped = build_quantized(module, config=config)
        for backward in (False, True):

            def run(wrapped: nn.Module = wrapped, backward: bool = backward) -> None:
                x_ref = x.detach().requires_grad_(backward)
                out = wrapped(x_ref)
                if backward:
                    assert isinstance(out, torch.Tensor)
                    out.sum().backward()

            ms = _median_time(run)
            mode_name = "fwd+bwd" if backward else "fwd"
            print(
                f"{'4x512x512':<18}{flags_name:<26}{mode_name:<9}{ms * 1e3:>9.2f} ",
                flush=True,
            )
    print()


def main() -> None:
    torch.manual_seed(0)
    print("== FakeQuantizer ==")
    bench_fake_quantizer()
    print("== QATWrapper (2-layer Linear block, hidden 512) ==")
    bench_qat_wrapper()


if __name__ == "__main__":
    main()
