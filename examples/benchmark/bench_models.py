"""CPU benchmark for the composite models in ``llminfra.models``.

Measures forward (and forward+backward for the training paths) wall time of
``CausalLMModel``, ``EncoderOnlyModel``, ``EncoderDecoderModel`` and
``MultimodalCausalLM`` across realistic shapes (batch 2-8, seq 128-2048,
hidden 256-768), with and without padding masks / labels.

Timings are medians over an adaptively chosen number of repeats. Run from
the repository root:

    python examples/benchmark/bench_models.py
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

# The package is not pip-installed in this repo; make the script runnable
# directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llminfra.models import (
    EncoderDecoderModel,
    EncoderOnlyModel,
    MultimodalCausalLM,
)
from llminfra.models.language import CausalLMModel


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


def _rand_ids(batch: int, seq: int, vocab: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab, (batch, seq), generator=generator)


def _rand_mask(batch: int, seq: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    mask = torch.rand(batch, seq, generator=generator) > 0.2
    mask[:, 0] = True  # keep at least one valid token per batch row
    return mask


def _print_pair(
    name: str, label_a: str, ms_a: float, label_b: str, ms_b: float
) -> None:
    print(
        f"{name:<12}{label_a}={ms_a * 1e3:8.1f}ms  {label_b}={ms_b * 1e3:8.1f}ms",
        flush=True,
    )


def _time_causal_lm(
    name: str,
    model: CausalLMModel,
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    variants = {
        "fwd-plain": lambda: model(ids),
        "fwd-mask": lambda: model(ids, attention_mask=mask),
        "fwd-labels": lambda: model(ids, attention_mask=mask, labels=labels),
    }
    row = f"{name:<10}"
    for variant, fn in variants.items():
        with torch.no_grad():
            row += f"{variant}={_median_time(fn) * 1e3:8.1f}ms  "

    def train_step() -> None:
        model.zero_grad(set_to_none=True)
        out = model(ids, attention_mask=mask, labels=labels)
        assert out.loss is not None
        out.loss.backward()

    row += f"train={_median_time(train_step) * 1e3:8.1f}ms"
    print(row, flush=True)


def _time_encoder(
    name: str,
    model: EncoderOnlyModel,
    ids: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    with torch.no_grad():
        plain = _median_time(lambda: model(ids))
        masked = _median_time(lambda: model(ids, attention_mask=mask))
    _print_pair(name, "fwd-plain", plain, "fwd-mask", masked)


def _time_encoder_decoder(
    name: str,
    model: EncoderDecoderModel,
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> None:
    with torch.no_grad():
        plain = _median_time(lambda: model(src, tgt))
        masked = _median_time(lambda: model(src, tgt, src_mask, tgt_mask))
    _print_pair(name, "fwd-plain", plain, "fwd-mask", masked)


def _time_multimodal(
    name: str,
    model: MultimodalCausalLM,
    ids: torch.Tensor,
    vision: torch.Tensor,
    grid: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    with torch.no_grad():
        plain = _median_time(lambda: model(ids, vision, grid))
        trained = _median_time(lambda: model(ids, vision, grid, labels=labels))
    _print_pair(name, "fwd", plain, "fwd-labels", trained)


def bench_causal_lm() -> None:
    # (name, batch, seq, hidden, layers, vocab)
    shapes = [
        ("lm-128", 4, 128, 256, 4, 4096),
        ("lm-512", 4, 512, 512, 4, 8192),
        ("lm-1024", 2, 1024, 512, 6, 8192),
        ("lm-2048", 2, 2048, 768, 4, 16384),
    ]
    for name, batch, seq, hidden, layers, vocab in shapes:
        model = CausalLMModel(
            vocab_size=vocab,
            hidden_size=hidden,
            num_layers=layers,
            num_heads=8,
            intermediate_size=hidden * 4,
            max_seq_len=seq,
            num_mtp_predictions=2,
        ).eval()
        _time_causal_lm(
            name,
            model,
            _rand_ids(batch, seq, vocab),
            _rand_mask(batch, seq),
            _rand_ids(batch, seq, vocab),
        )


def bench_encoder() -> None:
    shapes = [
        ("enc-128", 8, 128, 256, 4),
        ("enc-512", 4, 512, 512, 6),
        ("enc-1024", 2, 1024, 768, 4),
    ]
    for name, batch, seq, hidden, layers in shapes:
        model = EncoderOnlyModel(
            vocab_size=8192,
            hidden_size=hidden,
            num_layers=layers,
            num_heads=8,
            intermediate_size=hidden * 4,
            max_seq_len=seq,
            type_vocab_size=2,
        ).eval()
        _time_encoder(name, model, _rand_ids(batch, seq, 8192), _rand_mask(batch, seq))


def bench_encoder_decoder() -> None:
    shapes = [
        ("ed-128", 4, 128, 256, 3),
        ("ed-512", 2, 512, 512, 4),
        ("ed-1024", 2, 1024, 512, 4),
    ]
    for name, batch, seq, hidden, layers in shapes:
        model = EncoderDecoderModel(
            vocab_size=8192,
            hidden_size=hidden,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            num_heads=8,
            intermediate_size=hidden * 4,
            max_seq_len=seq,
        ).eval()
        _time_encoder_decoder(
            name,
            model,
            _rand_ids(batch, seq, 8192),
            _rand_ids(batch, seq, 8192),
            _rand_mask(batch, seq),
            _rand_mask(batch, seq),
        )


def bench_multimodal() -> None:
    shapes = [
        ("mm-early", "early", 4, 256, 256, 4),
        ("mm-cross", "cross_attention", 4, 256, 256, 4),
        ("mm-early-lg", "early", 2, 512, 512, 6),
    ]
    for name, fusion, batch, seq, hidden, layers in shapes:
        model = MultimodalCausalLM(
            vocab_size=8192,
            vision_dim=hidden,
            hidden_size=hidden,
            num_layers=layers,
            num_heads=8,
            intermediate_size=hidden * 4,
            mrope_section=(hidden // 8, hidden // 4, hidden // 8),
            fusion_mode=fusion,
            max_seq_len=1024 + seq,
        ).eval()
        generator = torch.Generator().manual_seed(0)
        # 8x8 patch grid -> 64 vision tokens per example.
        vision = torch.randn(batch, 64, hidden, generator=generator)
        _time_multimodal(
            name,
            model,
            _rand_ids(batch, seq, 8192),
            vision,
            torch.tensor([[1, 8, 8]] * batch),
            _rand_ids(batch, seq, 8192),
        )


def main() -> None:
    torch.manual_seed(0)
    print("== CausalLMModel ==")
    bench_causal_lm()
    print("== EncoderOnlyModel ==")
    bench_encoder()
    print("== EncoderDecoderModel ==")
    bench_encoder_decoder()
    print("== MultimodalCausalLM ==")
    bench_multimodal()


if __name__ == "__main__":
    main()
