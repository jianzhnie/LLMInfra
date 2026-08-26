"""Seeded differential fuzz tests for the attention implementations.

Strategy: a single master seed drives a ``torch.Generator`` that draws random
shapes (batch 1-3, seq 1-96 — deliberately not aligned to any block/chunk
size — heads 1-4, head_dim 8-64), random causal/padding-mask combinations
(no mask, random suffix masking, fully-masked rows) and dtypes (float32
everywhere; float64 for the paths that are exact enough to benefit from
tighter tolerances). Each implementation is then compared against a trusted
reference over the same random inputs:

- ``MultiHeadAttention`` / ``GroupedQueryAttention`` /
  ``MultiQueryAttention`` / ``SlidingWindowAttention`` against a
  straightforward manual attention (einsum + softmax) written below, reusing
  only the module's projection weights.
- ``flash_attention`` versions fa1-fa4 against the dense
  ``reference_attention`` baseline. These cases are float32-only on purpose:
  the tiled versions upcast scores and softmax statistics to float32
  internally, so a float64 case could not be checked any tighter.
- ``LinearAttention``'s chunked causal scan against the naive ``cumsum``
  formulation written inline below, with chunk sizes both smaller than,
  larger than, and not dividing the sequence length.

Everything is deterministic: case parameters are drawn sequentially from the
master seed, and each case's tensors come from a generator seeded by
``MASTER_SEED + case_index``. To reproduce a failure, rerun with the same
seed — or sweep seeds with e.g.::

    LLMINFRA_FUZZ_SEED=12345 pytest tests/attention/test_differential_fuzz.py

The pytest id of a failing case prints all of its drawn parameters, so a
single case can be re-run in isolation via ``pytest -k <id>``.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from llminfra import (
    GroupedQueryAttention,
    LinearAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    SlidingWindowAttention,
)
from llminfra.flash_attention import flash_attention
from llminfra.flash_attention.common import FlashAttentionConfig, reference_attention

MASTER_SEED = int(os.environ.get("LLMINFRA_FUZZ_SEED", "20260824"))

NUM_MODULE_CASES = 64
NUM_FLASH_CASES = 32
NUM_LINEAR_CASES = 32

# (rtol, atol) per dtype: near-exact for float64, looser for float32 where
# different summation orders (matmul vs einsum, online softmax, chunking)
# accumulate visible rounding error.
TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-9, "atol": 1e-11},
}

FLASH_VERSIONS = ["fa1", "fa2", "fa3", "fa4"]


# --- random case drawing (all hand-rolled on top of torch.Generator) ---------


def _randint(gen: torch.Generator, low: int, high: int) -> int:
    """Uniform integer in the inclusive range [low, high]."""
    return int(torch.randint(low, high + 1, (), generator=gen))


def _randbool(gen: torch.Generator) -> bool:
    return bool(_randint(gen, 0, 1))


def _randchoice(gen: torch.Generator, options: list[Any]) -> Any:
    return options[_randint(gen, 0, len(options) - 1)]


def _randdtype(gen: torch.Generator) -> torch.dtype:
    # ~1/4 of cases in float64 for the tighter checks.
    return torch.float64 if _randint(gen, 0, 3) == 0 else torch.float32


def _random_suffix_mask(gen: torch.Generator, batch: int, seq: int) -> torch.Tensor:
    """Bool mask keeping a random-length prefix per batch row (length may be 0)."""
    valid = torch.arange(seq).expand(batch, seq) < torch.randint(
        0, seq + 1, (batch, 1), generator=gen
    )
    return valid


def _make_key_padding(
    gen: torch.Generator, mode: str, batch: int, seq: int
) -> torch.Tensor | None:
    """Random 2D key padding mask (True = valid) or None."""
    if mode == "none":
        return None
    if mode == "suffix":
        return _random_suffix_mask(gen, batch, seq)
    # "empty-row": everything valid except batch row 0, which is fully masked.
    mask = torch.ones(batch, seq, dtype=torch.bool)
    mask[0] = False
    return mask


def _make_module_case(gen: torch.Generator, index: int) -> dict[str, Any]:
    heads = _randint(gen, 1, 4)
    kv_divisors = [d for d in range(1, heads + 1) if heads % d == 0]
    case: dict[str, Any] = {
        "index": index,
        "kind": _randchoice(gen, ["mha", "gqa", "mqa", "swa"]),
        "batch": _randint(gen, 1, 3),
        "seq": _randint(gen, 1, 96),
        "heads": heads,
        "head_dim": 8 * _randint(gen, 1, 8),
        "kv_groups": _randchoice(gen, kv_divisors),
        "window": _randint(gen, 1, 96),
        "causal": _randbool(gen),
        "mask_mode": _randchoice(gen, ["none", "suffix", "empty-row"]),
        "dtype": _randdtype(gen),
        "seed": MASTER_SEED + index,
    }
    return case


def _make_flash_case(gen: torch.Generator, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "batch": _randint(gen, 1, 3),
        "seq": _randint(gen, 1, 96),
        "heads": _randint(gen, 1, 4),
        "head_dim": 8 * _randint(gen, 1, 8),
        "causal": _randbool(gen),
        "mask_mode": _randchoice(gen, ["none", "suffix", "empty-row"]),
        # Small, randomly drawn tiles so seqs hit ragged tile boundaries.
        "block_q": _randchoice(gen, [8, 16, 32, 64]),
        "block_kv": _randchoice(gen, [8, 16, 32]),
        "num_stages": _randint(gen, 1, 3),
        "seed": MASTER_SEED + index,
    }


def _make_linear_case(gen: torch.Generator, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "batch": _randint(gen, 1, 3),
        "seq": _randint(gen, 1, 96),
        "heads": _randint(gen, 1, 4),
        "head_dim": 8 * _randint(gen, 1, 8),
        "feature_dim": 8 * _randint(gen, 1, 8),
        "kernel": _randchoice(gen, ["elu", "relu"]),
        # Chunk sizes both below, above and not dividing the seq length.
        "chunk_size": _randchoice(gen, [1, 2, 3, 7, 16, 64, 200]),
        "mask_mode": _randchoice(gen, ["none", "suffix", "empty-row"]),
        "dtype": _randdtype(gen),
        "seed": MASTER_SEED + index,
    }


def _draw_cases(make_case, count: int) -> list[dict[str, Any]]:
    gen = torch.Generator().manual_seed(MASTER_SEED)
    return [make_case(gen, index) for index in range(count)]


MODULE_CASES = _draw_cases(_make_module_case, NUM_MODULE_CASES)
FLASH_CASES = _draw_cases(_make_flash_case, NUM_FLASH_CASES)
LINEAR_CASES = _draw_cases(_make_linear_case, NUM_LINEAR_CASES)


def _case_id(case: dict[str, Any]) -> str:
    parts = [f"seed{case['seed']}"] + [
        f"{key}{case[key]}" for key in sorted(case) if key not in {"index", "seed"}
    ]
    return "-".join(str(p).replace("torch.", "") for p in parts)


# --- (a) nn.Module attention vs a manual einsum + softmax reference ----------


def _build_module(case: dict[str, Any]) -> torch.nn.Module:
    hidden = case["heads"] * case["head_dim"]
    if case["kind"] == "mha":
        module: torch.nn.Module = MultiHeadAttention(hidden, case["heads"], dropout=0.0)
    elif case["kind"] == "gqa":
        module = GroupedQueryAttention(
            hidden, case["heads"], num_kv_groups=case["kv_groups"], dropout=0.0
        )
    elif case["kind"] == "mqa":
        module = MultiQueryAttention(hidden, case["heads"], dropout=0.0)
    else:
        module = SlidingWindowAttention(
            hidden,
            case["heads"],
            window_size=case["window"],
            num_kv_groups=case["kv_groups"],
            dropout=0.0,
            causal=case["causal"],
        )
    module.eval()
    return module.to(case["dtype"])


def _split_qkv(
    module: torch.nn.Module, kind: str, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project and head-split with the module's own weights."""
    q = module.split_head(module.q_proj(x))
    if kind == "mha":
        k = module.split_head(module.k_proj(x))
        v = module.split_head(module.v_proj(x))
    elif kind == "gqa":
        k = module.split_head_grouped(module.k_proj(x))
        v = module.split_head_grouped(module.v_proj(x))
    elif kind == "mqa":
        # Single shared head, expanded so the einsum below stays uniform.
        k = module.split_head(module.k_proj(x), num_heads=1).expand(
            -1, module.num_heads, -1, -1
        )
        v = module.split_head(module.v_proj(x), num_heads=1).expand(
            -1, module.num_heads, -1, -1
        )
    else:  # swa
        k = module._split_kv(module.k_proj(x))
        v = module._split_kv(module.v_proj(x))
    return q, k, v


def _module_mask(case: dict[str, Any]) -> torch.Tensor | None:
    """Build the (batch, 1, q, kv) bool mask handed to the nn modules."""
    batch, seq = case["batch"], case["seq"]
    gen = torch.Generator().manual_seed(case["seed"] + 1)
    mask: torch.Tensor | None = None
    # SlidingWindowAttention owns its causality via the window; the other
    # kinds get an explicit causal mask.
    if case["causal"] and case["kind"] != "swa":
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).expand(
            batch, 1, seq, seq
        )
    padding = _make_key_padding(gen, case["mask_mode"], batch, seq)
    if padding is not None:
        padding = padding.view(batch, 1, 1, seq)
        mask = padding if mask is None else mask & padding
    return mask


def _manual_module_attention(
    module: torch.nn.Module, case: dict[str, Any], x: torch.Tensor
) -> torch.Tensor:
    """Straightforward dense reference: scaled einsum scores + softmax."""
    q, k, v = _split_qkv(module, case["kind"], x)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * module.scale_factor

    mask = _module_mask(case)
    if case["kind"] == "swa":
        seq = case["seq"]
        pos = torch.arange(seq)
        distance = pos.view(-1, 1) - pos.view(1, -1)
        if case["causal"]:
            window = (distance >= 0) & (distance <= case["window"])
        else:
            window = distance.abs() <= case["window"]
        window = window.view(1, 1, seq, seq)
        mask = window if mask is None else window & mask

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # Rows whose keys are all masked softmax to NaN; define them as zero.
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.einsum("bhqk,bhkd->bhqd", weights, v)
    return module.o_proj(module.combine_head(out))


@pytest.mark.parametrize("case", MODULE_CASES, ids=[_case_id(c) for c in MODULE_CASES])
def test_nn_module_matches_manual_attention(case: dict[str, Any]) -> None:
    module = _build_module(case)
    gen = torch.Generator().manual_seed(case["seed"])
    x = torch.randn(
        case["batch"],
        case["seq"],
        case["heads"] * case["head_dim"],
        generator=gen,
        dtype=case["dtype"],
    )

    actual = module(x, attention_mask=_module_mask(case))
    expected = _manual_module_attention(module, case, x)

    torch.testing.assert_close(actual, expected, **TOLERANCES[case["dtype"]])


# --- (b) flash_attention fa1-fa4 vs the dense reference_attention ------------


@pytest.mark.parametrize("version", FLASH_VERSIONS)
@pytest.mark.parametrize("case", FLASH_CASES, ids=[_case_id(c) for c in FLASH_CASES])
def test_flash_versions_match_reference_attention(
    version: str, case: dict[str, Any]
) -> None:
    gen = torch.Generator().manual_seed(case["seed"])
    shape = (case["batch"], case["heads"], case["seq"], case["head_dim"])
    q = torch.randn(shape, generator=gen)
    k = torch.randn(shape, generator=gen)
    v = torch.randn(shape, generator=gen)
    padding = _make_key_padding(
        torch.Generator().manual_seed(case["seed"] + 1),
        case["mask_mode"],
        case["batch"],
        case["seq"],
    )
    config = FlashAttentionConfig(
        block_size_q=case["block_q"],
        block_size_kv=case["block_kv"],
        num_stages=case["num_stages"],
        keep_debug_state=False,
    )

    actual = flash_attention(
        q,
        k,
        v,
        version=version,
        causal=case["causal"],
        key_padding_mask=padding,
        config=config,
    )
    expected = reference_attention(
        q, k, v, causal=case["causal"], key_padding_mask=padding
    )

    torch.testing.assert_close(actual, expected, **TOLERANCES[torch.float32])


# --- (c) LinearAttention chunked causal scan vs the naive cumsum -------------


def _naive_causal_linear_attention(
    module: LinearAttention, x: torch.Tensor, padding: torch.Tensor | None
) -> torch.Tensor:
    """Naive causal linear attention via an explicit per-step cumsum state."""
    query = module._feature_map(module._split(module.q_proj(x), module.feature_dim))
    key = module._feature_map(module._split(module.k_proj(x), module.feature_dim))
    value = module.split_head(module.v_proj(x))
    if padding is not None:
        key = key * padding.unsqueeze(-1)
        value = value * padding.unsqueeze(-1)

    # Running (feature_dim, head_dim) state per position: O(seq) materialized.
    kv_state = torch.einsum("bhsf,bhsd->bhsfd", key, value).cumsum(dim=2)
    out_unnorm = torch.einsum("bhsf,bhsfd->bhsd", query, kv_state)
    key_state = key.cumsum(dim=2)
    normalizer = torch.einsum("bhsf,bhsf->bhs", query, key_state)

    # Same sign-safe division rule as the implementation under test.
    eps = torch.finfo(normalizer.dtype).eps
    safe = torch.where(
        normalizer >= 0, normalizer.clamp_min(eps), normalizer.clamp_max(-eps)
    )
    out = out_unnorm / safe.unsqueeze(-1)
    out = out.where(normalizer.unsqueeze(-1) != 0, torch.zeros_like(out))
    return module.o_proj(module.combine_head(out))


@pytest.mark.parametrize("case", LINEAR_CASES, ids=[_case_id(c) for c in LINEAR_CASES])
def test_linear_attention_chunked_scan_matches_naive_cumsum(
    case: dict[str, Any],
) -> None:
    hidden = case["heads"] * case["head_dim"]
    module = LinearAttention(
        hidden,
        case["heads"],
        feature_dim=case["feature_dim"],
        kernel=case["kernel"],
        causal=True,
        chunk_size=case["chunk_size"],
        dropout=0.0,
    )
    module.eval()
    module = module.to(case["dtype"])

    gen = torch.Generator().manual_seed(case["seed"])
    x = torch.randn(case["batch"], case["seq"], hidden, generator=gen).to(case["dtype"])
    padding = _make_key_padding(
        torch.Generator().manual_seed(case["seed"] + 1),
        case["mask_mode"],
        case["batch"],
        case["seq"],
    )
    mask = padding.view(case["batch"], 1, case["seq"]) if padding is not None else None

    actual = module(x, attention_mask=mask)
    expected = _naive_causal_linear_attention(module, x, mask)

    torch.testing.assert_close(actual, expected, **TOLERANCES[case["dtype"]])
