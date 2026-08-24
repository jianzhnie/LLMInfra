"""Tests for the package-level module registries."""

import pytest
import torch

from llminfra import (
    ALiBiAttention,
    CompressedSparseAttention,
    GatedDeltaNet,
    MultiHeadAttention,
    RingAttention,
    SlidingWindowAttention,
    build_attention,
    build_positional_encoding,
    list_attentions,
)
from llminfra.positional import list_positional_encodings
from llminfra.positional.rope_scaling import YaRNParameters

HIDDEN_SIZE = 32
NUM_HEADS = 4


def test_build_attention_registry() -> None:
    assert isinstance(
        build_attention("mha", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS),
        MultiHeadAttention,
    )
    assert isinstance(
        build_attention(
            "swa",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            window_size=4,
        ),
        SlidingWindowAttention,
    )
    assert isinstance(
        build_attention(
            "gated_delta",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            feature_dim=8,
        ),
        GatedDeltaNet,
    )
    assert "hybrid" in list_attentions()

    with pytest.raises(ValueError, match="Unknown attention"):
        build_attention("unknown", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS)


def test_extended_attention_modules_are_registered() -> None:
    assert isinstance(
        build_attention("ring", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS),
        RingAttention,
    )
    assert isinstance(
        build_attention("alibi", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS),
        ALiBiAttention,
    )
    assert isinstance(
        build_attention(
            "compressed_sparse",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            compress_ratio=2,
        ),
        CompressedSparseAttention,
    )


# Minimum extra constructor kwargs required by each attention registry name.
ATTENTION_KWARGS: dict[str, dict[str, object]] = {
    "gqa": {"num_kv_groups": 2},
    "mla": {"q_latent_size": 8, "kv_latent_size": 8},
    "swa": {"window_size": 4},
    "block_sparse": {"block_size": 4},
    "compressed_sparse": {"compress_ratio": 2},
    "dsa": {"block_size": 4, "top_k": 2},
    "dynamic_sparse": {"block_size": 4, "top_k": 2},
    "msa": {"block_size": 4, "top_k": 2},
    "minimax_sparse": {"block_size": 4, "top_k": 2},
    "hca": {"fine_compress_ratio": 2, "coarse_compress_ratio": 4, "fine_top_k": 2},
    "hierarchical_compressed": {
        "fine_compress_ratio": 2,
        "coarse_compress_ratio": 4,
        "fine_top_k": 2,
    },
}


@pytest.mark.parametrize("name", list_attentions())
def test_every_registered_attention_builds_and_forwards(name: str) -> None:
    module = build_attention(
        name,
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        **ATTENTION_KWARGS.get(name, {}),
    )
    x = torch.randn(2, 8, HIDDEN_SIZE)
    output = module(x)
    if isinstance(output, tuple):
        output = output[0]
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


# Minimum extra kwargs required by each positional encoding name.
POSITIONAL_KWARGS: dict[str, dict[str, object]] = {
    "yarn": {"params": YaRNParameters(original_max_position_embeddings=64)},
    "ntk": {"original_max_position_embeddings": 64},
    "interpolation": {"original_max_position_embeddings": 64},
    "longrope": {"preset": "reference_uniform_256k"},
    "mrope": {"mrope_section": (4, 4, 8)},
    "2d": {"max_blocks": 4, "max_positions_per_block": 16},
}

# Score-bias encodings return (1, heads, q, k) instead of hidden states.
_BIAS_ENCODINGS = {"alibi", "t5_bias"}


@pytest.mark.parametrize("name", list_positional_encodings())
def test_every_registered_positional_encoding_builds_and_forwards(name: str) -> None:
    needs_heads = name in _BIAS_ENCODINGS
    module = build_positional_encoding(
        name,
        dim=HIDDEN_SIZE,
        num_heads=NUM_HEADS if needs_heads else None,
        max_seq_len=64,
        **POSITIONAL_KWARGS.get(name, {}),
    )
    x = torch.randn(2, 8, HIDDEN_SIZE)
    if name == "mrope":
        output = module(x, torch.arange(8).expand(3, 2, 8))
    else:
        output = module(x)
    if isinstance(output, tuple):
        output = output[0]
    expected = (1, NUM_HEADS, 8, 8) if needs_heads else x.shape
    assert output.shape == expected
    if name == "alibi":
        # ALiBi's causal bias is -inf above the diagonal by design.
        lower = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        assert torch.isfinite(output[..., lower]).all()
        assert torch.isinf(output[..., ~lower]).all()
    else:
        assert torch.isfinite(output).all()
