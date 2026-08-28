"""Tests for the shared BaseAttention helpers in llminfra/attention/base.py."""

import pytest
import torch

from llminfra import MultiHeadAttention
from llminfra.attention.base_attention import validate_attention_inputs


@pytest.fixture()
def module():
    return MultiHeadAttention(hidden_size=32, num_heads=4, dropout=0.0)


def test_split_combine_head_roundtrip(module):
    x = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(0))

    heads = module.split_head(x)
    assert heads.shape == (2, 4, 5, 8)

    combined = module.combine_head(heads)
    assert combined.shape == x.shape
    torch.testing.assert_close(combined, x)


def test_compute_attention_weights_are_normalized(module):
    scores = torch.randn(2, 4, 5, 5, generator=torch.Generator().manual_seed(0))
    weights = module.compute_attention_weights(scores)

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4, 5))


def test_compute_attention_weights_respects_mask(module):
    scores = torch.randn(1, 1, 3, 3, generator=torch.Generator().manual_seed(0))
    mask = torch.tensor([[[[1, 1, 0]]]])  # (batch, 1, 1, seq) broadcast mask

    weights = module.compute_attention_weights(scores, mask)

    assert weights[..., 2].abs().max().item() == 0.0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1, 3))


def test_3d_mask_broadcasts_over_heads(module):
    """A (batch, q, kv) mask must apply per batch, not align with the head dim.

    Regression test: broadcasting a 3D mask against (batch, heads, q, kv)
    scores right-aligns its batch dim with the head dim, so batch 0's mask
    was applied to head 0 of *every* batch (or raised when batch != heads).
    """
    gen = torch.Generator().manual_seed(0)
    scores = torch.randn(2, 4, 5, 5, generator=gen)
    mask3 = torch.ones(2, 5, 5, dtype=torch.bool)
    mask3[0, :, 3:] = False  # batch 0 sees the first 3 keys
    mask3[1, :, 1:] = False  # batch 1 sees only the first key

    weights3 = module.compute_attention_weights(scores, mask3)
    weights4 = module.compute_attention_weights(scores, mask3.unsqueeze(1))

    torch.testing.assert_close(weights3, weights4)
    assert weights3[1, :, :, 1:].abs().max().item() == 0.0


def test_3d_mask_end_to_end_matches_4d():
    """Modules passing the user mask straight through handle 3D masks."""
    gen = torch.Generator().manual_seed(0)
    module = MultiHeadAttention(hidden_size=16, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 16, generator=gen)
    mask3 = torch.ones(2, 5, 5, dtype=torch.bool)
    mask3[0, :, 3:] = False
    mask3[1, :, 1:] = False

    torch.testing.assert_close(
        module(x, attention_mask=mask3),
        module(x, attention_mask=mask3.unsqueeze(1)),
    )


def test_validate_attention_inputs_ok():
    batch, seq_len = validate_attention_inputs(torch.randn(2, 5, 8), None, num_heads=4)
    assert (batch, seq_len) == (2, 5)


def test_validate_attention_inputs_rejects_2d_hidden_state():
    with pytest.raises(ValueError, match="3D"):
        validate_attention_inputs(torch.randn(5, 8), None, num_heads=4)


def test_validate_attention_inputs_rejects_2d_mask():
    with pytest.raises(ValueError, match="3D or 4D"):
        validate_attention_inputs(torch.randn(2, 5, 8), torch.ones(2, 5), num_heads=4)


def test_validate_attention_inputs_rejects_mask_batch_mismatch():
    with pytest.raises(ValueError, match="batch size"):
        validate_attention_inputs(
            torch.randn(2, 5, 8), torch.ones(3, 1, 1, 5), num_heads=4
        )


def test_validate_attention_inputs_rejects_mask_seq_mismatch():
    with pytest.raises(ValueError, match="sequence length"):
        validate_attention_inputs(
            torch.randn(2, 5, 8), torch.ones(2, 1, 1, 7), num_heads=4
        )


def test_validate_attention_inputs_rejects_float_mask():
    """Additive float masks (0/-inf) must be rejected, not silently ignored."""
    with pytest.raises(ValueError, match="1/0"):
        validate_attention_inputs(
            torch.randn(2, 5, 8), torch.zeros(2, 1, 1, 5), num_heads=4
        )


def test_extra_repr_reports_configuration(module):
    assert "hidden_size=32" in module.extra_repr()
    assert "num_heads=4" in module.extra_repr()
