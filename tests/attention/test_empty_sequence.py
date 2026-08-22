"""Empty-sequence (seq_len == 0) handling across attention modules.

Zero-length inputs are admissible (``validate_attention_inputs`` does not
reject them, and the matmul/softmax-based modules process them naturally),
so every attention module must return a ``(batch, 0, hidden_size)`` output
instead of crashing on empty ``stack``/``cat``/reduction calls. Several
recurrent and block-sparse paths used to fail here; the chunked causal scan
in ``LinearAttention`` regressed on it when the scan was introduced.
"""

import pytest
import torch

from llminfra.attention import (
    BlockSparseAttention,
    CompressedSparseAttention,
    DynamicSparseAttention,
    GatedDeltaNet,
    HybridAttention,
    KimiDeltaAttention,
    LightningAttention,
    LinearAttention,
    MultiHeadAttention,
)

HIDDEN = 32
HEADS = 4
BATCH = 2


def _build_modules() -> dict[str, torch.nn.Module]:
    return {
        "multi_head": MultiHeadAttention(HIDDEN, HEADS),
        "linear_causal": LinearAttention(HIDDEN, HEADS, causal=True),
        "linear_noncausal": LinearAttention(HIDDEN, HEADS, causal=False),
        "gated_delta_net": GatedDeltaNet(HIDDEN, HEADS, normalize=True),
        "kimi_delta": KimiDeltaAttention(HIDDEN, HEADS),
        "lightning": LightningAttention(HIDDEN, HEADS, block_size=16),
        "hybrid_linear_route": HybridAttention(HIDDEN, HEADS),
        "block_sparse": BlockSparseAttention(
            HIDDEN, HEADS, block_size=4, top_k=2, dropout=0.0
        ),
        "compressed_sparse": CompressedSparseAttention(HIDDEN, HEADS, compress_ratio=3),
        "dynamic_sparse": DynamicSparseAttention(HIDDEN, HEADS, block_size=4, top_k=2),
    }


@pytest.mark.parametrize("name", sorted(_build_modules()))
def test_empty_sequence_returns_empty_output(name: str) -> None:
    module = _build_modules()[name].eval()
    hidden_state = torch.randn(BATCH, 0, HIDDEN, requires_grad=True)
    output = module(hidden_state)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (BATCH, 0, HIDDEN)
    assert output.dtype == hidden_state.dtype
    output.sum().backward()
    assert hidden_state.grad is not None
    assert hidden_state.grad.shape == (BATCH, 0, HIDDEN)


def test_hybrid_full_route_empty_sequence() -> None:
    module = HybridAttention(HIDDEN, HEADS).eval()
    hidden_state = torch.randn(BATCH, 0, HIDDEN)
    output = module(hidden_state, layer_index=module.linear_interval)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (BATCH, 0, HIDDEN)


def test_empty_sequence_then_nonempty_same_instance() -> None:
    """An empty call must not corrupt any internal state for later calls."""
    module = LinearAttention(HIDDEN, HEADS).eval()
    empty = torch.randn(BATCH, 0, HIDDEN)
    nonempty = torch.randn(BATCH, 5, HIDDEN)
    module(empty)
    first = module(nonempty)
    second = module(nonempty)
    assert first.shape == (BATCH, 5, HIDDEN)
    torch.testing.assert_close(first, second)
