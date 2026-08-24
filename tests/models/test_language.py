"""Tests for decoder-only and prefix language models."""

import pytest
import torch

from llminfra import CausalLMModel


def test_causal_lm_shape_and_gradient():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        max_seq_len=16,
        attention_name="gqa",
    )
    input_ids = torch.randint(0, 32, (2, 7))
    logits = model(input_ids)
    assert logits.shape == (2, 7, 32)
    logits.sum().backward()
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_causal_lm_with_padding_mask():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="gqa",
    )
    input_ids = torch.randint(0, 32, (2, 6))
    padding = torch.ones(2, 6, dtype=torch.bool)
    padding[0, -2:] = False
    logits = model(input_ids, attention_mask=padding)
    assert torch.isfinite(logits).all()


def test_causal_lm_with_moe():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="gqa",
        use_moe=True,
        num_experts=4,
        expert_top_k=2,
    )
    input_ids = torch.randint(0, 32, (2, 5))
    assert model(input_ids).shape == (2, 5, 32)


def test_causal_lm_with_hybrid_attention_returns_weights():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        attention_name="hybrid",
        attention_kwargs={
            "linear_interval": 1,
            "full_interval": 1,
            "linear_feature_dim": 8,
            "num_kv_groups": 1,
        },
    )
    input_ids = torch.randint(0, 32, (2, 5))
    logits, weights = model(input_ids, return_attention_weights=True)
    assert logits.shape == (2, 5, 32)
    assert weights.shape == (2, 2, 5, 5)


def test_causal_lm_ties_embeddings():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="mha",
        tie_word_embeddings=True,
    )
    assert model.lm_head.weight is model.embed_tokens.weight


def test_causal_lm_with_alibi():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="mha",
        positional="alibi",
    )
    input_ids = torch.randint(0, 32, (2, 6))
    logits = model(input_ids)
    assert logits.shape == (2, 6, 32)


def test_causal_lm_with_ring_attention():
    """RingAttention masks causally on its own, so the model omits the mask."""
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="ring",
    )
    input_ids = torch.randint(0, 32, (2, 6))
    output = model(input_ids, labels=input_ids.clone(), return_dict=True)
    assert output.logits.shape == (2, 6, 32)
    assert torch.isfinite(output.loss).all()
    # Padding masks are not supported by ring attention and must surface the
    # attention module's own error instead of being silently dropped.
    with pytest.raises(ValueError, match="RingAttention"):
        model(input_ids, attention_mask=torch.ones(2, 6, dtype=torch.bool))


def test_causal_lm_rejects_t5_bias_positional():
    """t5_bias is a score bias with no causal-attention consumer here."""
    with pytest.raises(ValueError, match="t5_bias"):
        CausalLMModel(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            intermediate_size=32,
            positional="t5_bias",
        )


def test_causal_lm_with_longrope():
    factors = [1.0] * 16
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=32,
        attention_name="mha",
        positional="longrope",
        positional_kwargs={
            "original_max_position_embeddings": 16,
            "long_factor": factors,
            "short_factor": factors,
        },
    )
    logits = model(torch.randint(0, 32, (2, 24)))
    assert logits.shape == (2, 24, 32)


def test_causal_lm_with_2d_position():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=16,
        attention_name="mha",
        positional="2d",
        positional_kwargs={
            "max_blocks": 4,
            "max_positions_per_block": 4,
        },
    )
    logits = model(torch.randint(0, 32, (2, 8)))
    assert logits.shape == (2, 8, 32)


def test_causal_lm_loss_matches_shifted_cross_entropy():
    """loss must score position i against labels[i + 1] (finite alone is not enough)."""
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="mha",
    ).eval()
    input_ids = torch.randint(0, 32, (2, 6))

    output = model(input_ids, labels=input_ids)
    logits = model(input_ids)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 32), input_ids[:, 1:].reshape(-1)
    )
    assert output.loss is not None
    torch.testing.assert_close(output.loss, expected)


def _make_prefix_model() -> CausalLMModel:
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        max_seq_len=16,
        attention_name="mha",
    )
    model.eval()
    return model


def test_causal_lm_prefix_region_is_bidirectional():
    model = _make_prefix_model()
    input_ids = torch.randint(0, 32, (1, 8))
    perturbed = input_ids.clone()
    perturbed[0, 2] = (perturbed[0, 2] + 1) % 32
    with torch.no_grad():
        base = model(input_ids, prefix_len=4)
        changed = model(perturbed, prefix_len=4)
    # Within the prefix, later tokens are visible to earlier positions...
    assert not torch.allclose(base[0, 0], changed[0, 0])
    # ...and prefix tokens remain visible after the prefix boundary.
    assert not torch.allclose(base[0, 6], changed[0, 6])


def test_causal_lm_outside_prefix_stays_causal():
    model = _make_prefix_model()
    input_ids = torch.randint(0, 32, (1, 8))
    perturbed = input_ids.clone()
    perturbed[0, 6] = (perturbed[0, 6] + 1) % 32
    with torch.no_grad():
        base = model(input_ids, prefix_len=3)
        changed = model(perturbed, prefix_len=3)
    # Position 6 is outside the prefix: positions before it must not change.
    assert torch.allclose(base[0, :6], changed[0, :6])
    assert not torch.allclose(base[0, 6:], changed[0, 6:])


def test_causal_lm_prefix_len_combines_with_padding_mask():
    model = _make_prefix_model()
    input_ids = torch.randint(0, 32, (2, 8))
    padding = torch.ones(2, 8, dtype=torch.bool)
    padding[0, 1] = False  # a padded token inside the prefix
    perturbed = input_ids.clone()
    perturbed[0, 1] = (perturbed[0, 1] + 1) % 32
    with torch.no_grad():
        base = model(input_ids, attention_mask=padding, prefix_len=3)
        changed = model(perturbed, attention_mask=padding, prefix_len=3)
    # The padding mask wins over prefix visibility: masked tokens are ignored.
    assert torch.allclose(base[0], changed[0])


def test_causal_lm_prefix_len_validation():
    model = _make_prefix_model()
    input_ids = torch.randint(0, 32, (1, 8))
    with pytest.raises(ValueError, match="prefix_len"):
        model(input_ids, prefix_len=0)
    with pytest.raises(ValueError, match="prefix_len"):
        model(input_ids, prefix_len=9)


def test_causal_lm_mtp_training_output_and_tied_weights():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        num_mtp_predictions=2,
        tie_word_embeddings=True,
    )
    input_ids = torch.randint(0, 32, (2, 6))
    output = model(input_ids, labels=input_ids, return_mtp=True)

    assert output.logits.shape == (2, 6, 32)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.mtp_logits is not None and len(output.mtp_logits) == 2
    assert model.mtp_head is not None
    assert all(
        head.weight is model.embed_tokens.weight for head in model.mtp_head.heads
    )
    output.loss.backward()
    assert model.embed_tokens.weight.grad is not None


def test_causal_lm_return_mtp_requires_configured_head():
    model = _make_prefix_model()
    with pytest.raises(ValueError, match="num_mtp_predictions"):
        model(torch.randint(0, 32, (1, 4)), return_mtp=True)


def test_causal_lm_default_gqa_supports_odd_num_heads():
    # num_heads=5 used to crash: the default num_kv_groups (5 // 2 = 2) does
    # not divide 5. The default must fall back to a valid divisor.
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=20,
        num_layers=1,
        num_heads=5,
        intermediate_size=32,
    )
    input_ids = torch.randint(0, 32, (2, 5))
    assert model(input_ids).shape == (2, 5, 32)


def test_causal_lm_moves_cpu_attention_mask_to_model_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        pytest.skip("requires an MPS or CUDA device")
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
    ).to(device)
    input_ids = torch.randint(0, 32, (1, 4), device=device)
    cpu_mask = torch.ones(1, 4, dtype=torch.bool)
    logits = model(input_ids, attention_mask=cpu_mask)
    assert logits.shape == (1, 4, 32)


def test_causal_lm_mtp_head_forward_runs_once_with_labels(monkeypatch):
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        num_mtp_predictions=2,
    )
    assert model.mtp_head is not None
    calls = 0
    original_forward = model.mtp_head.forward

    def counting_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.mtp_head, "forward", counting_forward)
    input_ids = torch.randint(0, 32, (2, 6))
    model(input_ids, labels=input_ids)
    assert calls == 1
