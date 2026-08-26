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


def _make_generate_model(max_seq_len: int = 32) -> CausalLMModel:
    torch.manual_seed(0)
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        max_seq_len=max_seq_len,
        attention_name="mha",
    )
    model.eval()
    return model


def test_generate_greedy_matches_manual_argmax_loop():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 5))
    output = model.generate(prompt, max_new_tokens=6)
    assert output.num_drafted == 0 and output.num_accepted == 0

    with torch.no_grad():
        expected = prompt
        for _ in range(6):
            next_token = model(expected)[:, -1].argmax(dim=-1)
            expected = torch.cat([expected, next_token[:, None]], dim=-1)
    assert torch.equal(output.sequences, expected)


def test_generate_greedy_is_deterministic():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 5))
    first = model.generate(prompt, max_new_tokens=5)
    second = model.generate(prompt, max_new_tokens=5)
    assert torch.equal(first.sequences, second.sequences)


def test_generate_respects_max_new_tokens():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))
    output = model.generate(prompt, max_new_tokens=3)
    assert output.sequences.shape == (2, 4 + 3)
    assert torch.equal(output.sequences[:, :4], prompt)


def test_generate_stops_at_eos_and_pads_remaining():
    model = _make_generate_model()
    # Zero logits make argmax pick token 0 everywhere, so every row emits
    # eos immediately and the loop stops after a single generated token.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    prompt = torch.randint(1, 32, (2, 4))
    output = model.generate(prompt, max_new_tokens=5, eos_token_id=0, pad_token_id=9)
    assert output.sequences.shape == (2, 4 + 1)
    assert torch.equal(output.sequences[:, 4], torch.zeros(2, dtype=torch.long))


def test_generate_eos_stop_is_per_row():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))

    def scripted_forward(input_ids, **kwargs):
        # Row 0 emits eos (3) on the first step; row 1 never does.
        logits = torch.zeros(2, input_ids.size(1), 32)
        logits[0, -1, 3] = 1.0
        logits[1, -1, 7] = 1.0
        return logits

    original_forward = model.forward
    model.forward = scripted_forward  # type: ignore[assignment]
    try:
        output = model.generate(
            prompt, max_new_tokens=3, eos_token_id=3, pad_token_id=9
        )
    finally:
        model.forward = original_forward  # type: ignore[assignment]

    assert output.sequences.shape == (2, 4 + 3)
    # Row 0: eos then padding; row 1: keeps generating to max_new_tokens.
    assert output.sequences[0, 4:].tolist() == [3, 9, 9]
    assert output.sequences[1, 4:].tolist() == [7, 7, 7]


def test_generate_temperature_sampling_runs_in_bounds():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))
    torch.manual_seed(1)
    output = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    generated = output.sequences[:, 4:]
    assert generated.shape == (2, 5)
    assert (generated >= 0).all() and (generated < 32).all()


def test_generate_speculative_matches_greedy_when_draft_is_target():
    model = _make_generate_model()
    prompt = torch.randint(0, 32, (2, 4))
    greedy = model.generate(prompt, max_new_tokens=6)
    speculative = model.generate(
        prompt,
        max_new_tokens=6,
        draft_model=model,
        num_speculative_tokens=3,
    )
    # Speculative decoding with an identical draft is lossless: same tokens,
    # and every drafted token is accepted.
    assert torch.equal(speculative.sequences, greedy.sequences)
    assert speculative.num_drafted > 0
    assert speculative.num_drafted == speculative.num_accepted


def test_generate_rejects_invalid_arguments():
    model = _make_generate_model(max_seq_len=8)
    prompt = torch.randint(0, 32, (2, 4))
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(prompt, max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(prompt, max_new_tokens=2, temperature=-0.5)
    with pytest.raises(ValueError, match="batch, seq_len"):
        model.generate(torch.randint(0, 32, (4,)))
    with pytest.raises(ValueError, match="at least one token"):
        model.generate(torch.empty(2, 0, dtype=torch.long))
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(prompt, max_new_tokens=6)
