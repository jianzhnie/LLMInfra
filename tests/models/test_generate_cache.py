"""Tests for the KV-cache decoding path of ``CausalLMModel.generate``."""

import pytest
import torch

from llminfra import CausalLMModel


def _make_model(attention_name="gqa", seed=0, **kwargs):
    torch.manual_seed(seed)
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        intermediate_size=32,
        max_seq_len=64,
        attention_name=attention_name,
        **kwargs,
    )
    return model.eval()


@pytest.mark.parametrize("attention_name", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("prompt_len", [1, 3, 8])
@pytest.mark.parametrize("max_new_tokens", [1, 5])
def test_cache_matches_naive_greedy(attention_name, prompt_len, max_new_tokens):
    model = _make_model(attention_name)
    generator = torch.Generator().manual_seed(7)
    prompt = torch.randint(0, 32, (2, prompt_len), generator=generator)
    naive = model.generate(prompt, max_new_tokens=max_new_tokens)
    cached = model.generate(prompt, max_new_tokens=max_new_tokens, use_cache=True)
    assert torch.equal(naive.sequences, cached.sequences)


def test_cache_matches_naive_with_temperature_sampling():
    model = _make_model("gqa")
    prompt = torch.randint(0, 32, (2, 4), generator=torch.Generator().manual_seed(7))
    torch.manual_seed(123)
    naive = model.generate(prompt, max_new_tokens=6, temperature=1.0)
    torch.manual_seed(123)
    cached = model.generate(prompt, max_new_tokens=6, temperature=1.0, use_cache=True)
    # Equal outputs under the same seed prove both paths consume the RNG
    # identically (one sampling draw per generated token).
    assert torch.equal(naive.sequences, cached.sequences)


@pytest.mark.parametrize(
    "sample_kwargs", [{"top_k": 5}, {"top_p": 0.8}, {"top_k": 5, "top_p": 0.9}]
)
def test_cache_matches_naive_with_top_k_top_p(sample_kwargs):
    model = _make_model("mha")
    prompt = torch.randint(0, 32, (2, 4), generator=torch.Generator().manual_seed(7))
    torch.manual_seed(123)
    naive = model.generate(prompt, max_new_tokens=6, temperature=0.8, **sample_kwargs)
    torch.manual_seed(123)
    cached = model.generate(
        prompt, max_new_tokens=6, temperature=0.8, use_cache=True, **sample_kwargs
    )
    assert torch.equal(naive.sequences, cached.sequences)


def test_cache_decode_steps_process_one_token():
    model = _make_model("gqa")
    prompt_len, max_new_tokens = 4, 5
    prompt = torch.randint(0, 32, (2, prompt_len))

    naive_lens = []
    original_forward = model.forward

    def recording_forward(input_ids=None, **kwargs):
        naive_lens.append(input_ids.size(1))
        return original_forward(input_ids=input_ids, **kwargs)

    model.forward = recording_forward
    model.generate(prompt, max_new_tokens=max_new_tokens)
    model.forward = original_forward
    # The naive path recomputes the full sequence at every step.
    assert naive_lens == [prompt_len + i for i in range(max_new_tokens)]

    cache_lens = []
    original_cached_forward = model._forward_with_cache

    def recording_cached_forward(input_ids, past_key_values=None, **kwargs):
        cache_lens.append(input_ids.size(1))
        return original_cached_forward(input_ids, past_key_values, **kwargs)

    model._forward_with_cache = recording_cached_forward
    model.generate(prompt, max_new_tokens=max_new_tokens, use_cache=True)
    model._forward_with_cache = original_cached_forward
    # The cache path prefills the prompt once, then feeds one token per step.
    assert cache_lens == [prompt_len] + [1] * (max_new_tokens - 1)


def test_prefill_logits_match_full_forward():
    model = _make_model("gqa")
    ids = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(7))
    logits_full = model(ids)
    logits_prefill, presents = model._forward_with_cache(ids)
    assert torch.allclose(logits_full, logits_prefill, atol=1e-5)
    assert len(presents) == model.num_layers
    for key, value in presents:
        assert key.shape == (2, model.num_heads, 6, model.hidden_size // 4)
        assert value.shape == key.shape


def test_incremental_logits_match_full_forward():
    model = _make_model("mqa")
    ids = torch.randint(0, 32, (2, 6), generator=torch.Generator().manual_seed(7))
    logits_full = model(ids)
    past = None
    step_logits = []
    for position in range(ids.size(1)):
        logits, past = model._forward_with_cache(ids[:, position : position + 1], past)
        step_logits.append(logits)
    logits_incremental = torch.cat(step_logits, dim=1)
    assert torch.allclose(logits_full, logits_incremental, atol=1e-5)


def test_cache_rejects_unsupported_attention():
    model = _make_model("linear")
    prompt = torch.randint(0, 32, (1, 3))
    with pytest.raises(ValueError, match="does not support KV-cache"):
        model.generate(prompt, max_new_tokens=2, use_cache=True)


def test_cache_rejects_unsupported_positional():
    model = _make_model("gqa", positional="sinusoidal")
    prompt = torch.randint(0, 32, (1, 3))
    with pytest.raises(ValueError, match="positional='rope' or 'none'"):
        model.generate(prompt, max_new_tokens=2, use_cache=True)


def test_cache_supports_none_positional():
    model = _make_model("mha", positional="none")
    prompt = torch.randint(0, 32, (2, 4), generator=torch.Generator().manual_seed(7))
    naive = model.generate(prompt, max_new_tokens=4)
    cached = model.generate(prompt, max_new_tokens=4, use_cache=True)
    assert torch.equal(naive.sequences, cached.sequences)


def test_cache_rejects_draft_model():
    model = _make_model("gqa")
    prompt = torch.randint(0, 32, (1, 3))
    with pytest.raises(ValueError, match="draft_model"):
        model.generate(prompt, max_new_tokens=2, use_cache=True, draft_model=model)


def test_chunked_prefill_matches_full_forward():
    """q_len > 1 decode chunks exercise the right-aligned causal mask and the
    RoPE offset slice ``cos[past_len:total_len]``; the concatenated logits
    must match a single full forward."""
    model = _make_model("gqa")
    ids = torch.randint(0, 32, (2, 11), generator=torch.Generator().manual_seed(3))
    logits_full = model(ids)
    past = None
    chunks = []
    for chunk in (ids[:, :4], ids[:, 4:7], ids[:, 7:]):
        logits, past = model._forward_with_cache(chunk, past)
        chunks.append(logits)
    assert torch.allclose(logits_full, torch.cat(chunks, dim=1), atol=1e-5)


def test_cache_attention_mask_matches_naive_forward():
    """A 2D padding keep-mask over all ``past_len + q_len`` key positions must
    reproduce the naive forward's logits at every non-padded position."""
    model = _make_model("gqa")
    ids = torch.randint(0, 32, (2, 8), generator=torch.Generator().manual_seed(3))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
    logits_full = model(ids, attention_mask=mask)
    past = None
    steps = []
    for position in range(ids.size(1)):
        logits, past = model._forward_with_cache(
            ids[:, position : position + 1],
            past,
            attention_mask=mask[:, : position + 1],
        )
        steps.append(logits)
    logits_incremental = torch.cat(steps, dim=1)
    # The naive forward zeroes logits at padded query positions; mask those
    # slots out of the comparison on both sides.
    keep = mask.bool().unsqueeze(-1)
    assert torch.allclose(logits_full * keep, logits_incremental * keep, atol=1e-5)


def test_generate_up_to_last_forward_capacity():
    """The largest sequence any forward pass ingests is prompt + max_new - 1:
    the final generated token is never fed back. A prompt that already fills
    max_seq_len must still be able to emit one token (and no more)."""
    torch.manual_seed(0)
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        intermediate_size=32,
        max_seq_len=8,
        attention_name="gqa",
    ).eval()
    prompt = torch.randint(0, 32, (1, 8), generator=torch.Generator().manual_seed(7))
    naive = model.generate(prompt, max_new_tokens=1)
    cached = model.generate(prompt, max_new_tokens=1, use_cache=True)
    assert naive.sequences.shape == (1, 9)
    assert torch.equal(naive.sequences, cached.sequences)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(prompt, max_new_tokens=2)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(prompt, max_new_tokens=2, use_cache=True)


def test_cache_eos_and_pad_match_naive():
    model = _make_model("gqa")
    prompt = torch.randint(0, 32, (2, 4), generator=torch.Generator().manual_seed(7))
    reference = model.generate(prompt, max_new_tokens=6)
    # Use a token the greedy path actually emits as eos so the stop logic runs.
    eos_token_id = int(reference.sequences[0, 5])
    naive = model.generate(
        prompt, max_new_tokens=6, eos_token_id=eos_token_id, pad_token_id=9
    )
    cached = model.generate(
        prompt,
        max_new_tokens=6,
        use_cache=True,
        eos_token_id=eos_token_id,
        pad_token_id=9,
    )
    assert torch.equal(naive.sequences, cached.sequences)
    row = cached.sequences[0]
    eos_positions = (row[4:] == eos_token_id).nonzero().flatten()
    assert len(eos_positions) == 1
    assert (row[4 + eos_positions[0] + 1 :] == 9).all()
