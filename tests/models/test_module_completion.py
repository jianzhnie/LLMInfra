"""Coverage for the complete mainstream Transformer reference surface."""

import pytest
import torch

import llminfra
from llminfra import (
    DynamicSparseAttention,
    EmbeddingHead,
    EncoderOnlyModel,
    HierarchicalCompressedAttention,
    KimiDeltaAttention,
    LearnedAbsolutePositionEmbedding,
    MiniMaxSparseAttention,
    NoPositionEncoding,
    PrefixLMModel,
    RewardModelHead,
    SequenceClassificationHead,
    SinusoidalPositionEmbedding,
    T5RelativePositionBias,
    TokenClassificationHead,
    build_attention,
    build_positional_encoding,
    get_activation,
    list_attentions,
    list_positional_encodings,
    pool_hidden_state,
)


def test_new_mainstream_modules_are_public_and_registered():
    names = {
        "DynamicSparseAttention",
        "EmbeddingHead",
        "EncoderOnlyModel",
        "HierarchicalCompressedAttention",
        "LearnedAbsolutePositionEmbedding",
        "MiniMaxSparseAttention",
        "NoPositionEncoding",
        "PrefixLMModel",
        "RewardModelHead",
        "SequenceClassificationHead",
        "SinusoidalPositionEmbedding",
        "T5RelativePositionBias",
        "TokenClassificationHead",
    }
    assert names <= set(llminfra.__all__)
    assert all(hasattr(llminfra, name) for name in names)
    assert {"dsa", "msa", "hca"} <= set(list_attentions())
    assert {"kda", "kimi_delta"} <= set(list_attentions())
    assert {"learned", "sinusoidal", "t5_bias", "none"} <= set(
        list_positional_encodings()
    )


def test_dynamic_sparse_attention_masks_future_and_trains_index_scores():
    module = DynamicSparseAttention(
        hidden_size=32,
        num_heads=4,
        block_size=2,
        top_k=2,
        max_seq_len=16,
        num_kv_groups=2,
    )
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    output, weights = module(hidden, return_attention_weights=True)

    assert output.shape == hidden.shape
    assert weights.shape == (2, 4, 8, 8)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    assert module.select_blocks(hidden).shape == (2, 4, 4, 2)

    routing = module.routing_scores(hidden)
    routing.diagonal(dim1=-2, dim2=-1).sum().backward(retain_graph=True)
    assert module.indexer.query_proj.weight.grad is not None
    output.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_minimax_sparse_attention_shares_indexer_within_gqa_group():
    module = MiniMaxSparseAttention(
        hidden_size=32,
        num_heads=4,
        num_kv_groups=2,
        block_size=2,
        top_k=2,
        max_seq_len=16,
    )
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    indices = module.select_blocks(hidden)
    torch.testing.assert_close(indices[:, 0], indices[:, 1])
    torch.testing.assert_close(indices[:, 2], indices[:, 3])
    assert module.routing_scores(hidden).shape == (2, 2, 4, 4)

    output = module(hidden)
    output.square().mean().backward()
    assert output.shape == hidden.shape
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_kda_delta_rule_is_causal_and_has_no_quadratic_weights():
    module = KimiDeltaAttention(
        hidden_size=32,
        num_heads=4,
        feature_dim=8,
        output_gate=True,
    ).eval()
    hidden = torch.randn(2, 8, 32, requires_grad=True)
    output = module(hidden)
    assert output.shape == hidden.shape

    causal_mask = torch.tril(torch.ones(2, 1, 8, 8, dtype=torch.bool))
    torch.testing.assert_close(
        module(hidden.detach(), attention_mask=causal_mask),
        module(hidden.detach()),
    )

    perturbed = hidden.detach().clone()
    perturbed[:, -1] += 10.0
    with torch.no_grad():
        baseline = module(hidden.detach())
        changed = module(perturbed)
    torch.testing.assert_close(baseline[:, :-1], changed[:, :-1])
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    with pytest.raises(ValueError, match="does not materialize"):
        module(hidden.detach(), return_attention_weights=True)


def test_hierarchical_compressed_attention_mixes_both_paths():
    module = HierarchicalCompressedAttention(
        hidden_size=32,
        num_heads=4,
        fine_compress_ratio=2,
        coarse_compress_ratio=4,
        fine_top_k=2,
        max_seq_len=16,
        num_kv_groups=2,
    )
    hidden = torch.randn(2, 16, 32, requires_grad=True)
    output, fine_weights = module(hidden, return_attention_weights=True)
    assert output.shape == hidden.shape
    assert fine_weights.shape == (2, 4, 16, 8)
    output.sum().backward()
    assert module.mix_logit.grad is not None
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()

    with pytest.raises(ValueError, match="divisible"):
        module(torch.randn(1, 14, 32))


def test_sparse_variants_build_from_registry():
    dsa = build_attention(
        "dsa",
        hidden_size=32,
        num_heads=4,
        block_size=2,
        top_k=2,
        max_seq_len=16,
    )
    msa = build_attention(
        "msa",
        hidden_size=32,
        num_heads=4,
        num_kv_groups=2,
        block_size=2,
        top_k=2,
        max_seq_len=16,
    )
    hca = build_attention(
        "hca",
        hidden_size=32,
        num_heads=4,
        fine_compress_ratio=2,
        coarse_compress_ratio=4,
        max_seq_len=16,
    )
    assert isinstance(dsa, DynamicSparseAttention)
    assert isinstance(msa, MiniMaxSparseAttention)
    assert isinstance(hca, HierarchicalCompressedAttention)


def test_classic_modules_and_factory():
    hidden = torch.zeros(2, 4, 8)
    sinusoidal = SinusoidalPositionEmbedding(8, max_seq_len=8)
    output = sinusoidal(hidden)
    torch.testing.assert_close(output[0, 0, 0::2], torch.zeros(4))
    torch.testing.assert_close(output[0, 0, 1::2], torch.ones(4))

    learned = LearnedAbsolutePositionEmbedding(8, max_seq_len=8)
    learned_output = learned(hidden)
    learned_output.sum().backward()
    assert learned.embedding.weight.grad is not None

    nope = NoPositionEncoding(8, max_seq_len=8)
    assert nope(hidden) is hidden
    assert isinstance(
        build_positional_encoding("learned", dim=8, max_seq_len=8),
        LearnedAbsolutePositionEmbedding,
    )
    assert isinstance(
        build_positional_encoding("sinusoidal", dim=8, max_seq_len=8),
        SinusoidalPositionEmbedding,
    )
    assert get_activation("gelu_exact") is not None
    torch.testing.assert_close(
        get_activation("swish")(hidden),
        get_activation("silu")(hidden),
    )


def test_t5_relative_position_bias_supports_cross_attention_lengths():
    bias = T5RelativePositionBias(
        num_heads=4,
        num_buckets=32,
        max_distance=128,
        max_seq_len=16,
    )
    values = bias((3, 5))
    assert values.shape == (1, 4, 3, 5)
    values.sum().backward()
    assert bias.relative_attention_bias.weight.grad is not None
    assert isinstance(
        build_positional_encoding(
            "t5_bias",
            dim=8,
            num_heads=4,
            max_seq_len=16,
        ),
        T5RelativePositionBias,
    )


def test_padding_aware_pooling_and_task_heads():
    hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).view(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
    expected_last = hidden[:, 1:3][torch.arange(2), torch.tensor([0, 1])]
    torch.testing.assert_close(pool_hidden_state(hidden, mask, "last"), expected_last)
    expected_mean = torch.stack([hidden[0, :2].mean(0), hidden[1, 1:3].mean(0)])
    torch.testing.assert_close(pool_hidden_state(hidden, mask, "mean"), expected_mean)

    sequence = SequenceClassificationHead(8, 3, pooling="mean")
    token = TokenClassificationHead(8, 5)
    reward = RewardModelHead(8)
    embedding = EmbeddingHead(8, output_size=6)
    assert sequence(hidden, mask).shape == (2, 3)
    assert token(hidden).shape == (2, 4, 5)
    assert reward(hidden, mask).shape == (2,)
    vectors = embedding(hidden, mask)
    assert vectors.shape == (2, 6)
    torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(2))


def test_encoder_only_model_is_bidirectional_and_runs_heads():
    head = SequenceClassificationHead(16, 3)
    model = EncoderOnlyModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        intermediate_size=32,
        max_seq_len=16,
        type_vocab_size=2,
        output_head=head,
        tie_word_embeddings=True,
    ).eval()
    input_ids = torch.randint(0, 32, (2, 7))
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[0, -2:] = False
    output = model(input_ids, attention_mask=mask)
    assert output.last_hidden_state.shape == (2, 7, 16)
    assert output.head_output is not None and output.head_output.shape == (2, 3)
    assert output.language_model_logits is not None
    assert output.language_model_logits.shape == (2, 7, 32)
    assert torch.count_nonzero(output.last_hidden_state[0, -2:]) == 0
    assert model.lm_head is not None
    assert model.lm_head.weight is model.embed_tokens.weight

    perturbed = input_ids.clone()
    perturbed[1, -1] = (perturbed[1, -1] + 1) % 32
    with torch.no_grad():
        base = model(input_ids).last_hidden_state
        changed = model(perturbed).last_hidden_state
    assert not torch.allclose(base[1, 0], changed[1, 0])


def test_prefix_lm_supports_per_example_prefix_lengths():
    model = PrefixLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        intermediate_size=32,
        max_seq_len=16,
        attention_name="mha",
    ).eval()
    input_ids = torch.randint(0, 32, (2, 8))
    perturbed = input_ids.clone()
    perturbed[:, 4] = (perturbed[:, 4] + 1) % 32
    prefix_lengths = torch.tensor([2, 5])
    with torch.no_grad():
        base = model(input_ids, prefix_lengths=prefix_lengths)
        changed = model(perturbed, prefix_lengths=prefix_lengths)

    torch.testing.assert_close(base[0, 0], changed[0, 0])
    assert not torch.allclose(base[1, 0], changed[1, 0])
    with pytest.raises(ValueError, match="shape"):
        model(input_ids, prefix_lengths=torch.tensor([2]))
