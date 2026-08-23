"""Regression tests for model-level mainstream Transformer extensions."""

import tempfile

import pytest
import torch

import llminfra
from llminfra import (
    DSparkDecoder,
    DSparkScheduler,
    HybridLayerStack,
    Mamba2Layer,
    MultimodalCausalLM,
    TieredKVCache,
    TopKRouter,
    build_multimodal_position_ids,
    build_positional_encoding,
    distributed_ring_attention,
)


def test_mainstream_modules_are_available_from_package_root():
    public_names = {
        "ClampedSwiGLUFFN",
        "CrossAttention",
        "DeepNorm",
        "EncoderDecoderModel",
        "ExpertChoiceRouter",
        "GeGLUFFN",
        "HybridSSMBlock",
        "LayerNorm",
        "LayerScale",
        "ReGLUFFN",
        "build_feed_forward",
        "router_z_loss",
    }
    assert public_names <= set(llminfra.__all__)
    assert all(hasattr(llminfra, name) for name in public_names)


def test_gumbel_router_has_straight_through_gradient():
    router = TopKRouter(
        hidden_size=8,
        num_experts=4,
        top_k=2,
        routing_strategy="gumbel",
        gumbel_temperature=0.5,
    )
    x = torch.randn(3, 8, requires_grad=True)
    weights, indices = router(x)
    assert indices.shape == (3, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
    weights[:, 0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_hybrid_layer_stack_threads_only_ssm_states():
    stack = HybridLayerStack(
        hidden_size=16,
        num_heads=4,
        intermediate_size=32,
        layer_map="linear:ssm:full:ssm",
        d_state=4,
        shared_full_attention=True,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)
    output, states = stack(x, return_state=True, scan="chunked", chunk_size=2)
    assert output.shape == x.shape
    assert states[0] is None and states[2] is None
    assert states[1] is not None and states[3] is not None
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mamba_streaming_state_includes_convolution_history():
    layer = Mamba2Layer(8, d_state=3, conv_kernel=3).eval()
    x = torch.randn(1, 7, 8)
    full, _ = layer(x)
    first, state = layer(x[:, :3])
    second, _ = layer(x[:, 3:], state=state)
    torch.testing.assert_close(torch.cat([first, second], dim=1), full)


def test_multimodal_position_ids_and_early_fusion():
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    positions = build_multimodal_position_ids(grid, text_length=3)
    assert positions.shape == (3, 2, 7)
    assert torch.equal(positions[:, 0, -3:], torch.tensor([[2, 3, 4]] * 3))

    model = MultimodalCausalLM(
        vocab_size=32,
        vision_dim=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=32,
        mrope_section=(1, 1, 6),
        max_seq_len=16,
        num_mtp_predictions=2,
    )
    ids = torch.randint(0, 32, (2, 3))
    vision = torch.randn(2, 4, 8)
    output = model(ids, vision, grid, labels=ids, return_mtp=True)
    assert output.logits.shape == (2, 3, 32)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.mtp_logits is not None and len(output.mtp_logits) == 2
    assert output.alignment_logits is not None


def test_longrope_reference_preset_factory():
    encoding = build_positional_encoding(
        "longrope",
        dim=8,
        preset="reference_uniform_256k",
    )
    output = encoding(torch.randn(1, 1, 8, 8))
    assert output.shape == (1, 1, 8, 8)
    assert torch.isfinite(output).all()


def test_tiered_kv_cache_promotes_and_evicts():
    with tempfile.TemporaryDirectory() as directory:
        cache = TieredKVCache(
            directory,
            max_hbm_entries=1,
            max_cpu_entries=1,
            hbm_device="cpu",
        )
        for sequence_id in range(3):
            key = torch.full((2, 1), float(sequence_id))
            cache.put(sequence_id, key, key + 1)
        assert cache.tier_counts() == {"hbm": 1, "cpu": 1, "nvme": 1}
        key, value = cache.get(0)
        torch.testing.assert_close(key, torch.zeros(2, 1))
        torch.testing.assert_close(value, torch.ones(2, 1))


def test_dspark_scheduler_and_decoder():
    scheduler = DSparkScheduler((1, 2, 4), initial_length=1)
    assert scheduler.update(1, 1) == 2
    assert scheduler.update(0, 2) == 1

    def model(input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8)
        logits[..., 1] = 5
        return logits

    decoder = DSparkDecoder(model, model, scheduler=DSparkScheduler((1, 2)))
    output = decoder(torch.zeros(2, 3, dtype=torch.long))
    assert output.shape[0] == 2
    assert decoder.scheduler.draft_length == 2


def test_distributed_ring_requires_initialized_process_group():
    q = torch.randn(1, 1, 2, 4)
    with pytest.raises(RuntimeError, match=r"torch\.distributed"):
        distributed_ring_attention(q, q, q)
