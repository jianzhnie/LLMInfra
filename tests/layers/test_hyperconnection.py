import pytest
import torch

from llminfra import ManifoldConstrainedHyperConnection, TransformerBlock


def test_hyperconnection_shape_constraint_and_gradients():
    module = ManifoldConstrainedHyperConnection(8, hc_mult=4, sinkhorn_iters=20)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    output = module(hidden, torch.randn_like(hidden))
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert module.logits.grad is not None


def test_hyperconnection_mixing_matrix_is_doubly_stochastic():
    module = ManifoldConstrainedHyperConnection(4, hc_mult=3, sinkhorn_iters=30)
    matrix = module.mixing_matrix()
    torch.testing.assert_close(matrix.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(matrix.sum(dim=-2), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_hyperconnection_logits_change_output():
    torch.manual_seed(0)
    module = ManifoldConstrainedHyperConnection(8, hc_mult=4).eval()
    hidden = torch.randn(2, 3, 8)
    branch = torch.randn_like(hidden)
    baseline = module(hidden, branch)
    with torch.no_grad():
        module.logits.copy_(torch.randn_like(module.logits) * 5)
    assert not torch.allclose(module(hidden, branch), baseline)


def test_hyperconnection_sinkhorn_stable_for_large_logits():
    module = ManifoldConstrainedHyperConnection(8, hc_mult=4).eval()
    with torch.no_grad():
        module.logits.fill_(100.0)
    matrix = module.mixing_matrix()
    assert torch.isfinite(matrix).all()
    output = module(torch.randn(2, 3, 8), torch.randn(2, 3, 8))
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    "kwargs", [{"hidden_size": 0}, {"hidden_size": 4, "hc_mult": 0}]
)
def test_hyperconnection_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        ManifoldConstrainedHyperConnection(**kwargs)


def test_transformer_block_integrates_mhc_on_both_residual_branches():
    block = TransformerBlock(
        hidden_size=16,
        num_heads=4,
        intermediate_size=32,
        manifold_hyper_connection=True,
        hc_mult=4,
        sinkhorn_iters=20,
    )
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    output = block(hidden)
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert block.attention_mhc is not None
    assert block.ffn_mhc is not None
    assert block.attention_mhc.logits.grad is not None
    assert block.ffn_mhc.logits.grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_residual": True},
        {"norm_style": "deepnorm"},
        {"parallel": True},
    ],
)
def test_transformer_block_rejects_incompatible_mhc_layouts(kwargs):
    with pytest.raises(ValueError, match="manifold_hyper_connection"):
        TransformerBlock(
            hidden_size=16,
            num_heads=4,
            intermediate_size=32,
            manifold_hyper_connection=True,
            **kwargs,
        )


def test_hyper_connection_validates_arguments():
    with pytest.raises(ValueError, match="sinkhorn_iters"):
        ManifoldConstrainedHyperConnection(16, sinkhorn_iters=0)
    layer = ManifoldConstrainedHyperConnection(16)
    with pytest.raises(ValueError, match="identical shapes"):
        layer(torch.randn(2, 4, 16), torch.randn(2, 5, 16))
    with pytest.raises(ValueError, match="last dimension"):
        layer(torch.randn(2, 4, 8), torch.randn(2, 4, 8))
