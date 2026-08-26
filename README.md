# LLMInfra

**English** | [简体中文](README.zh-CN.md)

LLMInfra is an educational PyTorch toolkit implementing modern Transformer,
attention, MoE, positional encoding, inference, and speculative decoding
components.

## Features

- Attention: MHA, MQA, GQA, MLA, sliding-window, block-sparse, compressed
  sparse, linear, Lightning, Gated DeltaNet, ALiBi, hybrid, and ring attention.
- FlashAttention: pure-PyTorch FA1-FA4 tiled forward and backward references.
- Positional encodings: learned and sinusoidal absolute positions, RoPE, YaRN,
  Dynamic NTK, LongRoPE, ALiBi, T5 relative bias, 2D positions, and MRoPE.
- Transformer layers: RMSNorm, LayerNorm, SwiGLU and gated FFNs, configurable
  Transformer blocks, Mamba2-style state-space layers, and hybrid stacks.
- Models: decoder-only, prefix-LM, encoder-only, encoder-decoder, multimodal,
  classification, reward, token, and embedding heads.
- Mixture of Experts: top-k and expert-choice routing, shared experts, latent
  MoE, expert parallelism, load-balancing loss, and router z-loss.
- Inference infrastructure: paged KV caches, disk and tiered KV offload, paged
  attention, and block-sparse indexing.
- Speculative decoding: standard draft verification, N-Gram, EAGLE 1-3,
  Medusa, MTP, DFlash block-diffusion drafting, and DSpark dynamic scheduling.
- Quantization: portable fake INT4, INT8, FP8 quantization and QAT wrappers.

The implementations prioritize clarity, explicit tensor shapes, numerical
correctness, and testability. They are reference and teaching implementations,
not replacements for optimized production kernels.

## Requirements

- Python 3.10 or newer
- PyTorch

## Installation

Install from the repository:

```bash
git clone https://github.com/jianzhnie/LLMInfra.git
cd LLMInfra
python -m pip install -e .
```

## Quick Start

```python
import torch

from llminfra import MultiHeadAttention

attention = MultiHeadAttention(hidden_size=512, num_heads=8)
hidden_state = torch.randn(2, 128, 512)

output = attention(hidden_state)
print(output.shape)  # torch.Size([2, 128, 512])

output, weights = attention(
    hidden_state,
    return_attention_weights=True,
)
print(weights.shape)  # torch.Size([2, 8, 128, 128])
```

Build modules through the public registry:

```python
from llminfra import build_attention, list_attentions

attention = build_attention(
    "gqa",
    hidden_size=512,
    num_heads=8,
    num_kv_groups=2,
)
print(list_attentions())
```

Use the educational FlashAttention interface:

```python
import torch

from llminfra import flash_attention

query = torch.randn(2, 8, 128, 64, requires_grad=True)
key = torch.randn(2, 8, 128, 64, requires_grad=True)
value = torch.randn(2, 8, 128, 64, requires_grad=True)

output = flash_attention(query, key, value, version="fa2", causal=True)
output.sum().backward()
```

## Package Layout

| Path | Responsibility |
|---|---|
| `llminfra/attention/` | Dense, sparse, linear, latent, and hybrid attention |
| `llminfra/flash_attention/` | Educational FlashAttention v1-v4 references |
| `llminfra/positional/` | Absolute, relative, rotary, and long-context positions |
| `llminfra/layers/` | FFN, normalization, SSM, and Transformer layers |
| `llminfra/models/` | Language, encoder, encoder-decoder, and multimodal models |
| `llminfra/moe/` | Routers, experts, and Mixture-of-Experts modules |
| `llminfra/inference/` | KV caches, paging, offload, and sparse indexing |
| `llminfra/spec_decode/` | N-Gram, EAGLE, Medusa, MTP, DFlash, and DSpark |
| `llminfra/quantization.py` | Fake quantization and QAT utilities |
| `examples/` | Runnable per-module benchmark scripts |
| `tests/` | Tests organized to mirror the source package |

Public APIs are re-exported from `llminfra`. Import implementation modules
directly only when extending or studying their internals.

## Development

Run the test suite:

```bash
pytest
```

Run formatting, lint, and type checks:

```bash
ruff format --check .
ruff check .
mypy llminfra --strict
pre-commit run --all-files
```

The CPU test suite covers functional behavior, masks, gradients, public APIs,
cache semantics, and speculative decoding. CUDA-specific FlashAttention tests
are skipped when CUDA is unavailable.

## Documentation

- [Chinese README](README.zh-CN.md)
- [Transformer Review 2026](docs/transformers_review_2026.md)
- [Package layout and naming conventions](llminfra/README.md)

## License

Licensed under the [Apache License 2.0](LICENSE).
