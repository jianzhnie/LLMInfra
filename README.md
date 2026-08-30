# LLMInfra

**English** | [简体中文](README.zh-CN.md)

LLMInfra is an educational PyTorch toolkit implementing modern Transformer,
attention, MoE, positional encoding, inference, generation, and speculative
decoding components.

## Features

- Attention: MHA, MQA, GQA, MLA, sliding-window, block-sparse, compressed
  sparse, dynamic sparse, MiniMax sparse, hierarchical compressed, FlashMLA,
  ALiBi, hybrid, and ring attention — plus logit soft-capping (Gemma 2),
  attention sinks (GPT-OSS), and sigmoid output gating (Qwen3-Next) as
  opt-in options.
- Linear-time sequence layers: Linear Attention, Lightning Attention,
  Gated DeltaNet, Kimi Delta Attention (KDA), RetNet retention, GLA, RWKV,
  and TTT-Linear.
- FlashAttention: pure-PyTorch FA1-FA4 tiled forward and backward references.
- Positional encodings: learned and sinusoidal absolute positions, RoPE, YaRN,
  Dynamic NTK, LongRoPE, partial RoPE, ALiBi (BLOOM slope schedule), T5
  relative bias, 2D positions, and MRoPE.
- Transformer layers: RMSNorm, LayerNorm, DeepNorm, LayerScale, SwiGLU and
  gated FFNs (incl. GPT-OSS clamped SwiGLU), configurable Transformer blocks,
  Mamba2-style state-space layers, manifold-constrained hyper-connections
  (mHC), and hybrid stacks.
- Models: decoder-only, prefix-LM, encoder-only, encoder-decoder, multimodal,
  classification, reward, token, and embedding heads. `CausalLMModel.generate`
  supports a naive full-recompute loop, a KV-cache incremental path
  (`use_cache=True`), and speculative decoding.
- Generation: top-k / top-p / min-p / repetition-penalty logits processors,
  plus standalone `generate` / `generate_with_cache` decode loops that can
  drive any model exposing the matching callables.
- Mixture of Experts: top-k and expert-choice routing, shared experts, latent
  MoE, expert parallelism, load-balancing loss, and router z-loss.
- Inference infrastructure: paged KV caches, quantized KV caches (KIVI-style
  per-channel K / per-token V, INT8/INT4), disk and tiered KV offload, paged
  attention, and block-sparse indexing.
- Speculative decoding: standard draft verification, N-Gram, EAGLE 1-3,
  Medusa, MTP, DFlash block-diffusion drafting, and DSpark dynamic scheduling.
- Quantization: portable fake INT4, INT8, FP8 (E4M3), MXFP4 and NVFP4
  quantization with QAT wrappers.
- Optimizers: Muon (Newton-Schulz orthogonalization with AdamW fallback) and
  MuonClip (Kimi K2 style QK-clip).

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

Generate from a small model, naive or with a KV cache:

```python
import torch

from llminfra import CausalLMModel

model = CausalLMModel(
    vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8,
    intermediate_size=512,
)
model.eval()  # disable dropout so the two paths are deterministic
prompt = torch.randint(0, 1000, (1, 8))

naive = model.generate(prompt, max_new_tokens=16)
cached = model.generate(prompt, max_new_tokens=16, use_cache=True)
assert torch.equal(naive.sequences, cached.sequences)  # same tokens, less work
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
| `llminfra/inference/` | KV caches (paged / quantized / tiered) and sparse indexing |
| `llminfra/spec_decode/` | N-Gram, EAGLE, Medusa, MTP, DFlash, and DSpark |
| `llminfra/generation.py` | Logits processors and greedy/sampling decode loops |
| `llminfra/optimizers.py` | Muon / MuonClip optimizers |
| `llminfra/quantization.py` | Fake quantization (INT4/INT8/FP8/MXFP4/NVFP4) and QAT |
| `examples/` | Benchmarks (`benchmark/`), training demos, and a real-model generate example |
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
cache semantics, generation, and speculative decoding. CUDA-specific
FlashAttention tests are skipped when CUDA is unavailable.

## Documentation

- [Chinese README](README.zh-CN.md)
- [Module ↔ paper ↔ transformers cross-reference](llminfra_guide.md)
- [Transformer Review 2026](docs/transformers_review_2026.md)
- [Package layout and naming conventions](llminfra/README.md)

## License

Licensed under the [Apache License 2.0](LICENSE).
