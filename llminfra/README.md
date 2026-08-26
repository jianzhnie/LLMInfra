# llminfra package layout

The package is organized by responsibility. Prefer imports from `llminfra`
for public APIs and use subpackages when working on an implementation.

| Directory or file | Responsibility |
|---|---|
| `attention/` | Full, grouped, latent, sparse, linear and hybrid attention |
| `positional/` | Absolute, relative, rotary and long-context position encodings |
| `layers/` | Feed-forward, normalization, SSM, residual and Transformer blocks |
| `moe/` | Routers, experts, latent MoE and expert-parallel references |
| `inference/` | KV paging/offload and block-sparse attention indexing |
| `models/` | Encoder, decoder, language, multimodal models and output heads |
| `flash_attention/` | Educational FlashAttention v1-v4 implementations |
| `quantization.py` | Fake quantization and QAT wrappers |
| `spec_decode/` | N-Gram, EAGLE, Medusa, MTP, DFlash and DSpark decoding |
| `module_registry.py` | Public factories and implementation registries |

## Naming conventions

- Files and functions use descriptive `snake_case` names.
- Classes use `PascalCase`; established acronyms retain their conventional
  spelling, for example `ALiBiAttention`, `RMSNorm` and `SwiGLUFFN`.
- Public constructors use `build_*`; discovery helpers use `list_*`.
- Registry lookups that resolve a registered name to its entry use `get_*`
  (for example `get_activation`, `get_longrope_preset` and
  `get_version_module`); registration helpers use `register_*`.
- Import the canonical modules directly: `feed_forward`, `normalization`,
  `mamba2`, `transformer_block`, `gated_feed_forward`, `hybrid_layers`
  and `hyper_connection`. Model implementations live in `models/encoder`,
  `models/encoder_decoder`, `models/language`, `models/multimodal` and
  `models/heads`.
