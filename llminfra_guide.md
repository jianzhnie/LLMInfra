# LLMInfra 模块 ↔ 论文 ↔ transformers 实现 对照索引

> `Paper` 链接指向 arXiv;`transformers` 链接指向 GitHub `huggingface/transformers` main 分支 `src/transformers/` 下的对应文件(文件级,不含行号)。

### Attention — 经典(`llminfra/attention/`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| MultiHeadAttention | GPT-2、LLaMA;`logit_softcap` ↔ Gemma 2/3;`attention_sink` ↔ GPT-OSS / StreamingLLM;`output_gate` ↔ Qwen3-Next 门控注意力 | [Paper: Vaswani 2017](https://arxiv.org/abs/1706.03762)、[Gemma 2(softcap)](https://arxiv.org/abs/2408.00118)、[StreamingLLM(sink)](https://arxiv.org/abs/2309.17453)、[GPT-OSS(sink)](https://arxiv.org/abs/2508.10925)、[Gated Attention(output_gate)](https://arxiv.org/abs/2505.06708)、[transformers: llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)、[gemma2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py)、[gpt_oss](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py)、[qwen3_next](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) |
| MultiQueryAttention | Falcon、PaLM | [Paper](https://arxiv.org/abs/1911.02150)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py) |
| GroupedQueryAttention | LLaMA-2/3、Qwen3;`output_gate` ↔ Qwen3-Next 门控注意力 | [Paper](https://arxiv.org/abs/2305.13245)、[Paper: Gated Attention(output_gate)](https://arxiv.org/abs/2505.06708)、[transformers: llama](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)、[qwen3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)、[qwen3_next](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) |
| MultiHeadLatentAttention | DeepSeek-V2/V3、Kimi K2 | [Paper](https://arxiv.org/abs/2412.19437)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) |
| SlidingWindowAttention | Mistral | [Paper](https://arxiv.org/abs/2310.06825)、[transformers: mistral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py)、[masking_utils](https://github.com/huggingface/transformers/blob/main/src/transformers/masking_utils.py) |
| ALiBiAttention | BLOOM | [Paper](https://arxiv.org/abs/2108.12409)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py) |
| HybridAttention | Qwen3-Next 式混合 | [Model](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)、[transformers: configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py)、[modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) |
| AttentionResidual | Kimi K3 AttnRes | Kimi K3(官方未公开论文,无 transformers 对应实现) |

### Attention — 稀疏/系统

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| BlockSparseAttention | MiniMax M3 | [Paper: NSA](https://arxiv.org/abs/2502.11089)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py) |
| CompressedSparseAttention | DeepSeek-V4 CSA | [Paper: NSA](https://arxiv.org/abs/2502.11089)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py) |
| QueryKeyBlockIndexer / DynamicSparseAttention | DeepSeek-V3.2 DSA / V4 Lightning Indexer | [Paper: DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)、[transformers: deepseek_v32](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py)、[deepseek_v4](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py) |
| MiniMaxSparseAttention | MiniMax M3 Lightning Indexer | [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py) |
| HierarchicalCompressedAttention | DeepSeek-V4 HCA | [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py) |
| FlashMLA | DeepSeek FlashMLA kernel | [Repo](https://github.com/deepseek-ai/FlashMLA)(本模块为接口模拟,docstring 已声明) |
| RingAttention | 长上下文训练系统 | [Paper](https://arxiv.org/abs/2310.01889) |

### Attention — 线性/SSM

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| LinearAttention | Linear Transformers | [Paper](https://arxiv.org/abs/2006.16236);transformers 无对应实现 |
| LightningAttention | TransnormerLLM / MiniMax-01 | [Paper: TransNormerLLM](https://arxiv.org/abs/2307.14995)、[Paper: MiniMax-01](https://arxiv.org/abs/2501.08313);transformers 无对应实现 |
| GatedDeltaNet | Qwen3-Next | [Paper](https://arxiv.org/abs/2412.06464)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) |
| Retention (RetNet) | RetNet | [Paper](https://arxiv.org/abs/2307.08621);transformers 无对应实现 |
| GatedLinearAttention (GLA) | GLA Transformer | [Paper](https://arxiv.org/abs/2312.06635);transformers 无对应实现,[fla 库](https://github.com/fla-org/flash-linear-attention)为事实参考 |
| RWKVLayer | RWKV-4(RWKV-LM) | [Paper](https://arxiv.org/abs/2305.13048)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rwkv/modeling_rwkv.py) |
| TTTLayer | TTT-Linear(测试时训练) | [Paper](https://arxiv.org/abs/2407.04620);transformers 无对应实现 |
| KimiDeltaAttention (KDA) | Kimi Linear | [Paper](https://arxiv.org/abs/2510.26692)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)(`torch_recurrent_gated_delta_rule`) |

### Positional(`llminfra/positional/`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| Sinusoidal / Learned / NoPE | Transformer / GPT-2、BERT | [Paper](https://arxiv.org/abs/1706.03762)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) |
| RotaryPositionEmbedding | LLaMA、Qwen 全系 | [Paper: RoFormer](https://arxiv.org/abs/2104.09864)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) |
| YaRNScaledRotaryEmbedding | Mistral/Qwen 长上下文变体 | [Paper](https://arxiv.org/abs/2309.00071)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) |
| DynamicNTKRotaryEmbedding | CodeLlama、Qwen 长上下文 | emozilla 2023(社区方案,无正式论文)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) |
| PositionInterpolation | LLaMA 上下文扩展 | [Paper](https://arxiv.org/abs/2306.15595)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) |
| LongRoPEScaledRotaryEmbedding | Phi-3 长上下文 | [Paper](https://arxiv.org/abs/2402.13753)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) |
| PartialRotaryPositionEmbedding | GPT-NeoX 系 | [transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py)(`partial_rotary_factor` 语义) |
| ALiBiBias | BLOOM | [Paper](https://arxiv.org/abs/2108.12409)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py) |
| T5RelativePositionBias | T5 | [Paper](https://arxiv.org/abs/1910.10683)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) |
| TwoDimensionalPositionEmbedding | 分块数据/表格场景 | transformers 无对应实现 |
| MultiModalRoPE (MRoPE) | Qwen2-VL | [Paper](https://arxiv.org/abs/2409.12191)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py) |

### MoE(`llminfra/moe/`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| ExpertFFN | Mixtral、DeepSeek-V3 | [transformers: mixtral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)、[deepseek_v3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) |
| TopKRouter(softmax / sigmoid+aux-free) | Mixtral / DeepSeek-V3 | [Paper: Switch](https://arxiv.org/abs/2101.03961)、[Paper: DeepSeek-V3](https://arxiv.org/abs/2412.19437)、[transformers: mixtral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)、[deepseek_v3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) |
| MixtureOfExperts(分派/合并) | Mixtral | [Paper](https://arxiv.org/abs/2401.04088)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py) |
| DeepSeekMoE(共享专家) | DeepSeek-V2/V3 | [Paper](https://arxiv.org/abs/2412.19437)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py) |
| ExpertChoiceRouter | Google Expert-Choice | [Paper](https://arxiv.org/abs/2202.09368); transformers 无对应实现 |
| ExpertParallelMoE | GShard/Switch 专家并行 | [Paper: GShard](https://arxiv.org/abs/2006.16668)、[Paper: Switch](https://arxiv.org/abs/2101.03961) |
| LatentMoE | Nemotron-3 Ultra、MAI-Base-1 | [Paper](https://arxiv.org/abs/2601.18089);transformers 无对应实现 |
| load_balance_loss / router_z_loss | Switch / ST-MoE | [Paper: Switch](https://arxiv.org/abs/2101.03961)、[Paper: ST-MoE](https://arxiv.org/abs/2202.08906) |

### Speculative Decoding(`llminfra/spec_decode/`)

| 模块 | 来源 | 参考 |
| --- | --- | --- |
| SpeculativeDecoder | Leviathan 2023 | [Paper](https://arxiv.org/abs/2211.17192)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)(`_speculative_sampling`) |
| NGramSpeculator | Prompt lookup | [Repo](https://github.com/apoorvumang/prompt-lookup-decoding)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/candidate_generator.py) |
| EagleSpeculator / EAGLE 1–3 | EAGLE 系列 | [Paper: EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840);transformers 无对应实现 |
| MedusaHead / medusa_loss | Medusa | [Paper](https://arxiv.org/abs/2401.10774); transformers 无对应实现 |
| MTPDecoder / MultiTokenPredictionHead / mtp_loss | DeepSeek-V3 MTP | [Paper](https://arxiv.org/abs/2412.19437) §2.2 |
| BlockDiffusionDrafter / DFlashDecoder / dflash_loss | DFlash 块扩散起草 | [Paper](https://arxiv.org/abs/2602.06036) §4;transformers 无对应实现 |
| DSparkDecoder / DSparkScheduler | DSpark 式动态块长调度 | 无公开论文; docstring 自述为可移植调度接口 |

### Layers(`llminfra/layers/`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| RMSNorm / LayerNorm / DeepNorm / LayerScale | LLaMA / 标准 LN / GLM-130B / CaiT | [Paper: RMSNorm](https://arxiv.org/abs/1910.07467)、[DeepNorm](https://arxiv.org/abs/2203.00555)、[CaiT](https://arxiv.org/abs/2103.17239)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) |
| SwiGLUFFN / GeGLU / ReGLU | LLaMA / GLU 变体 | [Paper: GLU Variants](https://arxiv.org/abs/2002.05202)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) |
| ClampedSwiGLUFFN | GPT-OSS | [Model Card](https://arxiv.org/abs/2508.10925)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py) |
| Mamba2Layer | Mamba-2 | [Paper](https://arxiv.org/abs/2405.21060)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py) |
| ManifoldConstrainedHyperConnection | mHC | [Paper: mHC](https://arxiv.org/abs/2512.24880)、[Paper: Hyper-Connections](https://arxiv.org/abs/2409.19606) |
| TransformerBlock | LLaMA/GPT-J/GLM-130B | [Paper: LLaMA](https://arxiv.org/abs/2302.13971)(pre-norm)、[GLM-130B](https://arxiv.org/abs/2210.02414)(DeepNorm)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) |
| HybridSSMBlock / HybridLayerStack | Zamba2、Qwen3-Next | [transformers: zamba2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/zamba2/modeling_zamba2.py)、[qwen3_next](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py) |

### Flash Attention(`llminfra/flash_attention/`)

transformers 无手写 FA 内核,权威参照为论文公式、[官方 CUDA 实现](https://github.com/Dao-AILab/flash-attention)与 `F.scaled_dot_product_attention`:

| 模块 | 核心公式对照 | 参照 |
| --- | --- | --- |
| `common/ops.py`(merge/LSE/梯度重算) | FA1 Alg.1 在线 softmax;FA2 Alg.2 延迟归一化 + LSE 重算 + rowsum 技巧 | [Paper: FA1](https://arxiv.org/abs/2205.14135)、[FA2](https://arxiv.org/abs/2307.08691)、[Code](https://github.com/Dao-AILab/flash-attention) |
| `flash_attention_v1/v2` | KV 外循环 / query-tile ownership + 延迟归一化 | [Paper: FA1](https://arxiv.org/abs/2205.14135)、[FA2](https://arxiv.org/abs/2307.08691)、[Code](https://github.com/Dao-AILab/flash-attention) |
| `flash_attention_v3` | ping-pong 双缓冲、FP8 per-tile 量化模拟 | [Paper: FA3](https://arxiv.org/abs/2407.08608)、[Code](https://github.com/Dao-AILab/flash-attention) |
| `flash_attention_v4` | wave 调度 + 三角色;selective rescale | [Blog: FA4](https://tridao.me/blog/2026/flash4/)、[Code](https://github.com/Dao-AILab/flash-attention) |

### Inference & Quantization

| 模块 | 对应 | 参考 |
| --- | --- | --- |
| PagedKVBlockAllocator / PagedAttentionCache / paged_attention | vLLM PagedAttention | [Paper](https://arxiv.org/abs/2309.06180)、[Code: vLLM](https://github.com/vllm-project/vllm);数值对照 SDPA |
| OnDiskKVStore / TieredKVCache | vLLM swap 机制近似 | [Paper](https://arxiv.org/abs/2309.06180) §4.4 |
| BlockSparseIndexer(inference) | NSA 式块选择 | [Paper](https://arxiv.org/abs/2502.11089) |
| FakeQuantizer / QuantizationConfig / QATWrapper | PyTorch fake-quant/QAT 标准;`mxfp4` ↔ GPT-OSS;`nvfp4` ↔ Blackwell 系 NVFP4 | [Code: torch.ao.quantization](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fake_quantize.py)、[Paper: STE](https://arxiv.org/abs/1308.3432)、[GPT-OSS(MXFP4)](https://arxiv.org/abs/2508.10925)、OCP Microscaling(MX)规范、原生 `torch.float8_e4m3fn`(OCP FP8) |

### Optimizers(`llminfra/optimizers.py`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| Muon(Newton-Schulz 正交化 + AdamW fallback) | Kimi K2、GLM-4.5 等 | [Paper](https://arxiv.org/abs/2502.16982)、[Code: KellerJordan/Muon](https://github.com/KellerJordan/Muon) |
| MuonClip(QK-clip) | Kimi K2 | [Paper: Kimi K2](https://arxiv.org/abs/2507.20534) §2 |

### Models(`llminfra/models/`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| EncoderOnlyModel / EncoderBlock / pool_hidden_state | BERT 式双向编码器 | [Paper](https://arxiv.org/abs/1810.04805)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) |
| DecoderBlock / CausalLMModel / PrefixLMModel(`generate` 可选 paged cache / speculative decoding) | GPT 式解码器 | [Paper: GPT-2(Radford 2019, 无 arXiv)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) |
| EncoderDecoderModel / CrossAttention | T5 / BART 式编码器-解码器 | [Paper: T5](https://arxiv.org/abs/1910.10683)、[BART](https://arxiv.org/abs/1910.13461)、[transformers: t5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)、[bart](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py) |
| MultimodalCausalLM / VisionEncoderAdapter / CrossAttentionFuser / build_multimodal_position_ids | LLaVA 式多模态(+ MRoPE 位置 id) | [Paper: LLaVA](https://arxiv.org/abs/2304.08485)、[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py) |
| RewardModelHead / SequenceClassificationHead / TokenClassificationHead / EmbeddingHead | RLHF 奖励模型 / 分类与嵌入头 | [Paper: InstructGPT](https://arxiv.org/abs/2203.02155) |

### 工程入口(无数值语义)

- `llminfra/module_registry.py`:`build_attention` / `list_attentions` / `build_positional_encoding` 工厂分发。
- `llminfra/flash_attention/fp8_support.py`:FA3 FP8 per-tile 量化的能力检测与守卫(见 FA3 行)。
- `llminfra/attention/base_attention.py`:所有注意力模块共享的 split/combine head、mask、softmax(含 softcap/sink)基建。
