# 任务:对照 transformers 实现审查并整理 LLMInfra 关键模块

## 背景

- 本项目:`/Users/robin/work_dir/LLMInfra/llminfra`(教学向 PyTorch 实现,测试与文档齐备)
- 参考实现:HuggingFace transformers 本地副本 `/Users/robin/work_dir/transformers`(`src/transformers/models/<model>/modeling_*.py`),以及各方法的原始论文

## 目标

逐个核对 LLMInfra 中关键模块的实现与 transformers 中对应模型实现、原始论文公式是否一致,产出一份"模块 ↔ 论文 ↔ 模型"的对照审查报告。

## 执行要求

1. **逐模块对照**:每个模块先在 transformers 本地副本中定位对应实现(给出具体文件:行号),再与 LLMInfra 实现逐项对比:数学公式、缩放因子、mask 语义、张量布局、边界条件。
2. **差异分级**:发现差异时区分——(a) 实现错误(bug);(b) 有意的教学简化(已在 docstring 声明);(c) 等价的不同写法。只有 (a) 需要修复。
3. **实证验证**:对任何判定的 bug,写一个最小复现脚本/测试证明它确实导致错误输出,再修复并补回归测试。不要仅凭阅读下结论。
4. **不破坏现状**:修复后 `python -m pytest tests -q` 必须全部通过;不改变公开 API;遵循仓库现有代码风格。

## 交付物

1. 更新后的对照表(见下方各表,补全 Model 列与论文/实现链接)
2. 差异清单:每个模块一段,说明与 transformers/论文的一致点与差异点
3. 确认的 bug 修复(若有),附复现证据与回归测试

## 模块清单与已知模型对应

### Attention(`llminfra/attention/`)

| 模块 | 对应模型 | 参考(论文/transformers 实现) |
| --- | --- | --- |
| MHA | GPT-2、LLaMA-1 | Vaswani 2017;`modeling_llama.py` |
| MQA | Falcon、PaLM | Shazeer 2019;`modeling_falcon.py` |
| GQA | LLaMA-2/3、Qwen2/3、Mistral | Ainslie 2023;`modeling_qwen3.py` |
| MLA | DeepSeek-V2/V3、Kimi K2 | DeepSeek-V3 tech report |
| SlidingWindowAttention | Mistral | `modeling_mistral.py` |
| BlockSparseAttention | BigBird 式块稀疏 | BigBird 2020 |
| CompressedSparseAttention | NSA(DeepSeek)压缩注意力 | NSA 2025 |
| DynamicSparseAttention / MiniMaxSparseAttention | MiniMax-01 | MiniMax-01 2025 |
| LinearAttention | Linear Transformers | Katharopoulos 2020 |
| LightningAttention | TransnormerLLM | OpenNLPLab 2023 |
| GatedDeltaNet | Qwen3-Next | Yang 2025(Qwen3-Next) |
| KimiDeltaAttention (KDA) | Kimi Linear | Moonshot 2025 |
| RingAttention | 长上下文训练系统 | Liu 2023 |
| ALiBiAttention | BLOOM | Press 2022;`modeling_bloom.py` |
| HybridAttention | Zamba / Qwen3-Next 式混合 | — |
| FlashMLA | DeepSeek FlashMLA 推理内核 | DeepSeek FlashMLA repo |

### Positional Encoding(`llminfra/positional/`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| Sinusoidal / Learned Absolute | Transformer / GPT-2、BERT | Vaswani 2017 |
| RoPE | LLaMA、Qwen 全系 | Su 2021(RoFormer);`modeling_llama.py` |
| YaRN | Mistral/Qwen 长上下文变体 | Peng 2023 |
| DynamicNTK | CodeLlama、Qwen 长上下文 | emozilla 2023 |
| PositionInterpolation | LLaMA 上下文扩展 | Chen 2023 |
| LongRoPE | Phi-3 长上下文 | Ding 2024 |
| ALiBiBias | BLOOM | Press 2022 |
| T5RelativePositionBias | T5 | Raffel 2020;`modeling_t5.py` |
| TwoDimensionalPositionEmbedding | 分块数据/表格类场景 | — |
| MultiModalRoPE (MRoPE) | Qwen2-VL | `modeling_qwen2_vl.py` |

### MoE(`llminfra/moe/`)

| 模块 | 对应模型 | 参考 |
| --- | --- | --- |
| TopKRouter + MixtureOfExperts | Switch Transformer、Mixtral | Fedus 2022;`modeling_mixtral.py` |
| ExpertChoiceRouter | Google Expert-Choice | Zhou 2022 |
| DeepSeekMoE(共享专家+细粒度+aux-free 均衡) | DeepSeek-V2/V3 | DeepSeek-V3 tech report |
| LatentMoE | 潜空间 MoE 变体 | — |
| ExpertParallelMoE | GShard/DeepSpeed 专家并行 | Lepikhin 2020 |

### Speculative Decoding(`llminfra/spec_decode/`)

| 模块 | 来源 | 参考 |
| --- | --- | --- |
| NGramSpeculator | 提示查找式草稿 | — |
| EagleSpeculator / EAGLE 1–3 | EAGLE 系列 | Li 2024/2025 |
| MedusaHead | Medusa | Cai 2024 |
| MTPDecoder / MultiTokenPredictionHead | DeepSeek-V3 MTP | DeepSeek-V3 tech report |
| DFlashDecoder / BlockDiffusionDrafter | DFlash(块扩散起草) | arXiv:2602.06036 |
| DSparkDecoder / DSparkScheduler | DSpark(动态块长调度) | — |

### 其他核心模块

- `llminfra/flash_attention/`:FA1–FA4 分块实现 ↔ Dao 2019–2024 系列论文,对照在线 softmax 合并与 LSE 重算梯度的公式
- `llminfra/layers/`:RMSNorm(LLaMA)、SwiGLU(PaLM)、Mamba2(Mamba-2 论文)、HyperConnection(mHC 论文)
- `llminfra/inference/`:PagedAttention(vLLM,Kwon 2023)、KV 分层卸载
- `llminfra/quantization.py`:fake quant / QAT 标准做法

## 验收标准

- 对照表中每个模块的 Model 列与参考列都已填实(附 transformers 本地文件:行号或论文编号)
- 每个确认修复的 bug 有复现脚本证据 + 回归测试,全套测试通过
- 教学简化与真实错误被明确区分,无凭阅读臆断的结论
