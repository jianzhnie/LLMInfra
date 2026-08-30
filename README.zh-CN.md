# LLMInfra

[English](README.md) | **简体中文**

LLMInfra 是一个 PyTorch 工具库,以清晰、可读、可测试的纯 PyTorch
代码实现现代大模型基础设施的核心组件:注意力机制、FlashAttention、位置编码、
Transformer 层、MoE、推理系统(KV cache 分页/量化/分层卸载)、生成与采样、
推测解码、量化与优化器。

实现优先考虑可读性、显式的张量形状、数值正确性与可测试性,是参考与教学实现,
不用于替代生产环境的优化算子。

## 特性

- **注意力机制**(全部继承统一基类 `BaseAttention`,接口一致):
  - **MHA / MQA / GQA** - 多头、多查询、分组查询注意力(可选 QK-Norm、
    logit soft-capping(Gemma 2)、attention sink(GPT-OSS)、
    输出门控(Qwen3-Next))
  - **MLA (Multi-Head Latent Attention)** - 多头潜空间注意力
  - **FlashMLA** - MLA + 潜空间 KV 缓存的分页推理模拟
  - **SWA (Sliding Window Attention)** - 滑动窗口注意力(与 Mistral/transformers
    窗口约定一致:含自身共 `window_size` 个位置)
  - **Block Sparse / Compressed Sparse / Dynamic Sparse / MiniMax Sparse /
    Hierarchical Compressed** - 块稀疏、压缩 KV + 稀疏选择等稀疏注意力家族
  - **Ring Attention** - 分块在线 softmax 精确注意力(含分布式接口)
  - **ALiBi / Hybrid / Attention Residual** - ALiBi 偏置(BLOOM 斜率表)、
    线性/全量按层交错、残差注意力(Kimi K3 风格)
- **线性时间序列模块**(state 递推家族):
  - **Linear / Lightning Attention** - 线性注意力(分块因果扫描)与分块内 softmax 变体
  - **Gated DeltaNet** - 门控线性注意力 `S = (1-g)S + g·k⊗v`
  - **Kimi Delta Attention (KDA)** - 带误差修正项的 delta 规则线性注意力
  - **RetNet (Retention)** - 固定衰减保留机制(并行/递推双路径)
  - **GLA (Gated Linear Attention)** - 数据依赖 per-feature 门控 `S = S⊙g + kᵀv`
  - **RWKV** - RWKV-4 时混 + 通道混双块(递推/并行双路径)
  - **TTT-Linear** - 测试时训练层(mini-batch 内部梯度下降)
- **FlashAttention 教学实现**:FA1–FA4 四个版本的纯 PyTorch 在线 softmax 分块
  实现(含 forward/backward),用于理解各代算法的结构演进
- **位置编码与长上下文扩展**:可学习/正弦绝对位置、RoPE、YaRN、Dynamic NTK、
  位置插值、LongRoPE、Partial RoPE、ALiBi、T5 相对偏置、2D 位置、MRoPE
- **Transformer 层**:RMSNorm、LayerNorm、DeepNorm、LayerScale、SwiGLU 及门控 FFN
  (含 GPT-OSS 截断 SwiGLU)、可配置 Transformer Block(多种归一化风格)、
  Mamba2 状态空间层、混合层堆栈、超连接(mHC)
- **模型级组合**:CausalLM(含前缀 LM)、Encoder、Encoder-Decoder、多模态模型,
  分类/奖励/embedding 等任务头,以及 Attention/Positional 注册表工厂
- **生成与采样**:`generate`(naive 全量重算)/ `generate_with_cache`(KV cache
  增量)两个独立解码循环;top-k / top-p / min-p / 重复惩罚 logits processors;
  `CausalLMModel.generate` 支持 naive、KV cache(`use_cache=True`)与推测解码
- **MoE 模块**:Top-k 与 Expert-Choice 路由、共享专家(DeepSeek 风格)、LatentMoE、
  专家并行、负载均衡损失与 router z-loss
- **推理系统**:分页 KV cache、量化 KV cache(KIVI 式 per-channel K / per-token V,
  INT8/INT4)、磁盘/分层 KV 卸载、PagedAttention、块稀疏索引
- **推测解码**:标准草稿-验证、N-Gram、EAGLE 1–3、Medusa、MTP、
  DFlash(块扩散并行起草)与 DSpark(动态块长调度)
- **量化**:便携式 fake INT4/INT8/FP8(E4M3)/MXFP4/NVFP4 量化与 QAT 包装器
- **优化器**:Muon(Newton-Schulz 正交化 + AdamW fallback)与 MuonClip
  (Kimi K2 式 QK-clip)
- **掩码约定**:`1(True)= 保留、0(False)= 屏蔽`,支持广播的 padding mask
  与完整 mask;可返回注意力权重用于可视化分析

## 安装

### 依赖

- Python >= 3.10
- PyTorch

### 从源码安装

```bash
git clone https://github.com/jianzhnie/LLMInfra.git
cd LLMInfra
python -m pip install -e .
```

## 快速开始

```python
import torch
from llminfra import MultiHeadAttention

attention = MultiHeadAttention(hidden_size=512, num_heads=8)
hidden_state = torch.randn(2, 128, 512)

output = attention(hidden_state)
print(output.shape)  # torch.Size([2, 128, 512])

output, attn_weights = attention(hidden_state, return_attention_weights=True)
print(attn_weights.shape)  # torch.Size([2, 8, 128, 128])
```

通过公共注册表按名字构建模块:

```python
from llminfra import build_attention, list_attentions

attention = build_attention("gqa", hidden_size=512, num_heads=8, num_kv_groups=2)
print(list_attentions())
```

教学版 FlashAttention 接口:

```python
import torch
from llminfra import flash_attention

q = torch.randn(2, 8, 128, 64, requires_grad=True)
k = torch.randn(2, 8, 128, 64, requires_grad=True)
v = torch.randn(2, 8, 128, 64, requires_grad=True)

out = flash_attention(q, k, v, version="fa2", causal=True)
out.sum().backward()  # 梯度走 fa2 的分块 backward
```

从小模型生成文本(naive 与 KV cache 两条路径,输出逐 token 相同):

```python
import torch
from llminfra import CausalLMModel

model = CausalLMModel(
    vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8,
    intermediate_size=512,
)
model.eval()  # 关掉 dropout,两条路径才有确定性可比
prompt = torch.randint(0, 1000, (1, 8))

naive = model.generate(prompt, max_new_tokens=16)                    # 每步全量重算
cached = model.generate(prompt, max_new_tokens=16, use_cache=True)   # prefill + 增量
assert torch.equal(naive.sequences, cached.sequences)
```

## 模块详解

### 1. Multi-Head Attention (MHA)

多头注意力机制,出自论文 *Attention is All You Need*。每个 Query、Key、Value 各有 `num_heads` 组独立的投影矩阵。

```python
from llminfra import MultiHeadAttention

attention = MultiHeadAttention(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    bias=True,
)
```

参数说明:
- `hidden_size`:输入和输出特征维度
- `num_heads`:注意力头数量,必须能整除 `hidden_size`
- `dropout`:注意力权重的 dropout 概率,默认 0.1
- `bias`:线性投影是否使用偏置,默认 True
- `qk_norm`:是否对 Q/K 做逐头 RMSNorm(Qwen3 风格),默认 False
- `logit_softcap`:Gemma 2 风格 logit 软截断,默认 None(关闭)
- `attention_sink`:是否加可学习的 sink logit(GPT-OSS 风格),默认 False
- `output_gate`:是否加 sigmoid 输出门控(Qwen3-Next 风格),默认 False

### 2. Multi-Query Attention (MQA)

多查询注意力机制,出自论文 *Fast Transformer Decoding: One Write-Head is All You Need*。所有 Query 头共享同一组 Key 和 Value 投影,显著减少显存占用和推理计算量。

```python
from llminfra import MultiQueryAttention

attention = MultiQueryAttention(
    hidden_size=512,
    num_heads=8,
    dropout=0.1,
    bias=True,
)
```

### 3. Group Query Attention (GQA)

分组查询注意力机制,出自论文 *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。MHA 与 MQA 的折中方案,将 Query 头分组,每组共享一组 Key/Value 头。

```python
from llminfra import GroupedQueryAttention

attention = GroupedQueryAttention(
    hidden_size=512,
    num_heads=8,
    num_kv_groups=2,  # G=2 时每组 4 头共享一组 KV
    dropout=0.1,
    bias=True,
)
```

特殊参数:
- `num_kv_groups`:KV 分组数量,必须能整除 `num_heads`
  - 当 `num_kv_groups == num_heads`:等价于 MHA
  - 当 `num_kv_groups == 1`:等价于 MQA
  - 当 `1 < num_kv_groups < num_heads`:标准 GQA
- `output_gate`:Qwen3-Next 风格 sigmoid 输出门控,默认 False

### 4. Multi-Head Latent Attention (MLA)

多头潜空间注意力,在注意力计算前先将特征投影到潜空间,再映射回原空间进行注意力计算。

```python
from llminfra import MultiHeadLatentAttention

attention = MultiHeadLatentAttention(
    hidden_size=512,
    num_heads=8,
    q_latent_size=256,    # Query 潜空间维度
    kv_latent_size=256,   # Key/Value 潜空间维度
    dropout=0.0,
    bias=True,
)
```

### 5. FlashAttention(教学实现)

`llminfra.flash_attention` 子包以纯 PyTorch 实现 FlashAttention v1–v4 的核心算法结构(在线 softmax、分块循环、LSE 重算梯度),提供函数式接口 `flash_attention`(类似 `F.scaled_dot_product_attention`)与 `FlashAttention` 模块包装,均支持 autograd:

```python
import torch
from llminfra import FlashAttention, flash_attention

q = torch.randn(2, 8, 128, 64, requires_grad=True)
k = torch.randn(2, 8, 128, 64, requires_grad=True)
v = torch.randn(2, 8, 128, 64, requires_grad=True)

# 函数式调用,version 选择 FA 版本(fa1/fa2/fa3/fa4)
out = flash_attention(q, k, v, version="fa2", causal=True)
out.sum().backward()

# 或者像普通 nn.Module 一样使用
attn = FlashAttention(version="fa3", causal=True)
out = attn(q, k, v)
```

四个版本各自暴露底层的 `forward` / `backward` 以及可微分的
`flash_attention_v1`–`flash_attention_v4` 函数,差异体现在循环结构、工作划分与调度方式上,分别对应各代论文的算法改进点。

### 6. Sliding Window Attention (SWA)

```python
from llminfra import SlidingWindowAttention

attention = SlidingWindowAttention(
    hidden_size=512,
    num_heads=8,
    window_size=128,  # 每个 query 可见的 key 数量(含自身,Mistral 约定)
    num_kv_groups=2,
)
```

### 7. Block Sparse Attention

```python
from llminfra import BlockSparseAttention

attention = BlockSparseAttention(
    hidden_size=512,
    num_heads=8,
    block_size=16,
    top_k=4,
)
```

### 8. 线性时间序列模块(Linear / GLA / RetNet / GDN / KDA / RWKV / TTT)

统一的 state 递推语义:padding 位置既不写入也不衰减 state;多数模块提供
chunked 并行与逐步递推两条数值一致的路径。

```python
from llminfra import (
    GatedDeltaNet,
    GatedLinearAttention,
    KimiDeltaAttention,
    LinearAttention,
    Retention,
    RWKVLayer,
    TTTLayer,
)

attention = LinearAttention(hidden_size=512, num_heads=8, feature_dim=64, kernel="elu")
retention = Retention(hidden_size=512, num_heads=8)          # RetNet,固定衰减
gla = GatedLinearAttention(hidden_size=512, num_heads=8)     # 数据依赖门控
gdn = GatedDeltaNet(hidden_size=512, num_heads=8)            # 插值式门控
kda = KimiDeltaAttention(hidden_size=512, num_heads=8)       # delta 规则修正
rwkv = RWKVLayer(hidden_size=512)                            # 时混 + 通道混
ttt = TTTLayer(hidden_size=512, chunk_size=16)               # 测试时训练
```

### 9. PagedAttention 与量化 KV cache

```python
from llminfra import PagedAttentionCache, QuantizedKVCache, paged_attention

cache = PagedAttentionCache(
    num_blocks=128,
    block_size=16,
    num_heads=8,
    head_dim=64,
)

# KIVI 式量化缓存:per-channel K / per-token V,最近 token 保留全精度
qcache = QuantizedKVCache(num_heads=8, head_dim=64, bits=4, residual_length=128)
```

### 10. 位置编码

```python
import torch
from llminfra import RotaryPositionEmbedding, ALiBiBias

rope = RotaryPositionEmbedding(dim=64, max_seq_len=4096)
x = torch.randn(2, 8, 128, 64)
y = rope(x)

alibi = ALiBiBias(num_heads=8, max_seq_len=4096)
bias = alibi(x)  # shape (1, 8, 128, 128)
```

YaRN、Dynamic NTK、LongRoPE 等长上下文变体分别通过
`YaRNScaledRotaryEmbedding`、`DynamicNTKRotaryEmbedding`、`LongRoPEScaledRotaryEmbedding` 使用,也可经 `build_positional_encoding` 工厂按名字构建。

### 11. MoE

```python
import torch
from llminfra import MixtureOfExperts, DeepSeekMoE

moe = MixtureOfExperts(
    hidden_size=256,
    num_experts=8,
    intermediate_size=512,
    top_k=2,
)
x = torch.randn(2, 32, 256)
y = moe(x)

ds_moe = DeepSeekMoE(
    hidden_size=256,
    num_routed_experts=8,
    num_shared_experts=1,
    intermediate_size=512,
    top_k=2,
)
y = ds_moe(x)
```

### 12. Hybrid Attention 与 Transformer Block

```python
import torch
from llminfra import HybridAttention, TransformerBlock, SwiGLUFFN

hybrid = HybridAttention(
    hidden_size=256,
    num_heads=8,
    linear_interval=3,
    full_interval=1,
)
block = TransformerBlock(
    hidden_size=256,
    num_heads=8,
    intermediate_size=512,
    attention=hybrid,
    ffn=SwiGLUFFN(256, 512),
)
x = torch.randn(2, 32, 256)
y = block(x, layer_index=3)
```

### 13. Gated DeltaNet 与 CausalLMModel

```python
import torch
from llminfra import GatedDeltaNet, CausalLMModel

gdn = GatedDeltaNet(hidden_size=256, num_heads=8, feature_dim=64)
x = torch.randn(2, 32, 256)
y = gdn(x)

model = CausalLMModel(
    vocab_size=1000,
    hidden_size=256,
    num_layers=4,
    num_heads=8,
    intermediate_size=512,
    attention_name="hybrid",
    use_moe=True,
)
input_ids = torch.randint(0, 1000, (2, 32))
logits = model(input_ids)
```

### 14. 生成与采样(`llminfra/generation.py`)

两个独立的教学解码循环,可驱动任何满足契约的模型:

```python
import torch
from llminfra import generate, generate_with_cache, TopKLogitsProcessor, LogitsProcessorList

# naive:每步全量重算
out = generate(logits_fn, prompt_ids, max_new_tokens=32, temperature=0.8,
               processors=LogitsProcessorList([TopKLogitsProcessor(50)]))

# KV cache:prefill 一次,之后每步只喂新 token
def step_fn(tokens, past):
    logits, new_past = model._forward_with_cache(tokens, past)
    return logits[:, -1], new_past

out = generate_with_cache(step_fn, prompt_ids, max_new_tokens=32)
```

`CausalLMModel.generate` 已内置两条路径(`use_cache=True` 切换)与推测解码
(`draft_model=...`);logits processors 组合顺序与 transformers 一致
(repetition → top-k → top-p → min-p)。真实模型示例见
`examples/generate_with_opt.py`(驱动本地 OPT-125m,两路径输出逐 token 相同)。

### 15. 优化器(`llminfra/optimizers.py`)

```python
from llminfra import CausalLMModel, Muon, MuonClip

# 隐藏层矩阵参数走 Muon(Newton-Schulz 正交化),embedding/1D 参数自动走 AdamW
model = CausalLMModel(
    vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8,
    intermediate_size=512,
)
optimizer = Muon.from_named_parameters(model.named_parameters(), lr=2e-3)

# MuonClip:当某注意力头的 logit 超过阈值时按 sqrt(τ/max) 缩放 Q/K 权重
# (教学版要求 num_kv_heads == num_heads,即 MHA;QK-clip 对 GQA/MQA 不适用)
mha_model = CausalLMModel(
    vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8,
    intermediate_size=512, attention_name="mha",
)
optimizer = MuonClip.from_named_parameters(mha_model.named_parameters(), lr=2e-3)
attn = mha_model.blocks[0].attention
optimizer.register_qk_params("layer0", attn.q_proj.weight, attn.k_proj.weight, num_heads=8)
optimizer.qk_clip(max_logits={"layer0": max_logit_per_head}, threshold=100.0)
```

## 使用 Attention Mask

掩码遵循 `1(True)= 保留,0(False)= 屏蔽` 约定,并对注意力分数广播,因此 padding mask 和完整 mask 都支持:

```python
batch_size = 2
seq_len = 128

# padding mask:屏蔽每个样本末尾的若干 key 位置
attention_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
attention_mask[0, 0, 0, -10:] = False  # 屏蔽第一个样本的最后 10 个 key

output = attention(hidden_state, attention_mask=attention_mask)
```

注意:
- 掩码必须是布尔(或整型 1/0);HuggingFace 风格的 0/-inf 加性 float 掩码会被显式拒绝。
- 若某一行所有 key 均被屏蔽,该行的注意力权重和输出定义为 0(而非 NaN)。

## 项目结构

| 路径 | 职责 |
|---|---|
| `llminfra/attention/` | 稠密、稀疏、线性、潜空间与混合注意力 |
| `llminfra/flash_attention/` | 教学版 FlashAttention v1–v4 |
| `llminfra/positional/` | 绝对、相对、旋转与长上下文位置编码 |
| `llminfra/layers/` | FFN、归一化、SSM(Mamba2)与 Transformer 层 |
| `llminfra/models/` | 语言、编码器、编码器-解码器与多模态模型 |
| `llminfra/moe/` | 路由器、专家与混合专家模块 |
| `llminfra/inference/` | KV 缓存(分页/量化/分层卸载)与稀疏索引 |
| `llminfra/spec_decode/` | N-Gram、EAGLE、Medusa、MTP、DFlash、DSpark |
| `llminfra/generation.py` | Logits processors 与贪心/采样解码循环 |
| `llminfra/optimizers.py` | Muon / MuonClip 优化器 |
| `llminfra/quantization.py` | Fake 量化(INT4/INT8/FP8/MXFP4/NVFP4)与 QAT 工具 |
| `examples/` | 性能基准(`benchmark/`)、训练示例与真实模型生成示例 |
| `tests/` | 按源码目录组织的测试 |

公共 API 均从 `llminfra` 顶层导出;仅在扩展或研究内部实现时才直接导入实现模块。

## 开发与测试

```bash
# 运行测试套件(CPU 全覆盖;CUDA 专属用例在无 GPU 时自动跳过)
python -m pytest -q

# 格式化与 lint(配置见 pyproject.toml)
ruff format --check .
ruff check .

# 类型检查(strict 模式零错误)
mypy llminfra --strict

# 全部 pre-commit 钩子
pre-commit run --all-files
```

## 文档

- [English README](README.md)
- [模块 ↔ 论文 ↔ transformers 对照索引](llminfra_guide.md)
- [包内结构与命名约定](llminfra/README.md)
- [Transformer Review 2026](docs/transformers_review_2026.md)

## 参考文献

1. Vaswani, A., et al. "Attention is All You Need." NeurIPS 2017.
2. Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." 2019.
3. Chen, W., et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." 2023.
4. Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." 2022.
5. Dao, T. "FlashAttention-2: Better Attention with Better Parallelism and Work Partitioning." 2023.
6. Shah, J., et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024.
7. Sun, Y., et al. "Retentive Network: A Successor to Transformer for Large Language Models." 2023.
8. Peng, B., et al. "RWKV: Reinventing RNNs for the Transformer Era." 2023.
9. Sun, Y., et al. "Learning to (Learn at Test Time): RNNs with Expressive Hidden States." 2024.
10. Jordan, K., et al. "Muon: An Optimizer for Hidden Layers" / "Muon is Scalable for LLM Training." 2024/2025.
11. Liu, Z., et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." 2024.
12. Chen, J., et al. "DFlash: Block Diffusion for Flash Speculative Decoding." 2026.

## 许可证

[Apache License 2.0](LICENSE)
