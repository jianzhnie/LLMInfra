"""Educational Lightning Attention implementation.

Lightning Attention, used by MiniMax-01, combines chunked linear attention
with intra-block softmax attention. This module keeps the chunkwise structure
readable; production implementations use custom tiling and kernel fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class LightningAttention(BaseAttention):
    """Chunked linear attention with intra-block softmax attention.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        feature_dim: Feature dimension of the linear kernel map.
        block_size: Number of tokens per intra-block softmax chunk.
        kernel: Feature map, one of ``"elu"``, ``"relu"`` or ``"linear"``.
        causal: Whether keys after the current query are masked out. The
            inter-block linear state is inherently causal (it only accumulates
            past blocks); this flag controls the intra-block softmax mask.
        dropout: Dropout applied to the output.
        bias: Whether linear projections use biases.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int | None = None,
        block_size: int = 64,
        kernel: str = "elu",
        causal: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if kernel not in {"elu", "relu", "linear"}:
            raise ValueError(f"Unknown kernel: {kernel}")
        self.feature_dim = feature_dim or self.head_dim
        self.block_size = int(block_size)
        self.kernel_name = kernel
        self.causal = bool(causal)

        self.q_proj = nn.Linear(hidden_size, num_heads * self.feature_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.feature_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run chunked Lightning Attention over ``hidden_state``."""
        if return_attention_weights:
            raise ValueError(
                "LightningAttention does not materialize attention weights"
            )
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        batch_size, seq_len, _ = hidden_state.size()

        raw_query = self._split(self.q_proj(hidden_state))
        raw_key = self._split(self.k_proj(hidden_state))
        value = self.split_head(self.v_proj(hidden_state))
        query = self._feature_map(raw_query)
        key = self._feature_map(raw_key)

        key_mask = self._key_padding_mask(attention_mask, batch_size)
        if key_mask is not None:
            key = key * key_mask.unsqueeze(-1)
            value = value * key_mask.unsqueeze(-1)
            raw_key = raw_key.masked_fill(~key_mask.unsqueeze(-1), float("-inf"))

        state = torch.zeros(
            batch_size,
            self.num_heads,
            self.feature_dim,
            self.head_dim,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        normalizer = torch.zeros(
            batch_size,
            self.num_heads,
            1,
            1,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )
        outputs: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(self.feature_dim)
        # The causal intra-block mask is identical for equal-length blocks;
        # build each distinct one only once per forward pass.
        future_masks: dict[int, torch.Tensor] = {}

        for start in range(0, seq_len, self.block_size):
            stop = min(start + self.block_size, seq_len)
            q_block = query[:, :, start:stop]
            k_block = key[:, :, start:stop]
            raw_q_block = raw_query[:, :, start:stop]
            raw_k_block = raw_key[:, :, start:stop]
            v_block = value[:, :, start:stop]

            scores = torch.matmul(raw_q_block, raw_k_block.transpose(-1, -2)) * scale
            if self.causal:
                # Within a block, query row i may only attend to keys j <= i.
                length = stop - start
                future = future_masks.get(length)
                if future is None:
                    future = torch.triu(
                        torch.ones(
                            length, length, dtype=torch.bool, device=hidden_state.device
                        ),
                        diagonal=1,
                    )
                    future_masks[length] = future
                scores = scores.masked_fill(future, float("-inf"))
            intra = torch.softmax(scores, dim=-1) @ v_block
            intra = torch.nan_to_num(intra, nan=0.0)

            # Sign-safe normalizer: elu/relu kernels give non-negative sums,
            # but the identity ("linear") kernel may produce negative ones.
            eps = torch.finfo(normalizer.dtype).eps
            safe_normalizer = torch.where(
                normalizer >= 0, normalizer.clamp_min(eps), normalizer.clamp_max(-eps)
            )
            inter = torch.matmul(q_block, state) / safe_normalizer
            outputs.append(intra + inter)

            state = state + torch.matmul(k_block.transpose(-1, -2), v_block)
            normalizer = normalizer + k_block.sum(dim=(2, 3), keepdim=True)

        if outputs:
            output: torch.Tensor = torch.cat(outputs, dim=2)
        else:
            # Empty sequence: no block was processed; an empty slice of
            # ``value`` keeps the autograd graph alive.
            output = value[:, :, :0]
        output = self.o_proj(self.combine_head(output))
        if self.training and self.dropout_prob > 0:
            output = self.dropout(output)
        return output

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """Split a projection into ``(batch, heads, seq, feature_dim)``."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(
            1, 2
        )

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_name == "elu":
            return F.elu(x) + 1.0
        if self.kernel_name == "relu":
            return F.relu(x)
        return x

    @staticmethod
    def _key_padding_mask(
        attention_mask: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # Causal structure is already enforced by the chunked recurrence.
            # Reduce over query positions to retain key-padding visibility.
            attention_mask = attention_mask.any(dim=-2)
        elif attention_mask.dim() != 3:
            raise ValueError("attention_mask must be 3D or 4D")
        elif attention_mask.size(1) != 1:
            attention_mask = attention_mask.any(dim=1, keepdim=True)
        if attention_mask.size(0) != batch_size:
            raise ValueError("attention_mask batch size must match hidden_state")
        return attention_mask.bool()

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"{super().extra_repr()}, feature_dim={self.feature_dim}, "
            f"block_size={self.block_size}, kernel={self.kernel_name}, "
            f"causal={self.causal}"
        )
