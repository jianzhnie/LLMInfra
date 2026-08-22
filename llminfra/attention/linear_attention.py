"""Educational linear attention implementation.

Linear attention replaces ``softmax(QK^T)`` with a kernel feature map so the
key/value information can be accumulated into a fixed-size recurrent state.
This is the core family behind Kimi Delta Attention and Qwen3-Next's Gated
DeltaNet layers, although the production versions add gating, chunkwise
parallelism and custom kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base_attention import BaseAttention, validate_attention_inputs


class LinearAttention(BaseAttention):
    """Kernelized linear attention with optional causal state accumulation.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        feature_dim: Feature dimension of the kernel feature map.
        kernel: One of ``"elu"``, ``"relu"`` or ``"linear"``.
        causal: Whether keys after the current query are masked out.
        chunk_size: Tokens per chunk in the causal scan. The causal path
            keeps a running ``(feature_dim, head_dim)`` state per head and
            only materializes one chunk at a time, so peak memory is
            ``O(chunk_size * feature_dim * head_dim + chunk_size**2)``
            instead of ``O(seq_len * feature_dim * head_dim)``. It is a
            performance knob and does not change the result.
        dropout: Dropout applied to the output when training.
        bias: Whether linear projections use biases.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int | None = None,
        kernel: str = "elu",
        causal: bool = True,
        chunk_size: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        if kernel not in {"elu", "relu", "linear"}:
            raise ValueError(f"Unknown linear attention kernel: {kernel}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

        self.feature_dim = feature_dim or self.head_dim
        self.kernel_name = kernel
        self.causal = bool(causal)
        self.chunk_size = int(chunk_size)

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
        """Run linear attention over ``hidden_state``.

        Linear attention does not materialize an ``(n, n)`` attention matrix,
        so ``return_attention_weights=True`` is not supported.
        """
        if return_attention_weights:
            raise ValueError("LinearAttention does not materialize attention weights")
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)

        query = self._split(self.q_proj(hidden_state), self.feature_dim)
        key = self._split(self.k_proj(hidden_state), self.feature_dim)
        value = self.split_head(self.v_proj(hidden_state))
        key_padding_mask = self._key_padding_mask(attention_mask, hidden_state.size(0))

        query = self._feature_map(query)
        key = self._feature_map(key)
        if key_padding_mask is not None:
            key = key * key_padding_mask.unsqueeze(-1)
            value = value * key_padding_mask.unsqueeze(-1)

        if self.causal:
            out_unnorm, normalizer = self._causal_scan(query, key, value)
        else:
            # Non-causal form: out = q · sum_s (k_s ⊗ v_s); the (f, d) state
            # is shared by all positions.
            kv_state = torch.einsum("bhsf,bhsd->bhfd", key, value)
            out_unnorm = torch.einsum("bhsf,bhfd->bhsd", query, kv_state)
            normalizer = torch.einsum("bhsf,bhf->bhs", query, key.sum(dim=2))

        # Sign-safe division: elu/relu kernels give non-negative normalizers,
        # but the identity ("linear") kernel may legitimately produce negative
        # ones. Only exactly-zero rows (fully masked) are zeroed out.
        eps = torch.finfo(normalizer.dtype).eps
        safe_normalizer = torch.where(
            normalizer >= 0, normalizer.clamp_min(eps), normalizer.clamp_max(-eps)
        )
        output: torch.Tensor = out_unnorm / safe_normalizer.unsqueeze(-1)
        output = output.where(normalizer.unsqueeze(-1) != 0, torch.zeros_like(output))
        output = self.combine_head(output)
        output = self.o_proj(output)
        if self.training and self.dropout_prob > 0:
            output = self.dropout(output)
        return output

    def _causal_scan(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked causal scan returning unnormalized outputs and normalizers.

        Causal form: ``out[s] = q_s · sum_{s'<=s} (k_s' ⊗ v_s')``. Instead of
        materializing the per-step ``(batch, heads, seq, f, d)`` cumsum, a
        running ``(f, d)`` state per head is carried across chunks: each chunk
        adds its intra-chunk causal contribution plus ``phi(q) @ state`` for
        the history. Peak memory drops from ``O(s·f·d)`` to
        ``O(chunk_size·f·d + chunk_size²)``. Masked keys/values are already
        zeroed by the caller, so they contribute nothing to the state.
        """
        batch_size, _, seq_len, _ = query.shape
        if seq_len == 0:
            # The chunk loop never runs for empty sequences and ``torch.cat``
            # rejects empty lists; empty slices keep the autograd graph alive.
            return value[:, :, :0], query[:, :, :0, 0]
        state = query.new_zeros(
            batch_size, self.num_heads, self.feature_dim, self.head_dim
        )
        key_state = query.new_zeros(batch_size, self.num_heads, self.feature_dim)
        out_parts: list[torch.Tensor] = []
        norm_parts: list[torch.Tensor] = []
        # Causal intra-chunk masks are identical for equal-length chunks;
        # build each distinct one only once per forward pass.
        future_masks: dict[int, torch.Tensor] = {}

        for start in range(0, seq_len, self.chunk_size):
            stop = min(start + self.chunk_size, seq_len)
            q_chunk = query[:, :, start:stop]
            k_chunk = key[:, :, start:stop]
            v_chunk = value[:, :, start:stop]

            # Intra-chunk: causally masked phi(q) phi(k)^T applied to v.
            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
            length = stop - start
            future = future_masks.get(length)
            if future is None:
                future = torch.triu(
                    torch.ones(length, length, dtype=torch.bool, device=query.device),
                    diagonal=1,
                )
                future_masks[length] = future
            scores = scores.masked_fill(future, 0.0)
            local = torch.matmul(scores, v_chunk)
            # Inter-chunk: attend to the running state of previous chunks.
            history = torch.matmul(q_chunk, state)
            out_parts.append(local + history)

            # Row sums of the causally masked scores are exactly
            # sum_{t<=s} phi(q_s)·phi(k_t), so no cumsum is needed.
            local_norm = scores.sum(dim=-1)
            history_norm = torch.matmul(q_chunk, key_state.unsqueeze(-1)).squeeze(-1)
            norm_parts.append(local_norm + history_norm)

            state = state + torch.matmul(k_chunk.transpose(-1, -2), v_chunk)
            key_state = key_state + k_chunk.sum(dim=2)

        return torch.cat(out_parts, dim=2), torch.cat(norm_parts, dim=2)

    def _split(self, x: torch.Tensor, feature_dim: int) -> torch.Tensor:
        """Split a projection into ``(batch, heads, seq, feature_dim)``."""
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, feature_dim).transpose(1, 2)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured non-negative kernel feature map."""
        if self.kernel_name == "elu":
            return F.elu(x) + 1.0
        if self.kernel_name == "relu":
            return F.relu(x)
        return x

    @staticmethod
    def _key_padding_mask(
        attention_mask: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        """Convert a BaseAttention-style mask to a ``(batch, 1, seq)`` mask."""
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # Causal structure is already enforced by the recurrence. Reduce
            # over query positions to retain only key-padding visibility.
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
            f"kernel={self.kernel_name}, causal={self.causal}, "
            f"chunk_size={self.chunk_size}"
        )
