"""Educational TTT (Test-Time Training) layer with a linear inner model.

TTT ("Learning to (Learn at Test Time): RNNs with Expressive Hidden States",
Sun et al. 2024, arXiv:2407.04620) makes the hidden state of a sequence
model a model in its own right: processing tokens *is* gradient descent on a
self-supervised loss, so the "state" is a weight matrix instead of a
fixed-size vector. This module implements the TTT-Linear variant. The inner
model is the linear map ``f(x) = W x``; every token contributes one gradient
step on the reconstruction loss ``l(W; x) = 1/2 ||W x_K - x_V||^2`` with
learned projections ``x_K = W_K x`` and ``x_V = W_V x``, and the output rule
is ``z_t = W_t x_{Q,t}`` where ``W_t`` already includes the gradient from
token ``t``. Updates follow the paper's mini-batch (chunked) form: inside a
chunk of ``chunk_size`` tokens, all gradients are evaluated at the
chunk-initial weights, which is what makes the dual form parallelizable.

Teaching simplifications: single-head only (the paper's multi-head TTT is
omitted), the inner model has no bias and no LayerNorm, there is no fused
dual-form kernel (each chunk materializes a
``(batch, chunk_size, hidden, hidden)`` cumulative-gradient tensor), and the
factor of 2 in the exact squared-loss gradient is absorbed into the learning
rate.
"""

from __future__ import annotations

import torch
from torch import nn

from .base_attention import validate_attention_inputs


class TTTLayer(nn.Module):
    """TTT-Linear layer: a linear inner model trained by online gradient descent.

    The per-sequence state ``W`` starts at the learned ``init_weight`` and is
    updated chunk by chunk. Within a chunk the per-token gradients
    ``(W_chunk_start k_s - v_s) k_s^T`` are accumulated inclusively, so the
    output at step ``t`` reads the inner model after the gradient steps of
    tokens up to and including ``t``. Masked (padded) tokens still produce an
    output but contribute no gradient step, matching the key-padding
    semantics of the other linear-time layers in this package.

    Args:
        hidden_size: Dimensionality of input and output features; the inner
            model is a square ``(hidden_size, hidden_size)`` weight matrix.
        chunk_size: Tokens per mini-batch of inner gradient descent. Larger
            chunks parallelize better but evaluate gradients at staler
            weights; ``chunk_size=1`` recovers the naive per-token form.
        inner_lr: Initial (or fixed) learning rate of the inner gradient
            descent, one value per output channel of the inner model.
        learnable_lr: Whether ``inner_lr`` is a learnable parameter. If
            False it is a fixed buffer.
        bias: Whether the k/v/q/o projections use biases.

    """

    def __init__(
        self,
        hidden_size: int,
        chunk_size: int = 16,
        inner_lr: float = 0.01,
        learnable_lr: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if inner_lr <= 0:
            raise ValueError(f"inner_lr must be > 0, got {inner_lr}")

        self.hidden_size = int(hidden_size)
        self.chunk_size = int(chunk_size)

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.init_weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))

        self.inner_lr: torch.Tensor
        lr = torch.full((hidden_size,), float(inner_lr))
        if learnable_lr:
            self.inner_lr = nn.Parameter(lr)
        else:
            self.register_buffer("inner_lr", lr)

        for projection in (self.k_proj, self.v_proj, self.q_proj, self.o_proj):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the TTT layer over ``hidden_state``.

        Args:
            hidden_state: Input tensor of shape (batch_size, seq_len,
                hidden_size).
            attention_mask: Optional mask in the 1/0 (or bool) convention of
                the other attention modules; 3D or 4D, reduced to its
                key-padding component since the update rule is already
                causal.

        """
        if hidden_state.dim() != 3 or hidden_state.size(-1) != self.hidden_size:
            raise ValueError(
                f"hidden_state must have shape (batch, seq, {self.hidden_size})"
            )
        # TTT-Linear is single-headed; the shared validator only checks
        # shapes and the 1/0 mask convention.
        validate_attention_inputs(hidden_state, attention_mask, num_heads=1)
        batch_size, seq_len, _ = hidden_state.size()
        if seq_len == 0:
            return hidden_state.new_zeros(batch_size, 0, self.hidden_size)
        key_padding_mask = self._key_padding_mask(attention_mask, batch_size)

        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)
        query = self.q_proj(hidden_state)
        weight = self.init_weight.unsqueeze(0).expand(
            batch_size, self.hidden_size, self.hidden_size
        )
        lr = self.inner_lr

        outputs: list[torch.Tensor] = []
        for start in range(0, seq_len, self.chunk_size):
            stop = min(start + self.chunk_size, seq_len)
            key_c = key[:, start:stop]
            value_c = value[:, start:stop]
            query_c = query[:, start:stop]

            # Mini-batch TTT: all gradients of the chunk are evaluated at the
            # chunk-initial weight, never at the partially updated one.
            residual = torch.einsum("bij,bsj->bsi", weight, key_c) - value_c
            if key_padding_mask is not None:
                # Padded tokens contribute no gradient step.
                residual = residual * key_padding_mask[:, start:stop].unsqueeze(-1).to(
                    residual.dtype
                )
            grads = torch.einsum("bsi,bsj->bsij", residual, key_c)
            # Inclusive cumsum: token t updates W before producing its output.
            cumulative = torch.cumsum(grads, dim=1)

            # z_t = (W - lr * cumulative_t) q_t, split into two contractions
            # so the chunk-initial weight is shared across the chunk.
            base = torch.einsum("bij,bsj->bsi", weight, query_c)
            correction = torch.einsum("bsij,bsj->bsi", cumulative, query_c)
            outputs.append(base - lr.view(1, 1, -1) * correction)

            weight = weight - lr.view(1, -1, 1) * cumulative[:, -1]
        output: torch.Tensor = self.o_proj(torch.cat(outputs, dim=1))
        return output

    @staticmethod
    def _key_padding_mask(
        attention_mask: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        """Convert a BaseAttention-style mask to a ``(batch, seq)`` mask."""
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # Causal structure is already enforced by the update rule. Reduce
            # over query positions to retain only key-padding visibility.
            attention_mask = attention_mask.any(dim=-2)
        elif attention_mask.dim() != 3:
            raise ValueError("attention_mask must be 3D or 4D")
        elif attention_mask.size(1) != 1:
            attention_mask = attention_mask.any(dim=1, keepdim=True)
        if attention_mask.size(0) != batch_size:
            raise ValueError("attention_mask batch size must match hidden_state")
        return attention_mask.squeeze(1).bool()

    def extra_repr(self) -> str:
        """Show the hidden size, chunk size and lr handling in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, chunk_size={self.chunk_size}, "
            f"learnable_lr={isinstance(self.inner_lr, nn.Parameter)}"
        )
