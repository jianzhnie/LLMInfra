"""Hybrid SSM/attention block that alternates sublayers by a pattern.

This is a teaching-level generalization of the hybrid layouts used by
Zamba (Mamba blocks interleaved with shared attention) and Qwen3-Next
(linear/SSM layers with occasional full attention): a ``pattern`` such as
``"ssm:ssm:attn"`` decides which sublayer type runs at each position.
Unlike the real architectures, the sublayers are composed sequentially
without residual connections or normalization, to keep the routing logic
front and center.
"""

from __future__ import annotations

import torch
from torch import nn

from ..attention.linear_attention import LinearAttention
from ..attention.multi_head_attention import MultiHeadAttention
from .feed_forward import SwiGLUFFN
from .normalization import RMSNorm
from .mamba2 import Mamba2Layer, Mamba2State

VALID_TOKENS = ("ssm", "attn")
HYBRID_LAYER_TYPES = ("linear", "ssm", "full")


def _causal_attention_mask(hidden_state: torch.Tensor) -> torch.Tensor:
    """Build a lower-triangular causal mask matching ``hidden_state``."""
    batch_size, seq_len = hidden_state.shape[:2]
    mask = torch.ones(
        seq_len, seq_len, dtype=torch.bool, device=hidden_state.device
    ).tril_()
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


class HybridSSMBlock(nn.Module):
    """Alternate Mamba2 SSM sublayers and attention sublayers by a pattern.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads, used when ``attention`` is
            ``None`` and a fresh ``MultiHeadAttention`` is created per
            ``"attn"`` token.
        pattern: Sublayer layout, either a ``":"``-separated string such as
            ``"ssm:ssm:attn"`` or a list of tokens. Each token is ``"ssm"``
            (a fresh ``Mamba2Layer``) or ``"attn"`` (an attention module).
        attention: Optional attention module used for every ``"attn"``
            token (weights are shared across positions). Defaults to one
            fresh ``MultiHeadAttention`` per ``"attn"`` token.
        d_state: SSM state dimension for each ``Mamba2Layer``.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        pattern: str | list[str] = "ssm:attn",
        attention: nn.Module | None = None,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        tokens = pattern.split(":") if isinstance(pattern, str) else list(pattern)
        if not tokens:
            raise ValueError("pattern must contain at least one token")
        invalid = [token for token in tokens if token not in VALID_TOKENS]
        if invalid:
            raise ValueError(
                f"unknown pattern tokens {invalid}; expected {list(VALID_TOKENS)}"
            )
        self.pattern = tokens

        layers: list[nn.Module] = []
        for token in tokens:
            if token == "ssm":
                layers.append(Mamba2Layer(hidden_size, d_state=d_state))
            elif attention is None:
                layers.append(MultiHeadAttention(hidden_size, num_heads))
            else:
                layers.append(attention)
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run each sublayer in pattern order.

        SSM sublayers return ``(output, state)``; only ``output`` is passed
        on and the state is dropped (no streaming across calls here).
        ``attention_mask`` is forwarded to every attention sublayer. When no
        mask is given, attention sublayers default to a causal
        (lower-triangular) mask so they cannot leak future tokens.

        Args:
            hidden_state: Input of shape ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional mask for the attention sublayers.

        Returns:
            Final hidden states, same shape as the input.

        """
        for layer in self.layers:
            if isinstance(layer, Mamba2Layer):
                hidden_state, _ = layer(hidden_state)
            else:
                layer_mask = attention_mask
                if layer_mask is None:
                    layer_mask = _causal_attention_mask(hidden_state)
                hidden_state = layer(hidden_state, attention_mask=layer_mask)
        return hidden_state

    def extra_repr(self) -> str:
        """Show the sublayer pattern and layer sizes in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"pattern={':'.join(self.pattern)}"
        )


class HybridLayerStack(nn.Module):
    """Pre-norm residual stack with a Linear/SSM/Full layer map.

    This module provides one model-level layout interface for Qwen3-Next-like
    linear/full mixtures, Zamba-like SSM/shared-attention mixtures, and custom
    three-way combinations. Every layer contains a mixer followed by a
    SwiGLU FFN. ``full`` layers may share one attention module to reproduce
    Zamba's shared-attention pattern.

    Args:
        hidden_size: Input/output feature dimension.
        num_heads: Head count for linear and full attention.
        intermediate_size: SwiGLU intermediate dimension.
        layer_map: Colon-separated string or sequence containing ``linear``,
            ``ssm`` and ``full``.
        d_state: State size for every SSM layer.
        shared_full_attention: Reuse one full-attention module for all
            ``full`` positions.
        full_attention: Optional custom full-attention module.
        dropout: Dropout on mixer and FFN residual branches.
        norm_eps: RMSNorm epsilon.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_map: str | list[str] = "linear:linear:ssm:full",
        d_state: int = 16,
        shared_full_attention: bool = False,
        full_attention: nn.Module | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        tokens = layer_map.split(":") if isinstance(layer_map, str) else list(layer_map)
        tokens = ["full" if token == "attn" else token for token in tokens]
        if not tokens:
            raise ValueError("layer_map must contain at least one layer")
        invalid = [token for token in tokens if token not in HYBRID_LAYER_TYPES]
        if invalid:
            raise ValueError(
                f"unknown layer types {invalid}; expected {list(HYBRID_LAYER_TYPES)}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.layer_map = tuple(tokens)
        self.shared_full_attention = bool(shared_full_attention)
        shared_attention = full_attention
        if shared_full_attention and shared_attention is None:
            shared_attention = MultiHeadAttention(hidden_size, num_heads)

        mixers: list[nn.Module] = []
        for token in tokens:
            if token == "linear":
                mixers.append(LinearAttention(hidden_size, num_heads, causal=True))
            elif token == "ssm":
                mixers.append(Mamba2Layer(hidden_size, d_state=d_state))
            elif shared_attention is not None:
                mixers.append(shared_attention)
            else:
                mixers.append(MultiHeadAttention(hidden_size, num_heads))
        self.mixers = nn.ModuleList(mixers)
        self.mixer_norms = nn.ModuleList(
            RMSNorm(hidden_size, eps=norm_eps) for _ in tokens
        )
        self.ffn_norms = nn.ModuleList(
            RMSNorm(hidden_size, eps=norm_eps) for _ in tokens
        )
        self.ffns = nn.ModuleList(
            SwiGLUFFN(hidden_size, intermediate_size) for _ in tokens
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        states: list[Mamba2State | None] | None = None,
        return_state: bool = False,
        scan: str = "recurrent",
        chunk_size: int = 16,
    ) -> torch.Tensor | tuple[torch.Tensor, list[Mamba2State | None]]:
        """Evaluate the configured layer map and optionally return SSM states.

        ``states`` is aligned with ``layer_map``; non-SSM entries must be
        ``None``. This makes state ownership explicit during streaming decode.
        When ``attention_mask`` is ``None``, ``full`` layers fall back to a
        causal (lower-triangular) mask; ``linear`` and ``ssm`` layers are
        causal by construction.
        """
        if states is None:
            states = [None] * len(self.mixers)
        if len(states) != len(self.mixers):
            raise ValueError("states must have one entry per layer_map position")

        next_states: list[Mamba2State | None] = []
        for layer_type, mixer, mixer_norm, ffn_norm, ffn, state in zip(
            self.layer_map,
            self.mixers,
            self.mixer_norms,
            self.ffn_norms,
            self.ffns,
            states,
            strict=True,
        ):
            normalized = mixer_norm(hidden_state)
            if layer_type == "ssm":
                if not isinstance(mixer, Mamba2Layer):
                    raise TypeError("ssm layer map entry must contain Mamba2Layer")
                mixed, next_state = mixer(
                    normalized,
                    state=state,
                    scan=scan,
                    chunk_size=chunk_size,
                )
                next_states.append(next_state)
            else:
                if state is not None:
                    raise ValueError("non-SSM layer state must be None")
                layer_mask = attention_mask
                if layer_mask is None and layer_type == "full":
                    # LinearAttention is already causal (causal=True); full
                    # attention defaults to a causal mask to match.
                    layer_mask = _causal_attention_mask(normalized)
                mixed = mixer(normalized, attention_mask=layer_mask)
                next_states.append(None)
            hidden_state = hidden_state + self.dropout(mixed)
            hidden_state = hidden_state + self.dropout(ffn(ffn_norm(hidden_state)))

        if return_state:
            return hidden_state, next_states
        return hidden_state

    def extra_repr(self) -> str:
        """Show the layer map and stack settings in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"layer_map={':'.join(self.layer_map)}, "
            f"shared_full_attention={self.shared_full_attention}"
        )
