"""Transformer building blocks that compose attention, norm and FFN modules."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from ..attention.attention_residual import AttentionResidual
from ..attention.hybrid_attention import HybridAttention
from ..attention.multi_head_attention import MultiHeadAttention
from .feed_forward import SwiGLUFFN
from .hyper_connection import ManifoldConstrainedHyperConnection
from .normalization import DeepNorm, LayerNorm, LayerScale, RMSNorm

_NORM_STYLES = ("pre", "post", "sandwich", "deepnorm")
_NORM_TYPES = ("rmsnorm", "layernorm")


class TransformerBlock(nn.Module):
    """Transformer block with configurable norm placement and sublayer layout.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        attention: Optional attention module. Defaults to ``MultiHeadAttention``.
            The module must accept ``attention_mask`` and
            ``return_attention_weights`` keyword arguments (the
            `BaseAttention` interface); ``HybridAttention`` additionally
            receives ``layer_index``.
        ffn: Optional FFN module. Defaults to ``SwiGLUFFN``.
        norm_eps: RMSNorm epsilon.
        pre_norm: Deprecated boolean shortcut for ``norm_style``. When not
            ``None`` it overrides ``norm_style``: ``True`` maps to ``"pre"``
            and ``False`` to ``"post"``. Kept for backward compatibility.
        norm_style: Where normalization is applied around each sublayer.
            ``"pre"`` normalizes the sublayer input (``x + sublayer(norm(x))``);
            ``"post"`` normalizes after the residual add
            (``norm(x + sublayer(x))``); ``"sandwich"`` normalizes both the
            sublayer input and the sublayer output before the residual add
            (``x + norm_out(sublayer(norm_in(x)))``).
        parallel: When ``True`` the attention and FFN sublayers run in
            parallel on the same (normalized) input and their outputs are
            summed into one residual, GPT-J style:
            ``x + attn(norm1(x)) + ffn(norm2(x))`` for ``norm_style="pre"``.
            For ``"post"`` the residual sum is normalized once afterwards;
            for ``"sandwich"`` each sublayer keeps its own input/output norms.
        attention_residual: When ``True`` the attention output is added back
            through a learned per-dimension gate (`AttentionResidual`)
            instead of a plain residual add.
        norm_type: Point-wise normalization used by pre/post/sandwich styles.
        layer_scale_init: Optional initial value for learned per-channel
            residual-branch scaling. ``None`` disables LayerScale.
        deepnorm_alpha: Residual multiplier for ``norm_style="deepnorm"``.
        manifold_hyper_connection: Enable mHC-style constrained residual
            mixing for both attention and FFN branches.
        hc_mult: Number of residual streams used by each mHC mixer.
        sinkhorn_iters: Sinkhorn normalization iterations for mHC.

    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention: nn.Module | None = None,
        ffn: nn.Module | None = None,
        norm_eps: float = 1e-5,
        pre_norm: bool | None = None,
        norm_style: str = "pre",
        parallel: bool = False,
        attention_residual: bool = False,
        norm_type: str = "rmsnorm",
        layer_scale_init: float | None = None,
        deepnorm_alpha: float = 1.0,
        manifold_hyper_connection: bool = False,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
    ) -> None:
        super().__init__()
        if pre_norm is not None:
            norm_style = "pre" if pre_norm else "post"
        if norm_style not in _NORM_STYLES:
            raise ValueError(
                f"Unknown norm_style: {norm_style!r} (expected one of {_NORM_STYLES})"
            )
        if norm_type not in _NORM_TYPES:
            raise ValueError(
                f"Unknown norm_type: {norm_type!r} (expected one of {_NORM_TYPES})"
            )
        if norm_style == "deepnorm" and attention_residual:
            raise ValueError("attention_residual cannot be combined with deepnorm")
        if manifold_hyper_connection and attention_residual:
            raise ValueError(
                "manifold_hyper_connection cannot be combined with attention_residual"
            )
        if manifold_hyper_connection and norm_style == "deepnorm":
            raise ValueError(
                "manifold_hyper_connection cannot be combined with deepnorm"
            )
        if manifold_hyper_connection and parallel:
            raise ValueError("manifold_hyper_connection requires sequential sublayers")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.norm_style = norm_style
        self.norm_type = norm_type
        self.parallel = bool(parallel)
        # Legacy attribute kept for backward compatibility; True only for the
        # pure pre-norm layout.
        self.pre_norm = norm_style == "pre"
        if attention is None:
            attention = MultiHeadAttention(hidden_size, num_heads)
        if ffn is None:
            ffn = SwiGLUFFN(hidden_size, intermediate_size)
        self.attention = attention
        self.ffn = ffn
        norm_class: type[nn.Module] = RMSNorm if norm_type == "rmsnorm" else LayerNorm
        # norm1/norm2 are only built when the style actually calls them:
        # "deepnorm" uses the DeepNorm modules below instead, and the
        # parallel "post" layout normalizes once (norm1) after the residual.
        self.norm1: nn.Module | None = None
        self.norm2: nn.Module | None = None
        if norm_style != "deepnorm":
            self.norm1 = norm_class(hidden_size, eps=norm_eps)
        if norm_style != "deepnorm" and not (norm_style == "post" and parallel):
            self.norm2 = norm_class(hidden_size, eps=norm_eps)
        # Post-sublayer norms, only used by the "sandwich" style.
        self.norm3: nn.Module | None = None
        self.norm4: nn.Module | None = None
        if norm_style == "sandwich":
            self.norm3 = norm_class(hidden_size, eps=norm_eps)
            self.norm4 = norm_class(hidden_size, eps=norm_eps)
        self.deepnorm1: DeepNorm | None = None
        self.deepnorm2: DeepNorm | None = None
        if norm_style == "deepnorm":
            self.deepnorm1 = DeepNorm(hidden_size, alpha=deepnorm_alpha, eps=norm_eps)
            self.deepnorm2 = DeepNorm(hidden_size, alpha=deepnorm_alpha, eps=norm_eps)
        scale_class = (
            (lambda: LayerScale(hidden_size, layer_scale_init))
            if layer_scale_init is not None
            else nn.Identity
        )
        self.attention_scale = scale_class()
        self.ffn_scale = scale_class()
        self.attn_res: AttentionResidual | None = None
        if attention_residual:
            self.attn_res = AttentionResidual(hidden_size)
        self.attention_mhc: ManifoldConstrainedHyperConnection | None = None
        self.ffn_mhc: ManifoldConstrainedHyperConnection | None = None
        if manifold_hyper_connection:
            self.attention_mhc = ManifoldConstrainedHyperConnection(
                hidden_size,
                hc_mult=hc_mult,
                sinkhorn_iters=sinkhorn_iters,
            )
            self.ffn_mhc = ManifoldConstrainedHyperConnection(
                hidden_size,
                hc_mult=hc_mult,
                sinkhorn_iters=sinkhorn_iters,
            )

    def _add_attention_residual(
        self, hidden_state: torch.Tensor, attention_output: torch.Tensor
    ) -> torch.Tensor:
        """Add the attention output back, optionally through the learned gate."""
        if self.attention_mhc is not None:
            return cast(
                torch.Tensor, self.attention_mhc(hidden_state, attention_output)
            )
        if self.attn_res is not None:
            return cast(torch.Tensor, self.attn_res(hidden_state, attention_output))
        return hidden_state + attention_output

    def _add_ffn_residual(
        self, hidden_state: torch.Tensor, ffn_output: torch.Tensor
    ) -> torch.Tensor:
        """Add the FFN branch through plain residual or mHC mixing."""
        if self.ffn_mhc is not None:
            return cast(torch.Tensor, self.ffn_mhc(hidden_state, ffn_output))
        return hidden_state + ffn_output

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
        layer_index: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one transformer block.

        ``layer_index`` is forwarded to ``HybridAttention`` when used, so the
        caller can reproduce Qwen3-Next/Kimi-style 3:1 linear/full layouts.
        """
        # Pre/sandwich feed the normalized input to the attention sublayer;
        # post feeds it the raw input and normalizes after the residual.
        if self.norm_style in {"post", "deepnorm"}:
            attention_input = hidden_state
        else:
            if self.norm1 is None:
                raise RuntimeError("norm1 was not initialized")
            attention_input = self.norm1(hidden_state)
        if isinstance(self.attention, HybridAttention):
            result = self.attention(
                attention_input,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
                layer_index=layer_index,
            )
        else:
            result = self.attention(
                attention_input,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
            )

        if return_attention_weights:
            attention_output, attention_weights = result
        else:
            attention_output = result

        if self.parallel:
            hidden_state = self._parallel_forward(hidden_state, attention_output)
        else:
            hidden_state = self._sequential_forward(hidden_state, attention_output)

        if return_attention_weights:
            return hidden_state, attention_weights
        return hidden_state

    def forward_with_cache(
        self,
        hidden_state: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_index: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one transformer block with a KV cache for incremental decoding.

        This mirrors :meth:`forward` exactly — the residual, norm-style,
        parallel and FFN logic is shared — but routes the attention sublayer
        through its ``forward_with_cache`` method so keys/values are
        appended to ``past_key_value`` instead of recomputed for the whole
        sequence. ``layer_index`` is accepted for interface parity with
        :meth:`forward`; it is only meaningful for ``HybridAttention``,
        which does not support caching.

        Args:
            hidden_state: New-token input of shape (batch_size, q_len,
                hidden_size).
            past_key_value: Optional cached ``(key, value)`` pair for this
                block from previous decode steps.
            attention_mask: Optional keep-mask broadcastable against the
                (batch_size, num_heads, q_len, past_len + q_len) scores.
            layer_index: Unused; kept for signature parity with
                :meth:`forward`.

        Returns:
            The block output of shape (batch_size, q_len, hidden_size) and
            the updated ``(key, value)`` cache for this block.

        Raises:
            ValueError: If the attention module does not implement
                ``forward_with_cache``.

        """
        del layer_index  # only HybridAttention consumes it; it cannot cache
        attention_forward_with_cache = getattr(
            self.attention, "forward_with_cache", None
        )
        if not callable(attention_forward_with_cache):
            raise ValueError(
                f"{type(self.attention).__name__} does not support KV-cache "
                "decoding; forward_with_cache is implemented by "
                "MultiHeadAttention, GroupedQueryAttention and "
                "MultiQueryAttention"
            )
        if self.norm_style in {"post", "deepnorm"}:
            attention_input = hidden_state
        else:
            if self.norm1 is None:
                raise RuntimeError("norm1 was not initialized")
            attention_input = self.norm1(hidden_state)
        attention_output, present = attention_forward_with_cache(
            attention_input,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
        )
        if self.parallel:
            hidden_state = self._parallel_forward(hidden_state, attention_output)
        else:
            hidden_state = self._sequential_forward(hidden_state, attention_output)
        return hidden_state, present

    def _sequential_forward(
        self, hidden_state: torch.Tensor, attention_output: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention then FFN, each wrapped in its own residual."""
        attention_output = self.attention_scale(attention_output)
        if self.norm_style == "deepnorm":
            if self.deepnorm1 is None or self.deepnorm2 is None:
                raise RuntimeError("deepnorm modules were not initialized")
            hidden_state = self.deepnorm1(hidden_state, attention_output)
            ffn_output = self.ffn_scale(self.ffn(hidden_state))
            return cast(torch.Tensor, self.deepnorm2(hidden_state, ffn_output))
        if self.norm_style == "sandwich":
            if self.norm3 is None or self.norm4 is None:
                raise RuntimeError("sandwich norms were not initialized")
            attention_output = self.norm3(attention_output)
        hidden_state = self._add_attention_residual(hidden_state, attention_output)
        if self.norm_style == "post":
            if self.norm1 is None or self.norm2 is None:
                raise RuntimeError("post-norm modules were not initialized")
            hidden_state = self.norm1(hidden_state)
            return cast(
                torch.Tensor,
                self.norm2(
                    self._add_ffn_residual(hidden_state, self.ffn(hidden_state))
                ),
            )
        if self.norm2 is None:
            raise RuntimeError("norm2 was not initialized")
        ffn_output = self.ffn_scale(self.ffn(self.norm2(hidden_state)))
        if self.norm_style == "sandwich":
            if self.norm4 is None:
                raise RuntimeError("sandwich FFN norm was not initialized")
            ffn_output = self.norm4(ffn_output)
        return self._add_ffn_residual(hidden_state, ffn_output)

    def _parallel_forward(
        self, hidden_state: torch.Tensor, attention_output: torch.Tensor
    ) -> torch.Tensor:
        """GPT-J style parallel block: FFN reads the same input as attention."""
        attention_output = self.attention_scale(attention_output)
        if self.norm_style == "deepnorm":
            if self.deepnorm1 is None:
                raise RuntimeError("deepnorm module was not initialized")
            combined = attention_output + self.ffn_scale(self.ffn(hidden_state))
            return cast(torch.Tensor, self.deepnorm1(hidden_state, combined))
        if self.norm_style == "post":
            if self.norm1 is None:
                raise RuntimeError("norm1 was not initialized")
            ffn_output = self.ffn_scale(self.ffn(hidden_state))
            hidden_state = self._add_attention_residual(hidden_state, attention_output)
            return cast(torch.Tensor, self.norm1(hidden_state + ffn_output))
        if self.norm2 is None:
            raise RuntimeError("norm2 was not initialized")
        ffn_output = cast(
            torch.Tensor, self.ffn_scale(self.ffn(self.norm2(hidden_state)))
        )
        if self.norm_style == "sandwich":
            if self.norm3 is None or self.norm4 is None:
                raise RuntimeError("sandwich norms were not initialized")
            attention_output = self.norm3(attention_output)
            ffn_output = cast(torch.Tensor, self.norm4(ffn_output))
        hidden_state = self._add_attention_residual(hidden_state, attention_output)
        return hidden_state + ffn_output

    def extra_repr(self) -> str:
        """Summarize the block's layout and norm settings in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"intermediate_size={self.intermediate_size}, "
            f"norm_style={self.norm_style!r}, norm_type={self.norm_type!r}, "
            f"parallel={self.parallel}, "
            f"attention_residual={self.attn_res is not None}, "
            f"manifold_hyper_connection={self.attention_mhc is not None}"
        )
