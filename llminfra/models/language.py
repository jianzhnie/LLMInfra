"""Composable causal language model built from LLMInfra modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.feed_forward import SwiGLUFFN
from ..layers.normalization import RMSNorm
from ..layers.transformer_block import TransformerBlock
from ..module_registry import build_attention, build_positional_encoding
from ..moe import DeepSeekMoE
from ..positional.multimodal_rope import MultiModalRotaryPositionEmbedding
from ..speculative_decoding.mtp import MultiTokenPredictionHead, mtp_loss
from .encoder_decoder import cached_causal_mask


@dataclass
class CausalLMOutput:
    """Structured causal-LM output for training and auxiliary heads."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None
    mtp_logits: list[torch.Tensor] | None = None


class CausalLMModel(nn.Module):
    """Teaching-level causal language model.

    This class composes embedding, positional encoding, transformer blocks,
    RMSNorm and an output head. It is intended for architecture experiments,
    not for reproducing a specific production model.

    Args:
        vocab_size: Vocabulary size for the embedding and LM head.
        hidden_size: Model hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        intermediate_size: FFN (or expert) intermediate dimension.
        max_seq_len: Maximum supported sequence length.
        attention_name: Registry name of the attention module (see
            ``list_attentions()``). Note: ``positional="alibi"`` forces
            ``attention_name="alibi"``, because ALiBi is applied as an
            attention-score bias rather than an input embedding.
            ``"ring"`` implements causal masking internally and does not
            support padding masks or prefix-LM masking.
        attention_kwargs: Extra keyword arguments for the attention module.
            For ``"gqa"``, ``num_kv_groups`` defaults to ``num_heads // 2``.
        positional: Positional encoding name (``rope``, ``yarn``, ``ntk``,
            ``alibi``, ``2d``, ...) or ``"none"``. ``"t5_bias"`` is a
            score-bias module with no causal-attention consumer in this
            package and is rejected.
        positional_kwargs: Extra keyword arguments for the positional module.
        use_moe: Replace the FFN with a ``DeepSeekMoE``.
        num_experts: Number of routed experts when ``use_moe`` is set.
        expert_top_k: Experts selected per token when ``use_moe`` is set.
        num_shared_experts: Always-on shared experts when ``use_moe`` is set.
        norm_eps: RMSNorm epsilon.
        tie_word_embeddings: Share the LM head weight with the token embedding.
        num_mtp_predictions: Number of future-token auxiliary prediction
            heads. Zero disables MTP.
        mtp_loss_weight: Weight applied to the MTP loss when labels are given.
        norm_type: ``"rmsnorm"`` or ``"layernorm"`` inside blocks.
        norm_style: ``"pre"``, ``"post"``, ``"sandwich"`` or ``"deepnorm"``.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 4096,
        attention_name: str = "gqa",
        attention_kwargs: dict[str, Any] | None = None,
        positional: str = "rope",
        positional_kwargs: dict[str, Any] | None = None,
        use_moe: bool = False,
        num_experts: int = 8,
        expert_top_k: int = 2,
        num_shared_experts: int = 1,
        norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        num_mtp_predictions: int = 0,
        mtp_loss_weight: float = 0.1,
        norm_type: str = "rmsnorm",
        norm_style: str = "pre",
        parallel_blocks: bool = False,
        layer_scale_init: float | None = None,
        attention_residual: bool = False,
        deepnorm_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("vocab_size, hidden_size and num_layers must be >= 1")
        if num_mtp_predictions < 0:
            raise ValueError("num_mtp_predictions must be >= 0")
        if mtp_loss_weight < 0:
            raise ValueError("mtp_loss_weight must be >= 0")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_seq_len = int(max_seq_len)
        if positional == "alibi":
            # ALiBi is an attention-score bias, not an input embedding: swap
            # in the ALiBi attention module and skip input-side encoding.
            attention_name = "alibi"
            positional = "none"
        if positional == "t5_bias":
            raise ValueError(
                "t5_bias is an attention-score bias and cannot be applied as "
                "a causal-LM input embedding"
            )
        self.attention_name = attention_name
        self.use_moe = bool(use_moe)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.num_mtp_predictions = int(num_mtp_predictions)
        self.mtp_loss_weight = float(mtp_loss_weight)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        attention_kwargs = dict(attention_kwargs or {})
        if attention_name == "alibi" and "max_seq_len" not in attention_kwargs:
            attention_kwargs["max_seq_len"] = max_seq_len
        if attention_name == "gqa" and "num_kv_groups" not in attention_kwargs:
            # Largest divisor of num_heads not exceeding num_heads // 2, so
            # the default stays valid even when num_heads is odd or prime.
            attention_kwargs["num_kv_groups"] = next(
                groups
                for groups in range(max(1, num_heads // 2), 0, -1)
                if num_heads % groups == 0
            )

        self.positional = (
            None
            if positional in {None, "none"}
            else build_positional_encoding(
                positional,
                dim=hidden_size,
                max_seq_len=max_seq_len,
                **dict(positional_kwargs or {}),
            )
        )

        self.blocks = nn.ModuleList(
            TransformerBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                attention=build_attention(
                    attention_name,
                    hidden_size,
                    num_heads,
                    **attention_kwargs,
                ),
                ffn=(
                    DeepSeekMoE(
                        hidden_size,
                        num_routed_experts=num_experts,
                        num_shared_experts=num_shared_experts,
                        intermediate_size=intermediate_size,
                        top_k=expert_top_k,
                    )
                    if use_moe
                    else SwiGLUFFN(hidden_size, intermediate_size)
                ),
                norm_eps=norm_eps,
                norm_type=norm_type,
                norm_style=norm_style,
                parallel=parallel_blocks,
                layer_scale_init=layer_scale_init,
                attention_residual=attention_residual,
                deepnorm_alpha=deepnorm_alpha,
            )
            for _ in range(num_layers)
        )

        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.mtp_head: MultiTokenPredictionHead | None = None
        if num_mtp_predictions:
            self.mtp_head = MultiTokenPredictionHead(
                hidden_size,
                vocab_size,
                num_predictions=num_mtp_predictions,
            )
            if tie_word_embeddings:
                for head in self.mtp_head.heads:
                    head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
        prefix_len: int | torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = False,
        return_mtp: bool = False,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | CausalLMOutput:
        """Run the model over token ids or pre-computed embeddings.

        Args:
            input_ids: Long tensor of shape ``(batch, seq_len)``.
            inputs_embeds: Optional embeddings shaped ``(batch, seq_len,
                hidden_size)``. Exactly one of ``input_ids`` and
                ``inputs_embeds`` must be supplied.
            attention_mask: Optional padding mask of shape
                ``(batch, seq_len)``, ``(batch, 1, seq_len)`` or
                ``(batch, 1, seq_len, seq_len)``.
            return_attention_weights: Return weights from the final block when
                the attention module supports them.
            prefix_len: Optional prefix-LM length, either one integer shared by
                the batch or a ``(batch,)`` tensor for per-example lengths.
                Prefix positions are bidirectionally visible; all remaining
                positions stay causal. ``None`` keeps the purely causal mask.
            labels: Optional next-token targets shaped like ``input_ids``.
                Supplying labels returns :class:`CausalLMOutput` with loss.
            return_dict: Return :class:`CausalLMOutput` instead of the legacy
                tensor/tuple result.
            return_mtp: Include auxiliary MTP logits. Requires the model to
                have been constructed with ``num_mtp_predictions > 0``.
            position_ids: Optional multimodal position ids. This is consumed
                by ``mrope``; other positional modules use implicit indices.

        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            if input_ids.dim() != 2:
                raise ValueError("input_ids must have shape (batch, seq_len)")
            batch_size, seq_len = input_ids.size()
            hidden_state = self.embed_tokens(input_ids)
            device = input_ids.device
        else:
            if inputs_embeds is None:
                raise ValueError("inputs_embeds is required when input_ids is omitted")
            if inputs_embeds.dim() != 3 or inputs_embeds.size(-1) != self.hidden_size:
                raise ValueError(
                    "inputs_embeds must have shape (batch, seq, hidden_size)"
                )
            batch_size, seq_len, _ = inputs_embeds.size()
            hidden_state = inputs_embeds
            device = inputs_embeds.device
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        if self.positional is not None:
            if position_ids is not None and isinstance(
                self.positional, MultiModalRotaryPositionEmbedding
            ):
                hidden_state = self.positional(hidden_state, position_ids)
            elif position_ids is not None:
                raise ValueError("position_ids are only supported by mrope")
            else:
                hidden_state = self.positional(hidden_state)

        # Padding positions must not leak their token embedding through the
        # residual stream. Key masking alone prevents other tokens from
        # attending to padding, but the padded query position would still
        # retain its own embedding through every residual connection.
        query_padding_mask: torch.Tensor | None = None
        if attention_mask is not None and attention_mask.dim() == 2:
            query_padding_mask = attention_mask[:, :, None].to(
                device=hidden_state.device,
                dtype=hidden_state.dtype,
            )
            hidden_state = hidden_state * query_padding_mask

        combined_mask: torch.Tensor | None = self._build_mask(
            attention_mask, batch_size, seq_len, device, prefix_len
        )
        # RingAttention is causal internally and rejects mask arguments; with
        # no padding or prefix the combined mask would be purely causal, so
        # omit it. A user-supplied mask still flows through and ring raises
        # its own error there, which is the documented contract.
        if (
            self.attention_name == "ring"
            and attention_mask is None
            and prefix_len is None
        ):
            combined_mask = None
        last_weights: torch.Tensor | None = None
        for layer_index, block in enumerate(self.blocks):
            wants_weights = (
                return_attention_weights and layer_index == self.num_layers - 1
            )
            result = block(
                hidden_state,
                attention_mask=combined_mask,
                return_attention_weights=wants_weights,
                layer_index=layer_index,
            )
            if wants_weights:
                hidden_state, last_weights = result
            else:
                hidden_state = result
            if query_padding_mask is not None:
                hidden_state = hidden_state * query_padding_mask

        hidden_state = self.norm(hidden_state)
        logits: torch.Tensor = self.lm_head(hidden_state)
        if query_padding_mask is not None:
            logits = logits * query_padding_mask
        mtp_logits: list[torch.Tensor] | None = None
        if return_mtp or labels is not None:
            if self.mtp_head is None and return_mtp:
                raise ValueError("return_mtp requires num_mtp_predictions > 0")
            if self.mtp_head is not None:
                mtp_logits = self.mtp_head(hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            if labels.shape != (batch_size, seq_len):
                raise ValueError("labels must have shape (batch, seq_len)")
            if seq_len < 2:
                raise ValueError("labels require a sequence length of at least 2")
            # Score position i against labels[i + 1]. Slicing logits[:, :-1]
            # would force a full (batch, seq, vocab) copy because the slice
            # is not viewable; shifting the (tiny) labels instead and
            # ignoring each row's final slot keeps logits a view. The set of
            # scored positions is unchanged, so the mean loss is identical.
            shifted_labels = torch.cat(
                [labels[:, 1:], labels.new_full((batch_size, 1), -100)], dim=1
            )
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                shifted_labels.reshape(-1),
            )
            if self.mtp_head is not None and self.mtp_loss_weight:
                loss = loss + self.mtp_loss_weight * mtp_loss(
                    self.mtp_head,
                    hidden_state,
                    labels,
                    logits_list=mtp_logits,
                )

        if return_dict or labels is not None or return_mtp:
            return CausalLMOutput(
                logits=logits,
                loss=loss,
                attention_weights=last_weights,
                mtp_logits=mtp_logits,
            )
        if return_attention_weights:
            if last_weights is None:
                raise ValueError(
                    "The configured attention module does not return weights"
                )
            return logits, last_weights
        return logits

    @staticmethod
    def _build_mask(
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        prefix_len: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine a user padding mask with a causal (or prefix-LM) mask.

        With ``prefix_len`` set, key positions before each example's prefix
        length are visible to every query position (bidirectional prefix),
        while later key positions follow the causal lower triangle. The
        result is combined with ``attention_mask`` (1 = keep) via logical AND.
        """
        causal = cached_causal_mask(seq_len, device).expand(
            batch_size, 1, seq_len, seq_len
        )
        if prefix_len is not None:
            if isinstance(prefix_len, int):
                prefix_lengths = torch.full(
                    (batch_size,), prefix_len, dtype=torch.long, device=device
                )
            elif isinstance(prefix_len, torch.Tensor):
                if prefix_len.shape != (batch_size,):
                    raise ValueError("prefix_len tensor must have shape (batch,)")
                prefix_lengths = prefix_len.to(device=device, dtype=torch.long)
            else:
                raise TypeError("prefix_len must be an int, tensor, or None")
            if not ((prefix_lengths >= 1) & (prefix_lengths <= seq_len)).all():
                raise ValueError(f"prefix_len values must be in [1, {seq_len}]")
            prefix_keys = (
                torch.arange(seq_len, device=device)[None, :] < prefix_lengths[:, None]
            ).view(batch_size, 1, 1, seq_len)
            causal = causal | prefix_keys
        if attention_mask is None:
            return causal
        attention_mask = attention_mask.to(device=device)
        if attention_mask.dim() == 2:
            padding = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            padding = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 4:
            padding = attention_mask
        else:
            raise ValueError("attention_mask must be 2D, 3D or 4D")
        if padding.size(0) != batch_size:
            raise ValueError("attention_mask batch size must match input_ids")
        return causal & padding.bool()

    def extra_repr(self) -> str:
        """Report the architecture configuration shown in the module ``repr``."""
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"attention_name={self.attention_name}, use_moe={self.use_moe}, "
            f"num_mtp_predictions={self.num_mtp_predictions}"
        )


class PrefixLMModel(CausalLMModel):
    """Causal language model with an explicit bidirectional prefix contract.

    ``prefix_lengths`` is required on every forward call and may vary across
    batch rows. The implementation otherwise shares all architecture options,
    output types, and training semantics with :class:`CausalLMModel`.
    """

    # The explicit-prefix contract intentionally narrows forward: the
    # parent's optional ``prefix_len`` becomes the required keyword-only
    # ``prefix_lengths`` so prefix-LM masking is never applied by accident.
    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None = None,
        *,
        prefix_lengths: int | torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
        labels: torch.Tensor | None = None,
        return_dict: bool = False,
        return_mtp: bool = False,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | CausalLMOutput:
        """Run prefix-LM masking with scalar or per-example prefix lengths."""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
            prefix_len=prefix_lengths,
            labels=labels,
            return_dict=return_dict,
            return_mtp=return_mtp,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )


__all__ = ["CausalLMModel", "CausalLMOutput", "PrefixLMModel"]
