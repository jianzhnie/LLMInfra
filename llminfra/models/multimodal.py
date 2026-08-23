"""Teaching-grade multimodal fusion interfaces for LLMInfra.

This module provides the minimal interfaces needed to plug vision features
into a text model: a ``VisionEncoderAdapter`` that projects pre-computed
patch features into the model's hidden size, and a ``CrossAttentionFuser``
that performs late fusion by letting text tokens attend to vision tokens.

Simplifications (documented for teaching purposes):

- ``VisionEncoderAdapter`` is a single linear projection, not a real ViT.
  It is intentionally *not* compatible with pretrained ViT checkpoints; it
  only defines the tensor interface ``(batch, num_patches, vision_dim) ->
  (batch, num_patches, hidden_size)``.
- ``CrossAttentionFuser`` reuses ``CrossAttention`` from
  ``llminfra.models.encoder_decoder`` because ``MultiHeadAttention`` only supports
  self-attention (query length == key/value length), while fusion requires
  text queries to attend over a different number of vision key/value tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .encoder_decoder import CrossAttention
from .heads import pool_hidden_state
from .language import CausalLMModel, CausalLMOutput


class VisionEncoderAdapter(nn.Module):
    """Project pre-computed vision patch features into the model hidden size.

    This is an interface skeleton: it assumes a vision encoder has already
    produced patch-level features and only learns the linear projection into
    the language model's representation space. It does not load or reproduce
    real ViT weights.

    Args:
        vision_dim: Feature dimension produced by the (external) vision
            encoder.
        hidden_size: Model hidden dimension to project into.
        bias: Whether to use a bias in the projection.

    """

    def __init__(self, vision_dim: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        if vision_dim < 1 or hidden_size < 1:
            raise ValueError("vision_dim and hidden_size must be >= 1")
        self.vision_dim = int(vision_dim)
        self.hidden_size = int(hidden_size)
        self.proj = nn.Linear(vision_dim, hidden_size, bias=bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features into the model hidden size.

        Args:
            vision_features: Tensor of shape
                ``(batch, num_patches, vision_dim)``.

        Returns:
            Tensor of shape ``(batch, num_patches, hidden_size)``.

        """
        if vision_features.dim() != 3:
            raise ValueError("vision_features must be 3D")
        if vision_features.size(-1) != self.vision_dim:
            raise ValueError(
                f"vision_features last dim must be {self.vision_dim}, "
                f"got {vision_features.size(-1)}"
            )
        projected: torch.Tensor = self.proj(vision_features)
        return projected

    def extra_repr(self) -> str:
        """Report the projection dimensions shown in the module ``repr``."""
        return f"vision_dim={self.vision_dim}, hidden_size={self.hidden_size}"


class CrossAttentionFuser(CrossAttention):
    """Late-fusion module: text tokens attend to vision tokens.

    Queries are projected from the text hidden state while keys and values
    are projected from (adapter-projected) vision features, so the number of
    vision tokens may differ from the number of text tokens. The attention
    math is inherited from ``CrossAttention`` (``nn.Linear`` projections plus
    ``BaseAttention.split_head``/``combine_head``/
    ``compute_attention_weights``).

    Args:
        hidden_size: Dimensionality of text and (projected) vision features.
        num_heads: Number of attention heads. Must divide ``hidden_size``.
        dropout: Dropout probability for attention weights.
        bias: Whether to use bias in the linear projections.

    """

    # Late fusion reuses the cross-attention signature (separate text/vision
    # inputs), which intentionally extends the BaseAttention contract.
    def forward(  # type: ignore[override]
        self,
        text_state: torch.Tensor,
        vision_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse vision information into the text hidden state.

        Args:
            text_state: Tensor of shape ``(batch, text_len, hidden_size)``.
            vision_state: Tensor of shape ``(batch, num_patches,
                hidden_size)``, typically the output of
                ``VisionEncoderAdapter``.
            attention_mask: Optional mask broadcastable against the
                ``(batch, num_heads, text_len, num_patches)`` score tensor,
                e.g. a ``(batch, 1, 1, num_patches)`` mask over valid
                patches. 1/True marks visible patches, 0/False masks them.
            return_attention_weights: Also return the attention weights.

        Returns:
            Fused tensor of shape ``(batch, text_len, hidden_size)``, and
            optionally the weights of shape ``(batch, num_heads, text_len,
            num_patches)``.

        """
        return super().forward(
            text_state,
            vision_state,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
        )


def _validate_image_grid(
    image_grid_thw: torch.Tensor,
    *,
    expected_image_tokens: int | None = None,
) -> int:
    """Validate ``image_grid_thw`` and return the per-example token count.

    Shared by :func:`build_multimodal_position_ids` and the cross-attention
    fusion path of :class:`MultimodalCausalLM` so the two validation sites
    cannot drift apart.
    """
    if image_grid_thw.dim() != 2 or image_grid_thw.size(-1) != 3:
        raise ValueError("image_grid_thw must have shape (batch, 3)")
    if (image_grid_thw < 1).any():
        raise ValueError("all image grid dimensions must be >= 1")
    token_counts = image_grid_thw.prod(dim=-1)
    if not torch.equal(token_counts, token_counts[:1].expand_as(token_counts)):
        raise ValueError("dense batches require equal image token counts")
    image_tokens = int(token_counts[0].item())
    if expected_image_tokens is not None and image_tokens != expected_image_tokens:
        raise ValueError(
            f"image grid produces {image_tokens} tokens, "
            f"but vision_features contains {expected_image_tokens}"
        )
    return image_tokens


def build_multimodal_position_ids(
    image_grid_thw: torch.Tensor,
    text_length: int,
    *,
    expected_image_tokens: int | None = None,
) -> torch.Tensor:
    """Build temporal/height/width ids for ``[vision, text]`` sequences.

    Args:
        image_grid_thw: Integer tensor shaped ``(batch, 3)``. Each row holds
            the temporal, height and width grid sizes after patch merging.
        text_length: Number of text tokens following the vision tokens.
        expected_image_tokens: Optional validation target for ``t * h * w``.

    Returns:
        Position ids shaped ``(3, batch, image_tokens + text_length)``.

    All examples in a dense batch must contain the same number of vision
    tokens. Text positions begin after the largest axis coordinate, matching
    the non-overlapping multimodal position convention used by mRoPE models.

    """
    if image_grid_thw.dim() != 2 or image_grid_thw.size(-1) != 3:
        raise ValueError("image_grid_thw must have shape (batch, 3)")
    if text_length < 0:
        raise ValueError("text_length must be >= 0")
    if (image_grid_thw < 1).any():
        raise ValueError("all image grid dimensions must be >= 1")
    token_counts = image_grid_thw.prod(dim=-1)
    if not torch.equal(token_counts, token_counts[:1].expand_as(token_counts)):
        raise ValueError("dense batches require equal image token counts")
    image_tokens = int(token_counts[0].item())
    if expected_image_tokens is not None and image_tokens != expected_image_tokens:
        raise ValueError(
            f"image grid produces {image_tokens} tokens, "
            f"but vision_features contains {expected_image_tokens}"
        )

    rows: list[torch.Tensor] = []
    for grid in image_grid_thw:
        temporal, height, width = (int(value.item()) for value in grid)
        t_ids = torch.arange(temporal, device=grid.device).view(-1, 1, 1)
        h_ids = torch.arange(height, device=grid.device).view(1, -1, 1)
        w_ids = torch.arange(width, device=grid.device).view(1, 1, -1)
        vision_ids = torch.stack(
            [
                t_ids.expand(temporal, height, width).reshape(-1),
                h_ids.expand(temporal, height, width).reshape(-1),
                w_ids.expand(temporal, height, width).reshape(-1),
            ]
        )
        text_start = max(temporal, height, width)
        text_ids = torch.arange(
            text_start,
            text_start + text_length,
            device=grid.device,
        ).expand(3, -1)
        rows.append(torch.cat([vision_ids, text_ids], dim=-1))
    return torch.stack(rows, dim=1)


@dataclass
class MultimodalCausalLMOutput:
    """Outputs from :class:`MultimodalCausalLM`."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None
    mtp_logits: list[torch.Tensor] | None = None
    alignment_logits: torch.Tensor | None = None


class MultimodalCausalLM(nn.Module):
    """End-to-end text/vision causal model with early or cross fusion.

    The class consumes pre-computed vision patch features rather than pixels.
    In ``early`` mode projected patches are prepended as a bidirectional
    prefix and receive temporal/height/width mRoPE ids. In
    ``cross_attention`` mode text embeddings first attend to projected vision
    features and then enter the causal LM. A cosine alignment head is exposed
    for contrastive or reward-style auxiliary objectives.

    This is a model-level reference architecture, not a pretrained ViT or a
    checkpoint-compatible implementation of a particular multimodal model.
    """

    def __init__(
        self,
        vocab_size: int,
        vision_dim: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        *,
        mrope_section: tuple[int, int, int],
        fusion_mode: str = "early",
        alignment_dim: int | None = None,
        max_seq_len: int = 4096,
        attention_name: str = "gqa",
        attention_kwargs: dict[str, Any] | None = None,
        # Forwarded verbatim to ``CausalLMModel``; the values are the
        # heterogeneous option types of that constructor, so they stay Any
        # (plain object/TypedDict were verified to break mypy here).
        **language_model_kwargs: Any,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"early", "cross_attention"}:
            raise ValueError("fusion_mode must be 'early' or 'cross_attention'")
        self.fusion_mode = fusion_mode
        self.hidden_size = int(hidden_size)
        self.vision_adapter = VisionEncoderAdapter(vision_dim, hidden_size)
        self.cross_attention = (
            CrossAttentionFuser(hidden_size, num_heads)
            if fusion_mode == "cross_attention"
            else None
        )
        self.language_model = CausalLMModel(
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
            max_seq_len=max_seq_len,
            attention_name=attention_name,
            attention_kwargs=attention_kwargs,
            positional="mrope",
            positional_kwargs={"mrope_section": mrope_section},
            **language_model_kwargs,
        )
        projection_dim = int(alignment_dim or hidden_size)
        self.text_alignment = nn.Linear(hidden_size, projection_dim, bias=False)
        self.vision_alignment = nn.Linear(hidden_size, projection_dim, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        vision_features: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        vision_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_attention_weights: bool = False,
        return_mtp: bool = False,
    ) -> MultimodalCausalLMOutput:
        """Fuse vision patches with text tokens and run the causal LM."""
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, text_len)")
        batch_size, text_length = input_ids.shape
        vision_state = self.vision_adapter(vision_features)
        if vision_state.size(0) != batch_size:
            raise ValueError("vision and text batch sizes must match")
        vision_length = vision_state.size(1)
        if self.fusion_mode == "cross_attention":
            # Early fusion re-validates the grid while building position ids;
            # the cross-attention path never builds them, so validate the
            # token count here. This mirrors the checks in
            # ``build_multimodal_position_ids`` without materializing the
            # (3, batch, tokens) id tensor that would be discarded.
            if image_grid_thw.dim() != 2 or image_grid_thw.size(-1) != 3:
                raise ValueError("image_grid_thw must have shape (batch, 3)")
            if (image_grid_thw < 1).any():
                raise ValueError("all image grid dimensions must be >= 1")
            token_counts = image_grid_thw.prod(dim=-1)
            if not torch.equal(token_counts, token_counts[:1].expand_as(token_counts)):
                raise ValueError("dense batches require equal image token counts")
            if int(token_counts[0].item()) != vision_length:
                raise ValueError(
                    f"image grid produces {int(token_counts[0].item())} tokens, "
                    f"but vision_features contains {vision_length}"
                )
        text_state = self.language_model.embed_tokens(input_ids)

        if self.fusion_mode == "early":
            output, text_offset = self._early_fusion(
                text_state,
                vision_state,
                image_grid_thw,
                attention_mask,
                vision_attention_mask,
                labels,
                return_attention_weights,
                return_mtp,
            )
        else:
            output, text_offset = self._cross_attention_fusion(
                text_state,
                vision_state,
                attention_mask,
                vision_attention_mask,
                labels,
                return_attention_weights,
                return_mtp,
            )

        # pool_hidden_state has a mask-free fast path; forward None instead of
        # an all-ones mask so alignment pooling skips the weights multiply.
        text_summary = pool_hidden_state(
            text_state,
            (
                None
                if attention_mask is None
                else self._normalize_mask(
                    attention_mask, batch_size, text_length, text_state.device
                )
            ),
            pooling="mean",
        )
        vision_summary = pool_hidden_state(
            vision_state,
            (
                None
                if vision_attention_mask is None
                else self._normalize_mask(
                    vision_attention_mask,
                    batch_size,
                    vision_state.size(1),
                    vision_state.device,
                )
            ),
            pooling="mean",
        )
        alignment_logits = F.cosine_similarity(
            self.text_alignment(text_summary),
            self.vision_alignment(vision_summary),
            dim=-1,
        )
        return MultimodalCausalLMOutput(
            logits=output.logits[:, text_offset:],
            loss=output.loss,
            attention_weights=output.attention_weights,
            mtp_logits=(
                None
                if output.mtp_logits is None
                else [logits[:, text_offset:] for logits in output.mtp_logits]
            ),
            alignment_logits=alignment_logits,
        )

    def _early_fusion(
        self,
        text_state: torch.Tensor,
        vision_state: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor | None,
        vision_attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        return_attention_weights: bool,
        return_mtp: bool,
    ) -> tuple[CausalLMOutput, int]:
        """Prepend vision patches as a bidirectional causal-LM prefix."""
        batch_size, text_length, _ = text_state.shape
        vision_length = vision_state.size(1)
        embeddings = torch.cat([vision_state, text_state], dim=1)
        text_mask = self._normalize_mask(
            attention_mask, batch_size, text_length, embeddings.device
        )
        vision_mask = self._normalize_mask(
            vision_attention_mask,
            batch_size,
            vision_length,
            embeddings.device,
        )
        combined_mask = torch.cat([vision_mask, text_mask], dim=-1)
        position_ids = build_multimodal_position_ids(
            image_grid_thw.to(embeddings.device),
            text_length,
            expected_image_tokens=vision_length,
        )
        combined_labels: torch.Tensor | None = None
        if labels is not None:
            if labels.shape != (batch_size, text_length):
                raise ValueError("labels must match input_ids")
            ignored = labels.new_full((batch_size, vision_length), -100)
            combined_labels = torch.cat([ignored, labels], dim=-1)
        output = self.language_model(
            inputs_embeds=embeddings,
            attention_mask=combined_mask,
            prefix_len=vision_length,
            position_ids=position_ids,
            labels=combined_labels,
            return_dict=True,
            return_attention_weights=return_attention_weights,
            return_mtp=return_mtp,
        )
        if not isinstance(output, CausalLMOutput):
            raise TypeError("language model must return CausalLMOutput")
        return output, vision_length

    def _cross_attention_fusion(
        self,
        text_state: torch.Tensor,
        vision_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
        vision_attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        return_attention_weights: bool,
        return_mtp: bool,
    ) -> tuple[CausalLMOutput, int]:
        """Fuse projected patches into text through cross attention."""
        if self.cross_attention is None:
            raise RuntimeError("cross-attention fuser is not initialized")
        batch_size, text_length, _ = text_state.shape
        vision_mask = self._normalize_mask(
            vision_attention_mask,
            batch_size,
            vision_state.size(1),
            text_state.device,
        )
        fused = text_state + self.cross_attention(
            text_state,
            vision_state,
            attention_mask=vision_mask[:, None, None],
        )
        positions = torch.arange(text_length, device=text_state.device)
        position_ids = positions.expand(3, batch_size, -1)
        output = self.language_model(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True,
            return_attention_weights=return_attention_weights,
            return_mtp=return_mtp,
        )
        if not isinstance(output, CausalLMOutput):
            raise TypeError("language model must return CausalLMOutput")
        return output, 0

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a validated two-dimensional keep mask."""
        if mask is None:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (batch_size, seq_len):
            raise ValueError(f"mask must have shape {(batch_size, seq_len)}")
        return mask.to(device=device, dtype=torch.bool)
