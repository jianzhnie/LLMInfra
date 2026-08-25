"""DFlash: block-diffusion speculative decoding (arXiv:2602.06036).

DFlash replaces the autoregressive draft model with a lightweight block
diffusion model: the whole draft block is denoised in a single parallel
forward pass, conditioned on hidden features extracted from the target model
and injected into every draft layer's key/value projections.

Simplifications versus the paper, for teaching purposes:

- Block positions use a learned absolute embedding instead of RoPE.
- ``dflash_loss`` samples one anchor block per sequence per call; the paper
  concatenates several blocks per sequence with a sparse block mask.
- Drafting is stateless across decoding cycles (no persistent KV cache
  reuse); callers pass fresh ``target_features`` for each block.
- Extracting/fusing multi-layer target hidden states is the caller's job;
  both APIs here consume the fused ``target_features`` directly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from llminfra.layers.feed_forward import SwiGLUFFN
from llminfra.layers.normalization import RMSNorm


class DraftLayer(nn.Module):
    """Pre-norm transformer layer with target features injected as KV entries."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        feature_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.norm1 = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Target context features enter as extra key/value pairs prepended to
        # every layer, so the conditioning signal never dilutes with depth.
        self.k_context = nn.Linear(feature_size, hidden_size, bias=False)
        self.v_context = nn.Linear(feature_size, hidden_size, bias=False)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend bidirectionally over the block and the injected context."""
        normalized = self.norm1(x)
        query = self._split_heads(self.q_proj(normalized))
        key = torch.cat(
            [
                self._split_heads(self.k_context(context)),
                self._split_heads(self.k_proj(normalized)),
            ],
            dim=2,
        )
        value = torch.cat(
            [
                self._split_heads(self.v_context(context)),
                self._split_heads(self.v_proj(normalized)),
            ],
            dim=2,
        )
        attn_mask = None
        if context_mask is not None:
            batch, seq = x.size(0), x.size(1)
            ctx_len = context.size(1)
            visible_context = context_mask[:, None, None, :].expand(
                batch, 1, seq, ctx_len
            )
            visible_block = torch.ones(
                batch, 1, seq, seq, dtype=torch.bool, device=x.device
            )
            attn_mask = torch.cat([visible_context, visible_block], dim=-1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        )
        attended = attended.transpose(1, 2).reshape(x.shape)
        x = x + self.o_proj(attended)
        # nn.Module.__call__ returns Any in torch's stubs; pin the type here.
        output: torch.Tensor = x + self.ffn(self.norm2(x))
        return output


class BlockDiffusionDrafter(nn.Module):
    """Lightweight block-diffusion draft model for DFlash.

    The input block has ``block_size`` positions: position 0 is the clean
    anchor token (produced by the target model) and the remaining positions
    hold ``mask_token_id``. Each masked position predicts its own token in
    parallel, so all ``block_size - 1`` draft tokens come from one forward
    pass. Fused target-model features are injected into every layer as extra
    key/value pairs.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Draft model hidden dimension.
        num_layers: Number of draft transformer layers.
        num_heads: Attention heads per draft layer.
        block_size: Tokens per block including the clean anchor; the number
            of drafted tokens per pass is ``block_size - 1``.
        mask_token_id: Token id marking positions to predict.
        target_feature_size: Dimension of the fused target features;
            defaults to ``hidden_size``.
        intermediate_size: FFN width; defaults to ``4 * hidden_size``.
        token_embedding: Optional embedding shared with the target model
            (the paper shares and freezes it); created if not given.
        lm_head: Optional LM head shared with the target model; created if
            not given.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 5,
        num_heads: int = 4,
        block_size: int = 16,
        mask_token_id: int = 0,
        target_feature_size: int | None = None,
        intermediate_size: int | None = None,
        token_embedding: nn.Embedding | None = None,
        lm_head: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if block_size < 2:
            raise ValueError("block_size must be >= 2 (anchor + one masked slot)")
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.target_feature_size = int(target_feature_size or hidden_size)
        self.token_embedding = token_embedding or nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(block_size, hidden_size)
        self.layers = nn.ModuleList(
            DraftLayer(
                hidden_size,
                num_heads,
                intermediate_size or 4 * hidden_size,
                self.target_feature_size,
            )
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = lm_head or nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_features: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denoise a masked block in one parallel pass.

        Args:
            input_ids: Block token ids of shape ``(batch, block_size)``;
                position 0 is the clean anchor, the rest are mask tokens.
            target_features: Fused target-model context of shape
                ``(batch, context_len, target_feature_size)``.
            context_mask: Optional boolean keep-mask of shape
                ``(batch, context_len)``; masked-out entries are excluded
                from every layer's injected KV (used in training to prevent
                leakage from features at positions after the anchor).

        Returns:
            Logits of shape ``(batch, block_size, vocab_size)``. Slot ``j``
            predicts the token at block position ``j``; draft tokens come
            from the masked slots ``logits[:, 1:]``.

        """
        if input_ids.dim() != 2 or input_ids.size(1) != self.block_size:
            raise ValueError(f"input_ids must have shape (batch, {self.block_size})")
        if (
            target_features.dim() != 3
            or target_features.size(0) != input_ids.size(0)
            or target_features.size(2) != self.target_feature_size
        ):
            raise ValueError(
                "target_features must have shape (batch, context_len, "
                f"{self.target_feature_size}) matching the input_ids batch size"
            )
        if context_mask is not None and context_mask.shape != target_features.shape[:2]:
            raise ValueError(
                "context_mask must have shape (batch, context_len) matching "
                "target_features"
            )
        positions = torch.arange(self.block_size, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, target_features, context_mask)
        logits: torch.Tensor = self.lm_head(self.final_norm(x))
        return logits

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"num_layers={len(self.layers)}, block_size={self.block_size}, "
            f"mask_token_id={self.mask_token_id}"
        )


class DFlashDecoder(nn.Module):
    """Speculative decoding with a block-diffusion drafter.

    Each call drafts ``block_size - 1`` tokens in a single parallel drafter
    pass and verifies them with the target model, row by row: greedy
    acceptance at ``temperature == 0``, rejection sampling otherwise
    (``min(1, p_target / p_draft)``, rejection resampled from
    ``norm(max(0, p_target - p_draft))``). Both modes emit exact target-model
    tokens, so decoding is lossless.

    Args:
        drafter: The block-diffusion draft model.
        target_model: Callable mapping ``(batch, seq)`` ids to logits.
        temperature: Sampling temperature; 0 selects argmax.
        append_bonus_token: When True, append one extra token sampled from
            the target logits after a fully accepted block.
        pad_token_id: Padding for rows that accepted fewer tokens.

    """

    def __init__(
        self,
        drafter: BlockDiffusionDrafter,
        target_model: Callable[[torch.Tensor], torch.Tensor],
        *,
        temperature: float = 0.0,
        append_bonus_token: bool = True,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.drafter = drafter
        self.target_model = target_model
        self.temperature = float(temperature)
        self.append_bonus_token = bool(append_bonus_token)
        self.pad_token_id = int(pad_token_id)
        self.last_num_drafted = 0
        self.last_num_accepted = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        target_features: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draft one block in parallel and verify it against the target model.

        Args:
            input_ids: Prefix token ids of shape ``(batch, seq)``; the final
                token anchors the draft block (in a decoding loop it is the
                bonus token from the previous verification cycle).
            target_features: Fused target-model features for the prefix,
                shape ``(batch, seq, target_feature_size)``.
            context_mask: Optional boolean keep-mask over the prefix.

        Returns:
            ``input_ids`` concatenated with the emitted tokens, right-padded
            to the longest row with ``pad_token_id``. Each row emits between
            1 (immediate rejection) and ``block_size`` (all drafts accepted
            plus the bonus token) new tokens.

        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq)")
        if input_ids.size(1) < 1:
            raise ValueError("input sequence must contain at least one token")
        if target_features.dim() != 3 or target_features.shape[:2] != input_ids.shape:
            raise ValueError(
                "target_features must have shape (batch, seq, feature_size) "
                "aligned with input_ids"
            )
        batch = input_ids.size(0)
        num_drafted = self.drafter.block_size - 1
        anchor = input_ids[:, -1:]
        masked = input_ids.new_full((batch, num_drafted), self.drafter.mask_token_id)
        block = torch.cat([anchor, masked], dim=-1)
        # One parallel pass drafts the whole block; slot k of draft_logits
        # predicts draft token k.
        draft_logits = self.drafter(block, target_features, context_mask)[:, 1:]
        drafted = self._sample(draft_logits)

        target_logits = self.target_model(torch.cat([input_ids, drafted], dim=-1))
        rows: list[torch.Tensor] = []
        total_accepted = 0
        for row in range(batch):
            decoded, num_accepted = self._verify_row(
                input_ids[row : row + 1],
                drafted[row : row + 1],
                draft_logits[row : row + 1],
                target_logits[row : row + 1],
            )
            rows.append(decoded[0])
            total_accepted += num_accepted
        max_length = max(row.numel() for row in rows)
        output = input_ids.new_full((batch, max_length), self.pad_token_id)
        for row, decoded in enumerate(rows):
            output[row, : decoded.numel()] = decoded
        self.last_num_drafted = batch * num_drafted
        self.last_num_accepted = total_accepted
        return output

    def _verify_row(
        self,
        input_ids: torch.Tensor,
        drafted: torch.Tensor,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Verify one row's draft tokens against the target distribution."""
        seq = input_ids.size(1)
        accepted: list[torch.Tensor] = []
        num_accepted = 0
        for step in range(drafted.size(1)):
            target_step_logits = target_logits[:, seq - 1 + step]
            draft_token = drafted[:, step : step + 1]
            if self.temperature == 0.0:
                target_token = torch.argmax(target_step_logits, dim=-1)[:, None]
                if torch.equal(draft_token, target_token):
                    accepted.append(draft_token)
                    num_accepted += 1
                    continue
                accepted.append(target_token)
                break
            if bool(
                self._accept_draft(
                    draft_logits[:, step], target_step_logits, draft_token
                ).item()
            ):
                accepted.append(draft_token)
                num_accepted += 1
            else:
                accepted.append(
                    self._sample_residual(draft_logits[:, step], target_step_logits)
                )
                break
        else:
            if self.append_bonus_token:
                accepted.append(self._sample(target_logits[:, -1])[:, None])
        return torch.cat([input_ids, *accepted], dim=-1), num_accepted

    def _accept_draft(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_token: torch.Tensor,
    ) -> torch.Tensor:
        """Accept a draft token with probability ``min(1, p_target/p_draft)``."""
        p_draft = torch.softmax(draft_logits / self.temperature, dim=-1)
        p_target = torch.softmax(target_logits / self.temperature, dim=-1)
        ratio = p_target.gather(-1, draft_token) / p_draft.gather(
            -1, draft_token
        ).clamp_min(1e-12)
        return torch.rand_like(ratio) < ratio.clamp(max=1.0)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        flat = probabilities.reshape(-1, probabilities.size(-1))
        return torch.multinomial(flat, 1).reshape(probabilities.shape[:-1])

    def _sample_residual(
        self, draft_logits: torch.Tensor, target_logits: torch.Tensor
    ) -> torch.Tensor:
        """Sample the correction distribution ``norm(max(0, p_target - p_draft))``."""
        p_draft = torch.softmax(draft_logits / self.temperature, dim=-1)
        p_target = torch.softmax(target_logits / self.temperature, dim=-1)
        residual = (p_target - p_draft).clamp_min(0.0)
        total = residual.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            total > 1e-12,
            residual / total.clamp_min(1e-12),
            p_target,
        )
        return torch.multinomial(probabilities, 1)

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return (
            f"block_size={self.drafter.block_size}, "
            f"temperature={self.temperature}, "
            f"append_bonus_token={self.append_bonus_token}"
        )


def dflash_loss(
    drafter: BlockDiffusionDrafter,
    input_ids: torch.Tensor,
    target_features: torch.Tensor,
    *,
    weight_decay: float | None = None,
) -> torch.Tensor:
    """Anchor-sampled block diffusion training loss.

    One random anchor position is sampled per sequence; the anchor token
    stays clean, the following ``block_size - 1`` positions are masked, and
    the drafter is trained to predict them in parallel. This matches the
    inference-time contract, where drafting always conditions on a clean
    target-produced token. Context features are masked beyond the anchor so
    no future information leaks into the draft. Earlier masked positions get
    exponentially larger weight, ``w_k = exp(-(k - 1) / decay)`` for block
    position ``k``, because an early error invalidates the rest of the block
    under speculative verification.

    Args:
        drafter: The block-diffusion draft model.
        input_ids: Clean token sequences of shape ``(batch, seq)`` with
            ``seq > block_size``.
        target_features: Fused target-model features, shape
            ``(batch, seq, target_feature_size)``.
        weight_decay: Decay constant ``decay`` of the positional weighting;
            defaults to ``block_size``.

    Returns:
        Scalar loss (weighted mean cross-entropy over masked positions).

    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape (batch, seq)")
    batch, seq = input_ids.shape
    block_size = drafter.block_size
    if seq <= block_size:
        raise ValueError(
            f"sequence length ({seq}) must exceed block_size ({block_size})"
        )
    if target_features.shape[:2] != input_ids.shape:
        raise ValueError(
            "target_features must have shape (batch, seq, feature_size) "
            "aligned with input_ids"
        )
    decay = float(weight_decay) if weight_decay is not None else float(block_size)
    if decay <= 0:
        raise ValueError("weight_decay must be > 0")

    rows = torch.arange(batch, device=input_ids.device)
    anchor = torch.randint(0, seq - block_size + 1, (batch,), device=input_ids.device)
    block = input_ids.new_full((batch, block_size), drafter.mask_token_id)
    block[:, 0] = input_ids[rows, anchor]
    # Hide context features after the anchor: target hidden states are causal,
    # so features at positions <= anchor carry no future information.
    context_mask = (
        torch.arange(seq, device=input_ids.device)[None, :] <= anchor[:, None]
    )
    logits = drafter(block, target_features, context_mask)
    offsets = anchor[:, None] + torch.arange(1, block_size, device=input_ids.device)
    labels = input_ids.gather(1, offsets)
    weights = torch.exp(
        -torch.arange(block_size - 1, device=input_ids.device, dtype=logits.dtype)
        / decay
    )
    per_token = F.cross_entropy(
        logits[:, 1:].reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view(batch, block_size - 1)
    return (per_token * weights).sum() / (weights.sum() * batch)


__all__ = ["BlockDiffusionDrafter", "DFlashDecoder", "dflash_loss"]
