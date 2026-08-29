"""Educational RWKV implementation (RWKV-4 time-mix and channel-mix blocks).

RWKV ("RWKV: Reinventing RNNs for the Transformer Era", Peng et al. 2023,
arXiv:2305.13048) replaces quadratic attention with a per-channel
exponential-decay recurrence (the WKV computation) plus token-shifted linear
mixing, so a layer runs in linear time with a constant-size state. The
formulas match the RWKV-4 reference in transformers'
``models/rwkv/modeling_rwkv.py`` (``RwkvSelfAttention`` and
``RwkvFeedForward``). Both a step-by-step recurrent scan and a naive
quadratic parallel form of the WKV computation are provided; they evaluate
the same recurrence.

Teaching simplifications: the multi-head WKV and extra state bookkeeping of
RWKV-5/6 are omitted, there is no custom CUDA kernel, the state is kept in
the activation dtype instead of float32, the parameter initialization uses
the layer-0 variant of the official schedule, and the ``rescale_every``
inference trick is not implemented.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base_attention import validate_attention_inputs


class RWKVTimeMix(nn.Module):
    """RWKV-4 time-mixing block (the attention replacement).

    Receptance, key and value are per-token linear interpolations of the
    current and the previous token (the "token shift"), followed by linear
    projections. The WKV recurrence then computes an exponentially decaying
    weighted average of past values, with logits given by the keys plus a
    per-channel "time-first" bonus for the current token:

    ``wkv_t = (sum_{i<t} exp(k_i - (t-1-i) * d) v_i + exp(u + k_t) v_t)``
    ``        / (sum_{i<t} exp(k_i - (t-1-i) * d) + exp(u + k_t))``

    where ``d = exp(time_decay)`` is the per-channel decay rate and
    ``u = time_first`` the bonus.

    Args:
        hidden_size: Dimensionality of input and output features.
        bias: Whether the linear projections use biases (RWKV-4 uses none).

    """

    def __init__(self, hidden_size: int, bias: bool = False) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        self.hidden_size = int(hidden_size)

        self.time_decay = nn.Parameter(torch.empty(hidden_size))
        self.time_first = nn.Parameter(torch.empty(hidden_size))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.output = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Apply the RWKV-4 layer-0 initialization schedule.

        Matches transformers' ``RwkvPreTrainedModel._init_weights`` with
        ``layer_id == 0``: per-channel log-decay speeds span -5 to 3 with
        exponent 0.7, the time-first bonus starts at ``log(0.3)`` plus a
        zigzag, and the mixing coefficients ramp linearly across channels.
        Projection weights are orthogonal with zero biases, as in the
        official code.
        """
        hidden_size = self.hidden_size
        position = torch.arange(hidden_size, dtype=torch.float32)
        decay_speed = -5.0 + 8.0 * (position / max(hidden_size - 1, 1)) ** 0.7
        zigzag = ((position + 1) % 3 - 1) * 0.5
        ramp = (position / hidden_size).view(1, 1, hidden_size)
        with torch.no_grad():
            self.time_decay.copy_(decay_speed)
            self.time_first.copy_(math.log(0.3) + zigzag)
            self.time_mix_key.copy_(ramp)
            self.time_mix_value.copy_(ramp)
            self.time_mix_receptance.copy_(ramp**0.5)
        for projection in (self.key, self.value, self.receptance, self.output):
            nn.init.orthogonal_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        scan: str = "recurrent",
    ) -> torch.Tensor:
        """Run time-mixing over ``hidden_state`` of shape ``(batch, seq, hidden)``.

        Args:
            hidden_state: Input tensor of shape (batch_size, seq_len,
                hidden_size).
            key_padding_mask: Optional ``(batch, seq_len)`` boolean mask,
                True marks valid tokens. Masked tokens are excluded from the
                WKV state entirely.
            scan: ``"recurrent"`` or the numerically equivalent naive
                quadratic ``"parallel"`` form.

        """
        shifted = F.pad(hidden_state, (0, 0, 1, -1))
        mixed_key = hidden_state * self.time_mix_key + shifted * (1 - self.time_mix_key)
        mixed_value = hidden_state * self.time_mix_value + shifted * (
            1 - self.time_mix_value
        )
        mixed_receptance = hidden_state * self.time_mix_receptance + shifted * (
            1 - self.time_mix_receptance
        )

        key = self.key(mixed_key)
        value = self.value(mixed_value)
        receptance = torch.sigmoid(self.receptance(mixed_receptance))
        if key_padding_mask is not None:
            # A -inf key contributes exactly zero to the WKV numerator and
            # denominator, so padded tokens never enter the state.
            key = key.masked_fill(~key_padding_mask.unsqueeze(-1), float("-inf"))
        if scan == "parallel":
            wkv = self._wkv_parallel(key, value)
        else:
            wkv = self._wkv_recurrent(key, value)
        output: torch.Tensor = self.output(receptance * wkv)
        return output

    def _wkv_recurrent(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Step-by-step WKV scan with the running (num, den, max) state.

        Mirrors ``rwkv_linear_attention_cpu`` in transformers'
        ``modeling_rwkv.py``: the state tracks the exponentially decayed
        sums rescaled by ``exp(-max_state)`` so the exponentials never
        overflow. A position contributes nothing when its denominator is
        exactly zero (its whole prefix is padded); its output is defined as
        zero instead of NaN.
        """
        batch_size, seq_len, channels = key.shape
        if seq_len == 0:
            return value
        time_decay = -torch.exp(self.time_decay)
        num_state = key.new_zeros(batch_size, channels)
        den_state = key.new_zeros(batch_size, channels)
        max_state = key.new_full((batch_size, channels), -1e38)
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            key_t = key[:, step]
            value_t = value[:, step]

            # WKV output at time t, rescaled by the running maximum.
            max_for_output = torch.maximum(max_state, key_t + self.time_first)
            e1 = torch.exp(max_state - max_for_output)
            e2 = torch.exp(key_t + self.time_first - max_for_output)
            numerator = e1 * num_state + e2 * value_t
            denominator = e1 * den_state + e2
            nonzero = denominator != 0
            safe_denominator = torch.where(
                nonzero, denominator, torch.ones_like(denominator)
            )
            outputs.append(
                torch.where(
                    nonzero,
                    numerator / safe_denominator,
                    torch.zeros_like(numerator),
                )
            )

            # Decay the state and absorb the current token for the next step.
            max_for_state = torch.maximum(max_state + time_decay, key_t)
            e1 = torch.exp(max_state + time_decay - max_for_state)
            e2 = torch.exp(key_t - max_for_state)
            num_state = e1 * num_state + e2 * value_t
            den_state = e1 * den_state + e2
            max_state = max_for_state
        return torch.stack(outputs, dim=1)

    def _wkv_parallel(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Naive quadratic WKV form, materializing all pairwise logits.

        Computes ``wkv_t`` directly from the defining formula: position
        ``i < t`` contributes ``exp(k_i - (t-1-i) * d)`` and position ``t``
        contributes ``exp(u + k_t)``. Rows are rescaled by their maximum
        logit before exponentiating, which keeps the result exact while
        avoiding overflow. Padded keys already carry ``-inf`` logits and
        therefore contribute zero.
        """
        _, seq_len, _ = key.shape
        if seq_len == 0:
            return value
        decay = torch.exp(self.time_decay)  # per-channel rate, positive
        position = torch.arange(seq_len, device=key.device)
        row = position[:, None]  # query index t
        col = position[None, :]  # key index i

        # History logits: k_i - (t - 1 - i) * d, valid only for i < t.
        distance = (row - col - 1).clamp(min=0)
        exponent = key.unsqueeze(1) - distance[..., None] * decay
        # Diagonal logits use the time-first bonus instead of the decay.
        is_diagonal = (row == col)[None, :, :, None]
        diagonal = (key + self.time_first).unsqueeze(2)
        exponent = torch.where(is_diagonal, diagonal, exponent)
        # Future positions (i > t) never contribute.
        is_future = (col > row)[None, :, :, None]
        exponent = exponent.masked_fill(is_future, float("-inf"))

        row_max = exponent.amax(dim=2, keepdim=True)
        # Fully masked rows have row_max == -inf; substituting 0 makes their
        # weights exactly zero instead of NaN.
        row_max = torch.where(
            torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
        )
        weights = torch.exp(exponent - row_max)
        numerator = torch.einsum("btic,bic->btc", weights, value)
        denominator = weights.sum(dim=2)
        nonzero = denominator != 0
        safe_denominator = torch.where(
            nonzero, denominator, torch.ones_like(denominator)
        )
        return torch.where(
            nonzero, numerator / safe_denominator, torch.zeros_like(numerator)
        )

    def extra_repr(self) -> str:
        """Show the hidden size in ``repr(self)``."""
        return f"hidden_size={self.hidden_size}"


class RWKVChannelMix(nn.Module):
    """RWKV-4 channel-mixing block (the feed-forward replacement).

    Token-shifted key/receptance mixing, a squared-ReLU key projection and a
    sigmoid receptance gate: ``sigmoid(W_r r) * (W_v (relu(W_k k))^2)``.

    Args:
        hidden_size: Dimensionality of input and output features.
        intermediate_size: Hidden dimension of the key projection. Defaults
            to ``4 * hidden_size``.
        bias: Whether the linear projections use biases (RWKV-4 uses none).

    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        if intermediate_size is not None and intermediate_size < 1:
            raise ValueError(f"intermediate_size must be >= 1, got {intermediate_size}")
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size or 4 * hidden_size)

        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, self.intermediate_size, bias=bias)
        self.value = nn.Linear(self.intermediate_size, hidden_size, bias=bias)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Apply the RWKV-4 layer-0 mixing schedule and orthogonal weights."""
        hidden_size = self.hidden_size
        ramp = (torch.arange(hidden_size, dtype=torch.float32) / hidden_size).view(
            1, 1, hidden_size
        )
        with torch.no_grad():
            self.time_mix_key.copy_(ramp)
            self.time_mix_receptance.copy_(ramp)
        for projection in (self.key, self.value, self.receptance):
            nn.init.orthogonal_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Run channel-mixing over ``hidden_state``, ``(batch, seq, hidden)``."""
        shifted = F.pad(hidden_state, (0, 0, 1, -1))
        mixed_key = hidden_state * self.time_mix_key + shifted * (1 - self.time_mix_key)
        mixed_receptance = hidden_state * self.time_mix_receptance + shifted * (
            1 - self.time_mix_receptance
        )
        key = torch.square(torch.relu(self.key(mixed_key)))
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(mixed_receptance))
        output: torch.Tensor = receptance * value
        return output

    def extra_repr(self) -> str:
        """Show the hidden and intermediate sizes in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class RWKVLayer(nn.Module):
    """Full RWKV-4 block: pre-LN time-mix and channel-mix with residuals.

    Mirrors the structure of transformers' ``RwkvBlock`` (without the
    layer-0-only ``pre_ln``): ``x + time_mix(ln1(x))`` followed by
    ``x + channel_mix(ln2(x))``. Padding follows the same 1/0 mask
    convention as the other attention modules: masked tokens are zeroed
    before the token shift so their contents never leak into valid
    positions, and they are excluded from the WKV state.

    Args:
        hidden_size: Dimensionality of input and output features.
        intermediate_size: Hidden dimension of the channel-mix key
            projection. Defaults to ``4 * hidden_size``.
        bias: Whether linear projections use biases (RWKV-4 uses none).
        layer_norm_eps: Epsilon of the two pre-LayerNorms.

    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        bias: bool = False,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
        self.hidden_size = int(hidden_size)

        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.time_mix = RWKVTimeMix(hidden_size, bias=bias)
        self.channel_mix = RWKVChannelMix(hidden_size, intermediate_size, bias=bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scan: str = "recurrent",
    ) -> torch.Tensor:
        """Run the RWKV block over ``hidden_state``.

        Args:
            hidden_state: Input tensor of shape (batch_size, seq_len,
                hidden_size).
            attention_mask: Optional mask in the 1/0 (or bool) convention of
                the other attention modules; 3D or 4D, reduced to its
                key-padding component since the recurrence is already causal.
            scan: ``"recurrent"`` or the numerically equivalent naive
                quadratic ``"parallel"`` WKV form.

        """
        if hidden_state.dim() != 3 or hidden_state.size(-1) != self.hidden_size:
            raise ValueError(
                f"hidden_state must have shape (batch, seq, {self.hidden_size})"
            )
        if scan not in {"recurrent", "parallel"}:
            raise ValueError(f"scan must be 'recurrent' or 'parallel', got {scan!r}")
        # RWKV-4 is single-headed; the shared validator only checks shapes
        # and the 1/0 mask convention.
        validate_attention_inputs(hidden_state, attention_mask, num_heads=1)
        batch_size, seq_len, _ = hidden_state.size()
        if seq_len == 0:
            return hidden_state.new_zeros(batch_size, 0, self.hidden_size)

        key_padding_mask = self._key_padding_mask(attention_mask, batch_size)
        if key_padding_mask is not None:
            # Zero padded tokens up front so the token shift never carries
            # their contents into valid positions.
            hidden_state = hidden_state * key_padding_mask.unsqueeze(-1).to(
                hidden_state.dtype
            )
        hidden = hidden_state + self.time_mix(
            self.ln1(hidden_state), key_padding_mask, scan
        )
        output: torch.Tensor = hidden + self.channel_mix(self.ln2(hidden))
        return output

    @staticmethod
    def _key_padding_mask(
        attention_mask: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        """Convert a BaseAttention-style mask to a ``(batch, seq)`` mask."""
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
        return attention_mask.squeeze(1).bool()

    def extra_repr(self) -> str:
        """Show the hidden and intermediate sizes in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.channel_mix.intermediate_size}"
        )
