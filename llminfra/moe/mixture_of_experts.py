"""Educational mixture-of-experts (MoE) modules.

These modules implement the routing and expert computation patterns used by
mainstream MoE models such as Qwen3, DeepSeekMoE, Mixtral, DBRX, Baichuan-M3
and Nemotron-3. They are teaching implementations: expert execution loops
over expert ids instead of using optimized group GEMM kernels.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn import functional as dist_nn


class ExpertFFN(nn.Module):
    """One expert's feed-forward network."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if activation not in {"silu", "relu", "gelu"}:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_name = activation
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the single-path FFN ``w2(act(w1(x)))`` to the input."""
        output: torch.Tensor = self.w2(self._activation(self.w1(x)))
        return output

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "silu":
            return F.silu(x)
        if self.activation_name == "relu":
            return F.relu(x)
        return F.gelu(x)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)


class TopKRouter(nn.Module):
    """Top-k expert router with optional training-time noise.

    Args:
        hidden_size: Input feature dimension.
        num_experts: Number of routed experts.
        top_k: Number of experts selected per token.
        add_noise: Enable the Switch/GShard-style noise used during training.
        noise_epsilon: Scale of the routing noise.
        dropout: Dropout applied to routing weights.
        scoring_func: Scoring applied to the selected experts' logits,
            ``"softmax"`` (default) or ``"sigmoid"``. With ``"sigmoid"`` the
            weights are ``sigmoid(logits)`` renormalized to sum to 1.
        aux_free_balance: Enable a DeepSeek-V3-style auxiliary-loss-free
            balancing bias. A gradient-free ``router_bias`` parameter is
            registered; top-k selection uses ``logits + router_bias`` while
            the routing weights are still computed from the unbiased logits.
            After each training-mode forward the bias is updated by a
            discrete step proportional to the sign of each expert's load
            violation. This is a teaching simplification: the real systems
            update the bias per training step with a tuned schedule.
        balance_update_rate: Step size ``u`` of the bias update.
        routing_strategy: ``"topk"`` (default) or ``"gumbel"``. Gumbel
            routing is a training-time relaxation only; in eval mode the
            router always falls back to deterministic top-k selection.
        gumbel_temperature: Temperature ``tau`` of the gumbel-softmax
            relaxation (gumbel strategy only).
        gumbel_hard: With the gumbel strategy, select experts through a
            straight-through hard mask. The forward weights are the unbiased
            softmax/sigmoid scores of the selected experts (matching the
            top-k path); the gumbel probabilities only provide gradients.

    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        add_noise: bool = False,
        noise_epsilon: float = 1e-2,
        dropout: float = 0.0,
        scoring_func: str = "softmax",
        aux_free_balance: bool = False,
        balance_update_rate: float = 1e-3,
        routing_strategy: str = "topk",
        gumbel_temperature: float = 1.0,
        gumbel_hard: bool = True,
    ) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= num_experts")
        if scoring_func not in {"softmax", "sigmoid"}:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")
        if routing_strategy not in {"topk", "gumbel"}:
            raise ValueError("routing_strategy must be 'topk' or 'gumbel'")
        if gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be > 0")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.add_noise = bool(add_noise)
        self.noise_epsilon = float(noise_epsilon)
        self.scoring_func = scoring_func
        self.aux_free_balance = bool(aux_free_balance)
        self.balance_update_rate = float(balance_update_rate)
        self.routing_strategy = routing_strategy
        self.gumbel_temperature = float(gumbel_temperature)
        self.gumbel_hard = bool(gumbel_hard)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.noise_proj = (
            nn.Linear(hidden_size, num_experts, bias=False) if add_noise else None
        )
        self.dropout = nn.Dropout(dropout)
        if aux_free_balance:
            # Gradient-free on purpose: balanced by discrete updates, not SGD.
            self.router_bias = nn.Parameter(
                torch.zeros(num_experts), requires_grad=False
            )
        else:
            self.register_parameter("router_bias", None)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(routing_weights, expert_indices)``.

        ``routing_weights`` has shape ``(batch, top_k)`` and
        ``expert_indices`` has shape ``(batch, top_k)``. With ``dropout=0``
        the weights sum to 1 along the last dimension (dropout during
        training rescales them, as usual).
        """
        logits = self.router(x)
        if self.training and self.add_noise and self.noise_proj is not None:
            # Switch/GShard-style: noise scaled by a learned per-expert std
            # and the configured epsilon. A constant shift would not change
            # the top-k ranking, so the epsilon must multiply the noise.
            noise = torch.randn_like(logits)
            noise = noise * F.softplus(self.noise_proj(x))
            logits = logits + noise * self.noise_epsilon

        # The balancing bias steers *which* experts are picked, but never
        # enters the weights (DeepSeek-V3 auxiliary-loss-free balancing).
        selection_logits = logits
        if self.router_bias is not None:
            selection_logits = logits + self.router_bias
        if self.training and self.routing_strategy == "gumbel":
            selection_probabilities = F.gumbel_softmax(
                selection_logits,
                tau=self.gumbel_temperature,
                hard=False,
                dim=-1,
            )
            _, indices = torch.topk(selection_probabilities, self.top_k, dim=-1)
            probabilities = selection_probabilities
            if self.router_bias is not None:
                # The bias only steers selection; weights stay unbiased.
                probabilities = F.gumbel_softmax(
                    logits,
                    tau=self.gumbel_temperature,
                    hard=False,
                    dim=-1,
                )
            if self.gumbel_hard:
                hard_mask = torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)
                probabilities = hard_mask + probabilities - probabilities.detach()
            weights = probabilities.gather(-1, indices)
            if self.gumbel_hard:
                # Forward values are the unbiased softmax/sigmoid scores of
                # the selected experts, matching the top-k path (and eval
                # mode); the gumbel probabilities only provide a
                # straight-through gradient.
                scores = self._score(logits.gather(-1, indices))
                weights = scores.detach() + weights - weights.detach()
            else:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            _, indices = torch.topk(selection_logits, self.top_k, dim=-1)
            weights = self._score(logits.gather(-1, indices))
        weights = self.dropout(weights)

        if self.training and self.router_bias is not None:
            self._update_router_bias(indices)
        return weights, indices

    def _score(self, selected_logits: torch.Tensor) -> torch.Tensor:
        """Turn selected experts' logits into normalized routing weights."""
        if self.scoring_func == "softmax":
            return torch.softmax(selected_logits, dim=-1)
        weights = torch.sigmoid(selected_logits)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def _update_router_bias(self, indices: torch.Tensor) -> None:
        """Step the bias by ``u * sign(target_load - load)`` per expert."""
        counts = torch.bincount(indices.reshape(-1), minlength=self.num_experts).to(
            torch.float32
        )
        target = indices.numel() / self.num_experts
        violation = target - counts
        self.router_bias.add_(self.balance_update_rate * torch.sign(violation))

    def routing_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw router logits without training noise."""
        logits: torch.Tensor = self.router(x)
        return logits

    def extra_repr(self) -> str:
        """Describe the router's configuration in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"top_k={self.top_k}, add_noise={self.add_noise}, "
            f"scoring_func={self.scoring_func}, "
            f"aux_free_balance={self.aux_free_balance}, "
            f"routing_strategy={self.routing_strategy}"
        )


class ExpertChoiceRouter(nn.Module):
    """Expert-choice routing: each expert selects its top tokens.

    Instead of each token picking its top-k experts, each expert picks its
    top-``top_tokens`` tokens, which guarantees perfectly balanced expert
    loads by construction (every expert receives the same number of tokens).

    Shape convention: ``forward`` returns ``(weights, token_indices)`` where
    both have shape ``(num_experts, top_tokens)``. ``token_indices[e, j]`` is
    the index (into the flattened token axis of the input) of the j-th token
    chosen by expert ``e``, and ``weights[e, j]`` is that token's routing
    weight for expert ``e``. Weights come from a softmax over the token axis,
    so for each expert the weights over all tokens (not only the chosen ones)
    sum to 1. A token may be chosen by several experts or by none.

    Args:
        hidden_size: Input feature dimension.
        num_experts: Number of routed experts.
        top_tokens: Number of tokens each expert selects.

    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_tokens: int,
    ) -> None:
        super().__init__()
        if top_tokens < 1:
            raise ValueError("top_tokens must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_tokens = int(top_tokens)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weights, token_indices)``, each ``(num_experts, top_tokens)``.

        Args:
            x: Token features of shape ``(num_tokens, hidden_size)`` with
                ``num_tokens >= top_tokens``.

        """
        if x.size(0) < self.top_tokens:
            raise ValueError(
                f"num_tokens ({x.size(0)}) must be >= top_tokens ({self.top_tokens})"
            )
        logits = self.router(x)  # (num_tokens, num_experts)
        expert_scores = logits.transpose(0, 1)  # (num_experts, num_tokens)
        scores = torch.softmax(expert_scores, dim=-1)
        weights, token_indices = torch.topk(scores, self.top_tokens, dim=-1)
        return weights, token_indices

    def extra_repr(self) -> str:
        """Describe the router's configuration in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"top_tokens={self.top_tokens}"
        )


class MixtureOfExperts(nn.Module):
    """Standard top-k routed mixture of experts.

    The module does not add a residual connection; callers should apply the
    transformer residual outside this layer.

    Args:
        expert_dropout: Probability of dropping a token's routed expert
            weight during training. Dropped weights are zeroed and the
            surviving experts of that token are renormalized to sum to 1
            (tokens whose experts were all dropped contribute nothing this
            step). Teaching simplification of expert-level dropout.

    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        top_k: int = 2,
        activation: str = "silu",
        add_router_noise: bool = False,
        bias: bool = True,
        expert_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= expert_dropout < 1.0:
            raise ValueError("expert_dropout must satisfy 0 <= p < 1")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.expert_dropout = float(expert_dropout)
        self.experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size, activation, bias)
            for _ in range(num_experts)
        )
        self.router = TopKRouter(
            hidden_size,
            num_experts,
            top_k=top_k,
            add_noise=add_router_noise,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to experts and return the combined output."""
        flat = x.reshape(-1, self.hidden_size)
        routing_weights, expert_indices = self.router(flat)
        if self.training and self.expert_dropout > 0.0:
            keep = torch.rand_like(routing_weights) >= self.expert_dropout
            routing_weights = routing_weights * keep
            totals = routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights / totals.clamp_min(1e-12)
        output = torch.zeros_like(flat)

        # Flatten the (token, k) assignments so each expert runs a single GEMM
        # over all of its tokens instead of top_k smaller masked slices;
        # index_add_ scatters the weighted outputs back to their tokens.
        flat_indices = expert_indices.reshape(-1)
        flat_weights = routing_weights.reshape(-1)
        token_ids = torch.arange(flat.size(0), device=flat.device).repeat_interleave(
            self.top_k
        )
        for expert_id in range(self.num_experts):
            hit = flat_indices == expert_id
            if not hit.any():
                continue
            selected = token_ids[hit]
            expert_output = self.experts[expert_id](flat[selected])
            output.index_add_(
                0, selected, flat_weights[hit].unsqueeze(-1) * expert_output
            )
        return output.view_as(x)

    def extra_repr(self) -> str:
        """Describe the layer's configuration in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"intermediate_size={self.intermediate_size}, top_k={self.top_k}, "
            f"expert_dropout={self.expert_dropout}"
        )


class DeepSeekMoE(nn.Module):
    """DeepSeek-style MoE with a small set of shared experts.

    The output is ``routed_experts(x) + sum(shared_experts(x))``. A residual
    connection is intentionally left to the transformer block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        intermediate_size: int,
        top_k: int = 6,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_shared_experts < 1:
            raise ValueError("num_shared_experts must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_routed_experts = int(num_routed_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.routed = MixtureOfExperts(
            hidden_size,
            num_routed_experts,
            intermediate_size,
            top_k=top_k,
            activation=activation,
            bias=bias,
        )
        self.shared_experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size, activation, bias)
            for _ in range(num_shared_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return routed output plus shared-expert output."""
        shared: torch.Tensor = self.shared_experts[0](x)
        for expert in self.shared_experts[1:]:
            shared = shared + expert(x)
        routed: torch.Tensor = self.routed(x)
        return routed + shared

    def extra_repr(self) -> str:
        """Describe the layer's configuration in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_routed_experts={self.num_routed_experts}, "
            f"num_shared_experts={self.num_shared_experts}, "
            f"intermediate_size={self.intermediate_size}, top_k={self.top_k}"
        )


class ExpertParallelMoE(nn.Module):
    """Expert-parallel MoE with a pure-PyTorch all-to-all reference path.

    In a distributed process group, token/expert assignments are sent to the
    owning rank with an autograd-aware ``all_to_all_single`` collective,
    evaluated by local experts, and sent back to the source rank for combine.
    Without an initialized process group the module retains a local-owner
    simulation mode, which is useful for inspecting one rank in unit tests.

    This implementation deliberately executes experts in Python loops. It
    demonstrates the communication contract and capacity policy, but a
    production implementation should pack tokens with fused kernels and use
    grouped GEMM for expert execution.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        intermediate_size: int,
        top_k: int = 2,
        world_size: int = 1,
        rank: int = 0,
        activation: str = "silu",
        bias: bool = True,
        capacity_factor: float | None = None,
        process_group: dist.ProcessGroup | None = None,
        use_distributed: bool | None = None,
    ) -> None:
        super().__init__()
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("world_size must be >= 1 and rank < world_size")
        if capacity_factor is not None and capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0")
        distributed_ready = dist.is_available() and dist.is_initialized()
        if use_distributed is True and not distributed_ready:
            raise RuntimeError("use_distributed=True requires an initialized group")
        self.use_distributed = (
            distributed_ready if use_distributed is None else use_distributed
        )
        self.process_group = process_group
        if self.use_distributed:
            actual_world_size = dist.get_world_size(process_group)
            actual_rank = dist.get_rank(process_group)
            if (world_size, rank) != (actual_world_size, actual_rank):
                raise ValueError(
                    "world_size/rank must match the initialized process group"
                )
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.capacity_factor = capacity_factor
        self.local_expert_ids = list(
            range(self.rank, self.num_experts, self.world_size)
        )
        self.experts = nn.ModuleList(
            ExpertFFN(hidden_size, intermediate_size, activation, bias)
            for _ in self.local_expert_ids
        )
        self.router = TopKRouter(
            hidden_size,
            num_experts,
            top_k=top_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route, execute local experts, and combine token outputs."""
        flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.router(flat)
        if self.use_distributed and self.world_size > 1:
            output = self._distributed_dispatch(flat, weights, indices)
        else:
            output = self._local_dispatch(flat, weights, indices)
        return output.view_as(x)

    def _local_dispatch(
        self,
        flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Execute only the experts owned by this rank."""
        output = torch.zeros_like(flat)
        for local_index, expert_id in enumerate(self.local_expert_ids):
            # Sum this expert's routing weight over the top-k slots in one
            # vectorized step instead of a per-slot Python loop.
            token_weights = (weights * (indices == expert_id)).sum(dim=-1)
            mask = token_weights > 0
            if self.capacity_factor is not None and mask.any():
                capacity = max(
                    1,
                    math.ceil(
                        self.capacity_factor
                        * flat.size(0)
                        * self.top_k
                        / self.num_experts
                    ),
                )
                selected = mask.nonzero(as_tuple=False).flatten()
                mask[selected[capacity:]] = False
            if mask.any():
                output[mask] += token_weights[mask].unsqueeze(-1) * self.experts[
                    local_index
                ](flat[mask])
        return output

    def _distributed_dispatch(
        self,
        flat: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch assignments to expert owners and return them to sources."""
        token_ids = torch.arange(flat.size(0), device=flat.device)
        assignment_tokens = token_ids.repeat_interleave(self.top_k)
        expert_ids = indices.reshape(-1)
        assignment_weights = weights.reshape(-1)
        destinations = expert_ids.remainder(self.world_size)
        order = torch.argsort(destinations, stable=True)
        send_counts = torch.bincount(destinations, minlength=self.world_size).to(
            dtype=torch.int64
        )
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(
            recv_counts,
            send_counts,
            group=self.process_group,
        )
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()
        total_received = int(recv_counts.sum().item())

        send_features = flat[assignment_tokens[order]].contiguous()
        received_features = flat.new_empty(total_received, self.hidden_size)
        # torch.distributed.nn.functional collectives are untyped in torch.
        received_features = dist_nn.all_to_all_single(  # type: ignore[no-untyped-call]
            received_features,
            send_features,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group,
        )
        received_weights = dist_nn.all_to_all_single(  # type: ignore[no-untyped-call]
            assignment_weights.new_empty(total_received),
            assignment_weights[order].contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group,
        )
        received_expert_ids = expert_ids.new_empty(total_received)
        dist.all_to_all_single(
            received_expert_ids,
            expert_ids[order].contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.process_group,
        )

        received_output = torch.zeros_like(received_features)
        for local_index, expert_id in enumerate(self.local_expert_ids):
            selected = (
                (received_expert_ids == expert_id).nonzero(as_tuple=False).flatten()
            )
            if self.capacity_factor is not None:
                capacity = max(
                    1,
                    math.ceil(
                        self.capacity_factor
                        * max(1, total_received)
                        / max(1, len(self.local_expert_ids))
                    ),
                )
                selected = selected[:capacity]
            if selected.numel() == 0:
                continue
            expert_output = self.experts[local_index](received_features[selected])
            received_output = received_output.index_add(
                0,
                selected,
                expert_output * received_weights[selected, None],
            )

        returned = dist_nn.all_to_all_single(  # type: ignore[no-untyped-call]
            flat.new_empty(order.numel(), self.hidden_size),
            received_output,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.process_group,
        )
        output = torch.zeros_like(flat)
        return output.index_add(0, assignment_tokens[order], returned)

    def extra_repr(self) -> str:
        """Describe the layer's configuration in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
            f"world_size={self.world_size}, rank={self.rank}, "
            f"local_expert_ids={self.local_expert_ids}, "
            f"use_distributed={self.use_distributed}"
        )


def load_balance_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Auxiliary load-balancing loss for top-k routing.

    Args:
        router_logits: Shape ``(num_tokens, num_experts)``.
        expert_indices: Shape ``(num_tokens, top_k)``.
        num_experts: Number of routed experts.

    """
    if router_logits.size(-1) != num_experts:
        raise ValueError("router_logits last dim must equal num_experts")
    if expert_indices.dim() != 2:
        raise ValueError("expert_indices must have shape (num_tokens, top_k)")
    if router_logits.size(0) == 0:
        # mean over an empty token axis would be NaN; no tokens, no loss.
        return router_logits.new_zeros(())
    # One bincount over the flattened (token, k) indices replaces the
    # per-slot Python loop; integer counts are exact either way.
    counts = torch.bincount(expert_indices.reshape(-1), minlength=num_experts).to(
        torch.float32
    )
    fraction = counts / max(1, expert_indices.numel())
    probabilities = torch.softmax(router_logits, dim=-1).mean(dim=0)
    return num_experts * (fraction * probabilities).sum()


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss: penalizes large-magnitude routing logits.

    Introduced by ST-MoE and used by many production MoE models to keep
    router logits in a numerically stable range. Defined as
    ``mean(logsumexp(logits, dim=-1) ** 2)``.

    Args:
        router_logits: Raw router logits of shape ``(num_tokens, num_experts)``.

    Returns:
        Scalar z-loss tensor.

    """
    log_z = torch.logsumexp(router_logits, dim=-1)
    return (log_z * log_z).mean()
