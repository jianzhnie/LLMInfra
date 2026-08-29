"""Muon and MuonClip optimizers for training LLMs.

This module provides educational PyTorch implementations of:

- **Muon** (MomentUm Orthogonalized by Newton-Schulz), which orthogonalizes
  the momentum of hidden-layer weight matrices via a quintic Newton-Schulz
  iteration before applying the update. See "Muon is Scalable for LLM
  Training" (Moonshot AI, arXiv:2502.16982) and the reference implementation
  at https://github.com/KellerJordan/Muon. Following both sources, 1D
  parameters (biases, norms) and explicitly excluded parameters (embeddings,
  LM head) fall back to an internal AdamW update.

- **MuonClip**, which adds the QK-clip mechanism from the Kimi K2 technical
  report (arXiv:2507.20534, Section 2, "MuonClip"): whenever the maximum
  observed QK logit of an attention head exceeds a threshold ``tau``, the
  ``q_proj``/``k_proj`` weights of that head are rescaled by
  ``sqrt(tau / max_logit)`` so that the logit is bounded by ``tau`` in the
  next forward pass.

The implementations here prioritize readability over kernel efficiency (no
distributed sharding, no fused operations) and are intended for
small-to-medium models and unit tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

__all__ = ["Muon", "MuonClip", "zeropower_via_newtonschulz5"]


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Compute an approximate orthogonalization of ``G`` via Newton-Schulz.

    Runs a quintic Newton-Schulz iteration that pushes the singular values of
    ``G`` towards 1, yielding a semi-orthogonal matrix with the same shape.
    The iteration is performed in bfloat16 for speed, following the reference
    implementation of Keller Jordan (https://github.com/KellerJordan/Muon) and
    "Muon is Scalable for LLM Training" (arXiv:2502.16982).

    The coefficients ``(3.4445, -4.7750, 2.0315)`` are tuned so that the
    iteration converges quickly from spectral norm ``<= 1``; the input is
    therefore normalized by its Frobenius norm first (Frobenius norm upper
    bounds the spectral norm).

    Args:
        G: Matrix (or batch of matrices) of shape ``(..., rows, cols)`` to
            orthogonalize. If ``rows > cols`` the iteration is run on the
            transpose so that the smaller dimension is iterated on.
        steps: Number of Newton-Schulz iterations. 5 steps is the standard
            choice from the reference implementation.

    Returns:
        A tensor with the same shape as ``G`` whose singular values are
        clustered near 1 (approximate semi-orthogonalization), in bfloat16.

    """
    if G.ndim < 2:
        raise ValueError(f"Expected a tensor with ndim >= 2, got ndim={G.ndim}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X: Tensor = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    # Normalize so the spectral norm is <= 1 before iterating.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(Optimizer):
    """Muon optimizer with an internal AdamW fallback for non-matrix params.

    Muon ("Muon is Scalable for LLM Training", arXiv:2502.16982) orthogonalizes
    the (optionally Nesterov) momentum of each matrix-shaped parameter with a
    Newton-Schulz iteration and applies the orthogonalized update scaled by
    ``0.2 * sqrt(max(rows, cols))``, which matches the RMS of the update to
    that of a typical AdamW update (the "RMS matching" convention of both
    Keller Jordan's reference and the Moonshot paper).

    Parameters are routed per parameter group via the ``use_muon`` flag:

    - ``use_muon=True``: Muon update (Newton-Schulz + momentum).
    - ``use_muon=False``: standard decoupled AdamW update.

    If a group does not specify ``use_muon``, it defaults to ``True`` exactly
    when every parameter in the group has ``ndim >= 2``, which is the
    convenient default judgment: hidden-layer matrices go to Muon while 1D
    biases/norms go to AdamW. Embedding and LM-head weights should be
    excluded explicitly (see :meth:`from_named_parameters`), as recommended
    by both Keller Jordan and Moonshot.

    Args:
        params: Iterable of parameters or parameter-group dicts. A group dict
            may set ``use_muon`` (bool) to force the routing; group-level
            values override the constructor defaults for other keys as usual.
        lr: Learning rate for the Muon groups.
        momentum: Momentum coefficient for the Muon groups.
        nesterov: Whether to use Nesterov-style momentum in the Muon groups.
        ns_steps: Number of Newton-Schulz iterations.
        weight_decay: Decoupled weight decay applied by both Muon and AdamW
            updates (i.e. ``p *= 1 - lr_eff * weight_decay``).
        adamw_lr: Learning rate for the AdamW fallback groups. Defaults to
            ``lr`` when ``None``.
        adamw_betas: ``(beta1, beta2)`` for the AdamW fallback groups.
        adamw_eps: Epsilon for the AdamW fallback groups.

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_lr: float | None = None,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if adamw_lr is not None and adamw_lr <= 0.0:
            raise ValueError(f"Invalid AdamW learning rate: {adamw_lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults: dict[str, Any] = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "use_muon": None,
            "adamw_lr": adamw_lr if adamw_lr is not None else lr,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
        }
        super().__init__(params, defaults)
        # Convenient default routing: matrix-only groups use Muon, anything
        # containing 1D parameters falls back to AdamW.
        for group in self.param_groups:
            if group["use_muon"] is None:
                group["use_muon"] = all(p.ndim >= 2 for p in group["params"])

    @classmethod
    def from_named_parameters(
        cls,
        named_parameters: Iterable[tuple[str, Tensor]],
        lr: float = 0.02,
        adamw_lr: float | None = None,
        **kwargs: Any,
    ) -> Muon:
        """Build a Muon optimizer with the standard Muon/AdamW split.

        Parameters whose name suggests an embedding or LM head
        (``"embed"``, ``"lm_head"``, ``"head"``) and 1D parameters
        (biases, norm gains) are routed to the AdamW fallback; every other
        parameter with ``ndim >= 2`` is routed to Muon. This mirrors the
        parameter split recommended by Keller Jordan's reference
        implementation and the Moonshot paper.

        Args:
            named_parameters: ``(name, parameter)`` pairs, e.g.
                ``model.named_parameters()``.
            lr: Learning rate for the Muon group.
            adamw_lr: Learning rate for the AdamW group. Defaults to ``lr``.
            **kwargs: Forwarded to :class:`Muon`.

        Returns:
            A :class:`Muon` instance with two parameter groups.

        """
        muon_params: list[Tensor] = []
        adamw_params: list[Tensor] = []
        for name, param in named_parameters:
            lowered = name.lower()
            excluded = any(key in lowered for key in ("embed", "lm_head", "head"))
            if param.ndim >= 2 and not excluded:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        groups: list[dict[str, Any]] = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True})
        if adamw_params:
            groups.append(
                {
                    "params": adamw_params,
                    "use_muon": False,
                    "lr": adamw_lr if adamw_lr is not None else lr,
                }
            )
        return cls(groups, lr=lr, adamw_lr=adamw_lr, **kwargs)

    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...
    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns
                the loss, as in the standard PyTorch optimizer API.

        Returns:
            The loss returned by ``closure``, or ``None``.

        """
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        """Apply the Muon update to every parameter of ``group``."""
        lr: float = group["lr"]
        momentum: float = group["momentum"]
        nesterov: bool = group["nesterov"]
        ns_steps: int = group["ns_steps"]
        weight_decay: float = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf: Tensor = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            update = grad.add(buf, alpha=momentum) if nesterov else buf
            # Orthogonalize the 2D view (fan-out x rest) of the update.
            update_2d = update.reshape(update.size(0), -1)
            orthogonal = zeropower_via_newtonschulz5(update_2d, ns_steps)
            rows, cols = update_2d.shape
            # RMS matching: scale so the update RMS ~= 0.2 (Keller Jordan /
            # Moonshot convention).
            scale = 0.2 * math.sqrt(max(rows, cols))
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            p.add_(orthogonal.reshape(p.shape).to(p.dtype), alpha=-lr * scale)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        """Apply a decoupled AdamW update to every parameter of ``group``."""
        lr: float = group["adamw_lr"]
        beta1, beta2 = group["adamw_betas"]
        eps: float = group["adamw_eps"]
        weight_decay: float = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)
                state["step"] = 0
            exp_avg: Tensor = state["exp_avg"]
            exp_avg_sq: Tensor = state["exp_avg_sq"]
            state["step"] += 1
            t: int = state["step"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**t
            bias_correction2 = 1.0 - beta2**t
            denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
            if weight_decay != 0.0:
                p.mul_(1.0 - lr * weight_decay)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


class MuonClip(Muon):
    """Muon with the QK-clip mechanism from Kimi K2 (arXiv:2507.20534).

    MuonClip adds the QK-clip trick of the Kimi K2 technical report (Section
    2, "MuonClip") on top of Muon: when the maximum QK attention logit
    observed for a head exceeds a threshold ``tau``, the ``q_proj`` and
    ``k_proj`` weights of that head are rescaled by
    ``sqrt(tau / max_logit)``. Because the logits are bilinear in the two
    projections, this bounds the post-clip logit by ``tau`` without changing
    the optimizer state, taming attention-logit explosions at scale.

    .. note::
        This is a teaching-oriented simplification. The production version in
        Kimi K2 is integrated with the training loop's forward hooks and
        logging system, which record per-head max logits automatically. Here
        the user registers the ``(q_proj, k_proj)`` weight pairs explicitly
        via :meth:`register_qk_params` and feeds observed max logits to
        :meth:`qk_clip`.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # name -> (q_param, k_param, num_heads)
        self._qk_registry: dict[str, tuple[Tensor, Tensor, int]] = {}
        # Registry metadata recovered by the last ``load_state_dict`` call.
        self._loaded_qk_registry: dict[str, int] = {}

    def register_qk_params(
        self,
        name: str,
        q_param: Tensor,
        k_param: Tensor,
        num_heads: int,
    ) -> None:
        """Register a ``q_proj``/``k_proj`` weight pair for QK-clipping.

        Args:
            name: Identifier of the attention layer; used as the key when
                passing observed max logits to :meth:`qk_clip`.
            q_param: Query projection weight of shape
                ``(num_heads * head_dim, in_features)``.
            k_param: Key projection weight of shape
                ``(num_kv_heads * head_dim, in_features)``. For simplicity
                ``num_kv_heads`` must equal ``num_heads`` (no GQA/MQA in this
                teaching implementation).
            num_heads: Number of attention heads.

        Raises:
            ValueError: If the weight shapes are inconsistent with
                ``num_heads``.

        """
        if q_param.ndim != 2 or k_param.ndim != 2:
            raise ValueError("q_param and k_param must be 2D weight matrices")
        if q_param.size(0) != k_param.size(0):
            raise ValueError(
                "This teaching implementation requires num_kv_heads == num_heads"
            )
        if q_param.size(0) % num_heads != 0:
            raise ValueError(
                f"q_param rows ({q_param.size(0)}) are not divisible by "
                f"num_heads ({num_heads})"
            )
        self._qk_registry[name] = (q_param, k_param, num_heads)

    @property
    def registered_qk_names(self) -> tuple[str, ...]:
        """Names of the attention layers registered for QK-clipping."""
        return tuple(self._qk_registry)

    @torch.no_grad()
    def qk_clip(
        self, max_logits: dict[str, Tensor], threshold: float
    ) -> dict[str, float]:
        """Clip QK weights of heads whose max logit exceeds ``threshold``.

        For each registered layer present in ``max_logits``, computes the
        per-head rescaling factor ``sqrt(threshold / max_logit)`` for every
        head whose observed max logit exceeds ``threshold`` and multiplies the
        corresponding rows of both the ``q_proj`` and ``k_proj`` weights by
        that factor. Since the attention logit is bilinear in the two
        projections, its post-clip upper bound is ``threshold``.

        Args:
            max_logits: Mapping from registered layer name to the max QK
                logits observed in the last forward pass. The tensor may be a
                scalar (one value shared by all heads) or have one element per
                head.
            threshold: Logit threshold ``tau`` (``100`` in Kimi K2).

        Returns:
            Mapping from layer name to the smallest rescaling factor applied
            (``1.0`` if the layer was left untouched). Layers absent
            from ``max_logits`` are skipped and omitted from the result.

        Raises:
            ValueError: If a max-logit tensor has neither one element nor one
                element per head.

        """
        if threshold <= 0.0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        applied: dict[str, float] = {}
        for name, (q_param, k_param, num_heads) in self._qk_registry.items():
            if name not in max_logits:
                continue
            logits = max_logits[name].detach().to(torch.float64).flatten()
            if logits.numel() == 1:
                logits = logits.expand(num_heads)
            elif logits.numel() != num_heads:
                raise ValueError(
                    f"max_logits[{name!r}] has {logits.numel()} elements; "
                    f"expected 1 or num_heads ({num_heads})"
                )
            head_dim = q_param.size(0) // num_heads
            worst = 1.0
            for head in range(num_heads):
                max_logit = float(logits[head])
                if max_logit <= threshold:
                    continue
                gamma = math.sqrt(threshold / max_logit)
                rows = slice(head * head_dim, (head + 1) * head_dim)
                q_param[rows].mul_(gamma)
                k_param[rows].mul_(gamma)
                worst = min(worst, gamma)
            applied[name] = worst
        return applied

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer state, including the QK-clip registry."""
        state = super().state_dict()
        # Param identity is already encoded by the packed optimizer state, so
        # only the registry metadata (names and head counts) needs saving.
        state["qk_registry"] = {
            name: num_heads for name, (_, _, num_heads) in self._qk_registry.items()
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state saved by :meth:`state_dict`.

        The QK-clip registry metadata is restored for inspection, but the
        weight tensors themselves are not (they belong to the model); call
        :meth:`register_qk_params` again after loading if you intend to keep
        clipping.

        """
        state_dict = dict(state_dict)
        self._loaded_qk_registry = state_dict.pop("qk_registry", {})
        super().load_state_dict(state_dict)
