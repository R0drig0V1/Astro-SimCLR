"""Optimizer and LR schedule of SimCLR pretraining: LARS and cosine-with-warmup (task 1.5).

LARS is a port of the legacy ``utils/lars.py`` (itself a translation of the TensorFlow
optimizer in google-research/simclr). The update, for every parameter ``w`` with gradient
``g`` (classic momentum, the only mode the legacy code implemented)::

    g      <- g + weight_decay * w
    ratio  <- eeta * ||w|| / ||g||       (or 1, see ``legacy_trust_ratio``)
    v      <- momentum * v + lr * ratio * g
    w      <- w - (momentum * v + lr * ratio * g)   if use_nesterov else   w - v

The schedule replaces the legacy timm 0.6.x ``CosineLRScheduler`` (``warmup_t=5``,
``warmup_lr_init=lr/3``, ``t_initial=350``, ``cycle_limit=1``, ``lr_min=1e-4``) that
``utils/training.py`` wrapped in a ``LambdaLR``, so timm is no longer a dependency. It is
stepped once per epoch.

Behaviour changes against the legacy code (also in docs/architecture.md):

- LARS works on any device: the legacy ``torch.Tensor([1.0]).to(g_norm.get_device())``
  crashed on CPU (``get_device()`` is -1) and would have mapped MPS to ``cuda:N``.
- The legacy trust-ratio test ``w_norm.ge(0)`` / ``g_norm.ge(0)`` is always true, so a zero
  gradient gives an inf/NaN ratio and a zero weight gives ratio 0 (zero-initialised biases
  never move). It stays the default (``legacy_trust_ratio=True``) because the fix changes
  ResNet results; ``legacy_trust_ratio=False`` uses ``.gt(0)``, so those cases get ratio 1.
- The deprecated ``add_(scalar, tensor)`` overload is replaced by ``add_(tensor, alpha=)``.
- Hyperparameters are read from each param group (the legacy step mixed group values with
  instance attributes); with a single group, as in every legacy run, nothing changes.
- ``p.grad`` is no longer modified in place by the weight-decay term.
- Dropped: the unused ``epoch`` argument and counter of ``step``, the never-implemented
  ``classic_momentum=False`` mode and the ``exclude_from_*`` lists, which the legacy step
  never consulted (its name filtering was a TODO).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, overload

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import ParamsT

__all__ = ["EETA_DEFAULT", "LARS", "cosine_warmup_lr", "cosine_warmup_scheduler"]

#: Default trust coefficient (``eeta``) of the legacy and TensorFlow LARS.
EETA_DEFAULT = 0.001


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (You, Gitman and Ginsburg, arXiv:1708.03888).

    Args:
        params: parameters or param groups, as for any torch optimizer.
        lr: learning rate (required).
        momentum: momentum factor.
        use_nesterov: apply the Nesterov correction ``momentum * v + scaled_lr * g``.
        weight_decay: L2 term added to the gradient before the trust ratio.
        eeta: trust coefficient.
        legacy_trust_ratio: True (default) keeps the legacy ``.ge(0)`` test, so a zero
            gradient makes the parameter NaN and a zero weight never moves; False uses
            ``.gt(0)``, giving those parameters a trust ratio of 1.

    Raises:
        ValueError: a negative ``lr``, ``momentum``, ``weight_decay`` or ``eeta``.
    """

    def __init__(
        self,
        params: ParamsT,
        *,
        lr: float,
        momentum: float = 0.9,
        use_nesterov: bool = False,
        weight_decay: float = 0.0,
        eeta: float = EETA_DEFAULT,
        legacy_trust_ratio: bool = True,
    ) -> None:
        for name, value in (
            ("lr", lr),
            ("momentum", momentum),
            ("weight_decay", weight_decay),
            ("eeta", eeta),
        ):
            if value < 0:
                msg = f"LARS {name} must be >= 0, got {value}"
                raise ValueError(msg)
        defaults: dict[str, Any] = {
            "lr": lr,
            "momentum": momentum,
            "use_nesterov": use_nesterov,
            "weight_decay": weight_decay,
            "eeta": eeta,
            "legacy_trust_ratio": legacy_trust_ratio,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Take one optimization step; ``closure`` re-evaluates the loss, if given.

        Parameters without a gradient are skipped. The trust ratio is computed on the
        parameter's own device and moved to the host once per parameter (as the legacy
        ``.item()`` did), so the arithmetic matches the legacy fixtures.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, group)
        return loss

    def _update(self, p: torch.Tensor, group: dict[str, Any]) -> None:
        assert p.grad is not None  # noqa: S101 -- checked by the caller; narrows the type
        lr: float = group["lr"]
        momentum: float = group["momentum"]
        grad = p.grad.add(p, alpha=group["weight_decay"])
        w_norm = torch.linalg.vector_norm(p)
        g_norm = torch.linalg.vector_norm(grad)
        if group["legacy_trust_ratio"]:
            usable = w_norm.ge(0) & g_norm.ge(0)
        else:
            usable = w_norm.gt(0) & g_norm.gt(0)
        trust_ratio = torch.where(
            usable, group["eeta"] * w_norm / g_norm, torch.ones_like(w_norm)
        ).item()
        scaled_lr = lr * trust_ratio
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        next_v: torch.Tensor = state["momentum_buffer"]
        next_v.mul_(momentum).add_(grad, alpha=scaled_lr)
        update = momentum * next_v + scaled_lr * grad if group["use_nesterov"] else next_v
        p.sub_(update)


def cosine_warmup_lr(
    epoch: int,
    base_lr: float,
    *,
    warmup_epochs: int = 5,
    warmup_lr_init: float | None = None,
    t_initial: int = 350,
    lr_min: float = 1e-4,
) -> float:
    """Learning rate at ``epoch`` of the legacy timm 0.6.x cosine schedule (one cycle).

    - ``epoch < warmup_epochs``: linear from ``warmup_lr_init`` (default ``base_lr / 3``)
      towards ``base_lr``; it never reaches ``base_lr`` (timm's ``warmup_prefix=False``).
    - ``warmup_epochs <= epoch < t_initial``: ``lr_min + (base_lr - lr_min) / 2 *
      (1 + cos(pi * epoch / t_initial))``; the cosine counts from epoch 0, so it is already
      ``warmup_epochs`` into its decay when the warmup ends.
    - ``epoch >= t_initial``: ``lr_min`` forever (``cycle_limit=1``).

    Raises:
        ValueError: a negative ``epoch`` or ``warmup_epochs``, or a non-positive
            ``t_initial``.
    """
    if epoch < 0 or warmup_epochs < 0 or t_initial <= 0:
        msg = (
            "need epoch >= 0, warmup_epochs >= 0 and t_initial > 0, got "
            f"{epoch}, {warmup_epochs} and {t_initial}"
        )
        raise ValueError(msg)
    if epoch < warmup_epochs:
        init = base_lr / 3 if warmup_lr_init is None else warmup_lr_init
        # Same operation order as timm, so the values match its fixtures bit for bit.
        return init + epoch * ((base_lr - init) / warmup_epochs)
    if epoch >= t_initial:
        return lr_min
    return lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * epoch / t_initial))


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    base_lr: float,
    *,
    warmup_epochs: int = 5,
    warmup_lr_init: float | None = None,
    t_initial: int = 350,
    lr_min: float = 1e-4,
) -> LambdaLR:
    """A ``LambdaLR`` that sets every group's lr to :func:`cosine_warmup_lr` each epoch.

    Like the legacy wrapper, the multiplier is ``cosine_warmup_lr(epoch) / base_lr``, so a
    group whose initial lr is ``base_lr`` follows the schedule exactly. It must be stepped
    once per epoch (Lightning ``interval: "epoch"``); on construction it already sets the
    epoch-0 value.

    Raises:
        ValueError: ``base_lr`` is not positive (the multiplier divides by it).
    """
    if base_lr <= 0:
        msg = f"base_lr must be > 0, got {base_lr}"
        raise ValueError(msg)

    def multiplier(epoch: int) -> float:
        lr = cosine_warmup_lr(
            epoch,
            base_lr,
            warmup_epochs=warmup_epochs,
            warmup_lr_init=warmup_lr_init,
            t_initial=t_initial,
            lr_min=lr_min,
        )
        return lr / base_lr

    return LambdaLR(optimizer, lr_lambda=multiplier)
