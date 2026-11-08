"""SimCLR's optimizer and learning-rate schedule (phase 5, task 5.0b; ADR 0018).

- **LARS** (:func:`build_lars`): the package's :class:`astro_simclr.optim.LARS` with the
  fixed trust ratio (``legacy_trust_ratio=False``: the ratio ``eeta * ||w|| / ||g||`` is
  used only when both norms are > 0, else 1), no Nesterov (google-research/simclr's
  default), momentum 0.9, ``eeta`` 0.001 and weight decay 1e-6 (Chen et al. 2020, App.
  B.1). BatchNorm parameters and biases are **excluded from both the weight decay and the
  trust ratio** (:func:`lars_param_groups`; SimCLR's ``exclude_from_weight_decay`` and
  ``exclude_from_layer_adaptation``), so they get plain SGD with momentum. The legacy
  configuration (one group, ``.ge(0)``) froze every zero-initialised parameter, the BN
  shifts included (the SimCLR review, finding A2).
- **Learning rate** (:func:`base_learning_rate`): ``0.3 * B / 256`` (the paper's linear
  scaling).
- **Schedule** (:func:`warmup_cosine_lr`): per optimizer step, a linear warmup to the base
  rate, then a cosine decay that reaches ``lr_min`` exactly at the run's final step. The
  schedule takes the run's length explicitly, so it cannot outlive or undershoot it (the
  legacy per-epoch schedule ignored ``max_epochs``: review finding A1).

Engine-free (ADR 0014): pure functions of ``step``, the optimizer steps done before the
step; the loop sets the rate with :func:`set_learning_rate` before each step.
"""

from __future__ import annotations

import math
from typing import Any, Final

from torch import nn
from torch.optim import Optimizer

from astro_simclr.optim import EETA_DEFAULT, EXCLUDE_KEY, LARS

__all__ = [
    "LR_PER_256",
    "MOMENTUM",
    "WEIGHT_DECAY",
    "base_learning_rate",
    "build_lars",
    "is_excluded",
    "lars_param_groups",
    "set_learning_rate",
    "warmup_cosine_lr",
]

#: The paper's base rate per 256 stamps (LARS, batch 256-8192).
LR_PER_256: Final = 0.3
#: The paper's weight decay (not applied to BatchNorm parameters and biases).
WEIGHT_DECAY: Final = 1e-6
#: LARS momentum.
MOMENTUM: Final = 0.9

#: The normalisation layers whose parameters are excluded (affine weight and bias).
_NORM_LAYERS: Final = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
)


def base_learning_rate(batch_size: int, per_256: float = LR_PER_256) -> float:
    """``per_256 * batch_size / 256``.

    Raises:
        ValueError: if ``batch_size < 1`` or ``per_256 <= 0``.
    """
    if batch_size < 1 or not per_256 > 0:
        raise ValueError(f"need batch_size >= 1 and per_256 > 0, got {batch_size}, {per_256}")
    return per_256 * batch_size / 256


def is_excluded(module: nn.Module, name: str) -> bool:
    """Whether parameter ``name`` of ``module`` (its own, not a child's) is excluded.

    Excluded: every parameter of a normalisation layer (BatchNorm, GroupNorm, LayerNorm)
    and every bias.
    """
    return isinstance(module, _NORM_LAYERS) or name == "bias"


def lars_param_groups(model: nn.Module, weight_decay: float = WEIGHT_DECAY) -> list[dict[str, Any]]:
    """Two LARS param groups of ``model``'s trainable parameters, adapted then excluded.

    The first group (convolution and linear weights) gets ``weight_decay`` and the trust
    ratio; the second (:func:`is_excluded`) gets ``weight_decay=0`` and
    ``exclude_from_adaptation=True`` (:data:`astro_simclr.optim.EXCLUDE_KEY`). Parameters
    keep ``model.parameters()`` order within each group; shared parameters appear once.

    Raises:
        ValueError: if ``model`` has no trainable parameter.
    """
    adapted: list[nn.Parameter] = []
    excluded: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            (excluded if is_excluded(module, name) else adapted).append(param)
    if not adapted and not excluded:
        raise ValueError("the model has no trainable parameter")
    return [
        {"params": adapted, "weight_decay": weight_decay},
        {"params": excluded, "weight_decay": 0.0, EXCLUDE_KEY: True},
    ]


def build_lars(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = WEIGHT_DECAY,
    momentum: float = MOMENTUM,
    eeta: float = EETA_DEFAULT,
) -> LARS:
    """LARS over :func:`lars_param_groups` of ``model`` (module docstring).

    ``lr`` is the initial rate; the loop overwrites it every step (:func:`set_learning_rate`).

    Raises:
        ValueError: as :class:`~astro_simclr.optim.LARS` and :func:`lars_param_groups`.
    """
    return LARS(
        lars_param_groups(model, weight_decay),
        lr=lr,
        momentum=momentum,
        use_nesterov=False,
        weight_decay=weight_decay,
        eeta=eeta,
        legacy_trust_ratio=False,
    )


def warmup_cosine_lr(
    step: int, *, base_lr: float, total_steps: int, warmup_steps: int, lr_min: float = 0.0
) -> float:
    """The learning rate of the step after ``step`` optimizer steps (0-based).

    - ``step < warmup_steps``: ``base_lr * (step + 1) / warmup_steps``, so the last warmup
      step runs at ``base_lr``;
    - then ``lr_min + (base_lr - lr_min) * (1 + cos(pi * t)) / 2`` with ``t = (step -
      warmup_steps) / (total_steps - 1 - warmup_steps)``, so the first step after the
      warmup runs at ``base_lr`` and the run's final step (``step = total_steps - 1``)
      exactly at ``lr_min``. With a single step after the warmup, that step is at
      ``lr_min``.

    Raises:
        ValueError: unless ``0 <= step < total_steps``, ``0 <= warmup_steps <
            total_steps`` and ``0 <= lr_min <= base_lr``.
    """
    if not 0 <= step < total_steps or not 0 <= warmup_steps < total_steps:
        msg = (
            "need 0 <= step < total_steps and 0 <= warmup_steps < total_steps, got "
            f"step {step}, warmup_steps {warmup_steps}, total_steps {total_steps}"
        )
        raise ValueError(msg)
    if not 0 <= lr_min <= base_lr:
        raise ValueError(f"need 0 <= lr_min <= base_lr, got {lr_min} and {base_lr}")
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - 1 - warmup_steps
    t = 1.0 if decay_steps == 0 else (step - warmup_steps) / decay_steps
    if t == 1.0:
        return lr_min  # exact, not lr_min + rounding of (1 + cos(pi)) / 2
    return lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))


def set_learning_rate(optimizer: Optimizer, lr: float) -> None:
    """Set ``lr`` on every param group of ``optimizer``."""
    for group in optimizer.param_groups:
        group["lr"] = lr
