"""The paper classifier's learning-rate schedule (plan row D3): constant or cosine, per step.

CD21 trains with a constant Adam learning rate (1e-3, ``lr_schedule: constant``, the
default). Row D3 adds a fixed cosine decay to zero over exactly the run's ``max_steps``,
with an optional linear warmup, so the final weights (``checkpoint_selection: final``) are
those of a run whose learning rate has annealed. :class:`LrSchedule` is the pure rule: the
factor of the base learning rate at optimizer step ``step`` (0-based, the number of steps
done before it, as every per-step draw of ADR 0014)::

    warmup   (step < w):  (step + 1) / w              (reaches 1 at the last warmup step)
    constant (step >= w): 1
    cosine   (step >= w): 0.5 * (1 + cos(pi * p)),  p = min(1, (step - w) / (T - w))

with ``w = warmup_steps`` and ``T = total_steps``. The cosine's last step (``T - 1``) has a
small positive factor, and a step past ``T`` (none: a cosine run has no horizon extension,
``config.Settings``) would have 0.

The plain task applies it with a ``torch.optim.lr_scheduler.LambdaLR`` over the rule
(:meth:`LrSchedule.scheduler`), stepped once after every optimizer step: at construction
it sets the learning rate of step 0, and each ``scheduler.step()`` the next step's, so the
learning rate used by step ``s`` is ``lr * factor(s)``. The plain task steps it in
``after_step`` and keeps its state in the task state (Lightning stepped it with
``interval: step``). With the defaults (constant, no
warmup) there is no scheduler at all, so a CD21 run is exactly what it was before the flag.
Nothing here imports Lightning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import LambdaLR

from astro_simclr.config import LR_SCHEDULES

__all__ = ["LrSchedule"]


@dataclass(frozen=True, slots=True)
class LrSchedule:
    """The learning-rate factor per optimizer step (module docstring).

    Raises:
        ValueError: for an unknown ``kind``, a negative ``warmup_steps``, a
            ``total_steps`` below 1, or a warmup that does not end before ``total_steps``.
    """

    kind: str = "constant"
    warmup_steps: int = 0
    total_steps: int = 1

    def __post_init__(self) -> None:
        if self.kind not in LR_SCHEDULES:
            msg = f"lr_schedule must be one of {', '.join(LR_SCHEDULES)}, got {self.kind!r}"
            raise ValueError(msg)
        if self.warmup_steps < 0 or self.total_steps < 1:
            msg = (
                "need warmup_steps >= 0 and total_steps >= 1, got "
                f"{self.warmup_steps} and {self.total_steps}"
            )
            raise ValueError(msg)
        if self.warmup_steps >= self.total_steps:
            msg = (
                f"lr_warmup_steps ({self.warmup_steps}) must be smaller than the run's "
                f"max_steps ({self.total_steps})"
            )
            raise ValueError(msg)

    @property
    def active(self) -> bool:
        """Whether the learning rate ever differs from the base one (a scheduler is needed)."""
        return self.kind != "constant" or self.warmup_steps > 0

    def factor(self, step: int) -> float:
        """The factor of the base learning rate at optimizer step ``step`` (0-based).

        Raises:
            ValueError: if ``step`` is negative.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        warmup = self.warmup_steps
        if step < warmup:
            return (step + 1) / warmup
        if self.kind == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / (self.total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def scheduler(self, optimizer: torch.optim.Optimizer) -> LambdaLR | None:
        """A ``LambdaLR`` applying :meth:`factor` to ``optimizer``; None when inactive.

        The lambda is a plain function, so the scheduler's ``state_dict`` holds plain
        values only (``LambdaLR`` saves no function), loadable with ``weights_only``.
        """
        if not self.active:
            return None

        def factor(step: int) -> float:
            return self.factor(step)

        return LambdaLR(optimizer, factor)
