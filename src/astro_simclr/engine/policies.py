"""The schedule policies of the plain loop: monitor, validation grid, horizon, stop, keep.

Each policy is a frozen dataclass whose pure rule is the contract that 4.1's loop follows
and that the 4.7(a) differential test checks against Lightning (ADR 0012):

- :class:`Monitor`: a strict improvement (``<`` for ``min``, ``>`` for ``max``); NaN never
  improves; the first finite value always does (``HorizonCallback._improves``);
- :class:`ValidationGrid`: validate after the optimizer steps that are multiples of the
  interval, never at step 0, never because an epoch ended, and never forced after the last
  step (Lightning's ``(total_batch_idx + 1) % val_check_batch == 0``);
- :class:`Horizon`: a new best after ``s`` steps sets the horizon to
  ``max(horizon, s + increment)``; a resume keeps ``max(initial, saved)``;
- :class:`EarlyStop`: Lightning's ``EarlyStopping`` semantics, counted in validations (the
  ``rj23`` schedule of task 3.4b; no current run uses it);
- :data:`Keep`: which weights a run ends with, ``best`` (restored, CD21) or ``stop`` (the
  last, RJ23's primary); ``rj23`` reports both from one run.

The loop logic itself (when the rules are applied) arrives in 4.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from astro_simclr.config import ConfigError

__all__ = [
    "EarlyStop",
    "Horizon",
    "Keep",
    "Monitor",
    "MonitorMode",
    "Schedule",
    "ValidationGrid",
    "ValidationTrigger",
]

#: Whether smaller (``min``) or larger (``max``) monitored values are better.
MonitorMode = Literal["min", "max"]
#: The weights a run ends with: the best validation's (restored) or the last step's.
Keep = Literal["stop", "best"]


@dataclass(frozen=True, slots=True)
class Monitor:
    """The validation metric that selects the best weights.

    Raises:
        ConfigError: for an empty name or a ``mode`` other than ``min`` / ``max``.
    """

    name: str = "loss_val"
    mode: MonitorMode = "min"

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("the monitor needs a metric name")
        if self.mode not in ("min", "max"):
            raise ConfigError(f"monitor mode must be 'min' or 'max', got {self.mode!r}")

    def improves(self, value: float, best: float | None) -> bool:
        """Whether ``value`` is a new best over ``best`` (``None``: no best yet).

        The comparison is strict and made on the Python floats of the float32 values, as
        ``HorizonCallback`` makes it: equal values do not improve, NaN never does.
        """
        if best is None:
            return not math.isnan(value)
        return value < best if self.mode == "min" else value > best


class ValidationTrigger(Protocol):
    """When the loop validates, as a function of the optimizer steps done."""

    def due(self, steps_done: int) -> bool:
        """Whether to validate after the step that brought the count to ``steps_done``."""


@dataclass(frozen=True, slots=True)
class ValidationGrid:
    """Validate after every ``every_n_steps`` optimizer steps (10 in the paper).

    Lightning (``check_val_every_n_epoch=None``, integer ``val_check_interval``) validates
    when ``(total_batch_idx + 1) % val_check_interval == 0``; with one optimizer step per
    batch ``total_batch_idx + 1`` is the steps done. So the epoch boundary (1,136 steps
    for ``red``) triggers nothing, and the last step is validated only when it is on the
    grid (always for the presets: 30,000 and ``s + 10,000`` with ``s`` on the grid).
    Task 3.4b may add a field (default off) for RJ23's "iteration 0 included".

    Raises:
        ConfigError: if ``every_n_steps`` is not positive.
    """

    every_n_steps: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.every_n_steps, bool) or self.every_n_steps < 1:
            raise ConfigError(f"every_n_steps must be >= 1, got {self.every_n_steps!r}")

    def due(self, steps_done: int) -> bool:
        """``steps_done >= 1`` and on the grid."""
        return steps_done >= 1 and steps_done % self.every_n_steps == 0


@dataclass(frozen=True, slots=True)
class Horizon:
    """ALeRCE's training horizon: ``initial`` steps, extended by each new best.

    ``increment`` 0 makes it a fixed cap (the ``rj23`` schedule's 100k iterations). The
    loop checks ``steps_done >= horizon`` after the step's validation, so a new best at the
    horizon step extends the run instead of ending it.

    Raises:
        ConfigError: for a non-positive ``initial`` or a negative ``increment``.
    """

    initial: int = 30_000
    increment: int = 10_000

    def __post_init__(self) -> None:
        if self.initial < 1 or self.increment < 0:
            msg = f"need initial >= 1 and increment >= 0, got {self.initial}, {self.increment}"
            raise ConfigError(msg)

    def extended(self, horizon: int, steps_done: int) -> int:
        """The horizon after a new best at ``steps_done``: ``max(horizon, s + increment)``."""
        return max(horizon, steps_done + self.increment)

    def resumed(self, saved: int) -> int:
        """The horizon of a resumed run: a larger ``initial`` extends it, nothing shortens it."""
        return max(self.initial, saved)


@dataclass(frozen=True, slots=True)
class EarlyStop:
    """Stop after ``patience`` validations without an improvement (Lightning's semantics).

    An improvement is ``current + min_delta < best`` (``min``; ``current - min_delta > best``
    for ``max``), the wait count resets on it; ``check_finite`` stops on a non-finite value;
    ``divergence_threshold`` stops as soon as the value is worse than it
    (``lightning/pytorch/callbacks/early_stopping.py``, ADR 0012). A contract only: the
    first user is task 3.4b's ``rj23`` schedule (patience 5, strict, ``min_delta`` 0).

    Raises:
        ConfigError: for a ``patience`` below 1 or a negative ``min_delta``.
    """

    patience: int
    min_delta: float = 0.0
    check_finite: bool = True
    divergence_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ConfigError(f"patience must be >= 1, got {self.patience}")
        if self.min_delta < 0:
            raise ConfigError(f"min_delta must be >= 0, got {self.min_delta}")


@dataclass(frozen=True, slots=True)
class Schedule:
    """Everything that decides when a run validates, extends, stops and what it keeps.

    The ``carrasco2021`` preset is ``Schedule()`` (``loss_val``/``min``, every 10 steps,
    horizon 30,000 + 10,000, ``keep=("best",)``, ``last`` every 1,000 steps).

    Raises:
        ConfigError: for an empty or repeated ``keep``, or a ``checkpoint_every_n_steps``
            that is not a positive multiple of the validation interval (a ``last``
            checkpoint is written after a validation, so a resume owes none).
    """

    monitor: Monitor = Monitor()
    validation: ValidationGrid = ValidationGrid()
    horizon: Horizon = Horizon()
    early_stop: EarlyStop | None = None
    keep: tuple[Keep, ...] = ("best",)
    checkpoint_every_n_steps: int | None = 1_000

    def __post_init__(self) -> None:
        if not self.keep or len(set(self.keep)) != len(self.keep):
            raise ConfigError(f"keep must name 'stop' and/or 'best' once each, got {self.keep}")
        if any(choice not in ("stop", "best") for choice in self.keep):
            raise ConfigError(f"keep must name 'stop' and/or 'best', got {self.keep}")
        every = self.checkpoint_every_n_steps
        interval = self.validation.every_n_steps
        if every is not None and (every < 1 or every % interval != 0):
            msg = (
                f"checkpoint_every_n_steps ({every}) must be a positive multiple of the "
                f"validation interval ({interval})"
            )
            raise ConfigError(msg)
