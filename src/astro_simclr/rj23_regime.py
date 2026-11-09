"""The training regime of Reyes-Jainaga et al. 2023 (RJ23), engine-free (phase 3, task 3.4b).

RJ23 (ApJL 952, L43, arXiv:2304.13080, sec. III and App. A; task 3.0's research of their MIT
code, ``github.com/alercebroker/multiscale_stamps``, whose training loop is described, never
copied) train each scenario's classifier (:mod:`astro_simclr.rj23_model`) with Keras' Adam
on class-balanced batches of 256, validate the macro-F1 every 500 iterations, stop after 5
validations without improvement and save the model **at the stopping iteration**. This
module holds those rules as pure objects, with no training engine (the phase-4 rule from B0
on: ``docs/phases/04-training-platform.md``). The plain engine's adapter,
:mod:`astro_simclr.tasks.rj23`, uses these objects as they are (as the Lightning adapter
did until task 4.8). The choices the sources leave open are pinned in ADR 0023.

- :class:`RJ23Schedule`, the **schedule policy**: :meth:`RJ23Schedule.decide` is a pure
  function of ``(step, history)``, the number of optimizer steps done and the monitored
  values of the validations so far, returning an :class:`RJ23Decision` (validate? stop?
  which weights to keep?):

  - validation after every ``eval_every`` (500) optimizer steps, **step 0 included** (the
    initial weights), so the ``k``-th value of the history is the one of step
    ``k * eval_every``;
  - patience ``patience`` (5) on a **strict** improvement (``>`` for ``max``, the
    macro-F1; NaN never improves): the run stops at the validation that makes the count of
    validations since the best reach ``patience``, as Keras' and Lightning's
    ``EarlyStopping`` with ``min_delta=0``;
  - a hard cap of ``max_steps`` (100,000) optimizer steps, a multiple of ``eval_every``,
    so the cap step is validated; a stop there that patience did not cause is
    ``reason="cap"`` (:attr:`RJ23Decision.capped`), reported per run;
  - ``keep``: the decision names the weights to snapshot now, ``best`` at each new best
    and ``stop`` at the stop, so one run yields both (:data:`KEEPS`); ``stop`` is RJ23's
    primary, ``best`` the alternative the plan reports next to it.

- :class:`ClassUniformStream`, the **per-sample class-uniform sampler** (RJ23's
  ``tf.data.Dataset.sample_from_datasets`` with uniform weights over the per-class
  datasets; not CD21's tiling, :mod:`astro_simclr.training.tiling`): each item of a batch
  draws its class uniformly, then an item of that class uniformly, with replacement. Batch
  ``b`` is drawn from :func:`astro_simclr.engine.randomness.host_generator` ``(seed, b,
  RESERVED_STREAMS["rj23_class_uniform"])`` only (ADR 0014's rule for a new flag), so it is
  a pure function of the stream definition and ``b`` (as :class:`~astro_simclr.training.
  tiling.PermutationStream`): the stored position of a resumed run is its step count, and
  it sees the batches of an uninterrupted one.

- :class:`KerasAdam` and :func:`keras_adam`, **Keras Adam parity**: ``beta_1=0.9``,
  ``beta_2=0.999``, ``epsilon=1e-7`` (Keras' defaults; torch's is 1e-8), with Keras'
  epsilon placement, ``theta -= lr_t m / (sqrt(v) + eps)``, ``lr_t = lr sqrt(1 - beta_2^t)
  / (1 - beta_1^t)``. torch's ``Adam`` adds epsilon to the bias-corrected ``sqrt(v_hat)``
  instead, which equals Keras' only with ``eps / sqrt(1 - beta_2^t)``, a step-dependent
  epsilon torch cannot take (ADR 0003 kept torch's placement for CD21; ADR 0023 says why
  RJ23 does not).

- :func:`predict_probabilities`, the evaluation every engine shares: softmax outputs in
  eval mode and without gradients, in fixed-size batches.

RJ23's other TF-default parities (BN momentum 0.99 and epsilon 1e-3, Glorot-uniform
kernels, TF 'same' pooling) are in the model, :mod:`astro_simclr.rj23_model` (ADR 0013).
Nothing here imports Lightning.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from astro_simclr.engine.randomness import RESERVED_STREAMS, host_generator

__all__ = [
    "KEEPS",
    "KERAS_ADAM_BETAS",
    "KERAS_ADAM_EPS",
    "RJ23_BATCH_SIZE",
    "RJ23_EVAL_EVERY",
    "RJ23_MAX_STEPS",
    "RJ23_PATIENCE",
    "STREAM",
    "ClassUniformStream",
    "Keep",
    "KerasAdam",
    "RJ23Decision",
    "RJ23Schedule",
    "StopReason",
    "keras_adam",
    "predict_probabilities",
]

#: Validate every this many optimizer steps (RJ23: every 500 iterations).
RJ23_EVAL_EVERY: Final = 500
#: Validations without a strict improvement before the stop (RJ23: patience 5).
RJ23_PATIENCE: Final = 5
#: The hard cap on optimizer steps (task 3.4b; RJ23 state none).
RJ23_MAX_STEPS: Final = 100_000
#: RJ23's class-balanced training batch.
RJ23_BATCH_SIZE: Final = 256
#: Keras ``Adam`` defaults: ``beta_1``, ``beta_2``.
KERAS_ADAM_BETAS: Final = (0.9, 0.999)
#: Keras ``Adam`` default ``epsilon`` (torch's ``Adam`` default is 1e-8).
KERAS_ADAM_EPS: Final = 1e-7
#: The ``engine.randomness`` stream number of :class:`ClassUniformStream`.
STREAM: Final = RESERVED_STREAMS["rj23_class_uniform"]

#: Which weights a run ends with: the stopping validation's or the best validation's.
Keep = Literal["stop", "best"]
#: Both, primary first (RJ23 saved the model at the stopping iteration).
KEEPS: Final[tuple[Keep, ...]] = ("stop", "best")
#: Why a run stopped: patience ran out, or the step cap was reached first.
StopReason = Literal["patience", "cap"]
MonitorMode = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class RJ23Decision:
    """What the loop does after ``step`` optimizer steps (:meth:`RJ23Schedule.decide`).

    Attributes:
        step: the optimizer steps done.
        validate: ``step`` is on the validation grid; the last value of the history is
            this step's.
        stop: end the run now (only ever at a validation).
        keep: the weights to snapshot now: ``"best"`` when this validation is a new best,
            ``"stop"`` when the run stops here (both when the cap step is a new best).
        reason: why the run stops (``None`` unless ``stop``).
        best_step: the step of the best validation so far (``None`` before any finite
            value).
        best_value: its monitored value.
        wait: validations since the best one (since the first, if none improved yet).
    """

    step: int
    validate: bool
    stop: bool
    keep: frozenset[Keep]
    reason: StopReason | None
    best_step: int | None
    best_value: float | None
    wait: int

    @property
    def capped(self) -> bool:
        """Whether the run stopped at the step cap rather than by patience."""
        return self.reason == "cap"


@dataclass(frozen=True, slots=True)
class RJ23Schedule:
    """RJ23's validation, patience, cap and keep rules (module docstring; ADR 0023).

    Attributes:
        eval_every: validate after every this many optimizer steps, step 0 included.
        patience: stop at the validation that makes this many validations since the
            best one.
        max_steps: the hard cap on optimizer steps, a multiple of ``eval_every``.
        mode: ``max`` (the macro-F1) or ``min``.

    Raises:
        ValueError: for a non-positive ``eval_every``, ``patience`` or ``max_steps``, a cap
            that is not a multiple of ``eval_every``, or an unknown ``mode``.
    """

    eval_every: int = RJ23_EVAL_EVERY
    patience: int = RJ23_PATIENCE
    max_steps: int = RJ23_MAX_STEPS
    mode: MonitorMode = "max"

    def __post_init__(self) -> None:
        for name in ("eval_every", "patience", "max_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_steps % self.eval_every:
            msg = (
                f"max_steps ({self.max_steps}) must be a multiple of eval_every "
                f"({self.eval_every}), so that the cap step is validated"
            )
            raise ValueError(msg)
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

    def validates(self, step: int) -> bool:
        """Whether the loop validates after ``step`` optimizer steps (0 included)."""
        return step >= 0 and step % self.eval_every == 0

    def validation_step(self, index: int) -> int:
        """The step of the ``index``-th validation (0-based): ``index * eval_every``."""
        return index * self.eval_every

    def improves(self, value: float, best: float | None) -> bool:
        """A strict improvement of ``value`` over ``best`` (``None``: none yet); NaN never."""
        if math.isnan(value):
            return False
        if best is None:
            return True
        return value > best if self.mode == "max" else value < best

    def decide(self, step: int, history: Sequence[float]) -> RJ23Decision:
        """The decision after ``step`` optimizer steps, given the validations so far.

        ``history`` holds the monitored value of every validation at a step ``<= step``,
        in order: ``step // eval_every + 1`` values (validation 0 is the initial weights).
        A pure function: the loop appends the value of a validation step before calling it,
        and calling it again on a restored history gives the same decision.

        Raises:
            ValueError: for a negative ``step`` or one beyond the cap, a history of the
                wrong length, or a history that goes on past a stop.
        """
        if isinstance(step, bool) or step < 0 or step > self.max_steps:
            raise ValueError(f"step must be in [0, {self.max_steps}], got {step!r}")
        expected = step // self.eval_every + 1
        if len(history) != expected:
            msg = (
                f"after {step} steps the history must hold {expected} validations "
                f"(steps 0, {self.eval_every}, ...), got {len(history)}"
            )
            raise ValueError(msg)
        best_value: float | None = None
        best_step: int | None = None
        wait = 0
        reason: StopReason | None = None
        for index, raw in enumerate(history):
            value = float(raw)
            at = self.validation_step(index)
            if reason is not None:
                msg = f"the history goes on after the run stopped ({reason}) at step {at}"
                raise ValueError(msg)
            if self.improves(value, best_value):
                best_value, best_step, wait = value, at, 0
            else:
                wait += 1
            if wait >= self.patience:
                reason = "patience"
            elif at >= self.max_steps:
                reason = "cap"
        validate = self.validates(step)
        stop = validate and reason is not None
        keep: set[Keep] = set()
        if validate and best_step == step:
            keep.add("best")
        if stop:
            keep.add("stop")
        return RJ23Decision(
            step=step,
            validate=validate,
            stop=stop,
            keep=frozenset(keep),
            reason=reason if stop else None,
            best_step=best_step,
            best_value=best_value,
            wait=wait,
        )


class ClassUniformStream:
    """Batches whose items draw a class uniformly, then one of its items uniformly.

    ``labels`` are the training labels (values in ``range(n_classes)``; ``n_classes``
    defaults to ``max + 1``), ``batch_size`` the items per batch and ``seed`` the stream's
    seed. :meth:`batch` gives the indices of batch ``b``: ``rng = host_generator(seed, b,
    STREAM)`` (:data:`STREAM`) draws the ``batch_size`` classes
    (``rng.integers(n_classes)``), then one index within each drawn class
    (``rng.integers(count_c)``), mapped to the class's ``k``-th item in label-array order.
    Items repeat within and across batches (draws with replacement), and a class's expected
    share of every batch is ``1 / n_classes`` whatever its size.

    :attr:`batches_per_pass` (``ceil(n / batch_size)``) is only the length of the
    logging "epoch" (the loop's derived epoch; Lightning's): the stream has no passes.

    Raises:
        ValueError: if ``labels`` is not a non-empty 1-D integer array with values in
            ``range(n_classes)``, a class has no item, ``batch_size`` is not positive or
            ``seed`` is negative.
    """

    def __init__(
        self,
        labels: npt.ArrayLike,
        batch_size: int,
        seed: int,
        *,
        n_classes: int | None = None,
    ) -> None:
        array = np.asarray(labels)
        if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
            msg = f"labels must be a non-empty 1-D integer array, got {array.dtype} {array.shape}"
            raise ValueError(msg)
        classes = int(array.max()) + 1 if n_classes is None else n_classes
        if array.min() < 0 or array.max() >= classes:
            raise ValueError(f"labels must be in range({classes})")
        counts = np.bincount(array, minlength=classes)
        if (counts == 0).any():
            empty = np.flatnonzero(counts == 0).tolist()
            raise ValueError(f"classes {empty} have no item: a class-uniform draw needs each")
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        if seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self.n = int(array.size)
        self.n_classes = classes
        self.batch_size = batch_size
        self.seed = seed
        self.class_counts: tuple[int, ...] = tuple(int(c) for c in counts)
        # The items of class c are _members[_offsets[c] : _offsets[c] + count_c].
        self._members = np.argsort(array, kind="stable").astype(np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self._counts = counts.astype(np.int64)

    @property
    def batches_per_pass(self) -> int:
        """``ceil(n / batch_size)``: the logging epoch (``engine.loop``'s derived epoch)."""
        return -(-self.n // self.batch_size)

    def batch(self, batch: int) -> npt.NDArray[np.int64]:
        """The ``batch_size`` item indices of batch ``batch`` (a pure function of it).

        Raises:
            ValueError: if ``batch`` is negative.
        """
        if batch < 0:
            raise ValueError(f"batch index must be >= 0, got {batch}")
        rng = host_generator(self.seed, batch, STREAM)
        drawn = rng.integers(0, self.n_classes, size=self.batch_size)
        within = rng.integers(0, self._counts[drawn])
        return self._members[self._offsets[drawn] + within]

    def iterate(self, start: int = 0) -> Iterator[npt.NDArray[np.int64]]:
        """Batches ``start, start + 1, ...`` without end (a resume starts at its step)."""
        batch = start
        while True:
            yield self.batch(batch)
            batch += 1

    def state(self) -> dict[str, Any]:
        """What defines the stream, for checkpoints (a resume must match it)."""
        return {
            "sampler": "class_uniform",
            "n": self.n,
            "class_counts": list(self.class_counts),
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


class KerasAdam(torch.optim.Optimizer):
    """Adam with Keras' update and defaults (module docstring), no weight decay.

    Per parameter, at step ``t`` (from 1)::

        m = beta_1 m + (1 - beta_1) g
        v = beta_2 v + (1 - beta_2) g^2
        theta -= lr sqrt(1 - beta_2^t) / (1 - beta_1^t) * m / (sqrt(v) + eps)

    as ``tf.keras.optimizers.Adam`` (Keras 2 and 3; ``amsgrad=False``). ``m`` and ``v``
    start at zero; the state holds ``step`` (an ``int``), ``exp_avg`` and ``exp_avg_sq``.

    Raises:
        ValueError: for a non-positive ``lr``, a beta outside ``[0, 1)`` or a negative
            ``eps``; in :meth:`step`, for a sparse gradient.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = KERAS_ADAM_BETAS,
        eps: float = KERAS_ADAM_EPS,
    ) -> None:
        if not lr > 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """One update of every parameter with a gradient; returns ``closure()`` if given."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise ValueError("KerasAdam does not support sparse gradients")
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)
        return loss


def keras_adam(params: Iterable[torch.nn.Parameter], lr: float) -> KerasAdam:
    """RJ23's optimizer: :class:`KerasAdam` with Keras' defaults and learning rate ``lr``.

    ``lr`` is the scenario's Table 2 rate (:data:`astro_simclr.rj23_model.RJ23_TABLE2`).

    Raises:
        ValueError: for a non-positive ``lr``.
    """
    return KerasAdam(params, lr=lr, betas=KERAS_ADAM_BETAS, eps=KERAS_ADAM_EPS)


def predict_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Softmax outputs ``(N, C)`` of ``model(images, positions)``, float32 on the CPU.

    Runs in eval mode (dropout off, BN on its moving statistics) under ``torch.no_grad``,
    ``batch_size`` rows at a time in order, then restores the model's previous mode. The
    inputs stay where they are (the device-resident splits of a run).

    Raises:
        ValueError: if ``batch_size`` is not positive, the inputs are empty or their
            lengths differ.
    """
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")
    n = int(images.shape[0])
    if n == 0 or int(positions.shape[0]) != n:
        msg = f"need N > 0 images and N positions, got {n} and {int(positions.shape[0])}"
        raise ValueError(msg)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            chunks = [
                torch.softmax(
                    model(
                        images[start : start + batch_size], positions[start : start + batch_size]
                    ),
                    dim=1,
                )
                .float()
                .cpu()
                for start in range(0, n, batch_size)
            ]
    finally:
        model.train(was_training)
    return torch.cat(chunks)
