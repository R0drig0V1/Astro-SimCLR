"""The plain training loop (task 4.1, ADR 0014): one visible loop in place of Lightning's.

:func:`fit` trains a :class:`~astro_simclr.engine.task.Task` on a training stream and
validates it on one evaluation source, following ADR 0014's checklist item by item:

- **C1, order of a run:** ADR 0010's precision and the determinism flags
  (:meth:`DevicePolicy.apply`), then :func:`~.randomness.seed_everything`, then the task
  built **on the CPU** by the caller's factory (so its initial weights do not depend on
  the device), then the module moved to the device, then ``Task.make_optimizer()``.
- **C2:** each step is forward, ``optimizer.zero_grad(set_to_none=True)``, backward,
  ``optimizer.step()`` (Lightning's closure order), then ``Task.after_step(step)`` (a
  scheduler step, a weight EMA update: Lightning's ``interval: step`` scheduler and
  ``on_train_batch_end``); ``module.zero_grad(set_to_none=True)`` before each validation,
  so every gradient is ``None`` while it runs.
- **C3:** a fit validation captures every submodule's ``training`` flag, calls
  ``eval()``, runs under ``torch.no_grad()`` and restores the flags (not a blanket
  ``train()``: a frozen encoder stays in ``eval``); the loop never calls ``train()`` at the
  start. The final evaluation (:func:`evaluate`) runs under ``torch.inference_mode()``.
- **C4, C5:** ``steps_done`` (optimizer steps done) is the only clock. After each step the
  validation trigger (:class:`~.policies.ValidationGrid`) decides; the policies are
  updated (a new best keeps the weights and extends the horizon); ``last`` is written if
  due; the loop ends when ``steps_done >= horizon`` *after* that, so a new best at the
  horizon step extends the run. No validation at an epoch boundary, none forced after the
  final step, none before training unless the grid validates at start (a fresh fit's
  initial weights, step 0: the ``rj23`` schedule). The epoch is derived, never counted:
  ``(steps - 1) // batches_per_pass``, and 0 for a validation at step 0 (Lightning's
  ``current_epoch`` before the first step).
- **C6:** ``train_step(batch, step=steps_done)`` and ``batch(steps_done)``: every per-step
  draw is keyed by the steps done before the step.
- **C7:** the monitored value is ``float(mean_of_batch_losses(...))`` (``loss_val``), or a
  ``score`` of the validation's outputs that the caller names in the monitor (the ``rj23``
  schedule's ``macro_f1_val``), compared strictly (:class:`~.policies.Monitor`); early
  stopping follows Lightning's ``EarlyStopping`` (:meth:`~.policies.EarlyStop.update`), or
  the schedule's :class:`~.policies.StopRule` decides on the whole history.
- **C8:** :meth:`DevicePolicy.apply`. **C9:** :func:`evaluate_kept` evaluates the weights the
  schedule keeps (``best``: restored; ``stop``: the last), ``stop`` first.
- **C10:** checkpoints ``ckpt/last.pt`` (the whole training state, every RNG stream
  included) after the validation of every ``checkpoint_every_n_steps`` steps and at the end
  of training, and ``ckpt/best.pt`` (the best weights) at the end. The resume order is the
  ADR's: checks, precision and seed, build on the CPU, ``model``, move, optimizer,
  ``task``, ``policy``, ``stream``, and the RNG streams last.
- **C11:** no gradient clipping, no accumulation, one optimizer, no autocast, no sanity
  validation, no batch transfer (the sources serve device batches).

Outputs, under ``run_dir`` (ADR 0015's layout) when one is given: ``scalars.jsonl``
(:mod:`.scalars`), ``ckpt/last.pt``, ``ckpt/best.pt``. There is **no progress bar**: a
``progress`` stream, when given, gets one line per validation. ``manifest.json`` is the
caller's (:mod:`.record`): the loop does not know the config it was built from (4.2).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Any, Final, Protocol

import torch
from torch import nn

from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.engine.checkpoint import (
    CHECKPOINT_FORMAT,
    BestWeights,
    Checkpoint,
    IdentityState,
    PolicyState,
    capture_rng,
    check_identity,
    check_numerics,
    load_checkpoint,
    numerics_record,
    restore_rng,
    save_checkpoint,
)
from astro_simclr.engine.data import BatchSource, EvalSource
from astro_simclr.engine.device import DevicePolicy
from astro_simclr.engine.policies import Keep, Schedule
from astro_simclr.engine.randomness import seed_everything
from astro_simclr.engine.scalars import ScalarLog
from astro_simclr.engine.task import EvalOutput, Task, mean_of_batch_losses
from astro_simclr.runs.layout import BEST_CHECKPOINT, LAST_CHECKPOINT, SCALARS_FILE

__all__ = [
    "MONITORED",
    "Evaluation",
    "FitResult",
    "Score",
    "TrainStream",
    "Validation",
    "evaluate",
    "evaluate_kept",
    "fit",
]

#: The metric a fit validation computes by default (the mean of the batch losses); a fit
#: given a ``score`` monitors the metric its schedule names instead (task 3.4b).
MONITORED: Final = "loss_val"

#: A validation metric of the fit's outputs (the ``rj23`` schedule's macro-F1).
Score = Callable[[Sequence[EvalOutput]], float]


class TrainStream(BatchSource, Protocol):
    """A training stream that can be resumed (``engine.tiers.TrainSource``)."""

    def resume(self, state: dict[str, Any], steps_done: int) -> None:
        """Check the saved stream definition, then expect step ``steps_done`` next."""


@dataclass(frozen=True, slots=True)
class Validation:
    """One fit validation, after the policies' update (what ``scalars.jsonl`` records).

    Attributes:
        steps_done: optimizer steps done when it ran (Lightning's ``global_step``).
        epoch: the pass running then, ``(steps_done - 1) // batches_per_pass``; 0 for
            the step-0 validation (the first pass, about to run; Lightning's
            ``current_epoch`` there).
        value: the monitored value (a Python float of the float32 mean).
        improved: whether it was a new best.
        horizon: the horizon after the update.
        wait_count: early stopping's count after the update (0 without early stopping).
        train_loss: mean training loss of the steps since the previous validation of
            this fit (``None`` if none ran in this fit, e.g. right after a resume).
    """

    steps_done: int
    epoch: int
    value: float
    improved: bool
    horizon: int
    wait_count: int
    train_loss: float | None


@dataclass(frozen=True, slots=True)
class FitResult:
    """What a fit leaves: the schedule's outcome and the best weights (``metrics.json``'s).

    ``last_step`` is the steps done at the end; ``best_epoch`` / ``last_epoch`` are derived
    as ``metrics.json`` derives them (ADR 0014, C5). ``validations`` are this fit's only (a
    resumed fit does not repeat the earlier ones). ``best_model`` is the best weights on
    the CPU (``None`` when the schedule does not keep ``best`` or no validation ran).
    ``history`` is every fit validation's value of the run, resumed ones included, for a
    schedule with a ``stop_rule`` (empty otherwise).
    """

    last_step: int
    horizon: int
    best_step: int | None
    best_score: float | None
    batches_per_pass: int
    stopped_early: bool
    validations: tuple[Validation, ...]
    resumed_from: Path | None
    steps: int
    fit_seconds: float
    best_model: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    history: tuple[float, ...] = ()

    @property
    def best_epoch(self) -> int | None:
        """The pass running at the best validation (0 for a step-0 best), or ``None``."""
        if self.best_step is None:
            return None
        return _validation_epoch(self.best_step, self.batches_per_pass)

    @property
    def last_epoch(self) -> int:
        """The pass running at the last step (``-1`` before any step)."""
        return _epoch(self.last_step, self.batches_per_pass)

    @property
    def seconds_per_step(self) -> float | None:
        """Wall-clock seconds per optimizer step of this fit (``None`` without steps)."""
        return self.fit_seconds / self.steps if self.steps else None


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One evaluation pass (every tensor on the CPU).

    Attributes:
        loss: the mean of the batch losses (the monitored value's definition).
        probabilities: (N, C) float32, in source order.
        targets: (N,) int64 true labels.
    """

    loss: float
    probabilities: torch.Tensor
    targets: torch.Tensor


def _epoch(steps: int, batches_per_pass: int) -> int:
    return (steps - 1) // batches_per_pass


def _validation_epoch(steps: int, batches_per_pass: int) -> int:
    # A validation at step 0 (ValidationGrid.at_start) sees the first pass about to run:
    # epoch 0, as Lightning's current_epoch, not the -1 of "the pass of the last step".
    return max(0, _epoch(steps, batches_per_pass))


def _versions() -> dict[str, str | None]:
    try:
        package = version("astro-simclr")
    except PackageNotFoundError:  # pragma: no cover - the package is always installed here
        package = None
    return {"astro_simclr": package, "torch": str(torch.__version__)}


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """``module.state_dict()`` copied to the CPU (a copy even when it is there already)."""
    return {key: value.detach().to("cpu", copy=True) for key, value in module.state_dict().items()}


@contextmanager
def _evaluating(module: nn.Module) -> Iterator[None]:
    """``eval()`` for the block, then every submodule's own ``training`` flag back (C3)."""
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, mode in modes:
            child.training = mode


@dataclass(slots=True)
class _Policy:
    """The schedule's mutable state during a fit (``PolicyState`` in a checkpoint)."""

    schedule: Schedule
    horizon: int
    best_score: float | None = None
    best_step: int | None = None
    wait_count: int = 0
    stop_score: float | None = None
    best_model: dict[str, torch.Tensor] | None = None
    history: list[float] = field(default_factory=list)

    @classmethod
    def restored(cls, schedule: Schedule, state: PolicyState) -> _Policy:
        """The policy of a resumed run (the monitor must be the same, C10).

        Raises:
            ConfigError: if the checkpoint selected on another monitor or mode.
        """
        monitor = schedule.monitor
        if (state["monitor"], state["mode"]) != (monitor.name, monitor.mode):
            msg = (
                f"the checkpoint selected on {state['monitor']} ({state['mode']}), this run on "
                f"{monitor.name} ({monitor.mode}); resume with the same monitor"
            )
            raise ConfigError(msg)
        return cls(
            schedule=schedule,
            horizon=schedule.horizon.resumed(state["horizon"]),
            best_score=state["best_score"],
            best_step=state["best_step"],
            wait_count=state["wait_count"],
            stop_score=state.get("stop_score"),
            best_model=state["best_model"],
            history=[float(value) for value in state.get("history", [])],
        )

    def state(self) -> PolicyState:
        """The checkpoint entry (plain values; the best weights are on the CPU already)."""
        state: PolicyState = {
            "monitor": self.schedule.monitor.name,
            "mode": self.schedule.monitor.mode,
            "horizon": self.horizon,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "wait_count": self.wait_count,
            "stop_score": self.stop_score,
            "best_model": self.best_model,
        }
        if self.schedule.stop_rule is not None:
            # Only with a stop rule: the carrasco2021 checkpoints keep their keys.
            state["history"] = list(self.history)
        return state

    def stopped(self, steps_done: int) -> bool:
        """Whether the stop rule has stopped the run at ``steps_done`` (a resumed run).

        A ``last`` checkpoint on the validation grid follows that step's validation, so its
        history ends with the value of its step and the rule decides again exactly as it did
        before the checkpoint. Off the grid (the final ``last`` of a run that ended at its
        horizon) the rule never stopped there: the horizon ends the run.
        """
        rule = self.schedule.stop_rule
        if rule is None or not self.history or not self.schedule.validation.due(steps_done):
            return False
        return rule.decide(steps_done, self.history).stop

    def observe(self, value: float, steps_done: int, module: nn.Module) -> tuple[bool, bool]:
        """Update after a validation of value ``value``; ``(improved, stop)``.

        A strict improvement keeps the weights (when ``best`` is kept) and extends the
        horizon (as Lightning's ``HorizonCallback`` did); early stopping, when configured,
        counts on its own best (``EarlyStopping``).
        """
        schedule = self.schedule
        improved = schedule.monitor.improves(value, self.best_score)
        if improved:
            self.best_score = value
            self.best_step = steps_done
            if "best" in schedule.keep:
                self.best_model = _cpu_state(module)
            self.horizon = schedule.horizon.extended(self.horizon, steps_done)
        stop = False
        if schedule.early_stop is not None:
            self.stop_score, self.wait_count, stop = schedule.early_stop.update(
                value, self.stop_score, self.wait_count, schedule.monitor.mode
            )
        if schedule.stop_rule is not None:
            self.history.append(value)
            stop = schedule.stop_rule.decide(steps_done, self.history).stop
        return improved, stop


class _Fit:
    """One fit's state and steps (:func:`fit` builds and drives it)."""

    def __init__(
        self,
        task: Task,
        train: TrainStream,
        validation: EvalSource,
        policy: _Policy,
        *,
        device: DevicePolicy,
        identity: IdentityState,
        run_dir: Path | None,
        steps_done: int,
        score: Score | None = None,
    ) -> None:
        self.task = task
        self.module = task.module
        self.train = train
        self.validation = validation
        self.policy = policy
        self.schedule = policy.schedule
        self.device = device
        self.identity = identity
        self.run_dir = run_dir
        self.steps_done = steps_done
        self.optimizer: torch.optim.Optimizer = task.make_optimizer()
        self.score = score
        self.validations: list[Validation] = []
        self.stopped_early = policy.stopped(steps_done)
        self._loss_sum: torch.Tensor | None = None
        self._loss_steps = 0
        self._versions = _versions()

    def done(self) -> bool:
        """Whether the run is over: the horizon is reached or early stopping fired."""
        return self.stopped_early or self.steps_done >= self.policy.horizon

    def step(self) -> None:
        """One optimizer step on stream batch ``steps_done`` (C2, C6)."""
        batch = self.train.batch(self.steps_done)
        loss = self.task.train_step(batch, self.steps_done)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        self.optimizer.step()
        self.task.after_step(self.steps_done)
        self.steps_done += 1
        # Summed on the device (no synchronisation); read at the next validation.
        detached = loss.detach()
        self._loss_sum = detached if self._loss_sum is None else self._loss_sum + detached
        self._loss_steps += 1

    def monitored_value(self) -> float:
        """The fit validation's monitored value (C2, C3, C7): ``loss_val``, or the score."""
        self.module.zero_grad(set_to_none=True)
        with _evaluating(self.module), torch.no_grad():
            outputs = [self.task.eval_step(batch) for batch in self.validation]
        if self.score is not None:
            return float(self.score(outputs))
        return float(mean_of_batch_losses([output.loss for output in outputs]))

    def _train_loss(self) -> float | None:
        if self._loss_sum is None:
            return None
        value = float(self._loss_sum) / self._loss_steps
        self._loss_sum, self._loss_steps = None, 0
        return value

    def validate(self) -> Validation:
        """Validate, update the policies, write ``last`` if due (C4, C5, C10)."""
        value = self.monitored_value()
        improved, stop = self.policy.observe(value, self.steps_done, self.module)
        self.stopped_early = stop
        record = Validation(
            steps_done=self.steps_done,
            epoch=_validation_epoch(self.steps_done, self.train.batches_per_pass),
            value=value,
            improved=improved,
            horizon=self.policy.horizon,
            wait_count=self.policy.wait_count,
            train_loss=self._train_loss(),
        )
        self.validations.append(record)
        every = self.schedule.checkpoint_every_n_steps
        if every is not None and self.steps_done > 0 and self.steps_done % every == 0:
            # After the policy update, so a resume owes no validation (as HorizonCallback);
            # never at step 0 (a fresh fit's start validation): there is nothing to resume.
            self.save_last()
        return record

    def checkpoint(self) -> Checkpoint:
        """The whole training state now (C10)."""
        return {
            "format_version": CHECKPOINT_FORMAT,
            "engine": "plain",
            "identity": self.identity,
            "steps_done": self.steps_done,
            "model": _cpu_state(self.module),
            "optimizer": self.optimizer.state_dict(),
            "task": self.task.state(),
            "policy": self.policy.state(),
            "stream": self.train.state(),
            "rng": capture_rng(self.device),
            "numerics": numerics_record(self.device),
            "versions": self._versions,
        }

    def save_last(self) -> None:
        """Write ``ckpt/last.pt`` (nothing without a run directory)."""
        if self.run_dir is not None:
            save_checkpoint(self.run_dir / LAST_CHECKPOINT, self.checkpoint())

    def save_best(self) -> None:
        """Write ``ckpt/best.pt`` if there are best weights (and a run directory)."""
        policy = self.policy
        if self.run_dir is None or policy.best_model is None:
            return
        if policy.best_step is None or policy.best_score is None:  # pragma: no cover
            raise RuntimeError("best weights without a best step: a policy bug")
        best: BestWeights = {
            "format_version": CHECKPOINT_FORMAT,
            "engine": "plain",
            "kind": "best",
            "identity": self.identity,
            "steps_done": policy.best_step,
            "monitor": self.schedule.monitor.name,
            "score": policy.best_score,
            "model": policy.best_model,
            "numerics": numerics_record(self.device),
            "versions": self._versions,
        }
        save_checkpoint(self.run_dir / BEST_CHECKPOINT, best)


def _check_schedule(schedule: Schedule, score: Score | None) -> None:
    if score is None and schedule.monitor.name != MONITORED:
        msg = (
            f"the plain loop computes {MONITORED} only, got the monitor "
            f"{schedule.monitor.name!r}: pass the score that computes it"
        )
        raise ConfigError(msg)


def _no_identity() -> IdentityState:
    return {"run_id": None, "attempt": None, "split_run": None}


def _resumed(checkpoint: Checkpoint, task: Task) -> Callable[[_Fit], None]:
    """Load ``model`` into the CPU module now; return the rest of the resume order (C10)."""
    task.module.load_state_dict(checkpoint["model"])

    def finish(run: _Fit) -> None:
        run.optimizer.load_state_dict(checkpoint["optimizer"])
        task.load_state(checkpoint["task"])
        run.train.resume(checkpoint["stream"], checkpoint["steps_done"])
        restore_rng(checkpoint["rng"])  # last: nothing in between may consume it

    return finish


def _progress_line(record: Validation, policy: _Policy) -> str:
    best = "best" if record.improved else f"best {policy.best_step}"
    name = policy.schedule.monitor.name
    return (
        f"step {record.steps_done} epoch {record.epoch} {name} {record.value:.6g} "
        f"({best}) horizon {record.horizon}"
    )


def _log(log: ScalarLog | None, record: Validation, started: float, name: str) -> None:
    if log is None:
        return
    log.write(
        {
            "step": record.steps_done,
            "epoch": record.epoch,
            name: record.value,
            "loss_train": record.train_loss,
            "improved": record.improved,
            "horizon": record.horizon,
            "wait_count": record.wait_count,
            "seconds": time.perf_counter() - started,
        }
    )


def fit(
    build: Callable[[], Task],
    train: TrainStream,
    validation: EvalSource,
    *,
    schedule: Schedule,
    device: DevicePolicy,
    seed: int,
    run_dir: Path | None = None,
    identity: IdentityState | None = None,
    resume: Path | None = None,
    on_validation: Callable[[Validation, Task], None] | None = None,
    on_step: Callable[[int], None] | None = None,
    progress: IO[str] | None = None,
    score: Score | None = None,
) -> tuple[Task, FitResult]:
    """Train the task ``build()`` makes until its horizon (or early stop); return it and the result.

    ``build`` is called once, after the precision, the flags and the seed are set, and must
    build the task on the CPU (C1); ``train`` and ``validation`` serve batches on
    ``device`` (their construction draws no random numbers). ``identity`` is stored in every
    checkpoint and checked on resume (default: no run id, no split run). ``resume`` is a
    ``ckpt/last.pt`` to continue from. ``on_validation(record, task)`` runs after each fit
    validation, after its checkpoint (the module holds the weights of that step; it must
    not change them). ``on_step(steps_done)`` runs after each optimizer step, before its
    validation (the preflight's throughput timer). ``progress`` gets one line per validation
    (default: nothing). ``score`` computes the monitored value from a validation's outputs
    (default: ``loss_val``, the mean of the batch losses); the schedule's monitor names it.

    The module ends with the **last** weights; :func:`evaluate_kept` restores the best.

    Raises:
        ConfigError: for a monitor other than ``loss_val`` without a ``score``; on resume,
            a checkpoint the
            plain engine cannot continue (:func:`~.checkpoint.check_resumable`), of another
            run (:func:`~.checkpoint.check_identity`), precision or determinism policy
            (:func:`~.checkpoint.check_numerics`) or monitor.
        DataError: on resume, a checkpoint of another training stream.
        ValueError: if the validation source is empty.
    """
    _check_schedule(schedule, score)
    current = _no_identity() if identity is None else identity
    checkpoint = None if resume is None else load_checkpoint(resume)
    if checkpoint is not None:
        check_identity(checkpoint["identity"], current)
        check_numerics(checkpoint["numerics"], numerics_record(device))
    device.apply()
    seed_everything(seed)
    task = build()
    finish = None if checkpoint is None else _resumed(checkpoint, task)
    task.module.to(device.device)
    if checkpoint is None:
        policy = _Policy(schedule, schedule.horizon.initial)
        start = 0
    else:
        policy = _Policy.restored(schedule, checkpoint["policy"])
        start = checkpoint["steps_done"]
    run = _Fit(
        task,
        train,
        validation,
        policy,
        device=device,
        identity=current,
        run_dir=run_dir,
        steps_done=start,
        score=score,
    )
    if finish is not None:
        finish(run)
    return task, _drive(run, resume, on_validation, on_step, progress)


def _reporter(
    run: _Fit,
    log: ScalarLog | None,
    started: float,
    on_validation: Callable[[Validation, Task], None] | None,
    progress: IO[str] | None,
) -> Callable[[], None]:
    """One fit validation and its reports: ``scalars.jsonl``, the progress line, the hook."""
    name = run.schedule.monitor.name

    def validate() -> None:
        record = run.validate()
        _log(log, record, started, name)
        if progress is not None:
            print(_progress_line(record, run.policy), file=progress, flush=True)
        if on_validation is not None:
            on_validation(record, run.task)

    return validate


def _drive(
    run: _Fit,
    resume: Path | None,
    on_validation: Callable[[Validation, Task], None] | None,
    on_step: Callable[[int], None] | None,
    progress: IO[str] | None,
) -> FitResult:
    """The loop proper: steps, validations, the final checkpoints (C4, C10)."""
    start = run.steps_done
    started = time.perf_counter()
    log = None
    if run.run_dir is not None:
        log = ScalarLog.open(
            run.run_dir / SCALARS_FILE, resume_from=None if resume is None else start
        )
    validate = _reporter(run, log, started, on_validation, progress)
    try:
        if run.schedule.validation.at_start and run.steps_done == 0:
            validate()  # the initial weights (a fresh fit: a checkpoint is never at 0)
        while not run.done():
            run.step()
            if on_step is not None:
                on_step(run.steps_done)
            if run.schedule.validation.due(run.steps_done):
                validate()
        seconds = time.perf_counter() - started
        if run.steps_done > start and run.schedule.checkpoint_every_n_steps is not None:
            run.save_last()  # the last weights, at the end of training (as HorizonCallback)
        run.save_best()
    finally:
        if log is not None:
            log.close()
    policy = run.policy
    return FitResult(
        last_step=run.steps_done,
        horizon=policy.horizon,
        best_step=policy.best_step,
        best_score=policy.best_score,
        batches_per_pass=run.train.batches_per_pass,
        stopped_early=run.stopped_early,
        validations=tuple(run.validations),
        resumed_from=resume,
        steps=run.steps_done - start,
        fit_seconds=seconds,
        best_model=policy.best_model,
        history=tuple(policy.history),
    )


def evaluate(task: Task, sources: Mapping[str, EvalSource]) -> dict[str, Evaluation]:
    """One pass of each source with the module's current weights, under ``inference_mode``.

    The module is in ``eval()`` for the passes and its submodules' modes are restored
    after (C3). The results are keyed like ``sources``.

    Raises:
        ValueError: if a source is empty.
    """
    module = task.module
    results: dict[str, Evaluation] = {}
    with _evaluating(module), torch.inference_mode():
        for name, source in sources.items():
            outputs = [task.eval_step(batch) for batch in source]
            loss = mean_of_batch_losses([output.loss for output in outputs])
            results[name] = Evaluation(
                loss=float(loss),
                probabilities=torch.cat([o.probabilities for o in outputs]).cpu(),
                targets=torch.cat([o.targets for o in outputs]).cpu(),
            )
    return results


def evaluate_kept(
    task: Task, result: FitResult, sources: Mapping[str, EvalSource], keep: tuple[Keep, ...]
) -> dict[Keep, dict[str, Evaluation]]:
    """The final evaluation of the weights the schedule keeps (C9).

    ``stop`` evaluates the module as :func:`fit` left it (the last weights) and runs first;
    ``best`` loads the best weights into the module, then evaluates (Lightning's
    ``validate``/``test(ckpt_path=best)``). The module ends with the best weights when
    ``best`` is kept.

    Raises:
        DataError: if ``best`` is kept but the fit has no best weights (no validation ran).
        ValueError: if a source is empty.
    """
    results: dict[Keep, dict[str, Evaluation]] = {}
    if "stop" in keep:
        results["stop"] = evaluate(task, sources)
    if "best" in keep:
        if result.best_model is None:
            raise DataError("no best weights to evaluate (did any validation run?)")
        task.module.load_state_dict(result.best_model)
        results["best"] = evaluate(task, sources)
    return results
