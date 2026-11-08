"""Thin Lightning adapters of the RJ23 regime: ``train-supervised --preset rj23-<scenario>``.

Task 3.4b. The rules live in engine-free objects (the phase-4 rule): the schedule policy,
the class-uniform sampler, Keras' Adam and the evaluation in :mod:`astro_simclr.rj23_regime`,
the parts of a run in :mod:`astro_simclr.rj23_data`, the classifiers in
:mod:`astro_simclr.rj23_model`; what needs no engine (the request, the inputs, the stream
check, the records and ``metrics.json``) is :mod:`astro_simclr.tasks.rj23`'s, shared with
the plain engine's adapter there, so both engines write the same files. This module only
wires them into Lightning:

- :class:`RJ23Module`: the scenario's :class:`~astro_simclr.rj23_model.RJ23Classifier`,
  unweighted cross-entropy, :func:`~astro_simclr.rj23_regime.keras_adam` with the
  configured learning rate. It logs ``loss_train`` only; validation is the callback's.
- :class:`RJ23DataModule`: every part of the run resident on the training device (as the
  ``carrasco2021`` schedule's ``DeviceTensorDataModule``); the training loader yields batch
  ``s`` of the :class:`~astro_simclr.rj23_regime.ClassUniformStream` at global step
  ``s``, so the stored position of a resume is the step count. Its checkpoint state is the
  stream definition and the run identity (splits SHA-256, mode, scenario); a resume with
  other data, batch size, seed or splits is refused.
- :class:`RJ23ScheduleCallback`: Lightning's own validation loop is off; the callback
  validates after step 0 (``on_train_start``) and after every step on the policy's grid
  (``on_train_batch_end``): the macro-F1 of the selection part
  (:func:`~astro_simclr.protocol.benchmark_metrics.macro_f1`), appended to the history and
  passed to :meth:`~astro_simclr.rj23_regime.RJ23Schedule.decide`. It snapshots the weights
  the decision keeps (``best``, ``stop``), stops the Trainer, and writes ``last.ckpt``
  after the validation of every ``checkpoint_every_n_steps`` steps and at the stop. Its
  state (history and snapshots) is in ``last.ckpt``.
- :func:`run_rj23`: one run end to end (below).

**A run** (:func:`run_rj23`; ``docs/architecture.md``, "training.rj23"): verify the object
splits against the alert table and the layout, build the parts of the protocol
(``replication``: train on Train, stop on the whole Validation, evaluate Validation and
Test; ``decision``, ``--screen``: stop on the selection half, evaluate both halves, Test
never read), seed, train until the policy stops, then evaluate **both** kept weights on the
evaluated parts. Outputs under ``output_dir``::

    checkpoints/last.ckpt          Lightning's full state (what --resume continues from)
    checkpoints/stop.ckpt          the weights at the stop     (plain torch dicts, read
    checkpoints/best.ckpt          the weights at the best      with weights_only=True)
    predictions_<stage>.npz        the primary keep's (rj23_keep) per-alert probabilities,
                                   index = candid (val / test, or selection / decision)
    predictions_<stage>_<keep>.npz the other keep's
    metrics.json                   below

``metrics.json`` follows the benchmark-run contract of ``bench-report`` (``seed``,
``machine``, ``benchmark.splits_sha256``, ``predictions.<stage>.file``) and adds the
run's identity (``run``: run id, attempt, config hash; ``manifest.json`` is written as for
every run, ADR 0015, with the object splits hashed as the run's splits), the
schedule's record (``schedule``: the history of ``[step, macro-F1]``, ``stopped_by``
``patience`` or ``cap``, ``capped``, the stop and best steps), both keeps' evaluations
(``keep``), the ``preset`` report and the data, config, versions and timing.

**Resume** (``--resume PATH|auto``) restores the weights, the optimizer, the step and the
callback's history and snapshots, and the batch stream continues exactly. Lightning does not
restore the global torch generator, so the dropout masks after a resume differ from an
uninterrupted run's (the plain engine's checkpoint holds every RNG stream: its resume is
exact, ``tasks.rj23``). An interrupted run (Ctrl-C) ends with an error that says to resume it.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback

from astro_simclr.config import RJ23, RJ23_MONITOR, ConfigError, Settings
from astro_simclr.data.rj23_input import ViewName
from astro_simclr.data.schema import DataError
from astro_simclr.experiments import (
    CHECKPOINT_DIR,
    LAST_CHECKPOINT,
    METRICS_FILE,
    check_accelerator,
    machine_record,
    resolve_resume,
)
from astro_simclr.protocol.benchmark_metrics import macro_f1
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.rj23_data import RJ23Part
from astro_simclr.rj23_model import BNPosition, build_rj23_classifier
from astro_simclr.rj23_regime import (
    KEEPS,
    ClassUniformStream,
    Keep,
    RJ23Decision,
    RJ23Schedule,
    keras_adam,
    predict_probabilities,
)
from astro_simclr.runs.record import start_run
from astro_simclr.tasks.rj23 import (
    RUN_IDENTITY_STATE,
    DevicePart,
    RJ23Inputs,
    RJ23Request,
    check_stream_state,
    keep_record,
    regime_schedule,
    rj23_inputs,
    rj23_metrics,
    schedule_record,
)
from astro_simclr.training.runs import apply_fp32_precision, build_trainer, seed_run

__all__ = [
    "RUN_IDENTITY_STATE",
    "RJ23DataModule",
    "RJ23Module",
    "RJ23Request",
    "RJ23ScheduleCallback",
    "run_rj23",
]

#: A batch: images (B, C, H, W), positions (B, 3), int64 labels (B,).
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class RJ23Module(L.LightningModule):
    """RJ23's classifier for ``scenario`` with its training step and Keras' Adam.

    The weights are drawn at construction from the global torch generator (seed first).

    Raises:
        ValueError: as :func:`~astro_simclr.rj23_model.build_rj23_classifier`, or for a
            non-positive ``lr`` (when the optimizer is built).
    """

    def __init__(
        self,
        *,
        scenario: ViewName,
        n_classes: int,
        bn_position: BNPosition = "after_concat",
        lr: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = build_rj23_classifier(scenario, n_classes=n_classes, bn_position=bn_position)

    def forward(self, images: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W)`` views and ``(N, 3)`` positions -> ``(N, n_classes)`` logits."""
        logits: torch.Tensor = self.model(images, positions)
        return logits

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """The mean, unweighted cross-entropy of ``batch`` (RJ23: the batches are balanced)."""
        images, positions, labels = batch
        loss = F.cross_entropy(self(images, positions), labels)
        self.log("loss_train", loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Keras' Adam (epsilon 1e-7, Keras' placement) at the configured rate."""
        return keras_adam(self.parameters(), self.lr)


class _ClassUniformLoader:
    """The training batches of one logging epoch, from the Trainer's global step on."""

    def __init__(
        self,
        part: DevicePart,
        stream: ClassUniformStream,
        position: Callable[[], int],
        identity: Mapping[str, object],
    ) -> None:
        self._part = part
        self._stream = stream
        self._position = position
        self._identity = dict(identity)

    def __len__(self) -> int:
        return self._stream.batches_per_pass

    def __iter__(self) -> Iterator[Batch]:
        start = self._position()
        end = (start // len(self) + 1) * len(self)
        device = self._part.images.device
        for batch in range(start, end):
            index = torch.from_numpy(self._stream.batch(batch)).to(device)
            yield self._part.images[index], self._part.positions[index], self._part.labels[index]

    def state_dict(self) -> dict[str, Any]:
        """The stream definition and the run identity (in every checkpoint)."""
        return {**self._stream.state(), RUN_IDENTITY_STATE: dict(self._identity)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Refuse a resume of another run; the position is the global step.

        Raises:
            ConfigError: if the checkpoint is from another splits file, mode or scenario.
            DataError: if its stream (items, class counts, batch size, seed) differs.
        """
        check_stream_state(state, self._stream, self._identity)


class RJ23DataModule(L.LightningDataModule):
    """The parts of an RJ23 run on the training device, and the class-uniform stream.

    ``parts`` are :func:`~astro_simclr.rj23_data.rj23_parts`'s; ``train`` names the part
    trained on. ``identity`` is kept in the training loader's checkpoint state. No
    validation or test loaders: :class:`RJ23ScheduleCallback` and :func:`run_rj23`
    evaluate :meth:`part` directly.

    Raises:
        ValueError: at construction, if ``train`` is not one of ``parts`` or the stream
            rejects its labels, batch size or seed.
        DataError: from :meth:`part` or :meth:`train_dataloader` before ``setup``.
    """

    def __init__(
        self,
        parts: Mapping[str, RJ23Part],
        *,
        train: str,
        batch_size: int,
        seed: int,
        n_classes: int,
        identity: Mapping[str, object],
    ) -> None:
        super().__init__()
        if train not in parts:
            raise ValueError(f"no part {train!r} to train on; parts: {sorted(parts)}")
        self._parts = dict(parts)
        self._train_name = train
        self._identity = dict(identity)
        self.stream = ClassUniformStream(parts[train].labels, batch_size, seed, n_classes=n_classes)
        self._device_parts: dict[str, DevicePart] | None = None

    def _device(self) -> torch.device:
        trainer = self.trainer
        return torch.device("cpu") if trainer is None else trainer.strategy.root_device

    def setup(self, stage: str) -> None:
        """Move every part to the Trainer's device (once per device)."""
        device = self._device()
        current = self._device_parts
        if current is None or current[self._train_name].images.device != device:
            self._device_parts = {
                name: DevicePart.of(part, device) for name, part in self._parts.items()
            }

    def part(self, name: str) -> DevicePart:
        """The device-resident part ``name``.

        Raises:
            DataError: before ``setup``, or for an unknown part.
        """
        if self._device_parts is None or name not in self._device_parts:
            raise DataError(f"RJ23DataModule has no part {name!r} (has setup() run?)")
        return self._device_parts[name]

    def _global_step(self) -> int:
        trainer = self.trainer
        return 0 if trainer is None else trainer.global_step

    def train_dataloader(self) -> _ClassUniformLoader:
        """The class-uniform stream from the current global step (module docstring)."""
        return _ClassUniformLoader(
            self.part(self._train_name), self.stream, self._global_step, self._identity
        )


def _clone(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in module.state_dict().items()}


class RJ23ScheduleCallback(Callback):
    """Apply :class:`~astro_simclr.rj23_regime.RJ23Schedule` to a Lightning fit.

    Args:
        schedule: the policy.
        datamodule: the run's data module (the selection part is read from it).
        selection: the part whose macro-F1 is monitored (``Validation``, or the
            ``selection`` half).
        n_classes: the classes of the macro-F1.
        eval_batch_size: rows per evaluation batch.
        last_checkpoint: where ``last.ckpt`` is written.
        last_every_n_steps: write it after the validations of the steps that are multiples
            of this (a multiple of the validation interval), and at the stop; None: only
            at the stop.

    After the fit, :attr:`decision` is the last decision, :attr:`history` the monitored
    values, :attr:`snapshots` the kept weights (``stop`` and ``best``, on the training
    device) and :attr:`snapshot_steps` their steps.

    Raises:
        ConfigError: on resume, if the checkpoint ran another schedule or selection part.
    """

    def __init__(
        self,
        schedule: RJ23Schedule,
        datamodule: RJ23DataModule,
        *,
        selection: str,
        n_classes: int,
        eval_batch_size: int,
        last_checkpoint: Path,
        last_every_n_steps: int | None = None,
    ) -> None:
        if last_every_n_steps is not None and last_every_n_steps % schedule.eval_every:
            msg = (
                f"last_every_n_steps ({last_every_n_steps}) must be a multiple of the "
                f"validation interval ({schedule.eval_every})"
            )
            raise ValueError(msg)
        self.schedule = schedule
        self.datamodule = datamodule
        self.selection = selection
        self.n_classes = n_classes
        self.eval_batch_size = eval_batch_size
        self.last_checkpoint = Path(last_checkpoint)
        self.last_every_n_steps = last_every_n_steps
        self.history: list[float] = []
        self.decision: RJ23Decision | None = None
        self.snapshots: dict[Keep, dict[str, torch.Tensor]] = {}
        self.snapshot_steps: dict[Keep, int] = {}
        #: The global step at which this fit started (0, or a resume's step); None if
        #: Lightning skipped the fit (a resume at ``max_steps``).
        self.start_step: int | None = None

    @property
    def stopped(self) -> bool:
        """Whether the policy has stopped the run."""
        return self.decision is not None and self.decision.stop

    def _evaluate(self, trainer: L.Trainer, pl_module: L.LightningModule) -> float:
        part = self.datamodule.part(self.selection)
        model = cast(RJ23Module, pl_module).model
        probabilities = predict_probabilities(
            model, part.images, part.positions, self.eval_batch_size
        )
        predicted = probabilities.argmax(dim=1).numpy()
        return macro_f1(part.part.labels, predicted, self.n_classes)

    def _validate(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        step = trainer.global_step
        value = self._evaluate(trainer, pl_module)
        self.history.append(value)
        decision = self.schedule.decide(step, self.history)
        self.decision = decision
        model = cast(RJ23Module, pl_module).model
        for keep in sorted(decision.keep):
            self.snapshots[keep] = _clone(model)
            self.snapshot_steps[keep] = step
        for logger in trainer.loggers:
            logger.log_metrics({RJ23_MONITOR: value}, step=step)
        if decision.stop:
            trainer.should_stop = True
        every = self.last_every_n_steps
        if step > 0 and (decision.stop or (every is not None and step % every == 0)):
            # After this step's validation and the snapshots: a resume owes nothing.
            trainer.save_checkpoint(self.last_checkpoint)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Validate the initial weights (step 0); a resumed run that had stopped ends."""
        self.start_step = trainer.global_step
        if not self.history and trainer.global_step == 0:
            self._validate(trainer, pl_module)
        elif self.stopped:
            trainer.should_stop = True

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """After the optimizer step: validate when the step is on the policy's grid."""
        if self.schedule.validates(trainer.global_step):
            self._validate(trainer, pl_module)

    def state_dict(self) -> dict[str, Any]:
        """The schedule, the history and the kept weights (on the CPU), for ``last.ckpt``."""
        return {
            "schedule": dataclasses.asdict(self.schedule),
            "selection": self.selection,
            "history": list(self.history),
            "snapshots": {
                keep: {key: tensor.cpu() for key, tensor in state.items()}
                for keep, state in self.snapshots.items()
            },
            "snapshot_steps": dict(self.snapshot_steps),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a resumed run's history and kept weights.

        Raises:
            ConfigError: if the checkpoint ran another schedule or selection part.
        """
        saved = (state_dict["schedule"], state_dict["selection"])
        if saved != (dataclasses.asdict(self.schedule), self.selection):
            msg = (
                f"the checkpoint ran the schedule {saved[0]} on {saved[1]!r}, this run "
                f"{dataclasses.asdict(self.schedule)} on {self.selection!r}"
            )
            raise ConfigError(msg)
        self.history = [float(value) for value in state_dict["history"]]
        self.snapshots = dict(state_dict["snapshots"])
        self.snapshot_steps = {k: int(v) for k, v in state_dict["snapshot_steps"].items()}
        if self.history:
            last = self.schedule.validation_step(len(self.history) - 1)
            self.decision = self.schedule.decide(last, self.history)


def _evaluate_keeps(
    module: RJ23Module,
    datamodule: RJ23DataModule,
    callback: RJ23ScheduleCallback,
    inputs: RJ23Inputs,
    settings: Settings,
) -> dict[str, Any]:
    """Predict the evaluated parts with both kept weights; write the weights and files."""
    out = settings.output_dir
    primary = cast(Keep, settings.rj23_keep)
    result: dict[str, Any] = {"primary": primary}
    for keep in KEEPS:
        weights = callback.snapshots[keep]
        checkpoint = f"{CHECKPOINT_DIR}/{keep}.ckpt"
        torch.save(
            {
                "keep": keep,
                "step": callback.snapshot_steps[keep],
                "hyper_parameters": dict(module.hparams),
                "state_dict": {key: tensor.cpu() for key, tensor in weights.items()},
            },
            out / checkpoint,
        )
        module.model.load_state_dict(weights)
        probabilities: dict[str, np.ndarray[Any, Any]] = {}
        for part_name, _ in inputs.plan.evaluated:
            part = datamodule.part(part_name)
            module.model.to(part.images.device)
            probabilities[part_name] = predict_probabilities(
                module.model, part.images, part.positions, settings.eval_batch_size
            ).numpy()
        result[keep] = keep_record(
            out,
            keep,
            primary,
            inputs,
            probabilities,
            step=callback.snapshot_steps[keep],
            checkpoint=checkpoint,
        )
    return result


def run_rj23(
    settings: Settings,
    request: RJ23Request,
    *,
    resume: str | None = None,
    callbacks: Sequence[Callback] = (),
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Train and evaluate one RJ23 run (module docstring); return and write its metrics.

    ``settings`` must have ``schedule: rj23`` (an ``rj23-<scenario>`` preset), with
    ``data_dir / dataset`` a ``raw-alerts`` layout. ``resume`` is ``None``, ``"auto"`` or a
    checkpoint path, as for ``experiments.run_command``.

    Raises:
        ConfigError: for another schedule, an unavailable accelerator, a missing
            ``--resume`` checkpoint, a resume of another run, or a fit that ended before
            the schedule stopped it (interrupted: resume it).
        DataError: if the layout is missing or not ``raw-alerts``, the alert table does
            not fit it, a part is empty, or the splits file is malformed or not of this
            table (``SplitsError``).
    """
    if settings.schedule != RJ23:
        raise ConfigError(f"run_rj23 needs schedule={RJ23}, got {settings.schedule}")
    check_accelerator(settings.accelerator)
    apply_fp32_precision(settings)
    inputs = rj23_inputs(settings, request)
    ckpt_path = resolve_resume(resume, settings.output_dir)
    plan, n_classes = inputs.plan, inputs.data.n_classes

    seed_run(settings)
    module = RJ23Module(
        scenario=cast(ViewName, settings.rj23_scenario),
        n_classes=n_classes,
        bn_position=cast(BNPosition, settings.rj23_bn_position),
        lr=settings.lr,
    )
    datamodule = RJ23DataModule(
        inputs.parts,
        train=plan.train,
        batch_size=settings.batch_size,
        seed=settings.seed,
        n_classes=n_classes,
        identity=inputs.identity,
    )
    out = settings.output_dir
    (out / CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    callback = RJ23ScheduleCallback(
        regime_schedule(settings),
        datamodule,
        selection=plan.selection,
        n_classes=n_classes,
        eval_batch_size=settings.eval_batch_size,
        last_checkpoint=out / CHECKPOINT_DIR / LAST_CHECKPOINT,
        last_every_n_steps=settings.checkpoint_every_n_steps,
    )
    trainer = build_trainer(
        settings,
        callbacks=[callback, *callbacks],
        enable_checkpointing=False,
        enable_progress_bar=progress_bar,
        max_epochs=-1,
        max_steps=settings.max_steps,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    decision = request.mode == "decision"
    record = start_run(
        "train-supervised",
        settings,
        data=inputs.path,
        machine=machine_record(trainer),
        split=SplitRequest(request.splits, None) if decision else None,
        object_splits=None if decision else request.splits,
        resumed=ckpt_path is not None,
    )
    try:
        cleaned = _fit_and_report(
            settings, request, trainer, (module, datamodule, callback), inputs, ckpt_path
        )
        cleaned["run"] = record.summary()
        text = json.dumps(cleaned, indent=2, allow_nan=False) + "\n"
        (out / METRICS_FILE).write_text(text, encoding="utf-8")
    except (KeyboardInterrupt, SystemExit):
        # Lightning turns a Ctrl-C during the fit into SystemExit.
        record.end("cancelled")
        raise
    except BaseException:
        record.end("failed")
        raise
    record.end("finished")
    return cleaned


def _fit_and_report(
    settings: Settings,
    request: RJ23Request,
    trainer: L.Trainer,
    built: tuple[RJ23Module, RJ23DataModule, RJ23ScheduleCallback],
    inputs: RJ23Inputs,
    ckpt_path: Path | None,
) -> dict[str, Any]:
    """Fit until the policy stops, evaluate both keeps; return the JSON-safe metrics.

    Raises:
        ConfigError: if the fit ended before the schedule stopped it (interrupted).
    """
    module, datamodule, callback = built
    out = settings.output_dir
    started = time.perf_counter()
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    seconds = time.perf_counter() - started
    if not callback.stopped:
        msg = (
            f"the run ended at step {trainer.global_step} before the {RJ23} schedule stopped "
            f"it (interrupted?): continue it with --resume auto (last.ckpt in {out})"
        )
        raise ConfigError(msg)
    keep = _evaluate_keeps(module, datamodule, callback, inputs, settings)
    # No on_train_start when Lightning skips the fit (a resume at the cap): no step ran.
    start = callback.start_step
    steps = 0 if start is None else trainer.global_step - start
    engine: dict[str, Any] = {
        "checkpoints": {
            "last": f"{CHECKPOINT_DIR}/{LAST_CHECKPOINT}",
            **{name: keep[name]["checkpoint"] for name in KEEPS},
        },
        "versions": {
            "astro_simclr": version("astro-simclr"),
            "torch": torch.__version__,
            "lightning": L.__version__,
        },
        "accelerator": type(trainer.accelerator).__name__,
        "machine": machine_record(trainer),
        "resumed_from": None if ckpt_path is None else str(ckpt_path),
        "timing": {
            "fit_seconds": seconds,
            "steps": steps,
            "seconds_per_step": seconds / steps if steps else None,
        },
    }
    return rj23_metrics(
        settings,
        request,
        inputs,
        schedule=schedule_record(callback.schedule, callback.selection, callback.history),
        keep=keep,
        last_step=trainer.global_step,
        engine=engine,
    )
