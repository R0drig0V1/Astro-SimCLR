"""The ``rj23`` schedule on the plain engine: its inputs, records and the run (task 3.4b).

Lightning-free (the engine-free rule, ADR 0014): the rules are
:mod:`astro_simclr.rj23_regime`'s policy objects (the schedule, the class-uniform stream,
Keras' Adam, the evaluation) and the parts are :mod:`astro_simclr.rj23_data`'s, used as
they are. This module holds the run's inputs and records and the plain engine's adapter:

- **Inputs and records** (the Lightning runner, removed in task 4.8, shared them):
  :class:`RJ23Request`,
  :func:`rj23_inputs` (the layout, the alert table, the object splits and the protocol's
  parts, verified), :func:`check_stream_state` (the resume check of the stream and the run
  identity), :func:`keep_record` (one kept weights' predictions files and evaluation),
  :func:`schedule_record` and :func:`rj23_metrics` (``metrics.json`` without the engine's
  entries). A plain run therefore writes the files a Lightning run wrote, with the same
  values.
- **The plain adapter**: :class:`RJ23Task` (the scenario's classifier, unweighted
  cross-entropy, :func:`~astro_simclr.rj23_regime.keras_adam`), :class:`RJ23TrainSource`
  (batch ``s`` of the :class:`~astro_simclr.rj23_regime.ClassUniformStream`, stream 10, at
  step ``s``), :class:`RJ23EvalSource` (a part in consecutive batches),
  :func:`macro_f1_score` (the monitored value) and :func:`rj23_schedule` (the engine's
  :class:`~astro_simclr.engine.Schedule`: ``macro_f1_val``/``max`` every
  ``val_every_n_steps`` from step 0, a fixed horizon at the cap, the
  :class:`~astro_simclr.rj23_regime.RJ23Schedule` as its stop rule, ``keep`` both).
- :func:`run_rj23_plain`: one run on the plain engine, ``tasks.plain.run_plain``'s path
  for ``schedule: rj23``.

**Batches.** A training or evaluation batch is ``(images, positions, labels)``: the
scenario's view ``(B, C, H, W)``, the positions on the unit sphere ``(B, 3)`` in the
features slot of the engine's :data:`~astro_simclr.engine.Batch`, and **int64** labels
``(B,)`` (not one-hot: the task and its sources go together, and the cross-entropy takes
class indices as Lightning's ``RJ23Module`` did).

**Plain outputs** under ``output_dir``: ``ckpt/last.pt`` (the whole training state, written
after the validations of every ``checkpoint_every_n_steps`` steps and at the stop: the
``stop`` weights), ``ckpt/best.pt`` (the ``best`` weights), ``scalars.jsonl``, the
predictions files and ``metrics.json`` as the Lightning run's, and ``manifest.json``
(``engine: plain``). A resume (``--resume auto``: ``ckpt/last.pt``) is exact, the dropout
masks included (every RNG stream is in the checkpoint, unlike Lightning's); a resumed run
that had stopped trains nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any, Final, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from astro_simclr.config import RJ23, RJ23_MONITOR, ConfigError, Settings, settings_to_plain
from astro_simclr.data.raw_alerts import RawAlerts, load_raw_alerts
from astro_simclr.data.rj23_input import LargeValues, ViewName
from astro_simclr.data.schema import DataError
from astro_simclr.engine import (
    Batch,
    DevicePolicy,
    EvalOutput,
    FitResult,
    Horizon,
    IdentityState,
    Monitor,
    Schedule,
    Task,
    TrainBatch,
    Validation,
    ValidationGrid,
    evaluate_kept,
    fit,
    recorded,
)
from astro_simclr.presets import preset_report
from astro_simclr.protocol.alerts import read_alert_table
from astro_simclr.protocol.benchmark_metrics import confusion_counts, macro_f1
from astro_simclr.protocol.object_splits import read_object_splits
from astro_simclr.protocol.predictions import Predictions, predictions_file, save_predictions
from astro_simclr.protocol.split_runs import SplitRequest, file_sha256
from astro_simclr.rj23_data import MODE_PARTS, MODES, RJ23Mode, RJ23Part, RunPlan, rj23_parts
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
from astro_simclr.runs.layout import BEST_CHECKPOINT, LAST_CHECKPOINT, METRICS_FILE
from astro_simclr.runs.record import QueueAttempt, start_run
from astro_simclr.tasks.common import check_accelerator, dataset_path
from astro_simclr.tasks.results import finite

__all__ = [
    "RUN_IDENTITY_STATE",
    "RJ23EvalSource",
    "RJ23Inputs",
    "RJ23Request",
    "RJ23Task",
    "RJ23TrainSource",
    "check_stream_state",
    "keep_record",
    "macro_f1_score",
    "rj23_inputs",
    "rj23_metrics",
    "rj23_schedule",
    "run_rj23_plain",
    "schedule_record",
]

#: The key of the run identity in the training stream's checkpoint state.
RUN_IDENTITY_STATE: Final = "rj23_run"
#: The engine of :func:`run_rj23_plain` (``manifest.json``, ``metrics.json["machine"]``).
ENGINE: Final = "plain"


@dataclass(frozen=True, slots=True)
class RJ23Request:
    """What an RJ23 run trains on besides its settings.

    Attributes:
        splits: the object-splits file (``make-object-splits``) of the layout's alerts.
        mode: ``replication`` (the paper protocol) or ``decision`` (``--screen``).
        alerts: the alert table CSV the splits were drawn from; None: the layout's own
            (grouped by ``oid``).

    Raises:
        ConfigError: for an unknown ``mode``.
    """

    splits: Path
    mode: RJ23Mode = "replication"
    alerts: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}, got {self.mode!r}")


@dataclass(frozen=True, slots=True)
class RJ23Inputs:
    """The verified inputs of one RJ23 run (:func:`rj23_inputs`).

    Attributes:
        path: the ``raw-alerts`` layout.
        data: its alerts.
        parts: the protocol's parts (``rj23_data.rj23_parts``).
        plan: which part trains, selects and is evaluated.
        identity: the run's identity in the stream state (splits SHA-256, mode, scenario).
    """

    path: Path
    data: RawAlerts
    parts: Mapping[str, RJ23Part]
    plan: RunPlan
    identity: Mapping[str, str]


def rj23_inputs(settings: Settings, request: RJ23Request) -> RJ23Inputs:
    """Read the layout and the alert table, verify the object splits, build the parts.

    Raises:
        ConfigError: for another schedule than ``rj23``.
        DataError: if the layout is missing or not ``raw-alerts``, the alert table does
            not fit it, a part is empty, or the splits file is malformed or not of this
            table (``SplitsError``).
    """
    if settings.schedule != RJ23:
        raise ConfigError(f"an RJ23 run needs schedule={RJ23}, got {settings.schedule}")
    path = dataset_path(settings)
    data = load_raw_alerts(path)
    splits = read_object_splits(request.splits)
    table = None if request.alerts is None else read_alert_table(request.alerts, data.label_names)
    scenario = cast(ViewName, settings.rj23_scenario)
    parts = rj23_parts(
        data,
        splits,
        scenario,
        mode=request.mode,
        table=table,
        large_values=cast(LargeValues, settings.rj23_large_values),
    )
    identity = {
        "splits_sha256": file_sha256(request.splits),
        "mode": request.mode,
        "scenario": scenario,
    }
    return RJ23Inputs(path, data, parts, MODE_PARTS[request.mode], identity)


def build_rj23_model(settings: Settings, n_classes: int) -> nn.Module:
    """The scenario's classifier, drawn from the global generator (call after seeding)."""
    return build_rj23_classifier(
        cast(ViewName, settings.rj23_scenario),
        n_classes=n_classes,
        bn_position=cast(BNPosition, settings.rj23_bn_position),
    )


def check_stream_state(
    state: Mapping[str, Any], stream: ClassUniformStream, identity: Mapping[str, object]
) -> None:
    """Refuse a resume of another run or stream (the training stream's checkpoint state).

    Raises:
        ConfigError: if the checkpoint is from another splits file, mode or scenario.
        DataError: if its stream (items, class counts, batch size, seed) differs.
    """
    stored = state.get(RUN_IDENTITY_STATE)
    if stored != dict(identity):
        msg = (
            f"the checkpoint is from the RJ23 run {stored}, this run is {dict(identity)}: "
            "a resume must continue the same run (use another --output-dir or no --resume)"
        )
        raise ConfigError(msg)
    saved = {key: value for key, value in state.items() if key != RUN_IDENTITY_STATE}
    if saved != stream.state():
        msg = (
            f"the checkpoint was trained on the stream {saved}, but this run's is "
            f"{stream.state()}: resume with the same data, batch_size and seed"
        )
        raise DataError(msg)


@dataclass(frozen=True, slots=True)
class DevicePart:
    """One part's arrays on the training device (and the part itself)."""

    images: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor
    part: RJ23Part

    @classmethod
    def of(cls, part: RJ23Part, device: torch.device | str) -> DevicePart:
        """``part``'s arrays as tensors on ``device``."""
        return cls(
            images=torch.from_numpy(np.ascontiguousarray(part.images)).to(device),
            positions=torch.from_numpy(np.ascontiguousarray(part.positions)).to(device),
            labels=torch.from_numpy(np.ascontiguousarray(part.labels)).to(device),
            part=part,
        )


class RJ23TrainSource:
    """The class-uniform training stream on the device, as the plain loop reads it.

    Batch ``s`` is :meth:`ClassUniformStream.batch` ``(s)`` gathered on the device: the
    batch the Lightning loader gave at global step ``s``. Its state is the stream definition and
    the run identity, checked on resume (:func:`check_stream_state`).
    """

    def __init__(
        self, part: DevicePart, stream: ClassUniformStream, identity: Mapping[str, object]
    ) -> None:
        self.part = part
        self.stream = stream
        self.identity = dict(identity)

    @property
    def batches_per_pass(self) -> int:
        """``ceil(n / batch_size)``: the derived epoch only (the stream has no passes)."""
        return self.stream.batches_per_pass

    def batch(self, step: int) -> TrainBatch:
        """``(images, positions, int64 labels)`` of stream batch ``step``."""
        index = torch.from_numpy(self.stream.batch(step)).to(self.part.images.device)
        return self.part.images[index], self.part.positions[index], self.part.labels[index]

    def state(self) -> dict[str, Any]:
        """The stream definition and the run identity (in every checkpoint)."""
        return {**self.stream.state(), RUN_IDENTITY_STATE: dict(self.identity)}

    def resume(self, state: dict[str, Any], steps_done: int) -> None:
        """Check the saved stream; the position is the step (a pure function of it).

        Raises:
            ConfigError: if the checkpoint is from another run.
            DataError: if it is of another stream.
        """
        check_stream_state(state, self.stream, self.identity)


class RJ23EvalSource:
    """A device part in consecutive batches of ``batch_size`` rows, the last one partial."""

    def __init__(self, part: DevicePart, batch_size: int) -> None:
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")
        self.part = part
        self.batch_size = batch_size

    def __len__(self) -> int:
        return -(-len(self.part.part) // self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        part, size = self.part, self.batch_size
        for start in range(0, len(part.part), size):
            stop = start + size
            yield part.images[start:stop], part.positions[start:stop], part.labels[start:stop]


class RJ23Task:
    """RJ23's classifier as an engine ``Task`` (Lightning's ``RJ23Module``, step for step).

    Built on the CPU from the global generator (``fit`` seeds first). ``eval_step`` gives
    :func:`~astro_simclr.rj23_regime.predict_probabilities` of its batch, the evaluation
    the Lightning runner used, and a loss computed from those probabilities (no run reports
    it: the monitored value is the macro-F1).

    Raises:
        ValueError: as :func:`~astro_simclr.rj23_model.build_rj23_classifier`, or for a
            non-positive ``lr`` (when the optimizer is built).
    """

    def __init__(self, settings: Settings, n_classes: int) -> None:
        self.net = build_rj23_model(settings, n_classes)
        self.lr = settings.lr

    @property
    def module(self) -> nn.Module:
        """The :class:`~astro_simclr.rj23_model.RJ23Classifier`."""
        return self.net

    def make_optimizer(self) -> torch.optim.Optimizer:
        """Keras' Adam (epsilon 1e-7, Keras' placement) at the configured rate."""
        return keras_adam(self.net.parameters(), self.lr)

    def train_step(self, batch: TrainBatch, step: int) -> torch.Tensor:
        """The mean, unweighted cross-entropy of the batch (the batches are balanced)."""
        images, positions, labels = batch[0], batch[1], batch[2]
        logits: torch.Tensor = self.net(images, positions)
        return F.cross_entropy(logits, labels)

    def after_step(self, step: int) -> None:
        """Nothing after the optimizer step: no scheduler, no weight EMA (ADR 0014, D3)."""

    def eval_step(self, batch: Batch) -> EvalOutput:
        """The batch's softmax outputs (on the CPU), labels and mean negative log-likelihood."""
        images, positions, labels = batch
        probabilities = predict_probabilities(self.net, images, positions, int(images.shape[0]))
        targets = labels.cpu()
        tiny = torch.finfo(probabilities.dtype).tiny
        loss = F.nll_loss(probabilities.clamp_min(tiny).log(), targets)
        return EvalOutput(loss=loss, probabilities=probabilities, targets=targets)

    def state(self) -> dict[str, Any]:
        """No state outside the weights: every draw is a function of the step."""
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Nothing to restore."""


def macro_f1_score(n_classes: int) -> Callable[[Sequence[EvalOutput]], float]:
    """The monitored value of a validation: the macro-F1 of the argmax predictions."""

    def score(outputs: Sequence[EvalOutput]) -> float:
        probabilities = torch.cat([output.probabilities for output in outputs])
        targets = torch.cat([output.targets for output in outputs]).numpy()
        return macro_f1(targets, probabilities.argmax(dim=1).numpy(), n_classes)

    return score


def regime_schedule(settings: Settings) -> RJ23Schedule:
    """The policy of ``settings`` (``val_every_n_steps``, ``patience``, ``max_steps``)."""
    return RJ23Schedule(
        eval_every=settings.val_every_n_steps,
        patience=settings.patience,
        max_steps=settings.max_steps,
    )


def rj23_schedule(settings: Settings) -> Schedule:
    """The ``rj23`` settings as the plain engine's :class:`~astro_simclr.engine.Schedule`.

    The validation grid starts at step 0, the horizon is the fixed cap (``increment``
    0), the :class:`~astro_simclr.rj23_regime.RJ23Schedule` decides the stop on the
    history, and both weights are kept. The loop's best (a strict improvement, NaN never)
    is :meth:`RJ23Schedule.improves`, so its best weights are the policy's ``best``.
    """
    return Schedule(
        monitor=Monitor(RJ23_MONITOR, "max"),
        validation=ValidationGrid(settings.val_every_n_steps, at_start=True),
        horizon=Horizon(settings.max_steps, 0),
        stop_rule=regime_schedule(settings),
        keep=KEEPS,
        checkpoint_every_n_steps=settings.checkpoint_every_n_steps,
    )


def _evaluation(
    labels: np.ndarray[Any, Any], predicted: np.ndarray[Any, Any], names: Sequence[str]
) -> dict[str, Any]:
    counts = confusion_counts(labels, predicted, len(names))
    rows = counts.sum(axis=1)
    recalls = {
        name: (float(counts[i, i] / rows[i]) if rows[i] else None) for i, name in enumerate(names)
    }
    present = [value for value in recalls.values() if value is not None]
    return {
        "n": int(labels.size),
        "accuracy": float(np.mean(labels == predicted)),
        "balanced_accuracy": sum(present) / len(present),
        "macro_f1": macro_f1(labels, predicted, len(names)),
        "recall_per_class": recalls,
        "confusion_counts": counts.tolist(),
    }


def keep_record(
    out: Path,
    keep: Keep,
    primary: Keep,
    inputs: RJ23Inputs,
    probabilities: Mapping[str, np.ndarray[Any, Any]],
    *,
    step: int,
    checkpoint: str,
) -> dict[str, Any]:
    """Write one kept weights' predictions files; return its ``metrics.json["keep"]`` entry.

    ``probabilities`` maps each evaluated part to its ``(N, C)`` softmax outputs. The
    primary keep's files are ``predictions_<stage>.npz`` (what ``bench-report`` reads),
    the other's ``predictions_<stage>_<keep>.npz``. ``checkpoint`` is the file of these
    weights, relative to ``out`` (each engine's own).
    """
    names = inputs.data.label_names
    evaluation: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for part_name, stage in inputs.plan.evaluated:
        part = inputs.parts[part_name]
        stored = Predictions(probabilities[part_name], part.labels, part.candids, len(names))
        file = predictions_file(stage if keep == primary else f"{stage}_{keep}")
        save_predictions(stored, out / file)
        predictions[stage] = {"file": file, "split": part_name, "n": len(part)}
        evaluation[stage] = _evaluation(part.labels, stored.predicted, names)
    return {
        "step": step,
        "checkpoint": checkpoint,
        "evaluation": evaluation,
        "predictions": predictions,
    }


def final_decision(schedule: RJ23Schedule, history: Sequence[float]) -> RJ23Decision:
    """The decision at the run's last validation (the stop, once the run has stopped)."""
    return schedule.decide(schedule.validation_step(len(history) - 1), history)


def schedule_record(
    schedule: RJ23Schedule, selection: str, history: Sequence[float]
) -> dict[str, Any]:
    """``metrics.json["schedule"]``: the policy, the stop and best steps, the history.

    Raises:
        ConfigError: if the history has not stopped (the fit was interrupted).
    """
    decision = final_decision(schedule, history)
    if not decision.stop or decision.best_step is None:
        raise ConfigError("the rj23 schedule has not stopped: the run was interrupted")
    return {
        "name": RJ23,
        "eval_every": schedule.eval_every,
        "patience": schedule.patience,
        "max_steps": schedule.max_steps,
        "selection": selection,
        "stopped_by": decision.reason,
        "capped": decision.capped,
        "stop_step": decision.step,
        "best_step": decision.best_step,
        "history": [[schedule.validation_step(i), value] for i, value in enumerate(history)],
    }


def _parts_record(parts: Mapping[str, RJ23Part], n_classes: int) -> dict[str, Any]:
    return {
        name: {
            "n": len(part),
            "class_counts": np.bincount(part.labels, minlength=n_classes).tolist(),
        }
        for name, part in parts.items()
    }


def _data_record(path: Path, data: RawAlerts) -> dict[str, Any]:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "kind": meta["kind"],
        "source": meta["source"],
        "label_names": list(data.label_names),
        "class_counts": data.class_counts(),
    }


def rj23_metrics(
    settings: Settings,
    request: RJ23Request,
    inputs: RJ23Inputs,
    *,
    schedule: dict[str, Any],
    keep: dict[str, Any],
    last_step: int,
    engine: Mapping[str, Any],
) -> dict[str, Any]:
    """``metrics.json`` of an RJ23 run, JSON-safe (non-finite floats are ``null``).

    ``schedule`` is :func:`schedule_record`'s, ``keep`` the ``primary`` choice and each
    :func:`keep_record`. ``engine`` holds what each engine records its own way
    (``checkpoints``, ``versions``, ``accelerator``, ``machine``, ``resumed_from``,
    ``timing``); the run's identity (``run``) is added by the runner.
    """
    data = inputs.data
    primary = keep[settings.rj23_keep]
    history = schedule["history"]
    best_value = next(value for step, value in history if step == schedule["best_step"])
    metrics: dict[str, Any] = {
        "command": "train-supervised",
        "seed": settings.seed,
        "best_step": schedule["best_step"],
        "stop_step": schedule["stop_step"],
        "last_step": last_step,
        "monitor": RJ23_MONITOR,
        "monitor_best": best_value,
        "labels": list(data.label_names),
        "schedule": schedule,
        "keep": keep,
        "evaluation": primary["evaluation"],
        "predictions": primary["predictions"],
        "benchmark": {
            "splits": str(request.splits),
            "splits_sha256": inputs.identity["splits_sha256"],
            "mode": request.mode,
            "alerts": None if request.alerts is None else str(request.alerts),
            "parts": _parts_record(inputs.parts, data.n_classes),
        },
        **engine,
        "data": _data_record(inputs.path, data),
        "config": settings_to_plain(settings),
        "preset": preset_report(settings),
    }
    return cast(dict[str, Any], finite(metrics))


def _split_arguments(request: RJ23Request) -> dict[str, Any]:
    """``start_run``'s split: a decision run is a ``screen``, a replication run ``whole``."""
    decision = request.mode == "decision"
    return {
        "split": SplitRequest(request.splits, None) if decision else None,
        "object_splits": None if decision else request.splits,
    }


def run_rj23_plain(
    settings: Settings,
    request: RJ23Request,
    *,
    resume: str | None = None,
    on_step: Callable[[int], None] | None = None,
    on_validation: Callable[[Validation, Task], None] | None = None,
    progress: IO[str] | None = None,
    queued: QueueAttempt | None = None,
) -> dict[str, Any]:
    """Train and evaluate one RJ23 run on the plain engine; return and write its metrics.

    The plain counterpart of the Lightning ``training.rj23.run_rj23`` (removed in task
    4.8; module docstring): the same inputs, the same batches, validations, stop and kept
    weights, the same files.
    ``resume`` is ``None``, ``"auto"`` (``ckpt/last.pt`` if any) or a ``ckpt/last.pt``
    path; the hooks are :func:`astro_simclr.engine.fit`'s. ``queued`` is a queued run's
    fixed attempt, machine class and run id (``runs.record``, "Queued runs"; task A3).

    Raises:
        ConfigError: for another schedule, an unavailable accelerator, more than one
            device or a Lightning dry-run limit, a missing ``--resume`` checkpoint, or a
            checkpoint the engine refuses (another run, policy or monitor).
        DataError: as :func:`rj23_inputs`, or a checkpoint of another stream.
    """
    # tasks.plain dispatches here: imported late, as it imports this module's caller.
    from astro_simclr.tasks.plain import (
        check_plain_runtime,
        plain_machine,
        resolve_accelerator,
        resolve_plain_resume,
    )

    if settings.schedule != RJ23:
        raise ConfigError(f"run_rj23_plain needs schedule={RJ23}, got {settings.schedule}")
    check_accelerator(settings.accelerator)
    check_plain_runtime(settings)
    inputs = rj23_inputs(settings, request)
    out = settings.output_dir
    ckpt_path = resolve_plain_resume(resume, out)
    policy = DevicePolicy(
        resolve_accelerator(settings.accelerator),
        settings.deterministic,
        cast(Literal["ieee", "tf32"], settings.fp32_precision),
    )
    plan, n_classes = inputs.plan, inputs.data.n_classes
    device_parts = {name: DevicePart.of(inputs.parts[name], policy.device) for name in plan.parts}
    stream = ClassUniformStream(
        inputs.parts[plan.train].labels, settings.batch_size, settings.seed, n_classes=n_classes
    )
    train = RJ23TrainSource(device_parts[plan.train], stream, inputs.identity)
    sources = {
        name: RJ23EvalSource(device_parts[name], settings.eval_batch_size) for name in plan.parts
    }
    out.mkdir(parents=True, exist_ok=True)
    policy.apply()  # the machine record reads the precision back (ADR 0010)
    machine = plain_machine(policy.device)
    record = start_run(
        "train-supervised",
        settings,
        data=inputs.path,
        machine=machine,
        resumed=ckpt_path is not None,
        engine=ENGINE,
        queued=queued,
        **_split_arguments(request),
    )
    identity: IdentityState = {
        "run_id": record.manifest.run_id,
        "attempt": record.manifest.attempt,
        "split_run": dict(inputs.identity),
    }
    with recorded(record):
        task, result = fit(
            lambda: RJ23Task(settings, n_classes),
            train,
            sources[plan.selection],
            schedule=rj23_schedule(settings),
            device=policy,
            seed=settings.seed,
            run_dir=out,
            identity=identity,
            resume=ckpt_path,
            on_validation=on_validation,
            on_step=on_step,
            progress=progress,
            score=macro_f1_score(n_classes),
        )
        engine = {
            # The stop weights are the last ones: ckpt/last.pt at the stop.
            "checkpoints": {
                "last": LAST_CHECKPOINT,
                "stop": LAST_CHECKPOINT,
                "best": BEST_CHECKPOINT,
            },
            "versions": {"astro_simclr": version("astro-simclr"), "torch": str(torch.__version__)},
            "accelerator": policy.device.type,
            "machine": machine,
            "resumed_from": None if ckpt_path is None else str(ckpt_path),
            "timing": {
                "fit_seconds": result.fit_seconds,
                "steps": result.steps,
                "seconds_per_step": result.seconds_per_step,
            },
        }
        metrics = _plain_metrics(settings, request, inputs, (task, result), sources, engine)
        metrics["run"] = record.summary()
        text = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        (out / METRICS_FILE).write_text(text, encoding="utf-8")
    return metrics


def _plain_metrics(
    settings: Settings,
    request: RJ23Request,
    inputs: RJ23Inputs,
    trained: tuple[Task, FitResult],
    sources: Mapping[str, RJ23EvalSource],
    engine: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate both kept weights (``stop`` first: the last), write the files, the metrics."""
    task, result = trained
    schedule = schedule_record(regime_schedule(settings), inputs.plan.selection, result.history)
    evaluated = {name: sources[name] for name, _ in inputs.plan.evaluated}
    kept = evaluate_kept(task, result, evaluated, KEEPS)
    steps = {"stop": schedule["stop_step"], "best": schedule["best_step"]}
    files = {"stop": LAST_CHECKPOINT, "best": BEST_CHECKPOINT}
    primary = cast(Keep, settings.rj23_keep)
    keep: dict[str, Any] = {"primary": primary}
    for name in KEEPS:
        probabilities = {part: kept[name][part].probabilities.numpy() for part in evaluated}
        keep[name] = keep_record(
            settings.output_dir,
            name,
            primary,
            inputs,
            probabilities,
            step=steps[name],
            checkpoint=files[name],
        )
    return rj23_metrics(
        settings,
        request,
        inputs,
        schedule=schedule,
        keep=keep,
        last_step=result.last_step,
        engine=engine,
    )
