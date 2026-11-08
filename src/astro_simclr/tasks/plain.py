"""``run_command`` on the plain engine: the paper classifier's run (task 4.6; ADR 0014).

:func:`run_plain` is ``experiments.run_command`` for the path the plain engine trains today,
``train-supervised`` of the paper classifier on the ``carrasco2021`` schedule (every 2.5
flag, split runs, ``fp32_precision``), without Lightning. It refuses what ``run_command``
refuses (``tasks.common``), trains with :func:`astro_simclr.engine.fit` and writes the same
outputs, so a plain run and a Lightning run of one config are compared file by file:

- ``metrics.json``: the keys of ``run_command``'s, with ``machine.engine`` ``plain``, the
  plain checkpoints (``ckpt/best.pt``, ``ckpt/last.pt``) and no Lightning version;
  ``best_epoch`` / ``last_epoch`` derived from the steps (ADR 0014, C5);
- ``manifest.json`` (ADR 0015) with ``engine: plain``, which is also part of the run id;
- ``scalars.jsonl``, ``ckpt/{last,best}.pt`` (the engine's), ``predictions_*.npz`` and the
  confusion-matrix figures (``tasks.results``, shared with Lightning).

**Order of a run.** The checks (accelerator, names, schedule, split, resume), the data
(``tasks.paper.paper_inputs``: the arrays the Lightning path trains on, on the ``device``
tier), the manifest (``running``), then :func:`~astro_simclr.engine.fit` (precision, flags,
seed, the task built on the CPU, ...), the final evaluation of the scored weights (the
restored best ones, or the last ones with ``checkpoint_selection: final``, plan row D3:
``Schedule.keep`` ``stop``), ``metrics.json`` and the manifest's ``finished``. A run
refused before the manifest exists never started and leaves none (``run_command``'s rule,
task A1).

**Resume.** ``resume="auto"`` continues from ``OUTPUT_DIR/ckpt/last.pt`` when it exists (a
fresh start otherwise); a path must exist. The engine refuses a Lightning checkpoint, another
run id (another config, data, machine class or split), another precision or determinism
policy and another stream. Because the run id holds the config, a resume cannot extend a
run by raising ``max_steps`` (Lightning allows it): that is a new run.

**Not on the plain engine yet:** the legacy commands and models (frozen at
``lightning-final``, plan 4.6), phase 3's ``rj23`` schedule (a follow-up), and
``limit_train_batches`` / ``limit_val_batches`` (Lightning's dry-run limits: a plain dry run
sets ``max_steps``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import IO, Any, Final, Literal, cast

import torch

from astro_simclr.config import CARRASCO2021, ConfigError, Settings, settings_to_plain
from astro_simclr.data.schema import DataError
from astro_simclr.engine import (
    Dataset,
    DevicePolicy,
    EvalBatches,
    FitResult,
    Horizon,
    IdentityState,
    Monitor,
    Schedule,
    StreamSpec,
    Task,
    TierChoice,
    TrainSource,
    Validation,
    ValidationGrid,
    evaluate_kept,
    fit,
    machine_record,
    open_eval,
    open_train,
    recorded,
)
from astro_simclr.presets import preset_report
from astro_simclr.protocol.split_runs import SplitRequest, SplitRun, open_split_run
from astro_simclr.runs.layout import BEST_CHECKPOINT, LAST_CHECKPOINT, METRICS_FILE
from astro_simclr.runs.record import RunRecord, start_run
from astro_simclr.tasks.common import (
    PAPER_STAMP_CLASSIFIER,
    check_accelerator,
    check_names,
    check_schedule,
    check_split,
    dataset_path,
    refuse_rj23,
)
from astro_simclr.tasks.paper import build_paper_task, check_paper_task, paper_inputs
from astro_simclr.tasks.results import (
    StageOutput,
    classifier_results,
    data_info,
    finite,
    selection_entry,
)
from astro_simclr.training.tensor_splits import TensorSplit

__all__ = ["ENGINE", "plain_machine", "resolve_accelerator", "resolve_plain_resume", "run_plain"]

#: The engine this module trains on (``manifest.json``, ``metrics.json["machine"]``).
ENGINE: Final = "plain"
#: The only monitor of the ``carrasco2021`` path; its mode.
_MONITOR: Final = Monitor("loss_val", "min")

Accelerator = Literal["cpu", "cuda", "mps"]


def resolve_accelerator(accelerator: str) -> Accelerator:
    """The device a ``Settings.accelerator`` names, as Lightning resolves it.

    ``auto``: CUDA, else MPS, else the CPU; ``gpu``: CUDA, else MPS. Call
    ``check_accelerator`` first (``gpu`` without either is refused there).

    Raises:
        ConfigError: for ``gpu`` on a machine with neither (``check_accelerator`` says it
            first) or an unknown name.
    """
    if accelerator in ("cpu", "cuda", "mps"):
        return cast(Accelerator, accelerator)
    if accelerator in ("auto", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        if accelerator == "auto":
            return "cpu"
    raise ConfigError(f"accelerator {accelerator!r} names no device here")


def resolve_plain_resume(resume: str | None, output_dir: Path) -> Path | None:
    """The plain checkpoint to resume from: ``None``, ``"auto"`` or a path.

    ``"auto"`` means ``output_dir/ckpt/last.pt`` if it exists, otherwise a fresh start (so
    the same command can be re-run after an interruption).

    Raises:
        ConfigError: if an explicit path does not exist.
    """
    if resume is None:
        return None
    if resume == "auto":
        last = output_dir / LAST_CHECKPOINT
        return last if last.is_file() else None
    path = Path(resume)
    if not path.is_file():
        raise ConfigError(f"--resume {resume}: no such checkpoint")
    return path


def plain_machine(device: torch.device) -> dict[str, Any]:
    """``metrics.json["machine"]`` of a plain run (``engine.machine_record``)."""
    return machine_record(device)


def _check_plain(command: str, settings: Settings) -> None:
    """What the plain engine trains (module docstring).

    Raises:
        ConfigError: for another command, model or schedule, ``devices`` other than 1, or a
            Lightning dry-run limit.
    """
    paper = command == "train-supervised" and settings.supervised_model == PAPER_STAMP_CLASSIFIER
    if not paper or settings.schedule != CARRASCO2021:
        msg = (
            f"the plain engine trains train-supervised with supervised_model="
            f"{PAPER_STAMP_CLASSIFIER} on schedule {CARRASCO2021} (task 4.6); got {command} "
            f"with {settings.supervised_model} on {settings.schedule}: use runtime.engine "
            "lightning"
        )
        raise ConfigError(msg)
    if settings.devices != 1:
        raise ConfigError(f"the plain engine trains on one device, got devices={settings.devices}")
    limits = [n for n in ("limit_train_batches", "limit_val_batches") if getattr(settings, n)]
    if limits:
        msg = (
            f"{', '.join(limits)} are Lightning's dry-run limits; the plain engine has none "
            "(set max_steps, val_every_n_steps and horizon_increment for a short run)"
        )
        raise ConfigError(msg)


def schedule_of(settings: Settings) -> Schedule:
    """The ``carrasco2021`` settings as the engine's :class:`~astro_simclr.engine.Schedule`.

    ``checkpoint_selection: final`` (plan row D3) keeps the last weights (``stop``), and no
    best ones; the default keeps the best (CD21).
    """
    final = settings.checkpoint_selection == "final"
    return Schedule(
        monitor=_MONITOR,
        validation=ValidationGrid(settings.val_every_n_steps),
        horizon=Horizon(settings.max_steps, settings.horizon_increment),
        keep=("stop",) if final else ("best",),
        checkpoint_every_n_steps=settings.checkpoint_every_n_steps,
    )


def _dataset(split: TensorSplit) -> Dataset:
    return Dataset(
        images=split.images,
        features=split.features,
        labels=split.labels,
        fits=split.fits,
        n_classes=split.n_classes,
    )


def _sources(
    settings: Settings, splits: dict[str, TensorSplit], device: torch.device
) -> tuple[TrainSource, dict[str, EvalBatches]]:
    """The Train stream (class-balanced, the run's seed) and the evaluation roles, on the
    ``device`` tier (``DeviceTensorDataModule``'s batches, ADR 0018)."""
    train = open_train(
        _dataset(splits["Train"]),
        TierChoice(),
        StreamSpec(batch_size=settings.batch_size, seed=settings.seed),
        device,
    )
    evals = {
        role: open_eval(
            _dataset(splits[role]), TierChoice(), device, batch_size=settings.eval_batch_size
        )
        for role in ("Validation", "Test")
    }
    return train, evals


def _pretrained_file(path: Path | None) -> Path | None:
    # experiments.pretrained_checkpoint's rule: a run directory means its best checkpoint.
    if path is None:
        return None
    return path / "checkpoints" / "best.ckpt" if path.is_dir() else path


def _base_metrics(
    command: str, settings: Settings, result: FitResult, device: torch.device, data: Path
) -> dict[str, Any]:
    """``run_command``'s base metrics, for a plain run."""
    return {
        "command": command,
        "seed": settings.seed,
        "best_epoch": result.best_epoch,
        "last_epoch": result.last_epoch,
        "best_step": result.best_step,
        "last_step": result.last_step,
        "monitor": settings.monitor,
        "monitor_best": result.best_score,
        "checkpoints": {"best": BEST_CHECKPOINT, "last": LAST_CHECKPOINT},
        "versions": {"astro_simclr": version("astro-simclr"), "torch": str(torch.__version__)},
        "accelerator": device.type,
        "machine": plain_machine(device),
        "data": data_info(data),
        "config": settings_to_plain(settings),
    }


def _metrics(
    command: str,
    settings: Settings,
    trained: tuple[Task, FitResult],
    record: RunRecord,
    inputs: tuple[Path, dict[str, Any], int, dict[str, EvalBatches], torch.device],
    resume: Path | None,
    split_run: SplitRun | None,
) -> dict[str, Any]:
    """Every entry of ``metrics.json`` (and the prediction and figure files it names)."""
    task, result = trained
    data, paper_data, n_tiled, sources, device = inputs
    metrics = _base_metrics(command, settings, result, device, data)
    metrics["run"] = record.summary()
    metrics["resumed_from"] = None if resume is None else str(resume)
    metrics["timing"] = {
        "fit_seconds": result.fit_seconds,
        "steps": result.steps,
        "seconds_per_step": result.seconds_per_step,
    }
    metrics["horizon"] = result.horizon
    metrics["paper_data"] = {**paper_data, "n_tiled": n_tiled}
    metrics["preset"] = preset_report(settings)
    if split_run is not None:
        metrics["protocol"] = split_run.describe()
    (keep,) = schedule_of(settings).keep
    scored = evaluate_kept(task, result, sources, (keep,))[keep]
    stages = {
        role: StageOutput.of(scored[role].probabilities, scored[role].targets, device)
        for role in ("Validation", "Test")
    }
    out = settings.output_dir
    metrics.update(
        classifier_results(command, stages["Validation"], stages["Test"], out, split_run)
    )
    metrics["selection"] = selection_entry(
        settings, best_step=result.best_step, last_step=result.last_step
    )
    if keep == "stop":
        metrics["checkpoints"]["best"] = None  # the scored weights are ckpt/last.pt's
    return metrics


def run_plain(
    command: str,
    settings: Settings,
    *,
    resume: str | None = None,
    split: SplitRequest | None = None,
    on_step: Callable[[int], None] | None = None,
    on_validation: Callable[[Validation, Task], None] | None = None,
    progress: IO[str] | None = None,
) -> dict[str, Any]:
    """Run ``command`` with ``settings`` on the plain engine; return the metrics (module docstring).

    ``resume`` is ``None``, ``"auto"`` or a ``ckpt/last.pt`` path; ``split`` makes it a split
    run. ``on_step`` / ``on_validation`` / ``progress`` are :func:`astro_simclr.engine.fit`'s
    (the preflight's timer, the tests' probes, one line per validation). The metrics are
    also written to ``output_dir/metrics.json``.

    Raises:
        AcceleratorError: if the accelerator is not available.
        ConfigError: on an invalid name, a config the plain engine does not train
            (module docstring), the paper model's schedule rules, a split run of another
            command, a missing ``--resume`` checkpoint, or a checkpoint the engine refuses
            (another run, policy or monitor, a Lightning checkpoint).
        DataError: if the dataset is missing or invalid, has no float32 stamps, a splits
            file does not belong to it, or a checkpoint is of another stream.
    """
    refuse_rj23(settings)
    check_accelerator(settings.accelerator)
    check_names(settings)
    _check_plain(command, settings)
    check_schedule(command, settings)
    data = dataset_path(settings)
    out = settings.output_dir
    ckpt_path = resolve_plain_resume(resume, out)
    split_run = None
    if split is not None:
        check_split(command, settings)
        split_run = open_split_run(split, data)
    policy = DevicePolicy(
        resolve_accelerator(settings.accelerator),
        settings.deterministic,
        cast(Literal["ieee", "tf32"], settings.fp32_precision),
    )
    try:
        check_paper_task(settings)
        source, paper_data = paper_inputs(settings, data, split_run)
        splits = source()
    except ValueError as err:
        # DataError is a ValueError too, and must keep its exit code (1, not 2).
        if isinstance(err, ConfigError | DataError):
            raise
        raise ConfigError(str(err)) from None
    train, sources = _sources(settings, splits, policy.device)
    out.mkdir(parents=True, exist_ok=True)
    policy.apply()  # the machine record reads the precision back (ADR 0010)
    record = start_run(
        command,
        settings,
        data=data,
        machine=plain_machine(policy.device),
        split=split,
        pretrained_file=_pretrained_file(settings.pretrained),
        resumed=ckpt_path is not None,
        engine=ENGINE,
    )
    identity: IdentityState = {
        "run_id": record.manifest.run_id,
        "attempt": record.manifest.attempt,
        "split_run": None if split_run is None else dict(split_run.identity()),
    }
    with recorded(record):
        trained = fit(
            lambda: build_paper_task(settings, paper_data),
            train,
            sources["Validation"],
            schedule=schedule_of(settings),
            device=policy,
            seed=settings.seed,
            run_dir=out,
            identity=identity,
            resume=ckpt_path,
            on_validation=on_validation,
            on_step=on_step,
            progress=progress,
        )
        inputs = (data, paper_data, len(train.order.positions), sources, policy.device)
        metrics = _metrics(command, settings, trained, record, inputs, ckpt_path, split_run)
        cleaned = cast(dict[str, Any], finite(metrics))
        text = json.dumps(cleaned, indent=2, allow_nan=False) + "\n"
        (out / METRICS_FILE).write_text(text, encoding="utf-8")
    return cleaned
