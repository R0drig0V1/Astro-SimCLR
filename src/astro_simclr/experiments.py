"""Run one training subcommand end to end: settings, data, module, Trainer and outputs.

This is what ``astro-simclr pretrain | linear-eval | fine-tune | train-supervised`` do
(task 1.7), kept out of ``cli`` so that the preflight and the tests call the same code. One
run is:

1. check that the requested accelerator exists (:func:`check_accelerator`): a reference
   config says ``accelerator: cuda``, and without CUDA it must fail here, clearly, instead
   of inside Lightning or, worse, train on the CPU;
2. open the prepared layout ``data_dir/<dataset>`` (:func:`dataset_path`);
3. ``seed_run`` (before any module is built: initial weights consume the generators);
4. build the module and its data module from the settings;
5. ``trainer.fit`` with EarlyStopping, the best-by-monitor checkpoint ``best.ckpt`` and the
   latest-epoch checkpoint ``last.ckpt`` (``build_last_checkpoint``: Lightning 2.6's
   ``save_last=True`` skips epochs without improvement), optionally resuming
   (``--resume PATH|auto``);
6. classifiers: ``validate`` and ``test`` on the best checkpoint, then the metrics JSON and
   the confusion-matrix figures.

Everything a run writes is under ``settings.output_dir``::

    OUT/checkpoints/best.ckpt     the selected epoch (monitor, mode from the metric)
    OUT/checkpoints/last.ckpt     the latest epoch (what --resume continues from)
    OUT/lightning_logs/version_N  TensorBoard logs and the Lightning hparams.yaml
    OUT/metrics.json              see below
    OUT/confusion_matrix_{val,test}.png   classifiers only
    OUT/predictions_{val,test}.npz        classifiers only: per-stamp probabilities

``metrics.json`` holds the command, the seed, the selected (``best_epoch``, 0-based) and
last epoch and global step (``best_step``, ``last_step``), the monitored value at the
best epoch, the package/torch/Lightning versions, the machine (``machine``: device type,
GPU name, torch, CUDA and cuDNN versions, platform; task 2.3: runs of one comparison must
share it), the data (prepared directory, source SHA-256 and the class counts of
``meta.json``) and the resolved settings. Classifiers add ``val_acc_best`` (micro accuracy
on Validation at the best checkpoint: with the default monitor ``accuracy_val`` this is the
best validation accuracy over epochs, the metric of the legacy tables),
``test_acc_at_best``, the macro-F1 and, per split, the balanced accuracy, the per-class
recall and the confusion matrices (normalised and counts). They also write the per-stamp
softmax probabilities of the selected checkpoint on each split
(``predictions_{val,test}.npz``, :mod:`astro_simclr.protocol.predictions`), listed under
``predictions``: what ``compare-variants`` pairs stamp by stamp. Non-finite numbers are
written as ``null`` so the file stays strict JSON.

**Resume** restores the epoch counter, the optimizer and LR-scheduler state and the
EarlyStopping / ModelCheckpoint state (Lightning's ``fit(ckpt_path=...)``). It does not
restore the global random generators, so a resumed run is not bit-identical to an
uninterrupted one (augmentations and shuffling of the later epochs differ).

**Schedules** (``Settings.schedule``, task 2.2a). The above is the ``epochs`` schedule
(EarlyStopping only if ``early_stopping``). ``carrasco2021`` is ALeRCE's iteration schedule
for ``train-supervised`` with ``supervised_model: paper_stamp_classifier``: the paper model
on device-resident 21x21 crops (``training.device_data``; the interim uint8 crops until
task 2.2c wires the float path), validation every ``val_every_n_steps`` steps, and a
``training.schedule.HorizonCallback`` in place of EarlyStopping and the best
``ModelCheckpoint``s; it also writes ``last.ckpt`` after the validation of every
``checkpoint_every_n_steps`` steps, and at the end. The outputs are
the same files; the metrics add ``horizon``, and every run's metrics hold ``best_step`` and
``last_step`` (optimizer steps). Its resume is exact: the batch stream is a function of the
step, and the horizon and best weights are in ``last.ckpt``.
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Protocol, cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from astro_simclr.augmentations import resolve_augmentation_name
from astro_simclr.config import CARRASCO2021, ConfigError, Settings, settings_to_plain
from astro_simclr.data.prepared import is_prepared, read_meta
from astro_simclr.data.schema import LABEL_NAMES, DataError
from astro_simclr.paper_model import PAPER_CROP
from astro_simclr.protocol.predictions import Predictions, predictions_file, save_predictions
from astro_simclr.training.datamodules import ClassifierDataModule, ContrastiveDataModule
from astro_simclr.training.device_data import DeviceTensorDataModule, interim_uint8_crops
from astro_simclr.training.metrics import confusion_counts, macro_f1
from astro_simclr.training.modules import (
    ClassifierModule,
    FineTuneModule,
    LinearEvalModule,
    PaperClassifierModule,
    SimCLRModule,
    SupervisedCEModule,
    SupervisedCEResNetModule,
)
from astro_simclr.training.runs import (
    build_callbacks,
    build_last_checkpoint,
    build_trainer,
    monitor_mode,
    seed_run,
)
from astro_simclr.training.schedule import HorizonCallback

__all__ = [
    "BEST_CHECKPOINT",
    "CHECKPOINT_DIR",
    "LAST_CHECKPOINT",
    "METRICS_FILE",
    "PAPER_STAMP_CLASSIFIER",
    "AcceleratorError",
    "check_accelerator",
    "dataset_path",
    "load_pretrained",
    "resolve_resume",
    "run_command",
]

#: Subdirectory of the output directory with the checkpoints.
CHECKPOINT_DIR: Final = "checkpoints"
#: The best-by-monitor checkpoint (``ModelCheckpoint(filename="best")``).
BEST_CHECKPOINT: Final = "best.ckpt"
#: The latest-epoch checkpoint (``training.runs.build_last_checkpoint``).
LAST_CHECKPOINT: Final = "last.ckpt"
#: The metrics JSON of a run.
METRICS_FILE: Final = "metrics.json"
#: ``train-supervised`` model names that are not ResNets.
_STAMP_CLASSIFIER: Final = "stamp_classifier"
#: ``train-supervised`` model of the ``carrasco2021`` schedule (task 2.2a).
PAPER_STAMP_CLASSIFIER: Final = "paper_stamp_classifier"


class _BestCheckpoint(Protocol):
    """What the metrics read from the best-checkpoint callback of either schedule."""

    @property
    def best_model_path(self) -> str: ...

    @property
    def best_model_score(self) -> torch.Tensor | None: ...


class AcceleratorError(ConfigError):
    """The requested accelerator is not available on this machine or in this torch build."""


def check_accelerator(accelerator: str) -> None:
    """Fail clearly if ``accelerator`` (a ``Settings.accelerator`` value) cannot be used.

    ``cuda`` needs ``torch.cuda.is_available()``, ``mps`` needs MPS, ``gpu`` either of them;
    ``cpu`` and ``auto`` always work.

    Raises:
        AcceleratorError: naming the torch build and pointing to the preflight and ADR 0001
            (a CPU build where a CUDA one was wanted is the usual cause).
    """
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    missing = (
        (accelerator == "cuda" and not cuda)
        or (accelerator == "mps" and not mps)
        or (accelerator == "gpu" and not (cuda or mps))
    )
    if missing:
        build = f"torch {torch.__version__}, CUDA {torch.version.cuda or 'none'}"
        msg = (
            f"accelerator {accelerator!r} was requested, but it is not available ({build}). "
            "Use --accelerator cpu|auto, or install the torch build this machine needs "
            "(astro-simclr preflight; ADR 0001)."
        )
        raise AcceleratorError(msg)


def dataset_path(settings: Settings) -> Path:
    """The prepared layout of the run: ``settings.data_dir / settings.dataset``.

    Raises:
        DataError: if it is not a prepared layout (the message says how to make one).
    """
    path = settings.data_dir / settings.dataset
    if not is_prepared(path):
        msg = (
            f"{path} is not a prepared dataset; run `astro-simclr prepare-data "
            f"{settings.dataset}.pkl --out {settings.data_dir}` first (or set data_dir / "
            "dataset)"
        )
        raise DataError(msg)
    return path


def resolve_resume(resume: str | None, output_dir: Path) -> Path | None:
    """The checkpoint to resume from: ``None``, ``"auto"`` or a path.

    ``"auto"`` means ``output_dir/checkpoints/last.ckpt`` if it exists, otherwise a fresh
    start (so the same command can be re-run after an interruption).

    Raises:
        ConfigError: if an explicit path does not exist.
    """
    if resume is None:
        return None
    if resume == "auto":
        last = output_dir / CHECKPOINT_DIR / LAST_CHECKPOINT
        return last if last.is_file() else None
    path = Path(resume)
    if not path.is_file():
        raise ConfigError(f"--resume {resume}: no such checkpoint")
    return path


def load_pretrained(path: Path | None) -> SimCLRModule:
    """Load the pretrained SimCLR module of ``linear-eval`` / ``fine-tune``, on the CPU.

    ``path`` is a checkpoint, or a pretrain output directory (its ``checkpoints/best.ckpt``).

    Raises:
        ConfigError: if ``path`` is unset, missing, or not a SimCLR checkpoint.
    """
    if path is None:
        raise ConfigError("this command needs the pretrain checkpoint: --pretrained PATH")
    if path.is_dir():
        path = path / CHECKPOINT_DIR / BEST_CHECKPOINT
    if not path.is_file():
        raise ConfigError(f"pretrained checkpoint {path}: no such file")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    if "encoder_name" not in hparams or "projection_dim" not in hparams:
        raise ConfigError(f"{path} is not a pretrain (SimCLR) checkpoint")
    return SimCLRModule.load_from_checkpoint(path, map_location="cpu", weights_only=True)


def _check_names(settings: Settings) -> None:
    try:
        resolve_augmentation_name(settings.augmentation)
    except ValueError as err:
        raise ConfigError(str(err)) from None


def _build_supervised(settings: Settings) -> ClassifierModule:
    common: dict[str, Any] = {
        "beta_loss": settings.beta_loss,
        "lr": settings.lr,
        "optimizer": settings.optimizer,
        "with_features": settings.with_features,
    }
    if settings.supervised_model == _STAMP_CLASSIFIER:
        return SupervisedCEModule(drop_rate=settings.drop_rate, **common)
    if not settings.supervised_model.startswith("resnet"):
        msg = (
            f"supervised_model {settings.supervised_model!r} is not {_STAMP_CLASSIFIER!r}, "
            f"{PAPER_STAMP_CLASSIFIER!r} or a ResNet (resnet18, resnet34, resnet50, resnet152)"
        )
        raise ConfigError(msg)
    return SupervisedCEResNetModule(resnet_model=settings.supervised_model, **common)


def _build_downstream(command: str, settings: Settings) -> ClassifierModule:
    pretrained = load_pretrained(settings.pretrained)
    if pretrained.image_size != settings.image_size:
        msg = (
            f"the pretrained encoder was trained on {pretrained.image_size} px, but "
            f"image_size is {settings.image_size}; set image_size={pretrained.image_size}"
        )
        raise ConfigError(msg)
    # Seed after loading: the head's initial weights must not depend on the checkpoint
    # loading path (which builds a module with random weights first).
    seed_run(settings)
    if command == "linear-eval":
        return LinearEvalModule(
            pretrained,
            beta_loss=settings.beta_loss,
            lr=settings.lr,
            optimizer=settings.optimizer,
            with_features=settings.with_features,
        )
    return FineTuneModule(
        pretrained,
        drop_rate=settings.drop_rate,
        beta_loss=settings.beta_loss,
        lr=settings.lr * settings.fine_tune_lr_factor,
        optimizer=settings.optimizer,
        with_features=settings.with_features,
    )


def _check_schedule(command: str, settings: Settings) -> None:
    """The ``carrasco2021`` schedule and the paper model go together, on 21x21 crops.

    Raises:
        ConfigError: if only one of them is chosen, the command is not ``train-supervised``,
            or ``image_size`` is not 21.
    """
    paper_model = command == "train-supervised" and (
        settings.supervised_model == PAPER_STAMP_CLASSIFIER
    )
    if settings.schedule == CARRASCO2021 and not paper_model:
        msg = (
            f"schedule {CARRASCO2021} trains only train-supervised with "
            f"supervised_model={PAPER_STAMP_CLASSIFIER}"
        )
        raise ConfigError(msg)
    if paper_model and settings.schedule != CARRASCO2021:
        raise ConfigError(f"{PAPER_STAMP_CLASSIFIER} needs schedule={CARRASCO2021}")
    if paper_model and settings.image_size != PAPER_CROP:
        msg = (
            f"{PAPER_STAMP_CLASSIFIER} takes {PAPER_CROP}x{PAPER_CROP} crops: "
            f"set image_size={PAPER_CROP}"
        )
        raise ConfigError(msg)


def _build_paper(settings: Settings, data: Path) -> tuple[ClassifierModule, DeviceTensorDataModule]:
    """The paper model and its device-resident data (the interim uint8 crops until 2.2c)."""
    seed_run(settings)
    module = PaperClassifierModule(
        beta_loss=settings.beta_loss,
        lr=settings.lr,
        optimizer=settings.optimizer,
        with_features=settings.with_features,
        dropout_active=settings.dropout_active,
        drop_rate=settings.drop_rate,
    )
    datamodule = DeviceTensorDataModule(
        lambda: interim_uint8_crops(data),
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        seed=settings.seed,
    )
    return module, datamodule


def _build(
    command: str, settings: Settings, data: Path
) -> tuple[L.LightningModule, L.LightningDataModule, frozenset[str]]:
    """The module, its data module and the metrics the module logs (its ``MONITORABLE``)."""
    datamodule: L.LightningDataModule
    _check_schedule(command, settings)
    if settings.schedule == CARRASCO2021:
        paper, datamodule = _build_paper(settings, data)
        return paper, datamodule, ClassifierModule.MONITORABLE
    if command == "pretrain":
        seed_run(settings)
        module: L.LightningModule = SimCLRModule(
            encoder_name=settings.encoder_name,
            image_size=settings.image_size,
            projection_dim=settings.projection_dim,
            temperature=settings.temperature,
            lr_encoder=settings.lr,
            optimizer_encoder=settings.optimizer,
            nt_xent_legacy_denominator=settings.nt_xent_legacy_denominator,
            lars_legacy_trust_ratio=settings.lars_legacy_trust_ratio,
        )
        datamodule = ContrastiveDataModule(
            data,
            image_size=settings.image_size,
            batch_size=settings.batch_size,
            augmentation=settings.augmentation,
            num_workers=settings.num_workers,
        )
        return module, datamodule, SimCLRModule.MONITORABLE
    if command == "train-supervised":
        seed_run(settings)
        module = _build_supervised(settings)
    elif command in ("linear-eval", "fine-tune"):
        module = _build_downstream(command, settings)
    else:
        raise ConfigError(f"unknown training command {command!r}")
    datamodule = ClassifierDataModule(
        data,
        image_size=settings.image_size,
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        augmentation=settings.augmentation,
        num_workers=settings.num_workers,
    )
    return module, datamodule, ClassifierModule.MONITORABLE


def _finite(value: object) -> object:
    """Recursively replace non-finite floats with None (strict JSON has no NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    return value


def _scalar(value: torch.Tensor | float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _data_info(path: Path) -> dict[str, Any]:
    meta = read_meta(path)
    return {
        "path": str(path),
        "kind": meta["kind"],
        "source": meta["source"],
        "class_counts": {split: info["class_counts"] for split, info in meta["splits"].items()},
    }


def _machine(trainer: L.Trainer) -> dict[str, Any]:
    """The machine and build a run used (task 2.3: one comparison never mixes them).

    The device is the one Lightning chose (``trainer.strategy.root_device``); no device is
    named here. ``gpu`` is the CUDA device name, ``None`` on other devices (MPS has no name
    API; ``platform`` and ``processor`` identify the Mac).
    """
    device = trainer.strategy.root_device
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    # torch.backends.cudnn is untyped; the casts state the documented signatures.
    cudnn_available = cast(Callable[[], bool], torch.backends.cudnn.is_available)
    cudnn_version = cast(Callable[[], int | None], torch.backends.cudnn.version)
    cudnn = cudnn_version() if cudnn_available() else None
    return {
        "device": device.type,
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": cudnn,
        "platform": platform.platform(),
        "processor": platform.machine(),
        "python": platform.python_version(),
    }


def _base_metrics(
    command: str,
    settings: Settings,
    trainer: L.Trainer,
    checkpoint: _BestCheckpoint,
    data: Path,
) -> dict[str, Any]:
    best = Path(checkpoint.best_model_path)
    best_epoch = best_step = None
    if best.is_file():
        saved = torch.load(best, map_location="cpu", weights_only=True)
        best_epoch, best_step = int(saved["epoch"]), int(saved["global_step"])
    return {
        "command": command,
        "seed": settings.seed,
        "best_epoch": best_epoch,
        "last_epoch": trainer.current_epoch - 1,
        "best_step": best_step,
        "last_step": trainer.global_step,
        "monitor": settings.monitor,
        "monitor_best": _scalar(checkpoint.best_model_score),
        "checkpoints": {
            "best": f"{CHECKPOINT_DIR}/{BEST_CHECKPOINT}",
            "last": f"{CHECKPOINT_DIR}/{LAST_CHECKPOINT}",
        },
        "versions": {
            "astro_simclr": version("astro-simclr"),
            "torch": torch.__version__,
            "lightning": L.__version__,
        },
        "accelerator": type(trainer.accelerator).__name__,
        "machine": _machine(trainer),
        "data": _data_info(data),
        "config": settings_to_plain(settings),
    }


def _recalls(counts: list[list[int]]) -> dict[str, float | None]:
    """Per class (by name), the recall: diagonal over row sum; ``None`` for an absent class."""
    return {
        name: (row[i] / sum(row) if sum(row) else None)
        for i, (name, row) in enumerate(zip(LABEL_NAMES, counts, strict=True))
    }


def _evaluation(module: ClassifierModule, stage: str) -> dict[str, Any]:
    y_pred, y_true = module.predictions[stage]
    counts = confusion_counts(y_pred, y_true).tolist()
    recalls = _recalls(counts)
    present = [value for value in recalls.values() if value is not None]
    return {
        "accuracy": float((y_pred == y_true).double().mean()),
        "balanced_accuracy": sum(present) / len(present),
        "macro_f1": macro_f1(y_pred, y_true),
        "recall_per_class": recalls,
        "confusion_matrix": module.confusion_matrices[stage].tolist(),
        "confusion_counts": counts,
    }


def _write_predictions(
    module: ClassifierModule, stage: str, split: str, out: Path
) -> dict[str, Any]:
    """Write the stage's per-stamp probabilities; return its ``metrics["predictions"]`` entry.

    The evaluation loaders are unshuffled, so the loader order is the split's file order
    and ``index`` is ``arange(N)``.
    """
    labels = module.predictions[stage][1].numpy().astype(np.int64)
    predictions = Predictions(
        module.probabilities[stage].float().numpy(),
        labels,
        np.arange(labels.size, dtype=np.int64),
    )
    name = predictions_file(stage)
    save_predictions(predictions, out / name)
    return {"file": name, "split": split, "n": int(labels.size)}


def _plot_confusion_matrix(matrix: list[list[float]], title: str, path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")  # no display on servers; only files are written
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow([[100 * v for v in row] for row in matrix], vmin=0, vmax=100, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{100 * value:.1f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=45)
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=9)
    fig.colorbar(image, ax=ax, label="% of the true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _classifier_results(
    command: str,
    trainer: L.Trainer,
    module: ClassifierModule,
    datamodule: L.LightningDataModule,
    best: str,
    out: Path,
) -> dict[str, Any]:
    if not best or not Path(best).is_file():
        raise DataError("no best checkpoint was written (did any validation run?)")
    trainer.validate(module, datamodule=datamodule, ckpt_path=best, verbose=False)
    val = _evaluation(module, "val")
    predictions = {"val": _write_predictions(module, "val", "Validation", out)}
    trainer.test(module, datamodule=datamodule, ckpt_path=best, verbose=False)
    test = _evaluation(module, "test")
    predictions["test"] = _write_predictions(module, "test", "Test", out)
    for stage, result in (("val", val), ("test", test)):
        title = f"{command}: {stage} accuracy {100 * result['accuracy']:.1f} %"
        _plot_confusion_matrix(
            result["confusion_matrix"], title, out / f"confusion_matrix_{stage}.png"
        )
    return {
        "val_acc_best": val["accuracy"],
        "test_acc_at_best": test["accuracy"],
        "val_macro_f1": val["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "labels": list(LABEL_NAMES),
        "evaluation": {"val": val, "test": test},
        "predictions": predictions,
        "figures": ["confusion_matrix_val.png", "confusion_matrix_test.png"],
    }


def _epochs_trainer(
    settings: Settings,
    monitorable: frozenset[str],
    callbacks: Sequence[Callback],
    progress_bar: bool,
) -> tuple[_BestCheckpoint, L.Trainer]:
    """The legacy schedule: epochs, EarlyStopping (unless disabled), best and last by epoch."""
    dirpath = settings.output_dir / CHECKPOINT_DIR
    early_stopping, checkpoint = build_callbacks(settings, monitorable, dirpath=dirpath)
    last = build_last_checkpoint(dirpath, Path(LAST_CHECKPOINT).stem)
    stopping = [early_stopping] if settings.early_stopping else []
    trainer = build_trainer(
        settings,
        callbacks=[*stopping, checkpoint, last, *callbacks],
        enable_progress_bar=progress_bar,
    )
    return checkpoint, trainer


def _carrasco2021_trainer(
    settings: Settings,
    monitorable: frozenset[str],
    callbacks: Sequence[Callback],
    progress_bar: bool,
) -> tuple[_BestCheckpoint, L.Trainer]:
    """ALeRCE's schedule: steps, validation every ``val_every_n_steps``, the horizon rule.

    No EarlyStopping, no ``ModelCheckpoint`` and no epoch limit (``max_steps`` and the
    horizon end the run); the ``HorizonCallback`` writes ``best.ckpt`` and, after the
    validation of every ``checkpoint_every_n_steps`` steps, ``last.ckpt``; no sanity check.
    """
    dirpath = settings.output_dir / CHECKPOINT_DIR
    horizon = HorizonCallback(
        dirpath=dirpath,
        filename=Path(BEST_CHECKPOINT).stem,
        monitor=settings.monitor,
        mode=monitor_mode(settings.monitor, monitorable),
        initial_horizon=settings.max_steps,
        increment=settings.horizon_increment,
        last_filename=Path(LAST_CHECKPOINT).stem,
        last_every_n_steps=settings.checkpoint_every_n_steps,
    )
    trainer = build_trainer(
        settings,
        callbacks=[horizon, *callbacks],
        enable_checkpointing=False,
        enable_progress_bar=progress_bar,
        max_epochs=-1,
        max_steps=settings.max_steps,
        val_check_interval=settings.val_every_n_steps,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )
    return horizon, trainer


def run_command(
    command: str,
    settings: Settings,
    *,
    resume: str | None = None,
    callbacks: Sequence[Callback] = (),
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Run the training subcommand ``command`` with ``settings``; return the metrics.

    ``command`` is ``pretrain``, ``linear-eval``, ``fine-tune`` or ``train-supervised``.
    ``resume`` is ``None``, ``"auto"`` or a checkpoint path (:func:`resolve_resume`).
    ``callbacks`` are added to the Trainer (the preflight's throughput timer);
    ``progress_bar=False`` hides Lightning's progress bar (the preflight prints its own
    summary). The metrics are also written to ``output_dir/metrics.json``.

    Raises:
        AcceleratorError: if the accelerator is not available.
        ConfigError: on an invalid name (augmentation, model, monitor), a missing or wrong
            pretrained checkpoint, a missing ``--resume`` checkpoint, or a ``carrasco2021``
            schedule with another command or model (and the paper model without it).
        DataError: if the dataset is missing, invalid or smaller than one batch.
    """
    check_accelerator(settings.accelerator)
    _check_names(settings)
    data = dataset_path(settings)
    ckpt_path = resolve_resume(resume, settings.output_dir)
    try:
        module, datamodule, monitorable = _build(command, settings, data)
    except ValueError as err:
        # DataError is a ValueError too, and must keep its exit code (1, not 2).
        if isinstance(err, ConfigError | DataError):
            raise
        raise ConfigError(str(err)) from None
    out = settings.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if settings.schedule == CARRASCO2021:
        checkpoint, trainer = _carrasco2021_trainer(settings, monitorable, callbacks, progress_bar)
    else:
        checkpoint, trainer = _epochs_trainer(settings, monitorable, callbacks, progress_bar)
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    metrics = _base_metrics(command, settings, trainer, checkpoint, data)
    metrics["resumed_from"] = None if ckpt_path is None else str(ckpt_path)
    if isinstance(checkpoint, HorizonCallback):
        metrics["horizon"] = checkpoint.horizon
    if isinstance(module, ClassifierModule):
        metrics.update(
            _classifier_results(
                command, trainer, module, datamodule, checkpoint.best_model_path, out
            )
        )
    else:
        metrics["pretrained_encoder"] = settings.encoder_name
    cleaned = cast(dict[str, Any], _finite(metrics))
    text = json.dumps(cleaned, indent=2, allow_nan=False) + "\n"
    (out / METRICS_FILE).write_text(text, encoding="utf-8")
    return cleaned
