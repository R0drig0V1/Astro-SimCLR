"""Run one training subcommand end to end: settings, data, module, Trainer and outputs.

This is what ``astro-simclr pretrain | linear-eval | fine-tune | train-supervised`` do
(task 1.7), kept out of ``cli`` so that the preflight and the tests call the same code. One
run is:

1. check that the requested accelerator exists (:func:`check_accelerator`): a reference
   config says ``accelerator: cuda``, and without CUDA it must fail here, clearly, instead
   of inside Lightning or, worse, train on the CPU;
2. set the float32 precision of convolutions and matmuls (``apply_fp32_precision``,
   ADR 0010: ``ieee`` by default, where torch would use TF32 convolutions on Ampere+), then
   open the prepared layout ``data_dir/<dataset>`` (:func:`dataset_path`);
3. ``seed_run`` (before any module is built: initial weights consume the generators);
4. build the module and its data module from the settings;
5. ``trainer.fit`` with EarlyStopping, the best-by-monitor checkpoint ``best.ckpt`` and the
   latest-epoch checkpoint ``last.ckpt`` (``build_last_checkpoint``: Lightning 2.6's
   ``save_last=True`` skips epochs without improvement), optionally resuming
   (``--resume PATH|auto``);
6. classifiers: ``validate`` and ``test`` on the best checkpoint, then the metrics JSON and
   the confusion-matrix figures.

Around the fit, the run's ``manifest.json`` (``runs.record``; plan task A1, ADR 0015):
``queued`` then ``running`` once the Trainer exists (its device names the machine class),
then ``finished`` after ``metrics.json`` is written, ``failed`` on an exception, or
``cancelled`` on Ctrl-C. A run refused before that (a bad name, missing data) writes none.

Everything a run writes is under ``settings.output_dir``::

    OUT/manifest.json             the run id, attempt, provenance and status (runs.manifest)
    OUT/checkpoints/best.ckpt     the selected epoch (monitor, mode from the metric)
    OUT/checkpoints/last.ckpt     the latest epoch (what --resume continues from)
    OUT/lightning_logs/version_N  TensorBoard logs and the Lightning hparams.yaml
    OUT/metrics.json              see below
    OUT/confusion_matrix_{val,test}.png   classifiers only
    OUT/predictions_{val,test}.npz        classifiers only: per-stamp probabilities
                                          (split runs: confusion_matrix_{selection,
                                          evaluation}.png, predictions_evaluation.npz)

``metrics.json`` holds the command, the seed, ``run`` (run id, attempt and config hash, as
in the manifest), the selected (``best_epoch``, 0-based) and
last epoch and global step (``best_step``, ``last_step``), the monitored value at the
best epoch, the package/torch/Lightning versions, the machine (``machine``: device type,
GPU name, torch, CUDA and cuDNN versions, platform, the effective float32 precision and
the engine, ``lightning``; task 2.3, ADR 0010 and task A1: runs of one comparison must
share it), the data (prepared directory,
source SHA-256 and the class counts of ``meta.json``) and the resolved settings.
Classifiers add ``val_acc_best`` (micro accuracy on Validation at the best checkpoint: with
the default monitor ``accuracy_val`` this is the best validation accuracy over epochs, the
metric of the legacy tables),
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
on device-resident 21x21 crops (``training.device_data``: the float path of task 2.1, or
the interim uint8 crops with ``paper_input: uint8``), Train without the stamps that leak
into Validation/Test (``exclude_leaked_train``), validation every ``val_every_n_steps`` steps,
and a
``training.schedule.HorizonCallback`` in place of EarlyStopping and the best
``ModelCheckpoint``s; it also writes ``last.ckpt`` after the validation of every
``checkpoint_every_n_steps`` steps, and at the end. The outputs are
the same files; the metrics add ``horizon``, ``paper_data`` (input, NaN approximation,
excluded Train indices, Train and tiled sizes) and ``preset`` (task 2.2c,
:func:`astro_simclr.presets.preset_report`: whether the run is the faithful preset, and
which keys and diagnostics differ), and every run's metrics hold ``best_step`` and
``last_step`` (optimizer steps). Its resume is exact: the batch stream is a function of the
step, and the horizon and best weights are in ``last.ckpt``.

**Split runs** (task 2.5-infra; :mod:`astro_simclr.protocol.split_runs`). With
``split=SplitRequest(file, fold)`` (``train-supervised --splits FILE --fold K | --screen``;
``carrasco2021`` schedule only) the run first verifies the splits file against the
prepared data, then trains on the fold's training part (or Train minus the excluded stamps,
for the screen), tiled as in the preset, selects on the fold's selection set (the screen's
selection half) with the same monitor, horizon rule and batch-64 means, and predicts the
evaluation part (the screen's decision half). Test is neither loaded nor predicted. The
outputs differ from a normal run's: ``predictions_evaluation.npz`` (``index``: the pool
index, or the Validation index for the screen), ``confusion_matrix_{selection,evaluation}
.png``, and in the metrics ``protocol`` (:meth:`SplitRun.describe`: mode, fold, the splits
file and its SHA-256, part sizes), ``selection_acc_best``, ``evaluation_accuracy``,
``evaluation_balanced_accuracy``, ``evaluation.{selection,evaluation}`` and
``predictions.evaluation``, instead of ``val_acc_best``, ``test_acc_at_best`` and the
``val``/``test`` entries. The Lightning logs still call the evaluation part ``test`` (it is
the data module's ``Test`` role). The run's identity (splits SHA-256, mode, fold) is in
every checkpoint, and ``--resume`` refuses, before any step, a checkpoint of another split
run or of a normal run (and a normal run refuses a split run's checkpoint).

**Timing.** Every run's metrics hold ``timing``: the wall-clock seconds of the fit loop, the
optimizer steps it ran (fewer than ``last_step`` after a resume) and ``seconds_per_step``
(validation included; the pilot's figure of task 2.2c). ``null`` when no step ran.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Protocol, cast

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback

from astro_simclr.config import CARRASCO2021, ConfigError, Settings, settings_to_plain
from astro_simclr.data.schema import DataError
from astro_simclr.presets import preset_report
from astro_simclr.protocol.split_runs import SplitRequest, SplitRun, open_split_run
from astro_simclr.runs.record import start_run
from astro_simclr.tasks.common import (
    PAPER_STAMP_CLASSIFIER,
    AcceleratorError,
    check_accelerator,
    dataset_path,
)
from astro_simclr.tasks.common import check_names as _check_names
from astro_simclr.tasks.common import check_schedule as _check_schedule
from astro_simclr.tasks.common import check_split as _check_split
from astro_simclr.tasks.common import refuse_rj23 as _refuse_rj23
from astro_simclr.tasks.paper import load_backbone, paper_inputs
from astro_simclr.tasks.results import StageOutput, classifier_results, data_info
from astro_simclr.tasks.results import finite as _finite
from astro_simclr.training.datamodules import ClassifierDataModule, ContrastiveDataModule
from astro_simclr.training.device_data import (
    SPLIT_RUN_STATE,
    DeviceTensorDataModule,
    check_split_identity,
)
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
    apply_fp32_precision,
    build_callbacks,
    build_last_checkpoint,
    build_trainer,
    fp32_precision_record,
    monitor_mode,
    seed_run,
)
from astro_simclr.training.schedule import HorizonCallback

__all__ = [
    "BEST_CHECKPOINT",
    "CHECKPOINT_DIR",
    "ENGINE",
    "LAST_CHECKPOINT",
    "METRICS_FILE",
    "PAPER_STAMP_CLASSIFIER",
    "AcceleratorError",
    "check_accelerator",
    "dataset_path",
    "load_pretrained",
    "machine_record",
    "paper_module",
    "pretrained_checkpoint",
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
#: The engine of every run of this module (``runs.manifest``; plan task A1). The plain
#: PyTorch engine of phase 4 will record ``plain``.
ENGINE: Final = "lightning"
#: ``train-supervised`` model names that are not ResNets.
_STAMP_CLASSIFIER: Final = "stamp_classifier"


class _BestCheckpoint(Protocol):
    """What the metrics read from the best-checkpoint callback of either schedule."""

    @property
    def best_model_path(self) -> str: ...

    @property
    def best_model_score(self) -> torch.Tensor | None: ...


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


def pretrained_checkpoint(path: Path | None) -> Path | None:
    """The checkpoint file ``--pretrained PATH`` names: ``PATH`` itself, or for a pretrain
    run directory its ``checkpoints/best.ckpt``; ``None`` without ``--pretrained``.

    The file may not exist (``load_pretrained`` reports that).
    """
    if path is None:
        return None
    return path / CHECKPOINT_DIR / BEST_CHECKPOINT if path.is_dir() else path


def load_pretrained(path: Path | None) -> SimCLRModule:
    """Load the pretrained SimCLR module of ``linear-eval`` / ``fine-tune``, on the CPU.

    ``path`` is a checkpoint, or a pretrain output directory (its ``checkpoints/best.ckpt``).

    Raises:
        ConfigError: if ``path`` is unset, missing, or not a SimCLR checkpoint.
    """
    path = pretrained_checkpoint(path)
    if path is None:
        raise ConfigError("this command needs the pretrain checkpoint: --pretrained PATH")
    if not path.is_file():
        raise ConfigError(f"pretrained checkpoint {path}: no such file")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    hparams = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    if "encoder_name" not in hparams or "projection_dim" not in hparams:
        raise ConfigError(f"{path} is not a pretrain (SimCLR) checkpoint")
    return SimCLRModule.load_from_checkpoint(path, map_location="cpu", weights_only=True)


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


def paper_module(settings: Settings) -> PaperClassifierModule:
    """The ``PaperClassifierModule`` of ``settings`` (the model drawn from the global generator).

    The paper model's fields, including the 2.5-models variant flags (``paper_arch``,
    ``group_fields``, ``context``, ``rot90_augment``) and the C5 / C6 augmentation
    (``aug_noise``, ``aug_feature_jitter``, ``synthetic_bogus``, ``aug_min_r2``) and the C7
    flux rescaling (``flux_rescale``, ``flux_k_min``, ``flux_k_max``), whose
    draws are seeded with ``seed``, and the ``feature_set`` diagnostic (ADR 0016).

    Raises:
        UnknownOptimizerError, ValueError: as ``PaperClassifierModule``.
    """
    return PaperClassifierModule(
        beta_loss=settings.beta_loss,
        lr=settings.lr,
        optimizer=settings.optimizer,
        with_features=settings.with_features,
        dropout_active=settings.dropout_active,
        drop_rate=settings.drop_rate,
        adam_betas=settings.adam_betas,
        paper_arch=settings.paper_arch,
        group_fields=settings.group_fields,
        context=settings.context,
        rot90_augment=settings.rot90_augment,
        augment_seed=settings.seed,
        aug_noise=settings.aug_noise,
        aug_feature_jitter=settings.aug_feature_jitter,
        synthetic_bogus=settings.synthetic_bogus,
        aug_min_r2=settings.aug_min_r2,
        flux_rescale=settings.flux_rescale,
        flux_k_min=settings.flux_k_min,
        flux_k_max=settings.flux_k_max,
        feature_set=settings.feature_set,
    )


def _build_paper(
    settings: Settings, data: Path, split_run: SplitRun | None = None
) -> tuple[ClassifierModule, DeviceTensorDataModule, dict[str, Any]]:
    """The paper model, its device-resident data and the ``paper_data`` metrics entry.

    The splits come from ``tasks.paper.paper_inputs`` (shared with the plain engine: the
    float path or the uint8 diagnostic, the leaked Train stamps removed, a split run's
    roles, the C5-C7 fits); they are built once, at the data module's first ``setup``.
    """
    source, info = paper_inputs(settings, data, split_run)
    seed_run(settings)
    module = paper_module(settings)
    _load_backbone_weights(settings, module, info)
    datamodule = DeviceTensorDataModule(
        source,
        batch_size=settings.batch_size,
        eval_batch_size=settings.eval_batch_size,
        seed=settings.seed,
        run_identity=None if split_run is None else split_run.identity(),
        # Augmented runs plan each step on the host, ahead of the step (ADR 0006, "Prefetch").
        planner=module.augment_plan if module.augmenting and _PREFETCH else None,
    )
    return module, datamodule, info


def _load_backbone_weights(
    settings: Settings, module: PaperClassifierModule, info: dict[str, Any]
) -> None:
    """``tasks.paper.load_backbone`` on ``module.model``, with this module's :data:`_DOWNLOAD`.

    Raises:
        ConfigError, WeightsError: as ``tasks.paper.load_backbone``.
    """
    load_backbone(settings, module.model, info, download=_DOWNLOAD)


#: The weights downloader of ``_load_backbone_weights`` (None: the Hugging Face Hub); the
#: tests replace it with a local file, so no test touches the network.
_DOWNLOAD: Callable[[Any], Path] | None = None


#: Plan the training augmentation in the loader (True) or in the step (False: the in-step
#: path, with host round trips; the same bits; kept for the equivalence test).
_PREFETCH = True


def _resume_identity(ckpt_path: Path) -> object:
    """The split-run identity stored in a ``carrasco2021`` checkpoint (``None``: a normal run).

    It is in the training loader's state (``device_data.SPLIT_RUN_STATE``), which Lightning
    keeps under ``loops.fit_loop.state_dict.combined_loader``. A checkpoint without it is
    from a normal run (or from before task 2.5-infra, when every run was a normal run).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    loops = checkpoint.get("loops", {}) if isinstance(checkpoint, dict) else {}
    states = loops.get("fit_loop", {}).get("state_dict", {}).get("combined_loader", [])
    for state in states if isinstance(states, list) else []:
        if isinstance(state, dict) and SPLIT_RUN_STATE in state:
            return state[SPLIT_RUN_STATE]
    return None


def _build(
    command: str, settings: Settings, data: Path, split_run: SplitRun | None = None
) -> tuple[L.LightningModule, L.LightningDataModule, frozenset[str], dict[str, Any] | None]:
    """The module, its data module, the metrics the module logs (its ``MONITORABLE``) and,
    for the paper model, its ``paper_data`` metrics entry (filled in at ``setup``)."""
    datamodule: L.LightningDataModule
    _check_schedule(command, settings)
    if settings.schedule == CARRASCO2021:
        paper, datamodule, info = _build_paper(settings, data, split_run)
        return paper, datamodule, ClassifierModule.MONITORABLE, info
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
        return module, datamodule, SimCLRModule.MONITORABLE, None
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
    return module, datamodule, ClassifierModule.MONITORABLE, None


def _scalar(value: torch.Tensor | float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def machine_record(trainer: L.Trainer) -> dict[str, Any]:
    """The machine and build a run used (task 2.3: one comparison never mixes them).

    The device is the one Lightning chose (``trainer.strategy.root_device``); no device is
    named here. ``gpu`` is the CUDA device name, ``None`` on other devices (MPS has no name
    API; ``platform`` and ``processor`` identify the Mac). ``fp32_precision`` (the effective
    float32 policy), ``fp32_backends`` and ``compute_capability`` come from
    :func:`~astro_simclr.training.runs.fp32_precision_record` (ADR 0010).
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
        "engine": ENGINE,
        **fp32_precision_record(device),
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
        "machine": machine_record(trainer),
        "data": data_info(data),
        "config": settings_to_plain(settings),
    }


def _stage(module: ClassifierModule, stage: str) -> StageOutput:
    """What ``validate`` / ``test`` left in ``module`` for ``stage`` (``val`` or ``test``)."""
    y_pred, y_true = module.predictions[stage]
    return StageOutput(
        module.probabilities[stage], y_pred, y_true, module.confusion_matrices[stage]
    )


def _classifier_results(
    command: str,
    trainer: L.Trainer,
    module: ClassifierModule,
    datamodule: L.LightningDataModule,
    best: str,
    out: Path,
    split_run: SplitRun | None = None,
) -> dict[str, Any]:
    """Validate and test the best weights, then ``tasks.results.classifier_results``.

    A split run's ``Test`` role is its evaluation part (``device_data.protocol_splits``),
    so ``trainer.test`` predicts it and never touches the real Test split.

    Raises:
        DataError: if no best checkpoint was written.
    """
    if not best or not Path(best).is_file():
        raise DataError("no best checkpoint was written (did any validation run?)")
    trainer.validate(module, datamodule=datamodule, ckpt_path=best, verbose=False)
    val = _stage(module, "val")
    trainer.test(module, datamodule=datamodule, ckpt_path=best, verbose=False)
    test = _stage(module, "test")
    return classifier_results(command, val, test, out, split_run)


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


class _FitTimer(Callback):
    """The wall-clock time and optimizer steps of the fit loop (``metrics["timing"]``)."""

    def __init__(self) -> None:
        self._start: tuple[float, int] | None = None
        self.seconds: float | None = None
        self.steps: int | None = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Start the clock at the first step (after setup and the data upload)."""
        self._start = (time.perf_counter(), trainer.global_step)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Stop the clock; the steps are those of this fit (a resume counts its own)."""
        if self._start is not None:
            start, step = self._start
            self.seconds = time.perf_counter() - start
            self.steps = trainer.global_step - step

    def report(self) -> dict[str, float | int | None]:
        """``fit_seconds``, ``steps`` and ``seconds_per_step`` (``None`` without steps)."""
        per_step = None
        if self.seconds is not None and self.steps:
            per_step = self.seconds / self.steps
        return {"fit_seconds": self.seconds, "steps": self.steps, "seconds_per_step": per_step}


def _add_results(
    metrics: dict[str, Any],
    command: str,
    settings: Settings,
    trainer: L.Trainer,
    built: tuple[L.LightningModule, L.LightningDataModule, _BestCheckpoint],
    paper_data: dict[str, Any] | None,
    split_run: SplitRun | None,
) -> None:
    """Add what follows the fit to ``metrics``: the horizon, the paper data and preset, the
    split, and the classifier results (or a pretrain run's encoder)."""
    module, datamodule, checkpoint = built
    if isinstance(checkpoint, HorizonCallback):
        metrics["horizon"] = checkpoint.horizon
    if paper_data is not None and isinstance(datamodule, DeviceTensorDataModule):
        tiled = datamodule.tiled_indices
        paper_data["n_tiled"] = None if tiled is None else len(tiled)
        metrics["paper_data"] = paper_data
        metrics["preset"] = preset_report(settings)
    if split_run is not None:
        metrics["protocol"] = split_run.describe()
    if isinstance(module, ClassifierModule):
        best = checkpoint.best_model_path
        out = settings.output_dir
        metrics.update(
            _classifier_results(command, trainer, module, datamodule, best, out, split_run)
        )
    else:
        metrics["pretrained_encoder"] = settings.encoder_name


def run_command(
    command: str,
    settings: Settings,
    *,
    resume: str | None = None,
    callbacks: Sequence[Callback] = (),
    progress_bar: bool = True,
    split: SplitRequest | None = None,
) -> dict[str, Any]:
    """Run the training subcommand ``command`` with ``settings``; return the metrics.

    ``command`` is ``pretrain``, ``linear-eval``, ``fine-tune`` or ``train-supervised``.
    ``resume`` is ``None``, ``"auto"`` or a checkpoint path (:func:`resolve_resume`).
    ``callbacks`` are added to the Trainer (the preflight's throughput timer);
    ``progress_bar=False`` hides Lightning's progress bar (the preflight prints its own
    summary). ``split`` makes it a split run (module docstring, "Split runs"). The metrics
    are also written to ``output_dir/metrics.json``.

    Raises:
        AcceleratorError: if the accelerator is not available.
        ConfigError: on an invalid name (augmentation, model, monitor), a missing or wrong
            pretrained checkpoint, a missing ``--resume`` checkpoint, or a ``carrasco2021``
            schedule with another command or model (and the paper model without it), or
            the ``rj23`` schedule (``training.rj23.run_rj23`` runs it); with
            ``split``, another command or schedule, or a fold the file does not have; a
            ``carrasco2021`` resume from a checkpoint of another split run (splits file,
            mode or fold), or across a split run and a normal run.
        DataError: if the dataset is missing, invalid or smaller than one batch, or (the
            paper model's ``paper_input: float32``) has no float32 stamps; with ``split``,
            a splits file that is malformed or does not belong to the prepared data
            (``SplitsError``).
    """
    _refuse_rj23(settings)
    check_accelerator(settings.accelerator)
    _check_names(settings)
    # Before any module is built (ADR 0010): torch's default would run TF32 convolutions.
    apply_fp32_precision(settings)
    data = dataset_path(settings)
    ckpt_path = resolve_resume(resume, settings.output_dir)
    split_run = None
    if split is not None:
        _check_split(command, settings)
        # Verify before anything is built or trained: other data must fail fast.
        split_run = open_split_run(split, data)
    if ckpt_path is not None and settings.schedule == CARRASCO2021:
        # Before any step: a resume across folds or modes would leak evaluation stamps
        # into training (the loader repeats this check when Lightning restores it).
        check_split_identity(
            _resume_identity(ckpt_path), None if split_run is None else split_run.identity()
        )
    try:
        module, datamodule, monitorable, paper_data = _build(command, settings, data, split_run)
    except ValueError as err:
        # DataError is a ValueError too, and must keep its exit code (1, not 2).
        if isinstance(err, ConfigError | DataError):
            raise
        raise ConfigError(str(err)) from None
    out = settings.output_dir
    out.mkdir(parents=True, exist_ok=True)
    timer = _FitTimer()
    extra = [*callbacks, timer]
    if settings.schedule == CARRASCO2021:
        checkpoint, trainer = _carrasco2021_trainer(settings, monitorable, extra, progress_bar)
    else:
        checkpoint, trainer = _epochs_trainer(settings, monitorable, extra, progress_bar)
    record = start_run(
        command,
        settings,
        data=data,
        machine=machine_record(trainer),
        split=split,
        pretrained_file=pretrained_checkpoint(settings.pretrained),
        resumed=ckpt_path is not None,
    )
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        metrics = _base_metrics(command, settings, trainer, checkpoint, data)
        metrics["run"] = record.summary()
        metrics["resumed_from"] = None if ckpt_path is None else str(ckpt_path)
        metrics["timing"] = timer.report()
        built = (module, datamodule, checkpoint)
        _add_results(metrics, command, settings, trainer, built, paper_data, split_run)
        cleaned = cast(dict[str, Any], _finite(metrics))
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
