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
                                          (split runs: confusion_matrix_{selection,
                                          evaluation}.png, predictions_evaluation.npz)

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

import dataclasses
import json
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Protocol, cast

import lightning as L
import numpy as np
import numpy.typing as npt
import torch
from lightning.pytorch.callbacks import Callback

from astro_simclr.augmentations import resolve_augmentation_name
from astro_simclr.config import CARRASCO2021, ConfigError, Settings, settings_to_plain
from astro_simclr.data.prepared import is_prepared, load_prepared, read_meta
from astro_simclr.data.schema import LABEL_NAMES, SPLITS, DataError
from astro_simclr.paper_model import PAPER_CROP
from astro_simclr.presets import preset_report
from astro_simclr.protocol.leakage import find_duplicates, train_exclusions
from astro_simclr.protocol.predictions import Predictions, predictions_file, save_predictions
from astro_simclr.protocol.split_runs import SplitRequest, SplitRun, open_split_run
from astro_simclr.training.class_augment import DEFAULT_MIN_R2, fit_exclusions, stamp_fits
from astro_simclr.training.datamodules import ClassifierDataModule, ContrastiveDataModule
from astro_simclr.training.device_data import (
    SPLIT_RUN_STATE,
    DeviceTensorDataModule,
    TensorSplit,
    check_split_identity,
    float_crops,
    interim_uint8_crops,
    protocol_splits,
    without_train_indices,
)
from astro_simclr.training.flux_rescale import fit_central_psf, psf_exclusions
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
    "paper_module",
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
    _check_paper_variant(paper_model, settings)


#: The flags of task 2.5-models (architecture) and of C5 / C6 (augmentation), and their
#: defaults (the faithful model).
_PAPER_VARIANT_DEFAULTS: Final[Mapping[str, object]] = {
    "paper_arch": "rot4",
    "group_fields": None,
    "context": "none",
    "rot90_augment": False,
    # Class-aware augmentation (plan rows C5 and C6; ADR 0006).
    "aug_noise": 0.0,
    "aug_feature_jitter": 0.0,
    "synthetic_bogus": 0.0,
    "aug_min_r2": DEFAULT_MIN_R2,
    # Transient flux rescaling (plan row C7; ADR 0007).
    "flux_rescale": 0.0,
    "flux_k_min": 0.5,
    "flux_k_max": 1.5,
}


def _check_paper_variant(paper_model: bool, settings: Settings) -> None:
    """The 2.5 flags apply to the paper model only: elsewhere they would be ignored.

    Raises:
        ConfigError: if a flag of :data:`_PAPER_VARIANT_DEFAULTS` is changed for another
            model.
    """
    if paper_model:
        return
    changed = [
        key for key, default in _PAPER_VARIANT_DEFAULTS.items() if getattr(settings, key) != default
    ]
    if changed:
        msg = f"{', '.join(changed)} apply only to supervised_model={PAPER_STAMP_CLASSIFIER}"
        raise ConfigError(msg)


def _leaked_train(settings: Settings, data: Path) -> list[int]:
    """The Train indices to drop: the stamps that duplicate a Validation or Test stamp.

    Found by ``protocol.leakage`` on the prepared uint8 stamps (``red``: ``[2995]``), the
    same check that ``make-splits`` stores and ``verify_splits`` repeats; ``[]`` when
    ``exclude_leaked_train`` is false.
    """
    if not settings.exclude_leaked_train:
        return []
    return train_exclusions(find_duplicates(load_prepared(data)))


def paper_module(settings: Settings) -> PaperClassifierModule:
    """The ``PaperClassifierModule`` of ``settings`` (the model drawn from the global generator).

    The paper model's fields, including the 2.5-models variant flags (``paper_arch``,
    ``group_fields``, ``context``, ``rot90_augment``) and the C5 / C6 augmentation
    (``aug_noise``, ``aug_feature_jitter``, ``synthetic_bogus``, ``aug_min_r2``) and the C7
    flux rescaling (``flux_rescale``, ``flux_k_min``, ``flux_k_max``), whose
    draws are seeded with ``seed``.

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
    )


def _build_paper(
    settings: Settings, data: Path, split_run: SplitRun | None = None
) -> tuple[ClassifierModule, DeviceTensorDataModule, dict[str, Any]]:
    """The paper model, its device-resident data and the ``paper_data`` metrics entry.

    The input is the float path (``paper_input: float32``, the paper's) or the interim uint8
    crops (``uint8``, a diagnostic); the leaked Train stamps are removed before the tiling.
    The splits are built once, at the data module's first ``setup``. With ``split_run``
    only Train and Validation are read, and the data module's roles are the run's parts
    (``device_data.protocol_splits``); the excluded Train stamps are the splits file's,
    which ``open_split_run`` has checked against a fresh leakage check.
    """
    if settings.paper_input == "float32" and "float32" not in read_meta(data)["images"]:
        # Fail before the leakage scan and the Trainer; float_crops says the same in setup.
        msg = (
            f"{data} has no float32 stamps (images_f32.npy), which the paper's input needs: "
            "run `astro-simclr prepare-data PKL --images float32 --overwrite` (or set "
            "paper_input=uint8, a diagnostic)"
        )
        raise DataError(msg)
    if split_run is None:
        excluded = _leaked_train(settings, data)
    else:
        excluded = list(split_run.splits.excluded_train)
    info: dict[str, Any] = {
        "input": settings.paper_input,
        "crop": PAPER_CROP,
        "nan_approximation": (
            settings.nan_approximation if settings.paper_input == "float32" else None
        ),
        "excluded_train": excluded,
    }

    def crops(names: Sequence[str]) -> dict[str, TensorSplit]:
        # One reader for both paths, so a split run gets the same input (and context
        # channels, task 2.5-models) as a paper-protocol run.
        if settings.paper_input == "float32":
            return float_crops(
                data,
                nan_approximation=settings.nan_approximation,
                names=names,
                context=settings.context != "none",
            )
        return interim_uint8_crops(data, names=names)

    def split_source(run: SplitRun) -> dict[str, TensorSplit]:
        pool = crops(("Train", "Validation"))  # never Test
        splits = protocol_splits(pool, run.parts(), run.splits.n_train)
        info["n_train"] = len(splits["Train"].labels)
        return splits

    def with_fits(splits: dict[str, TensorSplit]) -> dict[str, TensorSplit]:
        return _with_train_fits(settings, splits, info)

    def source() -> dict[str, TensorSplit]:
        if split_run is not None:
            return with_fits(split_source(split_run))
        splits = without_train_indices(crops(SPLITS), excluded)
        info["n_train"] = len(splits["Train"].labels)
        return with_fits(splits)

    seed_run(settings)
    module = paper_module(settings)
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


#: Plan the training augmentation in the loader (True) or in the step (False: the in-step
#: path, with host round trips; the same bits; kept for the equivalence test).
_PREFETCH = True


def _with_train_fits(
    settings: Settings, splits: dict[str, TensorSplit], info: dict[str, Any]
) -> dict[str, TensorSplit]:
    """``splits`` with the Train role's augmentation table, when C5 / C6 / C7 need it.

    C5 / C6 need each Train stamp's cross-channel fit (``class_augment.stamp_fits``), C7 its
    central PSF fit too (``flux_rescale.fit_central_psf``, appended). Computed once, on the
    run's own Train role (the fold's training part in a split run); the excluded shares go
    to ``info`` (``class_augment``, ``flux_rescale``).
    """
    if not (settings.aug_noise or settings.synthetic_bogus or settings.flux_rescale):
        return splits
    train = splits["Train"]
    fits = stamp_fits(train.images)
    info["class_augment"] = fit_exclusions(fits, train.labels, settings.aug_min_r2)
    if settings.flux_rescale:
        fits = np.concatenate([fits, fit_central_psf(train.images)], axis=1)
        info["flux_rescale"] = psf_exclusions(fits, train.labels)
    return {**splits, "Train": dataclasses.replace(train, fits=fits)}


def _check_split(command: str, settings: Settings) -> None:
    """A split run needs ``train-supervised`` on the ``carrasco2021`` schedule.

    Raises:
        ConfigError: on another command or schedule, or ``exclude_leaked_train: false``
            (the splits exclude the leaked stamps by construction).
    """
    if command != "train-supervised" or settings.schedule != CARRASCO2021:
        msg = (
            f"--splits trains only train-supervised on the {CARRASCO2021} schedule (e.g. "
            f"--preset {CARRASCO2021}), got {command} with schedule {settings.schedule}"
        )
        raise ConfigError(msg)
    if not settings.exclude_leaked_train:
        msg = (
            "--splits always leaves the leaked Train stamps out (they are in no fold): "
            "exclude_leaked_train must be true"
        )
        raise ConfigError(msg)


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
    module: ClassifierModule,
    stage: str,
    split: str,
    out: Path,
    *,
    index: npt.NDArray[np.int64] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Write the stage's per-stamp probabilities; return its ``metrics["predictions"]`` entry.

    The evaluation loaders are unshuffled, so the loader order is the split's file order
    and ``index`` is ``arange(N)`` by default; a split run passes its evaluation part's
    pool (or Validation) indices, in the same order. ``name`` is the file's stage name
    (default ``stage``).

    Raises:
        DataError: if ``index`` does not have one entry per predicted stamp.
    """
    labels = module.predictions[stage][1].numpy().astype(np.int64)
    if index is None:
        index = np.arange(labels.size, dtype=np.int64)
    if index.shape != labels.shape:
        raise DataError(f"{index.size} indices for {labels.size} predicted stamps")
    predictions = Predictions(module.probabilities[stage].float().numpy(), labels, index)
    file = predictions_file(name or stage)
    save_predictions(predictions, out / file)
    return {"file": file, "split": split, "n": int(labels.size)}


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


def _split_results(
    command: str,
    trainer: L.Trainer,
    module: ClassifierModule,
    datamodule: L.LightningDataModule,
    best: str,
    out: Path,
    split_run: SplitRun,
) -> dict[str, Any]:
    """A split run's results: the selection part (validate) and the evaluation part (test).

    The data module's ``Test`` role is the evaluation part (``device_data.protocol_splits``),
    so ``trainer.test`` predicts it and never touches the real Test split.
    """
    parts = split_run.parts()
    trainer.validate(module, datamodule=datamodule, ckpt_path=best, verbose=False)
    selection = _evaluation(module, "val")
    trainer.test(module, datamodule=datamodule, ckpt_path=best, verbose=False)
    evaluation = _evaluation(module, "test")
    predictions = {
        "evaluation": _write_predictions(
            module,
            "test",
            parts.evaluation_name,
            out,
            index=parts.evaluation_index,
            name="evaluation",
        )
    }
    figures = []
    for part, result in (("selection", selection), ("evaluation", evaluation)):
        title = (
            f"{command}: {split_run.mode} {part}: balanced accuracy "
            f"{100 * result['balanced_accuracy']:.1f} %"
        )
        figure = f"confusion_matrix_{part}.png"
        _plot_confusion_matrix(result["confusion_matrix"], title, out / figure)
        figures.append(figure)
    return {
        "selection_acc_best": selection["accuracy"],
        "evaluation_accuracy": evaluation["accuracy"],
        "evaluation_balanced_accuracy": evaluation["balanced_accuracy"],
        "labels": list(LABEL_NAMES),
        "evaluation": {"selection": selection, "evaluation": evaluation},
        "predictions": predictions,
        "figures": figures,
    }


def _classifier_results(
    command: str,
    trainer: L.Trainer,
    module: ClassifierModule,
    datamodule: L.LightningDataModule,
    best: str,
    out: Path,
    split_run: SplitRun | None = None,
) -> dict[str, Any]:
    if not best or not Path(best).is_file():
        raise DataError("no best checkpoint was written (did any validation run?)")
    if split_run is not None:
        return _split_results(command, trainer, module, datamodule, best, out, split_run)
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
            schedule with another command or model (and the paper model without it); with
            ``split``, another command or schedule, or a fold the file does not have; a
            ``carrasco2021`` resume from a checkpoint of another split run (splits file,
            mode or fold), or across a split run and a normal run.
        DataError: if the dataset is missing, invalid or smaller than one batch, or (the
            paper model's ``paper_input: float32``) has no float32 stamps; with ``split``,
            a splits file that is malformed or does not belong to the prepared data
            (``SplitsError``).
    """
    check_accelerator(settings.accelerator)
    _check_names(settings)
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
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    metrics = _base_metrics(command, settings, trainer, checkpoint, data)
    metrics["resumed_from"] = None if ckpt_path is None else str(ckpt_path)
    metrics["timing"] = timer.report()
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
        metrics.update(
            _classifier_results(
                command, trainer, module, datamodule, checkpoint.best_model_path, out, split_run
            )
        )
    else:
        metrics["pretrained_encoder"] = settings.encoder_name
    cleaned = cast(dict[str, Any], _finite(metrics))
    text = json.dumps(cleaned, indent=2, allow_nan=False) + "\n"
    (out / METRICS_FILE).write_text(text, encoding="utf-8")
    return cleaned
