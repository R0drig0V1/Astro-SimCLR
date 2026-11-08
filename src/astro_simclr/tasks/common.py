"""Run checks shared by both engines' runners (Lightning-free; task 4.6).

Moved from ``experiments`` (which re-exports them under their old names) so that the plain
engine's runner (``tasks.plain``) refuses exactly what ``run_command`` refuses, with the
same messages, without importing Lightning:

- :func:`check_accelerator` (:class:`AcceleratorError`) and :func:`dataset_path`;
- :func:`check_names` (the augmentation registry);
- :func:`check_schedule` and :func:`check_paper_variant`: the ``carrasco2021`` schedule,
  the paper model and its 2.5 flags go together;
- :func:`check_split`: a split run is ``train-supervised`` on ``carrasco2021``;
- :func:`refuse_rj23`: phase 3's ``rj23`` schedule has its own runner
  (``training.rj23.run_rj23``, task 3.4b), so both generic runners refuse it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import torch

from astro_simclr.augmentations import resolve_augmentation_name
from astro_simclr.config import CARRASCO2021, RJ23, ConfigError, Settings
from astro_simclr.data.prepared import is_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.paper_variants import paper_input_side
from astro_simclr.training.class_augment import DEFAULT_MIN_R2

__all__ = [
    "PAPER_STAMP_CLASSIFIER",
    "PAPER_VARIANT_DEFAULTS",
    "AcceleratorError",
    "check_accelerator",
    "check_names",
    "check_paper_variant",
    "check_schedule",
    "check_split",
    "dataset_path",
    "refuse_rj23",
]

#: ``train-supervised`` model of the ``carrasco2021`` schedule (task 2.2a).
PAPER_STAMP_CLASSIFIER: Final = "paper_stamp_classifier"


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


def refuse_rj23(settings: Settings) -> None:
    """The ``rj23`` schedule trains through ``training.rj23.run_rj23``, not a generic runner.

    Raises:
        ConfigError: for ``schedule: rj23`` (the message gives the command that trains it).
    """
    if settings.schedule == RJ23:
        msg = (
            f"schedule {RJ23} trains through training.rj23.run_rj23: `astro-simclr "
            "train-supervised --preset rj23-<scenario> --splits OBJECT_SPLITS`"
        )
        raise ConfigError(msg)


def check_names(settings: Settings) -> None:
    """Refuse an unknown augmentation name (``ConfigError``, not ``ValueError``)."""
    try:
        resolve_augmentation_name(settings.augmentation)
    except ValueError as err:
        raise ConfigError(str(err)) from None


def check_schedule(command: str, settings: Settings) -> None:
    """The ``carrasco2021`` schedule and the paper model go together, on 21x21 crops.

    ``image_size`` must be the side ``paper_arch`` takes: 21, or 63 (the whole stamp) for
    the ConvNeXt rows (plan row B8), so the recorded config says what the model saw.

    Raises:
        ConfigError: if only one of them is chosen, the command is not ``train-supervised``,
            or ``image_size`` is not the side of ``paper_arch``.
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
    side = paper_input_side(settings.paper_arch)
    if paper_model and settings.image_size != side:
        msg = (
            f"{PAPER_STAMP_CLASSIFIER} with paper_arch {settings.paper_arch} takes "
            f"{side}x{side} images: set image_size={side}"
        )
        raise ConfigError(msg)
    check_paper_variant(paper_model, settings)


#: The flags of task 2.5-models (architecture) and of C5 / C6 (augmentation), and their
#: defaults (the faithful model).
PAPER_VARIANT_DEFAULTS: Final[Mapping[str, object]] = {
    "paper_arch": "rot4",
    "group_fields": None,
    "context": "none",
    "rot90_augment": False,
    # The features the head reads (a 2.5 diagnostic; ADR 0016).
    "feature_set": "all",
    # The ConvNeXt backbone's initial weights (plan row B8; ADR 0022).
    "backbone_weights": "none",
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


def check_paper_variant(paper_model: bool, settings: Settings) -> None:
    """The 2.5 flags apply to the paper model only: elsewhere they would be ignored.

    Raises:
        ConfigError: if a flag of :data:`PAPER_VARIANT_DEFAULTS` is changed for another
            model.
    """
    if paper_model:
        return
    changed = [
        key for key, default in PAPER_VARIANT_DEFAULTS.items() if getattr(settings, key) != default
    ]
    if changed:
        msg = f"{', '.join(changed)} apply only to supervised_model={PAPER_STAMP_CLASSIFIER}"
        raise ConfigError(msg)


def check_split(command: str, settings: Settings) -> None:
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
