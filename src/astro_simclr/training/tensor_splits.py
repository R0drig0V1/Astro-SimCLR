"""The paper model's splits as arrays: :class:`TensorSplit` and the crop builders (Lightning-free).

Split out of ``training.device_data`` (task 4.6) so that the plain engine (ADR 0014) builds
exactly the arrays the Lightning ``DeviceTensorDataModule`` trains on without importing
Lightning; ``device_data`` re-exports every name, so existing imports are unchanged. The
design (the float path, the uint8 diagnostic, the leaked stamps, the roles of a split run)
is described in ``device_data``'s docstring.

- :class:`TensorSplit`: one labelled split, float32 crops (or whole stamps), the 23 kept
  features, int64 labels and the optional C5-C7 fits;
- :func:`float_crops` (the paper's input) and :func:`interim_uint8_crops` (the diagnostic);
- :func:`without_train_indices` (the leaked Train stamps) and :func:`protocol_splits` (a
  split run's roles);
- :func:`check_split_identity`: a resume must continue the same split run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.config import ConfigError
from astro_simclr.data.float_input import context_split, crop_split
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import (
    DISCARDED_FEATURES,
    SPLITS,
    STAMP_SHAPE,
    DataError,
    SplitData,
)
from astro_simclr.models import N_CLASSES, N_FEATURES, check_n_classes
from astro_simclr.paper_model import PAPER_CROP
from astro_simclr.protocol.leakage import train_keep_indices
from astro_simclr.protocol.split_runs import RunParts
from astro_simclr.training.class_augment import FIT_COLUMNS
from astro_simclr.training.flux_rescale import PSF_COLUMNS

__all__ = [
    "IMAGE_CHANNELS",
    "IMAGE_SIDES",
    "SPLIT_RUN_STATE",
    "Batch",
    "PlannedTrainBatch",
    "Planner",
    "SplitSource",
    "TensorSplit",
    "TrainBatch",
    "check_split_identity",
    "float_crops",
    "interim_uint8_crops",
    "protocol_splits",
    "without_train_indices",
]

#: The key of a split run's identity in the training loader's checkpoint state.
SPLIT_RUN_STATE: Final = "split_run"
#: The image channels a :class:`TensorSplit` may have: the crop, or the crop and the context.
IMAGE_CHANNELS = (3, 6)
#: The image sides a :class:`TensorSplit` may have: the paper's crop, or the whole stamp
#: (the ConvNeXt rows, plan row B8; whole stamps have 3 channels only).
IMAGE_SIDES = (PAPER_CROP, STAMP_SHAPE[0])

#: (crops, features, one-hot labels), as ``ClassifierModule`` expects.
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#: A training batch: a :data:`Batch`, plus the stamp fits when the Train split has them.
TrainBatch = Batch | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
#: A :data:`TrainBatch` followed by its uploaded augmentation plan.
PlannedTrainBatch = tuple[Any, ...]
#: ``planner(host_batch, step) -> plan``: ``PaperClassifierModule.augment_plan``.
Planner = Callable[[Any, int], dict[str, Any]]
_TRAIN, _VALIDATION, _TEST = SPLITS


@dataclass(frozen=True, slots=True)
class TensorSplit:
    """One labelled split, ready for the paper model.

    Attributes:
        images: float32 (N, C, S, S) crops, channels first; S is 21 and C is 3, or 6 with
            the context channels (:data:`IMAGE_CHANNELS`), or S is 63 (whole stamps) and C
            is 3 (:data:`IMAGE_SIDES`).
        features: float32 (N, 23), the kept alert features.
        labels: int64 (N,), values in ``range(n_classes)``.
        fits: float32 (N, 6) ``class_augment.stamp_fits`` of the crops, or (N, 11) with
            ``flux_rescale.fit_central_psf`` after them (C7), or None (the default). Only a
            training split that C5 / C6 / C7 augment carries them; the training batches
            then have a fourth element (:class:`DeviceTensorDataModule`).
        n_classes: the number of classes, the width of the one-hot labels (default 5,
            CD21's; an N-class dataset passes its own, task 3.1b).

    Raises:
        ValueError: if a dtype or shape is wrong, the lengths differ, N is 0, ``n_classes``
            is not an integer >= 2 or a label is out of range.
    """

    images: npt.NDArray[np.float32]
    features: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]
    fits: npt.NDArray[np.float32] | None = None
    n_classes: int = N_CLASSES

    def __post_init__(self) -> None:
        """Check dtypes, shapes, lengths and label values."""
        check_n_classes(self.n_classes)
        n = len(self.labels)
        channels = self.images.shape[1] if self.images.ndim == 4 else 0
        channels = channels if channels in IMAGE_CHANNELS else IMAGE_CHANNELS[0]
        side = self.images.shape[-1] if self.images.ndim == 4 else 0
        side = side if side in IMAGE_SIDES else PAPER_CROP
        if side != PAPER_CROP:
            channels = IMAGE_CHANNELS[0]  # whole stamps: no context channels
        expected = {
            "images": (np.float32, (n, channels, side, side)),
            "features": (np.float32, (n, N_FEATURES)),
            "labels": (np.int64, (n,)),
        }
        for name, (dtype, shape) in expected.items():
            array = getattr(self, name)
            if array.dtype != dtype or array.shape != shape:
                wanted = str(shape)
                if name == "images":  # (N, 3 or 6, 21, 21) or (N, 3, 63, 63)
                    choices = " or ".join(map(str, IMAGE_CHANNELS))
                    big = IMAGE_SIDES[1]
                    wanted = (
                        f"(N, {choices}, {PAPER_CROP}, {PAPER_CROP}) or (N, 3, {big}, {big}) "
                        f"with N = {n}"
                    )
                msg = (
                    f"TensorSplit.{name} must be {np.dtype(dtype)} {wanted}, "
                    f"got {array.dtype} {array.shape}"
                )
                raise ValueError(msg)
        widths = (len(FIT_COLUMNS), len(FIT_COLUMNS) + len(PSF_COLUMNS))
        if self.fits is not None and (
            self.fits.dtype != np.float32 or self.fits.shape not in {(n, w) for w in widths}
        ):
            msg = (
                f"TensorSplit.fits must be float32 ({n}, {widths[0]}) or ({n}, {widths[1]}), "
                f"got {self.fits.dtype} {self.fits.shape}"
            )
            raise ValueError(msg)
        if n == 0:
            raise ValueError("TensorSplit is empty")
        if self.labels.min() < 0 or self.labels.max() >= self.n_classes:
            raise ValueError(f"labels must be in range({self.n_classes})")


#: The splits of a run (``Train``, ``Validation``, ``Test``), or a callable that builds them.
SplitSource = Mapping[str, TensorSplit] | Callable[[], Mapping[str, TensorSplit]]


def check_split_identity(stored: object, current: Mapping[str, object] | None) -> None:
    """Refuse a resume across split runs: ``stored`` (from a checkpoint) must be ``current``.

    ``None`` is a normal run (no ``--splits``). Both are ``SplitRun.identity()`` mappings
    otherwise.

    Raises:
        ConfigError: if they differ (the message names both).
    """
    expected = None if current is None else dict(current)
    if stored == expected:
        return

    def describe(identity: object) -> str:
        if not isinstance(identity, Mapping):
            return "a normal run (no --splits)"
        part = "the screen" if identity.get("fold") is None else f"fold {identity.get('fold')}"
        return f"{part} of the splits file with SHA-256 {identity.get('splits_sha256')}"

    msg = (
        f"the checkpoint is from {describe(stored)}, but this run is {describe(expected)}: "
        "a resume must continue the same run (its weights saw other stamps; use another "
        "--output-dir or no --resume)"
    )
    raise ConfigError(msg)


def _centre_crop_unit(images: npt.NDArray[np.uint8], size: int) -> npt.NDArray[np.float32]:
    """(N, H, W, 3) uint8 -> (N, 3, size, size) float32 in [0, 1], the centre crop."""
    side = images.shape[1]
    start = (side - size) // 2
    crop = np.asarray(images[:, start : start + size, start : start + size, :], dtype=np.float32)
    return np.ascontiguousarray(crop.transpose(0, 3, 1, 2) / np.float32(255.0))


def _check_side(side: int) -> None:
    if side not in IMAGE_SIDES:
        raise ValueError(f"side must be one of {IMAGE_SIDES}, got {side}")


def _labelled(
    path: Path, name: str, split: SplitData
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """The 23 kept features (float32) and the labels (int64) of a labelled split.

    Raises:
        DataError: if the split has no labels or features (a pretraining file).
    """
    if split.labels is None or split.features is None:
        raise DataError(f"{path}: {name} is unlabelled; the paper model needs labels")
    kept = np.delete(np.asarray(split.features), list(DISCARDED_FEATURES), axis=1)
    return kept.astype(np.float32), np.array(split.labels, dtype=np.int64, copy=True)


def interim_uint8_crops(
    path: Path, *, names: Sequence[str] = SPLITS, side: int = PAPER_CROP
) -> dict[str, TensorSplit]:
    """The splits ``names`` of the prepared layout ``path``, as crops of its uint8 stamps.

    The stand-in of task 2.2a, now the ``paper_input: uint8`` diagnostic: the centre
    ``side`` x ``side`` (21, or 63 for the whole stamp) of the stored uint8 stamps, divided
    by 255 (no re-normalisation on the crop), with the 23 kept features (columns 13-15
    dropped, as in ``StampDataset``) as float32.

    Raises:
        DataError: if ``path`` is not a valid prepared layout of a classifier file.
        ValueError: if ``side`` is not one of :data:`IMAGE_SIDES`.
    """
    _check_side(side)
    data = load_prepared(path)
    splits: dict[str, TensorSplit] = {}
    for name in names:
        split = data.splits[name]
        features, labels = _labelled(path, name, split)
        splits[name] = TensorSplit(
            images=_centre_crop_unit(split.images, side),
            features=features,
            labels=labels,
        )
    return splits


def float_crops(
    path: Path,
    *,
    nan_approximation: bool = True,
    names: Sequence[str] = SPLITS,
    context: bool = False,
    side: int = PAPER_CROP,
) -> dict[str, TensorSplit]:
    """The splits ``names`` of the prepared layout ``path`` through the paper's float path.

    Each split's stored float32 stamps (``images_f32.npy``, read from the memory map in
    chunks) go through ``crop_split(images, 21, nan_approximation=...)`` (task 2.1: ALeRCE's
    ``image_crop_at_center`` -> ``image_normalize_by_image`` -> ``image_nan_to_num``),
    then channels first. With ``context``, ``context_split(images, nan_approximation=...)``
    adds 3 channels after the crop's (task 2.5-models). ``side=63`` crops the whole stamp
    (``crop_split(images, 63, ...)``: the same min-max and NaN approximation on all its
    pixels; the ConvNeXt rows, plan row B8). The features are the 23 kept ones, as float32.
    For ``red`` the crops take about 270 MB (twice that with the context, 9 times at 63),
    plus a transient copy of the same size per split.

    Raises:
        DataError: if ``path`` is not a valid prepared layout of a classifier file, or it
            has no float32 stamps (the message names the ``prepare-data`` command).
        ValueError: if ``side`` is not one of :data:`IMAGE_SIDES`, or ``context`` is set
            with whole stamps.
    """
    _check_side(side)
    if context and side != PAPER_CROP:
        raise ValueError("the context channels go with the 21x21 crop, not whole stamps")
    data = load_prepared(path)
    splits: dict[str, TensorSplit] = {}
    for name in names:
        split = data.splits[name]
        if split.images_f32 is None:
            msg = (
                f"{path} has no float32 stamps (images_f32.npy), which the paper's input "
                f"needs: run `astro-simclr prepare-data PKL --images float32 --overwrite` "
                "(or set paper_input=uint8, a diagnostic)"
            )
            raise DataError(msg)
        features, labels = _labelled(path, name, split)
        crops = crop_split(split.images_f32, side, nan_approximation=nan_approximation)
        if context:
            crops = np.concatenate(
                (crops, context_split(split.images_f32, nan_approximation=nan_approximation)),
                axis=-1,
            )
        splits[name] = TensorSplit(
            images=np.ascontiguousarray(crops.transpose(0, 3, 1, 2)),
            features=features,
            labels=labels,
        )
    return splits


def without_train_indices(
    splits: Mapping[str, TensorSplit], excluded: Sequence[int]
) -> dict[str, TensorSplit]:
    """``splits`` with the Train stamps ``excluded`` removed; Validation and Test unchanged.

    The kept Train stamps stay in file order, so the tiling and the stream are those of the
    reduced split (the plan, task 2.3: the leaked stamp leaves Train only). Train's ``fits``, if
    any, follow its stamps.

    Raises:
        ValueError: if an excluded index is outside the Train split.
    """
    train = splits[_TRAIN]
    keep = train_keep_indices(len(train.labels), excluded)
    reduced = TensorSplit(
        images=np.ascontiguousarray(train.images[keep]),
        features=np.ascontiguousarray(train.features[keep]),
        labels=np.ascontiguousarray(train.labels[keep]),
        fits=None if train.fits is None else np.ascontiguousarray(train.fits[keep]),
        n_classes=train.n_classes,
    )
    return {**splits, _TRAIN: reduced}


def protocol_splits(
    splits: Mapping[str, TensorSplit], parts: RunParts, n_train: int
) -> dict[str, TensorSplit]:
    """The roles of a split run, gathered from Train and Validation by pool index.

    ``splits`` holds at least ``Train`` (all ``n_train`` stamps, in file order: the pool
    index of Train stamp ``i`` is ``i``) and ``Validation`` (pool index ``n_train + j``).
    Returns ``Train`` = ``parts.training`` (the only stamps trained on; the data module
    tiles them), ``Validation`` = ``parts.selection`` and ``Test`` = ``parts.evaluation``,
    each in pool order, which for the screen halves is Validation file order. The roles
    have no ``fits``: they are computed on the run's Train role afterwards.

    Raises:
        DataError: if Train does not have ``n_train`` stamps (the splits belong to other
            data) or a pool index is outside Train + Validation.
    """
    train, validation = splits[_TRAIN], splits[_VALIDATION]
    if len(train.labels) != n_train:
        msg = f"the splits have {n_train} Train stamps, the data {len(train.labels)}"
        raise DataError(msg)
    n_pool = n_train + len(validation.labels)

    def gather(pool: npt.NDArray[np.int64]) -> TensorSplit:
        if pool.size and (pool.min() < 0 or pool.max() >= n_pool):
            raise DataError(f"pool indices must be in range({n_pool})")
        in_train = pool < n_train
        rows_train, rows_val = pool[in_train], pool[~in_train] - n_train

        def take(attribute: str) -> npt.NDArray[Any]:
            first = getattr(train, attribute)[rows_train]
            second = getattr(validation, attribute)[rows_val]
            return np.ascontiguousarray(np.concatenate([first, second]))

        return TensorSplit(
            images=take("images"),
            features=take("features"),
            labels=take("labels"),
            n_classes=train.n_classes,
        )

    return {
        _TRAIN: gather(parts.training),
        _VALIDATION: gather(parts.selection),
        _TEST: gather(parts.evaluation),
    }
