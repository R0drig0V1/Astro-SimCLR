"""Schemas of the ZTF stamp files, the in-memory representation, and the data errors.

Every reader in :mod:`astro_simclr.data` (the pickle loader and the prepared ``.npy``
layout) returns a :class:`StampData`: one :class:`SplitData` per split, with the stamps
already converted to uint8 (and, on request, also kept as float32). The two file kinds,
checked against the real files on 2026-11-07 (``docs/data.md``):

- **classifier** (``td_ztf_stamp_17_06_20*.pkl``): ``Train``, ``Validation`` and ``Test``,
  each with ``images`` (63x63x3 floats in [0, 1]), ``labels`` (integers 0-4, see
  :data:`LABEL_NAMES`) and ``features`` (26 values per stamp, see :data:`FEATURE_NAMES`).
- **pretraining** (``td_ztf_stamp_simclr_300.pkl``): ``Train`` holds only ``images``;
  ``Validation`` and ``Test`` are copies of the classifier splits.

Errors are :class:`DataError` subclasses (``ValueError``), and they name the file, split
and key involved, so that a bad file is diagnosed from the message alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "DISCARDED_FEATURES",
    "FEATURE_NAMES",
    "LABEL_NAMES",
    "SPLITS",
    "STAMP_SHAPE",
    "DataError",
    "Kind",
    "SchemaError",
    "SplitData",
    "StampData",
    "TruncatedFileError",
]

#: Class names, indexed by label (``utils/dataset.py:label_names``).
LABEL_NAMES: tuple[str, ...] = ("AGN", "SN", "VS", "Asteroid", "Bogus")

#: The 26 ALeRCE features stored per stamp, in file order (``utils/dataset.py``). The real
#: files hold them already standardized on the Train split (mean 0, std 1 per column),
#: except ``isdiffpos``, scaled to std 1 but not centred (its values are 0 and 2.12).
FEATURE_NAMES: tuple[str, ...] = (
    "sgscore1",
    "distpsnr1",
    "sgscore2",
    "distpsnr2",
    "sgscore3",
    "distpsnr3",
    "isdiffpos",
    "fwhm",
    "magpsf",
    "sigmapsf",
    "ra",
    "dec",
    "diffmaglim",
    "rb",
    "distnr",
    "magnr",
    "classtar",
    "ndethist",
    "ncovhist",
    "ecl_lat",
    "ecl_long",
    "gal_lat",
    "gal_long",
    "non_detections",
    "chinr",
    "sharpnr",
)

#: Feature columns the legacy datasets drop (``discarted_features=[13, 14, 15]``: ``rb``,
#: ``distnr``, ``magnr``), leaving the 23 that the classifiers read.
DISCARDED_FEATURES: tuple[int, ...] = (13, 14, 15)

#: Split names, in file order.
SPLITS: tuple[str, ...] = ("Train", "Validation", "Test")

#: Shape of one stamp: height, width and the (science, template, difference) channels.
STAMP_SHAPE: tuple[int, int, int] = (63, 63, 3)

#: ``classifier``: every split is labelled. ``pretraining``: ``Train`` holds only images.
type Kind = Literal["classifier", "pretraining"]


class DataError(ValueError):
    """A data file cannot be used: missing, truncated, corrupt or of the wrong schema."""


class TruncatedFileError(DataError):
    """A pickle ends before its STOP opcode (an interrupted download or copy)."""


class SchemaError(DataError):
    """A file does not match its schema; the message names the file, split and key.

    Attributes:
        path: the offending file or prepared directory.
        split: the split (``Train``, ...), or ``None`` for a file-level problem.
        key: the key inside the split (``images``, ...), or ``None``.
    """

    def __init__(
        self, path: Path, message: str, *, split: str | None = None, key: str | None = None
    ) -> None:
        self.path = path
        self.split = split
        self.key = key
        where = "/".join(part for part in (split, key) if part is not None)
        super().__init__(f"{path}: {where + ': ' if where else ''}{message}")


@dataclass(frozen=True, slots=True)
class SplitData:
    """One split, validated: ``images`` is uint8 (N, 63, 63, 3).

    ``labels`` is int64 (N,) with values in ``range(len(LABEL_NAMES))``; ``features`` is
    float64 (N, 26) with every value finite. Both are ``None`` for the unlabelled
    pretraining ``Train`` split. ``images_f32`` holds the stored stamps themselves as
    float32 (N, 63, 63, 3), finite and in [0, 1], for the paper's float input path
    (:mod:`astro_simclr.data.float_input`); it is ``None`` unless the data was loaded or
    prepared with ``float32=True`` (``prepare-data --images float32``). The arrays may be
    read-only memory maps.
    """

    images: npt.NDArray[np.uint8]
    labels: npt.NDArray[np.int64] | None
    features: npt.NDArray[np.float64] | None
    images_f32: npt.NDArray[np.float32] | None = None

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def class_counts(self) -> list[int] | None:
        """Number of stamps per class, indexed by label; ``None`` if the split is unlabelled."""
        if self.labels is None:
            return None
        counts = np.bincount(self.labels, minlength=len(LABEL_NAMES))
        return [int(c) for c in counts]


@dataclass(frozen=True, slots=True)
class StampData:
    """A whole data file: its kind and its splits (always the three of :data:`SPLITS`)."""

    kind: Kind
    splits: dict[str, SplitData]
