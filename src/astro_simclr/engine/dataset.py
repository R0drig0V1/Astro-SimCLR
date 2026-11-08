"""One dataset API over the prepared kinds, for every data tier (task 4.3, ADR 0016).

A :class:`Dataset` is a set of row-aligned arrays, the only thing the tiers (:mod:`.tiers`)
read:

- ``images``: float32 ``(N, ...)``, any per-row shape, as stored (the prepared stamps are
  ``(N, 63, 63, 3)``, channels last; the paper's crops ``(N, 3, 21, 21)``). It may be a
  read-only memory map: the ``memmap`` tier reads it in place, the other two copy it into
  memory once.
- ``features`` (float32 ``(N, F)``), ``labels`` (int64 ``(N,)``, with ``n_classes``) and
  ``fits`` (float32 ``(N, K)``, the ``class_augment`` stamp fits of C5-C7): optional, and
  always in memory (they are small next to the images).
- ``rows``: an optional selection of the stored rows, in order (a split of a pool, a fold).
  Row ``i`` of the dataset is stored row ``rows[i]``. The ``memmap`` tier keeps the
  selection and reads through it, so selecting a split of a large pool copies nothing.

The kinds and their adapters:

- ``classifier`` (:func:`prepared_split` on a classifier layout): the stored float32
  stamps (``images_f32.npy``, 63x63x3, a memory map), int64 labels and the 23 kept
  features as float32;
- ``unlabelled`` (:func:`prepared_split` on a ``pretraining`` layout's ``Train``): the
  images only;
- ``raw-alerts`` (:func:`raw_alerts_dataset`): the raw cutouts (``cutouts.npy``, NaN and
  inf kept), ``index["label"]``, and the 23 CD21 features as float32 (not standardised,
  NaN kept) when the layout has them; a split selects rows by ``candid``;
- ``arrays`` (:class:`Dataset` itself): any float32 images and columns, for example a
  ``device_data.TensorSplit``'s arrays (the paper's crops).

Only float32 stamps are served: the uint8 ``images.npy`` is the quantised legacy input
(plan finding F6); storing images as uint8 or float16 is an explicit, recorded tier choice
(``TierChoice.storage_dtype``), applied by the tier, never by this module. The model input
transform (channels first, the crop, the per-channel min-max) is the task's: it runs on the
device batch, so it is the same on every tier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.raw_alerts import load_raw_alerts
from astro_simclr.data.schema import DISCARDED_FEATURES, LABEL_NAMES, SPLITS, DataError

__all__ = [
    "COLUMNS",
    "Dataset",
    "DatasetKind",
    "prepared_split",
    "raw_alerts_dataset",
]

#: What a dataset was built from (module docstring).
type DatasetKind = Literal["classifier", "unlabelled", "raw-alerts", "arrays"]
#: The column names, in batch order.
COLUMNS: Final = ("images", "features", "labels", "fits")


@dataclass(frozen=True, slots=True)
class Dataset:
    """Row-aligned arrays and an optional row selection (module docstring).

    Raises:
        DataError: if ``images`` is not float32 with at least 2 dimensions, a column has
            another dtype, rank or row count, ``labels`` come without ``n_classes`` (or the
            reverse) or hold a value outside ``range(n_classes)``, ``rows`` is not a 1-D
            integer array of stored rows, or the dataset is empty.
    """

    images: npt.NDArray[np.float32]
    features: npt.NDArray[np.float32] | None = None
    labels: npt.NDArray[np.int64] | None = None
    fits: npt.NDArray[np.float32] | None = None
    n_classes: int | None = None
    rows: npt.NDArray[np.int64] | None = None
    kind: DatasetKind = "arrays"

    def __post_init__(self) -> None:
        """Check dtypes, shapes, row counts, labels and the selection."""
        if self.images.dtype != np.float32 or self.images.ndim < 2:
            msg = f"images must be float32 (N, ...), got {self.images.dtype} {self.images.shape}"
            raise DataError(msg)
        self._check_columns(int(self.images.shape[0]))
        self._check_labels()
        if self.rows is not None:
            object.__setattr__(self, "rows", self._checked_rows(self.rows))
        if len(self) == 0:
            raise DataError("the dataset is empty")

    def _check_columns(self, stored: int) -> None:
        for name, dtype, ndim in (
            ("features", np.float32, 2),
            ("labels", np.int64, 1),
            ("fits", np.float32, 2),
        ):
            array = getattr(self, name)
            if array is not None and (
                array.dtype != dtype or array.ndim != ndim or array.shape[0] != stored
            ):
                msg = (
                    f"{name} must be {np.dtype(dtype)} with {ndim} dimensions and {stored} "
                    f"rows, got {array.dtype} {array.shape}"
                )
                raise DataError(msg)

    def _check_labels(self) -> None:
        if (self.labels is None) != (self.n_classes is None):
            raise DataError("labels and n_classes go together")
        if self.labels is None or self.n_classes is None:
            return
        if isinstance(self.n_classes, bool) or self.n_classes < 2:
            raise DataError(f"n_classes must be an integer >= 2, got {self.n_classes!r}")
        if self.labels.size and (self.labels.min() < 0 or self.labels.max() >= self.n_classes):
            raise DataError(f"labels must be in range({self.n_classes})")

    def _checked_rows(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        stored = int(self.images.shape[0])
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise DataError(f"rows must be a 1-D integer array, got {rows.dtype} {rows.shape}")
        if rows.size and (rows.min() < 0 or rows.max() >= stored):
            raise DataError(f"rows must be in range({stored})")
        return rows.astype(np.int64, copy=False)

    def __len__(self) -> int:
        """The number of dataset rows (the selection's length, or the stored rows)."""
        return int(self.images.shape[0] if self.rows is None else self.rows.shape[0])

    @property
    def labelled(self) -> bool:
        """Whether the dataset has labels and features (what the classifier batch needs)."""
        return self.labels is not None and self.features is not None

    def stored_rows(self, rows: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """The stored rows of dataset rows ``rows`` (the identity without a selection)."""
        index = np.asarray(rows, dtype=np.int64)
        return index if self.rows is None else self.rows[index]

    def labels_of_rows(self) -> npt.NDArray[np.int64] | None:
        """The labels of the dataset rows, in order (what the tiling reads), or ``None``."""
        if self.labels is None:
            return None
        return self.labels if self.rows is None else self.labels[self.rows]

    def columns(self) -> dict[str, npt.NDArray[np.generic]]:
        """The present stored columns, by name, in :data:`COLUMNS` order (no selection applied)."""
        present = ((name, getattr(self, name)) for name in COLUMNS)
        return {name: array for name, array in present if array is not None}

    def select(self, rows: npt.ArrayLike) -> Dataset:
        """The dataset of rows ``rows`` of this one, in that order (composes selections).

        Raises:
            DataError: if a row is outside ``range(len(self))``, or ``rows`` is empty.
        """
        index = np.asarray(rows, dtype=np.int64).reshape(-1)
        if index.size and (index.min() < 0 or index.max() >= len(self)):
            raise DataError(f"rows must be in range({len(self)})")
        return replace(self, rows=self.stored_rows(index))

    def materialised(self) -> Dataset:
        """The same rows as in-memory, writable, contiguous arrays and no selection.

        What the ``device`` and ``pinned`` tiers hold; reads every selected image once.
        """

        rows = self.rows

        def take[G: np.generic](array: npt.NDArray[G]) -> npt.NDArray[G]:
            if rows is None:
                return np.array(array, copy=True, order="C")
            return np.ascontiguousarray(array[rows])

        return Dataset(
            images=take(self.images),
            features=None if self.features is None else take(self.features),
            labels=None if self.labels is None else take(self.labels),
            fits=None if self.fits is None else take(self.fits),
            n_classes=self.n_classes,
            kind=self.kind,
        )


def _kept_features(features: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    """The 23 kept features of the 26 stored ones, as float32 (``device_data._labelled``)."""
    kept = np.delete(np.asarray(features), list(DISCARDED_FEATURES), axis=1)
    return kept.astype(np.float32)


def prepared_split(path: Path, split: str) -> Dataset:
    """Split ``split`` of a prepared layout (format 1 or 2): ``classifier`` or ``unlabelled``.

    The images are the stored float32 stamps (``images_f32.npy``, memory-mapped, (N, 63,
    63, 3)); a labelled split also gets its labels (int64, ``n_classes`` 5) and its 23 kept
    features (float32, columns 13-15 dropped, as every classifier reads them). A
    ``pretraining`` layout's ``Train`` is ``unlabelled``: images only.

    Raises:
        DataError: if ``path`` is not a valid prepared layout (see ``load_prepared``),
            ``split`` is not one of ``Train``, ``Validation``, ``Test``, or the layout has
            no float32 stamps (the message names the ``prepare-data`` command).
    """
    if split not in SPLITS:
        raise DataError(f"split must be one of {SPLITS}, got {split!r}")
    data = load_prepared(path)
    content = data.splits[split]
    if content.images_f32 is None:
        msg = (
            f"{path} has no float32 stamps (images_f32.npy): run `astro-simclr prepare-data "
            "PKL --images float32 --overwrite` (the uint8 stamps are quantised, plan F6)"
        )
        raise DataError(msg)
    if content.labels is None or content.features is None:
        return Dataset(images=content.images_f32, kind="unlabelled")
    return Dataset(
        images=content.images_f32,
        features=_kept_features(content.features),
        labels=np.array(content.labels, dtype=np.int64, copy=True),
        n_classes=len(LABEL_NAMES),
        kind="classifier",
    )


def raw_alerts_dataset(path: Path, *, candids: Sequence[int] | None = None) -> Dataset:
    """The ``raw-alerts`` layout ``path`` (format 3), or the alerts ``candids`` of it, in order.

    The images are the raw cutouts (memory-mapped, (N, 63, 63, 3), NaN and inf kept: the
    task's input path normalises them); the labels are ``index["label"]`` with the layout's
    ``n_classes``; the features, when the layout has them, are its 23 CD21 features as
    float32 (not standardised, NaN kept). ``candids`` (a split of task 3.2) selects rows
    through ``RawAlerts.rows_of`` without copying the cutouts.

    Raises:
        DataError: if ``path`` is not a valid ``raw-alerts`` layout (``load_raw_alerts``),
            a ``candid`` is not in it, or ``candids`` is empty.
    """
    alerts = load_raw_alerts(path)
    rows = None if candids is None else alerts.rows_of(candids)
    features = None if alerts.features is None else alerts.features.astype(np.float32)
    return Dataset(
        images=alerts.cutouts,
        features=features,
        labels=alerts.labels,
        n_classes=alerts.n_classes,
        rows=rows,
        kind="raw-alerts",
    )
