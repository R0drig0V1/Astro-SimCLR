"""Per-stamp predictions of a classifier run: the file format, loading and seed ensembling.

Every classifier run (``experiments.run_command``) writes, next to ``metrics.json``, one
file per evaluated stage, ``predictions_val.npz`` and ``predictions_test.npz``, for the
checkpoint it selected. The comparisons of task 2.3 work on these, never on the summary
accuracies, because they pair the two variants stamp by stamp. Arrays (``np.savez``, read
with ``allow_pickle=False``):

=================  =====================  =================================================
``probabilities``  float32 (N, 5)         softmax of the logits, columns in ``LABEL_NAMES``
``labels``         int64 (N,)             the true class of each stamp
``index``          int64 (N,)             the stamp's index in the evaluated set: its
                                          position in the split (file order) for a normal
                                          run, the pool index for a CV fold
=================  =====================  =================================================

``metrics.json["predictions"][stage]`` names the file and the evaluated set (``split``:
``Validation``, ``Test``, or a fold's evaluation part).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from astro_simclr.data.schema import LABEL_NAMES, DataError

__all__ = [
    "PREDICTION_KEYS",
    "Predictions",
    "check_paired",
    "ensemble",
    "load_predictions",
    "predictions_file",
    "save_predictions",
    "seed_mean_correct",
]

#: The arrays of a predictions file.
PREDICTION_KEYS: Final = ("probabilities", "labels", "index")


def predictions_file(stage: str) -> str:
    """The file name of a stage's predictions: ``predictions_<stage>.npz``."""
    return f"predictions_{stage}.npz"


@dataclass(frozen=True, slots=True)
class Predictions:
    """The per-stamp output of one run on one evaluated set (module docstring)."""

    probabilities: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]
    index: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        n = self.labels.shape[0] if self.labels.ndim == 1 else -1
        if n <= 0 or self.probabilities.shape != (n, len(LABEL_NAMES)) or self.index.shape != (n,):
            msg = (
                f"predictions need probabilities (N, {len(LABEL_NAMES)}), labels (N,) and "
                f"index (N,) with N > 0, got {self.probabilities.shape}, "
                f"{self.labels.shape} and {self.index.shape}"
            )
            raise ValueError(msg)

    @property
    def predicted(self) -> npt.NDArray[np.int64]:
        """The predicted class of each stamp (argmax of the probabilities)."""
        return np.argmax(self.probabilities, axis=1).astype(np.int64)

    @property
    def correct(self) -> npt.NDArray[np.bool_]:
        """Whether each stamp is classified correctly."""
        return np.asarray(self.predicted == self.labels)


def save_predictions(predictions: Predictions, path: Path) -> None:
    """Write ``predictions`` to ``path`` (``.npz``, uncompressed).

    Raises:
        OSError: if the file cannot be written.
    """
    with path.open("wb") as file:
        np.savez(
            file,
            probabilities=predictions.probabilities.astype(np.float32),
            labels=predictions.labels.astype(np.int64),
            index=predictions.index.astype(np.int64),
        )


def load_predictions(path: Path) -> Predictions:
    """Read a predictions file (never unpickling: ``allow_pickle=False``).

    Raises:
        DataError: if the file is missing, unreadable, or lacks a key or has bad shapes.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in PREDICTION_KEYS}
    except (OSError, ValueError, KeyError) as err:
        raise DataError(f"{path}: not a predictions file ({err})") from None
    try:
        return Predictions(
            arrays["probabilities"].astype(np.float32),
            arrays["labels"].astype(np.int64),
            arrays["index"].astype(np.int64),
        )
    except ValueError as err:
        raise DataError(f"{path}: {err}") from None


def check_paired(runs: Sequence[Predictions]) -> None:
    """Check that every run evaluated the same stamps, in the same order, with the same labels.

    Raises:
        ValueError: if ``runs`` is empty or two runs differ in ``index`` or ``labels``.
    """
    if not runs:
        raise ValueError("no predictions given")
    first = runs[0]
    for position, run in enumerate(runs[1:], start=1):
        if not np.array_equal(run.index, first.index):
            raise ValueError(f"run {position} evaluated other stamps than run 0 (index differs)")
        if not np.array_equal(run.labels, first.labels):
            raise ValueError(f"run {position} has other labels than run 0 for the same stamps")


def ensemble(runs: Sequence[Predictions]) -> Predictions:
    """The seed ensemble: the mean of the runs' probabilities, stamp by stamp.

    This is what McNemar compares (plan: average the seeds' probabilities per stamp first;
    never pool seed x stamp pairs, which are not independent).

    Raises:
        ValueError: if the runs are not paired (:func:`check_paired`).
    """
    check_paired(runs)
    mean = np.mean([run.probabilities.astype(np.float64) for run in runs], axis=0)
    return Predictions(mean.astype(np.float32), runs[0].labels, runs[0].index)


def seed_mean_correct(runs: Sequence[Predictions]) -> npt.NDArray[np.float64]:
    """Per stamp, the fraction of runs (seeds) that classify it correctly.

    Its mean is the mean over seeds of the runs' accuracies, which is what the paired
    bootstrap resamples (plan: resample the stamps, averaging over seeds).

    Raises:
        ValueError: if the runs are not paired (:func:`check_paired`).
    """
    check_paired(runs)
    return np.mean([run.correct for run in runs], axis=0, dtype=np.float64)
