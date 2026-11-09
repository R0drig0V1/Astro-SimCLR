"""The phase-3 benchmark metrics: macro-F1, macro precision and recall, per class, confusion.

Task 3.2 of ``docs/phases/03-newer-benchmarks.md``. The RJ23 and C25 benchmarks (6 classes,
imbalanced) are scored with **macro-F1**, so this module computes, from integer labels and
predictions of any number of classes:

- per class: precision, recall and F1, with the support (true count) and the predicted
  count;
- their unweighted means over the classes (macro precision, macro recall, macro-F1) and
  the micro accuracy;
- the confusion matrix of counts, and the matrix with each row normalised to 100 (the
  percentages RJ23 shows: entry ``[i, j]`` is the percentage of true class ``i`` predicted
  as ``j``).

**Zero-division rule** (``docs/evaluation-protocol.md``, "Phase 3 benchmark metrics"):

- A class with true samples that is never predicted has precision 0 (0/0 is taken as 0),
  recall 0 and F1 0, and it **counts** in the macro means. Leaving it out would reward a
  model for never predicting a hard class.
- A class that is predicted but has no true sample has precision 0, recall 0 (0/0 taken as
  0) and F1 0, and it counts too: its false positives are real errors.
- A class **absent from both** the labels and the predictions has no defined score: its
  per-class values are ``None`` and it is left out of the macro means. This is the rule of
  :mod:`astro_simclr.training.metrics` (it leaves out classes with ``TP + FP + FN = 0``,
  as torchmetrics' macro averages did), so a phase-2 macro-F1 and this one agree.
- A row of the normalised confusion matrix with no true sample is all zeros (as
  ``training.metrics.confusion_matrix``).

**Prior-reweighted macro-F1.** Precision, and so macro-F1, depends on the class
prevalence of the evaluated set: a rare class's false positives come from the large
classes. To compare with a benchmark whose test set has other class counts (RJ23's), each
row of the confusion matrix is rescaled to a reference count, ``counts[i] / support_i x
reference_i`` (recall is unchanged), and macro precision and macro-F1 are recomputed from
it (:func:`reweighted_counts`). A row with no true sample cannot be rescaled and stays
zeros. Macro precision and macro-F1 are unchanged when every reference count is scaled by
one factor, so only the class *proportions* matter: :func:`rj23_reference` takes them from
RJ23's class totals (:data:`RJ23_CLASS_COUNTS`), which a stratified 60/20/20 split keeps in
its test set up to rounding. R5's headline metric is this reweighted macro-F1 (phase-3
plan, task 3.5).

F1 is computed as ``2 TP / (2 TP + FP + FN)``, which equals the harmonic mean of precision
and recall whenever both are defined and is 0 under the rule above when ``TP = 0``.
:func:`macro_f1_from_counts` does it on stacked count matrices, which is what the paired
bootstrap (:func:`astro_simclr.protocol.stats.paired_bootstrap_macro_f1`) resamples.

Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

__all__ = [
    "RJ23_CLASS_COUNTS",
    "BenchmarkMetrics",
    "ClassScores",
    "benchmark_metrics",
    "confusion_counts",
    "confusion_percent",
    "macro_f1",
    "macro_f1_from_counts",
    "reweighted_counts",
    "rj23_reference",
]

#: RJ23's class totals (Reyes-Jainaga et al. 2023; ``docs/phases/README.md``).
RJ23_CLASS_COUNTS: Final[dict[str, int]] = {
    "AGN": 9774,
    "SN": 3615,
    "VS": 10211,
    "Asteroid": 9180,
    "Bogus": 15950,
    "Satellite": 580,
}


def rj23_reference(label_names: Sequence[str]) -> tuple[float, ...] | None:
    """RJ23's class counts in the order of ``label_names``, or ``None`` if one is not RJ23's.

    The proportions are what the reweighting uses (module docstring), so the class totals
    serve for RJ23's test set.
    """
    if not all(name in RJ23_CLASS_COUNTS for name in label_names):
        return None
    return tuple(float(RJ23_CLASS_COUNTS[name]) for name in label_names)


Counts = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClassScores:
    """One class's scores; ``None`` where the class is absent from labels and predictions."""

    name: str
    support: int
    predicted: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """The scores of one set of predictions (module docstring for the rules)."""

    n: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: tuple[ClassScores, ...]
    counts: Counts
    percent: npt.NDArray[np.float64]
    #: Macro precision and macro-F1 with the rows rescaled to reference counts (``None``
    #: without a reference).
    reweighted_macro_precision: float | None = None
    reweighted_macro_f1: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready dict: the scalars, per-class scores by name and both matrices."""
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_class": {
                c.name: {
                    "support": c.support,
                    "predicted": c.predicted,
                    "precision": c.precision,
                    "recall": c.recall,
                    "f1": c.f1,
                }
                for c in self.per_class
            },
            "confusion_counts": self.counts.tolist(),
            "confusion_percent": self.percent.tolist(),
            "reweighted_macro_precision": self.reweighted_macro_precision,
            "reweighted_macro_f1": self.reweighted_macro_f1,
        }


def _labels_pair(
    labels: npt.ArrayLike, predicted: npt.ArrayLike, n_classes: int
) -> tuple[Counts, Counts]:
    true = np.asarray(labels)
    pred = np.asarray(predicted)
    if true.ndim != 1 or true.shape != pred.shape or true.size == 0:
        msg = (
            f"expected two non-empty 1-D label arrays of one shape, got {true.shape}, {pred.shape}"
        )
        raise ValueError(msg)
    if not (np.issubdtype(true.dtype, np.integer) and np.issubdtype(pred.dtype, np.integer)):
        raise ValueError("labels and predictions must be integer class indices")
    if n_classes < 2:
        raise ValueError(f"need at least 2 classes, got {n_classes}")
    for name, values in (("labels", true), ("predictions", pred)):
        if values.min() < 0 or values.max() >= n_classes:
            raise ValueError(f"{name} must be in range({n_classes})")
    return true.astype(np.int64), pred.astype(np.int64)


def confusion_counts(labels: npt.ArrayLike, predicted: npt.ArrayLike, n_classes: int) -> Counts:
    """The (n_classes, n_classes) matrix of counts: ``[i, j]`` = true ``i`` predicted ``j``.

    Raises:
        ValueError: if the inputs are not non-empty 1-D integer arrays of one shape, or a
            value is outside ``range(n_classes)``.
    """
    true, pred = _labels_pair(labels, predicted, n_classes)
    flat = np.bincount(true * n_classes + pred, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes).astype(np.int64)


def confusion_percent(counts: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Each row of a count matrix normalised to sum to 100; a row of zeros stays zeros.

    Raises:
        ValueError: if ``counts`` is not a square 2-D matrix.
    """
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    rows = matrix.sum(axis=1, keepdims=True)
    return np.divide(100 * matrix, rows, out=np.zeros_like(matrix), where=rows > 0)


def _ratio(num: npt.NDArray[np.float64], den: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``num / den`` with 0/0 taken as 0 (the zero-division rule)."""
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def _parts(
    counts: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"expected (..., C, C) count matrices, got shape {matrix.shape}")
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    fn = matrix.sum(axis=-1) - tp
    fp = matrix.sum(axis=-2) - tp
    return tp, fp, fn


def macro_f1_from_counts(counts: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Macro-F1 of stacked count matrices ``(..., C, C)``; returns shape ``(...)``.

    Classes with ``TP + FP + FN = 0`` (absent from labels and predictions) are left out of
    the mean; a matrix of zeros gives 0.

    Raises:
        ValueError: if the input is not a stack of square matrices.
    """
    tp, fp, fn = _parts(counts)
    den = 2 * tp + fp + fn
    present = den > 0
    f1 = _ratio(2 * tp, den)
    n_present = present.sum(axis=-1)
    return np.divide(
        (f1 * present).sum(axis=-1),
        n_present,
        out=np.zeros(n_present.shape, dtype=np.float64),
        where=n_present > 0,
    )


def reweighted_counts(counts: npt.ArrayLike, reference: Sequence[float]) -> npt.NDArray[np.float64]:
    """The count matrix with each row rescaled to sum to its reference count.

    A row with no true sample stays zeros (it cannot be rescaled).

    Raises:
        ValueError: if ``counts`` is not square, or ``reference`` has another length or a
            negative or non-finite value, or sums to 0.
    """
    matrix = np.asarray(counts, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {matrix.shape}")
    if ref.shape != (matrix.shape[0],) or not np.all(np.isfinite(ref)) or ref.min() < 0:
        raise ValueError(f"reference must hold {matrix.shape[0]} finite counts >= 0")
    if ref.sum() <= 0:
        raise ValueError("reference counts must not all be 0")
    return confusion_percent(matrix) / 100 * ref[:, None]


def _macro_precision(counts: npt.NDArray[np.float64]) -> float:
    tp, fp, fn = _parts(counts)
    present = (tp + fp + fn) > 0
    return float(_ratio(tp, tp + fp)[present].mean())


def macro_f1(labels: npt.ArrayLike, predicted: npt.ArrayLike, n_classes: int) -> float:
    """Macro-F1 of integer predictions (module docstring for the zero-division rule).

    Raises:
        ValueError: as :func:`confusion_counts`.
    """
    return float(macro_f1_from_counts(confusion_counts(labels, predicted, n_classes)))


def benchmark_metrics(
    labels: npt.ArrayLike,
    predicted: npt.ArrayLike,
    label_names: Sequence[str],
    *,
    reference_counts: Sequence[float] | None = None,
) -> BenchmarkMetrics:
    """Every benchmark score of integer predictions, one class per name of ``label_names``.

    With ``reference_counts`` (one per class) it also gives the prior-reweighted macro
    precision and macro-F1 (module docstring).

    Raises:
        ValueError: as :func:`confusion_counts` (with ``n_classes = len(label_names)``), or
            as :func:`reweighted_counts`.
    """
    n_classes = len(label_names)
    counts = confusion_counts(labels, predicted, n_classes)
    tp, fp, fn = _parts(counts)
    present = (tp + fp + fn) > 0
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    per_class = tuple(
        ClassScores(
            name,
            int(tp[c] + fn[c]),
            int(tp[c] + fp[c]),
            float(precision[c]) if present[c] else None,
            float(recall[c]) if present[c] else None,
            float(f1[c]) if present[c] else None,
        )
        for c, name in enumerate(label_names)
    )
    reweighted = None if reference_counts is None else reweighted_counts(counts, reference_counts)
    # At least one class is present: the inputs are non-empty.
    return BenchmarkMetrics(
        n=int(counts.sum()),
        accuracy=float(tp.sum() / counts.sum()),
        macro_precision=float(precision[present].mean()),
        macro_recall=float(recall[present].mean()),
        macro_f1=float(f1[present].mean()),
        per_class=per_class,
        counts=counts,
        percent=confusion_percent(counts),
        reweighted_macro_precision=None if reweighted is None else _macro_precision(reweighted),
        reweighted_macro_f1=None if reweighted is None else float(macro_f1_from_counts(reweighted)),
    )
