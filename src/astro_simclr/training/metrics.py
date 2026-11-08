"""The classification metrics and the confusion matrix, shared by every classifier module.

The legacy ``utils/training.py`` had four copies of a ``metrics`` method and four of a
``confusion_matrix`` method, written for torchmetrics 0.6. This module is the one
implementation that replaces them, on torchmetrics >= 1 (``task="multiclass"``).

Averaging, as in the legacy code:

- **accuracy is micro** (correct predictions / all predictions). torchmetrics 0.6
  ``Accuracy(num_classes=5)`` defaulted to micro; ``MulticlassAccuracy`` now defaults to
  *macro*, so ``average="micro"`` is passed explicitly. Every legacy accuracy (the Ray Tune
  tables, the reference numbers of the plan) is micro.
- **precision and recall are macro** (the legacy code asked for ``average='macro'``).
  Classes absent from both the predictions and the targets are left out of the mean, in
  torchmetrics 0.6 and 1.x alike.
- **the confusion matrix is normalised over the true labels** (``normalize="true"``: each
  row sums to 1); a class with no true sample gives a row of zeros, as in torchmetrics 0.6.

Inputs are integer class labels, already ``argmax``-ed. The functions work on the device of
their inputs (no device is named here).
"""

from __future__ import annotations

from typing import Final

import torch
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_confusion_matrix,
    multiclass_precision,
    multiclass_recall,
)

from astro_simclr.models import N_CLASSES

__all__ = ["METRIC_NAMES", "classification_metrics", "confusion_matrix"]

#: Keys of :func:`classification_metrics`, in logging order.
METRIC_NAMES: Final = ("accuracy", "precision", "recall")


def _check_labels(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    if y_pred.ndim != 1 or y_pred.shape != y_true.shape or y_pred.numel() == 0:
        msg = (
            "y_pred and y_true must be non-empty 1-D label tensors of the same shape, got "
            f"{tuple(y_pred.shape)} and {tuple(y_true.shape)}"
        )
        raise ValueError(msg)


def classification_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, *, num_classes: int = N_CLASSES
) -> dict[str, torch.Tensor]:
    """Micro accuracy, macro precision and macro recall of the labels ``y_pred``.

    Returns ``{"accuracy": ..., "precision": ..., "recall": ...}``, each a 0-d float tensor
    on the inputs' device.

    Raises:
        ValueError: if the inputs are not non-empty 1-D tensors of the same shape.
    """
    _check_labels(y_pred, y_true)
    return {
        "accuracy": multiclass_accuracy(y_pred, y_true, num_classes, average="micro"),
        "precision": multiclass_precision(y_pred, y_true, num_classes, average="macro"),
        "recall": multiclass_recall(y_pred, y_true, num_classes, average="macro"),
    }


def confusion_matrix(
    y_pred: torch.Tensor, y_true: torch.Tensor, *, num_classes: int = N_CLASSES
) -> torch.Tensor:
    """The (num_classes, num_classes) confusion matrix, rows normalised to sum to 1.

    Entry ``[i, j]`` is the fraction of the samples of true class ``i`` predicted as ``j``.

    Raises:
        ValueError: if the inputs are not non-empty 1-D tensors of the same shape.
    """
    _check_labels(y_pred, y_true)
    return multiclass_confusion_matrix(y_pred, y_true, num_classes, normalize="true")
