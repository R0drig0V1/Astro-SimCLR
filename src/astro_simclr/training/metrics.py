"""The confusion matrices and the macro-F1 of the classifier results, in plain torch.

``tasks.results`` writes these into every classifier's ``metrics.json``. They were
torchmetrics calls (``multiclass_confusion_matrix``, ``multiclass_f1_score``) until task
4.8 removed torchmetrics with Lightning; the functions below compute the same operations in
the same order and dtypes, so every recorded number is unchanged bit for bit (the golden
regression set, ``tests/test_engine_regression.py``, holds the evaluation entries of every
case):

- **the counts** (:func:`confusion_counts`): entry ``[i, j]`` counts the samples of true
  class ``i`` predicted as ``j``, an int64 ``bincount`` of ``true * num_classes + pred``;
- **the confusion matrix is normalised over the true labels** (:func:`confusion_matrix`;
  each row sums to 1): the float32 counts divided by their row sums; a class with no true
  sample gives a row of zeros, not NaN (as torchmetrics did, and torchmetrics 0.6 before
  it);
- **macro-F1** (:func:`macro_f1`) is the unweighted mean of the per-class F1, ``2 tp / (2 tp
  + fn + fp)`` in float32 (0 where the denominator is 0), over the classes present in the
  predictions or the targets: each present class's F1 is divided by their number, then
  summed (torchmetrics' ``_adjust_weights_safe_divide``, whose rounding this keeps).

Inputs are integer class labels, already ``argmax``-ed, in ``[0, num_classes)``. The
functions work on the device of their inputs (no device is named here).
"""

from __future__ import annotations

import torch

from astro_simclr.models import N_CLASSES

__all__ = ["confusion_counts", "confusion_matrix", "macro_f1"]


def _check_labels(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> None:
    if y_pred.ndim != 1 or y_pred.shape != y_true.shape or y_pred.numel() == 0:
        msg = (
            "y_pred and y_true must be non-empty 1-D label tensors of the same shape, got "
            f"{tuple(y_pred.shape)} and {tuple(y_true.shape)}"
        )
        raise ValueError(msg)
    for name, labels in (("y_pred", y_pred), ("y_true", y_true)):
        if labels.is_floating_point() or int(labels.min()) < 0 or int(labels.max()) >= num_classes:
            raise ValueError(f"{name} must hold integer labels in [0, {num_classes})")


def confusion_counts(
    y_pred: torch.Tensor, y_true: torch.Tensor, *, num_classes: int = N_CLASSES
) -> torch.Tensor:
    """The (num_classes, num_classes) confusion matrix of counts (int64, not normalised).

    Raises:
        ValueError: if the inputs are not non-empty 1-D tensors of the same shape, or hold
            a label outside ``[0, num_classes)``.
    """
    _check_labels(y_pred, y_true, num_classes)
    bins = y_true.to(torch.long) * num_classes + y_pred.to(torch.long)
    counts = torch.bincount(bins, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


def confusion_matrix(
    y_pred: torch.Tensor, y_true: torch.Tensor, *, num_classes: int = N_CLASSES
) -> torch.Tensor:
    """The (num_classes, num_classes) confusion matrix, rows normalised to sum to 1.

    Entry ``[i, j]`` is the fraction of the samples of true class ``i`` predicted as ``j``;
    the row of a class without true samples is zeros.

    Raises:
        ValueError: as :func:`confusion_counts`.
    """
    counts = confusion_counts(y_pred, y_true, num_classes=num_classes).float()
    matrix = counts / counts.sum(dim=-1, keepdim=True)
    return torch.nan_to_num(matrix, nan=0.0)


def macro_f1(y_pred: torch.Tensor, y_true: torch.Tensor, *, num_classes: int = N_CLASSES) -> float:
    """Macro-averaged F1 of the labels ``y_pred`` (classes absent from both are left out).

    Raises:
        ValueError: as :func:`confusion_counts`.
    """
    counts = confusion_counts(y_pred, y_true, num_classes=num_classes)
    tp = counts.diag()
    fp = counts.sum(0) - tp
    fn = counts.sum(1) - tp
    # The order and dtypes of torchmetrics' _fbeta_reduce with beta = 1 (module docstring).
    numerator = 2.0 * tp
    denominator = 2.0 * tp + 1.0 * fn + fp
    zero = torch.zeros((), dtype=numerator.dtype, device=numerator.device)
    score = torch.where(denominator != 0, numerator / denominator, zero)
    weights = torch.ones_like(score)
    weights[tp + fp + fn == 0] = 0.0
    total = weights.sum(-1, keepdim=True)
    weighted = torch.where(total != 0, weights * score / total, zero)
    return float(weighted.sum(-1))
