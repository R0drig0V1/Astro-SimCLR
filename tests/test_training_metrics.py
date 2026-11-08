"""Tests of the shared metrics and confusion matrix (astro_simclr.training.metrics)."""

from __future__ import annotations

import pytest
import torch

from astro_simclr.training.metrics import classification_metrics, confusion_matrix

# Imbalanced and fixed: 10 samples of class 0, 2 of class 1, one each of classes 2 and 3,
# none of class 4. The predictions get every class-0 sample right except one, and only one
# sample of the minority classes.
Y_TRUE = torch.tensor([0] * 10 + [1, 1, 2, 3])
Y_PRED = torch.tensor([0] * 9 + [1] + [1, 0, 0, 0])


def test_accuracy_is_the_legacy_micro_value() -> None:
    # The legacy torchmetrics 0.6 Accuracy(num_classes=5) was micro: correct / total.
    legacy_micro = (Y_PRED == Y_TRUE).sum().item() / len(Y_TRUE)  # 10 / 14
    # Macro (MulticlassAccuracy's default) would average the per-class recalls of the
    # classes present: (0.9 + 0.5 + 0 + 0) / 4.
    macro = (0.9 + 0.5 + 0.0 + 0.0) / 4
    accuracy = classification_metrics(Y_PRED, Y_TRUE)["accuracy"].item()
    assert accuracy == pytest.approx(legacy_micro, abs=1e-7)
    assert accuracy != pytest.approx(macro, abs=1e-3)


def test_precision_and_recall_are_macro_over_the_classes_present() -> None:
    metrics = classification_metrics(Y_PRED, Y_TRUE)
    # Class 4 is absent from both predictions and targets, so it is left out of the means.
    # Precision: class 0 9/12, class 1 1/2, classes 2 and 3 have no prediction (0).
    assert metrics["precision"].item() == pytest.approx((9 / 12 + 1 / 2 + 0 + 0) / 4)
    # Recall: 9/10, 1/2, 0, 0.
    assert metrics["recall"].item() == pytest.approx((9 / 10 + 1 / 2 + 0 + 0) / 4)


def test_confusion_matrix_is_normalised_over_the_true_labels() -> None:
    cm = confusion_matrix(Y_PRED, Y_TRUE)
    assert cm.shape == (5, 5)
    expected = torch.zeros(5, 5)
    expected[0, 0], expected[0, 1] = 0.9, 0.1
    expected[1, 0], expected[1, 1] = 0.5, 0.5
    expected[2, 0] = 1.0
    expected[3, 0] = 1.0
    # Class 4 has no true sample: a row of zeros, not NaN (as in torchmetrics 0.6).
    torch.testing.assert_close(cm, expected)


@pytest.mark.parametrize(
    ("y_pred", "y_true"),
    [
        (torch.tensor([0, 1]), torch.tensor([0, 1, 2])),
        (torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5, dtype=torch.long)),
        (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
    ],
)
def test_invalid_label_tensors_are_rejected(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="1-D label tensors"):
        classification_metrics(y_pred, y_true)
    with pytest.raises(ValueError, match="1-D label tensors"):
        confusion_matrix(y_pred, y_true)
