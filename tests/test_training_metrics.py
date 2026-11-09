"""Tests of the confusion matrices and macro-F1 in plain torch (astro_simclr.training.metrics).

The counts, the confusion matrix normalised over the true labels and the macro-F1 over the
classes present, on a fixed imbalanced example whose values are written out by hand, and
the input checks (shape, emptiness and the label range ``[0, num_classes)``).
"""

from __future__ import annotations

import pytest
import torch

from astro_simclr.training.metrics import confusion_counts, confusion_matrix, macro_f1

# Imbalanced and fixed: 10 samples of class 0, 2 of class 1, one each of classes 2 and 3,
# none of class 4. The predictions get every class-0 sample right except one, and only one
# sample of the minority classes.
Y_TRUE = torch.tensor([0] * 10 + [1, 1, 2, 3])
Y_PRED = torch.tensor([0] * 9 + [1] + [1, 0, 0, 0])


def test_confusion_matrix_is_normalised_over_the_true_labels() -> None:
    cm = confusion_matrix(Y_PRED, Y_TRUE)
    assert cm.shape == (5, 5)
    assert cm.dtype == torch.float32
    expected = torch.zeros(5, 5)
    expected[0, 0], expected[0, 1] = 0.9, 0.1
    expected[1, 0], expected[1, 1] = 0.5, 0.5
    expected[2, 0] = 1.0
    expected[3, 0] = 1.0
    # Class 4 has no true sample: a row of zeros, not NaN (as in torchmetrics 0.6).
    torch.testing.assert_close(cm, expected)
    torch.testing.assert_close(cm.sum(dim=1), torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    ("y_pred", "y_true"),
    [
        (torch.tensor([0, 1]), torch.tensor([0, 1, 2])),
        (torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5, dtype=torch.long)),
        (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
    ],
)
def test_invalid_label_tensors_are_rejected(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    for function in (confusion_counts, confusion_matrix, macro_f1):
        with pytest.raises(ValueError, match="1-D label tensors"):
            function(y_pred, y_true)


@pytest.mark.parametrize(
    ("y_pred", "y_true", "name"),
    [
        (torch.tensor([0, 5]), torch.tensor([0, 1]), "y_pred"),  # == num_classes
        (torch.tensor([0, 1]), torch.tensor([-1, 1]), "y_true"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0, 1]), "y_pred"),  # not integer labels
        (torch.tensor([0, 1]), torch.tensor([0, 9]), "y_true"),
    ],
)
def test_labels_outside_the_classes_are_rejected(
    y_pred: torch.Tensor, y_true: torch.Tensor, name: str
) -> None:
    # A bincount of true * num_classes + pred would silently file such a label under
    # another cell (or fail far from the cause), so it is refused first.
    for function in (confusion_counts, confusion_matrix, macro_f1):
        with pytest.raises(ValueError, match=rf"{name} must hold integer labels in \[0, 5\)"):
            function(y_pred, y_true)


def test_the_label_range_follows_num_classes() -> None:
    y_pred, y_true = torch.tensor([0, 2, 1]), torch.tensor([0, 1, 2])
    counts = confusion_counts(y_pred, y_true, num_classes=3)
    assert counts.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    with pytest.raises(ValueError, match=r"y_true must hold integer labels in \[0, 2\)"):
        confusion_counts(torch.tensor([0, 1, 1]), y_true, num_classes=2)
    # The labels of the five classes are fine with the default num_classes.
    assert confusion_counts(y_pred, y_true).shape == (5, 5)


def test_macro_f1_and_counts() -> None:
    counts = confusion_counts(Y_PRED, Y_TRUE)
    assert counts.dtype == torch.int64
    assert counts.sum() == len(Y_TRUE)
    assert counts.tolist() == [
        [9, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    # Per-class F1 over the classes present (4 is in neither): class 0: P 9/12, R 9/10;
    # class 1: P 1/2, R 1/2; classes 2 and 3: 0.
    f1_0 = 2 * (9 / 12) * (9 / 10) / (9 / 12 + 9 / 10)
    assert macro_f1(Y_PRED, Y_TRUE) == pytest.approx((f1_0 + 0.5) / 4)
    with pytest.raises(ValueError):
        macro_f1(Y_PRED[:2], Y_TRUE)


def test_macro_f1_counts_a_class_present_in_the_predictions_only() -> None:
    # Class 2 is never true but is predicted once: it is present, with F1 0, so it lowers
    # the mean (class 0: 2 tp, 1 fn -> 0.8; class 1: 1 tp -> 1; class 2: 0).
    y_true = torch.tensor([0, 0, 0, 1])
    y_pred = torch.tensor([0, 0, 2, 1])
    assert macro_f1(y_pred, y_true) == pytest.approx((0.8 + 1.0 + 0.0) / 3)
    # Perfect predictions of a subset of the classes: the absent classes are left out.
    assert macro_f1(y_true, y_true) == 1.0
    assert isinstance(macro_f1(y_true, y_true), float)
