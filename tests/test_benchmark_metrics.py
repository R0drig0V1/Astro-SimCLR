"""Tests of the phase-3 benchmark metrics: hand-computed macro-F1 and the zero-division rule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astro_simclr.protocol.benchmark_metrics import (
    RJ23_CLASS_COUNTS,
    benchmark_metrics,
    confusion_counts,
    confusion_percent,
    macro_f1,
    macro_f1_from_counts,
    reweighted_counts,
    rj23_reference,
)
from astro_simclr.training import metrics as training_metrics

NAMES3 = ("A", "B", "C")


def test_hand_computed_three_classes() -> None:
    # A: TP 2, FN 1, FP 0 -> P 1, R 2/3, F1 0.8
    # B: TP 1, FN 1, FP 1 -> P 1/2, R 1/2, F1 1/2
    # C: TP 1, FN 0, FP 1 -> P 1/2, R 1, F1 2/3
    labels = [0, 0, 0, 1, 1, 2]
    predicted = [0, 0, 1, 1, 2, 2]
    result = benchmark_metrics(labels, predicted, NAMES3)
    assert result.n == 6
    assert result.accuracy == pytest.approx(4 / 6)
    assert result.macro_precision == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert result.macro_recall == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    assert result.macro_f1 == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert [c.f1 for c in result.per_class] == pytest.approx([0.8, 0.5, 2 / 3])
    assert [c.support for c in result.per_class] == [3, 2, 1]
    assert [c.predicted for c in result.per_class] == [2, 2, 2]
    assert result.counts.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    expected = np.array([[200 / 3, 100 / 3, 0], [0, 50, 50], [0, 0, 100]])
    np.testing.assert_allclose(result.percent, expected)
    assert macro_f1(labels, predicted, 3) == pytest.approx(result.macro_f1)


def test_class_never_predicted_counts_as_zero() -> None:
    # C has a true sample but is never predicted: P = 0 (0/0 -> 0), R = 0, F1 = 0, counted.
    result = benchmark_metrics([0, 1, 2], [0, 1, 1], NAMES3)
    c = result.per_class[2]
    assert (c.precision, c.recall, c.f1) == (0.0, 0.0, 0.0)
    assert result.macro_f1 == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert result.macro_precision == pytest.approx((1 + 0.5 + 0) / 3)


def test_class_predicted_without_true_samples_counts_as_zero() -> None:
    result = benchmark_metrics([0, 0], [0, 1], ("A", "B"))
    b = result.per_class[1]
    assert (b.support, b.predicted, b.precision, b.recall, b.f1) == (0, 1, 0.0, 0.0, 0.0)
    assert result.macro_f1 == pytest.approx((2 / 3 + 0) / 2)
    assert result.percent[1].tolist() == [0.0, 0.0]  # a row without true samples


def test_class_absent_from_both_is_left_out() -> None:
    four = benchmark_metrics([0, 1, 2], [0, 1, 1], (*NAMES3, "D"))
    three = benchmark_metrics([0, 1, 2], [0, 1, 1], NAMES3)
    d = four.per_class[3]
    assert (d.precision, d.recall, d.f1) == (None, None, None)
    assert four.macro_f1 == pytest.approx(three.macro_f1)
    assert four.macro_recall == pytest.approx(three.macro_recall)
    assert four.as_dict()["per_class"]["D"]["f1"] is None


def test_perfect_and_all_wrong() -> None:
    assert macro_f1([0, 1, 2, 2], [0, 1, 2, 2], 3) == 1.0
    assert macro_f1([0, 1, 2], [1, 2, 0], 3) == 0.0
    assert float(macro_f1_from_counts(np.zeros((3, 3)))) == 0.0


def test_six_classes_agree_with_the_training_metrics() -> None:
    """Same rule as the phase-2 macro-F1 (torchmetrics), on random 6-class predictions."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(5, 60))
        # Use few classes in some trials, so that some are absent from both.
        high = int(rng.integers(2, 7))
        labels = rng.integers(0, high, size=n)
        predicted = rng.integers(0, high, size=n)
        expected = training_metrics.macro_f1(
            torch.from_numpy(predicted), torch.from_numpy(labels), num_classes=6
        )
        assert macro_f1(labels, predicted, 6) == pytest.approx(expected, abs=1e-6), trial


def test_stacked_counts() -> None:
    a = confusion_counts([0, 0, 1], [0, 1, 1], 2)
    b = confusion_counts([0, 1, 1], [0, 1, 1], 2)
    stacked = macro_f1_from_counts(np.stack([[a, b], [b, a]]))
    assert stacked.shape == (2, 2)
    assert stacked[0, 1] == 1.0
    assert stacked[0, 0] == pytest.approx(macro_f1([0, 0, 1], [0, 1, 1], 2))


def test_as_dict_is_json_ready() -> None:
    plain = benchmark_metrics([0, 1], [0, 0], ("A", "B")).as_dict()
    assert plain["confusion_counts"] == [[1, 0], [1, 0]]
    assert plain["confusion_percent"] == [[100.0, 0.0], [100.0, 0.0]]
    assert plain["per_class"]["B"] == {
        "support": 1,
        "predicted": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_input_checks() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        confusion_counts([], [], 3)
    with pytest.raises(ValueError, match="non-empty"):
        confusion_counts([0, 1], [0], 3)
    with pytest.raises(ValueError, match="integer"):
        confusion_counts([0.0, 1.0], [0, 1], 3)
    with pytest.raises(ValueError, match="at least 2"):
        confusion_counts([0], [0], 1)
    with pytest.raises(ValueError, match="labels must be in range"):
        confusion_counts([0, 3], [0, 1], 3)
    with pytest.raises(ValueError, match="predictions must be in range"):
        confusion_counts([0, 1], [0, -1], 3)
    with pytest.raises(ValueError, match="square"):
        confusion_percent(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="C, C"):
        macro_f1_from_counts(np.zeros(3))


def test_prior_reweighting_by_hand() -> None:
    # Class A: 8 true, 2 predicted as B; class B: 2 true, both right.
    labels = [0] * 8 + [1] * 2
    predicted = [0] * 6 + [1] * 2 + [1] * 2
    # Rescale to 100 A and 100 B: A row [75, 25], B row [0, 100].
    rescaled = reweighted_counts(confusion_counts(labels, predicted, 2), [100, 100])
    np.testing.assert_allclose(rescaled, np.array([[75.0, 25.0], [0.0, 100.0]]))
    result = benchmark_metrics(labels, predicted, ("A", "B"), reference_counts=[100, 100])
    # Precision of B rises from 2/4 to 100/125; recall does not change.
    f1_a, f1_b = 2 * 75 / (150 + 25), 2 * 100 / (200 + 25)
    assert result.reweighted_macro_f1 == pytest.approx((f1_a + f1_b) / 2)
    assert result.reweighted_macro_precision == pytest.approx((1 + 0.8) / 2)
    assert result.macro_f1 == pytest.approx((2 * 6 / 14 + 2 * 2 / 6) / 2)
    # Reweighting to the evaluated set's own counts changes nothing.
    same = benchmark_metrics(labels, predicted, ("A", "B"), reference_counts=[8, 2])
    assert same.reweighted_macro_f1 == pytest.approx(same.macro_f1)
    assert benchmark_metrics(labels, predicted, ("A", "B")).reweighted_macro_f1 is None


def test_reweighting_checks_and_empty_rows() -> None:
    counts = confusion_counts([0, 0], [0, 1], 2)  # B has no true sample: its row stays 0
    np.testing.assert_allclose(reweighted_counts(counts, [10, 10]), [[5.0, 5.0], [0.0, 0.0]])
    assert len(RJ23_CLASS_COUNTS) == 6
    assert rj23_reference(("Bogus", "SN")) == (15950.0, 3615.0)
    assert rj23_reference(("Bogus", "Galaxy")) is None
    # Only the proportions matter: scaling the reference changes nothing.
    labels, predicted = [0, 0, 0, 1, 1, 2], [0, 1, 1, 1, 2, 2]
    a = benchmark_metrics(labels, predicted, ("A", "B", "C"), reference_counts=[5, 3, 2])
    b = benchmark_metrics(labels, predicted, ("A", "B", "C"), reference_counts=[50, 30, 20])
    assert a.reweighted_macro_f1 == pytest.approx(b.reweighted_macro_f1)
    with pytest.raises(ValueError, match="square"):
        reweighted_counts(np.zeros((2, 3)), [1, 1])
    with pytest.raises(ValueError, match="2 finite"):
        reweighted_counts(counts, [1, 1, 1])
    with pytest.raises(ValueError, match="2 finite"):
        reweighted_counts(counts, [1, -1])
    with pytest.raises(ValueError, match="all be 0"):
        reweighted_counts(counts, [0, 0])
