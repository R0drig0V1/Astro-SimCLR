"""Tests of the per-stamp predictions files and seed ensembling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _protocol_support import predictions

from astro_simclr.data.schema import DataError
from astro_simclr.protocol.predictions import (
    Predictions,
    check_paired,
    ensemble,
    load_predictions,
    predictions_file,
    save_predictions,
    seed_mean_correct,
)


def test_round_trip(tmp_path: Path) -> None:
    pred = predictions([0, 1, 2], [0, 2, 2], index=[5, 6, 9])
    path = tmp_path / predictions_file("test")
    assert path.name == "predictions_test.npz"
    save_predictions(pred, path)
    loaded = load_predictions(path)
    np.testing.assert_array_equal(loaded.probabilities, pred.probabilities)
    np.testing.assert_array_equal(loaded.index, [5, 6, 9])
    assert loaded.correct.tolist() == [True, False, True]
    assert loaded.predicted.dtype == np.int64


def test_bad_files(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not a predictions file"):
        load_predictions(tmp_path / "missing.npz")
    partial = tmp_path / "partial.npz"
    np.savez(partial, labels=np.zeros(3, dtype=np.int64))
    with pytest.raises(DataError, match="not a predictions file"):
        load_predictions(partial)
    shapes = tmp_path / "shapes.npz"
    np.savez(shapes, probabilities=np.zeros((3, 4)), labels=np.zeros(3), index=np.zeros(3))
    with pytest.raises(DataError, match="probabilities"):
        load_predictions(shapes)


def test_shape_checks() -> None:
    with pytest.raises(ValueError, match="N > 0"):
        Predictions(np.zeros((0, 5), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64))
    with pytest.raises(ValueError):
        Predictions(np.zeros((2, 5), np.float32), np.zeros(2, np.int64), np.zeros(3, np.int64))


def test_ensemble_averages_probabilities_per_stamp() -> None:
    labels = [0, 1]
    a = predictions(labels, [0, 0])
    b = predictions(labels, [1, 1])
    c = predictions(labels, [0, 1])
    mean = ensemble([a, b, c])
    np.testing.assert_allclose(mean.probabilities[0, :2], [(0.6 + 0.1 + 0.6) / 3, 0.8 / 3])
    assert mean.predicted.tolist() == [0, 1]
    assert seed_mean_correct([a, b, c]).tolist() == pytest.approx([2 / 3, 2 / 3])


def test_check_paired() -> None:
    a = predictions([0, 1], [0, 1])
    with pytest.raises(ValueError, match="no predictions"):
        check_paired([])
    with pytest.raises(ValueError, match="index"):
        check_paired([a, predictions([0, 1], [0, 1], index=[1, 2])])
    with pytest.raises(ValueError, match="labels"):
        ensemble([a, predictions([1, 1], [0, 1])])


def test_n_classes(tmp_path: Path) -> None:
    """Five classes by default (phase 2); the benchmarks pass six."""
    six = Predictions(np.full((2, 6), 1 / 6, np.float32), np.zeros(2, np.int64), np.arange(2), 6)
    path = tmp_path / "six.npz"
    save_predictions(six, path)
    assert load_predictions(path, n_classes=6).n_classes == 6
    with pytest.raises(DataError, match=r"probabilities \(N, 5\)"):
        load_predictions(path)
    assert ensemble([six, six]).n_classes == 6
    with pytest.raises(ValueError, match="6 classes, run 0 5"):
        check_paired([predictions([0, 0], [0, 0], index=[0, 1]), six])
    with pytest.raises(ValueError, match="N > 0"):
        Predictions(np.zeros((2, 1), np.float32), np.zeros(2, np.int64), np.arange(2), 1)
