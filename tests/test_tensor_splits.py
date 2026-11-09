"""Tests of ``training.tensor_splits``: the paper model's splits as arrays (task 2.2a).

What the plain engine's data path takes from it besides the crops of
``test_data_float_input.py``: :class:`TensorSplit`'s checks, the split-run identity that a
resume must keep (:func:`check_split_identity`) and the uint8 diagnostic crops of the
prepared layout (:func:`interim_uint8_crops`). The batches the engine serves from these
arrays are ``test_engine_tiers.py``'s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astro_simclr.config import ConfigError
from astro_simclr.data.schema import DataError
from astro_simclr.training.tensor_splits import (
    SPLIT_RUN_STATE,
    TensorSplit,
    check_split_identity,
    interim_uint8_crops,
)


def _split(labels: list[int], offset: float = 0.0) -> TensorSplit:
    n = len(labels)
    # Image i is filled with offset + i, so a batch's images identify its indices.
    images = np.broadcast_to(
        (offset + np.arange(n, dtype=np.float32))[:, None, None, None], (n, 3, 21, 21)
    ).copy()
    features = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 23))
    return TensorSplit(images=images, features=features, labels=np.array(labels, dtype=np.int64))


def test_tensor_split_validates() -> None:
    good = _split([0, 1])
    with pytest.raises(ValueError, match="images"):
        TensorSplit(good.images.astype(np.float64), good.features, good.labels)
    with pytest.raises(ValueError, match="features"):
        TensorSplit(good.images, good.features[:, :3], good.labels)
    with pytest.raises(ValueError, match="empty"):
        TensorSplit(good.images[:0], good.features[:0], good.labels[:0])
    with pytest.raises(ValueError, match="labels must be in"):
        TensorSplit(good.images, good.features, np.array([0, 5], dtype=np.int64))
    with pytest.raises(ValueError, match="fits"):
        TensorSplit(good.images, good.features, good.labels, fits=np.zeros((2, 4), np.float32))
    # An N-class dataset widens the label range (task 3.1b).
    wide = TensorSplit(good.images, good.features, np.array([0, 6], dtype=np.int64), n_classes=7)
    assert wide.n_classes == 7


def test_split_identity_must_be_the_runs() -> None:
    assert SPLIT_RUN_STATE == "split_run"
    fold2 = {"splits_sha256": "a" * 64, "mode": "fold", "fold": 2}
    check_split_identity(fold2, fold2)
    check_split_identity(None, None)
    with pytest.raises(ConfigError, match=r"from fold 3 of .* but this run is fold 2"):
        check_split_identity({**fold2, "fold": 3}, fold2)
    with pytest.raises(ConfigError, match="from a normal run"):
        check_split_identity(None, fold2)
    with pytest.raises(ConfigError, match="this run is a normal run"):
        check_split_identity(fold2, None)
    screen = {"splits_sha256": "a" * 64, "mode": "screen", "fold": None}
    with pytest.raises(ConfigError, match=r"from the screen of .* but this run is fold 2"):
        check_split_identity(screen, fold2)


def test_interim_crops_of_the_prepared_layout(classifier_dir: Path, pretraining_dir: Path) -> None:
    splits = interim_uint8_crops(classifier_dir)
    train = splits["Train"]
    assert train.images.shape[1:] == (3, 21, 21)
    assert train.images.dtype == np.float32
    assert train.images.min() >= 0.0
    assert train.images.max() <= 1.0
    assert train.features.shape[1] == 23
    with pytest.raises(DataError, match="unlabelled"):
        interim_uint8_crops(pretraining_dir)
