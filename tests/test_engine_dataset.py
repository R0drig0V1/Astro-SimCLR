"""Tests of ``engine.dataset``: one API over the classifier, unlabelled and raw-alerts kinds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _training_support import N_PER_SPLIT

from astro_simclr.data import prepare
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.raw_alerts import write_raw_alerts
from astro_simclr.data.schema import DataError
from astro_simclr.data.synthetic import (
    PRETRAINING_FILE,
    make_synthetic,
    synthetic_cd21_features,
    synthetic_raw_alerts,
)
from astro_simclr.engine.dataset import Dataset, prepared_split, raw_alerts_dataset

_NAMES = ("AGN", "SN", "VS")


def _arrays(n: int = 6) -> Dataset:
    images = np.arange(n * 4, dtype=np.float32).reshape(n, 2, 2)
    features = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    labels = np.arange(n, dtype=np.int64) % 3
    return Dataset(images=images, features=features, labels=labels, n_classes=3)


def test_classifier_split_reads_the_float32_stamps(classifier_f32_dir: Path) -> None:
    dataset = prepared_split(classifier_f32_dir, "Validation")
    stored = load_prepared(classifier_f32_dir).splits["Validation"]
    assert dataset.kind == "classifier"
    assert dataset.labelled
    assert isinstance(dataset.images, np.memmap)  # read in place, not copied
    assert dataset.images.shape == (len(stored), 63, 63, 3)
    assert dataset.features is not None and dataset.features.dtype == np.float32
    assert dataset.features.shape == (len(stored), 23)
    assert stored.features is not None
    kept = np.delete(np.asarray(stored.features), [13, 14, 15], axis=1).astype(np.float32)
    assert np.array_equal(dataset.features, kept)
    assert dataset.labels is not None and stored.labels is not None
    assert np.array_equal(dataset.labels, stored.labels)
    assert dataset.n_classes == 5


def test_prepared_split_refuses_uint8_layouts_and_unknown_splits(
    classifier_dir: Path, classifier_f32_dir: Path
) -> None:
    with pytest.raises(DataError, match="images_f32"):
        prepared_split(classifier_dir, "Train")
    with pytest.raises(DataError, match="split must be"):
        prepared_split(classifier_f32_dir, "Holdout")


def test_pretraining_train_is_unlabelled(tmp_path: Path) -> None:
    make_synthetic(tmp_path / "raw", n_per_split=N_PER_SPLIT, seed=0)
    layout = prepare(tmp_path / "raw" / PRETRAINING_FILE, tmp_path / "out", images="float32")
    dataset = prepared_split(layout, "Train")
    assert dataset.kind == "unlabelled"
    assert not dataset.labelled
    assert dataset.labels is None and dataset.features is None
    assert dataset.labels_of_rows() is None
    assert set(dataset.columns()) == {"images"}
    assert prepared_split(layout, "Validation").kind == "classifier"


def test_raw_alerts_dataset_and_a_candid_split(tmp_path: Path) -> None:
    cutouts, columns = synthetic_raw_alerts(_NAMES, n_per_class=4, seed=1)
    features = synthetic_cd21_features(len(cutouts), seed=1)
    layout = write_raw_alerts(
        tmp_path / "pool", cutouts, columns, _NAMES, {"name": "t"}, features=features
    )
    whole = raw_alerts_dataset(layout)
    assert whole.kind == "raw-alerts"
    assert whole.n_classes == 3
    assert len(whole) == 12
    assert whole.features is not None and whole.features.dtype == np.float32
    assert np.array_equal(whole.features, features.astype(np.float32), equal_nan=True)
    candids = columns["candid"][[7, 2, 9]]
    split = raw_alerts_dataset(layout, candids=candids.tolist())
    assert len(split) == 3
    assert split.rows is not None and split.rows.tolist() == [7, 2, 9]
    assert isinstance(split.images, np.memmap)  # the selection copies no cutout
    labels = split.labels_of_rows()
    assert labels is not None and labels.tolist() == columns["label"][[7, 2, 9]].tolist()
    held = split.materialised()
    assert np.array_equal(held.images, cutouts[[7, 2, 9]], equal_nan=True)


def test_raw_alerts_without_features(tmp_path: Path) -> None:
    cutouts, columns = synthetic_raw_alerts(_NAMES, n_per_class=2, seed=0)
    layout = write_raw_alerts(tmp_path / "pool", cutouts, columns, _NAMES, {})
    dataset = raw_alerts_dataset(layout)
    assert dataset.features is None
    assert not dataset.labelled


def test_select_composes_and_materialised_copies() -> None:
    dataset = _arrays()
    first = dataset.select([5, 1, 3, 0])
    second = first.select([2, 0])
    assert second.rows is not None and second.rows.tolist() == [3, 5]
    assert len(second) == 2
    assert second.stored_rows([1]).tolist() == [5]
    held = second.materialised()
    assert held.rows is None
    assert np.array_equal(held.images, dataset.images[[3, 5]])
    assert held.labels is not None and held.labels.tolist() == [0, 2]
    assert held.images.flags.writeable and held.images.flags.c_contiguous
    whole = dataset.materialised()
    assert np.array_equal(whole.images, dataset.images)
    assert not np.shares_memory(whole.images, dataset.images)
    with pytest.raises(DataError, match="range"):
        first.select([4])


def test_dataset_validates() -> None:
    good = _arrays()
    with pytest.raises(DataError, match="images must be float32"):
        Dataset(images=good.images.astype(np.float64))
    with pytest.raises(DataError, match="images must be float32"):
        Dataset(images=np.zeros(3, dtype=np.float32))
    with pytest.raises(DataError, match="features"):
        Dataset(images=good.images, features=np.zeros((5, 3), dtype=np.float32))
    with pytest.raises(DataError, match="fits"):
        Dataset(images=good.images, fits=np.zeros((6, 3), dtype=np.float64))
    with pytest.raises(DataError, match="go together"):
        Dataset(images=good.images, labels=good.labels)
    with pytest.raises(DataError, match="n_classes"):
        Dataset(images=good.images, labels=np.zeros(6, dtype=np.int64), n_classes=1)
    with pytest.raises(DataError, match="labels must be in"):
        Dataset(images=good.images, labels=good.labels, n_classes=2)
    with pytest.raises(DataError, match="1-D integer"):
        Dataset(images=good.images, rows=np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(DataError, match="rows must be in"):
        Dataset(images=good.images, rows=np.array([6]))
    with pytest.raises(DataError, match="empty"):
        Dataset(images=good.images, rows=np.array([], dtype=np.int64))
    narrow = Dataset(images=good.images, rows=np.array([1, 2], dtype=np.int32))
    assert narrow.rows is not None and narrow.rows.dtype == np.int64
