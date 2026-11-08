"""Tests of the synthetic data generator (astro_simclr.data.synthetic)."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from astro_simclr.data import load_pickle
from astro_simclr.data.synthetic import (
    CLASSIFIER_FILES,
    PRETRAINING_FILE,
    class_pattern,
    make_synthetic,
)

ALL_FILES = [*CLASSIFIER_FILES, PRETRAINING_FILE]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    make_synthetic(out, n_per_split=40, seed=0)
    return out


def _raw(path: Path) -> dict[str, dict[str, list[object]]]:
    # The test's own freshly written file, so unpickling it is safe.
    raw: dict[str, dict[str, list[object]]] = pickle.loads(path.read_bytes())  # noqa: S301
    return raw


def test_writes_the_four_files(tmp_path: Path) -> None:
    paths = make_synthetic(tmp_path / "new" / "dir", n_per_split=5, seed=0)
    assert [p.name for p in paths] == ALL_FILES
    assert all(p.is_file() for p in paths)
    assert not list((tmp_path / "new" / "dir").glob("*.tmp"))


def test_same_seed_same_bytes_other_seed_other_bytes(tmp_path: Path) -> None:
    first = make_synthetic(tmp_path / "a", n_per_split=7, seed=11)
    second = make_synthetic(tmp_path / "b", n_per_split=7, seed=11)
    other = make_synthetic(tmp_path / "c", n_per_split=7, seed=12)
    for a, b, c in zip(first, second, other, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name
        assert a.read_bytes() != c.read_bytes(), a.name


def test_classifier_files_copy_the_real_schema(synthetic: Path) -> None:
    for name in CLASSIFIER_FILES:
        raw = _raw(synthetic / name)
        assert list(raw) == ["Train", "Validation", "Test"]
        for content in raw.values():
            assert list(content) == ["images", "features", "labels"]
            image = content["images"][0]
            assert isinstance(image, np.ndarray)
            assert image.dtype == np.float32 and image.shape == (63, 63, 3)
            assert image.min() >= 0 and image.max() <= 1
            row = content["features"][0]
            assert isinstance(row, np.ndarray) and row.dtype == object and row.shape == (26,)
            assert {type(v) for v in row} == {np.float64, float}
            assert all(type(label) is np.int64 for label in content["labels"])


def test_pretraining_file_copies_the_real_schema(synthetic: Path) -> None:
    raw = _raw(synthetic / PRETRAINING_FILE)
    assert list(raw["Train"]) == ["images"]
    image = raw["Train"]["images"][0]
    assert isinstance(image, np.ndarray) and image.dtype == np.float64
    full = _raw(synthetic / CLASSIFIER_FILES[0])
    assert pickle.dumps(raw["Validation"]) == pickle.dumps(full["Validation"])


def test_label_fraction_files_share_validation_and_test(synthetic: Path) -> None:
    full = load_pickle(synthetic / CLASSIFIER_FILES[0])
    for name in CLASSIFIER_FILES[1:]:
        data = load_pickle(synthetic / name)
        for split in ("Validation", "Test"):
            np.testing.assert_array_equal(data.splits[split].images, full.splits[split].images)
        train = data.splits["Train"]
        assert train.class_counts() == [8, 8, 8, 8, 8]
        # Train is drawn from the full Train split, with duplicates.
        full_rows = {row.tobytes() for row in full.splits["Train"].images}
        assert {row.tobytes() for row in train.images} <= full_rows


def test_every_split_is_class_balanced(synthetic: Path) -> None:
    data = load_pickle(synthetic / CLASSIFIER_FILES[0])
    for split in data.splits.values():
        assert split.class_counts() == [8, 8, 8, 8, 8]


def test_class_pattern_is_strong_in_the_central_crop(synthetic: Path) -> None:
    # A nearest-centroid classifier on the central 27x27 crop (the stamp encoders' input),
    # fitted on Train, must separate the classes on Test almost perfectly.
    data = load_pickle(synthetic / CLASSIFIER_FILES[0])

    def crop(images: np.ndarray) -> np.ndarray:
        return images[:, 18:45, 18:45].reshape(len(images), -1).astype(np.float64)

    train, test = data.splits["Train"], data.splits["Test"]
    assert train.labels is not None and test.labels is not None
    centroids = np.stack([crop(train.images)[train.labels == c].mean(0) for c in range(5)])
    distances = ((crop(test.images)[:, None] - centroids[None]) ** 2).sum(-1)
    accuracy = float((distances.argmin(1) == test.labels).mean())
    assert accuracy >= 0.95


def test_patterns_differ_between_classes() -> None:
    patterns = [class_pattern(label) for label in range(5)]
    for a in range(5):
        for b in range(a + 1, 5):
            assert not np.allclose(patterns[a], patterns[b]), (a, b)
    with pytest.raises(KeyError):
        class_pattern(5)


def test_too_few_stamps_per_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"n_per_split must be >= 5"):
        make_synthetic(tmp_path, n_per_split=4, seed=0)
