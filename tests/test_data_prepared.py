"""Tests of the prepared .npy layout (astro_simclr.data.prepared)."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from astro_simclr.data import DataError, SchemaError, load_pickle, load_prepared, prepare
from astro_simclr.data.files import sha256_file
from astro_simclr.data.prepared import FORMAT_VERSION, read_meta, verify_prepared
from astro_simclr.data.preprocessing import img_float2int
from astro_simclr.data.synthetic import CLASSIFIER_FILES, PRETRAINING_FILE, make_synthetic

FIXTURES = Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("raw")
    make_synthetic(out, n_per_split=10, seed=1)
    return out


@pytest.fixture
def prepared(raw_dir: Path, tmp_path: Path) -> Path:
    return prepare(raw_dir / CLASSIFIER_FILES[2], tmp_path)


def test_prepare_gives_the_legacy_uint8_arrays(tmp_path: Path) -> None:
    # Acceptance: float64 input, compared with the legacy img_float2int (fixture), not
    # only with the new loader.
    with np.load(FIXTURES / "prepare_legacy.npz") as npz:
        stamps, legacy = npz["prepare__images_float64"], npz["prepare__legacy_uint8"]
    assert stamps.dtype == np.float64
    # The fixture is discriminating: a float32 detour would not give the legacy bytes.
    detour = np.stack([img_float2int(s.astype(np.float32)) for s in stamps])
    assert (detour != legacy).sum() > 100

    # A pretraining-schema pickle: Train is a list of float64 stamps, like the 300k set.
    labelled = {
        "images": [stamps[0].astype(np.float32)],
        "labels": [np.int64(0)],
        "features": [np.zeros(26)],
    }
    source = tmp_path / "pre.pkl"
    content = {"Train": {"images": list(stamps)}, "Validation": labelled, "Test": labelled}
    source.write_bytes(pickle.dumps(content, protocol=4))
    dest = prepare(source, tmp_path / "out")
    images = np.load(dest / "Train" / "images.npy", allow_pickle=False)
    np.testing.assert_array_equal(images, legacy)


def test_layout_and_meta(prepared: Path, raw_dir: Path) -> None:
    source = raw_dir / CLASSIFIER_FILES[2]
    assert prepared.name == source.stem
    meta = read_meta(prepared)
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["kind"] == "classifier"
    assert meta["source"] == {
        "name": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    assert meta["schema"]["stamp_shape"] == [63, 63, 3]
    assert len(meta["schema"]["feature_names"]) == 26
    train = meta["splits"]["Train"]
    assert train["n"] == 10 and train["class_counts"] == [2, 2, 2, 2, 2]
    assert train["files"]["images.npy"]["dtype"] == "uint8"
    assert train["files"]["images.npy"]["shape"] == [10, 63, 63, 3]
    for split in ("Train", "Validation", "Test"):
        assert sorted(p.name for p in (prepared / split).iterdir()) == [
            "features.npy",
            "images.npy",
            "labels.npy",
        ]


def test_loads_memory_mapped_and_equal_to_the_pickle(prepared: Path, raw_dir: Path) -> None:
    from_pickle = load_pickle(raw_dir / CLASSIFIER_FILES[2])
    data = load_prepared(prepared)
    assert data.kind == "classifier"
    for split, content in data.splits.items():
        assert isinstance(content.images, np.memmap)
        assert not content.images.flags.writeable
        expected = from_pickle.splits[split]
        np.testing.assert_array_equal(content.images, expected.images)
        np.testing.assert_array_equal(np.asarray(content.labels), np.asarray(expected.labels))
        np.testing.assert_array_equal(np.asarray(content.features), np.asarray(expected.features))


def test_pretraining_layout_has_only_images_in_train(raw_dir: Path, tmp_path: Path) -> None:
    dest = prepare(raw_dir / PRETRAINING_FILE, tmp_path)
    assert [p.name for p in (dest / "Train").iterdir()] == ["images.npy"]
    data = load_prepared(dest)
    assert data.kind == "pretraining"
    assert data.splits["Train"].labels is None
    assert read_meta(dest)["splits"]["Train"]["class_counts"] is None


def test_prepare_is_deterministic(raw_dir: Path, tmp_path: Path) -> None:
    first = prepare(raw_dir / CLASSIFIER_FILES[1], tmp_path / "a")
    second = prepare(raw_dir / CLASSIFIER_FILES[1], tmp_path / "b")
    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    for file in files:
        assert (first / file).read_bytes() == (second / file).read_bytes(), file


def test_existing_destination_needs_overwrite(prepared: Path, raw_dir: Path) -> None:
    with pytest.raises(DataError, match=r"already exists; pass --overwrite"):
        prepare(raw_dir / CLASSIFIER_FILES[2], prepared.parent)
    (prepared / "Train" / "stale.txt").write_text("x")
    again = prepare(raw_dir / CLASSIFIER_FILES[2], prepared.parent, overwrite=True)
    assert not (again / "Train" / "stale.txt").exists()
    assert not any(p.name.endswith(".partial") for p in prepared.parent.iterdir())


def test_overwrite_never_deletes_a_foreign_directory(raw_dir: Path, tmp_path: Path) -> None:
    foreign = tmp_path / Path(CLASSIFIER_FILES[2]).stem
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep me")
    with pytest.raises(DataError, match=r"not a prepared layout; not replacing"):
        prepare(raw_dir / CLASSIFIER_FILES[2], tmp_path, overwrite=True)
    assert (foreign / "precious.txt").exists()


def test_stale_partial_directory_is_replaced(raw_dir: Path, tmp_path: Path) -> None:
    stem = Path(CLASSIFIER_FILES[2]).stem
    (tmp_path / f".{stem}.partial").mkdir()
    (tmp_path / f".{stem}.partial" / "junk").write_text("x")
    assert load_prepared(prepare(raw_dir / CLASSIFIER_FILES[2], tmp_path)).kind == "classifier"


def test_missing_source(tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"no such file"):
        prepare(tmp_path / "nope.pkl", tmp_path)


def _edit_meta(path: Path, **changes: object) -> None:
    meta = json.loads((path / "meta.json").read_text())
    meta.update(changes)
    (path / "meta.json").write_text(json.dumps(meta))


def test_meta_errors(prepared: Path, tmp_path: Path) -> None:
    with pytest.raises(DataError, match=r"not a prepared layout"):
        load_prepared(tmp_path / "nothing")
    _edit_meta(prepared, format_version=99)
    with pytest.raises(DataError, match=r"format version 99"):
        load_prepared(prepared)
    _edit_meta(prepared, format="other")
    with pytest.raises(DataError, match=r"not an astro-simclr-prepared file"):
        load_prepared(prepared)
    (prepared / "meta.json").write_text("{")
    with pytest.raises(DataError, match=r"invalid JSON"):
        load_prepared(prepared)


def test_meta_kind_and_splits_are_checked(prepared: Path) -> None:
    _edit_meta(prepared, kind="weird")
    with pytest.raises(SchemaError, match=r"unknown kind 'weird'"):
        load_prepared(prepared)
    _edit_meta(prepared, kind="classifier", splits={"Train": {}})
    with pytest.raises(SchemaError, match=r"must list the splits"):
        load_prepared(prepared)


def test_meta_must_list_the_schema_files(prepared: Path) -> None:
    meta = json.loads((prepared / "meta.json").read_text())
    del meta["splits"]["Test"]["files"]["features.npy"]
    (prepared / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SchemaError, match=r"Test: meta.json lists"):
        load_prepared(prepared)


def test_array_mismatches(prepared: Path) -> None:
    np.save(prepared / "Validation" / "labels.npy", np.zeros(10, dtype=np.int32))
    with pytest.raises(SchemaError, match=r"Validation/labels: labels.npy is int32"):
        load_prepared(prepared)


def test_array_disagreeing_with_meta(prepared: Path) -> None:
    np.save(prepared / "Test" / "labels.npy", np.zeros(4, dtype=np.int64))
    with pytest.raises(SchemaError, match=r"but meta.json says int64 \[10\]"):
        load_prepared(prepared)


def test_pickled_npy_is_refused(prepared: Path) -> None:
    np.save(prepared / "Test" / "labels.npy", np.array([None] * 10), allow_pickle=True)
    with pytest.raises(SchemaError, match=r"cannot load labels.npy"):
        load_prepared(prepared)


def _resave(path: Path, array: np.ndarray) -> None:
    np.save(path, array)
    meta_path = path.parents[1] / "meta.json"
    meta = json.loads(meta_path.read_text())
    info = meta["splits"][path.parent.name]["files"][path.name]
    info["shape"] = list(array.shape)
    meta_path.write_text(json.dumps(meta))


def test_content_checks(prepared: Path) -> None:
    _resave(prepared / "Train" / "labels.npy", np.full(10, 7, dtype=np.int64))
    with pytest.raises(SchemaError, match=r"Train/labels: labels outside 0..4"):
        load_prepared(prepared)
    _resave(prepared / "Train" / "labels.npy", np.zeros(9, dtype=np.int64))
    with pytest.raises(SchemaError, match=r"Train/labels: has 9 rows, but images has 10"):
        load_prepared(prepared)
    _resave(prepared / "Train" / "labels.npy", np.zeros(10, dtype=np.int64))
    _resave(prepared / "Train" / "features.npy", np.full((10, 26), np.nan))
    with pytest.raises(SchemaError, match=r"Train/features: has NaN"):
        load_prepared(prepared)


def test_empty_split_is_refused(prepared: Path) -> None:
    empty: dict[str, np.ndarray] = {
        "images": np.zeros((0, 63, 63, 3), np.uint8),
        "labels": np.zeros(0, np.int64),
        "features": np.zeros((0, 26)),
    }
    for key, array in empty.items():
        _resave(prepared / "Test" / f"{key}.npy", array)
    with pytest.raises(SchemaError, match=r"Test/images: is empty"):
        load_prepared(prepared)


def test_verify_detects_modified_and_missing_files(prepared: Path) -> None:
    verify_prepared(prepared)
    images = prepared / "Train" / "images.npy"
    blob = bytearray(images.read_bytes())
    blob[-1] ^= 0xFF
    images.write_bytes(bytes(blob))
    with pytest.raises(DataError, match=r"SHA-256 differs"):
        verify_prepared(prepared)
    images.unlink()
    with pytest.raises(DataError, match=r"missing"):
        verify_prepared(prepared)
