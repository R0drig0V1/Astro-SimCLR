"""Tests of the float32 stamps in the loader and the prepared layout, format v2 (task 2.1)."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from astro_simclr.cli import main
from astro_simclr.data import DataError, SchemaError, load_pickle, load_prepared, prepare
from astro_simclr.data.pickle_loader import read_pickle
from astro_simclr.data.prepared import (
    FORMAT_VERSION,
    IMAGE_DTYPES,
    READABLE_VERSIONS,
    read_meta,
    verify_prepared,
    write_prepared,
)
from astro_simclr.data.schema import SplitData, StampData
from astro_simclr.data.synthetic import CLASSIFIER_FILES, PRETRAINING_FILE, make_synthetic

SPLITS = ("Train", "Validation", "Test")


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("raw_f32")
    make_synthetic(out, n_per_split=10, seed=2)
    return out


@pytest.fixture
def both(raw_dir: Path, tmp_path: Path) -> tuple[Path, Path]:
    """The same pickle prepared uint8-only and with float32."""
    source = raw_dir / CLASSIFIER_FILES[2]
    return (
        prepare(source, tmp_path / "u8"),
        prepare(source, tmp_path / "f32", images="float32"),
    )


def test_versions() -> None:
    assert FORMAT_VERSION == 2
    assert READABLE_VERSIONS == (1, 2)
    assert IMAGE_DTYPES == ("uint8", "float32")


def test_loader_keeps_the_stored_float32_stamps(raw_dir: Path) -> None:
    source = raw_dir / CLASSIFIER_FILES[2]
    raw = read_pickle(source)
    assert isinstance(raw, dict)
    data = load_pickle(source, float32=True)
    for split in SPLITS:
        content = data.splits[split]
        assert content.images_f32 is not None and content.images_f32.dtype == np.float32
        np.testing.assert_array_equal(content.images_f32, np.stack(raw[split]["images"]))
    assert load_pickle(source).splits["Train"].images_f32 is None


def test_loader_rounds_float64_stamps_to_float32(raw_dir: Path) -> None:
    source = raw_dir / PRETRAINING_FILE  # Train holds float64 stamps
    raw = read_pickle(source)
    assert isinstance(raw, dict)
    train = load_pickle(source, float32=True).splits["Train"]
    assert train.labels is None and train.images_f32 is not None
    expected = np.stack(raw["Train"]["images"]).astype(np.float32)
    np.testing.assert_array_equal(train.images_f32, expected)


def test_float32_needs_values_inside_0_1(tmp_path: Path) -> None:
    # 1.001 rounds to 255, so the uint8 path accepts it; the float path would keep it.
    stamp = np.full((63, 63, 3), 0.5, dtype=np.float32)
    stamp[0, 0, 0] = 1.001
    split = {"images": [stamp], "labels": [np.int64(0)], "features": [np.zeros(26)]}
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps({name: split for name in SPLITS}, protocol=4))
    assert load_pickle(path).splits["Train"].images[0, 0, 0, 0] == 255
    with pytest.raises(SchemaError, match=r"Train/images: stamp 0: values outside \[0, 1\]"):
        load_pickle(path, float32=True)


def test_float32_layout(both: tuple[Path, Path], raw_dir: Path) -> None:
    u8, f32 = both
    meta_u8, meta_f32 = read_meta(u8), read_meta(f32)
    assert meta_u8["format_version"] == meta_f32["format_version"] == 2
    assert meta_u8["images"] == ["uint8"]
    assert meta_f32["images"] == ["uint8", "float32"]
    for split in SPLITS:
        assert sorted(p.name for p in (f32 / split).iterdir()) == [
            "features.npy",
            "images.npy",
            "images_f32.npy",
            "labels.npy",
        ]
        info = meta_f32["splits"][split]["files"]["images_f32.npy"]
        assert info["dtype"] == "float32" and info["shape"] == [10, 63, 63, 3]
        # The uint8 files are the same bytes either way (so old manifests still match).
        for name, entry in meta_u8["splits"][split]["files"].items():
            assert meta_f32["splits"][split]["files"][name] == entry
    data = load_prepared(f32)
    from_pickle = load_pickle(raw_dir / CLASSIFIER_FILES[2], float32=True)
    for split in SPLITS:
        content = data.splits[split]
        assert isinstance(content.images_f32, np.memmap)
        expected = from_pickle.splits[split].images_f32
        assert expected is not None
        np.testing.assert_array_equal(content.images_f32, expected)
    assert load_prepared(u8).splits["Train"].images_f32 is None
    verify_prepared(f32)


def test_pretraining_layout_with_float32(raw_dir: Path, tmp_path: Path) -> None:
    dest = prepare(raw_dir / PRETRAINING_FILE, tmp_path, images="float32")
    assert sorted(p.name for p in (dest / "Train").iterdir()) == ["images.npy", "images_f32.npy"]
    train = load_prepared(dest).splits["Train"]
    assert train.labels is None and train.images_f32 is not None


def test_version_1_layouts_are_still_read(both: tuple[Path, Path]) -> None:
    u8, f32 = both
    meta = json.loads((u8 / "meta.json").read_text())
    meta["format_version"] = 1
    del meta["images"]
    (u8 / "meta.json").write_text(json.dumps(meta))
    assert read_meta(u8)["images"] == ["uint8"]
    data = load_prepared(u8)
    assert data.splits["Test"].images_f32 is None
    verify_prepared(u8)

    # A version-1 meta.json never lists images_f32.npy.
    meta = json.loads((f32 / "meta.json").read_text())
    meta["format_version"] = 1
    (f32 / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(SchemaError, match=r"lists .*images_f32.npy"):
        load_prepared(f32)


@pytest.mark.parametrize("images", [None, "uint8", ["float32"], ["uint8", "float16"]])
def test_meta_images_is_checked(images: object, both: tuple[Path, Path]) -> None:
    u8, _ = both
    meta = json.loads((u8 / "meta.json").read_text())
    meta["images"] = images
    (u8 / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(DataError, match=r"images must list uint8"):
        read_meta(u8)


def test_unknown_version_names_the_readable_ones(both: tuple[Path, Path]) -> None:
    u8, _ = both
    meta = json.loads((u8 / "meta.json").read_text())
    meta["format_version"] = 3
    (u8 / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(DataError, match=r"format version 3; .* reads versions 1, 2"):
        load_prepared(u8)


def test_float32_files_are_checked(both: tuple[Path, Path]) -> None:
    _, f32 = both
    file = f32 / "Validation" / "images_f32.npy"
    array = np.load(file)
    array[0, 0, 0, 0] = 0.25 if array[0, 0, 0, 0] != 0.25 else 0.75
    np.save(file, array)
    with pytest.raises(DataError, match=r"images_f32.npy: SHA-256 differs"):
        verify_prepared(f32)
    np.save(file, array[:4])
    with pytest.raises(SchemaError, match=r"meta.json says float32 \[10, 63, 63, 3\]"):
        load_prepared(f32)
    np.save(file, array.astype(np.float64))
    with pytest.raises(SchemaError, match=r"images_f32.npy is float64"):
        load_prepared(f32)


def test_images_f32_rows_must_match(tmp_path: Path) -> None:
    images = np.zeros((2, 63, 63, 3), dtype=np.uint8)
    labels, features = np.zeros(2, dtype=np.int64), np.zeros((2, 26))
    short = SplitData(images, labels, features, np.zeros((1, 63, 63, 3), dtype=np.float32))
    full = SplitData(images, labels, features, np.zeros((2, 63, 63, 3), dtype=np.float32))
    source = {"name": "x.pkl", "size_bytes": 0, "sha256": "0" * 64}
    dest = write_prepared(
        StampData("classifier", {"Train": short, "Validation": full, "Test": full}),
        tmp_path / "x",
        source,
    )
    with pytest.raises(SchemaError, match=r"Train/images_f32: has 1 rows, but images has 2"):
        load_prepared(dest)
    plain = SplitData(images, labels, features)
    with pytest.raises(ValueError, match=r"every split or in none"):
        write_prepared(
            StampData("classifier", {"Train": full, "Validation": plain, "Test": plain}),
            tmp_path / "y",
            source,
        )


def test_prepare_rejects_an_unknown_image_dtype(raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"images must be one of"):
        prepare(raw_dir / CLASSIFIER_FILES[2], tmp_path, images="float16")  # type: ignore[arg-type]


def test_cli_prepare_data_images_float32(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = raw_dir / CLASSIFIER_FILES[1]
    assert main(["prepare-data", str(source), "--out", str(tmp_path), "--images", "float32"]) == 0
    dest = tmp_path / source.stem
    assert (dest / "Test" / "images_f32.npy").is_file()
    capsys.readouterr()
    assert main(["inspect-data", str(dest)]) == 0
    assert "images_f32.npy  float32 (10, 63, 63, 3)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["prepare-data", str(source), "--out", str(tmp_path), "--images", "float16"])
