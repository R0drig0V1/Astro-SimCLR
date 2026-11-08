"""Tests of the lenient pickle loader (astro_simclr.data.pickle_loader): schemas and errors."""

import datetime
import importlib
import io
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from astro_simclr.data import DataError, SchemaError, TruncatedFileError, load_pickle
from astro_simclr.data.pickle_loader import _NumpyOnlyUnpickler, validate
from astro_simclr.data.preprocessing import img_float2int

N = 6


def _split(n: int = N, *, as_arrays: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(n)
    images = [rng.uniform(0, 1, (63, 63, 3)).astype(np.float32) for _ in range(n)]
    features = [np.array(rng.normal(size=26), dtype=object) for _ in range(n)]
    labels = [np.int64(i % 5) for i in range(n)]
    if as_arrays:
        return {
            "images": np.stack(images),
            "labels": np.array(labels),
            "features": np.array(features, dtype=np.float64),
        }
    return {"images": images, "features": features, "labels": labels}


def _classifier(**kw: Any) -> dict[str, Any]:
    return {split: _split(**kw) for split in ("Train", "Validation", "Test")}


def _write(path: Path, obj: object) -> Path:
    path.write_bytes(pickle.dumps(obj, protocol=4))
    return path


def test_loads_lists_like_the_real_files(tmp_path: Path) -> None:
    raw = _classifier()
    data = load_pickle(_write(tmp_path / "c.pkl", raw))
    assert data.kind == "classifier"
    assert list(data.splits) == ["Train", "Validation", "Test"]
    train = data.splits["Train"]
    assert train.images.dtype == np.uint8 and train.images.shape == (N, 63, 63, 3)
    np.testing.assert_array_equal(train.images[2], img_float2int(raw["Train"]["images"][2]))
    assert train.labels is not None and train.labels.dtype == np.int64
    assert train.features is not None and train.features.dtype == np.float64
    assert train.features.shape == (N, 26)
    assert train.class_counts() == [2, 1, 1, 1, 1]


def test_loads_ndarrays_and_plain_ints(tmp_path: Path) -> None:
    raw = _classifier(as_arrays=True)
    raw["Test"]["labels"] = [int(x) for x in raw["Test"]["labels"]]
    data = load_pickle(_write(tmp_path / "c.pkl", raw))
    assert data.splits["Test"].labels is not None
    assert data.splits["Test"].labels.tolist() == [0, 1, 2, 3, 4, 0]


def test_pretraining_kind_keeps_train_unlabelled(tmp_path: Path) -> None:
    raw = _classifier()
    raw["Train"] = {"images": [np.full((63, 63, 3), 0.5) for _ in range(3)]}  # float64
    data = load_pickle(_write(tmp_path / "p.pkl", raw))
    assert data.kind == "pretraining"
    train = data.splits["Train"]
    assert len(train) == 3 and train.labels is None and train.features is None
    assert train.class_counts() is None
    assert data.splits["Validation"].labels is not None


def test_converts_in_the_stored_dtype(tmp_path: Path) -> None:
    # 0.5/255 + tiny: float64 rounds up, a float32 detour rounds the other way.
    value = (0.5 + 2e-9) / 255
    assert img_float2int(np.array(value)) != img_float2int(np.array(value, np.float32))
    raw = _classifier()
    raw["Train"] = {"images": [np.full((63, 63, 3), value)]}
    data = load_pickle(_write(tmp_path / "p.pkl", raw))
    assert data.splits["Train"].images[0, 0, 0, 0] == img_float2int(np.array(value))


Mutation = Callable[[dict[str, Any]], object]


def _set(split: str, key: str, value: object) -> Mutation:
    def mutate(raw: dict[str, Any]) -> object:
        raw[split][key] = value
        return raw

    return mutate


def _stamp(index: int, value: object) -> Mutation:
    def mutate(raw: dict[str, Any]) -> object:
        raw["Train"]["images"][index] = value
        return raw

    return mutate


def _drop(split: str, key: str | None = None) -> Mutation:
    def mutate(raw: dict[str, Any]) -> object:
        if key is None:
            del raw[split]
        else:
            del raw[split][key]
        return raw

    return mutate


VIOLATIONS: dict[str, tuple[Mutation, str | None, str | None, str]] = {
    "top level not a dict": (lambda raw: [raw], None, None, "top level must be a dict"),
    "missing split": (_drop("Test"), None, None, r"missing splits \['Test'\]"),
    "unexpected split": (
        lambda raw: {**raw, "Extra": {}},
        None,
        None,
        r"unexpected splits \['Extra'\]",
    ),
    "split not a dict": (
        lambda raw: {**raw, "Validation": []},
        "Validation",
        None,
        "split must be a dict",
    ),
    "missing key": (_drop("Validation", "features"), "Validation", None, "missing keys"),
    "unexpected key": (_set("Test", "ids", [1]), "Test", None, r"unexpected keys \['ids'\]"),
    "images not a sequence": (_set("Train", "images", 3), "Train", "images", "list or an"),
    "images empty": (_set("Train", "images", []), "Train", "images", "is empty"),
    "stamp wrong shape": (
        _stamp(4, np.zeros((60, 63, 3))),
        "Train",
        "images",
        r"stamp 4 has shape \(60, 63, 3\)",
    ),
    "stamp uint8": (
        _stamp(1, np.zeros((63, 63, 3), np.uint8)),
        "Train",
        "images",
        "stamp 1: .*floating point",
    ),
    "stamp nan": (_stamp(0, np.full((63, 63, 3), np.nan)), "Train", "images", "NaN or inf"),
    "stamp out of range": (_stamp(2, np.full((63, 63, 3), 2.0)), "Train", "images", r"\[0, 1\]"),
    "labels float": (_set("Train", "labels", [0.0] * N), "Train", "labels", "integers"),
    "labels bool": (_set("Train", "labels", [True] * N), "Train", "labels", "integers"),
    "labels too short": (_set("Train", "labels", [0] * (N - 1)), "Train", "labels", "6 integers"),
    "labels 2-d": (_set("Train", "labels", [[0]] * N), "Train", "labels", "6 integers"),
    "label out of range": (
        _set("Validation", "labels", [0, 1, 5, 0, 0, 0]),
        "Validation",
        "labels",
        "label 5 at 2",
    ),
    "negative label": (_set("Train", "labels", [-1] * N), "Train", "labels", "label -1"),
    "features wrong width": (
        _set("Test", "features", np.zeros((N, 23))),
        "Test",
        "features",
        r"\(6, 23\), expected \(6, 26\)",
    ),
    "features ragged": (
        _set("Test", "features", [np.zeros(26)] * (N - 1) + [np.zeros(3)]),
        "Test",
        "features",
        "must be numbers",
    ),
    "features not numeric": (
        _set("Test", "features", [["a"] * 26] * N),
        "Test",
        "features",
        "must be numbers",
    ),
    "features nan": (
        _set("Test", "features", np.where(np.eye(N, 26) > 0, np.nan, 0.0)),
        "Test",
        "features",
        "row 0 has NaN",
    ),
}


@pytest.mark.parametrize("case", list(VIOLATIONS))
def test_schema_violations_name_file_split_and_key(case: str, tmp_path: Path) -> None:
    mutate, split, key, match = VIOLATIONS[case]
    path = _write(tmp_path / "bad.pkl", mutate(_classifier()))
    with pytest.raises(SchemaError, match=match) as exc:
        load_pickle(path)
    assert exc.value.path == path
    assert (exc.value.split, exc.value.key) == (split, key)
    assert str(exc.value).startswith(str(path))


def test_validate_is_usable_without_a_file() -> None:
    with pytest.raises(SchemaError, match=r"^x.pkl: the top level"):
        validate(Path("x.pkl"), 42)


def test_truncated_file_fails_before_unpickling(tmp_path: Path) -> None:
    full = pickle.dumps(_classifier(), protocol=4)
    path = tmp_path / "cut.pkl"
    path.write_bytes(full[: len(full) // 2])
    with pytest.raises(TruncatedFileError, match=r"truncated .*STOP opcode"):
        load_pickle(path)


def test_truncated_file_ending_in_a_stop_byte(tmp_path: Path) -> None:
    # A cut that happens to end in b"." passes the fast check and fails while unpickling.
    full = pickle.dumps(_classifier(), protocol=4)
    path = tmp_path / "cut.pkl"
    path.write_bytes(full[:1000] + b".")
    with pytest.raises(DataError, match=r"truncated|cannot unpickle"):
        load_pickle(path)


def test_eof_while_unpickling_is_reported_as_truncation(tmp_path: Path) -> None:
    # PROTO 2, then BININT1 whose argument byte is "." (so the fast check passes); the
    # unpickler then reads past the end looking for the next opcode: EOFError.
    path = tmp_path / "eof.pkl"
    path.write_bytes(b"\x80\x02K.")
    with pytest.raises(TruncatedFileError, match=r"ends in the middle of the data"):
        load_pickle(path)


def test_empty_and_missing_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pkl"
    empty.write_bytes(b"")
    with pytest.raises(TruncatedFileError, match=r"empty"):
        load_pickle(empty)
    with pytest.raises(DataError, match=r"cannot open"):
        load_pickle(tmp_path / "missing.pkl")


def test_corrupt_bytes(tmp_path: Path) -> None:
    path = tmp_path / "junk.pkl"
    path.write_bytes(b"not a pickle at all.")
    with pytest.raises(DataError, match=r"cannot unpickle"):
        load_pickle(path)


def test_only_numpy_globals_are_resolved(tmp_path: Path) -> None:
    path = _write(tmp_path / "date.pkl", {"Train": datetime.date(2020, 6, 17)})
    with pytest.raises(DataError, match=r"global datetime.date is not allowed"):
        load_pickle(path)


@pytest.mark.parametrize("module", ["numpy.core.multiarray", "numpy._core.multiarray"])
def test_numpy_1x_and_2x_module_names_are_accepted(module: str) -> None:
    # The real files were written by numpy 1.x (numpy.core.multiarray); numpy 2 writes
    # numpy._core. Both resolve to the numpy 2 objects, without a deprecation warning.
    unpickler = _NumpyOnlyUnpickler(io.BytesIO())
    multiarray = importlib.import_module("numpy._core.multiarray")
    assert unpickler.find_class(module, "_reconstruct") is multiarray._reconstruct
    assert unpickler.find_class(module, "scalar") is multiarray.scalar
    assert unpickler.find_class("numpy", "dtype") is np.dtype
