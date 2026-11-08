"""Tests of the ``raw-alerts`` prepared kind (astro_simclr.data.raw_alerts, task 3.1b)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from astro_simclr.data import DataError, SchemaError, load_prepared, rj23_input
from astro_simclr.data.inspection import format_summary, summarize
from astro_simclr.data.m1_features import M1_FEATURE_NORM_STATS
from astro_simclr.data.prepared import (
    META_NAME,
    layout_kind,
    prepare,
    read_meta,
    read_meta_file,
)
from astro_simclr.data.raw_alerts import (
    CD21_FEATURES,
    CUTOUTS_NAME,
    FEATURES_NAME,
    FORMAT_VERSION,
    INDEX_DTYPE,
    INDEX_NAME,
    KIND,
    OID_PATTERN,
    RawAlerts,
    check_label_names,
    label_names_of,
    load_raw_alerts,
    make_index,
    verify_raw_alerts,
    write_raw_alerts,
)
from astro_simclr.data.synthetic import (
    CLASSIFIER_FILES,
    make_synthetic,
    synthetic_cd21_features,
    synthetic_raw_alerts,
)
from astro_simclr.preflight import build_manifest, prepared_dirs
from astro_simclr.protocol import alerts
from astro_simclr.protocol.alerts import RJ23_LABEL_NAMES

#: RJ23's six classes (task 3.2's names): a label set other than CD21's five.
RJ23_NAMES = RJ23_LABEL_NAMES
SOURCE = {"name": "synthetic", "seed": 0}

Columns = dict[str, npt.NDArray[Any]]


@pytest.fixture
def arrays() -> tuple[npt.NDArray[np.float32], Columns]:
    return synthetic_raw_alerts(RJ23_NAMES, n_per_class=4, seed=0)


@pytest.fixture
def layout(tmp_path: Path, arrays: tuple[npt.NDArray[np.float32], Columns]) -> Path:
    cutouts, columns = arrays
    return write_raw_alerts(tmp_path / "rj23", cutouts, columns, RJ23_NAMES, SOURCE)


def _rewrite_meta(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    meta = json.loads((path / META_NAME).read_text())
    change(meta)
    (path / META_NAME).write_text(json.dumps(meta))


# --- A synthetic 6-class layout loads -----------------------------------------------------


def test_six_class_layout_round_trips(
    layout: Path, arrays: tuple[npt.NDArray[np.float32], Columns]
) -> None:
    cutouts, columns = arrays
    data = load_raw_alerts(layout)
    assert isinstance(data, RawAlerts)
    assert len(data) == 24 and data.n_classes == 6 and data.label_names == RJ23_NAMES
    assert data.class_counts() == [4] * 6
    # Raw values kept bit for bit, NaN and inf included (the synthetic set has both).
    assert np.isnan(cutouts).any() and np.isinf(cutouts).any()
    np.testing.assert_array_equal(data.cutouts, cutouts)  # NaN == NaN for assert_array_equal
    assert data.cutouts.dtype == np.float32 and isinstance(data.cutouts, np.memmap)
    for name in INDEX_DTYPE.names or ():
        np.testing.assert_array_equal(data.index[name], columns[name])
    assert data.labels.dtype == np.int64
    np.testing.assert_array_equal(data.labels, columns["label"])
    verify_raw_alerts(layout)
    assert label_names_of(layout) == RJ23_NAMES


def test_meta_records_the_schema_and_hashes(layout: Path) -> None:
    meta = read_meta_file(layout)
    assert meta["format_version"] == FORMAT_VERSION == 3
    assert meta["kind"] == KIND
    assert meta["source"] == SOURCE
    assert meta["schema"]["label_names"] == list(RJ23_NAMES)
    assert meta["schema"]["channels"] == ["science", "reference", "difference"]
    assert [field for field, _ in meta["schema"]["index_fields"]] == list(INDEX_DTYPE.names or ())
    assert meta["schema"]["feature_names"] is None
    assert meta["n"] == 24 and meta["class_counts"] == [4] * 6
    assert sorted(meta["files"]) == [CUTOUTS_NAME, INDEX_NAME]
    for info in meta["files"].values():
        assert len(info["sha256"]) == 64
    assert sorted(p.name for p in layout.iterdir()) == sorted([META_NAME, CUTOUTS_NAME, INDEX_NAME])


def test_index_reads_without_pickle(layout: Path) -> None:
    index = np.load(layout / INDEX_NAME, allow_pickle=False)
    assert index.dtype == INDEX_DTYPE
    assert not index.dtype.hasobject


def test_writer_is_deterministic(
    tmp_path: Path, arrays: tuple[npt.NDArray[np.float32], Columns]
) -> None:
    cutouts, columns = arrays
    first = write_raw_alerts(tmp_path / "a", cutouts, columns, RJ23_NAMES, SOURCE)
    second = write_raw_alerts(tmp_path / "b", cutouts, make_index(columns), RJ23_NAMES, SOURCE)
    for name in (META_NAME, CUTOUTS_NAME, INDEX_NAME):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_any_number_of_classes(tmp_path: Path) -> None:
    cutouts, columns = synthetic_raw_alerts(("real", "bogus"), n_per_class=2, seed=3)
    data = load_raw_alerts(
        write_raw_alerts(tmp_path / "rb", cutouts, columns, ("real", "bogus"), {})
    )
    assert data.n_classes == 2


def test_make_index_accepts_pandas_like_object_strings(
    arrays: tuple[npt.NDArray[np.float32], Columns],
) -> None:
    _, columns = arrays
    columns["oid"] = columns["oid"].astype(object)  # what DataFrame.to_numpy() gives
    assert make_index(columns)["oid"][0] == "ZTF18syn00000"


def test_overwrite_rules(tmp_path: Path, arrays: tuple[npt.NDArray[np.float32], Columns]) -> None:
    cutouts, columns = arrays
    dest = write_raw_alerts(tmp_path / "x", cutouts, columns, RJ23_NAMES, SOURCE)
    with pytest.raises(DataError, match="already exists"):
        write_raw_alerts(dest, cutouts, columns, RJ23_NAMES, SOURCE)
    write_raw_alerts(dest, cutouts, columns, RJ23_NAMES, {"name": "again"}, overwrite=True)
    assert read_meta_file(dest)["source"] == {"name": "again"}
    other = tmp_path / "not-a-layout"
    other.mkdir()
    (other / "keep.txt").write_text("x")
    with pytest.raises(DataError, match="not a prepared layout"):
        write_raw_alerts(other, cutouts, columns, RJ23_NAMES, SOURCE, overwrite=True)
    assert (other / "keep.txt").exists()
    # A stale .partial from an interrupted run is replaced.
    stale = tmp_path / ".y.partial"
    stale.mkdir()
    write_raw_alerts(tmp_path / "y", cutouts, columns, RJ23_NAMES, SOURCE)
    assert not stale.exists()


# --- Malformed layouts are refused ----------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("candid", "dup", "candid .* is duplicated"),
        ("candid", 0, "candid <= 0"),
        ("label", 6, "label outside 0..5"),
        ("label", -1, "label outside 0..5"),
        ("fid", 4, "fid not in"),
        ("ra", 360.0, "ra outside"),
        ("dec", np.nan, "dec outside"),
        ("jd", np.inf, "jd that is not finite and > 0"),
        ("jd", 0.0, "jd that is not finite and > 0"),
        ("oid", "", "oid that does not match"),
        ("oid", " ZTF1", "oid that does not match"),
        ("oid", "ZTF1/x", "oid that does not match"),
        ("label_source", "", "empty label_source"),
        ("source_dataset", "", "empty source_dataset"),
        ("source_row", -1, "source_row < 0"),
    ],
)
def test_writer_refuses_bad_rows(
    tmp_path: Path,
    arrays: tuple[npt.NDArray[np.float32], Columns],
    column: str,
    value: object,
    message: str,
) -> None:
    cutouts, columns = arrays
    columns[column] = columns[column].copy()
    columns[column][5] = columns[column][2] if value == "dup" else value
    with pytest.raises(ValueError, match=message):
        write_raw_alerts(tmp_path / "bad", cutouts, columns, RJ23_NAMES, SOURCE)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c, cols: (c.astype(np.float64), cols), "cutouts must be float32"),
        (lambda c, cols: (c[:, :62], cols), "cutouts must be float32"),
        (lambda c, cols: (c[:-1], cols), "cutouts must be float32"),
        (lambda c, cols: (c, {k: v for k, v in cols.items() if k != "jd"}), "missing \\['jd'\\]"),
        (lambda c, cols: (c, {**cols, "extra": cols["jd"]}), "unexpected \\['extra'\\]"),
        (lambda c, cols: (c, {**cols, "jd": cols["jd"][:-1]}), "of one length"),
        (lambda c, cols: (c, {**cols, "label": cols["label"] * 1.0}), "must hold integers"),
        (lambda c, cols: (c, {**cols, "label": cols["label"] > 0}), "must hold integers"),
        (lambda c, cols: (c, {**cols, "oid": cols["source_row"]}), "must hold strings"),
        (lambda c, cols: (c, {**cols, "ra": cols["oid"]}), "must hold numbers"),
        (lambda c, cols: (c, {**cols, "oid": np.char.add(cols["oid"], "-long")}), "longer than"),
    ],
)
def test_writer_refuses_bad_arrays(
    tmp_path: Path,
    arrays: tuple[npt.NDArray[np.float32], Columns],
    change: Callable[[Any, Columns], tuple[Any, Columns]],
    message: str,
) -> None:
    cutouts, columns = change(*arrays)
    with pytest.raises(ValueError, match=message):
        write_raw_alerts(tmp_path / "bad", cutouts, columns, RJ23_NAMES, SOURCE)


def test_writer_refuses_a_wrong_structured_index(
    tmp_path: Path, arrays: tuple[npt.NDArray[np.float32], Columns]
) -> None:
    cutouts, columns = arrays
    index = make_index(columns)
    wider = index.astype([*INDEX_DTYPE.descr[:-1], ("source_row", "<i4")])
    with pytest.raises(ValueError, match="1-D array of"):
        write_raw_alerts(tmp_path / "bad", cutouts, wider, RJ23_NAMES, SOURCE)
    with pytest.raises(ValueError, match="the index is empty"):
        write_raw_alerts(tmp_path / "bad", cutouts[:0], index[:0], RJ23_NAMES, SOURCE)
    with pytest.raises(ValueError, match="source must be a mapping"):
        write_raw_alerts(tmp_path / "bad", cutouts, index, RJ23_NAMES, ["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON-serialisable"):
        write_raw_alerts(tmp_path / "bad", cutouts, index, RJ23_NAMES, {"x": object()})


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (("only",), "at least 2"),
        (("a", "a"), "repeat"),
        (("a", ""), "non-empty"),
        (("a", " b"), "stripped"),
        (("a", 3), "non-empty"),
    ],
)
def test_label_names_are_checked(names: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_label_names(names)


def _save_index(layout: Path, index: npt.NDArray[Any]) -> None:
    np.save(layout / INDEX_NAME, index, allow_pickle=False)


def test_loader_refuses_a_duplicated_candid(layout: Path) -> None:
    index = np.load(layout / INDEX_NAME)
    index["candid"][3] = index["candid"][1]
    _save_index(layout, index)
    with pytest.raises(SchemaError, match=r"candid \d+ is duplicated \(rows \[1, 3\]\)"):
        load_raw_alerts(layout)
    with pytest.raises(DataError, match="SHA-256 differs"):
        verify_raw_alerts(layout)


def test_loader_refuses_a_label_out_of_range(layout: Path) -> None:
    index = np.load(layout / INDEX_NAME)
    index["label"][0] = 6
    _save_index(layout, index)
    with pytest.raises(SchemaError, match=r"label outside 0\.\.5"):
        load_raw_alerts(layout)


def test_loader_refuses_fewer_label_names_than_labels(layout: Path) -> None:
    _rewrite_meta(layout, lambda m: m["schema"].update(label_names=list(RJ23_NAMES[:5])))
    with pytest.raises(SchemaError, match=r"label outside 0\.\.4"):
        load_raw_alerts(layout)


def test_loader_refuses_a_wrong_cutout_dtype_or_shape(layout: Path) -> None:
    cutouts = np.load(layout / CUTOUTS_NAME)
    np.save(layout / CUTOUTS_NAME, cutouts.astype(np.float64))
    with pytest.raises(SchemaError, match=r"meta\.json says float32"):
        load_raw_alerts(layout)
    # The same mistake recorded in meta.json too: the schema check still refuses it.
    _rewrite_meta(layout, lambda m: m["files"][CUTOUTS_NAME].update(dtype="float64"))
    with pytest.raises(SchemaError, match="cutouts must be float32"):
        load_raw_alerts(layout)


def test_loader_refuses_a_cutout_count_that_differs(layout: Path) -> None:
    cutouts = np.load(layout / CUTOUTS_NAME)[:-1]
    np.save(layout / CUTOUTS_NAME, cutouts)
    _rewrite_meta(layout, lambda m: m["files"][CUTOUTS_NAME].update(shape=list(cutouts.shape)))
    with pytest.raises(SchemaError, match=r"cutouts must be float32 \(24, 63, 63, 3\)"):
        load_raw_alerts(layout)


def test_loader_refuses_a_wrong_index_dtype(layout: Path) -> None:
    index = np.load(layout / INDEX_NAME)
    narrow = index.astype([*INDEX_DTYPE.descr[:-1], ("source_row", "<i4")])
    _save_index(layout, narrow)
    descr = str(narrow.dtype.descr)
    _rewrite_meta(layout, lambda m: m["files"][INDEX_NAME].update(dtype=descr))
    with pytest.raises(SchemaError, match="1-D array of"):
        load_raw_alerts(layout)


def test_loader_never_unpickles(layout: Path) -> None:
    np.save(layout / INDEX_NAME, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(SchemaError, match=r"cannot load index\.npy"):
        load_raw_alerts(layout)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda m: m.update(kind="classifier"), DataError, "not a raw-alerts layout"),
        (lambda m: m.update(format_version=2), DataError, "not a raw-alerts layout"),
        (lambda m: m.update(format="other"), DataError, "not an astro-simclr-prepared"),
        (lambda m: m.update(files={CUTOUTS_NAME: {}}), SchemaError, "must list"),
        (lambda m: m.update(n=23), SchemaError, "n=23"),
        (lambda m: m.update(class_counts=[24, 0, 0, 0, 0, 0]), SchemaError, "class_counts"),
        (lambda m: m["schema"].pop("label_names"), SchemaError, "no schema.label_names"),
        (lambda m: m["schema"].update(label_names=["a", "a"]), SchemaError, "repeat"),
        (lambda m: m["schema"].update(channels=["r", "s", "d"]), SchemaError, "schema differs"),
        (lambda m: m["schema"]["index_fields"].pop(), SchemaError, "schema differs"),
        (lambda m: m["files"][INDEX_NAME].update(shape=[3]), SchemaError, "meta.json says"),
    ],
)
def test_loader_refuses_a_bad_meta(
    layout: Path,
    change: Callable[[dict[str, Any]], None],
    error: type[Exception],
    message: str,
) -> None:
    _rewrite_meta(layout, change)
    with pytest.raises(error, match=message):
        load_raw_alerts(layout)


def test_loader_refuses_a_missing_file(layout: Path) -> None:
    (layout / INDEX_NAME).unlink()
    with pytest.raises(SchemaError, match=r"cannot load index\.npy"):
        load_raw_alerts(layout)
    with pytest.raises(DataError, match="missing"):
        verify_raw_alerts(layout)
    (layout / META_NAME).unlink()
    with pytest.raises(DataError, match="not a prepared layout"):
        load_raw_alerts(layout)


def test_verify_refuses_meta_without_files(layout: Path) -> None:
    _rewrite_meta(layout, lambda m: m.update(files=None))
    with pytest.raises(DataError, match="no files listed"):
        verify_raw_alerts(layout)


# --- The phase-2 readers and tools ------------------------------------------------------------


def test_phase2_readers_refuse_it_by_name(layout: Path) -> None:
    for reader in (read_meta, load_prepared):
        with pytest.raises(DataError, match="load_raw_alerts"):
            reader(layout)


def test_raw_reader_refuses_a_split_layout(tmp_path: Path) -> None:
    make_synthetic(tmp_path / "raw", n_per_split=10, seed=0)
    prepared = prepare(tmp_path / "raw" / CLASSIFIER_FILES[0], tmp_path / "prepared")
    with pytest.raises(DataError, match="not a raw-alerts layout"):
        load_raw_alerts(prepared)
    assert layout_kind(prepared) == "classifier"


def test_layout_kind(layout: Path, tmp_path: Path) -> None:
    assert layout_kind(layout) == KIND
    assert layout_kind(tmp_path / "missing") is None
    _rewrite_meta(layout, lambda m: m.update(kind=3))
    assert layout_kind(layout) is None


def test_inspect_data_summarises_it(layout: Path) -> None:
    summary = summarize(layout)
    assert summary["format"] == summary["kind"] == KIND
    assert summary["n"] == 24
    assert summary["class_counts"] == dict.fromkeys(RJ23_NAMES, 4)
    assert summary["label_sources"] == {"synthetic": 24}
    text = format_summary(summary)
    assert "24 alerts" in text and "Satellite 4" in text and "synthetic 24" in text
    assert "cutouts.npy  float32 (24, 63, 63, 3)" in text


def test_preflight_leaves_it_out(layout: Path, tmp_path: Path) -> None:
    make_synthetic(tmp_path / "raw", n_per_split=10, seed=0)
    prepared = prepare(tmp_path / "raw" / CLASSIFIER_FILES[0], tmp_path)
    assert prepared_dirs(tmp_path) == [prepared]
    assert list(build_manifest(tmp_path)["datasets"]) == [prepared.name]


def test_synthetic_raw_alerts_checks_its_arguments() -> None:
    with pytest.raises(ValueError, match=">= 2 classes"):
        synthetic_raw_alerts(("a",), 1, 0)
    with pytest.raises(ValueError, match="n_per_class >= 1"):
        synthetic_raw_alerts(("a", "b"), 0, 0)


# --- CD21 features, candid rows and task 3.2's alert table ----------------------------------


def test_cd21_features_are_m1s_23() -> None:
    assert tuple(M1_FEATURE_NORM_STATS) == CD21_FEATURES
    assert len(CD21_FEATURES) == 23


def test_features_round_trip(
    tmp_path: Path, arrays: tuple[npt.NDArray[np.float32], Columns]
) -> None:
    cutouts, columns = arrays
    features = synthetic_cd21_features(len(cutouts), seed=1)
    dest = write_raw_alerts(tmp_path / "f", cutouts, columns, RJ23_NAMES, SOURCE, features=features)
    data = load_raw_alerts(dest)
    assert data.features is not None
    np.testing.assert_array_equal(data.features, features)  # NaN kept
    assert np.isnan(data.features).any()
    meta = read_meta_file(dest)
    assert meta["schema"]["feature_names"] == list(CD21_FEATURES)
    assert sorted(meta["files"]) == [CUTOUTS_NAME, FEATURES_NAME, INDEX_NAME]
    verify_raw_alerts(dest)
    assert "features.npy" in format_summary(summarize(dest))
    assert (
        load_raw_alerts(
            write_raw_alerts(tmp_path / "g", cutouts, columns, RJ23_NAMES, SOURCE)
        ).features
        is None
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.astype(np.float32), "features must be float64"),
        (lambda f: f[:, :22], "features must be float64"),
        (lambda f: f[:-1], "features must be float64"),
        (lambda f: np.where(np.arange(23) == 8, np.inf, f), "infinite magpsf"),
    ],
)
def test_writer_refuses_bad_features(
    tmp_path: Path,
    arrays: tuple[npt.NDArray[np.float32], Columns],
    change: Callable[[npt.NDArray[np.float64]], npt.NDArray[Any]],
    message: str,
) -> None:
    cutouts, columns = arrays
    features = change(synthetic_cd21_features(len(cutouts), seed=1))
    with pytest.raises(ValueError, match=message):
        write_raw_alerts(tmp_path / "bad", cutouts, columns, RJ23_NAMES, SOURCE, features=features)


def test_loader_refuses_bad_features(
    tmp_path: Path, arrays: tuple[npt.NDArray[np.float32], Columns]
) -> None:
    cutouts, columns = arrays
    features = synthetic_cd21_features(len(cutouts), seed=1)
    dest = write_raw_alerts(tmp_path / "f", cutouts, columns, RJ23_NAMES, SOURCE, features=features)
    features[0, 0] = -np.inf
    np.save(dest / FEATURES_NAME, features)
    with pytest.raises(SchemaError, match="infinite sgscore1"):
        load_raw_alerts(dest)
    (dest / FEATURES_NAME).unlink()
    with pytest.raises(SchemaError, match=r"cannot load features\.npy"):
        load_raw_alerts(dest)
    # meta.json lists the file but the schema says there are no features: refused.
    _rewrite_meta(dest, lambda m: m["schema"].update(feature_names=None))
    with pytest.raises(SchemaError, match="must list exactly"):
        load_raw_alerts(dest)
    _rewrite_meta(dest, lambda m: m["schema"].update(feature_names=["x"]))
    with pytest.raises(SchemaError, match="schema differs"):
        load_raw_alerts(dest)


def test_rows_of_maps_candids_to_rows(layout: Path) -> None:
    data = load_raw_alerts(layout)
    candids = data.index["candid"][[5, 0, 17]]
    np.testing.assert_array_equal(data.rows_of(candids), [5, 0, 17])
    with pytest.raises(DataError, match="candid 7 is not in the layout"):
        data.rows_of([candids[0], 7])


def test_alert_table_from_raw(layout: Path) -> None:
    data = load_raw_alerts(layout)
    table = alerts.alert_table_from_raw(data)
    assert len(table) == len(data) and table.label_names == RJ23_NAMES
    assert table.group_key == "oid"
    rows = data.rows_of(table.candid)
    np.testing.assert_array_equal(data.index["oid"][rows], table.oid)
    np.testing.assert_array_equal(data.labels[rows], table.label)
    grouped = alerts.alert_table_from_raw(data, group=[f"g{i % 3}" for i in range(len(data))])
    assert grouped.group_key == "group"


def test_oid_rule_matches_the_alert_table() -> None:
    # The same character set as task 3.2's alert table, so every stored row is accepted.
    for oid in ("ZTF18abcdefg", "a_b-c.d", "bad oid", "x/y", ""):
        ours = OID_PATTERN.fullmatch(oid) is not None
        assert ours == (alerts.OID_PATTERN.fullmatch(oid) is not None)


def test_synthetic_cd21_features_checks_n() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        synthetic_cd21_features(0, 0)


def test_rj23_input_reads_a_loaded_layout(layout: Path) -> None:
    # Task 3.3's input path takes the memory-mapped cutouts and the index as they are.
    data = load_raw_alerts(layout)
    rows = data.rows_of(data.index["candid"][:5])
    views = rj23_input.rj23_views(data.cutouts[rows])
    for name, shape in rj23_input.VIEW_SHAPES.items():
        assert views[name].shape == (5, *shape)
        assert np.isfinite(views[name]).all()
    position = rj23_input.unit_sphere(data.index["ra"][rows], data.index["dec"][rows])
    np.testing.assert_allclose(np.linalg.norm(position, axis=1), 1.0, rtol=1e-6)
