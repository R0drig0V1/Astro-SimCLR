"""The ``raw-alerts`` prepared kind (format version 3, task 3.1b): raw cutouts, N classes.

The phase-2 layouts (:mod:`astro_simclr.data.prepared`, versions 1 and 2) hold CD21's
pickles: three fixed splits, five classes, stamps already min-max normalised, 26 features.
Phase 3 works from **raw** alert cutouts (task 3.1 downloads them) with its own label sets
(R5's five classes, RJ23's six, any later one), and draws object-grouped splits (task 3.2).
This kind holds such a dataset as one pool of rows, without splits::

    OUT/<name>/
        meta.json      format, version 3, kind "raw-alerts", the label names, the schema
                       (index fields, feature names), the source (provenance), n, the
                       class counts, and the dtype, shape and SHA-256 of every file
        cutouts.npy    float32 (N, 63, 63, 3): science, reference, difference, as the
                       alert stored them (NaN and inf kept)
        index.npy      structured (N,), one row per cutout (:data:`INDEX_DTYPE`)
        features.npy   float64 (N, 23), optional: CD21's features (:data:`CD21_FEATURES`)

Row ``i`` of ``index.npy`` (and of ``features.npy``) describes cutout ``i``. The phase-3
protocol (task 3.2, ADR 0008) names alerts by ``candid`` and groups them by ``oid``:
``protocol.alerts.alert_table_from_raw`` builds its alert table from the index, and
:meth:`RawAlerts.rows_of` maps the ``candid`` of a split back to rows.

**Features** (plan 3.1b: phase-2 models are evaluated on R5): the 23 alert features of
CD21 that M1 reads, in the order of ``m1_features.M1_FEATURE_NORM_STATS``, in CD21's
definitions (its Table A1, with ALeRCE's -999 sentinels already clipped) but **not
standardised**: ``(x - mean) / std`` with those statistics gives a phase-2 model's input.
NaN marks a value the alert lacks (the evaluation decides how to fill it); inf is refused.

**Why a structured ``.npy`` for the index** (rather than CSV or Parquet): numpy reads it with
``allow_pickle=False`` (a structured dtype of fixed-width strings and numbers has no object
field, so the file cannot carry a pickle); the dtype and shape are in the header and checked
before any value is used; floats round-trip bit for bit (``jd`` needs 15+ significant
digits); it memory-maps like the cutouts and is hashed the same way; and it needs no new
dependency. Its cost is fixed-width strings (:data:`OID_WIDTH`, :data:`TEXT_WIDTH`), which
the builder checks instead of letting numpy truncate silently. A human-readable copy for
review or for a frozen, committed index (ADR 0012) is an export of this file, not its source.

**No invalid-pixel mask is stored:** RJ23's mask (1 where a pixel is NaN or inf) is a pure
function of the cutouts, so storing it would only add a second copy that could disagree.
The input path of task 3.3 derives it.

:func:`write_raw_alerts` is the builder (task 3.1's fetcher calls it with its cutouts and
its per-row table); :func:`load_raw_alerts` opens a layout memory-mapped and refuses a
malformed one; :func:`verify_raw_alerts` checks the SHA-256 of every file. The writer
checks exactly what the loader checks, so it never writes a layout the loader refuses.
Downloaded data is untrusted: nothing here unpickles, and every value that the index
constrains is checked.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.data.files import sha256_file
from astro_simclr.data.prepared import (
    FORMAT,
    META_NAME,
    RAW_ALERTS_KIND,
    is_prepared,
    read_meta_file,
)
from astro_simclr.data.schema import (
    DISCARDED_FEATURES,
    FEATURE_NAMES,
    STAMP_SHAPE,
    DataError,
    SchemaError,
)

__all__ = [
    "CD21_FEATURES",
    "CHANNEL_NAMES",
    "CUTOUTS_NAME",
    "FEATURES_NAME",
    "FORMAT_VERSION",
    "INDEX_DTYPE",
    "INDEX_NAME",
    "KIND",
    "OID_PATTERN",
    "OID_WIDTH",
    "TEXT_WIDTH",
    "ZTF_FIDS",
    "RawAlerts",
    "check_label_names",
    "label_names_of",
    "load_raw_alerts",
    "make_index",
    "verify_raw_alerts",
    "write_raw_alerts",
]

#: ``meta.json["kind"]`` of this layout.
KIND: Final = RAW_ALERTS_KIND
#: ``meta.json["format_version"]`` of this layout (the split layouts are versions 1 and 2).
FORMAT_VERSION: Final = 3
#: The array files at the root of the layout (``features.npy`` is optional).
CUTOUTS_NAME: Final = "cutouts.npy"
INDEX_NAME: Final = "index.npy"
FEATURES_NAME: Final = "features.npy"
#: The 23 CD21 features of ``features.npy``, in column order: the 26 stored ones minus
#: ``DISCARDED_FEATURES`` (``rb``, ``distnr``, ``magnr``), as M1 reads them.
CD21_FEATURES: Final = tuple(
    name for i, name in enumerate(FEATURE_NAMES) if i not in DISCARDED_FEATURES
)
#: The cutout channels, in file order (the alert's science, reference and difference).
CHANNEL_NAMES: Final = ("science", "reference", "difference")
#: Maximum length of an ``oid`` (ZTF object ids have 12 characters, ``ZTF18abcdefg``).
OID_WIDTH: Final = 16
#: A valid ``oid``: the character set of ``protocol.alerts.OID_PATTERN`` (task 3.2's alert
#: table), within :data:`OID_WIDTH` (the field width bounds the length).
OID_PATTERN: Final = re.compile(r"[A-Za-z0-9_.\-]+")
#: Maximum length of ``label_source`` and ``source_dataset``.
TEXT_WIDTH: Final = 64
#: The ZTF filter ids ``fid`` may take: 1 (g), 2 (r) and 3 (i).
ZTF_FIDS: Final = (1, 2, 3)

#: One row of ``index.npy``. Little-endian, so the file is the same on every machine.
#:
#: - ``oid``: the ZTF object id (:data:`OID_PATTERN`); ``candid``: the alert's candidate
#:   id (> 0, unique);
#: - ``jd``: the alert's Julian date (finite, > 0); ``fid``: its filter (:data:`ZTF_FIDS`);
#: - ``ra``, ``dec``: degrees, ra in [0, 360), dec in [-90, 90];
#: - ``label``: the class, an index into the layout's label names;
#: - ``label_source``: where the label comes from (for example ``cd21``, ``zenodo-4279623``,
#:   ``tns``, ``visual``); never an ALeRCE classifier output (phase-3 rule);
#: - ``source_dataset``, ``source_row``: the table the row was built from and its row there
#:   (>= 0), so every row can be traced back.
INDEX_DTYPE: Final = np.dtype(
    [
        ("oid", f"<U{OID_WIDTH}"),
        ("candid", "<i8"),
        ("jd", "<f8"),
        ("fid", "<i8"),
        ("ra", "<f8"),
        ("dec", "<f8"),
        ("label", "<i8"),
        ("label_source", f"<U{TEXT_WIDTH}"),
        ("source_dataset", f"<U{TEXT_WIDTH}"),
        ("source_row", "<i8"),
    ]
)
_TEXT_FIELDS: Final = ("oid", "label_source", "source_dataset")
_INT_FIELDS: Final = ("candid", "fid", "label", "source_row")
_FLOAT_FIELDS: Final = ("jd", "ra", "dec")
_CUTOUT_DTYPE: Final = np.dtype("<f4")
_FEATURE_DTYPE: Final = np.dtype("<f8")


@dataclass(frozen=True, slots=True)
class RawAlerts:
    """A loaded ``raw-alerts`` layout (see the module docstring).

    Attributes:
        cutouts: float32 (N, 63, 63, 3), science / reference / difference, raw (NaN and
            inf kept); usually a read-only memory map.
        index: structured (N,) array of :data:`INDEX_DTYPE`, row ``i`` for cutout ``i``.
        label_names: the class names, indexed by ``index["label"]`` (at least two).
        features: float64 (N, 23) CD21 features (:data:`CD21_FEATURES`, not
            standardised, NaN where missing), or ``None`` if the layout has none.
    """

    cutouts: npt.NDArray[np.float32]
    index: npt.NDArray[np.void]
    label_names: tuple[str, ...]
    features: npt.NDArray[np.float64] | None = None

    def __len__(self) -> int:
        return int(self.index.shape[0])

    @property
    def n_classes(self) -> int:
        """The number of classes of the dataset (what an N-class classifier is built with)."""
        return len(self.label_names)

    @property
    def labels(self) -> npt.NDArray[np.int64]:
        """The labels, int64 (N,), values in ``range(n_classes)`` (a copy)."""
        return np.array(self.index["label"], dtype=np.int64, copy=True)

    def class_counts(self) -> list[int]:
        """Number of rows per class, indexed by label."""
        return [int(c) for c in np.bincount(self.index["label"], minlength=self.n_classes)]

    def rows_of(self, candids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """The rows of the alerts ``candids``, in the given order (how a split is gathered).

        Raises:
            DataError: if a ``candid`` is not in the layout.
        """
        wanted = np.asarray(candids, dtype=np.int64).reshape(-1)
        candid = self.index["candid"]
        order = np.argsort(candid, kind="stable")
        position = np.minimum(np.searchsorted(candid, wanted, sorter=order), len(order) - 1)
        rows = order[position]
        missing = wanted[candid[rows] != wanted]
        if missing.size:
            raise DataError(f"candid {int(missing[0])} is not in the layout")
        return rows.astype(np.int64)


def check_label_names(label_names: Sequence[str]) -> tuple[str, ...]:
    """Return ``label_names`` as a tuple after checking it.

    Raises:
        ValueError: if there are fewer than two names, a name is not a non-empty string
            without surrounding whitespace, or a name repeats.
    """
    names = tuple(label_names)
    if len(names) < 2:
        raise ValueError(f"label_names needs at least 2 classes, got {len(names)}")
    for name in names:
        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError(f"label names must be non-empty, stripped strings, got {name!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"label names repeat: {list(names)}")
    return names


def make_index(columns: Mapping[str, npt.ArrayLike]) -> npt.NDArray[np.void]:
    """Build an ``index.npy`` table (:data:`INDEX_DTYPE`) from one array per column.

    ``columns`` has exactly the fields of :data:`INDEX_DTYPE`, each 1-D and of the same
    length (a ``pandas`` frame converts with ``{c: df[c].to_numpy() for c in df}``). Text
    columns take strings (object arrays of ``str`` too), integer columns integers only (no
    float or bool cast), float columns floats or integers. Values are not range-checked
    here: :func:`write_raw_alerts` does that.

    Raises:
        ValueError: for a missing or extra column, a column that is not 1-D, lengths that
            differ, a column of the wrong kind, or a string longer than its field (numpy
            would truncate it silently).
    """
    names = set(INDEX_DTYPE.names or ())
    if set(columns) != names:
        missing, extra = sorted(names - set(columns)), sorted(set(columns) - names)
        raise ValueError(f"index columns: missing {missing}, unexpected {extra}")
    arrays = {name: np.asarray(columns[name]) for name in INDEX_DTYPE.names or ()}
    lengths = {name: array.shape for name, array in arrays.items()}
    if any(array.ndim != 1 for array in arrays.values()) or len(set(lengths.values())) != 1:
        raise ValueError(f"index columns must be 1-D of one length, got shapes {lengths}")
    table = np.empty(len(arrays["candid"]), dtype=INDEX_DTYPE)
    for name, array in arrays.items():
        table[name] = _column(name, array)
    return table


def _column(name: str, array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    if name in _TEXT_FIELDS:
        if array.dtype.kind == "O" and all(isinstance(v, str) for v in array):
            array = array.astype(str)
        if array.dtype.kind != "U":
            raise ValueError(f"index column {name!r} must hold strings, got {array.dtype}")
        width = INDEX_DTYPE[name].itemsize // 4
        too_long = np.char.str_len(array) > width
        if too_long.any():
            row = int(np.flatnonzero(too_long)[0])
            raise ValueError(f"{name} at row {row} is longer than {width}: {array[row]!r}")
        return array
    allowed = "iu" if name in _INT_FIELDS else "iuf"
    if array.dtype.kind not in allowed:
        what = "integers" if name in _INT_FIELDS else "numbers"
        raise ValueError(f"index column {name!r} must hold {what}, got {array.dtype}")
    return array


def write_raw_alerts(
    dest: Path,
    cutouts: npt.ArrayLike,
    index: npt.NDArray[np.void] | Mapping[str, npt.ArrayLike],
    label_names: Sequence[str],
    source: Mapping[str, Any],
    *,
    features: npt.ArrayLike | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a ``raw-alerts`` layout at ``dest`` and return ``dest``.

    The builder of task 3.1b, the interface for task 3.1's fetcher:

    - ``cutouts``: float32 (N, 63, 63, 3), science / reference / difference, raw; NaN and
      inf are stored as they are. Other float dtypes are refused rather than cast, so the
      stored values are exactly the fetcher's.
    - ``index``: the per-row table, a structured array of :data:`INDEX_DTYPE` or the
      columns of :func:`make_index`; row ``i`` describes cutout ``i``.
    - ``label_names``: the class names (any N >= 2, :func:`check_label_names`).
    - ``source``: the provenance, recorded as ``meta.json["source"]`` (JSON-serialisable:
      for example the fetch run, the label table and their SHA-256).
    - ``features``: optional float64 (N, 23), CD21's features of each alert
      (:data:`CD21_FEATURES`, module docstring), NaN where the alert lacks one.

    The rows keep the given order. The layout is written to
    a sibling ``.partial`` directory and renamed at the end, so an interrupted run never
    leaves a directory that looks complete. The writer is deterministic (no timestamps).

    Raises:
        ValueError: if an array or the index fails the checks of :func:`load_raw_alerts`
            (dtype, shape, label range, a duplicated ``candid``, an infinite feature, ...),
            the label names are invalid, or ``source`` is not a JSON object.
        DataError: if ``dest`` exists and ``overwrite`` is false, or ``dest`` exists and is
            not a prepared layout (it is never deleted then).
        OSError: if the destination cannot be written.
    """
    names = check_label_names(label_names)
    table = index if isinstance(index, np.ndarray) else make_index(index)
    array = np.asarray(cutouts)
    extra = None if features is None else np.asarray(features)
    problem = (
        _cutouts_problem(array, len(table))
        or _index_problem(table, len(names))
        or _features_problem(extra, len(table))
    )
    if problem:
        raise ValueError(problem)
    if not isinstance(source, Mapping):
        raise ValueError(f"source must be a mapping, got {type(source).__name__}")
    try:
        source_json = json.loads(json.dumps(dict(source)))  # before anything is written
    except (TypeError, ValueError) as err:
        raise ValueError(f"source must be JSON-serialisable ({err})") from None
    if dest.exists():
        if not overwrite:
            raise DataError(f"{dest} already exists; pass overwrite=True to replace it")
        if not is_prepared(dest):
            raise DataError(f"{dest} exists and is not a prepared layout; not replacing it")
    partial = dest.with_name(f".{dest.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    files: dict[str, Any] = {}
    arrays = [(CUTOUTS_NAME, array), (INDEX_NAME, table)]
    arrays += [] if extra is None else [(FEATURES_NAME, extra)]
    for name, values in arrays:
        file = partial / name
        np.save(file, np.ascontiguousarray(values), allow_pickle=False)
        files[name] = {
            "dtype": _dtype_text(values.dtype),
            "shape": list(values.shape),
            "sha256": sha256_file(file),
        }
    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "source": source_json,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "schema": _schema(names, with_features=extra is not None),
        "n": len(table),
        "class_counts": [int(c) for c in np.bincount(table["label"], minlength=len(names))],
        "files": files,
    }
    (partial / META_NAME).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest)
    partial.rename(dest)
    return dest


def label_names_of(path: Path) -> tuple[str, ...]:
    """The label names of the ``raw-alerts`` layout at ``path``, from ``meta.json`` only.

    Cheap (no array is opened): what a run reads to build an N-class classifier
    (``n_classes = len(label_names_of(path))``) before loading the data.

    Raises:
        DataError: if ``meta.json`` is missing, invalid, or not of a ``raw-alerts`` layout.
        SchemaError: if its label names are invalid.
    """
    return _label_names(path, _read_raw_meta(path))[0]


def load_raw_alerts(path: Path) -> RawAlerts:
    """Open the ``raw-alerts`` layout at ``path``; the cutouts are memory-mapped, read-only.

    Checks ``meta.json`` (format, version, kind, schema), every file against it (dtype and
    shape) and the index values: lengths, label range, a duplicated ``candid``, and the
    ranges of :data:`INDEX_DTYPE`, plus the class counts; ``features.npy``, when the schema
    lists feature names, must be float64 (N, 23) without inf. The cutout values are not
    checked (NaN and inf are legitimate); :func:`verify_raw_alerts` checks the hashes.

    Raises:
        DataError: if ``meta.json`` is missing or invalid, or not of this kind.
        SchemaError: if a file is missing, cannot be loaded without pickle, or does not
            match the schema, ``meta.json`` or the value rules.
    """
    meta = _read_raw_meta(path)
    names, with_features = _label_names(path, meta)
    expected = [CUTOUTS_NAME, INDEX_NAME, *([FEATURES_NAME] if with_features else [])]
    files = meta.get("files")
    if not isinstance(files, dict) or sorted(files) != sorted(expected):
        raise SchemaError(path, f"{META_NAME} must list exactly {expected}")
    cutouts = _load(path, CUTOUTS_NAME, files[CUTOUTS_NAME])
    index = np.array(_load(path, INDEX_NAME, files[INDEX_NAME]))  # small: read, then check
    features = _load(path, FEATURES_NAME, files[FEATURES_NAME]) if with_features else None
    problem = (
        _cutouts_problem(cutouts, len(index))
        or _index_problem(index, len(names))
        or _features_problem(features, len(index))
    )
    if problem:
        raise SchemaError(path, problem)
    counts = [int(c) for c in np.bincount(index["label"], minlength=len(names))]
    if meta.get("n") != len(index) or meta.get("class_counts") != counts:
        raise SchemaError(
            path,
            f"{META_NAME} says n={meta.get('n')!r}, class_counts={meta.get('class_counts')!r}; "
            f"the index has n={len(index)}, class_counts={counts}",
        )
    return RawAlerts(cutouts, index, names, features)


def verify_raw_alerts(path: Path) -> None:
    """Recompute the SHA-256 of every file of ``path`` and compare with ``meta.json``.

    Raises:
        DataError: if ``meta.json`` is invalid, a file is missing, or a hash differs.
    """
    meta = _read_raw_meta(path)
    files = meta.get("files")
    if not isinstance(files, dict):
        raise DataError(f"{path / META_NAME}: no files listed")
    for name, info in files.items():
        file = path / name
        if not file.is_file():
            raise DataError(f"{file}: missing")
        if not isinstance(info, dict) or sha256_file(file) != info.get("sha256"):
            raise DataError(f"{file}: SHA-256 differs from {META_NAME}; the file was modified")


def _read_raw_meta(path: Path) -> dict[str, Any]:
    meta = read_meta_file(path)
    if meta.get("kind") != KIND or meta.get("format_version") != FORMAT_VERSION:
        raise DataError(
            f"{path / META_NAME}: not a {KIND} layout (kind {meta.get('kind')!r}, "
            f"format version {meta.get('format_version')!r}; expected {KIND!r}, "
            f"{FORMAT_VERSION})"
        )
    return meta


def _schema(names: tuple[str, ...], *, with_features: bool) -> dict[str, Any]:
    return {
        "stamp_shape": list(STAMP_SHAPE),
        "channels": list(CHANNEL_NAMES),
        "label_names": list(names),
        "index_fields": [[name, INDEX_DTYPE[name].str] for name in INDEX_DTYPE.names or ()],
        "feature_names": list(CD21_FEATURES) if with_features else None,
    }


def _label_names(path: Path, meta: dict[str, Any]) -> tuple[tuple[str, ...], bool]:
    """The label names, and whether the layout has features, from a checked schema."""
    schema = meta.get("schema")
    raw_names = schema.get("label_names") if isinstance(schema, dict) else None
    if not isinstance(raw_names, list):
        raise SchemaError(path, f"{META_NAME} has no schema.label_names list")
    try:
        names = check_label_names(raw_names)
    except ValueError as err:
        raise SchemaError(path, f"{META_NAME}: {err}") from None
    with_features = isinstance(schema, dict) and schema.get("feature_names") is not None
    if schema != _schema(names, with_features=with_features):
        raise SchemaError(path, f"{META_NAME}: schema differs from this version's {KIND} schema")
    return names, with_features


def _dtype_text(dtype: np.dtype[Any]) -> str:
    # A structured dtype as its field list; a plain one as its name (``float32``).
    return str(dtype.descr) if dtype.names else str(dtype)


def _load(path: Path, name: str, info: Any) -> npt.NDArray[Any]:
    file = path / name
    try:
        array: npt.NDArray[Any] = np.load(file, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as err:
        raise SchemaError(path, f"cannot load {name} ({err})") from None
    recorded = (info.get("dtype"), info.get("shape")) if isinstance(info, dict) else (None, None)
    if recorded != (_dtype_text(array.dtype), list(array.shape)):
        raise SchemaError(
            path,
            f"{name} is {_dtype_text(array.dtype)} {list(array.shape)}, but {META_NAME} says "
            f"{recorded[0]} {recorded[1]}",
        )
    return array


def _cutouts_problem(cutouts: npt.NDArray[Any], n: int) -> str | None:
    """What is wrong with the cutouts array for an index of ``n`` rows, or ``None``."""
    expected = (n, *STAMP_SHAPE)
    if cutouts.dtype != _CUTOUT_DTYPE or cutouts.shape != expected:
        return f"cutouts must be float32 {expected}, got {cutouts.dtype} {cutouts.shape}"
    return None


def _features_problem(features: npt.NDArray[Any] | None, n: int) -> str | None:
    """What is wrong with the features array for an index of ``n`` rows, or ``None``."""
    if features is None:
        return None
    expected = (n, len(CD21_FEATURES))
    if features.dtype != _FEATURE_DTYPE or features.shape != expected:
        return f"features must be float64 {expected}, got {features.dtype} {features.shape}"
    infinite = np.isinf(features)
    if infinite.any():
        row, column = (int(i) for i in np.argwhere(infinite)[0])
        return f"features row {row} has an infinite {CD21_FEATURES[column]}"
    return None


def _index_problem(index: npt.NDArray[Any], n_classes: int) -> str | None:
    """The first rule of :data:`INDEX_DTYPE` that ``index`` breaks, or ``None``."""
    if index.dtype != INDEX_DTYPE or index.ndim != 1:
        return f"the index must be a 1-D array of {INDEX_DTYPE.descr}, got {index.dtype}"
    if len(index) == 0:
        return "the index is empty"
    checks: list[tuple[str, npt.NDArray[np.bool_]]] = [
        (
            f"an oid that does not match {OID_PATTERN.pattern}",
            np.fromiter(
                (OID_PATTERN.fullmatch(str(oid)) is None for oid in index["oid"]),
                dtype=bool,
                count=len(index),
            ),
        ),
        ("a candid <= 0", index["candid"] <= 0),
        ("a jd that is not finite and > 0", ~(np.isfinite(index["jd"]) & (index["jd"] > 0))),
        (f"a fid not in {ZTF_FIDS}", ~np.isin(index["fid"], ZTF_FIDS)),
        ("an ra outside [0, 360)", ~((index["ra"] >= 0) & (index["ra"] < 360))),
        ("a dec outside [-90, 90]", ~((index["dec"] >= -90) & (index["dec"] <= 90))),
        (
            f"a label outside 0..{n_classes - 1}",
            (index["label"] < 0) | (index["label"] >= n_classes),
        ),
        ("an empty label_source", index["label_source"] == ""),
        ("an empty source_dataset", index["source_dataset"] == ""),
        ("a source_row < 0", index["source_row"] < 0),
    ]
    for what, bad in checks:
        if bad.any():
            row = int(np.flatnonzero(bad)[0])
            return f"index row {row} has {what}: {index[row]!r}"
    candid = index["candid"]
    unique, counts = np.unique(candid, return_counts=True)
    if len(unique) != len(candid):
        dup = int(unique[counts > 1][0])
        rows = np.flatnonzero(candid == dup).tolist()
        return f"candid {dup} is duplicated (rows {rows})"
    return None
