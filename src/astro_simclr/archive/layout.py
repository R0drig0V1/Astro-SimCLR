"""The archive corpus layout: unlabelled raw cutouts, one shard directory per night.

The unlabelled sibling of the ``raw-alerts`` kind (:mod:`astro_simclr.data.raw_alerts`,
task 3.1b): the same float32 raw cutouts (science, reference, difference; NaN and inf kept)
and the same identity columns, without labels, and sharded by night so that an ingest can
be resumed night by night::

    OUT/
        corpus.json                 format, kind "archive-corpus", version, the selection
                                    rules and the exclusion list's SHA-256 (every night of
                                    the corpus has the same ones)
        nights/
            20190101/
                meta.json           kind "archive-night": the night, the source tarball
                                    (name, size, SHA-256, member counts), the selection,
                                    the exclusions, the per-reason counts, n, and the
                                    dtype, shape and SHA-256 of every file
                index.npy           structured (n,) (:data:`INDEX_DTYPE`)
                cutouts-00000.npy   float32 (<= rows_per_file, 63, 63, 3)
                cutouts-00001.npy   ...
            .20190102.partial/      a night being written (removed by the next ingest)

Row ``i`` of ``index.npy`` is row ``i`` of the concatenated cutout files, in file order.
The cutouts are split into files of at most ``rows_per_file`` rows so that the writer
holds one file in memory, not a whole night (a 2020 night may keep tens of thousands of
stamps, 47.6 KB each).

**Atomicity.** A night is written under ``nights/.<night>.partial``; every file is
fsynced, then the directory is renamed to ``nights/<night>`` and its parent fsynced. A
night directory therefore exists only once it is complete, which is what resume checks.
Replacing a night (``--overwrite``) moves the old one aside to ``.<night>.replaced``
first and removes it only after the new one is in place; :func:`recover_night` puts it
back if a crash came in between. ``corpus.json`` is written through a temporary file and
``os.replace``. One ingest at a time holds a corpus (:func:`corpus_lock`, an exclusive
``flock`` on ``OUT/.lock``), so two processes never write the same night.
The writer is deterministic (no timestamps), so the same tarball and rules give the same
bytes.

The ``format`` is not the prepared layout's, so the phase-2 readers refuse it. Nothing
here unpickles (``allow_pickle=False``), and the loader checks every file against
``meta.json`` before returning it.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.data.files import sha256_file
from astro_simclr.data.raw_alerts import OID_WIDTH
from astro_simclr.data.schema import STAMP_SHAPE, DataError, SchemaError

__all__ = [
    "CORPUS_INDEX_DTYPE",
    "CORPUS_KIND",
    "CORPUS_NAME",
    "FORMAT",
    "FORMAT_VERSION",
    "INDEX_DTYPE",
    "INDEX_NAME",
    "META_NAME",
    "NIGHTS_DIR",
    "NIGHT_KIND",
    "Night",
    "NightWriter",
    "check_corpus",
    "corpus_index",
    "corpus_lock",
    "excluded_rows",
    "iter_nights",
    "list_nights",
    "load_night",
    "night_dir",
    "read_night_meta",
    "recover_night",
    "verify_night",
]

#: ``format`` of ``corpus.json`` and of every night's ``meta.json``.
FORMAT: Final = "astro-simclr-archive"
#: ``format_version``; bumped on any change of the layout.
FORMAT_VERSION: Final = 1
CORPUS_KIND: Final = "archive-corpus"
NIGHT_KIND: Final = "archive-night"
CORPUS_NAME: Final = "corpus.json"
LOCK_NAME: Final = ".lock"
META_NAME: Final = "meta.json"
NIGHTS_DIR: Final = "nights"
INDEX_NAME: Final = "index.npy"
#: The cutout files of a night: ``cutouts-00000.npy``, ``cutouts-00001.npy``, ...
_CUTOUT_FILE: Final = re.compile(r"cutouts-(\d{5})\.npy")
_NIGHT: Final = re.compile(r"\d{8}")

#: One row of ``index.npy`` (little-endian, the same file on every machine):
#:
#: - ``oid``, ``candid``, ``jd``, ``fid``, ``ra``, ``dec``: as in the ``raw-alerts`` index
#:   (same widths and ranges);
#: - ``night``: the tarball's night, ``YYYYMMDD``;
#: - ``ndethist``: the candidate's ``ndethist`` (>= 0; the first-detection rule read it);
#: - ``isdiffpos``: true for a positive (science minus reference) subtraction;
#: - ``rb``, ``drb``: the real-bogus scores, NaN when the alert lacks them (``drb`` is
#:   absent before ZTF's schema 3.3), so later cuts can be applied without re-ingesting;
#: - ``member``: the alert's position among the tarball's members, for tracing.
INDEX_DTYPE: Final = np.dtype(
    [
        ("oid", f"<U{OID_WIDTH}"),
        ("candid", "<i8"),
        ("jd", "<f8"),
        ("fid", "<i8"),
        ("ra", "<f8"),
        ("dec", "<f8"),
        ("night", "<i8"),
        ("ndethist", "<i8"),
        ("isdiffpos", "?"),
        ("rb", "<f8"),
        ("drb", "<f8"),
        ("member", "<i8"),
    ]
)
_CUTOUT_DTYPE: Final = np.dtype("<f4")
#: One row of :func:`corpus_index`: where the alert is (night, row in that night) and who.
CORPUS_INDEX_DTYPE: Final = np.dtype(
    [("night", "<i8"), ("row", "<i8"), ("oid", f"<U{OID_WIDTH}"), ("candid", "<i8"), ("jd", "<f8")]
)


def night_dir(out: Path, night: int) -> Path:
    """The directory of ``night`` in the corpus at ``out``."""
    return out / NIGHTS_DIR / f"{night:08d}"


@contextmanager
def corpus_lock(out: Path) -> Iterator[None]:
    """Hold the corpus at ``out`` for one ingest (created if needed).

    Raises:
        DataError: if another process (or another open of the lock) holds it.
        OSError: if ``out`` cannot be created.
    """
    out.mkdir(parents=True, exist_ok=True)
    with (out / LOCK_NAME).open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DataError(f"{out}: another ingest is writing this corpus") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def recover_night(out: Path, night: int) -> None:
    """Put back a night that a crash left aside while it was being replaced.

    If ``nights/<night>`` is missing and ``.<night>.replaced`` holds a complete night, the
    replacement never landed: the old night is renamed back (the new one is redone).
    """
    final = night_dir(out, night)
    aside = final.with_name(f".{final.name}.replaced")
    if not final.exists() and (aside / META_NAME).is_file():
        aside.rename(final)
        _fsync_dir(final.parent)


def check_corpus(out: Path, record: Mapping[str, Any]) -> None:
    """Create ``out/corpus.json`` with ``record``, or check that it holds the same one.

    ``record`` is the corpus-wide part of an ingest (the selection rules and the
    exclusions); every night of a corpus must share it, so a resume with other rules or
    another exclusion list is refused rather than mixed in.

    Raises:
        DataError: if ``corpus.json`` exists and is unreadable, of another format, or
            records other rules.
        OSError: if ``out`` cannot be written.
    """
    file = out / CORPUS_NAME
    wanted = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "kind": CORPUS_KIND,
        **json.loads(json.dumps(dict(record))),
    }
    if not file.exists():
        out.mkdir(parents=True, exist_ok=True)
        _write_json(file, wanted)
        return
    found = _read_json(file)
    if found != wanted:
        changed = sorted(k for k in set(found) | set(wanted) if found.get(k) != wanted.get(k))
        raise DataError(
            f"{file}: this corpus was ingested with other {', '.join(changed)}; "
            "use another --out (or remove the corpus) to ingest with these rules"
        )


@dataclass
class NightWriter:
    """Write one night under its ``.partial`` directory, then commit it (module docstring).

    Add rows with :meth:`add`, then :meth:`commit` with the night's metadata. If the
    writer is dropped without a commit, the partial directory stays and the next writer
    of the night removes it.
    """

    out: Path
    night: int
    rows_per_file: int = 4096
    _rows: list[tuple[Any, ...]] = field(default_factory=list, init=False)
    _buffer: list[npt.NDArray[np.float32]] = field(default_factory=list, init=False)
    _files: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.rows_per_file < 1:
            raise ValueError(f"rows_per_file must be >= 1, got {self.rows_per_file}")
        self.final = night_dir(self.out, self.night)
        self.partial = self.final.with_name(f".{self.final.name}.partial")
        if self.partial.exists():
            shutil.rmtree(self.partial)
        self.partial.mkdir(parents=True)

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, row: tuple[Any, ...], cutouts: npt.NDArray[np.float32]) -> None:
        """Append one row (the fields of :data:`INDEX_DTYPE`, in order) and its cutouts."""
        if cutouts.dtype != _CUTOUT_DTYPE or cutouts.shape != STAMP_SHAPE:
            raise ValueError(f"cutouts must be float32 {STAMP_SHAPE}, got {cutouts.shape}")
        self._rows.append(row)
        self._buffer.append(cutouts)
        if len(self._buffer) == self.rows_per_file:
            self._flush()

    def commit(self, meta: Mapping[str, Any]) -> Path:
        """Write the index and ``meta.json``, then rename the night into place.

        ``meta`` is the night's record (source, selection, exclusions, counts); the
        format keys, ``n`` and ``files`` are added here. An existing complete night is
        replaced (the caller decides whether that is allowed).

        Raises:
            OSError: if a file cannot be written.
        """
        self._flush()
        index = np.array(self._rows, dtype=INDEX_DTYPE)
        self._save(INDEX_NAME, index)
        record = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "kind": NIGHT_KIND,
            "night": self.night,
            **json.loads(json.dumps(dict(meta))),
            "created_by": f"astro-simclr {version('astro-simclr')}",
            "n": len(index),
            "files": self._files,
        }
        _write_json(self.partial / META_NAME, record)
        _fsync_dir(self.partial)
        aside = self.final.with_name(f".{self.final.name}.replaced")
        if aside.exists():
            shutil.rmtree(aside)
        if self.final.exists():  # replaced: set aside until the new night is in place
            self.final.rename(aside)
        self.partial.rename(self.final)
        _fsync_dir(self.final.parent)
        if aside.exists():
            shutil.rmtree(aside)
        return self.final

    def _flush(self) -> None:
        if not self._buffer:
            return
        name = f"cutouts-{sum(1 for n in self._files if n != INDEX_NAME):05d}.npy"
        self._save(name, np.stack(self._buffer))
        self._buffer = []

    def _save(self, name: str, values: npt.NDArray[Any]) -> None:
        file = self.partial / name
        with file.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        self._files[name] = {
            "dtype": _dtype_text(values.dtype),
            "shape": list(values.shape),
            "sha256": sha256_file(file),
        }


@dataclass(frozen=True, slots=True)
class Night:
    """One loaded night.

    Attributes:
        path: its directory.
        meta: its ``meta.json``.
        index: structured (n,) of :data:`INDEX_DTYPE`.
        cutouts: the cutout files, memory-mapped read-only, in order; together (n, 63,
            63, 3) float32.
    """

    path: Path
    meta: dict[str, Any]
    index: npt.NDArray[np.void]
    cutouts: tuple[npt.NDArray[np.float32], ...]

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def cutout(self, row: int) -> npt.NDArray[np.float32]:
        """The (63, 63, 3) cutouts of ``row`` (a view into the memory map)."""
        if not 0 <= row < len(self):
            raise IndexError(f"row {row} outside 0..{len(self) - 1}")
        for chunk in self.cutouts:
            if row < len(chunk):
                stamp: npt.NDArray[np.float32] = chunk[row]
                return stamp
            row -= len(chunk)
        raise IndexError(row)  # pragma: no cover - the loader checked the lengths


def list_nights(out: Path) -> list[int]:
    """The complete nights of the corpus at ``out``, sorted (partial ones are not listed)."""
    nights = out / NIGHTS_DIR
    if not nights.is_dir():
        return []
    return sorted(
        int(p.name)
        for p in nights.iterdir()
        if p.is_dir() and _NIGHT.fullmatch(p.name) and (p / META_NAME).is_file()
    )


def load_night(path: Path) -> Night:
    """Open the night at ``path``; its cutouts are memory-mapped read-only.

    Checks ``meta.json`` (format, version, kind, the night matching the directory), that
    it lists exactly the index and consecutive cutout files, every file's dtype and shape
    against it, that the cutout rows add up to the index, and the index's values (oid
    width, candid > 0 and unique, fid, ra, dec, the night). :func:`verify_night` checks
    the hashes.

    Raises:
        DataError: if ``meta.json`` is missing or of another kind.
        SchemaError: if a file is missing, cannot be loaded without pickle, or disagrees
            with ``meta.json`` or the value rules.
    """
    meta = _read_night_meta(path)
    files = meta.get("files")
    if not isinstance(files, dict) or INDEX_NAME not in files:
        raise SchemaError(path, f"{META_NAME} lists no {INDEX_NAME}")
    chunks = sorted(name for name in files if name != INDEX_NAME)
    expected = [f"cutouts-{i:05d}.npy" for i in range(len(chunks))]
    if chunks != expected:
        raise SchemaError(path, f"{META_NAME} must list the cutout files {expected}, got {chunks}")
    index = np.array(_load(path, INDEX_NAME, files[INDEX_NAME]))
    if index.dtype != INDEX_DTYPE or index.ndim != 1:
        raise SchemaError(path, f"{INDEX_NAME} is {index.dtype}, expected {INDEX_DTYPE.descr}")
    cutouts = tuple(_load(path, name, files[name]) for name in chunks)
    for name, chunk in zip(chunks, cutouts, strict=True):
        if chunk.dtype != _CUTOUT_DTYPE or chunk.shape[1:] != STAMP_SHAPE:
            raise SchemaError(path, f"{name} must be float32 (k, 63, 63, 3), got {chunk.shape}")
    total = sum(len(chunk) for chunk in cutouts)
    if total != len(index) or meta.get("n") != len(index):
        raise SchemaError(
            path, f"{len(index)} index rows, {total} cutout rows, meta n={meta.get('n')!r}"
        )
    problem = _index_problem(index, int(meta["night"]))
    if problem:
        raise SchemaError(path, problem)
    return Night(path, meta, index, cutouts)


def verify_night(path: Path) -> None:
    """Recompute the SHA-256 of every file of the night at ``path`` against ``meta.json``.

    Raises:
        DataError: if ``meta.json`` is invalid, a file is missing, or a hash differs.
    """
    meta = _read_night_meta(path)
    files = meta.get("files")
    if not isinstance(files, dict):
        raise DataError(f"{path / META_NAME}: no files listed")
    for name, info in files.items():
        file = path / name
        if not file.is_file():
            raise DataError(f"{file}: missing")
        if not isinstance(info, dict) or sha256_file(file) != info.get("sha256"):
            raise DataError(f"{file}: SHA-256 differs from {META_NAME}; the file was modified")


def excluded_rows(index: npt.NDArray[np.void], oids: frozenset[str]) -> npt.NDArray[np.bool_]:
    """A mask of the rows of ``index`` (any table with an ``oid`` field) whose oid is in ``oids``.

    For an exclusion list that grew after a night was ingested (a new benchmark): the
    reader drops those rows instead of re-ingesting the tarball.
    """
    if not oids:
        return np.zeros(len(index), dtype=bool)
    return np.isin(index["oid"], np.array(sorted(oids), dtype=index.dtype["oid"]))


def corpus_index(out: Path, *, exclude: frozenset[str] = frozenset()) -> npt.NDArray[np.void]:
    """One alert per object over every complete night of the corpus at ``out``.

    Each night keeps one alert per ``oid`` already; across nights an object can appear
    again (an alert more than 30 days after its previous one, which its history does not
    show). This keeps the earliest alert of each ``oid`` (by ``jd``, then ``candid``) and
    drops every ``oid`` in ``exclude`` (an exclusion list that grew after the ingest).
    Rows are sorted by night and row (:data:`CORPUS_INDEX_DTYPE`), the order to read the
    cutouts in. The phase-5 corpus index is built from it.

    Raises:
        DataError: if a night is invalid (:func:`load_night`).
    """
    parts = []
    for night in list_nights(out):
        index = load_night(night_dir(out, night)).index
        part = np.empty(len(index), dtype=CORPUS_INDEX_DTYPE)
        part["night"], part["row"] = night, np.arange(len(index))
        for name in ("oid", "candid", "jd"):
            part[name] = index[name]
        parts.append(part)
    if not parts:
        return np.empty(0, dtype=CORPUS_INDEX_DTYPE)
    rows = np.concatenate(parts)
    order = np.lexsort((rows["candid"], rows["jd"], rows["oid"]))
    rows = rows[order]
    _, first = np.unique(rows["oid"], return_index=True)
    rows = rows[first]
    rows = rows[~excluded_rows(rows, exclude)]
    return rows[np.lexsort((rows["row"], rows["night"]))]


def read_night_meta(path: Path) -> dict[str, Any]:
    """The checked ``meta.json`` of the night at ``path`` (see :func:`load_night`)."""
    return _read_night_meta(path)


def iter_nights(out: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(night, meta)`` of every complete night of the corpus at ``out``, in order."""
    for night in list_nights(out):
        yield night, _read_night_meta(night_dir(out, night))


def _read_night_meta(path: Path) -> dict[str, Any]:
    meta = _read_json(path / META_NAME)
    if (
        meta.get("format") != FORMAT
        or meta.get("format_version") != FORMAT_VERSION
        or meta.get("kind") != NIGHT_KIND
    ):
        raise DataError(f"{path / META_NAME}: not an {NIGHT_KIND} (format {FORMAT_VERSION})")
    if not isinstance(meta.get("night"), int) or f"{meta['night']:08d}" != path.name:
        raise DataError(f"{path / META_NAME}: night {meta.get('night')!r} is not {path.name}")
    return meta


def _read_json(file: Path) -> dict[str, Any]:
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DataError(f"{file}: missing") from None
    except (OSError, ValueError) as err:
        raise DataError(f"{file}: unreadable ({err})") from None
    if not isinstance(value, dict):
        raise DataError(f"{file}: not a JSON object")
    return value


def _write_json(file: Path, value: Mapping[str, Any]) -> None:
    """Write ``value`` atomically: a temporary file, fsync, ``os.replace``, fsync the dir."""
    tmp = file.with_name(file.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, file)
    _fsync_dir(file.parent)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _dtype_text(dtype: np.dtype[Any]) -> str:
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
            f"{name} is {_dtype_text(array.dtype)} {list(array.shape)}, but {META_NAME} "
            f"says {recorded[0]} {recorded[1]}",
        )
    return array


def _index_problem(index: npt.NDArray[Any], night: int) -> str | None:
    """The first value rule ``index`` breaks, or ``None`` (an empty night is valid)."""
    if len(index) == 0:
        return None
    checks: list[tuple[str, npt.NDArray[np.bool_]]] = [
        ("an empty oid", index["oid"] == ""),
        ("a candid <= 0", index["candid"] <= 0),
        ("a jd that is not finite and > 0", ~(np.isfinite(index["jd"]) & (index["jd"] > 0))),
        ("a fid not in (1, 2, 3)", ~np.isin(index["fid"], (1, 2, 3))),
        ("an ra outside [0, 360)", ~((index["ra"] >= 0) & (index["ra"] < 360))),
        ("a dec outside [-90, 90]", ~((index["dec"] >= -90) & (index["dec"] <= 90))),
        (f"a night other than {night}", index["night"] != night),
        ("an ndethist < 0", index["ndethist"] < 0),
    ]
    for what, bad in checks:
        if bad.any():
            return f"index row {int(np.flatnonzero(bad)[0])} has {what}"
    if len(np.unique(index["candid"])) != len(index):
        return "a candid is duplicated"
    return None
