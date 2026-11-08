"""The raw alert store of ``fetch-alerts``: index, shards, snapshot and summary (task 3.1).

Layout (``docs/data.md``, "Raw alerts")::

    OUT/<split>/
        fetch.json          format, version, the prepared dataset (name, source SHA-256),
                            the split, its size and class counts, endpoints, tolerances
        index.csv           one line per attempt (append-only; the last line of a row wins)
        shard-00000.npz     one committed batch of rows (np.savez_compressed):
                              rows        int64   (n,)          split rows of the ok alerts
                              candid      int64   (n,)
                              cutouts     float32 (n, 63, 63, 3) science, template,
                                                                 difference; NaN kept
                              candidates  uint8   UTF-8 JSON list of the n candidate records
                              responses   uint8   UTF-8 JSON list of the raw responses
                                                  fetched by the batch: url, kind,
                                                  retrieved_at, sha256, offset, size
                              bodies      uint8   the responses' bytes, concatenated
        shard-00001.npz ...

Only ``ok`` alerts (stamp check passed) have cutouts in a shard, so no stored row fails
the check. Every index line, whatever its status, names the shard of its batch and lists
the responses it used (``kind:sha256``); a response served from the cache was stored by an
earlier line, so the snapshot of the whole directory holds every byte the index was
decided on (:class:`Snapshot`, ``fetch-alerts --from-snapshot``).

**Why this format.** The run appends for a day or two, gets interrupted, and resumes, so
nothing is ever rewritten. A batch is committed as one shard, written to a temporary file,
fsynced and renamed, the directory fsynced, and only then are its index lines appended
(flushed and fsynced): an index
line never points to a missing shard. A crash between the two leaves an orphan shard that
no line references; its rows are fetched again. A truncated last index line is dropped.
CSV needs no dependency and opens anywhere (Parquet would need pyarrow, over 100 MB, for
52k small rows), and one ``.npz`` per batch keeps the file count low (about 500 files for
``red``, not 200k response files) and loads with ``allow_pickle=False``: the records travel
as JSON text in uint8 arrays, never as Python objects.

Statuses (:data:`STATUSES`): ``ok`` (stored); ``mismatch`` (downloaded, no candidate passes
the check); ``ghost_candidate`` (no ALeRCE object within the radius: an object outside
ALeRCE's database, reported apart and left out of the recovery target); ``no_object``
(objects in the cone, none with a detection) and ``no_detection_match`` (not found);
``ambiguous`` (more than
one candidate passes, or too many to download); ``bad_alert`` (no candidate is a readable
alert); ``http_error`` (a timeout or server failure; retried by the next ``--resume``).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.alerts.parse import Scalar
from astro_simclr.alerts.snapshot import Response
from astro_simclr.alerts.transport import HttpError
from astro_simclr.data.schema import LABEL_NAMES, STAMP_SHAPE, DataError

__all__ = [
    "COLUMNS",
    "FORMAT",
    "FORMAT_VERSION",
    "GROUPS",
    "INDEX_NAME",
    "META_NAME",
    "MIN_RECOVERY",
    "RETRIED",
    "STATUSES",
    "IndexRow",
    "IndexWriter",
    "Recovered",
    "Shard",
    "Snapshot",
    "Summary",
    "duplicates",
    "format_duplicates",
    "format_summary",
    "latest_rows",
    "load_recovered",
    "next_shard_id",
    "read_index",
    "read_meta",
    "read_shard",
    "shard_path",
    "summarize",
    "write_meta",
    "write_shard",
]

#: ``fetch.json["format"]`` and its version (bumped on any change of the layout).
FORMAT = "astro-simclr-raw-alerts"
FORMAT_VERSION = 1
INDEX_NAME = "index.csv"
META_NAME = "fetch.json"

#: Every status, in report order.
STATUSES: tuple[str, ...] = (
    "ok",
    "mismatch",
    "ghost_candidate",
    "no_object",
    "no_detection_match",
    "ambiguous",
    "bad_alert",
    "http_error",
)
#: Statuses that ``--resume`` attempts again.
RETRIED: frozenset[str] = frozenset({"http_error"})
#: The acceptance's miss classes (plan-review amendment 3) -> their statuses.
GROUPS: dict[str, tuple[str, ...]] = {
    "recovered": ("ok",),
    "ghost": ("ghost_candidate",),
    "not_found": ("no_object", "no_detection_match"),
    "ambiguous": ("ambiguous",),
    "timeout": ("http_error",),
    "mismatch": ("mismatch", "bad_alert"),
}
#: Acceptance: the fraction of each class's non-ghost rows that must be ``ok``.
MIN_RECOVERY = 0.98
#: The full stamp side, as ``raw_shape`` writes it.
_FULL_SHAPE = f"{STAMP_SHAPE[0]}x{STAMP_SHAPE[1]}"

_SHARD_KEYS = ("rows", "candid", "cutouts", "candidates", "responses", "bodies")


@dataclass(frozen=True, slots=True)
class IndexRow:
    """One line of ``index.csv``: one attempt at one split row.

    ``target_*`` are the de-standardised values the search used. ``n_objects`` objects were
    in the cone and ``n_matches`` distinct candids matched the photometry. ``candidates``
    lists every matching ``oid:candid`` (space-separated). The candidate fields (``oid``
    to ``dec``) come from the chosen alert's own record: the ``ok`` one, or for a
    ``mismatch`` the closest one. ``max_abs_diff`` is that alert's
    ``max |minmax(raw) - stored|``; ``nan_pixels`` counts its raw NaNs over the three
    channels (padding excluded); ``raw_shape`` is ``HxW`` before a CCD-edge repair.
    ``shard`` is the batch the line was committed with; ``responses`` lists the raw
    responses it used (``kind:sha256``, space-separated). ``requests`` and ``seconds``
    are this attempt's cost, rate-limit waits included.
    """

    row: int
    label: int
    status: str
    target_ra: float
    target_dec: float
    target_magpsf: float
    target_sigmapsf: float
    n_objects: int | None = None
    n_matches: int | None = None
    candidates: str = ""
    oid: str = ""
    candid: int | None = None
    jd: float | None = None
    fid: int | None = None
    magpsf: float | None = None
    sigmapsf: float | None = None
    isdiffpos: str = ""
    rb: float | None = None
    drb: float | None = None
    ra: float | None = None
    dec: float | None = None
    raw_shape: str = ""
    nan_pixels: int | None = None
    max_abs_diff: float | None = None
    shard: int | None = None
    responses: str = ""
    requests: int = 0
    seconds: float = 0.0
    note: str = ""


#: Column name -> parser of its non-empty text (field types of :class:`IndexRow`).
_PARSERS: dict[str, type] = {
    f.name: {"int": int, "float": float, "str": str}[str(f.type).split(" | ", maxsplit=1)[0]]
    for f in fields(IndexRow)
}
COLUMNS: tuple[str, ...] = tuple(_PARSERS)


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    # One record per physical line, so a truncated tail is always a whole line.
    return str(value).replace("\r", " ").replace("\n", " ")


def _fsync_dir(path: Path) -> None:
    """Make a rename in ``path`` durable (POSIX; a no-op where directories cannot be opened)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _parse_row(record: dict[str, str]) -> IndexRow:
    values: dict[str, Any] = {}
    for name, parser in _PARSERS.items():
        text = record.get(name)
        if text is None:
            raise ValueError(f"missing column {name}")
        values[name] = text if parser is str else (None if text == "" else parser(text))
    row = IndexRow(**values)
    if row.status not in STATUSES:
        raise ValueError(f"unknown status {row.status!r}")
    return row


def read_index(split_dir: Path) -> list[IndexRow]:
    """Every line of ``split_dir/index.csv``, in file order (empty if there is none).

    A last line without its newline (an interrupted write) is ignored.

    Raises:
        DataError: if the header or a complete line is invalid.
    """
    path = split_dir / INDEX_NAME
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if text and tuple(reader.fieldnames or ()) != COLUMNS:
        raise DataError(f"{path}: unexpected columns {reader.fieldnames}")
    rows: list[IndexRow] = []
    for number, record in enumerate(reader, start=2):
        try:
            rows.append(_parse_row(record))
        except (ValueError, TypeError) as err:
            raise DataError(f"{path}: line {number}: {err}") from None
    return rows


def latest_rows(rows: Iterable[IndexRow]) -> dict[int, IndexRow]:
    """Split row -> its last index line (later attempts replace earlier ones)."""
    return {line.row: line for line in rows}


class IndexWriter:
    """Appends lines to ``split_dir/index.csv``, writing the header on a new file.

    On open, a last line without its newline (an interrupted write) is cut off, so the
    file stays a valid CSV. Every :meth:`append` is flushed and fsynced.
    """

    def __init__(self, split_dir: Path) -> None:
        self.path = split_dir / INDEX_NAME
        if self.path.is_file():
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):
                with self.path.open("r+b") as handle:
                    handle.truncate(data.rfind(b"\n") + 1)
        new = not self.path.is_file() or self.path.stat().st_size == 0
        self._new = new
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        if new:
            self._writer.writerow(COLUMNS)
            self._sync()

    def append(self, rows: Sequence[IndexRow]) -> None:
        """Append ``rows`` and make them durable."""
        for line in rows:
            values = asdict(line)
            self._writer.writerow([_format(values[name]) for name in COLUMNS])
        self._sync()

    def close(self) -> None:
        """Close the file."""
        self._handle.close()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        if self._new:
            _fsync_dir(self.path.parent)
            self._new = False


def write_meta(split_dir: Path, meta: dict[str, Any]) -> None:
    """Write ``split_dir/fetch.json`` (``format``, ``format_version``, ``created_by`` added)."""
    full = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        **meta,
        "created_by": f"astro-simclr {version('astro-simclr')}",
    }
    _atomic_write(split_dir / META_NAME, (json.dumps(full, indent=2) + "\n").encode("utf-8"))


def read_meta(split_dir: Path) -> dict[str, Any]:
    """Read and check ``split_dir/fetch.json``.

    Raises:
        DataError: if it is missing, not JSON, or of another format or version.
    """
    path = split_dir / META_NAME
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise DataError(f"{path}: cannot read ({err.strerror or err})") from None
    except json.JSONDecodeError as err:
        raise DataError(f"{path}: invalid JSON ({err})") from None
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise DataError(f"{path}: not an {FORMAT} file")
    if meta.get("format_version") != FORMAT_VERSION:
        raise DataError(f"{path}: format version {meta.get('format_version')!r}")
    return meta


@dataclass(frozen=True, slots=True)
class Shard:
    """The content of one shard file (module docstring)."""

    rows: npt.NDArray[np.int64]
    candid: npt.NDArray[np.int64]
    cutouts: npt.NDArray[np.float32]
    candidates: list[dict[str, Scalar]]
    responses: list[Response]


def shard_path(split_dir: Path, shard_id: int) -> Path:
    """``split_dir/shard-NNNNN.npz``."""
    return split_dir / f"shard-{shard_id:05d}.npz"


def next_shard_id(split_dir: Path) -> int:
    """One more than the largest shard id on disk (orphans included), or 0."""
    ids = [int(p.stem.removeprefix("shard-")) for p in split_dir.glob("shard-[0-9]*.npz")]
    return max(ids, default=-1) + 1


def _json_bytes(value: object) -> npt.NDArray[np.uint8]:
    return np.frombuffer(json.dumps(value).encode("utf-8"), dtype=np.uint8)


def write_shard(split_dir: Path, shard_id: int, shard: Shard) -> Path:
    """Write ``shard`` as shard ``shard_id`` atomically (temporary file, fsync, rename).

    Raises:
        ValueError: if the arrays disagree in length or have the wrong dtype or shape.
        OSError: if the file cannot be written.
    """
    _check_arrays(shard.rows, shard.candid, shard.cutouts, len(shard.candidates), "shard")
    records, offset = [], 0
    for r in shard.responses:
        records.append(
            {
                "url": r.url,
                "kind": r.kind,
                "retrieved_at": r.retrieved_at,
                "sha256": r.sha256,
                "offset": offset,
                "size": len(r.body),
            }
        )
        offset += len(r.body)
    bodies = np.frombuffer(b"".join(r.body for r in shard.responses), dtype=np.uint8)
    path = shard_path(split_dir, shard_id)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            rows=shard.rows,
            candid=shard.candid,
            cutouts=shard.cutouts,
            candidates=_json_bytes(shard.candidates),
            responses=_json_bytes(records),
            bodies=bodies,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(split_dir)  # the rename must be durable before the index lines name it
    return path


def _load_npz(path: Path, keys: Sequence[str]) -> dict[str, npt.NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as npz:
            if sorted(npz.files) != sorted(_SHARD_KEYS):
                raise DataError(f"{path}: keys {sorted(npz.files)}, expected {sorted(_SHARD_KEYS)}")
            return {key: npz[key] for key in keys}
    except DataError:
        raise
    except (OSError, ValueError) as err:
        raise DataError(f"{path}: cannot load ({err})") from None


def _json_list(array: npt.NDArray[Any], name: str) -> list[dict[str, Any]]:
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError(f"{name} must be uint8 (bytes,)")
    value = json.loads(array.tobytes().decode("utf-8"))
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"{name} must be a JSON list of objects")
    return value


def _response_records(array: npt.NDArray[Any], n_bodies: int) -> list[dict[str, Any]]:
    records = _json_list(array, "responses")
    for record in records:
        offset, size = record.get("offset"), record.get("size")
        if (
            not all(
                isinstance(record.get(k), str) for k in ("url", "kind", "retrieved_at", "sha256")
            )
            or not isinstance(offset, int)
            or not isinstance(size, int)
            or offset < 0
            or size < 0
            or offset + size > n_bodies
        ):
            raise ValueError(f"invalid response record {record!r}")
    return records


def read_shard(path: Path) -> Shard:
    """Load a shard without pickle and check it (the bodies included).

    Raises:
        DataError: if the file cannot be loaded or does not match the layout.
    """
    arrays = _load_npz(path, _SHARD_KEYS)
    try:
        candidates = _json_list(arrays["candidates"], "candidates")
        _check_arrays(
            arrays["rows"], arrays["candid"], arrays["cutouts"], len(candidates), str(path)
        )
        bodies = arrays["bodies"]
        if bodies.dtype != np.uint8 or bodies.ndim != 1:
            raise ValueError("bodies must be uint8 (bytes,)")
        responses = [
            Response(
                r["url"],
                r["kind"],
                r["retrieved_at"],
                r["sha256"],
                bodies[r["offset"] : r["offset"] + r["size"]].tobytes(),
            )
            for r in _response_records(arrays["responses"], len(bodies))
        ]
    except (ValueError, UnicodeDecodeError) as err:
        raise DataError(f"{path}: {err}") from None
    return Shard(arrays["rows"], arrays["candid"], arrays["cutouts"], candidates, responses)


def _check_arrays(
    rows: npt.NDArray[Any],
    candid: npt.NDArray[Any],
    cutouts: npt.NDArray[Any],
    n_c: int,
    where: str,
) -> None:
    n = len(rows)
    expected = (
        ("rows", rows, np.int64, (n,)),
        ("candid", candid, np.int64, (n,)),
        ("cutouts", cutouts, np.float32, (n, *STAMP_SHAPE)),
    )
    for name, array, dtype, shape in expected:
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(
                f"{where}: {name} is {array.dtype} {array.shape}, expected "
                f"{np.dtype(dtype)} {shape}"
            )
    if n_c != n:
        raise ValueError(f"{where}: {n_c} candidates for {n} rows")


class Snapshot:
    """The raw responses of a store, served by URL: the offline source of a rebuild.

    ``Snapshot(split_dir).fetch`` is a :data:`~astro_simclr.alerts.snapshot.RawFetch`:
    it returns the stored body and its original retrieval time, after checking the
    body's SHA-256. Every shard on disk is indexed, orphans included (their responses are
    genuine). A URL fetched more than once is served from its latest retrieval.

    Raises:
        DataError: if a shard cannot be loaded.
    """

    def __init__(self, split_dir: Path) -> None:
        self.split_dir = split_dir
        self._where: dict[str, tuple[int, dict[str, Any]]] = {}
        ids = sorted(int(p.stem.removeprefix("shard-")) for p in split_dir.glob("shard-[0-9]*.npz"))
        for shard_id in ids:
            path = shard_path(split_dir, shard_id)
            arrays = _load_npz(path, ("responses", "bodies"))
            try:
                records = _response_records(arrays["responses"], len(arrays["bodies"]))
            except (ValueError, UnicodeDecodeError) as err:
                raise DataError(f"{path}: {err}") from None
            for record in records:
                self._where[record["url"]] = (shard_id, record)
        self._bodies: tuple[int, npt.NDArray[np.uint8]] | None = None

    def __len__(self) -> int:
        """The number of distinct URLs in the snapshot."""
        return len(self._where)

    def __contains__(self, url: object) -> bool:
        """True if the snapshot holds a response for ``url``."""
        return url in self._where

    def iter_many(self, urls: Iterable[str]) -> Iterator[tuple[str, bytes, str, str]]:
        """(url, body, retrieval time, SHA-256) for the ``urls`` the snapshot holds.

        The URLs are grouped by shard and each shard's bodies are loaded once, so a bulk
        read of thousands of responses is cheap and holds one shard in memory at a time;
        URLs not in the snapshot are skipped. The order is by shard, then as given.

        Raises:
            DataError: if a shard cannot be loaded or a body does not match its SHA-256.
        """
        by_shard: dict[int, list[str]] = {}
        for url in dict.fromkeys(urls):
            if url in self._where:
                by_shard.setdefault(self._where[url][0], []).append(url)
        for shard_id, shard_urls in sorted(by_shard.items()):
            path = shard_path(self.split_dir, shard_id)
            bodies = _load_npz(path, ("bodies",))["bodies"]
            for url in shard_urls:
                record = self._where[url][1]
                body = bodies[record["offset"] : record["offset"] + record["size"]].tobytes()
                if hashlib.sha256(body).hexdigest() != record["sha256"]:
                    raise DataError(f"{path}: {url}: SHA-256 differs")
                yield url, body, str(record["retrieved_at"]), str(record["sha256"])

    def read_many(self, urls: Iterable[str]) -> dict[str, tuple[bytes, str, str]]:
        """URL -> (body, retrieval time, SHA-256): :meth:`iter_many` as a dict."""
        return {url: (body, at, sha) for url, body, at, sha in self.iter_many(urls)}

    def fetch(self, url: str, max_bytes: int) -> tuple[bytes, str]:
        """The stored body of ``url`` and its retrieval time.

        Raises:
            HttpError: if ``url`` is not in the snapshot or its body is larger than
                ``max_bytes`` (recorded as ``http_error`` by a rebuild).
            DataError: if the stored body does not match its SHA-256.
        """
        if url not in self._where:
            raise HttpError(url, None, "not in the snapshot")
        shard_id, record = self._where[url]
        if self._bodies is None or self._bodies[0] != shard_id:
            self._bodies = (
                shard_id,
                _load_npz(shard_path(self.split_dir, shard_id), ("bodies",))["bodies"],
            )
        body = self._bodies[1][record["offset"] : record["offset"] + record["size"]].tobytes()
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise DataError(f"{shard_path(self.split_dir, shard_id)}: {url}: SHA-256 differs")
        if len(body) > max_bytes:
            raise HttpError(url, None, f"response larger than {max_bytes} bytes")
        return body, str(record["retrieved_at"])


@dataclass(frozen=True, slots=True)
class Recovered:
    """The ``ok`` alerts of a split, sorted by split row (:func:`load_recovered`)."""

    rows: npt.NDArray[np.int64]
    candid: npt.NDArray[np.int64]
    cutouts: npt.NDArray[np.float32]
    candidates: list[dict[str, Scalar]]


def load_recovered(split_dir: Path) -> Recovered:
    """Load every ``ok`` alert that the index points to.

    Each row's last index line names its shard; orphan shards are ignored. Holds the
    cutouts in RAM (about 48 KB per alert).

    Raises:
        DataError: if the index or a shard is invalid, or a row is missing from its shard.
    """
    by_shard: dict[int, list[int]] = {}
    for line in latest_rows(read_index(split_dir)).values():
        if line.status != "ok":
            continue
        if line.shard is None:
            raise DataError(f"{split_dir}: row {line.row} is ok without a shard")
        by_shard.setdefault(line.shard, []).append(line.row)
    rows: list[int] = []
    candid: list[int] = []
    cutouts: list[npt.NDArray[np.float32]] = []
    candidates: list[dict[str, Scalar]] = []
    for shard_id, shard_rows in sorted(by_shard.items()):
        path = shard_path(split_dir, shard_id)
        arrays = _load_npz(path, ("rows", "candid", "cutouts", "candidates"))
        records = _json_list(arrays["candidates"], "candidates")
        position = {int(r): i for i, r in enumerate(arrays["rows"])}
        for row in shard_rows:
            if row not in position:
                raise DataError(f"{split_dir}: row {row} is not in shard {shard_id}")
            i = position[row]
            rows.append(row)
            candid.append(int(arrays["candid"][i]))
            cutouts.append(arrays["cutouts"][i])
            candidates.append(records[i])
    order = np.argsort(np.asarray(rows, dtype=np.int64), kind="stable")
    stacked = np.stack(cutouts) if cutouts else np.empty((0, *STAMP_SHAPE), dtype=np.float32)
    return Recovered(
        rows=np.asarray(rows, dtype=np.int64)[order],
        candid=np.asarray(candid, dtype=np.int64)[order],
        cutouts=stacked[order].astype(np.float32, copy=False),
        candidates=[candidates[i] for i in order],
    )


@dataclass(frozen=True, slots=True)
class Summary:
    """The state of one split of the store (:func:`summarize`).

    Attributes:
        split: the split name.
        n_rows: the split's size (from ``fetch.json``).
        class_sizes: class name -> rows of that class in the split.
        counts: class name -> status -> rows (latest line per row).
        edge: class name -> ``ok`` rows whose cutouts were not 63x63 (CCD edge).
        attempts: the number of index lines (attempts, retries included).
        requests: total requests over every attempt.
        seconds: total wall time over every attempt.
        max_abs_diff_ok: the largest verification difference of the ``ok`` rows.
    """

    split: str
    n_rows: int
    class_sizes: dict[str, int]
    counts: dict[str, dict[str, int]]
    edge: dict[str, int]
    attempts: int
    requests: int
    seconds: float
    max_abs_diff_ok: float | None

    @property
    def totals(self) -> dict[str, int]:
        """Status -> rows over every class."""
        return {s: sum(c[s] for c in self.counts.values()) for s in STATUSES}

    @property
    def done(self) -> int:
        """Rows with a line in the index."""
        return sum(self.totals.values())

    @property
    def ghosts(self) -> dict[str, int]:
        """Class name -> ``ghost_candidate`` rows (objects not in ALeRCE's database)."""
        return {name: c["ghost_candidate"] for name, c in self.counts.items()}

    def recovery(self) -> dict[str, float | None]:
        """Class name -> ``ok`` rows / non-ghost rows (``None`` if there are none)."""
        out: dict[str, float | None] = {}
        for name, size in self.class_sizes.items():
            target = size - self.ghosts[name]
            out[name] = self.counts[name]["ok"] / target if target > 0 else None
        return out

    def accepted(self) -> bool:
        """The acceptance: every class's non-ghost rows recovered at :data:`MIN_RECOVERY`."""
        return all(r is None or r >= MIN_RECOVERY for r in self.recovery().values())

    def to_json(self) -> dict[str, Any]:
        """The summary as plain JSON values."""
        return {
            "split": self.split,
            "n_rows": self.n_rows,
            "done": self.done,
            "class_sizes": self.class_sizes,
            "counts": self.counts,
            "totals": self.totals,
            "groups": {
                g: {name: sum(c[s] for s in ss) for name, c in self.counts.items()}
                for g, ss in GROUPS.items()
            },
            "recovery": self.recovery(),
            "accepted": self.accepted(),
            "ghosts": self.ghosts,
            "edge_ok": self.edge,
            "attempts": self.attempts,
            "requests": self.requests,
            "seconds": self.seconds,
            "seconds_per_attempt": self.seconds / self.attempts if self.attempts else None,
            "max_abs_diff_ok": self.max_abs_diff_ok,
        }


def summarize(split_dir: Path) -> Summary:
    """Counts per class and status of ``split_dir`` (the latest line of each row).

    Raises:
        DataError: if ``fetch.json`` or the index is invalid.
    """
    meta = read_meta(split_dir)
    lines = read_index(split_dir)
    counts = {name: dict.fromkeys(STATUSES, 0) for name in LABEL_NAMES}
    edge = dict.fromkeys(LABEL_NAMES, 0)
    ok_diffs: list[float] = []
    for line in latest_rows(lines).values():
        if not 0 <= line.label < len(LABEL_NAMES):
            raise DataError(f"{split_dir}: row {line.row} has label {line.label}")
        name = LABEL_NAMES[line.label]
        counts[name][line.status] += 1
        if line.status == "ok":
            if line.max_abs_diff is not None:
                ok_diffs.append(line.max_abs_diff)
            if line.raw_shape and line.raw_shape != _FULL_SHAPE:
                edge[name] += 1
    sizes = meta.get("class_counts", {})
    return Summary(
        split=str(meta.get("split", split_dir.name)),
        n_rows=int(meta.get("n_rows", 0)),
        class_sizes={name: int(sizes.get(name, 0)) for name in LABEL_NAMES},
        counts=counts,
        edge=edge,
        attempts=len(lines),
        requests=sum(line.requests for line in lines),
        seconds=sum(line.seconds for line in lines),
        max_abs_diff_ok=max(ok_diffs) if ok_diffs else None,
    )


def format_summary(summary: Summary) -> str:
    """Counts per class and status, the acceptance groups, recovery, CCD-edge and costs."""
    shown = [s for s in STATUSES if summary.totals[s] or s in ("ok", "mismatch")]
    width = max(len(name) for name in (*LABEL_NAMES, "total"))
    cols = [*shown, "edge", "recovery"]

    def cell(name: str, value: str) -> str:
        return f"{value:>{max(len(name), 6)}}"

    recovery = summary.recovery()
    lines = [
        f"{summary.split}: {summary.done}/{summary.n_rows} rows done",
        f"{'class':<{width}}  " + "  ".join(cell(c, c) for c in cols),
    ]
    for name, by_status in summary.counts.items():
        r = recovery[name]
        values = [str(by_status[s]) for s in shown]
        values += [str(summary.edge[name]), "-" if r is None else f"{100 * r:.1f}%"]
        lines.append(
            f"{name:<{width}}  " + "  ".join(cell(c, v) for c, v in zip(cols, values, strict=True))
        )
    total_values = [str(summary.totals[s]) for s in shown]
    total_values += [str(sum(summary.edge.values())), ""]
    lines.append(
        f"{'total':<{width}}  "
        + "  ".join(cell(c, v) for c, v in zip(cols, total_values, strict=True))
    )
    groups = ", ".join(f"{g} {sum(summary.totals[s] for s in ss)}" for g, ss in GROUPS.items())
    lines.append(f"groups: {groups}")
    per = summary.seconds / summary.attempts if summary.attempts else 0.0
    lines.append(
        f"attempts {summary.attempts}, requests {summary.requests}, "
        f"wall time {summary.seconds:.1f} s ({per:.2f} s per attempt)"
    )
    if summary.max_abs_diff_ok is not None:
        lines.append(f"max |minmax(raw) - stored| over ok rows: {summary.max_abs_diff_ok:.2e}")
    pending = summary.n_rows - summary.done
    verdict = "PASS" if summary.accepted() else "FAIL"
    suffix = f" ({pending} rows not attempted yet)" if pending else ""
    lines.append(
        f"acceptance (>= {100 * MIN_RECOVERY:.0f}% ok per class, ghosts excluded): "
        f"{verdict}{suffix}"
    )
    return "\n".join(lines) + "\n"


def duplicates(split_dirs: Iterable[Path]) -> dict[int, list[tuple[str, int]]]:
    """Candid -> the (split, row) pairs whose ``ok`` alert it is, for candids with several.

    Many rows -> one candid means the same alert is stored twice: within a split, or
    across splits (a leak, such as ``red``'s Train 2995 and Validation 115).

    Raises:
        DataError: if an index or ``fetch.json`` is invalid.
    """
    owners: dict[int, list[tuple[str, int]]] = {}
    for split_dir in split_dirs:
        split = str(read_meta(split_dir).get("split", split_dir.name))
        for line in latest_rows(read_index(split_dir)).values():
            if line.status == "ok" and line.candid is not None:
                owners.setdefault(line.candid, []).append((split, line.row))
    return {c: sorted(o) for c, o in sorted(owners.items()) if len(o) > 1}


def format_duplicates(found: dict[int, list[tuple[str, int]]], limit: int = 20) -> str:
    """One line per duplicated candid (at most ``limit``), and the count of cross-split ones."""
    cross = sum(1 for owners in found.values() if len({s for s, _ in owners}) > 1)
    lines = [f"duplicated candids: {len(found)} ({cross} across splits)"]
    for candid, owners in list(found.items())[:limit]:
        lines.append(f"  {candid}: " + ", ".join(f"{s} {r}" for s, r in owners))
    if len(found) > limit:
        lines.append(f"  ... {len(found) - limit} more")
    return "\n".join(lines) + "\n"
