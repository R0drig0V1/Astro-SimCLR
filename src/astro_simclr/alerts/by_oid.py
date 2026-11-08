"""Fetch detection lists and alerts by object id (task 3.5; extends task 3.1's ``fetch-alerts``).

Task 3.1 finds the alert of a *prepared row* from its features. The R5 reconstruction
(:mod:`astro_simclr.r5`) starts from *object ids* instead: the Zenodo light-curve labelled
set and the Bright Transient Survey list ``oid``\\ s, not alerts. For those it needs, per
object, the full detection list (to draw one alert with a stamp before ``JD_MAX``), then the
avro of the drawn ``candid``. This module fetches exactly such a list of items:

- ``detections OID``: ``GET {API_URL}/objects/{oid}/detections``;
- ``avro OID CANDID``: ``GET {AVRO_URL}/get_avro?oid=..&candid=..``, checked with
  :func:`~astro_simclr.alerts.parse.parse_alert` (a readable alert of that object and
  candid, cutouts at most 63x63).

Everything else is task 3.1's machinery, unchanged: one :class:`~astro_simclr.alerts.
transport.Fetcher` (one rate limit over both hosts, retries with backoff on timeouts, 5xx
and 429), the :class:`~astro_simclr.alerts.snapshot.Recorder` (every raw response kept with
its URL, retrieval time and SHA-256), the shard format of :mod:`~astro_simclr.alerts.store`
(one ``.npz`` per committed batch, here with no cutouts: the avro bodies *are* the alerts),
batches committed shard first, then their index lines, the stop callback (a signal never
cuts a commit), the cooldown after consecutive transient failures, and ``--resume``.

Layout::

    OUT/
        fetch.json        format "astro-simclr-oid-fetch", version 1, the endpoints
        index.csv         one line per attempt (append-only; the last line of an item wins):
                          kind, oid, candid, status, n_detections, raw_shape, shard,
                          responses, requests, seconds, note
        shard-00000.npz   the raw responses of one batch (store.write_shard, no cutouts)

Statuses (:data:`STATUSES`): ``ok``; ``not_found`` (HTTP 404, final: ALeRCE has no such
object or alert); ``refused`` (another final 4xx); ``bad_alert`` (the avro is not a readable
alert of that object and candid, final); ``http_error`` (transient: a timeout, 5xx or 429
after the retries, or malformed JSON; ``--resume`` tries it again). A detection list is
``ok`` when it is a JSON list of objects, however many usable records it has.

:class:`~astro_simclr.alerts.store.Snapshot` serves the stored responses by URL, so the R5
build reads every body it uses from the snapshot, offline, and its final statuses from
:func:`final_statuses`.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from astro_simclr.alerts.alerce import AlerceClient, ResponseError, is_oid
from astro_simclr.alerts.parse import AlertFormatError, parse_alert
from astro_simclr.alerts.snapshot import Recorder
from astro_simclr.alerts.store import Shard, next_shard_id, write_shard
from astro_simclr.alerts.transport import HttpError
from astro_simclr.data.schema import STAMP_SHAPE, DataError

__all__ = [
    "FINAL",
    "FORMAT",
    "FORMAT_VERSION",
    "KINDS",
    "STATUSES",
    "FetchItem",
    "ItemLine",
    "OidRunReport",
    "fetch_by_oid",
    "final_statuses",
    "read_items",
    "read_oid_index",
]

FORMAT: Final = "astro-simclr-oid-fetch"
FORMAT_VERSION: Final = 1
META_NAME: Final = "fetch.json"
INDEX_NAME: Final = "index.csv"
#: The two kinds of item.
KINDS: Final = ("detections", "avro")
#: Every status, in report order.
STATUSES: Final = ("ok", "not_found", "refused", "bad_alert", "http_error")
#: Statuses that ``--resume`` does not attempt again.
FINAL: Final = frozenset({"ok", "not_found", "refused", "bad_alert"})
#: Consecutive transient failures that trigger the cooldown (as task 3.1).
COOLDOWN_AFTER: Final = 3

Kind = Literal["detections", "avro"]


@dataclass(frozen=True, slots=True)
class FetchItem:
    """One request to make: an object's detection list, or one alert (``candid`` set)."""

    kind: Kind
    oid: str
    candid: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not is_oid(self.oid):
            raise ValueError(f"not a ZTF object id: {self.oid!r}")
        if (self.kind == "avro") != (self.candid is not None):
            raise ValueError("an avro item needs a candid, a detections item none")
        if self.candid is not None and not 0 < self.candid < 2**63:
            raise ValueError(f"candid must be a positive int64, got {self.candid}")

    def sort_key(self) -> tuple[str, str, int]:
        """A total order of items (kind, oid, candid), for deterministic item lists."""
        return (self.kind, self.oid, self.candid or 0)


@dataclass(frozen=True, slots=True)
class ItemLine:
    """One line of ``index.csv``: one attempt at one item (module docstring)."""

    kind: str
    oid: str
    candid: int | None
    status: str
    n_detections: int | None = None
    raw_shape: str = ""
    shard: int | None = None
    responses: str = ""
    requests: int = 0
    seconds: float = 0.0
    note: str = ""

    @property
    def item(self) -> FetchItem:
        """The item this line is about."""
        kind: Kind = "avro" if self.kind == "avro" else "detections"
        return FetchItem(kind, self.oid, self.candid)


_COLUMNS: Final = tuple(f.name for f in fields(ItemLine))
_INT: Final = frozenset({"candid", "n_detections", "shard", "requests"})


def read_items(path: Path) -> list[FetchItem]:
    """Read an items file: one item per line, ``OID`` (its detections) or ``OID CANDID``.

    Blank lines and lines starting with ``#`` are ignored; repeats are kept once, in the
    order of their first line.

    Raises:
        DataError: if the file cannot be read or a line is not an item (naming the line).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise DataError(f"{path}: cannot read the items file ({err})") from None
    items: dict[FetchItem, None] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        try:
            if len(parts) == 1:
                item = FetchItem("detections", parts[0])
            elif len(parts) == 2 and parts[1].isascii() and parts[1].isdigit():
                item = FetchItem("avro", parts[0], int(parts[1]))
            else:
                raise ValueError("expected OID or OID CANDID")
        except ValueError as err:
            raise DataError(f"{path}:{number}: {err}") from None
        items.setdefault(item, None)
    return list(items)


def read_oid_index(out: Path) -> list[ItemLine]:
    """Every line of ``out/index.csv`` in file order (empty if there is none).

    A last line without its newline (an interrupted write) is ignored.

    Raises:
        DataError: if the header or a complete line is invalid.
    """
    path = out / INDEX_NAME
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if text and tuple(reader.fieldnames or ()) != _COLUMNS:
        raise DataError(f"{path}: unexpected columns {reader.fieldnames}")
    lines: list[ItemLine] = []
    for number, record in enumerate(reader, start=2):
        try:
            values: dict[str, Any] = {}
            for name in _COLUMNS:
                raw = record[name]
                if name in _INT:
                    values[name] = None if raw == "" else int(raw)
                elif name == "seconds":
                    values[name] = float(raw)
                else:
                    values[name] = raw
            line = ItemLine(**values)
            if line.status not in STATUSES:
                raise ValueError(f"unknown status {line.status!r}")
            line.item  # noqa: B018 - validates kind, oid and candid
        except (KeyError, ValueError, TypeError) as err:
            raise DataError(f"{path}: line {number}: {err}") from None
        lines.append(line)
    return lines


def final_statuses(out: Path) -> dict[FetchItem, str]:
    """Item -> the status of its last attempt in ``out``, for final statuses only."""
    latest = {line.item: line.status for line in read_oid_index(out)}
    return {item: status for item, status in latest.items() if status in FINAL}


@dataclass(frozen=True, slots=True)
class OidRunReport:
    """What one :func:`fetch_by_oid` call did: items attempted and skipped, the statuses,
    requests, wall time and cooldowns."""

    attempted: int
    skipped: int
    counts: dict[str, int]
    requests: int
    seconds: float
    cooldowns: int


def _meta(api_url: str, avro_url: str) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "api_url": api_url,
        "avro_url": avro_url,
        "created_by": f"astro-simclr {version('astro-simclr')}",
    }


def _open(out: Path, client: AlerceClient, *, resume: bool) -> None:
    path = out / META_NAME
    if path.exists():
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise DataError(f"{path}: cannot read ({err})") from None
        if not isinstance(meta, dict) or meta.get("format") != FORMAT:
            raise DataError(f"{path}: not an {FORMAT} store")
        if meta.get("format_version") != FORMAT_VERSION:
            raise DataError(f"{path}: format version {meta.get('format_version')!r}")
        if not resume:
            raise DataError(f"{out} already holds a run; pass --resume to continue it")
        # The snapshot is keyed by URL: another host would make every stored item unseen.
        for key, value in (("api_url", client.api_url), ("avro_url", client.avro_url)):
            if meta.get(key) != value:
                raise DataError(f"{path}: {key} is {meta.get(key)!r}, this client uses {value!r}")
        return
    out.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_meta(client.api_url, client.avro_url), indent=2) + "\n")
    os.replace(tmp, path)


class _Writer:
    """Appends :class:`ItemLine` rows to ``index.csv`` (header on a new file, fsynced)."""

    def __init__(self, out: Path) -> None:
        path = out / INDEX_NAME
        if path.is_file():
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                with path.open("r+b") as handle:
                    handle.truncate(data.rfind(b"\n") + 1)
        new = not path.is_file() or path.stat().st_size == 0
        self._handle = path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        if new:
            self._writer.writerow(_COLUMNS)
            self._sync()

    def append(self, lines: Sequence[ItemLine]) -> None:
        for line in lines:
            values = asdict(line)
            self._writer.writerow([_text(values[name]) for name in _COLUMNS])
        self._sync()

    def close(self) -> None:
        self._handle.close()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value).replace("\r", " ").replace("\n", " ")


@dataclass(frozen=True, slots=True)
class _Outcome:
    line: ItemLine
    transient: bool


def _attempt(client: AlerceClient, item: FetchItem) -> _Outcome:
    """Fetch one item; never raises for a failed request (the status says it)."""
    line = ItemLine(item.kind, item.oid, item.candid, "http_error")
    try:
        if item.kind == "detections":
            found = client.detection_records(item.oid)
            return _Outcome(replace(line, status="ok", n_detections=len(found.records)), False)
        assert item.candid is not None  # noqa: S101 - FetchItem guarantees it
        body = client.alert_avro(item.oid, item.candid)
    except HttpError as err:
        status = err.status
        if status == 404:
            return _Outcome(replace(line, status="not_found", note=str(err)), False)
        transient = status is None or status >= 500 or status == 429
        return _Outcome(
            replace(line, status="http_error" if transient else "refused", note=str(err)), transient
        )
    except ResponseError as err:
        return _Outcome(replace(line, note=str(err)), True)
    try:
        alert = parse_alert(body, oid=item.oid, candid=item.candid)
    except AlertFormatError as err:
        return _Outcome(replace(line, status="bad_alert", note=str(err)), False)
    height, width = alert.raw_shape
    return _Outcome(replace(line, status="ok", raw_shape=f"{height}x{width}"), False)


def fetch_by_oid(
    items: Iterable[FetchItem],
    out: Path,
    *,
    client: AlerceClient,
    resume: bool = False,
    batch_size: int = 100,
    cooldown: float = 300.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int, int, ItemLine], None] | None = None,
    stop: Callable[[], bool] = lambda: False,
) -> OidRunReport:
    """Fetch ``items`` into the store ``out`` (module docstring); resumable.

    Items whose last attempt in ``out`` has a final status are skipped (repeats once).
    ``client``'s getter should be a :class:`~astro_simclr.alerts.snapshot.Recorder`, whose
    responses are stored in the shards. ``progress`` is called after each item with
    (position, items to attempt, line); ``stop`` is polled between items.

    Raises:
        DataError: if ``out`` holds another kind of store, or a run and ``resume`` is
            false, or its index is invalid.
        ValueError: if ``batch_size`` or ``cooldown`` is out of range.
        OSError: if the store cannot be written.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if cooldown < 0:
        raise ValueError(f"cooldown must be non-negative, got {cooldown}")
    _open(out, client, resume=resume)
    done = set(final_statuses(out))
    wanted = list(dict.fromkeys(items))
    pending = [item for item in wanted if item not in done]
    recorder = client.getter if isinstance(client.getter, Recorder) else None
    start, requests0 = clock(), client.requests
    counts = dict.fromkeys(STATUSES, 0)
    writer = _Writer(out)
    batch: list[ItemLine] = []
    shard_id = next_shard_id(out)
    streak = cooldowns = attempted = 0
    try:
        try:
            for position, item in enumerate(pending, start=1):
                if stop():
                    break
                if recorder is not None:
                    recorder.start_row()
                t0, r0 = clock(), client.requests
                outcome = _attempt(client, item)
                refs = " ".join(recorder.row_refs) if recorder is not None else ""
                line = replace(
                    outcome.line,
                    responses=refs,
                    requests=client.requests - r0,
                    seconds=clock() - t0,
                )
                attempted += 1
                counts[line.status] += 1
                batch.append(line)
                if progress is not None:
                    progress(position, len(pending), line)
                if len(batch) >= batch_size:
                    full, batch = batch, []
                    _commit(out, shard_id, full, writer, recorder)
                    shard_id += 1
                streak = streak + 1 if outcome.transient else 0
                if streak >= COOLDOWN_AFTER and cooldown > 0 and position < len(pending):
                    sleep(cooldown)
                    streak, cooldowns = 0, cooldowns + 1
        finally:
            if batch:
                _commit(out, shard_id, batch, writer, recorder)
    finally:
        writer.close()
    return OidRunReport(
        attempted=attempted,
        skipped=len(wanted) - len(pending),
        counts=counts,
        requests=client.requests - requests0,
        seconds=clock() - start,
        cooldowns=cooldowns,
    )


def _commit(
    out: Path,
    shard_id: int,
    batch: list[ItemLine],
    writer: _Writer,
    recorder: Recorder | None,
) -> None:
    """One shard with the batch's responses (no cutouts), then its index lines."""
    responses = recorder.pending() if recorder is not None else []
    shard = Shard(
        rows=np.empty(0, dtype=np.int64),
        candid=np.empty(0, dtype=np.int64),
        cutouts=np.empty((0, *STAMP_SHAPE), dtype=np.float32),
        candidates=[],
        responses=responses,
    )
    write_shard(out, shard_id, shard)
    writer.append([replace(line, shard=shard_id) for line in batch])
    if recorder is not None:
        recorder.drop_pending(len(responses))
