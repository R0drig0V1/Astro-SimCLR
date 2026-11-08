"""Resolve, download and verify the raw alerts of a prepared split (task 3.1, ``fetch-alerts``).

For each split row (the integrator's probe of 2026-11-08 and the plan-review amendments;
``docs/data.md``, "Raw alerts"):

1. **Target.** De-standardise ``ra``, ``dec``, ``magpsf`` and ``sigmapsf`` with M1's
   statistics (:func:`~astro_simclr.data.m1_features.destandardise`).
2. **Objects.** Cone search of :data:`~astro_simclr.alerts.alerce.RADIUS_ARCSEC` around
   (``ra``, ``dec``). None: ``ghost_candidate``, a row whose object is not in ALeRCE's
   database (RJ23's "ghost bogus": stamps known only from CD21's files). It is final: no
   retry beyond the transport's backoff and no wider tolerance; its de-standardised
   position stays in the index (``target_ra``, ``target_dec``) for later position groups.
   Objects in the cone but none with a detection: ``no_object``.
3. **Match.** For *every* object found, its detections; a detection matches when both
   ``|magpsf - target| <= 1e-4`` and ``|sigmapsf - target| <= 1e-4``
   (:data:`MATCH_TOLERANCE`). Matches are counted per distinct ``candid``; none is
   ``no_detection_match``.
4. **Ties.** ``jd`` is not a stored feature, so an AGN or VS with hundreds of detections
   can tie. Every matching candid is downloaded (up to :data:`MAX_CANDIDATES`; more is
   ``ambiguous`` without downloading) and the stamp check decides.
5. **Verify.** ALeRCE's normalisation of the raw cutouts, per-channel min-max over the full
   63x63 stamp without epsilon then NaN -> 0 (:func:`alerce_normalise`), against the
   stored ``images_f32`` row: a candidate passes when ``max |delta| <= 1e-6``
   (:data:`VERIFY_TOLERANCE`). Exactly one passing candidate is ``ok`` (stored); several
   are ``ambiguous``; none is ``mismatch`` (the closest is recorded, not stored), or
   ``bad_alert`` if no candidate was a readable alert.

A request that fails for good makes the row ``http_error``, which the next ``--resume``
tries again; every other status is final. That is a timeout, 5xx or 429 still failing
after the retries of :class:`~astro_simclr.alerts.transport.Fetcher`, malformed JSON
(both *transient*), or a final 4xx. After :data:`COOLDOWN_AFTER` transient failures in a
row the run pauses for ``cooldown`` seconds, so an outage of ALeRCE costs a pause, not
thousands of failed rows, when the run is unattended. If the cone search reports more
objects than its first page, a row that would end ``no_object``,
``no_detection_match``, ``mismatch`` or ``bad_alert`` is ``ambiguous`` instead, since an
unread object could hold its alert (an ``ok`` stands: the stamp check is conclusive).

**Order.** Rows are visited round-robin over the classes, in row order within each class
(:func:`interleaved_order`), so ``--limit N`` and an interrupted run cover the classes
evenly; Validation and Test are sorted by class. **Batches.** Every ``batch_size`` rows
are committed together: one shard (the ``ok`` cutouts and every raw response fetched),
then their index lines (:mod:`~astro_simclr.alerts.store`). The batch in progress is also
committed when the run stops: on an exception, or between rows when ``stop()`` turns
true (the CLI's SIGTERM, SIGHUP and Ctrl-C handlers set it, so a signal never cuts a
commit). A failed commit is not retried, and its responses stay pending; a hard kill loses
at most one batch. **Resume.** Rows with a final status in the index are skipped, except
an ``ok`` row whose shard is missing (a crash lost the rename), which is fetched again.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from astro_simclr.alerts.alerce import (
    API_URL,
    AVRO_URL,
    RADIUS_ARCSEC,
    AlerceClient,
    Detection,
    ResponseError,
)
from astro_simclr.alerts.parse import Alert, AlertFormatError, parse_alert
from astro_simclr.alerts.snapshot import Recorder
from astro_simclr.alerts.store import (
    RETRIED,
    STATUSES,
    IndexRow,
    IndexWriter,
    Shard,
    latest_rows,
    next_shard_id,
    read_index,
    read_meta,
    shard_path,
    write_meta,
    write_shard,
)
from astro_simclr.alerts.transport import HttpError
from astro_simclr.data.float_input import crop_then_normalise
from astro_simclr.data.m1_features import destandardise
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.prepared import read_meta as read_prepared_meta
from astro_simclr.data.schema import LABEL_NAMES, SPLITS, STAMP_SHAPE, DataError, SplitData

__all__ = [
    "COOLDOWN_AFTER",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_COOLDOWN",
    "MATCH_TOLERANCE",
    "MAX_CANDIDATES",
    "VERIFY_TOLERANCE",
    "RowResult",
    "RunReport",
    "Target",
    "alerce_normalise",
    "fetch_alerts",
    "fetch_row",
    "interleaved_order",
    "match_detections",
    "targets",
    "verify_cutouts",
]

#: Photometric match tolerance on ``magpsf`` and ``sigmapsf`` (mag). The stored features
#: are the alert's own float32 values; the API rounds them to about 1e-6.
MATCH_TOLERANCE = 1e-4
#: Acceptance tolerance of the stamp check (the stored stamps are float32 in [0, 1]).
VERIFY_TOLERANCE = 1e-6
#: Most tied candids downloaded for one row; more is ``ambiguous`` without downloading.
MAX_CANDIDATES = 16
#: Rows per committed batch: a hard kill loses at most this many rows (about 5 minutes).
DEFAULT_BATCH_SIZE = 100
#: Consecutive ``http_error`` rows that trigger a pause, and the pause in seconds.
COOLDOWN_AFTER = 3
DEFAULT_COOLDOWN = 300.0


@dataclass(frozen=True, slots=True)
class Target:
    """One split row and the de-standardised values that find its alert."""

    row: int
    label: int
    ra: float
    dec: float
    magpsf: float
    sigmapsf: float


def targets(split: SplitData) -> list[Target]:
    """The :class:`Target` of every row of a labelled split.

    Raises:
        DataError: if the split has no features or labels.
    """
    if split.features is None or split.labels is None:
        raise DataError("the split has no features or labels (an unlabelled split)")
    values = destandardise(split.features)
    return [
        Target(
            row=i,
            label=int(split.labels[i]),
            ra=float(values["ra"][i]),
            dec=float(values["dec"][i]),
            magpsf=float(values["magpsf"][i]),
            sigmapsf=float(values["sigmapsf"][i]),
        )
        for i in range(len(split))
    ]


def interleaved_order(labels: Sequence[int] | npt.NDArray[np.integer[Any]]) -> list[int]:
    """Row indices round-robin over the classes (ascending label), row order within each."""
    by_class: dict[int, list[int]] = {}
    for row, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(row)
    queues = [by_class[label] for label in sorted(by_class)]
    order: list[int] = []
    for depth in range(max((len(q) for q in queues), default=0)):
        order.extend(q[depth] for q in queues if depth < len(q))
    return order


def match_detections(
    target: Target,
    detections: Mapping[str, Sequence[Detection]],
    tolerance: float = MATCH_TOLERANCE,
) -> list[tuple[str, Detection]]:
    """The (oid, detection) pairs matching ``target``'s photometry, one per distinct candid.

    ``detections`` maps each object of the cone search to its detections. A candid listed
    twice (under one or two objects) counts once, with the first object that lists it.
    """
    found: dict[int, tuple[str, Detection]] = {}
    for oid, items in detections.items():
        for item in items:
            if (
                abs(item.magpsf - target.magpsf) <= tolerance
                and abs(item.sigmapsf - target.sigmapsf) <= tolerance
            ):
                found.setdefault(item.candid, (oid, item))
    return list(found.values())


def alerce_normalise(cutouts: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    """ALeRCE's normalisation of a raw ``(..., 63, 63, 3)`` stamp, in float64.

    Per channel ``x -= nanmin(x)``, ``x /= nanmax(x)`` over the full stamp, no epsilon,
    then NaN -> 0 (a constant or all-NaN channel becomes 0): the crop of
    :func:`~astro_simclr.data.float_input.crop_then_normalise` at the full size, without
    the NaN approximation.
    """
    return crop_then_normalise(
        np.asarray(cutouts, dtype=np.float64), crop=cutouts.shape[-2], nan_approximation=False
    )


def verify_cutouts(
    cutouts: npt.NDArray[np.floating[Any]], stored: npt.NDArray[np.floating[Any]]
) -> float:
    """``max |alerce_normalise(cutouts) - stored|`` over the stamp (float64).

    Raises:
        ValueError: if the shapes differ.
    """
    if cutouts.shape != stored.shape:
        raise ValueError(f"cutouts {cutouts.shape} and stored {stored.shape} differ in shape")
    stored64 = np.asarray(stored, dtype=np.float64)
    diff = float(np.abs(alerce_normalise(cutouts) - stored64).max())
    # An inf pixel can make the difference NaN; NaN would compare as "not larger".
    return diff if np.isfinite(diff) else float("inf")


@dataclass(frozen=True, slots=True)
class RowResult:
    """The outcome of one row: its index line, the alert when it is ``ok``, and whether an
    ``http_error`` was transient (a timeout, 5xx, 429 or malformed JSON; a final 4xx is
    not), which alone counts toward the cooldown."""

    line: IndexRow
    alert: Alert | None
    transient: bool = False


def fetch_row(
    client: AlerceClient,
    target: Target,
    stored: npt.NDArray[np.floating[Any]],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> RowResult:
    """Resolve, download and verify one row (module docstring); never raises for a miss.

    ``stored`` is the row's stored float stamp ``(63, 63, 3)``. When the client's getter
    is a :class:`~astro_simclr.alerts.snapshot.Recorder`, the line lists the responses
    the row used.
    """
    recorder = client.getter if isinstance(client.getter, Recorder) else None
    if recorder is not None:
        recorder.start_row()
    start, requests = clock(), client.requests
    line, alert, transient = _resolve(client, target, stored)
    refs = " ".join(recorder.row_refs) if recorder is not None else ""
    line = replace(
        line, responses=refs, requests=client.requests - requests, seconds=clock() - start
    )
    return RowResult(line, alert, transient)


@dataclass(frozen=True, slots=True)
class _Candidate:
    alert: Alert
    diff: float


#: Statuses that an unread second page of the cone search could change: the row becomes
#: ``ambiguous`` instead (an ``ok`` stands: the stamp check is conclusive).
_PAGE_DEPENDENT = frozenset({"no_object", "no_detection_match", "mismatch", "bad_alert"})


def _resolve(
    client: AlerceClient, target: Target, stored: npt.NDArray[np.floating[Any]]
) -> tuple[IndexRow, Alert | None, bool]:
    try:
        line, alert = _search(client, target, stored)
    except _RowFailed as failed:
        return failed.line, None, failed.transient
    if line.status in _PAGE_DEPENDENT and "more objects" in line.note:
        line = replace(line, status="ambiguous")
    return line, alert, False


class _RowFailed(Exception):
    """A request of the row failed for good: the ``http_error`` line, and if transient."""

    def __init__(self, line: IndexRow, transient: bool) -> None:
        self.line = line
        self.transient = transient


def _search(
    client: AlerceClient, target: Target, stored: npt.NDArray[np.floating[Any]]
) -> tuple[IndexRow, Alert | None]:
    line = IndexRow(
        row=target.row,
        label=target.label,
        status="http_error",
        target_ra=target.ra,
        target_dec=target.dec,
        target_magpsf=target.magpsf,
        target_sigmapsf=target.sigmapsf,
    )
    try:
        page = client.objects_near(target.ra, target.dec)
        line = replace(line, n_objects=len(page.oids), note="more objects" if page.has_next else "")
        if not page.oids:
            return replace(line, status="ghost_candidate"), None
        detections = {oid: client.detections(oid) for oid in page.oids}
        if not any(detections.values()):
            return replace(line, status="no_object"), None
        matches = match_detections(target, detections)
        listed = " ".join(f"{oid}:{d.candid}" for oid, d in matches)
        line = replace(line, n_matches=len(matches), candidates=listed)
        if not matches:
            return replace(line, status="no_detection_match"), None
        if len(matches) > MAX_CANDIDATES:
            note = f"{len(matches)} candidates, more than {MAX_CANDIDATES}: not downloaded"
            return replace(line, status="ambiguous", note=_join(line.note, note)), None
        found: list[_Candidate] = []
        errors: list[str] = []
        for oid, detection in matches:
            data = client.alert_avro(oid, detection.candid)
            try:
                alert = parse_alert(data, oid=oid, candid=detection.candid)
            except AlertFormatError as err:
                errors.append(f"{detection.candid}: {err}")
                continue
            found.append(_Candidate(alert, verify_cutouts(alert.cutouts, stored)))
    except (HttpError, ResponseError) as err:
        status = err.status if isinstance(err, HttpError) else None
        transient = status is None or status >= 500 or status == 429
        failed = replace(line, status="http_error", note=_join(line.note, str(err)))
        raise _RowFailed(failed, transient) from None
    line = replace(line, note=_join(line.note, *errors))
    return _decide(line, found)


def _decide(line: IndexRow, found: list[_Candidate]) -> tuple[IndexRow, Alert | None]:
    """The status from the downloaded candidates (module docstring, step 5)."""
    if not found:
        return replace(line, status="bad_alert"), None
    passing = [c for c in found if c.diff <= VERIFY_TOLERANCE]
    if len(passing) > 1:
        note = "several candidates pass the stamp check: " + " ".join(
            str(c.alert.candid) for c in passing
        )
        return replace(line, status="ambiguous", note=_join(line.note, note)), None
    best = passing[0] if passing else min(found, key=lambda c: c.diff)
    line = replace(_with_candidate(line, best.alert), max_abs_diff=best.diff)
    if passing:
        return replace(line, status="ok"), best.alert
    return replace(line, status="mismatch"), None


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def _with_candidate(line: IndexRow, alert: Alert) -> IndexRow:
    c = alert.candidate

    def number(name: str) -> float | None:
        value = c.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)

    fid = c.get("fid")
    height, width = alert.raw_shape
    padding = (STAMP_SHAPE[0] * STAMP_SHAPE[1] - height * width) * STAMP_SHAPE[2]
    return replace(
        line,
        oid=alert.oid,
        candid=alert.candid,
        jd=number("jd"),
        fid=fid if isinstance(fid, int) and not isinstance(fid, bool) else None,
        magpsf=number("magpsf"),
        sigmapsf=number("sigmapsf"),
        isdiffpos=str(c.get("isdiffpos") or ""),
        rb=number("rb"),
        drb=number("drb"),
        ra=number("ra"),
        dec=number("dec"),
        raw_shape=f"{height}x{width}",
        nan_pixels=int(np.isnan(alert.cutouts).sum()) - padding,
    )


@dataclass(frozen=True, slots=True)
class RunReport:
    """What one :func:`fetch_alerts` call did.

    Attributes:
        split_dir: ``OUT/<split>``.
        attempted: rows attempted by this call.
        skipped: rows skipped because the index already had a final status.
        counts: status -> rows of this call.
        requests: network requests of this call.
        seconds: wall time of this call.
        cooldowns: pauses taken after consecutive ``http_error`` rows.
    """

    split_dir: Path
    attempted: int
    skipped: int
    counts: dict[str, int]
    requests: int
    seconds: float
    cooldowns: int


def fetch_alerts(
    prepared_dir: Path,
    split: str,
    out: Path,
    *,
    client: AlerceClient,
    limit: int | None = None,
    resume: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cooldown: float = DEFAULT_COOLDOWN,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int, int, IndexRow], None] | None = None,
    stop: Callable[[], bool] = lambda: False,
) -> RunReport:
    """Fetch the raw alerts of ``prepared_dir``'s ``split`` into ``out/<split>/``.

    ``prepared_dir`` is a prepared layout with ``images_f32.npy`` (``prepare-data
    --images float32``). ``client``'s getter should be a
    :class:`~astro_simclr.alerts.snapshot.Recorder`, whose responses are stored in the
    shards. ``limit`` caps the rows attempted by this call (in :func:`interleaved_order`,
    after skipping the finished ones). ``progress`` is called after each row with
    (position, rows to attempt, index line). ``stop`` is polled between rows: when it
    returns true, the run commits its batch and returns (the CLI's signal handlers set it,
    so a SIGTERM never interrupts a commit).

    Raises:
        DataError: if the prepared layout is invalid or lacks float32 stamps; if
            ``out/<split>`` already holds a run and ``resume`` is false; if it holds a run
            of another dataset, source or split; or if its index is invalid.
        ValueError: if ``split``, ``limit``, ``batch_size`` or ``cooldown`` is invalid.
        OSError: if the store cannot be written.
    """
    _check_options(split, limit, batch_size, cooldown)
    data = load_prepared(prepared_dir).splits[split]
    if data.images_f32 is None:
        raise DataError(
            f"{prepared_dir}: no images_f32.npy; prepare it with prepare-data --images float32"
        )
    stored = data.images_f32
    rows = targets(data)
    split_dir = out / split
    _open_store(split_dir, _run_meta(prepared_dir, split, rows), resume=resume)
    latest = latest_rows(read_index(split_dir))
    # An ok row whose shard is missing (a crash lost the rename) is fetched again.
    done = {
        r
        for r, line in latest.items()
        if line.status not in RETRIED
        and not (
            line.status == "ok"
            and (line.shard is None or not shard_path(split_dir, line.shard).is_file())
        )
    }
    pending = [r for r in interleaved_order([t.label for t in rows]) if r not in done]
    if limit is not None:
        pending = pending[:limit]

    recorder = client.getter if isinstance(client.getter, Recorder) else None
    start, requests0 = clock(), client.requests
    counts = dict.fromkeys(STATUSES, 0)
    writer = IndexWriter(split_dir)
    batch: list[RowResult] = []
    shard_id = next_shard_id(split_dir)
    streak = cooldowns = attempted = 0
    try:
        try:
            for position, row in enumerate(pending, start=1):
                if stop():
                    break
                result = fetch_row(client, rows[row], stored[row], clock=clock)
                attempted += 1
                batch.append(result)
                counts[result.line.status] += 1
                if progress is not None:
                    progress(position, len(pending), result.line)
                if len(batch) >= batch_size:
                    # Emptied first: a failed commit is never retried by the finally below.
                    full, batch = batch, []
                    _commit(split_dir, shard_id, full, writer, recorder)
                    shard_id += 1
                streak = streak + 1 if result.transient else 0
                if streak >= COOLDOWN_AFTER and cooldown > 0 and position < len(pending):
                    sleep(cooldown)
                    streak, cooldowns = 0, cooldowns + 1
        finally:
            if batch:
                _commit(split_dir, shard_id, batch, writer, recorder)
    finally:
        writer.close()
    return RunReport(
        split_dir=split_dir,
        attempted=attempted,
        skipped=len(done),
        counts=counts,
        requests=client.requests - requests0,
        seconds=clock() - start,
        cooldowns=cooldowns,
    )


def _check_options(split: str, limit: int | None, batch_size: int, cooldown: float) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if cooldown < 0:
        raise ValueError(f"cooldown must be non-negative, got {cooldown}")


def _run_meta(prepared_dir: Path, split: str, rows: Sequence[Target]) -> dict[str, Any]:
    source = read_prepared_meta(prepared_dir).get("source", {})
    sizes = dict.fromkeys(LABEL_NAMES, 0)
    for target in rows:
        sizes[LABEL_NAMES[target.label]] += 1
    return {
        "dataset": prepared_dir.name,
        "source": {"name": source.get("name"), "sha256": source.get("sha256")},
        "split": split,
        "n_rows": len(rows),
        "class_counts": sizes,
        "api_url": API_URL,
        "avro_url": AVRO_URL,
        "radius_arcsec": RADIUS_ARCSEC,
        "match_tolerance": MATCH_TOLERANCE,
        "verify_tolerance": VERIFY_TOLERANCE,
        "max_candidates": MAX_CANDIDATES,
    }


#: ``fetch.json`` keys that must agree for ``--resume`` to continue a run: the data, and
#: every setting that decides a status.
_IDENTITY = (
    "dataset",
    "source",
    "split",
    "n_rows",
    "class_counts",
    "radius_arcsec",
    "match_tolerance",
    "verify_tolerance",
    "max_candidates",
)


def _open_store(split_dir: Path, meta: dict[str, Any], *, resume: bool) -> None:
    if (split_dir / "fetch.json").exists():
        if not resume:
            raise DataError(f"{split_dir} already holds a run; pass --resume to continue it")
        existing = read_meta(split_dir)
        for key in _IDENTITY:
            if existing.get(key) != meta[key]:
                raise DataError(
                    f"{split_dir}: {key} is {existing.get(key)!r} in fetch.json, but this run "
                    f"has {meta[key]!r}; use another --out"
                )
        return
    split_dir.mkdir(parents=True, exist_ok=True)
    write_meta(split_dir, meta)


def _commit(
    split_dir: Path,
    shard_id: int,
    batch: list[RowResult],
    writer: IndexWriter,
    recorder: Recorder | None,
) -> None:
    """Write the batch as one shard, then its index lines (in that order).

    The recorder's pending responses are dropped only once both are durable, so a commit
    that fails keeps them for the next one.
    """
    responses = recorder.pending() if recorder is not None else []
    ok = [(r.line.row, r.alert) for r in batch if r.alert is not None]
    cutouts = (
        np.stack([a.cutouts for _, a in ok]).astype(np.float32, copy=False)
        if ok
        else np.empty((0, *STAMP_SHAPE), dtype=np.float32)
    )
    shard = Shard(
        rows=np.asarray([row for row, _ in ok], dtype=np.int64),
        candid=np.asarray([a.candid for _, a in ok], dtype=np.int64),
        cutouts=cutouts,
        candidates=[a.candidate for _, a in ok],
        responses=responses,
    )
    write_shard(split_dir, shard_id, shard)
    writer.append([replace(r.line, shard=shard_id) for r in batch])
    if recorder is not None:
        recorder.drop_pending(len(responses))
