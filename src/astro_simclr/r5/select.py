"""The R5 selection: candidates per subclass, RJ23's cuts, the seeded walk (task 3.5).

The selection reads every byte it decides on from the **response snapshots**: task 3.1's
store of CD21 (``fetch-alerts --dataset ...``) and the by-oid store of the R5 build
(:mod:`astro_simclr.alerts.by_oid`). :class:`Responses` serves both by URL, with the final
failures of the by-oid store. :func:`evaluate` is a pure function of the inputs, the
snapshots and the seed; what it could not decide yet comes back as the
:class:`~astro_simclr.alerts.by_oid.FetchItem` list to download next
(:attr:`Selection.missing`). The online build alternates the two until nothing is missing;
the offline rebuild runs it once and refuses an incomplete snapshot. Both then write from
the same :class:`Selection`, so their outputs are the same bytes.

**Candidates** (the plan's table, :data:`~astro_simclr.r5.recipe.SUBCLASSES`):

- AGN and VS subclasses: the Zenodo rows of the subclass's ``classALeRCE``, one per
  ``oid``, in a seeded permutation (``default_rng([seed, stable_hash("sample:" + name)])``
  over the oid-sorted list), walked until the subclass reaches RJ23's count (its target);
- SN: Zenodo's SN (``SNIa``, ``SNII``, ``SNIbc``, ``SLSN``) in a seeded permutation, then
  the BTS SN that Zenodo lacks, in their own seeded permutation, walked until
  :data:`~astro_simclr.r5.recipe.SN_TARGET` in total. Zenodo first: RJ23's SN table is the
  successor of Zenodo's, so Zenodo's SN are the most RJ23-like; BTS tops them up;
- Asteroid and Bogus: every CD21 row of the class that task 3.1 resolved to an ``oid``
  (``ok``), one per ``oid``, in ``oid`` order, all kept that pass the cuts (RJ23 took all).
  Ghost rows (``ghost_candidate``, no ALeRCE object) and rows 3.1 could not resolve are
  dropped and counted.

**Decisions per candidate**, in order (the first that applies; the causes are
:data:`CAUSES`): a Bogus whose ``oid`` is in Zenodo or BTS (``crossmatch_known_object``:
RJ23 removed bogus with a cross-match to a known SN, VS or AGN); its sources disagree on
its class (``label_conflict``; a CD21 Bogus label never claims against a known object, so
the known object keeps its class); its detection list is missing (pending) or failed
for good; an Asteroid with ``ndet != 1`` or a Bogus with ``ndet >= 10`` (RJ23's cuts, on
the stored list before ``JD_MAX``); no eligible detection (``no_stamp_before_jd_max``); the
drawn alert is missing (pending), failed, unreadable, has invalid candidate fields, or is
not a full 63x63 stamp (``not_63x63``: RJ23 dropped non-square stamps; no redraw); else
**accepted**.

A walk stops once ``accepted + pending`` reaches the target, so the next fetch asks for just
enough; the walk is complete when nothing before that point is pending.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from astro_simclr.alerts.alerce import (
    API_URL,
    AVRO_URL,
    DetectionRecord,
    ResponseError,
    avro_url,
    detections_url,
    parse_detection_records,
)
from astro_simclr.alerts.by_oid import FetchItem, final_statuses
from astro_simclr.alerts.parse import AlertFormatError, parse_alert
from astro_simclr.alerts.store import Snapshot, latest_rows, read_index, read_meta
from astro_simclr.data.raw_alerts import ZTF_FIDS
from astro_simclr.data.schema import LABEL_NAMES as CD21_LABEL_NAMES
from astro_simclr.data.schema import SPLITS, DataError
from astro_simclr.r5.recipe import (
    BOGUS_MAX_NDET,
    JD_MAX,
    SEED,
    SN_TARGET,
    SUBCLASSES,
    Draw,
    bts_subclass,
    draw_alert,
    ndet,
    stable_hash,
)
from astro_simclr.r5.sources import BTS, ZENODO, SourceRow

__all__ = [
    "CAUSES",
    "AlertInfo",
    "Candidate",
    "Cd21Row",
    "Decision",
    "DetectionInfo",
    "Inputs",
    "Responses",
    "Selection",
    "evaluate",
    "read_cd21",
]

#: Every reason a candidate (or a source row) is not in R5, in report order.
CAUSES: Final = (
    "not_an_r5_subclass",
    "duplicate_oid",
    "cd21_ghost",
    "cd21_unresolved",
    "label_conflict",
    "crossmatch_known_object",
    "detections_unavailable",
    "detections_unreadable",
    "asteroid_ndet_not_1",
    "bogus_ndet_ge_10",
    "no_stamp_before_jd_max",
    "avro_unavailable",
    "avro_unreadable",
    "alert_fields_invalid",
    "not_63x63",
    "cd21_train_and_eval",
    "group_label_conflict",
)


@dataclass(frozen=True, slots=True)
class Cd21Row:
    """One row of task 3.1's store of CD21: its split, row, label, status, oid and position.

    ``ra`` and ``dec`` are the row's de-standardised position (``target_ra``,
    ``target_dec``), known for every row, ghosts included; ``oid`` is set for ``ok`` rows.
    """

    split: str
    row: int
    label: str
    status: str
    oid: str
    candid: int | None
    ra: float
    dec: float


def read_cd21(raw_dir: Path) -> tuple[str, list[Cd21Row]]:
    """The dataset name and every row of a ``fetch-alerts`` store with all three splits.

    Raises:
        DataError: if a split is missing or its ``fetch.json`` or index is invalid.
    """
    rows: list[Cd21Row] = []
    names: set[str] = set()
    for split in SPLITS:
        split_dir = raw_dir / split
        meta = read_meta(split_dir)
        names.add(str(meta.get("dataset")))
        for line in sorted(latest_rows(read_index(split_dir)).values(), key=lambda r: r.row):
            if not 0 <= line.label < len(CD21_LABEL_NAMES):
                raise DataError(f"{split_dir}: row {line.row} has label {line.label}")
            rows.append(
                Cd21Row(
                    split=split,
                    row=line.row,
                    label=CD21_LABEL_NAMES[line.label],
                    status=line.status,
                    oid=line.oid if line.status == "ok" else "",
                    candid=line.candid if line.status == "ok" else None,
                    ra=line.target_ra,
                    dec=line.target_dec,
                )
            )
    if len(names) != 1:
        raise DataError(f"{raw_dir}: the splits come from different datasets {sorted(names)}")
    return names.pop(), rows


@dataclass(frozen=True, slots=True)
class Inputs:
    """What the selection starts from: the two source tables and CD21's rows."""

    zenodo: Sequence[SourceRow]
    bts: Sequence[SourceRow]
    cd21_name: str
    cd21: Sequence[Cd21Row]


class Responses:
    """The stored responses of several snapshot directories, by URL, and the by-oid failures.

    ``dirs`` are searched in order (the first that holds a URL serves it), then
    ``fetched``, the by-oid store, whose final failure statuses (``not_found``,
    ``refused``, ``bad_alert``) are also reported by :meth:`failure`.
    """

    def __init__(self, dirs: Sequence[Path], fetched: Path | None) -> None:
        every = [*dirs, *([] if fetched is None else [fetched])]
        self._snapshots = [Snapshot(d) for d in every if d.is_dir()]
        statuses = final_statuses(fetched) if fetched is not None and fetched.is_dir() else {}
        self._failures = {item: s for item, s in statuses.items() if s != "ok"}

    def __contains__(self, url: object) -> bool:
        return any(url in s for s in self._snapshots)

    def iter_many(self, urls: Iterable[str]) -> Iterator[tuple[str, bytes, str]]:
        """(url, body, SHA-256) for the URLs some snapshot holds, the first one serving.

        Streams shard by shard (:meth:`~astro_simclr.alerts.store.Snapshot.iter_many`), so
        a caller that parses each body as it comes holds one shard at a time.
        """
        left = list(dict.fromkeys(urls))
        for snapshot in self._snapshots:
            here = [u for u in left if u in snapshot]
            for url, body, _, sha in snapshot.iter_many(here):
                yield url, body, sha
            served = set(here)
            left = [u for u in left if u not in served]

    def failure(self, item: FetchItem) -> str | None:
        """The final failure status of ``item`` in the by-oid store, if any."""
        return self._failures.get(item)


@dataclass(frozen=True, slots=True)
class DetectionInfo:
    """What the selection keeps of one detection list: its hash, ndet, eligible draw."""

    sha256: str
    ndet: int
    draw: Draw | None
    records: tuple[DetectionRecord, ...]
    error: str = ""


@dataclass(frozen=True, slots=True)
class AlertInfo:
    """What the selection keeps of one avro alert (the cutouts are read again at write time)."""

    sha256: str
    raw_shape: tuple[int, int] | None
    jd: float
    fid: int
    ra: float
    dec: float
    error: str = ""

    @property
    def full(self) -> bool:
        """True for a readable alert with a full 63x63 stamp and valid fields."""
        return not self.error and self.raw_shape == (63, 63)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One object a subclass may take, with where its label comes from."""

    oid: str
    label: str
    subclass: str
    label_source: str
    source_dataset: str
    source_row: int


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome for one visited candidate: ``accepted``, ``dropped`` (with a cause) or
    ``pending`` (``needs`` the items to fetch)."""

    candidate: Candidate
    status: str
    cause: str = ""
    detections: DetectionInfo | None = None
    alert: AlertInfo | None = None
    needs: tuple[FetchItem, ...] = ()


@dataclass
class Selection:
    """The result of :func:`evaluate`.

    Attributes:
        decisions: per walk (an AGN or VS subclass, ``SN``, ``Asteroid``, ``Bogus``), the
            visited candidates in walk order.
        available: per subclass name, the candidates it had (before any cut).
        excluded: (label, subclass, cause) -> count of source rows that never became a
            candidate (not an R5 subclass, a repeated oid, a CD21 ghost or unresolved row).
        missing: the items to fetch before the walks can complete (sorted).
        cd21_splits: oid -> the CD21 splits that hold it (for the cross-benchmark rule).
        lookup: the parsed responses the walks used (the first-stamp walk reuses it).
    """

    decisions: dict[str, list[Decision]] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    excluded: dict[tuple[str, str, str], int] = field(default_factory=dict)
    missing: list[FetchItem] = field(default_factory=list)
    cd21_splits: dict[str, set[str]] = field(default_factory=dict)
    lookup: _Lookup | None = None

    @property
    def complete(self) -> bool:
        """True when nothing is missing."""
        return not self.missing

    def accepted(self) -> list[Decision]:
        """The accepted decisions, by subclass in :data:`SUBCLASSES` order, then walk order."""
        return [d for ds in self.decisions.values() for d in ds if d.status == "accepted"]


class _Cache:
    """Parsed responses by URL, kept across the rounds of one build (bodies are dropped)."""

    def __init__(self) -> None:
        self.detections: dict[str, DetectionInfo] = {}
        self.alerts: dict[str, AlertInfo] = {}


def _detection_info(oid: str, body: bytes, sha: str, seed: int, jd_max: float) -> DetectionInfo:
    try:
        records = parse_detection_records(body, oid).records
    except ResponseError as err:
        return DetectionInfo(sha, 0, None, (), error=str(err))
    return DetectionInfo(
        sha, ndet(records, jd_max), draw_alert(oid, records, seed=seed, jd_max=jd_max), records
    )


def _alert_info(oid: str, candid: int, body: bytes, sha: str) -> AlertInfo:
    try:
        alert = parse_alert(body, oid=oid, candid=candid)
    except AlertFormatError as err:
        return AlertInfo(sha, None, 0.0, 0, 0.0, 0.0, error=f"unreadable: {err}")
    c = alert.candidate
    jd, fid, ra, dec = c.get("jd"), c.get("fid"), c.get("ra"), c.get("dec")
    assert isinstance(jd, float) and isinstance(fid, int)  # noqa: S101 - parse_alert checks
    assert isinstance(ra, float) and isinstance(dec, float)  # noqa: S101 - parse_alert checks
    valid = np.isfinite(jd) and jd > 0 and fid in ZTF_FIDS and 0 <= ra < 360 and -90 <= dec <= 90
    return AlertInfo(
        sha, alert.raw_shape, jd, fid, ra, dec, error="" if valid else "invalid fields"
    )


class _Lookup:
    """Detections and alerts of the snapshot, parsed once (bulk), for one evaluation."""

    def __init__(
        self, responses: Responses, cache: _Cache, seed: int, jd_max: float, api: str, avro: str
    ) -> None:
        self.responses = responses
        self.cache = cache
        self.seed, self.jd_max, self.api, self.avro = seed, jd_max, api, avro

    def load_detections(self, oids: Iterable[str]) -> None:
        urls = {detections_url(o, self.api): o for o in oids}
        todo = [u for u in urls if u not in self.cache.detections]
        for url, body, sha in self.responses.iter_many(todo):
            info = _detection_info(urls[url], body, sha, self.seed, self.jd_max)
            self.cache.detections[url] = info

    def load_alerts(self, pairs: Iterable[tuple[str, int]]) -> None:
        urls = {avro_url(o, c, self.avro): (o, c) for o, c in pairs}
        todo = [u for u in urls if u not in self.cache.alerts]
        for url, body, sha in self.responses.iter_many(todo):
            oid, candid = urls[url]
            self.cache.alerts[url] = _alert_info(oid, candid, body, sha)

    def detections(self, oid: str) -> DetectionInfo | str | None:
        """The parsed list, a final failure status, or ``None`` (not fetched yet)."""
        info = self.cache.detections.get(detections_url(oid, self.api))
        if info is not None:
            return info
        return self.responses.failure(FetchItem("detections", oid))

    def alert(self, oid: str, candid: int) -> AlertInfo | str | None:
        """The parsed alert, a final failure status, or ``None`` (not fetched yet)."""
        info = self.cache.alerts.get(avro_url(oid, candid, self.avro))
        if info is not None:
            return info
        return self.responses.failure(FetchItem("avro", oid, candid))


def _permuted(candidates: list[Candidate], key: str, seed: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda c: c.oid)
    rng = np.random.default_rng([seed, stable_hash(f"sample:{key}")])
    return [ordered[i] for i in rng.permutation(len(ordered))]


@dataclass
class _Pools:
    """Candidates per subclass name, the claims and the exclusions, from the inputs."""

    by_subclass: dict[str, list[Candidate]] = field(default_factory=dict)
    claims: dict[str, set[str]] = field(default_factory=dict)
    known: set[str] = field(default_factory=set)
    excluded: dict[tuple[str, str, str], int] = field(default_factory=dict)
    cd21_splits: dict[str, set[str]] = field(default_factory=dict)

    def exclude(self, label: str, subclass: str, cause: str) -> None:
        key = (label, subclass, cause)
        self.excluded[key] = self.excluded.get(key, 0) + 1

    def claim(self, oid: str, label: str) -> None:
        self.claims.setdefault(oid, set()).add(label)


#: Zenodo class -> (R5 label, subclass name).
_ZENODO_MAP: Final = {
    cls: (s.label, s.name) for s in SUBCLASSES if s.source != "cd21" for cls in s.source_classes
}


def _zenodo_label(cls: str) -> str:
    """The R5 class a Zenodo class claims (``other`` for a class R5 does not use)."""
    return _ZENODO_MAP[cls][0] if cls in _ZENODO_MAP else "other"


def _add_zenodo(pools: _Pools, rows: Iterable[SourceRow]) -> set[str]:
    """Zenodo's candidates; returns the oids of its SN (BTS repeats of them are dropped)."""
    seen: set[str] = set()
    sn: set[str] = set()
    for row in rows:
        pools.claim(row.oid, _zenodo_label(row.cls))
        pools.known.add(row.oid)
        if row.cls not in _ZENODO_MAP:
            pools.exclude("other", row.cls, "not_an_r5_subclass")
            continue
        label, sub = _ZENODO_MAP[row.cls]
        if row.oid in seen:
            pools.exclude(label, sub, "duplicate_oid")
            continue
        seen.add(row.oid)
        if label == "SN":
            sn.add(row.oid)
        pools.by_subclass[sub].append(Candidate(row.oid, label, sub, ZENODO, ZENODO, row.row))
    return sn


def _add_bts(pools: _Pools, rows: Iterable[SourceRow], zenodo_sn: set[str]) -> None:
    seen: set[str] = set()
    for row in rows:
        sub = bts_subclass(row.cls)
        pools.known.add(row.oid)
        if sub is None:
            pools.claim(row.oid, "other")
            pools.exclude("other", row.cls or "?", "not_an_r5_subclass")
            continue
        pools.claim(row.oid, "SN")
        if row.oid in seen or row.oid in zenodo_sn:
            # Already a candidate (a Zenodo SN or an earlier BTS row): the first one stands.
            pools.exclude("SN", sub, "duplicate_oid")
            continue
        seen.add(row.oid)
        pools.by_subclass[sub].append(Candidate(row.oid, "SN", sub, BTS, BTS, row.row))


def _add_cd21(pools: _Pools, name: str, rows: Iterable[Cd21Row]) -> None:
    seen: set[str] = set()
    for r in rows:
        if r.oid:
            # A CD21 Bogus label does not claim against a known object: RJ23's cross-match
            # cut drops the bogus and keeps the object (``crossmatch_known_object``).
            if r.label != "Bogus":
                pools.claim(r.oid, r.label)
            pools.cd21_splits.setdefault(r.oid, set()).add(r.split)
        if r.label not in ("Asteroid", "Bogus"):
            continue
        if r.status == "ghost_candidate":
            pools.exclude(r.label, r.label, "cd21_ghost")
        elif r.status != "ok":
            pools.exclude(r.label, r.label, "cd21_unresolved")
        elif r.oid in seen:
            pools.exclude(r.label, r.label, "duplicate_oid")
        else:
            seen.add(r.oid)
            candidate = Candidate(r.oid, r.label, r.label, "cd21", f"{name}/{r.split}", r.row)
            pools.by_subclass[r.label].append(candidate)


def _pools(inputs: Inputs) -> _Pools:
    pools = _Pools(by_subclass={s.name: [] for s in SUBCLASSES})
    zenodo_sn = _add_zenodo(pools, inputs.zenodo)
    _add_bts(pools, inputs.bts, zenodo_sn)
    _add_cd21(pools, inputs.cd21_name, inputs.cd21)
    return pools


def _detections_cause(c: Candidate, found: DetectionInfo) -> str:
    """The cut a candidate's detection list fails, or ``""``."""
    if found.error:
        return "detections_unreadable"
    if c.label == "Asteroid" and found.ndet != 1:
        return "asteroid_ndet_not_1"
    if c.label == "Bogus" and found.ndet >= BOGUS_MAX_NDET:
        return "bogus_ndet_ge_10"
    if found.draw is None:
        return "no_stamp_before_jd_max"
    return ""


def _alert_cause(alert: AlertInfo) -> str:
    """Why a drawn alert is refused, or ``""``."""
    if alert.error.startswith("unreadable"):
        return "avro_unreadable"
    if alert.error:
        return "alert_fields_invalid"
    if alert.raw_shape != (63, 63):
        return "not_63x63"
    return ""


def _decide(c: Candidate, lookup: _Lookup, pools: _Pools) -> Decision:
    if c.label == "Bogus" and c.oid in pools.known:
        return Decision(c, "dropped", "crossmatch_known_object")
    if len(pools.claims.get(c.oid, set()) | {c.label}) > 1:
        return Decision(c, "dropped", "label_conflict")
    found = lookup.detections(c.oid)
    if found is None:
        return Decision(c, "pending", needs=(FetchItem("detections", c.oid),))
    if isinstance(found, str):
        return Decision(c, "dropped", "detections_unavailable")
    cause = _detections_cause(c, found)
    if cause or found.draw is None:
        return Decision(c, "dropped", cause, detections=found)
    candid = found.draw.record.candid
    alert = lookup.alert(c.oid, candid)
    if alert is None:
        return Decision(c, "pending", detections=found, needs=(FetchItem("avro", c.oid, candid),))
    if isinstance(alert, str):
        return Decision(c, "dropped", "avro_unavailable", detections=found)
    cause = _alert_cause(alert)
    status = "dropped" if cause else "accepted"
    return Decision(c, status, cause, detections=found, alert=alert)


def _walk(
    order: Sequence[Candidate], target: int | None, decide: Callable[[Candidate], Decision]
) -> list[Decision]:
    decisions: list[Decision] = []
    accepted = pending = 0
    for candidate in order:
        if target is not None and accepted + pending >= target:
            break
        decision = decide(candidate)
        decisions.append(decision)
        accepted += decision.status == "accepted"
        pending += decision.status == "pending"
    return decisions


def _orders(pools: _Pools, seed: int) -> list[tuple[str, list[Candidate], int | None]]:
    """(walk name, candidates in walk order, target) per walk."""
    walks: list[tuple[str, list[Candidate], int | None]] = []
    for s in SUBCLASSES:
        if s.label in ("AGN", "VS"):
            walks.append((s.name, _permuted(pools.by_subclass[s.name], s.name, seed), s.target))
    sn = [s.name for s in SUBCLASSES if s.label == "SN"]
    zenodo_sn = [c for n in sn for c in pools.by_subclass[n] if c.label_source == ZENODO]
    bts_sn = [c for n in sn for c in pools.by_subclass[n] if c.label_source == BTS]
    walks.append(
        (
            "SN",
            _permuted(zenodo_sn, "SN/zenodo", seed) + _permuted(bts_sn, "SN/bts", seed),
            SN_TARGET,
        )
    )
    for name in ("Asteroid", "Bogus"):
        walks.append((name, sorted(pools.by_subclass[name], key=lambda c: c.oid), None))
    return walks


def evaluate(
    inputs: Inputs,
    responses: Responses,
    *,
    seed: int = SEED,
    jd_max: float = JD_MAX,
    cache: _Cache | None = None,
    api: str = API_URL,
    avro: str = AVRO_URL,
) -> Selection:
    """Run every walk on the snapshot (module docstring); a pure function of its inputs.

    ``cache`` (an internal object, reused across the rounds of one build) only saves
    parsing time; it never changes the result.
    """
    pools = _pools(inputs)
    lookup = _Lookup(responses, cache or _Cache(), seed, jd_max, api, avro)
    walks = _orders(pools, seed)
    everyone = [c for _, order, _ in walks for c in order]
    lookup.load_detections(c.oid for c in everyone)
    pairs = []
    for c in everyone:
        info = lookup.detections(c.oid)
        if isinstance(info, DetectionInfo) and info.draw is not None:
            pairs.append((c.oid, info.draw.record.candid))
    lookup.load_alerts(pairs)
    selection = Selection(
        available={name: len(pool) for name, pool in pools.by_subclass.items()},
        excluded=dict(pools.excluded),
        cd21_splits=pools.cd21_splits,
    )
    missing: dict[FetchItem, None] = {}
    for name, order, target in walks:
        decisions = _walk(order, target, lambda c: _decide(c, lookup, pools))
        selection.decisions[name] = decisions
        for d in decisions:
            missing.update(dict.fromkeys(d.needs))
    selection.missing = sorted(missing, key=FetchItem.sort_key)
    selection.lookup = lookup
    return selection
