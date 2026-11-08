"""Build R5: the selection, its grouped splits, the first-stamp view and the frozen outputs.

Task 3.5 of ``docs/phases/03-newer-benchmarks.md``; ADR 0020 has the reasons. One build:

1. **Select** (:func:`astro_simclr.r5.select.evaluate`) on the response snapshots; online,
   fetch what is missing (:func:`~astro_simclr.alerts.by_oid.fetch_by_oid`, the polite
   client of task 3.1) and repeat until nothing is (``max_rounds``); offline
   (``--from-snapshot``), refuse an incomplete snapshot.
2. **Group** the accepted objects together with **every** CD21 row (task 3.1's store,
   ghosts included): rows with the same ``oid``, or within 1.5" of each other (union-find
   across both benchmarks, :mod:`astro_simclr.protocol.groups`), are one group. An R5 group
   that holds a CD21 Train row *and* a CD21 Validation or Test row cannot satisfy the
   cross-benchmark rule (CD21 itself leaks there) and is dropped (``cd21_train_and_eval``);
   a group whose R5 objects carry two classes is dropped (``group_label_conflict``).
3. **Split** (:func:`~astro_simclr.protocol.object_splits.make_object_splits`, RJ23's
   60/20/20 by class, grouped) with the constraints of the 3.2 matrix: a group holding a
   CD21 Train row may only go to R5 Train; one holding a CD21 Validation or Test row only to
   R5 Validation or Test. :func:`~astro_simclr.protocol.grouping.assert_no_shared_groups`
   then checks the 3x3 matrix, which is recorded.
4. **First stamps** of R5 Test (RJ23's second view): the earliest eligible detection whose
   alert is a full 63x63 stamp (online, fetched as in step 1).
5. **Write** (all deterministic: same snapshot and code, same bytes)::

       OUT/
           r5/                 raw-alerts layout (format 3): one drawn alert per object
           r5-first/           raw-alerts layout: R5 Test's first stamps
           alerts.csv          the alert table of r5/ (oid, candid, jd, label, group)
           first-alerts.csv    the alert table of r5-first/
           splits.json         object splits of alerts.csv (with their constraints)
           index.csv           the frozen index: one line per R5 object (columns below)
           deviations.csv      target vs achieved per class and subclass, with the causes
           drops.csv           counts per class, subclass and cause
           eras.csv            class x year of the drawn alert
           split_counts.csv    class x split, objects
           build.json          parameters, input and output SHA-256, counts, the matrix

   ``index.csv`` columns: :data:`INDEX_COLUMNS`. Nothing written lists wall times or
   request counts, so an offline rebuild of the same snapshot reproduces every file byte for
   byte (:func:`check_expected` compares ``build.json``).

The layouts, ``alerts.csv``, ``splits.json`` and ``index.csv`` list the label set's objects
and stay out of git (ADR 0012); the tables and ``build.json`` hold counts and hashes only.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.alerts.alerce import AlerceClient, avro_url
from astro_simclr.alerts.by_oid import FetchItem, ItemLine, OidRunReport, fetch_by_oid
from astro_simclr.alerts.parse import parse_alert
from astro_simclr.data.files import sha256_file
from astro_simclr.data.raw_alerts import write_raw_alerts
from astro_simclr.data.schema import SPLITS, STAMP_SHAPE, DataError
from astro_simclr.protocol.alerts import AlertTable, write_alert_table
from astro_simclr.protocol.grouping import SPLIT_NAMES, assert_no_shared_groups
from astro_simclr.protocol.groups import neighbour_pairs, union_groups
from astro_simclr.protocol.object_splits import (
    ObjectSplits,
    make_object_splits,
    write_object_splits,
)
from astro_simclr.r5.recipe import (
    JD_MAX,
    JD_MAX_UTC,
    LABEL_NAMES,
    SEED,
    SN_TARGET,
    SUBCLASSES,
    first_stamp_order,
    year_of_jd,
)
from astro_simclr.r5.select import (
    CAUSES,
    AlertInfo,
    Decision,
    Inputs,
    Responses,
    Selection,
    _Cache,
    _Lookup,
    evaluate,
    read_cd21,
)
from astro_simclr.r5.sources import BTS, ZENODO, load_source, read_bts, read_zenodo

__all__ = [
    "INDEX_COLUMNS",
    "BuildResult",
    "Built",
    "build_r5",
    "check_expected",
    "load_inputs",
]

#: Radius of the cross-benchmark position groups (the 3.2 matrix), arcsec.
RADIUS_ARCSEC: Final = 1.5
#: Stream of the split draw (``make_object_splits``' seed is ``seed``; recorded).
BUILD_NAME: Final = "R5"
#: Columns of the frozen ``index.csv``.
INDEX_COLUMNS: Final = (
    "oid",
    "candid",
    "jd",
    "year",
    "label",
    "subclass",
    "label_source",
    "source_dataset",
    "source_row",
    "group",
    "split",
    "half",
    "ndet",
    "n_eligible",
    "draw_index",
    "first_candid",
    "detections_sha256",
    "avro_sha256",
    "first_avro_sha256",
    "cd21_splits",
)


def load_inputs(sources: Path, cd21_raw: Path) -> tuple[Inputs, dict[str, Any]]:
    """The selection's inputs and their provenance (the source records, CD21's index hashes).

    Raises:
        DataError: if a source is missing or changed, or CD21's store is invalid.
    """
    zenodo_path, zenodo_record = load_source(sources, ZENODO)
    bts_path, bts_record = load_source(sources, BTS)
    name, cd21 = read_cd21(cd21_raw)
    provenance = {
        "sources": {ZENODO: zenodo_record, BTS: bts_record},
        "cd21": {
            "dataset": name,
            "index_sha256": {s: sha256_file(cd21_raw / s / "index.csv") for s in SPLITS},
        },
    }
    inputs = Inputs(read_zenodo(zenodo_path), read_bts(bts_path), name, cd21)
    return inputs, provenance


@dataclass(frozen=True, slots=True)
class _Row:
    """One accepted R5 object, with what the outputs need."""

    decision: Decision
    alert: AlertInfo


@dataclass
class Built:
    """The selection with its groups, splits and first stamps (the outputs' content)."""

    selection: Selection
    rows: list[_Row]
    groups: list[str]
    dropped: dict[tuple[str, str, str], int]
    splits: ObjectSplits
    table: AlertTable
    matrix: list[list[int]]
    first: dict[str, tuple[int, AlertInfo]] = field(default_factory=dict)
    missing: list[FetchItem] = field(default_factory=list)


def _accepted_rows(selection: Selection) -> list[_Row]:
    rows = []
    for d in selection.accepted():
        assert d.alert is not None  # noqa: S101 - an accepted decision has its alert
        rows.append(_Row(d, d.alert))
    return sorted(
        rows,
        key=lambda r: (LABEL_NAMES.index(r.decision.candidate.label), r.decision.candidate.oid),
    )


def _components(rows: Sequence[_Row], cd21: Inputs) -> npt.NDArray[np.int64]:
    """Union-find over R5 rows then CD21 rows: same oid, or within :data:`RADIUS_ARCSEC`."""
    oids = [r.decision.candidate.oid for r in rows] + [c.oid for c in cd21.cd21]
    ra = np.asarray([r.alert.ra for r in rows] + [c.ra for c in cd21.cd21], dtype=np.float64)
    dec = np.asarray([r.alert.dec for r in rows] + [c.dec for c in cd21.cd21], dtype=np.float64)
    first, second = neighbour_pairs(ra, dec, RADIUS_ARCSEC)
    seen: dict[str, int] = {}
    extra_a: list[int] = []
    extra_b: list[int] = []
    for i, oid in enumerate(oids):
        if not oid:
            continue
        if oid in seen:
            extra_a.append(seen[oid])
            extra_b.append(i)
        else:
            seen[oid] = i
    a = np.concatenate([first, np.asarray(extra_a, dtype=np.int64)])
    b = np.concatenate([second, np.asarray(extra_b, dtype=np.int64)])
    return union_groups(len(oids), a, b)


def _count(store: dict[tuple[str, str, str], int], key: tuple[str, str, str]) -> None:
    store[key] = store.get(key, 0) + 1


def _group(
    rows: list[_Row], inputs: Inputs
) -> tuple[
    list[_Row], list[str], dict[str, tuple[str, ...]], list[str], dict[tuple[str, str, str], int]
]:
    """Kept rows, their group ids, the constraints, CD21's group per row, and the drops."""
    comp = _components(rows, inputs)
    n = len(rows)
    r5_comp, cd21_comp = comp[:n], comp[n:]
    names: dict[int, str] = {}
    labels: dict[int, set[str]] = {}
    for row, c in zip(rows, r5_comp.tolist(), strict=True):
        oid = row.decision.candidate.oid
        names[c] = min(names.get(c, oid), oid)
        labels.setdefault(c, set()).add(row.decision.candidate.label)
    splits: dict[int, set[str]] = {}
    for c21, c in zip(inputs.cd21, cd21_comp.tolist(), strict=True):
        splits.setdefault(c, set()).add(c21.split)
    dropped: dict[tuple[str, str, str], int] = {}
    kept: list[_Row] = []
    groups: list[str] = []
    for row, c in zip(rows, r5_comp.tolist(), strict=True):
        where = splits.get(c, set())
        cand = row.decision.candidate
        if "Train" in where and where & {"Validation", "Test"}:
            _count(dropped, (cand.label, cand.subclass, "cd21_train_and_eval"))
        elif len(labels[c]) > 1:
            _count(dropped, (cand.label, cand.subclass, "group_label_conflict"))
        else:
            kept.append(row)
            groups.append(names[c])
    allowed: dict[str, tuple[str, ...]] = {}
    for c, where in splits.items():
        if c in names and where:
            allowed[names[c]] = ("Train",) if "Train" in where else ("Validation", "Test")
    cd21_groups = [names.get(c, f"cd21-{c}") for c in cd21_comp.tolist()]
    return kept, groups, allowed, cd21_groups, dropped


def _split(
    selection: Selection, inputs: Inputs, seed: int
) -> tuple[
    list[_Row],
    list[str],
    dict[tuple[str, str, str], int],
    ObjectSplits,
    AlertTable,
    list[list[int]],
]:
    rows, groups, allowed, cd21_groups, dropped = _group(_accepted_rows(selection), inputs)
    if not rows:
        raise DataError("R5 has no objects: nothing was accepted")
    table = AlertTable.from_rows(
        [r.decision.candidate.oid for r in rows],
        [_candid(r) for r in rows],
        [r.alert.jd for r in rows],
        [LABEL_NAMES.index(r.decision.candidate.label) for r in rows],
        LABEL_NAMES,
        groups,
    )
    splits = make_object_splits(table, seed=seed, name=BUILD_NAME, allowed=allowed)
    r5_split = splits.split_of(table)
    cd21_split = [SPLIT_NAMES.index(c.split) for c in inputs.cd21]
    matrix = assert_no_shared_groups(table.group, r5_split, cd21_groups, cd21_split)
    return rows, groups, dropped, splits, table, matrix.tolist()


def _candid(row: _Row) -> int:
    detections = row.decision.detections
    assert detections is not None and detections.draw is not None  # noqa: S101 - accepted
    return detections.draw.record.candid


def _first_stamps(
    built_rows: Sequence[_Row], test_oids: set[str], lookup: _Lookup
) -> tuple[dict[str, tuple[int, AlertInfo]], list[FetchItem]]:
    """Per Test object, its first full stamp (candid, alert), and the avros still missing."""
    walks = {}
    for row in built_rows:
        oid = row.decision.candidate.oid
        if oid in test_oids:
            assert row.decision.detections is not None  # noqa: S101 - accepted
            walks[oid] = (row, first_stamp_order(row.decision.detections.records))
    lookup.load_alerts((oid, r.candid) for oid, (_, order) in walks.items() for r in order)
    first: dict[str, tuple[int, AlertInfo]] = {}
    missing: list[FetchItem] = []
    for oid, (row, order) in sorted(walks.items()):
        drawn = _candid(row)
        for record in order:
            if record.candid == drawn:
                first[oid] = (drawn, row.alert)
                break
            alert = lookup.alert(oid, record.candid)
            if alert is None:
                missing.append(FetchItem("avro", oid, record.candid))
                break
            if isinstance(alert, AlertInfo) and alert.full:
                first[oid] = (record.candid, alert)
                break
    return first, missing


def assemble(selection: Selection, inputs: Inputs, seed: int) -> Built:
    """Steps 2-4 of the module docstring on a complete selection."""
    rows, groups, dropped, splits, table, matrix = _split(selection, inputs, seed)
    test = set(table.oid[splits.rows(table, "Test")].tolist())
    assert selection.lookup is not None  # noqa: S101 - set by evaluate
    first, missing = _first_stamps(rows, test, selection.lookup)
    return Built(selection, rows, groups, dropped, splits, table, matrix, first, missing)


# -- writing ------------------------------------------------------------------------------


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if v is None else repr(v) if isinstance(v, float) else v for v in row])
    return buffer.getvalue().encode("utf-8")


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _cutouts(
    out: Path, name: str, wanted: Sequence[tuple[str, int]], responses: Responses
) -> npt.NDArray[np.float32]:
    """The raw cutouts of ``wanted`` (oid, candid), parsed again from the snapshot's avro.

    Held in a memory map next to the output, so a 50k-object build does not need 2.4 GB of
    RAM.
    """
    path = out / f".{name}-cutouts.npy"
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(wanted), *STAMP_SHAPE)
    )
    position = {avro_url(oid, candid): i for i, (oid, candid) in enumerate(wanted)}
    done = np.zeros(len(wanted), dtype=bool)
    for url, body, _ in responses.iter_many(position):
        i = position[url]
        oid, candid = wanted[i]
        array[i] = parse_alert(body, oid=oid, candid=candid).cutouts
        done[i] = True
    if not done.all():
        oid, candid = wanted[int(np.flatnonzero(~done)[0])]
        raise DataError(f"the avro of {oid}:{candid} is no longer in the snapshot")
    array.flush()
    return array


def _layout_index(
    entries: Sequence[tuple[Decision, int, AlertInfo]], dataset: str | None
) -> dict[str, npt.NDArray[Any]]:
    columns: dict[str, list[Any]] = {
        k: []
        for k in (
            "oid",
            "candid",
            "jd",
            "fid",
            "ra",
            "dec",
            "label",
            "label_source",
            "source_dataset",
            "source_row",
        )
    }
    for decision, candid, alert in entries:
        c = decision.candidate
        columns["oid"].append(c.oid)
        columns["candid"].append(candid)
        columns["jd"].append(alert.jd)
        columns["fid"].append(alert.fid)
        columns["ra"].append(alert.ra)
        columns["dec"].append(alert.dec)
        columns["label"].append(LABEL_NAMES.index(c.label))
        columns["label_source"].append(c.label_source)
        columns["source_dataset"].append(dataset or c.source_dataset)
        columns["source_row"].append(c.source_row)
    return {
        k: np.asarray(v, dtype=np.int64 if k in ("candid", "fid", "label", "source_row") else None)
        for k, v in columns.items()
    }


def _write_layout(
    out: Path,
    name: str,
    entries: Sequence[tuple[Decision, int, AlertInfo]],
    responses: Responses,
    source: dict[str, Any],
    dataset: str | None = None,
) -> None:
    wanted = [(d.candidate.oid, candid) for d, candid, _ in entries]
    cutouts = _cutouts(out, name, wanted, responses)
    try:
        write_raw_alerts(out / name, cutouts, _layout_index(entries, dataset), LABEL_NAMES, source)
    finally:
        del cutouts
        (out / f".{name}-cutouts.npy").unlink(missing_ok=True)


def _index_rows(built: Built) -> list[list[object]]:
    split_of = dict(
        zip(built.table.oid.tolist(), built.splits.split_of(built.table).tolist(), strict=True)
    )
    half_of = dict(
        zip(built.table.oid.tolist(), built.splits.half_of(built.table).tolist(), strict=True)
    )
    halves = ("selection", "decision")
    rows: list[list[object]] = []
    for row, group in zip(built.rows, built.groups, strict=True):
        d, c = row.decision, row.decision.candidate
        assert d.detections is not None and d.detections.draw is not None  # noqa: S101
        first = built.first.get(c.oid)
        part, half = split_of[c.oid], half_of[c.oid]
        rows.append(
            [
                c.oid,
                d.detections.draw.record.candid,
                row.alert.jd,
                year_of_jd(row.alert.jd),
                c.label,
                c.subclass,
                c.label_source,
                c.source_dataset,
                c.source_row,
                group,
                SPLIT_NAMES[part] if part >= 0 else "",
                halves[half] if half >= 0 else "",
                d.detections.ndet,
                d.detections.draw.n_eligible,
                d.detections.draw.index,
                first[0] if first else None,
                d.detections.sha256,
                row.alert.sha256,
                first[1].sha256 if first else None,
                " ".join(sorted(built.selection.cd21_splits.get(c.oid, set()))),
            ]
        )
    return rows


def _drops(built: Built) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = dict(built.selection.excluded)
    for decisions in built.selection.decisions.values():
        for d in decisions:
            if d.status == "dropped":
                _count(counts, (d.candidate.label, d.candidate.subclass, d.cause))
    for key, n in built.dropped.items():
        counts[key] = counts.get(key, 0) + n
    return counts


def _deviation_rows(built: Built, drops: dict[tuple[str, str, str], int]) -> list[list[object]]:
    achieved = Counter(
        (r.decision.candidate.label, r.decision.candidate.subclass) for r in built.rows
    )
    available = {(s.label, s.name): built.selection.available.get(s.name, 0) for s in SUBCLASSES}
    visited: Counter[tuple[str, str]] = Counter()
    for decisions in built.selection.decisions.values():
        for d in decisions:
            visited[(d.candidate.label, d.candidate.subclass)] += 1
    rows: list[list[object]] = []
    for s in SUBCLASSES:
        key = (s.label, s.name)
        target = s.target
        causes = sorted(
            ((cause, n) for (lab, sub, cause), n in drops.items() if (lab, sub) == key and n),
            key=lambda kv: CAUSES.index(kv[0]) if kv[0] in CAUSES else len(CAUSES),
        )
        why = "; ".join(f"{cause} {n}" for cause, n in causes)
        if not s.source_classes:
            why = _join(f"absent from {s.source}", why)
        elif target is not None and available[key] < target:
            why = _join(f"source holds {available[key]} < target", why)
        rows.append(
            [
                s.label,
                s.name,
                s.source,
                target,
                s.cap,
                available[key],
                visited[key],
                achieved[key],
                None if target is None else achieved[key] - target,
                why,
            ]
        )
    for label in LABEL_NAMES:
        subs = [s for s in SUBCLASSES if s.label == label]
        target = SN_TARGET if label == "SN" else sum(s.target or 0 for s in subs)
        total = sum(achieved[(label, s.name)] for s in subs)
        rows.append(
            [
                label,
                "(all)",
                "",
                target,
                None,
                sum(available[(label, s.name)] for s in subs),
                sum(visited[(label, s.name)] for s in subs),
                total,
                total - target,
                "",
            ]
        )
    return rows


def _join(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


def _era_rows(built: Built) -> tuple[list[str], list[list[object]]]:
    years = sorted({year_of_jd(r.alert.jd) for r in built.rows})
    counts = Counter((r.decision.candidate.label, year_of_jd(r.alert.jd)) for r in built.rows)
    header = ["label", *[str(y) for y in years], "total"]
    rows: list[list[object]] = []
    for label in LABEL_NAMES:
        values = [counts[(label, y)] for y in years]
        rows.append([label, *values, sum(values)])
    return header, rows


def _split_rows(built: Built) -> list[list[object]]:
    counts = built.splits.counts
    rows: list[list[object]] = []
    for k, label in enumerate(LABEL_NAMES):
        values = [counts[name]["groups"][k] for name in SPLIT_NAMES]
        total = sum(values)
        rows.append(
            [label, *values, total, *[round(v / total, 4) if total else None for v in values]]
        )
    return rows


def write_outputs(
    built: Built, out: Path, responses: Responses, provenance: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Write every output of the module docstring into ``out`` (a new directory).

    Returns the ``build.json`` content.
    """
    out.mkdir(parents=True)
    source = {"name": BUILD_NAME, "seed": seed, "jd_max": JD_MAX, "jd_max_utc": JD_MAX_UTC}
    main = [(r.decision, _candid(r), r.alert) for r in built.rows]
    _write_layout(out, "r5", main, responses, source)
    by_oid = {r.decision.candidate.oid: r for r in built.rows}
    group_of = {r.decision.candidate.oid: g for r, g in zip(built.rows, built.groups, strict=True)}
    first = [
        (by_oid[oid].decision, candid, alert)
        for oid, (candid, alert) in sorted(built.first.items())
    ]
    if first:
        _write_layout(
            out,
            "r5-first",
            first,
            responses,
            {**source, "view": "first-stamp"},
            dataset="r5-first-stamp",
        )
        first_table = AlertTable.from_rows(
            [d.candidate.oid for d, _, _ in first],
            [candid for _, candid, _ in first],
            [a.jd for _, _, a in first],
            [LABEL_NAMES.index(d.candidate.label) for d, _, _ in first],
            LABEL_NAMES,
            [group_of[d.candidate.oid] for d, _, _ in first],
        )
        write_alert_table(first_table, out / "first-alerts.csv")
    write_alert_table(built.table, out / "alerts.csv")
    write_object_splits(built.splits, out / "splits.json")
    _write(out / "index.csv", _csv_bytes(INDEX_COLUMNS, _index_rows(built)))
    drops = _drops(built)
    _write(
        out / "deviations.csv",
        _csv_bytes(
            (
                "label",
                "subclass",
                "source",
                "target",
                "cap",
                "available",
                "visited",
                "achieved",
                "deviation",
                "cause",
            ),
            _deviation_rows(built, drops),
        ),
    )
    drop_rows = sorted(
        ([lab, sub, cause, n] for (lab, sub, cause), n in drops.items()),
        key=lambda r: (str(r[0]), str(r[1]), CAUSES.index(str(r[2])) if r[2] in CAUSES else 99),
    )
    _write(out / "drops.csv", _csv_bytes(("label", "subclass", "cause", "count"), drop_rows))
    header, eras = _era_rows(built)
    _write(out / "eras.csv", _csv_bytes(header, eras))
    _write(
        out / "split_counts.csv",
        _csv_bytes(
            ("label", *SPLIT_NAMES, "total", *[f"{s}_fraction" for s in SPLIT_NAMES]),
            _split_rows(built),
        ),
    )
    names = [
        "alerts.csv",
        "splits.json",
        "index.csv",
        "deviations.csv",
        "drops.csv",
        "eras.csv",
        "split_counts.csv",
    ]
    names += ["first-alerts.csv"] if first else []
    layouts = ["r5", *(["r5-first"] if first else [])]
    files = {name: sha256_file(out / name) for name in sorted(names)}
    for layout in layouts:
        for path in sorted((out / layout).iterdir()):
            files[f"{layout}/{path.name}"] = sha256_file(path)
    record = {
        "format": "astro-simclr-r5-build",
        "format_version": 1,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "name": BUILD_NAME,
        "parameters": {
            "seed": seed,
            "jd_max": JD_MAX,
            "jd_max_utc": JD_MAX_UTC,
            "radius_arcsec": RADIUS_ARCSEC,
            "fractions": list(built.splits.fractions),
            "sn_target": SN_TARGET,
            "subclasses": [
                {
                    "label": s.label,
                    "name": s.name,
                    "source": s.source,
                    "source_classes": list(s.source_classes),
                    "target": s.target,
                    "cap": s.cap,
                }
                for s in SUBCLASSES
            ],
        },
        "inputs": provenance,
        "counts": {
            "objects": len(built.rows),
            "first_stamps": len(first),
            "by_class": {
                label: int(sum(1 for r in built.rows if r.decision.candidate.label == label))
                for label in LABEL_NAMES
            },
            "split_counts": built.splits.counts,
        },
        "alert_table_sha256": built.table.sha256,
        "cross_benchmark": {
            "against": provenance["cd21"]["dataset"],
            "rows": "R5 split",
            "columns": "CD21 split",
            "split_names": list(SPLIT_NAMES),
            "shared_groups": built.matrix,
        },
        "files": files,
    }
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# -- driver -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuildResult:
    """What :func:`build_r5` did.

    Attributes:
        complete: true if the outputs were written.
        rounds: selection rounds run.
        missing: items still missing (when not complete).
        fetched: the by-oid runs of this call.
        record: the ``build.json`` content (when complete).
    """

    complete: bool
    rounds: int
    missing: list[FetchItem]
    fetched: list[OidRunReport]
    record: dict[str, Any] | None


def _snapshot_dirs(cd21_raw: Path) -> list[Path]:
    return [cd21_raw / s for s in SPLITS]


def build_r5(
    sources: Path,
    cd21_raw: Path,
    fetched: Path,
    out: Path | None,
    *,
    seed: int = SEED,
    client: AlerceClient | None = None,
    max_rounds: int = 20,
    batch_size: int = 100,
    cooldown: float = 300.0,
    progress: Callable[[int, int, ItemLine], None] | None = None,
    log: Callable[[str], None] = lambda _: None,
    stop: Callable[[], bool] = lambda: False,
) -> BuildResult:
    """Build R5 (module docstring). ``client`` ``None`` is the offline rebuild.

    ``out`` ``None`` is a dry run: one round, nothing fetched or written, the missing items
    reported (the next round's request count).

    Raises:
        DataError: if an input is invalid, ``out`` exists, or (offline) the snapshot lacks
            a response the build needs.
    """
    if out is not None and out.exists():
        raise DataError(f"{out} already exists; pass another --out")
    inputs, provenance = load_inputs(sources, cd21_raw)
    cache = _Cache()
    runs: list[OidRunReport] = []
    built: Built | None = None
    missing: list[FetchItem] = []
    rounds = fetches = 0
    while True:
        # Every fetch is followed by an evaluation, so the result is never stale.
        rounds += 1
        responses = Responses(_snapshot_dirs(cd21_raw), fetched)
        selection = evaluate(inputs, responses, seed=seed, cache=cache)
        missing = selection.missing
        if not missing:
            built = assemble(selection, inputs, seed)
            missing = built.missing
        log(f"round {rounds}: {len(missing)} items missing")
        if not missing or out is None or client is None or stop() or fetches >= max_rounds:
            break
        fetches += 1
        run = fetch_by_oid(
            missing,
            fetched,
            client=client,
            resume=True,
            batch_size=batch_size,
            cooldown=cooldown,
            progress=progress,
            stop=stop,
        )
        runs.append(run)
        if run.attempted == 0 or run.counts["http_error"] == run.attempted:
            # (No body arrived, so a new evaluation would find the same items missing.)
            # Nothing new can arrive this call: every item failed transiently (an outage),
            # or the store already has a final status for what the snapshot lacks.
            log("the round fetched nothing usable: stopping (run build-r5 again later)")
            break
    if missing or built is None or out is None:
        if out is not None and client is None and missing:
            raise DataError(
                f"the snapshot lacks {len(missing)} responses the build needs "
                f"(first: {missing[0]}); run build-r5 online to fetch them"
            )
        return BuildResult(False, rounds, missing, runs, None)
    partial = out.with_name(f".{out.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    record = write_outputs(
        built, partial, Responses(_snapshot_dirs(cd21_raw), fetched), provenance, seed
    )
    os.replace(partial, out)
    return BuildResult(True, rounds, [], runs, record)


def check_expected(record: dict[str, Any], expected: Path) -> list[str]:
    """The output files whose SHA-256 differs from ``expected`` (a ``build.json``).

    Raises:
        DataError: if ``expected`` cannot be read.
    """
    try:
        old = json.loads(expected.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise DataError(f"{expected}: cannot read ({err})") from None
    before, after = old.get("files", {}), record["files"]
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
