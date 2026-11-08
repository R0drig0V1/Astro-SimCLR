"""Tests of the R5 build (task 3.5) on the offline world of ``_r5_support``.

The online build fetches from the fake service; the offline rebuild must then reproduce
every output byte for byte from the snapshots alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from _r5_support import FakeService, R5World, detections_json, make_r5_world

from astro_simclr.alerts.alerce import parse_detection_records
from astro_simclr.alerts.by_oid import FetchItem, fetch_by_oid, read_oid_index
from astro_simclr.data.raw_alerts import load_raw_alerts, verify_raw_alerts
from astro_simclr.data.schema import DataError
from astro_simclr.protocol.alerts import read_alert_table
from astro_simclr.protocol.object_splits import read_object_splits, verify_object_splits
from astro_simclr.r5.build import INDEX_COLUMNS, build_r5, check_expected
from astro_simclr.r5.recipe import JD_MAX, draw_alert
from astro_simclr.r5.select import (
    Candidate,
    Decision,
    Inputs,
    Responses,
    Selection,
    _walk,
    evaluate,
    read_cd21,
)
from astro_simclr.r5.sources import BTS, ZENODO, load_source, read_bts, read_zenodo


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[R5World, Path, Path]:
    root = tmp_path_factory.mktemp("r5")
    world = make_r5_world(root)
    out = root / "out"
    result = build_r5(
        world.sources, world.cd21, root / "fetched", out, client=world.service.client()
    )
    assert result.complete and result.rounds > 1
    return world, root, out


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_the_build_accepts_and_drops_as_the_recipe_says(built: tuple[R5World, Path, Path]) -> None:
    world, _, out = built
    index = {r["oid"]: r for r in rows(out / "index.csv")}
    for oid, outcome in world.expect.items():
        if outcome == "accepted":
            assert oid in index, oid
        else:
            assert oid not in index, (oid, outcome)
    drops = {
        (r["label"], r["subclass"], r["cause"]): int(r["count"]) for r in rows(out / "drops.csv")
    }
    for key in [
        ("AGN", "Blazar", "no_stamp_before_jd_max"),
        ("AGN", "AGN", "duplicate_oid"),
        ("AGN", "AGN", "label_conflict"),
        ("AGN", "QSO", "cd21_train_and_eval"),
        ("VS", "RRL", "avro_unavailable"),
        ("VS", "LPV", "not_63x63"),
        ("other", "CV/Nova", "not_an_r5_subclass"),
        ("other", "TDE", "not_an_r5_subclass"),
        ("SN", "SNIa", "duplicate_oid"),
        ("Asteroid", "Asteroid", "asteroid_ndet_not_1"),
        ("Bogus", "Bogus", "cd21_ghost"),
        ("Bogus", "Bogus", "cd21_unresolved"),
        ("Bogus", "Bogus", "bogus_ndet_ge_10"),
        ("Bogus", "Bogus", "crossmatch_known_object"),
    ]:
        assert drops.get(key, 0) >= 1, key
    counts = json.loads((out / "build.json").read_text())["counts"]["by_class"]
    assert counts == {"AGN": 12, "SN": 12, "VS": 12, "Asteroid": 12, "Bogus": 12}


def test_every_drawn_alert_is_the_seeded_draw_before_jd_max(
    built: tuple[R5World, Path, Path],
) -> None:
    world, _, out = built
    for r in rows(out / "index.csv"):
        obj = world.objects[r["oid"]]
        assert float(r["jd"]) <= JD_MAX
        records = parse_detection_records(detections_json(obj), obj.oid).records
        draw = draw_alert(obj.oid, records, seed=0)
        assert draw is not None
        assert int(r["candid"]) == draw.record.candid
        assert int(r["n_eligible"]) == draw.n_eligible and int(r["draw_index"]) == draw.index
        assert int(r["ndet"]) == len(
            {c for c, mjd, _ in obj.detections if mjd + 2400000.5 <= JD_MAX}
        )


def test_the_layouts_and_tables_are_consistent(built: tuple[R5World, Path, Path]) -> None:
    _, _, out = built
    verify_raw_alerts(out / "r5")
    data = load_raw_alerts(out / "r5")
    assert len(data) == 60 and data.label_names == ("AGN", "SN", "VS", "Asteroid", "Bogus")
    assert not np.isnan(data.cutouts).all()
    assert set(data.index["label_source"].tolist()) == {"zenodo-4279623", "bts", "cd21"}
    table = read_alert_table(out / "alerts.csv")
    splits = read_object_splits(out / "splits.json")
    verify_object_splits(splits, table)
    first = load_raw_alerts(out / "r5-first")
    test = set(table.oid[splits.rows(table, "Test")].tolist())
    assert set(first.index["oid"].tolist()) == test
    assert (first.index["source_dataset"] == "r5-first-stamp").all()
    header = rows(out / "index.csv")[0].keys()
    assert tuple(header) == INDEX_COLUMNS
    eras = rows(out / "eras.csv")
    assert [e["label"] for e in eras] == ["AGN", "SN", "VS", "Asteroid", "Bogus"]
    assert sum(int(e["total"]) for e in eras) == 60
    assert set(eras[0]) - {"label", "total"} <= {"2018", "2019", "2020", "2021", "2022"}


def test_first_stamp_is_the_earliest_full_stamp(built: tuple[R5World, Path, Path]) -> None:
    world, _, out = built
    for r in rows(out / "index.csv"):
        if r["split"] != "Test":
            assert r["first_candid"] == ""
            continue
        obj = world.objects[r["oid"]]
        stamped = sorted(
            (mjd, c) for c, mjd, s in obj.detections if s and mjd + 2400000.5 <= JD_MAX
        )
        full = [c for _, c in stamped if obj.shapes.get(c, (63, 63)) == (63, 63)]
        assert int(r["first_candid"]) == full[0]


def test_the_cross_benchmark_matrix_holds(built: tuple[R5World, Path, Path]) -> None:
    world, _, out = built
    index = {r["oid"]: r for r in rows(out / "index.csv")}
    assert index[world.pinned_train]["split"] == "Train"
    assert index[world.kept_from_train]["split"] in ("Validation", "Test")
    for r in index.values():
        where = set(r["cd21_splits"].split())
        if "Train" in where:
            assert r["split"] == "Train"
        if where & {"Validation", "Test"}:
            assert r["split"] != "Train"
    matrix = json.loads((out / "build.json").read_text())["cross_benchmark"]["shared_groups"]
    assert matrix[0][1] == matrix[0][2] == matrix[1][0] == matrix[2][0] == 0
    assert matrix[0][0] >= 9  # the pinned AGN, 8 Asteroids, 8 Bogus share CD21 Train
    splits = read_object_splits(out / "splits.json")
    assert splits.constraints[index[world.pinned_train]["group"]] == ("Train",)


def test_deviation_table_reports_targets_and_shortfalls(built: tuple[R5World, Path, Path]) -> None:
    _, _, out = built
    table = {(r["label"], r["subclass"]): r for r in rows(out / "deviations.csv")}
    eb = table[("VS", "EB")]
    assert (eb["target"], eb["cap"], eb["achieved"], eb["deviation"]) == (
        "2339",
        "2400",
        "6",
        "-2333",
    )
    assert "source holds 6 < target" in eb["cause"]
    assert "absent from zenodo" in table[("VS", "RSCVn")]["cause"]
    assert table[("VS", "ZZ")]["achieved"] == "0"
    assert table[("SN", "(all)")]["target"] == "3615"
    assert table[("Bogus", "(all)")]["achieved"] == "12"
    assert "cd21_ghost" in table[("Bogus", "Bogus")]["cause"]


def test_offline_rebuild_is_byte_identical(built: tuple[R5World, Path, Path]) -> None:
    world, root, out = built
    calls = len(world.service.calls)
    again = root / "again"
    result = build_r5(world.sources, world.cd21, root / "fetched", again, client=None)
    assert result.complete and len(world.service.calls) == calls  # no network
    assert result.record is not None
    assert check_expected(result.record, out / "build.json") == []
    files = sorted(p.relative_to(out) for p in out.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again) for p in again.rglob("*") if p.is_file())
    for name in files:
        assert (out / name).read_bytes() == (again / name).read_bytes(), name
    assert not list(again.glob(".*"))  # no temporary files left


def test_the_result_does_not_depend_on_what_else_was_fetched(
    built: tuple[R5World, Path, Path], tmp_path: Path
) -> None:
    """A store holding every response of the world (far more than the walks need, fetched
    in another order) gives the same outputs as the round-by-round online build."""
    world, _, out = built
    items = [FetchItem("detections", o.oid) for o in world.objects.values()]
    items += [
        FetchItem("avro", o.oid, c) for o in world.objects.values() for c, _, s in o.detections if s
    ]
    service = FakeService(bodies=dict(world.service.bodies))
    fetch_by_oid(list(reversed(items)), tmp_path / "all", client=service.client())
    result = build_r5(world.sources, world.cd21, tmp_path / "all", tmp_path / "o", client=None)
    assert result.complete and result.record is not None
    assert check_expected(result.record, out / "build.json") == []


def test_every_class_fills_every_split(built: tuple[R5World, Path, Path]) -> None:
    """The balance of constrained classes is reported, not verified (ADR 0020): check it."""
    _, _, out = built
    for r in rows(out / "split_counts.csv"):
        train, val, test = int(r["Train"]), int(r["Validation"]), int(r["Test"])
        assert min(train, val, test) >= 1 and train > max(val, test), r


def test_check_expected_names_the_files_that_differ(
    built: tuple[R5World, Path, Path], tmp_path: Path
) -> None:
    _, _, out = built
    record = json.loads((out / "build.json").read_text())
    record["files"]["index.csv"] = "0" * 64
    other = tmp_path / "build.json"
    other.write_text(json.dumps(record))
    current = json.loads((out / "build.json").read_text())
    assert check_expected(current, other) == ["index.csv"]
    with pytest.raises(DataError):
        check_expected(current, tmp_path / "missing.json")


def test_offline_build_refuses_an_incomplete_snapshot(tmp_path: Path) -> None:
    world = make_r5_world(tmp_path)
    with pytest.raises(DataError, match="snapshot lacks"):
        build_r5(world.sources, world.cd21, tmp_path / "fetched", tmp_path / "out", client=None)
    dry = build_r5(world.sources, world.cd21, tmp_path / "fetched", None, client=None)
    assert not dry.complete and dry.rounds == 1
    # CD21's detection lists are in task 3.1's snapshot: only Zenodo's and BTS's are
    # missing. An alert is missing only for a CD21 Bogus whose draw is not CD21's own alert.
    cd21_oids = {r.oid for r in read_cd21(world.cd21)[1] if r.oid}
    assert {i.oid for i in dry.missing if i.kind == "detections"}.isdisjoint(cd21_oids)
    avros = [i for i in dry.missing if i.kind == "avro"]
    assert avros and all(i.oid in cd21_oids for i in avros)
    assert not world.service.calls
    with pytest.raises(DataError, match="already exists"):
        (tmp_path / "exists").mkdir()
        build_r5(world.sources, world.cd21, tmp_path / "fetched", tmp_path / "exists")


def test_rounds_fetch_only_what_the_walk_needs(built: tuple[R5World, Path, Path]) -> None:
    world, root, _ = built
    lines = read_oid_index(root / "fetched")
    fetched = {line.item for line in lines}
    # CD21's detection lists come from task 3.1's snapshot, never fetched again.
    asteroid = next(o for o, e in world.expect.items() if e == "asteroid_ndet_not_1")
    assert FetchItem("detections", asteroid) not in fetched
    # Drops before any request: label conflicts, cross-matched bogus, duplicates.
    conflict = next(o for o, e in world.expect.items() if e == "label_conflict")
    assert FetchItem("detections", conflict) not in fetched
    assert len(world.service.calls) == len(lines)
    assert {line.status for line in lines} <= {"ok", "not_found"}


def test_the_walk_stops_at_the_target() -> None:
    cands = [
        Candidate(f"ZTF18aaaaa{c}a", "AGN", "AGN", "z", "z", i) for i, c in enumerate("abcdefgh")
    ]
    outcome = iter(["accepted", "dropped", "pending", "accepted", "accepted", "accepted"])
    decisions = _walk(cands, 3, lambda c: Decision(c, next(outcome)))
    assert [d.status for d in decisions] == ["accepted", "dropped", "pending", "accepted"]
    everything = _walk(cands, None, lambda c: Decision(c, "accepted"))
    assert len(everything) == len(cands)


def order(selection: Selection) -> list[str]:
    return [d.candidate.oid for d in selection.decisions["AGN"]]


def test_the_selection_is_seeded(tmp_path: Path) -> None:
    world = make_r5_world(tmp_path)
    zenodo, _ = load_source(world.sources, ZENODO)
    bts, _ = load_source(world.sources, BTS)
    name, cd21 = read_cd21(world.cd21)
    inputs = Inputs(read_zenodo(zenodo), read_bts(bts), name, cd21)
    responses = Responses([world.cd21 / s for s in ("Train", "Validation", "Test")], None)
    a = evaluate(inputs, responses, seed=0)
    b = evaluate(inputs, responses, seed=0)
    c = evaluate(inputs, responses, seed=7)
    assert order(a) == order(b)
    assert sorted(order(a)) == sorted(order(c)) and order(a) != order(c)
    assert a.missing == b.missing and not a.complete
