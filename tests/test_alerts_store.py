"""Tests of the raw alert store and of the fetch-alerts / alerts-summary commands (task 3.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _alerts_support import FakeAlerce, World, make_world

import astro_simclr.alerts.alerce as alerce_module
from astro_simclr.alerts.alerce import AlerceClient
from astro_simclr.alerts.snapshot import Recorder, Response
from astro_simclr.alerts.store import (
    COLUMNS,
    IndexRow,
    IndexWriter,
    Shard,
    Snapshot,
    format_duplicates,
    format_summary,
    load_recovered,
    next_shard_id,
    read_index,
    read_meta,
    read_shard,
    shard_path,
    summarize,
    write_meta,
    write_shard,
)
from astro_simclr.cli import main
from astro_simclr.data.schema import DataError


def _line(row: int, status: str = "ok", **kw: object) -> IndexRow:
    return IndexRow(row, 0, status, 1.0, 2.0, 19.0, 0.1, **kw)  # type: ignore[arg-type]


def test_index_round_trip_and_errors(tmp_path: Path) -> None:
    writer = IndexWriter(tmp_path)
    lines = [_line(0, candid=12, jd=2458000.5, note="a, b"), _line(1, "no_object")]
    writer.append(lines)
    writer.close()
    assert read_index(tmp_path) == lines
    assert (tmp_path / "index.csv").read_text().splitlines()[0] == ",".join(COLUMNS)
    (tmp_path / "index.csv").write_text("row,label\n0,0\n")
    with pytest.raises(DataError, match="unexpected columns"):
        read_index(tmp_path)
    good = ",".join(COLUMNS) + "\n"
    (tmp_path / "index.csv").write_text(good + "0,0,weird" + ",1.0" * 4 + "," * 24 + "\n")
    with pytest.raises(DataError, match="line 2"):
        read_index(tmp_path)
    assert read_index(tmp_path / "missing") == []


def test_meta_round_trip_and_errors(tmp_path: Path) -> None:
    write_meta(tmp_path, {"split": "Test"})
    assert read_meta(tmp_path)["split"] == "Test"
    (tmp_path / "fetch.json").write_text("{")
    with pytest.raises(DataError, match="invalid JSON"):
        read_meta(tmp_path)
    (tmp_path / "fetch.json").write_text(
        '{"format": "astro-simclr-raw-alerts", "format_version": 9}'
    )
    with pytest.raises(DataError, match="format version 9"):
        read_meta(tmp_path)
    (tmp_path / "fetch.json").write_text("[]")
    with pytest.raises(DataError, match="not an"):
        read_meta(tmp_path)
    with pytest.raises(DataError, match="cannot read"):
        read_meta(tmp_path / "missing")


def _shard(n: int = 2) -> Shard:
    cutouts = np.full((n, 63, 63, 3), np.nan, dtype=np.float32)
    responses = [
        Response("https://a", "avro", "T", "0" * 64, b"abc"),
        Response("https://b", "objects", "T", "1" * 64, b"{}"),
    ]
    return Shard(
        rows=np.arange(n, dtype=np.int64),
        candid=np.arange(n, dtype=np.int64) + 10,
        cutouts=cutouts,
        candidates=[{"candid": 10 + i} for i in range(n)],
        responses=responses,
    )


def test_shard_round_trip_and_validation(tmp_path: Path) -> None:
    path = write_shard(tmp_path, 3, _shard())
    assert path == shard_path(tmp_path, 3) and next_shard_id(tmp_path) == 4
    shard = read_shard(path)
    assert shard.rows.tolist() == [0, 1] and np.isnan(shard.cutouts).all()
    assert [r.body for r in shard.responses] == [b"abc", b"{}"]
    bad = _shard()
    with pytest.raises(ValueError, match="cutouts is float64"):
        write_shard(
            tmp_path,
            4,
            Shard(bad.rows, bad.candid, bad.cutouts.astype(np.float64), bad.candidates, []),
        )
    with pytest.raises(ValueError, match="1 candidates for 2 rows"):
        write_shard(tmp_path, 4, Shard(bad.rows, bad.candid, bad.cutouts, bad.candidates[:1], []))
    (tmp_path / "shard-00005.npz").write_bytes(b"junk")
    with pytest.raises(DataError, match="cannot load"):
        read_shard(tmp_path / "shard-00005.npz")
    np.savez(tmp_path / "shard-00006.npz", rows=np.zeros(1))
    with pytest.raises(DataError, match="keys"):
        read_shard(tmp_path / "shard-00006.npz")
    arrays = {k: v for k, v in np.load(path).items()}
    arrays["responses"] = np.frombuffer(json.dumps([{"url": "u"}]).encode(), dtype=np.uint8)
    np.savez(tmp_path / "shard-00007.npz", **arrays)
    with pytest.raises(DataError, match="invalid response record"):
        read_shard(tmp_path / "shard-00007.npz")
    arrays["responses"] = np.frombuffer(b"{}", dtype=np.uint8)
    np.savez(tmp_path / "shard-00008.npz", **arrays)
    with pytest.raises(DataError, match="JSON list"):
        read_shard(tmp_path / "shard-00008.npz")


def test_snapshot_detects_a_corrupted_body(tmp_path: Path) -> None:
    write_shard(tmp_path, 0, _shard())
    with pytest.raises(DataError, match="SHA-256 differs"):
        Snapshot(tmp_path).fetch("https://a", 100)


def test_load_recovered_errors(tmp_path: Path) -> None:
    writer = IndexWriter(tmp_path)
    writer.append([_line(0, shard=None)])
    writer.close()
    with pytest.raises(DataError, match="without a shard"):
        load_recovered(tmp_path)
    write_shard(tmp_path, 0, _shard(1))
    writer = IndexWriter(tmp_path)
    writer.append([_line(5, shard=0)])
    writer.close()
    with pytest.raises(DataError, match="row 0 is ok without a shard"):
        load_recovered(tmp_path)


def test_format_duplicates_truncates() -> None:
    found = {i: [("Train", i), ("Validation", i)] for i in range(3)}
    text = format_duplicates(found, limit=2)
    assert "3 (3 across splits)" in text and "... 1 more" in text


@pytest.fixture
def world(tmp_path: Path) -> World:
    return make_world(tmp_path)


def _patch_client(monkeypatch: pytest.MonkeyPatch, alerce: FakeAlerce) -> list[float]:
    rates: list[float] = []

    def make(rate: float) -> AlerceClient:
        rates.append(rate)
        return AlerceClient(Recorder(alerce.fetch, requests=lambda: alerce.requests))

    monkeypatch.setattr(alerce_module, "make_client", make)
    return rates


def test_cli_fetch_resume_summary_and_snapshot(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rates = _patch_client(monkeypatch, world.alerce)
    out = tmp_path / "raw"
    base = ["fetch-alerts", "--data-dir", str(world.prepared.parent), "--dataset", "world"]
    args = [*base, "--split", "Validation", "--out", str(out), "--rate", "1"]
    assert main([*args, "--limit", "4"]) == 0
    captured = capsys.readouterr()
    assert rates == [1.0] and "[1/4] row 0 AGN ok ZTF18aaaaaaa:100" in captured.err
    assert "attempted 4 rows (0 already done)" in captured.out
    assert main(args) == 1  # already holds a run, no --resume
    assert "--resume" in capsys.readouterr().err
    assert main([*args, "--resume", "--quiet", "--cooldown", "0"]) == 0
    captured = capsys.readouterr()
    assert (
        captured.err == ""
        and "acceptance (>= 98% ok per class, ghosts excluded): FAIL" in captured.out
    )
    assert main(["alerts-summary", str(out)]) == 1
    text = capsys.readouterr().out
    assert "Validation: 10/10 rows done" in text and "groups: recovered 4" in text
    assert "max |minmax(raw) - stored|" in text and "duplicated candids: 0" in text
    assert main(["alerts-summary", str(out), "--split", "Validation", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    summary = payload["splits"][0]
    assert summary["totals"]["ok"] == 4 and summary["edge_ok"]["Asteroid"] == 1
    assert summary["groups"]["ghost"]["SN"] == 1 and summary["ghosts"]["SN"] == 1
    assert summary["recovery"]["AGN"] == 1.0 and summary["recovery"]["SN"] == 1.0
    rebuilt = tmp_path / "rebuilt"
    snap = [
        *base,
        "--split",
        "Validation",
        "--out",
        str(rebuilt),
        "--from-snapshot",
        str(out),
        "--quiet",
    ]
    assert main(snap) == 0
    capsys.readouterr()
    assert summarize(rebuilt / "Validation").totals == summarize(out / "Validation").totals
    assert (
        main([*base, "--split", "Validation", "--out", str(out), "--from-snapshot", str(out)]) == 2
    )
    assert main([*args, "--rate", "0"]) == 2
    assert main([*args, "--resume", "--limit", "0"]) == 2
    assert main(["alerts-summary", str(tmp_path / "empty")]) == 1
    assert "no fetch-alerts run" in capsys.readouterr().err


def test_summary_accepts_a_fully_recovered_split(world: World, tmp_path: Path) -> None:
    out = tmp_path / "raw" / "Validation"
    out.mkdir(parents=True)
    write_meta(out, {"split": "Validation", "n_rows": 2, "class_counts": {"AGN": 2}})
    writer = IndexWriter(out)
    writer.append(
        [
            _line(0, max_abs_diff=1e-8, raw_shape="63x63"),
            _line(1, max_abs_diff=2e-8, raw_shape="63x50"),
        ]
    )
    writer.close()
    summary = summarize(out)
    assert summary.accepted() and summary.edge["AGN"] == 1 and summary.max_abs_diff_ok == 2e-8
    assert summary.recovery()["SN"] is None
    assert main(["alerts-summary", str(tmp_path / "raw")]) == 0


def test_recovery_excludes_ghost_rows(tmp_path: Path) -> None:
    write_meta(tmp_path, {"split": "Train", "n_rows": 3, "class_counts": {"Bogus": 3}})
    writer = IndexWriter(tmp_path)
    ghost = IndexRow(1, 4, "ghost_candidate", 10.5, -3.25, 19.0, 0.1, n_objects=0)
    writer.append([IndexRow(0, 4, "ok", 1.0, 2.0, 19.0, 0.1), ghost])
    writer.close()
    summary = summarize(tmp_path)
    assert summary.ghosts["Bogus"] == 1 and summary.recovery()["Bogus"] == 0.5
    assert summary.to_json()["groups"]["ghost"]["Bogus"] == 1
    assert read_index(tmp_path)[1].target_ra == 10.5  # the position groups' input
    writer = IndexWriter(tmp_path)
    writer.append([IndexRow(2, 4, "ghost_candidate", 1.0, 2.0, 19.0, 0.1)])
    writer.close()
    summary = summarize(tmp_path)
    assert summary.recovery()["Bogus"] == 1.0 and summary.accepted()
    assert "ghost_candidate" in format_summary(summary)


def test_newlines_in_text_fields_stay_on_one_line(tmp_path: Path) -> None:
    writer = IndexWriter(tmp_path)
    writer.append([_line(0, note="first\nsecond\r\nthird")])
    writer.close()
    assert len((tmp_path / "index.csv").read_text().splitlines()) == 2
    assert read_index(tmp_path)[0].note == "first second  third"


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGTERM"), reason="POSIX signals")
def test_cli_sigterm_stops_after_the_current_row(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import os
    import signal

    fetch = world.alerce.fetch

    def signalling(url: str, max_bytes: int) -> tuple[bytes, str]:
        if len(world.alerce.calls) == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return fetch(url, max_bytes)

    world.alerce.fetch = signalling  # type: ignore[method-assign]
    _patch_client(monkeypatch, world.alerce)
    before = signal.getsignal(signal.SIGTERM)
    out = tmp_path / "raw"
    args = ["fetch-alerts", "--data-dir", str(world.prepared.parent), "--dataset", "world"]
    code = main([*args, "--split", "Validation", "--out", str(out), "--quiet"])
    captured = capsys.readouterr()
    assert code == 128 + signal.SIGTERM and "stopping after the current row" in captured.err
    assert len(read_index(out / "Validation")) == 1  # the row in progress, committed
    assert signal.getsignal(signal.SIGTERM) == before
