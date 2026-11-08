"""Tests of the R5 commands (task 3.5): ``fetch-alerts --oids``, ``r5-sources``, ``build-r5``.

The online client is replaced by the offline fake service of ``_r5_support``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _r5_support import FakeService, make_r5_world

from astro_simclr import cli
from astro_simclr.alerts import alerce
from astro_simclr.alerts.alerce import avro_url, detections_url
from astro_simclr.r5 import sources


def use(monkeypatch: pytest.MonkeyPatch, service: FakeService) -> None:
    monkeypatch.setattr(alerce, "make_client", lambda rate: service.client())


def test_fetch_alerts_by_oid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    service = FakeService()
    service.bodies[detections_url("ZTF18aaaaaaa")] = b"[]"
    use(monkeypatch, service)
    items = tmp_path / "items.txt"
    items.write_text("ZTF18aaaaaaa\nZTF18aaaaaaa 5\n")
    out = tmp_path / "o"
    assert cli.main(["fetch-alerts", "--oids", str(items), "--out", str(out)]) == 0
    assert "ok 1, not_found 1" in capsys.readouterr().out
    assert service.calls == [detections_url("ZTF18aaaaaaa"), avro_url("ZTF18aaaaaaa", 5)]
    assert (
        cli.main(["fetch-alerts", "--oids", str(items), "--out", str(out), "--resume", "--quiet"])
        == 0
    )
    assert len(service.calls) == 2
    assert cli.main(["fetch-alerts", "--oids", str(items)]) == 2
    assert (
        cli.main(["fetch-alerts", "--oids", str(items), "--out", str(out), "--split", "Test"]) == 2
    )
    assert cli.main(["fetch-alerts", "--oids", str(items), "--out", str(out), "--rate", "0"]) == 2
    assert cli.main(["fetch-alerts", "--split", "Test"]) == 2
    assert "needs --dataset and --split" in capsys.readouterr().err


def test_r5_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: list[str] = []

    def fake_bts(dest: Path, *, get: object) -> Path:
        called.append("bts")
        return dest / "bts.csv"

    def fake_zenodo(dest: Path, *, keep_archive: bool) -> Path:
        called.append(f"zenodo keep={keep_archive}")
        return dest / "zenodo.csv"

    monkeypatch.setattr(sources, "download_bts", fake_bts)
    monkeypatch.setattr(sources, "download_zenodo", fake_zenodo)
    assert cli.main(["r5-sources", "--out", str(tmp_path)]) == 0
    assert (
        cli.main(["r5-sources", "--out", str(tmp_path), "--only", "zenodo", "--keep-archive"]) == 0
    )
    assert called == ["bts", "zenodo keep=False", "zenodo keep=True"]
    assert cli.main(["r5-sources", "--out", str(tmp_path), "--rate", "-1"]) == 2


def test_build_r5_online_offline_and_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = make_r5_world(tmp_path)
    use(monkeypatch, world.service)
    common = ["build-r5", "--sources", str(world.sources), "--cd21-raw", str(world.cd21)]
    common += ["--fetched", str(tmp_path / "fetched"), "--quiet"]
    assert cli.main([*common, "--dry-run"]) == 0
    assert "dry run: the next round fetches" in capsys.readouterr().out
    assert cli.main([*common, "--from-snapshot", "--out", str(tmp_path / "x")]) == 1
    assert "snapshot lacks" in capsys.readouterr().err
    assert cli.main([*common, "--out", str(tmp_path / "out")]) == 0
    assert "60 objects" in capsys.readouterr().out
    expect = str(tmp_path / "out" / "build.json")
    again = ["--from-snapshot", "--out", str(tmp_path / "again"), "--expect", expect]
    assert cli.main([*common, *again]) == 0
    assert "matches --expect byte for byte" in capsys.readouterr().out
    (tmp_path / "other.json").write_text('{"files": {}}')
    other = [
        "--from-snapshot",
        "--out",
        str(tmp_path / "b"),
        "--expect",
        str(tmp_path / "other.json"),
    ]
    assert cli.main([*common, *other]) == 1
    assert "differs from --expect" in capsys.readouterr().out
    assert cli.main(common) == 2  # neither --out nor --dry-run
    assert cli.main([*common, "--dry-run", "--expect", expect]) == 2
    assert cli.main([*common, "--dry-run", "--seed", "-1"]) == 2
    assert cli.main([*common, "--dry-run", "--max-rounds", "0"]) == 2


def test_build_r5_reports_an_incomplete_online_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = make_r5_world(tmp_path)
    use(monkeypatch, world.service)
    args = ["build-r5", "--sources", str(world.sources), "--cd21-raw", str(world.cd21)]
    args += ["--fetched", str(tmp_path / "f"), "--out", str(tmp_path / "out"), "--max-rounds", "1"]
    assert cli.main([*args, "--quiet"]) == 1
    assert "R5 not complete after 2 rounds" in capsys.readouterr().out  # 1 fetch, 2 selections
