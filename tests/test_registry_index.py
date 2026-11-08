"""Tests of ``astro_simclr.registry.index``: ``reindex`` of run roots and of the specs."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from _registry_support import (
    copy_replication,
    hide_spec_module,
    install_fake_spec_module,
    legacy_run,
    manifest_run,
    replication_spec,
    write_spec,
)

from astro_simclr.data.files import sha256_file
from astro_simclr.registry.home import RegistryError
from astro_simclr.registry.index import find_run_dirs, reindex
from astro_simclr.registry.store import load_records, open_registry


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = open_registry(tmp_path / "home" / "registry.sqlite")
    yield connection
    connection.close()


class CountingHasher:
    """``sha256_file`` that counts its calls."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.calls.append(path)
        return sha256_file(path)


def _dirs(conn: sqlite3.Connection) -> list[str]:
    return [str(record.run_dir) for record in load_records(conn)]


def test_find_run_dirs_prunes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    legacy_run(root / "a" / "run1")
    nested = legacy_run(root / "run2")
    legacy_run(nested / "inner")  # inside a run directory: not a run of its own
    legacy_run(root / ".hidden" / "run3")
    assert list(find_run_dirs(root)) == [root / "a" / "run1", root / "run2"]


def test_reindex_the_archive(tmp_path: Path, conn: sqlite3.Connection) -> None:
    root = copy_replication(tmp_path / "archive")
    hasher = CountingHasher()
    report = reindex(conn, [root], specs=tmp_path / "specs", hasher=hasher)
    assert report.indexed == {root.resolve(): 10}
    assert report.skipped == [] and report.missing_roots == [] and report.specs == 0
    assert len(load_records(conn)) == 10
    rows = conn.execute("SELECT kind, COUNT(*) FROM artefacts GROUP BY kind").fetchall()
    assert dict(rows) == {"metrics": 10, "predictions": 20, "figure": 20}
    first = len(hasher.calls)
    assert first == 50
    # A rescan rehashes nothing unchanged.
    reindex(conn, [], specs=tmp_path / "specs", hasher=hasher)
    assert len(hasher.calls) == first
    sha = conn.execute(
        "SELECT sha256 FROM artefacts WHERE path = 'predictions_val.npz' LIMIT 1"
    ).fetchone()[0]
    assert len(sha) == 64
    # --rebuild rehashes everything.
    reindex(conn, [], specs=tmp_path / "specs", rebuild=True, hasher=hasher)
    assert len(hasher.calls) == 2 * first


def test_rescan_follows_changes(tmp_path: Path, conn: sqlite3.Connection) -> None:
    root = tmp_path / "root"
    gone = legacy_run(root / "gone")
    changed = manifest_run(root / "changed", status="running")
    specs = tmp_path / "specs"
    reindex(conn, [root], specs=specs)
    assert {r.status for r in load_records(conn)} == {"finished", "running"}
    for path in gone.iterdir():
        path.unlink()
    gone.rmdir()
    manifest_run(changed, status="finished")
    hasher = CountingHasher()
    reindex(conn, [], specs=specs, hasher=hasher)
    records = load_records(conn)
    assert [(r.run_dir.name, r.status) for r in records] == [("changed", "finished")]
    assert {p.name for p in hasher.calls} == {"manifest.json", "metrics.json"}
    count = conn.execute("SELECT COUNT(*) FROM artefacts").fetchone()[0]
    assert count == 2


def test_unreadable_runs_are_skipped(tmp_path: Path, conn: sqlite3.Connection) -> None:
    root = tmp_path / "root"
    legacy_run(root / "good")
    bad = root / "bad"
    bad.mkdir(parents=True)
    (bad / "metrics.json").write_text("{", encoding="utf-8")
    report = reindex(conn, [root], specs=tmp_path / "specs")
    assert report.indexed == {root.resolve(): 1}
    assert [s.path.name for s in report.skipped] == ["bad"]
    assert "not valid JSON" in report.skipped[0].reason


def test_symlinks_are_not_followed(tmp_path: Path, conn: sqlite3.Connection) -> None:
    root = tmp_path / "root"
    run = legacy_run(root / "run")
    os.symlink(run / "metrics.json", run / "link.json")
    os.symlink(run, root / "alias")
    reindex(conn, [root], specs=tmp_path / "specs")
    assert _dirs(conn) == [str(run.resolve())]
    paths = {row[0] for row in conn.execute("SELECT path FROM artefacts")}
    assert paths == {"metrics.json"}


def test_roots(tmp_path: Path, conn: sqlite3.Connection) -> None:
    a = legacy_run(tmp_path / "a" / "run").parent
    b = legacy_run(tmp_path / "b" / "run").parent
    specs = tmp_path / "specs"
    reindex(conn, [a], specs=specs)
    reindex(conn, [b], specs=specs)
    assert len(load_records(conn)) == 2
    report = reindex(conn, [], specs=specs)  # every remembered root
    assert set(report.indexed) == {a.resolve(), b.resolve()}
    with pytest.raises(RegistryError, match="not a directory"):
        reindex(conn, [tmp_path / "nowhere"], specs=specs)
    (b / "run" / "metrics.json").unlink()
    (b / "run").rmdir()
    b.rmdir()
    report = reindex(conn, [], specs=specs)
    assert report.missing_roots == [b.resolve()]
    assert len(load_records(conn)) == 2  # an unmounted root keeps its rows


def test_nested_roots_index_a_run_once(tmp_path: Path, conn: sqlite3.Connection) -> None:
    outer = tmp_path / "outer"
    legacy_run(outer / "inner" / "run")
    specs = tmp_path / "specs"
    reindex(conn, [outer / "inner"], specs=specs)
    reindex(conn, [outer], specs=specs)
    reindex(conn, [], specs=specs)
    assert len(load_records(conn)) == 1


def test_specs_are_indexed(
    tmp_path: Path, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_spec_module(monkeypatch)
    specs = tmp_path / "specs"
    spec = replication_spec()
    write_spec(specs / f"{spec.spec_id}.yaml", spec)
    (specs / "broken.json").write_text("[]", encoding="utf-8")
    (specs / "notes.txt").write_text("not a spec", encoding="utf-8")
    report = reindex(conn, [], specs=specs)
    assert report.specs == 1
    assert [s.path.name for s in report.skipped] == ["broken.json"]
    rows = conn.execute("SELECT spec_id, name FROM specs").fetchall()
    assert rows == [(spec.spec_id, "replication")]
    (specs / f"{spec.spec_id}.yaml").unlink()
    reindex(conn, [], specs=specs)
    assert conn.execute("SELECT COUNT(*) FROM specs").fetchone() == (0,)


def test_specs_need_task_a3(
    tmp_path: Path, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    hide_spec_module(monkeypatch)
    specs = tmp_path / "specs"
    write_spec(specs / "x.yaml", replication_spec())
    report = reindex(conn, [], specs=specs)
    assert report.specs == 0 and report.skipped == []
