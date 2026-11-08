"""Tests of ``astro_simclr.registry.store`` (schema, migrations, rows) and ``registry.home``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from _registry_support import legacy_run, manifest_run

from astro_simclr.registry.home import (
    HOME_ENV,
    RegistryError,
    astro_simclr_home,
    check_outside_worktrees,
    registry_path,
    specs_dir,
)
from astro_simclr.registry.readers import read_run_dir
from astro_simclr.registry.store import (
    _RUN_COLUMNS,
    MIGRATIONS,
    SCHEMA_VERSION,
    insert_run,
    load_records,
    open_registry,
    record_row,
    schema_version,
    split_from_text,
    split_to_text,
    transaction,
)


def test_home_default_and_env(tmp_path: Path) -> None:
    assert astro_simclr_home({}) == Path.home() / ".astro-simclr"
    assert astro_simclr_home({HOME_ENV: ""}) == Path.home() / ".astro-simclr"
    assert astro_simclr_home({HOME_ENV: str(tmp_path)}) == tmp_path
    assert registry_path(environ={HOME_ENV: str(tmp_path)}) == (tmp_path / "registry.sqlite")
    assert specs_dir({HOME_ENV: str(tmp_path)}) == tmp_path / "specs"
    assert registry_path(tmp_path / "x.sqlite") == (tmp_path / "x.sqlite").resolve()


@pytest.mark.parametrize("git", ["dir", "file"])
def test_registry_inside_a_worktree_is_refused(tmp_path: Path, git: str) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    if git == "dir":
        (repo / ".git").mkdir()
    else:  # a linked worktree's .git is a file
        (repo / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="outside every worktree"):
        check_outside_worktrees(repo / "state" / "registry.sqlite")
    with pytest.raises(RegistryError, match="inside the git working tree"):
        registry_path(environ={HOME_ENV: str(repo / "state")})


def test_this_checkout_is_a_worktree() -> None:
    with pytest.raises(RegistryError):
        registry_path(Path(__file__).parent / "registry.sqlite")


def test_new_registry_is_at_the_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "registry.sqlite"
    conn = open_registry(path)
    assert schema_version(conn) == SCHEMA_VERSION == len(MIGRATIONS) == 1
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"roots", "runs", "artefacts", "specs"} <= tables
    conn.close()
    reopened = open_registry(path)  # idempotent
    assert schema_version(reopened) == 1
    reopened.close()


def test_migration_keeps_the_rows(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    conn = open_registry(path)
    conn.execute("INSERT INTO roots VALUES ('/r', 'now')")
    conn.close()
    v2 = (
        *MIGRATIONS,
        "ALTER TABLE roots ADD COLUMN note TEXT; CREATE INDEX roots_note ON roots (note)",
    )
    migrated = open_registry(path, migrations=v2)
    assert schema_version(migrated) == 2
    assert migrated.execute("SELECT path, note FROM roots").fetchall() == [("/r", None)]
    migrated.close()
    with pytest.raises(RegistryError, match="schema version 2, this astro-simclr reads up to 1"):
        open_registry(path)


def test_failed_migration_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    open_registry(path).close()
    broken = (*MIGRATIONS, "CREATE TABLE extra (a); NOT SQL")
    with pytest.raises(RegistryError, match="not a usable registry"):
        open_registry(path, migrations=broken)
    conn = open_registry(path)
    assert schema_version(conn) == 1
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    assert "extra" not in names
    conn.close()


def test_not_a_registry(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    path.write_bytes(b"this is not sqlite, just bytes " * 100)
    with pytest.raises(RegistryError, match="not a usable registry"):
        open_registry(path)
    (tmp_path / "file").touch()
    with pytest.raises(RegistryError, match="cannot open the registry"):
        open_registry(tmp_path / "file" / "registry.sqlite")


@pytest.mark.parametrize("split", [None, 0, 4, "screen"])
def test_split_round_trip(split: object) -> None:
    assert split_from_text(split_to_text(split)) == split  # type: ignore[arg-type]


def test_bad_split_text() -> None:
    with pytest.raises(RegistryError, match="not a split part"):
        split_from_text("fold-1")


def test_record_round_trip(tmp_path: Path) -> None:
    records = [
        read_run_dir(manifest_run(tmp_path / "m", split="screen")),
        read_run_dir(legacy_run(tmp_path / "l")),
    ]
    conn = open_registry(tmp_path / "registry.sqlite")
    with transaction(conn):
        for record in records:
            row = record_row(record, tmp_path, "now")
            assert tuple(row) == _RUN_COLUMNS
            insert_run(conn, row)
    loaded = load_records(conn)
    assert loaded == sorted(records, key=lambda r: str(r.run_dir))
    assert conn.row_factory is None
    conn.close()


def test_transaction_rolls_back(tmp_path: Path) -> None:
    conn = open_registry(tmp_path / "registry.sqlite")
    with pytest.raises(sqlite3.IntegrityError), transaction(conn):
        conn.execute("INSERT INTO roots VALUES ('/a', 'now')")
        conn.execute("INSERT INTO roots VALUES ('/a', 'again')")
    assert conn.execute("SELECT COUNT(*) FROM roots").fetchone() == (0,)
    conn.close()
