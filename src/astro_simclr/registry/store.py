"""The registry's SQLite file: schema, ``schema_version``, migrations, rows.

**Schema version** :data:`SCHEMA_VERSION`, stored in the file's ``PRAGMA user_version``
(0 for a new file). :func:`open_registry` applies the missing :data:`MIGRATIONS` in order,
each in its own transaction with the version bump, so a crash leaves the file at a whole
version. A file of a newer version than this code is refused: it was written by a newer
astro-simclr, and an older reader could misread it. As the index is derived,
``runs reindex --rebuild`` (or deleting the file) is always a way out, but a migration
keeps the remembered roots and the artefact hashes, which are slow to recompute.

Tables (version 1):

- ``roots``: the directories ``reindex`` scans, remembered so that a bare ``reindex``
  rescans them;
- ``runs``: one row per run directory (:class:`~astro_simclr.registry.readers.RunRecord`;
  ``split`` as text, ``'screen'`` or the fold's digits; mappings as canonical JSON);
- ``artefacts``: every file of a run directory, with its kind
  (``runs.layout.classify_artefact``), size, mtime and SHA-256 (presence is the row);
- ``specs``: the experiment specs found in ``$ASTRO_SIMCLR_HOME/specs`` (task A3), by id.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from astro_simclr.registry.home import RegistryError
from astro_simclr.registry.readers import RunRecord
from astro_simclr.runs.canonical import canonical_json
from astro_simclr.runs.identity import SCREEN, SplitPart

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "Artefact",
    "SpecRow",
    "insert_run",
    "load_records",
    "open_registry",
    "record_row",
    "schema_version",
    "split_from_text",
    "split_to_text",
    "transaction",
]

_V1: Final = """
CREATE TABLE roots (
    path TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);
CREATE TABLE runs (
    run_dir TEXT PRIMARY KEY,
    root TEXT NOT NULL,
    record_key TEXT NOT NULL,
    source TEXT NOT NULL,
    legacy INTEGER NOT NULL,
    config_format TEXT NOT NULL,
    run_id TEXT,
    attempt INTEGER,
    command TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    seed INTEGER NOT NULL,
    split TEXT,
    data_sha256 TEXT,
    splits_sha256 TEXT,
    pretrained_sha256 TEXT,
    data_source_sha256 TEXT,
    machine_class TEXT,
    engine TEXT NOT NULL,
    numerics_epoch INTEGER,
    git_sha TEXT,
    git_dirty INTEGER,
    uv_lock_sha256 TEXT,
    status TEXT NOT NULL,
    created_at TEXT,
    started_at TEXT,
    ended_at TEXT,
    preset TEXT,
    metrics_sha256 TEXT,
    machine_json TEXT,
    metrics_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE INDEX runs_by_key ON runs (record_key);
CREATE INDEX runs_by_run_id ON runs (run_id);
CREATE INDEX runs_by_config ON runs (command, config_sha256, seed);
CREATE TABLE artefacts (
    run_dir TEXT NOT NULL REFERENCES runs (run_dir) ON DELETE CASCADE,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY (run_dir, path)
);
CREATE TABLE specs (
    spec_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL
);
"""

#: The migrations: entry ``i`` takes a file from version ``i`` to ``i + 1``. Append only;
#: never edit a released entry (files in the wild are at its version).
MIGRATIONS: Final[tuple[str, ...]] = (_V1,)
#: The schema version this code writes and reads.
SCHEMA_VERSION: Final = len(MIGRATIONS)


@dataclass(frozen=True, slots=True)
class Artefact:
    """One file of a run directory (``path`` relative to it, POSIX separators)."""

    path: str
    kind: str
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SpecRow:
    """An experiment spec found by ``reindex``: its id, name and file."""

    spec_id: str
    name: str
    path: Path


def schema_version(conn: sqlite3.Connection) -> int:
    """The file's ``user_version``."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT``, or ``ROLLBACK`` on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _statements(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def _migrate(conn: sqlite3.Connection, migrations: Sequence[str]) -> None:
    current = schema_version(conn)
    if current > len(migrations):
        msg = (
            f"the registry is at schema version {current}, this astro-simclr reads up to "
            f"{len(migrations)}: upgrade, or rebuild it (runs reindex --rebuild)"
        )
        raise RegistryError(msg)
    for version in range(current, len(migrations)):
        with transaction(conn):
            for statement in _statements(migrations[version]):
                conn.execute(statement)
            # PRAGMA takes no parameters; the value is an int from range().
            conn.execute(f"PRAGMA user_version = {version + 1}")


def open_registry(path: Path, *, migrations: Sequence[str] = MIGRATIONS) -> sqlite3.Connection:
    """Open (creating it and its directory if needed) and migrate the registry at ``path``.

    The connection is in autocommit mode: writers use :func:`transaction`. Foreign keys
    are on, so deleting a run deletes its artefacts. ``migrations`` is for the tests.

    Raises:
        RegistryError: if the file cannot be opened, is not SQLite, or has a newer schema.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None)
    except (OSError, sqlite3.Error) as err:
        raise RegistryError(f"{path}: cannot open the registry ({err})") from None
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        _migrate(conn, migrations)
    except sqlite3.DatabaseError as err:
        conn.close()
        raise RegistryError(f"{path}: not a usable registry ({err})") from None
    except RegistryError as err:
        conn.close()
        raise RegistryError(f"{path}: {err}") from None
    return conn


def split_to_text(split: SplitPart) -> str | None:
    """A split part as the ``runs.split`` column holds it."""
    return None if split is None else str(split)


def split_from_text(text: str | None) -> SplitPart:
    """The ``runs.split`` column back as a split part.

    Raises:
        RegistryError: on a value :func:`split_to_text` does not write.
    """
    if text is None:
        return None
    if text == SCREEN:
        return SCREEN
    if text.isdigit():
        return int(text)
    raise RegistryError(f"runs.split {text!r} is not a split part")


def _json(value: object) -> str:
    return canonical_json(value)


def record_row(record: RunRecord, root: Path, indexed_at: str) -> dict[str, object]:
    """``record`` as a ``runs`` row (named parameters)."""
    return {
        "run_dir": str(record.run_dir),
        "root": str(root),
        "record_key": record.key,
        "source": record.source,
        "legacy": int(record.legacy),
        "config_format": record.config_format,
        "run_id": record.run_id,
        "attempt": record.attempt,
        "command": record.command,
        "config_sha256": record.config_sha256,
        "seed": record.seed,
        "split": split_to_text(record.split),
        "data_sha256": record.data_sha256,
        "splits_sha256": record.splits_sha256,
        "pretrained_sha256": record.pretrained_sha256,
        "data_source_sha256": record.data_source_sha256,
        "machine_class": record.machine_class,
        "engine": record.engine,
        "numerics_epoch": record.numerics_epoch,
        "git_sha": record.git_sha,
        "git_dirty": None if record.git_dirty is None else int(record.git_dirty),
        "uv_lock_sha256": record.uv_lock_sha256,
        "status": record.status,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "ended_at": record.ended_at,
        "preset": record.preset,
        "metrics_sha256": record.metrics_sha256,
        "machine_json": None if record.machine is None else _json(dict(record.machine)),
        "metrics_json": _json(dict(record.metrics)),
        "config_json": _json(dict(record.config)),
        "indexed_at": indexed_at,
    }


_RUN_COLUMNS: Final = tuple(
    line.split()[0]
    for line in _V1.split("CREATE TABLE runs (", 1)[1].split(");", 1)[0].strip().splitlines()
)
_INSERT_RUN: Final = (
    f"INSERT INTO runs ({', '.join(_RUN_COLUMNS)}) "  # noqa: S608 - fixed column names
    f"VALUES ({', '.join(':' + name for name in _RUN_COLUMNS)})"
)


def insert_run(conn: sqlite3.Connection, row: dict[str, object]) -> None:
    """Insert a :func:`record_row` into ``runs`` (its keys are the version-1 columns)."""
    conn.execute(_INSERT_RUN, row)


def _record(row: sqlite3.Row) -> RunRecord:
    machine: Any = None if row["machine_json"] is None else json.loads(row["machine_json"])
    return RunRecord(
        run_dir=Path(row["run_dir"]),
        source=row["source"],
        legacy=bool(row["legacy"]),
        config_format=row["config_format"],
        run_id=row["run_id"],
        attempt=row["attempt"],
        command=row["command"],
        config_sha256=row["config_sha256"],
        seed=row["seed"],
        split=split_from_text(row["split"]),
        data_sha256=row["data_sha256"],
        splits_sha256=row["splits_sha256"],
        pretrained_sha256=row["pretrained_sha256"],
        data_source_sha256=row["data_source_sha256"],
        machine_class=row["machine_class"],
        engine=row["engine"],
        numerics_epoch=row["numerics_epoch"],
        git_sha=row["git_sha"],
        git_dirty=None if row["git_dirty"] is None else bool(row["git_dirty"]),
        uv_lock_sha256=row["uv_lock_sha256"],
        status=row["status"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        preset=row["preset"],
        metrics_sha256=row["metrics_sha256"],
        machine=None if machine is None else MappingProxyType(machine),
        metrics=MappingProxyType(json.loads(row["metrics_json"])),
        config=MappingProxyType(json.loads(row["config_json"])),
    )


def load_records(conn: sqlite3.Connection) -> list[RunRecord]:
    """Every indexed record, ordered by run directory."""
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM runs ORDER BY run_dir").fetchall()
    finally:
        conn.row_factory = None
    return [_record(row) for row in rows]
