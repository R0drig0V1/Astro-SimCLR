"""``reindex``: scan run roots into the registry, and the specs directory.

A **root** is a directory of run directories at any depth (``results/phase2``, ``~/runs``,
the queue's run root). :func:`reindex` walks each root, without following symbolic links
and without entering hidden directories. Each directory that holds a ``manifest.json`` or
a ``metrics.json`` is a run directory: it is read
(:func:`~astro_simclr.registry.readers.read_run_dir`) and not descended into. The roots it
is given are remembered, and a call without roots rescans every remembered one.

Per root, in one transaction, the root's rows are replaced by what the scan found: a
run directory that disappeared loses its row, a changed one is reread. A directory that
cannot be read becomes a :class:`Skipped` entry with the reason, never a partial row.
A remembered root that no longer exists is reported and its rows are left as they are
(an unmounted disk is not a deletion); a root passed explicitly must exist.

**Artefacts.** Every file of a run directory gets a row with its kind
(``runs.layout.classify_artefact``), size, mtime and SHA-256. A file whose size and mtime
are those of its previous row keeps its recorded SHA-256 instead of being hashed again,
so a rescan of unchanged checkpoints is cheap; ``rebuild`` rehashes everything.

**Specs.** The experiment specs the queue accepted (task A3) are files in
``$ASTRO_SIMCLR_HOME/specs`` (``*.yaml``, ``*.yml``, ``*.json``). They are read with
``astro_simclr.runs.spec.load_spec`` and indexed by id and name, so that
``runs compare`` takes an id. Until task A3's module exists, no spec is indexed.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from astro_simclr.data.files import sha256_file
from astro_simclr.registry.home import RegistryError
from astro_simclr.registry.readers import RecordError, RunRecord, is_run_dir, read_run_dir
from astro_simclr.registry.spec_view import SpecUnavailableError, load_spec_file
from astro_simclr.registry.store import Artefact, insert_run, record_row, transaction
from astro_simclr.runs.layout import classify_artefact
from astro_simclr.runs.manifest import utc_now

__all__ = ["SPEC_SUFFIXES", "ReindexReport", "Skipped", "find_run_dirs", "reindex"]

#: The file suffixes of experiment specs in the specs directory.
SPEC_SUFFIXES: Final = frozenset({".yaml", ".yml", ".json"})


@dataclass(frozen=True, slots=True)
class Skipped:
    """A directory or spec file that was not indexed, and why."""

    path: Path
    reason: str


@dataclass(slots=True)
class ReindexReport:
    """What one :func:`reindex` did: per root, the runs indexed; what was skipped."""

    indexed: dict[Path, int] = field(default_factory=dict)
    missing_roots: list[Path] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    specs: int = 0


def find_run_dirs(root: Path) -> Iterator[Path]:
    """The run directories under ``root`` (module docstring), in sorted walk order."""
    for dirpath, dirnames, _ in os.walk(root, followlinks=False):
        here = Path(dirpath)
        if is_run_dir(here):
            dirnames[:] = []
            yield here
            continue
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))


def _files(run_dir: Path) -> Iterator[tuple[str, os.stat_result]]:
    for dirpath, dirnames, filenames in os.walk(run_dir, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            yield path.relative_to(run_dir).as_posix(), path.stat()


def _artefacts(
    run_dir: Path, previous: dict[str, Artefact], hasher: Callable[[Path], str]
) -> list[Artefact]:
    out = []
    for relative, stat in _files(run_dir):
        old = previous.get(relative)
        if old is not None and (old.size, old.mtime_ns) == (stat.st_size, stat.st_mtime_ns):
            sha = old.sha256
        else:
            sha = hasher(run_dir / relative)
        kind = str(classify_artefact(relative))
        out.append(Artefact(relative, kind, stat.st_size, stat.st_mtime_ns, sha))
    return out


def _previous_artefacts(conn: sqlite3.Connection, run_dir: Path) -> dict[str, Artefact]:
    rows = conn.execute(
        "SELECT path, kind, size, mtime_ns, sha256 FROM artefacts WHERE run_dir = ?",
        (str(run_dir),),
    ).fetchall()
    return {row[0]: Artefact(*row) for row in rows}


def _under(root: Path) -> tuple[str, str]:
    """``WHERE run_dir = ? OR run_dir LIKE ? ESCAPE '\\'`` parameters for ``root``'s rows."""
    escaped = str(root).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return str(root), escaped.rstrip("/") + "/%"


def _scan_root(
    conn: sqlite3.Connection,
    root: Path,
    report: ReindexReport,
    hasher: Callable[[Path], str],
) -> None:
    found: list[tuple[RunRecord, list[Artefact]]] = []
    for run_dir in find_run_dirs(root):
        try:
            record = read_run_dir(run_dir)
            artefacts = _artefacts(
                record.run_dir, _previous_artefacts(conn, record.run_dir), hasher
            )
        except (RecordError, OSError) as err:
            report.skipped.append(Skipped(run_dir, str(err)))
            continue
        found.append((record, artefacts))
    now = utc_now()
    with transaction(conn):
        conn.execute(
            "DELETE FROM runs WHERE root = ? OR run_dir = ? OR run_dir LIKE ? ESCAPE '\\'",
            (str(root), *_under(root)),
        )
        for record, artefacts in found:
            row = record_row(record, root, now)
            # A run under two roots (one root inside another) belongs to the last scan.
            conn.execute("DELETE FROM runs WHERE run_dir = ?", (row["run_dir"],))
            insert_run(conn, row)
            conn.executemany(
                "INSERT INTO artefacts (run_dir, path, kind, size, mtime_ns, sha256) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(row["run_dir"], a.path, a.kind, a.size, a.mtime_ns, a.sha256) for a in artefacts],
            )
    report.indexed[root] = len(found)


def _scan_specs(conn: sqlite3.Connection, specs: Path, report: ReindexReport) -> None:
    rows: list[tuple[str, str, str]] = []
    if specs.is_dir():
        for path in sorted(specs.iterdir()):
            if path.suffix not in SPEC_SUFFIXES or not path.is_file():
                continue
            try:
                spec = load_spec_file(path)
            except SpecUnavailableError:
                break
            except (ValueError, OSError) as err:
                report.skipped.append(Skipped(path, str(err)))
                continue
            rows.append((spec.spec_id, spec.name, str(path.resolve())))
    with transaction(conn):
        conn.execute("DELETE FROM specs")
        conn.executemany("INSERT OR REPLACE INTO specs VALUES (?, ?, ?)", rows)
    report.specs = len(rows)


def reindex(
    conn: sqlite3.Connection,
    roots: Sequence[Path] = (),
    *,
    specs: Path,
    rebuild: bool = False,
    hasher: Callable[[Path], str] = sha256_file,
) -> ReindexReport:
    """Scan ``roots`` (default: every remembered root) and ``specs`` into the registry.

    ``roots`` are made absolute and remembered. ``rebuild`` first empties the runs,
    artefacts and specs (the roots stay), so every file is reread and rehashed.
    ``hasher`` is for the tests.

    Raises:
        RegistryError: if a root in ``roots`` is not a directory.
    """
    report = ReindexReport()
    given = [root.expanduser().resolve() for root in roots]
    for root in given:
        if not root.is_dir():
            raise RegistryError(f"{root}: not a directory")
    now = utc_now()
    with transaction(conn):
        conn.executemany(
            "INSERT OR IGNORE INTO roots (path, added_at) VALUES (?, ?)",
            [(str(root), now) for root in given],
        )
        if rebuild:
            conn.execute("DELETE FROM runs")
            conn.execute("DELETE FROM specs")
    targets = given or [
        Path(row[0]) for row in conn.execute("SELECT path FROM roots ORDER BY path")
    ]
    for root in targets:
        if not root.is_dir():
            report.missing_roots.append(root)
            continue
        _scan_root(conn, root, report, hasher)
    _scan_specs(conn, specs, report)
    return report
