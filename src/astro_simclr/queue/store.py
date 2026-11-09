"""The durable queue: one SQLite file of jobs with heartbeat leases (task A3; ADR 0026).

The file is ``$ASTRO_SIMCLR_HOME/queue/queue.sqlite`` on the runner (the GPU machine),
checked by :func:`~astro_simclr.queue.location.check_queue_location` (a local disk, never
``/mnt/c`` or a synced folder). It is the runner's state and is never copied: the laptop
learns about runs from their run directories (``queue sync``, then ``runs reindex``).
Journal mode WAL with ``synchronous=FULL``: a committed claim or result survives a power
cut or ``wsl --shutdown``.

**Jobs.** One row per attempt of one cell of a spec: its run id, attempt, run directory and
the argv the worker runs (a JSON array, fixed when the job is queued). Statuses are the
manifest's (``runs.manifest``): ``queued -> running -> finished | failed | cancelled``, and
``running -> queued`` again when a lease expires or a job is killed by a signal.

**Leases.** :meth:`QueueStore.claim` moves the oldest admissible queued job to ``running``
under a fresh random lease token, valid ``lease_seconds``; the worker renews it with
:meth:`QueueStore.heartbeat`. Every later write of the job (:meth:`QueueStore.finish`,
:meth:`QueueStore.requeue`) names the token and is ignored if the token is no longer the
job's: a worker whose lease was taken over cannot overwrite the new holder's state.
:meth:`QueueStore.expired` lists running jobs whose lease ran out; the reaper requeues them
(:meth:`QueueStore.requeue` with their token), and the job resumes from ``last`` in the
same run directory. A job requeued more than ``max_resumes`` times is ``failed``.

**Attempts.** :meth:`QueueStore.add_job` gives a cell the attempt after the last one of its
run id; a cell whose run id already has a queued, running or finished attempt is not
queued again unless a new attempt is asked for (an identical-config rerun, ADR 0015).

**Peak memory.** :meth:`QueueStore.record_peak_memory` keeps the largest measured peak per
config hash and machine class, which admission reads (:mod:`~astro_simclr.queue.admission`).

All times are seconds since the epoch, passed in by the caller (a clock the tests control).
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from astro_simclr.queue.location import check_queue_location
from astro_simclr.runs.manifest import STATUSES, Status

__all__ = [
    "ACTIVE_STATUSES",
    "DEFAULT_MAX_RESUMES",
    "SCHEMA_VERSION",
    "Job",
    "NewJob",
    "QueueError",
    "QueueStore",
    "iso",
]

#: ``PRAGMA user_version`` of the schema below; a newer file is refused.
SCHEMA_VERSION: Final = 1
#: Requeues (lease expiries, kills) a job survives before it is ``failed``.
DEFAULT_MAX_RESUMES: Final = 5
#: Statuses that keep a run id from being queued again (without a new attempt).
ACTIVE_STATUSES: Final = frozenset({"queued", "running", "finished"})

_STATUS_LIST: Final = ", ".join(f"'{status}'" for status in STATUSES)
_SCHEMA: Final = f"""
CREATE TABLE IF NOT EXISTS specs (
    spec_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    scratch INTEGER NOT NULL,
    submitted_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id TEXT NOT NULL REFERENCES specs (spec_id),
    cell TEXT NOT NULL,
    run_id TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    git_sha TEXT NOT NULL,
    machine_class TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    cwd TEXT NOT NULL,
    argv TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_STATUS_LIST})),
    lease_token TEXT,
    lease_owner TEXT,
    lease_expires REAL,
    host TEXT,
    boot_id TEXT,
    pid INTEGER,
    slot TEXT,
    resumes INTEGER NOT NULL DEFAULT 0,
    max_resumes INTEGER NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    not_before REAL,
    exit_code INTEGER,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    ended_at REAL,
    UNIQUE (run_id, attempt)
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, machine_class, job_id);
CREATE TABLE IF NOT EXISTS peak_memory (
    config_sha256 TEXT NOT NULL,
    machine_class TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    measured_at REAL NOT NULL,
    PRIMARY KEY (config_sha256, machine_class)
);
"""


class QueueError(RuntimeError):
    """The queue file is unusable (newer schema, not a queue) or a request is invalid."""


def iso(seconds: float | None) -> str | None:
    """``seconds`` since the epoch as ISO 8601 UTC (for display), ``None`` for ``None``."""
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True, kw_only=True)
class NewJob:
    """What :meth:`QueueStore.add_job` needs of one cell (its attempt is chosen there)."""

    spec_id: str
    cell: str
    run_id: str
    config_sha256: str
    git_sha: str
    machine_class: str
    cwd: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    """One row of ``jobs`` (module docstring)."""

    job_id: int
    spec_id: str
    cell: str
    run_id: str
    config_sha256: str
    attempt: int
    git_sha: str
    machine_class: str
    run_dir: Path
    cwd: Path
    argv: tuple[str, ...]
    status: Status
    lease_token: str | None
    lease_owner: str | None
    lease_expires: float | None
    host: str | None
    boot_id: str | None
    pid: int | None
    slot: str | None
    resumes: int
    max_resumes: int
    cancel_requested: bool
    not_before: float | None
    exit_code: int | None
    error: str | None
    created_at: float
    started_at: float | None
    ended_at: float | None

    def to_plain(self) -> dict[str, object]:
        """The job as JSON values (``queue status --json``), times in ISO 8601 UTC."""
        return {
            "job_id": self.job_id,
            "spec_id": self.spec_id,
            "cell": self.cell,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "git_sha": self.git_sha,
            "machine_class": self.machine_class,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "lease_owner": self.lease_owner,
            "lease_expires": iso(self.lease_expires),
            "host": self.host,
            "pid": self.pid,
            "slot": self.slot,
            "resumes": self.resumes,
            "cancel_requested": self.cancel_requested,
            "exit_code": self.exit_code,
            "error": self.error,
            "created_at": iso(self.created_at),
            "started_at": iso(self.started_at),
            "ended_at": iso(self.ended_at),
        }


def _job(row: sqlite3.Row) -> Job:
    argv = json.loads(row["argv"])
    return Job(
        job_id=row["job_id"],
        spec_id=row["spec_id"],
        cell=row["cell"],
        run_id=row["run_id"],
        config_sha256=row["config_sha256"],
        attempt=row["attempt"],
        git_sha=row["git_sha"],
        machine_class=row["machine_class"],
        run_dir=Path(row["run_dir"]),
        cwd=Path(row["cwd"]),
        argv=tuple(str(arg) for arg in argv),
        status=cast(Status, row["status"]),
        lease_token=row["lease_token"],
        lease_owner=row["lease_owner"],
        lease_expires=row["lease_expires"],
        host=row["host"],
        boot_id=row["boot_id"],
        pid=row["pid"],
        slot=row["slot"],
        resumes=row["resumes"],
        max_resumes=row["max_resumes"],
        cancel_requested=bool(row["cancel_requested"]),
        not_before=row["not_before"],
        exit_code=row["exit_code"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


#: What :meth:`QueueStore.add_job` calls with the chosen attempt: it returns the run
#: directory and the argv of the job.
type CommandFor = Callable[[int], tuple[Path, Sequence[str]]]
#: What :meth:`QueueStore.claim` calls with the head job: the slot name, or ``None``.
type Admit = Callable[[Job], str | None]


class QueueStore:
    """The queue file (module docstring). Open with :meth:`open`; close with :meth:`close`."""

    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        """Wrap an open connection (use :meth:`open`)."""
        self._db = connection
        self.path = path

    @classmethod
    def open(cls, path: Path, *, mounts: Sequence[tuple[str, str]] | None = None) -> QueueStore:
        """Open (creating if needed) the queue file at ``path``.

        ``mounts`` is passed to :func:`~astro_simclr.queue.location.check_queue_location`.

        Raises:
            LocationError: if ``path`` is on an unsafe file system.
            QueueError: if the file has a newer schema or is not a queue database.
        """
        resolved = check_queue_location(path, mounts=mounts)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: transactions are explicit (BEGIN IMMEDIATE below).
        db = sqlite3.connect(resolved, timeout=30.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                msg = (
                    f"{resolved}: queue schema {version}; this version of astro-simclr reads "
                    f"up to {SCHEMA_VERSION}"
                )
                raise QueueError(msg)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(_SCHEMA)
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        except sqlite3.DatabaseError as err:
            db.close()
            raise QueueError(f"{resolved}: not a usable queue database ({err})") from None
        except QueueError:
            db.close()
            raise
        return cls(db, resolved)

    def close(self) -> None:
        """Close the connection."""
        self._db.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        # BEGIN IMMEDIATE takes the write lock first: two workers never claim one job.
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # --- specs and jobs ---------------------------------------------------------------------

    def add_spec(self, spec_id: str, name: str, path: Path, *, scratch: bool, now: float) -> None:
        """Record a submitted spec (once per spec id; a resubmission keeps the first)."""
        with self._write() as db:
            db.execute(
                "INSERT OR IGNORE INTO specs VALUES (?, ?, ?, ?, ?)",
                (spec_id, name, str(path), int(scratch), now),
            )

    def add_job(
        self,
        job: NewJob,
        command_for: CommandFor,
        *,
        now: float,
        new_attempt: bool = False,
        max_resumes: int = DEFAULT_MAX_RESUMES,
    ) -> tuple[Job, bool]:
        """Queue one cell; return its job and whether it was added (module docstring).

        Without ``new_attempt``, a run id that already has a queued, running or finished
        attempt is not queued again: that job is returned with ``False``. Otherwise the job
        gets the next attempt, and ``command_for(attempt)`` gives its run directory and argv.

        Raises:
            QueueError: if the spec was not added first.
        """
        with self._write() as db:
            if not new_attempt:
                placeholders = ", ".join("?" * len(ACTIVE_STATUSES))
                row = db.execute(
                    f"SELECT * FROM jobs WHERE run_id = ? AND status IN ({placeholders}) "  # noqa: S608
                    "ORDER BY attempt DESC LIMIT 1",
                    (job.run_id, *sorted(ACTIVE_STATUSES)),
                ).fetchone()
                if row is not None:
                    return _job(row), False
            last = db.execute(
                "SELECT MAX(attempt) FROM jobs WHERE run_id = ?", (job.run_id,)
            ).fetchone()[0]
            attempt = 1 if last is None else int(last) + 1
            run_dir, argv = command_for(attempt)
            try:
                cursor = db.execute(
                    "INSERT INTO jobs (spec_id, cell, run_id, config_sha256, attempt, git_sha, "
                    "machine_class, run_dir, cwd, argv, status, max_resumes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                    (
                        job.spec_id,
                        job.cell,
                        job.run_id,
                        job.config_sha256,
                        attempt,
                        job.git_sha,
                        job.machine_class,
                        str(run_dir),
                        str(job.cwd),
                        json.dumps([str(arg) for arg in argv]),
                        max_resumes,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as err:
                raise QueueError(f"cannot queue {job.cell}: {err}") from None
            row = db.execute("SELECT * FROM jobs WHERE job_id = ?", (cursor.lastrowid,)).fetchone()
        return _job(row), True

    def spec_ids(self, prefix: str) -> list[str]:
        """The spec ids that start with ``prefix``, sorted."""
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._db.execute(
            "SELECT spec_id FROM specs WHERE spec_id LIKE ? ESCAPE '\\' ORDER BY spec_id",
            (escaped + "%",),
        ).fetchall()
        return [str(row["spec_id"]) for row in rows]

    def job(self, job_id: int) -> Job | None:
        """The job ``job_id``, or ``None``."""
        row = self._db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else _job(row)

    def jobs(self, *, spec_id: str | None = None, status: Status | None = None) -> list[Job]:
        """Every job, oldest first, optionally of one spec (an id or a unique prefix) or status."""
        query, args = "SELECT * FROM jobs WHERE 1 = 1", []
        if spec_id is not None:
            query += " AND spec_id LIKE ? ESCAPE '\\'"
            args.append(spec_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if status is not None:
            query += " AND status = ?"
            args.append(status)
        rows = self._db.execute(query + " ORDER BY job_id", args).fetchall()
        return [_job(row) for row in rows]

    # --- leases -----------------------------------------------------------------------------

    def claim(
        self,
        *,
        owner: str,
        machine_class: str,
        host: str,
        boot_id: str | None,
        admit: Admit,
        now: float,
        lease_seconds: float,
    ) -> Job | None:
        """Lease the oldest queued job of ``machine_class`` that may start, or ``None``.

        Only the head of the queue is considered (first in, first out): if ``admit``
        returns ``None`` for it, nothing is claimed, so a large job is not starved by small
        ones behind it. A job with a ``not_before`` in the future is skipped.
        """
        with self._write() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE status = 'queued' AND machine_class = ? "
                "AND (not_before IS NULL OR not_before <= ?) ORDER BY job_id LIMIT 1",
                (machine_class, now),
            ).fetchone()
            if row is None:
                return None
            head = _job(row)
            slot = admit(head)
            if slot is None:
                return None
            token = secrets.token_hex(16)
            db.execute(
                "UPDATE jobs SET status = 'running', lease_token = ?, lease_owner = ?, "
                "lease_expires = ?, host = ?, boot_id = ?, slot = ?, pid = NULL, "
                "not_before = NULL, started_at = COALESCE(started_at, ?) WHERE job_id = ?",
                (token, owner, now + lease_seconds, host, boot_id, slot, now, head.job_id),
            )
            row = db.execute("SELECT * FROM jobs WHERE job_id = ?", (head.job_id,)).fetchone()
        return _job(row)

    def set_pid(self, job_id: int, token: str, pid: int) -> bool:
        """Record the process (and process group) of a leased job; ``False`` if not leased."""
        with self._write() as db:
            cursor = db.execute(
                "UPDATE jobs SET pid = ? WHERE job_id = ? AND lease_token = ?",
                (pid, job_id, token),
            )
        return cursor.rowcount == 1

    def heartbeat(self, leases: dict[int, str], *, now: float, lease_seconds: float) -> set[int]:
        """Renew the leases ``{job_id: token}``; return the ids whose lease is lost.

        A lease is lost when the job's token changed (the reaper requeued it) or the job is
        no longer running.
        """
        lost: set[int] = set()
        with self._write() as db:
            for job_id, token in leases.items():
                cursor = db.execute(
                    "UPDATE jobs SET lease_expires = ? WHERE job_id = ? AND lease_token = ? "
                    "AND status = 'running'",
                    (now + lease_seconds, job_id, token),
                )
                if cursor.rowcount != 1:
                    lost.add(job_id)
        return lost

    def cancel_requested(self, job_ids: Sequence[int]) -> set[int]:
        """The ids among ``job_ids`` whose cancellation was asked for."""
        if not job_ids:
            return set()
        placeholders = ", ".join("?" * len(job_ids))
        rows = self._db.execute(
            f"SELECT job_id FROM jobs WHERE cancel_requested = 1 AND job_id IN ({placeholders})",  # noqa: S608
            tuple(job_ids),
        ).fetchall()
        return {row["job_id"] for row in rows}

    def expired(self, *, now: float) -> list[Job]:
        """Running jobs whose lease ran out before ``now``."""
        rows = self._db.execute(
            "SELECT * FROM jobs WHERE status = 'running' AND lease_expires < ? ORDER BY job_id",
            (now,),
        ).fetchall()
        return [_job(row) for row in rows]

    def finish(
        self,
        job_id: int,
        token: str,
        status: Literal["finished", "failed", "cancelled"],
        *,
        now: float,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> bool:
        """End a leased job with ``status``; ``False`` (nothing written) if the lease is lost."""
        with self._write() as db:
            cursor = db.execute(
                "UPDATE jobs SET status = ?, exit_code = ?, error = ?, ended_at = ?, "
                "lease_token = NULL, lease_expires = NULL WHERE job_id = ? AND lease_token = ? "
                "AND status = 'running'",
                (status, exit_code, error, now, job_id, token),
            )
        return cursor.rowcount == 1

    def requeue(
        self,
        job_id: int,
        token: str,
        *,
        now: float,
        reason: str,
        exit_code: int | None = None,
        not_before: float | None = None,
        count: bool = True,
        expired_only: bool = False,
    ) -> Status | None:
        """Put a leased running job back in the queue, to resume from its ``last`` checkpoint.

        ``count`` adds one to its requeues; past ``max_resumes`` it is ``failed`` instead. A
        job whose cancellation was asked for is ``cancelled`` instead (its worker is gone).
        ``expired_only`` (the reaper) writes only if the lease has expired by ``now``.
        Returns the new status, or ``None`` if the lease is lost (nothing written).
        """
        with self._write() as db:
            query = (
                "SELECT resumes, max_resumes, cancel_requested FROM jobs WHERE job_id = ? "
                "AND lease_token = ? AND status = 'running'"
            )
            args: tuple[object, ...] = (job_id, token)
            if expired_only:
                query += " AND lease_expires < ?"
                args = (*args, now)
            row = db.execute(query, args).fetchone()
            if row is None:
                return None
            if row["cancel_requested"]:
                db.execute(
                    "UPDATE jobs SET status = 'cancelled', exit_code = ?, error = ?, ended_at = ?, "
                    "lease_token = NULL, lease_expires = NULL WHERE job_id = ?",
                    (exit_code, f"cancelled ({reason})", now, job_id),
                )
                return "cancelled"
            resumes = row["resumes"] + (1 if count else 0)
            if resumes > row["max_resumes"]:
                db.execute(
                    "UPDATE jobs SET status = 'failed', resumes = ?, exit_code = ?, error = ?, "
                    "ended_at = ?, lease_token = NULL, lease_expires = NULL WHERE job_id = ?",
                    (
                        resumes,
                        exit_code,
                        f"{reason}; gave up after {resumes - 1} resumes",
                        now,
                        job_id,
                    ),
                )
                return "failed"
            db.execute(
                "UPDATE jobs SET status = 'queued', resumes = ?, exit_code = ?, error = ?, "
                "not_before = ?, lease_token = NULL, lease_owner = NULL, lease_expires = NULL, "
                "pid = NULL, slot = NULL WHERE job_id = ?",
                (resumes, exit_code, reason, not_before, job_id),
            )
        return "queued"

    def cancel(self, job_ids: Sequence[int], *, now: float) -> dict[int, str]:
        """Cancel jobs: a queued one at once, a running one by asking its worker.

        Returns ``{job_id: "cancelled" | "cancel requested" | <status>}`` (a finished,
        failed or cancelled job is left as it is).
        """
        out: dict[int, str] = {}
        with self._write() as db:
            for job_id in job_ids:
                row = db.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if row is None:
                    out[job_id] = "no such job"
                elif row["status"] == "queued":
                    db.execute(
                        "UPDATE jobs SET status = 'cancelled', ended_at = ? WHERE job_id = ?",
                        (now, job_id),
                    )
                    out[job_id] = "cancelled"
                elif row["status"] == "running":
                    db.execute("UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?", (job_id,))
                    out[job_id] = "cancel requested"
                else:
                    out[job_id] = str(row["status"])
        return out

    def running(self, *, owner: str | None = None) -> list[Job]:
        """Running jobs, optionally of one lease owner."""
        if owner is None:
            rows = self._db.execute("SELECT * FROM jobs WHERE status = 'running'").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM jobs WHERE status = 'running' AND lease_owner = ?", (owner,)
            ).fetchall()
        return [_job(row) for row in rows]

    # --- peak memory ------------------------------------------------------------------------

    def record_peak_memory(
        self, config_sha256: str, machine_class: str, peak_bytes: int, *, now: float
    ) -> None:
        """Keep the largest measured peak of a config hash on a machine class."""
        with self._write() as db:
            db.execute(
                "INSERT INTO peak_memory VALUES (?, ?, ?, ?) ON CONFLICT "
                "(config_sha256, machine_class) DO UPDATE SET bytes = MAX(bytes, excluded.bytes), "
                "measured_at = excluded.measured_at",
                (config_sha256, machine_class, peak_bytes, now),
            )

    def peak_memory(self, config_sha256: str, machine_class: str) -> int | None:
        """The measured peak of a config hash on a machine class, in bytes, or ``None``."""
        row = self._db.execute(
            "SELECT bytes FROM peak_memory WHERE config_sha256 = ? AND machine_class = ?",
            (config_sha256, machine_class),
        ).fetchone()
        return None if row is None else int(row["bytes"])
