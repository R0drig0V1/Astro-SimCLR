"""The queue worker: claim, run, heartbeat, reap (task A3; ADR 0016).

One worker per machine runs the jobs of one machine class on its slots
(:mod:`~astro_simclr.queue.admission`). Each :meth:`Worker.step`:

1. **renews** the leases of its own jobs (:meth:`QueueStore.heartbeat`); a job whose lease
   was lost (requeued by another worker after this one stalled) is killed and forgotten;
2. **cancels** the jobs whose cancellation was asked for: SIGTERM to the process group,
   SIGKILL after the grace period, then ``cancelled`` in the queue and the manifest;
3. **collects** the jobs that ended (exit codes in :func:`outcome`);
4. **reaps** expired leases of other (dead) workers. An expired job that ran on this host
   in this boot may still be alive, an orphan of a worker killed with ``kill -9``: its
   process group is killed first, but only if the process still runs the job's run
   directory (``/proc/<pid>/cmdline`` or ``ps``), so a recycled pid is never killed. Then
   the job is requeued and resumes from ``last``. A job from another boot is gone already.
5. **claims** queued jobs while a slot admits the head of the queue, and starts each one as
   ``subprocess.Popen(job.argv, cwd=job.cwd, start_new_session=True)``: the argv stored at
   enqueue time, no shell, its own process group (so it can be killed as a whole), stdin
   closed, stdout and stderr appended to ``logs/<job id>.log``, ``CUDA_VISIBLE_DEVICES`` set
   to the slot's GPU.

:meth:`Worker.shutdown` (SIGTERM from systemd or the CLI) stops claiming, sends SIGTERM to
every job (``spec run-cell`` then sets its manifest back to ``queued``), waits for the grace
period, SIGKILLs what is left and requeues all of them at once, without waiting for the
lease to expire.

**Exit codes** (:func:`outcome`): 0 is ``finished``; ``job.EXIT_BUSY`` (the run directory is
held by another process) is a requeue after one lease period, not counted; death by a
signal (a negative return code) or an exit of 128 + a signal (``EXIT_TERMINATED``) is a
counted requeue, ``failed`` after ``max_resumes``; any other code is ``failed`` (an
exception is not retried: it would fail again).

The clock, the host name and the boot id are injectable, so the tests run every path with
real processes on CPU (``tests/test_queue_worker.py``).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Final, Literal

from astro_simclr.queue.admission import Slot, choose_slot
from astro_simclr.queue.job import CONCURRENT_JOBS_ENV, EXIT_BUSY, JOB_RESULT_FILE
from astro_simclr.queue.store import Job, QueueStore
from astro_simclr.runs.manifest import (
    TERMINAL_STATUSES,
    ManifestError,
    read_manifest,
    transition,
    write_manifest,
)

__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_POLL_SECONDS",
    "Outcome",
    "Worker",
    "cancel_manifest",
    "current_boot_id",
    "kill_group",
    "outcome",
    "process_runs",
]

#: How long a lease lasts without a heartbeat. A worker beats every poll (5 s), so 2
#: minutes survives a long pause (a slow disk, a suspended laptop on the other end does
#: not matter: the lease is local) and requeues a dead worker's jobs within 2 minutes.
DEFAULT_LEASE_SECONDS: Final = 120.0
#: Seconds between two steps.
DEFAULT_POLL_SECONDS: Final = 5.0
#: Seconds between SIGTERM and SIGKILL (Lightning writes no checkpoint on SIGTERM, so
#: this only lets the manifest be rewritten).
DEFAULT_GRACE_SECONDS: Final = 30.0

type Outcome = tuple[Literal["finished", "failed", "requeue", "retry"], str]


def outcome(returncode: int) -> Outcome:
    """What a job's exit code means (module docstring, "Exit codes").

    ``requeue`` is counted against ``max_resumes``; ``retry`` is not.
    """
    if returncode == 0:
        return "finished", "exit 0"
    if returncode == EXIT_BUSY:
        return "retry", "the run directory is held by another process"
    if returncode < 0:
        return "requeue", f"killed by signal {-returncode}"
    if returncode > 128 and returncode - 128 in {s.value for s in signal.Signals}:
        return "requeue", f"exit {returncode} (signal {returncode - 128})"
    return "failed", f"exit {returncode}"


def current_boot_id() -> str | None:
    """Linux's boot id (``/proc/sys/kernel/random/boot_id``), ``None`` elsewhere."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return None


def _command_line(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        pass
    try:
        # A fixed argv, no shell; `ps` from PATH (macOS and Linux without /proc).
        result = subprocess.run(  # noqa: S603
            ["ps", "-o", "command=", "-p", str(pid)],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def process_runs(pid: int, run_dir: Path) -> bool:
    """Whether process ``pid`` is alive and its command line names ``run_dir``.

    The check that keeps the reaper from killing a recycled pid.
    """
    line = _command_line(pid)
    return line is not None and str(run_dir) in line


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - another user's group: not ours
        return False
    return True


def kill_group(
    pgid: int,
    *,
    grace: float,
    first: signal.Signals = signal.SIGTERM,
    wait: Callable[[], bool] | None = None,
) -> None:
    """Send ``first`` to process group ``pgid``, then SIGKILL if it outlives ``grace``.

    ``wait`` reports whether the group is gone (default: ``killpg(pgid, 0)`` fails); a
    worker passes its own child's ``poll``, since an unreaped child stays a zombie.
    """
    gone = wait or (lambda: not _group_alive(pgid))
    try:
        os.killpg(pgid, first)
    except (ProcessLookupError, PermissionError):
        # No such group, or (macOS) a group left with zombies only: nothing to kill.
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if gone():
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + 10
    while not gone() and time.monotonic() < deadline:
        time.sleep(0.05)


@dataclass
class _Running:
    job: Job
    token: str
    process: subprocess.Popen[bytes]
    log: IO[bytes]
    slot: str
    cancelling: bool = False


@dataclass
class Worker:
    """A worker of one machine class on ``slots`` (module docstring)."""

    store: QueueStore
    machine_class: str
    slots: Sequence[Slot]
    log_dir: Path
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}")
    host: str = field(default_factory=socket.gethostname)
    boot_id: str | None = field(default_factory=current_boot_id)
    lease_seconds: float = DEFAULT_LEASE_SECONDS
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    clock: Callable[[], float] = time.time
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    _running: dict[int, _Running] = field(default_factory=dict)
    _stopping: bool = False

    @property
    def running_jobs(self) -> list[Job]:
        """The jobs this worker is running."""
        return [entry.job for entry in self._running.values()]

    def step(self) -> None:
        """One round: heartbeat, cancel, collect, reap, claim (module docstring)."""
        now = self.clock()
        leases = {job_id: entry.token for job_id, entry in self._running.items()}
        for job_id in self.store.heartbeat(leases, now=now, lease_seconds=self.lease_seconds):
            self._abandon(job_id)
        for job_id in self.store.cancel_requested(list(self._running)):
            self._cancel(job_id)
        self._collect()
        self._reap()
        if not self._stopping:
            self._claim_all()

    def run(self, *, poll_seconds: float = DEFAULT_POLL_SECONDS, idle_exit: bool = False) -> None:
        """Step until :meth:`shutdown` (or, with ``idle_exit``, until nothing is left to do).

        SIGTERM and SIGHUP call :meth:`shutdown`.
        """

        def on_signal(_signum: int, _frame: object) -> None:
            self._stopping = True

        previous = {
            signum: signal.signal(signum, on_signal) for signum in (signal.SIGTERM, signal.SIGHUP)
        }
        try:
            while not self._stopping:
                self.step()
                if idle_exit and not self._running and not self._has_work():
                    break
                time.sleep(poll_seconds)
        finally:
            self.shutdown()
            for signum, old in previous.items():
                signal.signal(signum, old)

    def _has_work(self) -> bool:
        return any(
            job.machine_class == self.machine_class for job in self.store.jobs(status="queued")
        )

    def shutdown(self) -> None:
        """Stop every job (SIGTERM, then SIGKILL after the grace) and requeue it at once."""
        self._stopping = True
        for job_id in list(self._running):
            entry = self._running.pop(job_id)
            _stop(entry, grace=self.grace_seconds)
            self.store.requeue(
                job_id,
                entry.token,
                now=self.clock(),
                reason="worker shut down",
                exit_code=entry.process.returncode,
                count=False,
            )

    # --- internals --------------------------------------------------------------------------

    def _abandon(self, job_id: int) -> None:
        """Our lease is gone (someone requeued the job): stop it, write nothing."""
        _stop(self._running.pop(job_id), grace=0, first=signal.SIGKILL)

    def _cancel(self, job_id: int) -> None:
        entry = self._running[job_id]
        if entry.cancelling:
            return
        entry.cancelling = True
        kill_group(
            entry.process.pid,
            grace=self.grace_seconds,
            wait=lambda: entry.process.poll() is not None,
        )

    def _collect(self) -> None:
        for job_id, entry in list(self._running.items()):
            code = entry.process.poll()
            if code is None:
                continue
            del self._running[job_id]
            entry.log.close()
            now = self.clock()
            if entry.cancelling:
                self.store.finish(job_id, entry.token, "cancelled", now=now, exit_code=code)
                cancel_manifest(entry.job.run_dir)
                continue
            kind, reason = outcome(code)
            if kind == "finished":
                peak = _peak_memory(entry.job.run_dir)
                if peak is not None:
                    self.store.record_peak_memory(
                        entry.job.config_sha256, entry.job.machine_class, peak, now=now
                    )
                self.store.finish(job_id, entry.token, "finished", now=now, exit_code=code)
            elif kind == "failed":
                self.store.finish(
                    job_id, entry.token, "failed", now=now, exit_code=code, error=reason
                )
            else:
                self.store.requeue(
                    job_id,
                    entry.token,
                    now=now,
                    reason=reason,
                    exit_code=code,
                    count=kind == "requeue",
                    not_before=now + self.lease_seconds if kind == "retry" else None,
                )

    def _reap(self) -> None:
        now = self.clock()
        for job in self.store.expired(now=now):
            if job.job_id in self._running or job.lease_token is None:
                continue  # our own lease (we stalled): the heartbeat renews it next step
            same_boot = job.host == self.host and job.boot_id == self.boot_id
            pid = job.pid
            if same_boot and pid is not None and process_runs(pid, job.run_dir):
                # Gone when no live process runs the job (a zombie has no command line).
                kill_group(
                    pid,
                    grace=self.grace_seconds,
                    wait=lambda: not process_runs(pid, job.run_dir),  # noqa: B023
                )
            self.store.requeue(
                job.job_id, job.lease_token, now=now, reason=f"lease of {job.lease_owner} expired"
            )

    def _peaks(self) -> dict[str, list[int | None]]:
        running: dict[str, list[int | None]] = {}
        for entry in self._running.values():
            peak = self.store.peak_memory(entry.job.config_sha256, entry.job.machine_class)
            running.setdefault(entry.slot, []).append(peak)
        return running

    def _claim_all(self) -> None:
        while True:
            running = self._peaks()

            def admit(job: Job, running: dict[str, list[int | None]] = running) -> str | None:
                peak = self.store.peak_memory(job.config_sha256, job.machine_class)
                slot = choose_slot(self.slots, peak, running)
                return None if slot is None else slot.name

            job = self.store.claim(
                owner=self.worker_id,
                machine_class=self.machine_class,
                host=self.host,
                boot_id=self.boot_id,
                admit=admit,
                now=self.clock(),
                lease_seconds=self.lease_seconds,
            )
            if job is None:
                return
            self._start(job, running)

    def _start(self, job: Job, running: Mapping[str, Sequence[int | None]]) -> None:
        if job.lease_token is None or job.slot is None:  # pragma: no cover - claim sets both
            return
        slot = next(slot for slot in self.slots if slot.name == job.slot)
        env = dict(self.env)
        env[CONCURRENT_JOBS_ENV] = str(len(running.get(slot.name, ())) + 1)
        if slot.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = (self.log_dir / f"{job.job_id}.log").open("ab")
        try:
            # The argv fixed at enqueue time: a list, never a shell string.
            process = subprocess.Popen(  # noqa: S603
                list(job.argv),
                cwd=job.cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as err:
            log.close()
            self.store.finish(
                job.job_id,
                job.lease_token,
                "failed",
                now=self.clock(),
                error=f"cannot start: {err}",
            )
            return
        self.store.set_pid(job.job_id, job.lease_token, process.pid)
        self._running[job.job_id] = _Running(job, job.lease_token, process, log, slot.name)


def _peak_memory(run_dir: Path) -> int | None:
    try:
        data = json.loads((run_dir / JOB_RESULT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    peak = data.get("peak_memory_bytes") if isinstance(data, dict) else None
    return peak if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0 else None


def _stop(entry: _Running, *, grace: float, first: signal.Signals = signal.SIGTERM) -> None:
    """Kill a child's process group, reap the child and close its log."""
    process = entry.process
    kill_group(process.pid, grace=grace, first=first, wait=lambda: process.poll() is not None)
    process.wait()
    entry.log.close()


def cancel_manifest(run_dir: Path) -> None:
    """Mark a cancelled attempt's manifest ``cancelled`` (if it has one and has not ended)."""
    try:
        manifest = read_manifest(run_dir)
    except ManifestError:
        return
    if manifest.status in TERMINAL_STATUSES:
        return
    try:
        write_manifest(run_dir, transition(manifest, "cancelled"))
    except (ManifestError, OSError) as err:  # pragma: no cover - a read-only run dir
        sys.stderr.write(f"astro-simclr: warning: {run_dir}: {err}\n")
