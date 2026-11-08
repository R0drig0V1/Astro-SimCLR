"""Crash-injection tests of the queue worker with real processes on CPU (task A3 acceptance).

The jobs are ``tests/fake_job.py`` processes (``_queue_support``): real children in their
own process groups, killed with real signals. The scenarios of the plan:

- ``kill -9`` of the *worker* (a worker process started with ``queue worker``): its job
  survives as an orphan, its lease expires, a second worker kills the orphan, requeues the
  job, and the job resumes from its "checkpoint";
- ``kill -9`` of the *job* with the worker alive: a counted requeue, then a resume;
- a lease taken over while its worker stalls: the old worker kills its child and writes
  nothing;
- cancel, shutdown, the exit codes, admission by measured peak memory.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from _queue_support import CLI, add_fake_job, wait_for

from astro_simclr.queue.admission import Slot
from astro_simclr.queue.runner import RunnerConfig
from astro_simclr.queue.store import Job, QueueStore
from astro_simclr.queue.worker import (
    Worker,
    cancel_manifest,
    current_boot_id,
    kill_group,
    outcome,
    process_runs,
)
from astro_simclr.runs.manifest import read_manifest

CPU = [Slot("cpu", None, None, 1)]


class Clock:
    """A clock the test moves forward."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QueueStore]:
    opened = QueueStore.open(tmp_path / "queue.sqlite", mounts=[])
    yield opened
    opened.close()


def _worker(store: QueueStore, tmp_path: Path, clock: Clock, **kwargs: object) -> Worker:
    slots = kwargs.pop("slots", CPU)
    options: dict[str, object] = {
        "worker_id": "w1",
        "host": "host-a",
        "boot_id": "boot-1",
        "lease_seconds": 10.0,
        "grace_seconds": 2.0,
        "clock": clock,
        **kwargs,
    }
    return Worker(store, "cpu-test", slots, tmp_path / "logs", **options)  # type: ignore[arg-type]


def _job(store: QueueStore, job: Job) -> Job:
    found = store.job(job.job_id)
    assert found is not None
    return found


def _settle(worker: Worker, store: QueueStore, job: Job, status: str) -> Job:
    def done() -> bool:
        worker.step()
        return _job(store, job).status == status

    wait_for(done)
    return _job(store, job)


def test_outcome_codes() -> None:
    assert outcome(0) == ("finished", "exit 0")
    assert outcome(75)[0] == "retry"
    assert outcome(-9) == ("requeue", "killed by signal 9")
    assert outcome(143)[0] == "requeue"
    assert outcome(137)[0] == "requeue"
    assert outcome(1) == ("failed", "exit 1")
    assert outcome(3)[0] == "failed"


def test_a_job_runs_and_its_peak_memory_is_recorded(store: QueueStore, tmp_path: Path) -> None:
    clock = Clock()
    job = add_fake_job(store, tmp_path / "r1", "ok")
    worker = _worker(store, tmp_path, clock)
    worker.step()
    assert [j.job_id for j in worker.running_jobs] == [job.job_id]
    done = _settle(worker, store, job, "finished")
    assert done.exit_code == 0
    assert store.peak_memory(job.config_sha256, "cpu-test") == 2**20
    assert (tmp_path / "logs" / f"{job.job_id}.log").exists()


def test_failure_is_not_retried(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "fail")
    worker = _worker(store, tmp_path, Clock())
    worker.step()
    failed = _settle(worker, store, job, "failed")
    assert failed.exit_code == 3 and failed.error == "exit 3" and failed.resumes == 0


def test_kill_9_of_the_job_requeues_and_resumes(store: QueueStore, tmp_path: Path) -> None:
    run_dir = tmp_path / "r1"
    job = add_fake_job(store, run_dir, "resume")
    worker = _worker(store, tmp_path, Clock())
    worker.step()
    wait_for(lambda: (run_dir / "last").exists() and int((run_dir / "last").read_text()) > 3)
    pid = _job(store, job).pid
    assert pid is not None
    os.kill(pid, signal.SIGKILL)

    def requeued() -> bool:  # the same step requeues and claims it again
        worker.step()
        return _job(store, job).resumes == 1

    wait_for(requeued)
    assert _job(store, job).exit_code == -9
    done = _settle(worker, store, job, "finished")
    assert done.resumes == 1
    assert int((run_dir / "resumed_from").read_text()) > 3  # it continued, not restarted
    assert (run_dir / "starts").read_text().splitlines() == ["resume", "resume"]


def _runner_home(tmp_path: Path) -> RunnerConfig:
    config = RunnerConfig(
        home=tmp_path / "home",
        repo=tmp_path,
        data_dir=tmp_path,
        runs_root=tmp_path / "runs",
        entry_point=CLI,
    )
    config.save()
    return config


def test_kill_9_of_the_worker_lease_expiry_then_resume(tmp_path: Path) -> None:
    """The plan's acceptance: kill -9, then lease expiry, then resume (another worker)."""
    config = _runner_home(tmp_path)
    store = QueueStore.open(config.queue_file, mounts=[])
    run_dir = config.runs_root / "r1"
    job = add_fake_job(store, run_dir, "resume")
    env = {**os.environ, "ASTRO_SIMCLR_HOME": str(config.home)}
    argv = [*CLI, "queue", "worker", "--machine-class", "cpu-test", "--lease-seconds", "3"]
    worker = subprocess.Popen(  # noqa: S603 - this interpreter's CLI
        [*argv, "--poll-seconds", "0.1"],
        env=env,
        start_new_session=True,
    )
    try:
        wait_for(lambda: (run_dir / "last").exists() and _job(store, job).pid is not None)
        running = _job(store, job)
        assert running.status == "running" and running.lease_owner is not None
        assert str(worker.pid) in running.lease_owner
        os.kill(worker.pid, signal.SIGKILL)  # the worker dies; its job is now an orphan
        worker.wait()
        orphan = running.pid
        assert orphan is not None and process_runs(orphan, run_dir)
        # The lease has not expired yet: a second worker leaves the job alone.
        clock = Clock()
        clock.now = running.lease_expires - 1  # type: ignore[operator]
        second = Worker(
            store,
            "cpu-test",
            CPU,
            config.log_dir,
            worker_id="w2",
            host=running.host or "",
            boot_id=running.boot_id,
            clock=clock,
            lease_seconds=10.0,
            grace_seconds=2.0,
        )
        second.step()
        assert _job(store, job).status == "running" and second.running_jobs == []
        # Past the lease: the orphan is killed, the job requeued and resumed by w2.
        clock.now = running.lease_expires + 1  # type: ignore[operator]
        second.step()
        wait_for(lambda: not process_runs(orphan, run_dir))
        assert [j.job_id for j in second.running_jobs] == [job.job_id]
        done = _settle(second, store, job, "finished")
        assert done.resumes == 1 and done.lease_owner == "w2"
        assert int((run_dir / "resumed_from").read_text()) > 0
    finally:
        if worker.poll() is None:  # pragma: no cover - only if an assertion failed early
            os.killpg(worker.pid, signal.SIGKILL)
        store.close()


def test_an_orphan_on_another_boot_is_requeued_without_a_kill(
    store: QueueStore, tmp_path: Path
) -> None:
    job = add_fake_job(store, tmp_path / "r1", "ok")
    clock = Clock()
    dead = _worker(store, tmp_path, clock, worker_id="dead", boot_id="boot-0")
    claimed = store.claim(
        owner="dead",
        machine_class="cpu-test",
        host="host-a",
        boot_id="boot-0",
        admit=lambda _job: "cpu",
        now=clock.now,
        lease_seconds=10,
    )
    assert claimed is not None and claimed.lease_token is not None
    store.set_pid(job.job_id, claimed.lease_token, os.getpid())  # a recycled pid: ourselves
    clock.now += 60
    alive = _worker(store, tmp_path, clock, worker_id="alive", boot_id="boot-1")
    alive.step()  # would kill the test process if it trusted the pid
    assert [j.job_id for j in alive.running_jobs] == [job.job_id]
    assert _settle(alive, store, job, "finished").resumes == 1
    assert dead.running_jobs == []


def test_a_recycled_pid_on_this_boot_is_not_killed(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "ok")
    clock = Clock()
    claimed = store.claim(
        owner="dead",
        machine_class="cpu-test",
        host="host-a",
        boot_id="boot-1",
        admit=lambda _job: "cpu",
        now=clock.now,
        lease_seconds=10,
    )
    assert claimed is not None and claimed.lease_token is not None
    store.set_pid(job.job_id, claimed.lease_token, os.getpid())
    assert not process_runs(os.getpid(), tmp_path / "r1")
    clock.now += 60
    alive = _worker(store, tmp_path, clock, worker_id="alive")
    alive.step()  # same host and boot, but the pid does not run r1: not killed
    assert _settle(alive, store, job, "finished").status == "finished"


def test_a_stalled_worker_loses_its_lease_and_writes_nothing(
    store: QueueStore, tmp_path: Path
) -> None:
    run_dir = tmp_path / "r1"
    job = add_fake_job(store, run_dir, "sleep")
    clock_a, clock_b = Clock(), Clock()
    a = _worker(store, tmp_path, clock_a, worker_id="a")
    a.step()
    wait_for(lambda: (run_dir / "last").exists())
    child = _job(store, job).pid
    assert child is not None
    clock_b.now = clock_a.now + 60  # a stalled for a minute
    b = _worker(store, tmp_path, clock_b, worker_id="b")
    b.step()  # b reaps a's lease: kills a's child (same host and boot), requeues, claims
    wait_for(lambda: _job(store, job).lease_owner == "b" and _job(store, job).pid != child)
    a.step()  # a's heartbeat is refused: it forgets the job without writing
    assert a.running_jobs == []
    current = _job(store, job)
    assert current.status == "running" and current.lease_owner == "b" and current.resumes == 1
    b.shutdown()
    assert _job(store, job).status == "queued"


def test_cancel_a_running_job(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "sleep")
    worker = _worker(store, tmp_path, Clock())
    worker.step()
    wait_for(lambda: (tmp_path / "r1" / "last").exists())
    assert store.cancel([job.job_id], now=1.0) == {job.job_id: "cancel requested"}
    cancelled = _settle(worker, store, job, "cancelled")
    assert cancelled.exit_code == 143


def test_cancel_escalates_to_sigkill(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "stubborn")
    worker = _worker(store, tmp_path, Clock(), grace_seconds=0.3)
    worker.step()
    wait_for(lambda: (tmp_path / "r1" / "last").exists())
    store.cancel([job.job_id], now=1.0)
    assert _settle(worker, store, job, "cancelled").exit_code == -9


def test_shutdown_requeues_without_counting(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "sleep")
    worker = _worker(store, tmp_path, Clock())
    worker.step()
    wait_for(lambda: (tmp_path / "r1" / "last").exists())
    worker.shutdown()
    requeued = _job(store, job)
    assert requeued.status == "queued" and requeued.resumes == 0
    assert requeued.error == "worker shut down" and requeued.exit_code == 143
    worker.step()  # a stopped worker claims nothing
    assert worker.running_jobs == []


def test_busy_run_directory_is_retried_after_a_lease(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "busy")
    clock = Clock()
    worker = _worker(store, tmp_path, clock)
    worker.step()
    retried = _settle(worker, store, job, "queued")
    assert retried.resumes == 0 and retried.not_before == clock.now + 10
    worker.step()
    assert worker.running_jobs == []  # not before the lease period


def test_a_job_that_cannot_start_fails(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "ok", argv=[str(tmp_path / "no-such-program")])
    worker = _worker(store, tmp_path, Clock())
    worker.step()
    failed = _job(store, job)
    assert failed.status == "failed" and "cannot start" in (failed.error or "")


def test_admission_by_measured_peak_memory(store: QueueStore, tmp_path: Path) -> None:
    gpu = [Slot("cuda:0", 0, 2**30, max_jobs=2)]
    measured = add_fake_job(store, tmp_path / "r1", "sleep", config_sha256="1" * 64)
    unmeasured = add_fake_job(store, tmp_path / "r2", "sleep", config_sha256="2" * 64)
    worker = _worker(store, tmp_path, Clock(), slots=gpu)
    worker.step()
    # The first job is alone; the second has no measured peak, so it waits.
    assert [j.job_id for j in worker.running_jobs] == [measured.job_id]
    store.record_peak_memory("1" * 64, "cpu-test", 2**20, now=1.0)
    store.record_peak_memory("2" * 64, "cpu-test", 2**20, now=1.0)
    worker.step()
    assert {j.job_id for j in worker.running_jobs} == {measured.job_id, unmeasured.job_id}
    started = _job(store, unmeasured)
    assert started.slot == "cuda:0"
    log = (tmp_path / "logs" / f"{unmeasured.job_id}.log").read_bytes()
    assert log == b""
    worker.shutdown()


def test_worker_environment(store: QueueStore, tmp_path: Path) -> None:
    script = tmp_path / "env.py"
    script.write_text(
        "import os, sys\nprint(os.environ['CUDA_VISIBLE_DEVICES'],"
        " os.environ['ASTRO_SIMCLR_CONCURRENT_JOBS'], sys.stdin.read() == '')\n"
    )
    job = add_fake_job(store, tmp_path / "r1", "ok", argv=[sys.executable, str(script)])
    worker = _worker(store, tmp_path, Clock(), slots=[Slot("cuda:1", 1, None)])
    worker.step()
    _settle(worker, store, job, "finished")
    assert (tmp_path / "logs" / f"{job.job_id}.log").read_text().split() == ["1", "1", "True"]


def test_run_with_idle_exit(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "ok")
    worker = _worker(store, tmp_path, Clock())
    worker.run(poll_seconds=0.02, idle_exit=True)
    assert _job(store, job).status == "finished"


def test_run_stops_on_sigterm(store: QueueStore, tmp_path: Path) -> None:
    job = add_fake_job(store, tmp_path / "r1", "sleep")
    worker = _worker(store, tmp_path, Clock())
    original = worker.step
    calls = {"n": 0}

    def step() -> None:
        original()
        calls["n"] += 1
        if calls["n"] == 3:
            os.kill(os.getpid(), signal.SIGTERM)  # as systemd stopping the service

    worker.step = step  # type: ignore[method-assign]
    worker.run(poll_seconds=0.05)
    assert _job(store, job).status == "queued"


def test_kill_group_of_a_finished_group_is_a_no_op() -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    process.wait()
    kill_group(process.pid, grace=0.1)  # nothing to kill: returns


def test_current_boot_id() -> None:
    boot = current_boot_id()
    if Path("/proc/sys/kernel/random/boot_id").exists():  # pragma: no cover - Linux CI
        assert boot
    else:
        assert boot is None


def test_cancel_manifest_without_a_manifest_is_a_no_op(tmp_path: Path) -> None:
    cancel_manifest(tmp_path)
    assert not list(tmp_path.iterdir())
    _ = read_manifest  # the manifest path is covered by test_queue_job.py
