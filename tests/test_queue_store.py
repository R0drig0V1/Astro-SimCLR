"""Tests of the queue file, its leases and its location checks (``queue.store``, ``queue.location``,
``queue.admission``; task A3)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from _queue_support import add_fake_job

from astro_simclr.queue.admission import Slot, admits, choose_slot
from astro_simclr.queue.location import LocationError, check_queue_location, read_mounts
from astro_simclr.queue.store import Job, NewJob, QueueError, QueueStore, iso

GIB = 2**30


@pytest.fixture
def store(tmp_path: Path) -> Iterator[QueueStore]:
    """An empty queue in ``tmp_path`` (no mount table: the path checks only)."""
    opened = QueueStore.open(tmp_path / "queue" / "queue.sqlite", mounts=[])
    yield opened
    opened.close()


def _claim(
    store: QueueStore, *, owner: str = "w1", now: float = 1.0, slot: str = "cpu"
) -> Job | None:
    return store.claim(
        owner=owner,
        machine_class="cpu-test",
        host="h",
        boot_id="b",
        admit=lambda _job: slot,
        now=now,
        lease_seconds=10.0,
    )


# --- location ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/mnt/c/Users/me/queue.sqlite",
        "/mnt/d",
        "/home/me/Dropbox/queue.sqlite",
        "/Users/me/Library/Mobile Documents/q.sqlite",
        "/home/me/OneDrive/x/q.sqlite",
    ],
)
def test_unsafe_locations_by_path(path: str) -> None:
    with pytest.raises(LocationError):
        check_queue_location(Path(path), mounts=[])


@pytest.mark.parametrize("fstype", ["9p", "drvfs", "nfs4", "cifs", "fuse.sshfs"])
def test_unsafe_locations_by_file_system(tmp_path: Path, fstype: str) -> None:
    mounts = [("/", "ext4"), (str(tmp_path.resolve()), fstype)]
    with pytest.raises(LocationError, match=fstype):
        check_queue_location(tmp_path / "q.sqlite", mounts=mounts)
    with pytest.raises(LocationError):
        QueueStore.open(tmp_path / "q.sqlite", mounts=mounts)


def test_safe_location(tmp_path: Path) -> None:
    mounts = [("/", "ext4"), ("/mnt/c", "9p"), (str(tmp_path.resolve().parent) + "x", "9p")]
    assert (
        check_queue_location(tmp_path / "q.sqlite", mounts=mounts)
        == (tmp_path / "q.sqlite").resolve()
    )
    # The longest matching mount wins: an ext4 mount inside a 9p one is fine.
    nested = [("/", "9p"), (str(tmp_path.resolve()), "ext4")]
    check_queue_location(tmp_path / "q.sqlite", mounts=nested)


def test_read_mounts(tmp_path: Path) -> None:
    table = tmp_path / "mounts"
    table.write_text(
        "/dev/sdc / ext4 rw 0 0\nC:\\134 /mnt/c 9p rw 0 0\nx /mnt/with\\040space 9p rw\nbad\n"
    )
    assert read_mounts(table) == [("/", "ext4"), ("/mnt/c", "9p"), ("/mnt/with space", "9p")]
    assert read_mounts(tmp_path / "missing") == []


# --- schema ------------------------------------------------------------------------------------


def test_reopen_keeps_jobs_and_refuses_newer_or_foreign_files(tmp_path: Path) -> None:
    path = tmp_path / "q.sqlite"
    store = QueueStore.open(path, mounts=[])
    add_fake_job(store, tmp_path / "r1", "ok")
    store.close()
    again = QueueStore.open(path, mounts=[])
    assert [job.status for job in again.jobs()] == ["queued"]
    again.close()
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version=99")
    db.close()
    with pytest.raises(QueueError, match="schema 99"):
        QueueStore.open(path, mounts=[])
    garbage = tmp_path / "garbage.sqlite"
    garbage.write_bytes(b"not a database at all" * 100)
    with pytest.raises(QueueError, match="not a usable"):
        QueueStore.open(garbage, mounts=[])


# --- jobs and attempts -------------------------------------------------------------------------


def test_add_job_attempts(store: QueueStore, tmp_path: Path) -> None:
    store.add_spec("s", "demo", tmp_path / "s.yaml", scratch=False, now=0.0)
    new = NewJob(
        spec_id="s",
        cell="a/s0/all",
        run_id="1" * 64,
        config_sha256="2" * 64,
        git_sha="3" * 40,
        machine_class="cpu-test",
        cwd=tmp_path,
    )

    def command_for(attempt: int) -> tuple[Path, list[str]]:
        return tmp_path / f"a{attempt}", ["true", str(attempt)]

    first, added = store.add_job(new, command_for, now=1.0)
    assert added and first.attempt == 1 and first.argv == ("true", "1")
    again, added = store.add_job(new, command_for, now=2.0)
    assert not added and again.job_id == first.job_id
    rerun, added = store.add_job(new, command_for, now=3.0, new_attempt=True)
    assert added and rerun.attempt == 2 and rerun.run_dir == tmp_path / "a2"
    # A failed attempt does not block a resubmission: it gets the next attempt.
    store.cancel([first.job_id, rerun.job_id], now=4.0)
    retry, added = store.add_job(new, command_for, now=5.0)
    assert added and retry.attempt == 3
    with pytest.raises(QueueError):
        store.add_job(_without_spec(new), command_for, now=6.0)


def _without_spec(new: NewJob) -> NewJob:
    return NewJob(
        spec_id="never-added",
        cell=new.cell,
        run_id="9" * 64,
        config_sha256=new.config_sha256,
        git_sha=new.git_sha,
        machine_class=new.machine_class,
        cwd=new.cwd,
    )


def test_jobs_filters(store: QueueStore, tmp_path: Path) -> None:
    one = add_fake_job(store, tmp_path / "r1", "ok", spec_id="abc_1")
    add_fake_job(store, tmp_path / "r2", "ok", spec_id="abd")
    assert [job.job_id for job in store.jobs(spec_id="abc_")] == [one.job_id]
    assert len(store.jobs(spec_id="ab")) == 2
    assert store.jobs(spec_id="a%") == []
    assert len(store.jobs(status="queued")) == 2
    assert store.job(999) is None
    plain = one.to_plain()
    assert plain["status"] == "queued" and plain["created_at"] == iso(0.0)
    assert iso(None) is None


# --- leases ------------------------------------------------------------------------------------


def test_claim_is_fifo_and_respects_admission(store: QueueStore, tmp_path: Path) -> None:
    first = add_fake_job(store, tmp_path / "r1", "ok")
    second = add_fake_job(store, tmp_path / "r2", "ok")
    other = add_fake_job(store, tmp_path / "r3", "ok", machine_class="cuda-x")
    blocked = store.claim(
        owner="w",
        machine_class="cpu-test",
        host="h",
        boot_id=None,
        admit=lambda _job: None,
        now=1.0,
        lease_seconds=10,
    )
    assert blocked is None  # the head is not admitted: nothing behind it jumps the queue
    claimed = _claim(store)
    assert claimed is not None and claimed.job_id == first.job_id
    assert claimed.status == "running" and claimed.lease_owner == "w1"
    assert claimed.lease_expires == 11.0 and claimed.started_at == 1.0
    nxt = _claim(store)
    assert nxt is not None and nxt.job_id == second.job_id
    assert _claim(store) is None
    assert store.job(other.job_id).status == "queued"  # type: ignore[union-attr]


def test_heartbeat_finish_and_fencing(store: QueueStore, tmp_path: Path) -> None:
    add_fake_job(store, tmp_path / "r1", "ok")
    job = _claim(store)
    assert job is not None and job.lease_token is not None
    token = job.lease_token
    assert store.set_pid(job.job_id, token, 1234)
    assert not store.set_pid(job.job_id, "stale", 1)
    assert store.heartbeat({job.job_id: token}, now=5.0, lease_seconds=10) == set()
    assert store.job(job.job_id).lease_expires == 15.0  # type: ignore[union-attr]
    assert store.expired(now=14.0) == []
    assert [j.job_id for j in store.expired(now=16.0)] == [job.job_id]
    # The reaper requeues it: the old holder's token is now stale.
    assert store.requeue(job.job_id, token, now=16.0, reason="lease expired") == "queued"
    assert store.heartbeat({job.job_id: token}, now=17.0, lease_seconds=10) == {job.job_id}
    assert not store.finish(job.job_id, token, "finished", now=17.0)
    assert store.requeue(job.job_id, token, now=17.0, reason="x") is None
    requeued = store.job(job.job_id)
    assert requeued is not None and requeued.status == "queued" and requeued.resumes == 1
    assert requeued.error == "lease expired" and requeued.pid is None
    again = _claim(store, owner="w2", now=18.0)
    assert again is not None and again.started_at == 1.0  # the first start is kept
    assert again.lease_token is not None
    assert store.finish(again.job_id, again.lease_token, "finished", now=20.0, exit_code=0)
    done = store.job(job.job_id)
    assert done is not None and done.status == "finished" and done.ended_at == 20.0
    assert done.lease_token is None


def test_requeue_gives_up_after_max_resumes(store: QueueStore, tmp_path: Path) -> None:
    add_fake_job(store, tmp_path / "r1", "ok")
    statuses = []
    for step in range(7):
        job = _claim(store, now=float(step))
        if job is None:
            break
        assert job.lease_token is not None
        statuses.append(store.requeue(job.job_id, job.lease_token, now=float(step), reason="kill"))
    assert statuses == ["queued"] * 5 + ["failed"]
    failed = store.jobs()[0]
    assert failed.status == "failed" and "gave up after 5 resumes" in (failed.error or "")


def test_requeue_not_counted_and_not_before(store: QueueStore, tmp_path: Path) -> None:
    add_fake_job(store, tmp_path / "r1", "ok")
    job = _claim(store)
    assert job is not None and job.lease_token is not None
    store.requeue(job.job_id, job.lease_token, now=1.0, reason="busy", not_before=50.0, count=False)
    assert store.jobs()[0].resumes == 0
    assert _claim(store, now=10.0) is None
    assert _claim(store, now=50.0) is not None


def test_cancel(store: QueueStore, tmp_path: Path) -> None:
    queued = add_fake_job(store, tmp_path / "r1", "ok")
    running = add_fake_job(store, tmp_path / "r2", "ok")
    store.cancel([queued.job_id], now=1.0)
    claimed = _claim(store)
    assert claimed is not None and claimed.job_id == running.job_id
    result = store.cancel([queued.job_id, running.job_id, 77], now=2.0)
    assert result == {
        queued.job_id: "cancelled",
        running.job_id: "cancel requested",
        77: "no such job",
    }
    assert store.cancel_requested([running.job_id, queued.job_id]) == {running.job_id}
    assert store.cancel_requested([]) == set()
    assert [job.job_id for job in store.running(owner="w1")] == [running.job_id]
    assert store.running(owner="other") == []
    assert len(store.running()) == 1


def test_peak_memory_keeps_the_largest(store: QueueStore) -> None:
    assert store.peak_memory("c", "m") is None
    store.record_peak_memory("c", "m", 5, now=1.0)
    store.record_peak_memory("c", "m", 3, now=2.0)
    assert store.peak_memory("c", "m") == 5
    store.record_peak_memory("c", "m", 9, now=3.0)
    assert store.peak_memory("c", "m") == 9
    assert store.peak_memory("c", "other") is None


# --- admission ---------------------------------------------------------------------------------


def test_admission_rules() -> None:
    gpu = Slot("cuda:0", 0, 4 * GIB, max_jobs=2)
    assert admits(gpu, None, [])  # alone: always
    assert admits(gpu, 1 * GIB, [1 * GIB])
    assert not admits(gpu, 3 * GIB, [1 * GIB])  # 4 GiB > 90 % of 4 GiB
    assert not admits(gpu, None, [1 * GIB])  # unmeasured: alone only
    assert not admits(gpu, 1 * GIB, [None])
    assert not admits(gpu, 1, [1, 1])  # max_jobs
    unknown = Slot("cuda:1", 1, None, max_jobs=2)
    assert not admits(unknown, 1, [1])
    single = Slot("cuda:2", 2, 4 * GIB)
    assert not admits(single, 1, [1])  # default: one job per GPU
    with pytest.raises(ValueError, match="max_jobs"):
        Slot("x", None, None, max_jobs=0)


def test_choose_slot_prefers_the_least_loaded() -> None:
    slots = [Slot("cuda:0", 0, 4 * GIB, 2), Slot("cuda:1", 1, 4 * GIB, 2)]
    assert choose_slot(slots, GIB, {}).name == "cuda:0"  # type: ignore[union-attr]
    chosen = choose_slot(slots, GIB, {"cuda:0": [GIB]})
    assert chosen is not None and chosen.name == "cuda:1"
    assert choose_slot(slots, None, {"cuda:0": [GIB], "cuda:1": [GIB]}) is None
