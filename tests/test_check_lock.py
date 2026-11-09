"""Tests of scripts/check-lock.sh: the machine-wide check lock, its FIFO queue and the pytest
worker cap.

scripts/check.sh sources the library; these tests source it (or run its ``run`` command) in
a fresh bash with the lock directory, hence the waiter directory next to it, moved under
``tmp_path``, so they never touch the real lock (which the running check may hold) and stay
parallel-safe. The environment is scrubbed of the variables the
running check exports (ASTRO_SIMCLR_CHECK_LOCK_HELD, PYTEST_XDIST_AUTO_NUM_WORKERS).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

LIBRARY = Path(__file__).resolve().parents[1] / "scripts" / "check-lock.sh"

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _env(lock: Path, **extra: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ASTRO_SIMCLR_CHECK_LOCK")
        and key not in {"GITHUB_ACTIONS", "PYTEST_XDIST_AUTO_NUM_WORKERS"}
    }
    # No release grace unless a test asks for one (the grace tests below).
    env.update(
        ASTRO_SIMCLR_CHECK_LOCK_DIR=str(lock),
        ASTRO_SIMCLR_CHECK_LOCK_POLL="1",
        ASTRO_SIMCLR_CHECK_LOCK_GRACE="0",
    )
    env.update(extra)
    return env


def _bash(script: str, lock: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 (bash on the test's own scripts)
        [str(BASH), "-c", f'source "{LIBRARY}"\n{script}'],
        env=_env(lock, **extra),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


# A process started with SIGINT ignored (`pytest &` from a non-interactive shell) passes
# the ignore on, and bash cannot trap a signal ignored at its start.
INT_IGNORED = pytest.mark.skipif(
    signal.getsignal(signal.SIGINT) == signal.SIG_IGN, reason="SIGINT ignored by the parent"
)


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_acquire_writes_the_owner_and_release_removes_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "locks" / "check.lock.d"  # the parent is created too
    out = _bash(
        'check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED held=$ASTRO_SIMCLR_CHECK_LOCK_HELD'
        ' self=$$"; cat "$ASTRO_SIMCLR_CHECK_LOCK_DIR/owner"; check_lock_release;'
        " check_lock_release",  # idempotent
        lock,
    )
    assert out.returncode == 0, out.stderr
    first, *owner = out.stdout.splitlines()
    fields = dict(part.split("=", 1) for part in first.split())
    assert fields["owned"] == "1" and fields["held"] == fields["self"]
    assert f"pid={fields['self']}" in owner and f"host={socket.gethostname()}" in owner
    assert not lock.exists()


@pytest.mark.parametrize("extra", [{"ASTRO_SIMCLR_CHECK_LOCK": "0"}, {"GITHUB_ACTIONS": "true"}])
def test_the_lock_is_skipped_when_disabled_or_on_github_actions(
    tmp_path: Path, extra: dict[str, str]
) -> None:
    lock = tmp_path / "check.lock.d"
    lock.mkdir()  # held by someone else: a disabled lock must not wait for it
    out = _bash('check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"', lock, **extra)
    assert out.returncode == 0 and out.stdout.strip() == "owned=0"
    assert lock.is_dir()


@pytest.mark.parametrize(
    "owner",
    [
        "pid={pid}\nhost={host}\n",
        "pid={pid}\n",  # no host: this host
        "astro-simclr integrator verify-merge pid {pid} Sun  8 Nov 2026 12:26:59 -03\n",
    ],
)
def test_a_lock_of_a_dead_pid_is_broken(tmp_path: Path, owner: str) -> None:
    lock = tmp_path / "check.lock.d"
    lock.mkdir()
    (lock / "owner").write_text(owner.format(pid=_dead_pid(), host=socket.gethostname()))
    out = _bash('check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"', lock)
    assert out.returncode == 0 and out.stdout.strip() == "owned=1"
    assert "broke a stale lock" in out.stderr
    assert not list(tmp_path.glob("*.stale.*"))  # the renamed stale lock is removed


@pytest.mark.parametrize(
    "owner",
    [
        None,  # between the holder's mkdir and its write, or a by-hand lock
        "started by hand\n",  # no pid
        "pid={dead}\nhost=another-machine\n",  # the pid is not ours to probe
        "pid={alive}\n",
    ],
)
def test_other_locks_are_not_stale(tmp_path: Path, owner: str | None) -> None:
    lock = tmp_path / "check.lock.d"
    lock.mkdir()
    if owner is not None:
        (lock / "owner").write_text(owner.format(dead=_dead_pid(), alive=os.getpid()))
    out = _bash('check_lock_is_stale "$ASTRO_SIMCLR_CHECK_LOCK_DIR" && echo stale', lock)
    assert out.stdout == ""


def test_a_waiter_takes_the_lock_once_the_holder_dies(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    lock.mkdir()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (lock / "owner").write_text(f"pid={holder.pid}\n")
    waiter = subprocess.Popen(  # noqa: S603
        [
            str(BASH),
            "-c",
            f'source "{LIBRARY}"; check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"',
        ],
        env=_env(lock),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(1.5)
        assert waiter.poll() is None  # still waiting for the live holder
        holder.kill()
        holder.wait()
        stdout, stderr = waiter.communicate(timeout=30)
    finally:
        holder.kill()
        waiter.kill()
    assert stdout.strip() == "owned=1"
    assert f"waiting for {lock}, place 1 of 1 (pid={holder.pid}" in stderr


def test_a_check_inside_the_holder_does_not_wait(tmp_path: Path) -> None:
    # The holder (verify-merge.sh, or check.sh itself) writes its pid and runs check.sh.
    lock = tmp_path / "check.lock.d"
    nested = f'bash -c \'source "{LIBRARY}"; check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"\''
    out = _bash(
        f'mkdir "$ASTRO_SIMCLR_CHECK_LOCK_DIR"; echo "verify-merge pid $$" '
        f'> "$ASTRO_SIMCLR_CHECK_LOCK_DIR/owner"; {nested}; '
        f"check_lock_acquire; {nested}",  # by the inherited ASTRO_SIMCLR_CHECK_LOCK_HELD
        lock,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["owned=0", "owned=0"]
    assert "held by ancestor" in out.stderr and lock.is_dir()


def test_release_leaves_a_lock_retaken_by_another_check(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    out = _bash(
        'check_lock_acquire; echo "pid=1" > "$ASTRO_SIMCLR_CHECK_LOCK_DIR/owner";'
        " check_lock_release",
        lock,
    )
    assert out.returncode == 0 and (lock / "owner").read_text() == "pid=1\n"


@pytest.mark.parametrize(
    ("ending", "code"),
    [
        ("exit 3", 3),
        ("false", 1),
        ("kill -TERM $$", 143),
        pytest.param("kill -INT $$", 130, marks=INT_IGNORED),
    ],
)
def test_the_traps_release_the_lock_on_every_exit(tmp_path: Path, ending: str, code: int) -> None:
    lock = tmp_path / "check.lock.d"
    out = _bash(f"set -e; check_lock_install_traps; check_lock_acquire; {ending}; exit 0", lock)
    assert out.returncode == code
    assert not lock.exists()


@pytest.mark.parametrize(
    ("system", "cores", "override", "workers"),
    [
        ("Darwin", 10, "", 4),  # min(4, cores / 2)
        ("Darwin", 6, "", 3),
        ("Darwin", 1, "", 1),  # never 0
        ("Linux", 3, "", 3),  # min(4, cores)
        ("Linux", 64, "", 4),
        ("Darwin", 10, "8", 8),  # PYTEST_XDIST_AUTO_NUM_WORKERS wins
    ],
)
def test_pytest_workers_are_capped(
    tmp_path: Path, system: str, cores: int, override: str, workers: int
) -> None:
    stubs = (
        f"uname() {{ echo {system}; }}; sysctl() {{ echo {cores}; }}; nproc() {{ echo {cores}; }}"
    )
    extra = {"PYTEST_XDIST_AUTO_NUM_WORKERS": override} if override else {}
    out = _bash(f"{stubs}; check_pytest_workers", tmp_path / "lock.d", **extra)
    assert out.stdout.strip() == str(workers)


def test_pytest_workers_on_this_machine_are_at_most_four(tmp_path: Path) -> None:
    out = _bash("check_pytest_workers", tmp_path / "lock.d")
    assert 1 <= int(out.stdout) <= 4


# The FIFO queue (module header of check-lock.sh): waiter files, oldest live waiter first.

TICKET = re.compile(r"^\d{13}\.\d{6}\.\d{10}$")


def _spawn(script: str, lock: Path, **extra: str) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603 (bash on the test's own scripts)
        [str(BASH), "-c", f'source "{LIBRARY}"\n{script}'],
        env=_env(lock, **extra),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _waiters(lock: Path, *, hidden: bool = False) -> list[str]:
    """The registered waiter names; with ``hidden``, also the dot files being written."""
    directory = lock.parent / "check.waiters"
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if hidden or not p.name.startswith("."))


def _until(condition: Callable[[], bool], timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.05)


def _until_waiters(lock: Path, count: int) -> None:
    _until(lambda: len(_waiters(lock)) == count)


def _held_by_a_live_process(lock: Path) -> subprocess.Popen[bytes]:
    lock.mkdir()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (lock / "owner").write_text(f"pid={holder.pid}\nhost={socket.gethostname()}\n")
    return holder


def test_the_ticket_has_a_fixed_width_and_sorts_by_time_then_pid(tmp_path: Path) -> None:
    out = _bash("check_lock_ticket 7; check_lock_ticket 123456; check_lock_ticket 5", tmp_path)
    tickets = out.stdout.split()
    assert len(tickets) == 3 and all(TICKET.match(ticket) for ticket in tickets), tickets
    assert tickets[0].endswith(".0000000007") and tickets[1].endswith(".0000123456")
    assert sorted(tickets) == tickets  # later tickets sort after, whatever the pid


@pytest.mark.parametrize("count", [2, 3])
def test_waiters_take_the_lock_in_arrival_order(tmp_path: Path, count: int) -> None:
    lock = tmp_path / "check.lock.d"
    order = tmp_path / "order"
    holder = _held_by_a_live_process(lock)
    waiters = []
    try:
        for index in range(count):
            # Each one registers before the next starts: arrival order is the index.
            waiters.append(
                _spawn(
                    f"check_lock_install_traps; check_lock_acquire; echo {index} >> '{order}';"
                    " sleep 1.2; check_lock_release",
                    lock,
                )
            )
            _until_waiters(lock, index + 1)
        shutil.rmtree(lock)  # the holder releases: all of them race for the lock
        for waiter in waiters:
            assert waiter.wait(timeout=60) == 0, waiter.stderr.read() if waiter.stderr else ""
    finally:
        holder.kill()
        for waiter in waiters:
            waiter.kill()
    assert order.read_text().split() == [str(index) for index in range(count)]
    assert _waiters(lock, hidden=True) == [] and not lock.exists()


def test_a_younger_waiter_does_not_take_a_free_lock(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    queue = tmp_path / "check.waiters"
    queue.mkdir()
    # An older live waiter (this pytest process; no pstart recorded) that has not yet
    # seen the lock free: the younger one must leave the lock to it.
    (queue / f"0000000000001.000000.0000000001-mc-sddp-{os.getpid()}").write_text("")
    out = _bash(
        'check_lock_register astro-simclr test; check_lock_is_first || echo "behind";'
        ' echo "place=$CHECK_LOCK_POSITION of $CHECK_LOCK_QUEUE"; check_lock_unregister',
        lock,
    )
    assert out.stdout.split() == ["behind", "place=2", "of", "2"]
    assert not lock.exists() and len(_waiters(lock)) == 1


@pytest.mark.parametrize("reused", [False, True])
def test_a_dead_waiter_is_skipped_and_removed(tmp_path: Path, reused: bool) -> None:
    lock = tmp_path / "check.lock.d"
    queue = tmp_path / "check.waiters"
    queue.mkdir()
    if reused:  # a live pid that started at another time: the pid was reused
        pid, content = os.getpid(), "pid={pid}\npstart=Thu Jan  1 00:00:00 1970\n"
    else:
        pid, content = _dead_pid(), "pid={pid}\n"
    dead = queue / f"0000000000001.000000.{pid:010d}-mc-sddp-{pid}"
    dead.write_text(content.format(pid=pid))
    out = _bash('check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"', lock)
    assert out.returncode == 0 and out.stdout.strip() == "owned=1", out.stderr
    assert "removed the file of a dead waiter" in out.stderr
    assert not dead.exists()


def test_a_waiter_of_another_host_is_not_probed(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    queue = tmp_path / "check.waiters"
    queue.mkdir()
    pid = _dead_pid()
    (queue / f"0000000000001.000000.{pid:010d}-mc-sddp-{pid}").write_text(
        f"pid={pid}\nhost=another-machine\n"
    )
    out = _bash("check_lock_register astro-simclr test; check_lock_is_first || echo behind", lock)
    assert out.stdout.strip() == "behind"


def test_the_waiter_file_is_removed_once_the_lock_is_taken(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    out = _bash(
        'check_lock_acquire; ls "$ASTRO_SIMCLR_CHECK_LOCK_WAITERS"; echo "owned=$CHECK_LOCK_OWNED"',
        lock,
    )
    assert out.returncode == 0 and out.stdout.split() == ["owned=1"], out.stderr
    assert lock.is_dir()


@pytest.mark.parametrize(
    ("sig", "code"), [(signal.SIGTERM, 143), pytest.param(signal.SIGINT, 130, marks=INT_IGNORED)]
)
def test_a_killed_waiter_removes_its_file(tmp_path: Path, sig: signal.Signals, code: int) -> None:
    lock = tmp_path / "check.lock.d"
    holder = _held_by_a_live_process(lock)
    waiter = _spawn("check_lock_install_traps; check_lock_acquire; echo acquired", lock)
    try:
        _until_waiters(lock, 1)
        name = _waiters(lock)[0]
        assert re.match(rf"^\d{{13}}\.\d{{6}}\.{waiter.pid:010d}-astro-simclr-{waiter.pid}$", name)
        waiter.send_signal(sig)
        assert waiter.wait(timeout=20) == code
    finally:
        holder.kill()
        waiter.kill()
    assert _waiters(lock, hidden=True) == []
    assert (lock / "owner").read_text().startswith(f"pid={holder.pid}\n")  # not ours to remove


def test_a_foreign_plain_mkdir_holder_is_respected(tmp_path: Path) -> None:
    # MC-SDDP's plain mkdir loop (or a by-hand lock): no owner file, no waiter file.
    lock = tmp_path / "check.lock.d"
    lock.mkdir()
    waiter = _spawn('check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"', lock)
    try:
        time.sleep(2.5)
        assert waiter.poll() is None and lock.is_dir()  # still waiting, lock untouched
        lock.rmdir()
        stdout, stderr = waiter.communicate(timeout=30)
    finally:
        waiter.kill()
    assert stdout.strip() == "owned=1"
    assert "no owner file" in stderr


def test_a_check_inside_the_holder_registers_no_waiter(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    nested = f'bash -c \'source "{LIBRARY}"; check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"\''
    out = _bash(
        f'check_lock_acquire; {nested}; ls "$ASTRO_SIMCLR_CHECK_LOCK_WAITERS"; '
        f'unset ASTRO_SIMCLR_CHECK_LOCK_HELD; {nested}; ls "$ASTRO_SIMCLR_CHECK_LOCK_WAITERS"',
        lock,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["owned=0", "owned=0"]  # by HELD, then by the owner pid


def _run(lock: Path, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 (bash on the test's own scripts)
        [str(BASH), str(LIBRARY), *args],
        env=_env(lock, **extra),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("code", [0, 1, 7])
def test_run_returns_the_command_exit_code_and_releases(tmp_path: Path, code: int) -> None:
    lock = tmp_path / "check.lock.d"
    out = _run(
        lock,
        "run",
        "--project",
        "mc-sddp",
        "--",
        "bash",
        "-c",
        'cat "$ASTRO_SIMCLR_CHECK_LOCK_DIR/owner"; exit "$0"',
        str(code),
    )
    assert out.returncode == code, out.stderr
    owner = dict(line.split("=", 1) for line in out.stdout.splitlines())
    assert owner["project"] == "mc-sddp" and owner["host"] == socket.gethostname()
    assert owner["command"].startswith("bash -c ") and set(owner) >= {"pid", "started"}
    assert not lock.exists() and _waiters(lock, hidden=True) == []


def test_run_inside_run_does_not_wait_for_itself(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    out = _run(lock, "run", "--project=outer", "--", str(BASH), str(LIBRARY), "run",
               "--project", "inner", "--", "true")  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert not lock.exists()


def test_run_is_disabled_with_the_lock(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    lock.mkdir()  # held: a disabled lock runs the command at once
    out = _run(lock, "run", "--project", "x", "--", "true", ASTRO_SIMCLR_CHECK_LOCK="0")
    assert out.returncode == 0 and lock.is_dir() and _waiters(lock, hidden=True) == []


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["check"],
        ["run", "--", "true"],  # no project
        ["run", "--project", "a/b", "--", "true"],  # not a file-name-safe project
        ["run", "--project", "x", "--"],  # no command
        ["run", "--project"],
        ["run", "--bogus"],
    ],
)
def test_run_usage_errors_exit_2(tmp_path: Path, args: list[str]) -> None:
    out = _run(tmp_path / "check.lock.d", *args)
    assert out.returncode == 2 and "usage:" in out.stderr
    assert _waiters(tmp_path / "check.lock.d", hidden=True) == []


@pytest.mark.parametrize("args", [["--help"], ["run", "--help"]])
def test_run_help(tmp_path: Path, args: list[str]) -> None:
    out = _run(tmp_path / "check.lock.d", *args)
    assert out.returncode == 0 and "usage:" in out.stderr


def test_stray_files_are_ignored_and_dead_partial_files_removed(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    queue = tmp_path / "check.waiters"
    queue.mkdir()
    stray = queue / "0-not-a-ticket"  # sorts first, but must not head the queue
    stray.write_text("")
    partial = queue / f".tmp.{_dead_pid()}"  # a waiter killed with SIGKILL mid-register
    partial.write_text("pid=")
    out = _bash('check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"', lock)
    assert out.returncode == 0 and out.stdout.strip() == "owned=1", out.stderr
    assert stray.exists() and not partial.exists()


def test_a_waiter_is_alive_whatever_the_reader_time_zone(tmp_path: Path) -> None:
    out = _bash(
        "export TZ=Asia/Tokyo; check_lock_register astro-simclr test;"
        ' export TZ=America/Santiago; check_lock_waiter_alive "$CHECK_LOCK_WAITER" && echo alive',
        tmp_path / "check.lock.d",
    )
    assert out.stdout.strip() == "alive", out.stderr


def test_a_lock_taken_without_its_owner_file_is_released(tmp_path: Path) -> None:
    # A signal (or a failed write) between the mkdir and the owner file: nobody would break
    # a lock without an owner file, so the holder's trap must remove it.
    lock = tmp_path / "check.lock.d"
    out = _bash(
        'mkdir "$ASTRO_SIMCLR_CHECK_LOCK_DIR"; CHECK_LOCK_OWNED=1; CHECK_LOCK_OWNER_WRITTEN=0;'
        " check_lock_release",
        lock,
    )
    assert out.returncode == 0 and not lock.exists()


def test_break_leaves_a_lock_retaken_after_it_was_judged_stale(tmp_path: Path) -> None:
    # Two waiters judged the same lock stale; the first broke it and took a fresh one, so
    # the second must not remove the fresh lock.
    lock = tmp_path / "check.lock.d"
    lock.mkdir()
    (lock / "owner").write_text(f"pid={os.getpid()}\n")
    out = _bash(f'check_lock_break "$ASTRO_SIMCLR_CHECK_LOCK_DIR" "pid={_dead_pid()}"', lock)
    assert out.returncode == 0 and (lock / "owner").read_text() == f"pid={os.getpid()}\n"
    assert not list(tmp_path.glob("*.stale.*"))


# The release grace (module header): a freed lock is left to pollers outside the queue.


@pytest.mark.parametrize(("grace", "order"), [("0", "A B F"), ("5", "A F B")])
def test_the_grace_lets_a_foreign_poller_in_between_two_queued_checks(
    tmp_path: Path, grace: str, order: str
) -> None:
    # MC-SDDP's plain mkdir loop, scaled down: one attempt every 4 s (30 s in real life,
    # against a 35 s grace), no waiter file. Without the grace, check B retakes the lock
    # that check A frees long before the foreign poller tries again.
    lock = tmp_path / "check.lock.d"
    log = tmp_path / "order"
    (tmp_path / "check.released").write_text("1\n")  # released long ago: A takes it at once
    holds = "check_lock_install_traps; check_lock_acquire; echo {name} >> '{log}'; sleep 2;"
    holds += " check_lock_release"
    first = _spawn(holds.format(name="A", log=log), lock, ASTRO_SIMCLR_CHECK_LOCK_GRACE=grace)
    others: list[subprocess.Popen[str] | subprocess.Popen[bytes]] = []
    try:
        _until(lambda: log.exists())
        second = _spawn(holds.format(name="B", log=log), lock, ASTRO_SIMCLR_CHECK_LOCK_GRACE=grace)
        others.append(second)
        _until_waiters(lock, 1)
        foreign = subprocess.Popen(  # noqa: S603
            [
                str(BASH),
                "-c",
                f"until mkdir '{lock}' 2>/dev/null; do sleep 4; done;"
                f" echo F >> '{log}'; sleep 0.5; rmdir '{lock}'",
            ]
        )
        others.append(foreign)
        for process in (first, second, foreign):
            assert process.wait(timeout=60) == 0
    finally:
        first.kill()
        for process in others:
            process.kill()
    assert log.read_text().split() == order.split()
    assert int((tmp_path / "check.released").read_text()) > 1  # B's release stamped it


def test_a_free_lock_with_no_known_release_gets_the_full_grace(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    start = time.monotonic()
    out = _bash(
        'check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"',
        lock,
        ASTRO_SIMCLR_CHECK_LOCK_GRACE="2",
    )
    assert out.returncode == 0 and out.stdout.strip() == "owned=1", out.stderr
    assert time.monotonic() - start >= 2
    assert "to pollers outside the queue" in out.stderr


def test_a_lock_free_for_longer_than_the_grace_is_taken_at_once(tmp_path: Path) -> None:
    lock = tmp_path / "check.lock.d"
    (tmp_path / "check.released").write_text(f"{int(time.time()) - 100}\n")
    out = _bash(
        'check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"; check_lock_release;'
        ' cat "$(check_lock_stamp_file)"',
        lock,
        ASTRO_SIMCLR_CHECK_LOCK_GRACE="30",
    )
    assert out.returncode == 0, out.stderr
    owned, stamp = out.stdout.split()
    assert owned == "owned=1" and "outside the queue" not in out.stderr
    assert abs(int(stamp) - time.time()) < 10  # the release refreshed the stamp


def test_breaking_a_stale_lock_starts_a_grace(tmp_path: Path) -> None:
    # Pollers outside the queue were waiting for the stale lock too.
    lock = tmp_path / "check.lock.d"
    lock.mkdir()
    (lock / "owner").write_text(f"pid={_dead_pid()}\n")
    (tmp_path / "check.released").write_text("1\n")
    out = _bash(
        'check_lock_acquire; echo "owned=$CHECK_LOCK_OWNED"',
        lock,
        ASTRO_SIMCLR_CHECK_LOCK_GRACE="2",
    )
    assert out.stdout.strip() == "owned=1", out.stderr
    assert "broke a stale lock" in out.stderr and "outside the queue" in out.stderr
