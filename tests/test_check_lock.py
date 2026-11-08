"""Tests of scripts/check-lock.sh: the machine-wide check lock and the pytest worker cap.

scripts/check.sh sources the library; these tests source it in a fresh bash with the lock
directory moved under ``tmp_path``, so they never touch the real lock (which the running
check may hold) and stay parallel-safe. The environment is scrubbed of the variables the
running check exports (ASTRO_SIMCLR_CHECK_LOCK_HELD, PYTEST_XDIST_AUTO_NUM_WORKERS).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
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
    env.update(ASTRO_SIMCLR_CHECK_LOCK_DIR=str(lock), ASTRO_SIMCLR_CHECK_LOCK_POLL="1")
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
    assert f"waiting for {lock} (pid={holder.pid}" in stderr


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
    ("ending", "code"), [("exit 3", 3), ("false", 1), ("kill -TERM $$", 143), ("kill -INT $$", 130)]
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
