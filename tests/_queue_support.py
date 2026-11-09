"""Helpers of the queue tests: specs, a fake job, git repositories and a fake ssh.

- :data:`FAKE_JOB` (``tests/fake_job.py``) is a stand-in for ``spec run-cell``: it takes
  ``--run-dir DIR --mode MODE`` (so the reaper's command-line check sees the run
  directory, as with the real argv) and keeps a "checkpoint" in ``DIR/last`` so that a
  resumed run can prove it continued.
- :data:`FAKE_SSH` (``tests/fake_ssh.py``) is a stand-in for ``ssh``: it drops the options
  and the host and runs the command through ``sh -c``, as sshd does, so quoting, stdin and
  exit codes are exercised for real; rsync's ``-e`` uses it as well.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from astro_simclr.queue.store import Job, NewJob, QueueStore

TESTS = Path(__file__).resolve().parent
FAKE_JOB = TESTS / "fake_job.py"
FAKE_SSH = TESTS / "fake_ssh.py"
#: The per-SHA CLI in the tests: this interpreter's package.
CLI = (sys.executable, "-m", "astro_simclr.cli")
SHA = "a" * 64  # a well-formed SHA-256 for identities

SPEC = """\
format: astro-simclr-experiment-spec
version: 1
name: {name}
command: train-supervised
preset: carrasco2021
settings:
  batch_size: 4
  max_steps: 2
  val_every_n_steps: 2
  horizon_increment: 0
  checkpoint_every_n_steps: 2
  accelerator: cpu
  deterministic: true
arms:
  baseline: {{}}
  wide: {{lr: 0.01}}
seeds: {seeds}
machine:
  class: cpu-test
  extra: cpu
scratch: {scratch}
"""


def spec_text(name: str = "demo", *, scratch: bool = False, seeds: str = "[0, 1]") -> str:
    """A two-arm short spec of the paper classifier on the plain engine (the default since
    task 4.8; ``carrasco2021``, 2 steps): the synthetic float32 data
    (``classifier_f32_dir``), two seeds by default."""
    return SPEC.format(name=name, scratch="true" if scratch else "false", seeds=seeds)


def git(root: Path, *args: str) -> str:
    """Run git in ``root``; fail the test on an error; return stdout."""
    argv = ["git", "-C", str(root), *args]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def init_repo(root: Path, files: dict[str, str] | None = None) -> str:
    """A git repository at ``root`` with one commit of ``files``; return its SHA."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "--quiet", "--initial-branch=main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "config", "commit.gpgsign", "false")
    for name, text in (files or {"README": "x\n"}).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "init")
    return git(root, "rev-parse", "HEAD")


def add_fake_job(
    store: QueueStore,
    run_dir: Path,
    mode: str,
    *,
    run_id: str | None = None,
    config_sha256: str = SHA,
    machine_class: str = "cpu-test",
    spec_id: str = "spec",
    argv: list[str] | None = None,
) -> Job:
    """Queue one fake job (``fake_job.py --run-dir run_dir --mode mode``), cwd its parent."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    store.add_spec(spec_id, "demo", run_dir.parent / "spec.yaml", scratch=False, now=0.0)
    command = argv or [sys.executable, str(FAKE_JOB), "--run-dir", str(run_dir), "--mode", mode]

    def command_for(_attempt: int) -> tuple[Path, list[str]]:
        return run_dir, command

    job, added = store.add_job(
        NewJob(
            spec_id=spec_id,
            cell=f"{run_dir.name}/s0/all",
            run_id=run_id or (str(abs(hash(run_dir))) * 8)[:64].ljust(64, "0"),
            config_sha256=config_sha256,
            git_sha="b" * 40,
            machine_class=machine_class,
            cwd=run_dir.parent,
        ),
        command_for,
        now=0.0,
    )
    assert added
    return job


def wait_for(condition: Callable[[], bool], timeout: float = 30.0) -> None:
    """Poll ``condition`` every 20 ms until it holds; fail after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)
