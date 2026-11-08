"""End-to-end tests of submission, the runner and sync over a fake ssh (task A3 acceptance).

A real "laptop" clone, a bare ``origin`` and a runner clone, all local git repositories; the
transport is :class:`~astro_simclr.queue.remote.SshTransport` with ``tests/fake_ssh.py`` in
place of ``ssh`` (it runs the command through ``sh -c``, as sshd does), and rsync's ``-e``
uses the same fake. ``uv`` is a fake script that records its argv; the per-SHA CLI is this
interpreter's package. The jobs are real dry ``spec run-cell`` processes on CPU.
"""

from __future__ import annotations

import ast
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from _queue_support import CLI, FAKE_SSH, SHA, git, init_repo, spec_text

from astro_simclr.config import Settings
from astro_simclr.queue.admission import Slot
from astro_simclr.queue.remote import SshTransport, SubmitError, submit, sync
from astro_simclr.queue.runner import RunnerConfig, RunnerError, enqueue, ensure_worktree
from astro_simclr.queue.store import QueueStore
from astro_simclr.queue.worker import Worker
from astro_simclr.runs.identity import GitState, run_identity
from astro_simclr.runs.manifest import new_manifest, read_manifest, transition, write_manifest
from astro_simclr.runs.spec import parse_spec

MACHINE = {"device": "cpu", "gpu": None, "torch": "x", "cuda": None, "fp32_precision": "ieee"}


class Env:
    """The laptop, origin, runner and transport of one test."""

    def __init__(self, tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.origin = tmp_path / "origin.git"
        self.laptop = tmp_path / "laptop"
        self.spec = self.laptop / "specs" / "demo.yaml"
        self.sha = init_repo(
            self.laptop, {"README": "x\n", "specs/demo.yaml": spec_text(seeds="[0]")}
        )
        git(tmp_path, "init", "--quiet", "--bare", str(self.origin))
        git(self.laptop, "remote", "add", "origin", str(self.origin))
        git(self.laptop, "push", "--quiet", "origin", "main")
        git(self.laptop, "fetch", "--quiet", "origin")
        self.runner_repo = tmp_path / "runner-repo"
        git(tmp_path, "clone", "--quiet", str(self.origin), str(self.runner_repo))
        self.uv_log = tmp_path / "uv.log"
        uv = tmp_path / "fake-uv"
        uv.write_text(f'#!/bin/sh\necho "$@" >> {self.uv_log}\n')
        uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
        self.config = RunnerConfig(
            home=tmp_path / "home",
            repo=self.runner_repo,
            data_dir=data_dir,
            runs_root=tmp_path / "home" / "runs",
            uv=str(uv),
            entry_point=CLI,
        )
        self.config.save()
        self.drop = tmp_path / "drop"
        monkeypatch.setenv("FAKE_SSH_ASTRO_HOME", str(self.config.home))
        monkeypatch.setenv("FAKE_SSH_HOME", str(tmp_path))
        monkeypatch.setenv("FAKE_SSH_DROP", str(self.drop))
        self.transport = SshTransport("rvidal@runner", ssh=(sys.executable, str(FAKE_SSH)))

    def submit(self, path: Path | None = None, **kwargs: object) -> dict[str, object]:
        report = submit(
            path or self.spec,
            self.transport,
            repo_root=self.laptop,
            remote_cli=CLI,
            **kwargs,  # type: ignore[arg-type]
        )
        assert isinstance(report, dict)
        return report

    def store(self) -> QueueStore:
        return QueueStore.open(self.config.queue_file, mounts=[])


@pytest.fixture
def env(
    tmp_path: Path, synthetic_prepared: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> Env:
    data_dir = next(iter(synthetic_prepared.values())).parent
    return Env(tmp_path, data_dir, monkeypatch)


def test_the_transport_quotes_every_argument(env: Env) -> None:
    tricky = ["a b", "it's", "$HOME", "; touch pwned", "`id`", '"q"', "*", ""]
    script = "import sys; print(sys.argv[1:])"
    result = env.transport.run([sys.executable, "-c", script, *tricky])
    assert result.returncode == 0, result.stderr
    assert ast.literal_eval(result.stdout.decode().strip()) == tricky
    assert not (env.tmp / "pwned").exists()
    echoed = env.transport.run(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"], stdin=b"spec"
    )
    assert echoed.stdout.decode().strip() == "spec"


def test_hosts_and_remote_paths_are_checked(env: Env) -> None:
    with pytest.raises(SubmitError, match="host name"):
        SshTransport("-oProxyCommand=evil")
    with pytest.raises(SubmitError, match="host name"):
        SshTransport("host; rm")
    with pytest.raises(SubmitError, match="rsync would expand"):
        env.transport.pull("runs/$(id)", env.tmp / "x")


def test_submit_run_and_sync(env: Env) -> None:
    """The whole path: submit over ssh, a per-SHA worktree, the worker, sync, reindex."""
    report = env.submit()
    spec = parse_spec(env.spec.read_text())
    assert report["spec_id"] == spec.spec_id
    worktree = env.config.worktrees / env.sha
    assert report["worktree"] == str(worktree)
    assert git(worktree, "rev-parse", "HEAD") == env.sha
    assert env.uv_log.read_text().split() == ["sync", "--locked", "--extra", "cpu"]
    added = report["added"]
    assert isinstance(added, list) and len(added) == 2
    assert (env.config.specs_dir / f"{spec.spec_id}.yaml").read_bytes() == env.spec.read_bytes()
    store = env.store()
    jobs = store.jobs()
    assert [job.cell for job in jobs] == [cell.key for cell in spec.cells()]
    first = jobs[0]
    assert first.argv[: len(CLI)] == CLI and first.argv[len(CLI) : len(CLI) + 2] == (
        "spec",
        "run-cell",
    )
    assert first.cwd == worktree and first.git_sha == env.sha
    assert first.run_dir.parent == env.config.runs_root / "demo"
    # A resubmission queues nothing new.
    again = env.submit()
    assert again["added"] == [] and len(again["existing"]) == 2  # type: ignore[arg-type]
    worker = Worker(store, "cpu-test", [Slot("cpu", None, None)], env.config.log_dir)
    worker.run(poll_seconds=0.05, idle_exit=True)
    statuses = {job.cell: (job.status, job.exit_code) for job in store.jobs()}
    assert set(statuses.values()) == {("finished", 0)}, statuses
    store.close()
    # A retained pretrain checkpoint, to be synced; a classifier's checkpoints are not.
    pretrain = env.config.runs_root / "other" / "p-a1"
    pretrain.mkdir(parents=True)
    (pretrain / "checkpoints").mkdir()
    (pretrain / "checkpoints" / "best.ckpt").write_bytes(b"best")
    (pretrain / "checkpoints" / "last.ckpt").write_bytes(b"last")
    identity = run_identity(
        "pretrain", Settings(), data_sha256=SHA, machine_class="cpu-test", engine="lightning"
    )
    manifest = new_manifest(
        "pretrain", Settings(), identity, git=GitState(None, None), uv_lock_sha256=None
    )
    running = transition(manifest, "running", machine=MACHINE)
    write_manifest(pretrain, transition(running, "finished"))
    (env.config.runs_root / "stray.sqlite").write_bytes(b"never copied")
    dest, specs = env.tmp / "laptop-runs", env.tmp / "laptop-home" / "specs"
    synced = sync(env.transport, dest, specs_dest=specs, remote_cli=CLI)
    assert synced.runs == 3
    assert synced.checkpoints == ("other/p-a1/checkpoints/best.ckpt",)
    assert synced.reindexed is False  # no `runs reindex` before the registry (A2) lands
    assert (dest / "other" / "p-a1" / "checkpoints" / "best.ckpt").read_bytes() == b"best"
    assert not (dest / "other" / "p-a1" / "checkpoints" / "last.ckpt").exists()
    for run_dir in (dest / "demo").iterdir():
        assert read_manifest(run_dir).status == "finished"
        assert not (run_dir / "checkpoints").exists() and not (run_dir / ".lock").exists()
        assert (run_dir / "metrics.json").exists()
    assert not list(dest.rglob("*.sqlite*"))
    assert (specs / f"{spec.spec_id}.yaml").exists()
    # Idempotent: a second sync (after a dropped one) gives the same tree.
    assert sync(env.transport, dest, specs_dest=specs, remote_cli=CLI, reindex=False).runs == 3


def test_sync_runs_reindex(env: Env) -> None:
    calls: list[list[str]] = []

    def commands(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    env.config.runs_root.mkdir(parents=True)
    env.config.specs_dir.mkdir(parents=True)
    report = sync(
        env.transport,
        env.tmp / "d",
        specs_dest=env.tmp / "s",
        remote_cli=CLI,
        commands=commands,
    )
    assert report.reindexed is True
    assert calls == [
        [sys.executable, "-m", "astro_simclr.cli", "runs", "reindex", str(env.tmp / "d")]
    ]

    def failing(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 1, b"", b"registry locked")

    with pytest.raises(SubmitError, match="registry locked"):
        sync(
            env.transport, env.tmp / "d", specs_dest=env.tmp / "s", remote_cli=CLI, commands=failing
        )


def test_a_dropped_connection(env: Env) -> None:
    env.drop.write_text("")
    with pytest.raises(SubmitError, match="ssh to rvidal@runner failed"):
        env.submit()
    with pytest.raises(SubmitError, match="ssh"):
        sync(env.transport, env.tmp / "d", specs_dest=env.tmp / "s", remote_cli=CLI)
    env.drop.unlink()
    assert len(env.submit()["added"]) == 2  # type: ignore[arg-type]


def test_local_refusals(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    (env.laptop / "README").write_text("uncommitted\n")
    with pytest.raises(SubmitError, match="uncommitted changes"):
        env.submit()
    git(env.laptop, "commit", "--quiet", "-am", "local only")
    with pytest.raises(SubmitError, match="push it first"):
        env.submit()
    outside = env.tmp / "outside.yaml"
    outside.write_text(spec_text())
    with pytest.raises(SubmitError, match="not committed"):
        env.submit(outside)
    with pytest.raises(SubmitError, match="not a git checkout"):
        submit(outside, env.transport, repo_root=env.tmp / "nowhere", remote_cli=CLI)
    # A scratch spec goes with a dirty, unpushed tree, with a warning: the runner fetches
    # nothing new, so the local-only commit is refused there.
    scratch = env.laptop / "specs" / "scratch.yaml"
    scratch.write_text(spec_text("scratch", scratch=True))
    with pytest.raises(SubmitError, match="refused"):
        env.submit(scratch)
    assert "not your uncommitted changes" in capsys.readouterr().err


def test_the_runner_refuses_an_edited_worktree(env: Env) -> None:
    env.submit()
    worktree = env.config.worktrees / env.sha
    (worktree / "README").write_text("hacked on the GPU machine\n")
    other = env.laptop / "specs" / "two.yaml"
    other.write_text(spec_text("two"))
    git(env.laptop, "add", "-A")
    git(env.laptop, "commit", "--quiet", "-m", "spec two")
    git(env.laptop, "push", "--quiet", "origin", "main")
    # A new commit gets its own (clean) worktree, fetched from origin.
    new_sha = git(env.laptop, "rev-parse", "HEAD")
    report = env.submit(other)
    assert report["worktree"] == str(env.config.worktrees / new_sha)
    # The edited one is refused for a non-scratch spec.
    store = env.store()
    with pytest.raises(RunnerError, match="local changes"):
        enqueue(env.config, store, env.spec.read_bytes(), sha=env.sha, now=time.time())
    scratch = spec_text("demo", scratch=True, seeds="[7]").encode()
    result = enqueue(env.config, store, scratch, sha=env.sha, now=time.time())
    assert len(result.added) == 2
    store.close()


def test_worktree_errors(env: Env) -> None:
    with pytest.raises(RunnerError, match="40-digit"):
        ensure_worktree(env.config, "abc")
    with pytest.raises(RunnerError, match="push it"):
        ensure_worktree(env.config, "f" * 40)
    dest = ensure_worktree(env.config, env.sha)
    assert ensure_worktree(env.config, env.sha) == dest  # reused
    git(dest, "checkout", "--quiet", "--detach", "HEAD")
    (dest / "new").write_text("x")
    git(dest, "add", "new")
    git(dest, "commit", "--quiet", "-m", "moved")
    with pytest.raises(RunnerError, match="remove it"):
        ensure_worktree(env.config, env.sha)


def test_enqueue_reports_a_failed_command(env: Env) -> None:
    failing_uv = env.tmp / "failing-uv"
    failing_uv.write_text("#!/bin/sh\necho 'lock out of date' >&2\nexit 2\n")
    failing_uv.chmod(0o755)
    config = RunnerConfig(
        home=env.config.home,
        repo=env.config.repo,
        data_dir=env.config.data_dir,
        runs_root=env.config.runs_root,
        uv=str(failing_uv),
        entry_point=CLI,
    )
    store = env.store()
    with pytest.raises(RunnerError, match=r"uv sync failed .*lock out of date"):
        enqueue(config, store, env.spec.read_bytes(), sha=env.sha, now=time.time())
    store.close()
    assert os.environ["FAKE_SSH_HOME"] == str(env.tmp)
