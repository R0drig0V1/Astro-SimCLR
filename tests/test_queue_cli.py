"""Tests of the ``spec`` and ``queue`` subcommands and of the worker hosting (``queue.cli``,
``queue.service``; task A3)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from _queue_support import CLI, add_fake_job, init_repo, spec_text

from astro_simclr.cli import main
from astro_simclr.queue.runner import RunnerConfig, RunnerError
from astro_simclr.queue.service import (
    SERVICE_NAME,
    service_commands,
    systemd_unit,
    tmux_has_session_argv,
    tmux_kill_session_argv,
    tmux_start_argv,
    tmux_target,
    unit_path,
)
from astro_simclr.queue.store import QueueStore


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``$ASTRO_SIMCLR_HOME`` in ``tmp_path``."""
    path = tmp_path / "home"
    monkeypatch.setenv("ASTRO_SIMCLR_HOME", str(path))
    return path


def _out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


def test_spec_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "s.yaml"
    spec.write_text(spec_text())
    assert main(["spec", "check", str(spec)]) == 0
    assert "demo: spec id" in _out(capsys)
    assert main(["spec", "check", str(spec), "--json"]) == 0
    data = json.loads(_out(capsys))
    assert len(data["cells"]) == 4
    spec.write_text("format: nope\n")
    assert main(["spec", "check", str(spec)]) == 2
    assert "not an astro-simclr-experiment-spec" in capsys.readouterr().err


def test_spec_expand_and_run_cell(
    tmp_path: Path, classifier_f32_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = classifier_f32_dir.parent
    repo = tmp_path / "repo"
    init_repo(repo)
    spec = tmp_path / "s.yaml"
    spec.write_text(spec_text(seeds="[0]"))
    base = ["--data-dir", str(data_dir), "--repo-root", str(repo)]
    assert main(["spec", "expand", str(spec), *base]) == 0
    assert len(json.loads(_out(capsys))["cells"]) == 2
    run_dir = tmp_path / "run"
    argv = ["spec", "run-cell", str(spec), "--cell", "wide/s0/all", "--attempt", "1"]
    assert main([*argv, "--run-dir", str(run_dir), *base]) == 0
    assert (run_dir / "manifest.json").exists()
    assert (
        main(["spec", "expand", str(spec), "--data-dir", str(tmp_path), "--repo-root", str(repo)])
        == 1
    )
    assert "not a prepared dataset" in capsys.readouterr().err


def test_queue_init_status_cancel_paths(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["queue", "status"]) == 1
    assert "queue init" in capsys.readouterr().err
    repo = tmp_path / "repo"
    repo.mkdir()
    init = ["queue", "init", "--repo", str(repo), "--data-dir", str(tmp_path), "--uv", "/x/uv"]
    assert main([*init, "--machine-class", "cpu-test"]) == 0
    config = RunnerConfig.load()
    assert config.repo == repo.resolve() and config.uv == "/x/uv"
    assert config.machine_classes == ("cpu-test",)
    assert config.runs_root == (home / "runs").resolve()
    capsys.readouterr()
    assert main(["queue", "paths"]) == 0
    assert json.loads(_out(capsys))["specs_dir"] == str(home / "specs")
    store = QueueStore.open(config.queue_file)
    claimed = add_fake_job(store, tmp_path / "r1", "ok", spec_id="s1")
    waiting = add_fake_job(store, tmp_path / "r2", "ok", spec_id="s2")
    store.claim(
        owner="w",
        machine_class="cpu-test",
        host="h",
        boot_id=None,
        admit=lambda job: "cpu" if job.job_id == claimed.job_id else None,
        now=1.0,
        lease_seconds=5,
    )
    store.close()
    assert main(["queue", "status"]) == 0
    text = _out(capsys)
    assert "running" in text and "queued" in text
    assert main(["queue", "status", "--json", "--spec", "s2"]) == 0
    assert [job["job_id"] for job in json.loads(_out(capsys))] == [waiting.job_id]
    assert main(["queue", "cancel", "--spec", "s"]) == 1  # two specs match: nothing cancelled
    assert "exactly one spec" in capsys.readouterr().err
    assert main(["queue", "cancel", "--spec", ""]) == 1
    capsys.readouterr()
    assert main(["queue", "cancel", "--spec", "s2", str(claimed.job_id)]) == 0
    text = _out(capsys)
    assert f"{claimed.job_id}: cancel requested" in text and f"{waiting.job_id}: cancelled" in text


def test_queue_enqueue_from_stdin(
    home: Path,
    tmp_path: Path,
    classifier_f32_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    sha = init_repo(repo)
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    RunnerConfig(
        home=home,
        repo=repo,
        data_dir=classifier_f32_dir.parent,
        runs_root=home / "runs",
        uv=str(uv),
        entry_point=CLI,
    ).save()
    import io

    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(spec_text().encode())))
    assert main(["queue", "enqueue", "--sha", sha, "-"]) == 0
    assert len(json.loads(_out(capsys))["added"]) == 4
    spec = tmp_path / "s.yaml"
    spec.write_text(spec_text())
    assert main(["queue", "enqueue", "--sha", sha, str(spec), "--new-attempt"]) == 0
    assert {job["attempt"] for job in json.loads(_out(capsys))["added"]} == {2}
    assert main(["queue", "enqueue", "--sha", "f" * 40, str(spec)]) == 1
    assert "git fetch failed" in capsys.readouterr().err


def test_queue_worker_cli(home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    RunnerConfig(home=home, repo=tmp_path, data_dir=tmp_path, runs_root=home / "runs").save()
    store = QueueStore.open(RunnerConfig.load().queue_file)
    job = add_fake_job(store, tmp_path / "r1", "ok")
    store.close()
    argv = [
        "queue",
        "worker",
        "--idle-exit",
        "--poll-seconds",
        "0.02",
        "--machine-class",
        "cpu-test",
    ]
    assert main(argv) == 0
    store = QueueStore.open(RunnerConfig.load().queue_file)
    assert store.job(job.job_id).status == "finished"  # type: ignore[union-attr]
    store.close()
    assert main(["queue", "worker", "--gpu", "0"]) == 2
    assert "needs --machine-class" in capsys.readouterr().err
    assert main(["queue", "worker", "--gpu", "0", "--slots-per-gpu", "2"]) == 2
    assert "--gpu-memory-mib" in capsys.readouterr().err


def test_queue_service(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    RunnerConfig(home=home, repo=tmp_path, data_dir=tmp_path, runs_root=home / "runs").save()
    assert main(["queue", "service", "--gpu", "0", "--machine-class", "cuda-x"]) == 0
    unit = _out(capsys)
    assert "Restart=always" in unit and '"--machine-class" "cuda-x"' in unit
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert main(["queue", "service", "--write"]) == 0
    assert (tmp_path / "xdg" / "systemd" / "user" / SERVICE_NAME).exists()
    assert "loginctl enable-linger" in _out(capsys)


def test_systemd_unit_quoting() -> None:
    unit = systemd_unit(
        ["/a b/python", "100%", 'say "hi"', "$HOME"], home=Path("/h$x"), working_dir=Path("/w%")
    )
    assert 'ExecStart="/a b/python" "100%%" "say \\"hi\\"" "$$HOME"' in unit
    assert "KillMode=mixed" in unit and "WorkingDirectory=/w%%" in unit
    assert 'Environment="ASTRO_SIMCLR_HOME=/h$x"' in unit
    assert unit_path(Path("/c")) == Path("/c/systemd/user") / SERVICE_NAME
    assert service_commands("me")[-1] == ["loginctl", "enable-linger", "me"]


def test_tmux_targets_are_exact() -> None:
    assert tmux_target("screen25") == "=screen25"
    assert tmux_has_session_argv("screen25") == ["tmux", "has-session", "-t", "=screen25"]
    assert tmux_kill_session_argv("w", socket="s") == [
        "tmux",
        "-L",
        "s",
        "kill-session",
        "-t",
        "=w",
    ]
    start = tmux_start_argv("w", ["/py", "a b"], home=Path("/h"))
    assert start[:5] == ["tmux", "new-session", "-d", "-s", "w"]
    assert start[5] == "env ASTRO_SIMCLR_HOME=/h /py 'a b'"
    for bad in ("", "a b", "a:b", "x.y", "=x"):
        with pytest.raises(ValueError, match="tmux session"):
            tmux_target(bad)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_tmux_prefix_regression() -> None:
    """``has-session -t screen25`` was true with only ``screen25c``; ``=screen25`` is not."""
    socket = f"astro-simclr-test-{uuid.uuid4().hex[:8]}"

    def tmux(argv: list[str]) -> int:
        return subprocess.run(argv, check=False).returncode  # noqa: S603 - fixed argvs

    assert tmux(["tmux", "-L", socket, "new-session", "-d", "-s", "screen25c", "sleep 30"]) == 0
    try:
        assert tmux(["tmux", "-L", socket, "has-session", "-t", "screen25"]) == 0  # the bug
        assert tmux(tmux_has_session_argv("screen25", socket=socket)) != 0
        assert tmux(tmux_has_session_argv("screen25c", socket=socket)) == 0
    finally:
        tmux(["tmux", "-L", socket, "kill-server"])


def test_queue_tmux_cli(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    RunnerConfig(home=home, repo=tmp_path, data_dir=tmp_path, runs_root=home / "runs").save()
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0 if len(calls) == 1 else 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert main(["queue", "tmux", "--session", "q1"]) == 0
    assert "already runs" in _out(capsys)
    assert calls == [["tmux", "has-session", "-t", "=q1"]]
    calls.clear()
    calls.append(["placeholder"])  # the next has-session "fails": start the session
    assert main(["queue", "tmux", "--session", "q1"]) == 1
    assert calls[-1][:5] == ["tmux", "new-session", "-d", "-s", "q1"]


def test_runner_config_errors(tmp_path: Path) -> None:
    with pytest.raises(RunnerError, match="queue init"):
        RunnerConfig.load(tmp_path)
    (tmp_path / "runner.yaml").write_text("repo: [")
    with pytest.raises(RunnerError, match="invalid YAML"):
        RunnerConfig.load(tmp_path)
    (tmp_path / "runner.yaml").write_text("repo: x\n")
    with pytest.raises(RunnerError, match="needs repo"):
        RunnerConfig.load(tmp_path)


def test_submit_and_status_over_a_host(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from astro_simclr.queue import remote

    def fake(_transport: object, argv: list[str], **_kwargs: object) -> object:
        return (
            [
                {
                    "job_id": 1,
                    "status": "queued",
                    "cell": "a/s0/all",
                    "attempt": 1,
                    "resumes": 0,
                    "run_dir": "/r",
                    "error": None,
                }
            ]
            if "status" in argv
            else {"ok": True}
        )

    monkeypatch.setattr(remote, "remote_json", fake)
    assert main(["queue", "status", "--host", "runner", "--spec", "abc"]) == 0
    assert "a/s0/all" in _out(capsys)
    assert main(["queue", "status", "--host=-bad"]) == 1
    assert "host name" in capsys.readouterr().err
