"""Tests of a job's fixed entry point, ``spec expand`` and ``spec run-cell`` (``queue.job``).

Dry ``train-supervised`` runs on the synthetic data, in-process: the manifest life cycle
(queued, running, finished; back to queued after a crash or SIGTERM; failed on an
exception), the run-directory lock, the dirty-tree refusal and the idempotent rerun.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
from pathlib import Path
from typing import Any

import pytest
from _queue_support import init_repo, spec_text

from astro_simclr import experiments
from astro_simclr.queue import job as job_module
from astro_simclr.queue.job import (
    EXIT_BUSY,
    EXIT_FAILED,
    EXIT_REFUSED,
    EXIT_TERMINATED,
    EXPANSION_FORMAT,
    JOB_RESULT_FILE,
    LOCK_FILE,
    expand,
    run_cell,
)
from astro_simclr.runs.identity import data_sha256
from astro_simclr.runs.manifest import new_manifest, read_manifest, transition, write_manifest
from astro_simclr.runs.spec import Cell, SpecError, expected_run_ids, parse_spec

CELL = "baseline/s0/all"


@pytest.fixture(scope="module")
def data_dir(synthetic_prepared: dict[str, Path]) -> Path:
    """The parent of the prepared synthetic datasets."""
    return next(iter(synthetic_prepared.values())).parent


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clean git checkout standing in for the per-SHA worktree."""
    root = tmp_path / "worktree"
    init_repo(root, {"README": "x\n", "configs/splits/s.json": "{}\n"})
    return root


def _spec(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / "specs" / "demo.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec_text() if text is None else text)
    return path


def _run(spec: Path, run_dir: Path, data_dir: Path, repo: Path, *, attempt: int = 1) -> int:
    return run_cell(spec, CELL, attempt=attempt, run_dir=run_dir, data_dir=data_dir, repo_root=repo)


def test_expand_matches_expected_run_ids(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    spec_file = _spec(tmp_path)
    document = expand(spec_file, data_dir=data_dir, repo_root=repo)
    assert document["format"] == EXPANSION_FORMAT
    spec = parse_spec(spec_file.read_text())
    assert document["spec_id"] == spec.spec_id
    data = data_sha256(data_dir / spec.arm_settings("baseline", 0).dataset)
    expected = expected_run_ids(spec, data_sha256=data, splits_sha256=None)
    cells = document["cells"]
    assert isinstance(cells, list)
    assert {entry["cell"]: entry["run_id"] for entry in cells} == {
        cell.key: run_id for cell, run_id in expected.items()
    }
    assert {entry["config_sha256"] for entry in cells} == {
        spec.arm_config_sha256(arm) for arm in ("baseline", "wide")
    }


def test_expand_hashes_the_splits_file(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    text = spec_text() + "splits: {file: configs/splits/s.json, screen: true}\n"
    document = expand(_spec(tmp_path, text), data_dir=data_dir, repo_root=repo)
    cells = document["cells"]
    assert isinstance(cells, list) and cells[0]["cell"] == "baseline/s0/screen"


def test_run_cell_writes_a_finished_manifest(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    spec_file = _spec(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    assert _run(spec_file, run_dir, data_dir, repo) == 0
    manifest = read_manifest(run_dir)
    assert manifest.status == "finished" and manifest.attempt == 1
    assert manifest.engine == "lightning" and manifest.identity.machine_class == "cpu-test"
    assert manifest.git.sha is not None and manifest.git.dirty is False
    assert manifest.machine is not None and manifest.machine["device"] == "cpu"
    assert manifest.config["output_dir"] == str(run_dir)
    assert (run_dir / "metrics.json").exists() and (run_dir / LOCK_FILE).exists()
    result = json.loads((run_dir / JOB_RESULT_FILE).read_text())
    assert result["peak_memory_bytes"] is None  # CPU: no device memory to measure
    spec = parse_spec(spec_file.read_text())
    ids = expected_run_ids(
        spec,
        data_sha256=manifest.identity.data_sha256,
        splits_sha256=None,
    )
    assert manifest.run_id == ids[Cell.parse(CELL)]
    # A rerun of a finished attempt does nothing and succeeds (a requeue after the end).
    before = (run_dir / "metrics.json").stat().st_mtime_ns
    assert _run(spec_file, run_dir, data_dir, repo) == 0
    assert (run_dir / "metrics.json").stat().st_mtime_ns == before
    # Another attempt number on the same directory is refused.
    assert _run(spec_file, run_dir, data_dir, repo, attempt=2) == EXIT_REFUSED


def test_a_crashed_attempt_resumes(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    """A manifest left ``running`` (kill -9, lost lease) goes back to queued, then runs."""
    spec_file = _spec(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    spec = parse_spec(spec_file.read_text())
    settings = spec.arm_settings("baseline", 0, data_dir=data_dir, output_dir=run_dir)
    identity = job_module.cell_identity(spec, Cell.parse(CELL), settings, repo_root=repo)
    from astro_simclr.runs.identity import git_state

    queued = new_manifest(
        spec.command, settings, identity, git=git_state(repo), uv_lock_sha256=None
    )
    machine = {"device": "cpu", "gpu": None, "torch": "x", "cuda": None, "fp32_precision": "ieee"}
    run_dir.mkdir(parents=True)
    write_manifest(run_dir, transition(queued, "running", machine=machine))
    assert _run(spec_file, run_dir, data_dir, repo) == 0
    assert read_manifest(run_dir).status == "finished"


def test_sigterm_puts_the_attempt_back_in_the_queue(
    tmp_path: Path, data_dir: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def terminated(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        os.kill(os.getpid(), signal.SIGTERM)  # the worker stopping us
        raise AssertionError("unreachable")  # pragma: no cover

    monkeypatch.setattr(experiments, "run_command", terminated)
    spec_file = _spec(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    assert _run(spec_file, run_dir, data_dir, repo) == EXIT_TERMINATED
    assert read_manifest(run_dir).status == "queued"
    assert signal.getsignal(signal.SIGTERM) is not None
    monkeypatch.undo()
    assert _run(spec_file, run_dir, data_dir, repo) == 0  # the next start resumes
    assert read_manifest(run_dir).status == "finished"


def test_an_exception_fails_the_attempt(
    tmp_path: Path, data_dir: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("diverged")

    monkeypatch.setattr(experiments, "run_command", broken)
    spec_file = _spec(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    assert _run(spec_file, run_dir, data_dir, repo) == EXIT_FAILED
    assert read_manifest(run_dir).status == "failed"
    monkeypatch.undo()
    assert _run(spec_file, run_dir, data_dir, repo) == EXIT_REFUSED  # a new attempt is needed


def test_a_held_run_directory_is_busy(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    with (run_dir / LOCK_FILE).open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        assert _run(_spec(tmp_path), run_dir, data_dir, repo) == EXIT_BUSY
    assert not (run_dir / "manifest.json").exists()


def test_a_dirty_tree_is_refused_unless_scratch(
    tmp_path: Path, data_dir: Path, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "README").write_text("edited on the GPU machine\n")
    run_dir = tmp_path / "runs" / "r"
    assert _run(_spec(tmp_path), run_dir, data_dir, repo) == EXIT_REFUSED
    assert "dirty" in capsys.readouterr().err
    scratch = _spec(tmp_path, spec_text(scratch=True))
    assert _run(scratch, tmp_path / "runs" / "s", data_dir, repo) == 0
    assert read_manifest(tmp_path / "runs" / "s").git.dirty is True
    outside = tmp_path / "not-git"
    outside.mkdir()
    assert _run(_spec(tmp_path), tmp_path / "runs" / "t", data_dir, outside) == EXIT_REFUSED


def test_an_unknown_cell_is_an_error(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    spec_file = _spec(tmp_path)
    with pytest.raises(SpecError, match="no cell"):
        run_cell(
            spec_file,
            "baseline/s7/all",
            attempt=1,
            run_dir=tmp_path / "r",
            data_dir=data_dir,
            repo_root=repo,
        )
    with pytest.raises(SpecError, match="no arm"):
        run_cell(
            spec_file,
            "nope/s0/all",
            attempt=1,
            run_dir=tmp_path / "r",
            data_dir=data_dir,
            repo_root=repo,
        )


def test_concurrent_jobs_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(job_module.CONCURRENT_JOBS_ENV, "3")
    assert job_module._concurrent_jobs() == 3
    monkeypatch.setenv(job_module.CONCURRENT_JOBS_ENV, "x")
    assert job_module._concurrent_jobs() == 1
    monkeypatch.setenv(job_module.CONCURRENT_JOBS_ENV, "0")
    assert job_module._concurrent_jobs() == 1
