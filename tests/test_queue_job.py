"""Tests of a job's fixed entry point, ``spec expand`` and ``spec run-cell`` (``queue.job``).

Short ``train-supervised`` runs of the paper classifier on the plain engine (the synthetic
float32 data), in-process: the manifest life cycle
(queued, running, finished; back to queued after a crash or SIGTERM; failed on an
exception), the run-directory lock, the dirty-tree refusal, the idempotent rerun and the
refusal of an ``engine: lightning`` spec (task 4.8).
"""

from __future__ import annotations

import dataclasses
import fcntl
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _queue_support import init_repo, spec_text
from _rj23_support import DATASET, rj23_layout

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
from astro_simclr.runs.identity import GitState, data_sha256
from astro_simclr.runs.manifest import new_manifest, read_manifest, transition, write_manifest
from astro_simclr.runs.spec import Cell, SpecError, expected_run_ids, parse_spec
from astro_simclr.tasks import plain

CELL = "baseline/s0/all"


@pytest.fixture(scope="module")
def data_dir(classifier_f32_dir: Path) -> Path:
    """The parent of the prepared synthetic dataset with float32 stamps."""
    return classifier_f32_dir.parent


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
    assert _run(spec_file, run_dir, data_dir, repo, attempt=3) == 0
    manifest = read_manifest(run_dir)
    assert manifest.status == "finished" and manifest.attempt == 3  # the queue's attempt
    assert manifest.engine == "plain" and manifest.identity.machine_class == "cpu-test"
    assert manifest.machine is not None and manifest.machine["device"] == "cpu"
    assert manifest.config["output_dir"] == str(run_dir)
    assert (run_dir / "metrics.json").exists() and (run_dir / LOCK_FILE).exists()
    result = json.loads((run_dir / JOB_RESULT_FILE).read_text())
    assert result["peak_memory_bytes"] is None  # CPU: no device memory to measure
    spec = parse_spec(spec_file.read_text())
    ids = expected_run_ids(spec, data_sha256=manifest.identity.data_sha256, splits_sha256=None)
    assert manifest.run_id == ids[Cell.parse(CELL)]
    # A rerun of a finished attempt does nothing and succeeds (a requeue after the end).
    before = (run_dir / "metrics.json").stat().st_mtime_ns
    assert _run(spec_file, run_dir, data_dir, repo, attempt=3) == 0
    assert (run_dir / "metrics.json").stat().st_mtime_ns == before
    # Another attempt number on the same directory is refused.
    assert _run(spec_file, run_dir, data_dir, repo, attempt=4) == EXIT_REFUSED


PLAIN_SPEC = """\
format: astro-simclr-experiment-spec
version: 1
name: plain-demo
command: train-supervised
settings:
  dataset: {dataset}
  schedule: carrasco2021
  supervised_model: paper_stamp_classifier
  image_size: 21
  eval_batch_size: 64
  early_stopping: false
  batch_size: 4
  max_steps: 4
  val_every_n_steps: 2
  horizon_increment: 0
  checkpoint_every_n_steps: 2
  accelerator: cpu
  deterministic: true
arms:
  baseline: {{}}
seeds: [0]
machine:
  class: cpu-test
engine: plain
scratch: false
"""


def test_a_plain_spec_trains_on_the_plain_engine(
    tmp_path: Path, classifier_f32_dir: Path, repo: Path
) -> None:
    """``engine: plain``: ``tasks.plain.run_plain`` with the queue's attempt; it resumes."""
    spec_file = _spec(tmp_path, PLAIN_SPEC.format(dataset=classifier_f32_dir.name))
    data = classifier_f32_dir.parent
    document = expand(spec_file, data_dir=data, repo_root=repo)
    assert document["engine"] == "plain"
    run_dir = tmp_path / "runs" / "p"
    spec = parse_spec(spec_file.read_text())
    settings = spec.arm_settings("baseline", 0, data_dir=data, output_dir=run_dir)
    identity = job_module.cell_identity(spec, Cell.parse(CELL), settings, repo_root=repo)
    # A crash before the first checkpoint: the manifest is left running, the same attempt.
    queued = new_manifest(
        spec.command, settings, identity, git=GitState(None, None), uv_lock_sha256=None, attempt=2
    )
    machine = {"device": "cpu", "gpu": None, "torch": "x", "cuda": None, "fp32_precision": "ieee"}
    run_dir.mkdir(parents=True)
    write_manifest(run_dir, transition(queued, "running", machine=machine))
    assert _run(spec_file, run_dir, data, repo, attempt=2) == 0
    manifest = read_manifest(run_dir)
    assert (manifest.status, manifest.engine, manifest.attempt) == ("finished", "plain", 2)
    assert manifest.identity.machine_class == "cpu-test"
    cells = document["cells"]
    assert isinstance(cells, list) and manifest.run_id == cells[0]["run_id"]
    assert (run_dir / "ckpt" / "last.pt").is_file() and (run_dir / JOB_RESULT_FILE).is_file()
    # The plain run id differs from the same cell's on Lightning.
    lightning = job_module.cell_identity(
        dataclasses.replace(spec, engine="lightning"), Cell.parse(CELL), settings, repo_root=repo
    )
    assert lightning.run_id != manifest.run_id


RJ23_SPEC = """\
format: astro-simclr-experiment-spec
version: 1
name: rj23-demo
command: train-supervised
preset: rj23-lowres
settings:
  dataset: {dataset}
  accelerator: cpu
  val_every_n_steps: 2
  max_steps: 6
  patience: 50
  batch_size: 8
  eval_batch_size: 8
  checkpoint_every_n_steps: 2
arms:
  baseline: {{}}
seeds: [0]
splits: {{file: configs/splits/objects.json, {mode}: true}}
machine:
  class: cpu-test
engine: plain
"""


@pytest.mark.parametrize("mode", ["screen", "whole"])
def test_a_queued_rj23_cell_runs_on_the_plain_engine(tmp_path: Path, mode: str) -> None:
    """An rj23 arm: the object splits go in an RJ23Request (decision or replication)."""
    data, objects = rj23_layout(tmp_path / "layout")
    root = tmp_path / "rj23-worktree"
    init_repo(root, {"configs/splits/objects.json": objects.read_text()})
    spec_file = _spec(tmp_path, RJ23_SPEC.format(dataset=DATASET, mode=mode))
    document = expand(spec_file, data_dir=data, repo_root=root)
    cells = document["cells"]
    assert isinstance(cells, list) and len(cells) == 1
    key = f"baseline/s0/{mode}"
    assert cells[0]["cell"] == key
    run_dir = tmp_path / "runs" / mode
    assert run_cell(spec_file, key, attempt=2, run_dir=run_dir, data_dir=data, repo_root=root) == 0
    manifest = read_manifest(run_dir)
    assert (manifest.status, manifest.engine, manifest.attempt) == ("finished", "plain", 2)
    assert manifest.identity.split == mode and manifest.identity.machine_class == "cpu-test"
    assert manifest.run_id == cells[0]["run_id"]
    assert (run_dir / "ckpt" / "last.pt").is_file() and (run_dir / JOB_RESULT_FILE).is_file()


def test_a_crashed_attempt_resumes(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    """A manifest left ``running`` (kill -9, SIGTERM, lost lease) continues, same attempt."""
    spec_file = _spec(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    spec = parse_spec(spec_file.read_text())
    settings = spec.arm_settings("baseline", 0, data_dir=data_dir, output_dir=run_dir)
    identity = job_module.cell_identity(spec, Cell.parse(CELL), settings, repo_root=repo)
    queued = new_manifest(
        spec.command, settings, identity, git=GitState(None, None), uv_lock_sha256=None
    )
    machine = {"device": "cpu", "gpu": None, "torch": "x", "cuda": None, "fp32_precision": "ieee"}
    run_dir.mkdir(parents=True)
    crashed = transition(queued, "running", machine=machine)
    write_manifest(run_dir, crashed)
    assert _run(spec_file, run_dir, data_dir, repo) == 0
    resumed = read_manifest(run_dir)
    assert resumed.status == "finished" and resumed.attempt == 1
    assert resumed.started_at == crashed.started_at  # the same attempt, continued


def test_a_signal_ends_the_process_without_writing(tmp_path: Path) -> None:
    code = (
        "import os, signal, time\n"
        "from astro_simclr.queue.job import _signals_exit\n"
        "with _signals_exit():\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    time.sleep(30)\n"
        f"open({str(tmp_path / 'reached')!r}, 'w').close()\n"
    )
    result = subprocess.run(  # noqa: S603 - this interpreter
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == EXIT_TERMINATED
    assert "stays resumable" in result.stderr
    assert not (tmp_path / "reached").exists()


def test_an_exception_fails_the_attempt(
    tmp_path: Path, data_dir: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("diverged")

    # After the manifest is written: run_plain records the attempt failed.
    monkeypatch.setattr(plain, "_base_metrics", broken)
    spec_file = _spec(tmp_path)
    run_dir = tmp_path / "runs" / "r"
    assert _run(spec_file, run_dir, data_dir, repo) == EXIT_FAILED
    assert read_manifest(run_dir).status == "failed"
    monkeypatch.undo()
    assert _run(spec_file, run_dir, data_dir, repo) == EXIT_REFUSED  # a new attempt is needed


def test_a_disagreeing_run_id_fails(
    tmp_path: Path, data_dir: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = job_module.cell_identity

    def other(*args: Any, **kwargs: Any) -> Any:
        import dataclasses

        return dataclasses.replace(real(*args, **kwargs), seed=99)

    monkeypatch.setattr(job_module, "cell_identity", other)
    assert _run(_spec(tmp_path), tmp_path / "runs" / "r", data_dir, repo) == EXIT_FAILED


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
    outside = tmp_path / "not-git"
    outside.mkdir()
    assert _run(_spec(tmp_path), tmp_path / "runs" / "t", data_dir, outside) == EXIT_REFUSED


def test_an_unknown_cell_is_an_error(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    spec_file = _spec(tmp_path)
    run_dir = tmp_path / "r"
    with pytest.raises(SpecError, match="no cell"):
        run_cell(
            spec_file,
            "baseline/s7/all",
            attempt=1,
            run_dir=run_dir,
            data_dir=data_dir,
            repo_root=repo,
        )
    with pytest.raises(SpecError, match="no arm"):
        run_cell(
            spec_file, "nope/s0/all", attempt=1, run_dir=run_dir, data_dir=data_dir, repo_root=repo
        )


def test_a_lightning_spec_is_refused(tmp_path: Path, data_dir: Path, repo: Path) -> None:
    # Task 4.8: the spec still parses (runs compare resolves its run ids), but no cell runs.
    spec_file = _spec(tmp_path, spec_text().replace("engine: plain", "engine: lightning"))
    run_dir = tmp_path / "runs" / "r"
    with pytest.raises(SpecError, match="lightning-final"):
        _run(spec_file, run_dir, data_dir, repo)
    assert not run_dir.exists()
