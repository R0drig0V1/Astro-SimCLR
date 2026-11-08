"""The fixed entry point of a queued job: ``spec expand`` and ``spec run-cell`` (task A3).

Both run with the code of the *submitted commit*, inside its per-SHA worktree on the runner
(:mod:`~astro_simclr.queue.runner`), so the settings, presets and hashes are that commit's,
not the runner's. The runner only ever calls them as argv arrays that name a spec file;
no shell, no interpolated string (ADR 0016).

- :func:`expand` (``spec expand SPEC --data-dir D --repo-root R``) prints, as JSON
  (:data:`EXPANSION_FORMAT`), the spec id and every cell with its run id and config hash.
  The run id needs the data's SHA-256 (from the prepared ``meta.json`` under ``D``) and the
  splits file's (under ``R``), which is why it runs on the machine that has the data.
- :func:`run_cell` (``spec run-cell SPEC --cell KEY --attempt N --run-dir DIR --data-dir D
  --repo-root R``) runs one cell with ``run_command`` and keeps its ``manifest.json``
  (ADR 0015) up to date:

  1. it takes an exclusive ``flock`` on ``DIR/.lock`` for its whole life, or exits with
     :data:`EXIT_BUSY` (75): two processes never write one run directory, even if a
     reaper mistook a live job for a dead one;
  2. it refuses a dirty or unknown git tree for a non-scratch spec (:data:`EXIT_REFUSED`);
  3. a new run directory gets a ``queued`` manifest; an existing one is read and must hold
     the same run id and attempt. A ``finished`` attempt exits 0 at once (a requeue after
     the job finished but before the worker recorded it); a ``failed`` or ``cancelled``
     one is refused; a ``running`` one is a crash (``kill -9``, a lost lease, a reboot) and
     goes back to ``queued`` (the manifest's expired-lease transition);
  4. ``running`` (with a machine record), then ``run_command(..., resume="auto")``: a
     fresh start, or the continuation of ``checkpoints/last.ckpt``;
  5. ``finished`` with the run's own machine record from ``metrics.json``, or ``failed``
     on an exception (exit 1). SIGTERM, SIGINT or SIGHUP (a worker shutting down) put it
     back to ``queued`` and exit with :data:`EXIT_TERMINATED` (143): the next run resumes;
  6. it writes ``DIR/job_result.json`` with the peak device memory (CUDA's
     ``max_memory_reserved``, ``null`` elsewhere), which the worker records for admission.

  ``ASTRO_SIMCLR_CONCURRENT_JOBS`` (set by the worker) is the manifest's
  ``concurrent_jobs``.

The engine is ``lightning`` until task 4.8.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import signal
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, Final

from astro_simclr.config import Settings
from astro_simclr.data.files import atomic_write_bytes, sha256_file
from astro_simclr.data.prepared import is_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.runs.identity import (
    Engine,
    RunIdentity,
    data_sha256,
    git_state,
    run_identity,
    uv_lock_sha256,
)
from astro_simclr.runs.manifest import (
    MANIFEST_FILE,
    Manifest,
    ManifestError,
    new_manifest,
    read_manifest,
    transition,
    write_manifest,
)
from astro_simclr.runs.spec import Cell, ExperimentSpec, SpecError, load_spec

__all__ = [
    "CONCURRENT_JOBS_ENV",
    "ENGINE",
    "EXIT_BUSY",
    "EXIT_FAILED",
    "EXIT_REFUSED",
    "EXIT_TERMINATED",
    "EXPANSION_FORMAT",
    "JOB_RESULT_FILE",
    "LOCK_FILE",
    "cell_identity",
    "expand",
    "run_cell",
]

#: The engine of queued runs (the plain engine replaces it with task 4.8).
ENGINE: Final[Engine] = "lightning"
#: ``format`` of :func:`expand`'s JSON (version 1).
EXPANSION_FORMAT: Final = "astro-simclr-spec-expansion"
#: The run-directory lock (never synced).
LOCK_FILE: Final = ".lock"
#: The job's result for the worker: the peak device memory.
JOB_RESULT_FILE: Final = "job_result.json"
#: Environment variable the worker sets: jobs sharing the device (the manifest's field).
CONCURRENT_JOBS_ENV: Final = "ASTRO_SIMCLR_CONCURRENT_JOBS"

#: Exit code of a run that failed (an exception in training).
EXIT_FAILED: Final = 1
#: Exit code of a cell refused before it starts (dirty tree, wrong or ended attempt).
EXIT_REFUSED: Final = 2
#: Exit code when another process holds the run directory (``EX_TEMPFAIL``).
EXIT_BUSY: Final = 75
#: Exit code after SIGTERM, SIGINT or SIGHUP: the attempt is back to ``queued``.
EXIT_TERMINATED: Final = 128 + signal.SIGTERM


class _Terminated(BaseException):
    """Raised by the signal handler: a BaseException, so no ``except Exception`` eats it."""


def _splits_sha256(spec: ExperimentSpec, repo_root: Path) -> str | None:
    return None if spec.splits is None else sha256_file(repo_root / spec.splits.file)


def _dataset(settings: Settings) -> Path:
    """``experiments.dataset_path`` without importing torch (``spec expand`` stays light)."""
    path = settings.data_dir / settings.dataset
    if not is_prepared(path):
        msg = (
            f"{path} is not a prepared dataset (the runner's data_dir holds the prepared "
            "layouts; astro-simclr prepare-data)"
        )
        raise DataError(msg)
    return path


def cell_identity(
    spec: ExperimentSpec,
    cell: Cell,
    settings: Settings,
    *,
    repo_root: Path,
    splits_sha256: str | None = None,
) -> RunIdentity:
    """The run identity of ``cell`` with its resolved ``settings`` (data read from disk).

    ``splits_sha256`` saves rehashing the splits file for every cell.

    Raises:
        DataError: if the dataset is not a prepared layout.
        OSError: if the splits file cannot be read.
    """
    splits = _splits_sha256(spec, repo_root) if splits_sha256 is None else splits_sha256
    return run_identity(
        spec.command,
        settings,
        data_sha256=data_sha256(_dataset(settings)),
        machine_class=spec.machine_class,
        engine=ENGINE,
        split=cell.split,
        splits_sha256=None if cell.split is None else splits,
    )


def expand(spec_path: Path, *, data_dir: Path, repo_root: Path) -> dict[str, object]:
    """The JSON document of ``spec expand`` (module docstring).

    Raises:
        SpecError: if the spec is invalid.
        DataError: if an arm's dataset is not prepared under ``data_dir``.
        OSError: if the splits file cannot be read.
    """
    spec = load_spec(spec_path)
    splits = _splits_sha256(spec, repo_root)
    cells: list[dict[str, object]] = []
    for cell in spec.cells():
        settings = spec.arm_settings(cell.arm, cell.seed, data_dir=data_dir)
        identity = cell_identity(spec, cell, settings, repo_root=repo_root, splits_sha256=splits)
        cells.append(
            {
                "cell": cell.key,
                "run_id": identity.run_id,
                "config_sha256": identity.config_sha256,
                "data_sha256": identity.data_sha256,
            }
        )
    return {
        "format": EXPANSION_FORMAT,
        "version": 1,
        "spec_id": spec.spec_id,
        "name": spec.name,
        "command": spec.command,
        "machine_class": spec.machine_class,
        "extra": spec.extra,
        "scratch": spec.scratch,
        "engine": ENGINE,
        "cells": cells,
    }


@contextmanager
def _locked(run_dir: Path) -> Iterator[bool]:
    """Hold ``run_dir/.lock`` exclusively; yield ``False`` if another process holds it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / LOCK_FILE).open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _signals_terminate() -> Iterator[None]:
    """Turn SIGTERM, SIGINT and SIGHUP into :class:`_Terminated` while the run lasts."""

    def handler(signum: int, _frame: FrameType | None) -> None:
        raise _Terminated(signum)

    names = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous = {signum: signal.signal(signum, handler) for signum in names}
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


def _start_machine(settings: Settings) -> dict[str, Any]:
    """The machine record of a starting run (the device Lightning's ``auto`` will pick).

    The finished manifest takes ``metrics.json``'s record instead, the device actually used.
    """
    import platform

    import torch

    from astro_simclr.training.runs import apply_fp32_precision, fp32_precision_record

    apply_fp32_precision(settings)
    wants = settings.accelerator
    if wants in ("auto", "gpu", "cuda") and torch.cuda.is_available():
        device = "cuda"
    elif wants in ("auto", "gpu", "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "processor": platform.machine(),
        **fp32_precision_record(torch.device(device)),
    }


def _peak_memory(device: object) -> int | None:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_reserved())


def _concurrent_jobs() -> int:
    try:
        return max(1, int(os.environ.get(CONCURRENT_JOBS_ENV, "1")))
    except ValueError:
        return 1


def _prepare_manifest(
    run_dir: Path, fresh: Manifest, attempt: int
) -> tuple[Manifest | None, str | None]:
    """The ``queued`` manifest to start from, or ``(None, reason)`` to refuse / skip."""
    if not (run_dir / MANIFEST_FILE).is_file():
        return fresh, None
    existing = read_manifest(run_dir)
    if existing.run_id != fresh.run_id or existing.attempt != attempt:
        return None, (
            f"{run_dir} holds run {existing.run_id[:16]} attempt {existing.attempt}, not "
            f"{fresh.run_id[:16]} attempt {attempt}"
        )
    if existing.status == "running":  # a crash or a lost lease: resume
        return transition(existing, "queued"), None
    if existing.status == "queued":
        return existing, None
    return None, f"attempt is {existing.status}"


def _train(
    spec: ExperimentSpec,
    cell: Cell,
    settings: Settings,
    queued: Manifest,
    *,
    run_dir: Path,
    repo_root: Path,
) -> int:
    """Steps 4-6 of the module docstring, from a ``queued`` manifest; the exit code."""
    from astro_simclr.experiments import run_command
    from astro_simclr.protocol.split_runs import SplitRequest

    manifest = transition(queued, "running", machine=_start_machine(settings))
    write_manifest(run_dir, manifest)
    split = None
    if spec.splits is not None and cell.split is not None:
        fold = cell.split if isinstance(cell.split, int) else None
        split = SplitRequest(repo_root / spec.splits.file, fold)
    try:
        with _signals_terminate():
            metrics = run_command(
                spec.command, settings, resume="auto", split=split, progress_bar=False
            )
    except _Terminated:
        write_manifest(run_dir, transition(manifest, "queued"))
        return EXIT_TERMINATED
    except Exception:
        traceback.print_exc()
        write_manifest(run_dir, transition(manifest, "failed"))
        return EXIT_FAILED
    machine = metrics.get("machine")
    if isinstance(machine, dict):
        # metrics.json always has the manifest's machine keys; keep the start record if not.
        with contextlib.suppress(ManifestError):
            manifest = dataclasses.replace(manifest, machine=machine)
    write_manifest(run_dir, transition(manifest, "finished"))
    device = manifest.machine["device"] if manifest.machine is not None else None
    result = {"format": "astro-simclr-job-result", "peak_memory_bytes": _peak_memory(device)}
    atomic_write_bytes(run_dir / JOB_RESULT_FILE, (json.dumps(result) + "\n").encode())
    return 0


def run_cell(
    spec_path: Path,
    cell_key: str,
    *,
    attempt: int,
    run_dir: Path,
    data_dir: Path,
    repo_root: Path,
) -> int:
    """Run one cell of a spec in ``run_dir`` (module docstring); return the exit code.

    Raises:
        SpecError: if the spec or the cell is invalid.
        ManifestError: if ``run_dir`` holds an invalid manifest.
        DataError: if the dataset is missing or invalid.
    """
    spec = load_spec(spec_path)
    cell = Cell.parse(cell_key)
    spec.arm(cell.arm)
    if cell not in spec.cells():
        raise SpecError(f"spec {spec.name} has no cell {cell_key}")
    settings = spec.arm_settings(cell.arm, cell.seed, data_dir=data_dir, output_dir=run_dir)
    with _locked(run_dir) as held:
        if not held:
            sys.stderr.write(f"astro-simclr: {run_dir} is held by another process\n")
            return EXIT_BUSY
        git = git_state(repo_root)
        if not spec.scratch and (git.sha is None or git.dirty):
            sys.stderr.write(
                f"astro-simclr: error: {repo_root} is {'dirty' if git.dirty else 'not a git tree'}"
                f"; spec {spec.name} is not scratch\n"
            )
            return EXIT_REFUSED
        identity = cell_identity(spec, cell, settings, repo_root=repo_root)
        fresh = new_manifest(
            spec.command,
            settings,
            identity,
            git=git,
            uv_lock_sha256=uv_lock_sha256(repo_root),
            attempt=attempt,
            concurrent_jobs=_concurrent_jobs(),
        )
        queued, reason = _prepare_manifest(run_dir, fresh, attempt)
        if queued is None:
            if reason == "attempt is finished":
                return 0
            sys.stderr.write(f"astro-simclr: error: {reason}\n")
            return EXIT_REFUSED
        return _train(spec, cell, settings, queued, run_dir=run_dir, repo_root=repo_root)
