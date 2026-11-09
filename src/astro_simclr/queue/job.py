"""The fixed entry point of a queued job: ``spec expand`` and ``spec run-cell`` (task A3).

Both run with the code of the *submitted commit*, inside its per-SHA worktree on the runner
(:mod:`~astro_simclr.queue.runner`), so the settings, presets and hashes are that commit's,
not the runner's. The runner only ever calls them as argv arrays that name a spec file:
no shell, no interpolated string (ADR 0026).

- :func:`expand` (``spec expand SPEC --data-dir D --repo-root R``) prints, as JSON
  (:data:`EXPANSION_FORMAT`), the spec id and every cell with its run id and config hash.
  The run id needs the data's SHA-256 (from the prepared ``meta.json`` under ``D``) and the
  splits file's (under ``R``), which is why it runs on the machine that has the data. It
  imports no torch.
- :func:`run_cell` (``spec run-cell SPEC --cell KEY --attempt N --run-dir DIR --data-dir D
  --repo-root R``) runs one cell:

  1. it takes an exclusive ``flock`` on ``DIR/.lock`` for its whole life, or exits with
     :data:`EXIT_BUSY` (75): two processes never write one run directory, even if a
     reaper mistook a live job for a dead one;
  2. it refuses a dirty or unknown git tree for a non-scratch spec (:data:`EXIT_REFUSED`);
  3. it reads the manifest already in ``DIR``, if any. A ``finished`` attempt exits 0 at
     once (a requeue after the job ended but before the worker recorded it); a
     ``failed`` or ``cancelled`` one, or one of another run id or attempt, is refused;
  4. on the spec's ``engine``, ``experiments.run_command`` (``lightning``) or
     ``tasks.plain.run_plain`` (``plain``; for an ``rj23`` arm with an ``RJ23Request`` of
     the object splits, a decision run for ``screen``, a replication for ``whole``), with
     ``resume="auto"`` and
     ``queued=QueueAttempt(N, spec machine class)``: a fresh start, or the continuation of
     ``checkpoints/last.ckpt`` (``ckpt/last.pt`` on the plain engine). Either writes the
     manifest (task A1, ``runs.record``): ``queued -> running``, then
     ``finished`` or ``failed``; an attempt left ``running`` by a crash goes
     ``running -> queued -> running`` and keeps its start time;
  5. SIGTERM, SIGINT or SIGHUP (the worker stopping it) end the process at once with
     ``128 + signal`` and write nothing: the manifest stays ``running``, exactly as after
     ``kill -9``, and the next start resumes it (the engine would otherwise record the
     attempt ``cancelled``, which is terminal);
  6. ``QueueAttempt`` also carries the cell's run id (the expansion's): ``start_run``
     refuses, before any step, to record another one. At the end it writes
     ``DIR/job_result.json`` with the peak device memory (CUDA's
     ``max_memory_reserved``, ``null`` elsewhere), which the worker records for admission.

  ``ASTRO_SIMCLR_CONCURRENT_JOBS`` (set by the worker) is the manifest's
  ``concurrent_jobs`` (``runs.record``).

The engine is part of the run id (ADR 0015), so :func:`expand` and :func:`cell_identity`
read it from the spec too.
"""

from __future__ import annotations

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
from typing import Final

from astro_simclr.config import RJ23, Settings
from astro_simclr.data.files import atomic_write_bytes, sha256_file
from astro_simclr.data.prepared import is_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.runs.identity import (
    SCREEN,
    RunIdentity,
    data_sha256,
    git_state,
    run_identity,
)
from astro_simclr.runs.manifest import MANIFEST_FILE, read_manifest
from astro_simclr.runs.spec import Cell, ExperimentSpec, SpecError, load_spec

__all__ = [
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

#: ``format`` of :func:`expand`'s JSON (version 1).
EXPANSION_FORMAT: Final = "astro-simclr-spec-expansion"
#: The run-directory lock (never synced).
LOCK_FILE: Final = ".lock"
#: The job's result for the worker: the peak device memory.
JOB_RESULT_FILE: Final = "job_result.json"

#: Exit code of a run that failed (an exception in training).
EXIT_FAILED: Final = 1
#: Exit code of a cell refused before it starts (dirty tree, wrong or ended attempt).
EXIT_REFUSED: Final = 2
#: Exit code when another process holds the run directory (``EX_TEMPFAIL``).
EXIT_BUSY: Final = 75
#: Exit code after SIGTERM (128 + 15); SIGINT and SIGHUP give 130 and 129.
EXIT_TERMINATED: Final = 128 + signal.SIGTERM


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
        engine=spec.engine,
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
        "engine": spec.engine,
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


def _exit_now(signum: int, _frame: FrameType | None) -> None:
    """End the process at once, writing nothing (module docstring, step 5)."""
    # os.write, not sys.stderr: a buffered write in a handler can raise "reentrant call".
    os.write(2, f"astro-simclr: signal {signum}: stopping; the attempt stays resumable\n".encode())
    os._exit(128 + signum)


@contextmanager
def _signals_exit() -> Iterator[None]:
    """While the run lasts, SIGTERM, SIGINT and SIGHUP call :func:`_exit_now`."""
    names = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    previous = {signum: signal.signal(signum, _exit_now) for signum in names}
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


def _peak_memory(device: object) -> int | None:
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_reserved())  # pragma: no cover - CUDA only


def _refusal(run_dir: Path, run_id: str, attempt: int) -> str | None:
    """Why the manifest in ``run_dir`` stops this attempt (``"finished"``: done already)."""
    if not (run_dir / MANIFEST_FILE).is_file():
        return None
    existing = read_manifest(run_dir)
    if existing.run_id != run_id or existing.attempt != attempt:
        return (
            f"{run_dir} holds run {existing.run_id[:16]} attempt {existing.attempt}, not "
            f"{run_id[:16]} attempt {attempt}"
        )
    if existing.status in ("queued", "running"):
        return None
    return existing.status


def _train(
    spec: ExperimentSpec,
    cell: Cell,
    settings: Settings,
    identity: RunIdentity,
    *,
    attempt: int,
    run_dir: Path,
    repo_root: Path,
) -> int:
    """Steps 4-6 of the module docstring; the exit code."""
    from astro_simclr.protocol.split_runs import SplitRequest
    from astro_simclr.runs.record import QueueAttempt

    split = None
    rj23 = None
    if spec.splits is not None and cell.split is not None:
        file = repo_root / spec.splits.file
        if settings.schedule == RJ23:
            from astro_simclr.tasks.rj23 import RJ23Request

            # The object splits travel in the request (runs.spec, "The rj23 schedule").
            rj23 = RJ23Request(file, "decision" if cell.split == SCREEN else "replication")
        else:
            split = SplitRequest(file, cell.split if isinstance(cell.split, int) else None)
    queued = QueueAttempt(attempt=attempt, machine_class=spec.machine_class, run_id=identity.run_id)
    try:
        with _signals_exit():
            if spec.engine == "plain":
                from astro_simclr.tasks.plain import run_plain

                run_plain(
                    spec.command, settings, resume="auto", split=split, rj23=rj23, queued=queued
                )
            else:
                from astro_simclr.experiments import run_command

                run_command(
                    spec.command,
                    settings,
                    resume="auto",
                    split=split,
                    progress_bar=False,
                    queued=queued,
                )
    except Exception:
        traceback.print_exc()
        return EXIT_FAILED
    manifest = read_manifest(run_dir)
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
        if not spec.scratch and (git.sha is None or git.dirty is not False):
            state = "dirty" if git.dirty else "not a git tree (or git status failed)"
            sys.stderr.write(
                f"astro-simclr: error: {repo_root} is {state}; spec {spec.name} is not scratch\n"
            )
            return EXIT_REFUSED
        identity = cell_identity(spec, cell, settings, repo_root=repo_root)
        refusal = _refusal(run_dir, identity.run_id, attempt)
        if refusal == "finished":
            return 0
        if refusal is not None:
            sys.stderr.write(f"astro-simclr: error: attempt refused: {refusal}\n")
            return EXIT_REFUSED
        return _train(
            spec, cell, settings, identity, attempt=attempt, run_dir=run_dir, repo_root=repo_root
        )
