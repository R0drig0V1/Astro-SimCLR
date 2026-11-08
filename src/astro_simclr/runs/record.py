"""Record a run's manifest from ``experiments.run_command`` (phase 4, task A1; ADR 0015).

:func:`start_run` computes the run's identity, decides its attempt from the manifest
already in the run directory, and writes ``manifest.json`` as ``queued`` then ``running``;
:meth:`RunRecord.end` writes the final status. ``run_command`` calls them around
``trainer.fit``: ``finished`` after ``metrics.json`` is written, ``failed`` on an exception,
``cancelled`` on ``KeyboardInterrupt`` or ``SystemExit`` (Lightning turns a Ctrl-C during
the fit into ``SystemExit``). A run that fails before the manifest exists (a bad config,
missing data) never started and leaves none.

**Attempts in today's layout.** ``--output-dir`` holds one run directory chosen by the
user, so it keeps the manifest of its latest attempt only (the queue, A3, gives each
attempt a directory of its own, ``runs.layout.run_dir_name``). Given the manifest already
there:

- none, or one of another run id: attempt 1;
- the same run id, resumed (``--resume``) from an attempt still ``queued`` or ``running``
  (a process killed before it could record anything): the same attempt continues,
  ``running -> queued -> running``, as after an expired lease, and keeps its start time;
- the same run id otherwise (a rerun, or a resume after ``finished``, ``failed`` or
  ``cancelled``, which are terminal): the next attempt.

A manifest that cannot be read is an error before training (``ConfigError``): it is never
silently replaced.

**Queued runs** (task A3, :class:`QueueAttempt`). The queue names each attempt's directory
and fixes its attempt number and machine class (the spec's, ADR 0015 rule 7) before the
run starts; ``spec run-cell`` passes them through ``run_command(queued=...)``. Then the
attempt is the queue's, not inferred: a fresh directory starts it, and a manifest of the
same run id and attempt still ``queued`` or ``running`` (a ``kill -9``, a lost lease, a
SIGTERM, a reboot) continues it, whether or not a ``last`` checkpoint was written yet. Any
other manifest in the directory is an error (``ConfigError``).

**Provenance.** The git state and the ``uv.lock`` SHA-256 are read from the source tree the
package runs from (the checkout holding ``pyproject.toml``); an installed wheel has neither
and records nulls. ``concurrent_jobs`` comes from the environment variable
``ASTRO_SIMCLR_CONCURRENT_JOBS`` (default 1): the hand-written queues that run two jobs on
one GPU set it, and the queue (A3) will.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from astro_simclr.config import ConfigError, Settings
from astro_simclr.data.files import sha256_file
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.runs.identity import (
    Engine,
    GitState,
    data_sha256,
    default_machine_class,
    git_state,
    run_identity,
    split_part,
    uv_lock_sha256,
)
from astro_simclr.runs.manifest import (
    MANIFEST_FILE,
    Manifest,
    ManifestError,
    Status,
    new_manifest,
    read_manifest,
    transition,
    write_manifest,
)

__all__ = [
    "CONCURRENT_JOBS_ENV",
    "QueueAttempt",
    "RunRecord",
    "concurrent_jobs_from_env",
    "source_root",
    "start_run",
]

#: Environment variable with the number of jobs sharing the device (default 1).
CONCURRENT_JOBS_ENV: Final = "ASTRO_SIMCLR_CONCURRENT_JOBS"


def source_root() -> Path | None:
    """The source checkout the package runs from, or ``None`` for an installed wheel.

    ``src/astro_simclr/runs/record.py`` sits three levels below the checkout's
    ``pyproject.toml``; in ``site-packages`` there is none.
    """
    root = Path(__file__).resolve().parents[3]
    return root if (root / "pyproject.toml").is_file() else None


def concurrent_jobs_from_env(environ: Mapping[str, str] | None = None) -> int:
    """``ASTRO_SIMCLR_CONCURRENT_JOBS`` as a positive integer, 1 when unset.

    Raises:
        ConfigError: if it is set to anything but a positive integer.
    """
    raw = (os.environ if environ is None else environ).get(CONCURRENT_JOBS_ENV)
    if raw is None or raw == "":
        return 1
    if not raw.isdigit() or int(raw) < 1:
        raise ConfigError(f"{CONCURRENT_JOBS_ENV}={raw!r}: expected a positive integer")
    return int(raw)


@dataclass(frozen=True, slots=True)
class QueueAttempt:
    """What the queue fixes of a run before it starts (module docstring, "Queued runs").

    ``attempt`` is the attempt the run directory is named after; ``machine_class`` is the
    spec's, recorded in the run id instead of the device's default class; ``run_id``, if
    given, is the run id the queue expanded the cell to: ``start_run`` refuses another one
    before training (hours later would waste the run).
    """

    attempt: int
    machine_class: str
    run_id: str | None = None


@dataclass(slots=True)
class RunRecord:
    """The manifest of a started attempt, kept in step with ``run_dir/manifest.json``."""

    run_dir: Path
    manifest: Manifest
    ended: bool = field(default=False)

    def summary(self) -> dict[str, object]:
        """What ``metrics.json["run"]`` records: run id, attempt and config hash."""
        return {
            "run_id": self.manifest.run_id,
            "attempt": self.manifest.attempt,
            "config_sha256": self.manifest.identity.config_sha256,
        }

    def end(self, status: Status, *, at: str | None = None) -> None:
        """Write the attempt's final status (``finished``, ``failed`` or ``cancelled``).

        A second call is ignored, so an error raised after ``finished`` was written cannot
        turn a finished attempt into a failed one.

        Raises:
            ManifestError: if ``status`` is not terminal.
            OSError: if the manifest cannot be written.
        """
        if self.ended:
            return
        if status not in ("finished", "failed", "cancelled"):
            raise ManifestError(f"a run ends finished, failed or cancelled, not {status}")
        self.manifest = transition(self.manifest, status, at=at)
        write_manifest(self.run_dir, self.manifest)
        self.ended = True


def _pretrained_sha256(settings: Settings, pretrained_file: Path | None) -> str | None:
    if settings.pretrained is None:
        return None
    if pretrained_file is None:
        raise ConfigError("a run with pretrained weights needs the checkpoint file to hash")
    return sha256_file(pretrained_file)


def _existing(run_dir: Path) -> Manifest | None:
    if not (run_dir / MANIFEST_FILE).is_file():
        return None
    try:
        return read_manifest(run_dir)
    except ManifestError as err:
        msg = (
            f"{err}; this output directory holds a manifest that cannot be read: move it "
            "away (or choose another --output-dir) rather than overwrite a run's record"
        )
        raise ConfigError(msg) from None


def start_run(
    command: str,
    settings: Settings,
    *,
    data: Path,
    machine: Mapping[str, object],
    split: SplitRequest | None = None,
    pretrained_file: Path | None = None,
    resumed: bool = False,
    engine: Engine = "lightning",
    root: Path | None = None,
    concurrent_jobs: int | None = None,
    at: str | None = None,
    queued: QueueAttempt | None = None,
) -> RunRecord:
    """Write the ``running`` manifest of a run about to train; return its record.

    ``data`` is the prepared layout, ``machine`` the run's machine record
    (``experiments._machine``: ``device``, ``gpu``, ``processor``, ...), ``split`` the split
    request, ``pretrained_file`` the resolved pretrained checkpoint (its content is hashed),
    ``resumed`` whether the fit resumes from a checkpoint. ``root`` is the source checkout
    for the git state and ``uv.lock`` (default :func:`source_root`); ``concurrent_jobs``
    defaults to :func:`concurrent_jobs_from_env`. The attempt follows the module
    docstring; ``queued`` fixes it and the machine class ("Queued runs").

    Raises:
        ConfigError: if the manifest already in ``settings.output_dir`` cannot be read, or
            ``ASTRO_SIMCLR_CONCURRENT_JOBS`` is invalid; with ``queued``, if the run id is
            not ``queued.run_id`` or the directory holds a manifest of another run,
            attempt, or an ended attempt.
        DataError: if ``data`` is not a readable prepared layout.
        OSError: if a hashed file or the manifest cannot be read or written.
    """
    run_dir = settings.output_dir
    processor = machine.get("processor")
    gpu = machine.get("gpu")
    identity = run_identity(
        command,
        settings,
        data_sha256=data_sha256(data),
        machine_class=queued.machine_class
        if queued is not None
        else default_machine_class(
            str(machine["device"]),
            gpu if isinstance(gpu, str) else None,
            processor if isinstance(processor, str) else "",
        ),
        engine=engine,
        split=split_part(split),
        splits_sha256=None if split is None else sha256_file(split.file),
        pretrained_sha256=_pretrained_sha256(settings, pretrained_file),
    )
    if queued is not None and queued.run_id is not None and queued.run_id != identity.run_id:
        msg = (
            f"this run's id is {identity.run_id}, but the queue expanded the cell to "
            f"{queued.run_id}: the spec expansion and runs.record disagree"
        )
        raise ConfigError(msg)
    jobs = concurrent_jobs_from_env() if concurrent_jobs is None else concurrent_jobs
    previous = _existing(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    unfinished = previous is not None and previous.status in ("queued", "running")
    same = previous is not None and previous.run_id == identity.run_id
    if (
        queued is not None
        and previous is not None
        and not (same and unfinished and previous.attempt == queued.attempt)
    ):
        msg = (
            f"{run_dir} holds run {previous.run_id[:16]} attempt {previous.attempt} "
            f"({previous.status}), not a resumable run {identity.run_id[:16]} attempt "
            f"{queued.attempt}"
        )
        raise ConfigError(msg)
    if previous is not None and same and unfinished and (resumed or queued is not None):
        start = previous if previous.status == "queued" else transition(previous, "queued", at=at)
    else:
        checkout = source_root() if root is None else root
        if queued is not None:
            attempt = queued.attempt
        else:
            attempt = previous.attempt + 1 if same and previous is not None else 1
        start = new_manifest(
            command,
            settings,
            identity,
            git=GitState(None, None) if checkout is None else git_state(checkout),
            uv_lock_sha256=None if checkout is None else uv_lock_sha256(checkout),
            attempt=attempt,
            concurrent_jobs=jobs,
            created_at=at,
        )
    write_manifest(run_dir, start)
    running = transition(start, "running", at=at, machine=machine)
    write_manifest(run_dir, running)
    return RunRecord(run_dir, running)
