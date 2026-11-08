"""The run-directory layout and the retention policy (task A0; ADR 0012).

One attempt of one run is one directory, named :func:`run_dir_name` (the run id's first
16 hex digits and the attempt: ``3f2a9c0d1e4b5a67-a1``) under a root the caller chooses
(the queue's, A3; today ``--output-dir``). Its files::

    manifest.json          the manifest (runs.manifest), written first, updated on each status
    metrics.json           the results (experiments.run_command)
    scalars.jsonl          local scalars, one JSON object per line (the plain engine)
    tb/                    TensorBoard event files (the plain engine)
    predictions_*.npz      per-stamp probabilities (protocol.predictions)
    ckpt/last.pt           the latest state, what a resume continues from (the plain engine)
    ckpt/best.pt           the selected weights (the plain engine)

A Lightning run (``engine: lightning``, until task 4.8) keeps today's names for the same
roles: ``checkpoints/last.ckpt``, ``checkpoints/best.ckpt`` and ``lightning_logs/``.
:func:`classify_artefact` knows both.

**Retention** (:func:`retention`): what to keep of an attempt, as a pure function of its
manifest and its files. Nothing deletes files yet; the registry (A2) and the runners (A3)
apply it.

- An attempt that is not ``finished`` keeps everything: a queued or running one resumes
  from ``last``; a failed or cancelled one is kept whole for diagnosis.
- A finished attempt drops ``last`` (nothing resumes a finished run) and keeps ``best``
  only for pretraining runs (``pretrain``, whose encoder ``linear-eval`` and ``fine-tune``
  read); a classifier's ``best`` is dropped, its predictions and metrics are the result.
- The manifest, metrics, scalars, TensorBoard logs, predictions and figures are kept
  forever, and so is every file the policy does not recognise: an unknown file is never
  deleted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final, Literal

from astro_simclr.runs.identity import check_sha256
from astro_simclr.runs.manifest import MANIFEST_FILE, Manifest, is_finished

__all__ = [
    "BEST_CHECKPOINT",
    "CHECKPOINT_DIR",
    "LAST_CHECKPOINT",
    "LIGHTNING_BEST",
    "LIGHTNING_LAST",
    "METRICS_FILE",
    "PRETRAINING_COMMANDS",
    "SCALARS_FILE",
    "TENSORBOARD_DIR",
    "ArtefactKind",
    "RetentionDecision",
    "classify_artefact",
    "retention",
    "run_dir_name",
]

#: ``metrics.json``: the results of a run.
METRICS_FILE: Final = "metrics.json"
#: ``scalars.jsonl``: local scalars of the plain engine, one JSON object per line.
SCALARS_FILE: Final = "scalars.jsonl"
#: ``tb/``: TensorBoard event files of the plain engine.
TENSORBOARD_DIR: Final = "tb"
#: ``ckpt/``: the plain engine's checkpoints.
CHECKPOINT_DIR: Final = "ckpt"
#: ``ckpt/last.pt``: the latest state, what a resume continues from.
LAST_CHECKPOINT: Final = f"{CHECKPOINT_DIR}/last.pt"
#: ``ckpt/best.pt``: the selected weights.
BEST_CHECKPOINT: Final = f"{CHECKPOINT_DIR}/best.pt"
#: The Lightning engine's ``last`` (``experiments.CHECKPOINT_DIR`` / ``LAST_CHECKPOINT``).
LIGHTNING_LAST: Final = "checkpoints/last.ckpt"
#: The Lightning engine's ``best``.
LIGHTNING_BEST: Final = "checkpoints/best.ckpt"
#: The commands whose ``best`` checkpoint is a product (an encoder other runs read).
PRETRAINING_COMMANDS: Final = frozenset({"pretrain"})

#: Hex digits of the run id in a run directory's name (64 bits: no collision in practice;
#: the manifest holds the full id).
_DIR_ID_DIGITS: Final = 16


class ArtefactKind(StrEnum):
    """The role of a file in a run directory (module docstring)."""

    MANIFEST = "manifest"
    METRICS = "metrics"
    SCALARS = "scalars"
    TENSORBOARD = "tensorboard"
    PREDICTIONS = "predictions"
    FIGURE = "figure"
    LAST_CHECKPOINT = "last_checkpoint"
    BEST_CHECKPOINT = "best_checkpoint"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class RetentionDecision:
    """Keep or delete one file, with the kind it was classed as and the rule's reason."""

    action: Literal["keep", "delete"]
    kind: ArtefactKind
    reason: str


def run_dir_name(run_id: str, attempt: int) -> str:
    """The directory name of one attempt: the run id's first 16 hex digits, ``-a<attempt>``.

    Raises:
        IdentityError: if ``run_id`` is not a hex SHA-256.
        ValueError: if ``attempt`` is not a positive integer.
    """
    check_sha256(run_id, "run_id")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f"attempt must be an integer >= 1, got {attempt!r}")
    return f"{run_id[:_DIR_ID_DIGITS]}-a{attempt}"


_EXACT: Final[Mapping[str, ArtefactKind]] = {
    MANIFEST_FILE: ArtefactKind.MANIFEST,
    METRICS_FILE: ArtefactKind.METRICS,
    SCALARS_FILE: ArtefactKind.SCALARS,
    LAST_CHECKPOINT: ArtefactKind.LAST_CHECKPOINT,
    LIGHTNING_LAST: ArtefactKind.LAST_CHECKPOINT,
    BEST_CHECKPOINT: ArtefactKind.BEST_CHECKPOINT,
    LIGHTNING_BEST: ArtefactKind.BEST_CHECKPOINT,
}
_LOG_DIRS: Final = frozenset({TENSORBOARD_DIR, "lightning_logs"})


def classify_artefact(path: str) -> ArtefactKind:
    """The kind of the file at ``path``, relative to the run directory (POSIX separators).

    ``predictions_*.npz`` at the top level are predictions, ``*.png`` there are figures
    (the confusion matrices), anything under ``tb/`` or ``lightning_logs/`` is a log; any
    other path, including a checkpoint under another name, is :attr:`ArtefactKind.OTHER`.
    """
    parts = PurePosixPath(path).parts
    name = PurePosixPath(path).name
    if path in _EXACT:
        return _EXACT[path]
    if len(parts) > 1 and parts[0] in _LOG_DIRS:
        return ArtefactKind.TENSORBOARD
    if len(parts) == 1 and name.startswith("predictions_") and name.endswith(".npz"):
        return ArtefactKind.PREDICTIONS
    if len(parts) == 1 and name.endswith(".png"):
        return ArtefactKind.FIGURE
    return ArtefactKind.OTHER


def _decide(kind: ArtefactKind, finished: bool, pretraining: bool) -> RetentionDecision:
    if not finished:
        return RetentionDecision("keep", kind, "the attempt is not finished: everything stays")
    if kind is ArtefactKind.LAST_CHECKPOINT:
        return RetentionDecision("delete", kind, "a finished run is never resumed")
    if kind is ArtefactKind.BEST_CHECKPOINT:
        if pretraining:
            return RetentionDecision("keep", kind, "a pretrained encoder other runs read")
        return RetentionDecision("delete", kind, "a classifier's result is its predictions")
    if kind is ArtefactKind.OTHER:
        return RetentionDecision("keep", kind, "an unknown file is never deleted")
    return RetentionDecision("keep", kind, "records and results are kept forever")


def retention(run: Manifest, artefacts: Iterable[str]) -> dict[str, RetentionDecision]:
    """Keep or delete each of ``artefacts`` (paths relative to ``run``'s directory).

    A pure function (module docstring, "Retention"): it reads only ``run.status`` and
    ``run.command`` and the paths, and deletes nothing.
    """
    finished = is_finished(run.status)
    pretraining = run.command in PRETRAINING_COMMANDS
    return {path: _decide(classify_artefact(path), finished, pretraining) for path in artefacts}
