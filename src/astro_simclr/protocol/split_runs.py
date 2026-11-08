"""Runs on a stored decision split: ``train-supervised --splits FILE --fold K | --screen``.

Task 2.5-infra of ``docs/phases/02-stamp-classifier.md``. Every 2.5 row is decided on the
stored splits of task 2.3 (:mod:`astro_simclr.protocol.splits`), never on test. A *split
run* trains one model on one part of them and predicts the part that the decision reads:

==========  =========================  =============================  =====================
mode        trains on (tiled)          selects the checkpoint on      predicts (decision)
==========  =========================  =============================  =====================
``fold``    fold K's training part     fold K's selection set         fold K's evaluation
                                       (100 per class)                part (pool index)
``screen``  Train minus excluded       the screen's selection half    the screen's decision
            (and, for a version-2      (50 per class of Validation)   half (Validation
            file, minus the Train                                     index)
            alerts grouped with
            Validation)
==========  =========================  =============================  =====================

A version-2 (grouped) file needs nothing else: its folds' training parts already leave out
the unused members of the selection groups (:class:`~astro_simclr.protocol.splits.CVFold`),
and :meth:`~astro_simclr.protocol.splits.ProtocolSplits.screen_training_excluded` gives the
screen's Train exclusions. ``describe()`` adds the grouping (``None`` for version 1).

Test is never loaded. The predictions' ``index`` follows ADR 0004 and what the decision
helpers of :mod:`astro_simclr.protocol.compare` check: the pool index for a fold (Train
``i`` is ``i``, Validation ``j`` is ``n_train + j``), the Validation index for the screen.

:func:`open_split_run` reads the splits file and **verifies it against the prepared data**
(:func:`~astro_simclr.protocol.splits.verify_splits`: the dataset fingerprint, the class
balance, the stratification and a fresh leakage check), so a run never trains on splits
made for other data. :meth:`SplitRun.parts` gives every part as sorted pool indices and
checks that they are disjoint (no stamp is both trained on and evaluated). The data module
gathers the stamps (``training.device_data.protocol_splits``) and ``experiments`` records
:meth:`SplitRun.describe` as ``metrics.json["protocol"]``, with the SHA-256 of the splits
file, which ``compare-variants --splits`` checks. :meth:`SplitRun.identity` goes into every
checkpoint, so ``--resume`` refuses a checkpoint of another fold, mode or splits file.

Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.config import ConfigError
from astro_simclr.protocol.leakage import train_keep_indices
from astro_simclr.protocol.splits import ProtocolSplits, read_splits, verify_splits

__all__ = [
    "FOLD",
    "SCREEN",
    "Mode",
    "RunParts",
    "SplitRequest",
    "SplitRun",
    "file_sha256",
    "open_split_run",
]

Mode = Literal["fold", "screen"]
#: ``metrics.json["protocol"]["mode"]`` of a fold run.
FOLD: Final = "fold"
#: ``metrics.json["protocol"]["mode"]`` of a screen run.
SCREEN: Final = "screen"

Indices = npt.NDArray[np.int64]


def file_sha256(path: Path) -> str:
    """The SHA-256 (hex) of the file at ``path``.

    Raises:
        OSError: if the file cannot be read.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitRequest:
    """What the command line asked for: a splits file and a fold, or the screen (``None``)."""

    file: Path
    fold: int | None

    @property
    def mode(self) -> Mode:
        """``fold`` or ``screen``."""
        return SCREEN if self.fold is None else FOLD


@dataclass(frozen=True, slots=True)
class RunParts:
    """The parts of one split run, as sorted pool indices (module docstring).

    ``evaluation_index`` is what the predictions file stores for the evaluation part: the
    pool indices for a fold, the Validation indices for the screen (in the same order as
    ``evaluation``). ``evaluation_name`` names the part in ``metrics.json``.
    """

    training: Indices
    selection: Indices
    evaluation: Indices
    evaluation_index: Indices
    evaluation_name: str


@dataclass(frozen=True, slots=True)
class SplitRun:
    """A verified splits file and the part of it one run uses."""

    file: Path
    sha256: str
    splits: ProtocolSplits
    fold: int | None

    @property
    def mode(self) -> Mode:
        """``fold`` or ``screen``."""
        return SCREEN if self.fold is None else FOLD

    def parts(self) -> RunParts:
        """The training, selection and evaluation parts (pool indices), checked disjoint.

        Raises:
            ValueError: if two parts share a stamp, or a part holds an excluded Train
                stamp (a malformed splits object; a file from :func:`read_splits` never
                does).
        """
        splits = self.splits
        n_train = splits.n_train
        if self.fold is None:
            # Grouped splits: also without the Train alerts grouped with Validation.
            training = train_keep_indices(n_train, splits.screen_training_excluded())
            parts = RunParts(
                training=training,
                selection=n_train + splits.screen.selection,
                evaluation=n_train + splits.screen.decision,
                evaluation_index=splits.screen.decision,
                evaluation_name="screen decision half (Validation)",
            )
        else:
            fold = splits.folds[self.fold]
            parts = RunParts(
                training=fold.training,
                selection=fold.selection,
                evaluation=fold.evaluation,
                evaluation_index=fold.evaluation,
                evaluation_name=f"fold {self.fold} evaluation part (pool)",
            )
        named = {
            "training": parts.training,
            "selection": parts.selection,
            "evaluation": parts.evaluation,
        }
        excluded = np.asarray(splits.excluded_train, dtype=np.int64)
        for name, part in named.items():
            if np.intersect1d(part, excluded).size:
                raise ValueError(f"the {name} part holds an excluded Train stamp")
        for first, second in combinations(named, 2):
            if np.intersect1d(named[first], named[second]).size:
                raise ValueError(f"the {first} and {second} parts share stamps")
        return parts

    def identity(self) -> dict[str, Any]:
        """What makes two split runs the same run: the splits SHA-256, the mode, the fold.

        Stored in every checkpoint of the run (the training loader's state,
        ``training.device_data``) and compared on ``--resume``: a resume from another fold
        would continue from weights trained on this fold's evaluation part. The file path
        is left out, so a moved splits file still resumes.
        """
        return {"splits_sha256": self.sha256, "mode": self.mode, "fold": self.fold}

    def describe(self) -> dict[str, Any]:
        """The ``metrics.json["protocol"]`` entry of the run (part sizes included)."""
        parts = self.parts()
        return {
            "mode": self.mode,
            "fold": self.fold,
            "splits_file": str(self.file),
            "splits_sha256": self.sha256,
            "dataset": self.splits.dataset.get("name"),
            "evaluation_part": parts.evaluation_name,
            "evaluation_index": "validation" if self.fold is None else "pool",
            "n_training": int(parts.training.size),
            "n_selection": int(parts.selection.size),
            "n_evaluation": int(parts.evaluation.size),
            "excluded_train": list(self.splits.excluded_train),
            "grouping": self._grouping(),
        }

    def _grouping(self) -> dict[str, Any] | None:
        """The grouping of a version-2 file (method, radius, groups), ``None`` for version 1."""
        grouping = self.splits.grouping
        if grouping is None:
            return None
        return {
            "method": grouping.method,
            "radius_arcsec": grouping.radius_arcsec,
            "n_groups": grouping.n_groups,
            "n_screen_excluded_train": len(self.splits.screen_excluded_train),
        }


def open_split_run(request: SplitRequest, prepared: Path) -> SplitRun:
    """Read ``request.file``, verify it against the prepared layout ``prepared``, pick the part.

    The verification is :func:`~astro_simclr.protocol.splits.verify_splits` (a few seconds
    on ``red``: it re-runs the leakage check on every stamp).

    Raises:
        ConfigError: if ``request.fold`` is not a fold of the file.
        SplitsError: if the file is unreadable or malformed, or was made for another
            dataset (its fingerprint: source and label SHA-256, split sizes) or does not
            verify on it.
        DataError: if ``prepared`` is not a valid prepared layout.
    """
    splits = read_splits(request.file)
    n_folds = len(splits.folds)
    if request.fold is not None and not 0 <= request.fold < n_folds:
        msg = f"--fold {request.fold}: {request.file} has folds 0-{n_folds - 1}"
        raise ConfigError(msg)
    verify_splits(splits, prepared)
    return SplitRun(request.file, file_sha256(request.file), splits, request.fold)
