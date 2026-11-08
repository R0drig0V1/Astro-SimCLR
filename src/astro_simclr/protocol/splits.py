"""The stored decision splits of the phase-2 protocol, and ``make-splits``.

Plan, task 2.3, "Decision protocol": decisions never use the metric of the stamps that chose
the checkpoint. Two split definitions serve that, both seeded and **stored** (the stored
file, not the generator, is the source of truth: numpy does not promise that a seeded
``Generator`` gives the same permutation in every version):

- **Screen**: Validation split into a *selection* half (checkpoint selection) and a
  *decision* half, ``screen_per_class`` (50) stamps of each class in each half. It can only
  drop a variant (:func:`astro_simclr.protocol.stats.screen_decision`).
- **Cross-validation** (the default for 2.5): stratified ``n_folds``-fold (5) CV on the
  *pool* Train + Validation. Fold ``k`` holds out its *evaluation* part; from the rest it
  carves its own checkpoint-*selection* set (``selection_per_class``, 100 of each class);
  what remains is its *training* part, the only stamps a run may tile (balance by
  repetition) and train on.

Pool indices number Train first, then Validation: Train stamp ``i`` is pool index ``i``,
Validation stamp ``j`` is ``n_train + j`` (:meth:`ProtocolSplits.locate`). The Train stamps
that duplicate a Validation or Test stamp (:mod:`astro_simclr.protocol.leakage`) are in no
fold: they are removed from Train only.

Randomness: each part has its own generator, ``np.random.default_rng([seed, part, ...])``,
independent of the global generators that ``seed_run`` sets (the splits are data
definitions, fixed across runs and seeds; a training seed must not move them). Every draw
works on sorted index arrays, so it does not depend on dictionary or file-system order.

File format (JSON, ``format`` ``astro-simclr-protocol-splits``, version 1)::

    dataset   name, source SHA-256, labels.npy SHA-256 and size of Train and Validation
    seed      the seed of every draw
    leakage   duplicates found and the excluded Train indices
    screen    per_class, selection and decision (sorted Validation indices)
    cv        n_folds, selection_per_class, fold_of (one character per pool index: the
              fold that evaluates it, or "-" for an excluded stamp) and selection (one
              sorted list of pool indices per fold)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.data.prepared import load_prepared, read_meta
from astro_simclr.data.schema import LABEL_NAMES, DataError, StampData
from astro_simclr.protocol.leakage import find_duplicates, train_exclusions

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "CVFold",
    "ProtocolSplits",
    "ScreenSplit",
    "SplitsError",
    "make_splits",
    "read_splits",
    "verify_splits",
    "write_splits",
]

#: ``format`` of a stored splits file.
FORMAT: Final = "astro-simclr-protocol-splits"
#: Bumped on any incompatible change of the file.
FORMAT_VERSION: Final = 1
#: Generator stream of each part (the second entry of the seed sequence).
_SCREEN_STREAM: Final = 0
_FOLD_STREAM: Final = 1
_SELECTION_STREAM: Final = 2
#: ``fold_of`` character of a stamp that is in no fold.
_EXCLUDED: Final = "-"
#: ``fold_of`` stores one digit per stamp, so at most 10 folds.
_MAX_FOLDS: Final = 10
#: The keys of ``dataset`` that identify the data (``name`` is informational: a renamed or
#: moved prepared directory holds the same data).
_IDENTITY_KEYS: Final = ("source_sha256", "labels_sha256", "n")

Indices = npt.NDArray[np.int64]


class SplitsError(DataError):
    """A splits file is malformed, or does not belong to the given prepared dataset."""


@dataclass(frozen=True, slots=True)
class ScreenSplit:
    """The 250/250 screen of Validation: two sorted, disjoint arrays of Validation indices."""

    selection: Indices
    decision: Indices


@dataclass(frozen=True, slots=True)
class CVFold:
    """One cross-validation fold, as sorted arrays of pool indices.

    ``evaluation`` is held out (the decision metric), ``selection`` chooses the checkpoint,
    ``training`` is the rest of the pool (the only part a run trains on and tiles). The
    three are disjoint and, with the excluded stamps, cover the pool.
    """

    fold: int
    evaluation: Indices
    selection: Indices
    training: Indices


@dataclass(frozen=True, slots=True)
class ProtocolSplits:
    """The stored splits of one prepared dataset (module docstring)."""

    dataset: dict[str, Any]
    seed: int
    duplicates: tuple[dict[str, object], ...]
    excluded_train: tuple[int, ...]
    screen_per_class: int
    screen: ScreenSplit
    selection_per_class: int
    folds: tuple[CVFold, ...]

    @property
    def n_train(self) -> int:
        """Number of Train stamps of the dataset (excluded ones included)."""
        return int(self.dataset["n"]["Train"])

    @property
    def n_validation(self) -> int:
        """Number of Validation stamps of the dataset."""
        return int(self.dataset["n"]["Validation"])

    def locate(self, pool: Indices) -> tuple[Indices, Indices]:
        """Split pool indices into ``(Train indices, Validation indices)``, each in order.

        This is what a data module needs to read a fold's stamps from the prepared layout.

        Raises:
            ValueError: if an index is outside the pool.
        """
        pool = np.asarray(pool, dtype=np.int64)
        if pool.size and (pool.min() < 0 or pool.max() >= self.n_train + self.n_validation):
            msg = f"pool indices must be in range({self.n_train + self.n_validation})"
            raise ValueError(msg)
        in_train = pool < self.n_train
        return pool[in_train], pool[~in_train] - self.n_train


def _per_class_draw(
    rng: np.random.Generator, candidates: Indices, labels: npt.NDArray[np.int64], n: int
) -> list[Indices]:
    """For each class, ``n`` indices of ``candidates`` with that label, without replacement.

    Raises:
        SplitsError: if a class has fewer than ``n`` candidates.
    """
    drawn = []
    for label, name in enumerate(LABEL_NAMES):
        pool = candidates[labels[candidates] == label]
        if pool.size < n:
            msg = f"class {name} has {pool.size} candidate stamps, fewer than {n}"
            raise SplitsError(msg)
        drawn.append(rng.choice(pool, size=n, replace=False))
    return drawn


def _make_screen(labels: npt.NDArray[np.int64], seed: int, per_class: int) -> ScreenSplit:
    rng = np.random.default_rng([seed, _SCREEN_STREAM])
    selection, decision = [], []
    for part in _per_class_draw(rng, np.arange(labels.size, dtype=np.int64), labels, 2 * per_class):
        selection.append(part[:per_class])
        decision.append(part[per_class:])
    return ScreenSplit(np.sort(np.concatenate(selection)), np.sort(np.concatenate(decision)))


def _assign_folds(
    labels: npt.NDArray[np.int64], usable: npt.NDArray[np.bool_], seed: int, n_folds: int
) -> npt.NDArray[np.int64]:
    """Per pool index, its evaluation fold (stratified), or -1 if not usable."""
    rng = np.random.default_rng([seed, _FOLD_STREAM])
    fold_of = np.full(labels.size, -1, dtype=np.int64)
    for label, name in enumerate(LABEL_NAMES):
        members = np.flatnonzero(usable & (labels == label))
        if members.size < n_folds:
            msg = f"class {name} has {members.size} stamps, fewer than {n_folds} folds"
            raise SplitsError(msg)
        for fold, chunk in enumerate(np.array_split(rng.permutation(members), n_folds)):
            fold_of[chunk] = fold
    return fold_of


def _build_fold(fold: int, fold_of: npt.NDArray[np.int64], selection: Indices) -> CVFold:
    rest = np.flatnonzero((fold_of >= 0) & (fold_of != fold)).astype(np.int64)
    selection = np.sort(selection.astype(np.int64))
    return CVFold(
        fold,
        np.flatnonzero(fold_of == fold).astype(np.int64),
        selection,
        np.setdiff1d(rest, selection, assume_unique=True).astype(np.int64),
    )


def _labels(data: StampData, path: Path) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The Train and Validation labels of a classifier file.

    Raises:
        DataError: if ``data`` is not a classifier file (an unlabelled split).
    """
    train, val = data.splits["Train"].labels, data.splits["Validation"].labels
    if data.kind != "classifier" or train is None or val is None:
        raise DataError(f"{path}: the protocol splits need a classifier file, got {data.kind}")
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def _dataset_identity(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    splits = meta["splits"]
    return {
        "name": path.name,
        "source_sha256": meta["source"]["sha256"],
        "labels_sha256": {
            split: splits[split]["files"]["labels.npy"]["sha256"]
            for split in ("Train", "Validation")
        },
        "n": {split: int(splits[split]["n"]) for split in ("Train", "Validation")},
    }


def make_splits(
    path: Path,
    *,
    seed: int = 0,
    n_folds: int = 5,
    selection_per_class: int = 100,
    screen_per_class: int = 50,
) -> ProtocolSplits:
    """Build the screen and the CV folds of the prepared classifier dataset at ``path``.

    Finds the cross-split duplicates first and leaves the duplicated Train stamps out of
    every fold. Deterministic for a given dataset, seed and numpy version.

    Raises:
        DataError: if ``path`` is not a valid prepared classifier layout.
        SplitsError: if a class is too small for the requested sizes, ``n_folds`` is not in
            2-10 (``fold_of`` stores one digit per stamp), or a size is not positive.
    """
    if not 2 <= n_folds <= _MAX_FOLDS:
        raise SplitsError(f"n_folds must be in 2-{_MAX_FOLDS}, got {n_folds}")
    if selection_per_class < 1 or screen_per_class < 1:
        raise SplitsError("selection_per_class and screen_per_class must be >= 1")
    meta = read_meta(path)
    data = load_prepared(path)
    duplicates = find_duplicates(data)
    excluded = train_exclusions(duplicates)
    train_labels, val_labels = _labels(data, path)
    screen = _make_screen(val_labels, seed, screen_per_class)

    labels = np.concatenate([train_labels, val_labels])
    usable = np.ones(labels.size, dtype=bool)
    usable[excluded] = False
    fold_of = _assign_folds(labels, usable, seed, n_folds)
    folds = []
    for fold in range(n_folds):
        rng = np.random.default_rng([seed, _SELECTION_STREAM, fold])
        rest = np.flatnonzero((fold_of >= 0) & (fold_of != fold)).astype(np.int64)
        # The training part must keep at least one stamp of each class.
        for label, name in enumerate(LABEL_NAMES):
            if np.count_nonzero(labels[rest] == label) <= selection_per_class:
                msg = (
                    f"fold {fold}: class {name} has too few stamps outside the fold for a "
                    f"selection set of {selection_per_class}"
                )
                raise SplitsError(msg)
        selection = np.concatenate(_per_class_draw(rng, rest, labels, selection_per_class))
        folds.append(_build_fold(fold, fold_of, selection))
    return ProtocolSplits(
        dataset=_dataset_identity(path, meta),
        seed=seed,
        duplicates=tuple(dup.as_dict() for dup in duplicates),
        excluded_train=tuple(excluded),
        screen_per_class=screen_per_class,
        screen=screen,
        selection_per_class=selection_per_class,
        folds=tuple(folds),
    )


def _fold_of(splits: ProtocolSplits) -> str:
    chars = [_EXCLUDED] * (splits.n_train + splits.n_validation)
    for fold in splits.folds:
        for index in fold.evaluation:
            chars[index] = str(fold.fold)
    return "".join(chars)


def _to_plain(splits: ProtocolSplits) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "dataset": splits.dataset,
        "seed": splits.seed,
        "pool": "Train indices 0..n_train-1, then Validation index j as n_train + j",
        "leakage": {
            "duplicates": list(splits.duplicates),
            "excluded_train": list(splits.excluded_train),
        },
        "screen": {
            "per_class": splits.screen_per_class,
            "selection": splits.screen.selection.tolist(),
            "decision": splits.screen.decision.tolist(),
        },
        "cv": {
            "n_folds": len(splits.folds),
            "selection_per_class": splits.selection_per_class,
            "fold_of": _fold_of(splits),
            "selection": [fold.selection.tolist() for fold in splits.folds],
        },
    }


def write_splits(splits: ProtocolSplits, path: Path) -> None:
    """Write ``splits`` as JSON to ``path`` (index lists on one line each; deterministic).

    Raises:
        OSError: if the file cannot be written.
    """
    text = json.dumps(_to_plain(splits), indent=2)
    # Keep each list of integers on one line: thousands of one-number lines help nobody.
    text = re.sub(
        r"\[\s*(-?\d+(?:,\s*-?\d+)*)\s*\]",
        lambda match: "[" + ", ".join(match.group(1).replace(",", " ").split()) + "]",
        text,
    )
    path.write_text(text + "\n", encoding="utf-8")


def _positive_int(value: object, what: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SplitsError(f"{path}: {what} must be a positive integer")
    return value


def _int_array(value: object, what: str, path: Path) -> Indices:
    if not isinstance(value, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise SplitsError(f"{path}: {what} must be a list of integers")
    array = np.asarray(value, dtype=np.int64)
    if array.size != np.unique(array).size or not np.array_equal(array, np.sort(array)):
        raise SplitsError(f"{path}: {what} must be sorted and without repeats")
    return array


def _parse_screen(raw: dict[str, Any], n_validation: int, path: Path) -> tuple[int, ScreenSplit]:
    per_class = _positive_int(raw.get("per_class"), "screen.per_class", path)
    selection = _int_array(raw.get("selection"), "screen.selection", path)
    decision = _int_array(raw.get("decision"), "screen.decision", path)
    expected = len(LABEL_NAMES) * per_class
    if selection.size != expected or decision.size != expected:
        raise SplitsError(f"{path}: the screen halves must hold per_class stamps of each class")
    if np.intersect1d(selection, decision).size:
        raise SplitsError(f"{path}: the screen halves overlap")
    if min(selection[0], decision[0]) < 0 or max(selection[-1], decision[-1]) >= n_validation:
        raise SplitsError(f"{path}: screen indices must be in range({n_validation})")
    return per_class, ScreenSplit(selection, decision)


def _parse_folds(
    raw: dict[str, Any], n_pool: int, excluded: tuple[int, ...], path: Path
) -> tuple[int, tuple[CVFold, ...]]:
    n_folds = _positive_int(raw.get("n_folds"), "cv.n_folds", path)
    per_class = _positive_int(raw.get("selection_per_class"), "cv.selection_per_class", path)
    fold_text, selections = raw.get("fold_of"), raw.get("selection")
    if not 2 <= n_folds <= _MAX_FOLDS:
        raise SplitsError(f"{path}: cv.n_folds must be in 2-{_MAX_FOLDS}")
    if (
        not isinstance(fold_text, str)
        or not isinstance(selections, list)
        or len(selections) != n_folds
    ):
        raise SplitsError(f"{path}: cv needs n_folds, selection_per_class, fold_of, selection")
    if len(fold_text) != n_pool or set(fold_text) - {_EXCLUDED, *map(str, range(n_folds))}:
        raise SplitsError(f"{path}: cv.fold_of must have one fold digit or '-' per pool stamp")
    fold_of = np.asarray([-1 if c == _EXCLUDED else int(c) for c in fold_text], dtype=np.int64)
    if sorted(np.flatnonzero(fold_of < 0).tolist()) != list(excluded):
        raise SplitsError(f"{path}: the stamps outside every fold must be the excluded ones")
    folds = []
    for fold, raw_selection in enumerate(selections):
        selection = _int_array(raw_selection, f"cv.selection[{fold}]", path)
        rest = np.flatnonzero((fold_of >= 0) & (fold_of != fold))
        if selection.size != len(LABEL_NAMES) * per_class or np.setdiff1d(selection, rest).size:
            raise SplitsError(
                f"{path}: cv.selection[{fold}] must hold {per_class} stamps per class, all "
                "outside the fold"
            )
        folds.append(_build_fold(fold, fold_of, selection))
    return per_class, tuple(folds)


def read_splits(path: Path) -> ProtocolSplits:
    """Read and check a splits file written by :func:`write_splits`.

    Checks the structure only (sizes, disjointness, ranges); :func:`verify_splits` checks
    that it belongs to a given prepared dataset, the class balance, the stratification and
    the excluded stamps.

    Raises:
        SplitsError: if the file cannot be read or is malformed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise SplitsError(f"{path}: cannot read the splits file ({err})") from None
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise SplitsError(f"{path}: not a {FORMAT} file")
    if raw.get("format_version") != FORMAT_VERSION:
        raise SplitsError(f"{path}: unsupported format_version {raw.get('format_version')!r}")
    try:
        dataset = raw["dataset"]
        n_train, n_validation = int(dataset["n"]["Train"]), int(dataset["n"]["Validation"])
        seed = int(raw["seed"])
        leakage = raw["leakage"]
        excluded = tuple(_int_array(leakage["excluded_train"], "excluded_train", path).tolist())
        duplicates = tuple(dict(d) for d in leakage["duplicates"])
        screen_raw, cv_raw = raw["screen"], raw["cv"]
    except (KeyError, TypeError, ValueError) as err:
        raise SplitsError(f"{path}: malformed splits file ({err!r})") from None
    if excluded and excluded[-1] >= n_train:
        raise SplitsError(f"{path}: excluded_train must be Train indices")
    screen_per_class, screen = _parse_screen(screen_raw, n_validation, path)
    selection_per_class, folds = _parse_folds(cv_raw, n_train + n_validation, excluded, path)
    return ProtocolSplits(
        dataset=dataset,
        seed=seed,
        duplicates=duplicates,
        excluded_train=excluded,
        screen_per_class=screen_per_class,
        screen=screen,
        selection_per_class=selection_per_class,
        folds=folds,
    )


def verify_splits(splits: ProtocolSplits, path: Path) -> None:
    """Check that ``splits`` belongs to the prepared dataset at ``path`` and is well drawn.

    - The source SHA-256, the SHA-256 of Train's and Validation's ``labels.npy`` (from
      ``meta.json``) and the split sizes must match the stored ones (the directory name
      may differ: a renamed or moved copy holds the same data).
    - With the real labels: every screen half holds ``screen_per_class`` stamps of each
      class, every fold's selection set ``selection_per_class``, and the folds are
      stratified (per class, the evaluation counts of two folds differ by at most 1).
    - The excluded Train stamps are exactly the ones the leakage check finds again
      (:func:`~astro_simclr.protocol.leakage.find_duplicates`; reads every stamp, a few
      seconds on ``red``).

    Raises:
        SplitsError: on a different dataset, an unbalanced or unstratified split, or other
            excluded stamps.
        DataError: if ``path`` is not a valid prepared layout.
    """
    expected = _dataset_identity(path, read_meta(path))
    if any(expected[key] != splits.dataset.get(key) for key in _IDENTITY_KEYS):
        msg = (
            f"the splits were made for {splits.dataset.get('name')!r} "
            f"(source {splits.dataset.get('source_sha256')}), not for {path} "
            f"(source {expected['source_sha256']}); regenerate them with `astro-simclr "
            "make-splits`"
        )
        raise SplitsError(msg)
    data = load_prepared(path)
    train_labels, val_labels = _labels(data, path)
    labels = np.concatenate([train_labels, val_labels])
    n_classes = len(LABEL_NAMES)
    per_fold = np.array(
        [np.bincount(labels[fold.evaluation], minlength=n_classes) for fold in splits.folds]
    )
    if np.any(per_fold.max(axis=0) - per_fold.min(axis=0) > 1):
        raise SplitsError(f"the folds are not stratified on {path}: counts {per_fold.tolist()}")
    excluded = train_exclusions(find_duplicates(data))
    if tuple(excluded) != splits.excluded_train:
        raise SplitsError(
            f"excluded_train {list(splits.excluded_train)} is not what the leakage check "
            f"finds on {path} ({excluded})"
        )
    checks = [
        ("screen.selection", val_labels[splits.screen.selection], splits.screen_per_class),
        ("screen.decision", val_labels[splits.screen.decision], splits.screen_per_class),
        *(
            (f"cv.selection[{fold.fold}]", labels[fold.selection], splits.selection_per_class)
            for fold in splits.folds
        ),
    ]
    for name, part, per_class in checks:
        counts = np.bincount(part, minlength=n_classes).tolist()
        if counts != [per_class] * n_classes:
            raise SplitsError(f"{name} is not balanced on {path}: class counts {counts}")
