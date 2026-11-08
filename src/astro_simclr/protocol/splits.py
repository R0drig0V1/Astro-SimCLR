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

**Grouping (format version 2).** Distinct alerts of one object (or one artefact) leak just
like a duplicated stamp (``docs/evaluation-protocol.md``, "Leakage"). Version 2 groups the
pool (:mod:`astro_simclr.protocol.groups`: by sky position within 1.5", or by an external
id) and keeps every group whole:

- **CV**: a class-stratified *group* k-fold. Groups are placed largest first (ties in
  seeded random order) into the fold where they least raise the per-class fill
  (``sum_c n_gc * count_fc / total_c``, lowest fold on a tie); single-class groups thus go
  to the fold with the fewest stamps of their class, and the singletons, placed last,
  level the folds. Per class, two folds differ by at most the largest number of that
  class's stamps in one group (1 without multi-stamp groups, as in version 1).
- **CV selection sets**: groups outside the evaluation fold, in seeded random order, are
  taken while a class of theirs still needs stamps; a taken (*selection*) group gives its
  members of the needed classes, at random up to the quota, and its other members are
  *unused* in that fold: neither selection nor training. So every selection set holds
  exactly ``selection_per_class`` of each class, and no group spans the selection set and
  the training part.
- **Screen**: Validation's groups are placed largest first (ties in seeded random order)
  into the half with room for all their stamps (the less filled one if both have room).
  A group that fits neither half is left out of the screen. If that leaves a class short
  of ``screen_per_class`` in a half (the *fallback*, never needed on ``red``), both halves
  are trimmed at random to the largest common count, which is stored as the screen's
  ``per_class`` and explained in ``grouping.screen_fallback``. The screen's runs train on
  Train minus ``excluded_train`` minus ``screen_excluded_train``: every Train alert whose
  group has a Validation member, as the leaked stamp leaves Train in version 1.

Randomness: each part has its own generator, ``np.random.default_rng([seed, part, ...])``,
independent of the global generators that ``seed_run`` sets (the splits are data
definitions, fixed across runs and seeds; a training seed must not move them). Every draw
works on sorted index arrays, so it does not depend on dictionary or file-system order.

File format (JSON, ``format`` ``astro-simclr-protocol-splits``, version 2; version 1 files
have no ``grouping`` and are still read)::

    dataset   name, source SHA-256, labels.npy SHA-256 and size of Train and Validation
    seed      the seed of every draw
    leakage   duplicates found and the excluded Train indices
    grouping  method (position or external), radius_arcsec, source (external ids: file
              name and SHA-256), n_groups over the pool, groups (each group of two or more
              pool indices, sorted), screen_excluded_train, screen_fallback (null or why)
    screen    per_class, selection and decision (sorted Validation indices)
    cv        n_folds, selection_per_class, fold_of (one character per pool index: the
              fold that evaluates it, or "-" for an excluded stamp) and selection (one
              sorted list of pool indices per fold)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.data.prepared import load_prepared, read_meta
from astro_simclr.data.schema import LABEL_NAMES, DataError, StampData
from astro_simclr.protocol.groups import (
    DEFAULT_RADIUS_ARCSEC,
    EXTERNAL,
    POSITION,
    Grouping,
    GroupingMethod,
    canonical_groups,
    grouping_for,
    load_group_ids,
)
from astro_simclr.protocol.leakage import find_duplicates, train_exclusions

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "POOL_SPLITS",
    "CVFold",
    "ProtocolSplits",
    "ScreenSplit",
    "SplitsError",
    "check_group_integrity",
    "make_splits",
    "read_splits",
    "verify_splits",
    "write_splits",
]

#: ``format`` of a stored splits file.
FORMAT: Final = "astro-simclr-protocol-splits"
#: Bumped on any incompatible change of the file (2: grouped splits).
FORMAT_VERSION: Final = 2
#: The versions :func:`read_splits` accepts.
_READABLE_VERSIONS: Final = (1, 2)
#: The splits of the pool, in pool order.
POOL_SPLITS: Final = ("Train", "Validation")
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
Labels = npt.NDArray[np.int64]


class SplitsError(DataError):
    """A splits file is malformed, or does not belong to the given prepared dataset."""


@dataclass(frozen=True, slots=True)
class ScreenSplit:
    """The 250/250 screen of Validation: two sorted, disjoint arrays of Validation indices."""

    selection: Indices
    decision: Indices


def _no_indices() -> Indices:
    return np.empty(0, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CVFold:
    """One cross-validation fold, as sorted arrays of pool indices.

    ``evaluation`` is held out (the decision metric), ``selection`` chooses the checkpoint,
    ``training`` is the rest of the pool (the only part a run trains on and tiles).
    ``unused`` (grouped splits only) are the members of the selection groups that are not
    in the selection set: they leave the training part so that no group spans the two.
    The four are disjoint and, with the excluded stamps, cover the pool.
    """

    fold: int
    evaluation: Indices
    selection: Indices
    training: Indices
    unused: Indices = field(default_factory=_no_indices)


@dataclass(frozen=True, slots=True)
class ProtocolSplits:
    """The stored splits of one prepared dataset (module docstring).

    ``grouping`` is the pool grouping of a version-2 file (``None`` for version 1, whose
    groups are taken as singletons); ``grouping_source`` the external id file's name and
    SHA-256 (``None`` for a position grouping); ``screen_excluded_train`` the Train
    indices whose group has a Validation member; ``screen_fallback`` why the screen holds
    fewer than the requested stamps per class (``None`` when it holds them all).
    """

    dataset: dict[str, Any]
    seed: int
    duplicates: tuple[dict[str, object], ...]
    excluded_train: tuple[int, ...]
    screen_per_class: int
    screen: ScreenSplit
    selection_per_class: int
    folds: tuple[CVFold, ...]
    grouping: Grouping | None = None
    grouping_source: dict[str, str] | None = None
    screen_excluded_train: tuple[int, ...] = ()
    screen_fallback: str | None = None

    @property
    def format_version(self) -> int:
        """2 for grouped splits, 1 otherwise."""
        return 1 if self.grouping is None else 2

    @property
    def n_train(self) -> int:
        """Number of Train stamps of the dataset (excluded ones included)."""
        return int(self.dataset["n"]["Train"])

    @property
    def n_validation(self) -> int:
        """Number of Validation stamps of the dataset."""
        return int(self.dataset["n"]["Validation"])

    def screen_training_excluded(self) -> tuple[int, ...]:
        """The Train indices a screen run may not train on: excluded or grouped with Validation."""
        return tuple(sorted({*self.excluded_train, *self.screen_excluded_train}))

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


# --- Drawing -------------------------------------------------------------------------------


def _group_counts(group_of: Indices, labels: Labels, rows: Indices) -> npt.NDArray[np.int64]:
    """``(n_groups, n_classes)``: the stamps of each class that ``rows`` put in each group."""
    n_groups = int(group_of.max()) + 1 if group_of.size else 0
    counts = np.zeros((n_groups, len(LABEL_NAMES)), dtype=np.int64)
    np.add.at(counts, (group_of[rows], labels[rows]), 1)
    return counts


def _placement_order(rng: np.random.Generator, counts: npt.NDArray[np.int64]) -> Indices:
    """The non-empty groups, largest first, ties in seeded random order."""
    present = np.flatnonzero(counts.sum(axis=1) > 0).astype(np.int64)
    shuffled = rng.permutation(present)
    return shuffled[np.argsort(-counts[shuffled].sum(axis=1), kind="stable")]


def _members_by_group(group_of: Indices, rows: Indices) -> dict[int, Indices]:
    """Group id -> its members among ``rows`` (sorted)."""
    rows = np.sort(rows)
    order = np.argsort(group_of[rows], kind="stable")
    keys, starts = np.unique(group_of[rows][order], return_index=True)
    chunks = np.split(rows[order], starts[1:])
    return {int(key): chunk for key, chunk in zip(keys.tolist(), chunks, strict=True)}


def _check_screen_supply(val_labels: Labels, per_class: int) -> None:
    for label, name in enumerate(LABEL_NAMES):
        n = int(np.count_nonzero(val_labels == label))
        if n < 2 * per_class:
            msg = f"class {name} has {n} candidate stamps, fewer than {2 * per_class}"
            raise SplitsError(msg)


def _make_screen(
    val_labels: Labels, val_groups: Indices, seed: int, per_class: int
) -> tuple[ScreenSplit, int, str | None]:
    """The screen halves (module docstring), the per-class count they hold, the fallback."""
    _check_screen_supply(val_labels, per_class)
    rng = np.random.default_rng([seed, _SCREEN_STREAM])
    rows = np.arange(val_labels.size, dtype=np.int64)
    groups = canonical_groups(val_groups)
    counts = _group_counts(groups, val_labels, rows)
    members = _members_by_group(groups, rows)
    fill = np.zeros((2, len(LABEL_NAMES)), dtype=np.int64)
    halves: list[list[Indices]] = [[], []]
    for group in _placement_order(rng, counts).tolist():
        need = counts[group]
        fits = [bool(np.all(fill[h] + need <= per_class)) for h in (0, 1)]
        if not any(fits):
            continue  # left out of the screen
        if all(fits):
            load = fill @ need
            half = int(rng.integers(2)) if load[0] == load[1] else int(np.argmin(load))
        else:
            half = fits.index(True)
        fill[half] += need
        halves[half].append(members[group])
    held = int(fill.min())
    fallback = None
    parts = [np.sort(np.concatenate(h)) if h else _no_indices() for h in halves]
    if held < per_class:
        if held < 1:
            raise SplitsError("the Validation groups leave no class in both screen halves")
        fallback = (
            f"the Validation groups fit only {held} stamps of every class in both halves "
            f"(requested {per_class}; per half and class {fill.tolist()}): each half was "
            f"trimmed at random to {held} per class"
        )
        trimmed = []
        for part in parts:
            keep = [
                rng.choice(part[val_labels[part] == label], size=held, replace=False)
                for label in range(len(LABEL_NAMES))
            ]
            trimmed.append(np.sort(np.concatenate(keep)))
        parts = trimmed
    return ScreenSplit(parts[0], parts[1]), held, fallback


def _assign_folds(
    labels: Labels, usable: npt.NDArray[np.bool_], group_of: Indices, seed: int, n_folds: int
) -> Indices:
    """Per pool index, its evaluation fold (stratified by class, grouped), or -1 if not usable."""
    rng = np.random.default_rng([seed, _FOLD_STREAM])
    rows = np.flatnonzero(usable).astype(np.int64)
    counts = _group_counts(group_of, labels, rows)
    totals = counts.sum(axis=0)
    for label, name in enumerate(LABEL_NAMES):
        if totals[label] < n_folds:
            msg = f"class {name} has {totals[label]} stamps, fewer than {n_folds} folds"
            raise SplitsError(msg)
    weight = 1.0 / totals
    fill = np.zeros((n_folds, len(LABEL_NAMES)), dtype=np.float64)
    fold_of_group = np.full(counts.shape[0], -1, dtype=np.int64)
    for group in _placement_order(rng, counts).tolist():
        need = counts[group]
        fold = int(np.argmin(fill @ (need * weight)))
        fold_of_group[group] = fold
        fill[fold] += need
    fold_of = np.full(labels.size, -1, dtype=np.int64)
    fold_of[rows] = fold_of_group[group_of[rows]]
    return fold_of


def _draw_selection(
    rng: np.random.Generator,
    rest: Indices,
    labels: Labels,
    group_of: Indices,
    per_class: int,
    fold: int,
) -> Indices:
    """A fold's selection set: ``per_class`` of each class, from whole groups (docstring).

    Raises:
        SplitsError: if a class runs out of candidates.
    """
    members = _members_by_group(group_of, rest)
    need = np.full(len(LABEL_NAMES), per_class, dtype=np.int64)
    taken: list[Indices] = []
    for group in rng.permutation(np.asarray(sorted(members), dtype=np.int64)).tolist():
        group_rows = members[group]
        group_labels = labels[group_rows]
        if not np.any(need[group_labels] > 0):
            continue
        for label in np.unique(group_labels).tolist():
            if need[label] == 0:
                continue
            candidates = group_rows[group_labels == label]
            if candidates.size > need[label]:
                candidates = rng.choice(candidates, size=int(need[label]), replace=False)
            taken.append(candidates)
            need[label] -= candidates.size
        if not need.any():
            break
    if need.any():
        short = LABEL_NAMES[int(np.argmax(need > 0))]
        msg = f"fold {fold}: class {short} has too few stamps outside the fold for a selection set"
        raise SplitsError(msg)
    return np.sort(np.concatenate(taken)).astype(np.int64)


def _build_fold(
    fold: int, fold_of: Indices, selection: Indices, group_of: Indices | None
) -> CVFold:
    rest = np.flatnonzero((fold_of >= 0) & (fold_of != fold)).astype(np.int64)
    selection = np.sort(selection.astype(np.int64))
    # Grouped: every member of a selection group leaves the training part.
    held = selection if group_of is None else rest[np.isin(group_of[rest], group_of[selection])]
    return CVFold(
        fold,
        np.flatnonzero(fold_of == fold).astype(np.int64),
        selection,
        np.setdiff1d(rest, held, assume_unique=True).astype(np.int64),
        np.setdiff1d(held, selection, assume_unique=True).astype(np.int64),
    )


def _labels(data: StampData, path: Path) -> tuple[Labels, Labels]:
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
            split: splits[split]["files"]["labels.npy"]["sha256"] for split in POOL_SPLITS
        },
        "n": {split: int(splits[split]["n"]) for split in POOL_SPLITS},
    }


def _screen_exclusions(group_of: Indices, n_train: int) -> tuple[int, ...]:
    """The Train indices whose group has a Validation member."""
    in_validation = np.unique(group_of[n_train:])
    return tuple(np.flatnonzero(np.isin(group_of[:n_train], in_validation)).tolist())


def _pool_grouping(
    data: StampData, radius_arcsec: float, group_ids_file: Path | None
) -> tuple[Grouping, dict[str, str] | None]:
    if group_ids_file is None:
        return grouping_for(data, POOL_SPLITS, radius_arcsec=radius_arcsec), None
    ids, source = load_group_ids(group_ids_file)
    return grouping_for(data, POOL_SPLITS, group_ids=ids), source


def make_splits(
    path: Path,
    *,
    seed: int = 0,
    n_folds: int = 5,
    selection_per_class: int = 100,
    screen_per_class: int = 50,
    radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
    group_ids_file: Path | None = None,
) -> ProtocolSplits:
    """Build the grouped screen and CV folds of the prepared classifier dataset at ``path``.

    Finds the cross-split duplicates first and leaves the duplicated Train stamps out of
    every fold. Groups the pool by position within ``radius_arcsec``, or by the ids of
    ``group_ids_file`` (:func:`~astro_simclr.protocol.groups.load_group_ids`), and keeps
    every group whole (module docstring). Deterministic for a given dataset, seed and
    numpy version.

    Raises:
        DataError: if ``path`` is not a valid prepared classifier layout, or the positions
            or group ids cannot be read.
        SplitsError: if a class is too small for the requested sizes, ``n_folds`` is not in
            2-10 (``fold_of`` stores one digit per stamp), or a size is not positive.
        ValueError: if ``radius_arcsec`` is not positive.
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
    n_train = train_labels.size
    grouping, source = _pool_grouping(data, radius_arcsec, group_ids_file)
    group_of = grouping.group_of
    screen, held, fallback = _make_screen(val_labels, group_of[n_train:], seed, screen_per_class)

    labels = np.concatenate([train_labels, val_labels])
    usable = np.ones(labels.size, dtype=bool)
    usable[excluded] = False
    fold_of = _assign_folds(labels, usable, group_of, seed, n_folds)
    folds = []
    for fold in range(n_folds):
        rng = np.random.default_rng([seed, _SELECTION_STREAM, fold])
        rest = np.flatnonzero((fold_of >= 0) & (fold_of != fold)).astype(np.int64)
        for label, name in enumerate(LABEL_NAMES):
            if np.count_nonzero(labels[rest] == label) <= selection_per_class:
                msg = (
                    f"fold {fold}: class {name} has too few stamps outside the fold for a "
                    f"selection set of {selection_per_class}"
                )
                raise SplitsError(msg)
        selection = _draw_selection(rng, rest, labels, group_of, selection_per_class, fold)
        built = _build_fold(fold, fold_of, selection, group_of)
        # The training part must keep at least one stamp of each class.
        missing = set(range(len(LABEL_NAMES))) - set(labels[built.training].tolist())
        if missing:
            name = LABEL_NAMES[min(missing)]
            msg = (
                f"fold {fold}: class {name} has too few stamps outside the fold's selection groups"
            )
            raise SplitsError(msg)
        folds.append(built)
    return ProtocolSplits(
        dataset=_dataset_identity(path, meta),
        seed=seed,
        duplicates=tuple(dup.as_dict() for dup in duplicates),
        excluded_train=tuple(excluded),
        screen_per_class=held,
        screen=screen,
        selection_per_class=selection_per_class,
        folds=tuple(folds),
        grouping=grouping,
        grouping_source=source,
        screen_excluded_train=_screen_exclusions(group_of, n_train),
        screen_fallback=fallback,
    )


# --- The file ------------------------------------------------------------------------------


def _fold_of(splits: ProtocolSplits) -> str:
    chars = [_EXCLUDED] * (splits.n_train + splits.n_validation)
    for fold in splits.folds:
        for index in fold.evaluation:
            chars[index] = str(fold.fold)
    return "".join(chars)


def _grouping_plain(splits: ProtocolSplits, grouping: Grouping) -> dict[str, Any]:
    return {
        "method": grouping.method,
        "radius_arcsec": grouping.radius_arcsec,
        "coordinates": (
            "ra, dec de-standardised with M1_FEATURE_NORM_STATS"
            if grouping.method == POSITION
            else None
        ),
        "source": splits.grouping_source,
        "rows": "pool indices",
        "n_groups": grouping.n_groups,
        "groups": grouping.multi(),
        "screen_excluded_train": list(splits.screen_excluded_train),
        "screen_fallback": splits.screen_fallback,
    }


def _to_plain(splits: ProtocolSplits) -> dict[str, Any]:
    plain: dict[str, Any] = {
        "format": FORMAT,
        "format_version": splits.format_version,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "dataset": splits.dataset,
        "seed": splits.seed,
        "pool": "Train indices 0..n_train-1, then Validation index j as n_train + j",
        "leakage": {
            "duplicates": list(splits.duplicates),
            "excluded_train": list(splits.excluded_train),
        },
    }
    if splits.grouping is not None:
        plain["grouping"] = _grouping_plain(splits, splits.grouping)
    plain["screen"] = {
        "per_class": splits.screen_per_class,
        "selection": splits.screen.selection.tolist(),
        "decision": splits.screen.decision.tolist(),
    }
    plain["cv"] = {
        "n_folds": len(splits.folds),
        "selection_per_class": splits.selection_per_class,
        "fold_of": _fold_of(splits),
        "selection": [fold.selection.tolist() for fold in splits.folds],
    }
    return plain


def write_splits(splits: ProtocolSplits, path: Path) -> None:
    """Write ``splits`` as JSON to ``path`` (index lists on one line each; deterministic).

    A version-1 object (read from a version-1 file) is written back as version 1.

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
    raw: dict[str, Any],
    n_pool: int,
    excluded: tuple[int, ...],
    group_of: Indices | None,
    path: Path,
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
        folds.append(_build_fold(fold, fold_of, selection, group_of))
    return per_class, tuple(folds)


def _parse_method(raw: dict[str, Any], path: Path) -> tuple[GroupingMethod, float | None]:
    method, radius = raw.get("method"), raw.get("radius_arcsec")
    if method == POSITION:
        if isinstance(radius, bool) or not isinstance(radius, int | float) or not radius > 0:
            raise SplitsError(f"{path}: grouping.radius_arcsec must be a positive number")
        return POSITION, float(radius)
    if method == EXTERNAL:
        if radius is not None:
            raise SplitsError(f"{path}: an external grouping has no radius_arcsec")
        return EXTERNAL, None
    raise SplitsError(f"{path}: grouping.method must be {POSITION!r} or {EXTERNAL!r}")


def _parse_groups(raw: object, n_pool: int, path: Path) -> Indices:
    """Canonical pool group ids from the stored multi-stamp groups (the rest: singletons)."""
    if not isinstance(raw, list):
        raise SplitsError(f"{path}: grouping.groups must be a list of index lists")
    label = np.arange(n_pool, dtype=np.int64)
    seen = np.zeros(n_pool, dtype=bool)
    for k, members_raw in enumerate(raw):
        members = _int_array(members_raw, f"grouping.groups[{k}]", path)
        if members.size < 2 or members[0] < 0 or members[-1] >= n_pool:
            msg = f"{path}: grouping.groups[{k}] must hold two or more pool indices"
            raise SplitsError(msg)
        if seen[members].any():
            raise SplitsError(f"{path}: grouping.groups[{k}] shares a stamp with another group")
        seen[members] = True
        label[members] = members[0]
    return canonical_groups(label)


def _parse_grouping(
    raw: object, n_pool: int, path: Path
) -> tuple[Grouping, dict[str, str] | None, tuple[int, ...], str | None]:
    """The ``grouping`` block of a version-2 file: the grouping, source, exclusions, fallback."""
    if not isinstance(raw, dict):
        raise SplitsError(f"{path}: a version-2 file needs a grouping block")
    method, radius = _parse_method(raw, path)
    grouping = Grouping(method, radius, _parse_groups(raw.get("groups"), n_pool, path))
    if raw.get("n_groups") != grouping.n_groups:
        msg = f"{path}: grouping.n_groups is not the number of groups ({grouping.n_groups})"
        raise SplitsError(msg)
    excluded = tuple(
        _int_array(raw.get("screen_excluded_train"), "screen_excluded_train", path).tolist()
    )
    source, fallback = raw.get("source"), raw.get("screen_fallback")
    if source is not None and not isinstance(source, dict):
        raise SplitsError(f"{path}: grouping.source must be null or an object")
    if fallback is not None and not isinstance(fallback, str):
        raise SplitsError(f"{path}: grouping.screen_fallback must be null or a string")
    return grouping, source, excluded, fallback


def check_group_integrity(splits: ProtocolSplits) -> None:
    """Check that no group spans two parts of ``splits`` (version 2; nothing for version 1).

    - Every group's stamps outside ``excluded_train`` are evaluated by one fold.
    - No group has a stamp in a fold's selection set and another in its training part.
    - No group has a stamp in each screen half.
    - ``screen_excluded_train`` is exactly the Train stamps whose group has a Validation
      member.

    Raises:
        SplitsError: naming the first group (by its smallest pool index) that breaks a rule.
    """
    grouping = splits.grouping
    if grouping is None:
        return
    group_of, n_train = grouping.group_of, splits.n_train
    if group_of.size != n_train + splits.n_validation:
        raise SplitsError("the grouping does not have one group per pool stamp")
    fold_of = np.full(group_of.size, -1, dtype=np.int64)
    for fold in splits.folds:
        fold_of[fold.evaluation] = fold.fold
    rows = np.flatnonzero(fold_of >= 0)
    low = np.full(grouping.n_groups, len(splits.folds), dtype=np.int64)
    high = np.full(grouping.n_groups, -1, dtype=np.int64)
    np.minimum.at(low, group_of[rows], fold_of[rows])
    np.maximum.at(high, group_of[rows], fold_of[rows])
    split_groups = np.flatnonzero((high >= 0) & (low != high))
    if split_groups.size:
        first = int(np.flatnonzero(group_of == split_groups[0])[0])
        raise SplitsError(f"the group of pool stamp {first} is evaluated by two folds")
    for fold in splits.folds:
        shared = np.intersect1d(group_of[fold.selection], group_of[fold.training])
        if shared.size:
            first = int(np.flatnonzero(group_of == shared[0])[0])
            msg = f"fold {fold.fold}: the group of pool stamp {first} is in selection and training"
            raise SplitsError(msg)
    screen = (
        group_of[n_train + splits.screen.selection],
        group_of[n_train + splits.screen.decision],
    )
    shared = np.intersect1d(*screen)
    if shared.size:
        first = int(np.flatnonzero(group_of == shared[0])[0])
        raise SplitsError(f"the group of pool stamp {first} is in both screen halves")
    if _screen_exclusions(group_of, n_train) != splits.screen_excluded_train:
        msg = "screen_excluded_train is not the Train stamps grouped with a Validation stamp"
        raise SplitsError(msg)


def read_splits(path: Path) -> ProtocolSplits:
    """Read and check a splits file written by :func:`write_splits` (version 1 or 2).

    Checks the structure (sizes, disjointness, ranges) and, for version 2, the group
    integrity (:func:`check_group_integrity`); :func:`verify_splits` checks that it belongs
    to a given prepared dataset, the class balance, the stratification, the excluded stamps
    and the grouping itself.

    Raises:
        SplitsError: if the file cannot be read or is malformed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise SplitsError(f"{path}: cannot read the splits file ({err})") from None
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise SplitsError(f"{path}: not a {FORMAT} file")
    file_version = raw.get("format_version")
    if file_version not in _READABLE_VERSIONS:
        raise SplitsError(f"{path}: unsupported format_version {file_version!r}")
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
    n_pool = n_train + n_validation
    grouping: Grouping | None = None
    source: dict[str, str] | None = None
    screen_excluded: tuple[int, ...] = ()
    fallback: str | None = None
    if file_version == 2:
        grouping, source, screen_excluded, fallback = _parse_grouping(
            raw.get("grouping"), n_pool, path
        )
    screen_per_class, screen = _parse_screen(screen_raw, n_validation, path)
    group_of = None if grouping is None else grouping.group_of
    selection_per_class, folds = _parse_folds(cv_raw, n_pool, excluded, group_of, path)
    splits = ProtocolSplits(
        dataset=dataset,
        seed=seed,
        duplicates=duplicates,
        excluded_train=excluded,
        screen_per_class=screen_per_class,
        screen=screen,
        selection_per_class=selection_per_class,
        folds=folds,
        grouping=grouping,
        grouping_source=source,
        screen_excluded_train=screen_excluded,
        screen_fallback=fallback,
    )
    try:
        check_group_integrity(splits)
    except SplitsError as err:
        raise SplitsError(f"{path}: {err}") from None
    return splits


def _stratification_slack(splits: ProtocolSplits, labels: Labels) -> npt.NDArray[np.int64]:
    """Per class, the allowed fold-count spread: 1, or the largest group's stamps of it."""
    slack = np.ones(len(LABEL_NAMES), dtype=np.int64)
    if splits.grouping is None:
        return slack
    usable = np.ones(labels.size, dtype=bool)
    usable[list(splits.excluded_train)] = False
    counts = _group_counts(splits.grouping.group_of, labels, np.flatnonzero(usable))
    return np.maximum(slack, counts.max(axis=0, initial=0))


def _verify_grouping(
    splits: ProtocolSplits, data: StampData, path: Path, group_ids_file: Path | None
) -> None:
    grouping = splits.grouping
    if grouping is None:
        return
    if grouping.method == POSITION:
        assert grouping.radius_arcsec is not None  # noqa: S101 - _parse_grouping's invariant
        fresh, _ = _pool_grouping(data, grouping.radius_arcsec, None)
        what = f'a fresh position grouping ({grouping.radius_arcsec}")'
    elif group_ids_file is not None:
        fresh, _ = _pool_grouping(data, DEFAULT_RADIUS_ARCSEC, group_ids_file)
        what = f"the group ids of {group_ids_file}"
    else:
        return  # external ids not given: the structure was checked by check_group_integrity
    if not np.array_equal(fresh.group_of, grouping.group_of):
        stored, found = grouping.n_groups, fresh.n_groups
        msg = (
            f"the stored groups ({stored}) are not what {what} finds on {path} ({found}); "
            "regenerate the splits with `astro-simclr make-splits`"
        )
        raise SplitsError(msg)


def verify_splits(
    splits: ProtocolSplits, path: Path, *, group_ids_file: Path | None = None
) -> None:
    """Check that ``splits`` belongs to the prepared dataset at ``path`` and is well drawn.

    - The source SHA-256, the SHA-256 of Train's and Validation's ``labels.npy`` (from
      ``meta.json``) and the split sizes must match the stored ones (the directory name
      may differ: a renamed or moved copy holds the same data).
    - With the real labels: every screen half holds ``screen_per_class`` stamps of each
      class, every fold's selection set ``selection_per_class``, and the folds are
      stratified (per class, the evaluation counts of two folds differ by at most 1, or,
      grouped, by the most stamps of that class in one group).
    - The excluded Train stamps are exactly the ones the leakage check finds again
      (:func:`~astro_simclr.protocol.leakage.find_duplicates`; reads every stamp, a few
      seconds on ``red``).
    - Version 2: no group spans two parts (:func:`check_group_integrity`), and the stored
      groups are what a fresh grouping finds: by position at the stored radius, or from
      ``group_ids_file`` for an external grouping (without it, an external grouping is
      checked for integrity only).

    Raises:
        SplitsError: on a different dataset, an unbalanced or unstratified split, other
            excluded stamps, a group in two parts or other groups.
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
    check_group_integrity(splits)
    # Before the stratification: folds drawn on other groups are mostly "unstratified" on
    # the stored ones, and the cause is the grouping.
    _verify_grouping(splits, data, path, group_ids_file)
    per_fold = np.array(
        [np.bincount(labels[fold.evaluation], minlength=n_classes) for fold in splits.folds]
    )
    if np.any(per_fold.max(axis=0) - per_fold.min(axis=0) > _stratification_slack(splits, labels)):
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
