"""The stored label-budget subsets of phase 5 (task 5.4), and ``make-label-budgets``.

Phase 5 asks whether SSL pretraining helps **with few labels**
(``docs/phases/05-self-supervised.md``, 5.4; plan review F4). The budgets count labelled
**objects per class** (10, 100 and 1,000 by default, plus 100 %), and every arm (SSL, random
init, scratch, ImageNet, Zoobot) trains on the **same** stored subsets, so arms are paired
by subset and the subset-to-subset variance is measured, not ignored
(:func:`astro_simclr.protocol.stats.hierarchical_paired_bootstrap`).

**Parts.** A subset is drawn from one *training part* only (:class:`BudgetPart`):

- a CD21 cross-validation fold's training part (:func:`cv_parts`, from the stored splits
  of :mod:`astro_simclr.protocol.splits`; rows are pool indices); the object is the
  position group (or the stamp, for an ungrouped version-1 file), named by the smallest
  pool index of its stamps in that training part;
- R5 Train (:func:`object_split_parts`, from the stored object splits of
  :mod:`astro_simclr.protocol.object_splits`; rows are alert-table rows); the object is the
  table's group key (the ``oid`` by default).

**The draw.** For part ``p`` and repeat ``r``, the part's objects (sorted ids) are put in a
seeded random order, ``np.random.default_rng([seed, 5, p, r]).permutation``. Per class, the
budget-``b`` subset takes the first ``b`` objects of that order that have an alert of that
class, with their alerts **of that class** in the training part (the rare object with alerts
of two classes counts once in each; its other alerts are unlabelled in that subset). So
the budgets are **nested** (subset ``r`` at 10 per class is inside subset ``r`` at 100), each
subset holds exactly ``b`` objects of every class, and repeats differ by their order only.

**The selection set** (for the arms that early-stop or choose a hyperparameter): per budget,
``selection_per_class[b]`` objects of each class, drawn from the same training part: per
class, the objects that come next in the same order, skipping every object of the subset
(of any class), so no object is in both. Its size is ``floor(selection_fraction x b)``
(at most 20 % of the budget, the plan's cap), lowered when needed to what the smallest class
has left after the subset in every part and repeat, so that it is the same for every class,
part and repeat (CD21 SN has about 1,084 objects in a fold's training part: at 1,000 it
leaves 83, not 200). It is recorded in the file. ``selection_fraction = 0`` stores no
selection set: the arms then train on the **fixed schedule** alternative (a declared epoch
count or a convex fit, no early stopping; ``docs/evaluation-protocol.md``, "Phase 5 label
budgets").

**100 %** is not stored: its subset is the whole training part, and its selection set is
the part's own (the fold's ``selection``; R5's Validation selection half), as in phases 2
and 3 (:func:`resolve`).

**The file** (JSON, ``format`` ``astro-simclr-label-budgets``, version 1) is the source of
truth, as for the stored splits (numpy does not promise one seeded permutation across
versions)::

    source              kind (cv-splits or object-splits), the parent file's name and
                        SHA-256, and the data fingerprint (CD21: the dataset identity of
                        the splits file; R5: the alert table's SHA-256)
    label_names, object_id, seed, n_repeats, budgets, include_full,
    selection_fraction, selection_per_class (per budget)
    parts               per part: its name and, per repeat, per class, ``objects`` (the
                        largest budget's subset, in draw order: budget b is its first b) and
                        ``selection`` (per budget, sorted)

:func:`verify_label_budgets` checks a file against the parts rebuilt from the verified
parent splits and the data. Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.protocol.alerts import AlertTable, read_alert_table
from astro_simclr.protocol.object_splits import (
    ObjectSplits,
    read_object_splits,
    verify_object_splits,
)
from astro_simclr.protocol.splits import ProtocolSplits, SplitsError, read_splits, verify_splits

__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_REPEATS",
    "DEFAULT_SELECTION_FRACTION",
    "FORMAT",
    "FORMAT_VERSION",
    "MAX_SELECTION_FRACTION",
    "BudgetError",
    "BudgetPart",
    "BudgetSubset",
    "LabelBudgets",
    "PartsSource",
    "SourceKind",
    "cv_parts",
    "cv_parts_from_files",
    "cv_source",
    "file_sha256",
    "make_label_budgets",
    "object_split_parts",
    "object_split_parts_from_files",
    "object_split_source",
    "read_label_budgets",
    "resolve",
    "verify_label_budgets",
    "write_label_budgets",
]

#: ``format`` of a stored label-budgets file.
FORMAT: Final = "astro-simclr-label-budgets"
#: Bumped on any incompatible change of the file.
FORMAT_VERSION: Final = 1
#: Labelled objects per class (plan 5.4).
DEFAULT_BUDGETS: Final = (10, 100, 1000)
#: Repeated subsets per part and budget: the probe needs >= 20 (plan 5.4).
DEFAULT_REPEATS: Final = 20
#: The selection set's size as a fraction of the budget.
DEFAULT_SELECTION_FRACTION: Final = 0.2
#: The plan's cap on that fraction ("at most 20 % of the budget").
MAX_SELECTION_FRACTION: Final = 0.2
#: Generator stream of the draw (second entry of the seed sequence; the phase-2 splits use
#: 0-2 and the object splits 0-1 with other seeds, so 5 keeps this draw apart in any case).
_BUDGET_STREAM: Final = 5

SourceKind = Literal["cv-splits", "object-splits"]
_SOURCE_KINDS: Final = ("cv-splits", "object-splits")
Rows = npt.NDArray[np.int64]
ObjectId = int | str


class BudgetError(SplitsError):
    """A label-budgets file is malformed, infeasible, or does not belong to its parent."""


@dataclass(frozen=True, slots=True)
class BudgetPart:
    """One training part a subset may be drawn from, as rows of a larger index space.

    ``rows`` are the part's rows (pool indices or alert-table rows; sorted, distinct),
    ``objects`` and ``labels`` the object id and class of each row (same order). ``selection``
    are the rows of the part's own selection set, used at 100 % only.
    """

    name: str
    rows: Rows
    objects: npt.NDArray[Any]
    labels: npt.NDArray[np.int64]
    selection: Rows


@dataclass(frozen=True, slots=True)
class BudgetSubset:
    """One resolved subset: the rows a run trains on and the rows it may select on.

    ``budget`` is the objects per class, or ``None`` for 100 %; ``repeat`` is 0 at 100 %.
    """

    part: str
    repeat: int
    budget: int | None
    labelled: Rows
    selection: Rows


@dataclass(frozen=True, slots=True)
class LabelBudgets:
    """A stored label-budgets file (module docstring).

    ``objects[p][r][c]`` is the largest budget's subset of class ``c`` in repeat ``r`` of
    part ``p``, in draw order; ``selection[p][r][c][b]`` the sorted selection set of budget
    ``b``. ``part_names`` follow the parts' order.
    """

    source: dict[str, Any]
    label_names: tuple[str, ...]
    object_id: str
    seed: int
    n_repeats: int
    budgets: tuple[int, ...]
    include_full: bool
    selection_fraction: float
    selection_per_class: dict[int, int]
    part_names: tuple[str, ...]
    objects: tuple[tuple[tuple[tuple[ObjectId, ...], ...], ...], ...]
    selection: tuple[tuple[tuple[dict[int, tuple[ObjectId, ...]], ...], ...], ...]

    def part_index(self, name: str) -> int:
        """The index of part ``name``.

        Raises:
            KeyError: if the file has no such part.
        """
        try:
            return self.part_names.index(name)
        except ValueError:
            raise KeyError(f"no part {name!r}; the file has {list(self.part_names)}") from None


# --- Parts ---------------------------------------------------------------------------------


def _first_member(group_of: Rows, rows: Rows) -> Rows:
    """Per row of ``rows``, the smallest of ``rows`` in its group: a stable, readable id."""
    group = group_of[rows]
    first = np.full(int(group_of.max()) + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first, group, rows)
    return first[group]


def cv_parts(splits: ProtocolSplits, pool_labels: npt.ArrayLike) -> list[BudgetPart]:
    """The training parts of the CV folds of ``splits``, named ``fold0``, ``fold1``, ...

    ``pool_labels`` are the labels of the pool (Train then Validation, the order of
    :class:`~astro_simclr.protocol.splits.ProtocolSplits`). The object of a stamp is the
    smallest pool index of its group's stamps in the fold's training part (so every id is a
    training row; the stamp itself in a version-1 file, which has no groups).

    Raises:
        BudgetError: if ``pool_labels`` does not have one label per pool stamp.
    """
    labels = np.asarray(pool_labels, dtype=np.int64)
    n_pool = splits.n_train + splits.n_validation
    if labels.shape != (n_pool,):
        raise BudgetError(f"expected {n_pool} pool labels, got shape {labels.shape}")
    group_of = (
        np.arange(n_pool, dtype=np.int64) if splits.grouping is None else splits.grouping.group_of
    )
    return [
        BudgetPart(
            f"fold{fold.fold}",
            fold.training,
            _first_member(group_of, fold.training),
            labels[fold.training],
            fold.selection,
        )
        for fold in splits.folds
    ]


def object_split_parts(splits: ObjectSplits, table: AlertTable) -> list[BudgetPart]:
    """R5's one training part, ``Train``: its alert rows, grouped by the table's group key.

    The selection set at 100 % is Validation's selection half. Call
    :func:`~astro_simclr.protocol.object_splits.verify_object_splits` first.
    """
    rows = splits.rows(table, "Train")
    return [
        BudgetPart(
            "Train",
            rows,
            np.asarray(table.group[rows]),
            np.asarray(table.label[rows], dtype=np.int64),
            splits.rows(table, "selection"),
        )
    ]


def cv_source(splits: ProtocolSplits, path: Path) -> dict[str, Any]:
    """The ``source`` of label budgets drawn from the stored CV splits file at ``path``.

    The fingerprint is the splits file's dataset identity (source, labels and sizes of the
    prepared data), which :func:`~astro_simclr.protocol.splits.verify_splits` checks.

    Raises:
        OSError: if the file cannot be read.
    """
    identity = {key: splits.dataset.get(key) for key in ("source_sha256", "labels_sha256", "n")}
    return {
        "kind": "cv-splits",
        "file": path.name,
        "sha256": file_sha256(path),
        "fingerprint": {"dataset": identity},
    }


def object_split_source(splits: ObjectSplits, path: Path) -> dict[str, Any]:
    """The ``source`` of label budgets drawn from the stored object splits file at ``path``.

    The fingerprint is the alert table's SHA-256 recorded in the splits (which
    :func:`~astro_simclr.protocol.object_splits.verify_object_splits` checks).

    Raises:
        OSError: if the file cannot be read.
    """
    return {
        "kind": "object-splits",
        "file": path.name,
        "sha256": file_sha256(path),
        "fingerprint": {"table_sha256": splits.table.get("sha256")},
    }


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes (the parent splits file's fingerprint).

    Raises:
        OSError: if the file cannot be read.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PartsSource:
    """The verified parts of a parent splits file, with what a budgets file records of it."""

    parts: list[BudgetPart]
    source: dict[str, Any]
    label_names: tuple[str, ...]
    object_id: str


def cv_parts_from_files(splits_path: Path, data_path: Path) -> PartsSource:
    """Read the CV splits at ``splits_path``, verify them on the prepared data, build the parts.

    Raises:
        SplitsError: if the splits file is malformed or does not belong to the data
            (:func:`~astro_simclr.protocol.splits.verify_splits`, which reads every stamp).
        DataError: if ``data_path`` is not a valid prepared classifier layout.
    """
    splits = read_splits(splits_path)
    verify_splits(splits, data_path)
    data = load_prepared(data_path)
    train, val = data.splits["Train"].labels, data.splits["Validation"].labels
    assert train is not None and val is not None  # noqa: S101 - verify_splits read them
    labels = np.concatenate([np.asarray(train), np.asarray(val)]).astype(np.int64)
    what = (
        "the stamp's pool index"
        if splits.grouping is None
        else "the smallest pool index of the object's group in the training part"
    )
    return PartsSource(
        cv_parts(splits, labels), cv_source(splits, splits_path), tuple(LABEL_NAMES), what
    )


def object_split_parts_from_files(
    splits_path: Path, alerts_path: Path, label_names: Sequence[str]
) -> PartsSource:
    """Read the alert table and its object splits, verify them, build R5's ``Train`` part.

    Raises:
        DataError: if the alert table cannot be read.
        SplitsError: if the object splits are malformed or do not belong to the table.
    """
    table = read_alert_table(alerts_path, label_names)
    splits = read_object_splits(splits_path)
    verify_object_splits(splits, table)
    return PartsSource(
        object_split_parts(splits, table),
        object_split_source(splits, splits_path),
        table.label_names,
        f"the alert table's {table.group_key}",
    )


# --- Drawing -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Order:
    """One part's objects and, per repeat and class, the object positions in draw order."""

    ids: npt.NDArray[Any]
    per_class: list[list[Rows]]  # [repeat][class] -> object indices into ids, in draw order


def _class_objects(part: BudgetPart, n_classes: int) -> tuple[npt.NDArray[Any], list[Rows]]:
    """The part's sorted object ids, and per class the indices of the objects having it."""
    ids, inverse = np.unique(part.objects, return_inverse=True)
    inverse = inverse.reshape(-1).astype(np.int64)
    having = [np.unique(inverse[part.labels == c]).astype(np.int64) for c in range(n_classes)]
    return ids, having


def _order(part: BudgetPart, index: int, seed: int, n_repeats: int, n_classes: int) -> _Order:
    ids, having = _class_objects(part, n_classes)
    per_repeat = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng([seed, _BUDGET_STREAM, index, repeat])
        rank = np.empty(ids.size, dtype=np.int64)
        rank[rng.permutation(ids.size)] = np.arange(ids.size, dtype=np.int64)
        per_repeat.append([objs[np.argsort(rank[objs], kind="stable")] for objs in having])
    return _Order(ids, per_repeat)


def _left_after(order: list[Rows], budget: int) -> list[Rows]:
    """Per class, the objects after the budget's subset that are in no class's subset."""
    taken = np.concatenate([objs[:budget] for objs in order])
    return [objs[budget:][~np.isin(objs[budget:], taken)] for objs in order]


def _check_parts(parts: Sequence[BudgetPart], n_classes: int) -> None:
    if not parts:
        raise BudgetError("no training part to draw from")
    names = [part.name for part in parts]
    if len(set(names)) != len(names):
        raise BudgetError(f"part names repeat: {names}")
    for part in parts:
        n = part.rows.size
        if part.objects.shape != (n,) or part.labels.shape != (n,):
            raise BudgetError(f"part {part.name}: rows, objects and labels differ in length")
        if n and (part.labels.min() < 0 or part.labels.max() >= n_classes):
            raise BudgetError(f"part {part.name}: labels must be in range({n_classes})")
        if np.unique(part.rows).size != n:
            raise BudgetError(f"part {part.name}: rows repeat")


def _check_settings(
    budgets: Sequence[int], n_repeats: int, selection_fraction: float
) -> tuple[int, ...]:
    values = tuple(int(b) for b in budgets)
    if not values or any(b < 1 for b in values) or list(values) != sorted(set(values)):
        raise BudgetError(f"budgets must be increasing positive integers, got {list(budgets)}")
    if n_repeats < 1:
        raise BudgetError(f"n_repeats must be >= 1, got {n_repeats}")
    if not 0 <= selection_fraction <= MAX_SELECTION_FRACTION:
        msg = (
            f"selection_fraction must be in [0, {MAX_SELECTION_FRACTION}] (at most 20 % of "
            f"the budget), got {selection_fraction}"
        )
        raise BudgetError(msg)
    return values


def make_label_budgets(
    parts: Sequence[BudgetPart],
    *,
    source: dict[str, Any],
    label_names: Sequence[str],
    object_id: str,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    n_repeats: int = DEFAULT_REPEATS,
    selection_fraction: float = DEFAULT_SELECTION_FRACTION,
    include_full: bool = True,
    seed: int = 0,
) -> LabelBudgets:
    """Draw the nested label-budget subsets and selection sets of ``parts`` (module docstring).

    ``source`` describes the parent (``kind``, ``file``, ``sha256``, ``fingerprint``) and is
    stored as is; ``object_id`` says what an object id is. Deterministic for given parts,
    settings and numpy version.

    Raises:
        BudgetError: on invalid settings or parts, a class with fewer objects than the
            largest budget in some part, or a selection set that would be empty (no object of
            some class left after the largest budget) while ``selection_fraction > 0``.
    """
    n_classes = len(label_names)
    values = _check_settings(budgets, n_repeats, selection_fraction)
    _check_parts(parts, n_classes)
    largest = values[-1]
    orders = [_order(part, i, seed, n_repeats, n_classes) for i, part in enumerate(parts)]
    for part, order in zip(parts, orders, strict=True):
        for c, objs in enumerate(order.per_class[0]):
            if objs.size < largest:
                msg = (
                    f"part {part.name}: class {label_names[c]} has {objs.size} objects, fewer "
                    f"than the largest budget {largest}"
                )
                raise BudgetError(msg)
    sizes: dict[int, int] = {}
    for budget in values:
        size = math.floor(round(selection_fraction * budget, 9))
        if size:
            left = min(
                rest.size
                for order in orders
                for per_class in order.per_class
                for rest in _left_after(per_class, budget)
            )
            if left < 1:
                msg = f"budget {budget}: some class has no object left for a selection set"
                raise BudgetError(msg)
            size = min(size, left)
        sizes[budget] = size
    objects = []
    selection = []
    for order in orders:
        objects.append(
            tuple(
                tuple(tuple(order.ids[objs[:largest]].tolist()) for objs in per_class)
                for per_class in order.per_class
            )
        )
        repeats = []
        for per_class in order.per_class:
            rest = {budget: _left_after(per_class, budget) for budget in values}
            repeats.append(
                tuple(
                    {
                        budget: tuple(sorted(order.ids[rest[budget][c][: sizes[budget]]].tolist()))
                        for budget in values
                    }
                    for c in range(n_classes)
                )
            )
        selection.append(tuple(repeats))
    return LabelBudgets(
        source=dict(source),
        label_names=tuple(label_names),
        object_id=object_id,
        seed=seed,
        n_repeats=n_repeats,
        budgets=values,
        include_full=include_full,
        selection_fraction=float(selection_fraction),
        selection_per_class=sizes,
        part_names=tuple(part.name for part in parts),
        objects=tuple(objects),
        selection=tuple(selection),
    )


# --- Resolving -----------------------------------------------------------------------------


def _rows_of(part: BudgetPart, per_class: Sequence[Sequence[ObjectId]]) -> Rows:
    """The part's rows of class ``c`` whose object is in ``per_class[c]``, for every ``c``."""
    keep = np.zeros(part.rows.size, dtype=bool)
    for c, ids in enumerate(per_class):
        if ids:
            keep |= (part.labels == c) & np.isin(part.objects, np.asarray(ids))
    return np.sort(part.rows[keep]).astype(np.int64)


def resolve(budgets: LabelBudgets, part: BudgetPart) -> list[BudgetSubset]:
    """Every subset of ``part``: per budget (increasing), every repeat, then 100 % if stored.

    The rows come from ``part`` (rebuilt from the verified parent splits and the data; run
    :func:`verify_label_budgets` first).

    Raises:
        KeyError: if the file has no part named ``part.name``.
    """
    p = budgets.part_index(part.name)
    subsets = []
    for budget in budgets.budgets:
        for repeat in range(budgets.n_repeats):
            classes = budgets.objects[p][repeat]
            labelled = _rows_of(part, [ids[:budget] for ids in classes])
            chosen = [budgets.selection[p][repeat][c][budget] for c in range(len(classes))]
            subsets.append(
                BudgetSubset(part.name, repeat, budget, labelled, _rows_of(part, chosen))
            )
    if budgets.include_full:
        full = np.sort(part.rows).astype(np.int64)
        subsets.append(BudgetSubset(part.name, 0, None, full, np.sort(part.selection)))
    return subsets


# --- Verification --------------------------------------------------------------------------


def _verify_part(budgets: LabelBudgets, part: BudgetPart) -> None:
    """Every stored object of the part has the stated class there; sizes and disjointness."""
    p = budgets.part_index(part.name)
    n_classes = len(budgets.label_names)
    ids, having = _class_objects(part, n_classes)
    known = [set(ids[objs].tolist()) for objs in having]
    largest = budgets.budgets[-1]
    for repeat in range(budgets.n_repeats):
        where = f"part {part.name}, repeat {repeat}"
        classes = budgets.objects[p][repeat]
        for c, objs in enumerate(classes):
            name = budgets.label_names[c]
            if len(objs) != largest or len(set(objs)) != largest:
                raise BudgetError(f"{where}: class {name} does not hold {largest} objects")
            if not set(objs) <= known[c]:
                raise BudgetError(f"{where}: an object of class {name} is not in the part")
        for budget in budgets.budgets:
            subset = {obj for objs in classes for obj in objs[:budget]}
            for c in range(n_classes):
                chosen = budgets.selection[p][repeat][c][budget]
                name = budgets.label_names[c]
                if len(chosen) != budgets.selection_per_class[budget]:
                    raise BudgetError(
                        f"{where}, budget {budget}: the selection set of class {name} does "
                        f"not hold {budgets.selection_per_class[budget]} objects"
                    )
                if not set(chosen) <= known[c] or set(chosen) & subset:
                    raise BudgetError(
                        f"{where}, budget {budget}: a selection object of class {name} is "
                        "not in the part, or is in the subset"
                    )


def verify_label_budgets(
    budgets: LabelBudgets, parts: Sequence[BudgetPart], *, source: dict[str, Any]
) -> None:
    """Check that ``budgets`` belongs to ``parts`` and its parent, and is well drawn.

    ``source`` is what :func:`make_label_budgets` would store for the parent at hand (its
    kind, SHA-256 and data fingerprint; the file name is informational). ``parts`` are
    rebuilt from the **verified** parent (:func:`~astro_simclr.protocol.splits.
    verify_splits` or :func:`~astro_simclr.protocol.object_splits.verify_object_splits`)
    and the data's labels. Checks: the kind, parent SHA-256 and fingerprint; the same part
    names; per part, repeat and class, the largest budget's distinct objects all having
    that class in the part; per budget, selection sets of the stored size, of that class,
    outside the subset.

    Raises:
        BudgetError: naming the first mismatch.
    """
    for key in ("kind", "sha256", "fingerprint"):
        if budgets.source.get(key) != source.get(key):
            msg = (
                f"the label budgets were made from {budgets.source.get('file')!r} "
                f"({key} {budgets.source.get(key)!r}), not from this parent "
                f"({source.get(key)!r}); regenerate them with `astro-simclr make-label-budgets`"
            )
            raise BudgetError(msg)
    names = tuple(part.name for part in parts)
    if names != budgets.part_names:
        raise BudgetError(f"the file's parts {list(budgets.part_names)} are not {list(names)}")
    _check_parts(parts, len(budgets.label_names))
    for part in parts:
        _verify_part(budgets, part)


# --- The file ------------------------------------------------------------------------------


def _to_plain(budgets: LabelBudgets) -> dict[str, Any]:
    names = budgets.label_names
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "source": budgets.source,
        "label_names": list(names),
        "object_id": budgets.object_id,
        "seed": budgets.seed,
        "n_repeats": budgets.n_repeats,
        "budgets": list(budgets.budgets),
        "include_full": budgets.include_full,
        "selection_fraction": budgets.selection_fraction,
        "selection_per_class": {str(b): n for b, n in budgets.selection_per_class.items()},
        "parts": [
            {
                "name": part_name,
                "repeats": [
                    {
                        names[c]: {
                            "objects": list(budgets.objects[p][r][c]),
                            "selection": {
                                str(b): list(ids) for b, ids in budgets.selection[p][r][c].items()
                            },
                        }
                        for c in range(len(names))
                    }
                    for r in range(budgets.n_repeats)
                ],
            }
            for p, part_name in enumerate(budgets.part_names)
        ],
    }


def write_label_budgets(budgets: LabelBudgets, path: Path) -> None:
    """Write ``budgets`` as JSON to ``path`` (id lists on one line each; deterministic).

    Raises:
        OSError: if the file cannot be written.
    """
    text = json.dumps(_to_plain(budgets), indent=1)
    # One line per id list: hundreds of thousands of one-id lines help nobody.
    text = re.sub(
        r"\[\s*((?:-?\d+|\"[^\"\n]*\")(?:,\s*(?:-?\d+|\"[^\"\n]*\"))*)\s*\]",
        lambda match: "[" + re.sub(r",\s+", ", ", match.group(1)) + "]",
        text,
    )
    path.write_text(text + "\n", encoding="utf-8")


def _id_list(value: object, what: str) -> tuple[ObjectId, ...]:
    if not isinstance(value, list) or not all(
        (isinstance(v, int) and not isinstance(v, bool)) or isinstance(v, str) for v in value
    ):
        raise BudgetError(f"{what} must be a list of object ids (integers or strings)")
    return tuple(value)


def _positive(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BudgetError(f"{what} must be a positive integer")
    return value


def _parse_settings(raw: dict[str, Any]) -> tuple[tuple[int, ...], int, float, dict[int, int]]:
    budgets_raw = raw.get("budgets")
    if not isinstance(budgets_raw, list):
        raise BudgetError("budgets must be a list of integers")
    n_repeats = _positive(raw.get("n_repeats"), "n_repeats")
    fraction = raw.get("selection_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise BudgetError("selection_fraction must be a number")
    budgets = _check_settings(
        [_positive(b, "a budget") for b in budgets_raw], n_repeats, float(fraction)
    )
    sizes_raw = raw.get("selection_per_class")
    if not isinstance(sizes_raw, dict) or sorted(sizes_raw) != sorted(map(str, budgets)):
        raise BudgetError("selection_per_class must give one size per budget")
    sizes = {}
    for budget in budgets:
        size = sizes_raw[str(budget)]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BudgetError("selection_per_class sizes must be non-negative integers")
        if size > math.floor(round(float(fraction) * budget, 9)):
            raise BudgetError(f"budget {budget}: the selection set exceeds the fraction's cap")
        sizes[budget] = size
    return budgets, n_repeats, float(fraction), sizes


def _parse_repeat(
    raw: object, names: tuple[str, ...], budgets: tuple[int, ...], what: str
) -> tuple[tuple[tuple[ObjectId, ...], ...], tuple[dict[int, tuple[ObjectId, ...]], ...]]:
    if not isinstance(raw, dict) or sorted(raw) != sorted(names):
        raise BudgetError(f"{what} must have one entry per class {list(names)}")
    objects = []
    selection = []
    for name in names:
        entry = raw[name]
        if not isinstance(entry, dict) or not isinstance(entry.get("selection"), dict):
            raise BudgetError(f"{what}.{name} needs objects and selection")
        objects.append(_id_list(entry.get("objects"), f"{what}.{name}.objects"))
        chosen = entry["selection"]
        if sorted(chosen) != sorted(map(str, budgets)):
            raise BudgetError(f"{what}.{name}.selection must have one list per budget")
        selection.append(
            {b: _id_list(chosen[str(b)], f"{what}.{name}.selection.{b}") for b in budgets}
        )
    return tuple(objects), tuple(selection)


def read_label_budgets(path: Path) -> LabelBudgets:
    """Read and check the structure of a file written by :func:`write_label_budgets`.

    :func:`verify_label_budgets` checks it against the parent splits and the data.

    Raises:
        BudgetError: if the file cannot be read or is malformed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise BudgetError(f"{path}: cannot read the label-budgets file ({err})") from None
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise BudgetError(f"{path}: not a {FORMAT} file")
    if raw.get("format_version") != FORMAT_VERSION:
        raise BudgetError(f"{path}: unsupported format_version {raw.get('format_version')!r}")
    try:
        return _parse(raw)
    except BudgetError as err:
        raise BudgetError(f"{path}: {err}") from None


def _parse(raw: dict[str, Any]) -> LabelBudgets:
    source, names_raw, parts_raw = raw.get("source"), raw.get("label_names"), raw.get("parts")
    if not isinstance(source, dict) or source.get("kind") not in _SOURCE_KINDS:
        raise BudgetError(f"source must be an object with kind in {list(_SOURCE_KINDS)}")
    if (
        not isinstance(names_raw, list)
        or len(names_raw) < 2
        or not all(isinstance(n, str) for n in names_raw)
        or len(set(names_raw)) != len(names_raw)
    ):
        raise BudgetError("label_names must be two or more distinct strings")
    names = tuple(names_raw)
    budgets, n_repeats, fraction, sizes = _parse_settings(raw)
    seed, object_id, include_full = raw.get("seed"), raw.get("object_id"), raw.get("include_full")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BudgetError("seed must be an integer")
    if not isinstance(object_id, str) or not isinstance(include_full, bool):
        raise BudgetError("object_id must be a string and include_full a boolean")
    if not isinstance(parts_raw, list) or not parts_raw:
        raise BudgetError("parts must be a non-empty list")
    part_names = []
    objects = []
    selection = []
    for k, part in enumerate(parts_raw):
        if not isinstance(part, dict) or not isinstance(part.get("name"), str):
            raise BudgetError(f"parts[{k}] needs a name")
        repeats = part.get("repeats")
        if not isinstance(repeats, list) or len(repeats) != n_repeats:
            raise BudgetError(f"parts[{k}] must hold n_repeats ({n_repeats}) repeats")
        parsed = [
            _parse_repeat(rep, names, budgets, f"parts[{k}].repeats[{r}]")
            for r, rep in enumerate(repeats)
        ]
        part_names.append(part["name"])
        objects.append(tuple(p[0] for p in parsed))
        selection.append(tuple(p[1] for p in parsed))
    if len(set(part_names)) != len(part_names):
        raise BudgetError(f"part names repeat: {part_names}")
    return LabelBudgets(
        source=source,
        label_names=names,
        object_id=object_id,
        seed=seed,
        n_repeats=n_repeats,
        budgets=budgets,
        include_full=include_full,
        selection_fraction=fraction,
        selection_per_class=sizes,
        part_names=tuple(part_names),
        objects=tuple(objects),
        selection=tuple(selection),
    )
