"""Group-aware, class-stratified assignment on per-row arrays, and the cross-benchmark rule.

Task 3.2 of ``docs/phases/03-newer-benchmarks.md``. Several rows (alerts, stamps) of one
*group* look alike: the alerts of one object (group key ``oid``) or, where there is no
``oid``, the stamps of one sky position (a position-group id, made elsewhere: this module
never builds groups, it takes **any per-row group-id array**). A split by row would put one
group on both sides, so these functions assign whole groups and stratify by class:

- :func:`group_rows`: the groups of a set of rows, each group's label and size, and the
  groups whose rows carry more than one label (a group has one class);
- :func:`assign_groups`: the core draw. Per class, in label order, with its own
  ``default_rng([seed, stream, class])``: shuffle the class's groups, then give each one, in
  that order, to the part whose row count is furthest below its target (``fraction x the
  class's rows``; ties to the earlier part). The order is random, not by size: a
  largest-first order balances the rows a little better, but sends the large groups to the
  first part and the small ones to the others, so the parts would differ in rows per group.
  A part is never chosen once it is at or above its target, so in every class no part
  ends more than ``g`` rows above its target and none more than ``(k - 1) g`` below
  (:func:`stratification_bound`; ``g`` is the class's largest group, ``k`` the number of
  parts);
- **constraints** (task 3.5): ``allowed`` restricts some groups to some parts, as the
  cross-benchmark rule needs (an R5 object that is in CD21's Train may only go to R5's
  Train). Within a class, the groups with one allowed part are placed first, then those
  with fewer allowed parts, then the free ones, each tier in the class's random order, and
  each group goes to the allowed part furthest below its target. The free groups therefore
  make up, as far as they can, for what the constraints push into one part; the bound above
  no longer holds for a class with constrained groups (:func:`check_stratified` is only
  meaningful for the others). Without ``allowed`` the draw is unchanged, number for number;
- :func:`grouped_split`: rows -> part, refusing (or dropping) groups with two labels;
  :func:`grouped_halves`: the 50/50 *selection* and *decision* halves of a set, for
  phase-2-style decisions on an imbalanced N-class set;
- :func:`check_stratified`: the bound above, and every part holding every class;
- :func:`shared_groups` / :func:`assert_no_shared_groups`: the **cross-benchmark rule**.
  Given the per-row group ids and split membership of two benchmarks (in one id space, such
  as ``oid``, or a position group built across both), count the groups each of the 3 x 3
  split pairs shares, and refuse the forbidden pairs: by default the phase-3 plan's matrix
  (:data:`CROSS_BENCHMARK_ALLOWED`: a group may be in both Trains, or in the Validation or
  Test of both; never in one's Train and the other's Validation or Test, either way). The
  pretraining exclusion lists are built from it.

Group ids may be strings or integers; they are compared as given. Nothing here imports
torch.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.protocol.splits import SplitsError

__all__ = [
    "CROSS_BENCHMARK_ALLOWED",
    "SPLIT_NAMES",
    "GroupRows",
    "assert_no_shared_groups",
    "assign_groups",
    "check_stratified",
    "group_rows",
    "grouped_halves",
    "grouped_split",
    "shared_groups",
    "stratification_bound",
]

#: The three parts of a benchmark, in tie order.
SPLIT_NAMES: Final = ("Train", "Validation", "Test")
#: The (split of A, split of B) pairs that may share groups (phase-3 plan, task 3.2):
#: Train/Train, and any pair of Validation and Test. Train with Validation or Test, in
#: either direction, is forbidden: a model trained on one benchmark would see an object of
#: the other's evaluation sets.
CROSS_BENCHMARK_ALLOWED: Final = frozenset(
    {
        ("Train", "Train"),
        ("Validation", "Validation"),
        ("Validation", "Test"),
        ("Test", "Validation"),
        ("Test", "Test"),
    }
)
#: Stream of :func:`grouped_halves`'s generator (:func:`grouped_split` uses 0 by default).
HALVES_STREAM: Final = 1
#: Tolerance of float comparisons of counts with targets, and of the fractions' sum.
_EPS: Final = 1e-9

Rows = npt.NDArray[np.int64]
Mask = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class GroupRows:
    """The groups of some rows: sorted ids, each one's label and row count, and conflicts.

    ``label`` of a conflicting group is its smallest label; ``conflicts`` maps each
    conflicting id to its sorted labels.
    """

    ids: npt.NDArray[Any]
    label: npt.NDArray[np.int64]
    size: npt.NDArray[np.int64]
    conflicts: dict[Any, tuple[int, ...]]


def _arrays(groups: npt.ArrayLike, labels: npt.ArrayLike) -> tuple[npt.NDArray[Any], Rows]:
    ids, classes = np.asarray(groups), np.asarray(labels)
    if ids.ndim != 1 or ids.shape != classes.shape:
        raise ValueError(
            f"groups and labels must be 1-D of one shape, got {ids.shape}, {classes.shape}"
        )
    if classes.size and (not np.issubdtype(classes.dtype, np.integer) or classes.min() < 0):
        raise ValueError("labels must be non-negative integers")
    return ids, classes.astype(np.int64)


def group_rows(groups: npt.ArrayLike, labels: npt.ArrayLike) -> GroupRows:
    """The groups of the rows, with label, size and label conflicts (class docstring).

    Raises:
        ValueError: if the arrays are not 1-D of one shape or the labels are not
            non-negative integers.
    """
    ids, classes = _arrays(groups, labels)
    unique, inverse, size = np.unique(ids, return_inverse=True, return_counts=True)
    low = np.full(unique.size, np.iinfo(np.int64).max, dtype=np.int64)
    high = np.full(unique.size, -1, dtype=np.int64)
    np.minimum.at(low, inverse, classes)
    np.maximum.at(high, inverse, classes)
    conflicts = {}
    for g in np.flatnonzero(low != high):
        conflicts[unique[g].item()] = tuple(sorted(set(classes[inverse == g].tolist())))
    return GroupRows(unique, low, size.astype(np.int64), conflicts)


def check_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    """The fractions as floats: at least 2, each finite and positive, summing to 1.

    Raises:
        SplitsError: otherwise.
    """
    values = tuple(float(f) for f in fractions)
    if (
        len(values) < 2
        or not all(np.isfinite(f) and f > 0 for f in values)
        or abs(sum(values) - 1) > _EPS
    ):
        raise SplitsError(f"fractions must be 2+ positive numbers summing to 1, got {values}")
    return values


def stratification_bound(sizes: npt.ArrayLike, n_parts: int = len(SPLIT_NAMES)) -> float:
    """How far below its target a part's row count of one class may end: ``(k - 1) g``.

    ``sizes`` are the class's group sizes, ``g`` the largest, ``k = n_parts``; no part ends
    more than ``g`` above its target. With singletons every part is within ``k - 1`` rows.
    """
    values = np.asarray(sizes)
    return float((n_parts - 1) * values.max()) if values.size else 0.0


def _allowed_mask(allowed: npt.ArrayLike | None, n_groups: int, n_parts: int) -> Mask | None:
    if allowed is None:
        return None
    mask = np.asarray(allowed, dtype=bool)
    if mask.shape != (n_groups, n_parts):
        raise ValueError(f"allowed must be ({n_groups}, {n_parts}) booleans, got {mask.shape}")
    if not mask.any(axis=1).all():
        raise SplitsError(f"group {int(np.flatnonzero(~mask.any(axis=1))[0])} has no allowed part")
    return mask


def assign_groups(
    sizes: npt.ArrayLike,
    labels: npt.ArrayLike,
    fractions: Sequence[float],
    *,
    seed: int,
    stream: int = 0,
    names: Sequence[str] | None = None,
    allowed: npt.ArrayLike | None = None,
) -> Rows:
    """The part of each group (module docstring, the core draw); deterministic per seed.

    ``sizes`` and ``labels`` are per group. ``names`` (class names by label) only make the
    error message readable. ``allowed``, optional, is a ``(groups, parts)`` boolean mask of
    the parts each group may go to (module docstring, "constraints").

    Raises:
        SplitsError: on invalid fractions, a group with no allowed part, or a class with
            groups too few for every part to get one.
        ValueError: if ``allowed`` has the wrong shape.
    """
    shares = np.asarray(check_fractions(fractions))
    size_array = np.asarray(sizes, dtype=np.int64)
    label_array = np.asarray(labels, dtype=np.int64)
    mask = _allowed_mask(allowed, size_array.size, shares.size)
    assigned = np.full(size_array.size, -1, dtype=np.int64)
    for c in np.unique(label_array).tolist():
        members = np.flatnonzero(label_array == c)
        rng = np.random.default_rng([seed, stream, c])
        targets = shares * size_array[members].sum()
        current = np.zeros(shares.size)
        order = members[rng.permutation(members.size)]
        if mask is not None:
            # The most constrained groups first; a stable sort keeps the random order
            # within a tier.
            order = order[np.argsort(mask[order].sum(axis=1), kind="stable")]
        for item in order:
            gap = targets - current
            if mask is not None:
                gap = np.where(mask[item], gap, -np.inf)
            part = int(np.argmax(gap))
            assigned[item] = part
            current[part] += size_array[item]
        if np.count_nonzero(current) < shares.size:
            name = names[c] if names is not None and c < len(names) else f"class {c}"
            msg = (
                f"{name} has {members.size} groups: too few for each of the "
                f"{shares.size} parts to get one"
            )
            if mask is not None and not mask[members].any(axis=0).all():
                empty = [int(k) for k in np.flatnonzero(~mask[members].any(axis=0))]
                msg = f"{name}: no group is allowed in part(s) {empty} by the constraints"
            raise SplitsError(msg)
    return assigned


def _refuse_conflicts(found: GroupRows, names: Sequence[str] | None) -> None:
    def label(c: int) -> str:
        return names[c] if names is not None and c < len(names) else str(c)

    items = sorted(found.conflicts.items(), key=lambda kv: str(kv[0]))
    shown = "; ".join(f"{g}: {', '.join(map(label, ls))}" for g, ls in items[:10])
    msg = (
        f"{len(items)} groups have rows with different labels ({shown}); a group has one "
        "label: fix the labels or drop those groups"
    )
    raise SplitsError(msg)


def grouped_split(
    groups: npt.ArrayLike,
    labels: npt.ArrayLike,
    fractions: Sequence[float],
    *,
    seed: int,
    stream: int = 0,
    drop_conflicts: bool = False,
    names: Sequence[str] | None = None,
    allowed: Mapping[Any, Collection[int]] | None = None,
) -> Rows:
    """Per row, its part (index into ``fractions``), or -1 for a dropped conflicting group.

    ``allowed`` maps a group id to the parts (indices) it may go to; groups not in it may
    go anywhere (module docstring, "constraints").

    Raises:
        SplitsError: on a group with two labels (unless ``drop_conflicts``), or as
            :func:`assign_groups`.
        ValueError: as :func:`group_rows`, or on a part index outside ``fractions``.
    """
    ids, _ = _arrays(groups, labels)
    found = group_rows(groups, labels)
    if found.conflicts and not drop_conflicts:
        _refuse_conflicts(found, names)
    keep = ~np.isin(found.ids, list(found.conflicts))
    part_of_group = np.full(found.ids.size, -1, dtype=np.int64)
    mask = None
    if allowed:
        mask = np.ones((found.ids.size, len(fractions)), dtype=bool)
        for i, group in enumerate(found.ids.tolist()):
            parts = allowed.get(group)
            if parts is not None:
                if any(not 0 <= k < len(fractions) for k in parts):
                    raise ValueError(f"group {group!r}: allowed parts {sorted(parts)} out of range")
                mask[i] = False
                mask[i, list(parts)] = True
        mask = mask[keep]
    part_of_group[keep] = assign_groups(
        found.size[keep],
        found.label[keep],
        fractions,
        seed=seed,
        stream=stream,
        names=names,
        allowed=mask,
    )
    return part_of_group[np.searchsorted(found.ids, ids)]


def grouped_halves(
    groups: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    seed: int,
    names: Sequence[str] | None = None,
) -> tuple[Rows, Rows]:
    """The *selection* and *decision* halves of the rows: sorted row positions.

    Grouped and class-stratified (50/50 of each class's rows, :func:`assign_groups` on its
    own stream), so a decision half never shares a group with the selection half that chose
    the checkpoint: the phase-2 screen, for an imbalanced N-class set.

    Raises:
        SplitsError: on a group with two labels, or a class with fewer than 2 groups.
        ValueError: as :func:`group_rows`.
    """
    part = grouped_split(groups, labels, (0.5, 0.5), seed=seed, stream=HALVES_STREAM, names=names)
    return np.flatnonzero(part == 0).astype(np.int64), np.flatnonzero(part == 1).astype(np.int64)


def check_stratified(
    groups: npt.ArrayLike,
    labels: npt.ArrayLike,
    part: npt.ArrayLike,
    fractions: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    part_names: Sequence[str] | None = None,
) -> None:
    """Check a grouped split: no group in two parts, and the stratification bound.

    ``part`` is per row (-1: in no part). For every class with rows in a part, each part
    holds at least one of its rows, at most ``g`` above and ``(k - 1) g`` below its target
    (:func:`stratification_bound`).

    Raises:
        SplitsError: naming the first violation.
        ValueError: as :func:`group_rows`.
    """
    shares = check_fractions(fractions)
    ids, classes = _arrays(groups, labels)
    parts = np.asarray(part, dtype=np.int64)
    used = parts >= 0
    # A group "labelled" with two parts is a group in two parts.
    if group_rows(ids[used], parts[used]).conflicts:
        raise SplitsError("a group has rows in two parts")
    found = group_rows(ids[used], classes[used])
    for c in np.unique(found.label).tolist():
        sizes = found.size[found.label == c]
        total = int(sizes.sum())
        for k, share in enumerate(shares):
            count = int(np.count_nonzero(used & (parts == k) & (classes == c)))
            if (
                count == 0
                or not -stratification_bound(sizes, len(shares)) - _EPS
                <= count - share * total
                <= int(sizes.max()) + _EPS
            ):
                name = names[c] if names is not None and c < len(names) else f"class {c}"
                where = part_names[k] if part_names is not None else f"part {k}"
                msg = (
                    f"{where} holds {count} {name} rows, outside the bound of its target "
                    f"{share * total:.1f} (largest group: {int(sizes.max())} rows)"
                )
                raise SplitsError(msg)


def shared_groups(
    groups_a: npt.ArrayLike,
    split_a: npt.ArrayLike,
    groups_b: npt.ArrayLike,
    split_b: npt.ArrayLike,
    *,
    n_splits: int = len(SPLIT_NAMES),
) -> list[list[npt.NDArray[Any]]]:
    """The groups shared by split ``i`` of benchmark A and split ``j`` of B, for all i, j.

    ``groups_*`` and ``split_*`` are per row of each benchmark (split indices into
    :data:`SPLIT_NAMES`, -1 for a row in no split). Returns an ``n_splits x n_splits`` table
    of sorted id arrays.

    Raises:
        ValueError: if a benchmark's arrays are not 1-D of one shape.
    """

    def members(groups: npt.ArrayLike, split: npt.ArrayLike) -> list[npt.NDArray[Any]]:
        ids, parts = np.asarray(groups), np.asarray(split, dtype=np.int64)
        if ids.ndim != 1 or ids.shape != parts.shape:
            raise ValueError(
                f"groups and split must be 1-D of one shape, got {ids.shape}, {parts.shape}"
            )
        return [np.unique(ids[parts == s]) for s in range(n_splits)]

    a, b = members(groups_a, split_a), members(groups_b, split_b)
    return [[np.intersect1d(a[i], b[j]) for j in range(n_splits)] for i in range(n_splits)]


def assert_no_shared_groups(
    groups_a: npt.ArrayLike,
    split_a: npt.ArrayLike,
    groups_b: npt.ArrayLike,
    split_b: npt.ArrayLike,
    *,
    allowed: Collection[tuple[str, str]] = CROSS_BENCHMARK_ALLOWED,
) -> npt.NDArray[np.int64]:
    """The cross-benchmark rule: no group in a forbidden pair of splits of A and B.

    ``allowed`` lists the ``(split of A, split of B)`` pairs that may share groups; the
    default is the plan's matrix (:data:`CROSS_BENCHMARK_ALLOWED`), and ``allowed=()``
    refuses all nine pairs. Returns the 3 x 3 matrix of shared-group counts (allowed pairs
    included), from which exclusion lists are built.

    Raises:
        SplitsError: naming every refused pair that shares groups, with a few ids.
        ValueError: as :func:`shared_groups`, or on an unknown split name in ``allowed``.
    """
    unknown = {name for pair in allowed for name in pair} - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown split names in allowed: {sorted(unknown)}")
    table = shared_groups(groups_a, split_a, groups_b, split_b)
    counts = np.array([[cell.size for cell in row] for row in table], dtype=np.int64)
    refused = [
        f"A {SPLIT_NAMES[i]} / B {SPLIT_NAMES[j]}: {cell.size} ({cell[:3].tolist()})"
        for i, row in enumerate(table)
        for j, cell in enumerate(row)
        if cell.size and (SPLIT_NAMES[i], SPLIT_NAMES[j]) not in allowed
    ]
    if refused:
        raise SplitsError("the benchmarks share groups: " + "; ".join(refused))
    return counts
