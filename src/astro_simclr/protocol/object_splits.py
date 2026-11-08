"""Object-grouped, class-stratified Train/Validation/Test splits of an alert table, stored.

Task 3.2 of ``docs/phases/03-newer-benchmarks.md``. Several alerts of one object look alike,
so a split by alert would leak an object from Train into Test. These splits assign whole
**groups** to Train, Validation or Test (default 60/20/20 of each class's alerts), so no
group is in two splits. The group is the alert table's
:attr:`~astro_simclr.protocol.alerts.AlertTable.group`: the ``oid`` by default, or a
``group`` column (such as a position-group id made elsewhere) when there is no reliable
``oid``. The draw and its guarantee are :func:`astro_simclr.protocol.grouping.
assign_groups`: within every class, each split's alert count ends at most ``g`` above and
``(k - 1) g`` below its target, ``g`` being the class's largest group.

**A group has one label.** :func:`make_object_splits` refuses a table where a group's
alerts carry two labels (``on_conflict="refuse"``, the default) or leaves those groups out
of every split and records them (``on_conflict="drop"``).

**Decision halves.** Validation is also split, grouped and class-stratified, into a
*selection* half (checkpoint selection) and a *decision* half (50/50 of each class's
alerts, :func:`~astro_simclr.protocol.grouping.grouped_halves`), so that phase-2-style
decisions (``docs/evaluation-protocol.md``) are possible on the imbalanced 6-class set.

**Constraints** (task 3.5, the cross-benchmark rule). ``allowed`` restricts named groups to
some splits: R5 puts an object of CD21's Train only in its own Train, and one of CD21's
Validation or Test only in Validation or Test
(:func:`~astro_simclr.protocol.grouping.assign_groups`, "constraints"). The stratification
bound is then checked only for the classes without a constrained group; the constraints
themselves are checked for every group.

The splits are **stored**, and the stored file is the source of truth (as for the phase-2
splits, ADR 0004: numpy does not promise one seeded permutation across versions). File
format (JSON, ``format`` ``astro-simclr-object-splits``, version 2; version 1 files, which
have no constraints, are still read)::

    table        name (informational), sha256 (AlertTable.sha256), n_alerts, n_groups
    group_key    "oid", or "group" (the table's own group column)
    label_names, seed, fractions (Train, Validation, Test)
    conflicts    the groups left out for conflicting labels: {group, labels}
    constraints  the constrained groups: [{splits, groups}], one entry per set of allowed
                 splits (version 2)
    groups       per split, the sorted group ids
    halves       selection and decision: the sorted group ids of Validation's halves
    counts       per split and half, alerts and groups per class (for a reader; verified)

:func:`verify_object_splits` checks a file against an alert table: the fingerprint and the
group key, every group in exactly one split, the conflicts, the halves, the counts and the
stratification bound.

Nothing here imports torch.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.protocol.alerts import OID_PATTERN, AlertTable, LabelConflict, label_conflicts
from astro_simclr.protocol.grouping import (
    SPLIT_NAMES,
    check_fractions,
    check_stratified,
    group_rows,
    grouped_halves,
    grouped_split,
)
from astro_simclr.protocol.splits import SplitsError

__all__ = [
    "DEFAULT_FRACTIONS",
    "FORMAT",
    "FORMAT_VERSION",
    "HALF_NAMES",
    "SPLIT_NAMES",
    "ObjectSplits",
    "make_object_splits",
    "read_object_splits",
    "verify_object_splits",
    "write_object_splits",
]

#: ``format`` of a stored object-splits file.
FORMAT: Final = "astro-simclr-object-splits"
#: Bumped on any incompatible change of the file (2: ``constraints``).
FORMAT_VERSION: Final = 2
#: The versions :func:`read_object_splits` reads.
READ_VERSIONS: Final = (1, 2)
#: RJ23's stratified 60/20/20.
DEFAULT_FRACTIONS: Final = (0.6, 0.2, 0.2)
#: The two halves of Validation.
HALF_NAMES: Final = ("selection", "decision")

OnConflict = Literal["refuse", "drop"]
Counts = dict[str, dict[str, list[int]]]
Member = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ObjectSplits:
    """Stored object splits of one alert table (module docstring).

    ``groups`` maps each split name to its sorted group ids, ``halves`` each half name to
    Validation's; ``counts`` maps each split and half name to its ``alerts`` and
    ``groups`` per class, in label order; ``table`` holds ``name``, ``sha256``,
    ``n_alerts`` and ``n_groups``. ``constraints`` maps each constrained group to the
    splits it may be in (empty when the splits were drawn without constraints).
    """

    table: dict[str, Any]
    group_key: str
    label_names: tuple[str, ...]
    seed: int
    fractions: tuple[float, float, float]
    conflicts: tuple[LabelConflict, ...]
    groups: dict[str, tuple[str, ...]]
    halves: dict[str, tuple[str, ...]]
    counts: Counts
    constraints: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def split_of(self, table: AlertTable) -> Member:
        """Per alert row of ``table``: its split's index in :data:`SPLIT_NAMES`, or -1.

        -1 marks an alert whose group is in no split (a dropped conflict, or unknown).
        """
        return _member(table, SPLIT_NAMES, self.groups)

    def half_of(self, table: AlertTable) -> Member:
        """Per alert row: 0 (selection half), 1 (decision half), or -1 (not in Validation)."""
        return _member(table, HALF_NAMES, self.halves)

    def rows(self, table: AlertTable, part: str) -> Member:
        """The sorted alert rows of ``table`` in a split or a half of Validation.

        Raises:
            ValueError: if ``part`` is not in :data:`SPLIT_NAMES` or :data:`HALF_NAMES`.
        """
        if part in SPLIT_NAMES:
            member = self.split_of(table) == SPLIT_NAMES.index(part)
        elif part in HALF_NAMES:
            member = self.half_of(table) == HALF_NAMES.index(part)
        else:
            raise ValueError(f"unknown part {part!r}; use one of {SPLIT_NAMES + HALF_NAMES}")
        return np.flatnonzero(member).astype(np.int64)


def _member(table: AlertTable, names: Sequence[str], parts: dict[str, tuple[str, ...]]) -> Member:
    lookup = {g: k for k, name in enumerate(names) for g in parts[name]}
    return np.asarray([lookup.get(g, -1) for g in table.group.tolist()], dtype=np.int64)


def _counts(table: AlertTable, member: Member, names: Sequence[str]) -> Counts:
    n_classes = len(table.label_names)
    result = {}
    for k, name in enumerate(names):
        rows = member == k
        found = group_rows(table.group[rows], table.label[rows])
        result[name] = {
            "alerts": np.bincount(table.label[rows], minlength=n_classes).tolist(),
            "groups": np.bincount(found.label, minlength=n_classes).tolist(),
        }
    return result


def _all_counts(table: AlertTable, split_of: Member, half_of: Member) -> Counts:
    return {**_counts(table, split_of, SPLIT_NAMES), **_counts(table, half_of, HALF_NAMES)}


def _parts(table: AlertTable, member: Member, names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    return {n: tuple(np.unique(table.group[member == k]).tolist()) for k, n in enumerate(names)}


def make_object_splits(
    table: AlertTable,
    *,
    seed: int = 0,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    on_conflict: OnConflict = "refuse",
    name: str = "",
    allowed: Mapping[str, Collection[str]] | None = None,
) -> ObjectSplits:
    """Draw the grouped, class-stratified splits and Validation's halves of ``table``.

    ``name`` is recorded as the table's name (informational). ``allowed`` maps group ids to
    the split names they may be in (module docstring, "Constraints"); ids the table lacks
    are ignored. Deterministic for a given table, seed, constraints and numpy version.

    Raises:
        SplitsError: on invalid fractions (three positive numbers summing to 1) or
            ``on_conflict``, a group with two labels and ``on_conflict="refuse"``, a class
            too small for every split (and half) to get one of its groups, or an unknown
            split name in ``allowed``.
    """
    shares = check_fractions(fractions)
    if len(shares) != len(SPLIT_NAMES):
        raise SplitsError(f"fractions must be {len(SPLIT_NAMES)} numbers (Train, Val, Test)")
    if on_conflict not in ("refuse", "drop"):
        raise SplitsError(f"on_conflict must be refuse or drop, got {on_conflict!r}")
    constraints = _constraints(table, allowed)
    split_of = grouped_split(
        table.group,
        table.label,
        shares,
        seed=seed,
        drop_conflicts=on_conflict == "drop",
        names=table.label_names,
        allowed={g: [SPLIT_NAMES.index(n) for n in names] for g, names in constraints.items()},
    )
    validation = np.flatnonzero(split_of == SPLIT_NAMES.index("Validation"))
    selection, decision = grouped_halves(
        table.group[validation], table.label[validation], seed=seed, names=table.label_names
    )
    half_of = np.full(len(table), -1, dtype=np.int64)
    half_of[validation[selection]] = 0
    half_of[validation[decision]] = 1
    conflicting = _conflicting(table)
    return ObjectSplits(
        table={
            "name": name,
            "sha256": table.sha256,
            "n_alerts": len(table),
            "n_groups": int(np.unique(table.group).size),
        },
        group_key=table.group_key,
        label_names=table.label_names,
        seed=seed,
        fractions=(shares[0], shares[1], shares[2]),
        conflicts=label_conflicts(table),
        groups=_parts(table, split_of, SPLIT_NAMES),
        halves=_parts(table, half_of, HALF_NAMES),
        counts=_all_counts(table, split_of, half_of),
        constraints={g: names for g, names in constraints.items() if g not in conflicting},
    )


def _conflicting(table: AlertTable) -> set[str]:
    return {c.group for c in label_conflicts(table)}


def _constraints(
    table: AlertTable, allowed: Mapping[str, Collection[str]] | None
) -> dict[str, tuple[str, ...]]:
    """The constraints on the table's own groups, split names in :data:`SPLIT_NAMES` order.

    A group allowed in every split is not a constraint and is left out.
    """
    if not allowed:
        return {}
    present = set(table.group.tolist())
    result: dict[str, tuple[str, ...]] = {}
    for group, names in sorted(allowed.items()):
        unknown = set(names) - set(SPLIT_NAMES)
        if unknown or not names:
            raise SplitsError(f"group {group!r}: allowed splits {sorted(names)} are invalid")
        ordered = tuple(n for n in SPLIT_NAMES if n in names)
        if group in present and len(ordered) < len(SPLIT_NAMES):
            result[group] = ordered
    return result


def _constraint_entries(constraints: Mapping[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    by_splits: dict[tuple[str, ...], list[str]] = {}
    for group, names in constraints.items():
        by_splits.setdefault(names, []).append(group)
    return [
        {"splits": list(names), "groups": sorted(groups)}
        for names, groups in sorted(
            by_splits.items(), key=lambda kv: [SPLIT_NAMES.index(n) for n in kv[0]]
        )
    ]


def _to_plain(splits: ObjectSplits) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created_by": f"astro-simclr {version('astro-simclr')}",
        "table": splits.table,
        "group_key": splits.group_key,
        "label_names": list(splits.label_names),
        "seed": splits.seed,
        "fractions": dict(zip(SPLIT_NAMES, splits.fractions, strict=True)),
        "conflicts": [{"group": c.group, "labels": list(c.labels)} for c in splits.conflicts],
        "constraints": _constraint_entries(splits.constraints),
        "groups": {name: list(splits.groups[name]) for name in SPLIT_NAMES},
        "halves": {name: list(splits.halves[name]) for name in HALF_NAMES},
        "counts": splits.counts,
    }


def write_object_splits(splits: ObjectSplits, path: Path) -> None:
    """Write ``splits`` as JSON (each list on one line; deterministic).

    Raises:
        OSError: if the file cannot be written.
    """
    text = json.dumps(_to_plain(splits), indent=2)
    # One line per list of numbers or ids: tens of thousands of one-item lines help nobody.
    text = re.sub(
        r"\[\s*((?:\"[^\"]*\"|-?\d+)(?:,\s*(?:\"[^\"]*\"|-?\d+))*)\s*\]",
        lambda match: "[" + ", ".join(re.split(r",\s*", match.group(1))) + "]",
        text,
    )
    path.write_text(text + "\n", encoding="utf-8")


def _id_list(value: object, what: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(v, str) and OID_PATTERN.fullmatch(v) for v in value
    ):
        raise SplitsError(f"{path}: {what} must be a list of valid group ids")
    if value != sorted(set(value)):
        raise SplitsError(f"{path}: {what} must be sorted and without repeats")
    return tuple(value)


def _parse_conflicts(
    value: object, names: tuple[str, ...], path: Path
) -> tuple[LabelConflict, ...]:
    if not isinstance(value, list):
        raise SplitsError(f"{path}: conflicts must be a list")
    conflicts = []
    for item in value:
        group = item.get("group") if isinstance(item, dict) else None
        labels = item.get("labels") if isinstance(item, dict) else None
        if (
            not isinstance(group, str)
            or not OID_PATTERN.fullmatch(group)
            or not isinstance(labels, list)
            or len(labels) < 2
            or not all(isinstance(label, str) and label in names for label in labels)
        ):
            raise SplitsError(f"{path}: each conflict needs a group and 2+ known labels")
        conflicts.append(LabelConflict(group, tuple(labels)))
    return tuple(conflicts)


def _parse_constraints(value: object, path: Path) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, list):
        raise SplitsError(f"{path}: constraints must be a list")
    result: dict[str, tuple[str, ...]] = {}
    for entry in value:
        names = entry.get("splits") if isinstance(entry, dict) else None
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) or n not in SPLIT_NAMES for n in names)
            or names != [n for n in SPLIT_NAMES if n in names]
            or len(names) == len(SPLIT_NAMES)
        ):
            raise SplitsError(f"{path}: each constraint needs 1-2 split names, in order")
        assert isinstance(entry, dict)  # noqa: S101 - narrowed above, for mypy
        for group in _id_list(entry.get("groups"), "constraints.groups", path):
            if group in result:
                raise SplitsError(f"{path}: group {group} has two constraints")
            result[group] = tuple(names)
    return result


def _disjoint(parts: dict[str, tuple[str, ...]], what: str, path: Path) -> set[str]:
    seen: set[str] = set()
    for name, ids in parts.items():
        if seen.intersection(ids):
            raise SplitsError(f"{path}: a group is in two {what} ({name} and an earlier one)")
        seen.update(ids)
    return seen


def _parse(raw: dict[str, Any], path: Path) -> ObjectSplits:
    try:
        names = tuple(str(n) for n in raw["label_names"])
        shares = check_fractions([raw["fractions"][name] for name in SPLIT_NAMES])
        counts = {
            n: {kind: [int(v) for v in raw["counts"][n][kind]] for kind in ("alerts", "groups")}
            for n in (*SPLIT_NAMES, *HALF_NAMES)
        }
        return ObjectSplits(
            table=dict(raw["table"]),
            group_key=raw["group_key"],
            label_names=names,
            seed=raw["seed"],
            fractions=(shares[0], shares[1], shares[2]),
            conflicts=_parse_conflicts(raw["conflicts"], names, path),
            groups={n: _id_list(raw["groups"][n], f"groups.{n}", path) for n in SPLIT_NAMES},
            halves={n: _id_list(raw["halves"][n], f"halves.{n}", path) for n in HALF_NAMES},
            counts=counts,
            constraints=_parse_constraints(raw.get("constraints", []), path),
        )
    except (KeyError, TypeError, ValueError, IndexError) as err:
        raise SplitsError(f"{path}: malformed object-splits file ({err!r})") from None


def read_object_splits(path: Path) -> ObjectSplits:
    """Read and check an object-splits file written by :func:`write_object_splits`.

    Checks the structure only (types, valid and sorted ids, disjoint splits and halves,
    halves inside Validation, fractions); :func:`verify_object_splits` checks it against an
    alert table.

    Raises:
        SplitsError: if the file cannot be read or is malformed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise SplitsError(f"{path}: cannot read the object-splits file ({err})") from None
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise SplitsError(f"{path}: not a {FORMAT} file")
    if raw.get("format_version") not in READ_VERSIONS:
        raise SplitsError(f"{path}: unsupported format_version {raw.get('format_version')!r}")
    splits = _parse(raw, path)
    seed = splits.seed
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or splits.group_key
        not in (
            "oid",
            "group",
        )
    ):
        raise SplitsError(f"{path}: seed must be an integer and group_key oid or group")
    table = splits.table
    if not isinstance(table.get("sha256"), str) or not isinstance(table.get("n_alerts"), int):
        raise SplitsError(f"{path}: table needs sha256 and n_alerts")
    seen = _disjoint(splits.groups, "splits", path)
    if not _disjoint(splits.halves, "halves", path) <= set(splits.groups["Validation"]):
        raise SplitsError(f"{path}: the halves must hold Validation groups only")
    if seen.intersection(c.group for c in splits.conflicts):
        raise SplitsError(f"{path}: a conflicting group is also in a split")
    for group, names in splits.constraints.items():
        if not any(group in splits.groups[n] for n in names):
            raise SplitsError(f"{path}: group {group} is outside its allowed splits {names}")
    return splits


def _check_identity(splits: ObjectSplits, table: AlertTable) -> None:
    if splits.table.get("sha256") != table.sha256 or splits.table.get("n_alerts") != len(table):
        msg = (
            "the object splits were made for another alert table "
            f"({splits.table.get('name')!r}, sha256 {splits.table.get('sha256')}), not this "
            f"one (sha256 {table.sha256}); regenerate them with `astro-simclr "
            "make-object-splits`"
        )
        raise SplitsError(msg)
    if splits.group_key != table.group_key or splits.label_names != table.label_names:
        raise SplitsError(
            f"group key or label names differ: {splits.group_key}, {splits.label_names} vs "
            f"{table.group_key}, {table.label_names}"
        )
    if splits.conflicts != label_conflicts(table):
        raise SplitsError("the recorded label conflicts are not the table's")


def verify_object_splits(splits: ObjectSplits, table: AlertTable) -> None:
    """Check that ``splits`` belongs to ``table`` and is well drawn.

    - The table's fingerprint (``sha256``), size, group key and label names match.
    - The conflicting groups are exactly the table's, and in no split; every other group of
      the table is in exactly one split, and no split names a group the table lacks; the
      halves cover Validation.
    - Within every class, every split (and half) holds at least one alert, within the
      draw's bound of its target (:func:`~astro_simclr.protocol.grouping.check_stratified`).
    - The stored per-part class counts are the table's.

    Raises:
        SplitsError: on any mismatch, naming it.
    """
    _check_identity(splits, table)
    expected = set(table.group.tolist()) - {c.group for c in splits.conflicts}
    stored = {g for name in SPLIT_NAMES for g in splits.groups[name]}
    if stored != expected:
        extra, missing = sorted(stored - expected), sorted(expected - stored)
        raise SplitsError(
            f"the splits do not cover the table's groups: {len(missing)} missing "
            f"({missing[:3]}), {len(extra)} unknown ({extra[:3]})"
        )
    halves = {g for name in HALF_NAMES for g in splits.halves[name]}
    if halves != set(splits.groups["Validation"]):
        raise SplitsError("the halves do not cover Validation's groups")
    if not set(splits.constraints) <= expected:
        raise SplitsError("a constrained group is not a group of the table")
    split_of, half_of = splits.split_of(table), splits.half_of(table)
    names = table.label_names
    for group, allowed in splits.constraints.items():
        if not any(group in splits.groups[n] for n in allowed):
            raise SplitsError(f"group {group} is outside its allowed splits {allowed}")
    # The bound holds only for the classes without a constrained group (module docstring).
    constrained = np.isin(table.group, list(splits.constraints))
    free = ~np.isin(table.label, np.unique(table.label[constrained]))
    check_stratified(
        table.group[free],
        table.label[free],
        split_of[free],
        splits.fractions,
        names=names,
        part_names=SPLIT_NAMES,
    )
    val = split_of == SPLIT_NAMES.index("Validation")
    check_stratified(
        table.group[val],
        table.label[val],
        half_of[val],
        (0.5, 0.5),
        names=names,
        part_names=HALF_NAMES,
    )
    if splits.counts != _all_counts(table, split_of, half_of):
        raise SplitsError("the stored per-part class counts are not the table's")
