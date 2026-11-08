"""The per-alert table of the phase-3 benchmarks: reading, fingerprint, objects, first stamps.

Task 3.2 of ``docs/phases/03-newer-benchmarks.md``. The object-grouped splits
(:mod:`astro_simclr.protocol.object_splits`) and the benchmark report
(:mod:`astro_simclr.protocol.bench_report`) read one row per labelled alert from a CSV
file, which the index of task 3.1 (``fetch-alerts``) can feed. Columns (header required,
any order; other columns, such as ``fid``, ``split`` and ``row``, are ignored):

==========  ==========================================================================
``oid``     the ZTF object id (``ZTF18abcdefg``): letters, digits, ``_``, ``-`` or ``.``,
            1-64 characters
``candid``  the ZTF alert id: a positive integer below 2**63, unique in the table
``jd``      the alert's Julian date: a finite number > 0
``label``   the class name, one of the label names in force (R5's 5 by default)
``group``   optional: the split's group key, when it is not the ``oid`` (for example a
            position-group id made elsewhere); same character set as ``oid``
==========  ==========================================================================

The file comes from downloaded data, so it is **untrusted**: it is parsed with this schema
only (the ``csv`` module, no evaluation of any field), and every row that breaks the
schema is an error naming its line.

:class:`AlertTable` keeps the rows sorted by ``candid``, so that its fingerprint
(:attr:`AlertTable.sha256`, SHA-256 of the canonical rows, with the group when there is a
``group`` column) does not depend on the file's row order. Per-alert predictions refer to
alerts by ``candid``. The splits group rows by :attr:`AlertTable.group` (the ``oid`` unless a
``group`` column is given: :attr:`AlertTable.group_key`); the first-stamp view always
groups by ``oid`` (the object).

**First stamp** (:func:`first_stamps`): the earliest alert of each object *in the table*
(smallest ``jd``; ties broken by the smaller ``candid``), which is RJ23's "first stamp"
view only if the table holds each object's first detection.

Nothing here imports torch.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.data.schema import LABEL_NAMES, DataError

__all__ = [
    "BENCHMARK_LABEL_NAMES",
    "COLUMNS",
    "OID_PATTERN",
    "RJ23_LABEL_NAMES",
    "AlertTable",
    "LabelConflict",
    "first_stamps",
    "label_conflicts",
    "read_alert_table",
    "write_alert_table",
]

#: The required columns of an alert table.
COLUMNS: Final = ("oid", "candid", "jd", "label")
#: The classes of R5, the phase-3 benchmark (RJ23's protocol without Satellite): the
#: phase-2 five, in their label order.
BENCHMARK_LABEL_NAMES: Final[tuple[str, ...]] = LABEL_NAMES
#: RJ23's 6 classes: the five, then Satellite (out of phase 3's scope; for a later 6-class
#: set).
RJ23_LABEL_NAMES: Final[tuple[str, ...]] = (*LABEL_NAMES, "Satellite")
#: A valid ``oid``.
OID_PATTERN: Final = re.compile(r"[A-Za-z0-9_.\-]{1,64}")
_CANDID: Final = re.compile(r"[0-9]{1,19}")
_MAX_CANDID: Final = 2**63 - 1


@dataclass(frozen=True, slots=True)
class LabelConflict:
    """A group whose alerts carry more than one label (a group has one class)."""

    group: str
    labels: tuple[str, ...]


def _check_ids(what: str, values: Sequence[str], n: int) -> None:
    if len(values) != n:
        raise DataError(f"{what} must have one value per row")
    bad = next((v for v in values if not OID_PATTERN.fullmatch(v)), None)
    if bad is not None:
        raise DataError(f"invalid {what} {bad!r}")


@dataclass(frozen=True, slots=True)
class AlertTable:
    """Labelled alerts sorted by ``candid`` (module docstring); arrays of one length N.

    ``label`` holds indices into ``label_names``; ``group`` is the split's group key per
    row, the ``oid`` itself when ``group_key`` is ``"oid"``. Build it with :meth:`from_rows`
    (or :func:`read_alert_table`), which checks and sorts the rows.
    """

    oid: npt.NDArray[np.str_]
    candid: npt.NDArray[np.int64]
    jd: npt.NDArray[np.float64]
    label: npt.NDArray[np.int64]
    label_names: tuple[str, ...]
    group: npt.NDArray[np.str_]
    group_key: Literal["oid", "group"]

    @classmethod
    def from_rows(
        cls,
        oid: Sequence[str],
        candid: Sequence[int],
        jd: Sequence[float],
        label: Sequence[int],
        label_names: Sequence[str],
        group: Sequence[str] | None = None,
    ) -> AlertTable:
        """Check the rows and sort them by ``candid``; ``group`` defaults to the ``oid``.

        Raises:
            DataError: on an empty table, unequal lengths, an invalid ``oid`` or group, a
                repeated or out-of-range ``candid``, a non-finite or non-positive ``jd``, a
                label outside ``range(len(label_names))``, or repeated label names.
        """
        names = tuple(label_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise DataError(f"need at least 2 distinct label names, got {names}")
        n = len(oid)
        if n == 0 or not len(candid) == len(jd) == len(label) == n:
            raise DataError("an alert table needs N > 0 rows with oid, candid, jd and label")
        _check_ids("oid", oid, n)
        if group is not None:
            _check_ids("group", group, n)
        ids = np.asarray(candid, dtype=np.int64)
        times = np.asarray(jd, dtype=np.float64)
        classes = np.asarray(label, dtype=np.int64)
        if ids.min() <= 0:
            raise DataError("candid must be positive")
        if not np.all(np.isfinite(times)) or times.min() <= 0:
            raise DataError("jd must be finite and positive")
        if classes.min() < 0 or classes.max() >= len(names):
            raise DataError(f"labels must be in range({len(names)})")
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        repeated = ids[1:][ids[1:] == ids[:-1]]
        if repeated.size:
            raise DataError(f"candid {int(repeated[0])} appears more than once")
        oids = np.asarray(oid, dtype=np.str_)[order]
        return cls(
            oids,
            ids,
            times[order],
            classes[order],
            names,
            oids if group is None else np.asarray(group, dtype=np.str_)[order],
            "oid" if group is None else "group",
        )

    def __len__(self) -> int:
        return int(self.candid.size)

    @property
    def sha256(self) -> str:
        """SHA-256 of the rows ``oid,candid,repr(jd),label name[,group]``, by ``candid``.

        The group is part of a row only with a ``group`` column (``group_key == "group"``).
        """
        digest = hashlib.sha256()
        rows = zip(self.oid, self.candid, self.jd, self.label, self.group, strict=True)
        for oid, candid, jd, label, group in rows:
            line = f"{oid},{int(candid)},{float(jd)!r},{self.label_names[int(label)]}"
            if self.group_key == "group":
                line += f",{group}"
            digest.update((line + "\n").encode())
        return digest.hexdigest()

    def rows_of(self, candids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """The row positions of ``candids``, in the given order.

        Raises:
            DataError: if a ``candid`` is not in the table.
        """
        wanted = np.asarray(candids, dtype=np.int64)
        rows = np.searchsorted(self.candid, wanted)
        clipped = np.minimum(rows, len(self) - 1)
        missing = wanted[self.candid[clipped] != wanted]
        if missing.size:
            raise DataError(f"candid {int(missing[0])} is not in the alert table")
        return clipped.astype(np.int64)


def _parse_row(
    row: dict[str, str | None], line: int, index: dict[str, int], path: Path
) -> tuple[str, int, float, int, str | None]:
    def field(name: str) -> str:
        value = row.get(name)
        if value is None:
            raise DataError(f"{path}:{line}: missing {name}")
        return value.strip()

    oid, candid, jd, label = (field(name) for name in COLUMNS)
    if not OID_PATTERN.fullmatch(oid):
        raise DataError(f"{path}:{line}: invalid oid {oid!r}")
    if not _CANDID.fullmatch(candid) or not 0 < int(candid) <= _MAX_CANDID:
        raise DataError(f"{path}:{line}: candid must be a positive integer below 2**63")
    try:
        time = float(jd)
    except ValueError:
        raise DataError(f"{path}:{line}: jd {jd!r} is not a number") from None
    if not math.isfinite(time) or time <= 0:
        raise DataError(f"{path}:{line}: jd must be finite and positive")
    if label not in index:
        raise DataError(f"{path}:{line}: unknown label {label!r}; expected one of {list(index)}")
    group = row.get("group")
    if group is not None:
        group = group.strip()
        if not OID_PATTERN.fullmatch(group):
            raise DataError(f"{path}:{line}: invalid group {group!r}")
    return oid, int(candid), time, index[label], group


def read_alert_table(path: Path, label_names: Sequence[str] = BENCHMARK_LABEL_NAMES) -> AlertTable:
    """Read an alert table CSV (module docstring) with the given label names.

    Raises:
        DataError: if the file cannot be read, lacks a required column, or a row breaks the
            schema (the message names the line), or :meth:`AlertTable.from_rows` refuses it.
    """
    index = {name: i for i, name in enumerate(label_names)}
    oids: list[str] = []
    candids: list[int] = []
    jds: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = [c for c in COLUMNS if c not in (reader.fieldnames or ())]
            if missing:
                raise DataError(f"{path}: missing columns {missing}; need {list(COLUMNS)}")
            has_group = "group" in (reader.fieldnames or ())
            for row in reader:
                oid, candid, jd, label, group = _parse_row(row, reader.line_num, index, path)
                if has_group:
                    if group is None:
                        raise DataError(f"{path}:{reader.line_num}: missing group")
                    groups.append(group)
                oids.append(oid)
                candids.append(candid)
                jds.append(jd)
                labels.append(label)
    except (OSError, UnicodeDecodeError, csv.Error) as err:
        raise DataError(f"{path}: cannot read the alert table ({err})") from None
    try:
        return AlertTable.from_rows(
            oids, candids, jds, labels, label_names, groups if has_group else None
        )
    except DataError as err:
        raise DataError(f"{path}: {err}") from None


def write_alert_table(table: AlertTable, path: Path) -> None:
    """Write ``table`` as CSV: the :data:`COLUMNS` (``jd`` with ``repr``, exact), and
    ``group`` when the table has its own group key.

    Raises:
        OSError: if the file cannot be written.
    """
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        own_group = table.group_key == "group"
        writer.writerow([*COLUMNS, "group"] if own_group else COLUMNS)
        rows = zip(table.oid, table.candid, table.jd, table.label, table.group, strict=True)
        for oid, candid, jd, label, group in rows:
            row = [oid, int(candid), repr(float(jd)), table.label_names[int(label)]]
            writer.writerow([*row, group] if own_group else row)


def label_conflicts(table: AlertTable) -> tuple[LabelConflict, ...]:
    """The groups (:attr:`AlertTable.group`) whose alerts carry two or more labels, sorted.

    Each conflict lists the group's labels in label order.
    """
    labels: dict[str, set[int]] = {}
    for group, label in zip(table.group.tolist(), table.label.tolist(), strict=True):
        labels.setdefault(group, set()).add(label)
    return tuple(
        LabelConflict(oid, tuple(table.label_names[c] for c in sorted(found)))
        for oid, found in sorted(labels.items())
        if len(found) > 1
    )


def first_stamps(table: AlertTable, rows: Iterable[int] | None = None) -> npt.NDArray[np.int64]:
    """For each object among ``rows`` (default: all), the row of its earliest alert.

    Earliest is the smallest ``jd``, ties broken by the smaller ``candid``. Returns sorted
    row positions (so sorted by ``candid``), one per object.
    """
    selected = (
        np.arange(len(table), dtype=np.int64) if rows is None else np.asarray(list(rows), np.int64)
    )
    if selected.size == 0:
        return selected
    order = np.lexsort((table.candid[selected], table.jd[selected], table.oid[selected]))
    ordered = selected[order]
    oids = table.oid[ordered]
    first = np.ones(ordered.size, dtype=bool)
    first[1:] = oids[1:] != oids[:-1]
    return np.sort(ordered[first])
