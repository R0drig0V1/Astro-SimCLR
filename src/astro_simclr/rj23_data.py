"""The parts of an RJ23 run on a ``raw-alerts`` layout and its object splits (task 3.4b).

Engine-free (numpy only; the phase-4 rule): the plain engine's RJ23 runner
(:mod:`astro_simclr.tasks.rj23`) builds its data here, as the Lightning adapter did
until task 4.8.

A run reads a ``raw-alerts`` layout (:mod:`astro_simclr.data.raw_alerts`, task 3.1b), its
stored object splits (:mod:`astro_simclr.protocol.object_splits`, task 3.2) and the alert
table those splits were drawn from (by default the layout's own,
:func:`~astro_simclr.protocol.alerts.alert_table_from_raw`, grouped by ``oid``; a CSV
table when the splits are grouped by another key). The splits are verified against the
table before anything is built, and every alert of the table must be in the layout with the
same label. Two protocols (plan 3.4b, "Protocol"; :data:`MODE_PARTS`):

- ``replication`` (the paper protocol, as task 2.4 did for CD21): train on ``Train``, stop
  and select on the **whole** ``Validation``, evaluate ``Validation`` and ``Test``;
- ``decision`` (a phase-3 decision, ``docs/evaluation-protocol.md``): train on ``Train``,
  stop and select on Validation's ``selection`` half, evaluate the ``selection`` and
  ``decision`` halves. ``Test`` is never read, not even its cutouts.

Each part (:class:`RJ23Part`) holds the scenario's view in torch's layout (``(N, C, H, W)``
float32; :func:`~astro_simclr.data.rj23_input.rj23_views` then
:func:`~astro_simclr.data.rj23_input.channels_first`), the position on the unit sphere
(:func:`~astro_simclr.data.rj23_input.unit_sphere`), the labels and the ``candid`` of every
row (the predictions' index, ADR 0008), in ``candid`` order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.data.raw_alerts import RawAlerts
from astro_simclr.data.rj23_input import (
    VIEW_NAMES,
    VIEW_SHAPES,
    LargeValues,
    ViewName,
    channels_first,
    rj23_views,
    unit_sphere,
)
from astro_simclr.data.schema import DataError
from astro_simclr.protocol.alerts import AlertTable, alert_table_from_raw
from astro_simclr.protocol.object_splits import ObjectSplits, verify_object_splits

__all__ = [
    "MODES",
    "MODE_PARTS",
    "RJ23Mode",
    "RJ23Part",
    "RunPlan",
    "rj23_parts",
    "rj23_rows",
]

RJ23Mode = Literal["replication", "decision"]
#: The protocols of a run (module docstring).
MODES: Final[tuple[RJ23Mode, ...]] = ("replication", "decision")


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Which parts a protocol trains on, selects on and evaluates.

    Attributes:
        train: the part trained on.
        selection: the part the schedule validates (stop and best).
        evaluated: the parts predicted at the end, each with its stage name in the
            predictions files (``val`` / ``test`` as ``bench-report`` reads them).
    """

    train: str
    selection: str
    evaluated: tuple[tuple[str, str], ...]

    @property
    def parts(self) -> tuple[str, ...]:
        """Every part the run reads, without repeats, in a fixed order."""
        names = [self.train, self.selection, *(part for part, _ in self.evaluated)]
        return tuple(dict.fromkeys(names))


#: Mode -> its parts (plan 3.4b: the replication stops on the whole validation set; a
#: decision on the selection half).
MODE_PARTS: Final[Mapping[RJ23Mode, RunPlan]] = {
    "replication": RunPlan("Train", "Validation", (("Validation", "val"), ("Test", "test"))),
    "decision": RunPlan(
        "Train", "selection", (("selection", "selection"), ("decision", "decision"))
    ),
}


@dataclass(frozen=True, slots=True)
class RJ23Part:
    """One part of a run, ready for :class:`~astro_simclr.rj23_model.RJ23Classifier`.

    Attributes:
        name: the part (``Train``, ``Validation``, ``Test``, ``selection``, ``decision``).
        images: float32 ``(N, C, H, W)``, the scenario's view.
        positions: float32 ``(N, 3)``, ``(x, y, z)`` on the unit sphere.
        labels: int64 ``(N,)``.
        candids: int64 ``(N,)``, ascending.

    Raises:
        ValueError: if the arrays disagree in length or N is 0.
    """

    name: str
    images: npt.NDArray[np.float32]
    positions: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]
    candids: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        n = len(self.labels)
        lengths = {len(self.images), len(self.positions), len(self.candids)}
        if n == 0 or lengths != {n}:
            raise ValueError(f"part {self.name!r} needs N > 0 rows of every array")

    def __len__(self) -> int:
        return len(self.labels)


def rj23_rows(
    data: RawAlerts, table: AlertTable, splits: ObjectSplits, part: str
) -> npt.NDArray[np.int64]:
    """The layout rows of ``part`` (a split or a half of Validation), in ``candid`` order.

    Raises:
        DataError: if an alert of the part is not in the layout, or its label there
            differs from the table's.
        ValueError: for an unknown ``part``.
    """
    table_rows = splits.rows(table, part)
    rows = data.rows_of(table.candid[table_rows])
    labels = data.index["label"][rows]
    if not np.array_equal(labels, table.label[table_rows]):
        bad = int(np.flatnonzero(labels != table.label[table_rows])[0])
        msg = (
            f"candid {int(table.candid[table_rows][bad])} has label {int(labels[bad])} in the "
            f"layout but {int(table.label[table_rows][bad])} in the alert table"
        )
        raise DataError(msg)
    return rows


def _part(
    data: RawAlerts,
    rows: npt.NDArray[np.int64],
    name: str,
    scenario: ViewName,
    large_values: LargeValues,
) -> RJ23Part:
    cutouts = np.asarray(data.cutouts[rows])
    view = rj23_views(cutouts, (scenario,), large_values=large_values)[scenario]
    index = data.index[rows]
    return RJ23Part(
        name=name,
        images=channels_first(view),
        positions=unit_sphere(index["ra"], index["dec"]),
        labels=np.asarray(index["label"], dtype=np.int64),
        candids=np.asarray(index["candid"], dtype=np.int64),
    )


def rj23_parts(
    data: RawAlerts,
    splits: ObjectSplits,
    scenario: ViewName,
    *,
    mode: RJ23Mode = "replication",
    table: AlertTable | None = None,
    large_values: LargeValues = "paper",
) -> dict[str, RJ23Part]:
    """The parts of :data:`MODE_PARTS` ``[mode]``, by name (module docstring).

    ``table`` is the alert table of the splits (default: the layout's own, grouped by
    ``oid``). The splits are verified against it first
    (:func:`~astro_simclr.protocol.object_splits.verify_object_splits`).

    Raises:
        ValueError: for an unknown ``scenario`` or ``mode``.
        DataError: if the table does not fit the layout (a missing alert, another label).
        SplitsError: if the splits do not belong to the table or are not well drawn.
    """
    if scenario not in VIEW_NAMES:
        raise ValueError(f"scenario must be one of {VIEW_NAMES}, got {scenario!r}")
    if mode not in MODE_PARTS:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if splits.label_names != data.label_names:
        msg = (
            f"the splits are for the classes {list(splits.label_names)}, the layout has "
            f"{list(data.label_names)}"
        )
        raise DataError(msg)
    alerts = alert_table_from_raw(data) if table is None else table
    verify_object_splits(splits, alerts)
    plan = MODE_PARTS[mode]
    parts = {
        name: _part(data, rj23_rows(data, alerts, splits, name), name, scenario, large_values)
        for name in plan.parts
    }
    expected = (VIEW_SHAPES[scenario][2], *VIEW_SHAPES[scenario][:2])
    assert all(p.images.shape[1:] == expected for p in parts.values())  # noqa: S101
    return parts
