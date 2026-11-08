"""The registry's view of an experiment spec: :class:`ComparisonPlan`.

Task A3 owns the spec format (``astro_simclr.runs.spec``: ``ExperimentSpec``, its YAML
file, its ``spec_id``). The registry reads a spec only through the structural interface
:class:`SpecLike`, agreed with A3: a name and an id, the training command, the machine
class, the splits file, the arms, the cells (arms x seeds x split parts) and each arm's
config hash. :func:`plan_from_spec` turns any such spec into a :class:`ComparisonPlan`, the
minimal frozen dataclass a comparison runs on. Tests and other callers can build a plan
directly.

``runs.spec`` is imported lazily (:func:`load_spec_file`), so this package works, and its
tests run, whether or not task A3's module is present.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

from astro_simclr.data.files import sha256_file
from astro_simclr.runs.identity import SplitPart, check_sha256

__all__ = [
    "SPEC_MODULE",
    "ComparisonPlan",
    "PlannedCell",
    "SpecLike",
    "SpecUnavailableError",
    "load_spec_file",
    "plan_from_spec",
]

#: The module of task A3 that defines and reads experiment specs.
SPEC_MODULE: Final = "astro_simclr.runs.spec"


class SpecUnavailableError(RuntimeError):
    """Experiment specs cannot be read: task A3's ``runs.spec`` is not installed."""


class _CellLike(Protocol):
    @property
    def arm(self) -> str: ...
    @property
    def seed(self) -> int: ...
    @property
    def split(self) -> SplitPart: ...


class _ArmLike(Protocol):
    @property
    def name(self) -> str: ...


class _SplitsLike(Protocol):
    @property
    def file(self) -> Path: ...


class SpecLike(Protocol):
    """What the registry reads of an experiment spec (``runs.spec.ExperimentSpec``)."""

    @property
    def name(self) -> str: ...
    @property
    def spec_id(self) -> str: ...
    @property
    def command(self) -> str: ...
    @property
    def machine_class(self) -> str | None: ...
    @property
    def splits(self) -> _SplitsLike | None: ...
    @property
    def arms(self) -> Sequence[_ArmLike]: ...
    def cells(self) -> Sequence[_CellLike]: ...
    def arm_config_sha256(self, arm: str) -> str: ...


@dataclass(frozen=True, slots=True)
class PlannedCell:
    """One cell a comparison needs: an arm (by name), a seed and a split part."""

    arm: str
    seed: int
    split: SplitPart

    def label(self) -> str:
        """``arm/seed=S`` or ``arm/seed=S/split=P``, for messages."""
        text = f"{self.arm}/seed={self.seed}"
        return text if self.split is None else f"{text}/split={self.split}"


def _sort_key(cell: PlannedCell) -> tuple[str, int, int, str]:
    # None, then folds in order, then "screen": a total order over mixed split parts.
    split = cell.split
    rank = 0 if split is None else 1 if isinstance(split, int) else 2
    return (cell.arm, cell.seed, rank, str(split))


@dataclass(frozen=True, slots=True, kw_only=True)
class ComparisonPlan:
    """What a comparison resolves: the spec's id and name, command, arms and cells.

    ``arms`` maps each arm's name to its config hash. ``machine_class`` and
    ``splits_sha256``, when set, are required of every run. Validated on construction.

    Raises:
        ValueError: on no cells, a cell of an unknown arm, an arm without cells, a repeated
            cell, two arms with one config hash (their runs could not be told apart), or a
            bad SHA-256.
    """

    spec_id: str
    name: str
    command: str
    arms: Mapping[str, str]
    cells: tuple[PlannedCell, ...]
    machine_class: str | None = None
    splits_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arms", MappingProxyType(dict(self.arms)))
        object.__setattr__(self, "cells", tuple(sorted(self.cells, key=_sort_key)))
        check_sha256(self.spec_id, "spec_id")
        for arm, sha in self.arms.items():
            check_sha256(sha, f"arm {arm} config_sha256")
        if len(set(self.arms.values())) != len(self.arms):
            raise ValueError(f"spec {self.name}: two arms share a config hash")
        if self.splits_sha256 is not None:
            check_sha256(self.splits_sha256, "splits_sha256")
        if not self.cells:
            raise ValueError(f"spec {self.name}: no cells")
        if len(set(self.cells)) != len(self.cells):
            raise ValueError(f"spec {self.name}: a cell appears twice")
        used = {cell.arm for cell in self.cells}
        unknown, unused = sorted(used - set(self.arms)), sorted(set(self.arms) - used)
        if unknown or unused:
            msg = f"spec {self.name}: cells of unknown arms {unknown}, arms without cells {unused}"
            raise ValueError(msg)

    @property
    def seeds(self) -> tuple[int, ...]:
        """The seeds of the cells, sorted."""
        return tuple(sorted({cell.seed for cell in self.cells}))

    @property
    def splits(self) -> tuple[SplitPart, ...]:
        """The split parts of the cells, in cell order."""
        return tuple(dict.fromkeys(cell.split for cell in self.cells))


def plan_from_spec(spec: SpecLike) -> ComparisonPlan:
    """The comparison plan of an experiment spec (module docstring).

    ``splits_sha256`` is the SHA-256 of the spec's splits file, read now.

    Raises:
        OSError: if the spec names a splits file that cannot be read.
        ValueError: if the plan is invalid (:class:`ComparisonPlan`).
    """
    arms = {arm.name: spec.arm_config_sha256(arm.name) for arm in spec.arms}
    cells = tuple(PlannedCell(cell.arm, cell.seed, cell.split) for cell in spec.cells())
    splits = spec.splits
    return ComparisonPlan(
        spec_id=spec.spec_id,
        name=spec.name,
        command=spec.command,
        arms=arms,
        cells=cells,
        machine_class=spec.machine_class,
        splits_sha256=None if splits is None else sha256_file(splits.file),
    )


def load_spec_file(path: Path) -> SpecLike:
    """Read the experiment spec file at ``path`` with task A3's ``runs.spec.load_spec``.

    Raises:
        SpecUnavailableError: if ``runs.spec`` is not installed.
        ValueError: if the spec is invalid (``SpecError`` is a ``ValueError``).
        OSError: if the file cannot be read.
    """
    try:
        module = importlib.import_module(SPEC_MODULE)
    except ModuleNotFoundError as err:
        if err.name != SPEC_MODULE:
            raise
        msg = f"experiment specs need task A3 ({SPEC_MODULE}), which this build lacks"
        raise SpecUnavailableError(msg) from None
    load = cast(Callable[[Path], SpecLike], module.load_spec)
    return load(path)
