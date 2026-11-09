"""Experiment specs: arms x seeds x folds (or the screen) on one machine class (task A3).

An experiment spec is a small YAML file that declares a set of runs before anything is
trained. The queue (:mod:`astro_simclr.queue`) expands it into one job per cell; the
registry (A2) checks that a comparison holds exactly its cells. Both read it through this
module, which imports neither torch nor sqlite. Format ``astro-simclr-experiment-spec``,
version 1 (:data:`SPEC_FORMAT`, :data:`SPEC_VERSION`)::

    format: astro-simclr-experiment-spec
    version: 1
    name: c5-flux-screen            # a slug; names the spec's run directories
    command: train-supervised       # the training subcommand (only this one in version 1)
    preset: carrasco2021            # optional: presets.PRESETS
    settings:                       # optional: overrides shared by every arm
      max_steps: 30000
    arms:                           # arm name -> overrides of the shared settings
      baseline: {}
      flux: {flux_rescale: 0.5}
    seeds: [0, 1, 2]
    splits:                         # optional; without it a run uses the file's own splits
      file: configs/splits/td_ztf_stamp_17_06_20_red-002.json   # relative to the checkout
      folds: [0, 1, 2, 3, 4]        # or `screen: true` (or, for rj23, `whole: true`)
    machine:
      class: cuda-nvidia-geforce-gtx-1050   # the run id's machine class (ADR 0015, rule 7)
      extra: cu126                  # the `uv sync --extra` of that machine (ADR 0001); optional
    engine: plain                   # required since task 4.8: plain (or lightning, read only)
    scratch: false                  # optional: true allows a dirty tree (never a result)

**Settings of a cell** (:meth:`ExperimentSpec.arm_settings`) resolve in the order of
``train-supervised`` on the command line: the ``Settings`` defaults, the subcommand's legacy
protocol (``hparams.COMMAND_DEFAULTS``), the preset, ``settings``, the arm's overrides, then
the cell's seed and the runner's ``data_dir`` and ``output_dir``. ``seed``, ``data_dir``,
``output_dir`` and ``pretrained`` cannot be set in a spec (:data:`RESERVED_KEYS`): the
first three belong to the cell and the runner, and a pretrained checkpoint has no
machine-independent name yet (phase 5).

**The rj23 schedule** (task 3.4b; on the plain engine since #62). An arm whose settings
resolve to ``schedule: rj23`` trains RJ23's protocol on a ``raw-alerts`` layout, and its
``splits.file`` is the layout's object splits (``make-object-splits``), never CV folds:
``screen: true`` is a decision run (split part ``screen``), ``whole: true`` a replication
run on the whole Train (split part ``whole``), as ``--screen`` / no ``--screen`` on the
command line. The alert table is the layout's own. Such a spec needs ``engine: plain``
(the queue does not drive Lightning's ``training.rj23.run_rj23``); ``whole: true`` is
refused for any other schedule.

**Engine.** ``engine`` is the training loop of every cell, and is required since task
4.8: ``plain`` (``tasks.plain.run_plain``: the paper classifier on ``carrasco2021`` and
RJ23's on ``rj23``; no ``limit_*_batches``) or ``lightning``. Before 4.8 a spec without
the key meant ``lightning``; a default of ``plain`` would have given such a file other
run ids and trained it on another engine without a word, so the key is required and its
absence is an error that says what to write. ``lightning`` still parses, so that ``runs
compare`` resolves the run ids of a spec that trained on Lightning, but
:func:`check_trainable` (``queue submit``, ``spec run-cell``) refuses it, naming the git
tag ``lightning-final``. Like ``runtime.engine`` of a nested config (ADR 0017), it is a run-id
component, not a setting: the arms' config hashes do not depend on it, their run ids do.

**Identity.** :attr:`ExperimentSpec.spec_id`, the experiment-spec id that a comparison
takes (A2), is the SHA-256 of the canonical JSON (``runs.canonical.canonical_json``) of
:meth:`ExperimentSpec.to_plain`: the parsed content, so a comment or a reordering of
keys does not change it, while any arm, seed, split, machine or ``scratch`` change does.
Each cell is one run id (:func:`expected_run_ids`); the seeds of one arm share the arm's
config hash (:meth:`ExperimentSpec.arm_config_sha256`), because the seed is an identity
key (ADR 0015, rule 4).

The registry depends on these names, so they are stable: ``name``, ``spec_id``,
``command``, ``machine_class``, ``engine``, ``splits`` (``None`` or ``.file``), ``arms`` (each
``.name``), ``cells()`` (each ``.arm``, the arm's name, ``.seed``, ``.split``) and
``arm_config_sha256(arm)``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import yaml

from astro_simclr.config import LIGHTNING_TAG, RJ23, ConfigError, Settings, settings_from_mapping
from astro_simclr.hparams import COMMAND_DEFAULTS
from astro_simclr.runs.canonical import CanonicalError, canonical_json, config_sha256, sha256_text
from astro_simclr.runs.identity import (
    ENGINES,
    SCREEN,
    WHOLE,
    Engine,
    IdentityError,
    SplitPart,
    run_identity,
)

__all__ = [
    "RESERVED_KEYS",
    "SPEC_COMMANDS",
    "SPEC_FORMAT",
    "SPEC_VERSION",
    "Arm",
    "Cell",
    "ExperimentSpec",
    "SpecError",
    "SpecSplits",
    "check_trainable",
    "expected_run_ids",
    "load_spec",
    "parse_spec",
]

#: ``format`` of a spec file.
SPEC_FORMAT: Final = "astro-simclr-experiment-spec"
#: The version this module reads and writes.
SPEC_VERSION: Final = 1
#: The training subcommands a version-1 spec may run (today's ``train-supervised``; the
#: plan's A3 scope). ``pretrain`` and the evaluations come with phase 5.
SPEC_COMMANDS: Final = ("train-supervised",)

#: Settings a spec may not set, with the reason.
RESERVED_KEYS: Final[Mapping[str, str]] = {
    "seed": "the cell's seed, from `seeds`",
    "data_dir": "the runner's data directory (the data's content is the run id's data SHA-256)",
    "output_dir": "the runner names the run directory",
    "pretrained": "a pretrained checkpoint has no machine-independent name yet (phase 5)",
}

_TOP_KEYS: Final = frozenset(
    {"format", "version", "name", "command", "preset", "settings", "arms", "seeds"}
    | {"splits", "machine", "engine", "scratch"}
)
_REQUIRED: Final = ("format", "version", "name", "command", "arms", "seeds", "machine")
_SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_EXTRA = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_CELL = re.compile(r"(?P<arm>[a-z0-9][a-z0-9._-]*)/s(?P<seed>\d+)/(?P<split>f\d+|screen|whole|all)")


class SpecError(ConfigError):
    """An experiment spec is malformed or names an invalid setting.

    A :class:`~astro_simclr.config.ConfigError`, so the command line exits with 2.
    """


def _slug(value: object, what: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        msg = (
            f"{what} must be a slug (lowercase letters, digits, '.', '_', '-'; at most 64), "
            f"got {value!r}"
        )
        raise SpecError(msg)
    return value


def _seed(value: object) -> int:
    # bool is an int: `seeds: [true]` is a typo, not seed 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecError(f"a seed must be an integer >= 0, got {value!r}")
    return value


def _overrides(value: object, what: str) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SpecError(f"{what} must be a mapping of setting names")
    reserved = sorted(set(value) & set(RESERVED_KEYS))
    if reserved:
        reasons = "; ".join(f"{key}: {RESERVED_KEYS[key]}" for key in reserved)
        raise SpecError(f"{what} may not set {', '.join(reserved)} ({reasons})")
    try:
        canonical_json(value)
    except CanonicalError as err:
        raise SpecError(f"{what}: {err}") from None
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class Arm:
    """One arm: a name (a slug) and its overrides of the spec's shared settings."""

    name: str
    overrides: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class SpecSplits:
    """The stored splits file of a spec's split runs, and its folds (``None``: the screen).

    ``file`` is relative to the checkout the runs are made from (``configs/splits/``).
    ``whole`` (``folds`` is then ``None``): an rj23 replication on the object splits' whole
    Train, split part ``whole`` (module docstring).
    """

    file: Path
    folds: tuple[int, ...] | None
    whole: bool = False


@dataclass(frozen=True, slots=True)
class Cell:
    """One run of a spec: an arm (its name), a seed and a split part.

    ``split`` is a fold, ``"screen"``, ``"whole"`` or ``None`` (``runs.identity.SplitPart``).
    """

    arm: str
    seed: int
    split: SplitPart

    @property
    def key(self) -> str:
        """The cell as text: ``<arm>/s<seed>/<f<fold>|screen|whole|all>`` (:meth:`parse`)."""
        split = self.split
        part = "all" if split is None else split if isinstance(split, str) else f"f{split}"
        return f"{self.arm}/s{self.seed}/{part}"

    @classmethod
    def parse(cls, key: str) -> Cell:
        """The cell whose :attr:`key` is ``key``.

        Raises:
            SpecError: if ``key`` is not of that form.
        """
        match = _CELL.fullmatch(key)
        if match is None:
            raise SpecError(f"cell {key!r} is not of the form ARM/sSEED/fFOLD|screen|whole|all")
        part = match["split"]
        split: SplitPart = (
            None
            if part == "all"
            else SCREEN
            if part == SCREEN
            else WHOLE
            if part == WHOLE
            else int(part[1:])
        )
        return cls(match["arm"], int(match["seed"]), split)

    def __lt__(self, other: object) -> bool:
        # A fold, "screen" and None cannot be compared directly: order by the key's parts.
        if not isinstance(other, Cell):
            return NotImplemented
        return _sort_key(self) < _sort_key(other)


def _sort_key(cell: Cell) -> tuple[str, int, int, int]:
    split = cell.split
    rank = 0 if split is None else 1 if split == SCREEN else 2 if split == WHOLE else 3
    return (cell.arm, cell.seed, rank, cell.split if isinstance(cell.split, int) else -1)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentSpec:
    """A parsed experiment spec (module docstring). Built by :func:`parse_spec`."""

    name: str
    command: str
    preset: str | None
    settings: Mapping[str, object]
    arms: tuple[Arm, ...]
    seeds: tuple[int, ...]
    splits: SpecSplits | None
    machine_class: str
    extra: str | None
    scratch: bool
    engine: Engine = "plain"

    def to_plain(self) -> dict[str, object]:
        """The spec as plain values, in the file's form (what :attr:`spec_id` hashes)."""
        splits: dict[str, object] | None = None
        if self.splits is not None:
            splits = {"file": self.splits.file.as_posix()}
            if self.splits.whole:
                splits["whole"] = True
            elif self.splits.folds is None:
                splits["screen"] = True
            else:
                splits["folds"] = list(self.splits.folds)
        return {
            "format": SPEC_FORMAT,
            "version": SPEC_VERSION,
            "name": self.name,
            "command": self.command,
            "preset": self.preset,
            "settings": dict(self.settings),
            "arms": {arm.name: dict(arm.overrides) for arm in self.arms},
            "seeds": list(self.seeds),
            "splits": splits,
            "machine": {"class": self.machine_class, "extra": self.extra},
            "engine": self.engine,
            "scratch": self.scratch,
        }

    @property
    def spec_id(self) -> str:
        """The experiment-spec id: the SHA-256 of the canonical JSON of :meth:`to_plain`."""
        return sha256_text(canonical_json(self.to_plain()))

    def arm(self, name: str) -> Arm:
        """The arm called ``name``.

        Raises:
            SpecError: if the spec has no such arm.
        """
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise SpecError(f"spec {self.name} has no arm {name!r}")

    def split_parts(self) -> tuple[SplitPart, ...]:
        """The split part of each run of an arm and seed: the folds, the screen, or none."""
        if self.splits is None:
            return (None,)
        if self.splits.whole:
            return (WHOLE,)
        if self.splits.folds is None:
            return (SCREEN,)
        return self.splits.folds

    def cells(self) -> tuple[Cell, ...]:
        """Every run of the spec, arms x seeds x split parts, in a stable order."""
        return tuple(
            sorted(
                Cell(arm.name, seed, split)
                for arm in self.arms
                for seed in self.seeds
                for split in self.split_parts()
            )
        )

    def _mapping(self, arm: Arm | str) -> dict[str, object]:
        chosen = self.arm(arm) if isinstance(arm, str) else arm
        data: dict[str, object] = dict(COMMAND_DEFAULTS[self.command])
        if self.preset is not None:
            from astro_simclr.presets import PRESETS

            data.update(PRESETS[self.preset])
        data.update(self.settings)
        data.update(chosen.overrides)
        return data

    def arm_settings(
        self,
        arm: Arm | str,
        seed: int,
        *,
        data_dir: Path = Path("dataset"),
        output_dir: Path = Path("outputs"),
    ) -> Settings:
        """The resolved settings of ``arm`` (an :class:`Arm` or its name) with ``seed``.

        ``data_dir`` and ``output_dir`` are non-semantic (ADR 0015): they do not change the
        config hash.

        Raises:
            SpecError: on an unknown arm, setting or invalid value.
        """
        data = self._mapping(arm)
        data.update({"seed": seed, "data_dir": data_dir, "output_dir": output_dir})
        try:
            return settings_from_mapping(data)
        except SpecError:
            raise
        except ConfigError as err:
            name = arm if isinstance(arm, str) else arm.name
            raise SpecError(f"spec {self.name}, arm {name}: {err}") from None

    def arm_config_sha256(self, arm: Arm | str) -> str:
        """The config hash of ``arm`` (an :class:`Arm` or its name), shared by its seeds.

        Raises:
            SpecError: on an unknown arm, setting or invalid value.
        """
        return config_sha256(self.command, self.arm_settings(arm, 0))


def _as_mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SpecError(f"{what} must be a mapping")
    return cast(Mapping[str, object], value)


def _parse_splits(value: object) -> SpecSplits | None:
    if value is None:
        return None
    splits = _as_mapping(value, "splits")
    unknown = sorted(set(splits) - {"file", "folds", "screen", "whole"})
    if unknown:
        raise SpecError(f"splits: unknown keys {', '.join(unknown)}")
    file = splits.get("file")
    if (
        not isinstance(file, str)
        or not file
        or Path(file).is_absolute()
        or ".." in Path(file).parts
    ):
        raise SpecError(f"splits.file must be a path inside the checkout, got {file!r}")
    screen, whole, folds = (
        splits.get("screen", False),
        splits.get("whole", False),
        splits.get("folds"),
    )
    for name, flag in (("screen", screen), ("whole", whole)):
        if flag is not True and flag is not False:
            raise SpecError(f"splits.{name} must be true or false, got {flag!r}")
    if [screen, whole, folds is not None].count(True) != 1:
        raise SpecError(
            "splits needs exactly one of `folds: [...]`, `screen: true` and `whole: true`"
        )
    if folds is None:
        return SpecSplits(Path(file), None, whole=whole is True)
    if not isinstance(folds, list) or not folds:
        raise SpecError(f"splits.folds must be a non-empty list, got {folds!r}")
    checked = tuple(_seed(fold) for fold in folds)
    if len(set(checked)) != len(checked):
        raise SpecError(f"splits.folds repeats a fold: {folds}")
    return SpecSplits(Path(file), checked)


def _parse_machine(value: object) -> tuple[str, str | None]:
    machine = _as_mapping(value, "machine")
    unknown = sorted(set(machine) - {"class", "extra"})
    if unknown:
        raise SpecError(f"machine: unknown keys {', '.join(unknown)}")
    machine_class = _slug(machine.get("class"), "machine.class")
    extra = machine.get("extra")
    if extra is not None and (not isinstance(extra, str) or not _EXTRA.fullmatch(extra)):
        raise SpecError(f"machine.extra must be a uv extra name or null, got {extra!r}")
    return machine_class, extra


def _parse_arms(value: object) -> tuple[Arm, ...]:
    arms = _as_mapping(value, "arms")
    if not arms:
        raise SpecError("arms must name at least one arm")
    return tuple(
        Arm(_slug(name, "an arm name"), _overrides(overrides, f"arm {name}"))
        for name, overrides in arms.items()
    )


def _parse_seeds(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SpecError(f"seeds must be a non-empty list, got {value!r}")
    seeds = tuple(_seed(seed) for seed in value)
    if len(set(seeds)) != len(seeds):
        raise SpecError(f"seeds repeats a seed: {value}")
    return seeds


def _parse_engine(data: Mapping[str, object]) -> Engine:
    """The spec's ``engine``, required since task 4.8 (module docstring, "Engine").

    Raises:
        SpecError: if the key is missing (saying what to write) or not an engine.
    """
    if "engine" not in data:
        msg = (
            "engine is required since task 4.8: `engine: plain` to train, or `engine: "
            "lightning` for a spec written before 4.8 (it meant Lightning), to compare its runs"
        )
        raise SpecError(msg)
    engine = data["engine"]
    if engine not in ENGINES:
        raise SpecError(f"engine {engine!r} is not one of: {', '.join(ENGINES)}")
    return engine


def _from_plain(data: Mapping[str, object]) -> ExperimentSpec:
    if data.get("format") != SPEC_FORMAT:
        raise SpecError(f"not an {SPEC_FORMAT} file (format {data.get('format')!r})")
    version = data.get("version")
    if type(version) is not int or version != SPEC_VERSION:
        raise SpecError(f"spec version {version!r}; this version reads {SPEC_VERSION}")
    missing = [key for key in _REQUIRED if key not in data]
    unknown = sorted(set(data) - _TOP_KEYS)
    if missing or unknown:
        raise SpecError(f"spec: missing keys {missing or 'none'}, unknown {unknown or 'none'}")
    command = data["command"]
    if command not in SPEC_COMMANDS:
        raise SpecError(f"command {command!r}: a spec runs one of {', '.join(SPEC_COMMANDS)}")
    preset = data.get("preset")
    if preset is not None:
        from astro_simclr.presets import PRESETS

        if preset not in PRESETS:
            raise SpecError(f"preset {preset!r} is not one of: {', '.join(PRESETS)}")
    scratch = data.get("scratch", False)
    if scratch is not True and scratch is not False:
        raise SpecError(f"scratch must be true or false, got {scratch!r}")
    machine_class, extra = _parse_machine(data["machine"])
    engine = _parse_engine(data)
    spec = ExperimentSpec(
        name=_slug(data["name"], "name"),
        command=command,
        preset=cast(str | None, preset),
        settings=_overrides(data.get("settings"), "settings"),
        arms=_parse_arms(data["arms"]),
        seeds=_parse_seeds(data["seeds"]),
        splits=_parse_splits(data.get("splits")),
        machine_class=machine_class,
        extra=extra,
        scratch=scratch,
        engine=engine,
    )
    for arm in spec.arms:  # every arm must resolve before anything is queued
        _check_schedule(spec, arm.name, spec.arm_settings(arm, spec.seeds[0]))
    return spec


def _check_schedule(spec: ExperimentSpec, arm: str, settings: Settings) -> None:
    """An rj23 arm's splits and engine; ``whole`` only for rj23 (module docstring)."""
    splits = spec.splits
    if settings.schedule != RJ23:
        if splits is not None and splits.whole:
            raise SpecError(f"arm {arm}: splits.whole is for schedule {RJ23} only")
        return
    if splits is None or splits.folds is not None:
        msg = (
            f"arm {arm}: schedule {RJ23} needs its object splits with `screen: true` (a "
            "decision run) or `whole: true` (a replication), not folds"
        )
        raise SpecError(msg)
    if spec.engine != "plain":
        raise SpecError(f"arm {arm}: schedule {RJ23} is queued on `engine: plain` only")


def check_trainable(spec: ExperimentSpec) -> None:
    """Refuse a spec this checkout cannot train: ``engine: lightning`` (task 4.8).

    Raises:
        SpecError: for ``engine: lightning``, naming the git tag that keeps the engine.
    """
    if spec.engine != "plain":
        msg = (
            f"spec {spec.name} has engine {spec.engine}, removed in task 4.8: queue it on "
            f"plain, or submit it from a checkout of the git tag {LIGHTNING_TAG}"
        )
        raise SpecError(msg)


def parse_spec(text: str) -> ExperimentSpec:
    """Parse and validate the YAML text of a spec (module docstring).

    Every arm's settings are resolved once, so an unknown setting or a bad value fails
    here, before anything is queued.

    Raises:
        SpecError: if the text is not valid YAML, not a version-1 spec, or any field,
            arm or setting is invalid.
    """
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise SpecError(f"the spec is not valid YAML: {err}") from None
    return _from_plain(_as_mapping(loaded, "a spec"))


def load_spec(path: Path) -> ExperimentSpec:
    """Read and validate the spec file at ``path`` (:func:`parse_spec`).

    Raises:
        SpecError: if the file cannot be read or is not a valid spec (the message names it).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise SpecError(f"cannot read spec {path}: {err}") from None
    try:
        return parse_spec(text)
    except SpecError as err:
        raise SpecError(f"{path}: {err}") from None


def expected_run_ids(
    spec: ExperimentSpec,
    *,
    data_sha256: str,
    splits_sha256: str | None,
    engine: Engine | None = None,
) -> dict[Cell, str]:
    """The run id of every cell, given the data and the splits file the runs read.

    ``splits_sha256`` is the SHA-256 of ``spec.splits.file`` (``None`` without splits);
    ``engine`` defaults to ``spec.engine``.

    Raises:
        SpecError: if ``splits_sha256`` is given without splits or the reverse, or a hash
            is malformed.
    """
    if (spec.splits is None) != (splits_sha256 is None):
        raise SpecError("splits_sha256 goes with the spec's splits")
    ids: dict[Cell, str] = {}
    for cell in spec.cells():
        settings = spec.arm_settings(cell.arm, cell.seed)
        try:
            identity = run_identity(
                spec.command,
                settings,
                data_sha256=data_sha256,
                machine_class=spec.machine_class,
                engine=spec.engine if engine is None else engine,
                split=cell.split,
                splits_sha256=None if cell.split is None else splits_sha256,
            )
        except IdentityError as err:
            raise SpecError(str(err)) from None
        ids[cell] = identity.run_id
    return ids
