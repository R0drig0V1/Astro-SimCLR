"""Builders for the registry tests: run directories (manifest and legacy), plans, a fake spec.

- :func:`manifest_run` writes a run directory as task A1's runners do: a
  ``manifest.json`` (A0's writer) and, once finished, a ``metrics.json``.
- :func:`legacy_run` writes a run directory with only a ``metrics.json``, in the form of
  the archived phase-2 runs (``results/phase2/replication``).
- :class:`FakeSpec` implements the interface agreed with task A3 (``runs.spec``):
  name, spec_id, command, machine_class, splits, arms, cells(), arm_config_sha256().
  :func:`install_fake_spec_module` puts a ``load_spec`` that reads it from JSON in place of
  ``astro_simclr.runs.spec``.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from _canonical_cases import case_settings

from astro_simclr.config import settings_to_plain
from astro_simclr.runs.canonical import canonical_json, config_sha256, sha256_text
from astro_simclr.runs.identity import GitState, SplitPart, run_identity
from astro_simclr.runs.manifest import new_manifest, transition, write_manifest

REPO = Path(__file__).resolve().parents[1]
#: The archived phase-2 replication: 5 seeds with features, 5 without (legacy runs).
REPLICATION = REPO / "results" / "phase2" / "replication"

MACHINE: dict[str, object] = {
    "device": "cuda",
    "gpu": "NVIDIA GeForce GTX 1050",
    "torch": "2.14.1+cu126",
    "cuda": "12.6",
    "fp32_precision": "ieee",
    "cudnn": 91002,
    "platform": "Linux-WSL2",
    "processor": "x86_64",
    "python": "3.13.13",
}
GTX1050 = "cuda-nvidia-geforce-gtx-1050"
DATA = "d" * 64
SPLITS = "5" * 64
T0, T1 = "2026-11-08T10:00:00+00:00", "2026-11-08T10:05:00+00:00"


def metrics_document(
    command: str, config: Mapping[str, object], seed: int, accuracy: float, **extra: object
) -> dict[str, object]:
    """A ``metrics.json`` of a classifier run (the keys the registry reads)."""
    stage = {"accuracy": accuracy, "balanced_accuracy": accuracy, "macro_f1": accuracy - 0.01}
    return {
        "command": command,
        "seed": seed,
        "best_step": 100,
        "last_step": 200,
        "monitor_best": -0.05,
        "machine": MACHINE,
        "data": {"source": {"sha256": "a" * 64}},
        "config": dict(config),
        "timing": {"seconds_per_step": 0.05},
        "evaluation": {"val": stage, "test": dict(stage)},
        **extra,
    }


def manifest_run(
    run_dir: Path,
    *,
    case: str = "carrasco2021",
    seed: int = 0,
    attempt: int = 1,
    status: str = "finished",
    accuracy: float = 0.9,
    split: SplitPart = None,
    machine: Mapping[str, object] = MACHINE,
    machine_class: str = GTX1050,
    engine: str = "lightning",
    numerics_epoch: int = 1,
    data_sha256: str = DATA,
    ended_at: str = "2026-11-08T11:00:00+00:00",
) -> Path:
    """Write a run directory with a manifest (and ``metrics.json`` if finished)."""
    command, settings = case_settings(case)
    settings = settings.replace(seed=seed, output_dir=run_dir)
    identity = run_identity(
        command,
        settings,
        data_sha256=data_sha256,
        machine_class=machine_class,
        engine=engine,  # type: ignore[arg-type]
        split=split,
        splits_sha256=None if split is None else SPLITS,
        numerics_epoch=numerics_epoch,
    )
    manifest = new_manifest(
        command,
        settings,
        identity,
        git=GitState("0" * 40, False),
        uv_lock_sha256="e" * 64,
        attempt=attempt,
        created_at=T0,
    )
    if status != "queued":
        manifest = transition(manifest, "running", at=T1, machine=machine)
    if status in ("finished", "failed", "cancelled"):
        manifest = transition(manifest, status, at=ended_at)  # type: ignore[arg-type]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, manifest)
    if status == "finished":
        doc = metrics_document(command, settings_to_plain(settings), seed, accuracy)
        doc["machine"] = dict(machine)
        (run_dir / "metrics.json").write_text(json.dumps(doc), encoding="utf-8")
    return run_dir


def legacy_run(
    run_dir: Path,
    *,
    case: str = "carrasco2021",
    seed: int = 0,
    accuracy: float = 0.9,
    drop: Sequence[str] = (),
    **extra: object,
) -> Path:
    """Write a run directory with only a ``metrics.json`` (a legacy, backfilled run).

    ``drop`` removes config keys, as an older ``Settings`` would not have had them.
    """
    command, settings = case_settings(case)
    config = settings_to_plain(settings.replace(seed=seed, output_dir=run_dir))
    for key in drop:
        del config[key]
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = metrics_document(command, config, seed, accuracy, **extra)
    (run_dir / "metrics.json").write_text(json.dumps(doc), encoding="utf-8")
    return run_dir


def copy_replication(dest: Path) -> Path:
    """A copy of the archived replication runs (metrics, predictions, figures)."""
    shutil.copytree(REPLICATION, dest)
    return dest


def case_config_sha256(case: str) -> str:
    """The config hash of :func:`_canonical_cases.case_settings` ``case``."""
    command, settings = case_settings(case)
    return config_sha256(command, settings)


@dataclass(frozen=True)
class FakeArm:
    """``runs.spec.Arm`` as far as the registry reads it."""

    name: str
    config_sha256: str


@dataclass(frozen=True)
class FakeCell:
    """``runs.spec.Cell``: the arm's name, a seed, a split part."""

    arm: str
    seed: int
    split: SplitPart


@dataclass(frozen=True)
class FakeSplits:
    """``runs.spec.SpecSplits`` as far as the registry reads it."""

    file: Path


@dataclass(frozen=True)
class FakeSpec:
    """An experiment spec with the interface agreed with task A3."""

    name: str
    command: str
    arms: tuple[FakeArm, ...]
    seeds: tuple[int, ...]
    split_parts: tuple[SplitPart, ...] = (None,)
    machine_class: str | None = GTX1050
    splits: FakeSplits | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def spec_id(self) -> str:
        """A content hash, as A3's."""
        plain = {
            "name": self.name,
            "command": self.command,
            "arms": [[a.name, a.config_sha256] for a in self.arms],
            "seeds": list(self.seeds),
            "splits": [s for s in self.split_parts],
            "machine_class": self.machine_class,
        }
        return sha256_text(canonical_json(plain))

    def cells(self) -> tuple[FakeCell, ...]:
        """Arms x seeds x split parts."""
        return tuple(
            FakeCell(arm.name, seed, split)
            for arm in self.arms
            for seed in self.seeds
            for split in self.split_parts
        )

    def arm_config_sha256(self, arm: FakeArm | str) -> str:
        """The arm's config hash."""
        name = arm if isinstance(arm, str) else arm.name
        return next(a.config_sha256 for a in self.arms if a.name == name)


def replication_spec(seeds: Sequence[int] = range(5), name: str = "replication") -> FakeSpec:
    """The phase-2 replication as a spec: the two presets x ``seeds``."""
    return FakeSpec(
        name=name,
        command="train-supervised",
        arms=(
            FakeArm("with-features", case_config_sha256("carrasco2021")),
            FakeArm("no-features", case_config_sha256("carrasco2021-no-features")),
        ),
        seeds=tuple(seeds),
    )


def write_spec(path: Path, spec: FakeSpec) -> Path:
    """Write ``spec`` as the JSON :func:`install_fake_spec_module`'s ``load_spec`` reads."""
    plain = {
        "name": spec.name,
        "command": spec.command,
        "arms": [[a.name, a.config_sha256] for a in spec.arms],
        "seeds": list(spec.seeds),
        "splits": list(spec.split_parts),
        "machine_class": spec.machine_class,
        "splits_file": None if spec.splits is None else str(spec.splits.file),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain), encoding="utf-8")
    return path


def _load_spec(path: Path) -> FakeSpec:
    plain = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plain, dict) or "name" not in plain:
        raise ValueError(f"{path}: not a spec")
    splits_file = plain["splits_file"]
    return FakeSpec(
        name=plain["name"],
        command=plain["command"],
        arms=tuple(FakeArm(name, sha) for name, sha in plain["arms"]),
        seeds=tuple(plain["seeds"]),
        split_parts=tuple(plain["splits"]),
        machine_class=plain["machine_class"],
        splits=None if splits_file is None else FakeSplits(Path(splits_file)),
    )


def install_fake_spec_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``astro_simclr.runs.spec`` a module whose ``load_spec`` reads :func:`write_spec`."""
    module = types.ModuleType("astro_simclr.runs.spec")
    module.load_spec = _load_spec  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "astro_simclr.runs.spec", module)


def hide_spec_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make importing ``astro_simclr.runs.spec`` fail, as before task A3 lands."""
    monkeypatch.setitem(sys.modules, "astro_simclr.runs.spec", None)
