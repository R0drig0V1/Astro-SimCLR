"""Comparisons by experiment spec: resolve the spec's cells to runs, or fail.

A comparison never takes a hand-picked list of run directories: it takes an experiment
spec (task A3), whose arms x seeds x split parts were fixed before the runs, so the set
of runs compared cannot drift from the pre-registration (plan review F10).
:func:`resolve` maps every cell of a :class:`~astro_simclr.registry.spec_view.ComparisonPlan`
to exactly one finished attempt, and fails, listing every problem at once, when:

- a cell has no finished run: the resolved set must equal arms x seeds x splits;
- a cell has more than one attempt and no explicit choice was made: ``use`` names the
  attempt (its key, a prefix of it, or its directory), or ``attempts="latest"`` takes the
  one that ended last (a legacy attempt has no end time, so it needs ``use``);
- two copies of one attempt (one key) disagree on their ``metrics.json``;
- the chosen runs mix machine signatures (device, GPU, torch, CUDA, effective float32
  precision; ``reference.machine_signature``, ADR 0010), engines (task A1) or numerics
  epochs (ADR 0015; a legacy run's is unknown), unless ``allow_mixed`` names that kind;
- the chosen runs were trained on different data: two data SHA-256s, or two source pickle
  SHA-256s. This is never allowed.

A run matches an arm when its command is the spec's and its config hash is the arm's; it
must also have the spec's machine class and splits file, when the spec names them.
Matching finished runs that are outside the cells (another seed, fold, machine or splits
file) are not used; the report counts them, so that a spec run on a subset of the seeds
is visible.

A copy of an attempt in another directory (the same :attr:`RunRecord.key`) is the same
attempt: it is listed, not counted twice.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, get_args

from astro_simclr.reference import machine_signature
from astro_simclr.registry.home import RegistryError
from astro_simclr.registry.readers import RunRecord
from astro_simclr.registry.spec_view import ComparisonPlan, PlannedCell

__all__ = [
    "MIXABLE",
    "AttemptPolicy",
    "ComparisonError",
    "Mixable",
    "Resolution",
    "arm_summary",
    "default_metrics",
    "format_report",
    "resolution_to_plain",
    "resolve",
]

#: What ``allow_mixed`` may allow: mixed machine signatures, engines, numerics epochs.
type Mixable = Literal["machines", "engines", "numerics"]
MIXABLE: Final[tuple[Mixable, ...]] = get_args(Mixable.__value__)
#: How a cell with several attempts is resolved without ``use``.
type AttemptPolicy = Literal["error", "latest"]

_MIN_PREFIX: Final = 6


class ComparisonError(RegistryError):
    """The registry's runs do not resolve the spec; ``problems`` lists every reason."""

    def __init__(self, spec: str, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"spec {spec} does not resolve:\n{lines}")


@dataclass(frozen=True, slots=True)
class Resolution:
    """A resolved comparison: one attempt per cell, its copies, and what was left out."""

    plan: ComparisonPlan
    chosen: Mapping[PlannedCell, RunRecord]
    copies: Mapping[PlannedCell, tuple[RunRecord, ...]]
    outside: tuple[RunRecord, ...]
    notes: tuple[str, ...]


def _matches_use(record: RunRecord, token: str) -> bool:
    if token == record.key or str(record.run_dir) == token:
        return True
    try:
        if Path(token).expanduser().resolve() == record.run_dir:
            return True
    except OSError:  # pragma: no cover - resolve() raises only on a symlink loop
        return False
    return len(token) >= _MIN_PREFIX and record.key.startswith(token)


def _end_time(record: RunRecord) -> datetime | None:
    return None if record.ended_at is None else datetime.fromisoformat(record.ended_at)


def _candidates(
    plan: ComparisonPlan, records: Iterable[RunRecord]
) -> tuple[dict[PlannedCell, list[RunRecord]], list[RunRecord]]:
    arm_of = {sha: arm for arm, sha in plan.arms.items()}
    wanted = set(plan.cells)
    candidates: dict[PlannedCell, list[RunRecord]] = defaultdict(list)
    outside: list[RunRecord] = []
    for record in records:
        arm = arm_of.get(record.config_sha256)
        if record.command != plan.command or arm is None or record.status != "finished":
            continue
        cell = PlannedCell(arm, record.seed, record.split)
        fits = (
            cell in wanted
            and plan.machine_class in (None, record.machine_class)
            and plan.splits_sha256 in (None, record.splits_sha256)
        )
        if fits:
            candidates[cell].append(record)
        else:
            outside.append(record)
    return candidates, outside


def _by_key(records: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    out: dict[str, list[RunRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda r: str(r.run_dir)):
        out[record.key].append(record)
    return out


def _choose(
    cell: PlannedCell,
    attempts: dict[str, list[RunRecord]],
    selected: str | None,
    policy: AttemptPolicy,
    problems: list[str],
) -> str | None:
    if selected is not None:
        return selected
    if len(attempts) == 1:
        return next(iter(attempts))
    keys = ", ".join(attempts)
    if policy == "latest":
        ends = {key: _end_time(copies[0]) for key, copies in attempts.items()}
        if all(end is not None for end in ends.values()):
            return max(ends, key=lambda key: (ends[key], key))
        problems.append(
            f"{cell.label()}: attempts {keys} include some without an end time (legacy): "
            "choose one with --use"
        )
        return None
    problems.append(
        f"{cell.label()}: {len(attempts)} attempts ({keys}): choose one with --use KEY "
        "or --attempts latest"
    )
    return None


def _selections(
    candidates: Mapping[PlannedCell, list[RunRecord]], use: Sequence[str], problems: list[str]
) -> dict[PlannedCell, str]:
    selected: dict[PlannedCell, str] = {}
    for token in use:
        hits = {
            (cell, record.key)
            for cell, records in candidates.items()
            for record in records
            if _matches_use(record, token)
        }
        if not hits:
            problems.append(f"--use {token}: matches no candidate run of this spec")
            continue
        for cell in sorted({cell for cell, _ in hits}, key=lambda c: c.label()):
            keys = sorted(key for hit_cell, key in hits if hit_cell == cell)
            previous = selected.get(cell)
            if len(keys) > 1:
                problems.append(f"--use {token}: ambiguous in {cell.label()} ({', '.join(keys)})")
            elif previous is not None and previous != keys[0]:
                problems.append(f"--use: two attempts chosen for {cell.label()}")
            else:
                selected[cell] = keys[0]
    return selected


def _check_mixed(
    chosen: Mapping[PlannedCell, RunRecord],
    allow_mixed: frozenset[Mixable],
    problems: list[str],
    notes: list[str],
) -> None:
    records = list(chosen.values())
    kinds: dict[Mixable, Counter[str]] = {
        "machines": Counter(
            repr(machine_signature({"machine": None if r.machine is None else dict(r.machine)}))
            for r in records
        ),
        "engines": Counter(r.engine for r in records),
        "numerics": Counter(
            "unknown (legacy)" if r.numerics_epoch is None else str(r.numerics_epoch)
            for r in records
        ),
    }
    for kind, counts in kinds.items():
        if len(counts) < 2:
            continue
        values = "; ".join(f"{value} ({n} runs)" for value, n in sorted(counts.items()))
        if kind in allow_mixed:
            notes.append(f"mixed {kind} allowed: {values}")
        else:
            problems.append(f"the runs mix {kind}: {values} (allow with --allow-mixed {kind})")
    for name, attribute in (("data", "data_sha256"), ("source data", "data_source_sha256")):
        shas = sorted({sha for r in records if (sha := getattr(r, attribute)) is not None})
        if len(shas) > 1:
            short = ", ".join(sha[:12] for sha in shas)
            problems.append(f"the runs were trained on different {name}: {short}")
    if any(r.legacy for r in records):
        notes.append(
            "legacy runs record no data SHA-256: their data is checked by the source "
            "pickle's SHA-256 only"
        )


def resolve(
    plan: ComparisonPlan,
    records: Iterable[RunRecord],
    *,
    use: Sequence[str] = (),
    attempts: AttemptPolicy = "error",
    allow_mixed: Iterable[Mixable] = (),
) -> Resolution:
    """One finished attempt per cell of ``plan`` from ``records`` (module docstring).

    Raises:
        ComparisonError: listing every missing cell, unresolved choice, inconsistent copy,
            unused ``use`` and disallowed mix.
    """
    problems: list[str] = []
    notes: list[str] = []
    candidates, outside = _candidates(plan, records)
    selected = _selections(candidates, use, problems)
    chosen: dict[PlannedCell, RunRecord] = {}
    copies: dict[PlannedCell, tuple[RunRecord, ...]] = {}
    for cell in plan.cells:
        by_key = _by_key(candidates.get(cell, []))
        if not by_key:
            problems.append(f"{cell.label()}: no finished run")
            continue
        key = _choose(cell, by_key, selected.get(cell), attempts, problems)
        if key is None:
            continue
        held = by_key[key]
        if len({r.metrics_sha256 for r in held}) > 1:
            dirs = ", ".join(str(r.run_dir) for r in held)
            problems.append(f"{cell.label()}: copies of {key} differ in metrics.json ({dirs})")
            continue
        chosen[cell], copies[cell] = held[0], tuple(held[1:])
    if chosen:
        _check_mixed(chosen, frozenset(allow_mixed), problems, notes)
    if problems:
        raise ComparisonError(f"{plan.name} ({plan.spec_id[:16]})", problems)
    if outside:
        keys = sorted({r.key for r in outside})
        notes.append(
            f"{len(keys)} finished attempts of these arms are outside the spec's cells "
            f"(another seed, split, machine class or splits file) and were not used: "
            + ", ".join(keys)
        )
    return Resolution(plan, chosen, copies, tuple(outside), tuple(notes))


def _stage_rank(metric: str) -> tuple[int, str]:
    stage = metric.split(".", 1)[0]
    return ({"val": 0, "test": 2}.get(stage, 1), metric)


def default_metrics(records: Iterable[RunRecord]) -> list[str]:
    """The balanced accuracy and macro-F1 of every stage all ``records`` have (val first)."""
    common: set[str] | None = None
    for record in records:
        keys = {k for k in record.metrics if k.endswith((".balanced_accuracy", ".macro_f1"))}
        common = keys if common is None else common & keys
    return sorted(common or (), key=_stage_rank)


def arm_summary(values: Sequence[float]) -> tuple[float, float | None, int]:
    """Mean, sample standard deviation (ddof=1; ``None`` for one value) and count."""
    n = len(values)
    mean = math.fsum(values) / n
    if n < 2:
        return mean, None, n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance), n


def _missing_metrics(resolution: Resolution, metrics: Sequence[str]) -> list[str]:
    return [
        f"{cell.label()} ({record.key}) has no metric {metric}"
        for cell, record in resolution.chosen.items()
        for metric in metrics
        if metric not in record.metrics
    ]


def _check_metrics(resolution: Resolution, metrics: Sequence[str]) -> None:
    problems = _missing_metrics(resolution, metrics)
    if problems:
        raise ComparisonError(resolution.plan.name, problems)


def _per_arm(resolution: Resolution, metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for cell, record in resolution.chosen.items():
        out[cell.arm].append(record.metrics[metric])
    return out


def resolution_to_plain(resolution: Resolution, metrics: Sequence[str]) -> dict[str, object]:
    """The resolution as JSON values (``runs compare --json``).

    Raises:
        ComparisonError: if a chosen run lacks one of ``metrics``.
    """
    _check_metrics(resolution, metrics)
    plan = resolution.plan
    arms: dict[str, dict[str, object]] = {arm: {} for arm in plan.arms}
    for metric in metrics:
        for arm, values in _per_arm(resolution, metric).items():
            mean, std, n = arm_summary(values)
            arms[arm][metric] = {"mean": mean, "std": std, "n": n}
    return {
        "spec_id": plan.spec_id,
        "name": plan.name,
        "command": plan.command,
        "metrics": list(metrics),
        "cells": [
            {
                "arm": cell.arm,
                "seed": cell.seed,
                "split": cell.split,
                "key": record.key,
                "run_dir": str(record.run_dir),
                "copies": [str(copy.run_dir) for copy in resolution.copies.get(cell, ())],
                "metrics": {metric: record.metrics[metric] for metric in metrics},
            }
            for cell, record in resolution.chosen.items()
        ],
        "arms": arms,
        "notes": list(resolution.notes),
    }


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]


def format_report(resolution: Resolution, metrics: Sequence[str]) -> list[str]:
    """The text report of ``runs compare``: the cells, then mean ± std per arm.

    A report, never a decision: the protocol's decisions are ``compare-variants``'
    (``docs/evaluation-protocol.md``), which takes the run directories listed here.

    Raises:
        ComparisonError: if a chosen run lacks one of ``metrics``.
    """
    _check_metrics(resolution, metrics)
    plan = resolution.plan
    splits = ", ".join("none" if s is None else str(s) for s in plan.splits)
    lines = [
        f"spec {plan.name} ({plan.spec_id}): {plan.command}",
        f"{len(plan.arms)} arms x {len(plan.seeds)} seeds x splits [{splits}] = "
        f"{len(plan.cells)} cells, all resolved",
        "",
    ]
    rows = [["arm", "seed", "split", "run", *metrics, "run_dir"]]
    for cell, record in resolution.chosen.items():
        values = [f"{record.metrics[m]:.4f}" for m in metrics]
        split = "-" if cell.split is None else str(cell.split)
        rows.append([cell.arm, str(cell.seed), split, record.key, *values, str(record.run_dir)])
    lines += _table(rows)
    lines.append("")
    summary = [["arm", "n", *metrics]]
    per_metric = {metric: _per_arm(resolution, metric) for metric in metrics}
    for arm in plan.arms:
        stats = []
        for metric in metrics:
            mean, std, _ = arm_summary(per_metric[metric][arm])
            stats.append(f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}")
        n = sum(cell.arm == arm for cell in resolution.chosen)
        summary.append([arm, str(n), *stats])
    lines += _table(summary)
    copies = sum(len(c) for c in resolution.copies.values())
    if copies:
        lines.append(f"note: {copies} copies of chosen attempts in other directories ignored")
    lines += [f"note: {note}" for note in resolution.notes]
    return lines
