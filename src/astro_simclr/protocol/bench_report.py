"""``bench-report``: the phase-3 benchmark report of runs with per-alert predictions.

Task 3.2 of ``docs/phases/03-newer-benchmarks.md``. A *benchmark run* (written by the
RJ23 / C25 runs of tasks 3.4 and 3.6) is a directory with a ``metrics.json`` and one
predictions file per evaluated split (:mod:`astro_simclr.protocol.predictions`, with
``n_classes`` = the number of label names and ``index`` = the alert's ``candid``). The
contract of ``metrics.json``:

=============================  ===========================================================
``seed``                       the run's seed (one run per seed and side)
``machine``                    the run's machine block (``engine.record.machine_record``:
                               device, gpu, torch, cuda); runs of one report must share it
``benchmark.splits_sha256``    SHA-256 of the object-splits file the run was trained on
``predictions.<stage>.file``   the predictions file of ``stage`` (``test`` or ``val``)
=============================  ===========================================================

:func:`benchmark_report` reads the object splits and the alert table, verifies them
(:func:`~astro_simclr.protocol.object_splits.verify_object_splits`), checks that every
run was trained on that splits file and predicted **exactly** the split's alerts with the
table's labels, and reports, on two views:

- **all alerts** of the split;
- **first stamp**: each object's earliest alert
  (:func:`~astro_simclr.protocol.alerts.first_stamps`), RJ23's second view.

For each view and side:

- macro-F1, macro precision, macro recall and accuracy as mean ± std (ddof = 1) over the
  seeds, and the macro-F1 of the seed ensemble (mean probabilities, then argmax);
- the **prior-reweighted** macro-F1 (rows rescaled to reference class counts, RJ23's class
  proportions by default: :mod:`astro_simclr.protocol.benchmark_metrics`), R5's headline;
- per class: the support (alerts and objects), precision, recall and F1 (mean ± std), and
  the **minimum detectable effect** of a recall difference
  (:func:`~astro_simclr.protocol.stats.minimum_detectable_effect`, on the class's objects:
  paired from the ensembles' discordance with a baseline, unpaired from the ensemble's
  recall without one);
- the confusion matrix in percent of each true class (mean over seeds).

With baseline runs it also reports the paired comparison, variant - baseline, on the **seed
ensembles**: the class-stratified paired bootstrap of the macro-F1 difference
(:func:`~astro_simclr.protocol.stats.paired_bootstrap_macro_f1`) and the exact McNemar test.
It is always a **report** (exit 0): choosing between variants on Test would make Test a
decision input (``docs/evaluation-protocol.md``).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from astro_simclr.config import ConfigError
from astro_simclr.protocol.alerts import AlertTable, first_stamps, read_alert_table
from astro_simclr.protocol.benchmark_metrics import (
    BenchmarkMetrics,
    benchmark_metrics,
    macro_f1,
    rj23_reference,
)
from astro_simclr.protocol.object_splits import read_object_splits, verify_object_splits
from astro_simclr.protocol.predictions import Predictions, ensemble, load_predictions
from astro_simclr.protocol.split_runs import file_sha256
from astro_simclr.protocol.stats import (
    mcnemar,
    minimum_detectable_effect,
    paired_bootstrap_macro_f1,
)
from astro_simclr.reference import load_run, machine_signature

__all__ = [
    "STAGES",
    "BenchmarkRun",
    "View",
    "benchmark_report",
    "load_benchmark_run",
]

#: Evaluated stage of a run -> the split it predicts.
STAGES: Final = {"test": "Test", "val": "Validation"}


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """One run's predictions on one split, sorted by ``candid``, with its identity."""

    path: str
    seed: object
    machine: tuple[object, ...] | None
    predictions: Predictions


@dataclass(frozen=True, slots=True)
class View:
    """An evaluation view: positions into the split's alerts, and each one's ``oid``."""

    name: str
    positions: npt.NDArray[np.int64]
    oid: npt.NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class _Side:
    """One side's runs and their seed ensemble."""

    name: str
    runs: Sequence[BenchmarkRun]
    ensemble: Predictions


def load_benchmark_run(
    run_dir: Path,
    *,
    stage: str,
    splits_sha256: str,
    table: AlertTable,
    rows: npt.NDArray[np.int64],
) -> BenchmarkRun:
    """Read a benchmark run's predictions of ``stage`` and align them to ``rows``.

    ``rows`` are the split's alert rows of ``table`` (sorted by ``candid``). The returned
    predictions follow that order.

    Raises:
        ConfigError: if ``metrics.json`` is unreadable, records another splits file (or
            none), or has no predictions for ``stage``; or if the predictions do not cover
            exactly the split's alerts, once each, with the table's labels.
        DataError: if the predictions file is missing or malformed.
    """
    metrics = load_run(run_dir)
    bench = metrics.get("benchmark")
    recorded = bench.get("splits_sha256") if isinstance(bench, dict) else None
    if recorded != splits_sha256:
        msg = (
            f"{run_dir}: trained on another object-splits file (benchmark.splits_sha256 "
            f"{recorded!r}, expected {splits_sha256})"
        )
        raise ConfigError(msg)
    entry = metrics.get("predictions", {}).get(stage)
    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
        raise ConfigError(f"{run_dir}: no per-alert {stage} predictions in metrics.json")
    root = run_dir.parent if run_dir.is_file() else run_dir
    raw = load_predictions(root / entry["file"], n_classes=len(table.label_names))
    order = np.argsort(raw.index, kind="stable")
    candids = raw.index[order]
    if not np.array_equal(candids, table.candid[rows]):
        msg = (
            f"{run_dir}: the {stage} predictions cover {raw.index.size} alerts, not exactly "
            f"the {rows.size} alerts of the split (by candid, once each)"
        )
        raise ConfigError(msg)
    if not np.array_equal(raw.labels[order], table.label[rows]):
        raise ConfigError(f"{run_dir}: the {stage} labels differ from the alert table's")
    aligned = Predictions(
        raw.probabilities[order], raw.labels[order], candids, len(table.label_names)
    )
    return BenchmarkRun(str(run_dir), metrics.get("seed"), machine_signature(metrics), aligned)


def _pct(value: float) -> str:
    return f"{100 * value:6.2f}"


def _mean_std(values: Sequence[float | None]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "   n/a"
    if len(present) == 1:
        return _pct(present[0])
    return f"{_pct(statistics.mean(present))} ± {100 * statistics.stdev(present):5.2f}"


def _check_runs(name: str, runs: Sequence[BenchmarkRun], out: list[str]) -> None:
    seeds = [run.seed for run in runs]
    if len(set(map(repr, seeds))) != len(seeds):
        raise ConfigError(f"the {name} runs repeat a seed ({seeds}); one run per seed")
    if len(runs) < 5:
        out.append(f"  WARN: {len(runs)} {name} run(s); the benchmarks use 5 seeds")


def _check_machines(runs: Sequence[BenchmarkRun], allow_mixed: bool, out: list[str]) -> None:
    machines = {run.machine for run in runs}
    if None in machines:
        out.append("  WARN: some runs do not record their machine")
        machines.discard(None)
    if len(machines) > 1:
        detail = "; ".join(sorted(map(str, machines)))
        if not allow_mixed:
            raise ConfigError(
                "the runs come from different machines, builds, float32 precisions or "
                f"engines ({detail}); a report never mixes them (pass --allow-mixed-machines "
                "to override)"
            )
        out.append(f"  WARN: mixed machines, builds, precisions or engines: {detail}")


def _reference(names: tuple[str, ...], given: Sequence[float] | None) -> tuple[float, ...] | None:
    if given is not None:
        if len(given) != len(names):
            raise ConfigError(f"--reference-counts needs {len(names)} values, got {len(given)}")
        return tuple(given)
    return rj23_reference(names)


def _mde(side: _Side, view: View, c: int, baseline: _Side | None) -> float:
    """The class's MDE of a recall difference on the view, as a fraction (module docstring)."""
    members = side.ensemble.labels[view.positions] == c
    n_objects = int(np.unique(view.oid[members]).size)
    correct = side.ensemble.correct[view.positions][members]
    if baseline is None:
        return minimum_detectable_effect(n_objects, rate=float(correct.mean()))
    other = baseline.ensemble.correct[view.positions][members]
    return minimum_detectable_effect(n_objects, discordant=float(np.mean(correct != other)))


def _describe(
    side: _Side,
    view: View,
    names: tuple[str, ...],
    reference: tuple[float, ...] | None,
    baseline: _Side | None,
    out: list[str],
) -> None:
    scores: list[BenchmarkMetrics] = [
        benchmark_metrics(
            run.predictions.labels[view.positions],
            run.predictions.predicted[view.positions],
            names,
            reference_counts=reference,
        )
        for run in side.runs
    ]
    labels = side.ensemble.labels[view.positions]
    ensembled = macro_f1(labels, side.ensemble.predicted[view.positions], len(names))
    out.append(
        f"  {side.name}: macro-F1 {_mean_std([s.macro_f1 for s in scores])}  "
        f"macro-P {_mean_std([s.macro_precision for s in scores])}  "
        f"macro-R {_mean_std([s.macro_recall for s in scores])}  "
        f"accuracy {_mean_std([s.accuracy for s in scores])}  "
        f"(%, mean ± std over {len(scores)} seeds); seed ensemble macro-F1 {_pct(ensembled)}"
    )
    if reference is not None:
        out.append(
            f"    prior-reweighted to {[round(r) for r in reference]}: macro-F1 "
            f"{_mean_std([s.reweighted_macro_f1 for s in scores])}  macro-P "
            f"{_mean_std([s.reweighted_macro_precision for s in scores])}"
        )
    mde = "MDE paired" if baseline is not None else "MDE"
    out.append(
        f"    {'class':10} {'alerts':>6} {'objects':>7}  {'precision':>15}  {'recall':>15}  "
        f"{'F1':>15}  {mde:>10}"
    )
    for c, name in enumerate(names):
        per = [s.per_class[c] for s in scores]
        n_objects = int(np.unique(view.oid[labels == c]).size)
        detectable = f"{100 * _mde(side, view, c, baseline):5.2f}" if n_objects else "  n/a"
        out.append(
            f"    {name:10} {per[0].support:6d} {n_objects:7d}  "
            f"{_mean_std([p.precision for p in per]):>15}  "
            f"{_mean_std([p.recall for p in per]):>15}  {_mean_std([p.f1 for p in per]):>15}  "
            f"{detectable:>10}"
        )
    percent = np.mean([s.percent for s in scores], axis=0)
    out.append(
        "    confusion (% of each true class, mean over seeds; rows true, columns predicted)"
    )
    out.append("    " + " " * 10 + "".join(f"{name[:8]:>9}" for name in names))
    for name, row in zip(names, percent, strict=True):
        out.append(f"    {name:10}" + "".join(f"{value:9.1f}" for value in row))


def _compare(
    baseline: _Side,
    variant: _Side,
    view: View,
    n_classes: int,
    n_boot: int,
    seed: int,
    out: list[str],
) -> None:
    interval = paired_bootstrap_macro_f1(
        baseline.ensemble.predicted[view.positions],
        variant.ensemble.predicted[view.positions],
        variant.ensemble.labels[view.positions],
        n_classes=n_classes,
        n_boot=n_boot,
        seed=seed,
    )
    test = mcnemar(
        baseline.ensemble.correct[view.positions], variant.ensemble.correct[view.positions]
    )
    out.append(
        f"  variant - baseline, seed-ensemble macro-F1: {100 * interval.observed:+.2f} "
        f"points, 95 % CI [{100 * interval.low:+.2f}, {100 * interval.high:+.2f}] (paired "
        f"class-stratified bootstrap, n_boot={n_boot}, seed {seed})"
    )
    out.append(
        f"  McNemar (seed ensembles): baseline only {test.only_baseline}, variant only "
        f"{test.only_variant}, p = {test.p_value:.4g}"
    )


def benchmark_report(
    run_dirs: Sequence[Path],
    splits_file: Path,
    alerts_file: Path,
    *,
    baseline_dirs: Sequence[Path] = (),
    stage: str = "test",
    reference_counts: Sequence[float] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    allow_mixed_machines: bool = False,
) -> list[str]:
    """The report's lines (module docstring): the runs and, if given, the paired comparison.

    ``run_dirs`` are the runs reported (the *variant* when ``baseline_dirs`` is given), one
    per seed. ``reference_counts`` (one per class) set the prior-reweighted macro-F1;
    without them, RJ23's class counts are used when every class is one of RJ23's, and the
    reweighting is left out otherwise. ``seed`` is the bootstrap's seed.

    Raises:
        ConfigError: on an unknown ``stage``, no runs, runs of another splits file, a
            repeated seed within a side, predictions of other alerts or labels, mixed
            machines (unless allowed), or reference counts of another length.
        DataError: if the alert table, the object-splits file (``SplitsError``) or a
            predictions file is unusable, or the splits do not verify against the table.
    """
    if stage not in STAGES:
        raise ConfigError(f"unknown stage {stage!r}; use one of {sorted(STAGES)}")
    if not run_dirs:
        raise ConfigError("bench-report needs at least one run")
    splits = read_object_splits(splits_file)
    names = splits.label_names
    reference = _reference(names, reference_counts)
    table = read_alert_table(alerts_file, names)
    verify_object_splits(splits, table)
    sha = file_sha256(splits_file)
    split = STAGES[stage]
    rows = splits.rows(table, split)

    def side(name: str, dirs: Sequence[Path]) -> _Side:
        runs = [
            load_benchmark_run(d, stage=stage, splits_sha256=sha, table=table, rows=rows)
            for d in dirs
        ]
        return _Side(name, runs, ensemble([run.predictions for run in runs]))

    baseline = side("baseline", baseline_dirs) if baseline_dirs else None
    variant = side("variant" if baseline else "runs", run_dirs)
    out = [
        f"bench-report: {split} of {splits_file} (seed {splits.seed}, groups by "
        f"{splits.group_key}; table {splits.table.get('name')!r}, sha256 {table.sha256[:12]})"
    ]
    sides = [variant] if baseline is None else [baseline, variant]
    for s in sides:
        _check_runs(s.name, s.runs, out)
    _check_machines([run for s in sides for run in s.runs], allow_mixed_machines, out)
    first = np.searchsorted(rows, first_stamps(table, rows)).astype(np.int64)
    oid = table.oid[rows]
    views = (
        View("all alerts", np.arange(rows.size, dtype=np.int64), oid),
        View("first stamp", first, oid[first]),
    )
    for view in views:
        n_objects = int(np.unique(view.oid).size)
        out.append(f"view {view.name}: {view.positions.size} alerts of {n_objects} objects")
        for s in sides:
            _describe(s, view, names, reference, baseline if s is variant else None, out)
        if baseline is not None:
            _compare(baseline, variant, view, len(names), n_boot, seed, out)
    note = " (Test is never a decision input)" if split == "Test" else ""
    out.append(f"bench-report: REPORT{note}")
    return out
