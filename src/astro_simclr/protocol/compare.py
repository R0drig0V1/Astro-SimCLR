"""Comparisons of two variants: ``compare-variants`` and the decision helpers.

- :func:`compare_variants` (CLI ``compare-variants``) reports, for one evaluated stage of
  two sets of runs (a baseline and a variant, one run per seed), the seed-mean accuracies,
  **McNemar on the seed ensembles** and the **paired bootstrap over stamps**. It always
  *reports*: on ``test`` because test is never a decision input, and on ``val`` because a
  normal run's Validation stamps chose its checkpoint. Runs from different machines or
  builds are refused (plan: a comparison never mixes them), unless explicitly allowed, and
  so are runs on different data (``metrics.json["data"]["source"]["sha256"]``): a normal
  run's ``index`` is ``arange(N)``, so only the data identity and the labels can show that
  two runs saw the same stamps.
- :func:`screen_comparison` and :func:`cv_comparison` are the *decision* helpers. They take
  predictions on the screen's decision half or on each CV fold's evaluation part
  (``splits.ProtocolSplits``) and apply the screen rule or the adoption rule. They check
  that the predictions cover exactly those stamps (``Predictions.index``), so a run
  evaluated on other stamps cannot be used by mistake. Whether the run chose its
  checkpoint on the matching selection set is the run's responsibility (its settings).
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np

from astro_simclr.config import ConfigError
from astro_simclr.protocol.predictions import (
    Predictions,
    check_paired,
    ensemble,
    load_predictions,
    seed_mean_correct,
)
from astro_simclr.protocol.splits import ProtocolSplits
from astro_simclr.protocol.stats import (
    BootstrapCI,
    McNemar,
    adoption_decision,
    balanced_accuracy,
    mcnemar,
    paired_bootstrap,
    screen_decision,
)
from astro_simclr.reference import load_run, machine_signature

__all__ = [
    "CVResult",
    "RunPredictions",
    "ScreenResult",
    "VariantComparison",
    "compare_variants",
    "cv_comparison",
    "load_run_predictions",
    "screen_comparison",
]

Stage = Literal["val", "test"]

#: Why each stage's comparison is a report and not a decision.
_NOT_A_DECISION: Final = {
    "test": "test is never a decision input: this comparison is archived and reported once "
    "per pre-registered comparison (docs/evaluation-protocol.md)",
    "val": "these Validation stamps chose each run's checkpoint, so this is not a decision "
    "input either: decide with the screen's decision half or the CV folds",
}


@dataclass(frozen=True, slots=True)
class RunPredictions:
    """One run's predictions on one stage, with what identifies the run."""

    path: str
    seed: object
    split: str
    machine: tuple[object, ...] | None
    #: SHA-256 of the source pickle of the run's prepared data (``metrics.json["data"]``).
    source: object
    predictions: Predictions


def load_run_predictions(run_dir: Path, stage: str) -> RunPredictions:
    """Read ``metrics.json`` of ``run_dir`` and the predictions file it names for ``stage``.

    Raises:
        ConfigError: if the metrics cannot be read or record no predictions for ``stage``
            (runs written before task 2.3 have none: re-run them).
        DataError: if the predictions file is missing or malformed.
    """
    metrics = load_run(run_dir)
    entry = metrics.get("predictions", {}).get(stage)
    if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
        msg = (
            f"{run_dir}: no per-stamp {stage} predictions in metrics.json (runs made before "
            "task 2.3 do not write them; re-run)"
        )
        raise ConfigError(msg)
    root = run_dir.parent if run_dir.is_file() else run_dir
    predictions = load_predictions(root / entry["file"])
    return RunPredictions(
        str(run_dir),
        metrics.get("seed"),
        str(entry.get("split")),
        machine_signature(metrics),
        metrics.get("data", {}).get("source", {}).get("sha256"),
        predictions,
    )


@dataclass(slots=True)
class VariantComparison:
    """The outcome of :func:`compare_variants`: the tests and the lines to print."""

    mcnemar: McNemar
    bootstrap: BootstrapCI
    lines: list[str] = field(default_factory=list)


def _describe(values: Sequence[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f} (n=1)"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} (n={len(values)})"


def _check_side(name: str, runs: Sequence[RunPredictions], out: list[str]) -> None:
    if not runs:
        raise ConfigError(f"compare-variants needs at least one {name} run")
    seeds = [run.seed for run in runs]
    if len(set(map(repr, seeds))) != len(seeds):
        raise ConfigError(f"the {name} runs repeat a seed ({seeds}); one run per seed")
    if len(runs) < 5:
        out.append(f"  WARN: {len(runs)} {name} run(s); the protocol uses 5 seeds")


def _check_machines(runs: Sequence[RunPredictions], allow_mixed: bool, out: list[str]) -> None:
    machines = {run.machine for run in runs}
    if None in machines:
        out.append("  WARN: some runs do not record their machine (made before task 2.3)")
        machines.discard(None)
    if len(machines) > 1:
        detail = "; ".join(sorted(map(str, machines)))
        if not allow_mixed:
            raise ConfigError(
                f"the runs come from different machines or builds ({detail}); a comparison "
                "never mixes them (pass --allow-mixed-machines to override)"
            )
        out.append(f"  WARN: mixed machines or builds: {detail}")


def compare_variants(
    baseline_dirs: Sequence[Path],
    variant_dirs: Sequence[Path],
    *,
    stage: Stage = "test",
    n_boot: int = 10_000,
    seed: int = 0,
    allow_mixed_machines: bool = False,
) -> VariantComparison:
    """Compare two sets of runs on ``stage``: McNemar and the paired bootstrap (a report).

    ``baseline_dirs`` and ``variant_dirs`` are run directories (one per seed). The
    difference is always ``variant - baseline``.

    Raises:
        ConfigError: on missing predictions, a repeated seed within a side, runs on
            different data, unpaired runs (other stamps or labels) or mixed machines
            without ``allow_mixed_machines``.
        DataError: if a predictions file is unreadable.
    """
    out: list[str] = []
    base = [load_run_predictions(path, stage) for path in baseline_dirs]
    variant = [load_run_predictions(path, stage) for path in variant_dirs]
    _check_side("baseline", base, out)
    _check_side("variant", variant, out)
    _check_machines([*base, *variant], allow_mixed_machines, out)
    sources = {repr(run.source) for run in [*base, *variant]}
    if len(sources) > 1:
        raise ConfigError(
            f"the runs were trained on different data (source SHA-256 {sorted(sources)}); "
            "paired stamps must be the same stamps"
        )
    splits = {run.split for run in [*base, *variant]}
    if len(splits) > 1:
        raise ConfigError(f"the runs evaluated different sets: {sorted(splits)}")
    try:
        check_paired([run.predictions for run in [*base, *variant]])
    except ValueError as err:
        raise ConfigError(f"the runs are not paired: {err}") from None

    base_preds = [run.predictions for run in base]
    variant_preds = [run.predictions for run in variant]
    labels = base_preds[0].labels
    test = mcnemar(ensemble(base_preds).correct, ensemble(variant_preds).correct)
    interval = paired_bootstrap(
        seed_mean_correct(base_preds),
        seed_mean_correct(variant_preds),
        labels,
        n_boot=n_boot,
        seed=seed,
    )
    lines = [f"compare-variants: {stage} ({splits.pop()}, {labels.size} stamps)", *out]
    for name, preds in (("baseline", base_preds), ("variant", variant_preds)):
        accuracies = [float(p.correct.mean()) for p in preds]
        ensembled = float(ensemble(preds).correct.mean())
        lines.append(f"  {name:8}  accuracy {_describe(accuracies)}; seed ensemble {ensembled:.4f}")
    lines += [
        f"  McNemar (seed ensembles): baseline only {test.only_baseline}, variant only "
        f"{test.only_variant}, exact p = {test.p_value:.4g}",
        f"  paired bootstrap over stamps (stratified by class, {interval.n_boot} resamples, "
        f"seed {seed}): variant - baseline = {interval.observed:+.4f}, "
        f"{100 * interval.confidence:.0f} % CI [{interval.low:+.4f}, {interval.high:+.4f}]",
        f"  note: {_NOT_A_DECISION[stage]}",
        "compare-variants: REPORT",
    ]
    return VariantComparison(test, interval, lines)


def _check_covers(runs: Sequence[Predictions], expected: np.ndarray, what: str) -> None:
    if not runs:
        raise ValueError(f"no predictions for {what}")
    for run in runs:
        if not np.array_equal(run.index, expected):
            raise ValueError(f"predictions do not cover exactly the stamps of {what}")
    check_paired(runs)


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """The screen on Validation's decision half: seed-mean accuracies and the rule's verdict."""

    baseline: float
    variant: float
    decision: Literal["drop", "keep"]


def screen_comparison(
    baseline: Sequence[Predictions], variant: Sequence[Predictions], splits: ProtocolSplits
) -> ScreenResult:
    """Apply the screen rule to runs evaluated on the screen's decision half.

    Each run's predictions must cover exactly ``splits.screen.decision`` (Validation
    indices), and must come from a run that chose its checkpoint on
    ``splits.screen.selection``.

    Raises:
        ValueError: if a side is empty or a run covers other stamps or labels.
    """
    _check_covers(baseline, splits.screen.decision, "the screen's decision half")
    _check_covers(variant, splits.screen.decision, "the screen's decision half")
    _check_covers([*baseline, *variant], splits.screen.decision, "the screen's decision half")
    base = float(np.mean([p.correct.mean() for p in baseline]))
    var = float(np.mean([p.correct.mean() for p in variant]))
    return ScreenResult(base, var, screen_decision(base, var))


@dataclass(frozen=True, slots=True)
class CVResult:
    """The CV comparison: per-fold balanced accuracies, the pooled interval and the rule."""

    baseline_folds: tuple[float, ...]
    variant_folds: tuple[float, ...]
    interval: BootstrapCI
    decision: Literal["adopt", "worse", "inconclusive"]


def cv_comparison(
    baseline: Mapping[int, Sequence[Predictions]],
    variant: Mapping[int, Sequence[Predictions]],
    splits: ProtocolSplits,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> CVResult:
    """Apply the adoption rule to a CV run of a flag against the baseline.

    ``baseline[k]`` and ``variant[k]`` are the runs (2 seeds in the plan) of fold ``k``,
    each evaluated on exactly ``splits.folds[k].evaluation`` (pool indices). Every pool
    stamp is held out by exactly one fold, so the per-stamp seed-mean scores of the folds
    are concatenated and the paired, class-stratified bootstrap of the **balanced
    accuracy** difference runs over all of them.

    Raises:
        ValueError: if a fold is missing on either side, or a run covers other stamps.
    """
    scores_base, scores_var, labels = [], [], []
    folds_base, folds_var = [], []
    for fold in splits.folds:
        what = f"fold {fold.fold}'s evaluation part"
        if fold.fold not in baseline or fold.fold not in variant:
            raise ValueError(f"no predictions for fold {fold.fold}")
        _check_covers(baseline[fold.fold], fold.evaluation, what)
        _check_covers(variant[fold.fold], fold.evaluation, what)
        _check_covers([*baseline[fold.fold], *variant[fold.fold]], fold.evaluation, what)
        base, var = seed_mean_correct(baseline[fold.fold]), seed_mean_correct(variant[fold.fold])
        fold_labels = baseline[fold.fold][0].labels
        folds_base.append(balanced_accuracy(base, fold_labels))
        folds_var.append(balanced_accuracy(var, fold_labels))
        scores_base.append(base)
        scores_var.append(var)
        labels.append(fold_labels)
    interval = paired_bootstrap(
        np.concatenate(scores_base),
        np.concatenate(scores_var),
        np.concatenate(labels),
        metric="balanced_accuracy",
        n_boot=n_boot,
        seed=seed,
    )
    return CVResult(tuple(folds_base), tuple(folds_var), interval, adoption_decision(interval))
