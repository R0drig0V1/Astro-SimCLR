"""Comparisons of two variants: ``compare-variants`` and the decision helpers.

- :func:`compare_variants` (CLI ``compare-variants``) reports, for one evaluated stage of
  two sets of runs (a baseline and a variant, one run per seed), the seed-mean accuracies,
  **McNemar on the seed ensembles** and the **paired bootstrap over stamps**. It always
  *reports*: on ``test`` because test is never a decision input, and on ``val`` because a
  normal run's Validation stamps chose its checkpoint. Runs from different machines,
  builds or effective float32 precisions (ADR 0010) are refused (plan: a comparison never
  mixes them), unless explicitly allowed, and so are runs on different data
  (``metrics.json["data"]["source"]["sha256"]``): a normal run's ``index`` is
  ``arange(N)``, so only the data identity and the labels can show that two runs saw the
  same stamps.
- :func:`screen_comparison` and :func:`cv_comparison` are the *decision* helpers. They take
  predictions on the screen's decision half or on each CV fold's evaluation part
  (``splits.ProtocolSplits``) and apply the screen rule or the adoption rule. They check
  that the predictions cover exactly those stamps (``Predictions.index``), so a run
  evaluated on other stamps cannot be used by mistake. Whether the run chose its
  checkpoint on the matching selection set is the run's responsibility (its settings).
  :func:`cv_comparison` applies the superiority rule of task 2.3 or, for the efficiency
  rows of task 2.5, the non-inferiority rule (``rule="non-inferiority"``, ``margin``).
- :func:`compare_decision` (CLI ``compare-variants --splits FILE``, task 2.5-infra) reads
  *split runs* (``train-supervised --splits FILE --fold K | --screen``,
  :mod:`astro_simclr.protocol.split_runs`): it checks that every run was trained on the
  given splits file (its SHA-256), in one mode, on the same data and machine, and that the
  two sides cover the same folds and seeds; then it applies :func:`cv_comparison` (folds)
  or :func:`screen_comparison` (screen; drop only) and returns a verdict that the CLI maps
  to its exit code: 0 for ``ADOPT`` / ``KEEP``, 1 for ``NOT ADOPTED`` / ``DROP``.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
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
from astro_simclr.protocol.split_runs import FOLD, SCREEN, file_sha256
from astro_simclr.protocol.splits import ProtocolSplits, read_splits
from astro_simclr.protocol.stats import (
    NON_INFERIORITY_MARGIN,
    SCREEN_DROP_THRESHOLD,
    Adoption,
    BootstrapCI,
    McNemar,
    Rule,
    adoption_decision,
    balanced_accuracy,
    check_margin,
    mcnemar,
    non_inferiority_decision,
    paired_bootstrap,
    screen_decision,
)
from astro_simclr.reference import load_run, machine_signature

__all__ = [
    "CVResult",
    "DecisionComparison",
    "RunPredictions",
    "ScreenResult",
    "SplitRunPredictions",
    "VariantComparison",
    "compare_decision",
    "compare_variants",
    "cv_comparison",
    "load_run_predictions",
    "load_split_run",
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
        if isinstance(metrics.get("protocol"), dict):
            msg = (
                f"{run_dir}: a --splits run has no {stage} predictions; compare split runs "
                "with compare-variants --splits FILE"
            )
            raise ConfigError(msg)
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
                f"the runs come from different machines, builds or float32 precisions "
                f"({detail}); a comparison never mixes them (pass --allow-mixed-machines to "
                "override)"
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
    """The CV comparison: per-fold and pooled balanced accuracies, the interval and the rule.

    ``baseline_pooled`` / ``variant_pooled`` are the balanced accuracies of the
    concatenated per-stamp seed-mean scores (every pool stamp once), whose difference is
    ``interval.observed``. ``mcnemar`` compares the seed ensembles over the same stamps.
    """

    baseline_folds: tuple[float, ...]
    variant_folds: tuple[float, ...]
    interval: BootstrapCI
    decision: Adoption
    rule: Rule = "superiority"
    margin: float | None = None
    baseline_pooled: float | None = None
    variant_pooled: float | None = None
    mcnemar: McNemar | None = None


def cv_comparison(
    baseline: Mapping[int, Sequence[Predictions]],
    variant: Mapping[int, Sequence[Predictions]],
    splits: ProtocolSplits,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    rule: Rule = "superiority",
    margin: float = NON_INFERIORITY_MARGIN,
) -> CVResult:
    """Apply the adoption rule to a CV run of a flag against the baseline.

    ``baseline[k]`` and ``variant[k]`` are the runs (2 seeds in the plan) of fold ``k``,
    each evaluated on exactly ``splits.folds[k].evaluation`` (pool indices). Every pool
    stamp is held out by exactly one fold, so the per-stamp seed-mean scores of the folds
    are concatenated and the paired, class-stratified bootstrap of the **balanced
    accuracy** difference runs over all of them. The pairing is by pool index: both sides
    must cover each fold's evaluation part in its stored (sorted) order, so position ``i``
    of the concatenation is the same stamp on both sides.

    ``rule`` is ``superiority`` (:func:`~astro_simclr.protocol.stats.adoption_decision`,
    task 2.3) or ``non-inferiority`` (:func:`~astro_simclr.protocol.stats.
    non_inferiority_decision` with ``margin``, the efficiency rows of task 2.5).

    Raises:
        ValueError: if a fold is missing on either side, a run covers other stamps, the
            rule is unknown or the margin is out of range.
    """
    if rule not in ("superiority", "non-inferiority"):
        raise ValueError(f"unknown rule {rule!r}; use superiority or non-inferiority")
    scores_base, scores_var, labels = [], [], []
    correct_base, correct_var = [], []
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
        correct_base.append(ensemble(baseline[fold.fold]).correct)
        correct_var.append(ensemble(variant[fold.fold]).correct)
    pooled_base, pooled_var = np.concatenate(scores_base), np.concatenate(scores_var)
    pooled_labels = np.concatenate(labels)
    interval = paired_bootstrap(
        pooled_base,
        pooled_var,
        pooled_labels,
        metric="balanced_accuracy",
        n_boot=n_boot,
        seed=seed,
    )
    if rule == "superiority":
        decision, used_margin = adoption_decision(interval), None
    else:
        decision, used_margin = non_inferiority_decision(interval, margin=margin), margin
    return CVResult(
        tuple(folds_base),
        tuple(folds_var),
        interval,
        decision,
        rule=rule,
        margin=used_margin,
        baseline_pooled=balanced_accuracy(pooled_base, pooled_labels),
        variant_pooled=balanced_accuracy(pooled_var, pooled_labels),
        mcnemar=mcnemar(np.concatenate(correct_base), np.concatenate(correct_var)),
    )


# --- compare-variants --splits FILE (task 2.5-infra) -----------------------------------------


@dataclass(frozen=True, slots=True)
class SplitRunPredictions:
    """One split run's evaluation-part predictions, with what identifies the run."""

    path: str
    seed: object
    mode: str
    fold: int | None
    splits_sha256: object
    machine: tuple[object, ...] | None
    source: object
    predictions: Predictions


def load_split_run(run_dir: Path) -> SplitRunPredictions:
    """Read a split run's ``metrics.json`` (its ``protocol`` entry) and evaluation predictions.

    Raises:
        ConfigError: if the metrics cannot be read, or the run is not a ``--splits`` run
            (no ``protocol`` entry or no ``evaluation`` predictions).
        DataError: if the predictions file is missing or malformed.
    """
    metrics = load_run(run_dir)
    protocol = metrics.get("protocol")
    entry = metrics.get("predictions", {}).get("evaluation")
    if (
        not isinstance(protocol, dict)
        or not isinstance(entry, dict)
        or not isinstance(entry.get("file"), str)
    ):
        msg = (
            f"{run_dir}: not a --splits run (no protocol entry or evaluation predictions in "
            "metrics.json): train it with train-supervised --splits FILE --fold K | --screen"
        )
        raise ConfigError(msg)
    root = run_dir.parent if run_dir.is_file() else run_dir
    fold = protocol.get("fold")
    return SplitRunPredictions(
        str(run_dir),
        metrics.get("seed"),
        str(protocol.get("mode")),
        fold if isinstance(fold, int) and not isinstance(fold, bool) else None,
        protocol.get("splits_sha256"),
        machine_signature(metrics),
        metrics.get("data", {}).get("source", {}).get("sha256"),
        load_predictions(root / entry["file"]),
    )


@dataclass(slots=True)
class DecisionComparison:
    """The outcome of :func:`compare_decision`: the verdict, the result and the lines."""

    mode: str
    verdict: str
    passed: bool
    cv: CVResult | None = None
    screen: ScreenResult | None = None
    lines: list[str] = field(default_factory=list)


def _cell(run: SplitRunPredictions) -> tuple[int | None, str]:
    return run.fold, repr(run.seed)


def _cells_text(cells: Iterable[tuple[int | None, str]]) -> str:
    """``fold 0 seed 1, ...`` (``seed 1, ...`` for the screen), sorted."""
    return ", ".join(
        f"seed {seed}" if fold is None else f"fold {fold} seed {seed}"
        for fold, seed in sorted(cells, key=lambda cell: (cell[0] or 0, cell[1]))
    )


def _check_cells(
    base: Sequence[SplitRunPredictions],
    variant: Sequence[SplitRunPredictions],
    splits: ProtocolSplits,
    mode: str,
) -> dict[int | None, list[str]]:
    """Check that both sides cover the same (fold, seed) cells, once each, every fold.

    Returns the seeds of each fold (the screen's "fold" is ``None``).

    Raises:
        ConfigError: on a repeated cell, a fold of the file missing (CV), a cell present
            on one side only, or folds with different numbers of seeds.
    """
    cells: dict[str, set[tuple[int | None, str]]] = {}
    for name, runs in (("baseline", base), ("variant", variant)):
        if not runs:
            raise ConfigError(f"compare-variants needs at least one {name} run")
        seen = [_cell(run) for run in runs]
        repeated = {cell for cell in seen if seen.count(cell) > 1}
        if repeated:
            msg = f"the {name} runs repeat {_cells_text(repeated)}; one run per fold and seed"
            raise ConfigError(msg)
        cells[name] = set(seen)
        if mode == FOLD:
            covered = {fold for fold, _ in seen}
            missing = sorted(f.fold for f in splits.folds if f.fold not in covered)
            if missing:
                raise ConfigError(f"the {name} runs miss fold(s) {missing} of the splits file")
    for name, other in (("baseline", "variant"), ("variant", "baseline")):
        extra = cells[name] - cells[other]
        if extra:
            msg = (
                f"in the {name} runs but not in the {other} runs: {_cells_text(extra)}; both "
                "sides must cover the same folds and seeds"
            )
            raise ConfigError(msg)
    seeds: dict[int | None, list[str]] = {}
    for fold, seed in sorted(cells["baseline"], key=lambda cell: (cell[0] or 0, cell[1])):
        seeds.setdefault(fold, []).append(seed)
    counts = {fold: len(fold_seeds) for fold, fold_seeds in seeds.items()}
    if len(set(counts.values())) > 1:
        # The pooled bootstrap would mix stamps scored over 1 and over 2 seeds.
        msg = (
            f"the folds have different numbers of seeds ({counts}); every fold needs the "
            "same seeds, so that every pool stamp is scored the same way"
        )
        raise ConfigError(msg)
    return seeds


def _check_identity(runs: Sequence[SplitRunPredictions], sha256: str, file: Path) -> str:
    """Check the splits file, the mode and the data of every run; return the mode.

    Raises:
        ConfigError: if a run was trained on another splits file, the runs mix modes or
            data, or a fold run has no fold.
    """
    for run in runs:
        if run.splits_sha256 != sha256:
            msg = (
                f"{run.path} was trained on another splits file (SHA-256 "
                f"{run.splits_sha256}), not on {file} ({sha256})"
            )
            raise ConfigError(msg)
        if run.mode not in (FOLD, SCREEN) or (run.mode == FOLD and run.fold is None):
            raise ConfigError(f"{run.path}: unknown split-run mode {run.mode!r}")
    modes = {run.mode for run in runs}
    if len(modes) > 1:
        raise ConfigError("the runs mix CV folds and the screen; compare one mode at a time")
    sources = {repr(run.source) for run in runs}
    if len(sources) > 1:
        raise ConfigError(
            f"the runs were trained on different data (source SHA-256 {sorted(sources)})"
        )
    return modes.pop()


def compare_decision(
    baseline_dirs: Sequence[Path],
    variant_dirs: Sequence[Path],
    splits_file: Path,
    *,
    rule: Rule | None = None,
    margin: float = NON_INFERIORITY_MARGIN,
    n_boot: int = 10_000,
    seed: int = 0,
    allow_mixed_machines: bool = False,
) -> DecisionComparison:
    """Decide a variant against the baseline from split runs (``compare-variants --splits``).

    ``baseline_dirs`` and ``variant_dirs`` are the run directories of
    ``train-supervised --splits splits_file``: one per fold and seed (CV), or one per
    seed (the screen). Both sides must cover the same cells, CV runs every fold of the file.

    - **CV** (fold runs): :func:`cv_comparison` with ``rule`` (default ``superiority``;
      ``non-inferiority`` adopts when the CI's lower bound is above ``-margin``). Verdict
      ``ADOPT`` or ``NOT ADOPTED``.
    - **Screen** runs: :func:`screen_comparison`, drop only. Verdict ``DROP`` or ``KEEP``
      (not dropped: the screen never adopts). ``rule`` must be unset.

    ``passed`` is true for ``ADOPT`` and ``KEEP``.

    Raises:
        ConfigError: if a directory is not a split run, a run was trained on another splits
            file, the runs mix modes, data or (without ``allow_mixed_machines``) machines,
            the sides cover other folds or seeds, a fold is missing, a rule is given for the
            screen, or a run's predictions do not cover its part.
        SplitsError: if ``splits_file`` cannot be read or is malformed.
        DataError: if a predictions file is unreadable.
    """
    if rule == "non-inferiority":
        try:
            check_margin(margin)
        except ValueError as err:
            raise ConfigError(str(err)) from None
    splits = read_splits(splits_file)
    sha256 = file_sha256(splits_file)
    base = [load_split_run(path) for path in baseline_dirs]
    variant = [load_split_run(path) for path in variant_dirs]
    out: list[str] = []
    mode = _check_identity([*base, *variant], sha256, splits_file)
    seeds = _check_cells(base, variant, splits, mode)
    machines = [
        RunPredictions(run.path, run.seed, run.mode, run.machine, run.source, run.predictions)
        for run in [*base, *variant]
    ]
    _check_machines(machines, allow_mixed_machines, out)
    if mode == SCREEN:
        return _decide_screen(base, variant, splits, rule, seeds, out)
    return _decide_cv(
        base,
        variant,
        splits,
        seeds,
        out,
        rule=rule or "superiority",
        margin=margin,
        n_boot=n_boot,
        seed=seed,
    )


def _by_fold(runs: Sequence[SplitRunPredictions]) -> dict[int, list[Predictions]]:
    """The runs' predictions per fold, ordered by seed (the same order on both sides)."""
    grouped: dict[int, list[tuple[str, Predictions]]] = {}
    for run in runs:
        if run.fold is None:  # excluded by _check_identity
            raise ConfigError(f"{run.path}: a fold run without a fold")
        grouped.setdefault(run.fold, []).append((repr(run.seed), run.predictions))
    return {
        fold: [p for _, p in sorted(items, key=lambda item: item[0])]
        for fold, items in grouped.items()
    }


def _by_seed(runs: Sequence[SplitRunPredictions]) -> list[Predictions]:
    return [run.predictions for run in sorted(runs, key=lambda run: repr(run.seed))]


def _seed_list(seeds: Sequence[str]) -> str:
    return "[" + ", ".join(seeds) + "]"


def _decide_cv(
    base: Sequence[SplitRunPredictions],
    variant: Sequence[SplitRunPredictions],
    splits: ProtocolSplits,
    seeds: Mapping[int | None, Sequence[str]],
    out: list[str],
    *,
    rule: Rule,
    margin: float,
    n_boot: int,
    seed: int,
) -> DecisionComparison:
    try:
        result = cv_comparison(
            _by_fold(base),
            _by_fold(variant),
            splits,
            n_boot=n_boot,
            seed=seed,
            rule=rule,
            margin=margin,
        )
    except ValueError as err:
        raise ConfigError(f"the runs cannot be compared: {err}") from None
    n_stamps = sum(fold.evaluation.size for fold in splits.folds)
    lines = [
        f"compare-variants: CV ({len(splits.folds)} folds, {n_stamps} pool stamps, "
        "balanced accuracy)",
        *out,
    ]
    for fold, b, v in zip(splits.folds, result.baseline_folds, result.variant_folds, strict=True):
        fold_seeds = seeds[fold.fold]
        if len(fold_seeds) < 2:
            lines.append(f"  WARN: fold {fold.fold} has {len(fold_seeds)} seed(s); the plan uses 2")
        lines.append(
            f"  fold {fold.fold}: baseline {b:.4f}, variant {v:.4f}, variant - baseline "
            f"{v - b:+.4f} ({fold.evaluation.size} stamps, seeds {_seed_list(fold_seeds)})"
        )
    interval = result.interval
    test = result.mcnemar
    lines.append(
        f"  pooled (seed means): baseline {result.baseline_pooled:.4f}, variant "
        f"{result.variant_pooled:.4f}"
    )
    if test is not None:
        lines.append(
            f"  McNemar (seed ensembles, all folds): baseline only {test.only_baseline}, "
            f"variant only {test.only_variant}, exact p = {test.p_value:.4g}"
        )
    lines.append(
        f"  paired bootstrap over pool stamps (paired by pool index, stratified by class, "
        f"{interval.n_boot} resamples, seed {seed}): variant - baseline = "
        f"{interval.observed:+.4f}, {100 * interval.confidence:.0f} % CI "
        f"[{interval.low:+.4f}, {interval.high:+.4f}]"
    )
    if rule == "superiority":
        lines.append(f"  rule: superiority (adopt only if the CI lies above 0): {result.decision}")
    else:
        lines.append(
            f"  rule: non-inferiority, margin {margin:.4f} ({100 * margin:.2f} points; adopt "
            f"if the CI lower bound is above {-margin:+.4f}): {result.decision}"
        )
    adopted = result.decision == "adopt"
    verdict = "ADOPT" if adopted else "NOT ADOPTED"
    lines.append(f"compare-variants: {verdict}" + ("" if adopted else f" ({result.decision})"))
    return DecisionComparison(FOLD, verdict, adopted, cv=result, lines=lines)


def _decide_screen(
    base: Sequence[SplitRunPredictions],
    variant: Sequence[SplitRunPredictions],
    splits: ProtocolSplits,
    rule: Rule | None,
    seeds: Mapping[int | None, Sequence[str]],
    out: list[str],
) -> DecisionComparison:
    if rule is not None:
        msg = (
            "the screen is drop-only: --rule applies to CV runs (the screen never adopts; "
            "its minimum detectable effect is about 4 points)"
        )
        raise ConfigError(msg)
    base_preds, variant_preds = _by_seed(base), _by_seed(variant)
    try:
        result = screen_comparison(base_preds, variant_preds, splits)
    except ValueError as err:
        raise ConfigError(f"the runs cannot be compared: {err}") from None
    test = mcnemar(ensemble(base_preds).correct, ensemble(variant_preds).correct)
    dropped = result.decision == "drop"
    verdict = "DROP" if dropped else "KEEP"
    lines = [
        f"compare-variants: screen ({splits.screen.decision.size} decision stamps of "
        f"Validation, seeds {_seed_list(seeds[None])})",
        *out,
        f"  accuracy (seed means): baseline {result.baseline:.4f}, variant "
        f"{result.variant:.4f}, variant - baseline {result.variant - result.baseline:+.4f}",
        f"  McNemar (seed ensembles): baseline only {test.only_baseline}, variant only "
        f"{test.only_variant}, exact p = {test.p_value:.4g}",
        f"  rule: drop if the variant is {100 * SCREEN_DROP_THRESHOLD:.0f} points or more "
        f"below the baseline; the screen never adopts: {result.decision}",
        f"compare-variants: {verdict}"
        + ("" if dropped else " (not dropped; decide with the CV folds)"),
    ]
    return DecisionComparison(SCREEN, verdict, not dropped, screen=result, lines=lines)
