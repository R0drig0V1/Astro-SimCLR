"""``report-phase2``: the phase-2 results report (``docs/results-phase2.md``) from run directories.

Task 2.6 of ``docs/phases/02-stamp-classifier.md`` asks for a document with the tables and
the confusion matrices of phase 2. :func:`build_report` writes it as deterministic
markdown from the archived runs, so that the document is regenerated, never edited by
hand. Every number comes from the runs' ``metrics.json``, their predictions files and
their ``manifest.json`` (task A1, when present); every statistic comes from the protocol
code that makes the decisions (:func:`~astro_simclr.reference.compare_reference` and
:func:`~astro_simclr.protocol.compare.compare_decision`), so the report cannot disagree
with ``compare-reference`` or ``compare-variants``.

**Inputs.** Each input is a directory with one subdirectory per *arm* (a configuration),
which holds the arm's run directories at any depth (``DIR/<arm>/.../metrics.json``):

- ``replication``: the CD21 replication (``results/phase2/replication``: ``with-features``
  and ``no-features``). Per arm, a table per seed (best step, validation and test accuracy
  of the selected checkpoint, SN test recall, s/step, test ECE) with mean ± std (ddof =
  1); then the
  CD21 gate on all the runs together (``compare-reference --ref CD21``, with the M1 oracle
  report: ``--oracle-report``, default ``m1_oracle.json`` in the directory).
- ``screen`` (with its splits file): screen runs (``train-supervised --splits FILE
  --screen``). Every arm against the baseline arm (default ``rot4``) by the screen rule
  (drop only), with McNemar and the ``compare-variants`` output.
- ``cv`` (with its splits file): CV fold runs (``--fold K``, 5 folds x 2 seeds). The
  balanced accuracy per run and per fold (seed means), the pooled values, the paired
  bootstrap CI, **both** verdicts (superiority, and non-inferiority with ``margin``) and
  McNemar, every arm against the baseline. The pre-registered rule of an arm is
  superiority, or non-inferiority for the arms listed in ``non_inferiority`` (the
  efficiency rows of task 2.5); the ``compare-variants`` output printed is that rule's.
  The per-fold table ends with the fold-to-fold standard deviation (ddof = 1), and a
  calibration table gives each arm's top-label ECE per run (mean ± std). Then, per arm,
  the pooled per-class recall and the confusion matrix.

**Secondary metrics.** The ECE (15 bins, ``protocol.calibration``, from the predictions
files) and the fold-to-fold standard deviation are reported for plan row D3's
pre-registered hypothesis; neither is a decision input (``docs/evaluation-protocol.md``).

The per-class recall and the confusion matrix of the pooled CV use the same scores as the
pooled balanced accuracy: per stamp, the fraction of the fold's seeds that predict each
class (every pool stamp is held out by exactly one fold). So the recall of a class is the
mean over seeds of the per-run recall, the diagonal of the row-normalised matrix is the
recall, and the mean of the recalls is the pooled balanced accuracy of the comparison.

**Throughput and provenance.** A table per section and arm gives the s/step of the runs
(``metrics.json["timing"]``), the GPU, and whether the runs were measured alone: the
``concurrent_jobs`` of their ``manifest.json`` (``ASTRO_SIMCLR_CONCURRENT_JOBS``). A run
with more than one job on its device is flagged, since its s/step is not a throughput
claim; a run without a manifest (the archived phase-2 runs) is "not recorded". The
provenance table adds the git commits and the number of config hashes (one expected per
arm) of the manifest runs and, with ``registry`` (a registry file of ``astro-simclr runs
reindex``, ADR 0024), how many runs are indexed and whether the index still holds the
current ``metrics.json`` (by its SHA-256).

**Determinism.** The output depends only on the inputs: arms and runs are sorted, the
bootstrap is seeded (``seed``, ``n_boot``), the paths are printed as given, and nothing
records the time or the machine the report was built on. Test metrics appear only in the
replication section (archived and reported once; the protocol's rule).

Nothing here imports torch.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from astro_simclr.config import ConfigError
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.protocol.compare import (
    CVResult,
    DecisionComparison,
    ScreenResult,
    SplitRunPredictions,
    compare_decision,
    load_run_predictions,
    load_split_run,
    run_ece,
)
from astro_simclr.protocol.predictions import Predictions, seed_mean_correct
from astro_simclr.protocol.split_runs import FOLD, SCREEN
from astro_simclr.protocol.splits import read_splits
from astro_simclr.protocol.stats import (
    NON_INFERIORITY_MARGIN,
    McNemar,
    Rule,
    adoption_decision,
    balanced_accuracy,
    check_margin,
    non_inferiority_decision,
)
from astro_simclr.reference import compare_reference, load_run

__all__ = [
    "DEFAULT_BASELINE_ARM",
    "Arm",
    "ReportInputs",
    "RunFacts",
    "build_report",
    "find_arms",
    "pooled_class_scores",
    "read_run_facts",
]

#: The baseline arm of the screen and the CV: the replication's rotation-tied model.
DEFAULT_BASELINE_ARM: Final = "rot4"
#: The M1 oracle report looked for in the replication directory.
ORACLE_FILE: Final = "m1_oracle.json"
_METRICS: Final = "metrics.json"
_MANIFEST: Final = "manifest.json"
_DASH: Final = "—"


@dataclass(frozen=True, slots=True)
class ReportInputs:
    """What :func:`build_report` reads (module docstring); at least one section is needed."""

    replication: Path | None = None
    oracle_report: Path | None = None
    screen: Path | None = None
    screen_splits: Path | None = None
    cv: Path | None = None
    cv_splits: Path | None = None
    baseline_arm: str = DEFAULT_BASELINE_ARM
    #: The CV arms whose pre-registered rule is non-inferiority.
    non_inferiority: frozenset[str] = field(default_factory=frozenset)
    margin: float = NON_INFERIORITY_MARGIN
    registry: Path | None = None
    n_boot: int = 10_000
    seed: int = 0
    allow_mixed_machines: bool = False


@dataclass(frozen=True, slots=True)
class Arm:
    """One configuration of a section: its name (the directory's) and its run directories."""

    name: str
    runs: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RunFacts:
    """What the throughput and provenance tables read from one run directory.

    ``concurrent_jobs``, ``git_sha`` and ``config_sha256`` come from ``manifest.json``
    and are ``None`` for a run without one.
    """

    path: Path
    seconds_per_step: float | None
    gpu: str
    metrics_sha256: str
    has_manifest: bool = False
    concurrent_jobs: int | None = None
    git_sha: str | None = None
    config_sha256: str | None = None


# --- inputs -----------------------------------------------------------------------------


def find_arms(root: Path) -> list[Arm]:
    """The arms of ``root``: its subdirectories that hold run directories, sorted by name.

    A run directory is a directory with a ``metrics.json``; it may sit at any depth under
    its arm (``fold0/seed1``, ...). Subdirectories without runs and plain files (READMEs,
    gate outputs) are skipped.

    Raises:
        ConfigError: if ``root`` is not a directory, holds a run directory directly (the
            layout is ``DIR/<arm>/<run>``) or holds no arm.
    """
    if not root.is_dir():
        raise ConfigError(f"{root}: not a directory")
    arms = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if (child / _METRICS).is_file():
            msg = (
                f"{child} is a run directory: {root} must hold one directory per arm "
                f"({root}/<arm>/<run>/{_METRICS})"
            )
            raise ConfigError(msg)
        runs = sorted(path.parent for path in child.rglob(_METRICS))
        if runs:
            arms.append(Arm(child.name, tuple(runs)))
    if not arms:
        raise ConfigError(f"{root}: no arm directories with runs ({root}/<arm>/<run>/{_METRICS})")
    return arms


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if np.isfinite(value) else None


def _seconds_per_step(metrics: Mapping[str, Any]) -> float | None:
    timing = metrics.get("timing")
    return _finite(timing.get("seconds_per_step")) if isinstance(timing, Mapping) else None


def read_run_facts(run_dir: Path) -> RunFacts:
    """The s/step, GPU and file hash of ``run_dir``, and its manifest's provenance.

    Raises:
        ConfigError: if ``metrics.json`` cannot be read, or a ``manifest.json`` is present
            but invalid (a report never silently drops what a run recorded).
    """
    from astro_simclr.runs.manifest import ManifestError, read_manifest

    metrics = load_run(run_dir)
    seconds = _seconds_per_step(metrics)
    machine = metrics.get("machine")
    gpu = _DASH
    if isinstance(machine, dict):
        gpu = str(machine.get("gpu") or machine.get("device") or _DASH)
    digest = hashlib.sha256((run_dir / _METRICS).read_bytes()).hexdigest()
    if not (run_dir / _MANIFEST).is_file():
        return RunFacts(run_dir, seconds, gpu, digest)
    try:
        manifest = read_manifest(run_dir / _MANIFEST)
    except ManifestError as err:
        raise ConfigError(f"{run_dir}: unreadable {_MANIFEST} ({err})") from None
    return RunFacts(
        run_dir,
        seconds,
        gpu,
        digest,
        has_manifest=True,
        concurrent_jobs=manifest.concurrent_jobs,
        git_sha=manifest.git.sha,
        config_sha256=manifest.identity.config_sha256,
    )


def _check_inputs(inputs: ReportInputs) -> None:
    if inputs.replication is None and inputs.screen is None and inputs.cv is None:
        raise ConfigError("report-phase2 needs at least one of --replication, --screen, --cv")
    pairs = (
        ("--screen", inputs.screen, "--screen-splits", inputs.screen_splits),
        ("--cv", inputs.cv, "--cv-splits", inputs.cv_splits),
    )
    for flag, directory, splits_flag, splits in pairs:
        if directory is not None and splits is None:
            raise ConfigError(f"{flag} needs {splits_flag} FILE (the splits file of its runs)")
        if directory is None and splits is not None:
            raise ConfigError(f"{splits_flag} is only used with {flag} DIR")
    if inputs.oracle_report is not None and inputs.replication is None:
        raise ConfigError("--oracle-report is only used with --replication DIR")
    if inputs.non_inferiority and inputs.cv is None:
        raise ConfigError("--non-inferiority names CV arms: it needs --cv DIR")
    if inputs.baseline_arm in inputs.non_inferiority:
        raise ConfigError(f"the baseline arm {inputs.baseline_arm!r} has no rule of its own")
    try:
        check_margin(inputs.margin)
    except ValueError as err:
        raise ConfigError(str(err)) from None


def _split_baseline(arms: Sequence[Arm], baseline: str, where: Path) -> tuple[Arm, list[Arm]]:
    by_name = {arm.name: arm for arm in arms}
    if baseline not in by_name:
        found = ", ".join(sorted(by_name))
        raise ConfigError(f"{where}: no baseline arm {baseline!r} (arms: {found})")
    variants = [arm for arm in arms if arm.name != baseline]
    if not variants:
        raise ConfigError(f"{where}: only the baseline arm {baseline!r}; nothing to compare")
    return by_name[baseline], variants


# --- formatting -------------------------------------------------------------------------


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|")


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(map(_cell, header)) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(_cell, row)) + " |" for row in rows]
    return lines


def _code(lines: Sequence[str]) -> list[str]:
    return ["```text", *lines, "```"]


def _num(value: float | None, digits: int = 4) -> str:
    return _DASH if value is None else f"{value:.{digits}f}"


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _mean_std(values: Sequence[float], digits: int = 4) -> str:
    if not values:
        return _DASH
    if len(values) == 1:
        return f"{values[0]:.{digits}f} (n=1)"
    mean, std = statistics.mean(values), statistics.stdev(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _mcnemar(test: McNemar | None) -> str:
    if test is None:
        return _DASH
    return f"{test.only_baseline} / {test.only_variant}, p = {test.p_value:.4g}"


def _seed_key(seed: object) -> tuple[int, object]:
    """Integers in numeric order, anything else after them in text order."""
    if isinstance(seed, int) and not isinstance(seed, bool):
        return (0, seed)
    return (1, repr(seed))


# --- replication ------------------------------------------------------------------------


def _sn_recall(metrics: Mapping[str, Any]) -> float | None:
    evaluation = metrics.get("evaluation")
    test = evaluation.get("test") if isinstance(evaluation, Mapping) else None
    recall = test.get("recall_per_class") if isinstance(test, Mapping) else None
    return _finite(recall.get("SN")) if isinstance(recall, Mapping) else None


def _test_ece(run_dir: Path, metrics: Mapping[str, Any]) -> float | None:
    """The run's test ECE: ``metrics.json``'s, else from its test predictions file, else None.

    Runs from before plan row D3 have no ``ece`` entry; their predictions (task 2.3 on)
    give the same number.
    """
    evaluation = metrics.get("evaluation")
    test = evaluation.get("test") if isinstance(evaluation, Mapping) else None
    if isinstance(test, Mapping) and "ece" in test:
        return _finite(test["ece"])
    try:
        return run_ece(load_run_predictions(run_dir, "test").predictions)
    except (ConfigError, ValueError):  # no predictions file (or a malformed one)
        return None


def _replication_arm(arm: Arm) -> list[str]:
    runs = sorted(
        ((path, load_run(path)) for path in arm.runs), key=lambda item: _seed_key(item[1]["seed"])
    )
    rows: list[list[object]] = []
    columns: dict[str, list[float]] = {"val": [], "test": [], "sn": [], "sps": [], "ece": []}
    for path, run in runs:
        values = {
            "val": _finite(run.get("val_acc_best")),
            "test": _finite(run.get("test_acc_at_best")),
            "sn": _sn_recall(run),
            "sps": _seconds_per_step(run),
            "ece": _test_ece(path, run),
        }
        for key, value in values.items():
            if value is not None:
                columns[key].append(value)
        step = run.get("best_step")
        best_step = f"{step:,}" if isinstance(step, int) and not isinstance(step, bool) else _DASH
        rows.append(
            [
                run.get("seed"),
                best_step,
                _num(values["val"], 3),
                _num(values["test"], 3),
                _num(values["sn"], 3),
                _num(values["sps"], 4),
                _num(values["ece"], 4),
            ]
        )
    rows.append(
        [
            "**mean ± std (ddof=1)**",
            "",
            _mean_std(columns["val"]),
            _mean_std(columns["test"]),
            _mean_std(columns["sn"], 3),
            _mean_std(columns["sps"]),
            _mean_std(columns["ece"]),
        ]
    )
    header = [
        "Seed",
        "best_step",
        "val_acc_best",
        "test (at best)",
        "SN test recall",
        "s/step",
        "test ECE",
    ]
    return [f"### {arm.name}", "", *_table(header, rows), ""]


def _replication_section(inputs: ReportInputs, root: Path) -> list[str]:
    arms = find_arms(root)
    oracle = inputs.oracle_report
    if oracle is None and (root / ORACLE_FILE).is_file():
        oracle = root / ORACLE_FILE
    out = [
        "## Replication (CD21, task 2.4)",
        "",
        f"Runs in `{root}`, one table per arm. Test accuracy is of the checkpoint selected "
        "on validation (archived and reported, never a decision input).",
        "",
    ]
    for arm in arms:
        out += _replication_arm(arm)
    gate = compare_reference(
        "CD21", [run for arm in arms for run in arm.runs], oracle_report=oracle
    )
    oracle_text = f"`{oracle}`" if oracle is not None else "none (the faithful tier needs it)"
    out += [
        "### CD21 gate",
        "",
        f"`compare-reference --ref CD21` on every run above; M1 oracle report: {oracle_text}.",
        "",
        f"**Gate: {gate.lines[-1]}**",
        "",
        *_code(gate.lines),
        "",
    ]
    return out


# --- screen -----------------------------------------------------------------------------


def _decide(
    inputs: ReportInputs, base: Arm, arm: Arm, splits: Path, rule: Rule | None
) -> DecisionComparison:
    return compare_decision(
        list(base.runs),
        list(arm.runs),
        splits,
        rule=rule,
        margin=inputs.margin,
        n_boot=inputs.n_boot,
        seed=inputs.seed,
        allow_mixed_machines=inputs.allow_mixed_machines,
    )


_MODES: Final = {FOLD: "CV fold runs (--fold K)", SCREEN: "screen runs (--screen)"}


def _screen_result(decision: DecisionComparison, where: Path) -> ScreenResult:
    if decision.mode != SCREEN or decision.screen is None:
        raise ConfigError(f"{where}: expected {_MODES[SCREEN]}, got {_MODES[decision.mode]}")
    return decision.screen


def _cv_result(decision: DecisionComparison, where: Path) -> CVResult:
    if decision.mode != FOLD or decision.cv is None:
        raise ConfigError(f"{where}: expected {_MODES[FOLD]}, got {_MODES[decision.mode]}")
    return decision.cv


def _screen_section(inputs: ReportInputs, root: Path, splits: Path) -> list[str]:
    base, variants = _split_baseline(find_arms(root), inputs.baseline_arm, root)
    rows = []
    details: list[str] = []
    for arm in variants:
        decision = _decide(inputs, base, arm, splits, None)
        result = _screen_result(decision, root)
        rows.append(
            [
                arm.name,
                len(arm.runs),
                _num(result.baseline),
                _num(result.variant),
                _signed(result.variant - result.baseline),
                _mcnemar(decision.mcnemar),
                decision.verdict,
            ]
        )
        details += [f"#### {arm.name} vs {base.name}", "", *_code(decision.lines), ""]
    header = [
        "Arm",
        "Runs",
        f"{base.name} accuracy",
        "Arm accuracy",
        "Arm - baseline",
        "McNemar (baseline only / arm only, p)",
        "Verdict",
    ]
    return [
        "## Screen (250/250 split of Validation, task 2.5)",
        "",
        f"Runs in `{root}`, splits file `{splits}`; every arm against `{base.name}` "
        f"({len(base.runs)} runs). Accuracy is the seed mean on the decision half. The "
        "screen only drops (an arm 4 points or more below the baseline); KEEP means not "
        "dropped, never adopted.",
        "",
        *_table(header, rows),
        "",
        "### compare-variants output",
        "",
        *details,
    ]


# --- cross-validation -------------------------------------------------------------------


def _by_cell(runs: Sequence[SplitRunPredictions]) -> dict[int, list[Predictions]]:
    """Each fold's predictions, ordered by seed."""
    grouped: dict[int, list[tuple[tuple[int, object], Predictions]]] = {}
    for run in runs:
        if run.fold is None:
            raise ConfigError(f"{run.path}: not a CV fold run")
        grouped.setdefault(run.fold, []).append((_seed_key(run.seed), run.predictions))
    return {
        fold: [pred for _, pred in sorted(items, key=lambda item: item[0])]
        for fold, items in sorted(grouped.items())
    }


def pooled_class_scores(
    folds: Mapping[int, Sequence[Predictions]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The pooled per-class recall and confusion counts of one arm's CV runs.

    ``folds[k]`` are the runs (seeds) of fold ``k``. Every stamp counts once, as the
    fraction of its fold's seeds that predict each class, so the counts of a row sum to
    the number of stamps of that class, and the recall of a class (the row-normalised
    diagonal) is the class mean of the seed-mean correctness that
    :func:`~astro_simclr.protocol.stats.balanced_accuracy` averages (module docstring).

    Returns:
        ``(recall, counts)``: ``recall`` (C,) and ``counts`` (C, C), rows the true class.

    Raises:
        ValueError: if ``folds`` is empty, a fold's runs are not paired, or a class has
            no stamp.
    """
    if not folds:
        raise ValueError("no folds")
    n_classes = next(iter(folds.values()))[0].n_classes
    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    scores, labels = [], []
    for runs in folds.values():
        scores.append(seed_mean_correct(runs))
        labels.append(runs[0].labels)
        for run in runs:
            np.add.at(counts, (run.labels, run.predicted), 1.0 / len(runs))
    score, label = np.concatenate(scores), np.concatenate(labels)
    present = np.bincount(label, minlength=n_classes)
    if np.any(present == 0):
        names = _class_names(n_classes)
        missing = [names[c] for c in np.flatnonzero(present == 0)]
        raise ValueError(f"no stamp of class(es) {missing} in the pooled folds")
    recall = np.array([score[label == c].mean() for c in range(n_classes)], dtype=np.float64)
    return recall, counts


def _class_names(n_classes: int) -> list[str]:
    return [LABEL_NAMES[c] if c < len(LABEL_NAMES) else f"class {c}" for c in range(n_classes)]


@dataclass(frozen=True, slots=True)
class _CVArm:
    """One CV arm's predictions per fold (ordered by seed) and its seeds per fold."""

    name: str
    folds: dict[int, list[Predictions]]
    seeds: dict[int, list[object]]


@dataclass(frozen=True, slots=True)
class _CVVariant:
    """A CV arm compared with the baseline under its pre-registered rule."""

    arm: _CVArm
    rule: Rule
    decision: DecisionComparison
    cv: CVResult


def _load_cv_arm(arm: Arm) -> _CVArm:
    runs = [load_split_run(path) for path in arm.runs]
    folds = _by_cell(runs)
    seeds: dict[int, list[object]] = {}
    for run in runs:
        if run.fold is not None:  # always (_by_cell refuses a run without a fold)
            seeds.setdefault(run.fold, []).append(run.seed)
    return _CVArm(arm.name, folds, {fold: sorted(s, key=_seed_key) for fold, s in seeds.items()})


def _cv_arms(inputs: ReportInputs, root: Path, splits: Path) -> tuple[_CVArm, list[_CVVariant]]:
    base, variants = _split_baseline(find_arms(root), inputs.baseline_arm, root)
    unknown = sorted(inputs.non_inferiority - {arm.name for arm in variants})
    if unknown:
        raise ConfigError(f"{root}: --non-inferiority names no arm here: {unknown}")
    base_arm = _load_cv_arm(base)
    compared = []
    for arm in variants:
        rule: Rule = "non-inferiority" if arm.name in inputs.non_inferiority else "superiority"
        decision = _decide(inputs, base, arm, splits, rule)
        cv = _cv_result(decision, root)
        compared.append(_CVVariant(_load_cv_arm(arm), rule, decision, cv))
    return base_arm, compared


def _per_run_table(arms: Sequence[_CVArm]) -> list[str]:
    cells = sorted(
        {(fold, seed) for arm in arms for fold, seeds in arm.seeds.items() for seed in seeds},
        key=lambda cell: (cell[0], _seed_key(cell[1])),
    )
    rows: list[list[object]] = []
    for fold, seed in cells:
        row: list[object] = [fold, seed]
        for arm in arms:
            seeds = arm.seeds.get(fold, [])
            pred = arm.folds[fold][seeds.index(seed)] if seed in seeds else None
            row.append(
                _DASH if pred is None else _num(balanced_accuracy(pred.correct, pred.labels))
            )
        rows.append(row)
    return _table(["Fold", "Seed", *(arm.name for arm in arms)], rows)


def _per_fold_table(base: _CVArm, variants: Sequence[_CVVariant], splits: Path) -> list[str]:
    first = variants[0].cv
    # The fold order of the CV results is the splits file's (cv_comparison's loop).
    fold_ids = [fold.fold for fold in read_splits(splits).folds]
    rows: list[list[object]] = []
    for position, fold in enumerate(fold_ids):
        values = [first.baseline_folds[position], *(v.cv.variant_folds[position] for v in variants)]
        rows.append([fold, *map(_num, values)])
    pooled = [first.baseline_pooled, *(v.cv.variant_pooled for v in variants)]
    rows.append(["**pooled**", *map(_num, pooled)])
    if len(fold_ids) > 1:
        spread = [first.baseline_folds, *(v.cv.variant_folds for v in variants)]
        rows.append(["**std over folds (ddof=1)**", *(_num(statistics.stdev(f)) for f in spread)])
    return _table(["Fold", base.name, *(v.arm.name for v in variants)], rows)


def _calibration_table(arms: Sequence[_CVArm]) -> list[str]:
    """Each arm's top-label ECE per run (mean ± std over its runs): a secondary metric."""
    rows = []
    for arm in arms:
        values = [run_ece(pred) for fold in sorted(arm.folds) for pred in arm.folds[fold]]
        rows.append([arm.name, len(values), _mean_std(values)])
    return _table(["Arm", "Runs", "ECE per run (mean ± std, ddof=1)"], rows)


def _decision_table(
    inputs: ReportInputs, base: _CVArm, variants: Sequence[_CVVariant]
) -> list[str]:
    rows = []
    for variant in variants:
        cv = variant.cv
        rows.append(
            [
                variant.arm.name,
                _num(cv.baseline_pooled),
                _num(cv.variant_pooled),
                _signed(cv.interval.observed),
                f"[{_signed(cv.interval.low)}, {_signed(cv.interval.high)}]",
                _mcnemar(cv.mcnemar),
                adoption_decision(cv.interval),
                non_inferiority_decision(cv.interval, margin=inputs.margin),
                f"{variant.rule}: **{variant.decision.verdict}**",
            ]
        )
    header = [
        "Arm",
        f"{base.name} pooled",
        "Arm pooled",
        "Arm - baseline",
        f"{100 * variants[0].cv.interval.confidence:.0f} % CI",
        "McNemar (baseline only / arm only, p)",
        "Superiority",
        f"Non-inferiority ({100 * inputs.margin:.2f} pt)",
        "Pre-registered rule",
    ]
    return _table(header, rows)


def _cv_section(inputs: ReportInputs, root: Path, splits: Path) -> list[str]:
    base, variants = _cv_arms(inputs, root, splits)
    out = [
        "## Cross-validation (5-fold CV on Train + Validation, task 2.5)",
        "",
        f"Runs in `{root}`, splits file `{splits}`; every arm against `{base.name}`. The "
        "metric is balanced accuracy on each fold's evaluation part; a fold's value is the "
        "mean over its seeds, and the pooled value is over every pool stamp once. The "
        "pre-registered rule of an arm is superiority (adopt if the 95 % CI of "
        "arm - baseline lies above 0) or, for the efficiency rows, non-inferiority (adopt "
        f"if its lower bound is above -{inputs.margin:.4f}, {100 * inputs.margin:.2f} "
        "points); both verdicts are shown.",
        "",
        "### Balanced accuracy per run",
        "",
        *_per_run_table([base, *(variant.arm for variant in variants)]),
        "",
        "### Balanced accuracy per fold (seed means)",
        "",
        *_per_fold_table(base, variants, splits),
        "",
        "### Decisions",
        "",
        *_decision_table(inputs, base, variants),
        "",
        "### Calibration (secondary)",
        "",
        "Top-label expected calibration error (15 equal-width bins) of each run on its "
        "fold's evaluation part; it decides nothing.",
        "",
        *_calibration_table([base, *(variant.arm for variant in variants)]),
        "",
        f"Paired bootstrap over pool stamps: {variants[0].cv.interval.n_boot} resamples, "
        f"seed {inputs.seed}, stratified by class. McNemar compares the seed ensembles over "
        "all folds.",
        "",
        "### compare-variants output",
        "",
    ]
    for variant in variants:
        title = f"#### {variant.arm.name} vs {base.name} ({variant.rule})"
        out += [title, "", *_code(variant.decision.lines), ""]
    out += _class_tables([base, *(variant.arm for variant in variants)])
    return out


def _class_tables(cv_arms: Sequence[_CVArm]) -> list[str]:
    scores = {}
    for arm in cv_arms:
        try:
            scores[arm.name] = pooled_class_scores(arm.folds)
        except ValueError as err:
            raise ConfigError(f"arm {arm.name}: {err}") from None
    n_classes = next(iter(scores.values()))[1].shape[0]
    classes = _class_names(n_classes)
    rows = [
        [name, *(_num(value, 3) for value in recall), _num(float(recall.mean()))]
        for name, (recall, _) in scores.items()
    ]
    out = [
        "### Per-class recall (pooled CV)",
        "",
        "Seed means over every pool stamp; the last column, their mean, is the pooled "
        "balanced accuracy.",
        "",
        *_table(["Arm", *classes, "Balanced accuracy"], rows),
        "",
        "### Confusion matrices (pooled CV)",
        "",
        "Percent of each true class (rows) predicted as each class (columns), with every "
        "stamp counted once as the fraction of its fold's seeds; the diagonal is the recall.",
        "",
    ]
    for name, (_, counts) in scores.items():
        percent = 100 * counts / counts.sum(axis=1, keepdims=True)
        matrix = [
            [cls, *(f"{v:.1f}" for v in row)] for cls, row in zip(classes, percent, strict=True)
        ]
        stamps = round(float(counts.sum()))
        out += [
            f"#### {name} ({stamps} stamps)",
            "",
            *_table(["True \\ predicted", *classes], matrix),
            "",
        ]
    return out


# --- throughput and provenance ----------------------------------------------------------


def _concurrency(facts: Sequence[RunFacts]) -> str:
    recorded = [fact.concurrent_jobs for fact in facts if fact.concurrent_jobs is not None]
    missing = len(facts) - len(recorded)
    parts = []
    shared = [jobs for jobs in recorded if jobs > 1]
    if shared:
        parts.append(
            f"**measured under concurrency** (up to {max(shared)} jobs; {len(shared)} of "
            f"{len(facts)} runs)"
        )
    elif recorded:
        parts.append("alone")
    if missing:
        parts.append(f"not recorded for {missing} of {len(facts)} runs")
    return "; ".join(parts)


def _registry_records(path: Path) -> dict[Path, str | None]:
    """Indexed run directory (resolved) -> the ``metrics.json`` SHA-256 the index holds."""
    from astro_simclr.registry.home import RegistryError
    from astro_simclr.registry.store import load_records, open_registry

    if not path.is_file():
        raise ConfigError(f"{path}: no registry file (build it with astro-simclr runs reindex)")
    try:
        conn = open_registry(path)
    except RegistryError as err:
        raise ConfigError(str(err)) from None
    try:
        return {record.run_dir.resolve(): record.metrics_sha256 for record in load_records(conn)}
    finally:
        conn.close()


def _registry_cell(facts: Sequence[RunFacts], index: Mapping[Path, str | None]) -> str:
    indexed = [fact for fact in facts if fact.path.resolve() in index]
    stale = [fact for fact in indexed if index[fact.path.resolve()] != fact.metrics_sha256]
    text = f"{len(indexed)} of {len(facts)} indexed"
    if stale:
        text += f", **{len(stale)} stale** (metrics.json changed since reindex)"
    return text


def _short_shas(values: Iterable[str | None]) -> str:
    shas = sorted({value[:7] for value in values if value})
    return ", ".join(shas) if shas else _DASH


def _resources_section(
    sections: Sequence[tuple[str, Sequence[Arm]]], registry: Path | None
) -> list[str]:
    index = None if registry is None else _registry_records(registry)
    throughput, provenance = [], []
    for section, arms in sections:
        for arm in arms:
            facts = [read_run_facts(path) for path in arm.runs]
            seconds = [fact.seconds_per_step for fact in facts if fact.seconds_per_step is not None]
            gpus = ", ".join(sorted({fact.gpu for fact in facts}))
            throughput.append(
                [section, arm.name, len(facts), gpus, _mean_std(seconds), _concurrency(facts)]
            )
            with_manifest = [fact for fact in facts if fact.has_manifest]
            hashes = {fact.config_sha256 for fact in with_manifest}
            config = _DASH if not with_manifest else str(len(hashes))
            if len(hashes) > 1:
                config += " (**mixed configs**)"
            row: list[object] = [
                section,
                arm.name,
                f"{len(with_manifest)} of {len(facts)}",
                _short_shas(fact.git_sha for fact in with_manifest),
                config,
            ]
            if index is not None:
                row.append(_registry_cell(facts, index))
            provenance.append(row)
    header = ["Section", "Arm", "Manifests", "git", "Config hashes"]
    if index is not None:
        header.append("Registry")
    registry_text = (
        f" Registry: `{registry}`." if registry is not None else " No registry was given."
    )
    return [
        "## Throughput",
        "",
        'Seconds per optimizer step of the fit (`metrics.json["timing"]`), mean ± std '
        "over the arm's runs. A run measured with other jobs on its device (the manifest's "
        "`concurrent_jobs` > 1) is not a throughput claim; the archived runs without a "
        "manifest did not record it.",
        "",
        *_table(["Section", "Arm", "Runs", "GPU", "s/step", "Concurrency"], throughput),
        "",
        "## Provenance",
        "",
        "From the runs' `manifest.json` (task A1): the git commits and the number of config "
        "hashes of each arm (one expected)." + registry_text,
        "",
        *_table(header, provenance),
        "",
    ]


# --- the report -------------------------------------------------------------------------


def _header(inputs: ReportInputs) -> list[str]:
    args = []
    for flag, value in (
        ("--replication", inputs.replication),
        ("--oracle-report", inputs.oracle_report),
        ("--screen", inputs.screen),
        ("--screen-splits", inputs.screen_splits),
        ("--cv", inputs.cv),
        ("--cv-splits", inputs.cv_splits),
        ("--registry", inputs.registry),
    ):
        if value is not None:
            args.append(f"{flag} {value}")
    if inputs.screen is not None or inputs.cv is not None:
        args.append(f"--baseline-arm {inputs.baseline_arm}")
    if inputs.non_inferiority:
        args.append("--non-inferiority " + " ".join(sorted(inputs.non_inferiority)))
    if inputs.cv is not None:
        args.append(f"--margin {inputs.margin}")
    args += [f"--n-boot {inputs.n_boot}", f"--seed {inputs.seed}"]
    if inputs.allow_mixed_machines:
        args.append("--allow-mixed-machines")
    return [
        "<!-- Generated by `astro-simclr report-phase2`; do not edit by hand: regenerate it "
        "from the runs. -->",
        "# Phase-2 results",
        "",
        "The results of phase 2 (`docs/phases/02-stamp-classifier.md`), judged by the rules of "
        "`docs/evaluation-protocol.md`. Generated with:",
        "",
        "```bash",
        "astro-simclr report-phase2 \\",
        *(f"    {arg} \\" for arg in args[:-1]),
        f"    {args[-1]}",
        "```",
        "",
    ]


def build_report(inputs: ReportInputs) -> str:
    """The phase-2 results report of ``inputs`` as markdown (module docstring).

    Raises:
        ConfigError: on inconsistent inputs (a section without its splits file, a missing
            baseline arm, an unknown ``non_inferiority`` arm, a bad margin), unreadable
            runs, runs that ``compare-reference`` or ``compare-variants --splits`` refuses
            (another splits file, mixed machines or data, unpaired folds or seeds), or an
            unusable registry file.
        DataError: if a predictions file is unreadable.
        SplitsError: if a splits file cannot be read.
    """
    _check_inputs(inputs)
    out = _header(inputs)
    resources: list[tuple[str, Sequence[Arm]]] = []
    if inputs.replication is not None:
        out += _replication_section(inputs, inputs.replication)
        resources.append(("replication", find_arms(inputs.replication)))
    # _check_inputs: a section's directory and its splits file come together.
    if inputs.screen is not None and inputs.screen_splits is not None:
        out += _screen_section(inputs, inputs.screen, inputs.screen_splits)
        resources.append(("screen", find_arms(inputs.screen)))
    if inputs.cv is not None and inputs.cv_splits is not None:
        out += _cv_section(inputs, inputs.cv, inputs.cv_splits)
        resources.append(("CV", find_arms(inputs.cv)))
    out += _resources_section(resources, inputs.registry)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"
