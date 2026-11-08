"""``compare-reference``: judge the runs of a reference experiment (R1-R4) against its gate.

The reference runs and their gates are defined in ``docs/phases/01-runnable-package.md``
("Reference runs" and "Definition of equivalent"). The metric is the best validation
accuracy over epochs (micro), ``val_acc_best`` of each run's ``metrics.json``; s is the
sample standard deviation (ddof=1), as in the legacy tables.

- **R1, R2 (pass rule)** against the legacy mean ``m_old`` ± ``s_old`` of ``n_old`` repeats:
  - one run (minimal set): ``|x - m_old| <= max(2 s_old sqrt(1 + 1/n_old), 0.01)``;
  - n >= 2 runs (full set: 5): ``|mean - m_old| <= max(2 sqrt(s_old**2/n_old + s_new**2/n),
    0.01)``.

  The plan writes ``/5`` for both terms; this is the same formula with the actual repeat
  counts. ``n_old`` is 5 for R1 and 3 for R2 (``results/hyperparameter_tuning_ce_resnet.csv``
  has 3 trials of that configuration), so R2 uses 3.
- **R3 (baseline)**: every run must have finite metrics, and every linear-evaluation run a
  validation accuracy above the majority-class baseline of its own Validation split (the
  class counts copied from ``meta.json`` into ``metrics.json``). Pretrain runs may be given
  too; their selected loss must be finite.
- **R4 (report)**: printed only.

The result is ``PASS``, ``FAIL`` or ``REPORT``; :func:`exit_code` maps it to 0, 1, 0.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from astro_simclr.config import ConfigError

__all__ = [
    "REFERENCES",
    "Comparison",
    "Reference",
    "compare_reference",
    "exit_code",
    "load_run",
    "pass_rule",
]

Gate = Literal["pass_rule", "baseline", "report"]
Result = Literal["PASS", "FAIL", "REPORT"]

#: Smallest tolerance of the pass rule (one accuracy point).
_MIN_TOLERANCE: Final = 0.01


@dataclass(frozen=True, slots=True)
class Reference:
    """One reference run: what it is, which runs it accepts and its gate."""

    name: str
    description: str
    #: ``metrics.json["command"]`` values accepted.
    commands: tuple[str, ...]
    gate: Gate
    #: ``config.supervised_model`` the runs must have (train-supervised references).
    supervised_model: str | None = None
    #: Legacy mean, sample std (ddof=1) and number of repeats (pass-rule references).
    mean: float | None = None
    std: float | None = None
    n: int | None = None
    source: str = ""
    #: Context printed with the result (not part of the gate).
    note: str = ""


#: The reference runs of phase 1 (plan: "Reference runs").
REFERENCES: Final[dict[str, Reference]] = {
    "R1": Reference(
        "R1",
        "CE stamp classifier, row-10 config (AdamW), tuning protocol",
        ("train-supervised",),
        "pass_rule",
        supervised_model="stamp_classifier",
        mean=0.9496,
        std=0.002191,
        n=5,
        source="results/summary_tuning_ce.csv, row 10 (5 trials)",
        note="external context: Carrasco-Davis et al. 2021 report test accuracy "
        "0.941 ± 0.004 (validation 0.950 ± 0.005)",
    ),
    "R2": Reference(
        "R2",
        "CE ResNet-18, hparams_best_ce_resnet.yaml",
        ("train-supervised",),
        "pass_rule",
        supervised_model="resnet18",
        mean=0.902667,
        std=0.009866,
        n=3,
        source="results/summary_tuning_ce_resnet.csv, row 6 (3 trials)",
    ),
    "R3": Reference(
        "R3",
        "SimCLR pretrain (31b10c9 settings), then linear evaluation at 1, 10 and 100 %",
        ("pretrain", "linear-eval"),
        "baseline",
        note="no legacy number: this result becomes the first recorded reference",
    ),
    "R4": Reference(
        "R4",
        "CE stamp classifier, R1 config without features, 1 % labels",
        ("train-supervised",),
        "report",
        supervised_model="stamp_classifier",
        note="the comparison point for R3",
    ),
}


@dataclass(slots=True)
class Comparison:
    """The outcome of :func:`compare_reference`: the result and the lines to print."""

    result: Result
    lines: list[str] = field(default_factory=list)


def load_run(run_dir: Path) -> dict[str, Any]:
    """Read ``run_dir/metrics.json`` (or ``run_dir`` itself, if it is the JSON file).

    Raises:
        ConfigError: if the file is missing or not a JSON object.
    """
    path = run_dir if run_dir.is_file() else run_dir / "metrics.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ConfigError(f"{run_dir}: cannot read the run's metrics ({err})") from None
    except json.JSONDecodeError as err:
        raise ConfigError(f"{path}: invalid JSON ({err})") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: not a metrics object")
    data["_path"] = str(run_dir)
    return data


def _as_float(value: object) -> float | None:
    """``value`` as a float if it is a finite number (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _finite(value: object) -> bool:
    return _as_float(value) is not None


def _describe(values: Sequence[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f} (n=1)"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f} (n={len(values)})"


def pass_rule(
    values: Sequence[float], mean_old: float, std_old: float, n_old: int
) -> tuple[bool, float, float]:
    """The plan's pass rule; returns ``(passed, |difference|, tolerance)``.

    One value uses the minimal-set rule, two or more the full-set rule (module docstring).

    Raises:
        ValueError: if ``values`` is empty or ``n_old < 1``.
    """
    if not values or n_old < 1:
        raise ValueError("pass_rule needs at least one value and n_old >= 1")
    if len(values) == 1:
        diff = abs(values[0] - mean_old)
        tolerance = max(2 * std_old * math.sqrt(1 + 1 / n_old), _MIN_TOLERANCE)
    else:
        std_new = statistics.stdev(values)
        diff = abs(statistics.mean(values) - mean_old)
        tolerance = max(
            2 * math.sqrt(std_old**2 / n_old + std_new**2 / len(values)), _MIN_TOLERANCE
        )
    return diff <= tolerance, diff, tolerance


def _check_runs(reference: Reference, runs: Sequence[dict[str, Any]]) -> None:
    if not runs:
        raise ConfigError("compare-reference needs at least one run directory")
    for run in runs:
        command = run.get("command")
        if command not in reference.commands:
            raise ConfigError(
                f"{run['_path']}: a {command!r} run cannot be compared with {reference.name} "
                f"(expects {', '.join(reference.commands)})"
            )
        model = run.get("config", {}).get("supervised_model")
        if (
            command == "train-supervised"
            and reference.supervised_model is not None
            and model != reference.supervised_model
        ):
            raise ConfigError(
                f"{run['_path']}: supervised_model {model!r}, but {reference.name} is "
                f"{reference.supervised_model!r}"
            )
    seen: dict[tuple[object, ...], str] = {}
    for run in runs:
        key = (run.get("command"), run.get("config", {}).get("dataset"), run.get("seed"))
        if key in seen:
            raise ConfigError(
                f"{run['_path']} and {seen[key]} are the same run (command, dataset, seed "
                f"{key[2]}); the seeds of a set must differ"
            )
        seen[key] = run["_path"]


def _accuracies(runs: Sequence[dict[str, Any]], key: str) -> list[float] | None:
    values: list[float] = []
    for run in runs:
        value = _as_float(run.get(key))
        if value is None:
            return None
        values.append(value)
    return values


def _gate_pass_rule(reference: Reference, runs: Sequence[dict[str, Any]], out: list[str]) -> bool:
    if reference.mean is None or reference.std is None or reference.n is None:
        raise ConfigError(f"{reference.name} has no legacy mean, std and n")  # pragma: no cover
    values = _accuracies(runs, "val_acc_best")
    if values is None:
        out.append("  val_acc_best is missing or not finite in some run")
        return False
    rule = "minimal-set rule" if len(values) == 1 else "full-set rule"
    passed, diff, tolerance = pass_rule(values, reference.mean, reference.std, reference.n)
    out.append(f"  val_acc_best: {_describe(values)}")
    out.append(
        f"  legacy:       {reference.mean:.4f} ± {reference.std:.4f} (n={reference.n}; "
        f"{reference.source})"
    )
    out.append(
        f"  |difference| = {diff:.4f} {'<=' if passed else '>'} tolerance {tolerance:.4f} ({rule})"
    )
    return passed


def _baseline(run: dict[str, Any]) -> float | None:
    counts = run.get("data", {}).get("class_counts", {}).get("Validation")
    if not counts or not all(isinstance(c, int) for c in counts) or sum(counts) == 0:
        return None
    return float(max(counts)) / float(sum(counts))


def _gate_baseline(runs: Sequence[dict[str, Any]], out: list[str]) -> bool:
    passed = True
    evaluated = 0
    for run in runs:
        dataset = run.get("config", {}).get("dataset")
        if run.get("command") == "pretrain":
            ok = _finite(run.get("monitor_best"))
            out.append(
                f"  pretrain {dataset}: {run.get('monitor')} = {run.get('monitor_best')}"
                f" -> {'finite' if ok else 'NOT FINITE'}"
            )
            passed &= ok
            continue
        evaluated += 1
        value = _as_float(run.get("val_acc_best"))
        test = _as_float(run.get("test_acc_at_best"))
        baseline = _baseline(run)
        ok = value is not None and test is not None and baseline is not None and value > baseline
        out.append(
            f"  linear-eval {dataset} (seed {run.get('seed')}): val_acc_best {value}, "
            f"test {test}, majority baseline {baseline} -> {'above' if ok else 'NOT above'}"
        )
        passed &= ok
    if evaluated == 0:
        out.append("  no linear-eval run given")
        return False
    return passed


def compare_reference(name: str, run_dirs: Sequence[Path]) -> Comparison:
    """Load the runs in ``run_dirs`` and apply the gate of reference ``name``.

    Raises:
        ConfigError: for an unknown reference, an unreadable run, a run of the wrong kind
            (command or model) or two runs with the same seed.
    """
    if name not in REFERENCES:
        raise ConfigError(f"unknown reference {name!r}; valid: {', '.join(REFERENCES)}")
    reference = REFERENCES[name]
    runs = [load_run(path) for path in run_dirs]
    _check_runs(reference, runs)
    lines = [f"{reference.name}: {reference.description}"]
    result: Result
    if reference.gate == "pass_rule":
        result = "PASS" if _gate_pass_rule(reference, runs, lines) else "FAIL"
    elif reference.gate == "baseline":
        result = "PASS" if _gate_baseline(runs, lines) else "FAIL"
    else:
        values = _accuracies(runs, "val_acc_best")
        lines.append(f"  val_acc_best: {'not finite' if values is None else _describe(values)}")
        result = "REPORT"
    tests = _accuracies(runs, "test_acc_at_best")
    if tests:
        lines.append(f"  test_acc_at_best: {_describe(tests)}")
    if reference.note:
        lines.append(f"  note: {reference.note}")
    lines.append(f"{reference.name}: {result}")
    return Comparison(result, lines)


def exit_code(result: Result) -> int:
    """0 for PASS and REPORT, 1 for FAIL."""
    return 1 if result == "FAIL" else 0
