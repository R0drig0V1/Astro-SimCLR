"""``compare-reference``: judge the runs of a reference experiment (R1-R4, CD21) against its gate.

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
- **R3 (baseline)**: the pretrain run and one linear-evaluation run per label fraction
  (1, 10 and 100 %: ``hparams.LABEL_FRACTION_DATASETS``) are all required, and every
  linear-evaluation run must have been trained on (one of) the given pretrain run(s). The
  pretrain loss must be finite, and every linear-evaluation run needs finite accuracies and
  a validation accuracy above the majority-class baseline of its own Validation split (the
  class counts copied from ``meta.json`` into ``metrics.json``).
- **R4 (report)**: printed only.
- **CD21 (two-tier gate, phase 2, task 2.3)**: the paper replication against Carrasco-Davis
  et al. 2021 (AJ 162, 231; arXiv:2008.03309), on the **test** accuracy of the checkpoint
  selected on validation (``test_acc_at_best``), over the with-features runs. The runs
  must use ``supervised_model: paper_stamp_classifier`` (task 2.2a), and a run whose
  config differs from the ``carrasco2021`` preset (``presets.PRESETS``, task 2.2c: another
  schedule, dataset or a diagnostic flag; ``with_features`` aside) gets a WARN line:
  - *faithful*: ``|mean - 0.941| <= 0.005`` **and** the M1 oracle passed
    (``--oracle-report FILE``: the JSON report of task 2.2b, whose top-level ``passed``
    must be ``true``);
  - *acceptable*: ``|mean - 0.941| <= 0.01``;
  - both also need the mean SN test recall within ±0.04 of 0.87 and the mean
    ``val_acc_best`` within ±0.01 of 0.950.

  The paper's sigma = 0.004 is ``np.std`` (ddof=0, ALeRCE's tables), about 0.0045 at ddof=1;
  both are printed for the runs. With sigma ≈ 0.0045 on both 5-run means, a perfect
  replication misses the faithful tier about 8 % of the time, so the plan allows one
  confirmatory 5-seed rerun (judged as its own set). The gate is defined on exactly 5
  seeds: another count gives REPORT (provisional), never PASS. The without-features runs
  (``with_features: false``, or ``feature_set: none``; ADR 0016) are compared with
  0.883 ± 0.006 and only reported (advisory). A ``feature_set: position`` run counts as
  with features, and its WARN line names the deviation.

Each run's resolved config is also compared with the distinguishing settings of its
reference config (``Reference.expected``: dataset, features, model, optimizer, protocol);
a difference is printed as a WARN line, not a failure (a deliberate variation may be
compared on purpose), so an R4 run given as R1 does not pass silently. A key missing from a
run's config (a run written before that setting existed) reads as the setting's default,
the behaviour the run had: the 2.5 flags (ADR 0005, ADR 0006) default to the faithful model,
so the replication runs of task 2.4 no longer warn about them.

Every gate also prints a WARN line when the runs come from different machines or builds
(``metrics.json["machine"]``, :func:`machine_signature`), or with different effective
float32 precisions (ADR 0010): the runs of one comparison use the same machine, build and
precision (plan, phase 2, "Machines").

The result is ``PASS`` (for CD21 with its tier), ``FAIL`` or ``REPORT``; :func:`exit_code`
maps it to 0, 1, 0.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from astro_simclr.config import ConfigError, Settings, settings_to_plain
from astro_simclr.hparams import LABEL_FRACTION_DATASETS
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS

__all__ = [
    "REFERENCES",
    "Comparison",
    "Reference",
    "compare_reference",
    "exit_code",
    "load_run",
    "machine_signature",
    "pass_rule",
    "read_oracle_report",
]

Gate = Literal["pass_rule", "baseline", "report", "two_tier"]
Tier = Literal["faithful", "acceptable"]
Result = Literal["PASS", "FAIL", "REPORT"]

#: Smallest tolerance of the pass rule (one accuracy point).
_MIN_TOLERANCE: Final = 0.01

#: CD21 (Carrasco-Davis et al. 2021), with features: test accuracy, mean and std (np.std,
#: ddof=0) of 5 runs on the balanced 100-per-class test set.
CD21_TEST_MEAN: Final = 0.941
CD21_TEST_STD: Final = 0.004
#: The tolerances of the two tiers on |mean - CD21_TEST_MEAN|.
CD21_FAITHFUL_TOLERANCE: Final = 0.005
CD21_ACCEPTABLE_TOLERANCE: Final = 0.01
#: SN recall on test (the paper's confusion matrix) and its tolerance.
CD21_SN_RECALL: Final = 0.87
CD21_SN_TOLERANCE: Final = 0.04
#: Mean validation accuracy of the paper (0.950 ± 0.005) and its tolerance.
CD21_VAL_MEAN: Final = 0.950
CD21_VAL_TOLERANCE: Final = 0.01
#: Without features (advisory): test accuracy mean and std (ddof=0).
CD21_NO_FEATURES_MEAN: Final = 0.883
CD21_NO_FEATURES_STD: Final = 0.006
#: The gate is defined on this many seeds.
CD21_SEEDS: Final = 5
#: Rounding of differences compared with a tolerance (0.946 - 0.941 is 0.0050000000000001).
_DIGITS: Final = 12


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
    #: Per command, settings of the reference config a run should have (WARN otherwise).
    expected: dict[str, dict[str, object]] = field(default_factory=dict)
    #: Runs with and without features are separate sets (a seed may appear in both).
    split_by_features: bool = False


#: The distinguishing settings of ``configs/reference/R1.yaml`` (R4 differs in two).
_R1_SETTINGS: Final[dict[str, object]] = {
    "dataset": LABEL_FRACTION_DATASETS[100],
    "supervised_model": "stamp_classifier",
    "image_size": 27,
    "augmentation": "without_aug",
    "batch_size": 45,
    "optimizer": "AdamW",
    "max_epochs": 220,
    "patience": 70,
}

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
        expected={"train-supervised": {**_R1_SETTINGS, "with_features": True}},
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
        expected={
            "train-supervised": {
                "dataset": LABEL_FRACTION_DATASETS[100],
                "supervised_model": "resnet18",
                "image_size": 63,
                "with_features": False,
                "augmentation": "rotation_v3",
                "batch_size": 30,
                "optimizer": "AdamW",
                "max_epochs": 220,
                "patience": 70,
            }
        },
    ),
    "R3": Reference(
        "R3",
        "SimCLR pretrain (31b10c9 settings), then linear evaluation at 1, 10 and 100 %",
        ("pretrain", "linear-eval"),
        "baseline",
        note="no legacy number: this result becomes the first recorded reference",
        expected={
            "pretrain": {
                "encoder_name": "pstamps",
                "batch_size": 550,
                "optimizer": "LARS",
                "augmentation": "astro8",
                "max_epochs": 850,
                "patience": 100,
            },
            "linear-eval": {
                "with_features": False,
                "augmentation": "without_aug",
                "batch_size": 45,
                "optimizer": "AdamW",
                "max_epochs": 220,
                "patience": 40,
            },
        },
    ),
    "R4": Reference(
        "R4",
        "CE stamp classifier, R1 config without features, 1 % labels",
        ("train-supervised",),
        "report",
        supervised_model="stamp_classifier",
        note="the comparison point for R3",
        expected={
            "train-supervised": {
                **_R1_SETTINGS,
                "dataset": LABEL_FRACTION_DATASETS[1],
                "with_features": False,
            }
        },
    ),
    "CD21": Reference(
        "CD21",
        "Carrasco-Davis et al. 2021 stamp classifier, paper replication (test accuracy)",
        ("train-supervised",),
        "two_tier",
        supervised_model="paper_stamp_classifier",
        mean=CD21_TEST_MEAN,
        std=CD21_TEST_STD,
        n=CD21_SEEDS,
        source="Carrasco-Davis et al. 2021, AJ 162, 231 (arXiv:2008.03309), std with ddof=0",
        note="a perfect replication misses the faithful tier about 8 % of the time; one "
        "confirmatory 5-seed rerun is allowed (plan, task 2.3)",
        # The carrasco2021 preset (task 2.2c): a diagnostic run gets a WARN line.
        expected={
            "train-supervised": {
                key: value
                for key, value in PRESETS[CARRASCO2021_PRESET].items()
                if key != "with_features"
            }
        },
        split_by_features=True,
    ),
}


@dataclass(slots=True)
class Comparison:
    """The outcome of :func:`compare_reference`: the result and the lines to print."""

    result: Result
    lines: list[str] = field(default_factory=list)
    #: The CD21 tier of a PASS (``None`` otherwise).
    tier: Tier | None = None


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


def machine_signature(run: dict[str, Any]) -> tuple[object, ...] | None:
    """What must match between runs of one comparison: device type, GPU, torch, CUDA and
    the effective float32 precision (``ieee`` or ``tf32``; ADR 0010).

    Read from ``metrics.json["machine"]`` (written since task 2.3); ``None`` if absent. A
    run written before ADR 0010 has no ``fp32_precision`` and reads as ``ieee``: every such
    run ran on hardware without TF32 (the GTX 1050, CPU or MPS; ``results/``).
    """
    machine = run.get("machine")
    if not isinstance(machine, dict):
        return None
    hardware = tuple(machine.get(key) for key in ("device", "gpu", "torch", "cuda"))
    return (*hardware, machine.get("fp32_precision", "ieee"))


def read_oracle_report(path: Path) -> bool:
    """Whether the M1 oracle report at ``path`` (task 2.2b) says the oracle passed.

    The report is a JSON object; its top-level ``passed`` must be a boolean.

    Raises:
        ConfigError: if the file cannot be read, is not JSON, or has no boolean ``passed``.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ConfigError(f"{path}: cannot read the oracle report ({err})") from None
    except json.JSONDecodeError as err:
        raise ConfigError(f"{path}: invalid JSON ({err})") from None
    if not isinstance(data, dict) or not isinstance(data.get("passed"), bool):
        raise ConfigError(f"{path}: an oracle report needs a top-level boolean 'passed'")
    return bool(data["passed"])


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
        config = run.get("config", {})
        key: tuple[object, ...] = (run.get("command"), config.get("dataset"), run.get("seed"))
        if reference.split_by_features:
            key = (*key, _with_features(run))
        if key in seen:
            raise ConfigError(
                f"{run['_path']} and {seen[key]} are the same run (command, dataset, seed "
                f"{key[2]}); the seeds of a set must differ"
            )
        seen[key] = run["_path"]


def _config_differences(reference: Reference, runs: Sequence[dict[str, Any]]) -> list[str]:
    lines = []
    defaults = settings_to_plain(Settings())
    for run in runs:
        expected = reference.expected.get(run.get("command", ""), {})
        config = {**defaults, **run.get("config", {})}
        different = [
            f"{key}={config.get(key)!r} (expected {value!r})"
            for key, value in expected.items()
            if config.get(key) != value
        ]
        if different:
            lines.append(
                f"  WARN: {run['_path']} differs from the {reference.name} config: "
                + ", ".join(different)
            )
    return lines


def _machine_warnings(runs: Sequence[dict[str, Any]]) -> list[str]:
    signatures = {machine_signature(run) for run in runs} - {None}
    if len(signatures) <= 1:
        return []
    detail = "; ".join(sorted(map(str, signatures)))
    return [
        f"  WARN: the runs come from different machines, builds or float32 precisions: {detail}"
    ]


def _with_features(run: dict[str, Any]) -> bool:
    """Whether the run's model read features: ``with_features`` and a ``feature_set`` not
    ``none`` (ADR 0016; a key missing from an older run reads as its default)."""
    config = run.get("config", {})
    return config.get("with_features") is not False and config.get("feature_set") != "none"


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


def _recalls(runs: Sequence[dict[str, Any]], stage: str, label: str) -> list[float] | None:
    values: list[float] = []
    for run in runs:
        recall = run.get("evaluation", {}).get(stage, {}).get("recall_per_class", {})
        value = _as_float(recall.get(label)) if isinstance(recall, dict) else None
        if value is None:
            return None
        values.append(value)
    return values


def _within(value: float, target: float, tolerance: float) -> bool:
    return round(abs(value - target), _DIGITS) <= tolerance


def _spread(values: Sequence[float]) -> str:
    """Mean, std with ddof=1 and with ddof=0 (the paper's convention)."""
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.4f} (n=1)"
    return (
        f"{mean:.4f} ± {statistics.stdev(values):.4f} (ddof=1; "
        f"{statistics.pstdev(values):.4f} with ddof=0, n={len(values)})"
    )


def _report_without_features(runs: Sequence[dict[str, Any]], out: list[str]) -> None:
    if not runs:
        return
    values = _accuracies(runs, "test_acc_at_best")
    if values is None:
        out.append("  without features: test_acc_at_best is missing or not finite in some run")
        return
    diff = abs(statistics.mean(values) - CD21_NO_FEATURES_MEAN)
    out.append(
        f"  without features (advisory, not gating): test accuracy {_spread(values)}; paper "
        f"{CD21_NO_FEATURES_MEAN:.3f} ± {CD21_NO_FEATURES_STD:.3f} (ddof=0); "
        f"|difference| = {diff:.4f}"
    )


def _gate_two_tier(
    runs: Sequence[dict[str, Any]], oracle: bool | None, out: list[str]
) -> tuple[Result, Tier | None]:
    with_features = [run for run in runs if _with_features(run)]
    _report_without_features([run for run in runs if not _with_features(run)], out)
    if not with_features:
        out.append(
            "  no run with features: nothing to gate (the without-features line is advisory)"
        )
        return "REPORT", None
    tests = _accuracies(with_features, "test_acc_at_best")
    vals = _accuracies(with_features, "val_acc_best")
    sn = _recalls(with_features, "test", "SN")
    if tests is None or vals is None or sn is None:
        out.append(
            "  FAIL: test_acc_at_best, val_acc_best or the SN test recall "
            "(evaluation.test.recall_per_class, written since task 2.3) is missing or not finite"
        )
        return "FAIL", None
    mean = statistics.mean(tests)
    diff = round(abs(mean - CD21_TEST_MEAN), _DIGITS)
    sn_ok = _within(statistics.mean(sn), CD21_SN_RECALL, CD21_SN_TOLERANCE)
    val_ok = _within(statistics.mean(vals), CD21_VAL_MEAN, CD21_VAL_TOLERANCE)
    out += [
        f"  test accuracy: {_spread(tests)}",
        f"  paper:         {CD21_TEST_MEAN:.3f} ± {CD21_TEST_STD:.3f} (ddof=0, n={CD21_SEEDS}; "
        f"about {CD21_TEST_STD * math.sqrt(CD21_SEEDS / (CD21_SEEDS - 1)):.4f} with ddof=1)",
        f"  |difference| = {diff:.4f} (faithful <= {CD21_FAITHFUL_TOLERANCE}, acceptable <= "
        f"{CD21_ACCEPTABLE_TOLERANCE})",
        f"  SN test recall: {_describe(sn)}; needs {CD21_SN_RECALL:.2f} ± {CD21_SN_TOLERANCE:.2f}"
        f" -> {'ok' if sn_ok else 'OUT'}",
        f"  val_acc_best:   {_describe(vals)}; needs {CD21_VAL_MEAN:.3f} ± "
        f"{CD21_VAL_TOLERANCE:.2f} -> {'ok' if val_ok else 'OUT'}",
        "  M1 oracle: "
        + {True: "passed", False: "FAILED", None: "no report given (--oracle-report)"}[oracle],
    ]
    best_steps = [run.get("best_step") for run in with_features if "best_step" in run]
    if best_steps:
        out.append(f"  best_step: {best_steps}")
    tier: Tier | None = None
    if sn_ok and val_ok and diff <= CD21_FAITHFUL_TOLERANCE and oracle is True:
        tier = "faithful"
    elif sn_ok and val_ok and diff <= CD21_ACCEPTABLE_TOLERANCE:
        tier = "acceptable"
        if diff <= CD21_FAITHFUL_TOLERANCE:
            out.append("  faithful needs the M1 oracle to have passed")
    if len(tests) != CD21_SEEDS:
        verdict = "FAIL" if tier is None else f"PASS ({tier})"
        out.append(
            f"  provisional: the gate is defined on {CD21_SEEDS} seeds, got {len(tests)}; "
            f"with these runs it would be {verdict}"
        )
        return "REPORT", None
    return ("FAIL", None) if tier is None else ("PASS", tier)


def _baseline(run: dict[str, Any]) -> float | None:
    counts = run.get("data", {}).get("class_counts", {}).get("Validation")
    if not counts or not all(isinstance(c, int) for c in counts) or sum(counts) == 0:
        return None
    return float(max(counts)) / float(sum(counts))


def _same_run(pretrained: object, pretrain_dirs: Sequence[str]) -> bool:
    """True if a linear-eval run's ``pretrained`` setting points at one of the pretrain runs
    (their directory, or its ``checkpoints/best.ckpt``)."""
    if not isinstance(pretrained, str):
        return False
    path = Path(pretrained).resolve()
    for run_dir in pretrain_dirs:
        root = Path(run_dir).resolve()
        if root.is_file():  # the metrics.json itself was given
            root = root.parent
        if path in (root, root / "checkpoints" / "best.ckpt"):
            return True
    return False


def _gate_pretrain(runs: Sequence[dict[str, Any]], out: list[str]) -> bool:
    if not runs:
        out.append("  FAIL: no pretrain run given (R3 needs it: its loss must be finite)")
        return False
    passed = True
    for run in runs:
        ok = _finite(run.get("monitor_best"))
        out.append(
            f"  pretrain {run['_path']}: {run.get('monitor')} = {run.get('monitor_best')}"
            f" -> {'finite' if ok else 'NOT FINITE'}"
        )
        passed &= ok
    return passed


def _gate_baseline(runs: Sequence[dict[str, Any]], out: list[str]) -> bool:
    pretrains = [run for run in runs if run.get("command") == "pretrain"]
    passed = _gate_pretrain(pretrains, out)
    pretrain_dirs = [run["_path"] for run in pretrains]
    seen: set[str] = set()
    for run in runs:
        if run.get("command") == "pretrain":
            continue
        config = run.get("config", {})
        dataset = config.get("dataset")
        seen.add(str(dataset))
        value = _as_float(run.get("val_acc_best"))
        test = _as_float(run.get("test_acc_at_best"))
        baseline = _baseline(run)
        ok = value is not None and test is not None and baseline is not None and value > baseline
        out.append(
            f"  linear-eval {dataset} (seed {run.get('seed')}): val_acc_best {value}, "
            f"test {test}, majority baseline {baseline} -> {'above' if ok else 'NOT above'}"
        )
        if pretrains and not _same_run(config.get("pretrained"), pretrain_dirs):
            out.append(
                f"  FAIL: {run['_path']} was trained on {config.get('pretrained')!r}, not on "
                "the given pretrain run"
            )
            ok = False
        passed &= ok
    missing = [
        f"{fraction} % ({dataset})"
        for fraction, dataset in LABEL_FRACTION_DATASETS.items()
        if dataset not in seen
    ]
    if missing:
        out.append(f"  FAIL: incomplete, no linear-eval run at {', '.join(missing)}")
        passed = False
    return passed


def compare_reference(
    name: str, run_dirs: Sequence[Path], *, oracle_report: Path | None = None
) -> Comparison:
    """Load the runs in ``run_dirs`` and apply the gate of reference ``name``.

    ``oracle_report`` is the M1 oracle report, used by CD21 only (its faithful tier).

    Raises:
        ConfigError: for an unknown reference, an unreadable run or oracle report, a run of
            the wrong kind (command or model), two runs with the same seed, or an oracle
            report given to another reference than CD21.
    """
    if name not in REFERENCES:
        raise ConfigError(f"unknown reference {name!r}; valid: {', '.join(REFERENCES)}")
    reference = REFERENCES[name]
    if oracle_report is not None and reference.gate != "two_tier":
        raise ConfigError(f"--oracle-report is only used by CD21, not by {name}")
    runs = [load_run(path) for path in run_dirs]
    _check_runs(reference, runs)
    lines = [
        f"{reference.name}: {reference.description}",
        *_config_differences(reference, runs),
        *_machine_warnings(runs),
    ]
    result: Result
    if reference.gate == "two_tier":
        oracle = None if oracle_report is None else read_oracle_report(oracle_report)
        result, tier = _gate_two_tier(runs, oracle, lines)
        if reference.note:
            lines.append(f"  note: {reference.note}")
        lines.append(f"{reference.name}: {result}" + (f" ({tier})" if tier else ""))
        return Comparison(result, lines, tier)
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
