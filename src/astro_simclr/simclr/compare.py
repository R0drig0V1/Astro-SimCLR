"""The paired go/no-go comparison of two arms' subset reports (phase 5, task 5.0b; ADR 0025).

**Subset reports.** ``simclr-probe`` (an arm: the SSL probe, the random-init probe) and
``simclr-scratch`` (the supervised scratch baseline) write one JSON report per label-subsets
file: a row per subset with its decision-half balanced accuracy. From
:data:`REPORT_VERSION` 2 a report also holds what a paired interval needs
(:func:`report_header`): the subsets and splits files (path and SHA-256), the decision
half's Validation rows and labels, and each subset's per-stamp ``decision_predictions``.
A version-1 report (written before, without ``report_version``) stays readable as a table;
the comparison refuses it with a message to re-run its command.

**The comparison** (:func:`compare_reports`). Both reports must be paired: the same subsets
file, splits file, decision rows and labels, and subsets. The interval is
:func:`~astro_simclr.protocol.stats.hierarchical_paired_bootstrap` of balanced accuracy,
variant - baseline: the subsets resampled with replacement, then the decision half's
**objects** within class (the splits file's groups: the alerts of one object move
together; a version-1 splits file makes each stamp an object).

**The verdict** (:func:`margin_verdict`; declared by the integrator before any baseline
result, user decision of 2026-11-09). At the primary budget, :data:`PRIMARY_PER_CLASS` =
10 per class, the variant (the SSL probe) must beat the baseline (the scratch model) by
**more than** :data:`GO_MARGIN` = 2 points of balanced accuracy, read as H1 is: the lower
bound of the paired 95 % interval above +0.02 (``go``), else ``no-go``. Other budgets (100
per class) are secondary: their interval is reported, with no verdict.

Nothing here imports torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

from astro_simclr.data.files import sha256_file
from astro_simclr.data.schema import LABEL_NAMES, DataError
from astro_simclr.protocol.splits import ProtocolSplits
from astro_simclr.protocol.stats import BootstrapCI, hierarchical_paired_bootstrap

__all__ = [
    "GO_MARGIN",
    "PRIMARY_PER_CLASS",
    "REPORT_VERSION",
    "Verdict",
    "compare_reports",
    "decision_groups",
    "margin_verdict",
    "read_report",
    "report_header",
    "summary",
]

#: The version of a subset report that holds per-stamp predictions (1: before them).
REPORT_VERSION: Final = 2
#: The go/no-go's primary budget (stamps per class).
PRIMARY_PER_CLASS: Final = 10
#: The declared margin: the interval's lower bound must exceed 2 points of balanced accuracy.
GO_MARGIN: Final = 0.02
#: Largest accepted margin (a fraction; 2.0 would be a margin typed in points).
_MAX_MARGIN: Final = 0.2
#: Rounding of a bound compared with the margin (as ``protocol.stats``).
_DIGITS: Final = 12

Verdict = Literal["go", "no-go"]


def _file(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def report_header(
    command: str,
    *,
    subsets_path: Path,
    splits_path: Path,
    decision_rows: npt.ArrayLike,
    decision_labels: npt.ArrayLike,
) -> dict[str, Any]:
    """The pairing keys of a version-2 subset report (module docstring).

    Raises:
        OSError: if a file cannot be read.
    """
    return {
        "report": command,
        "report_version": REPORT_VERSION,
        "subsets_file": _file(subsets_path),
        "splits_file": _file(splits_path),
        "decision_rows": np.asarray(decision_rows, dtype=np.int64).tolist(),
        "decision_labels": np.asarray(decision_labels, dtype=np.int64).tolist(),
    }


def summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    """The subsets' mean and standard deviation (``ddof=1``; 0 for one) of ``decision_bacc``."""
    scores = np.array([row["decision_bacc"] for row in rows], dtype=np.float64)
    return {
        "decision_bacc_mean": float(scores.mean()),
        "decision_bacc_std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
    }


def read_report(path: Path) -> dict[str, Any]:
    """Read a subset report (any version).

    Raises:
        DataError: if it is unreadable or not a JSON object with ``per_class`` and
            ``subsets``.
    """
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise DataError(f"{path}: cannot read the report: {err}") from err
    if not isinstance(report, dict) or "per_class" not in report or "subsets" not in report:
        raise DataError(f"{path}: not a simclr-probe or simclr-scratch report")
    return report


def _predictions(report: dict[str, Any], name: str) -> dict[int, npt.NDArray[np.int64]]:
    """The report's per-subset decision predictions, by subset number.

    Raises:
        DataError: for a version-1 report, or a subset row without predictions.
    """
    if report.get("report_version", 1) < REPORT_VERSION:
        raise DataError(
            f"{name} is a version-1 report, without per-stamp predictions: re-run its "
            "command (simclr-probe or simclr-scratch) to write them"
        )
    rows = {}
    for row in report["subsets"]:
        if "decision_predictions" not in row:
            raise DataError(f"{name}: subset {row.get('subset')} has no decision_predictions")
        rows[int(row["subset"])] = np.asarray(row["decision_predictions"], dtype=np.int64)
    return rows


def _check_paired(baseline: dict[str, Any], variant: dict[str, Any]) -> None:
    """Refuse two reports that are not paired by subset and stamp.

    Raises:
        DataError: naming the first key that differs.
    """
    for key in ("per_class", "subsets_file", "splits_file", "decision_rows", "decision_labels"):
        left, right = baseline.get(key), variant.get(key)
        if key.endswith("_file"):
            left = None if left is None else left.get("sha256")
            right = None if right is None else right.get("sha256")
        if left != right:
            raise DataError(f"the reports are not paired: their {key} differ")


def decision_groups(
    splits: ProtocolSplits, decision_rows: npt.ArrayLike
) -> npt.NDArray[np.int64] | None:
    """The object id of each decision-half Validation row (``None`` without grouping)."""
    if splits.grouping is None:
        return None
    rows = np.asarray(decision_rows, dtype=np.int64)
    return np.asarray(splits.grouping.group_of[splits.n_train + rows], dtype=np.int64)


def margin_verdict(interval: BootstrapCI, margin: float = GO_MARGIN) -> Verdict:
    """``go`` if the interval's lower bound is above ``margin`` (strictly), else ``no-go``.

    The bound is rounded to 12 digits first, so a bound of exactly ``margin`` is not above.

    Raises:
        ValueError: for a margin outside ``[0, 0.2]`` (a fraction, not points).
    """
    if not 0 <= margin <= _MAX_MARGIN:
        raise ValueError(
            f"the margin must be in [0, {_MAX_MARGIN}], got {margin}: it is a fraction of "
            "balanced accuracy (0.02 is 2 points)"
        )
    return "go" if round(interval.low - margin, _DIGITS) > 0 else "no-go"


def compare_reports(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    *,
    groups: npt.ArrayLike | None,
    margin: float = GO_MARGIN,
    primary_per_class: int = PRIMARY_PER_CLASS,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    n_classes: int = len(LABEL_NAMES),
) -> dict[str, Any]:
    """The paired interval of variant - baseline and, at the primary budget, the verdict.

    ``groups`` are the object ids of the decision rows (:func:`decision_groups`).

    Raises:
        DataError: if a report has no predictions or the reports are not paired.
        ValueError: as :func:`~astro_simclr.protocol.stats.hierarchical_paired_bootstrap`
            or :func:`margin_verdict`.
    """
    # The version first: a version-1 report lacks every pairing key, and saying so is clearer.
    left, right = _predictions(baseline, "the baseline"), _predictions(variant, "the variant")
    _check_paired(baseline, variant)
    if sorted(left) != sorted(right):
        raise DataError(f"the reports' subsets differ: {sorted(left)} vs {sorted(right)}")
    order = sorted(left)
    interval = hierarchical_paired_bootstrap(
        np.stack([left[r] for r in order]),
        np.stack([right[r] for r in order]),
        np.asarray(baseline["decision_labels"], dtype=np.int64),
        n_classes=n_classes,
        metric="balanced_accuracy",
        groups=groups,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
    )
    primary = int(baseline["per_class"]) == primary_per_class
    verdict = margin_verdict(interval, margin)
    labels = np.asarray(baseline["decision_labels"])
    n_objects = labels.size if groups is None else int(np.unique(np.asarray(groups)).size)
    return {
        "per_class": int(baseline["per_class"]),
        "n_subsets": len(order),
        "n_stamps": int(labels.size),
        "n_objects": n_objects,
        "metric": "balanced_accuracy",
        "baseline": {
            "report": baseline.get("report"),
            "decision_bacc_mean": baseline.get("decision_bacc_mean"),
        },
        "variant": {
            "report": variant.get("report"),
            "decision_bacc_mean": variant.get("decision_bacc_mean"),
        },
        "difference": interval.observed,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "confidence": interval.confidence,
        "n_boot": interval.n_boot,
        "bootstrap_seed": seed,
        "margin": margin,
        "primary": primary,
        "rule": (
            f"go iff the lower bound of the paired {confidence:.0%} interval of variant - "
            f"baseline exceeds {margin} at {primary_per_class} per class"
        ),
        "verdict": verdict if primary else None,
    }
