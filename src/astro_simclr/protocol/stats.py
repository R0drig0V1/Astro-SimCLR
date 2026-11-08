"""The statistics of the phase-2 protocol: McNemar, the paired bootstrap and the decision rules.

All inputs are per-stamp arrays of the *same* stamps, in the same order (paired):

- :func:`mcnemar`: the exact two-sided McNemar test on two boolean "correct" vectors. The
  plan applies it to **seed-ensembled** predictions (one prediction per stamp and variant:
  :func:`astro_simclr.protocol.predictions.ensemble`); pooling seed x stamp pairs would
  treat correlated pairs as independent and is not allowed.
- :func:`paired_bootstrap`: a percentile confidence interval of ``metric(variant) -
  metric(baseline)``, resampling stamps. Each stamp's score is the fraction of seeds that
  classify it correctly (``seed_mean_correct``), so the observed value is the difference
  of the seed-mean accuracies. Resampling is **stratified by class**: each class keeps its
  count, because the evaluation sets are balanced by design (100 or 50 per class; the CV
  folds are stratified), so the class mix is not a source of sampling noise. With
  ``metric="balanced_accuracy"`` (the CV metric) the classes weigh equally.
- :func:`adoption_decision`: the **superiority** rule of task 2.3: a flag is adopted only
  if the 95 % interval of its CV difference against the baseline excludes 0 on the
  positive side.
- :func:`non_inferiority_decision`: the **non-inferiority** rule of the efficiency rows of
  task 2.5 (B7, B2', B1'), which only need to be "as good": adopt when the interval's
  lower bound is above ``-margin`` (0.005, half a point of balanced accuracy).
- :func:`screen_decision`: the 250/250 screen can only **drop** a variant, when it is worse
  than the baseline by 4 points or more; it never adopts one.

The bootstrap draws from its own ``np.random.default_rng(seed)``, not from the global
generators, so a comparison is reproducible on its own. Nothing here imports torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "MAX_NON_INFERIORITY_MARGIN",
    "NON_INFERIORITY_MARGIN",
    "SCREEN_DROP_THRESHOLD",
    "Adoption",
    "BootstrapCI",
    "McNemar",
    "Metric",
    "Rule",
    "adoption_decision",
    "balanced_accuracy",
    "check_margin",
    "mcnemar",
    "non_inferiority_decision",
    "paired_bootstrap",
    "screen_decision",
]

Metric = Literal["accuracy", "balanced_accuracy"]
Adoption = Literal["adopt", "worse", "inconclusive"]
Screen = Literal["drop", "keep"]
#: The CV decision rules: ``superiority`` (task 2.3) or ``non-inferiority`` (efficiency rows).
Rule = Literal["superiority", "non-inferiority"]

#: The screen drops a variant worse than the baseline by this much or more (4 points).
SCREEN_DROP_THRESHOLD: Final = 0.04
#: The non-inferiority margin of the efficiency rows (plan, task 2.5): half a point.
NON_INFERIORITY_MARGIN: Final = 0.005
#: Largest accepted margin. It is a fraction of balanced accuracy; a larger value is far
#: more likely a margin typed in points (0.5 for half a point) than a real choice.
MAX_NON_INFERIORITY_MARGIN: Final = 0.05
#: Upper bound on the bootstrap's (resamples x stamps) block held in memory at once.
_BLOCK: Final = 2_000_000
#: Rounding of differences compared with a threshold (0.94 - 0.90 is 0.0400000000000003).
_DIGITS: Final = 12


@dataclass(frozen=True, slots=True)
class McNemar:
    """The exact McNemar test: discordant counts and the two-sided p-value.

    ``only_baseline``: stamps the baseline gets right and the variant wrong;
    ``only_variant``: the reverse.
    """

    n: int
    only_baseline: int
    only_variant: int
    p_value: float


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    """A percentile bootstrap interval of ``metric(variant) - metric(baseline)``."""

    metric: Metric
    observed: float
    low: float
    high: float
    confidence: float
    n_boot: int


def _check_pair(left: npt.NDArray[Any], right: npt.NDArray[Any]) -> None:
    if left.ndim != 1 or left.shape != right.shape or left.size == 0:
        msg = f"expected two non-empty 1-D arrays of one shape, got {left.shape}, {right.shape}"
        raise ValueError(msg)


def _paired_bool(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    left, right = np.asarray(a, dtype=np.bool_), np.asarray(b, dtype=np.bool_)
    _check_pair(left, right)
    return left, right


def _paired_float(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    _check_pair(left, right)
    return left, right


def mcnemar(correct_baseline: npt.ArrayLike, correct_variant: npt.ArrayLike) -> McNemar:
    """The exact two-sided McNemar test of two paired "correct" vectors.

    ``p = min(1, 2 P(X <= min(b, c)))`` with ``X ~ Binomial(b + c, 1/2)``, computed with
    exact integers (no normal approximation); ``p = 1`` when there is no discordant stamp.

    Raises:
        ValueError: if the inputs are not non-empty 1-D arrays of the same shape.
    """
    base, variant = _paired_bool(correct_baseline, correct_variant)
    only_baseline = int(np.count_nonzero(base & ~variant))
    only_variant = int(np.count_nonzero(~base & variant))
    n = only_baseline + only_variant
    if n == 0:
        return McNemar(int(base.size), 0, 0, 1.0)
    tail = sum(math.comb(n, k) for k in range(min(only_baseline, only_variant) + 1))
    p_value = min(1.0, 2 * tail / 2**n)
    return McNemar(int(base.size), only_baseline, only_variant, p_value)


def _class_indices(labels: npt.ArrayLike, size: int) -> list[npt.NDArray[np.int64]]:
    classes = np.asarray(labels, dtype=np.int64)
    if classes.shape != (size,):
        raise ValueError(f"labels must have shape ({size},), got {classes.shape}")
    return [np.flatnonzero(classes == c).astype(np.int64) for c in np.unique(classes)]


def _weights(groups: list[npt.NDArray[np.int64]], size: int, metric: Metric) -> list[float]:
    if metric == "accuracy":
        return [group.size / size for group in groups]
    if metric == "balanced_accuracy":
        return [1 / len(groups)] * len(groups)
    raise ValueError(f"unknown metric {metric!r}; use accuracy or balanced_accuracy")


def balanced_accuracy(score: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """The mean over the classes present of the mean per-stamp score (mean recall).

    ``score`` is per stamp: 0/1 correctness, or the fraction of seeds that are correct.

    Raises:
        ValueError: if ``score`` is not a non-empty 1-D array or ``labels`` differs in shape.
    """
    values, _ = _paired_float(score, score)
    groups = _class_indices(labels, values.size)
    return float(np.mean([values[group].mean() for group in groups]))


def paired_bootstrap(
    score_baseline: npt.ArrayLike,
    score_variant: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    metric: Metric = "accuracy",
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile interval of ``metric(variant) - metric(baseline)``, stratified by class.

    The scores are per stamp (module docstring); each resample draws, within every class,
    as many stamps as the class has, with replacement, and applies the same draw to both
    variants (that is what makes it paired). Deterministic for given inputs and ``seed``.

    Raises:
        ValueError: on unpaired inputs, an unknown metric, ``n_boot < 1`` or a confidence
            outside (0, 1).
    """
    base, variant = _paired_float(score_baseline, score_variant)
    if n_boot < 1 or not 0 < confidence < 1:
        raise ValueError(f"need n_boot >= 1 and 0 < confidence < 1, got {n_boot}, {confidence}")
    groups = _class_indices(labels, base.size)
    weights = _weights(groups, base.size, metric)
    diff = variant - base
    per_class = [diff[group] for group in groups]
    observed = float(sum(w * float(d.mean()) for w, d in zip(weights, per_class, strict=True)))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    block = max(1, _BLOCK // base.size)
    for start in range(0, n_boot, block):
        rows = min(block, n_boot - start)
        total = np.zeros(rows, dtype=np.float64)
        for weight, values in zip(weights, per_class, strict=True):
            draw = rng.integers(0, values.size, size=(rows, values.size))
            total += weight * values[draw].mean(axis=1)
        stats[start : start + rows] = total
    alpha = 1 - confidence
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(metric, observed, float(low), float(high), confidence, n_boot)


def adoption_decision(interval: BootstrapCI) -> Adoption:
    """The adoption rule: ``adopt`` only if the interval lies entirely above 0.

    ``worse`` if it lies entirely below 0; ``inconclusive`` otherwise. Only ``adopt``
    adopts the flag. Meant for the CV difference (:func:`astro_simclr.protocol.compare.
    cv_comparison`); applied to the test set it would make test a decision input.
    """
    if interval.low > 0:
        return "adopt"
    if interval.high < 0:
        return "worse"
    return "inconclusive"


def check_margin(margin: float) -> None:
    """Check a non-inferiority margin: a fraction in ``(0, MAX_NON_INFERIORITY_MARGIN]``.

    Raises:
        ValueError: otherwise; the message says that the margin is a fraction (a margin
            typed in points, such as 0.5, is the likely mistake).
    """
    if not 0 < margin <= MAX_NON_INFERIORITY_MARGIN:
        msg = (
            f"the non-inferiority margin must be in (0, {MAX_NON_INFERIORITY_MARGIN}], got "
            f"{margin}: it is a fraction of balanced accuracy (0.005 is half a point)"
        )
        raise ValueError(msg)


def non_inferiority_decision(
    interval: BootstrapCI, *, margin: float = NON_INFERIORITY_MARGIN
) -> Adoption:
    """The non-inferiority rule: ``adopt`` if the interval's lower bound is above ``-margin``.

    For changes that only need to be "as good" as the baseline (the efficiency rows of task
    2.5: fewer MACs or parameters), which the superiority rule (:func:`adoption_decision`)
    cannot adopt. ``worse`` if the whole interval is below ``-margin`` (inferior beyond the
    margin); ``inconclusive`` otherwise. ``margin`` is a fraction of the metric (0.005 is
    half a point). Differences are rounded like :func:`screen_decision`'s, so a bound of
    exactly ``-margin`` is not above it.

    Raises:
        ValueError: if ``margin`` is out of range (:func:`check_margin`).
    """
    check_margin(margin)
    if round(interval.low + margin, _DIGITS) > 0:
        return "adopt"
    if round(interval.high + margin, _DIGITS) < 0:
        return "worse"
    return "inconclusive"


def screen_decision(
    baseline: float, variant: float, *, threshold: float = SCREEN_DROP_THRESHOLD
) -> Screen:
    """The screen rule: ``drop`` if ``variant <= baseline - threshold``, else ``keep``.

    ``baseline`` and ``variant`` are the seed-mean accuracies on the screen's decision half.
    ``keep`` only means "not dropped": the screen's minimum detectable effect is about 4
    points, so it never adopts a variant.
    """
    return "drop" if round(variant - baseline, _DIGITS) <= -threshold else "keep"
