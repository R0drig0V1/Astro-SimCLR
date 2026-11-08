"""Golden tests of ``stats.hierarchical_paired_bootstrap`` (phase 5, task 5.4).

Two brute-force references on tiny cases:

- **exact enumeration**: every equally likely resample (subsets with replacement, then
  objects within class with replacement) is listed and scored with plain Python loops, so
  the exact bootstrap distribution is known; the function's Monte Carlo distribution must
  match it;
- **replayed draws**: the function's own draws (``_hierarchical_draws``, whose order is part
  of its contract) are scored again with the loop reference, resample by resample, and the
  interval recomputed from them must be the function's.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest

from astro_simclr.protocol.benchmark_metrics import (
    confusion_counts,
    macro_f1,
    macro_f1_from_counts,
    reweighted_counts,
)
from astro_simclr.protocol.stats import (
    _hierarchical_draws,
    hierarchical_paired_bootstrap,
)

N_CLASSES = 3


def _confusion(
    truth: Sequence[int], predicted: Sequence[int], weights: Sequence[float]
) -> list[list[float]]:
    counts = [[0.0] * N_CLASSES for _ in range(N_CLASSES)]
    for t, p, w in zip(truth, predicted, weights, strict=True):
        counts[t][p] += w
    return counts


def _balanced_accuracy(counts: list[list[float]]) -> float:
    recalls = [row[i] / sum(row) for i, row in enumerate(counts) if sum(row) > 0]
    return sum(recalls) / len(recalls)


def _macro_f1(counts: list[list[float]], reference: Sequence[float] | None = None) -> float:
    if reference is not None:
        counts = [
            [v / sum(row) * reference[i] if sum(row) else 0.0 for v in row]
            for i, row in enumerate(counts)
        ]
    scores = []
    for c in range(N_CLASSES):
        tp = counts[c][c]
        fn = sum(counts[c]) - tp
        fp = sum(counts[r][c] for r in range(N_CLASSES)) - tp
        if tp + fp + fn > 0:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def _score(counts: list[list[float]], metric: str, reference: Sequence[float] | None) -> float:
    if metric == "balanced_accuracy":
        return _balanced_accuracy(counts)
    return _macro_f1(counts, reference)


def _reference_statistic(
    base: np.ndarray,
    variant: np.ndarray,
    truth: Sequence[int],
    item_object: Sequence[int],
    subsets: Sequence[int],
    object_weights: Sequence[float],
    metric: str,
    reference: Sequence[float] | None = None,
) -> float:
    """Loop reference: mean over the drawn subsets of the paired difference."""
    weights = [object_weights[o] for o in item_object]
    total = 0.0
    for r in subsets:
        left = _score(_confusion(truth, base[r].tolist(), weights), metric, reference)
        right = _score(_confusion(truth, variant[r].tolist(), weights), metric, reference)
        total += right - left
    return total / len(subsets)


# A tiny case: 3 subsets; 2 objects of class 0, 2 of class 1 (one with two alerts), 1 of
# class 2; 6 items.
TRUTH = [0, 0, 1, 1, 1, 2]
GROUPS = ["a", "b", "c", "d", "d", "e"]
ITEM_OBJECT = [0, 1, 2, 3, 3, 4]
BY_CLASS = [[0, 1], [2, 3], [4]]
BASE = np.array([[0, 1, 1, 0, 1, 2], [0, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 2]])
VARIANT = np.array([[0, 0, 1, 1, 1, 2], [0, 1, 1, 1, 1, 2], [0, 0, 2, 1, 1, 0]])


def _exact_distribution(metric: str) -> dict[float, Fraction]:
    """Every resample, equally likely, and its statistic (rounded to merge float noise)."""
    n_runs = BASE.shape[0]
    outcomes: dict[float, Fraction] = {}
    object_draws = [list(itertools.product(objs, repeat=len(objs))) for objs in BY_CLASS]
    n_total = n_runs**n_runs * int(np.prod([len(d) for d in object_draws]))
    for subsets in itertools.product(range(n_runs), repeat=n_runs):
        for draws in itertools.product(*object_draws):
            weights = [0.0] * 5
            for draw in draws:
                for obj in draw:
                    weights[obj] += 1
            value = round(
                _reference_statistic(BASE, VARIANT, TRUTH, ITEM_OBJECT, subsets, weights, metric),
                9,
            )
            outcomes[value] = outcomes.get(value, Fraction(0)) + Fraction(1, n_total)
    return outcomes


@pytest.mark.parametrize("metric", ["balanced_accuracy", "macro_f1"])
def test_matches_the_exact_bootstrap_distribution(metric: str) -> None:
    exact = _exact_distribution(metric)
    assert sum(exact.values()) == 1
    n_boot = 40_000
    stats = _monte_carlo(metric, n_boot)
    values = np.array(sorted(exact))
    probs = np.array([float(exact[v]) for v in values])
    # Every Monte Carlo statistic is one of the enumerated outcomes...
    assert np.isin(np.round(stats, 9), values).all()
    # ...with the exact frequencies (within 5 binomial standard errors).
    freq = np.array([np.mean(np.round(stats, 9) == v) for v in values])
    assert np.all(np.abs(freq - probs) <= 5 * np.sqrt(probs * (1 - probs) / n_boot) + 1e-12)
    # And the mean of the exact distribution is close to the Monte Carlo mean.
    assert float(np.mean(stats)) == pytest.approx(float(probs @ values), abs=0.005)


def _monte_carlo(metric: str, n_boot: int, seed: int = 0) -> np.ndarray:
    """The function's own resample statistics, recomputed from its replayed draws."""
    rng = np.random.default_rng(seed)
    by_class = [np.asarray(objs, dtype=np.int64) for objs in BY_CLASS]
    subsets, weights = _hierarchical_draws(rng, n_boot, BASE.shape[0], by_class, 5)
    counts_base = np.stack([_counts(BASE[r], weights) for r in range(BASE.shape[0])], axis=1)
    counts_var = np.stack([_counts(VARIANT[r], weights) for r in range(BASE.shape[0])], axis=1)
    if metric == "balanced_accuracy":
        diff = _recall(counts_var).mean(-1) - _recall(counts_base).mean(-1)
    else:
        diff = macro_f1_from_counts(counts_var) - macro_f1_from_counts(counts_base)
    return np.take_along_axis(diff, subsets, axis=1).mean(axis=1)


def _recall(counts: np.ndarray) -> np.ndarray:
    recall: np.ndarray = np.diagonal(counts, axis1=-2, axis2=-1) / counts.sum(axis=-1)
    return recall


def _counts(predicted: np.ndarray, object_weights: np.ndarray) -> np.ndarray:
    item_weights = object_weights[:, ITEM_OBJECT]
    counts = np.zeros((object_weights.shape[0], N_CLASSES, N_CLASSES))
    for i, (t, p) in enumerate(zip(TRUTH, predicted.tolist(), strict=True)):
        counts[:, t, p] += item_weights[:, i]
    return counts


@pytest.mark.parametrize("metric", ["balanced_accuracy", "macro_f1"])
@pytest.mark.parametrize("reference", [None, (5.0, 1.0, 3.0)])
def test_replayed_draws_match_the_loop_reference(
    metric: str, reference: tuple[float, ...] | None
) -> None:
    n_boot, seed = 300, 4
    result = hierarchical_paired_bootstrap(
        BASE,
        VARIANT,
        TRUTH,
        n_classes=N_CLASSES,
        metric=metric,  # type: ignore[arg-type]
        groups=GROUPS,
        reference_counts=reference,
        n_boot=n_boot,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    by_class = [np.asarray(objs, dtype=np.int64) for objs in BY_CLASS]
    subsets, weights = _hierarchical_draws(rng, n_boot, BASE.shape[0], by_class, 5)
    stats = np.array(
        [
            _reference_statistic(
                BASE,
                VARIANT,
                TRUTH,
                ITEM_OBJECT,
                subsets[b].tolist(),
                weights[b].tolist(),
                metric,
                reference if metric == "macro_f1" else None,
            )
            for b in range(n_boot)
        ]
    )
    low, high = np.quantile(stats, [0.025, 0.975])
    assert result.low == pytest.approx(low, abs=1e-12)
    assert result.high == pytest.approx(high, abs=1e-12)
    observed = _reference_statistic(
        BASE,
        VARIANT,
        TRUTH,
        ITEM_OBJECT,
        range(3),
        [1.0] * 5,
        metric,
        reference if metric == "macro_f1" else None,
    )
    assert result.observed == pytest.approx(observed, abs=1e-12)
    assert result.metric == metric
    assert result.n_boot == n_boot


def test_observed_value_is_the_subset_mean_of_the_benchmark_metrics() -> None:
    rng = np.random.default_rng(1)
    truth = np.repeat(np.arange(N_CLASSES), 20)
    base = rng.integers(0, N_CLASSES, size=(4, truth.size))
    variant = np.where(rng.random((4, truth.size)) < 0.5, truth, base)
    result = hierarchical_paired_bootstrap(
        base, variant, truth, n_classes=N_CLASSES, metric="macro_f1", n_boot=50
    )
    expected = np.mean(
        [
            macro_f1(truth, variant[r], N_CLASSES) - macro_f1(truth, base[r], N_CLASSES)
            for r in range(4)
        ]
    )
    assert result.observed == pytest.approx(expected, abs=1e-12)
    reference = (1.0, 2.0, 7.0)
    reweighted = hierarchical_paired_bootstrap(
        base,
        variant,
        truth,
        n_classes=N_CLASSES,
        metric="macro_f1",
        reference_counts=reference,
        n_boot=50,
    )

    def rw(pred: np.ndarray) -> float:
        counts = reweighted_counts(confusion_counts(truth, pred, N_CLASSES), reference)
        return float(macro_f1_from_counts(counts))

    expected_rw = np.mean([rw(variant[r]) - rw(base[r]) for r in range(4)])
    assert reweighted.observed == pytest.approx(expected_rw, abs=1e-12)
    assert result.low <= result.observed <= result.high


def test_identical_arms_give_exactly_zero() -> None:
    result = hierarchical_paired_bootstrap(BASE, BASE, TRUTH, n_classes=N_CLASSES, n_boot=200)
    assert (result.observed, result.low, result.high) == (0.0, 0.0, 0.0)


def test_deterministic_and_seeded() -> None:
    def run(seed: int) -> tuple[float, float]:
        ci = hierarchical_paired_bootstrap(
            BASE, VARIANT, TRUTH, n_classes=N_CLASSES, groups=GROUPS, n_boot=500, seed=seed
        )
        return ci.low, ci.high

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_subset_variance_widens_the_interval() -> None:
    """Arms that differ only by subset: the subset level must add width."""
    rng = np.random.default_rng(2)
    truth = np.repeat(np.arange(N_CLASSES), 200)
    base = np.tile(truth, (10, 1))
    # Each variant subset gets a different share of errors: a large subset-to-subset spread.
    share = np.linspace(0.0, 0.5, 10)
    variant = np.stack(
        [np.where(rng.random(truth.size) < s, (truth + 1) % N_CLASSES, truth) for s in share]
    )
    one = hierarchical_paired_bootstrap(
        base[:1].repeat(10, 0), variant[5:6].repeat(10, 0), truth, n_classes=N_CLASSES
    )
    many = hierarchical_paired_bootstrap(base, variant, truth, n_classes=N_CLASSES)
    assert (many.high - many.low) > 2 * (one.high - one.low)


def test_large_case_runs_in_blocks() -> None:
    rng = np.random.default_rng(3)
    truth = np.repeat(np.arange(5), 400)
    base = rng.integers(0, 5, size=(20, truth.size))
    result = hierarchical_paired_bootstrap(base, base[::-1], truth, n_classes=5, n_boot=600)
    assert result.low <= 0 <= result.high


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"metric": "accuracy"}, "unknown metric"),
        ({"n_classes": 1}, "at least 2 classes"),
        ({"n_boot": 0}, "n_boot"),
        ({"confidence": 1.0}, "confidence"),
        ({"labels": [0, 0, 1, 1, 1, 3]}, "range"),
        ({"labels": [0.0, 0, 1, 1, 1, 2]}, "integer"),
        ({"labels": []}, "non-empty"),
        ({"predicted_variant": VARIANT[:2]}, "paired by subset"),
        ({"predicted_variant": VARIANT + 5}, "range"),
        ({"groups": ["a"] * 6}, "two classes"),
        ({"groups": ["a"]}, "shape"),
        ({"reference_counts": (1.0, 2.0)}, "reference_counts"),
        ({"reference_counts": (1.0, -2.0, 1.0)}, "reference_counts"),
        ({"reference_counts": (0.0, 0.0, 0.0)}, "reference_counts"),
    ],
)
def test_refuses_bad_inputs(kwargs: dict[str, object], match: str) -> None:
    args: dict[str, object] = {
        "predicted_baseline": BASE,
        "predicted_variant": VARIANT,
        "labels": TRUTH,
        "n_classes": N_CLASSES,
        "n_boot": 10,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        hierarchical_paired_bootstrap(**args)  # type: ignore[arg-type]
