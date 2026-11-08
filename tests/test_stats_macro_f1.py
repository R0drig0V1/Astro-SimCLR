"""Tests of the macro-F1 paired bootstrap (golden, against brute force) and of the MDE."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from _bench_support import noisy_predictions

from astro_simclr.protocol.benchmark_metrics import macro_f1
from astro_simclr.protocol.stats import (
    adoption_decision,
    minimum_detectable_effect,
    paired_bootstrap_macro_f1,
)


def _data(n: int = 120, n_classes: int = 4, seed: int = 0) -> npt.NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    # Imbalanced classes, each present.
    labels = np.concatenate(
        [np.arange(n_classes), rng.choice(n_classes, n - n_classes, p=[0.5, 0.3, 0.15, 0.05])]
    )
    return labels.astype(np.int64)


def _brute_force(
    base: npt.NDArray[np.int64],
    variant: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
    n_classes: int,
    n_boot: int,
    seed: int,
) -> tuple[float, npt.NDArray[np.float64]]:
    """The bootstrap written plainly: one resample at a time, macro-F1 from the labels.

    It consumes the generator as the implementation does (one draw per class, for every
    resample at once, in class order), so the two must agree to rounding.
    """
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    draws = [g[rng.integers(0, g.size, size=(n_boot, g.size))] for g in groups]

    def diff(index: npt.NDArray[np.int64]) -> float:
        def side(runs: npt.NDArray[np.int64]) -> float:
            return float(np.mean([macro_f1(labels[index], r[index], n_classes) for r in runs]))

        return side(variant) - side(base)

    stats = np.array(
        [diff(np.concatenate([d[b] for d in draws])) for b in range(n_boot)], dtype=np.float64
    )
    return diff(np.arange(labels.size)), stats


@pytest.mark.parametrize("runs", [1, 3])
def test_golden_against_brute_force(runs: int) -> None:
    labels = _data()
    base = np.stack([noisy_predictions(labels, 0.7, 4, seed=s) for s in range(runs)])
    variant = np.stack([noisy_predictions(labels, 0.8, 4, seed=10 + s) for s in range(runs)])
    observed, stats = _brute_force(base, variant, labels, 4, n_boot=300, seed=7)
    interval = paired_bootstrap_macro_f1(base, variant, labels, n_classes=4, n_boot=300, seed=7)
    assert interval.metric == "macro_f1"
    assert interval.observed == pytest.approx(observed, abs=1e-12)
    low, high = np.quantile(stats, [0.025, 0.975])
    assert (interval.low, interval.high) == pytest.approx((low, high), abs=1e-12)


def test_identical_sides_give_exactly_zero() -> None:
    labels = _data(n=200)
    runs = np.stack([noisy_predictions(labels, 0.75, 4, seed=s) for s in range(5)])
    interval = paired_bootstrap_macro_f1(runs, runs.copy(), labels, n_classes=4, n_boot=2000)
    assert (interval.observed, interval.low, interval.high) == (0.0, 0.0, 0.0)
    assert adoption_decision(interval) == "inconclusive"


def test_deterministic_and_seeded() -> None:
    labels = _data()
    a = noisy_predictions(labels, 0.6, 4, seed=1)
    b = noisy_predictions(labels, 0.9, 4, seed=2)
    first = paired_bootstrap_macro_f1(a, b, labels, n_classes=4, n_boot=500, seed=3)
    assert paired_bootstrap_macro_f1(a, b, labels, n_classes=4, n_boot=500, seed=3) == first
    other = paired_bootstrap_macro_f1(a, b, labels, n_classes=4, n_boot=500, seed=4)
    assert other.observed == first.observed and other.low != first.low
    # A clearly better variant: the interval is above 0; swapped sides mirror it.
    assert first.low > 0
    swapped = paired_bootstrap_macro_f1(b, a, labels, n_classes=4, n_boot=500, seed=3)
    assert swapped.observed == pytest.approx(-first.observed)


def test_blocks_do_not_change_the_draws_count() -> None:
    # More resamples than one memory block holds (2e6 // 120 = 16,666 rows per block).
    labels = _data()
    a = noisy_predictions(labels, 0.6, 4, seed=1)
    interval = paired_bootstrap_macro_f1(a, a, labels, n_classes=4, n_boot=17_000)
    assert interval.n_boot == 17_000 and interval.low == interval.high == 0.0


def test_input_checks() -> None:
    labels = _data(n=20)
    pred = labels.copy()
    with pytest.raises(ValueError, match="at least 2"):
        paired_bootstrap_macro_f1(pred, pred, labels, n_classes=1)
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap_macro_f1(pred, pred, labels, n_classes=4, n_boot=0)
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_macro_f1([], [], [], n_classes=4)
    with pytest.raises(ValueError, match="baseline"):
        paired_bootstrap_macro_f1(pred[:5], pred, labels, n_classes=4)
    with pytest.raises(ValueError, match="variant: predicted labels"):
        paired_bootstrap_macro_f1(pred, pred + 4, labels, n_classes=4)
    with pytest.raises(ValueError, match="labels must be in range"):
        paired_bootstrap_macro_f1(pred, pred, labels + 4, n_classes=4)


def test_minimum_detectable_effect() -> None:
    z = 1.959963984540054 + 0.8416212335729143
    # Paired, as the power note: 5 % discordant stamps on 500 -> about 2.8 points.
    assert minimum_detectable_effect(500, discordant=0.05) == pytest.approx(
        z * math.sqrt(0.05 / 500)
    )
    # Unpaired: RJ23's ~116 test satellites at 72 % recall.
    satellites = minimum_detectable_effect(116, rate=0.72)
    assert satellites == pytest.approx(z * math.sqrt(2 * 0.72 * 0.28 / 116))
    assert 0.16 < satellites < 0.17
    assert minimum_detectable_effect(10, rate=1.0) == 0.0
    with pytest.raises(ValueError, match="exactly one"):
        minimum_detectable_effect(10)
    with pytest.raises(ValueError, match="exactly one"):
        minimum_detectable_effect(10, discordant=0.1, rate=0.5)
    with pytest.raises(ValueError, match="n >= 1"):
        minimum_detectable_effect(0, rate=0.5)
    with pytest.raises(ValueError, match="n >= 1"):
        minimum_detectable_effect(10, rate=1.5)
