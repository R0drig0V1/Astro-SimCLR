"""Tests of the protocol statistics: McNemar, the paired bootstrap and the decision rules."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from astro_simclr.protocol.stats import (
    BootstrapCI,
    adoption_decision,
    balanced_accuracy,
    mcnemar,
    paired_bootstrap,
    screen_decision,
)


def _correct(n_both: int, only_a: int, only_b: int, neither: int) -> tuple[list[bool], list[bool]]:
    a = [True] * n_both + [True] * only_a + [False] * only_b + [False] * neither
    b = [True] * n_both + [False] * only_a + [True] * only_b + [False] * neither
    return a, b


def test_mcnemar_exact_p_value() -> None:
    a, b = _correct(80, 10, 2, 8)
    result = mcnemar(a, b)
    assert (result.n, result.only_baseline, result.only_variant) == (100, 10, 2)
    # 2 P(X <= 2), X ~ Bin(12, 1/2) = 2 (1 + 12 + 66) / 4096 (scipy.stats.binomtest: 0.0386).
    assert result.p_value == pytest.approx(158 / 4096)
    assert mcnemar(b, a).p_value == result.p_value  # symmetric


def test_mcnemar_edge_cases() -> None:
    assert mcnemar([True, False], [True, False]).p_value == 1.0  # no discordant stamp
    assert mcnemar(*_correct(0, 3, 3, 0)).p_value == 1.0  # capped at 1
    big = mcnemar(*_correct(0, 0, 3000, 0))  # exact integers do not overflow
    assert big.p_value == pytest.approx(2 / 2**3000) or big.p_value == 0.0
    with pytest.raises(ValueError, match="one shape"):
        mcnemar([True], [True, False])
    with pytest.raises(ValueError):
        mcnemar([], [])


def test_balanced_accuracy_weighs_classes_equally() -> None:
    labels = [0] * 8 + [1] * 2
    score = [1.0] * 8 + [0.0] * 2
    assert balanced_accuracy(score, labels) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="labels"):
        balanced_accuracy(score, labels[:5])


def test_bootstrap_degenerate_intervals() -> None:
    labels = np.repeat(np.arange(5), 20)
    same = np.linspace(0, 1, 100)
    zero = paired_bootstrap(same, same, labels, n_boot=200)
    assert (zero.observed, zero.low, zero.high) == (0.0, 0.0, 0.0)
    one = paired_bootstrap(np.zeros(100), np.ones(100), labels, n_boot=200)
    assert (one.observed, one.low, one.high) == (1.0, 1.0, 1.0)
    assert one == BootstrapCI("accuracy", 1.0, 1.0, 1.0, 0.95, 200)


def test_bootstrap_metric_weights() -> None:
    labels = np.array([0] * 8 + [1] * 2)
    base = np.zeros(10)
    variant = np.array([0.0] * 8 + [1.0] * 2)  # better only on the small class
    assert paired_bootstrap(base, variant, labels, n_boot=50).observed == pytest.approx(0.2)
    balanced = paired_bootstrap(base, variant, labels, metric="balanced_accuracy", n_boot=50)
    assert balanced.observed == pytest.approx(0.5)
    # Stratified: each class keeps its count, so a constant per-class difference has no spread.
    assert balanced.low == pytest.approx(0.5) and balanced.high == pytest.approx(0.5)


def test_bootstrap_is_deterministic_and_covers_the_truth() -> None:
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(5), 100)
    base = (rng.random(500) < 0.85).astype(float)
    variant = np.where(rng.random(500) < 0.5, 1.0, base)  # about +0.07
    first = paired_bootstrap(base, variant, labels, n_boot=2000, seed=3)
    assert first == paired_bootstrap(base, variant, labels, n_boot=2000, seed=3)
    assert first != paired_bootstrap(base, variant, labels, n_boot=2000, seed=4)
    assert first.low < first.observed < first.high
    assert first.observed == pytest.approx(variant.mean() - base.mean())
    # The width is about 2 x 1.96 x the paired standard error.
    se = np.std(variant - base) / math.sqrt(500)
    assert first.high - first.low == pytest.approx(2 * 1.96 * se, rel=0.2)


def test_bootstrap_blocks_do_not_change_the_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import astro_simclr.protocol.stats as stats

    monkeypatch.setattr(stats, "_BLOCK", 30)  # several blocks of 3 resamples
    labels = np.repeat(np.arange(5), 2)
    result = paired_bootstrap(np.zeros(10), np.linspace(0, 1, 10), labels, n_boot=7)
    assert result.n_boot == 7
    assert result.low <= result.observed <= result.high


def test_bootstrap_rejects_bad_arguments() -> None:
    labels = [0, 1]
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap([0, 1], [1, 1], labels, n_boot=0)
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap([0, 1], [1, 1], labels, confidence=1.0)
    with pytest.raises(ValueError, match="metric"):
        paired_bootstrap([0, 1], [1, 1], labels, metric=cast(Any, "f1"))


def test_adoption_rule() -> None:
    def ci(low: float, high: float) -> BootstrapCI:
        return BootstrapCI("balanced_accuracy", (low + high) / 2, low, high, 0.95, 100)

    assert adoption_decision(ci(0.001, 0.02)) == "adopt"
    assert adoption_decision(ci(-0.001, 0.02)) == "inconclusive"
    assert adoption_decision(ci(0.0, 0.02)) == "inconclusive"  # must exclude 0
    assert adoption_decision(ci(-0.03, -0.01)) == "worse"


def test_screen_rule_only_drops() -> None:
    assert screen_decision(0.94, 0.90) == "drop"  # exactly 4 points (float noise rounded)
    assert screen_decision(0.94, 0.89) == "drop"
    assert screen_decision(0.94, 0.901) == "keep"
    assert screen_decision(0.90, 0.99) == "keep"  # never "adopt"
