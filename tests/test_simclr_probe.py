"""Tests of the probe (label subsets, logistic regression) and the embedding diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from astro_simclr.data.schema import DataError
from astro_simclr.simclr.diagnostics import (
    alignment,
    dimension_std,
    effective_rank,
    knn_balanced_accuracy,
    knn_predict,
    uniformity,
)
from astro_simclr.simclr.probe import (
    WEIGHT_DECAY_GRID,
    array_sha256,
    draw_label_subsets,
    fit_probe,
    probe_screen,
    read_label_subsets,
    write_label_subsets,
)

LABELS = np.repeat(np.arange(5, dtype=np.int64), 30)  # 30 stamps per class


def _clusters(
    labels: np.ndarray, dim: int = 16, spread: float = 0.3, seed: int = 0
) -> torch.Tensor:
    """Gaussian clusters, one centre per class."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (5, dim))
    return torch.from_numpy(centres[labels] + spread * rng.normal(0, 1, (labels.size, dim)))


# --- label subsets ------------------------------------------------------------------------


def test_subsets_hold_per_class_stamps_of_the_candidates() -> None:
    candidates = np.arange(1, 150, 2)  # odd indices only
    subsets = draw_label_subsets(LABELS, candidates, per_class=4, n_subsets=3, seed=1)
    assert len(subsets.subsets) == 3 and subsets.per_class == 4
    for s in subsets.subsets:
        assert np.all(np.bincount(LABELS[s], minlength=5) == 4)
        assert np.all(s % 2 == 1) and np.all(np.diff(s) > 0)
    assert not np.array_equal(subsets.subsets[0], subsets.subsets[1])
    assert subsets.candidates_sha256 == array_sha256(np.sort(candidates))
    assert subsets.labels_sha256 == array_sha256(LABELS)


def test_subsets_are_nested_across_budgets_and_deterministic() -> None:
    small = draw_label_subsets(LABELS, np.arange(150), per_class=3, n_subsets=4, seed=7)
    large = draw_label_subsets(LABELS, np.arange(150)[::-1], per_class=10, n_subsets=4, seed=7)
    again = draw_label_subsets(LABELS, np.arange(150), per_class=3, n_subsets=4, seed=7)
    for s, big, same in zip(small.subsets, large.subsets, again.subsets, strict=True):
        assert set(s) <= set(big)
        np.testing.assert_array_equal(s, same)


def test_subsets_refuse_impossible_draws() -> None:
    with pytest.raises(ValueError, match="fewer than 31"):
        draw_label_subsets(LABELS, np.arange(150), per_class=31, n_subsets=1, seed=0)
    with pytest.raises(ValueError, match="repeat"):
        draw_label_subsets(LABELS, np.array([0, 0, 1]), per_class=1, n_subsets=1, seed=0)
    with pytest.raises(ValueError, match="range"):
        draw_label_subsets(LABELS, np.array([150]), per_class=1, n_subsets=1, seed=0)
    with pytest.raises(ValueError, match="per_class"):
        draw_label_subsets(LABELS, np.arange(150), per_class=0, n_subsets=1, seed=0)


def test_subsets_round_trip_and_are_checked_on_read(tmp_path: Path) -> None:
    subsets = draw_label_subsets(LABELS, np.arange(150), per_class=2, n_subsets=2, seed=3)
    path = tmp_path / "subsets.json"
    write_label_subsets(subsets, path)
    back = read_label_subsets(path, LABELS)
    assert back.per_class == 2 and back.seed == 3
    for a, b in zip(subsets.subsets, back.subsets, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(DataError, match="another Train label"):
        read_label_subsets(path, LABELS[::-1].copy())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"format": "other"}, "not a"),
        ({"format_version": 9}, "version"),
        ({"per_class": "x"}, "malformed"),
        ({"subsets": []}, "malformed"),
        ({"subsets": [[0, 0, 30, 60, 90, 120, 1, 31, 61, 91]]}, "distinct"),
    ],
)
def test_read_label_subsets_refuses_bad_files(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    subsets = draw_label_subsets(LABELS, np.arange(150), per_class=2, n_subsets=1, seed=0)
    path = tmp_path / "subsets.json"
    write_label_subsets(subsets, path)
    raw = json.loads(path.read_text())
    raw.update(change)
    path.write_text(json.dumps(raw))
    with pytest.raises(DataError, match=message):
        read_label_subsets(path)


def test_read_label_subsets_checks_the_classes_and_range(tmp_path: Path) -> None:
    subsets = draw_label_subsets(LABELS, np.arange(150), per_class=1, n_subsets=1, seed=0)
    path = tmp_path / "subsets.json"
    write_label_subsets(subsets, path)
    raw = json.loads(path.read_text())
    raw["subsets"] = [[0, 1, 2, 3, 4]]  # all of class 0
    path.write_text(json.dumps(raw))
    with pytest.raises(DataError, match="1 stamps per class"):
        read_label_subsets(path, LABELS)
    raw["subsets"] = [[0, 30, 60, 90, 999]]
    path.write_text(json.dumps(raw))
    with pytest.raises(DataError, match="outside"):
        read_label_subsets(path, LABELS)
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(DataError, match="cannot read"):
        read_label_subsets(tmp_path / "bad.json")


# --- probe --------------------------------------------------------------------------------


def test_probe_separates_clusters_and_is_deterministic() -> None:
    h = _clusters(LABELS)
    probe = fit_probe(h, LABELS, weight_decay=1e-3)
    assert probe.balanced_accuracy(h, LABELS) == 1.0
    again = fit_probe(h, LABELS, weight_decay=1e-3)
    assert torch.equal(probe.weight, again.weight) and torch.equal(probe.bias, again.bias)
    assert probe.logits(h.float()).dtype == torch.float64


def test_probe_reaches_the_optimum_of_its_objective() -> None:
    h = _clusters(LABELS, spread=2.0)  # overlapping: the optimum is interior
    wd = 0.1
    probe = fit_probe(h, LABELS, weight_decay=wd)
    weight = probe.weight.clone().requires_grad_(True)
    bias = probe.bias.clone().requires_grad_(True)
    x = (h - probe.mean) / probe.std
    loss = (
        F.cross_entropy(x @ weight.T + bias, torch.from_numpy(LABELS))
        + 0.5 * wd * weight.pow(2).sum()
    )
    torch.autograd.backward(loss)
    assert weight.grad is not None and bias.grad is not None
    assert float(weight.grad.abs().max()) < 1e-5 and float(bias.grad.abs().max()) < 1e-5


def test_probe_standardisation_and_constant_features() -> None:
    h = _clusters(LABELS)
    h[:, 0] = 3.0  # a constant feature
    probe = fit_probe(h, LABELS)
    assert float(probe.std[0]) == 1.0
    raw = fit_probe(h, LABELS, standardise=False)
    assert torch.equal(raw.mean, torch.zeros(16, dtype=torch.float64))


def test_probe_validates_inputs() -> None:
    h = _clusters(LABELS)
    with pytest.raises(ValueError, match="labels"):
        fit_probe(h, LABELS[:-1])
    with pytest.raises(ValueError, match="range"):
        fit_probe(h, LABELS + 1)
    with pytest.raises(ValueError, match="weight_decay"):
        fit_probe(h, LABELS, weight_decay=-1.0)


def test_probe_screen_selects_on_selection_and_scores_on_decision() -> None:
    train_h = _clusters(LABELS, spread=1.0)
    subsets = draw_label_subsets(LABELS, np.arange(150), per_class=5, n_subsets=3, seed=0)
    val_labels = np.repeat(np.arange(5, dtype=np.int64), 4)
    val_h = _clusters(val_labels, spread=1.0)  # same centres (same seed)
    report = probe_screen(
        train_h, LABELS, subsets, (val_h[:10], val_labels[:10]), (val_h[10:], val_labels[10:])
    )
    assert report["per_class"] == 5 and report["grid"] == list(WEIGHT_DECAY_GRID)
    assert len(report["subsets"]) == 3
    for row in report["subsets"]:
        assert row["weight_decay"] in WEIGHT_DECAY_GRID
        assert 0 <= row["decision_bacc"] <= 1
    scores = [row["decision_bacc"] for row in report["subsets"]]
    assert report["decision_bacc_mean"] == pytest.approx(np.mean(scores))
    one = draw_label_subsets(LABELS, np.arange(150), per_class=5, n_subsets=1, seed=0)
    single = probe_screen(
        train_h, LABELS, one, (val_h, val_labels), (val_h, val_labels), grid=[1.0]
    )
    assert single["decision_bacc_std"] == 0.0
    with pytest.raises(ValueError, match="grid"):
        probe_screen(train_h, LABELS, one, (val_h, val_labels), (val_h, val_labels), grid=[])


# --- diagnostics --------------------------------------------------------------------------


def test_effective_rank_bounds() -> None:
    gen = torch.Generator().manual_seed(0)
    iso = torch.randn(20000, 8, generator=gen)
    assert effective_rank(iso) == pytest.approx(8.0, rel=0.01)
    line = torch.randn(100, 1, generator=gen) * torch.ones(1, 8)
    assert effective_rank(line) == pytest.approx(1.0, abs=1e-6)
    assert effective_rank(torch.ones(10, 4)) == 0.0
    two = torch.zeros(50, 4)
    two[:, :2] = torch.randn(50, 2, generator=gen)
    assert 1.0 < effective_rank(two) <= 2.0 + 1e-9
    with pytest.raises(ValueError, match="N >= 2"):
        effective_rank(torch.ones(1, 4))


def test_dimension_std_detects_collapse() -> None:
    collapsed = torch.ones(10, 4)
    assert torch.all(dimension_std(collapsed) == 0)
    spread = torch.randn(20000, 64, generator=torch.Generator().manual_seed(1))
    assert float(dimension_std(spread).mean()) == pytest.approx(1 / math.sqrt(64), rel=0.02)


def test_alignment_and_uniformity_formulas() -> None:
    z = torch.randn(6, 3, generator=torch.Generator().manual_seed(2))
    assert alignment(z, 3 * z) == pytest.approx(0.0, abs=1e-12)
    assert alignment(z, -z) == pytest.approx(4.0)
    n = F.normalize(z.double(), dim=1)
    pairs = [
        math.exp(-2.0 * float((n[a] - n[b]).pow(2).sum()))
        for a in range(6)
        for b in range(a + 1, 6)
    ]
    assert uniformity(z) == pytest.approx(math.log(sum(pairs) / len(pairs)))
    with pytest.raises(ValueError, match="shape"):
        alignment(z, z[:, :2])


def test_knn_predicts_the_nearest_cluster() -> None:
    bank_labels = np.repeat(np.arange(5), 10)
    bank = _clusters(bank_labels, spread=0.1)
    queries = _clusters(np.arange(5), spread=0.1, seed=0)[:5]  # the same centres
    pred = knn_predict(bank, torch.from_numpy(bank_labels), queries, n_classes=5, k=5)
    assert pred.tolist() == [0, 1, 2, 3, 4]
    # k larger than the bank: every bank row votes.
    many = knn_predict(bank, torch.from_numpy(bank_labels), queries, n_classes=5, k=500)
    assert many.shape == (5,)
    score = knn_balanced_accuracy(
        bank, torch.from_numpy(bank_labels), queries, torch.arange(5), n_classes=5, k=5
    )
    assert score == 1.0


def test_knn_validates_inputs() -> None:
    bank = torch.zeros(3, 2)
    labels = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="shape"):
        knn_predict(bank, labels, torch.zeros(2, 3), n_classes=3)
    with pytest.raises(ValueError, match="range"):
        knn_predict(bank, labels, torch.zeros(1, 2), n_classes=2)
    with pytest.raises(ValueError, match="k >= 1"):
        knn_predict(bank, labels, torch.zeros(1, 2), n_classes=3, k=0)
    with pytest.raises(ValueError, match="one label per query"):
        knn_balanced_accuracy(bank, labels, torch.zeros(2, 2), torch.zeros(3), n_classes=3)
