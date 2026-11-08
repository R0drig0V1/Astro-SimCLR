"""Tests of the phase-5 frozen-feature probe runner (``protocol.probe``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astro_simclr.protocol.benchmark_metrics import confusion_counts, macro_f1
from astro_simclr.protocol.label_budgets import BudgetPart, make_label_budgets, resolve
from astro_simclr.protocol.probe import (
    fit_probe,
    paired_predictions,
    run_probes,
    summarise,
)
from astro_simclr.protocol.stats import hierarchical_paired_bootstrap

N_CLASSES = 3
DIM = 16


def _world(signal: float = 2.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, BudgetPart]:
    """Embeddings of 300 rows (rows 0-209: the training part; 210-299: evaluation)."""
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(N_CLASSES), 100)
    centres = rng.normal(size=(N_CLASSES, DIM))
    h = rng.normal(size=(labels.size, DIM)) + signal * centres[labels]
    rows = np.arange(210, dtype=np.int64)
    part = BudgetPart("fold0", rows, rows.copy(), labels[rows], rows[:30])
    return h, labels, part


def _subsets(part: BudgetPart, **kwargs: object) -> list:  # type: ignore[type-arg]
    budgets = make_label_budgets(
        [part],
        source={"kind": "cv-splits", "sha256": "", "fingerprint": {}},
        label_names=("A", "B", "C"),
        object_id="row",
        budgets=(5, 20),
        n_repeats=4,
        **kwargs,  # type: ignore[arg-type]
    )
    return resolve(budgets, part)


EVAL = {"fold0": np.arange(210, 300)}


def test_probe_learns_and_is_deterministic() -> None:
    h, labels, part = _world()
    subsets = _subsets(part)
    first = run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES)
    again = run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES)
    assert len(first) == len(subsets) == 2 * 4 + 1
    for a, b in zip(first, again, strict=True):
        assert np.array_equal(a.predicted, b.predicted)
        assert a.metrics == b.metrics
    for result, subset in zip(first, subsets, strict=True):
        assert result.n_labelled == subset.labelled.size
        assert result.selection_balanced_accuracy is None
        truth = labels[EVAL["fold0"]]
        assert result.metrics["macro_f1"] == pytest.approx(
            macro_f1(truth, result.predicted, N_CLASSES)
        )
        assert result.metrics["balanced_accuracy"] > 0.8
    full = first[-1]
    assert full.budget is None and full.n_labelled == 210


def test_plain_cross_entropy_objective_at_the_optimum() -> None:
    """The fit is a stationary point of mean CE + wd/2 ||W||^2 (bias unpenalised)."""
    h, labels, _ = _world(signal=0.5)
    x, y = torch.from_numpy(h[:60]), torch.from_numpy(labels[:60])
    probe = fit_probe(x, y, n_classes=N_CLASSES, weight_decay=0.1)
    weight = probe.weight.clone().requires_grad_(True)
    bias = probe.bias.clone().requires_grad_(True)
    z = (x - probe.mean) / probe.std
    loss = torch.nn.functional.cross_entropy(z @ weight.T + bias, y) + 0.05 * weight.pow(2).sum()
    torch.autograd.backward(loss)
    assert weight.grad is not None and bias.grad is not None
    assert float(weight.grad.abs().max()) < 1e-6
    assert float(bias.grad.abs().max()) < 1e-6
    assert probe.weight.dtype == torch.float64


def test_constant_feature_is_kept_finite() -> None:
    h, labels, _ = _world()
    h[:, 0] = 3.0
    probe = fit_probe(torch.from_numpy(h), torch.from_numpy(labels), n_classes=N_CLASSES)
    assert float(probe.std[0]) == 1.0
    assert torch.isfinite(probe.weight).all()


def test_grid_is_chosen_on_the_selection_set() -> None:
    h, labels, part = _world(signal=0.6)
    subsets = [s for s in _subsets(part) if s.budget == 20]
    grid = (1e-4, 1.0, 100.0)
    results = run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES, weight_decay=grid)
    for result, subset in zip(results, subsets, strict=True):
        assert result.weight_decay in grid
        scores = []
        for wd in grid:
            probe = fit_probe(
                torch.from_numpy(h[subset.labelled]),
                torch.from_numpy(labels[subset.labelled]),
                n_classes=N_CLASSES,
                weight_decay=wd,
            )
            pred = probe.predict(torch.from_numpy(h[subset.selection]))
            counts = confusion_counts(labels[subset.selection], pred, N_CLASSES)
            scores.append(float(np.mean(np.diag(counts) / counts.sum(axis=1))))
        assert result.weight_decay == grid[int(np.argmax(scores))]
        assert result.selection_balanced_accuracy == pytest.approx(max(scores))


def test_grid_needs_a_selection_set() -> None:
    h, labels, part = _world()
    subsets = _subsets(part, selection_fraction=0.0)
    with pytest.raises(ValueError, match="needs a selection set"):
        run_probes(h, labels, subsets[:1], EVAL, n_classes=N_CLASSES, weight_decay=(0.1, 1.0))


def test_reweighted_metric_and_float32() -> None:
    h, labels, part = _world()
    subsets = _subsets(part)[:2]
    results = run_probes(
        h,
        labels,
        subsets,
        EVAL,
        n_classes=N_CLASSES,
        reference_counts=(1.0, 2.0, 3.0),
        dtype=torch.float32,
    )
    assert "macro_f1_reweighted" in results[0].metrics


def test_pairing_and_the_hierarchical_interval() -> None:
    h, labels, part = _world(signal=0.8)
    subsets = _subsets(part)
    weak = h + np.random.default_rng(9).normal(scale=3.0, size=h.shape)
    strong = run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES)
    noisy = run_probes(weak, labels, subsets, EVAL, n_classes=N_CLASSES)
    base, variant = paired_predictions(noisy, strong, part="fold0", budget=20)
    assert base.shape == variant.shape == (4, 90)
    ci = hierarchical_paired_bootstrap(
        base, variant, labels[EVAL["fold0"]], n_classes=N_CLASSES, n_boot=500
    )
    assert ci.low > 0  # the clean embeddings beat the noisy ones
    full = paired_predictions(noisy, strong, part="fold0", budget=None)
    assert full[0].shape == (1, 90)
    with pytest.raises(ValueError, match="no probe"):
        paired_predictions(noisy, strong, part="fold1", budget=20)
    with pytest.raises(ValueError, match="repeats differ"):
        paired_predictions(noisy[:1], strong, part="fold0", budget=5)


def test_summarise_reports_subset_spread() -> None:
    h, labels, part = _world(signal=0.5)
    results = run_probes(h, labels, _subsets(part), EVAL, n_classes=N_CLASSES)
    table = summarise(results)
    assert [(row["budget"], row["n_subsets"]) for row in table] == [(5, 4), (20, 4), (None, 1)]
    values = [r.metrics["macro_f1"] for r in results if r.budget == 5]
    assert table[0]["macro_f1_mean"] == pytest.approx(np.mean(values))
    assert table[0]["macro_f1_std"] == pytest.approx(np.std(values, ddof=1))
    assert table[2]["balanced_accuracy_std"] == 0.0


def test_refuses_bad_inputs() -> None:
    h, labels, part = _world()
    subsets = _subsets(part)[:1]
    with pytest.raises(ValueError, match="grid"):
        run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES, weight_decay=())
    with pytest.raises(ValueError, match="grid"):
        run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES, weight_decay=(-1.0,))
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        run_probes(h[:, 0], labels, subsets, EVAL, n_classes=N_CLASSES)
    bad = h.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_probes(bad, labels, subsets, EVAL, n_classes=N_CLASSES)
    with pytest.raises(ValueError, match="labels must be"):
        run_probes(h, labels + 3, subsets, EVAL, n_classes=N_CLASSES)
    with pytest.raises(ValueError, match="rows must be"):
        run_probes(h, labels, subsets, {"fold0": np.array([400])}, n_classes=N_CLASSES)
    with pytest.raises(ValueError, match="no evaluation rows"):
        run_probes(h, labels, subsets, {}, n_classes=N_CLASSES)
    empty = type(subsets[0])("fold0", 0, 5, np.empty(0, np.int64), np.empty(0, np.int64))
    with pytest.raises(ValueError, match="no labelled row"):
        run_probes(h, labels, [empty], EVAL, n_classes=N_CLASSES)
    x, y = torch.from_numpy(h), torch.from_numpy(labels)
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        fit_probe(x, y[:-1], n_classes=N_CLASSES)
    with pytest.raises(ValueError, match="range"):
        fit_probe(x, y + 5, n_classes=N_CLASSES)
    with pytest.raises(ValueError, match="weight_decay"):
        fit_probe(x, y, n_classes=N_CLASSES, weight_decay=-1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_matches_cpu() -> None:  # pragma: no cover - CPU CI
    h, labels, part = _world()
    subsets = _subsets(part)[:2]
    cpu = run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES)
    gpu = run_probes(h, labels, subsets, EVAL, n_classes=N_CLASSES, device="cuda")
    for a, b in zip(cpu, gpu, strict=True):
        assert np.mean(a.predicted == b.predicted) > 0.98
