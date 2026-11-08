"""The frozen-feature probe runner of phase 5 (task 5.4): one linear probe per label subset.

Given stored embeddings ``h`` of shape ``(N, d)`` (one row per pool index or alert-table
row: the rows of :mod:`astro_simclr.protocol.label_budgets`), the labels and the resolved
subsets (:func:`~astro_simclr.protocol.label_budgets.resolve`), :func:`run_probes` fits one
probe per subset and scores it on the evaluation rows. Every arm (SSL, random init, ...) is
probed on the same subsets, so :func:`paired_predictions` hands two arms' predictions to
:func:`~astro_simclr.protocol.stats.hierarchical_paired_bootstrap` paired by subset.

**The probe** (:func:`fit_probe`): multinomial logistic regression, plain cross-entropy on
the logits (no softmax before the loss: the legacy linear evaluation's double softmax, the
SimCLR review's finding A3), with an L2 penalty ``weight_decay / 2 x ||W||^2`` on the
weights (not the bias). The features are standardised with the labelled rows' mean and
standard deviation (a constant feature is divided by 1). It is fitted by full-batch L-BFGS
(strong Wolfe line search) from zero weights, in float64 by default: the problem is convex,
so the fit needs no seed, learning rate or epoch count, and it is **deterministic** for a
given device and dtype (on a GPU, up to that device's reductions). This is the objective of
the task-5.0b pilot's probe (``simclr/probe.py``), on any device.

**Weight decay.** With one value, that value is used (a fixed, declared hyperparameter: the
same for every arm). With a grid, each subset's probe is chosen on that subset's
**selection set** (balanced accuracy; the first best in grid order), never on the
evaluation rows; a subset without a selection set (``selection_fraction = 0``) then is an
error.

Speed: the embeddings are moved to the device once; a fit at 1,000 objects per class (about
5,000 rows, d = 512) takes about a second on a loaded laptop CPU (tens of L-BFGS
iterations), so 20+ subsets x 3 budgets x a 4-value grid stay in minutes.

This module imports torch; the rest of :mod:`astro_simclr.protocol` does not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from astro_simclr.protocol.benchmark_metrics import (
    confusion_counts,
    macro_f1_from_counts,
    reweighted_counts,
)
from astro_simclr.protocol.label_budgets import BudgetSubset

__all__ = [
    "DEFAULT_WEIGHT_DECAY",
    "LinearProbe",
    "ProbeResult",
    "fit_probe",
    "paired_predictions",
    "run_probes",
    "summarise",
]

#: The declared default L2 penalty (per standardised feature).
DEFAULT_WEIGHT_DECAY: Final = 1e-3
#: L-BFGS iterations of a fit (the convex fit converges well before on these sizes).
_MAX_ITER: Final = 500


@dataclass(frozen=True, slots=True)
class LinearProbe:
    """A fitted probe: standardise, then ``logits = x @ weight.T + bias``."""

    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """(N, C) logits of (N, d) features, on the probe's device and dtype."""
        x = (h.to(self.weight.device, self.weight.dtype) - self.mean) / self.std
        return x @ self.weight.T + self.bias

    def predict(self, h: torch.Tensor) -> npt.NDArray[np.int64]:
        """(N,) predicted classes (argmax of the logits), as a CPU numpy array."""
        pred: npt.NDArray[np.int64] = self.logits(h).argmax(dim=1).cpu().numpy()
        return pred.astype(np.int64)


def fit_probe(
    h: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_classes: int,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_iter: int = _MAX_ITER,
) -> LinearProbe:
    """Fit the probe (module docstring) on ``h`` (N, d) and integer ``labels`` (N,).

    The fit runs on ``h``'s device and in its dtype (float64 recommended). The objective is
    ``mean CE(logits, y) + weight_decay / 2 x ||W||^2``.

    Raises:
        ValueError: for mismatched shapes, no row, labels outside ``range(n_classes)``, a
            negative ``weight_decay`` or ``max_iter < 1``.
    """
    if h.ndim != 2 or labels.shape != (h.shape[0],) or h.shape[0] < 1:
        raise ValueError(f"need (N, d) h and (N,) labels, got {tuple(h.shape)}, {labels.shape}")
    if int(labels.min()) < 0 or int(labels.max()) >= n_classes:
        raise ValueError(f"labels must be in range({n_classes})")
    if weight_decay < 0 or max_iter < 1:
        raise ValueError(
            f"need weight_decay >= 0 and max_iter >= 1, got {weight_decay}, {max_iter}"
        )
    x = h.detach()
    mean = x.mean(dim=0)
    std = x.std(dim=0, correction=0)
    std = torch.where(std > 0, std, torch.ones_like(std))
    x = (x - mean) / std
    y = labels.to(x.device, torch.int64)
    weight = torch.zeros(n_classes, x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    bias = torch.zeros(n_classes, dtype=x.dtype, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure() -> float:
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ weight.T + bias, y) + 0.5 * weight_decay * weight.pow(2).sum()
        torch.autograd.backward(loss)
        return float(loss.detach())

    # LBFGS.step is untyped in torch; the cast states the documented call.
    cast(Any, optimizer).step(closure)
    return LinearProbe(mean, std, weight.detach().clone(), bias.detach().clone())


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One subset's probe: the chosen penalty, its scores, the evaluation predictions.

    ``selection_balanced_accuracy`` is ``None`` when no grid was searched; ``metrics`` holds
    ``balanced_accuracy``, ``macro_f1`` and, with reference counts,
    ``macro_f1_reweighted``, on the evaluation rows; ``predicted`` are the predicted labels
    of the evaluation rows, in their given order.
    """

    part: str
    repeat: int
    budget: int | None
    n_labelled: int
    weight_decay: float
    selection_balanced_accuracy: float | None
    metrics: dict[str, float]
    predicted: npt.NDArray[np.int64]


def _balanced_accuracy(counts: npt.NDArray[np.int64]) -> float:
    support = counts.sum(axis=1)
    present = support > 0
    return float((np.diag(counts)[present] / support[present]).mean())


def _scores(
    truth: npt.NDArray[np.int64],
    predicted: npt.NDArray[np.int64],
    n_classes: int,
    reference_counts: Sequence[float] | None,
) -> dict[str, float]:
    counts = confusion_counts(truth, predicted, n_classes)
    scores = {
        "balanced_accuracy": _balanced_accuracy(counts),
        "macro_f1": float(macro_f1_from_counts(counts)),
    }
    if reference_counts is not None:
        reweighted = reweighted_counts(counts, reference_counts)
        scores["macro_f1_reweighted"] = float(macro_f1_from_counts(reweighted))
    return scores


def _check_grid(grid: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(v) for v in grid)
    if not values or any(not np.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"the weight-decay grid must be non-empty, finite and >= 0: {grid}")
    return values


def _choose(
    fits: Sequence[LinearProbe],
    h: torch.Tensor,
    truth: npt.NDArray[np.int64],
    selection: npt.NDArray[np.int64],
    n_classes: int,
) -> tuple[int, float]:
    """The first probe with the best selection-set balanced accuracy, and that score."""
    index = torch.from_numpy(selection).to(h.device)
    scores = [
        _balanced_accuracy(confusion_counts(truth[selection], probe.predict(h[index]), n_classes))
        for probe in fits
    ]
    best = int(np.argmax(scores))
    return best, scores[best]


def _inputs(
    embeddings: npt.ArrayLike | torch.Tensor,
    labels: npt.ArrayLike,
    n_classes: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, npt.NDArray[np.int64]]:
    """The embeddings on the device, in ``dtype``, and the labels, both checked."""
    h = torch.as_tensor(embeddings).to(device=device, dtype=dtype)
    truth = np.asarray(labels, dtype=np.int64)
    if h.ndim != 2 or truth.shape != (h.shape[0],):
        raise ValueError(f"need (N, d) embeddings and (N,) labels, got {tuple(h.shape)}")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("the embeddings hold NaN or infinite values")
    if truth.size and (truth.min() < 0 or truth.max() >= n_classes):
        raise ValueError(f"labels must be in range({n_classes})")
    return h, truth


def _rows(values: npt.ArrayLike, n: int, what: str) -> npt.NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or (array.size and (array.min() < 0 or array.max() >= n)):
        raise ValueError(f"{what}: rows must be a 1-D array in range({n})")
    return array


def run_probes(
    embeddings: npt.ArrayLike | torch.Tensor,
    labels: npt.ArrayLike,
    subsets: Sequence[BudgetSubset],
    evaluation: Mapping[str, npt.ArrayLike],
    *,
    n_classes: int,
    weight_decay: Sequence[float] = (DEFAULT_WEIGHT_DECAY,),
    reference_counts: Sequence[float] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    max_iter: int = _MAX_ITER,
) -> list[ProbeResult]:
    """Fit and score one probe per subset (module docstring), in the order of ``subsets``.

    ``embeddings`` (N, d) and ``labels`` (N,) are indexed by the subsets' rows;
    ``evaluation`` maps each part name to its evaluation rows (a CV fold's evaluation part,
    or R5 Test). ``weight_decay`` is one value or a grid chosen on each subset's selection
    set. Deterministic for given inputs, device and dtype.

    Raises:
        ValueError: on embeddings that are not (N, d) finite numbers, labels of another
            length or outside ``range(n_classes)``, rows outside ``range(N)``, a subset
            without labelled rows, a part without evaluation rows, an invalid grid, or a
            grid with a subset that has no selection set.
    """
    grid = _check_grid(weight_decay)
    h, truth = _inputs(embeddings, labels, n_classes, device, dtype)
    y = torch.from_numpy(truth).to(h.device)
    n = truth.size

    def rows(values: npt.ArrayLike, what: str) -> npt.NDArray[np.int64]:
        return _rows(values, n, what)

    eval_rows = {part: rows(r, f"evaluation[{part}]") for part, r in evaluation.items()}
    results = []
    for subset in subsets:
        where = f"part {subset.part}, repeat {subset.repeat}, budget {subset.budget}"
        labelled = rows(subset.labelled, where)
        if labelled.size == 0:
            raise ValueError(f"{where}: no labelled row")
        if subset.part not in eval_rows or eval_rows[subset.part].size == 0:
            raise ValueError(f"{where}: no evaluation rows for part {subset.part!r}")
        index = torch.from_numpy(labelled).to(h.device)
        fits = [
            fit_probe(h[index], y[index], n_classes=n_classes, weight_decay=wd, max_iter=max_iter)
            for wd in grid
        ]
        chosen, selection_score = 0, None
        if len(grid) > 1:
            selection = rows(subset.selection, where)
            if selection.size == 0:
                raise ValueError(f"{where}: a weight-decay grid needs a selection set")
            chosen, selection_score = _choose(fits, h, truth, selection, n_classes)
        target = eval_rows[subset.part]
        predicted = fits[chosen].predict(h[torch.from_numpy(target).to(h.device)])
        results.append(
            ProbeResult(
                subset.part,
                subset.repeat,
                subset.budget,
                int(labelled.size),
                grid[chosen],
                selection_score,
                _scores(truth[target], predicted, n_classes, reference_counts),
                predicted,
            )
        )
    return results


def paired_predictions(
    baseline: Sequence[ProbeResult],
    variant: Sequence[ProbeResult],
    *,
    part: str,
    budget: int | None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The two arms' ``(R, N_eval)`` predictions at one part and budget, paired by repeat.

    The input of :func:`~astro_simclr.protocol.stats.hierarchical_paired_bootstrap`.

    Raises:
        ValueError: if an arm has no result there, or the arms' repeats differ.
    """

    def stack(results: Sequence[ProbeResult], side: str) -> dict[int, npt.NDArray[np.int64]]:
        found = {r.repeat: r.predicted for r in results if r.part == part and r.budget == budget}
        if not found:
            raise ValueError(f"{side}: no probe at part {part!r}, budget {budget}")
        return found

    left, right = stack(baseline, "baseline"), stack(variant, "variant")
    if sorted(left) != sorted(right):
        raise ValueError(f"the arms' repeats differ: {sorted(left)} vs {sorted(right)}")
    order = sorted(left)
    return np.stack([left[r] for r in order]), np.stack([right[r] for r in order])


def summarise(results: Sequence[ProbeResult]) -> list[dict[str, Any]]:
    """Per part and budget (in first-seen order): the subsets' mean and std of each metric.

    The standard deviation is over subsets (``ddof=1``; 0 for one subset): the subset
    variance that plan review F4 asked to report.
    """
    cells: dict[tuple[str, int | None], list[ProbeResult]] = {}
    for result in results:
        cells.setdefault((result.part, result.budget), []).append(result)
    table = []
    for (part, budget), group in cells.items():
        row: dict[str, Any] = {"part": part, "budget": budget, "n_subsets": len(group)}
        for name in group[0].metrics:
            values = np.array([r.metrics[name] for r in group])
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        table.append(row)
    return table
