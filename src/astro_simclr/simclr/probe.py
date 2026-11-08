"""The linear probe on precomputed, frozen representations (phase 5, task 5.0b; ADR 0019).

**Label subsets** (:class:`LabelSubsets`): the labelled stamps a probe may train on at a
budget of ``per_class`` stamps per class, drawn once and stored in a file
(:func:`write_label_subsets`), so every arm (SSL, random init, scratch) trains on the same
subsets and the comparison is paired by subset. Subset ``r`` takes, for every class, the
first ``per_class`` stamps of a permutation of that class's candidates drawn from
``default_rng([seed, r, class])``: the permutation does not depend on ``per_class``, so the
budgets are **nested** (subset ``r`` at 10 per class is inside subset ``r`` at 100). The
candidates are the Train stamps a screened run may use (``ProtocolSplits.
screen_training_excluded`` left out), and the file records the SHA-256 of the candidate
indices and of the labels, so a file is refused on other data.

**The probe** (:func:`fit_probe`): multinomial logistic regression on ``h``, plain
cross-entropy on the logits (no double softmax: the legacy linear evaluation's
``softmax`` before ``P_stamps_loss`` compressed the loss, the SimCLR review's finding A3),
with an L2 penalty ``weight_decay / 2 * ||W||^2`` on the weights (not the bias). The
features are standardised with the training subset's mean and standard deviation. It is
fitted in float64 on the CPU by full-batch L-BFGS (strong Wolfe line search) from zero
weights: the problem is convex, so the fit is deterministic and does not depend on a
learning rate or an epoch count.

**Selection** (:func:`probe_screen`): the penalty is chosen from a declared grid on the
screen's **selection** half (balanced accuracy; the first best in grid order), and only
the chosen probe is scored on the **decision** half (task 5.0b evaluates on the CD21 screen
halves only).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from astro_simclr.data.schema import LABEL_NAMES, DataError
from astro_simclr.protocol.stats import balanced_accuracy

__all__ = [
    "SUBSETS_FORMAT",
    "SUBSETS_VERSION",
    "WEIGHT_DECAY_GRID",
    "LabelSubsets",
    "LinearProbe",
    "array_sha256",
    "draw_label_subsets",
    "fit_probe",
    "probe_screen",
    "read_label_subsets",
    "write_label_subsets",
]

#: ``format`` of a label-subsets file.
SUBSETS_FORMAT: Final = "astro-simclr-label-subsets"
#: Its version.
SUBSETS_VERSION: Final = 1
#: The declared L2 grid of the probe (``weight_decay``, per standardised feature).
WEIGHT_DECAY_GRID: Final = (1e-4, 1e-3, 1e-2, 1e-1)
#: L-BFGS iterations of a probe fit (the convex fit converges well before on these sizes).
_MAX_ITER: Final = 500


def array_sha256(values: npt.ArrayLike) -> str:
    """SHA-256 of an array's int64 little-endian bytes (indices or labels)."""
    data = np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(data.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class LabelSubsets:
    """Stored label subsets of one budget (module docstring).

    Attributes:
        per_class: stamps per class in each subset.
        seed: the draw's seed.
        n_classes: the classes (each subset holds ``per_class`` of each).
        candidates_sha256: :func:`array_sha256` of the sorted candidate Train indices.
        labels_sha256: :func:`array_sha256` of the whole Train label array.
        subsets: one sorted int64 array of Train indices per subset.
    """

    per_class: int
    seed: int
    n_classes: int
    candidates_sha256: str
    labels_sha256: str
    subsets: tuple[npt.NDArray[np.int64], ...]


def draw_label_subsets(
    labels: npt.NDArray[np.int64],
    candidates: npt.NDArray[np.int64],
    *,
    per_class: int,
    n_subsets: int,
    seed: int,
    n_classes: int = len(LABEL_NAMES),
) -> LabelSubsets:
    """Draw ``n_subsets`` nested label subsets of ``per_class`` stamps per class.

    ``labels`` is the whole Train label array, ``candidates`` the Train indices a subset
    may use (any order; duplicates are an error).

    Raises:
        ValueError: if ``per_class`` or ``n_subsets`` is below 1, a candidate is out of
            range or repeated, or a class has fewer than ``per_class`` candidates.
    """
    if per_class < 1 or n_subsets < 1:
        raise ValueError(f"need per_class >= 1 and n_subsets >= 1, got {per_class}, {n_subsets}")
    pool = np.sort(np.asarray(candidates, dtype=np.int64))
    if pool.size and (pool[0] < 0 or pool[-1] >= labels.size):
        raise ValueError(f"candidates must be Train indices in range({labels.size})")
    if np.unique(pool).size != pool.size:
        raise ValueError("candidates repeat an index")
    by_class = [pool[labels[pool] == c] for c in range(n_classes)]
    short = [
        LABEL_NAMES[c] if c < len(LABEL_NAMES) else str(c)
        for c, rows in enumerate(by_class)
        if rows.size < per_class
    ]
    if short:
        raise ValueError(f"fewer than {per_class} candidates in class(es) {', '.join(short)}")
    subsets = []
    for r in range(n_subsets):
        chosen = [
            np.random.default_rng([seed, r, c]).permutation(rows)[:per_class]
            for c, rows in enumerate(by_class)
        ]
        subsets.append(np.sort(np.concatenate(chosen)))
    return LabelSubsets(
        per_class, seed, n_classes, array_sha256(pool), array_sha256(labels), tuple(subsets)
    )


def write_label_subsets(subsets: LabelSubsets, path: Path) -> None:
    """Write ``subsets`` as JSON at ``path`` (the parent directory must exist)."""
    plain = {
        "format": SUBSETS_FORMAT,
        "format_version": SUBSETS_VERSION,
        "split": "Train",
        "per_class": subsets.per_class,
        "seed": subsets.seed,
        "n_classes": subsets.n_classes,
        "candidates_sha256": subsets.candidates_sha256,
        "labels_sha256": subsets.labels_sha256,
        "subsets": [s.tolist() for s in subsets.subsets],
    }
    path.write_text(json.dumps(plain, indent=1) + "\n")


def _check_labels(subsets: LabelSubsets, labels: npt.NDArray[np.int64], path: Path) -> None:
    """Refuse ``subsets`` unless they were drawn from ``labels``, ``per_class`` per class."""
    if array_sha256(labels) != subsets.labels_sha256:
        raise DataError(f"{path}: made for another Train label array")
    for s in subsets.subsets:
        if s.min() < 0 or s.max() >= labels.size:
            raise DataError(f"{path}: a subset holds an index outside range({labels.size})")
        if not np.all(np.bincount(labels[s], minlength=subsets.n_classes) == subsets.per_class):
            raise DataError(f"{path}: a subset does not hold {subsets.per_class} stamps per class")


def read_label_subsets(path: Path, labels: npt.NDArray[np.int64] | None = None) -> LabelSubsets:
    """Read and check a label-subsets file; with ``labels``, check it belongs to them.

    Raises:
        DataError: if the file is unreadable, of another format or version, malformed (a
            subset that is not ``per_class`` per class of distinct indices), or, with
            ``labels``, made for another label array.
    """
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise DataError(f"{path}: cannot read label subsets: {err}") from err
    if not isinstance(raw, dict) or raw.get("format") != SUBSETS_FORMAT:
        raise DataError(f"{path}: not a {SUBSETS_FORMAT} file")
    if raw.get("format_version") != SUBSETS_VERSION:
        raise DataError(
            f"{path}: version {raw.get('format_version')!r}, expected {SUBSETS_VERSION}"
        )
    try:
        per_class, seed, n_classes = int(raw["per_class"]), int(raw["seed"]), int(raw["n_classes"])
        subsets = tuple(np.asarray(s, dtype=np.int64) for s in raw["subsets"])
        result = LabelSubsets(
            per_class,
            seed,
            n_classes,
            str(raw["candidates_sha256"]),
            str(raw["labels_sha256"]),
            subsets,
        )
    except (KeyError, TypeError, ValueError) as err:
        raise DataError(f"{path}: malformed label subsets: {err}") from err
    if per_class < 1 or n_classes < 1 or not subsets:
        raise DataError(f"{path}: malformed label subsets: no subset, or a size below 1")
    for s in subsets:
        if s.ndim != 1 or s.size != per_class * n_classes or np.unique(s).size != s.size:
            raise DataError(f"{path}: a subset is not {per_class} distinct stamps per class")
    if labels is not None:
        _check_labels(result, labels, path)
    return result


@dataclass(frozen=True, slots=True)
class LinearProbe:
    """A fitted probe: standardisation, then ``logits = x @ weight.T + bias`` (float64, CPU)."""

    mean: torch.Tensor
    std: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """(N, C) float64 logits of (N, D) representations."""
        x = (h.detach().to("cpu", torch.float64) - self.mean) / self.std
        return x @ self.weight.T + self.bias

    def predict(self, h: torch.Tensor) -> npt.NDArray[np.int64]:
        """(N,) int64 predicted classes (argmax of the logits)."""
        pred: npt.NDArray[np.int64] = self.logits(h).argmax(dim=1).numpy().astype(np.int64)
        return pred

    def balanced_accuracy(self, h: torch.Tensor, labels: npt.NDArray[np.int64]) -> float:
        """Balanced accuracy on labelled representations."""
        correct = (self.predict(h) == labels).astype(np.float64)
        return balanced_accuracy(correct, labels)


def fit_probe(
    h: torch.Tensor,
    labels: npt.NDArray[np.int64],
    *,
    n_classes: int = len(LABEL_NAMES),
    weight_decay: float = 1e-3,
    max_iter: int = _MAX_ITER,
    standardise: bool = True,
) -> LinearProbe:
    """Fit the logistic-regression probe on (N, D) ``h`` (module docstring).

    The objective is ``mean CE(logits, y) + weight_decay / 2 * ||W||^2``. A feature with
    zero spread is divided by 1 (it stays constant). Deterministic for given inputs.

    Raises:
        ValueError: for mismatched shapes, labels outside ``range(n_classes)``, a negative
            ``weight_decay`` or ``max_iter < 1``.
    """
    if h.ndim != 2 or labels.shape != (h.shape[0],) or h.shape[0] < 1:
        raise ValueError(f"need (N, D) h and (N,) labels, got {tuple(h.shape)}, {labels.shape}")
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(f"labels must be in range({n_classes})")
    if weight_decay < 0 or max_iter < 1:
        raise ValueError(
            f"need weight_decay >= 0 and max_iter >= 1, got {weight_decay}, {max_iter}"
        )
    x = h.detach().to("cpu", torch.float64)
    dim = x.shape[1]
    if standardise:
        mean = x.mean(dim=0)
        std = x.std(dim=0, correction=0)
        std = torch.where(std > 0, std, torch.ones_like(std))
    else:
        mean, std = torch.zeros(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)
    x = (x - mean) / std
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    weight = torch.zeros(n_classes, dim, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
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

    # LBFGS.step is untyped in torch 2.14; the cast states the documented call.
    cast(Any, optimizer).step(closure)
    return LinearProbe(mean, std, weight.detach().clone(), bias.detach().clone())


def probe_screen(
    train_h: torch.Tensor,
    train_labels: npt.NDArray[np.int64],
    subsets: LabelSubsets,
    selection: tuple[torch.Tensor, npt.NDArray[np.int64]],
    decision: tuple[torch.Tensor, npt.NDArray[np.int64]],
    *,
    grid: Sequence[float] = WEIGHT_DECAY_GRID,
) -> dict[str, Any]:
    """Fit, select and score the probe on every subset (module docstring, "Selection").

    ``train_h`` (N_train, D) holds the representations of every Train stamp in file order;
    ``selection`` and ``decision`` are ``(h, labels)`` of the two screen halves.
    Returns ``{"per_class", "grid", "subsets": [{"subset", "weight_decay",
    "selection_bacc", "decision_bacc"}], "decision_bacc_mean", "decision_bacc_std"}``
    (the standard deviation over subsets, ``ddof=1``; 0 for one subset).

    Raises:
        ValueError: for an empty grid, or as :func:`fit_probe`.
    """
    if not grid:
        raise ValueError("the weight-decay grid is empty")
    rows: list[dict[str, Any]] = []
    for r, subset in enumerate(subsets.subsets):
        best: tuple[float, float, LinearProbe] | None = None
        for wd in grid:
            probe = fit_probe(
                train_h[torch.from_numpy(subset)],
                train_labels[subset],
                n_classes=subsets.n_classes,
                weight_decay=wd,
            )
            score = probe.balanced_accuracy(*selection)
            if best is None or score > best[0]:
                best = (score, wd, probe)
        assert best is not None  # noqa: S101 - the grid is not empty
        rows.append(
            {
                "subset": r,
                "weight_decay": best[1],
                "selection_bacc": best[0],
                "decision_bacc": best[2].balanced_accuracy(*decision),
            }
        )
    scores = np.array([row["decision_bacc"] for row in rows])
    return {
        "per_class": subsets.per_class,
        "grid": list(grid),
        "subsets": rows,
        "decision_bacc_mean": float(scores.mean()),
        "decision_bacc_std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
    }
