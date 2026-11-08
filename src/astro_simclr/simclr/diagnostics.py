"""Embedding diagnostics of SimCLR pretraining (plan 5.3; written for the 5.0b pilot).

Pure functions of tensors (no model, no engine), computed in float64 where a sum over many
terms is involved:

- :func:`effective_rank` of ``cov(h)``: ``exp(H(p))``, the entropy of the normalised
  eigenvalue spectrum (Roy and Vetterli 2007). Between 1 (every variance in one direction)
  and ``D`` (an isotropic spread); a falling value is dimensional collapse.
- :func:`dimension_std`: the per-dimension standard deviation of the L2-normalised ``z``.
  Near 0 everywhere is complete collapse; ``1 / sqrt(D)`` is what a spread over the sphere
  gives.
- :func:`alignment` and :func:`uniformity` (Wang and Isola 2020) of normalised embeddings:
  ``E ||f(x) - f(y)||^2`` over positive pairs (lower is better aligned) and ``log E
  exp(-t ||f(x) - f(y)||^2)`` over all distinct pairs (lower is more uniform; -4 is about
  the value of a uniform spread on a high-dimensional sphere for ``t = 2``).
- :func:`knn_predict`: the weighted kNN monitor of instance discrimination (Wu et al. 2018):
  cosine similarity to a labelled bank, the ``k`` nearest, each voting ``exp(sim / T)`` for
  its class. :func:`knn_balanced_accuracy` scores it with ``protocol.stats``'s balanced
  accuracy (the screen halves are balanced, so it equals the accuracy there).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from astro_simclr.protocol.stats import balanced_accuracy

__all__ = [
    "KNN_K",
    "KNN_TEMPERATURE",
    "UNIFORMITY_T",
    "alignment",
    "dimension_std",
    "effective_rank",
    "knn_balanced_accuracy",
    "knn_predict",
    "uniformity",
]

#: Neighbours of the kNN monitor (Wu et al. 2018 use 200 on ImageNet; 20 suits 5 classes
#: and banks of a few thousand).
KNN_K: Final = 20
#: Temperature of the kNN vote weights (Wu et al. 2018).
KNN_TEMPERATURE: Final = 0.07
#: ``t`` of the uniformity (Wang and Isola 2020).
UNIFORMITY_T: Final = 2.0
#: Queries per similarity block of the kNN (bounds the (block, bank) matrix).
_KNN_BLOCK: Final = 1024


def _check_2d(x: torch.Tensor, what: str, minimum: int = 1) -> None:
    if x.ndim != 2 or x.shape[0] < minimum:
        raise ValueError(f"{what} must be (N >= {minimum}, D), got {tuple(x.shape)}")


def effective_rank(h: torch.Tensor) -> float:
    """``exp`` of the entropy of the normalised eigenvalues of the covariance of (N, D) ``h``.

    The covariance is centred and unbiased, in float64; negative eigenvalues (rounding)
    count as 0. A zero covariance (constant ``h``) gives 0.

    Raises:
        ValueError: if ``h`` is not (N >= 2, D).
    """
    _check_2d(h, "h", minimum=2)
    x = h.detach().to(torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    cov = x.T @ x / (x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = float(eig.sum())
    if total <= 0:
        return 0.0
    p = eig[eig > 0] / total
    return math.exp(float(-(p * p.log()).sum()))


def dimension_std(z: torch.Tensor) -> torch.Tensor:
    """(D,) float64 standard deviation over the batch of each dimension of normalised ``z``.

    Raises:
        ValueError: if ``z`` is not (N >= 2, D).
    """
    _check_2d(z, "z", minimum=2)
    return F.normalize(z.detach().to(torch.float64), dim=1).std(dim=0)


def alignment(z_i: torch.Tensor, z_j: torch.Tensor, alpha: float = 2.0) -> float:
    """``mean ||n(z_i) - n(z_j)||^alpha`` over the positive pairs (rows) of normalised views.

    Raises:
        ValueError: if the inputs are not two (N >= 1, D) tensors of the same shape.
    """
    _check_2d(z_i, "z_i")
    if z_i.shape != z_j.shape:
        raise ValueError(f"z_i and z_j differ in shape: {tuple(z_i.shape)}, {tuple(z_j.shape)}")
    a = F.normalize(z_i.detach().to(torch.float64), dim=1)
    b = F.normalize(z_j.detach().to(torch.float64), dim=1)
    return float((a - b).norm(dim=1).pow(alpha).mean())


def uniformity(z: torch.Tensor, t: float = UNIFORMITY_T) -> float:
    """``log mean exp(-t ||n(z_a) - n(z_b)||^2)`` over the distinct pairs ``a < b``.

    Raises:
        ValueError: if ``z`` is not (N >= 2, D).
    """
    _check_2d(z, "z", minimum=2)
    x = F.normalize(z.detach().to(torch.float64), dim=1)
    sq = torch.pdist(x, p=2).pow(2)
    return float(torch.logsumexp(-t * sq, dim=0) - math.log(sq.numel()))


def knn_predict(
    bank: torch.Tensor,
    bank_labels: torch.Tensor,
    queries: torch.Tensor,
    *,
    n_classes: int,
    k: int = KNN_K,
    temperature: float = KNN_TEMPERATURE,
) -> torch.Tensor:
    """(Q,) int64 weighted-kNN class of each query against a labelled bank (module docstring).

    ``bank`` (M, D) and ``queries`` (Q, D) are representations (normalised here); with
    ``k > M`` every bank row votes. Ties of the vote go to the lowest class (argmax).

    Raises:
        ValueError: for mismatched shapes, labels outside ``range(n_classes)`` or a
            non-positive ``k`` or ``temperature``.
    """
    _check_2d(bank, "bank")
    _check_2d(queries, "queries")
    if bank.shape[1] != queries.shape[1] or bank_labels.shape != (bank.shape[0],):
        raise ValueError("bank, bank_labels and queries do not match in shape")
    if k < 1 or not temperature > 0:
        raise ValueError(f"need k >= 1 and temperature > 0, got {k}, {temperature}")
    labels = bank_labels.to(torch.int64)
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(f"bank labels must be in range({n_classes})")
    b = F.normalize(bank.detach().float(), dim=1)
    q_all = F.normalize(queries.detach().float(), dim=1)
    k = min(k, b.shape[0])
    out = []
    for start in range(0, q_all.shape[0], _KNN_BLOCK):
        q = q_all[start : start + _KNN_BLOCK]
        sim, idx = (q @ b.T).topk(k, dim=1)
        # A one-hot sum rather than scatter_add_: deterministic on every device.
        one_hot = F.one_hot(labels.to(q.device)[idx], n_classes).float()
        votes = (one_hot * (sim / temperature).exp()[..., None]).sum(dim=1)
        out.append(votes.argmax(dim=1))
    return torch.cat(out)


def knn_balanced_accuracy(
    bank: torch.Tensor,
    bank_labels: torch.Tensor,
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    n_classes: int,
    k: int = KNN_K,
    temperature: float = KNN_TEMPERATURE,
) -> float:
    """Balanced accuracy of :func:`knn_predict` on labelled queries.

    Raises:
        ValueError: as :func:`knn_predict`, or if ``query_labels`` is not (Q,).
    """
    if query_labels.shape != (queries.shape[0],):
        raise ValueError("query_labels must have one label per query")
    pred = knn_predict(
        bank, bank_labels, queries, n_classes=n_classes, k=k, temperature=temperature
    )
    truth = query_labels.cpu().numpy()
    correct = (pred.cpu().numpy() == truth).astype(np.float64)
    return balanced_accuracy(correct, truth)
