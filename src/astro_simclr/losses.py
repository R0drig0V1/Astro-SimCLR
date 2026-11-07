"""Training losses: the SimCLR contrastive loss ``NT_Xent`` and the classifier ``P_stamps_loss``.

Ported from the legacy ``losses/losses.py`` (task 1.3 of phase 1). Behaviour changes against
the legacy code (also listed in ``docs/architecture.md``):

- ``NT_Xent`` builds its negative mask from the batch it receives. The legacy class built
  masks only for its constructor ``batch_size`` and for a hard-coded 100 (the eval batch),
  and any other batch size failed. ``NT_Xent`` therefore takes no ``batch_size``.
- ``NT_Xent`` computes the cosine-similarity matrix with one matrix product instead of the
  legacy broadcast ``CosineSimilarity(dim=2)``, which allocated an (N, N, D) tensor. The
  values agree with the legacy ones to float32 rounding (the characterization tests use
  ``atol=1e-6``).
- ``P_stamps_loss`` takes no ``batch_size``: the legacy class stored it and never read it.

The class names are the legacy ones, so that the legacy configs, docs and results still
name them. The NT-Xent denominator keeps the legacy formula by default (ADR 0002,
``docs/decisions/0002-nt-xent-denominator.md``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["NT_Xent", "P_stamps_loss"]

#: Lower bound on the product of two norms in the cosine similarity: the ``eps`` of the
#: legacy ``nn.CosineSimilarity`` (and of ``F.cosine_similarity``), so that a zero vector
#: gives similarity 0 instead of NaN, as it did in the legacy code.
_COSINE_EPS = 1e-8


class NT_Xent(nn.Module):
    """Normalized temperature-scaled cross-entropy (SimCLR, Chen et al. 2020).

    For a batch of B positive pairs ``(z_i[k], z_j[k])`` there are N = 2B views. Each view
    ``a`` has one positive ``p(a)`` (the other view of the same stamp) and 2(B - 1)
    negatives (every view of every other stamp). With ``s(a, b) = cos(z_a, z_b) / T``:

    - ``legacy_denominator=True`` (default, the legacy formula):
      ``loss_a = -s(a, p(a)) + log(1 + sum_neg exp(s(a, n)))``. The legacy code put a zero
      logit where the positive belongs, so the positive term in the denominator is
      ``exp(0) = 1`` instead of ``exp(s(a, p(a)))``. This is close to the decoupled
      contrastive loss (DCL, Yeh et al. 2021), which drops the positive from the denominator.
    - ``legacy_denominator=False`` (the paper's formula):
      ``loss_a = -s(a, p(a)) + log(exp(s(a, p(a))) + sum_neg exp(s(a, n)))``.

    The loss is the mean of ``loss_a`` over the N views. The choice and its consequences are
    recorded in ADR 0002.

    There is no all-gather: on several devices each one would only see its own negatives,
    which changes the loss (hence ``Settings.devices`` defaults to 1).
    """

    def __init__(self, temperature: float, *, legacy_denominator: bool = True) -> None:
        """Create the loss.

        Args:
            temperature: the softmax temperature T (> 0).
            legacy_denominator: use the legacy denominator (``1`` in place of the positive
                term); False gives the paper's formula. See the class docstring.

        Raises:
            ValueError: if ``temperature`` is not positive.
        """
        super().__init__()
        if not temperature > 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.legacy_denominator = legacy_denominator

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss of a batch of positive pairs.

        Args:
            z_i: projections of the first views, shape (B, D), B >= 1.
            z_j: projections of the second views, same shape; ``z_j[k]`` is the positive
                of ``z_i[k]``.

        Returns:
            ``(loss, pos, neg)`` as in the legacy code: the mean loss (differentiable), and,
            without gradient, the mean cosine similarity of the positive pairs and of the
            negative pairs (the legacy code multiplied the scaled similarities back by T).
            With B = 1 there are no negatives, so ``neg`` is NaN (as in the legacy code).

        Raises:
            ValueError: if the inputs are not two 2-D tensors of the same shape with B >= 1.
        """
        if z_i.ndim != 2 or z_i.shape != z_j.shape or z_i.shape[0] < 1:
            raise ValueError(
                "z_i and z_j must both have shape (B, D) with B >= 1, "
                f"got {tuple(z_i.shape)} and {tuple(z_j.shape)}"
            )
        batch = z_i.shape[0]
        n = 2 * batch
        z = torch.cat((z_i, z_j), dim=0)

        # Same formula as nn.CosineSimilarity: x.y / sqrt(max(|x|^2 |y|^2, eps^2)).
        sq_norm = (z * z).sum(dim=1)
        denom = torch.sqrt(torch.clamp_min(sq_norm[:, None] * sq_norm[None, :], _COSINE_EPS**2))
        sim = (z @ z.T) / denom / self.temperature

        index = torch.arange(n, device=z.device)
        partner = (index + batch) % n
        self_mask = index[:, None] == index[None, :]
        pos_mask = partner[:, None] == index[None, :]
        positives = sim[index, partner]

        # Row a of the logits holds every view but a itself; the positive column keeps
        # s(a, p(a)) (paper) or is replaced by 0, i.e. exp(0) = 1 (legacy).
        logits = sim.masked_fill(self_mask, float("-inf"))
        if self.legacy_denominator:
            logits = logits.masked_fill(pos_mask, 0.0)
        loss = (torch.logsumexp(logits, dim=1) - positives).mean()

        with torch.no_grad():
            neg_mask = ~(self_mask | pos_mask)
            neg = sim[neg_mask].mean() * self.temperature
            pos = positives.mean() * self.temperature
        return loss, pos, neg


class P_stamps_loss(nn.Module):
    """Cross-entropy with entropy regularization: ``CE(logits, y) - beta * H(softmax(logits))``.

    ``y`` is the argmax of the one-hot (or probability) targets, and both terms are batch
    means. A positive ``beta`` rewards high-entropy predictions, which discourages
    over-confident classifiers (the legacy stamp-classifier loss). The gradient flows through
    both factors of the entropy, as in the legacy code.
    """

    def __init__(self, beta: float) -> None:
        """Create the loss.

        Args:
            beta: weight of the entropy term (``beta_loss`` in the legacy configs).
        """
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the mean loss of a batch.

        Args:
            logits: unnormalized class scores, shape (B, C).
            targets: one-hot (or probability) targets, shape (B, C); only their argmax is
                used, so ties go to the first maximal class (``torch.argmax``).

        Returns:
            The scalar mean loss.

        Raises:
            ValueError: if ``logits`` and ``targets`` are not 2-D tensors of the same shape.
        """
        if logits.ndim != 2 or logits.shape != targets.shape:
            raise ValueError(
                "logits and targets must both have shape (B, C), "
                f"got {tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        log_p = F.log_softmax(logits, dim=1)
        cross_entropy = F.nll_loss(log_p, torch.argmax(targets, dim=1))
        entropy = -(log_p.exp() * log_p).sum(dim=1).mean()
        return cross_entropy - self.beta * entropy
