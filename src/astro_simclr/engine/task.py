"""The ``Task`` protocol: what the plain loop trains, and the monitored value (ADR 0014).

A task owns a model, its loss, its optimizer and its per-step randomness; the loop owns
everything Lightning did around them (ADR 0014, "Checklist"): the ``train()`` / ``eval()``
modes, ``zero_grad(set_to_none=True)``, ``backward``, ``optimizer.step()``, ``no_grad``
during the fit's validations, the validation grid, the horizon, checkpoints and the RNG
streams. A task never reads a global step counter of its own: the loop passes ``step``,
the number of optimizer steps done before the batch, which is what Lightning's
``LightningModule.global_step`` is inside ``training_step`` (the augmentations of
``PaperClassifierModule`` are keyed by it).

Contracts only (task B0): the loop that calls a task arrives in 4.1, the paper classifier
task in 4.6.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn

from astro_simclr.engine.data import Batch, TrainBatch

__all__ = ["EvalOutput", "Task", "mean_of_batch_losses"]


@dataclass(frozen=True, slots=True)
class EvalOutput:
    """What one evaluation batch gives the loop (every tensor detached).

    Attributes:
        loss: the batch's mean loss, a 0-d float32 tensor (one entry of
            :func:`mean_of_batch_losses`).
        probabilities: (B, C) softmax of the logits, the per-stamp outputs of the
            comparisons (task 2.3).
        targets: (B,) int64 true labels.
    """

    loss: torch.Tensor
    probabilities: torch.Tensor
    targets: torch.Tensor


class Task(Protocol):
    """One trainable model with its loss, optimizer and per-step randomness.

    The loop's order for a run (ADR 0014, "Seeding" and "Resume"): seed every global
    stream, build the task (its ``module`` on the CPU, so initial weights do not depend on
    the device), move ``module`` to the device, then call :meth:`make_optimizer` (the
    optimizer is built on the moved parameters, as Lightning's ``strategy.setup`` does).
    """

    @property
    def module(self) -> nn.Module:
        """The model: what the loop moves, switches between train/eval and checkpoints."""

    def make_optimizer(self) -> torch.optim.Optimizer:
        """The optimizer over the trainable parameters; called once, after the device move."""

    def train_step(self, batch: TrainBatch, step: int) -> torch.Tensor:
        """The 0-d training loss of ``batch`` (graph attached; the loop does the backward).

        ``step`` is the number of optimizer steps done before this one (0 for the first
        step): every per-step draw (rotations, C5-C7, SimCLR views) is keyed by it.
        """

    def after_step(self, step: int) -> None:
        """Called after the optimizer step of step ``step`` (``train_step``'s ``step``).

        What follows each update outside the optimizer: a learning-rate scheduler's step,
        a weight EMA's update (plan row D3). It runs before the step's validation, if one
        is due. A task with nothing to do there does nothing.
        """

    def eval_step(self, batch: Batch) -> EvalOutput:
        """One evaluation batch, under the loop's ``eval()`` and no-grad context."""

    def state(self) -> dict[str, Any]:
        """Task state that is not in ``module.state_dict()`` (plain values; may be empty)."""

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore :meth:`state` on resume (after the weights, before the first step)."""


def mean_of_batch_losses(losses: Sequence[torch.Tensor]) -> torch.Tensor:
    """The monitored validation loss: ``torch.stack(losses).mean()``, in float32.

    The mean of the per-batch mean losses in loader order, the last (partial) batch
    weighing as much as a full one: TF's "mean of batch-64 means", what
    ``ClassifierModule._log_epoch`` logs as ``loss_val`` and Lightning passes on unchanged
    (ADR 0014, "Monitor value"). It is not the per-sample mean.

    Raises:
        ValueError: if ``losses`` is empty (no validation batch), or a loss is not a 0-d
            float32 tensor.
    """
    if not losses:
        raise ValueError("no validation batch: the monitored loss is undefined")
    for loss in losses:
        if loss.dim() != 0 or loss.dtype != torch.float32:
            msg = f"each batch loss must be a 0-d float32 tensor, got {loss.dtype} {loss.shape}"
            raise ValueError(msg)
    return torch.stack(list(losses)).mean()
