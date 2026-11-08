"""The paper classifier on the plain engine, for the tests (the 4.7(a) differential test).

A **test adapter**, not task 4.6's task: :class:`PaperTask` wraps today's
``PaperClassifierModule`` (a ``LightningModule``, used here as a plain ``nn.Module``) in the
engine's ``Task`` protocol, and :func:`plain_paper_run` runs ``run_command``'s
``carrasco2021`` path on :func:`astro_simclr.engine.fit` instead of a Lightning
``Trainer``. Everything that decides the numbers is shared with the Lightning path, so a
difference between the engines is the loop's:

- the data are the ones ``experiments._build_paper`` builds (the float crops, the leaked
  stamps removed, a split run's parts, C5-C7's fits), served by the engine's ``device``
  tier (``torch.equal`` to ``DeviceTensorDataModule``, ADR 0018);
- the model is ``experiments.paper_module(settings)``, built by the engine after its seed;
- a training step is the module's own ``augment_plan`` (the host draws of step ``step``)
  and ``apply_plan``, then its ``forward`` and ``compute_loss``: what
  ``PaperClassifierModule.training_step`` does with the loader's plan.

It lives in ``tests/`` because it imports Lightning (the engine must not, ADR 0014).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from astro_simclr import experiments
from astro_simclr.config import Settings
from astro_simclr.engine import (
    Batch,
    Dataset,
    DevicePolicy,
    EvalOutput,
    Evaluation,
    FitResult,
    Horizon,
    IdentityState,
    Monitor,
    Schedule,
    StreamSpec,
    Task,
    TierChoice,
    TrainBatch,
    Validation,
    ValidationGrid,
    evaluate_kept,
    fit,
    open_eval,
    open_train,
)
from astro_simclr.protocol.split_runs import SplitRequest, open_split_run
from astro_simclr.training.class_augment import upload_plan
from astro_simclr.training.device_data import TensorSplit
from astro_simclr.training.modules import AugmentBatch, ClassifierBatch, PaperClassifierModule
from astro_simclr.training.runs import monitor_mode


class PaperTask:
    """``PaperClassifierModule`` as an engine ``Task`` (module docstring)."""

    def __init__(self, paper: PaperClassifierModule) -> None:
        self.paper = paper

    @property
    def module(self) -> nn.Module:
        """The whole ``PaperClassifierModule``: its ``state_dict`` keys are Lightning's."""
        return self.paper

    def make_optimizer(self) -> torch.optim.Optimizer:
        """``configure_optimizers()``: the optimizer with the configured betas, no schedule."""
        return cast(torch.optim.Optimizer, self.paper.configure_optimizers())

    def train_step(self, batch: TrainBatch, step: int) -> torch.Tensor:
        """The planned augmentation of step ``step``, then the loss (``training_step``)."""
        paper = self.paper
        rows = batch
        if paper.augmenting:
            host = cast(ClassifierBatch | AugmentBatch, tuple(t.detach().cpu() for t in rows))
            plan = upload_plan(paper.augment_plan(host, step), rows[0].device)
            images, features, targets = paper.apply_plan(rows, plan)
        else:
            images, features, targets = rows[0], rows[1], rows[2]
        return paper.compute_loss(paper(images, features), targets)

    def eval_step(self, batch: Batch) -> EvalOutput:
        """``ClassifierModule._step``'s loss and probabilities, detached."""
        images, features, targets = batch
        logits = self.paper(images, features)
        loss = self.paper.compute_loss(logits, targets)
        return EvalOutput(
            loss=loss.detach(),
            probabilities=logits.softmax(dim=1).detach(),
            targets=targets.argmax(dim=1),
        )

    def state(self) -> dict[str, Any]:
        """No state outside the weights: every draw is a function of the step."""
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Nothing to restore."""


def _dataset(split: TensorSplit) -> Dataset:
    return Dataset(
        images=split.images,
        features=split.features,
        labels=split.labels,
        fits=split.fits,
        n_classes=split.n_classes,
    )


def schedule_of(settings: Settings) -> Schedule:
    """The ``carrasco2021`` settings as the engine's ``Schedule``."""
    return Schedule(
        monitor=Monitor(settings.monitor, monitor_mode(settings.monitor, frozenset({"loss_val"}))),
        validation=ValidationGrid(settings.val_every_n_steps),
        horizon=Horizon(settings.max_steps, settings.horizon_increment),
        checkpoint_every_n_steps=settings.checkpoint_every_n_steps,
    )


@dataclass(frozen=True, slots=True)
class PlainRun:
    """A plain run's outcome: the task, the fit result and the final evaluation."""

    task: PaperTask
    result: FitResult
    #: The kept weights' evaluation, keyed by role (``Validation``, ``Test``).
    evaluation: dict[str, Evaluation]
    #: The initial weights (CPU), as built before the device move.
    initial: dict[str, torch.Tensor]


def plain_paper_run(
    settings: Settings,
    *,
    run_dir: Path | None = None,
    split: SplitRequest | None = None,
    resume: Path | None = None,
    on_validation: Callable[[Validation, Task], None] | None = None,
) -> PlainRun:
    """``run_command("train-supervised", settings)``'s ``carrasco2021`` path, on the plain engine.

    ``run_dir`` defaults to ``settings.output_dir``. The roles are the Lightning path's:
    with ``split``, ``Validation`` is the selection part and ``Test`` the evaluation part.
    """
    data = experiments.dataset_path(settings)
    split_run = None if split is None else open_split_run(split, data)
    _, datamodule, _ = experiments._build_paper(settings, data, split_run)
    splits = datamodule._resolve()  # the TensorSplits the Lightning run trains on
    device = DevicePolicy(
        cast(Any, settings.accelerator),
        settings.deterministic,
        cast(Any, settings.fp32_precision),
    )
    target = device.device
    train = open_train(
        _dataset(splits["Train"]),
        TierChoice(),
        StreamSpec(batch_size=settings.batch_size, seed=settings.seed),
        target,
    )
    sources = {
        name: open_eval(
            _dataset(splits[name]), TierChoice(), target, batch_size=settings.eval_batch_size
        )
        for name in ("Validation", "Test")
    }
    identity: IdentityState = {
        "run_id": None,
        "attempt": None,
        "split_run": None if split_run is None else dict(split_run.identity()),
    }
    initial: dict[str, torch.Tensor] = {}

    def build() -> Task:
        task = PaperTask(experiments.paper_module(settings))
        initial.update({k: v.detach().clone() for k, v in task.module.state_dict().items()})
        return task

    schedule = schedule_of(settings)
    task, result = fit(
        build,
        train,
        sources["Validation"],
        schedule=schedule,
        device=device,
        seed=settings.seed,
        run_dir=settings.output_dir if run_dir is None else run_dir,
        identity=identity,
        resume=resume,
        on_validation=on_validation,
    )
    evaluation = evaluate_kept(task, result, sources, schedule.keep)["best"]
    return PlainRun(cast(PaperTask, task), result, evaluation, initial)
