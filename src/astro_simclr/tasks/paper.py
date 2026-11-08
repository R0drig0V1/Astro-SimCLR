"""The paper classifier as a plain-engine task, and its data (task 4.6; ADR 0014).

What ``experiments._build_paper`` and ``PaperClassifierModule`` do for the Lightning path,
without Lightning, sharing every piece that decides the numbers:

- :func:`paper_inputs`: the run's splits as ``training.tensor_splits.TensorSplit`` (the
  float path or the uint8 diagnostic, the leaked Train stamps removed, a split run's
  roles, the C5-C7 fits of the Train role) and the ``paper_data`` metrics entry. The
  Lightning path's ``_build_paper`` calls it too, so both engines train on the same arrays.
- :func:`load_backbone` (the ConvNeXt pretrained backbone, ADR 0022), also shared.
- :class:`PaperNet`: the model under the attribute ``model`` (``build_paper_model``), so its
  ``state_dict`` keys are ``PaperClassifierModule``'s; the loss ``P_stamps_loss`` is its
  ``criterion``; with ``ema_decay`` (plan row D3) the weight EMA ``training.ema.EmaWeights``
  is its ``ema``, and its forward in ``eval()`` mode runs the EMA weights. With
  ``feature_set: none`` the forward ignores the features, as the Lightning module does.
- :class:`PaperTask`: the engine ``Task``. A training step plans the augmentation of step
  ``step`` on a host copy of the batch (``training.paper_augment.PaperAugment``, the same
  object the Lightning module delegates to), uploads the plan, applies it, then computes
  the loss: ``PaperClassifierModule.training_step`` with the loader's plan (the
  host-planned augmentation of PR #38). The optimizer is ``build_optimizer`` over every
  parameter but the EMA's, with the configured Adam betas, and the learning-rate schedule
  of plan row D3 (``training.lr_schedule``) when one is set; ``after_step`` steps that
  schedule and updates the EMA, as the Lightning module's scheduler and
  ``on_train_batch_end`` do.
- :func:`build_paper_task`: the task of a ``Settings``, built on the CPU (the loop calls
  it after seeding, ADR 0014, C1).

``tests/test_engine_differential.py`` holds the two engines bit for bit equal on it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from astro_simclr.config import CONVNEXT_ARCHS, ConfigError, Settings
from astro_simclr.data.prepared import load_prepared, read_meta
from astro_simclr.data.schema import SPLITS, DataError
from astro_simclr.engine.data import Batch, TrainBatch
from astro_simclr.engine.task import EvalOutput
from astro_simclr.feature_sets import feature_columns
from astro_simclr.losses import P_stamps_loss
from astro_simclr.models import N_CLASSES
from astro_simclr.paper_variants import build_paper_model, paper_input_side
from astro_simclr.protocol.leakage import find_duplicates, train_exclusions
from astro_simclr.protocol.split_runs import SplitRun
from astro_simclr.training.class_augment import fit_exclusions, stamp_fits, upload_plan
from astro_simclr.training.ema import EmaWeights, trainable_parameters
from astro_simclr.training.flux_rescale import fit_central_psf, psf_exclusions
from astro_simclr.training.lr_schedule import LrSchedule
from astro_simclr.training.optimizers import PAPER_ADAM_BETAS, build_optimizer, check_optimizer_name
from astro_simclr.training.paper_augment import PaperAugment, check_augment_context
from astro_simclr.training.tensor_splits import (
    TensorSplit,
    float_crops,
    interim_uint8_crops,
    protocol_splits,
    without_train_indices,
)

__all__ = [
    "PaperNet",
    "PaperTask",
    "build_paper_task",
    "check_paper_task",
    "leaked_train",
    "load_backbone",
    "lr_schedule_of",
    "paper_augment",
    "paper_inputs",
    "with_train_fits",
]

#: The weights downloader of :func:`load_backbone` when the caller passes none (None: the
#: Hugging Face Hub); tests replace it with a local file, so no test touches the network.
DOWNLOAD: Callable[[Any], Path] | None = None


def leaked_train(settings: Settings, data: Path) -> list[int]:
    """The Train indices to drop: the stamps that duplicate a Validation or Test stamp.

    Found by ``protocol.leakage`` on the prepared uint8 stamps (``red``: ``[2995]``), the
    same check that ``make-splits`` stores and ``verify_splits`` repeats; ``[]`` when
    ``exclude_leaked_train`` is false.
    """
    if not settings.exclude_leaked_train:
        return []
    return train_exclusions(find_duplicates(load_prepared(data)))


def with_train_fits(
    settings: Settings, splits: dict[str, TensorSplit], info: dict[str, Any]
) -> dict[str, TensorSplit]:
    """``splits`` with the Train role's augmentation table, when C5 / C6 / C7 need it.

    C5 / C6 need each Train stamp's cross-channel fit (``class_augment.stamp_fits``), C7 its
    central PSF fit too (``flux_rescale.fit_central_psf``, appended). Computed once, on the
    run's own Train role (the fold's training part in a split run); the excluded shares go
    to ``info`` (``class_augment``, ``flux_rescale``).
    """
    if not (settings.aug_noise or settings.synthetic_bogus or settings.flux_rescale):
        return splits
    train = splits["Train"]
    fits = stamp_fits(train.images)
    info["class_augment"] = fit_exclusions(fits, train.labels, settings.aug_min_r2)
    if settings.flux_rescale:
        fits = np.concatenate([fits, fit_central_psf(train.images)], axis=1)
        info["flux_rescale"] = psf_exclusions(fits, train.labels)
    return {**splits, "Train": dataclasses.replace(train, fits=fits)}


def paper_inputs(
    settings: Settings, data: Path, split_run: SplitRun | None = None
) -> tuple[Callable[[], dict[str, TensorSplit]], dict[str, Any]]:
    """The run's splits (a callable that builds them) and its ``paper_data`` metrics entry.

    The input is the float path (``paper_input: float32``, the paper's) or the interim uint8
    crops (``uint8``, a diagnostic); the leaked Train stamps are removed before the tiling.
    With ``split_run`` only Train and Validation are read, and the roles are the run's
    parts (``protocol_splits``); the excluded Train stamps are the splits file's, which
    ``open_split_run`` has checked against a fresh leakage check. The callable fills
    ``n_train`` (and the C5-C7 exclusions) into the entry when it runs.

    Raises:
        DataError: if the float path is asked for and the data have no float32 stamps.
    """
    if settings.paper_input == "float32" and "float32" not in read_meta(data)["images"]:
        # Fail before the leakage scan and the training setup; float_crops says it too.
        msg = (
            f"{data} has no float32 stamps (images_f32.npy), which the paper's input needs: "
            "run `astro-simclr prepare-data PKL --images float32 --overwrite` (or set "
            "paper_input=uint8, a diagnostic)"
        )
        raise DataError(msg)
    if split_run is None:
        excluded = leaked_train(settings, data)
    else:
        excluded = list(split_run.splits.excluded_train)
    side = paper_input_side(settings.paper_arch)
    info: dict[str, Any] = {
        "input": settings.paper_input,
        "crop": side,
        "nan_approximation": (
            settings.nan_approximation if settings.paper_input == "float32" else None
        ),
        "excluded_train": excluded,
    }

    def crops(names: Sequence[str]) -> dict[str, TensorSplit]:
        # One reader for both paths, so a split run gets the same input (and context
        # channels, task 2.5-models) as a paper-protocol run.
        if settings.paper_input == "float32":
            return float_crops(
                data,
                nan_approximation=settings.nan_approximation,
                names=names,
                context=settings.context != "none",
                side=side,
            )
        return interim_uint8_crops(data, names=names, side=side)

    def source() -> dict[str, TensorSplit]:
        if split_run is not None:
            pool = crops(("Train", "Validation"))  # never Test
            splits = protocol_splits(pool, split_run.parts(), split_run.splits.n_train)
        else:
            splits = without_train_indices(crops(SPLITS), excluded)
        info["n_train"] = len(splits["Train"].labels)
        return with_train_fits(settings, splits, info)

    return source, info


def load_backbone(
    settings: Settings,
    model: nn.Module,
    info: dict[str, Any],
    *,
    download: Callable[[Any], Path] | None = None,
) -> None:
    """Start the ConvNeXt backbone of ``model`` from ``backbone_weights`` (plan row B8; ADR 0022).

    Nothing happens for ``none`` (from scratch: no network access). Otherwise the pinned
    file is downloaded (``download``, default :data:`DOWNLOAD`, or the Hugging Face cache),
    verified and loaded into the backbone, after the seeded initialisation, and recorded in
    ``info["backbone_weights"]``. A resumed run loads it too, then the checkpoint replaces
    every weight.

    Raises:
        ConfigError: if ``model`` is not a ConvNeXt or the weights do not fit ``paper_arch``.
        WeightsError: as ``convnext_model.fetch_backbone_weights`` and
            ``load_backbone_weights`` (a ``DataError``: the CLI prints it on one line).
    """
    if settings.backbone_weights == "none":
        return
    from astro_simclr.convnext_model import (
        ConvNeXtStampClassifier,
        backbone_weights_for,
        fetch_backbone_weights,
        load_backbone_weights,
    )

    weights = backbone_weights_for(settings.backbone_weights, settings.paper_arch)
    if weights is None or not isinstance(model, ConvNeXtStampClassifier):
        # Settings refuses it already (zoobot needs convnext_pico); kept for direct callers.
        raise ConfigError(f"backbone_weights need a ConvNeXt paper_arch, not {settings.paper_arch}")
    fetched = fetch_backbone_weights(weights, download=DOWNLOAD if download is None else download)
    load_backbone_weights(model, weights, fetched)
    info["backbone_weights"] = weights.record()


def paper_augment(settings: Settings) -> PaperAugment:
    """The training augmentation of ``settings`` (``PaperClassifierModule``'s).

    Raises:
        ValueError: as :class:`PaperAugment` and ``check_augment_context``.
    """
    augment = PaperAugment(
        seed=settings.seed,
        rot90=settings.rot90_augment,
        noise=settings.aug_noise,
        feature_jitter=settings.aug_feature_jitter,
        synthetic_bogus=settings.synthetic_bogus,
        min_r2=settings.aug_min_r2,
        flux_rescale=settings.flux_rescale,
        flux_k_min=settings.flux_k_min,
        flux_k_max=settings.flux_k_max,
    )
    check_augment_context(
        settings.synthetic_bogus, settings.flux_rescale, settings.context, N_CLASSES
    )
    return augment


def lr_schedule_of(settings: Settings) -> LrSchedule:
    """The learning-rate schedule of ``settings`` (plan row D3), over ``max_steps``.

    Raises:
        ValueError: as :class:`~astro_simclr.training.lr_schedule.LrSchedule`.
    """
    return LrSchedule(settings.lr_schedule, settings.lr_warmup_steps, settings.max_steps)


class PaperNet(nn.Module):
    """The paper model (``model``) and its loss (``criterion``), as ``PaperClassifierModule``.

    ``reads_features`` is whether the head reads the features (false for ``feature_set:
    none``, or without features): the forward then passes ``None``. ``ema`` (plan row D3)
    is the weight EMA of ``model``, or None; registered after ``model``, so the
    ``state_dict`` keys and order are the Lightning module's. With an EMA the forward runs
    the EMA weights in ``eval()`` mode (every validation and evaluation), ``model`` in
    ``train()`` mode.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: P_stamps_loss,
        *,
        reads_features: bool,
        ema: EmaWeights | None = None,
    ):
        super().__init__()
        self.criterion = criterion
        self.model = model
        self.reads_features = reads_features
        self.ema = ema

    def forward(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Crops (B, C, S, S) and features (B, 23) -> (B, n_classes) logits."""
        net = self.model if self.ema is None or self.training else self.ema.shadow
        out: torch.Tensor = net(images, features if self.reads_features else None)
        return out


class PaperTask:
    """The paper classifier as an engine ``Task`` (module docstring).

    ``lr_schedule`` (plan row D3) is the learning-rate schedule; the default is CD21's
    constant rate (no scheduler).

    Raises:
        UnknownOptimizerError: if ``optimizer`` is not a known name.
        ValueError: if ``adam_betas`` is not a key of ``PAPER_ADAM_BETAS``.
    """

    def __init__(
        self,
        net: PaperNet,
        augment: PaperAugment,
        *,
        optimizer: str,
        lr: float,
        adam_betas: str = "code",
        lr_schedule: LrSchedule | None = None,
    ) -> None:
        check_optimizer_name(optimizer)
        if adam_betas not in PAPER_ADAM_BETAS:
            valid = ", ".join(PAPER_ADAM_BETAS)
            raise ValueError(f"adam_betas must be one of {valid}, got {adam_betas!r}")
        self.net = net
        self.augment = augment
        self.optimizer_name = optimizer
        self.lr = lr
        self.adam_betas = adam_betas
        self.lr_schedule = LrSchedule() if lr_schedule is None else lr_schedule
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    @property
    def module(self) -> nn.Module:
        """The :class:`PaperNet` (its ``state_dict`` keys are the Lightning module's)."""
        return self.net

    def make_optimizer(self) -> torch.optim.Optimizer:
        """``build_optimizer`` over every parameter but the EMA's, with the configured betas.

        With an active ``lr_schedule`` the task also builds its ``LambdaLR`` here (which
        sets step 0's learning rate); :meth:`after_step` steps it.
        """
        betas = PAPER_ADAM_BETAS[self.adam_betas]
        optimizer = build_optimizer(
            self.optimizer_name, trainable_parameters(self.net), self.lr, adam_betas=betas
        )
        self.scheduler = self.lr_schedule.scheduler(optimizer)
        return optimizer

    def train_step(self, batch: TrainBatch, step: int) -> torch.Tensor:
        """The loss of step ``step``'s batch after its planned augmentation.

        Raises:
            ValueError: as ``PaperAugment.plan`` (C5-C7 without the stamp fits).
        """
        if self.augment.active:
            host = tuple(tensor.detach().cpu() for tensor in batch)
            planned = self.augment.plan(host, step)  # type: ignore[arg-type]
            plan = upload_plan(planned, batch[0].device)
            images, features, targets = self.augment.apply(batch, plan)
        else:
            images, features, targets = batch[0], batch[1], batch[2]
        loss: torch.Tensor = self.net.criterion(self.net(images, features), targets)
        return loss

    def after_step(self, step: int) -> None:
        """After optimizer step ``step``: the scheduler's step, then the EMA update (D3)."""
        if self.scheduler is not None:
            self.scheduler.step()
        if self.net.ema is not None:
            self.net.ema.update(self.net.model, step)

    def eval_step(self, batch: Batch) -> EvalOutput:
        """The batch's loss, probabilities and labels (``ClassifierModule._step``'s)."""
        images, features, targets = batch
        logits = self.net(images, features)
        loss = self.net.criterion(logits, targets)
        return EvalOutput(
            loss=loss.detach(),
            probabilities=logits.softmax(dim=1).detach(),
            targets=targets.argmax(dim=1),
        )

    def state(self) -> dict[str, Any]:
        """The scheduler's state, if any (the EMA is in the weights; every draw is a
        function of the step). Empty for a constant learning rate."""
        if self.scheduler is None:
            return {}
        return {"lr_scheduler": self.scheduler.state_dict()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore the scheduler's state (after the optimizer's, which holds the rate).

        Raises:
            ValueError: if the state and this task disagree on having a scheduler (another
                schedule; the run id refuses it first).
        """
        saved = state.get("lr_scheduler")
        if (saved is None) != (self.scheduler is None):
            raise ValueError("the checkpoint was saved with another learning-rate schedule")
        if self.scheduler is not None and saved is not None:
            self.scheduler.load_state_dict(dict(saved))


def check_paper_task(settings: Settings) -> None:
    """Every check of :func:`build_paper_task` that needs no model (run before the manifest).

    Raises:
        UnknownOptimizerError, ValueError: an unknown optimizer, betas, augmentation value,
            or feature set; C5's noise, C6 or C7 on a ConvNeXt.
    """
    check_optimizer_name(settings.optimizer)
    lr_schedule_of(settings)
    if settings.adam_betas not in PAPER_ADAM_BETAS:
        valid = ", ".join(PAPER_ADAM_BETAS)
        raise ValueError(f"adam_betas must be one of {valid}, got {settings.adam_betas!r}")
    feature_columns(settings.feature_set, with_features=settings.with_features)
    if settings.paper_arch in CONVNEXT_ARCHS and paper_augment(settings).needs_fits:
        # C5's noise, C6 and C7 are defined on the 21x21 crop (ADR 0022).
        raise ValueError(
            f"aug_noise, synthetic_bogus and flux_rescale work on the 21x21 crop, not on "
            f"the whole stamps of {settings.paper_arch}"
        )


def build_paper_task(
    settings: Settings, info: dict[str, Any], *, download: Callable[[Any], Path] | None = None
) -> PaperTask:
    """The task of ``settings``, on the CPU, from the global generator (call after seeding).

    The model is built exactly as ``PaperClassifierModule`` builds it (the loss first, then
    ``build_paper_model`` with the same arguments: the same draws), then the ConvNeXt
    backbone weights are loaded if configured (:func:`load_backbone`), then the weight EMA
    is copied from it if ``ema_decay`` is set (no draw).

    Raises:
        UnknownOptimizerError, ValueError: as ``PaperClassifierModule`` (an unknown
            optimizer, betas, augmentation value, variant or feature set; C5's noise, C6
            or C7 on a ConvNeXt).
        WeightsError: as :func:`load_backbone`.
    """
    check_paper_task(settings)
    columns = feature_columns(settings.feature_set, with_features=settings.with_features)
    reads_features = bool(columns)
    criterion = P_stamps_loss(settings.beta_loss)
    augment = paper_augment(settings)
    model = build_paper_model(
        settings.paper_arch,
        group_fields=settings.group_fields,
        context=settings.context,
        with_features=settings.with_features,
        dropout_active=settings.dropout_active,
        drop_rate=settings.drop_rate,
        feature_set=settings.feature_set,
    )
    load_backbone(settings, model, info, download=download)
    # After the backbone weights, so the EMA starts from them (plan row D3).
    ema = (
        None
        if settings.ema_decay is None
        else EmaWeights(model, settings.ema_decay, warmup=settings.ema_warmup)
    )
    net = PaperNet(model, criterion, reads_features=reads_features, ema=ema)
    return PaperTask(
        net,
        augment,
        optimizer=settings.optimizer,
        lr=settings.lr,
        adam_betas=settings.adam_betas,
        lr_schedule=lr_schedule_of(settings),
    )
