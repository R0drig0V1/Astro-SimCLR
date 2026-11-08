"""The Lightning 2 training modules: SimCLR pretraining and the four classifiers (task 1.6).

Ports of the legacy ``utils/training.py`` (PyTorch Lightning 1.4):

| Class | Legacy class | Trains |
|---|---|---|
| :class:`SimCLRModule` | ``SimCLR`` | encoder + projector with ``NT_Xent`` (LARS + cosine warmup) |
| :class:`LinearEvalModule` | ``Linear_classifier`` | a linear head on a frozen pretrained encoder |
| :class:`FineTuneModule` | ``Fine_tuning`` | the pretrained encoder and a new head |
| :class:`SupervisedCEModule` | ``Supervised_Cross_Entropy`` | ``StampClassifier`` from scratch |
| :class:`SupervisedCEResNetModule` | ``Supervised_Cross_Entropy_Resnet`` | a ResNet + linear head |
| :class:`PaperClassifierModule` | (new, task 2.2a) | the paper model or a 2.5 variant |

What changed for Lightning 2 (the arithmetic of every step is the legacy one):

- ``training_epoch_end(outputs)`` and friends were removed. Each step appends its detached
  predictions, labels and loss to an explicit per-stage buffer; ``on_*_epoch_end`` reads
  and clears it. The epoch loss is still the mean of the per-batch losses.
- ``self.log`` replaces ``self.logger.experiment.add_scalar``, so any logger works. The
  legacy TensorBoard names are kept (``Loss/Train``, ``Accuracy/Validation``, ...); their
  x-axis is now the global step (Lightning's convention) instead of the epoch.
- The metrics and the confusion matrix come from :mod:`astro_simclr.training.metrics`
  (micro accuracy, explicitly), not from four copies of the code.
- Datasets and loaders moved to :mod:`astro_simclr.training.datamodules`; the modules
  receive batches only.

The names that ``Settings.monitor`` may select are each module's :attr:`MONITORABLE`. The
classifiers log ``accuracy_val`` (the legacy monitor) and ``loss_val``; SimCLR logs
``loss_train_enc``. Other behaviour changes are listed on each class and in
``docs/architecture.md``.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Final, cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from astro_simclr.config import (
    CONVNEXT_ARCHS,
    DEFAULT_HYBRID_DEPTH,
    DEFAULT_HYBRID_DIM,
    DEFAULT_HYBRID_HEADS,
    DEFAULT_HYBRID_READOUT,
    HYBRID_FLAGS,
)
from astro_simclr.feature_sets import feature_columns
from astro_simclr.losses import NT_Xent, P_stamps_loss
from astro_simclr.models import (
    N_CLASSES,
    N_FEATURES,
    LinearClassifier,
    SimCLRNet,
    StampClassifier,
    ThreeLayerClassifier,
    build_encoder,
    check_n_classes,
)
from astro_simclr.optim import cosine_warmup_scheduler
from astro_simclr.paper_model import PAPER_DROP_RATE
from astro_simclr.paper_variants import build_paper_model
from astro_simclr.training.class_augment import (
    DEFAULT_MIN_R2,
    JITTER_STREAM,
    NOISE_STREAM,
    SYNTHETIC_STREAM,
    apply_noise,
    apply_synthetic,
    augment_rng,
    noise_augment,
    plan_jitter,
    plan_noise,
    plan_synthetic,
    synthetic_bogus,
    upload_plan,
)
from astro_simclr.training.flux_rescale import FLUX_STREAM, apply_flux, flux_rescale, plan_flux
from astro_simclr.training.metrics import METRIC_NAMES, classification_metrics, confusion_matrix
from astro_simclr.training.optimizers import (
    PAPER_ADAM_BETAS,
    build_optimizer,
    check_optimizer_name,
)

__all__ = [
    "ClassifierModule",
    "EncoderClassifier",
    "FineTuneModule",
    "LinearEvalModule",
    "PaperClassifierModule",
    "SimCLRModule",
    "SupervisedCEModule",
    "SupervisedCEResNetModule",
]

#: A classifier batch: images (B, 3, S, S), features (B, 23), one-hot labels (B, n_classes).
ClassifierBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#: A paper-model training batch with the stamp fits (B, 6) of ``class_augment.stamp_fits``.
AugmentBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
#: A training batch followed by its uploaded augmentation plan (``augment_plan``).
PlannedBatch = tuple[Any, ...]
#: A SimCLR batch: the two views (B, 3, S, S) of each stamp.
PairBatch = tuple[torch.Tensor, torch.Tensor]

#: Stage -> the suffix of its legacy TensorBoard names.
_STAGE_NAMES: Final = {"train": "Train", "val": "Validation", "test": "Test"}
#: Hidden width of the fine-tuning MLP head of the stamp encoders (legacy value).
_FINE_TUNE_HIDDEN: Final = 64


def _hparams_ignore(n_classes: int, *names: str) -> list[str]:
    """The ``save_hyperparameters(ignore=...)`` list of a classifier with ``n_classes``.

    ``n_classes`` is recorded only when it is not the default (5): the hyperparameters of
    the phase-2 modules, and so their ``hparams.yaml`` and checkpoints, stay exactly what
    they were before the parameter existed, and ``load_from_checkpoint`` rebuilds them with
    the default. An N-class module records it, and reloads with it.
    """
    return [*names, *(["n_classes"] if n_classes == N_CLASSES else [])]


@dataclass(frozen=True, slots=True)
class _StepOutput:
    """What one classifier step leaves in its epoch buffer (all detached)."""

    y_pred: torch.Tensor
    y_true: torch.Tensor
    loss: torch.Tensor
    #: softmax of the logits (B, n_classes): the per-stamp outputs of the comparisons (2.3).
    probabilities: torch.Tensor


class ClassifierModule(L.LightningModule):
    """Shared steps, epoch-end metrics and optimizer of the four classifier modules.

    Each step computes ``loss = compute_loss(logits, one_hot_targets)`` with
    ``P_stamps_loss(beta_loss)`` and buffers ``argmax(softmax(logits))``, the true labels
    and the loss (and the softmax probabilities). At the end of each train, validation and
    test epoch it logs, under the legacy names, ``Loss/<Stage>`` (mean of the batch
    losses), ``Accuracy/<Stage>`` (micro),
    ``Precision/<Stage>`` and ``Recall/<Stage>`` (macro). Validation and test also log
    ``accuracy_val`` / ``loss_val`` (``accuracy_test`` / ``loss_test``) on the progress bar,
    and store the row-normalised confusion matrix, on the CPU, in
    ``confusion_matrices["val"]`` / ``["test"]`` (the labels in ``predictions``, the
    probabilities in ``probabilities``), except in the validations inside ``fit``: those
    only feed the monitor, and ``trainer.validate`` / ``trainer.test`` fill these.

    Subclasses define ``forward(x_img, x_feat)``, and may override :meth:`compute_loss`
    and :meth:`trainable_parameters`. ``n_classes`` (default 5, CD21's classes) sizes the
    metrics and the confusion matrices; each subclass builds its network with it.

    Raises:
        UnknownOptimizerError: if ``optimizer`` is not a known name (at construction, not
            when training starts).
        ValueError: if ``n_classes`` is not an integer >= 2.
    """

    #: Metrics that ``Settings.monitor`` may name for this module.
    MONITORABLE: ClassVar[frozenset[str]] = frozenset({"accuracy_val", "loss_val"})

    def __init__(
        self,
        *,
        beta_loss: float,
        lr: float,
        optimizer: str,
        with_features: bool,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        check_optimizer_name(optimizer)
        self.n_classes = check_n_classes(n_classes)
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer_name = optimizer
        self.with_features = with_features
        self.criterion = P_stamps_loss(beta_loss)
        self._outputs: dict[str, list[_StepOutput]] = {stage: [] for stage in _STAGE_NAMES}
        #: The last confusion matrix of each evaluation stage (``"val"``, ``"test"``).
        self.confusion_matrices: dict[str, torch.Tensor] = {}
        #: The last ``(y_pred, y_true)`` of each evaluation stage, on the CPU (for the
        #: metrics JSON of the CLI: macro-F1 and the confusion-matrix counts).
        self.predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        #: The last softmax probabilities (N, n_classes) of each evaluation stage, on the CPU, in
        #: loader order (the per-stamp predictions files of the CLI, task 2.3).
        self.probabilities: dict[str, torch.Tensor] = {}

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """The training loss of a batch: ``P_stamps_loss(logits, one_hot_targets)``."""
        loss: torch.Tensor = self.criterion(logits, targets)
        return loss

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """The parameters the optimizer updates (all of them, unless overridden)."""
        return self.parameters()

    def _step(self, batch: ClassifierBatch, stage: str) -> torch.Tensor:
        x_img, x_feat, targets = batch
        logits = self(x_img, x_feat)
        loss = self.compute_loss(logits, targets)
        probabilities = logits.softmax(dim=1).detach()
        y_pred = probabilities.argmax(dim=1)
        y_true = targets.argmax(dim=1)
        self._outputs[stage].append(
            _StepOutput(y_pred, y_true.detach(), loss.detach(), probabilities)
        )
        return loss

    def training_step(self, batch: ClassifierBatch, batch_idx: int) -> torch.Tensor:
        """One optimisation step; returns the loss Lightning backpropagates."""
        return self._step(batch, "train")

    def validation_step(self, batch: ClassifierBatch, batch_idx: int) -> torch.Tensor:
        """One validation batch (buffered for :meth:`on_validation_epoch_end`)."""
        return self._step(batch, "val")

    def test_step(self, batch: ClassifierBatch, batch_idx: int) -> torch.Tensor:
        """One test batch (buffered for :meth:`on_test_epoch_end`)."""
        return self._step(batch, "test")

    def on_train_epoch_start(self) -> None:
        """Start the epoch with an empty buffer (an interrupted epoch may have left one)."""
        self._outputs["train"].clear()

    def on_validation_epoch_start(self) -> None:
        """Start the validation pass with an empty buffer."""
        self._outputs["val"].clear()

    def on_test_epoch_start(self) -> None:
        """Start the test pass with an empty buffer."""
        self._outputs["test"].clear()

    def on_train_epoch_end(self) -> None:
        """Log the training metrics of the epoch and clear its buffer."""
        self._log_epoch("train")

    def on_validation_epoch_end(self) -> None:
        """Log the validation metrics (including the monitored ones) and clear the buffer."""
        self._log_epoch("val")

    def on_test_epoch_end(self) -> None:
        """Log the test metrics and clear the buffer."""
        self._log_epoch("test")

    def _log_epoch(self, stage: str) -> None:
        outputs, self._outputs[stage] = self._outputs[stage], []
        if not outputs:
            return
        y_pred = torch.cat([out.y_pred for out in outputs])
        y_true = torch.cat([out.y_true for out in outputs])
        loss = torch.stack([out.loss for out in outputs]).mean()
        # The labels are argmaxes of n_classes-wide tensors, always in range: skip
        # torchmetrics' value checks (extra kernels and, on a GPU, a sync every validation).
        metrics = classification_metrics(
            y_pred, y_true, num_classes=self.n_classes, validate_args=False
        )
        name = _STAGE_NAMES[stage]
        self.log(f"Loss/{name}", loss)
        for key in METRIC_NAMES:
            self.log(f"{key.capitalize()}/{name}", metrics[key])
        if stage != "train":
            # The legacy monitor names, on the progress bar only (as in the legacy code).
            self.log(f"accuracy_{stage}", metrics["accuracy"], logger=False, prog_bar=True)
            self.log(f"loss_{stage}", loss, logger=False, prog_bar=True)
            if self.trainer.state.fn == TrainerFn.FITTING:
                # The fit's validations only select: the run's outputs come from
                # validate/test(ckpt_path=best), so their copies would be thrown away.
                return
            self.confusion_matrices[stage] = confusion_matrix(
                y_pred, y_true, num_classes=self.n_classes
            ).cpu()
            self.predictions[stage] = (y_pred.cpu(), y_true.cpu())
            self.probabilities[stage] = torch.cat([out.probabilities for out in outputs]).cpu()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """The optimizer ``optimizer`` over :meth:`trainable_parameters`, no LR schedule."""
        return build_optimizer(self.optimizer_name, self.trainable_parameters(), self.lr)


class EncoderClassifier(nn.Module):
    """An encoder, then a head on its output, concatenated with batch-normalised features.

    ``forward(x_img, x_feat)`` is ``head(cat(encoder(x_img), bn(x_feat)))`` with features,
    ``head(encoder(x_img))`` without (``x_feat`` is then ignored). The legacy code built the
    same network as ``nn.Sequential(encoder, BatchNorm1d(23), head)`` and indexed it.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module, *, with_features: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.bn = nn.BatchNorm1d(N_FEATURES) if with_features else None
        self.head = head

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) images and (B, 23) features -> (B, n_classes) logits."""
        h = self.encoder(x_img)
        if self.bn is not None:
            h = torch.cat((h, self.bn(x_feat)), dim=1)
        out: torch.Tensor = self.head(h)
        return out


def _n_inputs(encoder_features: int, with_features: bool) -> int:
    return encoder_features + (N_FEATURES if with_features else 0)


class SupervisedCEModule(ClassifierModule):
    """The supervised stamp classifier (legacy ``Supervised_Cross_Entropy``).

    ``StampClassifier(drop_rate, with_features)`` trained from scratch with
    ``P_stamps_loss(beta_loss)``. Images must be 27x27.

    Behaviour changes: the legacy ``batch_size``, ``image_size``, ``augmentation`` and
    ``data_path`` arguments belong to ``ClassifierDataModule`` now; a test loop exists (the
    legacy module had only a test dataloader).
    """

    def __init__(
        self,
        *,
        drop_rate: float,
        beta_loss: float,
        lr: float,
        optimizer: str,
        with_features: bool = True,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(
            beta_loss=beta_loss,
            lr=lr,
            optimizer=optimizer,
            with_features=with_features,
            n_classes=n_classes,
        )
        self.model = StampClassifier(drop_rate, with_features, n_classes)
        self.save_hyperparameters(ignore=_hparams_ignore(n_classes))

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, n_classes) logits."""
        out: torch.Tensor = self.model(x_img, x_feat if self.with_features else None)
        return out


def _check_cd21_augment(
    synthetic_bogus: float, flux_rescale: float, context: str, n_classes: int
) -> None:
    """Refuse C6 / C7 where they cannot apply: with context channels, or not on CD21's classes.

    Raises:
        ValueError: if ``flux_rescale`` or ``synthetic_bogus`` is set with ``context`` other
            than ``none``, or with ``n_classes`` other than 5.
    """
    if flux_rescale and context != "none":
        raise ValueError("flux_rescale changes no context channels: needs context='none'")
    if synthetic_bogus and context != "none":
        raise ValueError("synthetic_bogus builds no context channels: needs context='none'")
    if (synthetic_bogus or flux_rescale) and n_classes != N_CLASSES:
        # C6 and C7 use CD21's label indices (Asteroid and Bogus sources; never Bogus).
        raise ValueError("synthetic_bogus and flux_rescale need the 5 CD21 classes")


class PaperClassifierModule(ClassifierModule):
    """The paper-faithful stamp classifier, ``paper_model.PaperStampClassifier`` (task 2.2a).

    Trained from scratch with ``P_stamps_loss(beta_loss)`` on 21x21 crops (the batches of
    ``device_data.DeviceTensorDataModule``). ``dropout_active=False`` is M1's training
    (dropout layers present but inactive); ``drop_rate`` is their rate when active.
    ``adam_betas`` names the betas of ``Adam`` / ``AdamW`` (``optimizers.PAPER_ADAM_BETAS``):
    ``code`` (0.9, 0.999) is ALeRCE's training (task 2.2c), ``paper`` (0.5, 0.9) a diagnostic.

    The variants of task 2.5-models (``paper_variants.build_paper_model``): ``paper_arch``
    (``rot4``, faithful; ``single``; ``p4``; ``p4m``), ``group_fields`` and ``context``
    (``none`` or ``avgpool63``: the batches then have 6 channels); ``convnext_atto`` and
    ``convnext_pico`` (plan row B8, ADR 0022) take the whole stamps, (B, 3, 63, 63) batches,
    and refuse C5's noise, C6 and C7 (defined on the crop); ``hybrid`` and ``hybrid_p4``
    (plan row B9, ``hybrid_model``) are conv-attention hybrids on the crop, shaped by
    ``hybrid_dim``, ``hybrid_depth``, ``hybrid_heads`` and ``hybrid_readout`` (recorded in
    the hyperparameters only when they are not their defaults). The defaults build the
    faithful model with the same draws from the global generator. ``rot90_augment`` rotates
    every training stamp by ``k * 90`` degrees, ``k`` uniform in 0..3, drawn from a CPU
    generator seeded with ``[augment_seed, global_step]``: independent of the device and of
    the other random draws, and the same after a ``--resume``. Validation and test are
    never augmented.

    The class-aware augmentation of plan rows C5 and C6 (:mod:`training.class_augment`,
    ADR 0006), before the rotation (:meth:`augmented_batch`): ``synthetic_bogus`` replaces
    that fraction of the batch with synthetic hot-pixel Bogus, ``aug_noise`` adds background
    noise to the other rows
    (the difference follows the science and the template), ``aug_feature_jitter`` jitters
    the continuous features of every row; ``aug_min_r2`` excludes the stamps whose
    difference the cross-channel fit does not explain. Each has its own generator seeded
    with ``[augment_seed, global_step, stream]``. ``aug_noise`` and ``synthetic_bogus``
    need training batches with the stamp fits as a fourth element (a Train
    ``TensorSplit`` with ``fits``).

    The transient flux rescaling of plan row C7 (:mod:`training.flux_rescale`, ADR 0007),
    after C6 and before C5's noise: with probability ``flux_rescale`` a non-Bogus row's
    central transient (a Gaussian fitted to the difference) becomes ``k`` times brighter,
    ``k`` log-uniform in ``[flux_k_min, flux_k_max]`` raised to keep the row an alert, and
    ``magpsf`` / ``sigmapsf`` follow. It needs the 11-column fits (``flux_rescale.
    AUGMENT_COLUMNS``) in the batches, and no context channels.

    ``feature_set`` (:mod:`astro_simclr.feature_sets`, ADR 0016) picks the features the
    model's head reads from the 23 of every batch: ``all`` (faithful), ``position`` (the 6
    sky coordinates) or ``none`` (exactly the ``with_features=False`` module: the same model
    and draws, and ``with_features`` reads false). It is recorded in the hyperparameters
    only when it is not ``all``, so the faithful module's ``hparams.yaml`` and checkpoints
    are what they were before it existed.

    Raises:
        UnknownOptimizerError: if ``optimizer`` is not a known name.
        ValueError: if ``adam_betas`` is not a key of ``PAPER_ADAM_BETAS``, ``augment_seed``
            or an augmentation strength is negative, ``synthetic_bogus`` is not in [0, 1),
            ``aug_min_r2`` not in [0, 1], ``flux_rescale`` not in [0, 1], not ``0 <
            flux_k_min <= flux_k_max``, ``synthetic_bogus`` or ``flux_rescale`` is set with
            context channels or with ``n_classes`` other than 5, ``n_classes`` is not an
            integer >= 2, or the variant or ``feature_set`` is invalid
            (``build_paper_model``).
    """

    def __init__(
        self,
        *,
        beta_loss: float,
        lr: float,
        optimizer: str,
        with_features: bool = True,
        dropout_active: bool = False,
        drop_rate: float = PAPER_DROP_RATE,
        adam_betas: str = "code",
        paper_arch: str = "rot4",
        group_fields: int | None = None,
        context: str = "none",
        rot90_augment: bool = False,
        augment_seed: int = 0,
        aug_noise: float = 0.0,
        aug_feature_jitter: float = 0.0,
        synthetic_bogus: float = 0.0,
        aug_min_r2: float = DEFAULT_MIN_R2,
        flux_rescale: float = 0.0,
        flux_k_min: float = 0.5,
        flux_k_max: float = 1.5,
        n_classes: int = N_CLASSES,
        feature_set: str = "all",
        hybrid_dim: int = DEFAULT_HYBRID_DIM,
        hybrid_depth: int = DEFAULT_HYBRID_DEPTH,
        hybrid_heads: int = DEFAULT_HYBRID_HEADS,
        hybrid_readout: str = DEFAULT_HYBRID_READOUT,
    ) -> None:
        # Fails here on an unknown set or position without features; none: no features.
        reads_features = bool(feature_columns(feature_set, with_features=with_features))
        super().__init__(
            beta_loss=beta_loss,
            lr=lr,
            optimizer=optimizer,
            with_features=reads_features,
            n_classes=n_classes,
        )
        if adam_betas not in PAPER_ADAM_BETAS:
            valid = ", ".join(PAPER_ADAM_BETAS)
            raise ValueError(f"adam_betas must be one of {valid}, got {adam_betas!r}")
        if augment_seed < 0:
            raise ValueError(f"augment_seed must be >= 0, got {augment_seed}")
        if not (aug_noise >= 0 and aug_feature_jitter >= 0):
            raise ValueError("aug_noise and aug_feature_jitter must be >= 0")
        if not 0 <= synthetic_bogus < 1:
            raise ValueError(f"synthetic_bogus must be in [0, 1), got {synthetic_bogus}")
        if not 0 <= aug_min_r2 <= 1:
            raise ValueError(f"aug_min_r2 must be in [0, 1], got {aug_min_r2}")
        if not 0 <= flux_rescale <= 1:
            raise ValueError(f"flux_rescale must be in [0, 1], got {flux_rescale}")
        if not 0 < flux_k_min <= flux_k_max:
            raise ValueError("flux_k_min and flux_k_max must satisfy 0 < min <= max")
        _check_cd21_augment(synthetic_bogus, flux_rescale, context, n_classes)
        if paper_arch in CONVNEXT_ARCHS and (aug_noise or synthetic_bogus or flux_rescale):
            # C5's noise, C6 and C7 are defined on the 21x21 crop (ADR 0022).
            raise ValueError(
                f"aug_noise, synthetic_bogus and flux_rescale work on the 21x21 crop, not on "
                f"the whole stamps of {paper_arch}"
            )
        self.adam_betas = adam_betas
        self.rot90_augment = rot90_augment
        self.augment_seed = augment_seed
        self.aug_noise = aug_noise
        self.aug_feature_jitter = aug_feature_jitter
        self.synthetic_bogus = synthetic_bogus
        self.aug_min_r2 = aug_min_r2
        self.flux_rescale = flux_rescale
        self.flux_k_min = flux_k_min
        self.flux_k_max = flux_k_max
        #: Whether any training augmentation is on (the run then plans it on the host).
        self.augmenting = bool(
            rot90_augment or aug_noise or aug_feature_jitter or synthetic_bogus or flux_rescale
        )
        self.model = build_paper_model(
            paper_arch,
            group_fields=group_fields,
            context=context,
            with_features=with_features,
            dropout_active=dropout_active,
            drop_rate=drop_rate,
            n_classes=n_classes,
            feature_set=feature_set,
            hybrid_dim=hybrid_dim,
            hybrid_depth=hybrid_depth,
            hybrid_heads=hybrid_heads,
            hybrid_readout=hybrid_readout,
        )
        # Flags at their defaults stay out of the hyperparameters, so the faithful module's
        # hparams.yaml and checkpoints are what they were before the flags existed.
        hybrid = {
            "hybrid_dim": hybrid_dim,
            "hybrid_depth": hybrid_depth,
            "hybrid_heads": hybrid_heads,
            "hybrid_readout": hybrid_readout,
        }
        defaults = [key for key, value in hybrid.items() if value == HYBRID_FLAGS[key]]
        defaults += ["feature_set"] if feature_set == "all" else []
        self.save_hyperparameters(ignore=_hparams_ignore(n_classes, *defaults))

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Crops (B, C, S, S) and features -> (B, n_classes) logits.

        S is 21 (C: 3, or 6 with context; the hybrids too), or 63 for the ConvNeXt archs
        (C: 3).
        """
        out: torch.Tensor = self.model(x_img, x_feat if self.with_features else None)
        return out

    def rotation_steps(self, n: int, step: int) -> torch.Tensor:
        """The ``k`` (int64, CPU, values 0..3) of ``n`` training stamps at global step ``step``.

        Drawn from a NumPy generator seeded with ``[augment_seed, step]`` only (its
        ``SeedSequence`` mixes both into the whole state, unlike a 32-bit torch seed), so the
        same run on another device, or resumed, rotates the same stamps the same way.
        """
        rng = np.random.default_rng([self.augment_seed, step])
        return torch.from_numpy(rng.integers(0, 4, size=n, dtype=np.int64))

    def training_step(
        self, batch: ClassifierBatch | AugmentBatch | PlannedBatch, batch_idx: int
    ) -> torch.Tensor:
        """One optimisation step, on the augmented batch (the batch itself when all off).

        A batch whose last element is a dict carries its augmentation plan, computed and
        uploaded ahead by the training loader (``DeviceTensorDataModule(planner=
        augment_plan)``): only :meth:`apply_plan` runs here, with no host round trip.
        Otherwise :meth:`augmented_batch` plans and applies in the step.

        Raises:
            ValueError: as :meth:`augmented_batch`.
        """
        if isinstance(batch[-1], dict):
            plan = batch[-1]
            rows = cast(ClassifierBatch | AugmentBatch, tuple(batch[:-1]))
            return self._step(self.apply_plan(rows, plan), "train")
        return self._step(
            self.augmented_batch(cast(ClassifierBatch | AugmentBatch, batch)), "train"
        )

    def augmented_batch(self, batch: ClassifierBatch | AugmentBatch) -> ClassifierBatch:
        """The training batch after C6 / C7 / C5, the rotation and the jitter, when enabled.

        :meth:`augment_plan` on a host copy of ``batch`` at the current global step, then
        :meth:`apply_plan` on ``batch`` (the in-step path, with host round trips; the
        training loader normally plans ahead). Returns ``(images, features, one-hot
        targets)``; ``batch`` is not modified.

        Raises:
            ValueError: as :meth:`augment_plan`.
        """
        host = cast(ClassifierBatch | AugmentBatch, tuple(t.detach().cpu() for t in batch))
        plan = self.augment_plan(host, self.global_step)
        return self.apply_plan(batch, upload_plan(plan, batch[0].device))

    def augment_plan(self, batch: ClassifierBatch | AugmentBatch, step: int) -> dict[str, Any]:
        """The host part of the augmentation of a training batch at global step ``step``.

        ``batch`` is on the CPU (the Train rows of the step: crops, features, one-hot
        targets and, for C5 / C6 / C7, the stamp fits). Every draw and decision of C6, C7,
        C5's noise, the rotation and the jitter is made here, with the same NumPy calls and
        float64 arithmetic as the in-step functions (``class_augment.plan_*``,
        ``flux_rescale.plan_flux``), from ``[augment_seed, step, stream]`` (rotation:
        ``[augment_seed, step]``): device-independent and exact on resume (ADR 0006, 0007).
        Returns a dict of CPU tensors (keys ``c6``, ``c7``, ``c5``, ``rot``, ``jitter`` for
        the augmentations that change the batch), for :meth:`apply_plan`.

        The class-aware augmentation runs **before** the rotation: the stamp fits, and C7's
        PSF centroid in particular, describe the unrotated crop (``fit_central_psf``), so a
        rotated off-centre transient would get its rescaled flux at the wrong place (found in
        review of PR #22). C5 / C6 are rotation-agnostic, so the order only matters for C7.

        Raises:
            ValueError: if ``aug_noise``, ``synthetic_bogus`` or ``flux_rescale`` is on and
                the batch has no stamp fits (a fourth element), or ``flux_rescale`` is on and
                the fits have no PSF columns.
        """
        images, features = batch[0], batch[1]
        plan: dict[str, Any] = {}
        if self.aug_noise or self.synthetic_bogus or self.flux_rescale:
            if len(batch) != 4:
                msg = (
                    "aug_noise, synthetic_bogus and flux_rescale need the stamp fits in the "
                    "training batches: build the Train TensorSplit with fits="
                    "class_augment.stamp_fits (plus flux_rescale.fit_central_psf for C7)"
                )
                raise ValueError(msg)
            plan.update(self._class_plan(batch, step))
        if self.rot90_augment:
            plan["rot"] = self.rotation_steps(images.shape[0], step)
        if self.aug_feature_jitter:
            rng = augment_rng(self.augment_seed, step, JITTER_STREAM)
            plan["jitter"] = plan_jitter(tuple(features.shape), self.aug_feature_jitter, rng)
        return plan

    def _class_plan(self, batch: AugmentBatch, step: int) -> dict[str, Any]:
        """C6, C7 and C5's noise of :meth:`augment_plan`, in that order (host tensors)."""
        images, features, targets, fits_tensor = batch
        fits = fits_tensor.numpy()
        plan: dict[str, Any] = {}
        replaced = None
        if self.synthetic_bogus:
            rng = augment_rng(self.augment_seed, step, SYNTHETIC_STREAM)
            c6 = plan_synthetic(
                images, features, targets, fits, self.synthetic_bogus, rng, min_r2=self.aug_min_r2
            )
            replaced = c6.pop("replaced")
            if replaced.any():
                plan["c6"] = c6
        if self.flux_rescale:
            rng = augment_rng(self.augment_seed, step, FLUX_STREAM)
            c7 = plan_flux(
                images,
                features,
                targets,
                fits,
                self.flux_rescale,
                (self.flux_k_min, self.flux_k_max),
                rng,
                skip=replaced,
            )
            c7.pop("k")
            if c7:
                plan["c7"] = c7
        if self.aug_noise:
            rng = augment_rng(self.augment_seed, step, NOISE_STREAM)
            c5 = plan_noise(
                fits, images.shape[-1], self.aug_noise, rng, min_r2=self.aug_min_r2, skip=replaced
            )
            if c5 is not None:
                plan["c5"] = c5
        return plan

    def apply_plan(
        self, batch: ClassifierBatch | AugmentBatch, plan: dict[str, Any]
    ) -> ClassifierBatch:
        """The device part: :meth:`augment_plan`'s ``plan``, on ``batch``'s device, applied.

        The same torch operations as the in-step functions (``class_augment.apply_*``,
        ``flux_rescale.apply_flux``, the rotation, the jitter's addition), so the result is
        bit for bit the in-step one on every device; nothing is copied back to the host.
        """
        images, features, targets = batch[0], batch[1], batch[2]
        if "c6" in plan:
            images, features, targets = apply_synthetic(images, features, targets, plan["c6"])
        if "c7" in plan:
            images, features = apply_flux(images, features, plan["c7"])
        if "c5" in plan and len(batch) == 4:
            images = apply_noise(images, batch[3], plan["c5"])
        if "rot" in plan:
            rotations = torch.stack([torch.rot90(images, j, dims=(2, 3)) for j in range(4)])
            images = rotations[plan["rot"], torch.arange(images.shape[0], device=images.device)]
        if "jitter" in plan:
            features = features + plan["jitter"].to(features.dtype)
        return images, features, targets

    def class_augment(
        self, x_img: torch.Tensor, x_feat: torch.Tensor, targets: torch.Tensor, fits: torch.Tensor
    ) -> ClassifierBatch:
        """C6, C7, then C5's noise on a training batch, drawn for the current global step.

        Synthetic Bogus rows (``synthetic_bogus``) are not rescaled nor noised again. Returns
        the new ``(images, features, targets)``; the inputs are not modified.
        """
        step = self.global_step
        replaced = None
        if self.synthetic_bogus:
            rng = augment_rng(self.augment_seed, step, SYNTHETIC_STREAM)
            x_img, x_feat, targets, replaced = synthetic_bogus(
                x_img, x_feat, targets, fits, self.synthetic_bogus, rng, min_r2=self.aug_min_r2
            )
        if self.flux_rescale:
            rng = augment_rng(self.augment_seed, step, FLUX_STREAM)
            x_img, x_feat, _ = flux_rescale(
                x_img,
                x_feat,
                targets,
                fits,
                self.flux_rescale,
                (self.flux_k_min, self.flux_k_max),
                rng,
                skip=replaced,
            )
        if self.aug_noise:
            rng = augment_rng(self.augment_seed, step, NOISE_STREAM)
            x_img, _ = noise_augment(
                x_img, fits, self.aug_noise, rng, min_r2=self.aug_min_r2, skip=replaced
            )
        return x_img, x_feat, targets

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """The optimizer with the betas ``adam_betas`` names, no LR schedule."""
        return build_optimizer(
            self.optimizer_name,
            self.trainable_parameters(),
            self.lr,
            adam_betas=PAPER_ADAM_BETAS[self.adam_betas],
        )


class SupervisedCEResNetModule(ClassifierModule):
    """A ResNet classifier trained from scratch (legacy ``Supervised_Cross_Entropy_Resnet``).

    ``resnet_model`` (``resnet18``, ``resnet34``, ``resnet50`` or ``resnet152``, no
    pretrained weights) with ``fc`` removed, then ``LinearClassifier`` on its output and,
    with features, the batch-normalised 23 features.

    Behaviour changes: the unused legacy ``drop_rate`` argument is gone; ``resnet34`` is
    accepted; an unknown name raises ``ValueError`` (legacy: ``UnboundLocalError``).

    Raises:
        ValueError: if ``resnet_model`` is not a ResNet name.
    """

    def __init__(
        self,
        *,
        beta_loss: float,
        lr: float,
        optimizer: str,
        with_features: bool = True,
        resnet_model: str = "resnet18",
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(
            beta_loss=beta_loss,
            lr=lr,
            optimizer=optimizer,
            with_features=with_features,
            n_classes=n_classes,
        )
        if not resnet_model.startswith("resnet"):
            msg = f"resnet_model must be a ResNet (resnet18/34/50/152), got {resnet_model!r}"
            raise ValueError(msg)
        encoder = build_encoder(resnet_model)
        head = LinearClassifier(_n_inputs(encoder.out_features, with_features), n_classes)
        self.model = EncoderClassifier(encoder.module, head, with_features=with_features)
        self.save_hyperparameters(ignore=_hparams_ignore(n_classes))

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, n_classes) logits."""
        out: torch.Tensor = self.model(x_img, x_feat)
        return out


class SimCLRModule(L.LightningModule):
    """SimCLR pretraining (legacy ``SimCLR``): ``SimCLRNet`` trained with ``NT_Xent``.

    Each step takes a batch of view pairs, projects both views and computes
    ``NT_Xent(temperature, legacy_denominator=nt_xent_legacy_denominator)``. The optimizer
    (``LARS`` in every legacy run) follows ``cosine_warmup_scheduler``, stepped once per
    epoch. At each epoch end it logs ``Loss_encoder/Train`` and ``loss_train_enc`` (the mean
    of the batch losses; the legacy monitor), ``lr`` (the learning rate of that epoch),
    ``positive_mean`` and ``negative_mean`` (mean cosine similarities).

    ``nt_xent_legacy_denominator`` is the run setting ``Settings.nt_xent_legacy_denominator``
    (ADR 0002); ``lars_legacy_trust_ratio`` is LARS's ``legacy_trust_ratio``. Both default to
    the legacy behaviour.

    Behaviour changes: ``encoder_name`` goes through ``build_encoder`` (every ``stamp<k>``
    and ``pstamps`` work); any optimizer name of the factory works (legacy: only ``LARS``);
    the unused ``method`` argument and the loader arguments (``augmentation``,
    ``batch_size_encoder``, data paths) are gone (see ``ContrastiveDataModule``); ``lr`` is
    the rate used during the epoch (the legacy value was read after the scheduler step, so
    it was the next epoch's rate); there is no validation or test loop (the legacy loaders
    fed only the t-SNE visualisation, which is deferred).

    Raises:
        UnknownOptimizerError: if ``optimizer_encoder`` is not a known name.
        ValueError: for an unknown ``encoder_name`` or a non-positive ``temperature``.
    """

    MONITORABLE: ClassVar[frozenset[str]] = frozenset({"loss_train_enc"})

    def __init__(
        self,
        *,
        encoder_name: str,
        image_size: int,
        projection_dim: int,
        temperature: float,
        lr_encoder: float,
        optimizer_encoder: str = "LARS",
        nt_xent_legacy_denominator: bool = True,
        lars_legacy_trust_ratio: bool = True,
    ) -> None:
        super().__init__()
        check_optimizer_name(optimizer_encoder)
        self.encoder_name = encoder_name
        self.image_size = image_size
        self.lr_encoder = lr_encoder
        self.optimizer_encoder = optimizer_encoder
        self.lars_legacy_trust_ratio = lars_legacy_trust_ratio
        self.criterion_encoder = NT_Xent(temperature, legacy_denominator=nt_xent_legacy_denominator)
        encoder = build_encoder(encoder_name)
        #: Dimension of the encoder's representation ``h``.
        self.n_features_encoder = encoder.out_features
        self.clr = SimCLRNet(encoder.module, encoder.out_features, projection_dim)
        self._outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._epoch_lr = float("nan")
        self.save_hyperparameters()

    @property
    def encoder(self) -> nn.Module:
        """The encoder (images -> representations ``h``), shared with ``clr``."""
        return self.clr.encoder

    def forward(
        self, x_img_i: torch.Tensor, x_img_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two views -> ``(h_i, h_j, z_i, z_j)``."""
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = self.clr(
            x_img_i, x_img_j
        )
        return out

    def training_step(self, batch: PairBatch, batch_idx: int) -> torch.Tensor:
        """NT-Xent of the batch's projections; buffers the loss and the similarities."""
        x_img_i, x_img_j = batch
        _, _, z_i, z_j = self.clr(x_img_i, x_img_j)
        losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.criterion_encoder(z_i, z_j)
        loss, pos, neg = losses
        self._outputs.append((loss.detach(), pos, neg))
        return loss

    def on_train_epoch_start(self) -> None:
        """Clear the buffer and record the learning rate this epoch trains with."""
        self._outputs.clear()
        self._epoch_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])

    def on_train_epoch_end(self) -> None:
        """Log the epoch's mean loss, learning rate and similarities; clear the buffer."""
        outputs, self._outputs = self._outputs, []
        if not outputs:
            return
        loss, pos, neg = (torch.stack(values).mean() for values in zip(*outputs, strict=True))
        self.log("Loss_encoder/Train", loss)
        self.log("loss_train_enc", loss, logger=False, prog_bar=True)
        self.log("lr", self._epoch_lr, prog_bar=True)
        self.log("positive_mean", pos)
        self.log("negative_mean", neg)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """The encoder optimizer and the legacy cosine-with-warmup schedule (per epoch)."""
        optimizer = build_optimizer(
            self.optimizer_encoder,
            self.clr.parameters(),
            self.lr_encoder,
            lars_legacy_trust_ratio=self.lars_legacy_trust_ratio,
        )
        scheduler = cosine_warmup_scheduler(optimizer, self.lr_encoder)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr_cur",
            },
        }


def _as_pretrained(pretrained: SimCLRModule | Mapping[str, Any]) -> SimCLRModule:
    """The SimCLR module itself, or one rebuilt (random weights) from its hyperparameters.

    The mapping form is what ``load_from_checkpoint`` passes back: the downstream modules
    store ``pretrained.hparams`` (plain values) instead of the module, and the checkpoint's
    ``state_dict`` then restores the encoder weights.
    """
    if isinstance(pretrained, SimCLRModule):
        return pretrained
    return SimCLRModule(**pretrained)


class _DownstreamModule(ClassifierModule):
    """A classifier on a pretrained encoder: stores the encoder spec, checks the image size."""

    image_size: int

    def _save_pretrained_spec(self, pretrained: SimCLRModule) -> None:
        # The module itself cannot go into the hyperparameters (it is not a plain value);
        # its own hyperparameters are enough to rebuild its architecture on load.
        self.save_hyperparameters(ignore=_hparams_ignore(self.n_classes, "pretrained"))
        self.hparams["pretrained"] = dict(pretrained.hparams)

    def setup(self, stage: str) -> None:
        """Fail early if the data module crops stamps to another size than the encoder's.

        Raises:
            ValueError: if the trainer's data module has an ``image_size`` different from
                the pretrained encoder's (the stamp encoders would fail later, the ResNets
                would silently train on another input).
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        size = getattr(datamodule, "image_size", None)
        if size is not None and size != self.image_size:
            msg = (
                f"the data module's image_size is {size}, but the pretrained encoder was "
                f"trained on {self.image_size} px"
            )
            raise ValueError(msg)


class LinearEvalModule(_DownstreamModule):
    """Linear evaluation of a pretrained SimCLR encoder (legacy ``Linear_classifier``).

    The encoder is a frozen copy of ``pretrained.encoder``: it stays in eval mode (also
    while training) and runs without gradient. Its output is divided by its L2 norm and,
    with features, concatenated with the 23 raw features (no batch norm, as in the legacy
    code); ``LinearClassifier`` maps that to ``n_classes`` logits and is the only thing
    trained.

    **Legacy loss, kept:** the legacy module passed ``softmax(logits)``, not the logits, to
    ``P_stamps_loss``, which applies its own softmax. :meth:`compute_loss` does the same, so
    results stay comparable; it is a known quirk, not a fix to make in phase 1.

    ``pretrained`` is the SimCLR module, or its hyperparameters (what
    ``load_from_checkpoint`` passes: the checkpoint stores them as ``hparams["pretrained"]``
    and its ``state_dict`` restores the weights).

    Behaviour changes: the encoder is copied, so training never touches ``pretrained`` (the
    legacy module held the whole SimCLR module); the image size is ``pretrained.image_size``
    and a data module with another ``image_size`` is rejected in ``setup``; the unused legacy
    ``drop_rate`` and ``batch_size`` arguments are gone.
    """

    def __init__(
        self,
        pretrained: SimCLRModule | Mapping[str, Any],
        *,
        beta_loss: float,
        lr: float,
        optimizer: str,
        with_features: bool,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(
            beta_loss=beta_loss,
            lr=lr,
            optimizer=optimizer,
            with_features=with_features,
            n_classes=n_classes,
        )
        pretrained = _as_pretrained(pretrained)
        self.image_size = pretrained.image_size
        self.encoder = copy.deepcopy(pretrained.encoder).requires_grad_(False).eval()
        n_inputs = _n_inputs(pretrained.n_features_encoder, with_features)
        self.classifier = LinearClassifier(n_inputs, n_classes)
        self._save_pretrained_spec(pretrained)

    def train(self, mode: bool = True) -> LinearEvalModule:
        """Set the train/eval mode of the head; the frozen encoder always stays in eval."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, n_classes) logits of the linear head."""
        with torch.no_grad():
            h = self.encoder(x_img)
            h = h * (1 / torch.norm(h, dim=1, keepdim=True))
        x = torch.cat((h, x_feat), dim=1) if self.with_features else h
        out: torch.Tensor = self.classifier(x)
        return out

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """The legacy loss: ``P_stamps_loss(softmax(logits), targets)`` (see the class doc)."""
        loss: torch.Tensor = self.criterion(logits.softmax(dim=1), targets)
        return loss

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Only the linear head's parameters."""
        return self.classifier.parameters()


class FineTuneModule(_DownstreamModule):
    """Fine-tuning of a pretrained SimCLR encoder with a new head (legacy ``Fine_tuning``).

    A copy of ``pretrained.encoder`` (no normalisation of its output) feeds an
    :class:`EncoderClassifier`; everything is trained. The head depends on the encoder:

    - ResNets: ``LinearClassifier`` (legacy);
    - stamp encoders (``pstamps``, ``stamp<k>``): ``ThreeLayerClassifier`` with hidden
      width 64 and dropout ``drop_rate`` (legacy for ``pstamps``).

    Behaviour changes: ``stamp<k>`` encoders get the ``pstamps`` head (legacy: no branch,
    so ``self.model`` was never set and the module failed); the encoder is copied, so
    ``pretrained`` is not modified (the legacy module trained the caller's encoder in place).
    As in :class:`LinearEvalModule`, ``pretrained`` may be the SimCLR hyperparameters (for
    ``load_from_checkpoint``), and the data module's ``image_size`` must match.
    """

    def __init__(
        self,
        pretrained: SimCLRModule | Mapping[str, Any],
        *,
        drop_rate: float,
        beta_loss: float,
        lr: float,
        optimizer: str,
        with_features: bool = True,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__(
            beta_loss=beta_loss,
            lr=lr,
            optimizer=optimizer,
            with_features=with_features,
            n_classes=n_classes,
        )
        pretrained = _as_pretrained(pretrained)
        self.image_size = pretrained.image_size
        n_inputs = _n_inputs(pretrained.n_features_encoder, with_features)
        head: nn.Module
        if pretrained.encoder_name.startswith("resnet"):
            head = LinearClassifier(n_inputs, n_classes)
        else:
            head = ThreeLayerClassifier(
                n_inputs, _FINE_TUNE_HIDDEN, _FINE_TUNE_HIDDEN, n_classes, drop_rate
            )
        encoder = copy.deepcopy(pretrained.encoder)
        self.model = EncoderClassifier(encoder, head, with_features=with_features)
        self._save_pretrained_spec(pretrained)

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, n_classes) logits."""
        out: torch.Tensor = self.model(x_img, x_feat)
        return out
