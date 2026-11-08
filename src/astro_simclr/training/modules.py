"""The Lightning 2 training modules: SimCLR pretraining and the four classifiers (task 1.6).

Ports of the legacy ``utils/training.py`` (PyTorch Lightning 1.4):

| Class | Legacy class | Trains |
|---|---|---|
| :class:`SimCLRModule` | ``SimCLR`` | encoder + projector with ``NT_Xent`` (LARS + cosine warmup) |
| :class:`LinearEvalModule` | ``Linear_classifier`` | a linear head on a frozen pretrained encoder |
| :class:`FineTuneModule` | ``Fine_tuning`` | the pretrained encoder and a new head |
| :class:`SupervisedCEModule` | ``Supervised_Cross_Entropy`` | ``StampClassifier`` from scratch |
| :class:`SupervisedCEResNetModule` | ``Supervised_Cross_Entropy_Resnet`` | a ResNet + linear head |

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
from typing import Any, ClassVar, Final

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from astro_simclr.losses import NT_Xent, P_stamps_loss
from astro_simclr.models import (
    N_CLASSES,
    N_FEATURES,
    LinearClassifier,
    SimCLRNet,
    StampClassifier,
    ThreeLayerClassifier,
    build_encoder,
)
from astro_simclr.optim import cosine_warmup_scheduler
from astro_simclr.training.metrics import METRIC_NAMES, classification_metrics, confusion_matrix
from astro_simclr.training.optimizers import build_optimizer, check_optimizer_name

__all__ = [
    "ClassifierModule",
    "EncoderClassifier",
    "FineTuneModule",
    "LinearEvalModule",
    "SimCLRModule",
    "SupervisedCEModule",
    "SupervisedCEResNetModule",
]

#: A classifier batch: images (B, 3, S, S), features (B, 23), one-hot labels (B, 5).
ClassifierBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
#: A SimCLR batch: the two views (B, 3, S, S) of each stamp.
PairBatch = tuple[torch.Tensor, torch.Tensor]

#: Stage -> the suffix of its legacy TensorBoard names.
_STAGE_NAMES: Final = {"train": "Train", "val": "Validation", "test": "Test"}
#: Hidden width of the fine-tuning MLP head of the stamp encoders (legacy value).
_FINE_TUNE_HIDDEN: Final = 64


@dataclass(frozen=True, slots=True)
class _StepOutput:
    """What one classifier step leaves in its epoch buffer (all detached)."""

    y_pred: torch.Tensor
    y_true: torch.Tensor
    loss: torch.Tensor


class ClassifierModule(L.LightningModule):
    """Shared steps, epoch-end metrics and optimizer of the four classifier modules.

    Each step computes ``loss = compute_loss(logits, one_hot_targets)`` with
    ``P_stamps_loss(beta_loss)`` and buffers ``argmax(softmax(logits))``, the true labels
    and the loss. At the end of each train, validation and test epoch it logs, under the
    legacy names, ``Loss/<Stage>`` (mean of the batch losses), ``Accuracy/<Stage>`` (micro),
    ``Precision/<Stage>`` and ``Recall/<Stage>`` (macro). Validation and test also log
    ``accuracy_val`` / ``loss_val`` (``accuracy_test`` / ``loss_test``) on the progress bar,
    and store the row-normalised confusion matrix, on the CPU, in
    ``confusion_matrices["val"]`` / ``["test"]``.

    Subclasses define ``forward(x_img, x_feat)``, and may override :meth:`compute_loss`
    and :meth:`trainable_parameters`.

    Raises:
        UnknownOptimizerError: if ``optimizer`` is not a known name (at construction, not
            when training starts).
    """

    #: Metrics that ``Settings.monitor`` may name for this module.
    MONITORABLE: ClassVar[frozenset[str]] = frozenset({"accuracy_val", "loss_val"})

    def __init__(self, *, beta_loss: float, lr: float, optimizer: str, with_features: bool) -> None:
        super().__init__()
        check_optimizer_name(optimizer)
        self.beta_loss = beta_loss
        self.lr = lr
        self.optimizer_name = optimizer
        self.with_features = with_features
        self.criterion = P_stamps_loss(beta_loss)
        self._outputs: dict[str, list[_StepOutput]] = {stage: [] for stage in _STAGE_NAMES}
        #: The last confusion matrix of each evaluation stage (``"val"``, ``"test"``).
        self.confusion_matrices: dict[str, torch.Tensor] = {}

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
        y_pred = logits.softmax(dim=1).argmax(dim=1)
        y_true = targets.argmax(dim=1)
        self._outputs[stage].append(_StepOutput(y_pred.detach(), y_true.detach(), loss.detach()))
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
        metrics = classification_metrics(y_pred, y_true)
        name = _STAGE_NAMES[stage]
        self.log(f"Loss/{name}", loss)
        for key in METRIC_NAMES:
            self.log(f"{key.capitalize()}/{name}", metrics[key])
        if stage != "train":
            # The legacy monitor names, on the progress bar only (as in the legacy code).
            self.log(f"accuracy_{stage}", metrics["accuracy"], logger=False, prog_bar=True)
            self.log(f"loss_{stage}", loss, logger=False, prog_bar=True)
            self.confusion_matrices[stage] = confusion_matrix(y_pred, y_true).cpu()

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
    ) -> None:
        super().__init__(
            beta_loss=beta_loss, lr=lr, optimizer=optimizer, with_features=with_features
        )
        self.model = StampClassifier(drop_rate, with_features)
        self.save_hyperparameters()

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, 5) logits."""
        out: torch.Tensor = self.model(x_img, x_feat if self.with_features else None)
        return out


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
    ) -> None:
        super().__init__(
            beta_loss=beta_loss, lr=lr, optimizer=optimizer, with_features=with_features
        )
        if not resnet_model.startswith("resnet"):
            msg = f"resnet_model must be a ResNet (resnet18/34/50/152), got {resnet_model!r}"
            raise ValueError(msg)
        encoder = build_encoder(resnet_model)
        head = LinearClassifier(_n_inputs(encoder.out_features, with_features), N_CLASSES)
        self.model = EncoderClassifier(encoder.module, head, with_features=with_features)
        self.save_hyperparameters()

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, 5) logits."""
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
        self.save_hyperparameters(ignore=["pretrained"])
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
    code); ``LinearClassifier`` maps that to 5 logits and is the only thing trained.

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
    ) -> None:
        super().__init__(
            beta_loss=beta_loss, lr=lr, optimizer=optimizer, with_features=with_features
        )
        pretrained = _as_pretrained(pretrained)
        self.image_size = pretrained.image_size
        self.encoder = copy.deepcopy(pretrained.encoder).requires_grad_(False).eval()
        n_inputs = _n_inputs(pretrained.n_features_encoder, with_features)
        self.classifier = LinearClassifier(n_inputs, N_CLASSES)
        self._save_pretrained_spec(pretrained)

    def train(self, mode: bool = True) -> LinearEvalModule:
        """Set the train/eval mode of the head; the frozen encoder always stays in eval."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, 5) logits of the linear head."""
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
    ) -> None:
        super().__init__(
            beta_loss=beta_loss, lr=lr, optimizer=optimizer, with_features=with_features
        )
        pretrained = _as_pretrained(pretrained)
        self.image_size = pretrained.image_size
        n_inputs = _n_inputs(pretrained.n_features_encoder, with_features)
        head: nn.Module
        if pretrained.encoder_name.startswith("resnet"):
            head = LinearClassifier(n_inputs, N_CLASSES)
        else:
            head = ThreeLayerClassifier(
                n_inputs, _FINE_TUNE_HIDDEN, _FINE_TUNE_HIDDEN, N_CLASSES, drop_rate
            )
        encoder = copy.deepcopy(pretrained.encoder)
        self.model = EncoderClassifier(encoder, head, with_features=with_features)
        self._save_pretrained_spec(pretrained)

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        """Images and features -> (B, 5) logits."""
        out: torch.Tensor = self.model(x_img, x_feat)
        return out
