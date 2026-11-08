"""Tests of the Lightning modules (astro_simclr.training.modules) on the synthetic data.

Acceptance of plan task 1.6: ``fast_dev_run`` passes for every module, a seeded two-epoch
SimCLR run gives the same loss twice, the LR scheduler steps once per epoch, and the logged
accuracy is the legacy micro value. Training runs on the CPU, and on MPS/CUDA when present.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from _training_support import ACCELERATORS
from torch.utils.data import DataLoader, TensorDataset

from astro_simclr.config import Settings
from astro_simclr.losses import P_stamps_loss
from astro_simclr.models import (
    LinearClassifier,
    SimCLRNet,
    StampClassifier,
    ThreeLayerClassifier,
    build_encoder,
)
from astro_simclr.optim import cosine_warmup_lr
from astro_simclr.training import (
    ClassifierDataModule,
    ClassifierModule,
    ContrastiveDataModule,
    FineTuneModule,
    LinearEvalModule,
    SimCLRModule,
    SupervisedCEModule,
    SupervisedCEResNetModule,
    build_callbacks,
    build_trainer,
    seed_run,
)

CLASSIFIER_HPARAMS: dict[str, Any] = {"beta_loss": 0.01, "lr": 0.002, "optimizer": "AdamW"}


def _settings(tmp_path: Path, accelerator: str = "cpu", **changes: Any) -> Settings:
    return Settings(accelerator=accelerator, output_dir=tmp_path, **changes)


def _quiet_trainer(settings: Settings, **kwargs: Any) -> L.Trainer:
    defaults: dict[str, Any] = {
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
    }
    return build_trainer(settings, **{**defaults, **kwargs})


def _simclr(encoder_name: str = "stamp1", **changes: Any) -> SimCLRModule:
    hparams: dict[str, Any] = {
        "encoder_name": encoder_name,
        "image_size": 27,
        "projection_dim": 16,
        "temperature": 0.1791,
        "lr_encoder": 0.258,
    }
    return SimCLRModule(**{**hparams, **changes})


def _contrastive(path: Path, batch_size: int = 5) -> ContrastiveDataModule:
    return ContrastiveDataModule(path, image_size=27, batch_size=batch_size, augmentation="astro7")


def _classifier_data(path: Path, augmentation: str = "rotation_v3") -> ClassifierDataModule:
    return ClassifierDataModule(
        path, image_size=27, batch_size=5, eval_batch_size=5, augmentation=augmentation
    )


CLASSIFIERS: dict[str, Callable[[], ClassifierModule]] = {
    "supervised_ce": lambda: SupervisedCEModule(drop_rate=0.2, **CLASSIFIER_HPARAMS),
    "supervised_ce_no_features": lambda: SupervisedCEModule(
        drop_rate=0.2, with_features=False, **CLASSIFIER_HPARAMS
    ),
    "supervised_ce_resnet": lambda: SupervisedCEResNetModule(**CLASSIFIER_HPARAMS),
    "linear_eval": lambda: LinearEvalModule(_simclr(), with_features=True, **CLASSIFIER_HPARAMS),
    "linear_eval_no_features": lambda: LinearEvalModule(
        _simclr("pstamps"), with_features=False, **CLASSIFIER_HPARAMS
    ),
    "fine_tune_stamp": lambda: FineTuneModule(_simclr(), drop_rate=0.5, **CLASSIFIER_HPARAMS),
    "fine_tune_pstamps": lambda: FineTuneModule(
        _simclr("pstamps"), drop_rate=0.5, with_features=False, **CLASSIFIER_HPARAMS
    ),
    "fine_tune_resnet": lambda: FineTuneModule(
        _simclr("resnet18"), drop_rate=0.5, **CLASSIFIER_HPARAMS
    ),
}


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_fast_dev_run_simclr(accelerator: str, tmp_path: Path, pretraining_dir: Path) -> None:
    settings = _settings(tmp_path, accelerator)
    seed_run(settings)
    module = _simclr()
    trainer = _quiet_trainer(settings, fast_dev_run=True)
    trainer.fit(module, datamodule=_contrastive(pretraining_dir))
    assert torch.isfinite(trainer.callback_metrics["loss_train_enc"])


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("name", list(CLASSIFIERS))
def test_fast_dev_run_classifiers(
    name: str, accelerator: str, tmp_path: Path, classifier_dir: Path
) -> None:
    settings = _settings(tmp_path, accelerator)
    seed_run(settings)
    module = CLASSIFIERS[name]()
    datamodule = _classifier_data(classifier_dir)
    trainer = _quiet_trainer(settings, fast_dev_run=True)
    trainer.fit(module, datamodule=datamodule)
    metrics = trainer.callback_metrics
    for key in ("accuracy_val", "loss_val", "Loss/Train", "Accuracy/Train", "Recall/Validation"):
        assert torch.isfinite(metrics[key]), key
    (results,) = trainer.test(module, datamodule=datamodule, verbose=False)
    assert 0.0 <= results["Accuracy/Test"] <= 1.0
    assert module.confusion_matrices["test"].shape == (5, 5)
    assert module.confusion_matrices["val"].device.type == "cpu"


def _run_simclr(tmp_path: Path, path: Path, seed: int, epochs: int = 2) -> list[float]:
    """Train SimCLR for ``epochs`` epochs; return every step's loss and the last epoch mean."""
    settings = _settings(tmp_path, max_epochs=epochs, seed=seed)
    seed_run(settings)
    module = _simclr()
    losses: list[float] = []

    class Record(L.Callback):
        def on_train_batch_end(self, trainer: L.Trainer, *args: Any) -> None:
            outputs = args[1]  # (pl_module, outputs, batch, batch_idx)
            losses.append(float(outputs["loss"] if isinstance(outputs, dict) else outputs))

    trainer = _quiet_trainer(settings, callbacks=[Record()])
    trainer.fit(module, datamodule=_contrastive(path))
    return [*losses, float(trainer.callback_metrics["loss_train_enc"])]


def test_seeded_simclr_run_is_reproducible(tmp_path: Path, pretraining_dir: Path) -> None:
    first = _run_simclr(tmp_path / "a", pretraining_dir, seed=3)
    second = _run_simclr(tmp_path / "b", pretraining_dir, seed=3)
    assert len(first) == 2 * 2 + 1  # 2 epochs of 2 batches, then the epoch mean
    assert first == second  # bit for bit, on the same machine
    assert _run_simclr(tmp_path / "c", pretraining_dir, seed=4) != first


def test_lr_scheduler_steps_once_per_epoch(tmp_path: Path, pretraining_dir: Path) -> None:
    epochs = 7  # the 5-epoch warmup and two epochs of the cosine
    settings = _settings(tmp_path, max_epochs=epochs)
    seed_run(settings)
    module = _simclr()
    seen: list[tuple[int, int, float]] = []
    logged: list[float] = []

    class Record(L.Callback):
        def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            (config,) = trainer.lr_scheduler_configs
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            seen.append((trainer.current_epoch, config.scheduler.last_epoch, lr))

        def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            logged.append(float(trainer.callback_metrics["lr"]))

    trainer = _quiet_trainer(settings, callbacks=[Record()])
    # Batches of 2: five optimizer steps per epoch, so a per-step scheduler would show.
    trainer.fit(module, datamodule=_contrastive(pretraining_dir, batch_size=2))
    assert trainer.global_step == 5 * epochs
    (config,) = trainer.lr_scheduler_configs
    assert (config.interval, config.frequency, config.name) == ("epoch", 1, "lr_cur")
    for epoch, last_epoch, lr in seen:
        assert last_epoch == epoch
        assert lr == pytest.approx(cosine_warmup_lr(epoch, 0.258), abs=1e-12)
    assert [epoch for epoch, _, _ in seen] == list(range(epochs))
    # The logged lr is the rate the last epoch trained with.
    assert logged == [pytest.approx(cosine_warmup_lr(epochs - 1, 0.258))]
    assert config.scheduler.last_epoch == epochs


class _FixedLogits(ClassifierModule):
    """A classifier whose logits are its input features, to test the epoch-end metrics.

    With ``wrong`` set, the logits are rolled by one class, so every prediction that was
    right becomes wrong. ``lr=0``: training never changes anything.
    """

    def __init__(self) -> None:
        super().__init__(beta_loss=0.0, lr=0.0, optimizer="SGD", with_features=True)
        self.unused = torch.nn.Linear(1, 1)
        self.wrong = False

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        logits = x_feat.roll(1, dims=1) if self.wrong else x_feat
        # A zero term through a parameter, so the training loss has a gradient.
        return logits + 0 * self.unused.weight.sum()


def _fixed_loader(n: int = 10, batch_size: int = 4) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Batches whose features are the one-hot true labels: ``_FixedLogits`` gets them right."""
    labels = torch.arange(n) % 5
    one_hot = torch.nn.functional.one_hot(labels, 5)
    return DataLoader(TensorDataset(torch.zeros(n, 1), one_hot.float(), one_hot), batch_size)


def test_each_validation_pass_starts_from_an_empty_buffer(tmp_path: Path) -> None:
    module = _FixedLogits()
    trainer = _quiet_trainer(_settings(tmp_path))
    trainer.validate(module, dataloaders=_fixed_loader(), verbose=False)
    assert trainer.callback_metrics["accuracy_val"].item() == 1.0
    module.wrong = True
    trainer.validate(module, dataloaders=_fixed_loader(), verbose=False)
    # 0, not 0.5: the first pass's predictions are gone.
    assert trainer.callback_metrics["accuracy_val"].item() == 0.0
    assert module.confusion_matrices["val"].diagonal().sum().item() == 0.0


def test_each_training_epoch_starts_from_an_empty_buffer(tmp_path: Path) -> None:
    accuracies: list[float] = []

    class Flip(L.Callback):
        def on_train_epoch_start(self, trainer: L.Trainer, pl_module: Any) -> None:
            if trainer.current_epoch == 1:
                accuracies.append(float(trainer.callback_metrics["Accuracy/Train"]))
            pl_module.wrong = trainer.current_epoch == 1

        def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            accuracies.append(float(trainer.callback_metrics["Accuracy/Train"]))

    trainer = _quiet_trainer(
        _settings(tmp_path, max_epochs=2), callbacks=[Flip()], limit_val_batches=0
    )
    trainer.fit(_FixedLogits(), train_dataloaders=_fixed_loader())
    assert accuracies == [1.0, 0.0]


def test_early_stopping_stops_a_flat_run(tmp_path: Path) -> None:
    # accuracy_val is constant, so with patience 1 the run stops after its second epoch.
    settings = _settings(tmp_path, max_epochs=10, patience=1)
    early, checkpoint = build_callbacks(
        settings, _FixedLogits.MONITORABLE, dirpath=tmp_path / "ckpt"
    )
    trainer = _quiet_trainer(settings, callbacks=[early, checkpoint], enable_checkpointing=True)
    trainer.fit(_FixedLogits(), train_dataloaders=_fixed_loader(), val_dataloaders=_fixed_loader())
    assert trainer.current_epoch == 2
    assert early.stopped_epoch == 1
    assert checkpoint.best_model_score is not None
    assert checkpoint.best_model_score.item() == 1.0


def test_logged_accuracy_is_the_legacy_micro_value(tmp_path: Path) -> None:
    # Imbalanced and fixed: 9 samples of class 0, 3 of class 1, one each of 2, 3 and 4.
    y_true = torch.tensor([0] * 9 + [1] * 3 + [2, 3, 4])
    y_pred = torch.tensor([0] * 8 + [3] + [1, 0, 0] + [2, 0, 0])
    logits = torch.nn.functional.one_hot(y_pred, 5).float()
    targets = torch.nn.functional.one_hot(y_true, 5)
    data = TensorDataset(torch.zeros(len(y_true), 1), logits, targets)
    module = _FixedLogits()
    trainer = _quiet_trainer(_settings(tmp_path))
    # Batches of 4: the epoch value must come from all 15 predictions, not a batch mean.
    trainer.validate(module, dataloaders=DataLoader(data, batch_size=4), verbose=False)
    legacy_micro = (y_pred == y_true).float().mean().item()  # 10 / 15
    assert trainer.callback_metrics["accuracy_val"].item() == pytest.approx(legacy_micro)
    assert trainer.callback_metrics["Accuracy/Validation"].item() == pytest.approx(legacy_micro)
    expected_cm = torch.zeros(5, 5)
    expected_cm[0, 0], expected_cm[0, 3] = 8 / 9, 1 / 9
    expected_cm[1, 1], expected_cm[1, 0] = 1 / 3, 2 / 3
    expected_cm[2, 2], expected_cm[3, 0], expected_cm[4, 0] = 1.0, 1.0, 1.0
    torch.testing.assert_close(module.confusion_matrices["val"], expected_cm)


def test_modules_initialise_like_the_legacy_order() -> None:
    # Nothing that draws random numbers runs before the network, so a seed gives the same
    # initial weights as the bare model (and as the legacy module).
    for build_module, build_bare in (
        (
            lambda: SupervisedCEModule(drop_rate=0.2, **CLASSIFIER_HPARAMS).model,
            lambda: StampClassifier(0.2, True),
        ),
        (
            lambda: _simclr(projection_dim=8).clr,
            lambda: SimCLRNet(build_encoder("stamp1").module, 128, 8),
        ),
    ):
        L.seed_everything(5)
        module_state = build_module().state_dict()
        L.seed_everything(5)
        bare_state = build_bare().state_dict()
        assert module_state.keys() == bare_state.keys()
        for key, value in module_state.items():
            torch.testing.assert_close(value, bare_state[key])


def test_linear_eval_trains_only_the_head(tmp_path: Path, classifier_dir: Path) -> None:
    settings = _settings(tmp_path, max_epochs=2)
    seed_run(settings)
    pretrained = _simclr()
    pretrained_state = {k: v.clone() for k, v in pretrained.state_dict().items()}
    module = LinearEvalModule(pretrained, with_features=True, **CLASSIFIER_HPARAMS)
    encoder_before = {k: v.clone() for k, v in module.encoder.state_dict().items()}
    head_before = module.classifier.model.weight.clone()
    modes: list[bool] = []

    class Record(L.Callback):
        def on_train_batch_start(self, trainer: L.Trainer, pl_module: Any, *args: Any) -> None:
            modes.append(pl_module.encoder.training)

    trainer = _quiet_trainer(settings, callbacks=[Record()])
    trainer.fit(module, datamodule=_classifier_data(classifier_dir, "center_crop"))
    # The encoder (BatchNorm statistics included) never changes and is never in train mode.
    assert modes and not any(modes)
    for key, value in module.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[key])
    assert not torch.equal(module.classifier.model.weight, head_before)
    # The optimizer holds only the head; the caller's module is untouched (it was copied).
    (optimizer,) = trainer.optimizers
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in module.classifier.parameters()
    }
    for key, value in pretrained.state_dict().items():
        torch.testing.assert_close(value, pretrained_state[key])
    assert module.classifier.model.in_features == 128 + 23
    assert module.hparams["with_features"] is True
    # The encoder spec (plain values), not the module, is what the hyperparameters keep.
    assert module.hparams["pretrained"] == dict(pretrained.hparams)


def test_linear_eval_keeps_the_legacy_softmax_loss() -> None:
    module = LinearEvalModule(_simclr(), with_features=False, **CLASSIFIER_HPARAMS)
    logits = torch.tensor([[2.0, 0.0, -1.0, 0.5, 0.1], [0.0, 3.0, 0.0, 0.0, 0.0]])
    targets = torch.nn.functional.one_hot(torch.tensor([0, 2]), 5)
    legacy = P_stamps_loss(0.01)(logits.softmax(dim=1), targets)
    torch.testing.assert_close(module.compute_loss(logits, targets), legacy)
    assert not torch.allclose(
        module.compute_loss(logits, targets), P_stamps_loss(0.01)(logits, targets)
    )


def test_fine_tune_heads_and_copy(tmp_path: Path, classifier_dir: Path) -> None:
    resnet = FineTuneModule(_simclr("resnet18"), drop_rate=0.5, **CLASSIFIER_HPARAMS)
    assert isinstance(resnet.model.head, LinearClassifier)
    assert resnet.model.head.model.in_features == 512 + 23
    pstamps = FineTuneModule(
        _simclr("pstamps"), drop_rate=0.5, with_features=False, **CLASSIFIER_HPARAMS
    )
    assert isinstance(pstamps.model.head, ThreeLayerClassifier)
    assert pstamps.model.bn is None
    settings = _settings(tmp_path, max_epochs=1)
    seed_run(settings)
    pretrained = _simclr()
    before = {k: v.clone() for k, v in pretrained.state_dict().items()}
    module = FineTuneModule(pretrained, drop_rate=0.5, **CLASSIFIER_HPARAMS)
    assert isinstance(module.model.head, ThreeLayerClassifier)  # stamp<k>: the pstamps head
    encoder_before = {k: v.clone() for k, v in module.model.encoder.state_dict().items()}
    _quiet_trainer(settings).fit(module, datamodule=_classifier_data(classifier_dir))
    changed = [
        k
        for k, v in module.model.encoder.state_dict().items()
        if not torch.equal(v, encoder_before[k])
    ]
    assert changed  # the encoder is trained
    for key, value in pretrained.state_dict().items():
        torch.testing.assert_close(value, before[key])  # ... but on a copy


def test_resnet_classifier_rejects_other_encoders() -> None:
    with pytest.raises(ValueError, match="must be a ResNet"):
        SupervisedCEResNetModule(resnet_model="stamp1", **CLASSIFIER_HPARAMS)
    module = SupervisedCEResNetModule(
        resnet_model="resnet18", with_features=False, **CLASSIFIER_HPARAMS
    )
    assert module.model.bn is None
    assert isinstance(module.model.head, LinearClassifier)
    assert module.model.head.model.in_features == 512


def test_selection_callbacks_on_simclr_and_classifiers(
    tmp_path: Path, pretraining_dir: Path, classifier_dir: Path
) -> None:
    # SimCLR selects on its training loss; checkpoints are reloadable and never versioned.
    settings = _settings(tmp_path, max_epochs=2, monitor="loss_train_enc", patience=5)
    for _ in range(2):
        seed_run(settings)
        callbacks = build_callbacks(settings, SimCLRModule.MONITORABLE, dirpath=tmp_path / "ckpt")
        trainer = _quiet_trainer(settings, callbacks=list(callbacks), enable_checkpointing=True)
        trainer.fit(_simclr(), datamodule=_contrastive(pretraining_dir))
    assert sorted(p.name for p in (tmp_path / "ckpt").iterdir()) == ["best.ckpt"]
    early, checkpoint = callbacks
    assert checkpoint.best_model_score is not None
    assert early.wait_count <= 1
    restored = SimCLRModule.load_from_checkpoint(tmp_path / "ckpt" / "best.ckpt")
    assert restored.encoder_name == "stamp1"
    assert restored.hparams["nt_xent_legacy_denominator"] is True

    # A classifier can select on the validation loss (mode min), not only on accuracy_val.
    settings = _settings(tmp_path, max_epochs=3, monitor="loss_val")
    seed_run(settings)
    early, checkpoint = build_callbacks(
        settings, SupervisedCEModule.MONITORABLE, dirpath=tmp_path / "ce"
    )
    val_losses: list[float] = []

    class Record(L.Callback):
        def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            if not trainer.sanity_checking:
                val_losses.append(float(trainer.callback_metrics["loss_val"]))

    trainer = _quiet_trainer(
        settings, callbacks=[early, checkpoint, Record()], enable_checkpointing=True
    )
    module = SupervisedCEModule(drop_rate=0.2, **CLASSIFIER_HPARAMS)
    trainer.fit(module, datamodule=_classifier_data(classifier_dir))
    assert checkpoint.mode == "min"
    assert len(val_losses) == 3
    assert checkpoint.best_model_score is not None
    assert float(checkpoint.best_model_score) == pytest.approx(min(val_losses))
    # Checkpoints hold only plain types, so the safe torch.load default can read them.
    for path in (tmp_path / "ce" / "best.ckpt", tmp_path / "ckpt" / "best.ckpt"):
        content = torch.load(path, weights_only=True)
        assert content["datamodule_hyper_parameters"]["image_size"] == 27


def test_simclr_takes_the_nt_xent_setting() -> None:
    settings = Settings(nt_xent_legacy_denominator=False)
    module = _simclr(nt_xent_legacy_denominator=settings.nt_xent_legacy_denominator)
    assert module.criterion_encoder.legacy_denominator is False
    assert _simclr().criterion_encoder.legacy_denominator is True


def _fixed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(4, 3, 27, 27, generator=generator), torch.rand(4, 23, generator=generator)


@pytest.mark.parametrize(
    "build",
    [
        lambda pretrained: LinearEvalModule(pretrained, with_features=True, **CLASSIFIER_HPARAMS),
        lambda pretrained: FineTuneModule(pretrained, drop_rate=0.5, **CLASSIFIER_HPARAMS),
    ],
    ids=["linear_eval", "fine_tune"],
)
def test_downstream_modules_restore_from_checkpoint(
    build: Callable[[SimCLRModule], ClassifierModule], tmp_path: Path, classifier_dir: Path
) -> None:
    settings = _settings(tmp_path, max_epochs=1)
    seed_run(settings)
    module = build(_simclr("pstamps"))
    trainer = _quiet_trainer(settings)
    trainer.fit(module, datamodule=_classifier_data(classifier_dir))
    path = tmp_path / "module.ckpt"
    trainer.save_checkpoint(path)
    # The hyperparameters hold the encoder spec, not the module, and stay plain values.
    assert torch.load(path, weights_only=True)["hyper_parameters"]["pretrained"][
        "encoder_name"
    ] == ("pstamps")
    L.seed_everything(123)  # the rebuilt encoder starts from other random weights...
    restored = type(module).load_from_checkpoint(path)
    module.eval()
    restored.eval()
    x_img, x_feat = _fixed_batch()
    with torch.no_grad():  # ... and the checkpoint's state_dict restores the trained ones
        torch.testing.assert_close(restored(x_img, x_feat), module(x_img, x_feat))
    assert restored.image_size == 27


def test_downstream_modules_reject_another_image_size(tmp_path: Path, classifier_dir: Path) -> None:
    module = LinearEvalModule(_simclr("resnet18"), with_features=False, **CLASSIFIER_HPARAMS)
    datamodule = ClassifierDataModule(classifier_dir, image_size=63, batch_size=5)
    with pytest.raises(ValueError, match="image_size is 63, but the pretrained encoder"):
        _quiet_trainer(_settings(tmp_path), fast_dev_run=True).fit(module, datamodule=datamodule)
