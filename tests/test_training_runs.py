"""Tests of the optimizer factory, the run helpers and the data modules (no training here)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from _training_support import N_PER_SPLIT
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar

from astro_simclr.augmentations import UnknownAugmentationError
from astro_simclr.config import ConfigError, Settings
from astro_simclr.data import DataError
from astro_simclr.optim import LARS
from astro_simclr.training import (
    ClassifierDataModule,
    ContrastiveDataModule,
    SimCLRModule,
    SupervisedCEModule,
    build_callbacks,
    build_trainer,
    seed_run,
)
from astro_simclr.training.optimizers import (
    OPTIMIZER_NAMES,
    UnknownOptimizerError,
    build_optimizer,
)
from astro_simclr.training.runs import build_last_checkpoint, monitor_mode


def _params() -> list[torch.nn.Parameter]:
    return list(torch.nn.Linear(3, 2).parameters())


def test_optimizers_keep_the_legacy_hyperparameters() -> None:
    adamw = build_optimizer("AdamW", _params(), 0.01)
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["betas"] == (0.5, 0.9)
    adam = build_optimizer("Adam", _params(), 0.01)
    assert type(adam) is torch.optim.Adam
    assert adam.defaults["betas"] == (0.5, 0.9)
    sgd = build_optimizer("SGD", _params(), 0.01)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.defaults["momentum"] == 0
    lars = build_optimizer("LARS", _params(), 0.258, lars_legacy_trust_ratio=False)
    assert isinstance(lars, LARS)
    assert lars.defaults["use_nesterov"] is True
    assert lars.defaults["momentum"] == 0.9
    assert lars.defaults["weight_decay"] == 0.0
    assert lars.defaults["legacy_trust_ratio"] is False
    assert lars.defaults["lr"] == 0.258


def test_unknown_optimizer_lists_the_valid_names() -> None:
    with pytest.raises(UnknownOptimizerError, match="valid names: AdamW, Adam, SGD, LARS"):
        build_optimizer("RMSprop", _params(), 0.01)
    # Modules fail at construction, not when training starts (legacy: UnboundLocalError).
    with pytest.raises(UnknownOptimizerError):
        SupervisedCEModule(drop_rate=0.2, beta_loss=0.0, lr=0.01, optimizer="adamw")
    with pytest.raises(UnknownOptimizerError):
        SimCLRModule(
            encoder_name="stamp1",
            image_size=27,
            projection_dim=8,
            temperature=0.5,
            lr_encoder=0.1,
            optimizer_encoder="Lars",
        )
    assert set(OPTIMIZER_NAMES) == {"AdamW", "Adam", "SGD", "LARS"}


def test_seed_run_seeds_torch() -> None:
    assert seed_run(Settings(seed=7)) == 7
    first = torch.rand(3)
    seed_run(Settings(seed=7))
    torch.testing.assert_close(torch.rand(3), first)


def test_monitor_mode_follows_the_metric() -> None:
    assert monitor_mode("accuracy_val", SupervisedCEModule.MONITORABLE) == "max"
    assert monitor_mode("loss_val", SupervisedCEModule.MONITORABLE) == "min"
    assert monitor_mode("loss_train_enc", SimCLRModule.MONITORABLE) == "min"
    # SimCLR has no validation loop, so its default protocol monitor cannot be used.
    with pytest.raises(ConfigError, match="valid: loss_train_enc"):
        monitor_mode("accuracy_val", SimCLRModule.MONITORABLE)
    with pytest.raises(ConfigError, match="accuracy_val, loss_val"):
        monitor_mode("Accuracy/Validation", SupervisedCEModule.MONITORABLE)


def test_callbacks_read_the_settings(tmp_path: Path) -> None:
    settings = Settings(monitor="loss_val", patience=7, min_delta=0.02)
    early, checkpoint = build_callbacks(
        settings, SupervisedCEModule.MONITORABLE, dirpath=tmp_path, filename="ce"
    )
    assert isinstance(early, EarlyStopping)
    assert (early.monitor, early.patience, early.mode) == ("loss_val", 7, "min")
    # Lightning stores min_delta signed by the mode.
    assert abs(early.min_delta) == pytest.approx(0.02)
    assert isinstance(checkpoint, ModelCheckpoint)
    assert (checkpoint.monitor, checkpoint.mode, checkpoint.save_top_k) == ("loss_val", "min", 1)
    assert checkpoint.filename == "ce"
    assert checkpoint.dirpath == str(tmp_path)
    assert checkpoint._enable_version_counter is False


def test_trainer_takes_hardware_and_protocol_from_the_settings(tmp_path: Path) -> None:
    settings = Settings(accelerator="cpu", devices=1, max_epochs=3, output_dir=tmp_path)
    trainer = build_trainer(settings, logger=False)
    assert trainer.max_epochs == 3
    assert trainer.accelerator.__class__.__name__ == "CPUAccelerator"
    assert trainer.default_root_dir == str(tmp_path)
    assert torch.backends.cudnn.benchmark is True  # the legacy benchmark=True
    deterministic = build_trainer(
        settings.replace(deterministic=True), logger=False, fast_dev_run=True
    )
    assert torch.backends.cudnn.benchmark is False
    assert deterministic.fast_dev_run is True


def test_classifier_datamodule_builds_the_three_splits(classifier_dir: Path) -> None:
    dm = ClassifierDataModule(
        classifier_dir, image_size=27, batch_size=3, eval_batch_size=4, augmentation="rotation_v3"
    )
    dm.setup("fit")
    datasets = dm.datasets
    dm.setup("test")  # a second call keeps the same datasets
    assert dm.datasets is datasets
    train = list(dm.train_dataloader())
    # drop_last: 10 stamps in batches of 3 give 3 batches.
    assert len(train) == N_PER_SPLIT // 3
    x_img, x_feat, y = train[0]
    assert x_img.shape == (3, 3, 27, 27)
    assert x_feat.shape == (3, 23)
    assert y.shape == (3, 5)
    assert y.sum(dim=1).tolist() == [1, 1, 1]
    val = list(dm.val_dataloader())
    assert [len(batch[0]) for batch in val] == [4, 4, 2]
    # Evaluation always uses the centre crop: deterministic, so two passes agree.
    torch.testing.assert_close(
        next(iter(dm.test_dataloader()))[0], next(iter(dm.test_dataloader()))[0]
    )


def test_classifier_datamodule_rejects_bad_input(
    classifier_dir: Path, pretraining_dir: Path
) -> None:
    with pytest.raises(UnknownAugmentationError):
        ClassifierDataModule(classifier_dir, image_size=27, batch_size=2, augmentation="nope")
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        ClassifierDataModule(classifier_dir, image_size=27, batch_size=0)
    with pytest.raises(ValueError, match="num_workers"):
        ClassifierDataModule(classifier_dir, image_size=27, batch_size=2, num_workers=-1)
    with pytest.raises(DataError, match="expected a classifier file"):
        ClassifierDataModule(pretraining_dir, image_size=27, batch_size=2).setup("fit")
    with pytest.raises(DataError, match="fewer than batch_size"):
        ClassifierDataModule(classifier_dir, image_size=27, batch_size=11).setup("fit")


def test_batch_size_is_checked_only_for_fit(classifier_dir: Path, pretraining_dir: Path) -> None:
    # Evaluating trains nothing, so a batch_size larger than Train is fine there ...
    dm = ClassifierDataModule(classifier_dir, image_size=27, batch_size=11, eval_batch_size=4)
    for stage in ("validate", "test"):
        dm.setup(stage)
    assert sum(len(batch[0]) for batch in dm.test_dataloader()) == N_PER_SPLIT
    # ... but fit still fails, even after the datasets were built for another stage.
    with pytest.raises(DataError, match="fewer than batch_size"):
        dm.setup("fit")
    contrastive = ContrastiveDataModule(
        pretraining_dir, image_size=27, batch_size=11, augmentation="simclr"
    )
    contrastive.setup("predict")
    with pytest.raises(DataError, match="fewer than batch_size"):
        contrastive.setup("fit")


def test_contrastive_datamodule_yields_view_pairs(pretraining_dir: Path) -> None:
    dm = ContrastiveDataModule(pretraining_dir, image_size=27, batch_size=4, augmentation="astro8")
    with pytest.raises(RuntimeError, match="setup"):
        dm.train_dataloader()
    dm.setup("fit")
    batches = list(dm.train_dataloader())
    assert len(batches) == N_PER_SPLIT // 4
    x_i, x_j = batches[0]
    assert x_i.shape == x_j.shape == (4, 3, 27, 27)
    assert not torch.equal(x_i, x_j)  # two independent random views
    with pytest.raises(DataError, match="fewer than batch_size"):
        ContrastiveDataModule(
            pretraining_dir, image_size=27, batch_size=11, augmentation="simclr"
        ).setup("fit")
    with pytest.raises(ValueError, match="num_workers"):
        ContrastiveDataModule(
            pretraining_dir, image_size=27, batch_size=2, augmentation="simclr", num_workers=-1
        )


def test_divergence_and_finiteness_come_from_the_settings(tmp_path: Path) -> None:
    settings = Settings(divergence_threshold=0.1, check_finite=False)
    early_stopping, _ = build_callbacks(settings, frozenset({"accuracy_val"}), dirpath=tmp_path)
    assert early_stopping.divergence_threshold == 0.1
    assert early_stopping.check_finite is False


def test_last_checkpoint_has_no_monitor(tmp_path: Path) -> None:
    last = build_last_checkpoint(tmp_path)
    assert last.monitor is None
    assert last.filename == "last"
    assert last.save_top_k == 1


def test_trainer_batch_limits(tmp_path: Path) -> None:
    trainer = build_trainer(Settings(output_dir=tmp_path, accelerator="cpu"))
    assert trainer.limit_train_batches == 1.0
    limited = build_trainer(
        Settings(
            output_dir=tmp_path, accelerator="cpu", limit_train_batches=2, limit_val_batches=0.5
        )
    )
    assert (limited.limit_train_batches, limited.limit_val_batches) == (2, 0.5)


def test_progress_bar_refresh_rate_comes_from_the_settings(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path, accelerator="cpu")
    bar = build_trainer(settings, logger=False).progress_bar_callback
    assert isinstance(bar, TQDMProgressBar)
    assert bar.refresh_rate == settings.progress_bar_refresh_rate == 50
    every = build_trainer(settings.replace(progress_bar_refresh_rate=1), logger=False)
    assert isinstance(every.progress_bar_callback, TQDMProgressBar)
    assert every.progress_bar_callback.refresh_rate == 1
    # 0 hides the bar, and so does an explicit enable_progress_bar=False (whatever the rate).
    hidden = build_trainer(settings.replace(progress_bar_refresh_rate=0), logger=False)
    assert hidden.progress_bar_callback is None
    off = build_trainer(settings, logger=False, enable_progress_bar=False)
    assert off.progress_bar_callback is None
