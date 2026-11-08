"""The nested presets: the flat ones of ``presets`` and ``hparams``, migrated (task 4.2).

Each preset is a whole :class:`~astro_simclr.config_tree.schema.Config`, written out here
rather than raised from its flat form, so that the hash migration test
(``tests/test_config_tree_presets.py``) compares two independent definitions: each one must
lower to the flat preset's flat view and config hash exactly (ADR 0015, rule 10; ADR 0017).

- ``carrasco2021`` and ``carrasco2021-no-features``: ``presets.PRESETS`` on top of the
  ``train-supervised`` protocol (the replication of Carrasco-Davis et al. 2021, ADR 0003;
  every value and its provenance is in ``configs/phase2/carrasco2021.yaml``);
- ``<command>-legacy``: the legacy protocol of each training subcommand,
  ``hparams.COMMAND_DEFAULTS`` on top of the ``Settings`` defaults. A nested config file
  without ``preset`` starts from the one of its command (:data:`LEGACY_PRESETS`), as the
  flat subcommands do.

The preset values listed explicitly below are the ones the flat presets list (so a reader
sees the replication's full definition); the rest are the shared defaults. The deviation
report of a run (``presets.preset_report``) is still computed on the lowered settings.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Final

from astro_simclr.config import CARRASCO2021_EVAL_BATCH_SIZE, ConfigError
from astro_simclr.config_tree.schema import (
    AdamOptim,
    Carrasco2021Schedule,
    Config,
    Data,
    EpochsSchedule,
    Eval,
    FineTuneModel,
    LARSOptim,
    LinearEvalModel,
    PaperAugment,
    PaperStampClassifier,
    Runtime,
    SimCLRModel,
    TransformsAugment,
)

__all__ = ["LEGACY_PRESETS", "PRESETS", "preset"]

_CARRASCO2021: Final = Config(
    # num_workers 0 (the data stay on the device); float32 convolutions (ADR 0010).
    runtime=Runtime(num_workers=0, fp32_precision="ieee"),
    data=Data(
        dataset="td_ztf_stamp_17_06_20_red-002",
        image_size=21,
        batch_size=64,
        with_features=True,
    ),
    # The paper's 4-rotation model on the float input (tasks 2.1, 2.2a; ADR 0005).
    model=PaperStampClassifier(
        arch="rot4",
        group_fields=None,
        context="none",
        feature_set="all",
        input="float32",
        nan_approximation=True,
        exclude_leaked_train=True,
        dropout_active=False,
        drop_rate=0.5,
        beta_loss=0.5,
    ),
    # TF1 AdamOptimizer(1e-3, 0.9, 0.999).
    optim=AdamOptim(kind="Adam", lr=1e-3, betas="code"),
    schedule=Carrasco2021Schedule(
        max_steps=30_000,
        val_every_n_steps=10,
        horizon_increment=10_000,
        monitor="loss_val",
    ),
    # Every 2.5 augmentation off (ADR 0006, 0007).
    augment=PaperAugment(
        rot90=False,
        noise=0.0,
        feature_jitter=0.0,
        synthetic_bogus=0.0,
        min_r2=0.5,
        flux_rescale=0.0,
        flux_k_min=0.5,
        flux_k_max=1.5,
    ),
    eval=Eval(batch_size=CARRASCO2021_EVAL_BATCH_SIZE),
)

_PRETRAIN: Final = Config(
    # iter_clr.py: max 400 epochs, EarlyStopping(loss_train_enc, 0.002, patience 100).
    data=Data(dataset="td_ztf_stamp_simclr_300", image_size=27, batch_size=450),
    model=SimCLRModel(encoder="resnet18", projection_dim=300, temperature=0.17910250439416514),
    optim=LARSOptim(lr=1.258059613629328),
    schedule=EpochsSchedule(
        max_epochs=400, patience=100, min_delta=0.002, monitor="loss_train_enc"
    ),
    augment=TransformsAugment(name="simclr"),
)

#: Preset name -> its nested config.
PRESETS: Final[Mapping[str, Config]] = {
    "carrasco2021": _CARRASCO2021,
    "carrasco2021-no-features": dataclasses.replace(
        _CARRASCO2021, data=dataclasses.replace(_CARRASCO2021.data, with_features=False)
    ),
    "pretrain-legacy": _PRETRAIN,
    "linear-eval-legacy": Config(model=LinearEvalModel()),
    "fine-tune-legacy": Config(model=FineTuneModel()),
    # The Settings defaults are the train-supervised protocol (iter_ce_stamp.py).
    "train-supervised-legacy": Config(),
}

#: Training subcommand -> the preset of its legacy protocol.
LEGACY_PRESETS: Final[Mapping[str, str]] = {
    "pretrain": "pretrain-legacy",
    "linear-eval": "linear-eval-legacy",
    "fine-tune": "fine-tune-legacy",
    "train-supervised": "train-supervised-legacy",
}


def preset(name: str) -> Config:
    """The nested preset ``name``.

    Raises:
        ConfigError: if there is no such preset (the message lists them).
    """
    if name not in PRESETS:
        raise ConfigError(f"unknown preset {name!r}; presets: {', '.join(PRESETS)}")
    return PRESETS[name]
