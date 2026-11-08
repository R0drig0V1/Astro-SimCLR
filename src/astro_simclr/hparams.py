"""Per-subcommand defaults and the key mapping of the legacy ``results/hparams_best_*.yaml``.

The legacy entry points hard-coded their training protocol (max_epochs, EarlyStopping
patience and min_delta, the monitor) and loaded their hyperparameters from
``results/hparams_best_*.yaml``, whose keys are the legacy module arguments. Here:

- :data:`COMMAND_DEFAULTS` holds, per training subcommand, the settings that differ from
  the :class:`~astro_simclr.config.Settings` defaults, so that each subcommand's protocol
  defaults to its legacy ``iter_*.py`` values. Only ``pretrain`` differs (``iter_clr.py``:
  400 epochs, patience 100, ``min_delta`` 0.002 on ``loss_train_enc``, and its
  hyperparameters); the classifier subcommands use the ``Settings`` defaults, which are the
  ``iter_ce_stamp.py`` / ``linear_classifier.py`` / ``fine_tuning.py`` protocol.
- :func:`load_legacy_hparams` maps a legacy hyperparameter file to settings, per
  subcommand, through the documented tables :data:`CLASSIFIER_KEYS` and
  :data:`SIMCLR_KEYS`. A key outside the table is an error (a typo or a file from another
  code version must not be silently dropped). Some keys are known but belong to another
  subcommand; they are *skipped* and reported, never applied.

Two kinds of legacy file exist, told apart by their keys:

- **classifier files** (``hparams_best_ce.yaml``, ``hparams_best_ce_resnet.yaml``): the
  ``Supervised_Cross_Entropy`` arguments, used by ``train-supervised``, ``linear-eval`` and
  ``fine-tune`` (``linear_classifier.py`` and ``fine_tuning.py`` read
  ``hparams_best_ce.yaml``);
- **SimCLR files** (``hparams_best_simclr.yaml``): both the encoder (``*_encoder``,
  ``encoder_name``, ``projection_dim``, ``temperature``) and the classifier
  (``*_classifier``) arguments. ``pretrain`` takes the first group, ``linear-eval`` and
  ``fine-tune`` the second; ``augmentation`` is the *pretraining* augmentation there.

Augmentation names may be legacy aliases (``without_aug``, ``rotation_v3``, ...): they are
passed through, and the data modules resolve them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from astro_simclr.config import ConfigError, read_settings_file

__all__ = [
    "CLASSIFIER_KEYS",
    "COMMANDS",
    "COMMAND_DEFAULTS",
    "SIMCLR_KEYS",
    "LegacyHparams",
    "load_legacy_hparams",
    "map_legacy_hparams",
]

#: The training subcommands.
COMMANDS: Final = ("pretrain", "linear-eval", "fine-tune", "train-supervised")

#: Settings that differ from the ``Settings`` defaults, per subcommand (legacy protocol).
COMMAND_DEFAULTS: Final[Mapping[str, Mapping[str, object]]] = {
    # iter_clr.py: Trainer(max_epochs=400), EarlyStopping("loss_train_enc", min_delta=0.002,
    # patience=100), and its hparams dict (lr 1.258, batch 450, resnet18, simclr).
    "pretrain": {
        "max_epochs": 400,
        "patience": 100,
        "min_delta": 0.002,
        "monitor": "loss_train_enc",
        "dataset": "td_ztf_stamp_simclr_300",
        "augmentation": "simclr",
        "batch_size": 450,
        "encoder_name": "resnet18",
        "image_size": 27,
        "lr": 1.258059613629328,
        "optimizer": "LARS",
        "projection_dim": 300,
        "temperature": 0.17910250439416514,
    },
    # linear_classifier.py, fine_tuning.py and iter_ce_stamp.py: 220 epochs, patience 40,
    # min_delta 0.001 on accuracy_val, and hparams_best_ce.yaml (the Settings defaults).
    "linear-eval": {},
    "fine-tune": {},
    "train-supervised": {},
}

#: Classifier hyperparameter file: legacy key -> setting (identity names).
CLASSIFIER_KEYS: Final[Mapping[str, str]] = {
    "augmentation": "augmentation",
    "batch_size": "batch_size",
    "beta_loss": "beta_loss",
    "drop_rate": "drop_rate",
    "image_size": "image_size",
    "lr": "lr",
    "optimizer": "optimizer",
    "with_features": "with_features",
}

#: SimCLR hyperparameter file: per subcommand, legacy key -> setting, or ``None`` for a key
#: that belongs to another subcommand (skipped and reported).
SIMCLR_KEYS: Final[Mapping[str, Mapping[str, str | None]]] = {
    "pretrain": {
        "augmentation": "augmentation",
        "batch_size_encoder": "batch_size",
        "encoder_name": "encoder_name",
        "image_size": "image_size",
        "lr_encoder": "lr",
        "optimizer_encoder": "optimizer",
        "projection_dim": "projection_dim",
        "temperature": "temperature",
        # Used by linear-eval / fine-tune.
        "batch_size_classifier": None,
        "beta_loss": None,
        "drop_rate_classifier": None,
        "lr_classifier": None,
        "optimizer_classifier": None,
        "with_features": None,
    },
    "linear-eval": {
        "batch_size_classifier": "batch_size",
        "beta_loss": "beta_loss",
        "drop_rate_classifier": "drop_rate",
        "image_size": "image_size",
        "lr_classifier": "lr",
        "optimizer_classifier": "optimizer",
        "with_features": "with_features",
        # Fixed by the pretrain checkpoint (or the pretraining augmentation).
        "augmentation": None,
        "batch_size_encoder": None,
        "encoder_name": None,
        "lr_encoder": None,
        "optimizer_encoder": None,
        "projection_dim": None,
        "temperature": None,
    },
}

#: Keys whose only accepted value is checked and then dropped: ``method`` (only SimCLR is
#: ported; SupCon is in the backlog) and ``balanced_batch`` (the current legacy module has
#: no such argument, so ``false`` is the only value with a defined meaning).
_CHECKED_KEYS: Final[Mapping[str, object]] = {"method": "simclr", "balanced_batch": False}
#: Keys that only a SimCLR file has.
_SIMCLR_MARKERS: Final = frozenset({"batch_size_encoder", "lr_encoder", "optimizer_encoder"})


class LegacyHparams(dict[str, object]):
    """The settings mapped from a legacy file; :attr:`skipped` lists the keys left out."""

    def __init__(self, values: Mapping[str, object], skipped: list[str]) -> None:
        super().__init__(values)
        #: Known keys that belong to another subcommand (not applied).
        self.skipped = skipped


def _table(command: str, data: Mapping[str, object], source: str) -> Mapping[str, str | None]:
    if command not in COMMANDS:
        raise ConfigError(f"unknown training command {command!r}")
    if _SIMCLR_MARKERS & set(data):
        if command == "train-supervised":
            raise ConfigError(
                f"{source} is a SimCLR hyperparameter file; train-supervised "
                "takes a classifier file (hparams_best_ce*.yaml)"
            )
        return SIMCLR_KEYS["pretrain" if command == "pretrain" else "linear-eval"]
    if command == "pretrain":
        raise ConfigError(
            f"{source} is a classifier hyperparameter file; pretrain takes a "
            "SimCLR file (hparams_best_simclr.yaml)"
        )
    return CLASSIFIER_KEYS


def map_legacy_hparams(
    command: str, data: Mapping[str, object], source: str = "hyperparameter file"
) -> LegacyHparams:
    """Map the legacy hyperparameters ``data`` to settings for ``command``.

    Raises:
        ConfigError: for an unknown command, a file of the wrong kind for ``command``, a key
            outside the mapping table (the message lists the valid keys), or a checked key
            (``method``, ``balanced_batch``) with another value than the supported one.
    """
    table = _table(command, data, source)
    values: dict[str, object] = {}
    skipped: list[str] = []
    unknown = sorted(set(data) - set(table) - set(_CHECKED_KEYS))
    if unknown:
        valid = ", ".join(sorted({*table, *_CHECKED_KEYS}))
        raise ConfigError(
            f"{source}: unknown key(s) {', '.join(unknown)} for {command}; valid keys: {valid}"
        )
    for key, value in data.items():
        if key in _CHECKED_KEYS:
            if value != _CHECKED_KEYS[key]:
                raise ConfigError(
                    f"{source}: {key}={value!r} is not supported (only {_CHECKED_KEYS[key]!r})"
                )
            continue
        target = table[key]
        if target is None:
            skipped.append(key)
        else:
            values[target] = value
    return LegacyHparams(values, skipped)


def load_legacy_hparams(path: Path, command: str) -> LegacyHparams:
    """Read the legacy hyperparameter file ``path`` and map it for ``command``.

    Raises:
        ConfigError: if the file cannot be read or parsed (``yaml.safe_load``), or as
            :func:`map_legacy_hparams`.
    """
    return map_legacy_hparams(command, read_settings_file(path), str(path))
