"""Typed run settings: defaults, a YAML file, then ``key=value`` overrides from the CLI.

The legacy code hard-coded its hardware settings in ``utils/config.py`` (``gpus: 1``) and its
training protocol in every entry point (max_epochs, EarlyStopping patience and min_delta,
the checkpoint monitor, the eval batch size). Here they are one frozen dataclass,
:class:`Settings`, so every run records exactly what it used.

The fields are flat, so that any of them can be overridden with ``--set key=value``. Besides
the hardware, paths and protocol, they hold the hyperparameters of the four training
subcommands (task 1.7); each subcommand reads the fields it needs, and the others are
recorded but unused (``docs/architecture.md``, section ``config``, lists which). The
``schedule`` field (task 2.2a) chooses between the legacy epoch protocol and the iteration
schedule of Carrasco-Davis et al. 2021, whose own fields it validates.

Resolution order (later wins): the dataclass defaults, an optional ``defaults`` mapping (a
subcommand's legacy protocol), the YAML file, the overrides. YAML is read with
``yaml.safe_load`` only (the legacy ``FullLoader`` can build arbitrary objects). Unknown
keys and values of the wrong type are errors, never silently ignored: a typo in a reference
config must not change an experiment.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "ACCELERATORS",
    "CARRASCO2021",
    "CARRASCO2021_EVAL_BATCH_SIZE",
    "EPOCHS",
    "SCHEDULES",
    "ConfigError",
    "Settings",
    "load_settings",
    "parse_override",
    "read_settings_file",
    "settings_from_mapping",
    "settings_to_plain",
]

#: ``Settings.schedule`` of the legacy epoch protocol.
EPOCHS: Final = "epochs"
#: ``Settings.schedule`` of the Carrasco-Davis et al. 2021 iteration protocol (task 2.2a).
CARRASCO2021: Final = "carrasco2021"
#: The accepted ``Settings.schedule`` values.
SCHEDULES: Final = (EPOCHS, CARRASCO2021)
#: The evaluation batch the ``carrasco2021`` schedule requires (TF's batch of 64).
CARRASCO2021_EVAL_BATCH_SIZE: Final = 64

#: Accelerator names accepted by ``Settings.accelerator`` (the Lightning names this project
#: supports). ``auto`` picks CUDA, then MPS, then CPU.
ACCELERATORS = frozenset({"auto", "cpu", "cuda", "gpu", "mps"})


class ConfigError(ValueError):
    """A settings file or override is invalid: unknown key, wrong type or bad value."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Hardware, paths, reproducibility, training protocol and loss options of one run.

    The protocol defaults are the legacy classifier values (``iter_ce_stamp.py``,
    ``linear_classifier.py`` and ``fine_tuning.py``); the ``pretrain`` subcommand replaces
    them with the ``iter_clr.py`` values (``hparams.COMMAND_DEFAULTS``). The classifier
    hyperparameter defaults are ``results/hparams_best_ce.yaml`` with ``AdamW`` (the
    optimizer of every tuned trial behind it; the plan's known-defects table).
    """

    #: Lightning accelerator; see :data:`ACCELERATORS`. Reference configs set ``cuda`` so
    #: that a missing GPU fails instead of training on CPU.
    accelerator: str = "auto"
    #: Number of devices, or a Lightning device string. Defaults to 1: ``NT_Xent`` has no
    #: all-gather, so multi-device training would change the loss.
    devices: int | str = 1
    #: DataLoader worker processes; 0 loads in the main process.
    num_workers: int = 0
    #: Directory with the prepared dataset layout.
    data_dir: Path = Path("dataset")
    #: Every output of a run (checkpoints, logs, figures, metrics) goes under this directory.
    output_dir: Path = Path("outputs")
    #: Seed for ``lightning.seed_everything(seed, workers=True)``.
    seed: int = 0
    #: ``Trainer(deterministic=...)``. False keeps the legacy ``benchmark=True`` behaviour.
    deterministic: bool = False
    #: Upper bound on training epochs.
    max_epochs: int = 220
    #: EarlyStopping patience, in epochs.
    patience: int = 40
    #: Smallest change of ``monitor`` that EarlyStopping counts as an improvement.
    min_delta: float = 0.001
    #: Metric watched by EarlyStopping and the best checkpoint.
    monitor: str = "accuracy_val"
    #: Batch size of the validation and test loaders.
    eval_batch_size: int = 100
    #: ``NT_Xent(legacy_denominator=...)``: True keeps the legacy denominator
    #: ``1 + sum_neg exp(s_neg)``, so that runs stay comparable with ``results/``; False uses
    #: the paper's ``exp(s_pos) + sum_neg exp(s_neg)`` (ADR 0002).
    nt_xent_legacy_denominator: bool = True
    #: EarlyStopping ``check_finite``: stop when the monitored value becomes NaN or inf.
    check_finite: bool = True
    #: EarlyStopping ``divergence_threshold``: stop as soon as the monitored value is worse
    #: than this (the legacy tuning protocol used 0.1 on ``accuracy_val``). None: never.
    divergence_threshold: float | None = None
    #: ``Trainer(limit_train_batches=...)``: an int is a number of batches per epoch, a float
    #: a fraction. None: every batch. The dry mode sets 2.
    limit_train_batches: int | float | None = None
    #: ``Trainer(limit_val_batches=...)``, as ``limit_train_batches``.
    limit_val_batches: int | float | None = None

    # --- Data and model hyperparameters of the training subcommands (task 1.7) -------------

    #: Prepared layout to train on: the directory ``data_dir/<dataset>`` (a pickle stem).
    #: The default is the received 100 % file, the deduplicated original (``docs/data.md``).
    dataset: str = "td_ztf_stamp_17_06_20_red-002"
    #: Side of the centre crop fed to the network (the stamp networks need 27).
    image_size: int = 27
    #: Training batch size (SimCLR: pairs per batch).
    batch_size: int = 45
    #: Learning rate (SimCLR: the encoder's base rate before the cosine schedule).
    lr: float = 0.002495379490073152
    #: Optimizer name: ``AdamW``, ``Adam``, ``SGD`` or ``LARS``.
    optimizer: str = "AdamW"
    #: Training augmentation: a registry name or a legacy alias (``augmentations``).
    augmentation: str = "center_crop"
    #: ``P_stamps_loss`` entropy weight of the classifiers.
    beta_loss: float = 0.016063800410738464
    #: Dropout of the stamp classifier and of the fine-tuning head.
    drop_rate: float = 0.2
    #: Concatenate the 23 alert features to the image representation.
    with_features: bool = True
    #: ``train-supervised`` network: ``stamp_classifier`` or a ResNet name.
    supervised_model: str = "stamp_classifier"
    #: ``pretrain`` encoder (``build_encoder`` names: ``stamp<k>``, ``pstamps``, ``resnet*``).
    encoder_name: str = "resnet18"
    #: ``pretrain`` projection-head output size.
    projection_dim: int = 300
    #: ``pretrain`` NT-Xent temperature.
    temperature: float = 0.17910250439416514
    #: LARS ``legacy_trust_ratio``: True keeps the legacy always-true ``.ge(0)`` test.
    lars_legacy_trust_ratio: bool = True
    #: ``linear-eval`` / ``fine-tune``: the pretrain checkpoint, or a pretrain run directory
    #: (its ``checkpoints/best.ckpt``).
    pretrained: Path | None = None
    #: ``fine-tune`` trains with ``lr * fine_tune_lr_factor`` (legacy ``fine_tuning.py``: 0.2).
    fine_tune_lr_factor: float = 0.2
    #: ``paper_stamp_classifier``: apply its dropout layers (rate ``drop_rate``). False is
    #: faithful: M1 trained with its dropout inactive (task 2.2a).
    dropout_active: bool = False

    # --- Training schedule (task 2.2a) ------------------------------------------------------

    #: ``epochs`` (the legacy schedule: ``max_epochs``, EarlyStopping, best by epoch) or
    #: ``carrasco2021`` (ALeRCE's: steps, validation every ``val_every_n_steps``, the horizon
    #: rule; ``training.schedule``). See :data:`SCHEDULES`.
    schedule: str = "epochs"
    #: ``carrasco2021``: the initial horizon, in optimizer steps (the Trainer's ``max_steps``).
    max_steps: int = 30_000
    #: ``carrasco2021``: validate (and select) every this many optimizer steps.
    val_every_n_steps: int = 10
    #: ``carrasco2021``: a new best at step ``s`` raises the horizon to ``s + horizon_increment``.
    horizon_increment: int = 10_000
    #: ``carrasco2021``: write ``last.ckpt`` (what ``--resume`` continues from) after the
    #: validation of every this many optimizer steps (a multiple of ``val_every_n_steps``).
    checkpoint_every_n_steps: int = 1_000
    #: ``epochs``: stop early with EarlyStopping (``patience``, ``min_delta``). The
    #: ``carrasco2021`` schedule has none and requires false.
    early_stopping: bool = True

    def __post_init__(self) -> None:
        """Check the values that a type alone cannot (ranges and names).

        Raises:
            ConfigError: if any value is out of range.
        """
        if self.accelerator not in ACCELERATORS:
            valid = ", ".join(sorted(ACCELERATORS))
            raise ConfigError(f"accelerator {self.accelerator!r} is not one of: {valid}")
        if isinstance(self.devices, int) and self.devices < 1:
            raise ConfigError(f"devices must be >= 1, got {self.devices}")
        for name in ("max_epochs", "eval_batch_size"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("num_workers", "patience", "seed"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.min_delta < 0:
            raise ConfigError(f"min_delta must be >= 0, got {self.min_delta}")
        for name in ("monitor", "dataset", "optimizer", "augmentation", "encoder_name"):
            if not getattr(self, name):
                raise ConfigError(f"{name} must not be empty")
        self._check_hyperparameters()
        self._check_schedule()

    def _check_hyperparameters(self) -> None:
        for name in ("image_size", "batch_size", "projection_dim"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("lr", "temperature", "fine_tune_lr_factor"):
            if not getattr(self, name) > 0:
                raise ConfigError(f"{name} must be > 0, got {getattr(self, name)}")
        if not 0 <= self.drop_rate < 1:
            raise ConfigError(f"drop_rate must be in [0, 1), got {self.drop_rate}")
        for name in ("limit_train_batches", "limit_val_batches"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ConfigError(f"{name} must be > 0, got {value}")

    def _check_schedule(self) -> None:
        if self.schedule not in SCHEDULES:
            valid = ", ".join(SCHEDULES)
            raise ConfigError(f"schedule {self.schedule!r} is not one of: {valid}")
        for name in ("max_steps", "val_every_n_steps", "checkpoint_every_n_steps"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.horizon_increment < 0:
            raise ConfigError(f"horizon_increment must be >= 0, got {self.horizon_increment}")
        if self.schedule != CARRASCO2021:
            return
        # The schedule's definitions, which a setting must not silently change.
        if self.eval_batch_size != CARRASCO2021_EVAL_BATCH_SIZE:
            msg = (
                f"schedule {CARRASCO2021} needs eval_batch_size={CARRASCO2021_EVAL_BATCH_SIZE} "
                "(its selection loss is the mean of batch-64 means), got "
                f"{self.eval_batch_size}"
            )
            raise ConfigError(msg)
        if self.early_stopping:
            msg = f"schedule {CARRASCO2021} has no EarlyStopping: set early_stopping=false"
            raise ConfigError(msg)
        if self.num_workers != 0:
            msg = f"schedule {CARRASCO2021} keeps the data on the device: set num_workers=0"
            raise ConfigError(msg)
        if self.checkpoint_every_n_steps % self.val_every_n_steps != 0:
            # last.ckpt is written after a validation (training.schedule.HorizonCallback).
            msg = (
                f"schedule {CARRASCO2021} writes last.ckpt after a validation: "
                f"checkpoint_every_n_steps ({self.checkpoint_every_n_steps}) must be a "
                f"multiple of val_every_n_steps ({self.val_every_n_steps})"
            )
            raise ConfigError(msg)

    def replace(self, **changes: Any) -> Settings:
        """Return a copy with ``changes`` applied, validated like :func:`settings_from_mapping`.

        Raises:
            ConfigError: on an unknown key or a value of the wrong type.
        """
        return settings_from_mapping({**dataclasses.asdict(self), **changes})


def _coerce_int(value: object) -> int:
    # bool is a subclass of int; `devices: true` is a typo, not 1.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise TypeError("an integer")


def _coerce_float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    # YAML 1.1 (PyYAML) reads `1e-3` as a string: only `1.0e-3` is a float there.
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise TypeError("a number")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("true or false")


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    raise TypeError("a string")


def _coerce_none(value: object) -> None:
    if value is None:
        return None
    raise TypeError("null")


def _coerce_path(value: object) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    raise TypeError("a non-empty path")


_COERCERS: dict[type, Callable[[object], Any]] = {
    int: _coerce_int,
    float: _coerce_float,
    bool: _coerce_bool,
    str: _coerce_str,
    Path: _coerce_path,
    type(None): _coerce_none,
}


def _coerce(hint: Any, value: object) -> Any:
    """Convert ``value`` to the field type ``hint`` (a plain type or a union of them).

    Raises:
        TypeError: with a description of the expected type, if no member accepts it.
    """
    if get_origin(hint) in (Union, types.UnionType):
        expected = []
        for member in get_args(hint):
            try:
                return _coerce(member, value)
            except TypeError as err:
                expected.append(str(err))
        raise TypeError(" or ".join(expected))
    return _COERCERS[hint](value)


def settings_from_mapping(data: Mapping[str, object]) -> Settings:
    """Build :class:`Settings` from a mapping, checking keys and value types.

    Missing keys keep their defaults. Integers are accepted for float fields, numeric
    strings such as ``1e-3`` (which YAML 1.1 leaves as strings) for float fields, and
    strings for path fields; nothing else is converted.

    Raises:
        ConfigError: on an unknown key, a value of the wrong type or an out-of-range value.
    """
    hints = get_type_hints(Settings)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        valid = ", ".join(sorted(hints))
        raise ConfigError(f"unknown setting(s) {', '.join(unknown)}; valid settings: {valid}")
    values: dict[str, Any] = {}
    for key, value in data.items():
        try:
            values[key] = _coerce(hints[key], value)
        except TypeError as err:
            raise ConfigError(f"setting {key!r} must be {err}, got {value!r}") from None
    return Settings(**values)


def parse_override(text: str) -> tuple[str, object]:
    """Split a ``key=value`` override; the value is parsed as a YAML scalar.

    So ``seed=3`` gives an int, ``deterministic=true`` a bool and ``data_dir=/x`` a string.
    Only the first ``=`` splits, so values may contain ``=``.

    Raises:
        ConfigError: if there is no ``=``, the key is empty or the value is not valid YAML.
    """
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ConfigError(f"override {text!r} is not of the form key=value")
    try:
        value: object = yaml.safe_load(raw)
    except yaml.YAMLError as err:
        raise ConfigError(f"override {text!r}: invalid value ({err})") from None
    return key, value


def read_settings_file(path: Path) -> dict[str, object]:
    """Read a YAML settings file into a mapping, without checking the keys.

    An empty file gives ``{}``.

    Raises:
        ConfigError: if the file cannot be read, is not valid YAML or is not a mapping with
            string keys.
    """
    try:
        loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ConfigError(f"cannot read settings file {path}: {err}") from None
    except yaml.YAMLError as err:
        raise ConfigError(f"settings file {path} is not valid YAML: {err}") from None
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(k, str) for k in loaded):
        raise ConfigError(f"settings file {path} must be a mapping of setting names")
    return dict(loaded)


def load_settings(
    path: Path | None = None,
    overrides: Iterable[str] = (),
    *,
    defaults: Mapping[str, object] | None = None,
) -> Settings:
    """Resolve the settings of a run: defaults, then the YAML file ``path``, then ``overrides``.

    ``defaults`` (for example a subcommand's legacy protocol) sits between the dataclass
    defaults and the file. An empty YAML file means "no change".

    Raises:
        ConfigError: if the file cannot be read, is not valid YAML, is not a mapping, or any
            key or value is invalid (see :func:`settings_from_mapping`).
    """
    data: dict[str, object] = dict(defaults or {})
    if path is not None:
        data.update(read_settings_file(path))
    for text in overrides:
        key, value = parse_override(text)
        data[key] = value
    return settings_from_mapping(data)


def settings_to_plain(settings: Settings) -> dict[str, object]:
    """The settings as plain YAML/JSON values (paths as strings), in field order.

    The result is itself a valid settings mapping: ``settings_from_mapping`` of it gives
    ``settings`` back.
    """
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in dataclasses.asdict(settings).items()
    }
