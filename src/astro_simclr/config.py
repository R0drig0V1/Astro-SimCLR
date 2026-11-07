"""Typed run settings: defaults, a YAML file, then ``key=value`` overrides from the CLI.

The legacy code hard-coded its hardware settings in ``utils/config.py`` (``gpus: 1``) and its
training protocol in every entry point (max_epochs, EarlyStopping patience and min_delta,
the checkpoint monitor, the eval batch size). Here they are one frozen dataclass,
:class:`Settings`, so every run records exactly what it used.

Resolution order (later wins): the dataclass defaults, the YAML file, the overrides. YAML is
read with ``yaml.safe_load`` only (the legacy ``FullLoader`` can build arbitrary objects).
Unknown keys and values of the wrong type are errors, never silently ignored: a typo in a
reference config must not change an experiment.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "ACCELERATORS",
    "ConfigError",
    "Settings",
    "load_settings",
    "parse_override",
    "settings_from_mapping",
]

#: Accelerator names accepted by ``Settings.accelerator`` (the Lightning names this project
#: supports). ``auto`` picks CUDA, then MPS, then CPU.
ACCELERATORS = frozenset({"auto", "cpu", "cuda", "gpu", "mps"})


class ConfigError(ValueError):
    """A settings file or override is invalid: unknown key, wrong type or bad value."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """Hardware, paths, reproducibility and training protocol of one run.

    The protocol defaults are the legacy classifier values (``iter_ce_stamp.py``,
    ``linear_classifier.py`` and ``fine_tuning.py``); pretraining configs override them.
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
        if not self.monitor:
            raise ConfigError("monitor must not be empty")

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
    raise TypeError("a number")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError("true or false")


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value
    raise TypeError("a string")


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

    Missing keys keep their defaults. Integers are accepted for float fields and strings
    for path fields; nothing else is converted.

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


def load_settings(path: Path | None = None, overrides: Iterable[str] = ()) -> Settings:
    """Resolve the settings of a run: defaults, then the YAML file ``path``, then ``overrides``.

    An empty YAML file means "all defaults".

    Raises:
        ConfigError: if the file cannot be read, is not valid YAML, is not a mapping, or any
            key or value is invalid (see :func:`settings_from_mapping`).
    """
    data: dict[str, object] = {}
    if path is not None:
        try:
            loaded: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as err:
            raise ConfigError(f"cannot read settings file {path}: {err}") from None
        except yaml.YAMLError as err:
            raise ConfigError(f"settings file {path} is not valid YAML: {err}") from None
        if loaded is not None:
            if not isinstance(loaded, dict) or not all(isinstance(k, str) for k in loaded):
                raise ConfigError(f"settings file {path} must be a mapping of setting names")
            data.update(loaded)
    for text in overrides:
        key, value = parse_override(text)
        data[key] = value
    return settings_from_mapping(data)
