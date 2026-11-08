"""Seeding, model selection and the Trainer of one run, all from ``config.Settings``.

The legacy entry points (``iter_*.py``, ``linear_classifier.py``, ``fine_tuning.py``)
hard-coded these: ``Trainer(gpus=..., benchmark=True, max_epochs=...)``, an EarlyStopping
and a ``ModelCheckpoint_V2`` (a subclass that overrode a private method to drop the ``-v1``
suffix) on ``accuracy_val`` or ``loss_train_enc``, and no seed. Here:

- :func:`apply_fp32_precision` sets torch's float32 policy of convolutions and matmuls from
  ``fp32_precision`` (ADR 0010), and :func:`fp32_precision_record` reads back what a run
  used (``metrics.json["machine"]``); both are ``engine.precision``'s, shared with the
  plain engine;
- :func:`seed_run` calls ``lightning.seed_everything(seed, workers=True)``; it must run
  before the modules are built, because their initial weights consume the generators;
- :func:`build_callbacks` reads ``monitor``, ``patience`` and ``min_delta`` from the
  settings (nothing is hard-coded, so a config can select on ``loss_val``), checks that the
  module logs the monitored metric, and derives ``mode`` from it;
- :func:`build_trainer` takes ``accelerator`` and ``devices`` from the settings (the only
  place a device is chosen; ``gpus=`` no longer exists in Lightning 2), and the progress
  bar's refresh rate from ``progress_bar_refresh_rate``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Literal

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import Logger

from astro_simclr.config import ConfigError, Settings
from astro_simclr.engine.precision import apply_fp32_precision as _apply_fp32_precision
from astro_simclr.engine.precision import fp32_precision_record

__all__ = [
    "MONITOR_MODES",
    "apply_fp32_precision",
    "build_callbacks",
    "build_last_checkpoint",
    "build_trainer",
    "fp32_precision_record",
    "monitor_mode",
    "seed_run",
]

#: Every metric a module can be selected on, and whether larger (``max``) or smaller
#: (``min``) is better.
MONITOR_MODES: Final[dict[str, Literal["min", "max"]]] = {
    "accuracy_val": "max",
    "loss_val": "min",
    "loss_train_enc": "min",
}


def apply_fp32_precision(settings: Settings) -> None:
    """Set the float32 precision of cuDNN convolutions and of matmuls from the settings.

    ``engine.precision.apply_fp32_precision(settings.fp32_precision)``: one implementation
    for both engines (ADR 0010; the details, and why matmuls use the legacy API, are there).
    Call it at the start of every run, before any module is built; the setting is
    process-global, so it also replaces whatever an earlier run in the same process set.
    """
    _apply_fp32_precision(settings.fp32_precision)


def seed_run(settings: Settings) -> int:
    """Seed Python, numpy and torch (and the DataLoader workers) with ``settings.seed``.

    Returns the seed. Call it first in every run, before building modules or data modules.
    """
    return L.seed_everything(settings.seed, workers=True)


def monitor_mode(monitor: str, monitorable: frozenset[str]) -> Literal["min", "max"]:
    """The EarlyStopping/ModelCheckpoint ``mode`` of ``monitor``.

    ``monitorable`` is what the module logs (its ``MONITORABLE``).

    Raises:
        ConfigError: if the module does not log ``monitor``; the message lists what it logs.
    """
    if monitor not in monitorable or monitor not in MONITOR_MODES:
        valid = ", ".join(sorted(monitorable))
        msg = f"monitor {monitor!r} is not a metric this module logs; valid: {valid}"
        raise ConfigError(msg)
    return MONITOR_MODES[monitor]


def build_callbacks(
    settings: Settings,
    monitorable: frozenset[str],
    *,
    dirpath: Path,
    filename: str = "best",
) -> tuple[EarlyStopping, ModelCheckpoint]:
    """EarlyStopping and the best-by-monitor checkpoint of a run.

    Both watch ``settings.monitor``. EarlyStopping stops after ``settings.patience`` epochs
    without an improvement larger than ``settings.min_delta``, on a non-finite value if
    ``settings.check_finite``, and as soon as the value is worse than
    ``settings.divergence_threshold`` if that is set. The checkpoint keeps only the best
    epoch, as ``dirpath/<filename>.ckpt``; with ``enable_version_counter=False`` a rerun
    overwrites it instead of writing ``<filename>-v1.ckpt`` (what the legacy
    ``ModelCheckpoint_V2`` did). For the latest epoch, see :func:`build_last_checkpoint`.

    Raises:
        ConfigError: if the module does not log ``settings.monitor``.
    """
    mode = monitor_mode(settings.monitor, monitorable)
    early_stopping = EarlyStopping(
        monitor=settings.monitor,
        min_delta=settings.min_delta,
        patience=settings.patience,
        mode=mode,
        check_finite=settings.check_finite,
        divergence_threshold=settings.divergence_threshold,
    )
    checkpoint = ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=settings.monitor,
        mode=mode,
        save_top_k=1,
        enable_version_counter=False,
    )
    return early_stopping, checkpoint


def build_last_checkpoint(dirpath: Path, filename: str = "last") -> ModelCheckpoint:
    """A checkpoint of the latest epoch, ``dirpath/<filename>.ckpt``, overwritten every epoch.

    This is what ``--resume`` continues from. It is a second, monitor-less
    ``ModelCheckpoint`` rather than ``save_last=True`` on the best checkpoint: since
    Lightning 2.6, ``save_last`` writes only in the steps where the top-k checkpoint was
    also written, so after an epoch without improvement ``last.ckpt`` would be stale (and a
    resume would repeat epochs and lose the EarlyStopping count).
    """
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=None,
        save_top_k=1,
        enable_version_counter=False,
    )


def build_trainer(
    settings: Settings,
    *,
    callbacks: Sequence[Callback] = (),
    logger: Logger | bool = True,
    **trainer_kwargs: Any,
) -> L.Trainer:
    """A ``lightning.Trainer`` configured from ``settings``.

    ``accelerator``, ``devices``, ``max_epochs``, ``deterministic`` and (when set)
    ``limit_train_batches`` / ``limit_val_batches`` come from the settings; ``benchmark`` is
    the legacy ``True`` unless the run is deterministic (cuDNN ignores determinism when
    benchmarking). Outputs go under ``settings.output_dir``. The progress bar, unless
    ``enable_progress_bar=False`` is passed or ``settings.progress_bar_refresh_rate`` is 0,
    is a ``TQDMProgressBar`` redrawn every ``progress_bar_refresh_rate`` batches.
    ``trainer_kwargs`` override or extend these (for example ``fast_dev_run=True`` or
    ``limit_train_batches=2``).
    """
    callbacks = list(callbacks)
    rate = settings.progress_bar_refresh_rate
    progress_bar = bool(trainer_kwargs.pop("enable_progress_bar", True)) and rate > 0
    if progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=rate))
    kwargs: dict[str, Any] = {
        "accelerator": settings.accelerator,
        "devices": settings.devices,
        "max_epochs": settings.max_epochs,
        "deterministic": settings.deterministic,
        "benchmark": not settings.deterministic,
        "default_root_dir": settings.output_dir,
        "callbacks": callbacks,
        "logger": logger,
        "enable_model_summary": False,
        "enable_progress_bar": progress_bar,
    }
    for name in ("limit_train_batches", "limit_val_batches"):
        value = getattr(settings, name)
        if value is not None:
            kwargs[name] = value
    kwargs.update(trainer_kwargs)
    return L.Trainer(**kwargs)
