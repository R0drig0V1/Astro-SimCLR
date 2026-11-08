"""Training on Lightning 2: modules, data modules, metrics, optimizers and run helpers.

Submodules (``docs/architecture.md``, section "training", has the design):

- :mod:`~astro_simclr.training.modules`: ``SimCLRModule``, ``LinearEvalModule``,
  ``FineTuneModule``, ``SupervisedCEModule``, ``SupervisedCEResNetModule``;
- :mod:`~astro_simclr.training.datamodules`: ``ClassifierDataModule`` and
  ``ContrastiveDataModule`` over the prepared layout;
- :mod:`~astro_simclr.training.metrics`: the shared metrics and confusion matrix;
- :mod:`~astro_simclr.training.optimizers`: the optimizer factory;
- :mod:`~astro_simclr.training.runs`: seeding, EarlyStopping/ModelCheckpoint and the
  Trainer, from ``config.Settings``;
- task 2.2a, the ``carrasco2021`` schedule: :mod:`~astro_simclr.training.tiling` (class
  balance and the shuffled batch stream), :mod:`~astro_simclr.training.device_data`
  (``DeviceTensorDataModule``) and :mod:`~astro_simclr.training.schedule`
  (``HorizonCallback``); ``modules.PaperClassifierModule``;
- plan rows C5 and C6: :mod:`~astro_simclr.training.class_augment` (the paper model's
  class-aware training augmentation; ADR 0006).
- plan row D3: :mod:`~astro_simclr.training.ema` (the weight EMA) and
  :mod:`~astro_simclr.training.lr_schedule` (the per-step learning-rate schedule).

A run is: ``apply_fp32_precision(settings)`` (ADR 0010), ``seed_run(settings)``, build
the module and its data module, ``build_callbacks(settings, Module.MONITORABLE, dirpath=...)``,
``build_trainer(settings, callbacks=...)``, then ``trainer.fit(module, datamodule)``.
Importing this package does **not** import Lightning (task 4.6): the names above load on
first use (PEP 562), so the Lightning-free submodules (``class_augment``, ``flux_rescale``,
``ema``, ``lr_schedule``, ``paper_augment``, ``tensor_splits``, ``metrics``, ``optimizers``,
``tiling``) serve the plain
engine (ADR 0014) too. ``modules``, ``datamodules``, ``device_data``, ``runs`` and
``schedule`` import Lightning.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from astro_simclr.training.datamodules import ClassifierDataModule, ContrastiveDataModule
    from astro_simclr.training.modules import (
        ClassifierModule,
        FineTuneModule,
        LinearEvalModule,
        SimCLRModule,
        SupervisedCEModule,
        SupervisedCEResNetModule,
    )
    from astro_simclr.training.runs import (
        apply_fp32_precision,
        build_callbacks,
        build_trainer,
        seed_run,
    )

#: Where each lazily loaded name lives.
_LAZY: Final[dict[str, str]] = {
    "ClassifierDataModule": "datamodules",
    "ContrastiveDataModule": "datamodules",
    "ClassifierModule": "modules",
    "FineTuneModule": "modules",
    "LinearEvalModule": "modules",
    "SimCLRModule": "modules",
    "SupervisedCEModule": "modules",
    "SupervisedCEResNetModule": "modules",
    "apply_fp32_precision": "runs",
    "build_callbacks": "runs",
    "build_trainer": "runs",
    "seed_run": "runs",
}

__all__ = [
    "ClassifierDataModule",
    "ClassifierModule",
    "ContrastiveDataModule",
    "FineTuneModule",
    "LinearEvalModule",
    "SimCLRModule",
    "SupervisedCEModule",
    "SupervisedCEResNetModule",
    "apply_fp32_precision",
    "build_callbacks",
    "build_trainer",
    "seed_run",
]


def __getattr__(name: str) -> Any:
    """Load ``name`` from its submodule on first use (importing it imports Lightning).

    Raises:
        AttributeError: for a name this package does not export.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
