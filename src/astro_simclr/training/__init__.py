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

A run is: ``seed_run(settings)``, build the module and its data module,
``build_callbacks(settings, Module.MONITORABLE, dirpath=...)``,
``build_trainer(settings, callbacks=...)``, then ``trainer.fit(module, datamodule)``.
Importing this package imports Lightning; the CLI imports it inside its handlers.
"""

from astro_simclr.training.datamodules import ClassifierDataModule, ContrastiveDataModule
from astro_simclr.training.modules import (
    ClassifierModule,
    FineTuneModule,
    LinearEvalModule,
    SimCLRModule,
    SupervisedCEModule,
    SupervisedCEResNetModule,
)
from astro_simclr.training.runs import build_callbacks, build_trainer, seed_run

__all__ = [
    "ClassifierDataModule",
    "ClassifierModule",
    "ContrastiveDataModule",
    "FineTuneModule",
    "LinearEvalModule",
    "SimCLRModule",
    "SupervisedCEModule",
    "SupervisedCEResNetModule",
    "build_callbacks",
    "build_trainer",
    "seed_run",
]
