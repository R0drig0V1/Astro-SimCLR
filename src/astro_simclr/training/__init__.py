"""The engine-free training pieces the plain engine's tasks use (ADR 0014).

Submodules (``docs/architecture.md``, section "training", has the design):

- :mod:`~astro_simclr.training.metrics`: the confusion matrices and macro-F1 of the
  classifier results;
- :mod:`~astro_simclr.training.optimizers`: the optimizer factory;
- task 2.2a, the ``carrasco2021`` schedule: :mod:`~astro_simclr.training.tiling` (class
  balance and the shuffled batch stream) and :mod:`~astro_simclr.training.tensor_splits`
  (the float32 / uint8 crops, the split runs' roles and the resume identity check);
- plan rows C5 and C6: :mod:`~astro_simclr.training.class_augment` (the paper model's
  class-aware training augmentation; ADR 0006), C7:
  :mod:`~astro_simclr.training.flux_rescale`, and
  :mod:`~astro_simclr.training.paper_augment` (the per-step augmentation policy that
  composes them);
- plan row D3: :mod:`~astro_simclr.training.ema` (the weight EMA) and
  :mod:`~astro_simclr.training.lr_schedule` (the per-step learning-rate schedule).

Until task 4.8 this package also held the Lightning modules, data modules, callbacks and
Trainer helpers (``modules``, ``datamodules``, ``device_data``, ``runs``, ``schedule``,
``rj23``); they are frozen at the git tag ``lightning-final``. Nothing here imports
Lightning.
"""
