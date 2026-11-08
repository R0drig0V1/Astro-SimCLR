"""Named settings presets: ``train-supervised --preset carrasco2021`` (task 2.2c).

A preset is a mapping of :class:`~astro_simclr.config.Settings` fields, applied by the CLI
right after the subcommand's legacy protocol (``hparams.COMMAND_DEFAULTS``), so that
``--hparams``, ``--config``, ``--set`` and the explicit flags still override it
(``cli.training_settings``). ``configs/phase2/carrasco2021.yaml`` and
``configs/phase2/carrasco2021-no-features.yaml`` hold the same values as files, with their
provenance; a test keeps them equal to :data:`PRESETS`.

The ``carrasco2021`` preset is the paper-faithful replication of Carrasco-Davis et al. 2021
(the ALeRCE stamp classifier, model "M1"), where ALeRCE's TF1 code, not the paper text, is
the authority (ADR 0003):

- the paper model (``paper_stamp_classifier``) on the float path: the stored float32 stamps,
  ``crop_split(21)`` with the NaN approximation, zero-padded to 27 by the model;
- the 23 features (``with_features``; the ``-no-features`` variant drops them, an assumption:
  M1's hyperparameters on the 100-per-class split); ``feature_set: all``, so that the
  ``position`` and ``none`` diagnostics of ADR 0016 are deviations of ``carrasco2021``;
- Train tiled to the largest class; the leaked Train stamp (``red`` Train 2995) removed
  from Train only; batch 64;
- Adam (0.9, 0.999), epsilon 1e-8, lr 1e-3, no weight decay; ``CE - 0.5 H``;
- the iteration schedule of task 2.2a: 30,000 steps, validation (and selection on the
  validation loss, batch 64 in file order) every 10 steps, horizon +10,000 per new best, no
  EarlyStopping; a constant learning rate and no weight EMA (plan row D3's flags off);
- dropout layers present but inactive (``dropout_active: false``; rate 0.5 when active);
- float32 convolutions and matmuls (``fp32_precision: ieee``; ADR 0010), never TF32.

**Diagnostics** (:data:`DIAGNOSTICS`, plan task 2.4) are single settings that move away from
the preset on purpose: ``adam_betas: paper``, ``dropout_active: true``,
``nan_approximation: false`` and ``paper_input: uint8``. Any change to a preset key is
recorded by :func:`preset_report` in the run's ``metrics.json["preset"]`` (``faithful``
false, and the key in ``deviations``), so a diagnostic run can never pass for the
replication.

**The ``rj23-<scenario>`` presets** (task 3.4b; ADR 0023), one per input of Reyes-Jainaga et
al. 2023 (``full``, ``cropped``, ``lowres``, ``multiscale``): the scenario's Table 2
classifier (``supervised_model: rj23``, ``rj23_model.RJ23_TABLE2``) with its Table 2 learning
rate, Keras' Adam (``optimizer: Adam`` is built by ``rj23_regime.keras_adam``: epsilon 1e-7,
Keras' placement), batch 256 drawn class-uniformly per sample, and the ``rj23`` schedule:
the validation macro-F1 every 500 steps from step 0, patience 5, a 100,000-step cap, the
weights at the stop as the primary result (``rj23_keep: stop``). One preset per scenario,
named as the scenario's ``VIEW_NAMES`` entry, so that the learning rate belongs to the
preset and a run with another one is recorded as a deviation (``preset_report``). The data
(``dataset``: a ``raw-alerts`` layout) and the splits (``--splits``) are not part of it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Final

from astro_simclr.config import (
    CARRASCO2021,
    CARRASCO2021_EVAL_BATCH_SIZE,
    RJ23,
    RJ23_MODEL,
    RJ23_MONITOR,
    RJ23_SCENARIOS,
    Settings,
)

__all__ = [
    "CARRASCO2021_NO_FEATURES",
    "CARRASCO2021_PRESET",
    "DIAGNOSTICS",
    "PRESETS",
    "RJ23_LEARNING_RATES",
    "RJ23_PRESETS",
    "preset_for",
    "preset_report",
]

#: Name of the with-features preset (the gated replication).
CARRASCO2021_PRESET: Final = CARRASCO2021
#: Name of the without-features variant (advisory in the CD21 gate).
CARRASCO2021_NO_FEATURES: Final = f"{CARRASCO2021}-no-features"

_CARRASCO2021: Final[Mapping[str, object]] = {
    # Model and input (tasks 2.1, 2.2a): float32 stamps, 21x21 crop, NaN approximation.
    "dataset": "td_ztf_stamp_17_06_20_red-002",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "paper_input": "float32",
    "nan_approximation": True,
    "with_features": True,
    "dropout_active": False,
    "drop_rate": 0.5,
    # Data: the leaked Train stamp out of Train only (task 2.3); tiling is always on.
    "exclude_leaked_train": True,
    "batch_size": 64,
    "eval_batch_size": CARRASCO2021_EVAL_BATCH_SIZE,
    "num_workers": 0,
    # Optimizer and loss: TF1 AdamOptimizer(1e-3, 0.9, 0.999), CE - 0.5 H.
    "optimizer": "Adam",
    "adam_betas": "code",
    "lr": 1e-3,
    "beta_loss": 0.5,
    # Schedule (task 2.2a): selection on the validation loss every 10 steps.
    "schedule": CARRASCO2021,
    "monitor": "loss_val",
    "max_steps": 30_000,
    "val_every_n_steps": 10,
    "horizon_increment": 10_000,
    "early_stopping": False,
    # Architecture (task 2.5-models; ADR 0005): the paper's 4-rotation model, no context.
    # Listed so that a variant run (another value) is recorded as a deviation.
    "paper_arch": "rot4",
    "group_fields": None,
    "context": "none",
    "rot90_augment": False,
    # The ConvNeXt backbone's initial weights (plan row B8; ADR 0022): listed so that a
    # Zoobot-initialised run is recorded as a deviation.
    "backbone_weights": "none",
    # The features the head reads (a 2.5 diagnostic; ADR 0016): all 23. Listed so that a
    # position-only or images-only run is recorded as a deviation.
    "feature_set": "all",
    # Class-aware augmentation (plan rows C5 and C6; ADR 0006): off. Listed for the same
    # reason, so that an augmented run is recorded as a deviation.
    "aug_noise": 0.0,
    "aug_feature_jitter": 0.0,
    "synthetic_bogus": 0.0,
    "aug_min_r2": 0.5,
    # Transient flux rescaling (plan row C7; ADR 0007): off, listed for the same reason.
    "flux_rescale": 0.0,
    "flux_k_min": 0.5,
    "flux_k_max": 1.5,
    # Selection, learning-rate schedule and weight EMA (plan row D3): CD21's selection on the
    # validation loss, a constant learning rate, no EMA. Listed so that a D3 run is
    # recorded as a deviation.
    "checkpoint_selection": "best_val_loss",
    "lr_schedule": "constant",
    "lr_warmup_steps": 0,
    "ema_decay": None,
    "ema_warmup": False,
    # Precision (ADR 0010): float32 convolutions and matmuls, as on the GTX 1050 (no TF32).
    # Listed so that a TF32 run (``fp32_precision: tf32``) is recorded as a deviation.
    "fp32_precision": "ieee",
}

#: RJ23's Table 2 learning rates per scenario, as ``rj23_model.RJ23_TABLE2`` records them
#: (repeated here so that the CLI does not import torch to resolve settings; a test keeps
#: them equal).
RJ23_LEARNING_RATES: Final[Mapping[str, float]] = {
    "full": 7.589e-4,
    "cropped": 3.4627e-4,
    "lowres": 1.125e-3,
    "multiscale": 7.445e-4,
}


def _rj23(scenario: str) -> Mapping[str, object]:
    return {
        # Model and input (tasks 3.3, 3.4; ADRs 0009, 0013).
        "supervised_model": RJ23_MODEL,
        "rj23_scenario": scenario,
        "rj23_bn_position": "after_concat",
        "rj23_large_values": "paper",
        "with_features": False,
        # Optimizer (Keras Adam, task 3.4b) and the class-uniform batch of 256.
        "optimizer": "Adam",
        "lr": RJ23_LEARNING_RATES[scenario],
        "batch_size": 256,
        "num_workers": 0,
        # Schedule (task 3.4b): macro-F1 every 500 steps from step 0, patience 5, 100k cap.
        "schedule": RJ23,
        "monitor": RJ23_MONITOR,
        "val_every_n_steps": 500,
        "patience": 5,
        "max_steps": 100_000,
        "early_stopping": False,
        "rj23_keep": "stop",
        # Precision (ADR 0010), as the carrasco2021 presets.
        "fp32_precision": "ieee",
    }


#: Preset name -> settings of the four RJ23 scenarios (``rj23-full``, ...).
RJ23_PRESETS: Final[Mapping[str, Mapping[str, object]]] = {
    f"{RJ23}-{scenario}": _rj23(scenario) for scenario in RJ23_SCENARIOS
}

#: Preset name -> the settings it sets (``train-supervised`` only).
PRESETS: Final[Mapping[str, Mapping[str, object]]] = {
    CARRASCO2021_PRESET: _CARRASCO2021,
    CARRASCO2021_NO_FEATURES: {**_CARRASCO2021, "with_features": False},
    **RJ23_PRESETS,
}

#: The documented diagnostic flags of the ``carrasco2021`` presets (plan, task 2.4), as the
#: value that turns each one on.
DIAGNOSTICS: Final[Mapping[str, object]] = {
    "dropout_active": True,
    "adam_betas": "paper",
    "nan_approximation": False,
    "paper_input": "uint8",
}


def preset_for(settings: Settings) -> str | None:
    """The preset a run's settings belong to: by schedule and ``with_features`` / scenario.

    A ``carrasco2021`` schedule run is judged against ``carrasco2021`` (with features) or
    ``carrasco2021-no-features``, an ``rj23`` run against ``rj23-<rj23_scenario>``; any
    other run has no preset (``None``).
    """
    if settings.schedule == RJ23:
        return f"{RJ23}-{settings.rj23_scenario}"
    if settings.schedule != CARRASCO2021:
        return None
    return CARRASCO2021_PRESET if settings.with_features else CARRASCO2021_NO_FEATURES


def preset_report(settings: Settings) -> dict[str, Any] | None:
    """How a run's settings compare with its preset (``metrics.json["preset"]``).

    Returns ``None`` outside a preset (:func:`preset_for`), else ``name``, ``faithful``
    (no preset key changed), ``deviations`` (key -> ``{"value", "preset"}``, in preset
    order) and ``diagnostics`` (the :data:`DIAGNOSTICS` keys among them that are turned
    on; ``carrasco2021`` presets only, empty for ``rj23``). Keys outside the preset (seed,
    paths, accelerator, checkpoint interval) are not compared.
    """
    name = preset_for(settings)
    if name is None:
        return None
    values = dataclasses.asdict(settings)
    deviations = {
        key: {"value": values[key], "preset": expected}
        for key, expected in PRESETS[name].items()
        if values[key] != expected
    }
    diagnostics = (
        [key for key, on in DIAGNOSTICS.items() if values[key] == on]
        if settings.schedule == CARRASCO2021
        else []
    )
    return {
        "name": name,
        "faithful": not deviations,
        "deviations": deviations,
        "diagnostics": diagnostics,
    }
