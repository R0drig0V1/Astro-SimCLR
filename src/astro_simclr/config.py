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
schedule of Carrasco-Davis et al. 2021, whose own fields it validates; the paper model's input,
leakage and Adam fields (task 2.2c) are set by the ``carrasco2021`` preset
(:mod:`astro_simclr.presets`). The ``rj23`` schedule (task 3.4b) trains RJ23's classifiers on a
``raw-alerts`` layout with their regime (``rj23_regime``); its ``rj23_*`` fields are set by
the ``rj23-<scenario>`` presets.

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
    "ADAM_BETAS_CHOICES",
    "BACKBONE_WEIGHTS_CHOICES",
    "CARRASCO2021",
    "CARRASCO2021_EVAL_BATCH_SIZE",
    "CHECKPOINT_SELECTIONS",
    "CONTEXTS",
    "CONVNEXT_ARCHS",
    "EPOCHS",
    "FEATURE_SETS",
    "FP32_PRECISIONS",
    "GROUP_ARCHS",
    "LR_SCHEDULES",
    "PAPER_ARCHS",
    "PAPER_INPUTS",
    "RJ23",
    "RJ23_BN_POSITIONS",
    "RJ23_KEEPS",
    "RJ23_LARGE_VALUES",
    "RJ23_MODEL",
    "RJ23_MONITOR",
    "RJ23_SCENARIOS",
    "SCHEDULES",
    "ConfigError",
    "Settings",
    "coerce_value",
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
#: ``Settings.schedule`` of the Reyes-Jainaga et al. 2023 regime (task 3.4b).
RJ23: Final = "rj23"
#: The accepted ``Settings.schedule`` values.
SCHEDULES: Final = (EPOCHS, CARRASCO2021, RJ23)
#: ``Settings.supervised_model`` of the ``rj23`` schedule: ``rj23_model.RJ23Classifier``.
RJ23_MODEL: Final = "rj23"
#: The validation metric of the ``rj23`` schedule: the macro-F1 on the selection part.
RJ23_MONITOR: Final = "macro_f1_val"
#: ``Settings.rj23_scenario``: RJ23's four inputs (``data.rj23_input.VIEW_NAMES``).
RJ23_SCENARIOS: Final = ("full", "cropped", "lowres", "multiscale")
#: ``Settings.rj23_bn_position`` (``rj23_model.BNPosition``; ADR 0013's sensitivity arm).
RJ23_BN_POSITIONS: Final = ("after_concat", "before_concat")
#: ``Settings.rj23_keep``: the primary weights of a run (``rj23_regime.KEEPS``).
RJ23_KEEPS: Final = ("stop", "best")
#: ``Settings.rj23_large_values`` (``data.rj23_input.LargeValues``; ADR 0009).
RJ23_LARGE_VALUES: Final = ("paper", "code")
#: The evaluation batch the ``carrasco2021`` schedule requires (TF's batch of 64).
CARRASCO2021_EVAL_BATCH_SIZE: Final = 64

#: ``Settings.paper_input``: ``float32`` is the paper's input (2.1's float path), ``uint8``
#: the interim crops of the prepared uint8 stamps (task 2.2a; a diagnostic).
PAPER_INPUTS: Final = ("float32", "uint8")
#: ``Settings.adam_betas``: ``code`` is ALeRCE's code (0.9, 0.999), ``paper`` the paper text's
#: unused (0.5, 0.9) (a diagnostic; ADR 0003).
ADAM_BETAS_CHOICES: Final = ("code", "paper")

#: The ``paper_arch`` values that are timm ConvNeXt backbones on the whole 63x63 stamp
#: (plan row B8; ``convnext_model``, ADR 0022).
CONVNEXT_ARCHS: Final = ("convnext_atto", "convnext_pico")
#: ``Settings.paper_arch`` (task 2.5-models): the faithful 4-rotation model, its single pass,
#: the single-pass p4 / p4m group-equivariant CNNs (``paper_variants.PAPER_ARCHS``) and the
#: 4-rotation ConvNeXt-atto / pico at 63 px (:data:`CONVNEXT_ARCHS`).
PAPER_ARCHS: Final = ("rot4", "single", "p4", "p4m", *CONVNEXT_ARCHS)
#: The ``paper_arch`` values that are G-CNNs (the only ones ``group_fields`` applies to).
GROUP_ARCHS: Final = ("p4", "p4m")
#: ``Settings.backbone_weights`` (plan row B8; ADR 0022): ``none`` trains from scratch;
#: ``zoobot`` starts ``convnext_pico``'s backbone from the pinned Zoobot weights
#: (``convnext_model.BACKBONE_WEIGHTS``), downloaded at run time.
BACKBONE_WEIGHTS_CHOICES: Final = ("none", "zoobot")
#: ``Settings.context`` (task 2.5-models; ``paper_variants.CONTEXTS``).
CONTEXTS: Final = ("none", "avgpool63")

#: ``Settings.feature_set`` (task 2.5 diagnostic; ADR 0016): the alert features the paper
#: model's head reads: ``all`` 23 (faithful), ``position`` the 6 sky coordinates, ``none``
#: (the images only; the same model as ``with_features: false``). See
#: :mod:`astro_simclr.feature_sets`.
FEATURE_SETS: Final = ("all", "position", "none")

#: ``Settings.checkpoint_selection`` (plan row D3): ``best_val_loss`` scores the weights of
#: the best validation loss (CD21's protocol), ``final`` the weights at the end of the run
#: (the EMA shadow's with ``ema_decay``); validation still runs and is logged.
CHECKPOINT_SELECTIONS: Final = ("best_val_loss", "final")
#: ``Settings.lr_schedule`` (plan row D3; ``training.lr_schedule``): ``constant`` (CD21's
#: Adam 1e-3) or ``cosine`` over exactly ``max_steps``.
LR_SCHEDULES: Final = ("constant", "cosine")

#: ``Settings.fp32_precision`` (ADR 0010): ``ieee`` keeps float32 convolutions and matmuls
#: in full float32 on every device; ``tf32`` lets Ampere and newer CUDA GPUs run them in
#: TensorFloat-32 (10-bit mantissa; Pascal, CPU and MPS have no TF32 and stay float32).
FP32_PRECISIONS: Final = ("ieee", "tf32")

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
    #: float32 convolutions and matmuls; see :data:`FP32_PRECISIONS` (ADR 0010). Applied by
    #: ``training.runs.apply_fp32_precision`` before any module is built, and recorded with
    #: its effect in ``metrics.json["machine"]``. torch's own default is TF32 convolutions.
    fp32_precision: str = "ieee"
    #: Upper bound on training epochs.
    max_epochs: int = 220
    #: EarlyStopping patience, in epochs; for the ``rj23`` schedule, in validations.
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
    #: Redraw Lightning's progress bar every this many batches (``TQDMProgressBar``); 0
    #: hides it. Lightning's default, 1, costs ~0.4 ms a step on a 30,000-step run
    #: (``carrasco2021``); no effect on the numbers.
    progress_bar_refresh_rate: int = 50

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

    #: ``epochs`` (the legacy schedule: ``max_epochs``, EarlyStopping, best by epoch),
    #: ``carrasco2021`` (ALeRCE's: steps, validation every ``val_every_n_steps``, the horizon
    #: rule; ``training.schedule``) or ``rj23`` (RJ23's: validation every
    #: ``val_every_n_steps`` from step 0, ``patience`` validations, the ``max_steps`` cap;
    #: ``rj23_regime``). See :data:`SCHEDULES`.
    schedule: str = "epochs"
    #: ``carrasco2021``: the initial horizon, in optimizer steps (the Trainer's ``max_steps``).
    #: ``rj23``: the hard cap, a multiple of ``val_every_n_steps``.
    max_steps: int = 30_000
    #: ``carrasco2021`` / ``rj23``: validate (and select) every this many optimizer steps.
    val_every_n_steps: int = 10
    #: ``carrasco2021``: a new best at step ``s`` raises the horizon to ``s + horizon_increment``.
    horizon_increment: int = 10_000
    #: ``carrasco2021``: write ``last.ckpt`` (what ``--resume`` continues from) after the
    #: validation of every this many optimizer steps (a multiple of ``val_every_n_steps``).
    checkpoint_every_n_steps: int = 1_000
    #: ``epochs``: stop early with EarlyStopping (``patience``, ``min_delta``). The
    #: ``carrasco2021`` schedule has none and requires false.
    early_stopping: bool = True

    # --- The paper model's input, data and optimizer (task 2.2c; ADR 0003) -------------------

    #: ``paper_stamp_classifier`` input; see :data:`PAPER_INPUTS`. ``float32`` reads
    #: ``images_f32.npy`` (``prepare-data --images float32``) and applies
    #: ``data.float_input.crop_split`` (21x21, per-crop min-max, NaN -> 0).
    paper_input: str = "float32"
    #: ``paper_input: float32``: ``crop_split(nan_approximation=...)``. False treats every
    #: stored zero as a real pixel (a diagnostic of task 2.4).
    nan_approximation: bool = True
    #: ``paper_stamp_classifier``: drop the Train stamps that duplicate a Validation or Test
    #: stamp (``protocol.leakage``; ``red`` Train 2995), from Train only.
    exclude_leaked_train: bool = True
    #: ``paper_stamp_classifier``: the Adam / AdamW betas; see :data:`ADAM_BETAS_CHOICES`.
    #: The phase-1 modules ignore it and keep the legacy (0.5, 0.9).
    adam_betas: str = "code"

    # --- Paper-model architecture variants (task 2.5-models; ADR 0005) ----------------------

    #: ``paper_stamp_classifier`` trunk; see :data:`PAPER_ARCHS`. ``rot4`` (faithful) runs
    #: the paper's trunk on the 4 rotations and averages after fc1; ``single`` runs it once
    #: (not invariant); ``p4`` / ``p4m`` are single-pass group-equivariant CNNs
    #: (``paper_variants``; plan rows B7 and B2').
    paper_arch: str = "rot4"
    #: ``paper_arch: convnext_*``: the backbone's initial weights; see
    #: :data:`BACKBONE_WEIGHTS_CHOICES`. ``zoobot`` needs ``paper_arch: convnext_pico`` and
    #: network access (or the Hugging Face cache) at the start of the run.
    backbone_weights: str = "none"
    #: ``paper_arch: p4 | p4m``: fields of conv1-conv2 (conv3-conv5 have twice as many).
    #: None: ``32 // |G|``, the paper's 32/64 maps (p4 8/16, p4m 4/8 fields). Only the
    #: G-CNNs accept it.
    group_fields: int | None = None
    #: ``paper_stamp_classifier`` context input; see :data:`CONTEXTS`. ``avgpool63`` adds
    #: the 63x63 stamp average-pooled to 21x21 as 3 more channels (plan row B1';
    #: ``data.float_input.context_split``). Needs ``paper_input: float32``.
    context: str = "none"
    #: ``paper_stamp_classifier``: rotate each training stamp by a random multiple of 90
    #: degrees (drawn from ``seed`` and the step, so a resumed run repeats it). The control
    #: arm of the single-pass model; a no-op in exact arithmetic for the invariant archs.
    rot90_augment: bool = False
    #: ``paper_stamp_classifier``: the alert features its head reads; see
    #: :data:`FEATURE_SETS`. ``all`` is faithful; ``position`` keeps ``ra``, ``dec`` and the
    #: ecliptic and galactic coordinates of the 23 (the features BN and fc2 shrink to them);
    #: ``none`` drops them, bit-identical to ``with_features: false``. ``position`` needs
    #: ``with_features: true``. A diagnostic: how much an images-only or images+position
    #: model loses (ADR 0016).
    feature_set: str = "all"

    # --- Paper-model class-aware augmentation (plan rows C5 and C6; ADR 0006) ---------------

    #: ``paper_stamp_classifier`` C5: the largest background-noise scale added to the
    #: science and template of each training crop, in units of the crop channel's own
    #: background sigma (``training.class_augment``); the difference follows them. 0: off.
    aug_noise: float = 0.0
    #: C5: the sigma, in standardised units, of the Gaussian jitter on the continuous
    #: training features (``class_augment.DISCRETE_FEATURES`` never change). 0: off.
    aug_feature_jitter: float = 0.0
    #: C6: the fraction of each training batch replaced by synthetic hot-pixel Bogus,
    #: in [0, 1). 0: off.
    synthetic_bogus: float = 0.0
    #: C5 / C6: a stamp whose ``diff ~ a*sci + b*tmpl + c`` fit has a lower R^2 is not
    #: noised nor used as a synthetic Bogus source. In [0, 1].
    aug_min_r2: float = 0.5

    # --- Paper-model transient flux rescaling (plan row C7; ADR 0007) ------------------------

    #: ``paper_stamp_classifier`` C7: the probability that a training row (not Bogus) has its
    #: central transient rescaled by ``k`` (``training.flux_rescale``), in [0, 1]. 0: off.
    flux_rescale: float = 0.0
    #: C7: the smallest ``k`` (the alert limit may raise it per stamp); 0 < it <= flux_k_max.
    flux_k_min: float = 0.5
    #: C7: the largest ``k``.
    flux_k_max: float = 1.5

    # --- The RJ23 regime (task 3.4b; ADR 0023) ----------------------------------------------

    #: ``rj23``: the input and its Table 2 classifier; see :data:`RJ23_SCENARIOS`.
    rj23_scenario: str = "full"
    #: ``rj23``: the BN after the position concat (ADR 0013) or before it (the sensitivity
    #: arm); see :data:`RJ23_BN_POSITIONS`.
    rj23_bn_position: str = "after_concat"
    #: ``rj23``: the weights whose predictions are the run's primary ones, ``stop`` (RJ23's)
    #: or ``best``; both are evaluated and written. See :data:`RJ23_KEEPS`.
    rj23_keep: str = "stop"
    #: ``rj23``: ``data.rj23_input.normalise(large_values=...)``; see :data:`RJ23_LARGE_VALUES`.
    rj23_large_values: str = "paper"

    # --- Paper-model weight EMA, learning-rate schedule and selection (plan row D3) ---------

    #: ``carrasco2021``: which weights the run scores; see :data:`CHECKPOINT_SELECTIONS`.
    #: ``final`` needs ``horizon_increment: 0`` (the run's length must not depend on the
    #: validation either).
    checkpoint_selection: str = "best_val_loss"
    #: ``carrasco2021``: the learning rate per step; see :data:`LR_SCHEDULES`. ``cosine``
    #: anneals to 0 over exactly ``max_steps`` and needs ``horizon_increment: 0``.
    lr_schedule: str = "constant"
    #: ``carrasco2021``: linear warmup of the learning rate over the first this many steps
    #: (``training.lr_schedule``), fewer than ``max_steps``; 0: none. Needs
    #: ``horizon_increment: 0``.
    lr_warmup_steps: int = 0
    #: ``carrasco2021``: the decay of the weight EMA (``training.ema``), in (0, 1), e.g.
    #: 0.999; the model is then validated, selected and scored with the EMA weights.
    #: None: off.
    ema_decay: float | None = None
    #: ``ema_decay``: ramp the decay as ``min(ema_decay, (1 + step) / (10 + step))``.
    ema_warmup: bool = False

    def __post_init__(self) -> None:
        """Check the values that a type alone cannot (ranges and names).

        Raises:
            ConfigError: if any value is out of range.
        """
        if self.accelerator not in ACCELERATORS:
            valid = ", ".join(sorted(ACCELERATORS))
            raise ConfigError(f"accelerator {self.accelerator!r} is not one of: {valid}")
        self._check_fp32_precision()
        if isinstance(self.devices, int) and self.devices < 1:
            raise ConfigError(f"devices must be >= 1, got {self.devices}")
        for name in ("max_epochs", "eval_batch_size"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("num_workers", "patience", "seed", "progress_bar_refresh_rate"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.min_delta < 0:
            raise ConfigError(f"min_delta must be >= 0, got {self.min_delta}")
        for name in ("monitor", "dataset", "optimizer", "augmentation", "encoder_name"):
            if not getattr(self, name):
                raise ConfigError(f"{name} must not be empty")
        self._check_hyperparameters()
        self._check_paper_model()
        self._check_paper_variants()
        self._check_schedule()
        self._check_rj23()
        self._check_weight_averaging()

    def _check_fp32_precision(self) -> None:
        if self.fp32_precision not in FP32_PRECISIONS:
            valid = ", ".join(FP32_PRECISIONS)
            raise ConfigError(f"fp32_precision {self.fp32_precision!r} is not one of: {valid}")

    def _check_paper_model(self) -> None:
        for name, valid in (("paper_input", PAPER_INPUTS), ("adam_betas", ADAM_BETAS_CHOICES)):
            value = getattr(self, name)
            if value not in valid:
                raise ConfigError(f"{name} {value!r} is not one of: {', '.join(valid)}")

    def _check_paper_variants(self) -> None:
        for name, valid in (
            ("paper_arch", PAPER_ARCHS),
            ("context", CONTEXTS),
            ("feature_set", FEATURE_SETS),
            ("backbone_weights", BACKBONE_WEIGHTS_CHOICES),
        ):
            value = getattr(self, name)
            if value not in valid:
                raise ConfigError(f"{name} {value!r} is not one of: {', '.join(valid)}")
        self._check_convnext()
        if self.group_fields is not None:
            if self.paper_arch not in GROUP_ARCHS:
                msg = (
                    f"group_fields applies to paper_arch {' or '.join(GROUP_ARCHS)} only, "
                    f"not {self.paper_arch!r}"
                )
                raise ConfigError(msg)
            if self.group_fields < 1:
                raise ConfigError(f"group_fields must be >= 1, got {self.group_fields}")
        self._check_class_augment()
        self._check_flux_rescale()
        if self.context != "none" and self.paper_input != "float32":
            msg = (
                f"context {self.context!r} is part of the float input path: it needs "
                f"paper_input=float32, got {self.paper_input!r}"
            )
            raise ConfigError(msg)

    def _check_convnext(self) -> None:
        """The ConvNeXt rows (B8) take the whole stamp: the 21x21-crop flags do not apply."""
        if self.backbone_weights == "zoobot" and self.paper_arch != "convnext_pico":
            msg = (
                "backbone_weights 'zoobot' exists for paper_arch convnext_pico only, "
                f"not {self.paper_arch!r}"
            )
            raise ConfigError(msg)
        if self.paper_arch not in CONVNEXT_ARCHS:
            return
        on_crops = [
            name
            for name, off in (
                ("context", "none"),
                ("aug_noise", 0.0),
                ("synthetic_bogus", 0.0),
                ("flux_rescale", 0.0),
            )
            if getattr(self, name) != off
        ]
        if on_crops:
            msg = (
                f"{', '.join(on_crops)} work on the 21x21 crop, but paper_arch "
                f"{self.paper_arch!r} takes the whole 63x63 stamp: turn them off"
            )
            raise ConfigError(msg)

    def _check_flux_rescale(self) -> None:
        if not 0 <= self.flux_rescale <= 1:
            raise ConfigError(f"flux_rescale must be in [0, 1], got {self.flux_rescale}")
        if not 0 < self.flux_k_min <= self.flux_k_max:
            msg = (
                "flux_k_min and flux_k_max must satisfy 0 < flux_k_min <= flux_k_max, got "
                f"{self.flux_k_min} and {self.flux_k_max}"
            )
            raise ConfigError(msg)
        if self.flux_rescale and self.paper_input != "float32":
            msg = (
                "flux_rescale re-normalises the crops like the float input path: it needs "
                f"paper_input=float32, got {self.paper_input!r}"
            )
            raise ConfigError(msg)
        if self.flux_rescale and self.context != "none":
            msg = (
                "flux_rescale changes the 21x21 crop only, not the pooled context: it needs "
                f"context=none, got {self.context!r}"
            )
            raise ConfigError(msg)

    def _check_class_augment(self) -> None:
        for name in ("aug_noise", "aug_feature_jitter"):
            if not getattr(self, name) >= 0:
                raise ConfigError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0 <= self.synthetic_bogus < 1:
            raise ConfigError(f"synthetic_bogus must be in [0, 1), got {self.synthetic_bogus}")
        if not 0 <= self.aug_min_r2 <= 1:
            raise ConfigError(f"aug_min_r2 must be in [0, 1], got {self.aug_min_r2}")
        on_crops = [name for name in ("aug_noise", "synthetic_bogus") if getattr(self, name)]
        if on_crops and self.paper_input != "float32":
            msg = (
                f"{', '.join(on_crops)} re-normalise the crops like the float input path: "
                f"they need paper_input=float32, got {self.paper_input!r}"
            )
            raise ConfigError(msg)
        if self.synthetic_bogus and self.context != "none":
            msg = (
                "synthetic_bogus builds the 21x21 crop only, not a synthetic context: it "
                f"needs context=none, got {self.context!r}"
            )
            raise ConfigError(msg)
        if self.feature_set == "position" and not self.with_features:
            raise ConfigError("feature_set 'position' needs with_features=true")
        if self.aug_feature_jitter and not self.with_features:
            raise ConfigError("aug_feature_jitter needs with_features=true")
        if self.aug_feature_jitter and self.feature_set == "none":
            # The head reads no feature: the jitter would be drawn and then ignored.
            raise ConfigError("aug_feature_jitter needs feature_set 'all' or 'position'")

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

    def _check_rj23(self) -> None:
        choices = (
            ("rj23_scenario", RJ23_SCENARIOS, "full"),
            ("rj23_bn_position", RJ23_BN_POSITIONS, "after_concat"),
            ("rj23_keep", RJ23_KEEPS, "stop"),
            ("rj23_large_values", RJ23_LARGE_VALUES, "paper"),
        )
        for name, valid, _ in choices:
            if getattr(self, name) not in valid:
                msg = f"{name} {getattr(self, name)!r} is not one of: {', '.join(valid)}"
                raise ConfigError(msg)
        if self.schedule == RJ23:
            self._check_rj23_regime()
            return
        changed = [name for name, _, default in choices if getattr(self, name) != default]
        if changed:
            raise ConfigError(f"{', '.join(changed)} apply only to schedule={RJ23}")
        if self.supervised_model == RJ23_MODEL:
            raise ConfigError(f"supervised_model={RJ23_MODEL} needs schedule={RJ23}")

    def _check_rj23_regime(self) -> None:
        # The regime's definitions (rj23_regime.RJ23Schedule), which no setting may bend.
        rules = (
            (self.supervised_model == RJ23_MODEL, f"supervised_model={RJ23_MODEL}"),
            (not self.with_features, "with_features=false (RJ23 read no alert features)"),
            (self.monitor == RJ23_MONITOR, f"monitor={RJ23_MONITOR} (the selection macro-F1)"),
            (not self.early_stopping, "early_stopping=false (its patience is the schedule's)"),
            (self.num_workers == 0, "num_workers=0 (the data stays on the device)"),
            (self.patience >= 1, "patience >= 1"),
            (
                self.max_steps % self.val_every_n_steps == 0,
                "max_steps a multiple of val_every_n_steps (the cap step is validated)",
            ),
            (
                self.checkpoint_every_n_steps % self.val_every_n_steps == 0,
                "checkpoint_every_n_steps a multiple of val_every_n_steps",
            ),
        )
        problems = [what for ok, what in rules if not ok]
        if problems:
            raise ConfigError(f"schedule {RJ23} needs {'; '.join(problems)}")

    def _check_weight_averaging(self) -> None:
        """The plan row D3 flags (EMA, learning-rate schedule, selection; ADR 0014 rules)."""
        for name, valid in (
            ("checkpoint_selection", CHECKPOINT_SELECTIONS),
            ("lr_schedule", LR_SCHEDULES),
        ):
            value = getattr(self, name)
            if value not in valid:
                raise ConfigError(f"{name} {value!r} is not one of: {', '.join(valid)}")
        if self.ema_decay is not None and not 0 < self.ema_decay < 1:
            raise ConfigError(f"ema_decay must be in (0, 1), got {self.ema_decay}")
        if self.ema_warmup and self.ema_decay is None:
            raise ConfigError("ema_warmup ramps the EMA decay: it needs ema_decay")
        if self.lr_warmup_steps < 0:
            raise ConfigError(f"lr_warmup_steps must be >= 0, got {self.lr_warmup_steps}")
        changed = [
            name
            for name, off in (
                ("checkpoint_selection", "best_val_loss"),
                ("lr_schedule", "constant"),
                ("lr_warmup_steps", 0),
                ("ema_decay", None),
            )
            if getattr(self, name) != off
        ]
        if changed and self.schedule != CARRASCO2021:
            # Only the paper model's modules (both engines) read them.
            msg = f"{', '.join(changed)} apply only to schedule {CARRASCO2021}"
            raise ConfigError(msg)
        # A fixed schedule over max_steps, or scoring the final weights, needs a run whose
        # length the validation does not decide.
        fixed = [name for name in changed if name != "ema_decay"]
        if fixed and self.horizon_increment != 0:
            msg = (
                f"{', '.join(fixed)} need a run of exactly max_steps: set "
                f"horizon_increment=0 (got {self.horizon_increment})"
            )
            raise ConfigError(msg)
        if self.lr_warmup_steps >= self.max_steps and self.lr_warmup_steps:
            msg = (
                f"lr_warmup_steps ({self.lr_warmup_steps}) must be smaller than max_steps "
                f"({self.max_steps})"
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


def coerce_value(hint: Any, value: object) -> Any:
    """Convert ``value`` to the type ``hint`` with the rules of :func:`settings_from_mapping`.

    ``hint`` is one of the field types of :class:`Settings` (``int``, ``float``, ``bool``,
    ``str``, ``Path``, ``None``) or a union of them. The nested config
    (:mod:`astro_simclr.config_tree`) uses it, so that both forms accept exactly the same
    values.

    Raises:
        TypeError: with a description of the expected type, if ``value`` does not fit.
    """
    return _coerce(hint, value)


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
