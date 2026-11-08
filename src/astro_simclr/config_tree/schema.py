"""The nested run configuration: sections, tagged unions, lowering and raising (task 4.2).

The flat :class:`~astro_simclr.config.Settings` has about 60 fields, and each training
subcommand reads only some of them. The nested :class:`Config` groups them in eight
sections (ADR 0017), and the sections whose shape depends on a choice are **tagged
unions**: their ``kind`` picks a variant, and a variant holds only the fields that choice
reads.

========  ===========================================================================
section   variants (``kind``) or fields
========  ===========================================================================
run       ``seed``, ``output_dir``
runtime   ``engine``, ``accelerator``, ``devices``, ``num_workers``, ``deterministic``,
          ``fp32_precision``, ``progress_bar_refresh_rate``
data      ``dir``, ``dataset``, ``image_size``, ``batch_size``, ``with_features``
model     ``stamp_classifier``, ``resnet``, ``paper_stamp_classifier``
          (``train-supervised``); ``simclr`` (``pretrain``); ``linear_eval``;
          ``fine_tune``. The model decides the training subcommand (:attr:`Config.command`)
optim     ``Adam`` / ``AdamW``, ``SGD``, ``LARS``
schedule  ``epochs`` (the legacy protocol), ``carrasco2021`` (ALeRCE's iterations)
augment   ``transforms`` (a registry augmentation), ``paper`` (the paper model's
          training augmentation: rot90, C5, C6, C7)
eval      ``batch_size``, ``limit_batches``
========  ===========================================================================

**Defaults.** Every field's default is the default of the ``Settings`` field it lowers to,
so ``Config()`` lowers to ``Settings()`` exactly: the ``train-supervised`` legacy protocol.
Named starting points are presets (:mod:`astro_simclr.config_tree.presets`).

**Lowering** (:func:`flat_view`, :meth:`Config.to_settings`). Each field names its flat
field in its metadata (``flat``); a variant may also fix flat fields (``FLAT_FIXED``: the
``carrasco2021`` schedule has no EarlyStopping). The flat view is the defaults of
``Settings``, overwritten by the fields the config has. It is what the run executes (the
current engine takes a ``Settings``) and what the config hash is defined on (ADR 0015,
rule 10): :func:`config_sha256` goes through
:func:`~astro_simclr.runs.canonical.semantic_config_from_plain` with ``complete=True``.

**Raising** (:func:`config_from_settings`) is the inverse, for the flat shim. The variant
is chosen from the flat values; a flat field that the chosen variants do not hold must be
at its default, or the conversion is refused with the keys named (:class:`ConfigError`):
silently dropping it would change the config hash, and keeping it would need a field the
run does not read. The old subcommands still accept such files.

**Validation.** Every :class:`Config` is valid by construction: its ``__post_init__``
checks the tags, the cross-section rules (the paper model goes with the ``carrasco2021``
schedule and the ``paper`` augmentation, and only with them) and then builds the lowered
``Settings``, whose checks (ranges, names) run with their messages translated to the
nested paths.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, get_type_hints

from astro_simclr.config import (
    CARRASCO2021,
    EPOCHS,
    ConfigError,
    Settings,
    coerce_value,
    settings_from_mapping,
    settings_to_plain,
)
from astro_simclr.runs.canonical import (
    canonical_json,
    config_sha256_of,
    semantic_config_from_plain,
)

__all__ = [
    "ENGINES",
    "RESNETS",
    "SECTIONS",
    "AdamOptim",
    "Carrasco2021Schedule",
    "Config",
    "Data",
    "EpochsSchedule",
    "Eval",
    "FineTuneModel",
    "LARSOptim",
    "LinearEvalModel",
    "PaperAugment",
    "PaperStampClassifier",
    "ResNetClassifier",
    "Run",
    "Runtime",
    "SGDOptim",
    "SimCLRModel",
    "StampClassifier",
    "TransformsAugment",
    "config_from_mapping",
    "config_from_settings",
    "config_json",
    "config_sha256",
    "config_to_plain",
    "flat_view",
    "semantic_config_of",
]

#: ``runtime.engine`` values. ``plain`` (ADR 0014) joins when its loop lands (tasks 4.1, 4.6).
ENGINES: Final = ("lightning",)
#: ``model.name`` of the ``resnet`` variant (``models.build_encoder``'s torchvision ResNets).
RESNETS: Final = ("resnet18", "resnet34", "resnet50", "resnet152")

_DEFAULTS: Final[Mapping[str, object]] = {
    f.name: getattr(Settings(), f.name) for f in dataclasses.fields(Settings)
}
_UNSET: Final = object()


def _flat(name: str, default: object = _UNSET) -> Any:
    """A field that lowers to the flat field ``name``, with that field's default.

    ``default`` overrides the default (a tag that is not the flat default, such as
    ``SGD``). The flat name is checked here, so a typo fails at import.
    """
    if name not in _DEFAULTS:  # pragma: no cover - a programming error, caught at import
        raise AssertionError(f"{name!r} is not a Settings field")
    value = _DEFAULTS[name] if default is _UNSET else default
    return field(default=value, metadata={"flat": name})


def _tag(kind: str) -> Any:
    """The ``kind`` field of a variant whose tag lowers to no flat field."""
    return field(default=kind)


def _check_kind(obj: Any) -> None:
    kinds: tuple[str, ...] = obj.KINDS
    if obj.kind not in kinds:
        msg = f"{type(obj).__name__} kind must be one of: {', '.join(kinds)}; got {obj.kind!r}"
        raise ConfigError(msg)


# --- Plain sections ------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Run:
    """Identity and outputs of the run: the seed (a run-id component, ADR 0015) and where
    every output goes."""

    seed: int = _flat("seed")
    output_dir: Path = _flat("output_dir")  # noqa: RUF009 - a dataclasses.field


@dataclass(frozen=True, kw_only=True)
class Runtime:
    """Where and how the run computes: engine, device, determinism, precision (ADR 0010)."""

    #: The training loop; see :data:`ENGINES`. Not a flat field: it is the run id's
    #: ``engine`` component (ADR 0015), never part of the config hash.
    engine: str = "lightning"
    accelerator: str = _flat("accelerator")
    devices: int | str = _flat("devices")
    num_workers: int = _flat("num_workers")
    deterministic: bool = _flat("deterministic")
    fp32_precision: str = _flat("fp32_precision")
    progress_bar_refresh_rate: int = _flat("progress_bar_refresh_rate")

    def __post_init__(self) -> None:
        """Check the engine name (the other values are checked by ``Settings``).

        Raises:
            ConfigError: if ``engine`` is not in :data:`ENGINES`.
        """
        if self.engine not in ENGINES:
            msg = (
                f"runtime.engine {self.engine!r} is not one of: {', '.join(ENGINES)} "
                "(the plain engine lands with tasks 4.1 and 4.6)"
            )
            raise ConfigError(msg)


@dataclass(frozen=True, kw_only=True)
class Data:
    """The prepared dataset and what the training loader feeds."""

    #: Parent directory of the prepared datasets (the run reads ``dir/<dataset>``).
    dir: Path = _flat("data_dir")  # noqa: RUF009 - a dataclasses.field
    dataset: str = _flat("dataset")
    image_size: int = _flat("image_size")
    #: Training batch size (SimCLR: pairs per batch).
    batch_size: int = _flat("batch_size")
    #: Feed (and concatenate) the 23 alert features.
    with_features: bool = _flat("with_features")


@dataclass(frozen=True, kw_only=True)
class Eval:
    """The validation and test loaders."""

    batch_size: int = _flat("eval_batch_size")
    #: ``Trainer(limit_val_batches=...)``: int batches, float fraction, null every batch.
    limit_batches: int | float | None = _flat("limit_val_batches")


# --- model: tagged union -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class StampClassifier:
    """The phase-1 stamp classifier (``train-supervised``, ``SupervisedCEModule``)."""

    KINDS: ClassVar = ("stamp_classifier",)
    COMMAND: ClassVar = "train-supervised"
    kind: str = _flat("supervised_model", "stamp_classifier")
    drop_rate: float = _flat("drop_rate")
    #: ``P_stamps_loss`` entropy weight.
    beta_loss: float = _flat("beta_loss")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class ResNetClassifier:
    """A torchvision ResNet classifier (``train-supervised``, ``SupervisedCEResNetModule``)."""

    KINDS: ClassVar = ("resnet",)
    COMMAND: ClassVar = "train-supervised"
    kind: str = _tag("resnet")
    #: One of :data:`RESNETS`.
    name: str = _flat("supervised_model", "resnet18")
    beta_loss: float = _flat("beta_loss")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind`` or a name outside :data:`RESNETS`."""
        _check_kind(self)
        if self.name not in RESNETS:
            raise ConfigError(f"model.name {self.name!r} is not one of: {', '.join(RESNETS)}")


@dataclass(frozen=True, kw_only=True)
class PaperStampClassifier:
    """The paper model of Carrasco-Davis et al. 2021 and its 2.5 variants (ADR 0003, 0005).

    Its input fields (``input``, ``nan_approximation``, ``exclude_leaked_train``) are here,
    not in ``data``, because only this model reads them.
    """

    KINDS: ClassVar = ("paper_stamp_classifier",)
    COMMAND: ClassVar = "train-supervised"
    kind: str = _flat("supervised_model", "paper_stamp_classifier")
    #: ``rot4`` (faithful), ``single``, ``p4``, ``p4m`` (``config.PAPER_ARCHS``).
    arch: str = _flat("paper_arch")
    #: ``p4`` / ``p4m`` only: fields of conv1-conv2; null is ``32 // |G|``.
    group_fields: int | None = _flat("group_fields")
    #: ``none`` or ``avgpool63`` (``config.CONTEXTS``).
    context: str = _flat("context")
    #: ``float32`` (the paper's input) or ``uint8`` (a diagnostic).
    input: str = _flat("paper_input")
    nan_approximation: bool = _flat("nan_approximation")
    exclude_leaked_train: bool = _flat("exclude_leaked_train")
    #: Apply the dropout layers (rate ``drop_rate``); false is faithful.
    dropout_active: bool = _flat("dropout_active")
    drop_rate: float = _flat("drop_rate")
    beta_loss: float = _flat("beta_loss")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class SimCLRModel:
    """SimCLR pretraining: an encoder, its projection head and NT-Xent (``pretrain``)."""

    KINDS: ClassVar = ("simclr",)
    COMMAND: ClassVar = "pretrain"
    kind: str = _tag("simclr")
    #: A ``models.build_encoder`` name: ``stamp<k>``, ``pstamps``, ``resnet*``.
    encoder: str = _flat("encoder_name")
    projection_dim: int = _flat("projection_dim")
    temperature: float = _flat("temperature")
    #: NT-Xent's legacy denominator (ADR 0002).
    legacy_denominator: bool = _flat("nt_xent_legacy_denominator")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class LinearEvalModel:
    """A linear classifier on a frozen pretrained encoder (``linear-eval``)."""

    KINDS: ClassVar = ("linear_eval",)
    COMMAND: ClassVar = "linear-eval"
    kind: str = _tag("linear_eval")
    #: The pretrain checkpoint, or a pretrain run directory (a run-id component).
    pretrained: Path | None = _flat("pretrained")  # noqa: RUF009 - a dataclasses.field
    beta_loss: float = _flat("beta_loss")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class FineTuneModel:
    """A pretrained encoder and a head, trained together (``fine-tune``)."""

    KINDS: ClassVar = ("fine_tune",)
    COMMAND: ClassVar = "fine-tune"
    kind: str = _tag("fine_tune")
    pretrained: Path | None = _flat("pretrained")  # noqa: RUF009 - a dataclasses.field
    beta_loss: float = _flat("beta_loss")
    drop_rate: float = _flat("drop_rate")
    #: The run trains with ``optim.lr * lr_factor``.
    lr_factor: float = _flat("fine_tune_lr_factor")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


Model = (
    StampClassifier
    | ResNetClassifier
    | PaperStampClassifier
    | SimCLRModel
    | LinearEvalModel
    | FineTuneModel
)


# --- optim: tagged union -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class AdamOptim:
    """Adam or AdamW (``training.optimizers``)."""

    KINDS: ClassVar = ("AdamW", "Adam")
    kind: str = _flat("optimizer")
    lr: float = _flat("lr")
    #: ``code`` (0.9, 0.999) or ``paper`` (0.5, 0.9); read by the paper model only.
    betas: str = _flat("adam_betas")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class SGDOptim:
    """Plain SGD."""

    KINDS: ClassVar = ("SGD",)
    kind: str = _flat("optimizer", "SGD")
    lr: float = _flat("lr")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class LARSOptim:
    """LARS (``optim.LARS``), the SimCLR optimizer."""

    KINDS: ClassVar = ("LARS",)
    kind: str = _flat("optimizer", "LARS")
    lr: float = _flat("lr")
    #: Keep the legacy always-true trust-ratio test.
    legacy_trust_ratio: bool = _flat("lars_legacy_trust_ratio")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


Optim = AdamOptim | SGDOptim | LARSOptim


# --- schedule: tagged union ----------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class EpochsSchedule:
    """The legacy epoch protocol: ``max_epochs``, EarlyStopping, the best epoch."""

    KINDS: ClassVar = (EPOCHS,)
    kind: str = _flat("schedule", EPOCHS)
    max_epochs: int = _flat("max_epochs")
    #: The metric EarlyStopping and the best checkpoint watch.
    monitor: str = _flat("monitor")
    early_stopping: bool = _flat("early_stopping")
    patience: int = _flat("patience")
    min_delta: float = _flat("min_delta")
    check_finite: bool = _flat("check_finite")
    divergence_threshold: float | None = _flat("divergence_threshold")
    limit_train_batches: int | float | None = _flat("limit_train_batches")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class Carrasco2021Schedule:
    """ALeRCE's iteration schedule (task 2.2a): steps, validation grid, the horizon rule.

    It has no EarlyStopping, so it lowers ``early_stopping: false`` (``FLAT_FIXED``).
    """

    KINDS: ClassVar = (CARRASCO2021,)
    FLAT_FIXED: ClassVar[Mapping[str, object]] = {"early_stopping": False}
    kind: str = _flat("schedule", CARRASCO2021)
    max_steps: int = _flat("max_steps")
    val_every_n_steps: int = _flat("val_every_n_steps")
    horizon_increment: int = _flat("horizon_increment")
    #: ``last.ckpt`` interval: a multiple of ``val_every_n_steps`` (not semantic, ADR 0015).
    checkpoint_every_n_steps: int = _flat("checkpoint_every_n_steps")
    monitor: str = _flat("monitor")
    limit_train_batches: int | float | None = _flat("limit_train_batches")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


Schedule = EpochsSchedule | Carrasco2021Schedule


# --- augment: tagged union -----------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TransformsAugment:
    """A named augmentation of the ``augmentations`` registry (the phase-1 data modules)."""

    KINDS: ClassVar = ("transforms",)
    kind: str = _tag("transforms")
    #: A registry name or a legacy alias.
    name: str = _flat("augmentation")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


@dataclass(frozen=True, kw_only=True)
class PaperAugment:
    """The paper model's training augmentation: rot90, C5, C6, C7 (ADR 0006, 0007).

    Every one is off at its default, which is the faithful replication.
    """

    KINDS: ClassVar = ("paper",)
    kind: str = _tag("paper")
    rot90: bool = _flat("rot90_augment")
    #: C5: the largest background-noise scale, in units of the crop's background sigma.
    noise: float = _flat("aug_noise")
    #: C5: the jitter sigma on the continuous features.
    feature_jitter: float = _flat("aug_feature_jitter")
    #: C6: the fraction of each batch replaced by synthetic Bogus.
    synthetic_bogus: float = _flat("synthetic_bogus")
    #: C5 / C6: the R^2 below which a stamp is neither noised nor a C6 source.
    min_r2: float = _flat("aug_min_r2")
    #: C7: the probability that a non-Bogus row has its transient rescaled.
    flux_rescale: float = _flat("flux_rescale")
    flux_k_min: float = _flat("flux_k_min")
    flux_k_max: float = _flat("flux_k_max")

    def __post_init__(self) -> None:
        """Raises: ConfigError: on another ``kind``."""
        _check_kind(self)


Augment = TransformsAugment | PaperAugment

#: Section name -> its classes; a section with more than one is a tagged union, and its
#: first class is the default variant.
SECTIONS: Final[Mapping[str, tuple[type, ...]]] = {
    "run": (Run,),
    "runtime": (Runtime,),
    "data": (Data,),
    "model": (
        StampClassifier,
        ResNetClassifier,
        PaperStampClassifier,
        SimCLRModel,
        LinearEvalModel,
        FineTuneModel,
    ),
    "optim": (AdamOptim, SGDOptim, LARSOptim),
    "schedule": (EpochsSchedule, Carrasco2021Schedule),
    "augment": (TransformsAugment, PaperAugment),
    "eval": (Eval,),
}


# --- The config ----------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Config:
    """One run: eight sections, valid by construction (module docstring, "Validation")."""

    run: Run = field(default_factory=Run)
    runtime: Runtime = field(default_factory=Runtime)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=StampClassifier)
    optim: Optim = field(default_factory=AdamOptim)
    schedule: Schedule = field(default_factory=EpochsSchedule)
    augment: Augment = field(default_factory=TransformsAugment)
    eval: Eval = field(default_factory=Eval)

    def __post_init__(self) -> None:
        """Check the section types, the cross-section rules and the lowered settings.

        Raises:
            ConfigError: if a section holds another class than its own, the paper model is
                combined with another schedule or augmentation (or the reverse), or the
                lowered ``Settings`` is invalid (the message names the nested paths).
        """
        for name, classes in SECTIONS.items():
            if not isinstance(getattr(self, name), classes):
                kinds = ", ".join(type_.__name__ for type_ in classes)
                raise ConfigError(f"section {name} must be one of: {kinds}")
        paper = isinstance(self.model, PaperStampClassifier)
        if paper != isinstance(self.schedule, Carrasco2021Schedule):
            msg = (
                "model.kind paper_stamp_classifier and schedule.kind carrasco2021 go "
                f"together (start from preset {CARRASCO2021}); got model.kind "
                f"{self.model.kind} with schedule.kind {self.schedule.kind}"
            )
            raise ConfigError(msg)
        if paper != isinstance(self.augment, PaperAugment):
            msg = (
                "model.kind paper_stamp_classifier and augment.kind paper go together; got "
                f"model.kind {self.model.kind} with augment.kind {self.augment.kind}"
            )
            raise ConfigError(msg)
        self.to_settings()

    @property
    def command(self) -> str:
        """The training subcommand this config runs (decided by ``model``)."""
        command: str = type(self.model).COMMAND
        return command

    def to_settings(self) -> Settings:
        """The lowered flat settings (:func:`flat_view`), validated.

        Raises:
            ConfigError: if ``Settings`` rejects a value; the message adds the nested path
                of every flat name it mentions.
        """
        try:
            return settings_from_mapping(flat_view(self))
        except ConfigError as err:
            raise ConfigError(_translate(str(err), self)) from None


def _sections(config: Config) -> list[tuple[str, Any]]:
    return [(name, getattr(config, name)) for name in SECTIONS]


def _nested_names(config: Config) -> dict[str, str]:
    """Flat field name -> nested path, for the fields of ``config``'s variants."""
    names: dict[str, str] = {}
    for section, obj in _sections(config):
        for f in dataclasses.fields(obj):
            if "flat" in f.metadata:
                names[f.metadata["flat"]] = f"{section}.{f.name}"
    return names


def _translate(message: str, config: Config) -> str:
    names = _nested_names(config)
    words = dict.fromkeys(re.findall(r"[a-z][a-z0-9_]*", message))
    pairs = [f"{word} is {names[word]}" for word in words if word in names]
    return f"{message} (in the nested config: {'; '.join(pairs)})" if pairs else message


def _plain_value(value: object) -> object:
    return str(value) if isinstance(value, Path) else value


def flat_view(config: Config) -> dict[str, object]:
    """The flat view of ``config``: every ``Settings`` field name, as plain values.

    The ``Settings`` defaults, overwritten by each field's ``flat`` target and each
    variant's ``FLAT_FIXED``. Its keys are exactly the ``Settings`` fields, which is what
    :func:`~astro_simclr.runs.canonical.semantic_config_from_plain` checks with
    ``complete=True``.
    """
    view = settings_to_plain(Settings())
    for _, obj in _sections(config):
        for f in dataclasses.fields(obj):
            if "flat" in f.metadata:
                view[f.metadata["flat"]] = _plain_value(getattr(obj, f.name))
        view.update(getattr(type(obj), "FLAT_FIXED", {}))
    return view


def semantic_config_of(config: Config) -> dict[str, object]:
    """The semantic config (ADR 0015) of ``config``, through its flat view.

    Raises:
        ConfigError: if the flat view does not hold exactly the ``Settings`` fields.
    """
    return semantic_config_from_plain(config.command, flat_view(config), complete=True)


def config_sha256(config: Config) -> str:
    """The config hash of ``config``: equal to the flat settings' hash (ADR 0015, rule 10)."""
    return config_sha256_of(semantic_config_of(config))


def config_to_plain(config: Config) -> dict[str, dict[str, object]]:
    """``config`` as plain nested values (paths as strings), ``kind`` first in a union.

    ``config_from_mapping`` of the result gives ``config`` back.
    """
    return {
        name: {f.name: _plain_value(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        for name, obj in _sections(config)
    }


def config_json(config: Config) -> str:
    """The canonical JSON of the nested config (sorted keys, defaults materialised).

    For recording and comparing nested configs; the run identity hashes the flat view
    (:func:`config_sha256`), not this text.
    """
    return canonical_json(config_to_plain(config))


# --- Building from plain values ------------------------------------------------------------


def _variant(section: str, kind: object) -> type:
    classes = SECTIONS[section]
    for cls in classes:
        if kind in getattr(cls, "KINDS", ()):
            return cls
    kinds = [k for cls in classes for k in getattr(cls, "KINDS", ())]
    raise ConfigError(f"{section}.kind {kind!r} is not one of: {', '.join(kinds)}")


def _build_section(section: str, values: Mapping[str, object]) -> Any:
    """One section from plain values: the variant by ``kind``, keys and types checked."""
    classes = SECTIONS[section]
    if len(classes) > 1:
        cls = _variant(section, values.get("kind", classes[0]().kind))
    else:
        cls = classes[0]
    hints = get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(values) - set(names))
    if unknown:
        where = f"{section} ({section}.kind {values['kind']})" if "kind" in values else section
        msg = (
            f"unknown key(s) {', '.join(f'{section}.{key}' for key in unknown)}; "
            f"{where} takes: {', '.join(names)}"
        )
        raise ConfigError(msg)
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        try:
            kwargs[key] = coerce_value(hints[key], value)
        except TypeError as err:
            raise ConfigError(f"{section}.{key} must be {err}, got {value!r}") from None
    return cls(**kwargs)


def config_from_mapping(data: Mapping[str, object]) -> Config:
    """Build a :class:`Config` from plain nested values (``config_to_plain``'s form).

    A missing section takes its default (the first variant of a union); a union section
    without ``kind`` is its first variant. Values are coerced as in
    :func:`~astro_simclr.config.settings_from_mapping`.

    Raises:
        ConfigError: on an unknown section or key (the message names the path and the keys
            the variant takes), a section that is not a mapping, an unknown ``kind``, a
            value of the wrong type, or an invalid config (:class:`Config`).
    """
    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        msg = f"unknown section(s) {', '.join(unknown)}; sections: {', '.join(SECTIONS)}"
        raise ConfigError(msg)
    sections: dict[str, Any] = {}
    for name, values in data.items():
        if not isinstance(values, Mapping):
            raise ConfigError(f"section {name} must be a mapping, got {values!r}")
        sections[name] = _build_section(name, values)
    return Config(**sections)


# --- Raising the flat settings -------------------------------------------------------------


def _model_class(command: str, view: Mapping[str, object]) -> type:
    if command == "pretrain":
        return SimCLRModel
    if command == "linear-eval":
        return LinearEvalModel
    if command == "fine-tune":
        return FineTuneModel
    if command != "train-supervised":
        raise ConfigError(f"unknown training command {command!r}")
    name = view["supervised_model"]
    if name == "stamp_classifier":
        return StampClassifier
    if name == "paper_stamp_classifier":
        return PaperStampClassifier
    if name in RESNETS:
        return ResNetClassifier
    valid = ", ".join(("stamp_classifier", "paper_stamp_classifier", *RESNETS))
    raise ConfigError(f"supervised_model {name!r} is not one of: {valid}")


def _raise_section(cls: type, view: Mapping[str, object], section: str) -> Any:
    values = {
        f.name: view[f.metadata["flat"]] for f in dataclasses.fields(cls) if "flat" in f.metadata
    }
    if len(SECTIONS[section]) > 1:
        kinds: tuple[str, ...] = getattr(cls, "KINDS")  # noqa: B009 - not every class has it
        values = {"kind": kinds[0], **values}
    return _build_section(section, values)


def config_from_settings(command: str, settings: Settings, *, engine: str = "lightning") -> Config:
    """The nested config of a flat run (``command`` with ``settings``): the shim.

    The variants are chosen from the flat values (``supervised_model`` and the command,
    ``optimizer``, ``schedule``; the paper model takes the ``paper`` augmentation). The
    result lowers back to ``settings`` exactly, or the conversion is refused.

    Raises:
        ConfigError: on an unknown command, ``supervised_model`` or ``optimizer``, if the
            nested config is invalid, or if a flat field that the chosen variants do not
            hold is not at its default (the message lists each one).
    """
    view = settings_to_plain(settings)
    model_cls = _model_class(command, view)
    optim_cls = _variant("optim", view["optimizer"])
    schedule_cls = _variant("schedule", view["schedule"])
    augment_cls = PaperAugment if model_cls is PaperStampClassifier else TransformsAugment
    config = Config(
        run=_raise_section(Run, view, "run"),
        runtime=dataclasses.replace(_raise_section(Runtime, view, "runtime"), engine=engine),
        data=_raise_section(Data, view, "data"),
        model=_raise_section(model_cls, view, "model"),
        optim=_raise_section(optim_cls, view, "optim"),
        schedule=_raise_section(schedule_cls, view, "schedule"),
        augment=_raise_section(augment_cls, view, "augment"),
        eval=_raise_section(Eval, view, "eval"),
    )
    lowered = flat_view(config)
    # Compared with the types: 1 == 1.0 == True in Python, not in the hash.
    lost = {
        key: value
        for key, value in view.items()
        if type(lowered[key]) is not type(value) or lowered[key] != value
    }
    if lost:
        shown = ", ".join(f"{key}={value!r}" for key, value in lost.items())
        msg = (
            f"the flat settings do not fit the nested config of {command} "
            f"(model.kind {config.model.kind}, optim.kind {config.optim.kind}, "
            f"schedule.kind {config.schedule.kind}, augment.kind {config.augment.kind}): "
            f"{shown} is not read by that run, so only its default is accepted; drop it, "
            "or run it with the flat subcommand"
        )
        raise ConfigError(msg)
    return config
