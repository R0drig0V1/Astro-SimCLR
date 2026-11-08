"""Resolving a run's nested config: preset, file, ``--set`` overrides, and the flat shim.

``astro-simclr run CONFIG [--set ...]`` (task 4.2; ADR 0017) resolves its config here.

**A nested file** is a YAML mapping whose keys are section names (``run``, ``runtime``,
``data``, ``model``, ``optim``, ``schedule``, ``augment``, ``eval``), each a mapping, and
optionally ``preset: NAME``. It resolves, later wins:

1. the base: ``preset`` if given (:data:`~astro_simclr.config_tree.presets.PRESETS`), else
   the legacy preset of the command its ``model.kind`` implies (``train-supervised`` without
   one), as the flat subcommands start from their legacy protocol;
2. the file's sections;
3. the dotted overrides ``--set section.key=value`` (an unknown path is an error);
4. the flat overrides ``--set key=value`` (no dot; the shim): applied to the flat view,
   then raised back (:func:`~astro_simclr.config_tree.schema.config_from_settings`).

**The tag rule.** Steps 2 and 3 merge a section key by key, except that a ``kind`` that
differs from the base's replaces the whole section: the base's fields belong to another
variant. The dotted overrides form one patch first, so their order does not matter
(``model.kind=resnet model.name=resnet50`` and the reverse are the same).

**A flat file** (any other mapping: ``Settings`` field names, as the flat subcommands read)
needs ``command``: it resolves as ``<command> --config FILE`` does without ``--preset`` or
``--hparams`` (``hparams.COMMAND_DEFAULTS``, then the file), then it is raised to the
nested form, refused if that would lose a key, and the overrides apply as above.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from astro_simclr.config import (
    ConfigError,
    parse_override,
    read_settings_file,
    settings_from_mapping,
)
from astro_simclr.config_tree.presets import LEGACY_PRESETS, preset
from astro_simclr.config_tree.schema import (
    SECTIONS,
    Config,
    config_from_mapping,
    config_from_settings,
    config_to_plain,
    flat_view,
)
from astro_simclr.hparams import COMMAND_DEFAULTS, COMMANDS

__all__ = [
    "PRESET_KEY",
    "is_nested",
    "merge_sections",
    "resolve_config",
    "resolve_mapping",
    "split_overrides",
]

#: The top-level key of a nested file that names its base preset.
PRESET_KEY: Final = "preset"

Tree = dict[str, dict[str, object]]


def is_nested(data: Mapping[str, object]) -> bool:
    """Whether a config mapping is nested: section keys (and ``preset``), sections mappings.

    ``schedule`` is both a section and a flat field; a flat file has a string there. An
    empty mapping is nested (the base preset alone).
    """
    for key, value in data.items():
        if key == PRESET_KEY:
            continue
        if key not in SECTIONS or not isinstance(value, Mapping):
            return False
    return True


def merge_sections(base: Mapping[str, Mapping[str, object]], patch: Mapping[str, object]) -> Tree:
    """``patch``'s sections merged onto ``base`` by the tag rule (module docstring).

    Raises:
        ConfigError: on an unknown section, or a section that is not a mapping with string
            keys.
    """
    out: Tree = {name: dict(values) for name, values in base.items()}
    for name, values in patch.items():
        if name not in SECTIONS:
            msg = f"unknown section {name!r}; sections: {', '.join(SECTIONS)}"
            raise ConfigError(msg)
        if not isinstance(values, Mapping) or not all(isinstance(k, str) for k in values):
            raise ConfigError(f"section {name} must be a mapping of keys, got {values!r}")
        current = out.get(name, {})
        if "kind" in values and values["kind"] != current.get("kind"):
            current = {}
        out[name] = {**current, **values}
    return out


def split_overrides(overrides: Iterable[str]) -> tuple[Tree, dict[str, object]]:
    """Split ``--set`` items into the dotted patch and the flat keys (the shim).

    ``section.key=value`` goes into the patch (later wins), ``key=value`` into the flat
    mapping; values are YAML scalars (``config.parse_override``).

    Raises:
        ConfigError: on a malformed item, an unknown section, or a path of more than two
            parts (the sections hold no nested mappings).
    """
    patch: Tree = {}
    flat: dict[str, object] = {}
    for text in overrides:
        key, value = parse_override(text)
        if "." not in key:
            flat[key] = value
            continue
        section, _, name = key.partition(".")
        if section not in SECTIONS:
            msg = f"--set {key}: unknown section {section!r}; sections: {', '.join(SECTIONS)}"
            raise ConfigError(msg)
        if not name or "." in name:
            raise ConfigError(f"--set {key}: a path is section.key")
        patch.setdefault(section, {})[name] = value
    return patch, flat


def _base_command(sections: Mapping[str, object]) -> str:
    model = sections.get("model")
    kind = model.get("kind") if isinstance(model, Mapping) else None
    if kind is None:
        return "train-supervised"
    for cls in SECTIONS["model"]:
        if kind in getattr(cls, "KINDS", ()):
            command: str = getattr(cls, "COMMAND")  # noqa: B009 - model classes only
            return command
    return "train-supervised"  # the unknown kind fails when the section is built


def _apply_flat(config: Config, flat: Mapping[str, object]) -> Config:
    if not flat:
        return config
    settings = settings_from_mapping({**flat_view(config), **flat})
    return config_from_settings(config.command, settings, engine=config.runtime.engine)


def resolve_mapping(
    data: Mapping[str, object], overrides: Iterable[str] = (), *, command: str | None = None
) -> Config:
    """Resolve a config mapping (a file's content) and ``--set`` items (module docstring).

    ``command`` is required for a flat mapping and refused for a non-empty nested one
    (whose ``model`` decides the command).

    Raises:
        ConfigError: on an unknown preset, section, path, key, kind or command, a value of
            the wrong type, a flat mapping without ``command`` (or a nested one with it), a
            flat mapping that does not fit the nested form, or an invalid result.
    """
    patch, flat = split_overrides(overrides)
    nested = is_nested(data) and not (command is not None and not data)
    if nested:
        if command is not None:
            raise ConfigError("--command is for flat config files; a nested one names its model")
        name = data.get(PRESET_KEY)
        if name is not None and not isinstance(name, str):
            raise ConfigError(f"preset must be a preset name, got {name!r}")
        sections = {key: value for key, value in data.items() if key != PRESET_KEY}
        base = preset(name if name is not None else LEGACY_PRESETS[_base_command(sections)])
        tree = merge_sections(config_to_plain(base), sections)
    else:
        if command is None:
            valid = ", ".join(COMMANDS)
            raise ConfigError(
                "a flat config (Settings keys) needs the training command (--command "
                f"{valid}); a nested one has section keys: {', '.join(SECTIONS)}"
            )
        if command not in COMMANDS:
            raise ConfigError(f"unknown training command {command!r}")
        settings = settings_from_mapping({**COMMAND_DEFAULTS[command], **data})
        tree = config_to_plain(config_from_settings(command, settings))
    config = config_from_mapping(merge_sections(tree, patch))
    return _apply_flat(config, flat)


def resolve_config(
    path: Path, overrides: Iterable[str] = (), *, command: str | None = None
) -> Config:
    """Resolve the config file ``path`` (nested or flat) and ``--set`` items.

    Raises:
        ConfigError: if the file cannot be read, is not a YAML mapping, or as
            :func:`resolve_mapping`.
    """
    return resolve_mapping(read_settings_file(path), overrides, command=command)
