"""The canonical semantic config of a run and its SHA-256, the config hash (task A0; ADR 0013).

Two runs whose semantic configs are equal train the same thing; the config hash names that
thing, independently of where the run writes, which device it picked or which seed it drew.

**Canonical JSON** (:func:`canonical_json`): sorted keys, no whitespace (separators ``,`` and
``:``), ASCII only (``\\uXXXX`` escapes), floats as Python's ``repr`` (the shortest string
that reads back to the same double: ``0.001``, ``1e-08``), ``-0.0`` written as ``0.0``,
integers and floats kept apart (``2`` is not ``2.0``: ``limit_train_batches`` gives them
different meanings), and NaN or infinity refused. The output is a pure function of the
value, so its SHA-256 is stable across machines, Python versions and processes.

**Semantic config** (:func:`semantic_config`): the document hashed is::

    {"canonical_version": 1, "command": "<training subcommand>", "settings": {...}}

where ``settings`` is the *resolved* :class:`~astro_simclr.config.Settings` (defaults
materialised: a key left out of the YAML and a key set to its default give the same hash)
without

- :data:`NON_SEMANTIC_KEYS`: keys that change where or how fast a run computes, not what
  it computes (each with its reason in the mapping);
- :data:`IDENTITY_KEYS`: keys that are semantic but are their own components of the run
  id (:mod:`astro_simclr.runs.identity`), so that the seeds of one arm share one config
  hash;
- the keys of :data:`HASH_NEUTRAL_DEFAULTS` that hold their neutral value.

Every other ``Settings`` field is semantic, so a new field is hashed by default: that is
the safe direction (a field nobody classified can only make two equal runs look different,
never two different runs look equal).

**Changing the canonical form.** The fixtures ``tests/fixtures/runs/canonical_configs.json``
pin the canonical JSON and the hash of the ``carrasco2021`` presets and of several 2.5
variants, so any change below fails a test. A PR that

- adds a ``Settings`` field whose default reproduces the behaviour before it adds the field
  and that default to :data:`HASH_NEUTRAL_DEFAULTS` (existing hashes stay valid);
- adds a field that changes what existing configs compute, or changes the default of a
  semantic field, changes the hashes on purpose: it regenerates the fixtures, says so in
  the PR, and bumps ``NUMERICS_EPOCH`` if the numerics move;
- changes the rules of this module bumps :data:`CANONICAL_VERSION` (every hash changes).

**Contract for the nested config (task 4.2).** The hash is defined on the *flat view*, a
plain mapping keyed by today's ``Settings`` field names, so a new config representation
proves hash equality by lowering itself to that view: it calls
:func:`semantic_config_from_plain` with ``complete=True`` (which refuses a lowering that
forgets or invents a key) and compares :func:`config_sha256_of` with the flat preset's;
:func:`semantic_diff` names the keys that differ when they do not match. When the flat
``Settings`` is retired, the flat view (its field names, types and this module's rules)
stays the definition of the hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Final

from astro_simclr.config import ConfigError, Settings, settings_to_plain
from astro_simclr.hparams import COMMANDS

__all__ = [
    "CANONICAL_VERSION",
    "HASH_NEUTRAL_DEFAULTS",
    "IDENTITY_KEYS",
    "MISSING",
    "NON_SEMANTIC_KEYS",
    "CanonicalError",
    "canonical_json",
    "config_sha256",
    "config_sha256_of",
    "semantic_config",
    "semantic_config_from_plain",
    "semantic_diff",
    "sha256_text",
]

#: Version of the canonical form, part of every hashed document. Bumped only when the rules
#: of this module change (every config hash changes then).
CANONICAL_VERSION: Final = 1

#: Settings left out of the config hash because they do not change what a run computes,
#: with the reason. The run id does not include them either; the manifest records them.
NON_SEMANTIC_KEYS: Final[Mapping[str, str]] = {
    "accelerator": (
        "the device the run asks for: the device it ran on is the run id's machine class, "
        "and 'auto' and 'cuda' on one GPU compute the same thing"
    ),
    "data_dir": (
        "where the prepared data lives: its content is the run id's data SHA-256 (the "
        "dataset name, which selects the label fraction, stays semantic)"
    ),
    "output_dir": "where the run writes its outputs",
    "num_workers": (
        "DataLoader processes, a speed setting: the carrasco2021 schedule requires 0 (the "
        "data is device-resident); on the legacy CPU-worker path it changes which worker "
        "draws which augmentation (per-worker seeding), a different draw of the same "
        "distribution, treated like GPU non-determinism: a new attempt, not a new run"
    ),
    "progress_bar_refresh_rate": (
        "how often the terminal progress bar is redrawn: display only, it touches no tensor"
    ),
    "checkpoint_every_n_steps": (
        "how often last.ckpt is written, the resume granularity: saving a checkpoint "
        "draws no random numbers and changes no training state"
    ),
}

#: Settings left out of the config hash because they are components of the run id of
#: their own (``runs.identity.RunIdentity``), with the reason.
IDENTITY_KEYS: Final[Mapping[str, str]] = {
    "seed": (
        "the run id's seed component: the seeds of one arm then share one config hash, "
        "which is what a comparison groups by"
    ),
    "pretrained": (
        "a path to a checkpoint: what matters is its content, the run id's "
        "pretrained_sha256, not where the file is"
    ),
}

#: Settings fields added after canonical version 1, each with the value that reproduces
#: the behaviour before it. At that value the field is left out of the canonical form, so
#: the hashes of existing configs (and of the manifests that recorded them) do not move.
#: Empty in version 1: every field that existed then is materialised.
HASH_NEUTRAL_DEFAULTS: Final[Mapping[str, object]] = {}

_EXCLUDED: Final = frozenset(NON_SEMANTIC_KEYS) | frozenset(IDENTITY_KEYS)


class _Missing:
    """The type of :data:`MISSING`."""

    def __repr__(self) -> str:
        return "MISSING"


#: :func:`semantic_diff`'s value for a key one side does not have.
MISSING: Final = _Missing()


class CanonicalError(ValueError):
    """A value has no canonical JSON form (a non-finite float, a non-string key, ...)."""


def _plain(value: object, where: str) -> object:
    """``value`` as JSON-ready Python values, normalised for :func:`canonical_json`."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalError(f"{where}: {value!r} has no JSON form")
        # -0.0 == 0.0, but repr tells them apart: one canonical zero.
        return 0.0 if value == 0 else value
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError(f"{where}: key {key!r} is not a string")
            out[key] = _plain(item, f"{where}.{key}")
        return out
    if isinstance(value, list | tuple):
        return [_plain(item, f"{where}[{i}]") for i, item in enumerate(value)]
    raise CanonicalError(f"{where}: {type(value).__name__} {value!r} has no canonical form")


def canonical_json(value: object) -> str:
    """The canonical JSON text of ``value`` (module docstring, "Canonical JSON").

    Accepts ``None``, bools, ints, finite floats, strings, paths (as POSIX strings), and
    mappings with string keys and lists or tuples of those.

    Raises:
        CanonicalError: on a non-finite float, a non-string key or another type.
    """
    plain = _plain(value, "$")
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """The hex SHA-256 of ``text`` encoded as UTF-8 (canonical JSON is ASCII anyway)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_command(command: str) -> None:
    if command not in COMMANDS:
        raise ConfigError(f"command {command!r} is not one of: {', '.join(COMMANDS)}")


def semantic_config_from_plain(
    command: str, config: Mapping[str, object], *, complete: bool = True
) -> dict[str, object]:
    """The semantic config (module docstring) of a flat plain-values settings mapping.

    ``config`` is :func:`~astro_simclr.config.settings_to_plain`'s form: ``Settings`` field
    names, plain JSON values. With ``complete`` (the default; the nested config's lowering
    of task 4.2) its keys must be exactly today's ``Settings`` fields. With
    ``complete=False`` any keys are accepted: a manifest written by an older version may
    lack a field added since (with its :data:`HASH_NEUTRAL_DEFAULTS` value) or hold one
    removed since, and its recorded hash must still be recomputable.

    Raises:
        ConfigError: on an unknown command or, with ``complete``, a missing or unknown key.
        CanonicalError: if a value has no canonical form.
    """
    _check_command(command)
    if complete:
        fields = {field.name for field in dataclasses.fields(Settings)}
        missing, unknown = sorted(fields - set(config)), sorted(set(config) - fields)
        if missing or unknown:
            msg = (
                "a flat settings view needs exactly the Settings fields: "
                f"missing {missing or 'none'}, unknown {unknown or 'none'}"
            )
            raise ConfigError(msg)
    settings: dict[str, object] = {}
    for key, value in config.items():
        if key in _EXCLUDED:
            continue
        plain = _plain(value, key)
        # Compared as canonical text: 1 == 1.0 == True in Python, not in the hash.
        neutral = HASH_NEUTRAL_DEFAULTS.get(key, MISSING)
        if neutral is not MISSING and canonical_json(plain) == canonical_json(neutral):
            continue
        settings[key] = plain
    return {"canonical_version": CANONICAL_VERSION, "command": command, "settings": settings}


def semantic_config(command: str, settings: Settings) -> dict[str, object]:
    """The semantic config of a run of ``command`` with ``settings`` (module docstring).

    Raises:
        ConfigError: if ``command`` is not a training subcommand.
    """
    return semantic_config_from_plain(command, settings_to_plain(settings))


def config_sha256_of(semantic: Mapping[str, object]) -> str:
    """The hex SHA-256 of the canonical JSON of a semantic config (the config hash).

    Raises:
        CanonicalError: if a value has no canonical form.
    """
    return sha256_text(canonical_json(semantic))


def config_sha256(command: str, settings: Settings) -> str:
    """The config hash of a run of ``command`` with ``settings``.

    Raises:
        ConfigError: if ``command`` is not a training subcommand.
    """
    return config_sha256_of(semantic_config(command, settings))


def semantic_diff(
    a: Mapping[str, object], b: Mapping[str, object]
) -> dict[str, tuple[object, object]]:
    """The keys where two semantic configs differ: key -> (value in ``a``, value in ``b``).

    ``command`` and ``canonical_version`` are compared like the settings (under those
    names); a key one side lacks has :data:`MISSING` there. Empty when the hashes agree.
    """
    flat_a = _flatten(a)
    flat_b = _flatten(b)
    diff: dict[str, tuple[object, object]] = {}
    for key in sorted(set(flat_a) | set(flat_b)):
        value_a, value_b = flat_a.get(key, MISSING), flat_b.get(key, MISSING)
        # Compared as canonical text: 2 == 2.0 == True in Python, not in the hash.
        if (
            value_a is MISSING
            or value_b is MISSING
            or canonical_json(value_a) != canonical_json(value_b)
        ):
            diff[key] = (value_a, value_b)
    return diff


def _flatten(semantic: Mapping[str, object]) -> dict[str, object]:
    out = {key: value for key, value in semantic.items() if key != "settings"}
    settings = semantic.get("settings")
    if isinstance(settings, Mapping):
        out.update(settings)
    return out
