"""The hash migration test of task 4.2 (ADR 0015, rule 10; ADR 0017).

Every flat preset, lowered through ``semantic_config_from_plain(complete=True)``, has the
config hash of its nested form, and the nested forms of the pinned 2.5 variants have the
hashes pinned in ``tests/fixtures/runs/canonical_configs.json`` (task A0). The flat view of
each nested preset is also compared key by key, so a difference names its keys.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, get_type_hints

import pytest
from _canonical_cases import CONFIG_CASES, FIXTURE, case_settings

from astro_simclr.cli import build_parser, training_settings
from astro_simclr.config import (
    Settings,
    read_settings_file,
    settings_from_mapping,
    settings_to_plain,
)
from astro_simclr.config_tree import (
    LEGACY_PRESETS,
    PRESETS,
    SECTIONS,
    config_from_settings,
    config_sha256,
    flat_view,
    resolve_config,
    resolve_mapping,
)
from astro_simclr.config_tree.schema import semantic_config_of
from astro_simclr.hparams import COMMAND_DEFAULTS
from astro_simclr.presets import PRESETS as FLAT_PRESETS
from astro_simclr.runs.canonical import (
    config_sha256_of,
    semantic_config,
    semantic_config_from_plain,
    semantic_diff,
)

ROOT: Final = Path(__file__).resolve().parents[1]

#: The nested form of each pinned case of ``_canonical_cases.CONFIG_CASES``: a preset and
#: dotted overrides, written independently of the flat case.
NESTED_CASES: Final[Mapping[str, tuple[str, tuple[str, ...]]]] = {
    "carrasco2021": ("carrasco2021", ()),
    "carrasco2021-no-features": ("carrasco2021-no-features", ()),
    "carrasco2021+p4": ("carrasco2021", ("model.arch=p4",)),
    "carrasco2021+p4m-fields8": ("carrasco2021", ("model.arch=p4m", "model.group_fields=8")),
    "carrasco2021+single-rot90": ("carrasco2021", ("model.arch=single", "augment.rot90=true")),
    "carrasco2021+avgpool63": ("carrasco2021", ("model.context=avgpool63",)),
    "carrasco2021+c5": ("carrasco2021", ("augment.noise=1.0", "augment.feature_jitter=0.1")),
    "carrasco2021+c6": ("carrasco2021", ("augment.synthetic_bogus=0.1",)),
    "carrasco2021+c7": ("carrasco2021", ("augment.flux_rescale=0.5",)),
    "carrasco2021+tf32": ("carrasco2021", ("runtime.fp32_precision=tf32",)),
    "carrasco2021+position": ("carrasco2021", ("model.feature_set=position",)),
    "carrasco2021+feature-set-none": ("carrasco2021", ("model.feature_set=none",)),
    "carrasco2021+convnext-pico": (
        "carrasco2021",
        ("model.arch=convnext_pico", "data.image_size=63"),
    ),
    "carrasco2021+convnext-pico-zoobot": (
        "carrasco2021",
        ("model.arch=convnext_pico", "data.image_size=63", "model.backbone_weights=zoobot"),
    ),
    "carrasco2021+hybrid": ("carrasco2021", ("model.arch=hybrid",)),
    "carrasco2021+hybrid-p4-wide-cls": (
        "carrasco2021",
        (
            "model.arch=hybrid_p4",
            "model.group_fields=16",
            "model.hybrid_dim=128",
            "model.hybrid_depth=4",
            "model.hybrid_heads=8",
            "model.hybrid_readout=cls",
        ),
    ),
    "pretrain-legacy": ("pretrain-legacy", ()),
    "train-supervised-legacy": ("train-supervised-legacy", ()),
}


def _flat_preset(name: str, command: str) -> Settings:
    """The flat definition of nested preset ``name``: the CLI's resolution of it."""
    return settings_from_mapping({**COMMAND_DEFAULTS[command], **FLAT_PRESETS.get(name, {})})


@pytest.mark.parametrize("name", list(PRESETS))
def test_each_flat_preset_lowers_to_the_hash_of_its_nested_form(name: str) -> None:
    nested = PRESETS[name]
    command = nested.command
    flat = _flat_preset(name, command)
    # The flat view, key by key (non-semantic keys too).
    view, expected = flat_view(nested), settings_to_plain(flat)
    assert {k: (view[k], expected[k]) for k in expected if view[k] != expected[k]} == {}
    # The semantic config and its hash, through the A0 contract.
    lowered = semantic_config_from_plain(command, view, complete=True)
    assert semantic_diff(lowered, semantic_config(command, flat)) == {}
    assert config_sha256(nested) == config_sha256_of(semantic_config(command, flat))


def test_every_preset_kind_has_a_flat_counterpart() -> None:
    assert set(LEGACY_PRESETS.values()) | set(FLAT_PRESETS) == set(PRESETS)
    for command, name in LEGACY_PRESETS.items():
        assert PRESETS[name].command == command


@pytest.mark.parametrize("name", list(CONFIG_CASES))
def test_nested_cases_have_the_pinned_hashes(name: str) -> None:
    pinned = json.loads(FIXTURE.read_text(encoding="utf-8"))["configs"][name]
    base, overrides = NESTED_CASES[name]
    nested = resolve_mapping({"preset": base}, overrides)
    command, flat = case_settings(name)
    assert nested.command == command
    assert semantic_diff(semantic_config_of(nested), semantic_config(command, flat)) == {}
    assert config_sha256(nested) == pinned["config_sha256"]


def test_nested_cases_cover_the_pinned_cases() -> None:
    assert set(NESTED_CASES) == set(CONFIG_CASES)


@pytest.mark.parametrize("name", list(CONFIG_CASES))
def test_flat_cases_raise_losslessly(name: str) -> None:
    command, flat = case_settings(name)
    nested = config_from_settings(command, flat)
    assert nested.to_settings() == flat
    assert config_sha256(nested) == config_sha256_of(semantic_config(command, flat))


_FLAT_FILES: Final = {
    "configs/reference/R1.yaml": "train-supervised",
    "configs/reference/R2.yaml": "train-supervised",
    "configs/reference/R3-pretrain.yaml": "pretrain",
    "configs/reference/R3-linear-eval.yaml": "linear-eval",
    "configs/reference/R4.yaml": "train-supervised",
    "configs/phase2/carrasco2021.yaml": "train-supervised",
    "configs/phase2/carrasco2021-no-features.yaml": "train-supervised",
    "configs/phase3/rj23-full.yaml": "train-supervised",
    "configs/phase3/rj23-cropped.yaml": "train-supervised",
    "configs/phase3/rj23-lowres.yaml": "train-supervised",
    "configs/phase3/rj23-multiscale.yaml": "train-supervised",
}


def test_flat_files_list_every_reference_config() -> None:
    files = {p.relative_to(ROOT).as_posix() for p in ROOT.glob("configs/*/*.yaml")}
    assert {f for f in files if not f.startswith("configs/runs/")} == set(_FLAT_FILES)


@pytest.mark.parametrize(("path", "command"), list(_FLAT_FILES.items()))
def test_the_shim_reads_the_flat_configs_with_their_old_hash(path: str, command: str) -> None:
    # What `<command> --config PATH` resolved to: the legacy protocol, then the file (the
    # legacy commands are frozen at lightning-final since task 4.8; their configs still read).
    old = settings_from_mapping({**COMMAND_DEFAULTS[command], **read_settings_file(ROOT / path)})
    if command == "train-supervised":
        args = build_parser().parse_args([command, "--config", str(ROOT / path)])
        assert training_settings(command, args) == old
    nested = resolve_config(ROOT / path, command=command)
    assert nested.to_settings() == old
    assert config_sha256(nested) == config_sha256_of(semantic_config(command, old))


def test_every_flat_field_has_one_nested_home() -> None:
    """A Settings field added without a nested field fails here (ADR 0017)."""
    homes: dict[str, set[str]] = {}
    for section, classes in SECTIONS.items():
        for cls in classes:
            fixed = getattr(cls, "FLAT_FIXED", {})
            targets = [f.metadata["flat"] for f in dataclasses.fields(cls) if "flat" in f.metadata]
            for flat in [*targets, *fixed]:
                homes.setdefault(flat, set()).add(section)
    assert set(homes) == {f.name for f in dataclasses.fields(Settings)}
    # One section per flat field, so two sections never write the same key.
    assert {flat: sections for flat, sections in homes.items() if len(sections) > 1} == {}


#: Flat fields that a variant's tag (or the ResNet name) lowers to: their default is the
#: variant's own, not the flat default.
_TAG_TARGETS: Final = ("supervised_model", "optimizer", "schedule")


def test_nested_defaults_are_the_flat_defaults() -> None:
    defaults = settings_to_plain(Settings())
    for classes in SECTIONS.values():
        for cls in classes:
            obj = cls()
            for f in dataclasses.fields(cls):
                if f.metadata.get("flat") not in (None, *_TAG_TARGETS):
                    value = getattr(obj, f.name)
                    plain = str(value) if isinstance(value, Path) else value
                    assert plain == defaults[f.metadata["flat"]], (cls.__name__, f.name)


def test_nested_fields_have_the_flat_types() -> None:
    """Both forms coerce a value the same way, so the raise check compares like with like."""
    flat_hints = get_type_hints(Settings)
    for classes in SECTIONS.values():
        for cls in classes:
            hints = get_type_hints(cls)
            for f in dataclasses.fields(cls):
                if "flat" in f.metadata:
                    assert hints[f.name] == flat_hints[f.metadata["flat"]], (cls.__name__, f.name)
