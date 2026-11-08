"""The configs and identities pinned by ``tests/fixtures/runs/canonical_configs.json``.

The fixture freezes the canonical form (task A0; ADR 0015): the canonical JSON and the
config hash of the ``carrasco2021`` presets and of several 2.5 variants, and the run id of
a few identities. ``tests/test_runs_canonical.py`` recomputes them. A change that moves a
hash on purpose (module docstring of ``astro_simclr.runs.canonical``, "Changing the
canonical form") regenerates the file with::

    uv run python tests/_canonical_cases.py

and says so in its PR; the diff of the canonical JSON shows exactly which keys moved.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from astro_simclr.config import Settings, settings_from_mapping
from astro_simclr.hparams import COMMAND_DEFAULTS
from astro_simclr.presets import CARRASCO2021_NO_FEATURES, CARRASCO2021_PRESET, PRESETS
from astro_simclr.runs.canonical import canonical_json, config_sha256_of, semantic_config
from astro_simclr.runs.identity import SCREEN, RunIdentity, run_identity

FIXTURE: Final = Path(__file__).parent / "fixtures" / "runs" / "canonical_configs.json"

_CD21: Final = PRESETS[CARRASCO2021_PRESET]

#: name -> (command, the settings on top of the command's legacy protocol).
CONFIG_CASES: Final[Mapping[str, tuple[str, Mapping[str, object]]]] = {
    "carrasco2021": ("train-supervised", _CD21),
    "carrasco2021-no-features": ("train-supervised", PRESETS[CARRASCO2021_NO_FEATURES]),
    # 2.5 architecture variants (ADR 0005).
    "carrasco2021+p4": ("train-supervised", {**_CD21, "paper_arch": "p4"}),
    "carrasco2021+p4m-fields8": (
        "train-supervised",
        {**_CD21, "paper_arch": "p4m", "group_fields": 8},
    ),
    "carrasco2021+single-rot90": (
        "train-supervised",
        {**_CD21, "paper_arch": "single", "rot90_augment": True},
    ),
    "carrasco2021+avgpool63": ("train-supervised", {**_CD21, "context": "avgpool63"}),
    # 2.5 augmentation rows C5, C6, C7 (ADR 0006, 0007) and the TF32 deviation (ADR 0010).
    "carrasco2021+c5": (
        "train-supervised",
        {**_CD21, "aug_noise": 1.0, "aug_feature_jitter": 0.1},
    ),
    "carrasco2021+c6": ("train-supervised", {**_CD21, "synthetic_bogus": 0.1}),
    "carrasco2021+c7": ("train-supervised", {**_CD21, "flux_rescale": 0.5}),
    "carrasco2021+tf32": ("train-supervised", {**_CD21, "fp32_precision": "tf32"}),
    # The feature_set diagnostic (ADR 0016): all is hash-neutral, position and none are not.
    "carrasco2021+position": ("train-supervised", {**_CD21, "feature_set": "position"}),
    "carrasco2021+feature-set-none": ("train-supervised", {**_CD21, "feature_set": "none"}),
    # Plan row B8 (ADR 0022): backbone_weights none is hash-neutral, zoobot is not.
    "carrasco2021+convnext-pico": (
        "train-supervised",
        {**_CD21, "paper_arch": "convnext_pico", "image_size": 63},
    ),
    "carrasco2021+convnext-pico-zoobot": (
        "train-supervised",
        {**_CD21, "paper_arch": "convnext_pico", "image_size": 63, "backbone_weights": "zoobot"},
    ),
    # Plan row B9: the hybrids' shape flags are hash-neutral at their defaults only.
    "carrasco2021+hybrid": ("train-supervised", {**_CD21, "paper_arch": "hybrid"}),
    "carrasco2021+hybrid-p4-wide-cls": (
        "train-supervised",
        {
            **_CD21,
            "paper_arch": "hybrid_p4",
            "group_fields": 16,
            "hybrid_dim": 128,
            "hybrid_depth": 4,
            "hybrid_heads": 8,
            "hybrid_readout": "cls",
        },
    ),
    # The legacy protocols (phase 1).
    "pretrain-legacy": ("pretrain", {}),
    "train-supervised-legacy": ("train-supervised", {}),
}

_DATA: Final = "d" * 64
_SPLITS: Final = "5" * 64


def case_settings(name: str) -> tuple[str, Settings]:
    """The command and resolved settings of :data:`CONFIG_CASES` ``name`` (as the CLI)."""
    command, values = CONFIG_CASES[name]
    return command, settings_from_mapping({**COMMAND_DEFAULTS[command], **values})


def identity_cases() -> dict[str, RunIdentity]:
    """A few identities: a normal run, the plain engine, a fold, the screen, another machine."""
    command, cd21 = case_settings("carrasco2021")
    gtx1050 = "cuda-nvidia-geforce-gtx-1050"
    return {
        "carrasco2021/seed0": run_identity(
            command, cd21, data_sha256=_DATA, machine_class=gtx1050, engine="lightning"
        ),
        "carrasco2021/seed0/plain": run_identity(
            command, cd21, data_sha256=_DATA, machine_class=gtx1050, engine="plain"
        ),
        "carrasco2021/seed3/fold2": run_identity(
            command,
            cd21.replace(seed=3),
            data_sha256=_DATA,
            machine_class=gtx1050,
            engine="lightning",
            split=2,
            splits_sha256=_SPLITS,
        ),
        "carrasco2021/seed0/screen": run_identity(
            command,
            cd21,
            data_sha256=_DATA,
            machine_class=gtx1050,
            engine="lightning",
            split=SCREEN,
            splits_sha256=_SPLITS,
        ),
        "carrasco2021/seed0/cpu-epoch2": run_identity(
            command,
            cd21,
            data_sha256=_DATA,
            machine_class="cpu-arm64",
            engine="lightning",
            numerics_epoch=2,
        ),
    }


def build_fixture() -> dict[str, object]:
    """The fixture's content, computed by the current code."""
    configs = {}
    for name in CONFIG_CASES:
        command, settings = case_settings(name)
        semantic = semantic_config(command, settings)
        configs[name] = {
            "canonical_json": canonical_json(semantic),
            "config_sha256": config_sha256_of(semantic),
        }
    run_ids = {
        name: {"identity": identity.to_plain(), "run_id": identity.run_id}
        for name, identity in identity_cases().items()
    }
    return {"configs": configs, "run_ids": run_ids}


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(build_fixture(), indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {FIXTURE}\n")
