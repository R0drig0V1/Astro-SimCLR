"""Tests of ``astro_simclr.runs.canonical``: canonical JSON, semantic config, config hash."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from _canonical_cases import CONFIG_CASES, FIXTURE, build_fixture, case_settings

from astro_simclr.config import ConfigError, Settings, load_settings, settings_to_plain
from astro_simclr.hparams import COMMAND_DEFAULTS
from astro_simclr.runs import canonical
from astro_simclr.runs.canonical import (
    CANONICAL_VERSION,
    HASH_NEUTRAL_DEFAULTS,
    IDENTITY_KEYS,
    MISSING,
    NON_SEMANTIC_KEYS,
    CanonicalError,
    canonical_json,
    config_sha256,
    config_sha256_of,
    semantic_config,
    semantic_config_from_plain,
    semantic_diff,
)
from astro_simclr.runs.identity import run_identity

FIELDS = {field.name for field in dataclasses.fields(Settings)}
CONFIGS = Path(__file__).parents[1] / "configs" / "phase2"


def _fixture() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


# --- The pinned hashes -----------------------------------------------------------------------


def test_fixture_is_what_the_code_computes() -> None:
    # A failure here means the canonical form moved: see the module docstring of
    # astro_simclr.runs.canonical ("Changing the canonical form").
    assert build_fixture() == _fixture()


def test_fixture_covers_the_presets_and_the_variants() -> None:
    configs = _fixture()["configs"]
    assert set(configs) == set(CONFIG_CASES)
    assert {"carrasco2021", "carrasco2021-no-features"} <= set(configs)
    hashes = [entry["config_sha256"] for entry in configs.values()]
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize(
    ("name", "file"),
    [
        ("carrasco2021", "carrasco2021.yaml"),
        ("carrasco2021-no-features", "carrasco2021-no-features.yaml"),
    ],
)
def test_preset_files_hash_like_the_presets(name: str, file: str) -> None:
    command, preset = case_settings(name)
    from_file = load_settings(CONFIGS / file, defaults=COMMAND_DEFAULTS[command])
    assert config_sha256(command, from_file) == config_sha256(command, preset)
    assert config_sha256(command, preset) == _fixture()["configs"][name]["config_sha256"]


# --- Canonical JSON --------------------------------------------------------------------------


def test_canonical_json_sorts_keys_without_whitespace() -> None:
    assert canonical_json({"b": 1, "a": {"d": [1, 2], "c": None}}) == (
        '{"a":{"c":null,"d":[1,2]},"b":1}'
    )


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.1, "0.1"),
        (1e-3, "0.001"),
        (1e-8, "1e-08"),
        (1e16, "1e+16"),
        (0.17910250439416514, "0.17910250439416514"),
        (-0.0, "0.0"),
        (2, "2"),
        (2.0, "2.0"),
        (True, "true"),
        ("ñ", '"\\u00f1"'),
        (PurePosixPath("a/b"), '"a/b"'),
        ((1, "x"), '[1,"x"]'),
    ],
)
def test_canonical_json_values(value: object, text: str) -> None:
    assert canonical_json(value) == text


def test_floats_read_back_exactly() -> None:
    value = 0.002495379490073152
    assert float(canonical_json(value)) == value


@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, {1: "x"}, {"a": {2}}, object(), b"x"]
)
def test_canonical_json_refuses_values_without_a_form(value: object) -> None:
    with pytest.raises(CanonicalError):
        canonical_json(value)


# --- The key classification ------------------------------------------------------------------


def test_excluded_keys_are_settings_fields_with_a_reason() -> None:
    assert set(NON_SEMANTIC_KEYS) <= FIELDS
    assert set(IDENTITY_KEYS) <= FIELDS
    assert not set(NON_SEMANTIC_KEYS) & set(IDENTITY_KEYS)
    assert all(reason.strip() for reason in {**NON_SEMANTIC_KEYS, **IDENTITY_KEYS}.values())
    assert set(NON_SEMANTIC_KEYS) == {
        "accelerator",
        "data_dir",
        "output_dir",
        "progress_bar_refresh_rate",
        "checkpoint_every_n_steps",
        "rj23_keep",
    }
    assert set(IDENTITY_KEYS) == {"seed", "pretrained"}


def test_hash_neutral_defaults_are_semantic_fields() -> None:
    assert set(HASH_NEUTRAL_DEFAULTS) <= FIELDS - set(NON_SEMANTIC_KEYS) - set(IDENTITY_KEYS)
    for key, value in HASH_NEUTRAL_DEFAULTS.items():
        assert getattr(Settings(), key) == value


def test_semantic_config_holds_every_other_field() -> None:
    semantic = semantic_config("train-supervised", Settings())
    assert semantic["canonical_version"] == CANONICAL_VERSION
    assert semantic["command"] == "train-supervised"
    settings = semantic["settings"]
    assert isinstance(settings, dict)
    # The hash-neutral fields are semantic, but left out at their neutral values: num_workers
    # 0, feature_set all (ADR 0016) and the rj23_* defaults (task 3.4b; ADR 0018).
    neutral = {
        "num_workers",
        "feature_set",
        "rj23_scenario",
        "rj23_bn_position",
        "rj23_large_values",
    }
    assert set(HASH_NEUTRAL_DEFAULTS) == neutral
    assert set(settings) == FIELDS - set(NON_SEMANTIC_KEYS) - set(IDENTITY_KEYS) - neutral
    with_workers = semantic_config("train-supervised", Settings(num_workers=2))["settings"]
    assert isinstance(with_workers, dict)
    assert with_workers["num_workers"] == 2
    position = semantic_config("train-supervised", Settings(feature_set="position"))["settings"]
    assert isinstance(position, dict)
    assert position["feature_set"] == "position"


@pytest.mark.parametrize(
    "changes",
    [
        {"accelerator": "cpu"},
        {"data_dir": "/elsewhere/dataset"},
        {"output_dir": "elsewhere/run"},
        {"progress_bar_refresh_rate": 0},
        {"checkpoint_every_n_steps": 2_000},
        {"seed": 7},
        {"pretrained": "/runs/encoder/checkpoints/best.ckpt"},
    ],
)
def test_excluded_keys_do_not_move_the_hash(changes: dict[str, object]) -> None:
    for name in ("carrasco2021", "train-supervised-legacy"):
        command, settings = case_settings(name)
        assert config_sha256(command, settings.replace(**changes)) == config_sha256(
            command, settings
        )


def _other(value: object) -> object:
    """A value of another canonical form than ``value``."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 2 + 1
    if isinstance(value, str):
        return value + "-other"
    return 1


@pytest.mark.parametrize("key", sorted(FIELDS - set(NON_SEMANTIC_KEYS) - set(IDENTITY_KEYS)))
def test_every_semantic_key_moves_the_hash(key: str) -> None:
    plain = settings_to_plain(Settings())
    changed = {**plain, key: _other(plain[key])}
    base = config_sha256_of(semantic_config_from_plain("train-supervised", plain))
    assert config_sha256_of(semantic_config_from_plain("train-supervised", changed)) != base


@pytest.mark.parametrize(
    ("workers_a", "workers_b", "same"), [(0, 0, True), (0, 2, False), (2, 3, False)]
)
def test_num_workers_is_semantic_except_at_zero(workers_a: int, workers_b: int, same: bool) -> None:
    # Per-worker seeding: the worker count decides the augmentation draws (legacy path).
    command, settings = case_settings("train-supervised-legacy")

    def run_id(workers: int) -> str:
        identity = run_identity(
            command,
            settings.replace(num_workers=workers),
            data_sha256="d" * 64,
            machine_class="cpu-arm64",
            engine="lightning",
        )
        return identity.run_id

    assert (run_id(workers_a) == run_id(workers_b)) is same


def test_num_workers_zero_keeps_the_pinned_hashes() -> None:
    # 0 is hash-neutral: the hash is the one the config would have without the field.
    command, settings = case_settings("carrasco2021")
    plain = settings_to_plain(settings)
    del plain["num_workers"]
    without = semantic_config_from_plain(command, plain, complete=False)
    pinned = _fixture()["configs"]["carrasco2021"]["config_sha256"]
    assert config_sha256(command, settings) == config_sha256_of(without) == pinned


def test_the_command_moves_the_hash() -> None:
    settings = Settings()
    assert config_sha256("linear-eval", settings) != config_sha256("fine-tune", settings)


def test_defaults_are_materialised() -> None:
    implicit = Settings()
    explicit = Settings(lr=Settings().lr, schedule="epochs", group_fields=None)
    assert config_sha256("train-supervised", implicit) == config_sha256(
        "train-supervised", explicit
    )


def test_an_int_for_a_float_field_hashes_as_the_float() -> None:
    # The YAML `lr: 1` is coerced to 1.0 by settings_from_mapping, so it is the same run.
    one = load_settings(overrides=["lr=1"])
    assert config_sha256("train-supervised", one) == config_sha256(
        "train-supervised", Settings(lr=1.0)
    )


def test_int_and_float_limits_differ() -> None:
    # limit_train_batches: 2 is two batches, 2.0 a fraction (200 %): two configs.
    batches = Settings(limit_train_batches=2)
    fraction = Settings(limit_train_batches=2.0)
    assert config_sha256("train-supervised", batches) != config_sha256("train-supervised", fraction)


def test_hash_neutral_defaults_are_left_out_at_their_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A field added after version 1 with a neutral default keeps old hashes valid.
    old = config_sha256("train-supervised", Settings())
    monkeypatch.setattr(
        canonical, "HASH_NEUTRAL_DEFAULTS", {**HASH_NEUTRAL_DEFAULTS, "rot90_augment": False}
    )
    assert config_sha256("train-supervised", Settings()) != old  # rot90_augment left out
    plain = settings_to_plain(Settings())
    del plain["rot90_augment"]
    without = semantic_config_from_plain("train-supervised", plain, complete=False)
    assert config_sha256("train-supervised", Settings()) == config_sha256_of(without)
    on = Settings(rot90_augment=True)
    on_settings = semantic_config("train-supervised", on)["settings"]
    assert isinstance(on_settings, dict)
    assert on_settings["rot90_augment"] is True


# --- The flat view and the 4.2 contract ------------------------------------------------------


def test_a_complete_flat_view_must_have_every_field() -> None:
    plain = settings_to_plain(Settings())
    del plain["lr"]
    with pytest.raises(ConfigError, match=r"missing \['lr'\]"):
        semantic_config_from_plain("train-supervised", plain)
    with pytest.raises(ConfigError, match=r"unknown \['model.lr'\]"):
        semantic_config_from_plain(
            "train-supervised", {**settings_to_plain(Settings()), "model.lr": 1}
        )


def test_an_incomplete_view_is_accepted_without_complete() -> None:
    semantic = semantic_config_from_plain(
        "pretrain", {"lr": 1.0, "output_dir": "x"}, complete=False
    )
    assert semantic["settings"] == {"lr": 1.0}


def test_a_lowered_view_proves_hash_equality() -> None:
    # What the 4.2 migration test does: lower the nested preset to the flat view, hash it.
    command, settings = case_settings("carrasco2021")
    model: dict[str, object] = {"paper_arch": "rot4"}
    rest = {key: value for key, value in settings_to_plain(settings).items() if key not in model}
    lowered: dict[str, object] = {**rest, **model}
    semantic = semantic_config_from_plain(command, lowered)
    assert semantic_diff(semantic, semantic_config(command, settings)) == {}
    assert config_sha256_of(semantic) == _fixture()["configs"]["carrasco2021"]["config_sha256"]


def test_unknown_command() -> None:
    with pytest.raises(ConfigError, match="command 'run'"):
        semantic_config("run", Settings())


# --- semantic_diff ---------------------------------------------------------------------------


def test_semantic_diff_names_the_keys_that_differ() -> None:
    a = semantic_config("train-supervised", Settings())
    b = semantic_config("pretrain", Settings(lr=0.5, limit_train_batches=2))
    c = semantic_config("pretrain", Settings(lr=0.5, limit_train_batches=2.0))
    assert semantic_diff(a, a) == {}
    assert semantic_diff(a, b) == {
        "command": ("train-supervised", "pretrain"),
        "limit_train_batches": (None, 2),
        "lr": (Settings().lr, 0.5),
    }
    assert semantic_diff(b, c) == {"limit_train_batches": (2, 2.0)}
    partial = semantic_config_from_plain("pretrain", {"lr": 0.5}, complete=False)
    assert semantic_diff(partial, b)["batch_size"] == (MISSING, Settings().batch_size)
    assert repr(MISSING) == "MISSING"


def test_rj23_fields_hash_only_when_they_differ() -> None:
    # Task 3.4b: the rj23_* fields at their defaults leave every older hash unchanged; an
    # rj23 run's scenario and arm are hashed when they are not the defaults, its primary
    # keep never (both keeps are computed).
    from astro_simclr.presets import PRESETS

    full = Settings().replace(**PRESETS["rj23-full"])
    lowres = Settings().replace(**PRESETS["rj23-lowres"])
    semantic = semantic_config("train-supervised", lowres)["settings"]
    assert isinstance(semantic, dict) and semantic["rj23_scenario"] == "lowres"
    full_semantic = semantic_config("train-supervised", full)["settings"]
    assert isinstance(full_semantic, dict) and "rj23_scenario" not in full_semantic
    arm = full.replace(rj23_bn_position="before_concat")
    assert config_sha256("train-supervised", arm) != config_sha256("train-supervised", full)
    best = full.replace(rj23_keep="best")
    assert config_sha256("train-supervised", best) == config_sha256("train-supervised", full)
