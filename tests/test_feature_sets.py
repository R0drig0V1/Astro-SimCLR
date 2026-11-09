"""Tests of the ``feature_set`` diagnostic of the paper model (``feature_sets``; ADR 0016).

1. **The columns.** ``position`` is the 6 sky coordinates of the 23 kept features, checked
   by name against the stored schema and ``class_augment.KEPT_FEATURE_NAMES``.
2. **``all`` is the faithful model**: ``tests/test_paper_variants.py`` checks it against the
   golden fixture ``tests/fixtures/paper_model/faithful.npz`` (``feature_set`` is one of its
   flags at their defaults); here, its hyperparameters and config hash are unchanged.
3. **``none`` is bit-identical to ``with_features: false``**: the same initial tensors and
   logits for every ``paper_arch``, and the same trained checkpoint and predictions bytes
   after a CPU run through ``run_command``.
4. **``position`` reads only the coordinates**: the features BN and fc2 shrink to 6 inputs,
   the other 17 columns do not change the logits, and it trains and reloads.
5. The settings' checks, the preset deviation and the CD21 gate's grouping.

The split runs with ``feature_set`` and the 2.5 flags are in ``test_split_runs.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from astro_simclr.config import FEATURE_SETS, ConfigError, Settings, settings_from_mapping
from astro_simclr.convnext_model import ConvNeXtStampClassifier
from astro_simclr.data.schema import DISCARDED_FEATURES, FEATURE_NAMES
from astro_simclr.experiments import paper_module, run_command
from astro_simclr.feature_sets import (
    KEPT_FEATURES,
    POSITION_FEATURES,
    feature_columns,
    feature_names,
)
from astro_simclr.hybrid_model import HybridStampClassifier
from astro_simclr.models import N_FEATURES
from astro_simclr.paper_model import PaperStampClassifier
from astro_simclr.paper_variants import (
    PAPER_ARCHS,
    GroupStampClassifier,
    PaperHead,
    build_paper_model,
    count_parameters,
    paper_input_side,
)
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS, preset_report
from astro_simclr.reference import compare_reference
from astro_simclr.runs.canonical import config_sha256, semantic_config
from astro_simclr.training.class_augment import KEPT_FEATURE_NAMES
from astro_simclr.training.modules import PaperClassifierModule

_OPTIM: dict[str, Any] = {"beta_loss": 0.5, "lr": 1e-3, "optimizer": "Adam"}


def _preset(**changes: object) -> Settings:
    return settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], **changes})


def _inputs(n: int = 6, seed: int = 0, side: int = 21) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, side, side, generator=generator, dtype=torch.float64)
    features = torch.randn(n, N_FEATURES, generator=generator, dtype=torch.float64)
    return images, features


def _assert_same_state(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for key in sa:
        assert torch.equal(sa[key], sb[key]), key


# --- 1. The columns ------------------------------------------------------------------------


def test_position_is_the_six_sky_coordinates() -> None:
    assert KEPT_FEATURES == KEPT_FEATURE_NAMES
    assert len(KEPT_FEATURES) == N_FEATURES == len(FEATURE_NAMES) - len(DISCARDED_FEATURES)
    assert set(POSITION_FEATURES) == {"ra", "dec", "gal_lat", "gal_long", "ecl_lat", "ecl_long"}
    assert feature_columns("position") == (10, 11, 16, 17, 18, 19)
    assert feature_names("position") == POSITION_FEATURES
    # The stored columns behind them (the 26 of the schema, columns 13-15 dropped).
    stored = [FEATURE_NAMES.index(name) for name in POSITION_FEATURES]
    kept = [i for i in range(len(FEATURE_NAMES)) if i not in DISCARDED_FEATURES]
    assert [kept.index(i) for i in stored] == list(feature_columns("position"))


def test_all_and_none() -> None:
    assert FEATURE_SETS == ("all", "position", "none")
    assert feature_columns("all") == tuple(range(N_FEATURES))
    assert feature_names("all") == KEPT_FEATURES
    assert feature_columns("none") == () == feature_names("none")
    # Without features nothing is read, whatever the set (except position, a contradiction).
    assert feature_columns("all", with_features=False) == ()
    assert feature_columns("none", with_features=False) == ()
    with pytest.raises(ValueError, match="'position' needs with_features=True"):
        feature_columns("position", with_features=False)
    with pytest.raises(ValueError, match="feature_set must be one of all, position, none"):
        feature_columns("colour")


# --- 2. all: the faithful module's hyperparameters and hash ---------------------------------


def test_all_keeps_the_hyperparameters_and_the_config_hash() -> None:
    assert "feature_set" not in PaperClassifierModule(**_OPTIM).hparams
    assert "feature_set" not in PaperClassifierModule(**_OPTIM, feature_set="all").hparams
    assert PaperClassifierModule(**_OPTIM, feature_set="position").hparams["feature_set"] == (
        "position"
    )
    faithful = _preset()
    settings = semantic_config("train-supervised", faithful)["settings"]
    assert isinstance(settings, dict)
    assert "feature_set" not in settings
    position = faithful.replace(feature_set="position")
    assert config_sha256("train-supervised", position) != config_sha256(
        "train-supervised", faithful
    )


# --- 3. none is bit-identical to with_features: false --------------------------------------


@pytest.mark.parametrize("arch", PAPER_ARCHS)
def test_none_is_the_without_features_model(arch: str) -> None:
    torch.manual_seed(3)
    without = build_paper_model(arch, with_features=False).double().eval()
    torch.manual_seed(3)
    none = build_paper_model(arch, feature_set="none").double().eval()
    assert type(none) is type(without)
    _assert_same_state(none, without)
    images, features = _inputs(side=paper_input_side(arch))
    with torch.no_grad():
        assert torch.equal(none(images, features), without(images, None))


def test_none_module_is_the_without_features_module() -> None:
    torch.manual_seed(5)
    without = paper_module(_preset(with_features=False))
    torch.manual_seed(5)
    none = paper_module(_preset(feature_set="none"))
    assert none.with_features is False
    _assert_same_state(none, without)


def _smoke(classifier_f32_dir: Path, out: Path, **changes: object) -> Settings:
    return _preset(
        accelerator="cpu",
        deterministic=True,
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=out,
        batch_size=4,
        max_steps=4,
        val_every_n_steps=2,
        horizon_increment=0,
        checkpoint_every_n_steps=2,
        **changes,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_none_trains_bit_identically(classifier_f32_dir: Path, tmp_path: Path) -> None:
    runs = {}
    for name, changes in (("without", {"with_features": False}), ("none", {"feature_set": "none"})):
        out = tmp_path / name
        metrics = run_command(
            "train-supervised", _smoke(classifier_f32_dir, out, **changes), progress_bar=False
        )
        runs[name] = (out, metrics)
    (without_dir, without), (none_dir, none) = runs["without"], runs["none"]
    for key in ("best_step", "val_acc_best", "test_acc_at_best", "loss_val_best"):
        assert none.get(key) == without.get(key), key
    for stage in ("val", "test"):
        file = f"predictions_{stage}.npz"
        assert _sha(none_dir / file) == _sha(without_dir / file)
    states = [
        torch.load(d / "checkpoints" / "best.ckpt", weights_only=True)["state_dict"]
        for d in (without_dir, none_dir)
    ]
    assert list(states[0]) == list(states[1])
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key
    # It is a deviation of carrasco2021, not the faithful no-features preset.
    assert none["preset"]["name"] == CARRASCO2021_PRESET
    assert none["preset"]["deviations"]["feature_set"] == {"value": "none", "preset": "all"}


# --- 4. position reads only the coordinates -------------------------------------------------


@pytest.mark.parametrize("arch", PAPER_ARCHS)
def test_position_reads_only_the_coordinates(arch: str) -> None:
    torch.manual_seed(0)
    model = build_paper_model(arch, feature_set="position").double().eval()
    grouped = isinstance(
        model, GroupStampClassifier | ConvNeXtStampClassifier | HybridStampClassifier
    )
    head = model.head if grouped else model
    assert isinstance(head, PaperStampClassifier | PaperHead)
    assert head.bn_features is not None and head.bn_features.num_features == 6
    assert head.fc2.in_features == 64 + 6
    # The index is not part of the state: the keys are the faithful model's.
    torch.manual_seed(0)
    faithful = build_paper_model(arch)
    assert list(model.state_dict()) == list(faithful.state_dict())
    images, features = _inputs(side=paper_input_side(arch))
    other = [i for i in range(N_FEATURES) if i not in feature_columns("position")]
    changed = features.clone()
    changed[:, other] = torch.randn(len(features), len(other), dtype=torch.float64)
    moved = features.clone()
    moved[:, feature_columns("position")[0]] += 3.0
    with torch.no_grad():
        logits = model(images, features)
        assert torch.equal(model(images, changed), logits)
        assert not torch.allclose(model(images, moved), logits)


def test_position_parameter_count_and_errors() -> None:
    model = PaperStampClassifier(feature_set="position")
    # 439,557 without features, plus BN gamma/beta (2 x 6) and fc2's 6 x 64 weights.
    assert count_parameters(model) == 439_557 + 12 + 6 * 64 == 439_953
    images, features = _inputs(n=2)
    with pytest.raises(ValueError, match=r"x_feat of shape \(N, 23\)"):
        model.double()(images, features[:, :6])
    with pytest.raises(ValueError, match="'position' needs with_features=True"):
        PaperStampClassifier(with_features=False, feature_set="position")
    with pytest.raises(ValueError, match="feature_set must be one of"):
        GroupStampClassifier("p4", feature_set="sky")
    with pytest.raises(ValueError, match="'position' needs with_features=True"):
        PaperClassifierModule(**_OPTIM, with_features=False, feature_set="position")


def test_position_trains_and_reloads(classifier_f32_dir: Path, tmp_path: Path) -> None:
    settings = _smoke(classifier_f32_dir, tmp_path, feature_set="position", paper_arch="p4")
    metrics = run_command("train-supervised", settings, progress_bar=False)
    assert metrics["last_step"] == 4
    assert metrics["config"]["feature_set"] == "position"
    assert metrics["preset"]["faithful"] is False
    assert metrics["preset"]["deviations"]["feature_set"] == {"value": "position", "preset": "all"}
    checkpoint = torch.load(tmp_path / "checkpoints" / "best.ckpt", weights_only=True)
    assert checkpoint["hyper_parameters"]["feature_set"] == "position"
    restored = PaperClassifierModule(**checkpoint["hyper_parameters"])
    restored.load_state_dict(checkpoint["state_dict"])
    model = restored.model
    assert isinstance(model, GroupStampClassifier)
    assert model.head.bn_features is not None
    assert model.head.bn_features.num_features == 6


# --- 5. Settings, preset and the CD21 gate --------------------------------------------------


def test_settings_check_the_feature_set() -> None:
    assert Settings().feature_set == "all"
    assert _preset(feature_set="position", aug_feature_jitter=0.1).feature_set == "position"
    with pytest.raises(ConfigError, match="feature_set 'sky' is not one of: all, position, none"):
        _preset(feature_set="sky")
    with pytest.raises(ConfigError, match="feature_set 'position' needs with_features=true"):
        _preset(feature_set="position", with_features=False)
    with pytest.raises(ConfigError, match="aug_feature_jitter needs feature_set"):
        _preset(feature_set="none", aug_feature_jitter=0.1)
    # with_features: false and none agree.
    assert _preset(feature_set="none", with_features=False).feature_set == "none"


def test_preset_deviation() -> None:
    assert PRESETS[CARRASCO2021_PRESET]["feature_set"] == "all"
    report = preset_report(_preset(feature_set="position"))
    assert report is not None
    assert report["faithful"] is False
    assert report["deviations"] == {"feature_set": {"value": "position", "preset": "all"}}


def test_feature_set_needs_the_paper_model(classifier_f32_dir: Path, tmp_path: Path) -> None:
    settings = Settings(
        accelerator="cpu",
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=tmp_path,
        feature_set="none",
    )
    with pytest.raises(ConfigError, match="feature_set apply only to"):
        run_command("train-supervised", settings, progress_bar=False)


def _gate_run(root: Path, name: str, seed: int, feature_set: str) -> Path:
    out = root / name
    out.mkdir()
    metrics = {
        "command": "train-supervised",
        "seed": seed,
        # with_features stays true: feature_set alone makes it a without-features run.
        "config": {**PRESETS[CARRASCO2021_PRESET], "feature_set": feature_set},
        "val_acc_best": 0.89,
        "test_acc_at_best": 0.885,
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    return out


def test_cd21_counts_none_as_without_features(tmp_path: Path) -> None:
    runs = [_gate_run(tmp_path, f"n{seed}", seed, "none") for seed in range(5)]
    comparison = compare_reference("CD21", runs)
    text = "\n".join(comparison.lines)
    assert comparison.result == "REPORT"
    assert "without features (advisory, not gating)" in text
    assert "feature_set='none' (expected 'all')" in text
