"""The ``carrasco2021`` preset (task 2.2c): CLI wiring, the float path, leakage, Adam, gate.

The acceptance smoke run is :func:`test_cli_smoke_run_and_cd21_gate`: ``train-supervised
--preset carrasco2021`` on synthetic float data on the CPU, then ``compare-reference --ref
CD21`` on its output. The schedule is shortened with ``--set`` (the preset's other values
are kept), which the run records as deviations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _training_support import LEAKED_TRAIN

from astro_simclr.cli import _TRAINING_PRESETS, main
from astro_simclr.config import ConfigError, Settings, read_settings_file, settings_from_mapping
from astro_simclr.data import DataError, load_prepared
from astro_simclr.data.float_input import crop_split
from astro_simclr.experiments import run_command
from astro_simclr.presets import (
    CARRASCO2021_NO_FEATURES,
    CARRASCO2021_PRESET,
    DIAGNOSTICS,
    PRESETS,
    preset_for,
    preset_report,
)
from astro_simclr.reference import REFERENCES, compare_reference
from astro_simclr.training.device_data import (
    TensorSplit,
    float_crops,
    interim_uint8_crops,
    without_train_indices,
)
from astro_simclr.training.modules import PaperClassifierModule

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "phase2"

#: A short schedule for the synthetic data (2-step epochs at batch 4).
SHORT = {
    "batch_size": 4,
    "max_steps": 6,
    "val_every_n_steps": 3,
    "horizon_increment": 3,
    "checkpoint_every_n_steps": 3,
}


def _preset_settings(
    data: Path, out: Path, name: str = CARRASCO2021_PRESET, **changes: Any
) -> Settings:
    return settings_from_mapping(
        {
            **PRESETS[name],
            **SHORT,
            "accelerator": "cpu",
            "deterministic": True,
            "data_dir": data.parent,
            "dataset": data.name,
            "output_dir": out,
            **changes,
        }
    )


# --- The preset values and the config files -------------------------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_config_files_hold_the_preset(name: str) -> None:
    on_disk = read_settings_file(CONFIGS / f"{name}.yaml")
    assert settings_from_mapping(on_disk) == settings_from_mapping(PRESETS[name])
    assert sorted(on_disk) == sorted(PRESETS[name])


def test_preset_values() -> None:
    preset = PRESETS[CARRASCO2021_PRESET]
    assert set(_TRAINING_PRESETS) == set(PRESETS)
    settings = settings_from_mapping(preset)
    assert settings.schedule == "carrasco2021"
    assert (settings.optimizer, settings.adam_betas, settings.lr) == ("Adam", "code", 1e-3)
    assert (settings.beta_loss, settings.batch_size, settings.eval_batch_size) == (0.5, 64, 64)
    assert (settings.max_steps, settings.val_every_n_steps, settings.horizon_increment) == (
        30_000,
        10,
        10_000,
    )
    assert not settings.early_stopping and not settings.dropout_active
    assert settings.paper_input == "float32" and settings.nan_approximation
    assert settings.exclude_leaked_train and settings.with_features
    assert PRESETS[CARRASCO2021_NO_FEATURES] == {**preset, "with_features": False}
    # The CD21 gate expects the preset (with_features splits the set instead).
    expected = REFERENCES["CD21"].expected["train-supervised"]
    assert expected == {k: v for k, v in preset.items() if k != "with_features"}


def test_settings_validate_the_new_fields() -> None:
    with pytest.raises(ConfigError, match="paper_input 'float64'"):
        Settings(paper_input="float64")
    with pytest.raises(ConfigError, match="adam_betas 'tf'"):
        Settings(adam_betas="tf")


def test_preset_report_records_deviations_and_diagnostics(tmp_path: Path) -> None:
    assert preset_for(Settings()) is None
    assert preset_report(Settings()) is None
    faithful = settings_from_mapping({**PRESETS[CARRASCO2021_PRESET], "seed": 3})
    assert preset_report(faithful) == {
        "name": CARRASCO2021_PRESET,
        "faithful": True,
        "deviations": {},
        "diagnostics": [],
    }
    without = settings_from_mapping(PRESETS[CARRASCO2021_NO_FEATURES])
    assert preset_for(without) == CARRASCO2021_NO_FEATURES
    without_report = preset_report(without)
    assert without_report is not None and without_report["faithful"] is True
    diagnostic = faithful.replace(**DIAGNOSTICS)
    report = preset_report(diagnostic)
    assert report is not None
    assert report["faithful"] is False
    assert report["diagnostics"] == list(DIAGNOSTICS)
    assert report["deviations"]["adam_betas"] == {"value": "paper", "preset": "code"}


# --- Adam ------------------------------------------------------------------------------------


def _optimizer(**kwargs: Any) -> torch.optim.Optimizer:
    module = PaperClassifierModule(beta_loss=0.5, lr=1e-3, optimizer="Adam", **kwargs)
    optimizer = module.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    return optimizer


def test_paper_module_uses_the_code_betas() -> None:
    (group,) = _optimizer().param_groups
    assert group["betas"] == (0.9, 0.999)
    assert (group["eps"], group["weight_decay"], group["lr"]) == (1e-8, 0.0, 1e-3)
    (paper,) = _optimizer(adam_betas="paper").param_groups
    assert paper["betas"] == (0.5, 0.9)
    with pytest.raises(ValueError, match="adam_betas"):
        _optimizer(adam_betas="tf")


# --- The float path and the leakage removal --------------------------------------------------


def test_float_crops_are_the_float_path(classifier_f32_dir: Path) -> None:
    data = load_prepared(classifier_f32_dir)
    for approximation in (True, False):
        splits = float_crops(classifier_f32_dir, nan_approximation=approximation)
        for name, split in splits.items():
            stored = data.splits[name].images_f32
            assert stored is not None
            expected = crop_split(np.asarray(stored), 21, nan_approximation=approximation)
            np.testing.assert_array_equal(split.images, expected.transpose(0, 3, 1, 2))
            assert split.images.flags.c_contiguous
            assert split.features.shape == (len(stored), 23)


def test_float_crops_need_the_float32_stamps(classifier_dir: Path) -> None:
    with pytest.raises(DataError, match="--images float32"):
        float_crops(classifier_dir)


def test_without_train_indices(classifier_dir: Path) -> None:
    splits = interim_uint8_crops(classifier_dir)
    reduced = without_train_indices(splits, [0, 4])
    train = splits["Train"]
    np.testing.assert_array_equal(reduced["Train"].labels, np.delete(train.labels, [0, 4]))
    np.testing.assert_array_equal(reduced["Train"].images, np.delete(train.images, [0, 4], axis=0))
    assert reduced["Validation"] is splits["Validation"]
    assert isinstance(reduced["Train"], TensorSplit)
    assert without_train_indices(splits, [])["Train"].labels.size == train.labels.size
    with pytest.raises(ValueError, match="range"):
        without_train_indices(splits, [len(train.labels)])


# --- The acceptance smoke run: CLI, metrics, gate --------------------------------------------


def _cli(data: Path, out: Path, preset: str, *extra: str) -> int:
    overrides = [f"{key}={value}" for key, value in SHORT.items()]
    return main(
        [
            "train-supervised",
            "--preset",
            preset,
            "--data-dir",
            str(data.parent),
            "--output-dir",
            str(out),
            "--accelerator",
            "cpu",
            "--seed",
            "0",
            "--set",
            *overrides,
            *extra,
        ]
    )


def test_cli_smoke_run_and_cd21_gate(
    classifier_f32_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "carrasco2021"
    assert _cli(classifier_f32_dir, out, CARRASCO2021_PRESET) == 0
    assert "seconds_per_step" in capsys.readouterr().out
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    config = metrics["config"]
    assert config["supervised_model"] == "paper_stamp_classifier"
    assert (config["adam_betas"], config["paper_input"], config["batch_size"]) == (
        "code",
        "float32",
        4,
    )
    # The acceptance keys: best_step, machine and per-class recall.
    assert metrics["best_step"] % 3 == 0
    assert metrics["machine"]["device"] == "cpu"
    for stage in ("val", "test"):
        recall = metrics["evaluation"][stage]["recall_per_class"]
        assert list(recall) == metrics["labels"]
        assert (out / metrics["predictions"][stage]["file"]).is_file()
    # The leaked stamp left Train only; Validation and Test keep every stamp.
    assert metrics["paper_data"]["excluded_train"] == [LEAKED_TRAIN]
    assert metrics["paper_data"]["n_train"] == 9
    assert metrics["paper_data"]["n_tiled"] == 10
    assert metrics["predictions"]["val"]["n"] == metrics["predictions"]["test"]["n"] == 10
    assert metrics["paper_data"]["nan_approximation"] is True
    # The shortened schedule is recorded as deviations from the preset.
    preset = metrics["preset"]
    assert preset["name"] == CARRASCO2021_PRESET and preset["faithful"] is False
    assert set(preset["deviations"]) == {
        "batch_size",
        "max_steps",
        "val_every_n_steps",
        "horizon_increment",
    }
    assert preset["diagnostics"] == []
    timing = metrics["timing"]
    assert timing["steps"] == metrics["last_step"]
    assert timing["seconds_per_step"] > 0

    # compare-reference accepts the output: one seed and no oracle report -> REPORT (exit 0),
    # with the tier it would be (at most "acceptable": the faithful tier needs the oracle).
    comparison = compare_reference("CD21", [out])
    text = "\n".join(comparison.lines)
    assert comparison.result == "REPORT"
    assert "M1 oracle: no report given" in text
    assert "provisional: the gate is defined on 5 seeds" in text
    assert "best_step: [" in text
    assert "differs from the CD21 config: batch_size=4" in text
    assert main(["compare-reference", str(out), "--ref", "CD21"]) == 0


def test_cli_no_features_variant_and_a_diagnostic(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "no-features"
    code = _cli(
        classifier_f32_dir, out, CARRASCO2021_NO_FEATURES, "adam_betas=paper", "dropout_active=true"
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["with_features"] is False
    assert metrics["preset"]["name"] == CARRASCO2021_NO_FEATURES
    assert metrics["preset"]["diagnostics"] == ["dropout_active", "adam_betas"]
    checkpoint = torch.load(out / "checkpoints" / "best.ckpt", weights_only=True)
    assert checkpoint["hyper_parameters"]["adam_betas"] == "paper"
    assert not any(key.startswith("model.bn_features") for key in checkpoint["state_dict"])
    lines = compare_reference("CD21", [out]).lines
    assert any("advisory" in line for line in lines)


def test_uint8_diagnostic_without_leakage_removal(classifier_dir: Path, tmp_path: Path) -> None:
    settings = _preset_settings(
        classifier_dir,
        tmp_path,
        paper_input="uint8",
        exclude_leaked_train=False,
        max_steps=3,
        horizon_increment=0,
    )
    metrics = run_command("train-supervised", settings, progress_bar=False)
    assert metrics["paper_data"] == {
        "input": "uint8",
        "crop": 21,
        "nan_approximation": None,
        "excluded_train": [],
        "n_train": 10,
        "n_tiled": 10,
    }
    assert metrics["preset"]["diagnostics"] == ["paper_input"]


def test_float_path_on_uint8_only_data_is_a_data_error(
    classifier_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cli(classifier_dir, tmp_path / "run", CARRASCO2021_PRESET) == 1
    assert "--images float32" in capsys.readouterr().err
