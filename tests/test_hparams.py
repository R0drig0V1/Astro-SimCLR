"""Tests of the subcommand defaults, the legacy hyperparameter mapping and configs/reference."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from astro_simclr.config import ConfigError, Settings, load_settings, settings_from_mapping
from astro_simclr.hparams import (
    COMMAND_DEFAULTS,
    COMMANDS,
    load_legacy_hparams,
    map_legacy_hparams,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REFERENCE = ROOT / "configs" / "reference"


def test_command_defaults_are_valid_settings() -> None:
    for command in COMMANDS:
        settings_from_mapping(COMMAND_DEFAULTS[command])
    pretrain = settings_from_mapping(COMMAND_DEFAULTS["pretrain"])
    # iter_clr.py: Trainer(max_epochs=400), EarlyStopping(loss_train_enc, 0.002, 100).
    assert (pretrain.max_epochs, pretrain.patience, pretrain.min_delta) == (400, 100, 0.002)
    assert pretrain.monitor == "loss_train_enc"
    assert (pretrain.optimizer, pretrain.batch_size, pretrain.lr) == (
        "LARS",
        450,
        1.258059613629328,
    )
    # The classifiers keep the Settings defaults: the legacy iter_ce_stamp.py protocol.
    assert COMMAND_DEFAULTS["train-supervised"] == {}


def test_classifier_file_maps_to_identical_names() -> None:
    mapped = load_legacy_hparams(RESULTS / "hparams_best_ce.yaml", "train-supervised")
    assert mapped["batch_size"] == 45
    assert mapped["optimizer"] == "Adam"  # the file's value; R1's config corrects it
    assert mapped["augmentation"] == "without_aug"  # legacy alias, passed through
    assert mapped.skipped == []
    settings_from_mapping(mapped)
    resnet = load_legacy_hparams(RESULTS / "hparams_best_ce_resnet.yaml", "linear-eval")
    assert (resnet["image_size"], resnet["augmentation"]) == (63, "rotation_v3")


def test_simclr_file_splits_between_pretrain_and_downstream() -> None:
    path = RESULTS / "hparams_best_simclr.yaml"
    pretrain = load_legacy_hparams(path, "pretrain")
    assert pretrain["batch_size"] == 150  # batch_size_encoder
    assert pretrain["lr"] == pytest.approx(12.258059613629328)  # lr_encoder
    assert pretrain["optimizer"] == "LARS"
    assert pretrain["encoder_name"] == "pstamps"
    assert "lr_classifier" in pretrain.skipped
    settings_from_mapping({**COMMAND_DEFAULTS["pretrain"], **pretrain})

    for command in ("linear-eval", "fine-tune"):
        downstream = load_legacy_hparams(path, command)
        assert downstream["batch_size"] == 60  # batch_size_classifier
        assert downstream["lr"] == pytest.approx(0.001956409593991059)
        assert downstream["optimizer"] == "AdamW"
        assert downstream["drop_rate"] == 0.2
        # The pretraining augmentation must not become the classifier's.
        assert "augmentation" not in downstream
        assert {"augmentation", "lr_encoder", "encoder_name"} <= set(downstream.skipped)


@pytest.mark.parametrize(
    ("command", "data", "match"),
    [
        ("train-supervised", {"lr": 0.1, "lrr": 1}, "unknown key.*lrr.*valid keys"),
        ("pretrain", {"lr": 0.1}, "classifier hyperparameter file; pretrain takes"),
        ("train-supervised", {"lr_encoder": 0.1}, "SimCLR hyperparameter file"),
        ("pretrain", {"lr_encoder": 1.0, "method": "supcon"}, "method='supcon' is not supported"),
        ("train-supervised", {"balanced_batch": True}, "balanced_batch=True is not supported"),
        ("frobnicate", {}, "unknown training command"),
    ],
)
def test_mapping_errors(command: str, data: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        map_legacy_hparams(command, data)


def test_checked_keys_with_supported_values_are_dropped() -> None:
    mapped = map_legacy_hparams("train-supervised", {"balanced_batch": False, "lr": 0.1})
    assert dict(mapped) == {"lr": 0.1}


# --- configs/reference/ -------------------------------------------------------------------

_REFERENCE_COMMANDS = {
    "R1.yaml": "train-supervised",
    "R2.yaml": "train-supervised",
    "R3-pretrain.yaml": "pretrain",
    "R3-linear-eval.yaml": "linear-eval",
    "R4.yaml": "train-supervised",
}


def _reference(name: str) -> Settings:
    return load_settings(REFERENCE / name, defaults=COMMAND_DEFAULTS[_REFERENCE_COMMANDS[name]])


def test_every_reference_config_is_listed() -> None:
    assert sorted(path.name for path in REFERENCE.glob("*.yaml")) == sorted(_REFERENCE_COMMANDS)


@pytest.mark.parametrize("name", sorted(_REFERENCE_COMMANDS))
def test_reference_configs_fail_fast_without_a_gpu(name: str) -> None:
    raw = yaml.safe_load((REFERENCE / name).read_text())
    # Stated in the file, not inherited from a default.
    assert raw["accelerator"] == "cuda"
    assert raw["devices"] == 1
    assert "num_workers" in raw
    assert (REFERENCE / name).read_text().startswith("# ")  # the provenance comment
    _reference(name)


def test_r1_is_the_row_10_config_with_adamw() -> None:
    r1 = _reference("R1.yaml")
    legacy = yaml.safe_load((RESULTS / "hparams_best_ce.yaml").read_text())
    for key in ("batch_size", "beta_loss", "drop_rate", "image_size", "lr", "with_features"):
        assert getattr(r1, key) == legacy[key], key
    assert r1.optimizer == "AdamW"
    assert (r1.max_epochs, r1.patience, r1.min_delta) == (220, 70, 0.001)
    assert (r1.check_finite, r1.divergence_threshold) == (True, 0.1)
    r4 = _reference("R4.yaml")
    assert r4.replace(with_features=True, dataset=r1.dataset) == r1


def test_r2_matches_hparams_best_ce_resnet() -> None:
    r2 = _reference("R2.yaml")
    legacy = yaml.safe_load((RESULTS / "hparams_best_ce_resnet.yaml").read_text())
    for key, value in legacy.items():
        assert getattr(r2, key) == value, key
    assert r2.supervised_model == "resnet18"


def test_r3_uses_the_31b10c9_settings() -> None:
    pretrain = _reference("R3-pretrain.yaml")
    assert (pretrain.encoder_name, pretrain.batch_size, pretrain.optimizer) == (
        "pstamps",
        550,
        "LARS",
    )
    assert pretrain.lr == pytest.approx(0.258059613629328)
    assert (pretrain.max_epochs, pretrain.patience, pretrain.monitor) == (
        850,
        100,
        "loss_train_enc",
    )
    linear = _reference("R3-linear-eval.yaml")
    r1 = _reference("R1.yaml")
    assert (linear.lr, linear.batch_size, linear.optimizer) == (r1.lr, r1.batch_size, "AdamW")
    assert (linear.with_features, linear.augmentation, linear.patience) == (
        False,
        "without_aug",
        40,
    )


@pytest.mark.parametrize("name", ["R1", "R2", "R3", "R4"])
def test_compare_reference_expectations_match_the_configs(name: str) -> None:
    from astro_simclr.reference import REFERENCES

    files = {
        "R1": {"train-supervised": "R1.yaml"},
        "R2": {"train-supervised": "R2.yaml"},
        "R3": {"pretrain": "R3-pretrain.yaml", "linear-eval": "R3-linear-eval.yaml"},
        "R4": {"train-supervised": "R4.yaml"},
    }[name]
    expected = REFERENCES[name].expected
    assert set(expected) == set(files)
    for command, file in files.items():
        settings = _reference(file)
        for key, value in expected[command].items():
            assert getattr(settings, key) == value, (name, key)


def test_reference_datasets_are_the_received_files() -> None:
    from astro_simclr.data.synthetic import CLASSIFIER_FILES
    from astro_simclr.hparams import LABEL_FRACTION_DATASETS

    assert _reference("R1.yaml").dataset == LABEL_FRACTION_DATASETS[100]
    assert _reference("R2.yaml").dataset == LABEL_FRACTION_DATASETS[100]
    assert _reference("R4.yaml").dataset == LABEL_FRACTION_DATASETS[1]
    assert Settings().dataset == LABEL_FRACTION_DATASETS[100]
    # make-synthetic writes the same names, so the configs run on synthetic data too.
    assert {Path(name).stem for name in CLASSIFIER_FILES} == set(LABEL_FRACTION_DATASETS.values())
