"""Tests for astro_simclr.config: defaults, YAML loading, overrides and validation."""

from pathlib import Path

import pytest

from astro_simclr.config import (
    ConfigError,
    Settings,
    load_settings,
    parse_override,
    settings_from_mapping,
    settings_to_plain,
)


def test_defaults_are_the_legacy_classifier_protocol() -> None:
    settings = load_settings()
    assert settings == Settings()
    assert settings.accelerator == "auto"
    assert settings.devices == 1
    assert (settings.max_epochs, settings.patience, settings.min_delta) == (220, 40, 0.001)
    assert (settings.monitor, settings.eval_batch_size) == ("accuracy_val", 100)


def test_nt_xent_denominator_defaults_to_legacy() -> None:
    assert Settings().nt_xent_legacy_denominator is True
    settings = load_settings(overrides=["nt_xent_legacy_denominator=false"])
    assert settings.nt_xent_legacy_denominator is False
    with pytest.raises(ConfigError, match="true or false"):
        load_settings(overrides=["nt_xent_legacy_denominator=1"])


def test_yaml_then_overrides(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("seed: 3\ndata_dir: /data\nmax_epochs: 10\naccelerator: cpu\n")
    settings = load_settings(path, ["seed=7", "deterministic=true", "min_delta=0"])
    assert settings.seed == 7  # the override wins over the file
    assert settings.data_dir == Path("/data")
    assert settings.max_epochs == 10
    assert settings.accelerator == "cpu"
    assert settings.deterministic is True
    assert settings.min_delta == 0.0
    assert isinstance(settings.min_delta, float)


def test_empty_yaml_means_defaults(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_settings(path) == Settings()


@pytest.mark.parametrize(("raw", "expected"), [("devices=2", 2), ("devices=auto", "auto")])
def test_devices_accepts_int_or_string(raw: str, expected: object) -> None:
    assert load_settings(overrides=[raw]).devices == expected


def test_safe_load_rejects_python_tags(tmp_path: Path) -> None:
    path = tmp_path / "evil.yaml"
    path.write_text("seed: !!python/object/apply:os.system ['true']\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_settings(path)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("- 1\n- 2\n", "must be a mapping"),
        ("1: 2\n", "must be a mapping"),
        ("seed: [\n", "not valid YAML"),
    ],
)
def test_bad_files(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ConfigError, match=match):
        load_settings(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_settings(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"sed": 1}, "unknown setting.*sed.*valid settings"),
        ({"seed": "1"}, "'seed' must be an integer"),
        ({"seed": True}, "'seed' must be an integer"),
        ({"deterministic": 1}, "true or false"),
        ({"min_delta": "x"}, "a number"),
        ({"monitor": 3}, "a string"),
        ({"data_dir": ""}, "non-empty path"),
        ({"devices": 1.5}, "an integer or a string"),
        ({"accelerator": "tpu"}, "accelerator 'tpu' is not one of"),
        ({"devices": 0}, "devices must be >= 1"),
        ({"max_epochs": 0}, "max_epochs must be >= 1"),
        ({"eval_batch_size": 0}, "eval_batch_size must be >= 1"),
        ({"patience": -1}, "patience must be >= 0"),
        ({"num_workers": -1}, "num_workers must be >= 0"),
        ({"seed": -1}, "seed must be >= 0"),
        ({"min_delta": -0.1}, "min_delta must be >= 0"),
        ({"monitor": ""}, "monitor must not be empty"),
    ],
)
def test_invalid_values(data: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        settings_from_mapping(data)


def test_paths_accept_path_objects() -> None:
    assert settings_from_mapping({"output_dir": Path("o")}).output_dir == Path("o")


def test_replace_validates() -> None:
    settings = Settings().replace(seed=5)
    assert settings.seed == 5
    with pytest.raises(ConfigError):
        settings.replace(seed="5")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("seed=3", ("seed", 3)),
        (" monitor = loss_val", ("monitor", "loss_val")),
        ("data_dir=/a=b", ("data_dir", "/a=b")),
        ("deterministic=false", ("deterministic", False)),
    ],
)
def test_parse_override(text: str, expected: tuple[str, object]) -> None:
    assert parse_override(text) == expected


@pytest.mark.parametrize("text", ["seed", "=3", "seed=[", "seed=: :"])
def test_parse_override_errors(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_override(text)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"image_size": 0}, "image_size must be >= 1"),
        ({"projection_dim": 0}, "projection_dim must be >= 1"),
        ({"lr": 0}, "lr must be > 0"),
        ({"temperature": -1}, "temperature must be > 0"),
        ({"fine_tune_lr_factor": 0}, "fine_tune_lr_factor must be > 0"),
        ({"drop_rate": 1.0}, r"drop_rate must be in \[0, 1\)"),
        ({"limit_train_batches": 0}, "limit_train_batches must be > 0"),
        ({"limit_val_batches": -0.5}, "limit_val_batches must be > 0"),
        ({"limit_val_batches": True}, "an integer or a number or null"),
        ({"dataset": ""}, "dataset must not be empty"),
        ({"divergence_threshold": "x"}, "a number or null"),
        ({"pretrained": 3}, "a non-empty path or null"),
    ],
)
def test_invalid_hyperparameters(data: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        settings_from_mapping(data)


def test_optional_fields_accept_null_and_numbers() -> None:
    settings = load_settings(
        overrides=[
            "limit_train_batches=2",
            "limit_val_batches=0.5",
            "divergence_threshold=null",
            "pretrained=runs/pretrain",
        ]
    )
    assert settings.limit_train_batches == 2
    assert isinstance(settings.limit_train_batches, int)
    assert settings.limit_val_batches == 0.5
    assert settings.divergence_threshold is None
    assert settings.pretrained == Path("runs/pretrain")
    assert load_settings(overrides=["divergence_threshold=1"]).divergence_threshold == 1.0


def test_defaults_layer_sits_between_dataclass_and_file(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("patience: 5\n")
    settings = load_settings(path, ["seed=2"], defaults={"patience": 100, "max_epochs": 400})
    assert (settings.patience, settings.max_epochs, settings.seed) == (5, 400, 2)


def test_settings_to_plain_round_trips() -> None:
    settings = Settings(pretrained=Path("p"), data_dir=Path("d"), limit_val_batches=2)
    plain = settings_to_plain(settings)
    assert plain["pretrained"] == "p"
    assert settings_from_mapping(plain) == settings
