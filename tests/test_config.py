"""Tests for astro_simclr.config: defaults, YAML loading, overrides and validation."""

from pathlib import Path

import pytest

from astro_simclr.config import (
    ConfigError,
    Settings,
    load_settings,
    parse_override,
    settings_from_mapping,
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
