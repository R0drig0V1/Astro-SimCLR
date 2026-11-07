"""Tests for astro_simclr.cli, called in-process through main(argv)."""

from pathlib import Path

import pytest
import yaml

from astro_simclr.cli import main
from astro_simclr.config import Settings, settings_from_mapping


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "show-config" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("astro-simclr ")


def test_show_config_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("seed: 4\n")
    assert main(["show-config", "--config", str(path), "--set", "devices=2"]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert settings_from_mapping(printed) == Settings(seed=4, devices=2)


def test_invalid_setting_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show-config", "--set", "sed=1"]) == 2
    assert capsys.readouterr().err.startswith("astro-simclr: error: unknown setting")


def test_unknown_command_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2
