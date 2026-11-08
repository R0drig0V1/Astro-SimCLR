"""Tests of the data subcommands of the CLI (inspect-data, make-synthetic, prepare-data)."""

import json
from pathlib import Path

import pytest

from astro_simclr.cli import main
from astro_simclr.data.synthetic import CLASSIFIER_FILES, PRETRAINING_FILE

ALL_FILES = [*CLASSIFIER_FILES, PRETRAINING_FILE]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli")
    assert main(["make-synthetic", "--out", str(out), "--n-per-split", "10", "--seed", "5"]) == 0
    return out


@pytest.mark.parametrize("name", ALL_FILES)
def test_inspect_data_runs_on_every_synthetic_file(
    name: str, synthetic: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["inspect-data", str(synthetic / name)]) == 0
    out = capsys.readouterr().out
    assert "format: pickle" in out and "sha256: " in out
    assert "Validation: 10 stamps" in out
    assert "images    list[10] of ndarray float32 (63, 63, 3)" in out
    assert "classes: AGN 2, SN 2, VS 2, Asteroid 2, Bogus 2" in out
    if name == PRETRAINING_FILE:
        assert "kind: pretraining" in out
        assert "images  list[10] of ndarray float64 (63, 63, 3)" in out


def test_inspect_data_json(synthetic: Path, capsys: pytest.CaptureFixture[str]) -> None:
    capsys.readouterr()
    assert main(["inspect-data", "--json", str(synthetic / CLASSIFIER_FILES[1])]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["format"] == "pickle" and summary["kind"] == "classifier"
    assert len(summary["sha256"]) == 64
    train = summary["splits"]["Train"]
    assert train["n"] == 10
    assert train["keys"]["features"] == "list[10] of ndarray object (26,)"
    assert train["keys"]["labels"] == "list[10] of int64"
    assert train["class_counts"] == {"AGN": 2, "SN": 2, "VS": 2, "Asteroid": 2, "Bogus": 2}


def test_prepare_then_inspect_the_layout(
    synthetic: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = synthetic / PRETRAINING_FILE
    assert main(["prepare-data", str(source), "--out", str(tmp_path)]) == 0
    dest = tmp_path / source.stem
    assert capsys.readouterr().out == f"wrote {dest}\n"
    assert main(["inspect-data", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "format: prepared   kind: pretraining" in out
    assert f"source: {source.name}" in out and "SHA-256 match meta.json" in out
    assert "images.npy  uint8 (10, 63, 63, 3)" in out
    assert main(["inspect-data", "--json", str(dest)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["splits"]["Train"]["class_counts"] is None
    assert summary["source"]["name"] == source.name


def test_prepare_defaults_to_the_data_dir_setting(synthetic: Path, tmp_path: Path) -> None:
    source = synthetic / CLASSIFIER_FILES[2]
    args = ["prepare-data", str(source), "--set", f"data_dir={tmp_path / 'data'}"]
    assert main(args) == 0
    assert (tmp_path / "data" / source.stem / "meta.json").is_file()
    assert main(args) == 1  # exists
    assert main([*args, "--overwrite"]) == 0


def test_make_synthetic_seed_comes_from_the_settings(tmp_path: Path) -> None:
    common = ["make-synthetic", "--n-per-split", "5"]
    assert main([*common, "--out", str(tmp_path / "a"), "--set", "seed=9"]) == 0
    assert main([*common, "--out", str(tmp_path / "b"), "--seed", "9"]) == 0
    for name in ALL_FILES:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_make_synthetic_rejects_tiny_splits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["make-synthetic", "--out", str(tmp_path), "--n-per-split", "2"]) == 2
    assert "n_per_split must be >= 5" in capsys.readouterr().err


def test_data_errors_exit_1_with_one_line(
    synthetic: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    truncated = tmp_path / "cut.pkl"
    truncated.write_bytes((synthetic / CLASSIFIER_FILES[2]).read_bytes()[:-100])
    capsys.readouterr()
    for path, message in [
        (truncated, "the pickle is truncated"),
        (tmp_path / "missing.pkl", "no such file or directory"),
        (tmp_path, "a directory without meta.json"),
    ]:
        assert main(["inspect-data", str(path)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("astro-simclr: error: ") and message in err
        assert err.count("\n") == 1
    assert main(["prepare-data", str(truncated), "--out", str(tmp_path / "o")]) == 1
    assert "truncated" in capsys.readouterr().err
    assert not (tmp_path / "o" / "cut").exists()


def test_invalid_settings_are_still_usage_errors(synthetic: Path) -> None:
    assert main(["inspect-data", str(synthetic / PRETRAINING_FILE), "--set", "sed=1"]) == 2
