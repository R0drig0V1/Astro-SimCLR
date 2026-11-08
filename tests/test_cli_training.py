"""Tests of the training subcommands' settings, errors and outputs (``experiments``, ``cli``).

The full chain and ``--resume`` are in ``test_smoke.py``; these cover the resolution order
of the settings, every clear failure the plan asks for, and the metrics JSON details.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from astro_simclr.cli import build_parser, main, training_settings
from astro_simclr.config import ConfigError, Settings
from astro_simclr.experiments import (
    AcceleratorError,
    _finite,
    _recalls,
    check_accelerator,
    load_pretrained,
    resolve_resume,
)
from astro_simclr.protocol.predictions import load_predictions

ROOT = Path(__file__).resolve().parents[1]
_DRY = ["max_epochs=1", "limit_train_batches=1", "limit_val_batches=1", "num_workers=0"]


@pytest.fixture(scope="module")
def data_dir(synthetic_prepared: dict[str, Path]) -> Path:
    """The directory that holds the four prepared synthetic datasets."""
    return next(iter(synthetic_prepared.values())).parent


def _args(command: str, *argv: str) -> Settings:
    args = build_parser().parse_args([command, *argv])
    return training_settings(command, args)


def _run(command: str, data: Path, out: Path, *argv: str) -> int:
    base = [command, "--data-dir", str(data), "--output-dir", str(out), "--accelerator", "cpu"]
    return main([*base, *argv])


@pytest.fixture(scope="module")
def runs(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One dry pretrain run and one dry supervised run, shared by the tests below."""
    root = tmp_path_factory.mktemp("runs")
    pretrain, supervised = root / "pretrain", root / "supervised"
    assert (
        _run("pretrain", data_dir, pretrain, "--set", "encoder_name=stamp1", "batch_size=4", *_DRY)
        == 0
    )
    assert _run("train-supervised", data_dir, supervised, "--set", "batch_size=5", *_DRY) == 0
    return {"pretrain": pretrain, "supervised": supervised}


# --- Settings resolution -------------------------------------------------------------------


def test_resolution_order(tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text("optimizer: AdamW\nbatch_size: 7\nseed: 3\n")
    settings = _args(
        "train-supervised",
        "--hparams",
        str(ROOT / "results" / "hparams_best_ce.yaml"),
        "--config",
        str(config),
        "--set",
        "batch_size=9",
        "lr=0.5",
        "--seed",
        "4",
        "--data-dir",
        "d",
        "--output-dir",
        "o",
        "--accelerator",
        "cpu",
    )
    assert settings.optimizer == "AdamW"  # the config corrects the legacy file's "Adam"
    assert settings.beta_loss == pytest.approx(0.016063800410738464)  # from --hparams
    assert (settings.batch_size, settings.lr) == (9, 0.5)  # --set wins over the config
    assert settings.seed == 4  # the flag wins over everything
    assert (settings.data_dir, settings.output_dir) == (Path("d"), Path("o"))
    assert settings.accelerator == "cpu"


def test_pretrain_gets_the_legacy_protocol() -> None:
    settings = _args("pretrain")
    assert (settings.monitor, settings.max_epochs, settings.patience) == (
        "loss_train_enc",
        400,
        100,
    )
    assert settings.dataset == "td_ztf_stamp_simclr_300"


def test_skipped_legacy_keys_are_reported(capsys: pytest.CaptureFixture[str]) -> None:
    settings = _args("pretrain", "--hparams", str(ROOT / "results" / "hparams_best_simclr.yaml"))
    assert settings.batch_size == 150
    assert "not used by pretrain: batch_size_classifier" in capsys.readouterr().err


def test_set_takes_several_items(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show-config", "--set", "seed=1", "devices=2", "--set", "max_epochs=3"]) == 0
    out = capsys.readouterr().out
    assert "seed: 1\n" in out
    assert "devices: 2\n" in out
    assert "max_epochs: 3\n" in out


# --- Clear failures ------------------------------------------------------------------------


@pytest.mark.parametrize("accelerator", ["cuda", "gpu", "mps"])
def test_missing_accelerator_fails_clearly(
    accelerator: str,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    code = main(
        [
            "train-supervised",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path),
            "--accelerator",
            accelerator,
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert f"accelerator {accelerator!r} was requested, but it is not available" in err
    assert "preflight" in err
    assert not (tmp_path / "checkpoints").exists()  # failed before training


def test_cpu_and_auto_always_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    check_accelerator("cpu")
    check_accelerator("auto")
    with pytest.raises(AcceleratorError, match="CUDA"):
        check_accelerator("cuda")


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        (
            ["--set", "dataset=nope"],
            1,
            "nope is not a prepared dataset; run `astro-simclr prepare-data",
        ),
        (["--set", "augmentation=astro"], 2, "astro"),
        (["--set", "supervised_model=vgg"], 2, "supervised_model 'vgg' is not"),
        (["--set", "monitor=loss_train_enc"], 2, "monitor 'loss_train_enc' is not a metric"),
        (["--set", "optimizer=RMSprop"], 2, "unknown optimizer"),
        (["--resume", "missing.ckpt"], 2, "--resume missing.ckpt: no such checkpoint"),
        (["--set", "batch_size=1000"], 1, "fewer than batch_size 1000"),
    ],
)
def test_supervised_errors(
    argv: list[str],
    code: int,
    message: str,
    data_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run("train-supervised", data_dir, tmp_path, *argv) == code
    assert message in capsys.readouterr().err


def test_pretrain_rejects_the_classifier_monitor(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _run("pretrain", data_dir, tmp_path, "--set", "monitor=accuracy_val", "batch_size=4") == 2
    )
    assert "valid: loss_train_enc" in capsys.readouterr().err


def test_downstream_errors(
    runs: dict[str, Path], data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run("linear-eval", data_dir, tmp_path) == 2
    assert "needs the pretrain checkpoint" in capsys.readouterr().err
    assert _run("linear-eval", data_dir, tmp_path, "--pretrained", str(tmp_path / "x.ckpt")) == 2
    assert "no such file" in capsys.readouterr().err
    assert _run("fine-tune", data_dir, tmp_path, "--pretrained", str(runs["supervised"])) == 2
    assert "is not a pretrain (SimCLR) checkpoint" in capsys.readouterr().err
    code = _run(
        "fine-tune",
        data_dir,
        tmp_path,
        "--pretrained",
        str(runs["pretrain"]),
        "--set",
        "image_size=63",
    )
    assert code == 2
    assert "trained on 27 px, but image_size is 63" in capsys.readouterr().err


def test_load_pretrained_accepts_a_directory_or_a_file(runs: dict[str, Path]) -> None:
    by_dir = load_pretrained(runs["pretrain"])
    by_file = load_pretrained(runs["pretrain"] / "checkpoints" / "best.ckpt")
    assert by_dir.encoder_name == by_file.encoder_name == "stamp1"


def test_resolve_resume(tmp_path: Path) -> None:
    assert resolve_resume(None, tmp_path) is None
    assert resolve_resume("auto", tmp_path) is None  # nothing to resume: a fresh start
    last = tmp_path / "checkpoints" / "last.ckpt"
    last.parent.mkdir()
    last.write_bytes(b"")
    assert resolve_resume("auto", tmp_path) == last
    assert resolve_resume(str(last), tmp_path) == last
    with pytest.raises(ConfigError):
        resolve_resume(str(tmp_path / "other.ckpt"), tmp_path)


# --- Outputs -------------------------------------------------------------------------------


def test_outputs_stay_under_the_output_dir(runs: dict[str, Path]) -> None:
    out = runs["supervised"]
    names = {path.relative_to(out).parts[0] for path in out.rglob("*")}
    assert names == {
        "checkpoints",
        "lightning_logs",
        "manifest.json",  # task A1
        "metrics.json",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
        "predictions_val.npz",
        "predictions_test.npz",
    }
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["data"]["class_counts"]["Validation"] == [2, 2, 2, 2, 2]
    assert metrics["config"]["batch_size"] == 5
    counts = metrics["evaluation"]["val"]["confusion_counts"]
    assert sum(map(sum, counts)) == 10  # every validation stamp, counted once


def test_the_registry_reads_what_runs_write(runs: dict[str, Path], tmp_path: Path) -> None:
    """Task A2 on task A1's output: the run directories index as manifest runs."""
    from astro_simclr.registry.index import reindex
    from astro_simclr.registry.store import load_records, open_registry

    conn = open_registry(tmp_path / "registry.sqlite")
    try:
        report = reindex(conn, [runs["pretrain"].parent], specs=tmp_path / "specs")
        assert report.skipped == []
        records = {record.command: record for record in load_records(conn)}
        kinds = dict(
            conn.execute("SELECT path, kind FROM artefacts WHERE path LIKE 'checkpoints/%'")
        )
    finally:
        conn.close()
    assert set(records) == {"pretrain", "train-supervised"}
    for record in records.values():
        assert not record.legacy and record.status == "finished" and record.engine == "lightning"
        assert record.run_id is not None and record.numerics_epoch is not None
    assert records["train-supervised"].metrics["val.balanced_accuracy"] >= 0
    assert kinds["checkpoints/best.ckpt"] == "best_checkpoint"
    assert kinds["checkpoints/last.ckpt"] == "last_checkpoint"


def test_metrics_record_the_protocol_fields(runs: dict[str, Path]) -> None:
    """Task 2.3: machine, per-class recall, balanced accuracy and per-stamp predictions."""
    out = runs["supervised"]
    metrics = json.loads((out / "metrics.json").read_text())
    machine = metrics["machine"]
    assert machine["device"] == "cpu" and machine["gpu"] is None
    assert machine["torch"] == torch.__version__ and machine["cuda"] == torch.version.cuda
    assert {"platform", "processor", "python", "cudnn"} <= set(machine)
    assert machine["fp32_precision"] == "ieee"  # ADR 0010: the default, recorded
    for stage, split in (("val", "Validation"), ("test", "Test")):
        evaluation = metrics["evaluation"][stage]
        counts = evaluation["confusion_counts"]
        recalls = evaluation["recall_per_class"]
        assert list(recalls) == list(metrics["labels"])
        assert recalls["SN"] == pytest.approx(counts[1][1] / sum(counts[1]))
        assert evaluation["balanced_accuracy"] == pytest.approx(sum(recalls.values()) / 5)
        entry = metrics["predictions"][stage]
        assert entry == {"file": f"predictions_{stage}.npz", "split": split, "n": 10}
        predictions = load_predictions(out / entry["file"])
        assert predictions.index.tolist() == list(range(10))
        np.testing.assert_allclose(predictions.probabilities.sum(axis=1), 1.0, rtol=1e-5)
        # The saved probabilities give the confusion counts of the metrics.
        recount = np.zeros((5, 5), dtype=int)
        np.add.at(recount, (predictions.labels, predictions.predicted), 1)
        assert recount.tolist() == counts
        assert evaluation["accuracy"] == pytest.approx(predictions.correct.mean())


def test_recalls_of_an_absent_class() -> None:
    counts = [[2, 0, 0, 0, 0], [1, 1, 0, 0, 0], [0] * 5, [0, 0, 0, 3, 1], [0, 0, 0, 0, 0]]
    assert _recalls(counts) == {"AGN": 1.0, "SN": 0.5, "VS": None, "Asteroid": 0.75, "Bogus": None}


def test_resnet_supervised_model(data_dir: Path, tmp_path: Path) -> None:
    assert (
        _run(
            "train-supervised",
            data_dir,
            tmp_path,
            "--set",
            "supervised_model=resnet18",
            "with_features=false",
            "batch_size=5",
            *_DRY,
        )
        == 0
    )
    assert json.loads((tmp_path / "metrics.json").read_text())["config"]["supervised_model"] == (
        "resnet18"
    )


def test_non_finite_values_become_null() -> None:
    assert _finite({"a": [1.0, float("nan")], "b": (float("inf"), "x"), "c": 2}) == {
        "a": [1.0, None],
        "b": [None, "x"],
        "c": 2,
    }
