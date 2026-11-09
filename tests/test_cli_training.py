"""Tests of ``train-supervised``'s settings, errors and outputs (``cli``, ``tasks.plain``).

The resolution order of the settings, every clear failure the plan asks for, the metrics
JSON details, and the legacy commands removed with Lightning (task 4.8), which exit 2
naming the git tag that keeps them. Full runs of every flag are in
``test_engine_regression.py``; resumes in ``test_engine_resume.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from astro_simclr.cli import REMOVED_COMMANDS, build_parser, main, training_settings
from astro_simclr.config import Settings
from astro_simclr.protocol.predictions import load_predictions
from astro_simclr.tasks.common import AcceleratorError, check_accelerator
from astro_simclr.tasks.results import finite as _finite
from astro_simclr.tasks.results import recalls_of as _recalls

ROOT = Path(__file__).resolve().parents[1]
#: A short carrasco2021 run: 3 steps of batch 4, one validation, no horizon extension.
_SHORT = [
    "max_steps=3",
    "val_every_n_steps=3",
    "horizon_increment=0",
    "checkpoint_every_n_steps=3",
    "batch_size=4",
]


def _args(command: str, *argv: str) -> Settings:
    args = build_parser().parse_args([command, *argv])
    return training_settings(command, args)


def _run(data: Path, out: Path, *argv: str) -> int:
    base = ["train-supervised", "--preset", "carrasco2021", "--data-dir", str(data)]
    return main([*base, "--output-dir", str(out), "--accelerator", "cpu", *argv])


@pytest.fixture(scope="module")
def run(classifier_f32_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One short paper-classifier run through the CLI, shared by the tests below."""
    out = tmp_path_factory.mktemp("runs") / "supervised"
    assert _run(classifier_f32_dir.parent, out, "--set", *_SHORT) == 0
    return out


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
    classifier_f32_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    argv = ["train-supervised", "--preset", "carrasco2021", "--data-dir"]
    argv += [str(classifier_f32_dir.parent), "--output-dir", str(tmp_path)]
    assert main([*argv, "--accelerator", accelerator]) == 2
    err = capsys.readouterr().err
    assert f"accelerator {accelerator!r} was requested, but it is not available" in err
    assert "preflight" in err
    assert not (tmp_path / "manifest.json").exists()  # failed before the run started


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
        (["--set", "supervised_model=resnet18"], 2, "frozen at the git tag lightning-final"),
        (["--set", "optimizer=RMSprop"], 2, "unknown optimizer"),
        (["--resume", "missing.pt"], 2, "--resume missing.pt: no such checkpoint"),
        (["--set", "limit_train_batches=2"], 2, "Lightning's dry-run limits"),
    ],
)
def test_supervised_errors(
    argv: list[str],
    code: int,
    message: str,
    classifier_f32_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(classifier_f32_dir.parent, tmp_path, *argv) == code
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("command", REMOVED_COMMANDS)
def test_the_legacy_commands_name_the_tag(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Task 4.8: they trained on Lightning; any arguments, even invalid ones, get the pointer.
    assert main([command, "--output-dir", str(tmp_path / "out"), "--bogus"]) == 2
    err = capsys.readouterr().err
    assert f"{command} trained on Lightning" in err and "lightning-final" in err
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):  # not a subcommand of the parser any more
        build_parser().parse_args([command])
    capsys.readouterr()


# --- Outputs -------------------------------------------------------------------------------


def test_outputs_stay_under_the_output_dir(run: Path) -> None:
    names = {path.relative_to(run).parts[0] for path in run.rglob("*")}
    assert names == {
        "ckpt",
        "manifest.json",  # task A1
        "metrics.json",
        "scalars.jsonl",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
        "predictions_val.npz",
        "predictions_test.npz",
    }
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["data"]["class_counts"]["Validation"] == [2, 2, 2, 2, 2]
    assert metrics["config"]["batch_size"] == 4
    counts = metrics["evaluation"]["val"]["confusion_counts"]
    assert sum(map(sum, counts)) == 10  # every validation stamp, counted once


def test_the_registry_reads_what_runs_write(run: Path, tmp_path: Path) -> None:
    """Task A2 on task A1's output: the run directory indexes as a manifest run."""
    from astro_simclr.registry.index import reindex
    from astro_simclr.registry.store import load_records, open_registry

    conn = open_registry(tmp_path / "registry.sqlite")
    try:
        report = reindex(conn, [run.parent], specs=tmp_path / "specs")
        assert report.skipped == []
        records = load_records(conn)
        kinds = dict(conn.execute("SELECT path, kind FROM artefacts WHERE path LIKE 'ckpt/%'"))
    finally:
        conn.close()
    [record] = records
    assert record.command == "train-supervised"
    assert not record.legacy and record.status == "finished" and record.engine == "plain"
    assert record.run_id is not None and record.numerics_epoch is not None
    assert record.metrics["val.balanced_accuracy"] >= 0
    assert kinds["ckpt/best.pt"] == "best_checkpoint"
    assert kinds["ckpt/last.pt"] == "last_checkpoint"


def test_metrics_record_the_protocol_fields(run: Path) -> None:
    """Task 2.3: machine, per-class recall, balanced accuracy and per-stamp predictions."""
    out = run
    metrics = json.loads((out / "metrics.json").read_text())
    machine = metrics["machine"]
    assert machine["device"] == "cpu" and machine["gpu"] is None
    assert machine["torch"] == torch.__version__ and machine["cuda"] == torch.version.cuda
    assert {"platform", "processor", "python", "cudnn"} <= set(machine)
    assert machine["fp32_precision"] == "ieee"  # ADR 0010: the default, recorded
    assert machine["engine"] == "plain"
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


def test_non_finite_values_become_null() -> None:
    assert _finite({"a": [1.0, float("nan")], "b": (float("inf"), "x"), "c": 2}) == {
        "a": [1.0, None],
        "b": [None, "x"],
        "c": 2,
    }
