"""Tests of the plain engine's tasks and runner (task 4.6): ``astro_simclr.tasks``.

What the differential test (``test_engine_differential.py``) does not cover: the refusals
of :func:`run_plain`, the accelerator and resume resolution, ``astro-simclr run`` with
``runtime.engine: plain``, that the plain path (runner, CLI, preflight's CD21 run) never
imports Lightning, the engine-free throughput timer and the preflight's CD21 dry run.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from astro_simclr.cli import main
from astro_simclr.config import ConfigError, Settings
from astro_simclr.data.schema import DataError
from astro_simclr.preflight import ThroughputTimer, estimate, lightning_timer, run_preflight
from astro_simclr.preflight import Timed as PreflightTimed
from astro_simclr.tasks.paper import check_paper_task, paper_augment
from astro_simclr.tasks.plain import resolve_accelerator, resolve_plain_resume, run_plain

ROOT = Path(__file__).resolve().parents[1]

_CD21: dict[str, Any] = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "batch_size": 4,
    "max_steps": 4,
    "val_every_n_steps": 2,
    "horizon_increment": 0,
    "checkpoint_every_n_steps": 2,
}


def _settings(data: Path, out: Path, **changes: Any) -> Settings:
    base = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=data.parent,
        dataset=data.name,
        output_dir=out,
        **_CD21,
    )
    return base.replace(**changes)


def _no_lightning(code: str) -> None:
    """Run ``code`` in a fresh interpreter, then fail if it imported Lightning."""
    check = (
        f"{code}\nimport sys\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('lightning', "
        "'pytorch_lightning'))\nassert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", check], check=True, cwd=ROOT)  # noqa: S603


# --- what the plain runner trains, and what it refuses ---------------------------------


def test_run_plain_writes_a_plain_run(classifier_f32_dir: Path, tmp_path: Path) -> None:
    metrics = run_plain("train-supervised", _settings(classifier_f32_dir, tmp_path))
    assert metrics == json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["machine"]["engine"] == "plain"
    assert metrics["checkpoints"] == {"best": "ckpt/best.pt", "last": "ckpt/last.pt"}
    assert metrics["last_step"] == 4 and metrics["best_step"] in (2, 4)
    assert metrics["accelerator"] == "cpu" and "lightning" not in metrics["versions"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["engine"], manifest["status"]) == ("plain", "finished")
    for name in ("scalars.jsonl", "ckpt/last.pt", "ckpt/best.pt", "predictions_test.npz"):
        assert (tmp_path / name).is_file(), name


@pytest.mark.parametrize(
    ("command", "changes", "match"),
    [
        ("pretrain", {"schedule": "epochs", "supervised_model": "stamp_classifier"}, "plain"),
        (
            "train-supervised",
            {"schedule": "epochs", "supervised_model": "stamp_classifier", "image_size": 27},
            "plain engine trains",
        ),
        ("train-supervised", {"devices": 2}, "one device"),
        ("train-supervised", {"limit_val_batches": 2}, "dry-run limits"),
        ("train-supervised", {"image_size": 63}, "image_size=21"),
    ],
)
def test_run_plain_refusals(
    command: str, changes: dict[str, Any], match: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=match):
        run_plain(command, _settings(classifier_f32_dir, tmp_path, **changes))
    assert not (tmp_path / "manifest.json").exists()  # refused before it started


def test_run_plain_refuses_bad_resumes(classifier_f32_dir: Path, tmp_path: Path) -> None:
    settings = _settings(classifier_f32_dir, tmp_path)
    with pytest.raises(ConfigError, match="no such checkpoint"):
        run_plain("train-supervised", settings, resume=str(tmp_path / "missing.pt"))
    lightning = tmp_path / "old.ckpt"
    torch.save({"pytorch-lightning_version": "2.6.6", "state_dict": {}}, lightning)
    with pytest.raises(ConfigError, match="Lightning checkpoint"):
        run_plain("train-supervised", settings, resume=str(lightning))
    # A refused resume after the manifest was written is a failed attempt.
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"
    run_plain("train-supervised", settings)
    with pytest.raises(ConfigError, match="run_id"):  # another config: another run
        run_plain("train-supervised", settings.replace(lr=0.5), resume="auto")


def test_run_plain_needs_float32_stamps(classifier_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(DataError, match="no float32 stamps"):
        run_plain("train-supervised", _settings(classifier_dir, tmp_path))


def test_paper_task_checks(classifier_f32_dir: Path, tmp_path: Path) -> None:
    settings = _settings(classifier_f32_dir, tmp_path)
    check_paper_task(settings)
    assert not paper_augment(settings).active
    with pytest.raises(ValueError, match="context"):
        paper_augment(settings.replace(context="avgpool63", flux_rescale=0.5))


def test_accelerator_and_resume_resolution(tmp_path: Path) -> None:
    assert resolve_accelerator("cpu") == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "mps"
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        expected = "cpu"
    assert resolve_accelerator("auto") == expected
    if expected != "cpu":
        assert resolve_accelerator("gpu") == expected
    else:
        with pytest.raises(ConfigError, match="names no device"):
            resolve_accelerator("gpu")
    assert resolve_plain_resume(None, tmp_path) is None
    assert resolve_plain_resume("auto", tmp_path) is None
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "last.pt").write_bytes(b"x")
    assert resolve_plain_resume("auto", tmp_path) == tmp_path / "ckpt" / "last.pt"
    assert resolve_plain_resume(str(tmp_path / "ckpt" / "last.pt"), tmp_path) is not None


# --- the CLI, and the engine-free imports ----------------------------------------------


def test_cli_run_with_the_plain_engine(classifier_f32_dir: Path, tmp_path: Path) -> None:
    config = tmp_path / "c.yaml"
    out = tmp_path / "out"
    config.write_text(
        yaml.safe_dump(
            {
                "preset": "carrasco2021",
                "run": {"output_dir": str(out)},
                "runtime": {"engine": "plain", "accelerator": "cpu"},
                "data": {
                    "dir": str(classifier_f32_dir.parent),
                    "dataset": classifier_f32_dir.name,
                    "batch_size": 4,
                },
            }
        ),
        encoding="utf-8",
    )
    flat = ["max_steps=2", "val_every_n_steps=2", "checkpoint_every_n_steps=2"]
    assert main(["run", str(config), "--set", "horizon_increment=0", *flat]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine"] == "plain" and manifest["identity"]["engine"] == "plain"
    assert (out / "ckpt" / "best.pt").is_file()


def test_the_plain_path_imports_no_lightning() -> None:
    _no_lightning(
        "import astro_simclr.tasks.plain, astro_simclr.preflight, astro_simclr.cli, "
        "astro_simclr.training, astro_simclr.training.paper_augment, "
        "astro_simclr.training.tensor_splits"
    )


def test_training_names_load_lazily() -> None:
    import astro_simclr.training as training

    assert training.apply_fp32_precision.__module__ == "astro_simclr.training.runs"
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = training.missing


# --- the preflight: the engine-free timer and the CD21 dry run --------------------------


def test_throughput_timer_and_its_lightning_adapter() -> None:
    timer = ThroughputTimer()
    device = torch.device("cpu")
    timer.tick(device)
    timer.tick(device)
    assert len(timer.intervals) == 1
    timer.reset()
    timer.tick(device)
    assert len(timer.intervals) == 1  # the step after a reset is not timed
    assert timer.steps_per_second is not None
    adapter = lightning_timer(timer)
    adapter.on_train_epoch_start(None, None)
    module = SimpleNamespace(device=device)  # a LightningModule's device
    adapter.on_train_batch_end(None, module, None, None, 0)
    adapter.on_train_batch_end(None, module, None, None, 1)
    assert len(timer.intervals) == 2
    assert ThroughputTimer().steps_per_second is None


def test_estimate_counts_cd21_in_steps() -> None:
    cd21 = PreflightTimed("CD21", "red", 64, 220, 64, 10.0, 20, planned_steps=36_000)
    step = estimate([cd21], lambda dataset: 50_000)
    assert step.lines[0].startswith("CD21: >= 5.0 GPU-h full set, 1.0 minimal (36000 steps")


def test_preflight_cd21_runs_without_lightning(tmp_path: Path) -> None:
    code = (
        "from pathlib import Path\n"
        "from astro_simclr.preflight import run_preflight\n"
        f"ok = run_preflight(output_dir=Path({str(tmp_path)!r}), synthetic=True, "
        "accelerator='cpu', reference_dir=Path('configs/reference'), runs=['CD21'], "
        "n_per_split=40, steps=2)\n"
        "assert ok\n"
    )
    _no_lightning(code)
    chain = tmp_path / "dry-chain" / "CD21"
    assert json.loads((chain / "manifest.json").read_text())["engine"] == "plain"


def test_preflight_skips_cd21_without_float32_stamps(classifier_dir: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "td_ztf_stamp_17_06_20_red-002").symlink_to(classifier_dir, target_is_directory=True)
    stream = io.StringIO()
    passed = run_preflight(
        output_dir=tmp_path / "out",
        data_dir=data,
        accelerator="cpu",
        reference_dir=ROOT / "configs" / "reference",
        runs=["CD21"],
        stream=stream,
    )
    assert not passed
    assert "not prepared with float32 stamps" in stream.getvalue()
