"""End-to-end smoke test of the CLI: the whole chain, in-process, on the CPU.

``make-synthetic`` -> ``prepare-data --images float32`` -> ``train-supervised --preset
carrasco2021`` (the paper classifier on the plain engine) -> ``train-supervised --resume
auto``, each through ``cli.main([...])`` on a tiny synthetic dataset in ``tmp_path``. It
runs inside ``scripts/check.sh`` with no skip marker.

Until task 4.8 the chain was ``pretrain`` -> ``linear-eval`` -> ``fine-tune`` ->
``train-supervised`` on Lightning; those commands are frozen at the git tag
``lightning-final`` (their exit 2 is tested in ``test_cli_training.py``).

The run is short (60 steps of batch 8, a validation every 10) but long enough that the
classifier learns the synthetic classes: the plan asks for a validation accuracy of at
least 0.5 with seed 0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from astro_simclr.cli import main
from astro_simclr.data.synthetic import CLASSIFIER_FILES
from astro_simclr.engine import load_best, load_checkpoint

#: Stamps per split: enough for a few batches of 8, small enough to stay fast.
_N = 40
#: A short carrasco2021 run (no horizon extension, so it ends at max_steps).
_SHORT = [
    "batch_size=8",
    "max_steps=60",
    "val_every_n_steps=10",
    "horizon_increment=0",
    "checkpoint_every_n_steps=10",
]


def _finite_numbers(value: object) -> bool:
    """True if every number in a JSON value is finite (``null`` is not)."""
    if value is None:
        return False
    if isinstance(value, bool | str):
        return True
    if isinstance(value, int | float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    return True


def _metrics(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "metrics.json").read_text())
    return data


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """make-synthetic and prepare-data (float32 stamps) through the CLI: the data directory."""
    root = tmp_path_factory.mktemp("smoke")
    raw, data = root / "raw", root / "data"
    assert main(["make-synthetic", "--out", str(raw), "--n-per-split", str(_N)]) == 0
    source = raw / CLASSIFIER_FILES[0]
    assert main(["prepare-data", str(source), "--out", str(data), "--images", "float32"]) == 0
    return data


def _train(data: Path, out: Path, *extra: str) -> int:
    base = ["train-supervised", "--preset", "carrasco2021", "--data-dir", str(data)]
    base += ["--output-dir", str(out), "--accelerator", "cpu", "--seed", "0", *extra]
    return main([*base, "--set", *_SHORT])


def test_full_chain(prepared: Path, tmp_path: Path) -> None:
    out = tmp_path / "supervised"
    assert _train(prepared, out) == 0
    assert (out / "ckpt" / "best.pt").is_file() and (out / "ckpt" / "last.pt").is_file()
    metrics = _metrics(out)
    assert metrics["seed"] == metrics["config"]["seed"] == 0
    assert metrics["machine"]["engine"] == "plain"
    assert _finite_numbers(metrics["monitor_best"]) and metrics["best_step"] is not None
    for key in ("val_acc_best", "test_acc_at_best", "val_macro_f1", "test_macro_f1"):
        assert _finite_numbers(metrics[key]), key
    assert _finite_numbers(metrics["evaluation"])
    assert len(metrics["evaluation"]["test"]["confusion_matrix"]) == 5
    assert (out / "confusion_matrix_val.png").stat().st_size > 0
    assert (out / "confusion_matrix_test.png").stat().st_size > 0
    assert metrics["val_acc_best"] >= 0.5
    assert metrics["last_step"] == 60


def test_resume_of_a_finished_run_reproduces_it(prepared: Path, tmp_path: Path) -> None:
    # The final last.pt is at the horizon: a resume trains nothing and reports the same
    # results (exact mid-run resumes are in test_engine_resume.py).
    out = tmp_path / "run"
    assert _train(prepared, out) == 0
    first = _metrics(out)
    assert load_checkpoint(out / "ckpt" / "last.pt")["steps_done"] == 60
    assert _train(prepared, out, "--resume", "auto") == 0
    again = _metrics(out)
    assert again["timing"]["steps"] == 0
    assert again["resumed_from"] == str(out / "ckpt" / "last.pt")
    for key in ("best_step", "monitor_best", "val_acc_best", "test_acc_at_best", "evaluation"):
        assert again[key] == first[key], key
    assert again["run"]["run_id"] == first["run"]["run_id"]
    assert again["run"]["attempt"] == first["run"]["attempt"] + 1
    assert load_best(out / "ckpt" / "best.pt")["steps_done"] == first["best_step"]
