"""Tests of the ``rj23`` schedule's plain path (``tasks.rj23``, task 3.4b) beyond the
golden regression set (``test_engine_regression.py``, which pins its runs to what the
Lightning runner computed) and the resume tests (``test_engine_resume.py``):
what ``run_plain`` refuses, the sources, the stream check, the outputs of a plain run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from _rj23_support import DATASET, rj23_layout

from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.data.schema import DataError
from astro_simclr.engine import load_best, load_checkpoint
from astro_simclr.presets import PRESETS
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.rj23_regime import ClassUniformStream
from astro_simclr.runs.manifest import read_manifest
from astro_simclr.tasks.plain import run_plain
from astro_simclr.tasks.rj23 import (
    RUN_IDENTITY_STATE,
    DevicePart,
    RJ23EvalSource,
    RJ23Request,
    check_stream_state,
    rj23_inputs,
    run_rj23_plain,
    schedule_record,
)


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return rj23_layout(tmp_path_factory.mktemp("tasks-rj23"))


def _settings(data_dir: Path, out: Path, **changes: object) -> Settings:
    values: dict[str, object] = {
        **PRESETS["rj23-lowres"],
        "dataset": DATASET,
        "data_dir": data_dir,
        "output_dir": out,
        "accelerator": "cpu",
        "val_every_n_steps": 2,
        "max_steps": 6,
        "patience": 50,
        "batch_size": 8,
        "eval_batch_size": 8,
        "checkpoint_every_n_steps": 2,
        **changes,
    }
    return settings_from_mapping(values)


def test_a_plain_run_writes_its_outputs(layout: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, splits = layout
    out = tmp_path / "run"
    metrics = run_plain("train-supervised", _settings(data_dir, out), rj23=RJ23Request(splits))
    assert metrics["schedule"]["stopped_by"] == "cap" and metrics["schedule"]["capped"]
    assert [step for step, _ in metrics["schedule"]["history"]] == [0, 2, 4, 6]
    assert metrics["checkpoints"] == {
        "last": "ckpt/last.pt",
        "stop": "ckpt/last.pt",
        "best": "ckpt/best.pt",
    }
    assert load_checkpoint(out / "ckpt" / "last.pt")["steps_done"] == 6
    assert load_best(out / "ckpt" / "best.pt")["steps_done"] == metrics["best_step"]
    names = sorted(p.name for p in out.glob("predictions_*.npz"))
    assert names == [
        "predictions_test.npz",
        "predictions_test_best.npz",
        "predictions_val.npz",
        "predictions_val_best.npz",
    ]
    scalars = [json.loads(line) for line in (out / "scalars.jsonl").read_text().splitlines()]
    assert [r["step"] for r in scalars] == [0, 2, 4, 6]
    assert all("macro_f1_val" in r for r in scalars)
    manifest = read_manifest(out / "manifest.json")
    assert (manifest.engine, manifest.status) == ("plain", "finished")
    assert json.loads((out / "metrics.json").read_text()) == metrics


def test_run_plain_refuses_a_bad_rj23_request(layout: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, splits = layout
    settings = _settings(data_dir, tmp_path / "x")
    request = RJ23Request(splits)
    with pytest.raises(ConfigError, match="takes its splits in the RJ23 request"):
        run_plain("train-supervised", settings, rj23=request, split=SplitRequest(splits, None))
    with pytest.raises(ConfigError, match="trains train-supervised with supervised_model=rj23"):
        run_plain("pretrain", settings, rj23=request)
    with pytest.raises(ConfigError, match="an RJ23 request needs schedule=rj23"):
        run_plain("train-supervised", Settings(), rj23=request)
    with pytest.raises(ConfigError, match="run_rj23_plain needs schedule=rj23"):
        run_rj23_plain(Settings(), request)
    with pytest.raises(ConfigError, match="an RJ23 run needs schedule=rj23"):
        rj23_inputs(Settings(), request)
    with pytest.raises(ConfigError, match="one device"):
        run_plain("train-supervised", settings.replace(devices=2), rj23=request)


def test_a_plain_resume_of_another_run_is_refused(
    layout: tuple[Path, Path], tmp_path: Path
) -> None:
    data_dir, splits = layout
    out = tmp_path / "run"
    run_plain("train-supervised", _settings(data_dir, out), rj23=RJ23Request(splits))
    with pytest.raises(ConfigError, match="resume the same run"):
        run_plain(
            "train-supervised",
            _settings(data_dir, out, seed=1),
            rj23=RJ23Request(splits),
            resume="auto",
        )


def test_sources_and_the_stream_check(layout: tuple[Path, Path]) -> None:
    data_dir, splits = layout
    inputs = rj23_inputs(_settings(data_dir, data_dir / "unused"), RJ23Request(splits))
    part = DevicePart.of(inputs.parts["Validation"], "cpu")
    source = RJ23EvalSource(part, 3)
    batches = list(source)
    assert len(source) == len(batches) == -(-len(part.part) // 3)
    assert torch.equal(torch.cat([b[2] for b in batches]), part.labels)
    with pytest.raises(ValueError, match="positive"):
        RJ23EvalSource(part, 0)

    stream = ClassUniformStream(inputs.parts["Train"].labels, 8, 0, n_classes=5)
    state = {**stream.state(), RUN_IDENTITY_STATE: dict(inputs.identity)}
    check_stream_state(state, stream, inputs.identity)
    with pytest.raises(ConfigError, match="from the RJ23 run"):
        check_stream_state({**state, RUN_IDENTITY_STATE: {}}, stream, inputs.identity)
    with pytest.raises(DataError, match="trained on the stream"):
        check_stream_state({**state, "seed": 3}, stream, inputs.identity)


def test_schedule_record_needs_a_stopped_history() -> None:
    from astro_simclr.rj23_regime import RJ23Schedule

    with pytest.raises(ConfigError, match="has not stopped"):
        schedule_record(RJ23Schedule(eval_every=2, patience=5, max_steps=10), "Validation", [0.1])


def test_the_plain_rj23_path_imports_no_lightning() -> None:
    code = (
        "import sys, astro_simclr.tasks.plain, astro_simclr.tasks.rj23; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('lightning', "
        "'pytorch_lightning')); assert not bad, bad"
    )
    # A fresh interpreter: this process has imported Lightning already.
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 (fixed argv)
