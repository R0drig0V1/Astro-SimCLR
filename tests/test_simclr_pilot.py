"""Tests of the 5.0b pilot loop and its CLI (``astro_simclr.simclr.pilot``, ``.commands``).

A CPU smoke of a few steps on a small prepared layout (4 Train and 4 Validation stamps per
class, uint8 images) with stored splits (2 + 2 per class in the screen halves): the outputs,
the schedule's end, an exact resume, the refusals, activation checkpointing (the same
numbers, a resume that turns it on, the CUDA peak-memory keys), and the four ``simclr-*``
subcommands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _protocol_support import make_layout

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.engine.randomness import RESERVED_STREAMS
from astro_simclr.protocol.splits import make_splits, write_splits
from astro_simclr.simclr import pilot as pilot_module
from astro_simclr.simclr.pilot import (
    CKPT_FORMAT,
    BatchOrder,
    PilotConfig,
    load_checkpoint,
    load_pilot_data,
    resolve_device,
    train_pilot,
)
from astro_simclr.simclr.probe import read_label_subsets

TRAIN_PER_CLASS, VAL_PER_CLASS = 4, 4


@pytest.fixture(scope="module")
def layout(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(prepared dir, splits file)."""
    root = tmp_path_factory.mktemp("pilot")
    prepared = make_layout(root / "layout", train=TRAIN_PER_CLASS, validation=VAL_PER_CLASS, test=1)
    splits = make_splits(prepared, selection_per_class=1, screen_per_class=2)
    path = root / "splits.json"
    write_splits(splits, path)
    return prepared, path


def _config(layout: tuple[Path, Path], **changes: Any) -> PilotConfig:
    prepared, splits = layout
    base = PilotConfig(
        data=str(prepared),
        splits=str(splits),
        epochs=None,
        steps=4,
        batch_size=4,
        warmup_epochs=0.3,
        device="cpu",
        log_every=1,
        checkpoint_every=2,
        diagnostics_every=2,
        keep_every_epochs=1,
        diag_samples=8,
        knn_bank_per_class=2,
        embed_batch=16,
    )
    return replace(base, **changes)


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_simclr_package_imports_no_lightning() -> None:
    code = (
        "import sys, astro_simclr.simclr.pilot, astro_simclr.simclr.commands; "
        "roots = ('lightning', 'pytorch_lightning'); "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in roots); "
        "assert not bad, bad"
    )
    # A fresh interpreter: this process may have imported Lightning already.
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 (fixed argv)


def test_config_validation(layout: tuple[Path, Path]) -> None:
    with pytest.raises(ConfigError, match="exactly one"):
        _config(layout, epochs=3)
    with pytest.raises(ConfigError, match="exactly one"):
        _config(layout, steps=None)
    with pytest.raises(ConfigError, match="batch_size >= 2"):
        _config(layout, batch_size=1)
    with pytest.raises(ConfigError, match="temperature"):
        _config(layout, temperature=0.0)
    with pytest.raises(ConfigError, match="warmup_epochs"):
        _config(layout, warmup_epochs=-1.0)
    with pytest.raises(ConfigError, match="tier"):
        _config(layout, tier="disk")


def test_batch_order_is_a_permutation_per_pass() -> None:
    order = BatchOrder(n=10, batch_size=3, seed=4)
    assert order.steps_per_pass == 3
    first = np.concatenate([order.batch(s) for s in range(3)])
    assert first.size == 9 and np.unique(first).size == 9
    expected = np.random.default_rng([4, 0, RESERVED_STREAMS["simclr_batch_order"]]).permutation(10)
    np.testing.assert_array_equal(first, expected[:9])
    second = np.concatenate([order.batch(s) for s in range(3, 6)])
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(order.batch(4), order.batch(4))


def test_load_pilot_data_excludes_the_screen_exclusions(layout: tuple[Path, Path]) -> None:
    data = load_pilot_data(_config(layout))
    assert data.train.shape == (5 * TRAIN_PER_CLASS, 3, 63, 63)
    assert data.train.dtype == torch.float32 and float(data.train.max()) <= 1.0
    assert data.validation.shape[0] == 5 * VAL_PER_CLASS
    assert data.screen is not None and data.record["input"].startswith("images/255")
    assert data.record["n_pretrain"] == data.rows.size
    assert set(data.record["excluded_train"]).isdisjoint(data.rows.tolist())
    no_splits = load_pilot_data(_config(layout, splits=None))
    assert no_splits.screen is None and no_splits.rows.size == 5 * TRAIN_PER_CLASS
    with pytest.raises(ConfigError, match="fewer than a batch"):
        load_pilot_data(_config(layout, batch_size=500))


def test_load_pilot_data_prefers_the_float32_stamps(classifier_f32_dir: Path) -> None:
    data = load_pilot_data(PilotConfig(data=str(classifier_f32_dir), epochs=1, batch_size=2))
    assert data.record["input"] == "images_f32"


def test_resolve_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    with pytest.raises(ConfigError, match="unknown device"):
        resolve_device("tpu")


@pytest.fixture(scope="module")
def finished(layout: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A finished 4-step pilot run directory."""
    out = tmp_path_factory.mktemp("run")
    train_pilot(_config(layout), out)
    return out


def test_pilot_writes_every_output(finished: Path) -> None:
    record = json.loads((finished / "pilot.json").read_text())
    assert record["schedule"]["total_steps"] == 4
    assert record["schedule"]["base_lr"] == pytest.approx(0.3 * 4 / 256)
    assert record["noise_fields"] is None
    metrics = _lines(finished / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [1, 2, 3, 4]
    assert all(np.isfinite(float(str(m["loss"]))) for m in metrics)
    assert metrics[-1]["lr"] == 0.0  # the cosine ends exactly at lr_min at the final step
    diagnostics = _lines(finished / "diagnostics.jsonl")
    assert [d["step"] for d in diagnostics] == [2, 4]
    keys = {"effective_rank_h", "z_std_mean", "z_std_min", "alignment", "uniformity"}
    assert keys <= set(diagnostics[0])
    assert diagnostics[-1]["knn_selection_bacc"] is not None
    for name in ("last.pt", "final.pt"):
        payload = load_checkpoint(finished / "ckpt" / name)
        assert payload["format"] == CKPT_FORMAT and payload["steps_done"] == 4


def test_pilot_dumps_the_embeddings(finished: Path) -> None:
    folder = finished / "embeddings"
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["step"] == 4 and meta["view"] == "centre-57"
    train_h = np.load(folder / "train_h.npy")
    assert train_h.shape == (5 * TRAIN_PER_CLASS, 512) and train_h.dtype == np.float32
    assert np.load(folder / "validation_h.npy").shape == (5 * VAL_PER_CLASS, 512)
    assert np.load(folder / "train_labels.npy").shape == (5 * TRAIN_PER_CLASS,)


def test_pilot_resume_is_exact(layout: tuple[Path, Path], finished: Path, tmp_path: Path) -> None:
    config = _config(layout)
    stopped = train_pilot(config, tmp_path, stop_at=2)
    assert stopped.steps_done == 2 and not (tmp_path / "ckpt" / "final.pt").exists()
    resumed = train_pilot(config, tmp_path, resume=True)
    assert resumed.steps_done == 4
    whole = load_checkpoint(finished / "ckpt" / "final.pt")["model"]
    again = load_checkpoint(tmp_path / "ckpt" / "final.pt")["model"]
    assert all(torch.equal(whole[k], again[k]) for k in whole)
    assert [m["step"] for m in _lines(tmp_path / "metrics.jsonl")] == [1, 2, 3, 4]
    assert _lines(tmp_path / "metrics.jsonl") == _lines(finished / "metrics.jsonl")


def test_pilot_refuses_to_overwrite_or_resume_another_config(
    layout: tuple[Path, Path], finished: Path
) -> None:
    with pytest.raises(ConfigError, match="already holds a run"):
        train_pilot(_config(layout), finished)
    with pytest.raises(ConfigError, match="another config"):
        train_pilot(_config(layout, temperature=0.2), finished, resume=True)


def _assert_same_final(a: Path, b: Path) -> None:
    """The final checkpoints' weights, BatchNorm buffers and LARS state are equal bit for bit."""
    first, second = (
        load_checkpoint(a / "ckpt" / "final.pt"),
        load_checkpoint(b / "ckpt" / "final.pt"),
    )
    assert all(torch.equal(first["model"][k], second["model"][k]) for k in first["model"])
    states = first["optimizer"]["state"], second["optimizer"]["state"]
    for index, entry in states[0].items():
        assert all(torch.equal(v, states[1][index][k]) for k, v in entry.items())


def test_pilot_activation_checkpointing_changes_no_number(
    layout: tuple[Path, Path], finished: Path, tmp_path: Path
) -> None:
    train_pilot(_config(layout, activation_checkpointing=True), tmp_path)
    _assert_same_final(finished, tmp_path)
    for name in ("metrics.jsonl", "diagnostics.jsonl"):
        assert _lines(tmp_path / name) == _lines(finished / name)
    np.testing.assert_array_equal(
        np.load(tmp_path / "embeddings" / "train_h.npy"),
        np.load(finished / "embeddings" / "train_h.npy"),
    )
    assert json.loads((tmp_path / "pilot.json").read_text())["config"]["activation_checkpointing"]
    # Off (the default), the record is the one written before the flag existed.
    assert (
        "activation_checkpointing"
        not in json.loads((finished / "pilot.json").read_text())["config"]
    )


def test_pilot_resume_may_turn_activation_checkpointing_on(
    layout: tuple[Path, Path], finished: Path, tmp_path: Path
) -> None:
    train_pilot(_config(layout), tmp_path, stop_at=2)
    resumed = train_pilot(_config(layout, activation_checkpointing=True), tmp_path, resume=True)
    assert resumed.steps_done == 4 and resumed.model.activation_checkpointing
    _assert_same_final(finished, tmp_path)
    assert _lines(tmp_path / "metrics.jsonl") == _lines(finished / "metrics.jsonl")
    saved = load_checkpoint(tmp_path / "ckpt" / "final.pt")["config"]
    assert saved["activation_checkpointing"] is True
    # Any other change is still refused.
    with pytest.raises(ConfigError, match="another config"):
        train_pilot(
            _config(layout, activation_checkpointing=True, temperature=0.2), tmp_path, resume=True
        )


def test_memory_row_only_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    assert pilot_module._memory_row(torch.device("cpu")) == {}
    assert pilot_module._memory_row(torch.device("mps")) == {}
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _d: 7 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _d: 8 * 2**30)
    assert pilot_module._memory_row(torch.device("cuda")) == {
        "cuda_max_memory_allocated": 7 * 2**30,
        "cuda_max_memory_reserved": 8 * 2**30,
    }


def test_pilot_with_noise_on_the_host_tier(layout: tuple[Path, Path], tmp_path: Path) -> None:
    config = _config(layout, splits=None, noise_scale=1.0, tier="host", batch_size=10)
    run = train_pilot(config, tmp_path, stop_at=2)  # 2 of 4 noised steps, no final dump
    assert run.order.steps_per_pass == 2  # 20 stamps, batch 10
    # keep_every_epochs=1: a kept checkpoint at the end of the first pass.
    assert sorted(p.name for p in (tmp_path / "ckpt").glob("*.pt")) == [
        "last.pt",
        "step-0000002.pt",
    ]
    assert run.fits is not None and run.fits.shape == (5 * TRAIN_PER_CLASS, 6)
    record = json.loads((tmp_path / "pilot.json").read_text())
    assert record["noise_fields"] == "device-dependent"
    diagnostics = _lines(tmp_path / "diagnostics.jsonl")
    assert diagnostics[-1]["knn_selection_bacc"] is None  # no splits file, no monitor


def _cli(*args: str | Path | float) -> int:
    return main([str(a) for a in args])


def test_cli_pilot_embed_subsets_and_probe(
    layout: tuple[Path, Path], finished: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepared, splits = layout
    run = tmp_path / "run"
    pilot = ("simclr-pilot", prepared, "--output-dir", run, "--device", "cpu", "--batch-size", 4)
    # 1 of 2 steps: a time-boxed call (no final dump), then the refusal without --resume.
    once = ("--steps", 2, "--stop-at", 1, "--activation-checkpointing")
    assert _cli(*pilot, "--splits", splits, *once) == 0
    assert "1 steps done" in capsys.readouterr().out
    assert json.loads((run / "pilot.json").read_text())["config"]["activation_checkpointing"]
    assert _cli(*pilot, "--splits", splits, "--steps", 2) == 2

    random_init = tmp_path / "random"
    embed = ("simclr-embed", prepared, "--device", "cpu", "--embed-batch", 16, "--output-dir")
    assert _cli(*embed, random_init, "--random-init", "--seed", 3) == 0
    meta = json.loads((random_init / "meta.json").read_text())
    assert meta["source"] == {"random_init": True, "seed": 3} and meta["step"] == 0
    from_ckpt = tmp_path / "from_ckpt"
    assert _cli(*embed, from_ckpt, "--checkpoint", finished / "ckpt" / "final.pt") == 0
    np.testing.assert_allclose(
        np.load(from_ckpt / "train_h.npy"),
        np.load(finished / "embeddings" / "train_h.npy"),
        rtol=1e-5,
        atol=1e-6,
    )

    subsets = tmp_path / "subsets.json"
    make = ("make-label-subsets", prepared, "--splits", splits, "--per-class")
    assert _cli(*make, 2, "--n-subsets", 3, "--out", subsets) == 0
    assert len(read_label_subsets(subsets).subsets) == 3
    assert _cli(*make, 2, "--out", subsets) == 2  # exists
    assert _cli(*make, 99, "--out", tmp_path / "big.json") == 2  # too few stamps

    report = tmp_path / "probe.json"
    capsys.readouterr()
    probe = ("simclr-probe", finished / "embeddings", "--subsets", subsets, "--splits", splits)
    assert _cli(*probe, "--out", report, "--weight-decay", 0.01) == 0
    assert "decision-half balanced accuracy" in capsys.readouterr().out
    table = json.loads(report.read_text())
    assert table["grid"] == [0.01] and len(table["subsets"]) == 3
    assert table["embeddings"]["step"] == 4
