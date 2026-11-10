"""The pilot with the optional views of task 5.2 on (``simclr.pilot``, ``simclr-pilot``).

On a small prepared layout (4 Train and 4 Validation stamps per class), for each optional
view: a 4-step CPU run is deterministic and resumes exactly (stopped at 2, resumed, equal
to the whole run bit for bit), its numbers differ from the run without it, ``pilot.json``
and the checkpoints record its strength (omitted at 0), and a resume that changes it is
refused. The CLI passes every flag through. With every view off the pilot is unchanged:
``tests/test_simclr_views_unchanged.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from _protocol_support import make_layout

from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.simclr.pilot import PilotConfig, load_checkpoint, train_pilot

FLAGS = {
    "scale_jitter": 0.3,
    "blur_sigma": 1.0,
    "intensity_jitter": 0.4,
    "rotation_degrees": 6.0,
}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_layout(tmp_path_factory.mktemp("views") / "layout", train=4, validation=4, test=1)


def _config(prepared: Path, **changes: Any) -> PilotConfig:
    base = PilotConfig(
        data=str(prepared),
        epochs=None,
        steps=4,
        batch_size=4,
        warmup_epochs=0.3,
        noise_scale=1.0,
        device="cpu",
        log_every=1,
        checkpoint_every=2,
        diagnostics_every=4,
        keep_every_epochs=None,
        diag_samples=8,
        knn_bank_per_class=2,
        embed_batch=16,
    )
    return replace(base, **changes)


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _final(path: Path) -> dict[str, torch.Tensor]:
    model: dict[str, torch.Tensor] = load_checkpoint(path / "ckpt" / "final.pt")["model"]
    return model


@pytest.fixture(scope="module")
def plain(prepared: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("plain")
    train_pilot(_config(prepared), out)
    return out


@pytest.mark.parametrize("flag", list(FLAGS))
def test_a_view_is_deterministic_resumes_exactly_and_is_recorded(
    flag: str, prepared: Path, plain: Path, tmp_path: Path
) -> None:
    config = _config(prepared, **{flag: FLAGS[flag]})
    whole, part = tmp_path / "whole", tmp_path / "part"
    train_pilot(config, whole)
    assert train_pilot(config, part, stop_at=2).steps_done == 2
    assert train_pilot(config, part, resume=True).steps_done == 4
    first, second = _final(whole), _final(part)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert _lines(part / "metrics.jsonl") == _lines(whole / "metrics.jsonl")
    # The view changes the numbers (the run without it is ``plain``).
    assert _lines(whole / "metrics.jsonl")[0]["loss"] != _lines(plain / "metrics.jsonl")[0]["loss"]
    record = json.loads((whole / "pilot.json").read_text())["config"]
    assert record[flag] == FLAGS[flag]
    assert not set(FLAGS) - {flag} & set(record)  # the others are omitted at 0
    assert load_checkpoint(whole / "ckpt" / "final.pt")["config"][flag] == FLAGS[flag]
    # A resume may not change the view's strength, nor turn it off.
    for value in (FLAGS[flag] / 2, 0.0):
        with pytest.raises(ConfigError, match="another config"):
            train_pilot(_config(prepared, **{flag: value}), part, resume=True)


def test_a_run_without_the_views_refuses_to_resume_with_one(prepared: Path, plain: Path) -> None:
    assert not set(FLAGS) & set(json.loads((plain / "pilot.json").read_text())["config"])
    with pytest.raises(ConfigError, match="another config"):
        train_pilot(_config(prepared, intensity_jitter=0.2), plain, resume=True)


def test_the_config_refuses_a_bad_strength(prepared: Path) -> None:
    with pytest.raises(ConfigError, match="rotation_degrees must be <="):
        _config(prepared, rotation_degrees=7.0)
    with pytest.raises(ConfigError, match="blur_sigma"):
        _config(prepared, blur_sigma=-1.0)


def test_cli_passes_every_view_flag(prepared: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli"
    code = main(
        [
            "simclr-pilot",
            str(prepared),
            "--output-dir",
            str(out),
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--scale-jitter",
            "0.2",
            "--blur-sigma",
            "0.5",
            "--intensity-jitter",
            "0.3",
            "--rotation-degrees",
            "4",
            "--diagnostics-every",
            "2",
            "--knn-bank-per-class",
            "2",
        ]
    )
    assert code == 0
    record = json.loads((out / "pilot.json").read_text())["config"]
    assert {k: record[k] for k in FLAGS} == {
        "scale_jitter": 0.2,
        "blur_sigma": 0.5,
        "intensity_jitter": 0.3,
        "rotation_degrees": 4.0,
    }
