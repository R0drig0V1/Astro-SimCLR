"""The pilot with every optional view off is the pilot of before them, bit for bit.

The views of plan 5.2's ablations (scale jitter, blur, intensity, rotation; ADR 0025,
amendment "stronger views and their admission test") are off by default. Off, the pilot
must draw the same numbers and train the same weights as the code of main before they
existed (``40b7470``). The reference was recorded from that code: the SHA-256 of the
training views of a fixed batch (with and without C5 noise), and of a 4-step CPU pilot run
with noise (every logged loss, the final weights, BatchNorm buffers and LARS momenta).

CPU kernels round differently across instruction sets, so the reference holds on the
platform that recorded it (``_engine_cases.platform_key``; ``darwin-arm64-default``) and
the test skips elsewhere, as the engine goldens do. ``ASTRO_SIMCLR_RECORD_GOLDEN=1``
records this platform's reference instead of checking it (only to add a platform, or with
a numerics change said in the PR).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _engine_cases import platform_key
from _protocol_support import make_layout

from astro_simclr.protocol.splits import make_splits, write_splits
from astro_simclr.simclr.pilot import PilotConfig, load_checkpoint, train_pilot
from astro_simclr.simclr.views import (
    STAMP_SIDE,
    ViewConfig,
    draw_view_params,
    make_views,
    noise_generator,
)
from astro_simclr.stamp_relation import stamp_fits

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "simclr_views_unchanged"
RECORD = os.environ.get("ASTRO_SIMCLR_RECORD_GOLDEN") == "1"


def _sha(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_sha(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _views_record() -> dict[str, str]:
    rng = np.random.default_rng(5)
    x = torch.from_numpy(rng.random((6, 3, STAMP_SIDE, STAMP_SIDE), dtype=np.float32))
    params = draw_view_params(2, 9, 6)
    plain = make_views(x, params, ViewConfig())
    fits = torch.from_numpy(stamp_fits(x.numpy()))
    noised = make_views(
        x,
        params,
        ViewConfig(noise_scale=3.0, noise_min_r2=-1.0),
        fits=fits,
        generator=noise_generator(2, 9, x.device),
    )
    return {"views": _sha(*plain), "noised_views": _sha(*noised)}


def _run_record(tmp_path: Path) -> dict[str, Any]:
    prepared = make_layout(tmp_path / "layout", train=4, validation=4, test=1)
    splits = tmp_path / "splits.json"
    write_splits(make_splits(prepared, selection_per_class=1, screen_per_class=2), splits)
    config = PilotConfig(
        data=str(prepared),
        splits=str(splits),
        epochs=None,
        steps=4,
        batch_size=4,
        warmup_epochs=0.3,
        noise_scale=3.0,
        device="cpu",
        log_every=1,
        checkpoint_every=2,
        diagnostics_every=2,
        keep_every_epochs=1,
        diag_samples=8,
        knn_bank_per_class=2,
        embed_batch=16,
    )
    out = tmp_path / "run"
    train_pilot(config, out)
    final = load_checkpoint(out / "ckpt" / "final.pt")
    momenta = {
        f"{index}.{name}": value
        for index, entry in final["optimizer"]["state"].items()
        for name, value in entry.items()
    }
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    return {
        "losses": [row["loss"] for row in metrics],
        "model": _state_sha(final["model"]),
        "optimizer": _state_sha(momenta),
        "config": final["config"],
    }


def test_the_pilot_is_unchanged_with_the_optional_views_off(tmp_path: Path) -> None:
    path = GOLDEN_DIR / f"{platform_key()}.json"
    record = {**_views_record(), **_run_record(tmp_path)}
    # The paths of the temporary layout differ from run to run; the rest is the record.
    record["config"] = {k: v for k, v in record["config"].items() if k not in ("data", "splits")}
    if RECORD:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return
    if not path.is_file():
        pytest.skip(f"no reference for platform {platform_key()} (module docstring)")
    assert record == json.loads(path.read_text())
