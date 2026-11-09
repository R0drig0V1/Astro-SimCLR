"""The plain engine's regression cases and the helpers that run and fingerprint them.

Shared by ``test_engine_regression.py`` (the golden regression set, task 4.8: what replaced
the 4.7(a) Lightning differential test) and ``test_engine_resume.py`` (the resume tests).
The schedules are those of the differential test, so that the goldens recorded before
Lightning was removed (``tests/fixtures/engine_golden/``) are the outputs that test held
bit for bit equal to Lightning's:

- the paper classifier on ``carrasco2021`` (:data:`SCHEDULE`): validation every 3 steps, an
  initial horizon of 10 and an increment of 10, ``lr`` 3e-2, so the first validation's best
  always extends the run and the extensions end within a few dozen steps;
- RJ23's classifier on ``rj23`` (:data:`RJ23`): validation every 2 steps from step 0,
  patience 2, a 40-step cap.

**Adding a flag** (CLAUDE.md, "New training flags"): add a case to :data:`FLAGS` (or
:data:`RJ23_CASES`), then record its golden (``test_engine_regression.py``'s docstring).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from astro_simclr.config import Settings, settings_from_mapping
from astro_simclr.data import prepare
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.engine import Task, Validation
from astro_simclr.presets import PRESETS
from astro_simclr.protocol.splits import make_splits, write_splits

#: The ``carrasco2021`` schedule of every paper case (module docstring).
SCHEDULE: dict[str, Any] = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "monitor": "loss_val",
    "optimizer": "Adam",
    # Large enough that Validation's loss stops improving within 22-43 steps on 10 stamps
    # (at 1e-3 it improves for up to 133): every case extends the horizon, and stays fast.
    "lr": 3e-2,
    "beta_loss": 0.5,
    "batch_size": 4,
    "max_steps": 10,
    "val_every_n_steps": 3,
    "horizon_increment": 10,
    "checkpoint_every_n_steps": 3,
}

#: Every class-aware augmentation on (C5, C6, C7, p4).
AUGMENT: dict[str, Any] = {
    "paper_arch": "p4",
    "aug_noise": 1.0,
    "aug_feature_jitter": 0.1,
    "synthetic_bogus": 0.25,
    "aug_min_r2": 0.0,
    "flux_rescale": 0.5,
}

#: Plan row D3 all on: EMA with warmup, cosine with warmup, the final weights scored (a
#: fixed horizon: no extension). The decay is small so that 10 steps move the EMA.
D3: dict[str, Any] = {
    "ema_decay": 0.9,
    "ema_warmup": True,
    "lr_schedule": "cosine",
    "lr_warmup_steps": 3,
    "checkpoint_selection": "final",
    "horizon_increment": 0,
}

#: The 2.5 and D3 flags, one case each (the faithful model first): the regression set.
FLAGS: list[Any] = [
    pytest.param({}, id="faithful"),
    pytest.param({"paper_arch": "single"}, id="single"),
    pytest.param({"paper_arch": "p4"}, id="p4"),
    pytest.param({"paper_arch": "p4m", "group_fields": 4}, id="p4m"),
    pytest.param({"paper_arch": "convnext_atto", "image_size": 63}, id="convnext_atto"),
    pytest.param({"context": "avgpool63"}, id="context"),
    pytest.param({"rot90_augment": True}, id="rot90"),
    pytest.param({"aug_noise": 1.0, "aug_feature_jitter": 0.1, "aug_min_r2": 0.0}, id="C5"),
    pytest.param({"synthetic_bogus": 0.25, "aug_min_r2": 0.0}, id="C6"),
    pytest.param({"flux_rescale": 0.5}, id="C7"),
    pytest.param({"feature_set": "position"}, id="feature_set-position"),
    pytest.param({"feature_set": "none", "rot90_augment": True}, id="feature_set-none"),
    pytest.param({"fp32_precision": "tf32"}, id="fp32-tf32"),
    pytest.param(AUGMENT, id="C5-C7-p4"),
    # Plan row B9: the conv-attention hybrids (both trunks, both readouts).
    pytest.param({"paper_arch": "hybrid", "hybrid_readout": "cls"}, id="hybrid-cls"),
    pytest.param(
        {"paper_arch": "hybrid_p4", "hybrid_dim": 32, "hybrid_heads": 2, "rot90_augment": True},
        id="hybrid_p4",
    ),
    pytest.param({"ema_decay": 0.9}, id="ema"),
    pytest.param({"checkpoint_selection": "final", "horizon_increment": 0}, id="final"),
    pytest.param(
        {"lr_schedule": "cosine", "lr_warmup_steps": 2, "horizon_increment": 0}, id="cosine"
    ),
    pytest.param({"lr_warmup_steps": 4, "horizon_increment": 0}, id="constant-warmup"),
    pytest.param(D3, id="D3"),
    pytest.param({**AUGMENT, **D3, "paper_arch": "p4m"}, id="D3-C5-C7-p4m"),
]

#: Split runs: (flags, fold or None for the screen).
SPLIT_RUNS: list[Any] = [
    pytest.param(AUGMENT, 2, id="fold-augmented"),
    pytest.param({"rot90_augment": True}, None, id="screen"),
]

#: A short RJ23 regime: validation every 2 steps from step 0, patience 2, a 40-step cap.
RJ23: dict[str, Any] = {
    "val_every_n_steps": 2,
    "max_steps": 40,
    "patience": 2,
    "batch_size": 16,
    "eval_batch_size": 8,
    "checkpoint_every_n_steps": 2,
    "deterministic": True,
    "progress_bar_refresh_rate": 0,
}

#: (preset, changes, protocol, expected stop reason): each scenario, both protocols, the
#: sensitivity flags and both ways to stop.
RJ23_CASES: list[Any] = [
    pytest.param("rj23-lowres", {}, "replication", "patience", id="lowres"),
    pytest.param("rj23-multiscale", {}, "decision", "patience", id="multiscale-decision"),
    pytest.param(
        "rj23-full",
        {"rj23_bn_position": "before_concat", "rj23_keep": "best"},
        "replication",
        "patience",
        id="full-bn-before-keep-best",
    ),
    pytest.param(
        "rj23-cropped",
        {"patience": 50, "max_steps": 8, "rj23_large_values": "code"},
        "replication",
        "cap",
        id="cropped-capped",
    ),
]

#: ``metrics.json`` entries that name the engine, the machine or the run, not what it
#: computed (the run id holds the machine class and the data's hash of a temporary path's
#: layout, the timing is wall-clock).
ENGINE_ENTRIES = ("accelerator", "checkpoints", "versions", "run", "timing", "machine")

#: One validation: (steps done, the monitored value, the weights then).
Validations = list[tuple[int, float, dict[str, torch.Tensor]]]


def paper_settings(data: Path, out: Path, **changes: Any) -> Settings:
    """The paper classifier's ``Settings`` on :data:`SCHEDULE`, with ``changes``."""
    base = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=data.parent,
        dataset=data.name,
        output_dir=out,
        **SCHEDULE,
    )
    return base.replace(**changes)


def rj23_settings(data_dir: Path, out: Path, dataset: str, preset: str, **changes: Any) -> Settings:
    """RJ23's ``Settings`` of ``preset`` on :data:`RJ23`, with ``changes``."""
    values: dict[str, Any] = {
        **PRESETS[preset],
        **RJ23,
        "dataset": dataset,
        "data_dir": data_dir,
        "output_dir": out,
        "accelerator": "cpu",
        **changes,
    }
    return settings_from_mapping(values)


def make_split_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(prepared dir, splits file) with 25 stamps per split and one leaked Train stamp."""
    raw = tmp_path_factory.mktemp("cases_raw")
    out = tmp_path_factory.mktemp("cases_prepared")
    make_synthetic(raw, n_per_split=25, seed=1)
    source = raw / CLASSIFIER_FILES[0]
    # Our own synthetic file, written just above: unpickling it is safe.
    content = pickle.loads(source.read_bytes())  # noqa: S301
    for key in ("images", "features"):
        content["Train"][key][3] = content["Validation"][key][7]  # the leaked stamp
    source.write_bytes(pickle.dumps(content))
    prepared = prepare(source, out, images="float32")
    splits = make_splits(prepared, selection_per_class=1, screen_per_class=1)
    path = tmp_path_factory.mktemp("cases_splits") / "splits.json"
    write_splits(splits, path)
    return prepared, path


def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of ``module``'s ``state_dict``."""
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """SHA-256 of every tensor's name, dtype, shape and bytes, in the ``state_dict``'s order."""
    digest = hashlib.sha256()
    for key, tensor in state.items():
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{key}|{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PlainProbe:
    """The plain run's fit validations (an ``on_validation`` hook)."""

    def __init__(self) -> None:
        self.validations: Validations = []

    def __call__(self, record: Validation, task: Task) -> None:
        self.validations.append((record.steps_done, record.value, cpu_state(task.module)))


def assert_same_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    """The two ``state_dict`` copies hold the same keys and bit-identical tensors."""
    assert left.keys() == right.keys()
    for key, tensor in left.items():
        assert torch.equal(tensor, right[key]), key


def assert_same_validations(expected: Validations, actual: Validations) -> None:
    """The same steps, the same monitored values and bit-identical weights after each."""
    assert [(s, v) for s, v, _ in actual] == [(s, v) for s, v, _ in expected]
    for (step, _, left), (_, _, right) in zip(expected, actual, strict=True):
        try:
            assert_same_states(left, right)
        except AssertionError as err:
            raise AssertionError(f"weights differ after the validation of step {step}") from err


def _without_paths(value: Any) -> Any:
    """``value`` with every absolute path string replaced by its last component."""
    if isinstance(value, dict):
        return {k: _without_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_paths(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def computed(metrics: dict[str, Any]) -> dict[str, Any]:
    """``metrics`` without what names the engine, the machine or the run, and without paths.

    The JSON round trip makes it compare equal to a recorded golden (tuples become lists;
    Python floats survive it exactly).
    """
    kept: dict[str, Any] = {k: v for k, v in metrics.items() if k not in ENGINE_ENTRIES}
    kept["resumed_from"] = metrics["resumed_from"] is not None
    # Where the run read and wrote (temporary directories), not what it computed.
    kept["config"] = {
        k: v for k, v in metrics["config"].items() if k not in ("data_dir", "output_dir")
    }
    if "keep" in kept:  # rj23: each keep's checkpoint is a path of its own engine's layout
        kept["keep"] = {
            name: entry
            if not isinstance(entry, dict)
            else {k: v for k, v in entry.items() if k != "checkpoint"}
            for name, entry in kept["keep"].items()
        }
    plain: dict[str, Any] = json.loads(json.dumps(_without_paths(kept)))
    return plain


def predictions_digests(run_dir: Path) -> dict[str, dict[str, str]]:
    """SHA-256 of every array (dtype and bytes) of every ``predictions_*.npz`` of a run."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(run_dir.glob("predictions_*.npz")):
        with np.load(path) as arrays:
            out[path.name] = {
                key: hashlib.sha256(
                    arrays[key].dtype.str.encode() + np.ascontiguousarray(arrays[key]).tobytes()
                ).hexdigest()
                for key in sorted(arrays.files)
            }
    return out
