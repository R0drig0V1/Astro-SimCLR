"""The 4.7(a) CPU differential test: Lightning's ``run_command`` and the plain ``run_plain``.

ADR 0014, section 6, and plan task 4.7(a), a CI gate until Lightning is removed (4.8). Since
task 4.6 it runs the real plain path, ``tasks.plain.run_plain`` (the paper classifier task,
its data, the final evaluation and ``metrics.json``), against today's Lightning
``experiments.run_command``, in one process on the synthetic prepared data, on a schedule
with a horizon extension (validation every 3 steps, an initial horizon of 10 and an
increment of 10, so the first validation's best always extends the run; ``lr`` 3e-2 so that
the extensions end within a few dozen steps). The two runs must agree **bit for bit**:

- the initial weights (``build_paper_task`` after the seed, against the Lightning module's);
- after every fit validation: the steps done, the monitored ``loss_val`` (the same Python
  float) and every tensor of the ``state_dict`` (``torch.equal``);
- the whole ``metrics.json``, except what names the engine or the run (the machine's
  ``engine``, the checkpoint paths, the Lightning version, the accelerator's class name,
  the run id, the output directory and the timing): the steps, horizon, epochs and
  ``monitor_best``, ``paper_data``, ``preset``, ``protocol``, every evaluation entry
  (accuracies, macro-F1, recalls, both confusion matrices);
- the per-stamp predictions files (``predictions_*.npz``), array by array;
- the manifests: ``engine: plain`` and ``lightning``, both ``finished``.

It is parametrised over the 2.5 flags on main (``paper_arch`` ``single`` / ``p4`` / ``p4m`` /
``convnext_atto``, ``context``, ``rot90_augment``, C5's noise and jitter, C6, C7,
``feature_set``, ``fp32_precision``), split runs (a fold and the screen), a mid-run resume of
the plain engine and each accelerator present. A flag added to one engine only therefore
fails CI (CLAUDE.md, "New training flags are engine-free").

The second half ports the semantics of the three Lightning-bound resume tests (ADR 0014,
C10) to the plain runner: ``test_resume_from_mid_epoch_last_checkpoint``,
``test_resume_at_the_horizon_still_writes_best`` and ``test_augmented_resume_is_exact``
(here bit-exact, where the Lightning test allows ``atol=1e-6``).
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pytest
import torch
from _training_support import ACCELERATORS
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

from astro_simclr.config import Settings
from astro_simclr.data import prepare
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.engine import (
    Task,
    Validation,
    load_best,
    load_checkpoint,
    read_scalars,
    seed_everything,
)
from astro_simclr.experiments import run_command
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.protocol.splits import make_splits, write_splits
from astro_simclr.tasks.paper import build_paper_task
from astro_simclr.tasks.plain import run_plain

_SCHEDULE: dict[str, Any] = {
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

#: Every class-aware augmentation on, as ``test_split_runs._AUGMENT`` (C5, C6, C7, p4).
_AUGMENT: dict[str, Any] = {
    "paper_arch": "p4",
    "aug_noise": 1.0,
    "aug_feature_jitter": 0.1,
    "synthetic_bogus": 0.25,
    "aug_min_r2": 0.0,
    "flux_rescale": 0.5,
}

#: The 2.5 flags, one case each (and the faithful model first).
_FLAGS: list[Any] = [
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
    pytest.param(_AUGMENT, id="C5-C7-p4"),
]

#: Split runs: (flags, fold or None for the screen).
_SPLIT_RUNS: list[Any] = [
    pytest.param(_AUGMENT, 2, id="fold-augmented"),
    pytest.param({"rot90_augment": True}, None, id="screen"),
]

#: ``metrics.json`` entries that name the engine or the run, not what it computed.
_ENGINE_ENTRIES = ("accelerator", "checkpoints", "versions", "run", "timing")

Validations = list[tuple[int, float, dict[str, torch.Tensor]]]


def _settings(data: Path, out: Path, **changes: Any) -> Settings:
    base = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=data.parent,
        dataset=data.name,
        output_dir=out,
        **_SCHEDULE,
    )
    return base.replace(**changes)


@pytest.fixture(scope="module")
def split_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """(prepared dir, splits file) with 25 stamps per split, as ``test_split_runs``'s."""
    raw = tmp_path_factory.mktemp("diff_raw")
    out = tmp_path_factory.mktemp("diff_prepared")
    make_synthetic(raw, n_per_split=25, seed=1)
    source = raw / CLASSIFIER_FILES[0]
    # Our own synthetic file, written just above: unpickling it is safe.
    content = pickle.loads(source.read_bytes())  # noqa: S301
    for key in ("images", "features"):
        content["Train"][key][3] = content["Validation"][key][7]  # the leaked stamp
    source.write_bytes(pickle.dumps(content))
    prepared = prepare(source, out, images="float32")
    splits = make_splits(prepared, selection_per_class=1, screen_per_class=1)
    path = tmp_path_factory.mktemp("diff_splits") / "splits.json"
    write_splits(splits, path)
    return prepared, path


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


class _LightningProbe(Callback):
    """What the Lightning run did: its initial weights and every fit validation."""

    def __init__(self) -> None:
        self.initial: dict[str, torch.Tensor] | None = None
        self.validations: Validations = []

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.initial = _cpu_state(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.state.fn == TrainerFn.FITTING:
            value = float(trainer.callback_metrics["loss_val"])
            self.validations.append((trainer.global_step, value, _cpu_state(pl_module)))


class _PlainProbe:
    """The plain run's fit validations (an ``on_validation`` hook)."""

    def __init__(self) -> None:
        self.validations: Validations = []

    def __call__(self, record: Validation, task: Task) -> None:
        self.validations.append((record.steps_done, record.value, _cpu_state(task.module)))


class _CopyLast(_PlainProbe):
    """Also copy ``ckpt/last.pt`` (written just before) to ``dst`` at the step-``step`` one."""

    def __init__(self, step: int, src: Path, dst: Path) -> None:
        super().__init__()
        self.step, self.src, self.dst = step, src, dst

    def __call__(self, record: Validation, task: Task) -> None:
        super().__call__(record, task)
        if record.steps_done == self.step:
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, self.dst)


def _lightning(
    settings: Settings, split: SplitRequest | None = None
) -> tuple[dict[str, Any], _LightningProbe]:
    probe = _LightningProbe()
    metrics = run_command(
        "train-supervised", settings, callbacks=[probe], progress_bar=False, split=split
    )
    return metrics, probe


def _assert_same_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key, tensor in left.items():
        assert torch.equal(tensor, right[key]), key


def _assert_same_validations(lightning: Validations, plain: Validations) -> None:
    assert [(s, v) for s, v, _ in plain] == [(s, v) for s, v, _ in lightning]
    for (step, _, expected), (_, _, actual) in zip(lightning, plain, strict=True):
        try:
            _assert_same_states(expected, actual)
        except AssertionError as err:
            raise AssertionError(f"weights differ after the validation of step {step}") from err


def _computed(metrics: dict[str, Any]) -> dict[str, Any]:
    """``metrics`` without what names the engine or the run (module docstring)."""
    kept = {k: v for k, v in metrics.items() if k not in _ENGINE_ENTRIES}
    kept["machine"] = {k: v for k, v in metrics["machine"].items() if k != "engine"}
    kept["config"] = {k: v for k, v in metrics["config"].items() if k != "output_dir"}
    kept["resumed_from"] = metrics["resumed_from"] is not None
    return kept


def _assert_same_outputs(lightning_dir: Path, plain_dir: Path) -> None:
    """The predictions files, array by array, and the two manifests."""
    names = sorted(p.name for p in lightning_dir.glob("predictions_*.npz"))
    assert names and names == sorted(p.name for p in plain_dir.glob("predictions_*.npz"))
    for name in names:
        with np.load(lightning_dir / name) as left, np.load(plain_dir / name) as right:
            assert sorted(left.files) == sorted(right.files)
            for key in left.files:
                assert left[key].dtype == right[key].dtype
                assert np.array_equal(left[key], right[key]), (name, key)
    for path, engine in ((lightning_dir, "lightning"), (plain_dir, "plain")):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        assert (manifest["engine"], manifest["status"]) == (engine, "finished")


def _assert_engines_agree(
    settings: Settings, plain_out: Path, split: SplitRequest | None = None
) -> dict[str, Any]:
    expected, lightning = _lightning(settings, split)
    probe = _PlainProbe()
    plain = settings.replace(output_dir=plain_out)
    metrics = run_plain("train-supervised", plain, split=split, on_validation=probe)
    seed_everything(settings.seed)
    initial = _cpu_state(build_paper_task(settings, {}).module)
    assert lightning.initial is not None
    _assert_same_states(lightning.initial, initial)
    _assert_same_validations(lightning.validations, probe.validations)
    assert _computed(metrics) == _computed(expected)
    assert metrics["horizon"] > _SCHEDULE["max_steps"]  # the schedule did extend
    assert metrics["machine"]["engine"] == "plain"
    _assert_same_outputs(settings.output_dir, plain_out)
    return metrics


# --- the differential test ------------------------------------------------------------


@pytest.mark.parametrize("flags", _FLAGS)
def test_engines_agree_bit_for_bit(
    flags: dict[str, Any], classifier_f32_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(classifier_f32_dir, tmp_path / "lightning", **flags)
    _assert_engines_agree(settings, tmp_path / "plain")


@pytest.mark.parametrize(("flags", "fold"), _SPLIT_RUNS)
def test_engines_agree_on_split_runs(
    flags: dict[str, Any], fold: int | None, split_data: tuple[Path, Path], tmp_path: Path
) -> None:
    prepared, path = split_data
    settings = _settings(prepared, tmp_path / "lightning", **flags)
    metrics = _assert_engines_agree(settings, tmp_path / "plain", SplitRequest(path, fold))
    assert "protocol" in metrics and "evaluation_accuracy" in metrics


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_engines_agree_on_each_accelerator(
    accelerator: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    # Deterministic kernels on the same device: the same ops in the same order (4.7(b)'s
    # check in small; on CPU it repeats the C5-C7 case).
    settings = _settings(
        classifier_f32_dir, tmp_path / "lightning", accelerator=accelerator, **_AUGMENT
    )
    _assert_engines_agree(settings, tmp_path / "plain")


def test_mid_run_resume_agrees_with_an_uninterrupted_lightning_run(
    classifier_f32_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(classifier_f32_dir, tmp_path / "lightning", **_AUGMENT)
    expected, lightning = _lightning(settings)
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = tmp_path / "mid" / "last.pt"
    run_plain(
        "train-supervised",
        settings.replace(output_dir=whole),
        on_validation=_CopyLast(9, whole / "ckpt" / "last.pt", saved),
    )
    probe = _PlainProbe()
    metrics = run_plain(
        "train-supervised",
        settings.replace(output_dir=resumed),
        resume=str(saved),
        on_validation=probe,
    )
    _assert_same_validations([v for v in lightning.validations if v[0] > 9], probe.validations)
    assert metrics["resumed_from"] == str(saved)
    assert {**_computed(metrics), "resumed_from": False} == _computed(expected)
    _assert_same_outputs(settings.output_dir, resumed)


# --- the resume tests of the Lightning path, ported (ADR 0014, C10) -------------------


def test_resume_from_mid_epoch_last_checkpoint(classifier_f32_dir: Path, tmp_path: Path) -> None:
    # last.pt every 3 steps, on the validation grid, in epochs of 2 steps: the step-9 copy
    # is mid-epoch and was written after the step-9 validation, which the resume must not
    # lose nor repeat.
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = resumed / "ckpt" / "last.pt"
    copy = _CopyLast(9, whole / "ckpt" / "last.pt", saved)
    expected = run_plain(
        "train-supervised", _settings(classifier_f32_dir, whole), on_validation=copy
    )
    checkpoint = load_checkpoint(saved)
    assert checkpoint["steps_done"] == 9
    assert checkpoint["policy"]["horizon"] > 10  # extended before the checkpoint
    assert checkpoint["policy"]["best_step"] is not None

    probe = _PlainProbe()
    metrics = run_plain(
        "train-supervised",
        _settings(classifier_f32_dir, resumed),
        resume="auto",
        on_validation=probe,
    )
    assert [s for s, _, _ in copy.validations][:3] == [3, 6, 9]
    _assert_same_validations([v for v in copy.validations if v[0] > 9], probe.validations)
    for key in ("best_step", "last_step", "horizon", "monitor_best", "val_acc_best"):
        assert metrics[key] == expected[key], key
    assert metrics["resumed_from"] == str(saved)
    assert metrics["timing"]["steps"] == expected["last_step"] - 9
    _assert_same_states(
        load_best(whole / "ckpt" / "best.pt")["model"],
        load_best(resumed / "ckpt" / "best.pt")["model"],
    )
    # The resumed directory's scalars: only its own validations (none was logged there).
    steps = [record["step"] for record in read_scalars(resumed / "scalars.jsonl")]
    assert steps == [s for s, _, _ in probe.validations]


def test_resume_at_the_horizon_still_writes_best(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    settings = _settings(classifier_f32_dir, out)
    expected = run_plain("train-supervised", settings)
    best = load_best(out / "ckpt" / "best.pt")
    (out / "ckpt" / "best.pt").unlink()
    assert load_checkpoint(out / "ckpt" / "last.pt")["steps_done"] == expected["horizon"]
    # The final last.pt is at the horizon: no step runs, and best.pt is written again.
    metrics = run_plain("train-supervised", settings, resume="auto")
    assert metrics["timing"]["steps"] == 0
    for key in ("best_step", "monitor_best", "val_acc_best", "test_acc_at_best", "evaluation"):
        assert metrics[key] == expected[key], key
    assert metrics["best_step"] == best["steps_done"]
    assert metrics["monitor_best"] == best["score"]
    _assert_same_states(best["model"], load_best(out / "ckpt" / "best.pt")["model"])
    # A resume after "finished" is the next attempt of the same run (ADR 0015, amendment A1).
    assert metrics["run"]["attempt"] == expected["run"]["attempt"] + 1
    assert metrics["run"]["run_id"] == expected["run"]["run_id"]


def test_augmented_resume_is_exact(split_data: tuple[Path, Path], tmp_path: Path) -> None:
    # The C5 / C6 / C7 draws are a function of [seed, step]: a resumed fold run repeats them,
    # so its predictions are the uninterrupted run's, bit for bit.
    prepared, path = split_data
    changes = {
        "max_steps": 6,
        "val_every_n_steps": 2,
        "horizon_increment": 0,
        "checkpoint_every_n_steps": 2,
        **_AUGMENT,
    }
    whole, mid = tmp_path / "whole", tmp_path / "mid.pt"
    split = SplitRequest(path, 2)
    expected = run_plain(
        "train-supervised",
        _settings(prepared, whole, **changes),
        split=split,
        on_validation=_CopyLast(2, whole / "ckpt" / "last.pt", mid),
    )
    resumed = tmp_path / "resumed"
    metrics = run_plain(
        "train-supervised", _settings(prepared, resumed, **changes), split=split, resume=str(mid)
    )
    assert metrics["last_step"] == expected["last_step"] == 6
    assert metrics["evaluation"] == expected["evaluation"]
    with (
        np.load(whole / "predictions_evaluation.npz") as left,
        np.load(resumed / "predictions_evaluation.npz") as right,
    ):
        assert np.array_equal(left["probabilities"], right["probabilities"])
