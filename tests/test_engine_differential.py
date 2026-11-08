"""The 4.7(a) CPU differential test: Lightning's ``carrasco2021`` path and the plain engine.

ADR 0014, section 6, and plan task 4.7(a), a CI gate until Lightning is removed (4.8): both
engines run in one process on the synthetic prepared data, on a schedule with a horizon
extension (validation every 3 steps, an initial horizon of 10 and an increment of 10, so
the first validation's best always extends the run; ``lr`` 3e-2 so that the extensions
end within a few dozen steps), and must agree **bit for bit**:

- the initial weights (C1);
- after every fit validation: the steps done, the monitored ``loss_val`` (the same Python
  float) and every tensor of the ``state_dict`` (``torch.equal``: parameters and the batch
  norms' running statistics);
- ``best_step``, the horizon, ``last_step``, ``monitor_best`` and the derived epochs;
- the final evaluation on the restored best weights: the per-stamp probabilities of both
  roles (``torch.equal``) and the accuracies ``metrics.json`` reports.

It is parametrised over the 2.5 flags that exist (``paper_arch`` ``p4`` / ``p4m`` /
``single``, ``context``, ``rot90_augment``, C5's noise and jitter, C6, C7, ``feature_set``),
split runs (a fold and the screen) and a mid-run resume of the plain engine. A flag added
to one engine only therefore fails CI (CLAUDE.md, "New training flags are engine-free").

The second half ports the semantics of the three Lightning-bound resume tests (ADR 0014,
C10) to the plain engine: ``test_resume_from_mid_epoch_last_checkpoint``,
``test_resume_at_the_horizon_still_writes_best`` and ``test_augmented_resume_is_exact``
(here bit-exact, where the Lightning test allows ``atol=1e-6``).
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from _plain_paper import PlainRun, plain_paper_run
from _training_support import ACCELERATORS
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

from astro_simclr.config import Settings
from astro_simclr.data import prepare
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.engine import Task, Validation, load_best, load_checkpoint, read_scalars
from astro_simclr.experiments import run_command
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.protocol.splits import make_splits, write_splits

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
    pytest.param({"context": "avgpool63"}, id="context"),
    pytest.param({"rot90_augment": True}, id="rot90"),
    pytest.param({"aug_noise": 1.0, "aug_feature_jitter": 0.1, "aug_min_r2": 0.0}, id="C5"),
    pytest.param({"synthetic_bogus": 0.25, "aug_min_r2": 0.0}, id="C6"),
    pytest.param({"flux_rescale": 0.5}, id="C7"),
    pytest.param({"feature_set": "position"}, id="feature_set-position"),
    pytest.param({"feature_set": "none", "rot90_augment": True}, id="feature_set-none"),
    pytest.param(_AUGMENT, id="C5-C7-p4"),
    # Plan row B9: the conv-attention hybrids (both trunks, both readouts).
    pytest.param({"paper_arch": "hybrid", "hybrid_readout": "cls"}, id="hybrid-cls"),
    pytest.param(
        {"paper_arch": "hybrid_p4", "hybrid_dim": 32, "hybrid_heads": 2, "rot90_augment": True},
        id="hybrid_p4",
    ),
]

#: Split runs: (flags, fold or None for the screen).
_SPLIT_RUNS: list[Any] = [
    pytest.param(_AUGMENT, 2, id="fold-augmented"),
    pytest.param({"rot90_augment": True}, None, id="screen"),
]


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
    """What the Lightning run did: initial weights, every fit validation, final outputs."""

    def __init__(self) -> None:
        self.initial: dict[str, torch.Tensor] | None = None
        self.validations: list[tuple[int, float, dict[str, torch.Tensor]]] = []
        self.final: dict[str, torch.Tensor] = {}

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.initial = _cpu_state(pl_module)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.state.fn == TrainerFn.FITTING:
            value = float(trainer.callback_metrics["loss_val"])
            self.validations.append((trainer.global_step, value, _cpu_state(pl_module)))
        else:
            self.final["Validation"] = pl_module.probabilities["val"].clone()  # type: ignore[index]

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.final["Test"] = pl_module.probabilities["test"].clone()  # type: ignore[index]


class _PlainProbe:
    """The plain run's fit validations (an ``on_validation`` hook)."""

    def __init__(self) -> None:
        self.validations: list[tuple[int, float, dict[str, torch.Tensor]]] = []

    def __call__(self, record: Validation, task: Task) -> None:
        self.validations.append((record.steps_done, record.value, _cpu_state(task.module)))


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


def _assert_same_validations(
    lightning: list[tuple[int, float, dict[str, torch.Tensor]]],
    plain: list[tuple[int, float, dict[str, torch.Tensor]]],
) -> None:
    assert [(s, v) for s, v, _ in plain] == [(s, v) for s, v, _ in lightning]
    for (step, _, expected), (_, _, actual) in zip(lightning, plain, strict=True):
        try:
            _assert_same_states(expected, actual)
        except AssertionError as err:
            raise AssertionError(f"weights differ after the validation of step {step}") from err


def _accuracy(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    return float((probabilities.argmax(dim=1) == targets).double().mean())


def _assert_same_outcome(metrics: dict[str, Any], run: PlainRun, *, split: bool) -> None:
    result = run.result
    assert result.best_step == metrics["best_step"]
    assert result.last_step == metrics["last_step"]
    assert result.horizon == metrics["horizon"]
    assert result.horizon > _SCHEDULE["max_steps"]  # the schedule did extend
    assert result.best_score == metrics["monitor_best"]
    assert result.best_epoch == metrics["best_epoch"]
    assert result.last_epoch == metrics["last_epoch"]
    validation, test = run.evaluation["Validation"], run.evaluation["Test"]
    keys = (
        ("selection_acc_best", "evaluation_accuracy")
        if split
        else ("val_acc_best", "test_acc_at_best")
    )
    assert _accuracy(validation.probabilities, validation.targets) == metrics[keys[0]]
    assert _accuracy(test.probabilities, test.targets) == metrics[keys[1]]


def _assert_engines_agree(
    settings: Settings, plain_out: Path, split: SplitRequest | None = None
) -> PlainRun:
    metrics, lightning = _lightning(settings, split)
    probe = _PlainProbe()
    run = plain_paper_run(settings.replace(output_dir=plain_out), split=split, on_validation=probe)
    assert lightning.initial is not None
    _assert_same_states(lightning.initial, run.initial)
    _assert_same_validations(lightning.validations, probe.validations)
    _assert_same_outcome(metrics, run, split=split is not None)
    assert torch.equal(run.evaluation["Validation"].probabilities, lightning.final["Validation"])
    assert torch.equal(run.evaluation["Test"].probabilities, lightning.final["Test"])
    return run


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
    _assert_engines_agree(settings, tmp_path / "plain", SplitRequest(path, fold))


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_engines_agree_on_each_accelerator(
    accelerator: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    # Deterministic kernels on the same device: the same ops in the same order (4.7(b)'s
    # check in small; on CPU it repeats the faithful case).
    settings = _settings(
        classifier_f32_dir, tmp_path / "lightning", accelerator=accelerator, **_AUGMENT
    )
    _assert_engines_agree(settings, tmp_path / "plain")


class _CopyLast:
    """At the validation of step ``step``, copy ``last.pt`` (written just before) to ``dst``."""

    def __init__(self, step: int, src: Path, dst: Path) -> None:
        self.step, self.src, self.dst = step, src, dst
        self.inner = _PlainProbe()

    def __call__(self, record: Validation, task: Task) -> None:
        self.inner(record, task)
        if record.steps_done == self.step:
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, self.dst)


def test_mid_run_resume_agrees_with_an_uninterrupted_lightning_run(
    classifier_f32_dir: Path, tmp_path: Path
) -> None:
    settings = _settings(classifier_f32_dir, tmp_path / "lightning", **_AUGMENT)
    metrics, lightning = _lightning(settings)
    whole = tmp_path / "whole"
    saved = tmp_path / "mid" / "last.pt"
    plain_paper_run(
        settings.replace(output_dir=whole),
        on_validation=_CopyLast(9, whole / "ckpt" / "last.pt", saved),
    )
    probe = _PlainProbe()
    resumed = plain_paper_run(
        settings.replace(output_dir=tmp_path / "resumed"), resume=saved, on_validation=probe
    )
    _assert_same_validations([v for v in lightning.validations if v[0] > 9], probe.validations)
    _assert_same_outcome(metrics, resumed, split=False)
    assert torch.equal(resumed.evaluation["Test"].probabilities, lightning.final["Test"])


# --- the resume tests of the Lightning path, ported (ADR 0014, C10) -------------------


def test_resume_from_mid_epoch_last_checkpoint(classifier_f32_dir: Path, tmp_path: Path) -> None:
    # last.pt every 3 steps, on the validation grid, in epochs of 2 steps: the step-9 copy
    # is mid-epoch and was written after the step-9 validation, which the resume must not
    # lose nor repeat.
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = resumed / "ckpt" / "last.pt"
    copy = _CopyLast(9, whole / "ckpt" / "last.pt", saved)
    expected = plain_paper_run(_settings(classifier_f32_dir, whole), on_validation=copy)
    checkpoint = load_checkpoint(saved)
    assert checkpoint["steps_done"] == 9
    assert expected.result.batches_per_pass == 2
    assert checkpoint["policy"]["horizon"] > 10  # extended before the checkpoint
    assert checkpoint["policy"]["best_step"] is not None

    probe = _PlainProbe()
    run = plain_paper_run(_settings(classifier_f32_dir, resumed), resume=saved, on_validation=probe)
    assert [v.steps_done for v in expected.result.validations][:3] == [3, 6, 9]
    assert [s for s, _, _ in probe.validations] == [
        v.steps_done for v in expected.result.validations if v.steps_done > 9
    ]
    _assert_same_validations([v for v in copy.inner.validations if v[0] > 9], probe.validations)
    for key in ("best_step", "last_step", "horizon", "best_score"):
        assert getattr(run.result, key) == getattr(expected.result, key), key
    assert run.result.resumed_from == saved
    assert run.result.steps == expected.result.last_step - 9
    _assert_same_states(
        load_best(whole / "ckpt" / "best.pt")["model"],
        load_best(resumed / "ckpt" / "best.pt")["model"],
    )
    # The scalars of the resumed directory: the step-9 checkpoint's run had none there, so
    # only the resumed validations; a resume in the same directory keeps steps <= 9.
    steps = [record["step"] for record in read_scalars(resumed / "scalars.jsonl")]
    assert steps == [s for s, _, _ in probe.validations]


def test_resume_at_the_horizon_still_writes_best(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    expected = plain_paper_run(_settings(classifier_f32_dir, out))
    best = load_best(out / "ckpt" / "best.pt")
    (out / "ckpt" / "best.pt").unlink()
    last = out / "ckpt" / "last.pt"
    assert load_checkpoint(last)["steps_done"] == expected.result.horizon
    # The final last.pt is at the horizon: no step runs, and best.pt is written again.
    run = plain_paper_run(_settings(classifier_f32_dir, out), resume=last)
    assert run.result.steps == 0
    assert run.result.validations == ()
    assert run.result.best_step == expected.result.best_step == best["steps_done"]
    assert run.result.best_score == expected.result.best_score == best["score"]
    rewritten = load_best(out / "ckpt" / "best.pt")
    _assert_same_states(best["model"], rewritten["model"])
    for role in ("Validation", "Test"):
        assert torch.equal(
            run.evaluation[role].probabilities, expected.evaluation[role].probabilities
        )
    # The scalars of the first fit are kept (all of them are at steps <= the horizon).
    assert len(read_scalars(out / "scalars.jsonl")) == len(expected.result.validations)


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
    expected = plain_paper_run(
        _settings(prepared, whole, **changes),
        split=split,
        on_validation=_CopyLast(2, whole / "ckpt" / "last.pt", mid),
    )
    run = plain_paper_run(
        _settings(prepared, tmp_path / "resumed", **changes), split=split, resume=mid
    )
    assert run.result.last_step == expected.result.last_step == 6
    for role in ("Validation", "Test"):
        assert torch.equal(
            run.evaluation[role].probabilities, expected.evaluation[role].probabilities
        )
