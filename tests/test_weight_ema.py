"""Plan row D3: the weight EMA, the learning-rate schedule, the final-weights selection, ECE.

The pure pieces against hand computations (``training.ema``, ``training.lr_schedule``,
``protocol.calibration``), the ``Settings`` rules, and runs of the plain engine
(``tasks.plain.run_plain``) on the synthetic float32 data (``classifier_f32_dir``): the EMA
and the scheduler state are in the checkpoints and a resume restores them exactly;
``checkpoint_selection: final`` scores the last weights and lets the validation decide
nothing. The bit-for-bit runs of every D3 flag are ``test_engine_regression.py``'s goldens.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _training_support import ACCELERATORS
from torch import nn

from astro_simclr.config import ConfigError, Settings
from astro_simclr.engine import Task, Validation, evaluate, load_checkpoint, seed_everything
from astro_simclr.engine.loop import _Fit
from astro_simclr.losses import P_stamps_loss
from astro_simclr.protocol.calibration import expected_calibration_error
from astro_simclr.protocol.phase2_report import _replication_arm, find_arms
from astro_simclr.registry.readers import metrics_summary
from astro_simclr.tasks.paper import PaperNet, build_paper_task, paper_inputs
from astro_simclr.tasks.plain import _sources, run_plain
from astro_simclr.tasks.results import StageOutput, evaluation_entry
from astro_simclr.training.ema import EmaWeights, ema_decay_at, trainable_parameters
from astro_simclr.training.lr_schedule import LrSchedule

_SCHEDULE: dict[str, Any] = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "monitor": "loss_val",
    "optimizer": "Adam",
    "lr": 3e-2,
    "beta_loss": 0.5,
    "batch_size": 4,
    "max_steps": 10,
    "val_every_n_steps": 3,
    "horizon_increment": 0,
    "checkpoint_every_n_steps": 3,
}

#: Every D3 flag on (a fixed horizon of ``max_steps``).
_D3: dict[str, Any] = {
    "ema_decay": 0.9,
    "ema_warmup": True,
    "lr_schedule": "cosine",
    "lr_warmup_steps": 3,
    "checkpoint_selection": "final",
}


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


class _Tiny(nn.Module):
    """A linear layer and a BatchNorm: parameters and buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        out: torch.Tensor = self.norm(self.linear(x))
        return out


def _fill(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


# --- training.ema --------------------------------------------------------------------------


def test_ema_decay_warmup_is_the_tf_ramp() -> None:
    assert ema_decay_at(0.999, 0, warmup=False) == 0.999
    assert ema_decay_at(0.999, 0, warmup=True) == 1 / 10
    assert ema_decay_at(0.999, 8, warmup=True) == 9 / 18
    assert ema_decay_at(0.999, 10**6, warmup=True) == 0.999  # the ramp passed it
    assert ema_decay_at(0.5, 20, warmup=True) == 0.5
    with pytest.raises(ValueError, match="step"):
        ema_decay_at(0.9, -1, warmup=True)


def test_ema_update_matches_a_hand_computation() -> None:
    model = _Tiny()
    _fill(model, 1.0)
    ema = EmaWeights(model, 0.75)
    _fill(model, 3.0)
    ema.update(model, step=0)
    # 0.75 * 1 + 0.25 * 3 = 1.5 (exact in binary).
    assert all(torch.equal(p, torch.full_like(p, 1.5)) for p in ema.shadow.parameters())
    _fill(model, 5.5)
    ema.update(model, step=1)
    # 0.75 * 1.5 + 0.25 * 5.5 = 2.5.
    assert all(torch.equal(p, torch.full_like(p, 2.5)) for p in ema.shadow.parameters())
    # The model is not touched.
    assert all(torch.equal(p, torch.full_like(p, 5.5)) for p in model.parameters())


def test_ema_warmup_update_matches_a_hand_computation() -> None:
    model = _Tiny()
    _fill(model, 0.0)
    ema = EmaWeights(model, 0.999, warmup=True)
    expected = 0.0
    for step, value in enumerate((10.0, 20.0, 30.0)):
        _fill(model, value)
        ema.update(model, step)
        decay = (1 + step) / (10 + step)
        expected = decay * expected + (1 - decay) * value
        for parameter in ema.shadow.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, expected), rtol=1e-6)
    # By hand: 0.9 * 10 = 9; (2/11) * 9 + (9/11) * 20 = 18; (3/12) * 18 + (9/12) * 30 = 27.
    assert expected == pytest.approx(27.0)


def test_ema_copies_the_batchnorm_buffers() -> None:
    model = _Tiny()
    ema = EmaWeights(model, 0.9)
    model.train()
    model(torch.randn(8, 2))  # moves running_mean / running_var and num_batches_tracked
    model(torch.randn(8, 2) + 3)
    ema.update(model, step=0)
    for name, buffer in model.named_buffers():
        assert torch.equal(dict(ema.shadow.named_buffers())[name], buffer), name
    assert int(dict(ema.shadow.named_buffers())["norm.num_batches_tracked"]) == 2


def test_ema_is_frozen_and_out_of_the_optimizer() -> None:
    model = _Tiny()
    holder = nn.Module()
    holder.model = model
    assert list(trainable_parameters(holder)) == list(holder.parameters())  # no EMA: all
    holder.ema = EmaWeights(model, 0.9)
    trainable = list(trainable_parameters(holder))
    assert trainable == list(model.parameters())
    assert all(not p.requires_grad for p in holder.ema.parameters())
    assert all(p.requires_grad for p in trainable)


def test_ema_reset_and_mismatched_models() -> None:
    model = _Tiny()
    ema = EmaWeights(model, 0.9)
    _fill(model, 7.0)
    ema.reset(model)
    assert all(torch.equal(p, torch.full_like(p, 7.0)) for p in ema.shadow.parameters())
    with pytest.raises(ValueError, match="shadow"):
        ema.update(nn.Linear(2, 1), 0)
    for decay in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="ema_decay"):
            EmaWeights(model, decay)


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_ema_lives_on_the_models_device(accelerator: str) -> None:
    model = _Tiny()
    net = PaperNet(model, P_stamps_loss(0.5), reads_features=False, ema=EmaWeights(model, 0.9))
    net.to(accelerator)
    assert net.ema is not None
    devices = {t.device.type for t in [*net.ema.parameters(), *net.ema.buffers()]}
    assert devices == {accelerator}
    net.ema.update(net.model, 0)  # same device: no transfer error


def test_paper_net_evaluates_with_the_ema_weights() -> None:
    model = _Tiny()
    _fill(model, 1.0)
    net = PaperNet(model, P_stamps_loss(0.5), reads_features=False, ema=EmaWeights(model, 0.5))
    _fill(model, 3.0)
    x = torch.ones(4, 2)
    net.train()
    train_out = net(x, torch.zeros(4, 23))
    net.eval()
    assert net.ema is not None
    eval_out = net(x, torch.zeros(4, 23))
    assert torch.equal(eval_out, net.ema.shadow.eval()(x))
    assert not torch.equal(eval_out, model.eval()(x))
    assert train_out.shape == eval_out.shape


# --- training.lr_schedule ------------------------------------------------------------------


def test_lr_schedule_factors_match_a_hand_computation() -> None:
    cosine = LrSchedule("cosine", warmup_steps=2, total_steps=6)
    # Warmup 1/2, 2/2; then 0.5 (1 + cos(pi p)) with p = (s - 2) / 4.
    expected = [
        0.5,
        1.0,
        1.0,
        0.5 * (1 + math.cos(math.pi / 4)),
        0.5,
        0.5 * (1 - math.cos(math.pi / 4)),
    ]
    assert [cosine.factor(s) for s in range(6)] == pytest.approx(expected, abs=1e-15)
    assert cosine.factor(6) == pytest.approx(0.0, abs=1e-15)
    assert cosine.factor(100) == pytest.approx(0.0, abs=1e-15)  # clamped past the end
    constant = LrSchedule("constant", warmup_steps=4, total_steps=10)
    assert [constant.factor(s) for s in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    assert not LrSchedule().active
    assert LrSchedule().scheduler(torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=1.0)) is None


def test_lr_scheduler_sets_each_steps_rate() -> None:
    schedule = LrSchedule("cosine", warmup_steps=1, total_steps=5)
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = schedule.scheduler(optimizer)
    assert scheduler is not None
    used = []
    for _ in range(5):
        used.append(optimizer.param_groups[0]["lr"])
        parameter.grad = torch.ones(1)
        optimizer.step()
        scheduler.step()
    assert used == pytest.approx([0.1 * schedule.factor(s) for s in range(5)])
    assert scheduler.state_dict()["last_epoch"] == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "linear"}, "lr_schedule"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"total_steps": 0}, "total_steps"),
        ({"warmup_steps": 5, "total_steps": 5}, "smaller"),
    ],
)
def test_lr_schedule_rejects_invalid_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LrSchedule(**kwargs)
    with pytest.raises(ValueError, match="step"):
        LrSchedule().factor(-1)


# --- protocol.calibration ------------------------------------------------------------------


def test_ece_matches_a_hand_computation() -> None:
    probabilities = np.array(
        [
            [0.9, 0.1],  # conf 0.9 (bin 14: (13/15, 14/15]), correct
            [0.9, 0.1],  # conf 0.9, wrong
            [0.3, 0.7],  # conf 0.7 (bin 11: (10/15, 11/15]), correct
            [0.6, 0.4],  # conf 0.6 (bin 9: (8/15, 9/15]), correct
        ]
    )
    labels = np.array([0, 1, 1, 0])
    # Bin 14: 2 stamps, accuracy 0.5, confidence 0.9 -> 2/4 * 0.4 = 0.2.
    # Bin 11: 1 stamp, 1.0 vs 0.7 -> 1/4 * 0.3 = 0.075; bin 9: 1/4 * 0.4 = 0.1.
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.375)
    # One bin: |accuracy - mean confidence| = |0.75 - 0.775|.
    assert expected_calibration_error(probabilities, labels, n_bins=1) == pytest.approx(0.025)
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(perfect, np.array([0, 1])) == 0.0


def test_ece_bins_are_right_closed() -> None:
    # A confidence of exactly 8/15 is in bin (7/15, 8/15], with the 0.5 one, not with 0.54.
    edge = 8 / 15
    probabilities = np.array([[edge, 1 - edge], [0.5, 0.5], [0.54, 0.46]])
    labels = np.array([0, 0, 1])
    low = abs(2 - (edge + 0.5))  # bin 8: both correct
    high = abs(0 - 0.54)  # bin 9: wrong
    expected = (low + high) / 3
    assert expected_calibration_error(probabilities, labels) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("probabilities", "labels", "n_bins", "message"),
    [
        (np.zeros((0, 2)), np.zeros(0), 15, "probabilities"),
        (np.zeros((2, 1)), np.zeros(2), 15, "probabilities"),
        (np.full((2, 2), 0.5), np.zeros(3), 15, "labels"),
        (np.full((2, 2), 0.5), np.zeros(2), 0, "n_bins"),
    ],
)
def test_ece_rejects_invalid_inputs(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        expected_calibration_error(probabilities, labels, n_bins=n_bins)


def test_evaluation_entry_records_the_ece() -> None:
    probabilities = torch.tensor([[0.9, 0.05, 0.05, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0, 0.0]])
    stage = StageOutput.of(probabilities, torch.tensor([0, 0]), torch.device("cpu"))
    entry = evaluation_entry(stage)
    assert entry["ece"] == pytest.approx(expected_calibration_error(probabilities.numpy(), [0, 0]))
    assert entry["ece"] == pytest.approx((abs(1 - 0.9) + abs(0 - 0.8)) / 2, rel=1e-6)


# --- Settings ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"checkpoint_selection": "best"}, "checkpoint_selection"),
        ({"lr_schedule": "step"}, "lr_schedule"),
        ({"ema_decay": 1.0}, "ema_decay"),
        ({"ema_decay": 0.0}, "ema_decay"),
        ({"ema_warmup": True}, "needs ema_decay"),
        ({"lr_warmup_steps": -1}, "lr_warmup_steps"),
        ({"checkpoint_selection": "final", "horizon_increment": 10}, "horizon_increment=0"),
        ({"lr_schedule": "cosine", "horizon_increment": 10}, "horizon_increment=0"),
        ({"lr_warmup_steps": 2, "horizon_increment": 10}, "horizon_increment=0"),
        ({"lr_warmup_steps": 10}, "smaller than max_steps"),
        (
            {
                "schedule": "epochs",
                "eval_batch_size": 100,
                "early_stopping": True,
                "ema_decay": 0.9,
            },
            "apply only to schedule carrasco2021",
        ),
    ],
)
def test_settings_refuse_invalid_d3_flags(
    changes: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match=message):
        _settings(tmp_path / "data", tmp_path / "out", **changes)


def test_settings_accept_ema_with_the_horizon_rule(tmp_path: Path) -> None:
    # The EMA alone keeps CD21's selection and horizon extension (it selects among EMA weights).
    settings = _settings(tmp_path / "data", tmp_path / "out", ema_decay=0.999, horizon_increment=10)
    assert settings.checkpoint_selection == "best_val_loss"


# --- runs -------------------------------------------------------------------------------------


class _CopyLast:
    """An ``on_validation`` hook: after step ``step``'s validation, copy ``src`` to ``dst``.

    ``ckpt/last.pt`` is written in the validation of a checkpoint step, before the hook.
    """

    def __init__(self, step: int, src: Path, dst: Path) -> None:
        self.step, self.src, self.dst = step, src, dst

    def __call__(self, record: Validation, task: Task) -> None:
        if record.steps_done == self.step:
            self.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.src, self.dst)


def _last_state(out: Path) -> dict[str, torch.Tensor]:
    return load_checkpoint(out / "ckpt" / "last.pt")["model"]


def test_a_resume_restores_the_ema_and_schedule_exactly(
    classifier_f32_dir: Path, tmp_path: Path
) -> None:
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    saved = resumed / "ckpt" / "last.pt"
    copy = _CopyLast(6, whole / "ckpt" / "last.pt", saved)
    expected = run_plain(
        "train-supervised", _settings(classifier_f32_dir, whole, **_D3), on_validation=copy
    )
    checkpoint = load_checkpoint(saved)
    assert checkpoint["steps_done"] == 6
    shadow = [key for key in checkpoint["model"] if key.startswith("ema.shadow.")]
    assert shadow and len(shadow) * 2 == len(checkpoint["model"])  # model + its EMA
    assert checkpoint["task"]["lr_scheduler"]["last_epoch"] == 6
    metrics = run_plain(
        "train-supervised", _settings(classifier_f32_dir, resumed, **_D3), resume="auto"
    )
    assert metrics["resumed_from"] is not None
    assert metrics["evaluation"] == expected["evaluation"]
    last_whole, last_resumed = _last_state(whole), _last_state(resumed)
    assert last_whole.keys() == last_resumed.keys()
    assert all(torch.equal(last_whole[k], last_resumed[k]) for k in last_whole)
    # The final EMA differs from the final model: it is a real average, not a copy.
    model_keys = [k for k in last_whole if k.startswith("model.") and "num_batches" not in k]
    assert any(
        not torch.equal(last_whole[k], last_whole[k.replace("model.", "ema.shadow.", 1)])
        for k in model_keys
    )


def test_plain_checkpoint_carries_the_ema_and_the_scheduler(
    classifier_f32_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    run_plain("train-supervised", _settings(classifier_f32_dir, out, **_D3))
    checkpoint = load_checkpoint(out / "ckpt" / "last.pt")
    assert checkpoint["steps_done"] == 10
    assert any(key.startswith("ema.shadow.") for key in checkpoint["model"])
    assert checkpoint["task"]["lr_scheduler"]["last_epoch"] == 10
    # The default task carries no state at all (a CD21 checkpoint is what it was).
    default = tmp_path / "default"
    run_plain("train-supervised", _settings(classifier_f32_dir, default))
    plain = load_checkpoint(default / "ckpt" / "last.pt")
    assert plain["task"] == {}
    assert not any(key.startswith("ema.") for key in plain["model"])


def _test_probabilities(out: Path) -> np.ndarray:
    with np.load(out / "predictions_test.npz") as predictions:
        return np.asarray(predictions["probabilities"])


def test_final_selection_scores_the_last_weights(classifier_f32_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "final"
    settings = _settings(classifier_f32_dir, out, max_steps=30, **_D3)
    metrics = run_plain("train-supervised", settings)
    # Validation still ran, was logged and tracked; it chose nothing.
    assert metrics["best_step"] is not None and metrics["monitor_best"] is not None
    assert (out / "scalars.jsonl").read_text(encoding="utf-8").count("\n") == 10
    assert metrics["selection"] == {
        "checkpoint_selection": "final",
        "scored_step": 30,
        "ema": True,
        "ece_bins": 15,
    }
    assert metrics["checkpoints"]["best"] is None
    assert not (out / "ckpt" / "best.pt").exists()
    # The scored weights are last.pt's (its EMA, in eval mode): evaluate them again.
    seed_everything(settings.seed)
    task = build_paper_task(settings, {})
    task.module.load_state_dict(load_checkpoint(out / "ckpt" / "last.pt")["model"])
    source, _ = paper_inputs(settings, classifier_f32_dir)
    _, evals = _sources(settings, source(), torch.device("cpu"))
    again = evaluate(task, {"Test": evals["Test"]})["Test"]
    assert np.array_equal(again.probabilities.numpy(), _test_probabilities(out))
    assert metrics["evaluation"]["test"]["ece"] == pytest.approx(
        expected_calibration_error(again.probabilities.numpy(), again.targets.numpy())
    )


def test_final_selection_lets_the_validation_decide_nothing(
    classifier_f32_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(classifier_f32_dir, tmp_path / "run", max_steps=30, **_D3)
    expected = run_plain("train-supervised", settings)
    # Any validation values, here alternating, give the same training and the same scores.
    values = iter([5.0, 1.0] * 50)
    monkeypatch.setattr(_Fit, "monitored_value", lambda self: next(values))
    other = tmp_path / "other"
    metrics = run_plain("train-supervised", settings.replace(output_dir=other))
    assert metrics["monitor_best"] != expected["monitor_best"]  # the patch took effect
    assert metrics["evaluation"] == expected["evaluation"]
    assert metrics["last_step"] == expected["last_step"] == 30
    assert np.array_equal(_test_probabilities(other), _test_probabilities(tmp_path / "run"))


def test_best_val_loss_selection_keeps_choosing_on_validation(
    classifier_f32_dir: Path, tmp_path: Path
) -> None:
    # The same run but the default selection: it scores best_step's weights, so (with a
    # best before the end) other test predictions than the final weights'.
    final = run_plain(
        "train-supervised", _settings(classifier_f32_dir, tmp_path / "f", max_steps=30, **_D3)
    )
    best = run_plain(
        "train-supervised",
        _settings(
            classifier_f32_dir,
            tmp_path / "b",
            max_steps=30,
            **{**_D3, "checkpoint_selection": "best_val_loss"},
        ),
    )
    assert best["selection"]["scored_step"] == best["best_step"] == final["best_step"]
    assert (tmp_path / "b" / "ckpt" / "best.pt").is_file()
    if best["best_step"] != best["last_step"]:
        assert not np.array_equal(
            _test_probabilities(tmp_path / "b"), _test_probabilities(tmp_path / "f")
        )


def test_reports_show_the_scored_step_of_a_final_run(
    classifier_f32_dir: Path, tmp_path: Path
) -> None:
    # A final run's scores are the last step's weights: the replication table and the
    # registry must say so, not print the validation-best step next to them.
    root = tmp_path / "replication"
    out = root / "d3" / "seed-0"
    metrics = run_plain("train-supervised", _settings(classifier_f32_dir, out, max_steps=30, **_D3))
    assert metrics["best_step"] != metrics["last_step"] == 30  # they differ in this run
    (arm,) = find_arms(root)
    table = "\n".join(_replication_arm(arm))
    assert "| Seed | scored step | val acc (scored) | test (scored) |" in table
    assert "| 0 | 30 | " in table
    assert f"| 0 | {metrics['best_step']} | " not in table
    summary = metrics_summary(metrics)
    assert summary["scored_step"] == 30.0
    assert summary["best_step"] == float(metrics["best_step"])
