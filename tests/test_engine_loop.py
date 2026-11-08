"""Tests of the plain loop (task 4.1, ADR 0014) on a small toy task.

The toy task is a linear classifier on 4x4 "images" with an **active dropout** (it draws
from torch's global generator at every step, unlike the faithful paper model) and a
frozen submodule kept in ``eval``. That makes two checklist items observable: a resume is
exact only if every global RNG stream is checkpointed and restored last (C10), and a
validation restores each submodule's own mode instead of calling ``train()`` (C3). The
paper model on both engines is ``test_engine_differential.py``'s.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pytest
import torch
from _training_support import ACCELERATORS
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn

from astro_simclr.config import ConfigError, Settings
from astro_simclr.data.schema import DataError
from astro_simclr.engine import (
    CHECKPOINT_KEYS,
    Batch,
    Dataset,
    DevicePolicy,
    EarlyStop,
    EvalOutput,
    Horizon,
    IdentityState,
    Monitor,
    Schedule,
    StreamSpec,
    Task,
    TierChoice,
    TrainBatch,
    Validation,
    ValidationGrid,
    evaluate,
    evaluate_kept,
    fit,
    load_best,
    load_checkpoint,
    machine_record,
    open_eval,
    open_train,
    read_scalars,
    recorded,
    seed_everything,
)
from astro_simclr.engine.checkpoint import (
    BEST_KEYS,
    capture_rng,
    check_identity,
    check_numerics,
    numerics_record,
    restore_rng,
    save_checkpoint,
)
from astro_simclr.engine.precision import apply_fp32_precision, fp32_precision_record
from astro_simclr.engine.scalars import ScalarLog
from astro_simclr.engine.tiers import EvalBatches, TrainSource
from astro_simclr.runs.manifest import read_manifest
from astro_simclr.runs.record import start_run

_N = 40
_CPU = DevicePolicy("cpu", deterministic=True)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(16, 16)
        self.frozen.eval()  # a frozen encoder: must stay in eval across validations
        self.head = nn.Linear(16 + 2, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        h = self.frozen(images.flatten(1))
        out: torch.Tensor = self.head(torch.cat([self.dropout(h), features], dim=1))
        return out


class ToyTask:
    """An engine ``Task``: cross-entropy, Adam, and a step counter as task state."""

    def __init__(self) -> None:
        self.net = _Toy()
        self.trained_steps: list[int] = []
        self.seen: list[dict[str, Any]] = []

    @property
    def module(self) -> nn.Module:
        return self.net

    def make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.net.head.parameters(), lr=0.05)

    def train_step(self, batch: TrainBatch, step: int) -> torch.Tensor:
        self.trained_steps.append(step)
        images, features, one_hot = batch[0], batch[1], batch[2]
        logits = self.net(images, features)
        return nn.functional.cross_entropy(logits, one_hot.argmax(dim=1))

    def eval_step(self, batch: Batch) -> EvalOutput:
        images, features, one_hot = batch
        self.seen.append(
            {
                "training": self.net.training,
                "frozen_training": self.net.frozen.training,
                "grad_enabled": torch.is_grad_enabled(),
                "inference": torch.is_inference_mode_enabled(),
                "grads": [p.grad for p in self.net.parameters()],
            }
        )
        logits = self.net(images, features)
        targets = one_hot.argmax(dim=1)
        loss = nn.functional.cross_entropy(logits, targets)
        return EvalOutput(loss.detach(), logits.softmax(dim=1).detach(), targets)

    def state(self) -> dict[str, Any]:
        return {"trained": len(self.trained_steps)}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.restored = dict(state)


def _dataset(seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    return Dataset(
        images=rng.random((_N, 1, 4, 4), dtype=np.float32),
        features=rng.random((_N, 2), dtype=np.float32),
        labels=(np.arange(_N) % 3).astype(np.int64),
        n_classes=3,
    )


def _sources(device: str = "cpu") -> tuple[TrainSource, EvalBatches]:
    train = open_train(_dataset(0), TierChoice(), StreamSpec(batch_size=8, seed=3), device)
    return train, open_eval(_dataset(1), TierChoice(), device, batch_size=16)


def _schedule(**changes: Any) -> Schedule:
    values: dict[str, Any] = {
        "validation": ValidationGrid(4),
        "horizon": Horizon(10, 6),
        "checkpoint_every_n_steps": 4,
    }
    values.update(changes)
    return Schedule(**values)


def _fit(run_dir: Path | None, **kwargs: Any) -> tuple[ToyTask, Any]:
    schedule = kwargs.pop("schedule", _schedule())
    device = kwargs.pop("device", _CPU)
    train, validation = _sources(device.accelerator)
    task, result = fit(
        ToyTask,
        train,
        validation,
        schedule=schedule,
        device=device,
        seed=7,
        run_dir=run_dir,
        **kwargs,
    )
    assert isinstance(task, ToyTask)
    return task, result


def _states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


# --- the loop's checklist items -------------------------------------------------------


def test_grid_horizon_and_step_keys(tmp_path: Path) -> None:
    task, result = _fit(tmp_path)
    steps = [v.steps_done for v in result.validations]
    # Every 4 steps, never at the 5-step pass boundary, none forced after the last step.
    assert steps == list(range(4, result.last_step + 1, 4))
    assert result.batches_per_pass == 5
    assert result.last_step == result.horizon >= 10
    assert task.trained_steps == list(range(result.last_step))  # C6: steps done before
    assert result.horizon == max(10, result.best_step + 6)
    assert result.best_epoch == (result.best_step - 1) // 5
    assert result.last_epoch == (result.last_step - 1) // 5
    assert result.steps == result.last_step and result.resumed_from is None
    assert result.seconds_per_step is not None and result.seconds_per_step > 0
    improved = [v for v in result.validations if v.improved]
    assert improved[-1].steps_done == result.best_step
    assert result.best_score == improved[-1].value
    assert all(isinstance(v.train_loss, float) for v in result.validations)


def test_validation_mode_gradients_and_contexts() -> None:
    task, result = _fit(None)
    fit_seen = task.seen
    assert fit_seen
    for seen in fit_seen:
        assert seen["training"] is False and seen["frozen_training"] is False
        assert seen["grad_enabled"] is False and seen["inference"] is False  # no_grad (C3)
        assert all(grad is None for grad in seen["grads"])  # zero_grad before (C2)
    # The modes are restored per submodule, not by a blanket train().
    assert task.net.training is True and task.net.frozen.training is False
    task.seen.clear()
    _, validation = _sources()
    final = evaluate_kept(task, result, {"Validation": validation}, ("stop", "best"))
    assert all(seen["inference"] for seen in task.seen)  # the final evaluation (C3, C9)
    assert task.net.training is True and task.net.frozen.training is False
    assert final["best"]["Validation"].loss == result.best_score
    assert final["best"]["Validation"].probabilities.shape == (_N, 3)
    assert _states_equal(task.net.state_dict(), result.best_model)


def test_the_monitor_is_strict_and_extends_at_the_horizon() -> None:
    # A new best at the horizon step extends the run (the done check follows the update).
    _, result = _fit(None, schedule=_schedule(horizon=Horizon(4, 4), validation=ValidationGrid(4)))
    first = result.validations[0]
    assert first.steps_done == 4 and first.improved and first.horizon == 8
    assert result.last_step > 4


def test_increment_zero_is_a_fixed_cap_and_off_grid_end() -> None:
    _, result = _fit(None, schedule=_schedule(horizon=Horizon(10, 0)))
    assert result.last_step == result.horizon == 10
    assert [v.steps_done for v in result.validations] == [4, 8]  # nothing forced at 10


def test_keep_stop_only_keeps_no_best_weights(tmp_path: Path) -> None:
    task, result = _fit(tmp_path, schedule=_schedule(keep=("stop",)))
    assert result.best_model is None and result.best_step is not None
    assert not (tmp_path / "ckpt" / "best.pt").exists()
    last = {k: v.clone() for k, v in task.net.state_dict().items()}
    _, validation = _sources()
    final = evaluate_kept(task, result, {"Validation": validation}, ("stop",))
    assert set(final) == {"stop"}
    assert _states_equal(task.net.state_dict(), last)
    with pytest.raises(DataError, match="no best weights"):
        evaluate_kept(task, result, {"Validation": validation}, ("best",))


def test_no_validation_means_no_best(tmp_path: Path) -> None:
    _, result = _fit(
        tmp_path,
        schedule=_schedule(
            validation=ValidationGrid(20), horizon=Horizon(5, 0), checkpoint_every_n_steps=20
        ),
    )
    assert result.validations == () and result.best_step is None
    assert not (tmp_path / "ckpt" / "best.pt").exists()
    assert load_checkpoint(tmp_path / "ckpt" / "last.pt")["steps_done"] == 5


def test_only_loss_val_is_monitored_without_a_score() -> None:
    with pytest.raises(ConfigError, match="computes loss_val only"):
        _fit(None, schedule=_schedule(monitor=Monitor("accuracy_val", "max")))


def test_evaluate_refuses_an_empty_source() -> None:
    class _Empty:
        def __iter__(self) -> Any:
            return iter(())

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="no validation batch"):
        evaluate(ToyTask(), {"x": _Empty()})


# --- early stopping: Lightning's rule ---------------------------------------------------


_SEQUENCES = [
    [3.0, 2.0, 2.0, 2.5, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.95, 0.94, 0.80, 0.79, 0.78, 0.77],
    [1.0, math.nan, 0.5],
    [1.0, 2.0, 50.0, 0.1],
    [math.inf, 1.0],
]


@pytest.mark.parametrize("values", _SEQUENCES)
@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize(
    "rule",
    [
        EarlyStop(patience=2),
        EarlyStop(patience=1, min_delta=0.05),
        EarlyStop(patience=3, check_finite=False, divergence_threshold=10.0),
    ],
)
def test_early_stop_is_lightning_s(values: list[float], mode: str, rule: EarlyStop) -> None:
    reference = EarlyStopping(
        "loss_val",
        min_delta=rule.min_delta,
        patience=rule.patience,
        mode=mode,
        check_finite=rule.check_finite,
        divergence_threshold=rule.divergence_threshold,
    )
    best, wait = None, 0
    for value in values:
        current = torch.tensor(value, dtype=torch.float32)
        expected_stop, _ = reference._evaluate_stopping_criteria(current)
        best, wait, stop = rule.update(float(current), best, wait, mode)  # type: ignore[arg-type]
        assert (stop, wait) == (expected_stop, reference.wait_count), value
        initial = math.inf if mode == "min" else -math.inf
        assert (initial if best is None else best) == float(reference.best_score)
        if stop:
            break


def test_early_stop_ends_the_run(tmp_path: Path) -> None:
    schedule = _schedule(horizon=Horizon(10_000, 0), early_stop=EarlyStop(patience=1))
    _, result = _fit(tmp_path, schedule=schedule)
    assert result.stopped_early
    assert result.validations[-1].wait_count == 1 and not result.validations[-1].improved
    assert result.last_step == result.validations[-1].steps_done < 10_000
    assert load_checkpoint(tmp_path / "ckpt" / "last.pt")["steps_done"] == result.last_step


@dataclass(frozen=True)
class _Stop:
    stop: bool


class _ScriptedRule:
    """A stop rule that stops once the history holds ``length`` values (any ``StopRule``)."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.calls: list[tuple[int, tuple[float, ...]]] = []

    def decide(self, step: int, history: Sequence[float]) -> _Stop:
        self.calls.append((step, tuple(history)))
        return _Stop(len(history) >= self.length)


def _accuracy(outputs: Sequence[EvalOutput]) -> float:
    probabilities = torch.cat([o.probabilities for o in outputs])
    targets = torch.cat([o.targets for o in outputs])
    return float((probabilities.argmax(dim=1) == targets).float().mean())


def _rj23_like(rule: _ScriptedRule, **changes: Any) -> Schedule:
    values: dict[str, Any] = {
        "monitor": Monitor("accuracy_val", "max"),
        "validation": ValidationGrid(4, at_start=True),
        "horizon": Horizon(40, 0),
        "stop_rule": rule,
        "keep": ("stop", "best"),
        "checkpoint_every_n_steps": 4,
    }
    values.update(changes)
    return Schedule(**values)


def test_at_start_score_and_stop_rule(tmp_path: Path) -> None:
    rule = _ScriptedRule(3)
    task, result = _fit(tmp_path, schedule=_rj23_like(rule), score=_accuracy)
    # Step 0 (before any step), then the grid; the rule stopped at the third value.
    assert [v.steps_done for v in result.validations] == [0, 4, 8]
    assert result.validations[0].train_loss is None and result.validations[0].epoch == -1
    assert result.stopped_early and result.last_step == 8
    assert [step for step, _ in rule.calls] == [0, 4, 8]
    assert rule.calls[-1][1] == result.history == tuple(v.value for v in result.validations)
    assert all(0.0 <= value <= 1.0 for value in result.history)  # the score, not a loss
    assert task.trained_steps == list(range(8))
    # last.pt after the step-4 and step-8 validations (never at step 0), with the history.
    last = load_checkpoint(tmp_path / "ckpt" / "last.pt")
    assert last["steps_done"] == 8 and last["policy"]["history"] == list(result.history)
    records = read_scalars(tmp_path / "scalars.jsonl")
    assert [r["step"] for r in records] == [0, 4, 8]
    assert all("accuracy_val" in r and "loss_val" not in r for r in records)


def test_a_resumed_run_that_had_stopped_trains_nothing(tmp_path: Path) -> None:
    _, first = _fit(tmp_path, schedule=_rj23_like(_ScriptedRule(3)), score=_accuracy)
    rule = _ScriptedRule(3)
    _, again = _fit(
        tmp_path,
        schedule=_rj23_like(rule),
        score=_accuracy,
        resume=tmp_path / "ckpt" / "last.pt",
    )
    assert again.steps == 0 and again.validations == () and again.stopped_early
    assert again.history == first.history and again.best_step == first.best_step
    assert rule.calls == [(8, first.history)]  # decided once, on the restored history


def test_a_mid_run_resume_keeps_the_history(tmp_path: Path) -> None:
    whole, part = tmp_path / "whole", tmp_path / "part"
    _, expected = _fit(whole, schedule=_rj23_like(_ScriptedRule(4)), score=_accuracy)
    _fit(part, schedule=_rj23_like(_ScriptedRule(4), horizon=Horizon(4, 0)), score=_accuracy)
    # A step-4 checkpoint of another horizon: the run id is not checked without one.
    _, resumed = _fit(
        part,
        schedule=_rj23_like(_ScriptedRule(4)),
        score=_accuracy,
        resume=part / "ckpt" / "last.pt",
    )
    assert resumed.history == expected.history
    assert [v.steps_done for v in resumed.validations] == [8, 12]


def test_an_off_grid_horizon_ends_a_stop_rule_run(tmp_path: Path) -> None:
    # The last checkpoint at step 10 is off the grid (every 4): the rule is not asked there.
    schedule = _rj23_like(_ScriptedRule(99), horizon=Horizon(10, 0))
    _, first = _fit(tmp_path, schedule=schedule, score=_accuracy)
    assert first.last_step == 10 and not first.stopped_early
    rule = _ScriptedRule(99)
    _, again = _fit(
        tmp_path,
        schedule=_rj23_like(rule, horizon=Horizon(10, 0)),
        score=_accuracy,
        resume=tmp_path / "ckpt" / "last.pt",
    )
    assert again.steps == 0 and not again.stopped_early and rule.calls == []


def test_the_carrasco2021_checkpoint_has_no_history(tmp_path: Path) -> None:
    _, result = _fit(tmp_path)
    assert "history" not in load_checkpoint(tmp_path / "ckpt" / "last.pt")["policy"]
    assert result.history == ()


def test_a_schedule_takes_one_stop_rule() -> None:
    with pytest.raises(ConfigError, match="early_stop or stop_rule"):
        _rj23_like(_ScriptedRule(2), early_stop=EarlyStop(patience=1))


# --- checkpoints and exact resume -------------------------------------------------------


def test_checkpoints_are_plain_values(tmp_path: Path) -> None:
    _, result = _fit(tmp_path, identity={"run_id": "a" * 64, "attempt": 2, "split_run": None})
    last = load_checkpoint(tmp_path / "ckpt" / "last.pt")
    assert set(last) == CHECKPOINT_KEYS
    assert last["engine"] == "plain" and last["steps_done"] == result.last_step
    assert last["identity"]["attempt"] == 2
    assert last["task"] == {"trained": result.last_step}
    assert last["policy"]["best_step"] == result.best_step
    assert last["stream"]["batch_size"] == 8 and last["stream"]["seed"] == 3
    assert last["numerics"]["fp32_precision"] == "ieee" and last["numerics"]["deterministic"]
    assert set(last["versions"]) == {"astro_simclr", "torch"}
    best = load_best(tmp_path / "ckpt" / "best.pt")
    assert set(best) == BEST_KEYS
    assert best["steps_done"] == result.best_step and best["score"] == result.best_score
    assert _states_equal(best["model"], result.best_model)
    with pytest.raises(ConfigError, match="best weights only"):
        load_checkpoint(tmp_path / "ckpt" / "best.pt")
    with pytest.raises(ConfigError, match="not a plain engine's best"):
        load_best(tmp_path / "ckpt" / "last.pt")
    torch.save([1, 2], tmp_path / "list.pt")
    with pytest.raises(ConfigError, match="not a checkpoint"):
        load_checkpoint(tmp_path / "list.pt")
    torch.save({**best, "format_version": 99}, tmp_path / "old.pt")
    with pytest.raises(ConfigError, match="another format"):
        load_best(tmp_path / "old.pt")
    assert not list((tmp_path / "ckpt").glob(".*partial"))  # the atomic writes cleaned up


class _Keep:
    """Copy each validation's weights, and ``last.pt`` at ``step``."""

    def __init__(self, step: int | None = None, src: Path | None = None, dst: Path | None = None):
        self.step, self.src, self.dst = step, src, dst
        self.states: list[tuple[int, float, dict[str, torch.Tensor]]] = []

    def __call__(self, record: Validation, task: Task) -> None:
        state = {k: v.clone() for k, v in task.module.state_dict().items()}
        self.states.append((record.steps_done, record.value, state))
        if record.steps_done == self.step and self.src is not None and self.dst is not None:
            self.dst.write_bytes(self.src.read_bytes())


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_resume_is_exact_with_global_draws(accelerator: str, tmp_path: Path) -> None:
    # Dropout draws from the device's global generator at every step (CPU, CUDA or MPS):
    # only a checkpoint with the RNG streams, restored last, continues with the same draws.
    device = DevicePolicy(accelerator, deterministic=True)  # type: ignore[arg-type]
    whole = tmp_path / "whole"
    saved = tmp_path / "step8.pt"
    keep = _Keep(8, whole / "ckpt" / "last.pt", saved)
    _, expected = _fit(whole, on_validation=keep, device=device)
    resumed_keep = _Keep()
    task, result = _fit(
        tmp_path / "resumed", resume=saved, on_validation=resumed_keep, device=device
    )
    assert task.restored == {"trained": 8}
    assert task.trained_steps == list(range(8, expected.last_step))
    later = [entry for entry in keep.states if entry[0] > 8]
    assert [(s, v) for s, v, _ in resumed_keep.states] == [(s, v) for s, v, _ in later]
    assert all(_states_equal(a[2], b[2]) for a, b in zip(later, resumed_keep.states, strict=True))
    assert (result.best_step, result.horizon, result.last_step) == (
        expected.best_step,
        expected.horizon,
        expected.last_step,
    )
    assert _states_equal(result.best_model, expected.best_model)


def test_resume_without_the_rng_streams_is_not_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The control of the test above: skipping the RNG restore changes the dropout masks.
    whole = tmp_path / "whole"
    saved = tmp_path / "step8.pt"
    keep = _Keep(8, whole / "ckpt" / "last.pt", saved)
    _fit(whole, on_validation=keep)
    monkeypatch.setattr("astro_simclr.engine.loop.restore_rng", lambda state: None)
    resumed = _Keep()
    _fit(tmp_path / "resumed", resume=saved, on_validation=resumed)
    later = [entry for entry in keep.states if entry[0] > 8]
    assert not _states_equal(later[0][2], resumed.states[0][2])


def test_resume_refusals(tmp_path: Path) -> None:
    identity: IdentityState = {"run_id": "b" * 64, "attempt": 1, "split_run": None}
    _fit(tmp_path, identity=identity)
    last = tmp_path / "ckpt" / "last.pt"
    with pytest.raises(ConfigError, match="run_id"):
        _fit(None, resume=last, identity={**identity, "run_id": "c" * 64})
    with pytest.raises(ConfigError, match="split_run"):
        _fit(None, resume=last, identity={**identity, "split_run": {"fold": 1}})
    # Another attempt of the same run resumes (a resume after a failure).
    _fit(None, resume=last, identity={**identity, "attempt": 2})
    train, validation = _sources()
    with pytest.raises(ConfigError, match="precision and determinism"):
        fit(
            ToyTask,
            train,
            validation,
            schedule=_schedule(),
            device=DevicePolicy("cpu"),
            seed=7,
            identity=identity,
            resume=last,
        )
    with pytest.raises(ConfigError, match="same monitor"):
        _fit(
            None,
            resume=last,
            identity=identity,
            schedule=_schedule(monitor=Monitor("loss_val", "max")),
        )
    other = open_train(_dataset(0), TierChoice(), StreamSpec(batch_size=8, seed=4), "cpu")
    with pytest.raises(DataError, match="stream"):
        fit(
            ToyTask,
            other,
            validation,
            schedule=_schedule(),
            device=_CPU,
            seed=7,
            identity=identity,
            resume=last,
        )
    lightning = tmp_path / "lightning.ckpt"
    torch.save({"pytorch-lightning_version": L.__version__, "state_dict": {}}, lightning)
    with pytest.raises(ConfigError, match="Lightning checkpoint"):
        _fit(None, resume=lightning)


def test_a_larger_horizon_extends_a_resumed_run(tmp_path: Path) -> None:
    _, first = _fit(tmp_path)
    last = tmp_path / "ckpt" / "last.pt"
    _, longer = _fit(None, resume=last, schedule=_schedule(horizon=Horizon(first.last_step + 8, 6)))
    assert longer.last_step >= first.last_step + 8
    assert longer.steps == longer.last_step - first.last_step


def test_identity_and_numerics_checks() -> None:
    current = {"run_id": None, "attempt": None, "split_run": None}
    check_identity({"run_id": None, "split_run": None, "attempt": 3}, current)  # type: ignore[arg-type]
    record = numerics_record(_CPU)
    check_numerics({**record, "fp32_backends": {}}, record)  # only the policy is compared
    with pytest.raises(ConfigError, match="deterministic"):
        check_numerics({**record, "deterministic": False}, record)


def test_rng_capture_and_restore() -> None:
    state = capture_rng(_CPU)
    assert state["torch_cuda"] is None and state["torch_mps"] is None
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    expected = (random.random(), np.random.random(), torch.rand(1))  # noqa: S311
    restore_rng(loaded)
    assert (random.random(), np.random.random()) == expected[:2]  # noqa: S311
    assert torch.equal(torch.rand(1), expected[2])
    bad = {**state, "torch_cuda": [torch.zeros(1)] * (torch.cuda.device_count() + 1)}
    with pytest.raises(ConfigError, match="CUDA generator states"):
        restore_rng(bad)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_rng_capture_on_mps() -> None:
    state = capture_rng(DevicePolicy("mps"))
    assert state["torch_mps"] is not None
    restore_rng(state)


def test_save_checkpoint_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ckpt" / "last.pt"
    save_checkpoint(path, {"x": 1})  # type: ignore[arg-type]

    def broken(obj: object, target: Path) -> None:
        Path(target).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, {"x": 2})  # type: ignore[arg-type]
    monkeypatch.undo()
    assert torch.load(path, weights_only=True) == {"x": 1}


# --- seeding, flags, precision ---------------------------------------------------------


def test_seed_everything_is_lightning_s() -> None:
    L.seed_everything(123, verbose=False)
    lightning = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    assert seed_everything(123) == 123
    assert random.getstate() == lightning[0]
    assert np.array_equal(np.random.get_state()[1], lightning[1])
    assert torch.equal(torch.get_rng_state(), lightning[2])
    for bad in (-1, 2**32, True):
        with pytest.raises(ValueError, match="seed must be"):
            seed_everything(bad)


@pytest.mark.parametrize("deterministic", [True, False])
def test_device_policy_applies_its_flags(
    deterministic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    saved = (torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    try:
        policy = DevicePolicy("cpu", deterministic=deterministic, fp32_precision="tf32")
        policy.apply()
        assert torch.backends.cudnn.benchmark is (not deterministic)
        assert torch.are_deterministic_algorithms_enabled() is deterministic
        assert torch.get_float32_matmul_precision() == "high"
        expected = ":4096:8" if deterministic else None
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == expected
    finally:
        apply_fp32_precision("ieee")
        torch.backends.cudnn.benchmark = saved[0]
        torch.use_deterministic_algorithms(saved[1])
    assert policy.device == torch.device("cpu")


def test_precision_refuses_unknown_values() -> None:
    with pytest.raises(ConfigError, match="fp32_precision"):
        apply_fp32_precision("bf16")
    assert fp32_precision_record(torch.device("cpu"))["fp32_precision"] == "ieee"


# --- local scalars, progress, the manifest ------------------------------------------------


def test_scalars_and_no_progress_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, result = _fit(tmp_path)
    assert capsys.readouterr() == ("", "")  # no progress bar, no prints
    records = read_scalars(tmp_path / "scalars.jsonl")
    assert [r["step"] for r in records] == [v.steps_done for v in result.validations]
    first = records[0]
    assert set(first) == {
        "step",
        "epoch",
        "loss_val",
        "loss_train",
        "improved",
        "horizon",
        "wait_count",
        "seconds",
    }
    assert first["loss_val"] == result.validations[0].value
    stream = io.StringIO()
    _fit(None, progress=stream)
    lines = stream.getvalue().splitlines()
    assert len(lines) == len(result.validations) and lines[0].startswith("step 4 epoch 0 loss_val")


def test_scalar_log_resume_and_nan(tmp_path: Path) -> None:
    path = tmp_path / "scalars.jsonl"
    with ScalarLog.open(path) as log:
        for step in (10, 20, 30):
            log.write({"step": step, "loss_val": math.nan if step == 30 else 1.0})
    assert read_scalars(path)[-1]["loss_val"] is None
    with ScalarLog.open(path, resume_from=20) as log:
        log.write({"step": 30, "loss_val": 0.5})
    assert [(r["step"], r["loss_val"]) for r in read_scalars(path)] == [
        (10, 1.0),
        (20, 1.0),
        (30, 0.5),
    ]
    with ScalarLog.open(path):
        pass
    assert read_scalars(path) == []
    path.write_text('{"loss_val": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="integer step"):
        ScalarLog.open(path, resume_from=0)
    path.write_text("[1]\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        read_scalars(path)
    assert read_scalars(tmp_path / "missing.jsonl") == []


def test_manifest_of_a_plain_run(classifier_f32_dir: Path, tmp_path: Path) -> None:
    # The manifest's lifecycle around a plain fit: engine plain, running, then finished.
    settings = Settings(
        accelerator="cpu",
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=tmp_path,
    )
    machine = machine_record(torch.device("cpu"))
    assert machine["engine"] == "plain" and machine["fp32_precision"] == "ieee"
    record = start_run(
        "train-supervised", settings, data=classifier_f32_dir, machine=machine, engine="plain"
    )
    assert read_manifest(tmp_path).status == "running"
    identity = {
        "run_id": record.manifest.run_id,
        "attempt": record.manifest.attempt,
        "split_run": None,
    }
    with recorded(record):
        _fit(tmp_path, identity=identity)
    manifest = read_manifest(tmp_path)
    assert manifest.status == "finished" and manifest.engine == "plain"
    assert manifest.identity.engine == "plain"
    assert load_checkpoint(tmp_path / "ckpt" / "last.pt")["identity"]["run_id"] == manifest.run_id
    assert json.loads((tmp_path / "manifest.json").read_text())["machine"]["engine"] == "plain"


@pytest.mark.parametrize(
    ("error", "status"), [(RuntimeError, "failed"), (KeyboardInterrupt, "cancelled")]
)
def test_recorded_ends_failed_or_cancelled(
    error: type[BaseException], status: str, classifier_f32_dir: Path, tmp_path: Path
) -> None:
    settings = Settings(
        accelerator="cpu",
        data_dir=classifier_f32_dir.parent,
        dataset=classifier_f32_dir.name,
        output_dir=tmp_path,
    )
    record = start_run(
        "train-supervised",
        settings,
        data=classifier_f32_dir,
        machine=machine_record(torch.device("cpu")),
        engine="plain",
    )
    with pytest.raises(error), recorded(record):
        raise error
    assert read_manifest(tmp_path).status == status
