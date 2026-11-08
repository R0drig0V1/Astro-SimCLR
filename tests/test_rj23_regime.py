"""Tests of the engine-free RJ23 regime (astro_simclr.rj23_regime, task 3.4b).

The schedule policy, the class-uniform sampler, Keras' Adam (against a hand-computed step)
and the shared evaluation, with no training engine.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from astro_simclr.engine.randomness import RESERVED_STREAMS, host_generator
from astro_simclr.rj23_regime import (
    KEEPS,
    KERAS_ADAM_BETAS,
    KERAS_ADAM_EPS,
    RJ23_BATCH_SIZE,
    RJ23_EVAL_EVERY,
    RJ23_MAX_STEPS,
    RJ23_PATIENCE,
    STREAM,
    ClassUniformStream,
    KerasAdam,
    RJ23Schedule,
    keras_adam,
    predict_probabilities,
)


def test_engine_free_modules_import_no_lightning() -> None:
    code = (
        "import sys, astro_simclr.rj23_regime, astro_simclr.rj23_data; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('lightning', "
        "'pytorch_lightning')); assert not bad, bad"
    )
    # A fresh interpreter: this process may have imported Lightning already.
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 (fixed argv)


# --- The schedule policy --------------------------------------------------------------------


def test_defaults_are_rj23s() -> None:
    schedule = RJ23Schedule()
    assert (schedule.eval_every, schedule.patience, schedule.max_steps, schedule.mode) == (
        RJ23_EVAL_EVERY,
        RJ23_PATIENCE,
        RJ23_MAX_STEPS,
        "max",
    )
    assert (RJ23_EVAL_EVERY, RJ23_PATIENCE, RJ23_MAX_STEPS, RJ23_BATCH_SIZE) == (
        500,
        5,
        100_000,
        256,
    )
    assert KEEPS == ("stop", "best")


def test_validation_grid_includes_step_zero() -> None:
    schedule = RJ23Schedule()
    assert [s for s in range(0, 2001) if schedule.validates(s)] == [0, 500, 1000, 1500, 2000]
    assert not schedule.validates(-500)
    assert schedule.validation_step(3) == 1500


def test_step_zero_is_the_first_best() -> None:
    decision = RJ23Schedule().decide(0, [0.4])
    assert decision.validate and not decision.stop
    assert decision.keep == {"best"}
    assert (decision.best_step, decision.best_value, decision.wait) == (0, 0.4, 0)


def test_stop_at_patience_on_strict_improvement() -> None:
    schedule = RJ23Schedule(eval_every=500, patience=5)
    # Best at step 500; an equal value is no improvement; five validations without one.
    history = [0.1, 0.2, 0.2, 0.15, 0.2, 0.19, 0.2]
    for k in range(1, len(history)):
        decision = schedule.decide(500 * k, history[: k + 1])
        assert decision.stop == (k == len(history) - 1)
        assert decision.wait == k - 1
    assert decision.reason == "patience" and not decision.capped
    assert decision.keep == {"stop"}
    assert (decision.best_step, decision.best_value) == (500, 0.2)
    assert decision.step == 3000


def test_new_best_resets_the_wait() -> None:
    schedule = RJ23Schedule(eval_every=10, patience=2)
    assert schedule.decide(20, [0.5, 0.4, 0.3]).stop  # two validations since the best
    decision = schedule.decide(30, [0.5, 0.4, 0.6, 0.55])
    assert not decision.stop and decision.wait == 1 and decision.best_step == 20


def test_between_validations_nothing_happens() -> None:
    schedule = RJ23Schedule(eval_every=500, patience=1)
    decision = schedule.decide(501, [0.3, 0.4])
    assert not decision.validate and not decision.stop and decision.keep == frozenset()
    assert decision.reason is None


def test_cap_stops_and_is_flagged() -> None:
    schedule = RJ23Schedule(eval_every=500, patience=5, max_steps=2000)
    history = [0.1, 0.2, 0.3, 0.4, 0.5]  # improving to the end
    decision = schedule.decide(2000, history)
    assert decision.stop and decision.capped and decision.reason == "cap"
    assert decision.keep == {"stop", "best"}  # the cap step is also the best
    assert not schedule.decide(1500, history[:4]).stop


def test_patience_at_the_cap_is_patience() -> None:
    schedule = RJ23Schedule(eval_every=1, patience=2, max_steps=2)
    decision = schedule.decide(2, [0.5, 0.4, 0.3])
    assert decision.stop and decision.reason == "patience" and not decision.capped


def test_nan_never_improves() -> None:
    schedule = RJ23Schedule(eval_every=1, patience=2)
    decision = schedule.decide(1, [math.nan, 0.1])
    assert decision.best_step == 1 and decision.wait == 0
    assert schedule.decide(2, [0.1, math.nan, math.nan]).stop
    first = schedule.decide(0, [math.nan])
    assert first.best_step is None and first.best_value is None and first.keep == frozenset()


def test_min_mode() -> None:
    decision = RJ23Schedule(eval_every=1, patience=3, mode="min").decide(2, [1.0, 0.5, 0.7])
    assert decision.best_step == 1 and decision.wait == 1


def test_decide_is_pure() -> None:
    schedule = RJ23Schedule(eval_every=2, patience=2)
    history = [0.1, 0.3, 0.2]
    assert schedule.decide(4, history) == schedule.decide(4, list(history))


@pytest.mark.parametrize(
    ("step", "history", "message"),
    [
        (-1, [0.1], "step must be"),
        (100_500, [0.1] * 202, "step must be"),
        (500, [0.1], "must hold 2 validations"),
        (0, [], "must hold 1 validations"),
        (1500, [0.5, 0.4, 0.3, 0.2], "goes on after the run stopped"),
    ],
)
def test_decide_refuses_an_inconsistent_history(
    step: int, history: list[float], message: str
) -> None:
    schedule = RJ23Schedule(patience=2)
    with pytest.raises(ValueError, match=message):
        schedule.decide(step, history)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eval_every": 0}, "eval_every"),
        ({"patience": 0}, "patience"),
        ({"max_steps": True}, "max_steps"),
        ({"max_steps": 1250}, "multiple of eval_every"),
        ({"mode": "up"}, "mode"),
    ],
)
def test_schedule_refuses_bad_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RJ23Schedule(**kwargs)  # type: ignore[arg-type]


# --- The class-uniform sampler ------------------------------------------------------------


def _imbalanced() -> np.ndarray:
    # Three classes of very different sizes, interleaved: 600, 30 and 3 items.
    labels = np.concatenate([np.zeros(600), np.ones(30), np.full(3, 2)]).astype(np.int64)
    return np.random.default_rng(1).permutation(labels)


def test_classes_are_drawn_uniformly() -> None:
    labels = _imbalanced()
    stream = ClassUniformStream(labels, batch_size=256, seed=0)
    drawn = np.concatenate([stream.batch(b) for b in range(200)])  # 51,200 draws
    shares = np.bincount(labels[drawn], minlength=3) / drawn.size
    # Binomial sd of a 1/3 share over 51,200 draws is 0.0021: 5 sd.
    np.testing.assert_allclose(shares, 1 / 3, atol=0.011)


def test_items_are_uniform_within_their_class() -> None:
    labels = _imbalanced()
    stream = ClassUniformStream(labels, batch_size=256, seed=3)
    drawn = np.concatenate([stream.batch(b) for b in range(200)])
    members = np.flatnonzero(labels == 1)
    counts = np.array([np.sum(drawn == m) for m in members])
    # Each of the 30 items of class 1: about 51,200 / 3 / 30 = 569 draws (sd ~23).
    assert counts.min() > 450 and counts.max() < 690
    assert set(np.unique(drawn)) <= set(range(labels.size))
    assert set(np.flatnonzero(labels == 2)) <= set(drawn.tolist())


def test_batches_are_pure_functions_of_seed_and_step() -> None:
    labels = _imbalanced()
    stream = ClassUniformStream(labels, batch_size=64, seed=7)
    again = ClassUniformStream(labels.copy(), batch_size=64, seed=7)
    other = ClassUniformStream(labels, batch_size=64, seed=8)
    np.testing.assert_array_equal(stream.batch(5), again.batch(5))
    assert not np.array_equal(stream.batch(5), other.batch(5))
    assert not np.array_equal(stream.batch(5), stream.batch(6))
    assert stream.batch(5).dtype == np.int64 and stream.batch(5).shape == (64,)


def test_resume_from_a_stored_position_sees_the_same_batches() -> None:
    stream = ClassUniformStream(_imbalanced(), batch_size=32, seed=2)
    uninterrupted = [b for _, b in zip(range(10), stream.iterate(), strict=False)]
    position = 6  # what a checkpoint stores: the steps done
    resumed = [b for _, b in zip(range(4), stream.iterate(position), strict=False)]
    for left, right in zip(uninterrupted[position:], resumed, strict=True):
        np.testing.assert_array_equal(left, right)


def test_batches_come_from_the_reserved_host_generator() -> None:
    labels = np.array([0, 0, 0, 1, 2, 2])
    stream = ClassUniformStream(labels, batch_size=5, seed=4)
    assert STREAM == RESERVED_STREAMS["rj23_class_uniform"] == 10
    rng = host_generator(4, 9, STREAM)
    classes = rng.integers(0, 3, size=5)
    within = rng.integers(0, np.bincount(labels)[classes])
    members = {0: [0, 1, 2], 1: [3], 2: [4, 5]}
    expected = [members[int(c)][int(k)] for c, k in zip(classes, within, strict=True)]
    assert stream.batch(9).tolist() == expected


def test_stream_state_and_pass_length() -> None:
    stream = ClassUniformStream(np.array([0, 1, 1, 2, 2, 2]), batch_size=4, seed=1)
    assert stream.state() == {
        "sampler": "class_uniform",
        "n": 6,
        "class_counts": [1, 2, 3],
        "batch_size": 4,
        "seed": 1,
    }
    assert stream.batches_per_pass == 2
    assert stream.n_classes == 3


@pytest.mark.parametrize(
    ("labels", "kwargs", "message"),
    [
        (np.array([], dtype=np.int64), {}, "non-empty 1-D integer"),
        (np.array([[0, 1]]), {}, "non-empty 1-D integer"),
        (np.array([0.0, 1.0]), {}, "non-empty 1-D integer"),
        (np.array([0, 3]), {"n_classes": 3}, r"range\(3\)"),
        (np.array([-1, 0]), {}, r"range\(1\)"),
        (np.array([0, 2]), {}, r"classes \[1\] have no item"),
        (np.array([0, 1]), {"batch_size": 0}, "batch_size"),
        (np.array([0, 1]), {"seed": -1}, "seed"),
    ],
)
def test_stream_refuses_bad_input(labels: np.ndarray, kwargs: dict[str, int], message: str) -> None:
    args = {"batch_size": 2, "seed": 0, **kwargs}
    with pytest.raises(ValueError, match=message):
        ClassUniformStream(labels, **args)


def test_stream_refuses_a_negative_batch() -> None:
    with pytest.raises(ValueError, match="batch index"):
        ClassUniformStream(np.array([0, 1]), 2, 0).batch(-1)


# --- Keras Adam ------------------------------------------------------------------------------


def _scalar_run() -> list[float]:
    theta = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = keras_adam([theta], lr=0.1)
    values = []
    for grad in (0.5, -0.25):
        theta.grad = torch.tensor([grad], dtype=torch.float64)
        optimizer.step()
        values.append(float(theta.detach()))
    return values


def test_keras_adam_matches_a_hand_computed_step() -> None:
    # theta0 = 1, lr = 0.1, gradients 0.5 then -0.25, Keras defaults:
    # t=1: m = 0.05, v = 2.5e-4, lr_t = 0.1 sqrt(1 - 0.999) / (1 - 0.9) = 0.0316227766,
    #      theta1 = 1 - 0.0316227766 * 0.05 / (sqrt(2.5e-4) + 1e-7) = 0.9000006324515
    # t=2: m = 0.02, v = 3.1225e-4, lr_t = 0.1 sqrt(1 - 0.999^2) / (1 - 0.9^2) = 0.0235316725,
    #      theta2 = theta1 - 0.0235316725 * 0.02 / (sqrt(3.1225e-4) + 1e-7) = 0.8733670792
    first, second = _scalar_run()
    assert first == pytest.approx(0.9000006324515321, abs=1e-13)
    assert second == pytest.approx(0.8733670792079219, abs=1e-13)


def test_keras_epsilon_placement_differs_from_torchs() -> None:
    # torch adds eps to sqrt(v_hat): theta1 = 1 - 0.1 * 0.5 / (0.5 + 1e-7) = 0.90000002.
    theta = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    torch_adam = torch.optim.Adam([theta], lr=0.1, betas=KERAS_ADAM_BETAS, eps=KERAS_ADAM_EPS)
    theta.grad = torch.tensor([0.5], dtype=torch.float64)
    torch_adam.step()
    assert float(theta.detach()) == pytest.approx(1 - 0.05 / (0.5 + 1e-7), abs=1e-13)
    assert abs(float(theta.detach()) - _scalar_run()[0]) > 5e-7


def test_keras_adam_equals_torch_adam_without_epsilon() -> None:
    torch.manual_seed(0)
    start = torch.randn(4, 3, dtype=torch.float64)
    ours = nn.Parameter(start.clone())
    theirs = nn.Parameter(start.clone())
    keras = KerasAdam([ours], lr=0.01, eps=0.0)
    reference = torch.optim.Adam([theirs], lr=0.01, betas=KERAS_ADAM_BETAS, eps=0.0)
    for _ in range(5):
        grad = torch.randn(4, 3, dtype=torch.float64)
        ours.grad, theirs.grad = grad.clone(), grad.clone()
        keras.step()
        reference.step()
    torch.testing.assert_close(ours, theirs, rtol=1e-12, atol=1e-12)


def test_keras_adam_defaults_and_state() -> None:
    weight = nn.Parameter(torch.zeros(2))
    unused = nn.Parameter(torch.zeros(2))
    optimizer = keras_adam([weight, unused], lr=1e-3)
    group = optimizer.param_groups[0]
    assert (group["betas"], group["eps"], group["lr"]) == ((0.9, 0.999), 1e-7, 1e-3)
    weight.grad = torch.ones(2)
    calls: list[int] = []

    def closure() -> float:
        calls.append(1)
        return 2.0

    assert optimizer.step(closure) == 2.0
    assert calls == [1]
    assert optimizer.state[weight]["step"] == 1
    assert unused not in optimizer.state  # no gradient: untouched
    assert torch.all(unused == 0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"lr": 0.0}, "lr"), ({"betas": (0.9, 1.0)}, "betas"), ({"eps": -1.0}, "eps")],
)
def test_keras_adam_refuses_bad_hyperparameters(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        KerasAdam([nn.Parameter(torch.zeros(1))], **kwargs)  # type: ignore[arg-type]


def test_keras_adam_refuses_sparse_gradients() -> None:
    weight = nn.Parameter(torch.zeros(3))
    weight.grad = torch.zeros(3).to_sparse()
    with pytest.raises(ValueError, match="sparse"):
        keras_adam([weight], lr=1e-3).step()


# --- Evaluation ------------------------------------------------------------------------------


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc(self.dropout(images.flatten(1)[:, :1] + positions))
        return out


def test_predict_probabilities_is_eval_mode_and_batch_free() -> None:
    torch.manual_seed(0)
    model = _Toy()
    images, positions = torch.randn(10, 1, 2, 2), torch.randn(10, 4)
    whole = predict_probabilities(model, images, positions, batch_size=10)
    pieces = predict_probabilities(model, images, positions, batch_size=3)
    torch.testing.assert_close(whole, pieces)
    torch.testing.assert_close(whole.sum(dim=1), torch.ones(10))
    assert whole.dtype == torch.float32 and whole.device.type == "cpu"
    assert model.training  # restored
    model.eval()
    predict_probabilities(model, images, positions, batch_size=4)
    assert not model.training


@pytest.mark.parametrize(
    ("n_images", "n_positions", "batch_size", "message"),
    [(3, 3, 0, "batch_size"), (0, 0, 2, "N > 0"), (3, 2, 2, "N positions")],
)
def test_predict_probabilities_refuses_bad_input(
    n_images: int, n_positions: int, batch_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        predict_probabilities(
            _Toy(), torch.zeros(n_images, 1, 2, 2), torch.zeros(n_positions, 4), batch_size
        )
