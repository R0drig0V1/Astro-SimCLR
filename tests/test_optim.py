"""Tests for astro_simclr.optim: LARS and the cosine-with-warmup LR schedule.

Fixtures (tests/fixtures/optim/legacy_optim.npz) come from scripts/characterize/optim/: the
legacy utils/lars.py on CPU (get_device monkeypatched) and timm 0.6.13 CosineLRScheduler
wrapped in LambdaLR, as in the legacy SimCLR configure_optimizers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from astro_simclr.optim import LARS, cosine_warmup_lr, cosine_warmup_scheduler

FIXTURE = Path(__file__).parent / "fixtures" / "optim" / "legacy_optim.npz"
ATOL = 1e-6


@pytest.fixture(scope="module")
def legacy() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def test_fixture_records_provenance(legacy: dict[str, np.ndarray]) -> None:
    assert len(str(legacy["legacy_sha"])) == 40
    assert str(legacy["torch_version"]).startswith("2.")
    assert str(legacy["timm_version"]).startswith("0.6.")


# --- LARS -----------------------------------------------------------------------------------


def _params(legacy: dict[str, np.ndarray]) -> list[torch.nn.Parameter]:
    return [
        torch.nn.Parameter(torch.from_numpy(legacy[f"lars_init_{i}"]).clone()) for i in range(3)
    ]


@pytest.mark.parametrize("weight_decay", [0.0, 1e-4])
@pytest.mark.parametrize("nesterov", [False, True])
def test_lars_matches_legacy_steps(
    legacy: dict[str, np.ndarray], nesterov: bool, weight_decay: float
) -> None:
    params = _params(legacy)
    opt = LARS(params, lr=0.258, momentum=0.9, use_nesterov=nesterov, weight_decay=weight_decay)
    tag = f"nesterov{int(nesterov)}_wd{weight_decay:g}"
    for step in range(2):
        for i, p in enumerate(params):
            p.grad = torch.from_numpy(legacy[f"lars_grad_step{step}_{i}"]).clone()
        opt.step()
        for i, p in enumerate(params):
            want = legacy[f"lars_{tag}_step{step}_{i}"]
            np.testing.assert_allclose(p.detach().numpy(), want, rtol=0, atol=ATOL)


@pytest.mark.parametrize("legacy_trust_ratio", [True, False])
def test_lars_one_step_by_hand(legacy_trust_ratio: bool) -> None:
    w = torch.nn.Parameter(torch.tensor([3.0, 4.0]))  # ||w|| = 5
    w.grad = torch.tensor([0.6, 0.8])  # ||g|| = 1
    LARS([w], lr=2.0, momentum=0.9, eeta=0.01, legacy_trust_ratio=legacy_trust_ratio).step()
    # Both norms are positive, so both flags give ratio eeta * 5 / 1 = 0.05.
    expected = torch.tensor([3.0, 4.0]) - 2.0 * 0.05 * torch.tensor([0.6, 0.8])
    torch.testing.assert_close(w.detach(), expected, rtol=0, atol=ATOL)


def test_lars_nesterov_second_step_by_hand() -> None:
    w = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    opt = LARS([w], lr=1.0, momentum=0.5, use_nesterov=True, eeta=1.0)
    g = torch.tensor([0.6, 0.8])
    w.grad = g.clone()
    opt.step()  # ratio 5, v = 5 g, update = 0.5 * 5 g + 5 g = 7.5 g
    expected = torch.tensor([3.0, 4.0]) - 7.5 * g
    torch.testing.assert_close(w.detach(), expected, rtol=0, atol=ATOL)


def test_lars_zero_gradient_legacy_gives_nan() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    w.grad = torch.zeros(2)
    LARS([w], lr=0.1).step()
    # Legacy .ge(0): ratio = eeta * ||w|| / 0 = inf, and inf * 0 = NaN in the buffer.
    assert torch.isnan(w).all()


def test_lars_zero_gradient_strict_leaves_weight() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    w.grad = torch.zeros(2)
    LARS([w], lr=0.1, legacy_trust_ratio=False).step()
    assert torch.equal(w.detach(), torch.tensor([1.0, 2.0]))


def test_lars_zero_weight_legacy_never_moves() -> None:
    b = torch.nn.Parameter(torch.zeros(3))
    opt = LARS([b], lr=0.1, use_nesterov=True)
    for _ in range(3):
        b.grad = torch.tensor([1.0, -2.0, 0.5])
        opt.step()
    # Legacy .ge(0): ratio = eeta * 0 / ||g|| = 0, so a zero-initialised bias stays at 0.
    assert torch.equal(b.detach(), torch.zeros(3))


def test_lars_zero_weight_strict_takes_plain_momentum_step() -> None:
    b = torch.nn.Parameter(torch.zeros(3))
    g = torch.tensor([1.0, -2.0, 0.5])
    b.grad = g.clone()
    LARS([b], lr=0.1, use_nesterov=True, legacy_trust_ratio=False).step()
    # Ratio 1: v = 0.1 g, update = 0.9 * 0.1 g + 0.1 g.
    torch.testing.assert_close(b.detach(), -(0.09 + 0.1) * g, rtol=0, atol=ATOL)


@pytest.mark.parametrize("legacy_trust_ratio", [True, False])
def test_lars_zero_weight_and_gradient(legacy_trust_ratio: bool) -> None:
    w = torch.nn.Parameter(torch.zeros(2))
    w.grad = torch.zeros(2)
    LARS([w], lr=0.1, legacy_trust_ratio=legacy_trust_ratio).step()
    if legacy_trust_ratio:
        assert torch.isnan(w).all()  # 0 / 0
    else:
        assert torch.equal(w.detach(), torch.zeros(2))


def test_lars_skips_parameters_without_grad_and_runs_closure() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0]))
    frozen = torch.nn.Parameter(torch.tensor([5.0]))
    opt = LARS([w, frozen], lr=0.1)

    def closure() -> float:
        opt.zero_grad()
        loss = (w**2).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return float(loss.detach())

    assert opt.step(closure) == pytest.approx(1.0)
    assert w.grad is not None
    assert float(w.detach()) < 1.0
    assert float(frozen.detach()) == 5.0
    assert "momentum_buffer" not in opt.state[frozen]


def test_lars_does_not_modify_grad() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    w.grad = torch.tensor([0.5, 0.5])
    LARS([w], lr=0.1, weight_decay=0.1).step()
    assert torch.equal(w.grad, torch.tensor([0.5, 0.5]))


def test_lars_reads_hyperparameters_from_param_groups() -> None:
    a = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    b = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    groups: list[dict[str, Any]] = [{"params": [a]}, {"params": [b], "lr": 0.0}]
    opt = LARS(groups, lr=1.0)
    a.grad = torch.tensor([0.6, 0.8])
    b.grad = torch.tensor([0.6, 0.8])
    opt.step()
    assert not torch.equal(a.detach(), torch.tensor([3.0, 4.0]))
    assert torch.equal(b.detach(), torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("kwarg", ["lr", "momentum", "weight_decay", "eeta"])
def test_lars_rejects_negative_hyperparameters(kwarg: str) -> None:
    kwargs: dict[str, Any] = {"lr": 0.1, kwarg: -1.0}
    with pytest.raises(ValueError, match=kwarg):
        LARS([torch.nn.Parameter(torch.zeros(1))], **kwargs)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_lars_runs_on_mps() -> None:  # pragma: no cover - hardware dependent
    w = torch.nn.Parameter(torch.tensor([3.0, 4.0], device="mps"))
    w.grad = torch.tensor([0.6, 0.8], device="mps")
    LARS([w], lr=1.0).step()
    assert w.device.type == "mps"


# --- LR schedule ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_lr", [0.258, 1.258])
def test_scheduler_drives_optimizer_lr_like_legacy(
    legacy: dict[str, np.ndarray], base_lr: float
) -> None:
    want = legacy[f"lr_{base_lr:g}"]
    assert len(want) == 401  # epochs 0-400
    opt = LARS([torch.nn.Parameter(torch.zeros(1))], lr=base_lr, use_nesterov=True)
    sched = cosine_warmup_scheduler(opt, base_lr)
    got = []
    for _ in range(len(want)):
        got.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    got_arr = np.array(got)
    np.testing.assert_allclose(got_arr[:5], want[:5], rtol=0, atol=ATOL)  # warmup
    np.testing.assert_allclose(got_arr[5:350], want[5:350], rtol=0, atol=ATOL)  # cosine
    np.testing.assert_allclose(got_arr[350:], want[350:], rtol=0, atol=ATOL)  # plateau
    assert np.all(got_arr[350:] == pytest.approx(1e-4))


def test_schedule_shape_by_hand() -> None:
    lr = 0.3
    assert cosine_warmup_lr(0, lr) == pytest.approx(0.1)
    assert cosine_warmup_lr(4, lr) == pytest.approx(0.1 + 4 * (0.2 / 5))
    # The cosine counts from epoch 0 (no warmup prefix).
    expected = 1e-4 + 0.5 * (lr - 1e-4) * (1 + math.cos(math.pi * 5 / 350))
    assert cosine_warmup_lr(5, lr) == pytest.approx(expected)
    assert cosine_warmup_lr(349, lr) > 1e-4
    assert cosine_warmup_lr(350, lr) == 1e-4
    assert cosine_warmup_lr(10_000, lr) == 1e-4


def test_schedule_options() -> None:
    assert cosine_warmup_lr(0, 1.0, warmup_lr_init=0.0) == 0.0
    assert cosine_warmup_lr(0, 1.0, warmup_epochs=0) == pytest.approx(1.0)
    assert cosine_warmup_lr(10, 1.0, t_initial=10, lr_min=0.5) == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [{"epoch": -1}, {"epoch": 0, "warmup_epochs": -1}, {"epoch": 0, "t_initial": 0}],
)
def test_schedule_rejects_bad_arguments(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="need epoch"):
        cosine_warmup_lr(base_lr=0.1, **kwargs)


def test_scheduler_rejects_non_positive_base_lr() -> None:
    opt = LARS([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    with pytest.raises(ValueError, match="base_lr"):
        cosine_warmup_scheduler(opt, 0.0)
