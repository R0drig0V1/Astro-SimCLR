"""Tests of the SimCLR network, objective, LARS groups and schedule (``astro_simclr.simclr``)."""

from __future__ import annotations

import itertools
import math
from contextlib import nullcontext
from typing import Any

import pytest
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock

from astro_simclr.optim import EXCLUDE_KEY, LARS
from astro_simclr.simclr import network
from astro_simclr.simclr.network import (
    PROJECTION_DIM,
    REPRESENTATION_DIM,
    ProjectionHead,
    SimCLRNetwork,
    checkpointed_encoder,
    encoder_segments,
    resnet18_cifar,
)
from astro_simclr.simclr.objective import SimCLRLoss
from astro_simclr.simclr.optim import (
    base_learning_rate,
    build_lars,
    lars_param_groups,
    set_learning_rate,
    warmup_cosine_lr,
)

# --- network ------------------------------------------------------------------------------


def test_resnet18_cifar_stem_and_output() -> None:
    torch.manual_seed(0)
    net = resnet18_cifar()
    assert isinstance(net.conv1, nn.Conv2d)
    assert net.conv1.kernel_size == (3, 3) and net.conv1.stride == (1, 1)
    assert net.conv1.padding == (1, 1) and net.conv1.bias is None
    assert isinstance(net.maxpool, nn.Identity) and isinstance(net.fc, nn.Identity)
    sizes: dict[str, int] = {}
    for name in ("conv1", "layer1", "layer2", "layer3", "layer4"):
        getattr(net, name).register_forward_hook(
            lambda _m, _i, out, name=name: sizes.__setitem__(name, out.shape[-1])
        )
    h = net.eval()(torch.zeros(2, 3, 57, 57))
    assert h.shape == (2, REPRESENTATION_DIM)
    assert sizes == {"conv1": 57, "layer1": 57, "layer2": 29, "layer3": 15, "layer4": 8}


def test_resnet18_cifar_zero_inits_the_residual_branches() -> None:
    blocks = [m for m in resnet18_cifar().modules() if isinstance(m, BasicBlock)]
    assert len(blocks) == 8
    assert all(torch.count_nonzero(b.bn2.weight) == 0 for b in blocks)
    plain = [
        m for m in resnet18_cifar(zero_init_residual=False).modules() if isinstance(m, BasicBlock)
    ]
    assert all(torch.count_nonzero(b.bn2.weight) > 0 for b in plain)
    with pytest.raises(ValueError, match="in_channels"):
        resnet18_cifar(in_channels=0)


def test_resnet18_cifar_accepts_a_mask_channel() -> None:
    assert resnet18_cifar(in_channels=4).eval()(torch.zeros(1, 4, 57, 57)).shape == (1, 512)


def test_projection_head_layers() -> None:
    head = ProjectionHead()
    kinds = [type(m) for m in head.layers]
    assert kinds == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Linear]
    first, last = head.layers[0], head.layers[3]
    assert isinstance(first, nn.Linear) and first.bias is None
    assert (first.in_features, first.out_features) == (512, 512)
    assert isinstance(last, nn.Linear) and last.out_features == PROJECTION_DIM


def test_simclr_network_returns_h_and_z() -> None:
    torch.manual_seed(0)
    h, z = SimCLRNetwork()(torch.rand(4, 3, 57, 57))
    assert h.shape == (4, REPRESENTATION_DIM) and z.shape == (4, PROJECTION_DIM)


def test_simclr_network_init_depends_only_on_the_seed() -> None:
    torch.manual_seed(5)
    a = SimCLRNetwork().state_dict()
    torch.manual_seed(5)
    b = SimCLRNetwork().state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


# --- activation checkpointing -------------------------------------------------------------


def _train(checkpointing: bool, steps: int = 3, batch: int = 4) -> dict[str, Any]:
    """``steps`` LARS steps of a seeded network on seeded views; the state they leave."""
    torch.manual_seed(0)
    model = SimCLRNetwork(activation_checkpointing=checkpointing)
    optimizer = build_lars(model, lr=0.3)
    loss_fn = SimCLRLoss(0.2)
    gen = torch.Generator().manual_seed(1)
    losses, grads = [], []
    for _ in range(steps):
        x = torch.rand(2 * batch, 3, 57, 57, generator=gen)
        model.train()
        _, z = model(x)
        loss, _, _ = loss_fn(z[:batch], z[batch:])
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
        optimizer.step()
        losses.append(loss.detach())
    return {
        "losses": losses,
        "grads": grads,
        "state": model.state_dict(),
        "optimizer": optimizer.state_dict()["state"],
    }


def _assert_same_training(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert all(torch.equal(x, y) for x, y in zip(a["losses"], b["losses"], strict=True))
    for ga, gb in zip(a["grads"], b["grads"], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(ga, gb, strict=True))
    assert a["state"].keys() == b["state"].keys()
    assert all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])
    for index, entry in a["optimizer"].items():
        other = b["optimizer"][index]
        assert all(torch.equal(v, other[k]) for k, v in entry.items() if torch.is_tensor(v))


def test_activation_checkpointing_trains_bit_for_bit_the_same() -> None:
    # Losses, every gradient, the weights, the BatchNorm buffers and the LARS momenta.
    plain, checkpointed = _train(False), _train(True)
    _assert_same_training(plain, checkpointed)
    assert int(checkpointed["state"]["encoder.bn1.num_batches_tracked"]) == 3


def test_activation_checkpointing_restores_the_batchnorm_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without the restore, the recompute updates every BatchNorm of the encoder twice a
    # step: the guard is what keeps the buffers equal (and the head is not checkpointed).
    plain = _train(False)
    monkeypatch.setattr(network, "_contexts", lambda _m: (nullcontext(), nullcontext()))
    doubled = _train(True)["state"]
    assert int(doubled["encoder.bn1.num_batches_tracked"]) == 6
    assert int(doubled["encoder.layer4.1.bn2.num_batches_tracked"]) == 6
    assert int(doubled["head.layers.1.num_batches_tracked"]) == 3
    assert not torch.equal(
        doubled["encoder.layer1.0.bn1.running_mean"],
        plain["state"]["encoder.layer1.0.bn1.running_mean"],
    )


def test_restore_buffers_on_an_early_stop() -> None:
    # The checkpoint's early stop ends a recompute by raising inside the context.
    bn = nn.BatchNorm2d(2).train()
    before = {k: v.clone() for k, v in bn.state_dict().items()}
    with pytest.raises(RuntimeError, match="stop"), network._restore_buffers(bn):
        bn(torch.randn(3, 2, 4, 4))
        raise RuntimeError("stop")
    assert all(torch.equal(before[k], v) for k, v in bn.state_dict().items())


def test_activation_checkpointing_only_in_train_mode_with_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def counting(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return checkpointed_encoder(encoder, x)

    monkeypatch.setattr(network, "checkpointed_encoder", counting)
    torch.manual_seed(0)
    model = SimCLRNetwork(activation_checkpointing=True)
    x = torch.rand(4, 3, 57, 57)
    with torch.no_grad():
        model.train()(x)
    model.eval()(x)
    assert calls == []
    model.train()(x)
    assert calls == [1]
    model.activation_checkpointing = False  # a plain attribute: no state
    model(x)
    assert calls == [1]


def test_encoder_segments_compose_to_the_encoder() -> None:
    torch.manual_seed(0)
    encoder = resnet18_cifar().eval()
    segments = encoder_segments(encoder)
    assert len(segments) == 9  # the stem and the eight residual blocks
    assert all(isinstance(m, BasicBlock) for m, _ in segments[1:])
    x = torch.rand(2, 3, 57, 57)
    with torch.no_grad():
        out = x
        for _, function in segments:
            out = function(out)
        pooled = nn.functional.adaptive_avg_pool2d(out, (1, 1))  # torchvision's avgpool
        assert torch.equal(torch.flatten(pooled, 1), encoder(x))
    with pytest.raises(TypeError, match="ResNet"):
        encoder_segments(nn.Linear(2, 2))


# --- objective ----------------------------------------------------------------------------


def _brute_force_nt_xent(z_i: torch.Tensor, z_j: torch.Tensor, t: float) -> float:
    """NT-Xent of Chen et al. 2020, eq. 1, averaged over all 2B anchors, by explicit loops."""
    z = torch.cat([z_i, z_j]).double()
    n, b = z.shape[0], z_i.shape[0]
    total = 0.0
    for a in range(n):
        pos = (a + b) % n
        sims = [
            math.exp(float(torch.nn.functional.cosine_similarity(z[a], z[k], dim=0)) / t)
            for k in range(n)
        ]
        denominator = sum(s for k, s in enumerate(sims) if k != a)
        total += -math.log(sims[pos] / denominator)
    return total / n


@pytest.mark.parametrize(("batch", "dim", "t"), [(1, 4, 0.5), (5, 8, 0.2), (16, 128, 0.5)])
def test_simclr_loss_matches_the_brute_force_reference(batch: int, dim: int, t: float) -> None:
    gen = torch.Generator().manual_seed(batch)
    z_i = torch.randn(batch, dim, generator=gen)
    z_j = z_i + 0.5 * torch.randn(batch, dim, generator=gen)
    loss, pos, neg = SimCLRLoss(t)(z_i, z_j)
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(_brute_force_nt_xent(z_i, z_j, t), rel=1e-5)
    assert -1 <= float(pos) <= 1
    if batch == 1:
        assert math.isnan(float(neg))


def test_simclr_loss_is_the_paper_mode_and_float32() -> None:
    loss = SimCLRLoss(0.3)
    assert loss.temperature == 0.3 and loss.loss.legacy_denominator is False
    z = torch.randn(4, 8)
    out, _, _ = loss(z.bfloat16(), z.bfloat16())
    assert out.dtype == torch.float32
    with pytest.raises(ValueError, match="temperature"):
        SimCLRLoss(0.0)


# --- LARS groups --------------------------------------------------------------------------


def test_lars_param_groups_exclude_norm_layers_and_biases() -> None:
    model = SimCLRNetwork()
    adapted, excluded = lars_param_groups(model, weight_decay=1e-6)
    assert adapted["weight_decay"] == 1e-6 and EXCLUDE_KEY not in adapted
    assert excluded["weight_decay"] == 0.0 and excluded[EXCLUDE_KEY] is True
    names = {id(p): n for n, p in model.named_parameters()}
    adapted_names = {names[id(p)] for p in adapted["params"]}
    excluded_names = {names[id(p)] for p in excluded["params"]}
    assert len(adapted_names) + len(excluded_names) == len(names)
    assert all(n.endswith("weight") and "bn" not in n for n in adapted_names)
    assert "encoder.bn1.weight" in excluded_names and "encoder.bn1.bias" in excluded_names
    assert "head.layers.1.weight" in excluded_names  # the head's BatchNorm
    assert "head.layers.3.bias" in excluded_names
    assert "head.layers.0.weight" in adapted_names and "encoder.conv1.weight" in adapted_names
    # Convolution and linear weights are >= 2-D; every excluded parameter is 1-D.
    assert all(p.ndim >= 2 for p in adapted["params"])
    assert all(p.ndim == 1 for p in excluded["params"])


def test_lars_param_groups_skip_frozen_and_shared_parameters() -> None:
    shared = nn.Linear(2, 2)
    model = nn.Sequential(shared, shared, nn.Linear(2, 2))
    model[2].weight.requires_grad_(False)
    adapted, excluded = lars_param_groups(model)
    assert len(adapted["params"]) == 1 and len(excluded["params"]) == 2
    with pytest.raises(ValueError, match="no trainable"):
        lars_param_groups(nn.ReLU())


def test_excluded_parameters_get_plain_sgd_with_momentum() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    twin = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
    twin.load_state_dict(model.state_dict())
    for net in (model, twin):  # only the excluded (1-D) parameters train, in both
        for p in net.parameters():
            p.requires_grad_(p.ndim == 1)
    lars = build_lars(model, lr=0.1, weight_decay=0.5)
    excluded_twin = lars_param_groups(twin)[1]["params"]
    sgd = torch.optim.SGD(excluded_twin, lr=0.1, momentum=0.9, weight_decay=0.0)
    x = torch.randn(8, 3)
    for _ in range(3):
        for net, opt in ((model, lars), (twin, sgd)):
            opt.zero_grad()
            net(x).pow(2).sum().backward()
            opt.step()
    for (name, p), q in zip(model.named_parameters(), twin.parameters(), strict=True):
        if p.ndim == 1:
            torch.testing.assert_close(p, q, msg=name)
    # The zero-initialised BatchNorm shift moves (the legacy LARS kept it at 0).
    norm = model[1]
    assert isinstance(norm, nn.BatchNorm1d) and torch.count_nonzero(norm.bias) > 0


def test_adapted_parameters_get_the_trust_ratio_and_weight_decay() -> None:
    torch.manual_seed(0)
    layer = nn.Linear(3, 2, bias=False)
    w0 = layer.weight.detach().clone()
    opt = build_lars(layer, lr=0.5, weight_decay=0.01, eeta=0.001)
    layer(torch.ones(1, 3)).sum().backward()
    assert layer.weight.grad is not None
    g = layer.weight.grad + 0.01 * w0
    ratio = 0.001 * w0.norm() / g.norm()
    opt.step()
    torch.testing.assert_close(layer.weight.detach(), w0 - 0.5 * ratio * g)


def test_lars_trust_ratio_is_one_when_a_norm_is_zero() -> None:
    weight = nn.Parameter(torch.zeros(2, 2))
    opt = LARS([weight], lr=0.1, legacy_trust_ratio=False)
    weight.grad = torch.ones(2, 2)
    opt.step()
    torch.testing.assert_close(weight.detach(), torch.full((2, 2), -0.1))


def test_lars_without_the_exclusion_key_is_unchanged() -> None:
    """A group without EXCLUDE_KEY (every legacy run) still takes the LARS update."""
    weight = nn.Parameter(torch.ones(2))
    opt = LARS([{"params": [weight]}], lr=1.0, eeta=0.5, legacy_trust_ratio=False)
    weight.grad = torch.full((2,), 2.0)
    opt.step()
    torch.testing.assert_close(weight.detach(), torch.full((2,), 1.0 - 0.5 * 0.5 * 2.0))


def test_build_lars_settings() -> None:
    opt = build_lars(SimCLRNetwork(), lr=0.3)
    for group in opt.param_groups:
        assert group["momentum"] == 0.9 and group["use_nesterov"] is False
        assert group["legacy_trust_ratio"] is False and group["eeta"] == 0.001
    assert opt.param_groups[0]["weight_decay"] == 1e-6


# --- schedule -----------------------------------------------------------------------------


def test_base_learning_rate_scales_with_the_batch() -> None:
    assert base_learning_rate(256) == pytest.approx(0.3)
    assert base_learning_rate(4096) == pytest.approx(4.8)
    assert base_learning_rate(32, per_256=0.6) == pytest.approx(0.075)
    with pytest.raises(ValueError, match="batch_size"):
        base_learning_rate(0)


def test_warmup_cosine_warms_up_per_step_then_ends_exactly_at_lr_min() -> None:
    def lr(step: int) -> float:
        return warmup_cosine_lr(step, base_lr=1.2, total_steps=100, warmup_steps=10, lr_min=0.01)

    lrs = [lr(s) for s in range(100)]
    assert lrs[0] == pytest.approx(0.12)
    assert lrs[9] == pytest.approx(1.2) and lrs[10] == pytest.approx(1.2)
    assert all(b > a for a, b in itertools.pairwise(lrs[:10]))
    assert all(b <= a for a, b in itertools.pairwise(lrs[10:]))
    assert lrs[-1] == 0.01  # exactly, at the run's final step
    assert lrs[-2] > 0.01
    halfway = lr(10 + 89 // 2)
    assert halfway == pytest.approx(0.01 + 1.19 * 0.5 * (1 + math.cos(math.pi * 44 / 89)))


def test_warmup_cosine_edge_cases() -> None:
    assert warmup_cosine_lr(0, base_lr=1.0, total_steps=5, warmup_steps=0) == 1.0
    assert warmup_cosine_lr(4, base_lr=1.0, total_steps=5, warmup_steps=0) == 0.0
    # One step after the warmup: it is the final step, so it runs at lr_min.
    assert warmup_cosine_lr(3, base_lr=1.0, total_steps=4, warmup_steps=3, lr_min=0.1) == 0.1
    assert warmup_cosine_lr(0, base_lr=1.0, total_steps=1, warmup_steps=0) == 0.0
    for bad in (
        {"step": 5, "total_steps": 5, "warmup_steps": 0},
        {"step": 0, "total_steps": 5, "warmup_steps": 5},
        {"step": -1, "total_steps": 5, "warmup_steps": 0},
    ):
        with pytest.raises(ValueError, match="total_steps"):
            warmup_cosine_lr(base_lr=1.0, **bad)
    with pytest.raises(ValueError, match="lr_min"):
        warmup_cosine_lr(0, base_lr=1.0, total_steps=5, warmup_steps=0, lr_min=2.0)


def test_set_learning_rate_sets_every_group() -> None:
    opt = build_lars(SimCLRNetwork(), lr=0.3)
    set_learning_rate(opt, 0.07)
    assert [g["lr"] for g in opt.param_groups] == [0.07, 0.07]
