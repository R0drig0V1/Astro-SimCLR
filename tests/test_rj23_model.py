"""Tests for RJ23's classifiers (``astro_simclr.rj23_model``, task 3.4; ADR 0013).

Table 2 as published; the output and trunk shapes of the four scenarios on views built by
``rj23_input``; the parameter counts (hand-computed literals and the documented formula);
the TF-default parity golden tests (TF 'same' padding and max-pooling, Keras batch norm,
Glorot-uniform with zero biases), all computed by hand without TF; the dihedral invariance
of the averaged representation, on the views and on raw stamps (exact where the views make
it so, and the one-pixel crop shift of ADR 0009 where they do not); determinism; and CPU
against MPS or CUDA where present.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import torch
from _training_support import ACCELERATORS
from torch import nn

from astro_simclr.data import rj23_input as rj
from astro_simclr.rj23_model import (
    KERAS_BN_EPS,
    KERAS_BN_MOMENTUM,
    MAXPOOL,
    N_VIEWS,
    RJ23_TABLE2,
    KerasBatchNorm1d,
    RJ23Architecture,
    RJ23Classifier,
    TFSameMaxPool2d,
    build_rj23_classifier,
    dihedral_views,
    init_keras_defaults,
    rj23_parameter_count,
    tf_same_padding,
)

AnyFloat = npt.NDArray[np.floating]

#: The eight elements of D4 on channels-first torch images (N, C, H, W).
D4_TORCH: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda t: t,
    "rot90": lambda t: torch.rot90(t, 1, dims=(2, 3)),
    "rot180": lambda t: torch.rot90(t, 2, dims=(2, 3)),
    "rot270": lambda t: torch.rot90(t, 3, dims=(2, 3)),
    "flip_rows": lambda t: torch.flip(t, dims=(2,)),
    "flip_cols": lambda t: torch.flip(t, dims=(3,)),
    "transpose": lambda t: t.transpose(2, 3),
    "anti_transpose": lambda t: torch.rot90(t.transpose(2, 3), 2, dims=(2, 3)),
}

#: The same eight on channels-last numpy stamps (N, H, W, C).
D4_NUMPY: dict[str, Callable[[AnyFloat], AnyFloat]] = {
    "identity": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1, axes=(1, 2)),
    "rot180": lambda a: np.rot90(a, 2, axes=(1, 2)),
    "rot270": lambda a: np.rot90(a, 3, axes=(1, 2)),
    "flip_rows": lambda a: a[:, ::-1],
    "flip_cols": lambda a: a[:, :, ::-1],
    "transpose": lambda a: np.swapaxes(a, 1, 2),
    "anti_transpose": lambda a: np.rot90(np.swapaxes(a, 1, 2), 2, axes=(1, 2)),
}

#: Which axes of the source stamp each element reverses (rows, columns).
REVERSED_AXES = {
    "identity": (False, False),
    "rot90": (False, True),
    "rot180": (True, True),
    "rot270": (True, False),
    "flip_rows": (True, False),
    "flip_cols": (False, True),
    "transpose": (False, False),
    "anti_transpose": (True, True),
}

SCENARIOS = list(RJ23_TABLE2)


def raw_stamps(n: int = 3, seed: int = 0) -> npt.NDArray[np.float64]:
    """Random raw cutouts (N, 63, 63, 3) with signs, a wide range and an invalid pixel."""
    rng = np.random.default_rng(seed)
    stamps = rng.normal(size=(n, 63, 63, 3)) * rng.uniform(0.5, 500.0, size=(n, 1, 1, 3))
    stamps[0, :3, :5, 0] = np.nan
    return stamps


def view_tensor(stamps: AnyFloat, scenario: rj.ViewName) -> torch.Tensor:
    """The scenario's view of raw stamps as a channels-first float64 tensor."""
    view = rj.rj23_views(np.ascontiguousarray(stamps), [scenario])[scenario]
    return torch.from_numpy(rj.channels_first(view)).double()


def positions(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    ra, dec = rng.uniform(0, 360, n), rng.uniform(-30, 90, n)
    return torch.from_numpy(rj.unit_sphere(ra, dec))


def seeded_model(scenario: rj.ViewName, seed: int = 0, **kwargs: object) -> RJ23Classifier:
    torch.manual_seed(seed)
    return build_rj23_classifier(scenario, **kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------
# Table 2


def test_table2_as_published() -> None:
    # RJ23 Table 2 (arXiv:2304.13080v2, App. A), read column by column.
    expected = {
        "full": ((37, MAXPOOL, 60, MAXPOOL, 29, MAXPOOL, 26), 0.71726, 31, 7.589e-4),
        "cropped": ((112, 27, MAXPOOL, 92, 121), 0.866799, 133, 3.4627e-4),
        "lowres": ((63, 31, MAXPOOL, 108, 14), 0.73065, 22, 1.125e-3),
        "multiscale": ((76, 18, 54, 28), 0.8478, 69, 7.445e-4),
    }
    assert set(RJ23_TABLE2) == set(rj.VIEW_NAMES)
    for scenario, (layers, dropout, dense, lr) in expected.items():
        arch = RJ23_TABLE2[scenario]  # type: ignore[index]
        assert arch.scenario == scenario
        assert arch.layers == layers
        assert arch.dropout == dropout
        assert arch.dense == dense
        assert arch.learning_rate == lr
        assert arch.first_kernel_size == 3
    assert RJ23_TABLE2["multiscale"].input_shape == (24, 8, 8)
    assert RJ23_TABLE2["full"].input_shape == (6, 63, 63)


# ----------------------------------------------------------------------------------------
# Shapes and parameter counts


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("n_classes", [5, 6])
def test_output_and_trunk_shapes(scenario: rj.ViewName, n_classes: int) -> None:
    model = seeded_model(scenario, n_classes=n_classes).eval()
    x = view_tensor(raw_stamps(3), scenario).float()
    channels, side = RJ23_TABLE2[scenario].conv_widths[-1], 8
    with torch.no_grad():
        assert model.trunk(dihedral_views(x)).shape == (N_VIEWS * 3, channels, side, side)
        rep = model.representation(x)
        assert rep.shape == (3, 64 * channels)
        assert model(x, positions(3)).shape == (3, n_classes)
    assert rep.dtype == torch.float32


def test_trunk_pools_63_to_32_to_16_to_8() -> None:
    model = seeded_model("full")
    sides = []
    x = torch.zeros(1, 6, 63, 63)
    for layer in model.trunk:
        x = layer(x)
        if isinstance(layer, TFSameMaxPool2d):
            sides.append(x.shape[-1])
    assert sides == [32, 16, 8]


#: Hand-computed with the module docstring's formula, K = 5, BN after the concat.
#: full: (9*6+1)*37 + (9*37+1)*60 + (9*60+1)*29 + (9*29+1)*26 = 44,576; D = 64*26 = 1664;
#: + 2*1667 + 1668*31 + 32*5 = 99,778.
#: multiscale: (9*24+1)*76 + (9*76+1)*18 + (9*18+1)*54 + (9*54+1)*28 = 51,260; D = 1792;
#: + 2*1795 + 1796*69 + 70*5 = 179,124.
PARAMETER_COUNTS = {"full": 99_778, "cropped": 1_202_808, "lowres": 86_648, "multiscale": 179_124}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_parameter_counts(scenario: rj.ViewName) -> None:
    arch = RJ23_TABLE2[scenario]
    for bn_position, n_classes in [("after_concat", 5), ("before_concat", 5), ("after_concat", 6)]:
        model = build_rj23_classifier(scenario, n_classes=n_classes, bn_position=bn_position)  # type: ignore[arg-type]
        counted = sum(p.numel() for p in model.parameters() if p.requires_grad)
        formula = rj23_parameter_count(arch, n_classes=n_classes, bn_position=bn_position)  # type: ignore[arg-type]
        assert counted == formula
    assert rj23_parameter_count(arch) == PARAMETER_COUNTS[scenario]
    # The sensitivity arm's BN has 3 features (6 parameters) fewer.
    assert rj23_parameter_count(arch, bn_position="before_concat") == PARAMETER_COUNTS[scenario] - 6
    # One more class adds dense + 1 parameters.
    assert rj23_parameter_count(arch, n_classes=6) == PARAMETER_COUNTS[scenario] + arch.dense + 1


def test_first_kernel_size_widens_only_the_first_convolution() -> None:
    base = RJ23_TABLE2["multiscale"]
    arch = RJ23Architecture(
        scenario="multiscale",
        layers=base.layers,
        dense=base.dense,
        dropout=base.dropout,
        learning_rate=base.learning_rate,
        first_kernel_size=5,
    )
    model = RJ23Classifier(arch)
    convs = [m for m in model.trunk if isinstance(m, nn.Conv2d)]
    assert [c.kernel_size for c in convs] == [(5, 5), (3, 3), (3, 3), (3, 3)]
    assert rj23_parameter_count(arch) == rj23_parameter_count(base) + (25 - 9) * 24 * 76
    assert sum(p.numel() for p in model.parameters()) == rj23_parameter_count(arch)


# ----------------------------------------------------------------------------------------
# TF 'same' padding and max-pooling (hand-computed)


@pytest.mark.parametrize(
    ("size", "kernel", "stride", "expected"),
    [
        (63, 2, 2, (32, 0, 1)),  # the full trunk's first pooling: one pixel after
        (32, 2, 2, (16, 0, 0)),
        (16, 2, 2, (8, 0, 0)),
        (63, 3, 1, (63, 1, 1)),  # a 3x3 'same' convolution
        (5, 4, 2, (3, 1, 2)),  # total 3: one before, two after
        (4, 2, 1, (4, 0, 1)),
        (2, 5, 3, (1, 1, 2)),
    ],
)
def test_tf_same_padding(
    size: int, kernel: int, stride: int, expected: tuple[int, int, int]
) -> None:
    assert tf_same_padding(size, kernel, stride) == expected


def test_tf_same_max_pool_ignores_the_padding_on_negative_inputs() -> None:
    x = -torch.tensor([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]).reshape(1, 1, 3, 3)
    out = TFSameMaxPool2d(2)(x)
    # Windows {0,1}x{0,1}, {0,1}x{2}, {2}x{0,1}, {2}x{2}; zero padding would give 0s.
    torch.testing.assert_close(out, torch.tensor([[[[-1.0, -3.0], [-7.0, -9.0]]]]))


def test_tf_same_max_pool_with_padding_before() -> None:
    # Width 5, kernel 4, stride 2: one pixel before, two after; windows over the columns
    # {0, 1, 2}, {1, 2, 3, 4} and {3, 4}.
    x = torch.tensor([-3.0, -1, -4, -5, -2]).reshape(1, 1, 1, 5)
    out = TFSameMaxPool2d(4, stride=2)(x)
    torch.testing.assert_close(out, torch.tensor([[[[-1.0, -1.0, -2.0]]]]))
    assert "padding='same'" in repr(TFSameMaxPool2d(4, stride=2))


def test_tf_same_max_pool_matches_torch_ceil_mode_when_padding_is_after() -> None:
    x = torch.randn(2, 3, 63, 63, generator=torch.Generator().manual_seed(0)) - 5
    for side in (63, 32, 16):
        y = x[..., :side, :side]
        expected = nn.MaxPool2d(2, ceil_mode=True)(y)
        torch.testing.assert_close(TFSameMaxPool2d(2)(y), expected, rtol=0, atol=0)
    assert TFSameMaxPool2d(2)(x).shape[-2:] == (32, 32)


def test_tf_same_max_pool_backward_reaches_only_valid_pixels() -> None:
    x = (-torch.arange(9.0).reshape(1, 1, 3, 3)).requires_grad_()
    TFSameMaxPool2d(2)(x).sum().backward()
    assert x.grad is not None
    # The maxima of the four windows are pixels (0,0), (0,2), (2,0) and (2,2).
    expected = torch.tensor([[1.0, 0, 1], [0, 0, 0], [1, 0, 1]]).reshape(1, 1, 3, 3)
    torch.testing.assert_close(x.grad, expected)


# ----------------------------------------------------------------------------------------
# Keras batch normalisation (hand-computed)


def test_keras_batch_norm_defaults_and_one_training_step() -> None:
    bn = KerasBatchNorm1d(2)
    assert bn.eps == KERAS_BN_EPS == 1e-3
    assert bn.momentum == KERAS_BN_MOMENTUM == pytest.approx(1 - 0.99)
    torch.testing.assert_close(bn.weight.detach(), torch.ones(2))
    torch.testing.assert_close(bn.bias.detach(), torch.zeros(2))
    x = torch.tensor([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0], [6.0, 1.0]], dtype=torch.float64)
    bn = bn.double().train()
    out = bn(x)
    # Feature 0: mean 3, biased variance (9 + 1 + 1 + 9) / 4 = 5 (unbiased 20/3).
    expected0 = (x[:, 0] - 3.0) / math.sqrt(5.0 + 1e-3)
    torch.testing.assert_close(out[:, 0], expected0)
    # Feature 1 is constant: mean 1, variance 0, so (x - 1) / sqrt(eps) = 0.
    torch.testing.assert_close(out[:, 1], torch.zeros(4, dtype=torch.float64))
    # Moving statistics: 0.99 * old + 0.01 * batch, with the biased variance.
    assert bn.running_mean is not None and bn.running_var is not None
    torch.testing.assert_close(bn.running_mean, torch.tensor([0.03, 0.01], dtype=torch.float64))
    torch.testing.assert_close(bn.running_var, torch.tensor([1.04, 0.99], dtype=torch.float64))
    assert int(bn.num_batches_tracked) == 1  # type: ignore[arg-type]
    # torch's BatchNorm1d would have taken the unbiased 20/3 for the moving variance.
    reference = nn.BatchNorm1d(2, eps=1e-3, momentum=0.01).double().train()
    reference(x)
    assert reference.running_var is not None
    assert float(reference.running_var[0]) == pytest.approx(0.99 + 0.01 * 20 / 3)
    # Evaluation uses the moving statistics: (x - 0.03) / sqrt(1.04 + 1e-3).
    bn.eval()
    expected_eval = (x[:, 0] - 0.03) / math.sqrt(1.04 + 1e-3)
    torch.testing.assert_close(bn(x)[:, 0], expected_eval)


def test_keras_batch_norm_output_and_gradient_equal_torch_in_training() -> None:
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(16, 5, generator=gen, dtype=torch.float64)
    ours, theirs = KerasBatchNorm1d(5).double(), nn.BatchNorm1d(5, eps=1e-3).double()
    with torch.no_grad():
        for layer in (ours, theirs):
            layer.weight.copy_(torch.linspace(0.5, 2.0, 5))
            layer.bias.copy_(torch.linspace(-1.0, 1.0, 5))
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    out_a, out_b = ours(xa), theirs(xb)
    torch.testing.assert_close(out_a, out_b)
    (out_a**3).sum().backward()
    (out_b**3).sum().backward()
    assert xa.grad is not None and xb.grad is not None and ours.weight.grad is not None
    torch.testing.assert_close(xa.grad, xb.grad)
    assert theirs.weight.grad is not None
    torch.testing.assert_close(ours.weight.grad, theirs.weight.grad)


def test_keras_batch_norm_rejects_a_single_sample_in_training() -> None:
    with pytest.raises(ValueError, match="more than 1 value"):
        KerasBatchNorm1d(3)(torch.ones(1, 3))


# ----------------------------------------------------------------------------------------
# Glorot-uniform kernels, zero biases (hand-computed bounds)


def test_glorot_uniform_bounds_and_zero_biases() -> None:
    model = seeded_model("full", seed=1)
    convs = [m for m in model.trunk if isinstance(m, nn.Conv2d)]
    # sqrt(6 / (fan_in + fan_out)), Keras' fans: k^2 c_in and k^2 c_out, or the dense widths.
    bounds: dict[nn.Conv2d | nn.Linear, float] = {
        convs[0]: math.sqrt(6 / (9 * 6 + 9 * 37)),  # 0.12451...
        convs[1]: math.sqrt(6 / (9 * 37 + 9 * 60)),  # 0.08287...
        model.fc: math.sqrt(6 / (1667 + 31)),  # 0.05944...
        model.logits: math.sqrt(6 / (31 + 5)),  # 0.40825
    }
    assert bounds[convs[0]] == pytest.approx(0.124515, abs=1e-6)
    assert bounds[model.logits] == pytest.approx(0.408248, abs=1e-6)
    for layer, bound in bounds.items():
        weight = layer.weight.detach()
        assert float(weight.abs().max()) <= bound
        assert float(weight.abs().max()) > 0.9 * bound
        assert layer.bias is not None
        assert torch.count_nonzero(layer.bias) == 0
    # A uniform on [-l, l] has variance l^2 / 3 (20,000+ samples: within 5 %).
    fc = model.fc.weight.detach().double()
    assert float(fc.var()) == pytest.approx(bounds[model.fc] ** 2 / 3, rel=0.05)
    for conv in convs:
        assert conv.bias is not None and torch.count_nonzero(conv.bias) == 0
    assert torch.equal(model.bn.weight.detach(), torch.ones_like(model.bn.weight))
    assert torch.count_nonzero(model.bn.bias) == 0


def test_init_keras_defaults_resets_a_trained_model() -> None:
    # The constructor draws torch's default init first, so compare two re-inits instead.
    model = seeded_model("multiscale", seed=4)
    torch.manual_seed(4)
    init_keras_defaults(model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()(view_tensor(raw_stamps(4), "multiscale").float(), positions(4))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    assert model.bn.running_mean is not None and torch.count_nonzero(model.bn.running_mean) > 0
    torch.manual_seed(4)
    init_keras_defaults(model)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


# ----------------------------------------------------------------------------------------
# The 8 views and the invariance of the averaged representation


def test_dihedral_views_order() -> None:
    x = torch.arange(2 * 1 * 3 * 3, dtype=torch.float32).reshape(2, 1, 3, 3)
    views = dihedral_views(x)
    assert views.shape == (16, 1, 3, 3)
    for k in range(4):
        rotated = torch.rot90(x, k, dims=(2, 3))
        torch.testing.assert_close(views[k * 2 : k * 2 + 2], rotated)
        torch.testing.assert_close(views[8 + k * 2 : 8 + k * 2 + 2], torch.flip(rotated, (2,)))
    # The 8 views of one sample are distinct (an asymmetric input) ...
    per_sample = views.view(8, 2, 9)[:, 0]
    assert len({tuple(v.tolist()) for v in per_sample}) == 8
    # ... and any D4 transform of the input only permutes them.
    for g in D4_TORCH.values():
        moved = dihedral_views(g(x)).view(8, 2, 9)[:, 0]
        assert {tuple(v.tolist()) for v in moved} == {tuple(v.tolist()) for v in per_sample}


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("name", list(D4_TORCH))
def test_representation_is_invariant_on_the_views(scenario: rj.ViewName, name: str) -> None:
    model = seeded_model(scenario).double().eval()
    with torch.no_grad():
        first = model.trunk[0]
        assert isinstance(first, nn.Conv2d) and first.bias is not None
        nn.init.uniform_(first.bias, -0.3, 0.3)  # nonzero biases: a stricter check
    x = view_tensor(raw_stamps(2, seed=5), scenario)
    pos = positions(2).double()
    g = D4_TORCH[name]
    with torch.no_grad():
        torch.testing.assert_close(
            model.representation(g(x)), model.representation(x), rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(model(g(x), pos), model(x, pos), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", list(D4_NUMPY))
def test_full_scenario_is_invariant_to_rotating_the_raw_stamp(name: str) -> None:
    # The normalisation is per channel and the full view is the whole stamp, so rotating
    # or flipping the raw stamp rotates the view exactly (ADR 0009).
    model = seeded_model("full").double().eval()
    stamps = raw_stamps(2, seed=6)
    with torch.no_grad():
        moved = model.representation(view_tensor(D4_NUMPY[name](stamps), "full"))
        torch.testing.assert_close(
            moved, model.representation(view_tensor(stamps, "full")), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_is_invariant_to_transposing_the_raw_stamp(scenario: rj.ViewName) -> None:
    # The transpose maps each crop window (equal on both axes) and each pooling grid
    # (padded at the end of both axes) onto itself, so it is exact in every scenario.
    model = seeded_model(scenario).double().eval()
    stamps = raw_stamps(2, seed=8)
    with torch.no_grad():
        moved = model.representation(view_tensor(D4_NUMPY["transpose"](stamps), scenario))
        torch.testing.assert_close(
            moved, model.representation(view_tensor(stamps, scenario)), rtol=1e-6, atol=1e-9
        )


@pytest.mark.parametrize("name", list(D4_NUMPY))
def test_cropped_scenario_sees_the_mirrored_window_of_a_rotated_stamp(name: str) -> None:
    # ADR 0009: the 16x16 crop 24:40 is centred on pixel 31.5, so crop(g(x)) is g of the
    # window mirrored about the stamp's centre (23:39 along each axis g reverses). The
    # averaged representation therefore equals that of the shifted window, and differs from
    # the unrotated crop's unless g keeps the window (identity and transpose).
    model = seeded_model("cropped").double().eval()
    stamps = raw_stamps(2, seed=9)
    normalised = rj.normalise(stamps)
    rev_rows, rev_cols = REVERSED_AXES[name]
    rows = slice(23, 39) if rev_rows else slice(24, 40)
    cols = slice(23, 39) if rev_cols else slice(24, 40)
    window = torch.from_numpy(rj.channels_first(normalised[:, rows, cols])).double()
    with torch.no_grad():
        moved = model.representation(view_tensor(D4_NUMPY[name](stamps), "cropped"))
        torch.testing.assert_close(moved, model.representation(window), rtol=1e-12, atol=1e-12)
        unrotated = model.representation(view_tensor(stamps, "cropped"))
    if name in ("identity", "transpose"):
        torch.testing.assert_close(moved, unrotated, rtol=1e-12, atol=1e-12)
    else:
        assert not torch.allclose(moved, unrotated, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("scenario", ["lowres", "multiscale"])
def test_pooled_scenarios_are_not_invariant_to_rotating_the_raw_stamp(
    scenario: rj.ViewName,
) -> None:
    # The pooled grids pad at the end (63 = 15*4 + 3, 63 = 31*2 + 1), so a 90-degree
    # rotation of the stamp changes the coarse views, not only their orientation.
    model = seeded_model(scenario).double().eval()
    stamps = raw_stamps(2, seed=10)
    with torch.no_grad():
        moved = model.representation(view_tensor(D4_NUMPY["rot90"](stamps), scenario))
        unrotated = model.representation(view_tensor(stamps, scenario))
    assert not torch.allclose(moved, unrotated, rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------------------
# The head and the sensitivity arm


def test_head_order_and_bn_positions() -> None:
    for bn_position, features in [("after_concat", 1667), ("before_concat", 1664)]:
        model = seeded_model("full", bn_position=bn_position)
        assert model.bn.num_features == features
        assert model.fc.in_features == 1667
        assert model.dropout.p == RJ23_TABLE2["full"].dropout
    # After the concat the position goes through the BN: a constant position in a training
    # batch is normalised to 0, so the logits do not depend on its value.
    model = seeded_model("multiscale").train()
    model.dropout.p = 0.0
    x = view_tensor(raw_stamps(4), "multiscale").float()
    ones = torch.ones(4, 3)
    torch.testing.assert_close(model(x, ones), model(x, 5 * ones))
    # Before the concat it does not: the position reaches fc unnormalised.
    model = seeded_model("multiscale", bn_position="before_concat").train()
    model.dropout.p = 0.0
    assert not torch.allclose(model(x, ones), model(x, 5 * ones))


def test_dropout_acts_on_the_representation_in_training_only() -> None:
    model = seeded_model("lowres")
    x = view_tensor(raw_stamps(4), "lowres").float()
    pos = positions(4)
    model.eval()
    with torch.no_grad():
        first, second = model(x, pos), model(x, pos)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    model.train()
    torch.manual_seed(0)
    a = model(x, pos)
    torch.manual_seed(1)
    b = model(x, pos)
    assert not torch.allclose(a, b)


# ----------------------------------------------------------------------------------------
# Determinism


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_seeded_builds_and_steps_are_deterministic(scenario: rj.ViewName) -> None:
    x = view_tensor(raw_stamps(4, seed=11), scenario).float()
    pos = positions(4)
    labels = torch.tensor([0, 1, 2, 3])
    results = []
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        for _ in range(2):
            model = seeded_model(scenario, seed=12).train()
            torch.manual_seed(13)  # the dropout mask
            loss: torch.Tensor = nn.functional.cross_entropy(model(x, pos), labels)
            loss.backward()  # type: ignore[no-untyped-call]
            grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            results.append((model.state_dict(), loss.detach(), grads))
    finally:
        torch.use_deterministic_algorithms(previous)
    (state_a, loss_a, grads_a), (state_b, loss_b, grads_b) = results
    for key in state_a:
        torch.testing.assert_close(state_a[key], state_b[key], rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    assert len(grads_a) == len(list(seeded_model(scenario).parameters()))
    for ga, gb in zip(grads_a, grads_b, strict=True):
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def test_different_seeds_give_different_weights() -> None:
    a, b = seeded_model("full", seed=0), seeded_model("full", seed=1)
    assert not torch.equal(a.fc.weight, b.fc.weight)


# ----------------------------------------------------------------------------------------
# Devices


@pytest.mark.parametrize("accelerator", ACCELERATORS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_device_matches_cpu(accelerator: str, scenario: rj.ViewName) -> None:
    device = torch.device(accelerator)
    model = seeded_model(scenario, seed=2).eval()
    x = view_tensor(raw_stamps(3, seed=12), scenario).float()
    pos = positions(3)
    with torch.no_grad():
        expected = model(x, pos)
        moved = model.to(device)
        got = moved(x.to(device), pos.to(device)).cpu()
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    # A training step runs on the device and updates the Keras BN like the CPU does.
    moved.train()
    moved.dropout.p = 0.0
    cpu_model = seeded_model(scenario, seed=2).train()
    cpu_model.dropout.p = 0.0
    loss = moved(x.to(device), pos.to(device)).sum()
    loss.backward()
    cpu_model(x, pos)
    assert moved.bn.running_var is not None and cpu_model.bn.running_var is not None
    torch.testing.assert_close(
        moved.bn.running_var.cpu(), cpu_model.bn.running_var, rtol=1e-4, atol=1e-4
    )
    assert all(p.grad is not None for p in moved.parameters())


@pytest.mark.parametrize("accelerator", ACCELERATORS)
def test_tf_same_max_pool_on_device(accelerator: str) -> None:
    x = -torch.arange(2 * 3 * 63 * 63, dtype=torch.float32).reshape(2, 3, 63, 63)
    expected = TFSameMaxPool2d(2)(x)
    got = TFSameMaxPool2d(2)(x.to(accelerator)).cpu()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


# ----------------------------------------------------------------------------------------
# Errors and the engine-free rule


def _arch(**changes: object) -> RJ23Architecture:
    base = {
        "scenario": "multiscale",
        "layers": (4, 4),
        "dense": 3,
        "dropout": 0.5,
        "learning_rate": 1e-3,
    }
    base.update(changes)
    return RJ23Architecture(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: RJ23Classifier(_arch(layers=(4, MAXPOOL, 4))), "end on 8x8"),
        (lambda: RJ23Classifier(_arch(scenario="full")), "end on 8x8"),
        (lambda: RJ23Classifier(_arch(layers=(MAXPOOL, 4))), "start with a convolution"),
        (lambda: RJ23Classifier(_arch(layers=())), "start with a convolution"),
        (lambda: RJ23Classifier(_arch(layers=(4, 0))), ">= 1"),
        (lambda: RJ23Classifier(_arch(dense=0)), ">= 1"),
        (lambda: RJ23Classifier(_arch(dropout=1.0)), "dropout"),
        (lambda: RJ23Classifier(_arch(first_kernel_size=4)), "odd"),
        (lambda: RJ23Classifier(_arch(scenario="tiny")), "scenario"),
        (lambda: RJ23Classifier(_arch(), n_classes=1), "n_classes"),
        (lambda: RJ23Classifier(_arch(), bn_position="middle"), "bn_position"),  # type: ignore[arg-type]
        (lambda: build_rj23_classifier("tiny"), "scenario"),  # type: ignore[arg-type]
        (lambda: tf_same_padding(0, 2, 2), ">= 1"),
        (lambda: TFSameMaxPool2d(0), ">= 1"),
        (lambda: TFSameMaxPool2d(2)(torch.zeros(3, 4, 4)), r"\(N, C, H, W\)"),
        (lambda: dihedral_views(torch.zeros(1, 1, 4, 5)), "square"),
    ],
)
def test_errors(make: Callable[[], object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_forward_input_errors() -> None:
    model = build_rj23_classifier("cropped")
    with pytest.raises(ValueError, match=r"cropped model needs images of shape \(N, 6, 16, 16\)"):
        model(torch.zeros(2, 6, 8, 8), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="x_pos must be"):
        model(torch.zeros(2, 6, 16, 16), torch.zeros(2, 2))
    with pytest.raises(ValueError, match="x_pos must be"):
        model(torch.zeros(2, 6, 16, 16), torch.zeros(3, 3))


def test_module_does_not_import_lightning() -> None:
    code = (
        "import sys; import astro_simclr.rj23_model; "
        "sys.exit(1 if any('lightning' in m for m in sys.modules) else 0)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603
