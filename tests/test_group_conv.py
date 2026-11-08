"""Tests of ``group_conv``: the p4 / p4m group, the layers' equivariance and their fans.

The equivariance law checked for every layer and every group element ``g``: transforming
the input by ``g`` transforms each output map spatially by ``g`` and permutes the
orientation axis by ``h -> g^-1 h``, that is ``layer(T_g x)[:, f, h] = T_g(layer(x)[:, f,
g^-1 h])``. Biases are randomised first: the layers initialise them to 0, which would hide
a bias expanded in the wrong order.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from astro_simclr.group_conv import (
    GROUPS,
    Group,
    GroupConv,
    GroupReadout,
    LiftingConv,
    orientation_mean,
    transform,
)


def _randomise_biases(layer: LiftingConv | GroupConv) -> None:
    with torch.no_grad():
        layer.bias.uniform_(-0.5, 0.5)


def _regular_input(group: Group, fields: int, side: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        2, fields * group.order, side, side, generator=generator, dtype=torch.float64
    )


def _act_on_regular(group: Group, x: torch.Tensor, g: int) -> torch.Tensor:
    """The action of ``g`` on a regular map: spatial ``T_g`` and ``h -> g^-1 h`` on orientations."""
    n, channels, height, width = x.shape
    fields = x.reshape(n, channels // group.order, group.order, height, width)
    permuted = fields[:, :, [group.mul[group.inv[g]][h] for h in range(group.order)]]
    return group.apply(permuted, g).reshape(n, channels, height, width)


@pytest.mark.parametrize(("name", "order"), [("p4", 4), ("p4m", 8)])
def test_group_is_a_group(name: str, order: int) -> None:
    group = Group(name)
    assert group.order == order
    assert group.elements[0] == (0, 0)
    elements = range(order)
    for a in elements:
        assert group.mul[0][a] == group.mul[a][0] == a
        assert group.mul[a][group.inv[a]] == group.mul[group.inv[a]][a] == 0
    for a, b, c in itertools.product(elements, repeat=3):
        assert group.mul[group.mul[a][b]][c] == group.mul[a][group.mul[b][c]]
    # The table agrees with composing the transforms.
    probe = torch.arange(16.0).reshape(4, 4)
    for a, b in itertools.product(elements, repeat=2):
        assert torch.equal(
            group.apply(group.apply(probe, b), a), group.apply(probe, group.mul[a][b])
        )


def test_p4m_contains_the_flips_and_unknown_groups_fail() -> None:
    probe = torch.arange(9.0).reshape(3, 3)
    group = Group("p4m")
    assert any(torch.equal(group.apply(probe, g), probe.flip(-1)) for g in range(8))
    assert any(torch.equal(group.apply(probe, g), probe.T) for g in range(8))
    assert torch.equal(transform(probe, 1, 0), probe.flip(-1))
    with pytest.raises(ValueError, match="p4, p4m"):
        Group("c8")


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize(("kernel", "padding", "side"), [(4, 0, 9), (5, 2, 7), (3, 0, 8)])
def test_lifting_conv_is_equivariant(name: str, kernel: int, padding: int, side: int) -> None:
    group = Group(name)
    torch.manual_seed(0)
    layer = LiftingConv(group, 3, 2, kernel, padding).double()
    _randomise_biases(layer)
    x = torch.randn(2, 3, side, side, dtype=torch.float64)
    out = layer(x)
    assert out.shape == (2, 2 * group.order, side + 2 * padding - kernel + 1, out.shape[-1])
    for g in range(group.order):
        torch.testing.assert_close(
            layer(group.apply(x, g)), _act_on_regular(group, out, g), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize(("kernel", "padding", "side"), [(5, 2, 6), (3, 0, 7), (6, 0, 6)])
def test_group_conv_is_equivariant(name: str, kernel: int, padding: int, side: int) -> None:
    group = Group(name)
    torch.manual_seed(1)
    layer = GroupConv(group, 2, 3, kernel, padding).double()
    _randomise_biases(layer)
    x = _regular_input(group, 2, side)
    out = layer(x)
    assert out.shape[1] == 3 * group.order
    for g in range(group.order):
        torch.testing.assert_close(
            layer(_act_on_regular(group, x, g)), _act_on_regular(group, out, g), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("name", GROUPS)
def test_orientation_mean_of_a_readout_is_invariant(name: str) -> None:
    group = Group(name)
    torch.manual_seed(2)
    readout = GroupReadout(group, 2, 4, kernel_size=6).double()
    _randomise_biases(readout)
    x = _regular_input(group, 2, 6)
    expected = orientation_mean(readout(x), group.order)
    assert expected.shape == (2, 4, 1, 1)
    for g in range(group.order):
        torch.testing.assert_close(
            orientation_mean(readout(_act_on_regular(group, x, g)), group.order),
            expected,
            rtol=0,
            atol=1e-12,
        )
    with pytest.raises(ValueError, match="6x6 maps"):
        readout(_regular_input(group, 2, 7))


def test_expanded_kernels_are_plain_conv2d_kernels() -> None:
    group = Group("p4")
    lift = LiftingConv(group, 3, 8, 4)
    kernel, bias = lift.expanded()
    assert kernel.shape == (32, 3, 4, 4)
    assert bias.shape == (32,)
    # Orientation 0 is the stored filter; orientation r its rotation by r * 90 degrees.
    assert torch.equal(kernel.view(8, 4, 3, 4, 4)[:, 0], lift.weight)
    assert torch.equal(kernel.view(8, 4, 3, 4, 4)[:, 1], torch.rot90(lift.weight, 1, (-2, -1)))
    conv = GroupConv(group, 8, 16, 5, padding=2)
    assert conv.expanded()[0].shape == (64, 32, 5, 5)  # the paper's conv3 shape
    params = sum(p.numel() for p in conv.parameters())
    assert params == 16 * 8 * 4 * 25 + 16  # a quarter of the expanded kernel, plus biases


@pytest.mark.parametrize(
    ("layer", "fans"),
    [
        # Keras' convolution fans of the expanded kernel (out * |G|, in, k, k).
        (LiftingConv(Group("p4"), 3, 8, 4), (3 * 16, 8 * 4 * 16)),
        (GroupConv(Group("p4"), 8, 16, 5, 2), (8 * 4 * 25, 16 * 4 * 25)),
        (GroupConv(Group("p4m"), 4, 8, 5, 2), (4 * 8 * 25, 8 * 8 * 25)),
        # The readout is a dense layer: each input feeds each output once.
        (GroupReadout(Group("p4"), 16, 64, 6), (16 * 4 * 36, 64 * 4)),
    ],
)
def test_glorot_fans_and_bounds(layer: LiftingConv | GroupConv, fans: tuple[int, int]) -> None:
    assert layer.fans() == fans
    torch.manual_seed(0)
    layer.reset_parameters()
    bound = math.sqrt(6 / sum(fans))
    assert layer.weight.abs().max() <= bound
    assert layer.weight.abs().max() > 0.9 * bound
    assert torch.all(layer.bias == 0)
    # For the convolutions these are xavier_uniform_'s fans of the expanded kernel.
    if not isinstance(layer, GroupReadout):
        expanded = layer.expanded()[0]
        assert fans == torch.nn.init._calculate_fan_in_and_fan_out(expanded)


def test_orientation_mean_checks_the_channel_count() -> None:
    x = torch.arange(16.0).reshape(1, 8, 1, 2)
    torch.testing.assert_close(orientation_mean(x, 4)[0, :, 0, 0], torch.tensor([3.0, 11.0]))
    with pytest.raises(ValueError, match="multiple of the group order 3"):
        orientation_mean(x, 3)
