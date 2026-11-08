"""Tests of ``group_conv``: the p4 / p4m group, the layers' equivariance and their fans.

The equivariance law checked for every layer and every group element ``g``: transforming
the input by ``g`` transforms each output map spatially by ``g`` and permutes the
orientation axis by ``h -> g^-1 h``, that is ``layer(T_g x)[:, f, h] = T_g(layer(x)[:, f,
g^-1 h])``. Biases are randomised first: the layers initialise them to 0, which would hide
a bias expanded in the wrong order.

The kernel expansion is one precomputed gather (``group_conv`` module docstring). It is
checked bit for bit, values and gradients, against the transform-by-transform expansion
it replaced (:func:`_transform_expansion`, kept here verbatim as the reference), for every
layer type of the p4 and p4m paper models, on the CPU and on MPS or CUDA when present, and
under ``torch.use_deterministic_algorithms(True)``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import pytest
import torch
from _training_support import ACCELERATORS

from astro_simclr.config import CONTEXTS
from astro_simclr.group_conv import (
    GROUPS,
    Group,
    GroupConv,
    GroupReadout,
    LiftingConv,
    _gather_indices,
    orientation_mean,
    transform,
)
from astro_simclr.paper_variants import build_paper_model


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


def _transform_expansion(layer: LiftingConv | GroupConv) -> torch.Tensor:
    """The expanded kernel by transforms: ``expanded()`` before the gather (commit 13eda35).

    Kept verbatim as the reference of the bit-equality tests: values and gradients. Its
    stack of slices is the old way around the advanced index's MPS scatter.
    """
    group, order, k = layer.group, layer.group.order, layer.kernel_size
    if isinstance(layer, LiftingConv):
        kernel = torch.stack([group.apply(layer.weight, g) for g in range(order)], dim=1)
        return kernel.reshape(layer.out_fields * order, layer.in_channels, k, k)
    permutations = tuple([group.mul[group.inv[g]][h] for h in range(order)] for g in range(order))
    slices = layer.weight.unbind(dim=2)
    kernel = torch.stack(
        [
            group.apply(torch.stack([slices[h] for h in permutations[g]], 2), g)
            for g in range(order)
        ],
        dim=1,
    )
    return kernel.reshape(layer.out_fields * order, layer.in_fields * order, k, k)


def _layers(name: str) -> Iterator[LiftingConv | GroupConv]:
    """Every group layer of the paper models of ``name`` (both contexts), then small ones."""
    for context in CONTEXTS:
        torch.manual_seed(0)
        model = build_paper_model(name, context=context)
        layers = [m for m in model.modules() if isinstance(m, LiftingConv | GroupConv)]
        assert {type(m) for m in layers} == {LiftingConv, GroupConv, GroupReadout}
        yield from layers
    group = Group(name)
    torch.manual_seed(1)
    yield from (
        LiftingConv(group, 3, 2, 5, padding=2),  # odd kernel: a centre pixel
        GroupConv(group, 2, 3, 3),
        GroupConv(group, 1, 1, 1),
    )


def _assert_gather_is_transform_expansion(layer: LiftingConv | GroupConv) -> None:
    kernel, bias = layer.expanded()
    reference = _transform_expansion(layer)
    assert kernel.dtype == reference.dtype
    assert torch.equal(kernel, reference)
    assert torch.equal(bias, layer.bias.repeat_interleave(layer.group.order))
    upstream = torch.randn(
        kernel.shape, generator=torch.Generator().manual_seed(2), dtype=kernel.dtype
    ).to(kernel.device)
    (grad,) = torch.autograd.grad(kernel, layer.weight, upstream)
    (reference_grad,) = torch.autograd.grad(reference, layer.weight, upstream)
    # Bit-equal, not close: the |G| copies are summed in the transform expansion's order.
    assert torch.equal(grad, reference_grad)


@pytest.mark.parametrize("device", ACCELERATORS)
@pytest.mark.parametrize("name", GROUPS)
def test_gathered_kernel_is_the_transform_expansion(name: str, device: str) -> None:
    for layer in _layers(name):
        _assert_gather_is_transform_expansion(layer.to(device))


@pytest.fixture
def deterministic() -> Iterator[None]:
    """``torch.use_deterministic_algorithms(True)`` for one test, then the previous mode."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("device", ACCELERATORS)
@pytest.mark.parametrize("name", GROUPS)
def test_gathered_kernel_backward_is_deterministic(name: str, device: str) -> None:
    # index_select's own backward (index_add_) raises here on MPS; the gather's does not.
    for layer in _layers(name):
        layer.to(device)
        _assert_gather_is_transform_expansion(layer)
        kernel = layer.expanded()[0]
        grads = [
            torch.autograd.grad(layer.expanded()[0], layer.weight, torch.ones_like(kernel))[0]
            for _ in range(2)
        ]
        assert torch.equal(grads[0], grads[1])


def test_gather_buffers_follow_the_module_and_stay_out_of_checkpoints() -> None:
    layer = GroupConv(Group("p4m"), 2, 3, 5, padding=2)
    assert set(layer.state_dict()) == {"weight", "bias"}
    # 2 x 3 x 8 x 25 parameters, each copied 8 times: 64-bit index and its inverse.
    assert layer._kernel_index.shape == (3 * 8 * 2 * 8 * 25,)
    assert layer._kernel_inverse.shape == (8, 2 * 3 * 8 * 25)
    layer.double()
    assert layer._kernel_index.dtype == layer._kernel_inverse.dtype == torch.int64
    assert layer.expanded()[0].dtype == torch.float64
    fresh = GroupConv(Group("p4m"), 2, 3, 5, padding=2)
    fresh.load_state_dict(layer.float().state_dict())  # strict: no missing buffer keys
    assert torch.equal(fresh.expanded()[0], layer.expanded()[0])


def test_gather_indices_reject_a_layout_that_is_not_a_g_fold_copy() -> None:
    shape = torch.Size((2, 3))
    index, inverse = _gather_indices(torch.arange(6).repeat(2), shape, 2)
    assert torch.equal(index, torch.arange(6).repeat(2))
    # Each entry's copies, from the last position to the first.
    assert torch.equal(inverse, torch.stack([torch.arange(6, 12), torch.arange(6)]))
    with pytest.raises(RuntimeError, match="every entry 2 times"):
        _gather_indices(torch.tensor([0, 0, 1, 2, 3, 4, 5, 5, 5, 1, 2, 3]), shape, 2)
