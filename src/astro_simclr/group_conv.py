"""Hand-written p4 and p4m group-equivariant convolutions (task 2.5-models).

Cohen & Welling 2016 ("Group equivariant convolutional networks") for the two groups of
square-pixel symmetries: p4 (the 4 rotations by 90 degrees, C4) and p4m (rotations and
mirror flips, D4). Every layer expands its parameters into an ordinary convolution kernel
and runs one plain ``F.conv2d``, so it works on every backend torch supports (CPU, CUDA,
MPS) without a new dependency (no escnn: the portability rule of phase 1, and escnn's
``py3nj`` needs a Fortran compiler). The forward cost is exactly that of a plain CNN with
``fields * |G|`` channels.

Conventions:

- A group element ``g = (m, r)`` acts on the last two axes of an array as
  ``T_g(x) = rot90(flip^m(x), r)``, where ``flip`` reverses the last axis and ``rot90`` is
  ``torch.rot90`` (counter-clockwise, the rotation of the paper model). p4 has ``m = 0``
  only (4 elements), p4m has 8. Element 0 is the identity.
- A *regular* feature map of ``F`` fields has ``F * |G|`` channels in ``(field, g)``
  order: channel ``f * |G| + g`` is the response of field ``f`` to the filter transformed by
  ``g``.
- Lifting convolution (an image to a regular map): ``out[o, g] = T_g(W[o]) * x``.
- Group convolution (regular to regular): ``out[o, g] = sum_h T_g(W[o, g^-1 h]) * f[h]``,
  where ``*`` is torch's cross-correlation over the input fields.

With these, transforming the input by ``g`` transforms every regular map spatially by
``g`` and permutes its orientation axis by left multiplication. :func:`orientation_mean`
(the mean over the ``|G|`` maps of each field) is then invariant to the orientation part.

**Kernel expansion as one gather.** The expanded kernel is pure data movement: every entry
is a copy of one parameter entry. Each layer therefore computes, once at construction, the
flat source index of every expanded entry (by running the transforms above on an
``arange`` of the parameter's shape) and keeps it as a non-persistent buffer, so it follows
the module across devices and stays out of ``state_dict``. A call is then one
``index_select`` of the flattened parameter instead of up to ~``|G|^2`` ``select`` / ``rot90``
/ ``stack`` ops per layer (p4's training step issued ~200 of them). The values are the
same bits as the transform-by-transform expansion. The backward is not ``index_select``'s
own: that is an accumulating scatter (``index_add_``), which has no deterministic MPS
kernel (``Trainer(deterministic=True)`` would fail there) and sums the ``|G|`` copies of
each parameter entry in a different order than the transform expansion's autograd graph,
so the gradients would change in the last bits. :class:`_KernelGather` instead gathers the
``|G|`` copies with a precomputed inverse index and adds them in a fixed order, the one in
which autograd accumulated them for the transform expansion (from the last output
orientation to the first): no scatter, deterministic on CPU, CUDA and MPS, and the same
gradient bits (``tests/test_group_conv.py`` checks both against the transform expansion).

**Geometry.** Exact equivariance also needs every spatial operation to map its sampling
grid onto itself under the rotations (and flips) about the image centre: valid
convolutions of any kernel size, "same" convolutions with odd kernels and symmetric
padding, and 2x2 pooling of an even side. The paper's geometry (27 -> 24 -> 12 -> 6)
satisfies it; ``tests/test_paper_variants.py`` checks the whole model numerically, because
an odd side before a 2x2 pool or an asymmetric pad breaks it silently.

**Initialisation (Glorot-uniform, as the faithful model).** The fans are those of the
expanded linear map, so a group layer starts at the scale of the plain layer it replaces:

- :class:`LiftingConv` and :class:`GroupConv` follow Keras' convolution convention on the
  expanded kernel ``(out_fields * |G|, in_channels, k, k)``: ``fan_in = in_channels * k^2``
  and ``fan_out = out_fields * |G| * k^2`` (``in_channels`` is ``in_fields * |G|`` for a
  group convolution). These are ``nn.init.xavier_uniform_``'s fans of the expanded kernel.
- :class:`GroupReadout` (the paper's fc1 recast as a group convolution whose kernel covers
  the whole map, so the output is 1x1) is a dense layer in disguise: each input unit feeds
  each output unit once, not ``k^2`` times. Its fans are those of the dense layer,
  ``fan_in = in_fields * |G| * k^2`` and ``fan_out = out_fields * |G|``.

Each parameter entry appears ``|G|`` times in the expanded kernel, so the per-entry variance
is that of the expanded map. Biases are one per field (shared over the orientations, as
equivariance requires) and are initialised by the model, not here (zero by default).
"""

from __future__ import annotations

import math
from typing import Any, Final, Literal

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "GROUPS",
    "Group",
    "GroupConv",
    "GroupName",
    "GroupReadout",
    "LiftingConv",
    "orientation_mean",
    "transform",
]

#: The supported group names.
GroupName = Literal["p4", "p4m"]
#: :data:`GroupName` values, in order.
GROUPS: Final[tuple[GroupName, ...]] = ("p4", "p4m")


def transform(x: torch.Tensor, flip: int, rotation: int) -> torch.Tensor:
    """``rot90(flip^flip(x), rotation)`` on the last two axes (see the module docstring)."""
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, rotation, dims=(-2, -1))


class Group:
    """p4 or p4m acting on square arrays: its elements, Cayley table and inverses.

    ``elements[g]`` is ``(m, r)``; ``mul[a][b]`` is the index of ``a * b``, defined by
    ``T_{a*b} = T_a . T_b``; ``inv[a]`` the index of ``a^-1``. The table is computed by
    applying the transforms to an asymmetric probe, so it is consistent with
    :func:`transform` by construction.

    Raises:
        ValueError: for a name other than ``p4`` or ``p4m``.
    """

    def __init__(self, name: str) -> None:
        if name not in GROUPS:
            raise ValueError(f"group must be one of {', '.join(GROUPS)}, got {name!r}")
        self.name = name
        flips = (0,) if name == "p4" else (0, 1)
        self.elements: tuple[tuple[int, int], ...] = tuple((m, r) for m in flips for r in range(4))
        probe = torch.arange(9.0).reshape(3, 3)  # no symmetry: each transform is distinct
        images = [self.apply(probe, g) for g in range(self.order)]

        def index(image: torch.Tensor) -> int:
            return next(k for k, candidate in enumerate(images) if torch.equal(candidate, image))

        self.mul: tuple[tuple[int, ...], ...] = tuple(
            tuple(index(self.apply(self.apply(probe, b), a)) for b in range(self.order))
            for a in range(self.order)
        )
        self.inv: tuple[int, ...] = tuple(self.mul[a].index(0) for a in range(self.order))

    @property
    def order(self) -> int:
        """``|G|``: 4 for p4, 8 for p4m."""
        return len(self.elements)

    def apply(self, x: torch.Tensor, g: int) -> torch.Tensor:
        """``T_g(x)`` on the last two axes of ``x``."""
        flip, rotation = self.elements[g]
        return transform(x, flip, rotation)


def _glorot_(weight: torch.Tensor, fan_in: int, fan_out: int) -> None:
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -bound, bound)


def _lifting_layout(group: Group, weight: torch.Tensor) -> torch.Tensor:
    """:class:`LiftingConv`'s expansion by transforms: ``(O, I, k, k) -> (O |G|, I, k, k)``."""
    out_fields, in_channels, k, _ = weight.shape
    kernel = torch.stack([group.apply(weight, g) for g in range(group.order)], dim=1)
    return kernel.reshape(out_fields * group.order, in_channels, k, k)


def _group_layout(group: Group, weight: torch.Tensor) -> torch.Tensor:
    """:class:`GroupConv`'s expansion by transforms: ``(O, I, |G|, k, k) -> (O |G|, I |G|, k, k)``.

    For output orientation ``g`` the input orientations are permuted by ``h -> g^-1 h`` and
    the result transformed by ``T_g`` (module docstring). Used on an index tensor only, so
    the advanced index has no backward.
    """
    out_fields, in_fields, order, k, _ = weight.shape
    kernel = torch.stack(
        [
            group.apply(weight[:, :, [group.mul[group.inv[g]][h] for h in range(order)]], g)
            for g in range(order)
        ],
        dim=1,
    )  # (out_fields, |G| out, in_fields, |G| in, k, k)
    return kernel.reshape(out_fields * order, in_fields * order, k, k)


def _gather_indices(
    layout: torch.Tensor, shape: torch.Size, order: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The forward and inverse indices of :class:`_KernelGather` for an expansion ``layout``.

    ``layout`` is the expansion of ``arange(numel).reshape(shape)``, so its flat entries
    are the source index of each expanded entry (the forward index). The inverse index,
    ``(order, numel)``, lists for each parameter entry the ``order`` flat positions that
    copy it, from the last position to the first: the order in which the transform
    expansion's autograd graph accumulated them (module docstring).

    Raises:
        RuntimeError: if some parameter entry is not copied exactly ``order`` times (a
            layout bug; the gather would then not be the transform expansion).
    """
    index = layout.reshape(-1)
    numel = math.prod(shape)
    if not torch.equal(torch.bincount(index, minlength=numel), torch.full((numel,), order)):
        raise RuntimeError(f"the kernel layout does not copy every entry {order} times")
    positions = torch.argsort(index, stable=True).reshape(numel, order)
    return index, positions.flip(1).T.contiguous()


class _KernelGather(torch.autograd.Function):
    """``weight.flatten()[index]`` whose backward is a gather and an ordered sum, not a scatter.

    ``forward(weight, index, inverse)`` returns the flat expanded kernel. ``backward``
    gathers the ``|G|`` copies of every parameter entry's gradient with ``inverse`` and adds
    them row by row in ``inverse``'s order: deterministic on every backend, and bit-equal
    to the gradient of the transform expansion (module docstring).
    """

    @staticmethod
    def forward(
        ctx: Any, weight: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(inverse)
        ctx.weight_shape = weight.shape
        return weight.reshape(-1).index_select(0, index)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (inverse,) = ctx.saved_tensors
        copies = grad.reshape(-1).index_select(0, inverse.reshape(-1)).view(inverse.shape)
        total = copies[0]
        for row in copies[1:]:  # a fixed order, not a reduction kernel's
            total = total + row
        return total.view(ctx.weight_shape), None, None


def _gathered_kernel(
    weight: torch.Tensor, index: torch.Tensor, inverse: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    """The expanded kernel of ``shape``: one :class:`_KernelGather` of ``weight``."""
    flat: torch.Tensor = _KernelGather.apply(weight, index, inverse)
    return flat.view(shape)


class LiftingConv(nn.Module):
    """Image to regular map: ``(N, in_channels, H, W) -> (N, out_fields * |G|, H', W')``.

    ``weight`` is ``(out_fields, in_channels, k, k)`` and ``bias`` ``(out_fields,)``; the
    expanded kernel stacks ``T_g(weight)`` for every ``g``. ``padding`` is symmetric, as
    ``F.conv2d``'s integer padding. :meth:`reset_parameters` draws the Glorot-uniform kernel
    (module docstring) and zeroes the bias.
    """

    _kernel_index: torch.Tensor
    _kernel_inverse: torch.Tensor

    def __init__(
        self, group: Group, in_channels: int, out_fields: int, kernel_size: int, padding: int = 0
    ) -> None:
        super().__init__()
        self.group = group
        self.in_channels = in_channels
        self.out_fields = out_fields
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_fields, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_fields))
        layout = _lifting_layout(group, torch.arange(self.weight.numel()).view(self.weight.shape))
        self._kernel_shape = layout.shape
        index, inverse = _gather_indices(layout, self.weight.shape, group.order)
        # Not persistent: derived from the shapes, so checkpoints keep only the parameters.
        self.register_buffer("_kernel_index", index, persistent=False)
        self.register_buffer("_kernel_inverse", inverse, persistent=False)
        self.reset_parameters()

    def fans(self) -> tuple[int, int]:
        """``(fan_in, fan_out)`` of the Glorot initialisation (module docstring)."""
        area = self.kernel_size**2
        return self.in_channels * area, self.out_fields * self.group.order * area

    def reset_parameters(self) -> None:
        """Glorot-uniform kernel with :meth:`fans`, zero bias."""
        _glorot_(self.weight, *self.fans())
        nn.init.zeros_(self.bias)

    def expanded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The plain ``conv2d`` kernel ``(out_fields * |G|, in, k, k)`` and its bias.

        One gather of :attr:`weight` (module docstring): the kernel stacks ``T_g(weight)``
        over ``g`` along the output fields.
        """
        kernel = _gathered_kernel(
            self.weight, self._kernel_index, self._kernel_inverse, self._kernel_shape
        )
        return kernel, self.bias.repeat_interleave(self.group.order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One ``F.conv2d`` with the expanded kernel."""
        kernel, bias = self.expanded()
        return F.conv2d(x, kernel, bias, padding=self.padding)


class GroupConv(nn.Module):
    """Regular map to regular map: ``(N, in_fields * |G|, H, W) -> (N, out_fields * |G|, ...)``.

    ``weight`` is ``(out_fields, in_fields, |G|, k, k)`` and ``bias`` ``(out_fields,)``. For
    output orientation ``g``, the filter is ``T_g`` applied to the weight with its input
    orientations permuted by ``h -> g^-1 h`` (module docstring). :meth:`reset_parameters`
    draws the Glorot-uniform kernel with :meth:`fans` and zeroes the bias.
    """

    _kernel_index: torch.Tensor
    _kernel_inverse: torch.Tensor

    def __init__(
        self, group: Group, in_fields: int, out_fields: int, kernel_size: int, padding: int = 0
    ) -> None:
        super().__init__()
        self.group = group
        self.in_fields = in_fields
        self.out_fields = out_fields
        self.kernel_size = kernel_size
        self.padding = padding
        order = group.order
        self.weight = nn.Parameter(
            torch.empty(out_fields, in_fields, order, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_fields))
        layout = _group_layout(group, torch.arange(self.weight.numel()).view(self.weight.shape))
        self._kernel_shape = layout.shape
        index, inverse = _gather_indices(layout, self.weight.shape, order)
        # Not persistent: derived from the shapes, so checkpoints keep only the parameters.
        self.register_buffer("_kernel_index", index, persistent=False)
        self.register_buffer("_kernel_inverse", inverse, persistent=False)
        self.reset_parameters()

    def fans(self) -> tuple[int, int]:
        """``(fan_in, fan_out)``: Keras' convolution fans of the expanded kernel."""
        order, area = self.group.order, self.kernel_size**2
        return self.in_fields * order * area, self.out_fields * order * area

    def reset_parameters(self) -> None:
        """Glorot-uniform kernel with :meth:`fans`, zero bias."""
        _glorot_(self.weight, *self.fans())
        nn.init.zeros_(self.bias)

    def expanded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The plain ``conv2d`` kernel ``(out_fields * |G|, in_fields * |G|, k, k)`` and bias.

        One gather of :attr:`weight` (module docstring), laid out as :func:`_group_layout`.
        """
        kernel = _gathered_kernel(
            self.weight, self._kernel_index, self._kernel_inverse, self._kernel_shape
        )
        return kernel, self.bias.repeat_interleave(self.group.order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One ``F.conv2d`` with the expanded kernel."""
        kernel, bias = self.expanded()
        return F.conv2d(x, kernel, bias, padding=self.padding)


class GroupReadout(GroupConv):
    """A valid :class:`GroupConv` whose kernel covers the whole map: a dense layer, 1x1 output.

    The group version of the paper's fc1: ``(N, in_fields * |G|, k, k) ->
    (N, out_fields * |G|, 1, 1)``. Only the initialisation differs from :class:`GroupConv`:
    its fans are the dense layer's (module docstring).
    """

    def __init__(self, group: Group, in_fields: int, out_fields: int, kernel_size: int) -> None:
        super().__init__(group, in_fields, out_fields, kernel_size, padding=0)

    def fans(self) -> tuple[int, int]:
        """``(fan_in, fan_out)`` of the dense map: ``in_fields |G| k^2`` and ``out_fields |G|``."""
        order = self.group.order
        return self.in_fields * order * self.kernel_size**2, self.out_fields * order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The convolution; the map must be ``kernel_size`` square (the output is 1x1).

        Raises:
            ValueError: if the spatial size of ``x`` is not ``kernel_size`` x ``kernel_size``.
        """
        k = self.kernel_size
        if tuple(x.shape[-2:]) != (k, k):
            raise ValueError(f"GroupReadout needs {k}x{k} maps, got {tuple(x.shape[-2:])}")
        return super().forward(x)


def orientation_mean(x: torch.Tensor, order: int) -> torch.Tensor:
    """The mean over the ``order`` orientations of each field: (N, F * order, ...) -> (N, F, ...).

    Raises:
        ValueError: if the channel count is not a multiple of ``order``.
    """
    channels = x.shape[1]
    if channels % order:
        raise ValueError(f"{channels} channels are not a multiple of the group order {order}")
    return x.reshape(x.shape[0], channels // order, order, *x.shape[2:]).mean(dim=2)
