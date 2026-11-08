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
from typing import Final, Literal

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


class LiftingConv(nn.Module):
    """Image to regular map: ``(N, in_channels, H, W) -> (N, out_fields * |G|, H', W')``.

    ``weight`` is ``(out_fields, in_channels, k, k)`` and ``bias`` ``(out_fields,)``; the
    expanded kernel stacks ``T_g(weight)`` for every ``g``. ``padding`` is symmetric, as
    ``F.conv2d``'s integer padding. :meth:`reset_parameters` draws the Glorot-uniform kernel
    (module docstring) and zeroes the bias.
    """

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
        """The plain ``conv2d`` kernel ``(out_fields * |G|, in, k, k)`` and its bias."""
        order, k = self.group.order, self.kernel_size
        kernel = torch.stack([self.group.apply(self.weight, g) for g in range(order)], dim=1)
        kernel = kernel.reshape(self.out_fields * order, self.in_channels, k, k)
        return kernel, self.bias.repeat_interleave(order)

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
        # permutations[g][h] = g^-1 h: which stored input orientation filter g reads at h.
        self._permutations = tuple(
            [group.mul[group.inv[g]][h] for h in range(order)] for g in range(order)
        )
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
        """The plain ``conv2d`` kernel ``(out_fields * |G|, in_fields * |G|, k, k)`` and bias."""
        order, k = self.group.order, self.kernel_size
        # The permutation is a stack of slices, not an advanced index: the backward of an
        # index is an accumulating scatter, which has no deterministic MPS implementation
        # (Trainer(deterministic=True) would fail there).
        slices = self.weight.unbind(dim=2)
        kernel = torch.stack(
            [
                self.group.apply(torch.stack([slices[h] for h in self._permutations[g]], 2), g)
                for g in range(order)
            ],
            dim=1,
        )  # (out_fields, |G| out, in_fields, |G| in, k, k)
        kernel = kernel.reshape(self.out_fields * order, self.in_fields * order, k, k)
        return kernel, self.bias.repeat_interleave(order)

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
