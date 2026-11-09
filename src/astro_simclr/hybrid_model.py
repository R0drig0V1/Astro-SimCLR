"""Conv-attention hybrid stamp classifiers (plan row B9 of task 2.5).

Hybrids keep a CNN's locality bias and add global self-attention on top: the transformer
family with the best small-data record (CCT, CoAtNet, MaxViT; the research report of
2026-11-08 behind rows B8 and B9). :class:`HybridStampClassifier` puts one in the paper
model's frame, so that it is one ``paper_arch`` flag against the replication
(``paper_variants.build_paper_model``):

1. **A conv trunk down to its last feature grid**, on the paper's 21x21 crop (zero-padded to
   27): 27 -> 24 -> 12 -> 6, so a 6x6 grid of 36 tokens.

   - ``hybrid``: the paper model's five convolutions (32/64 filters, ReLU, two 2x2
     max-pools): 64 channels per token.
   - ``hybrid_p4``: the p4 G-CNN trunk of ``GroupStampClassifier`` (plan row B7;
     ``group_fields`` F, default 8, so F/2F fields), then **the mean over the 4
     orientations of each field** (``group_conv.orientation_mean``): 2F channels per token
     (16 by default).
2. **Tokens:** a linear embedding to ``dim``, plus a learned 2-D positional embedding (one
   vector per grid cell for ``hybrid``; one per C4 orbit of cells for ``hybrid_p4``, point
   4). With ``readout="cls"`` a learned CLS token is prepended (it has no position).
3. **``depth`` pre-norm transformer encoder blocks** (:class:`EncoderBlock`): LayerNorm,
   ``heads``-head self-attention, residual; LayerNorm, an MLP (``dim`` -> 2 ``dim`` ->
   ``dim``, GELU), residual. A final LayerNorm, then the readout: the mean of the 36 grid
   tokens (``mean``) or the CLS token (``cls``).
4. **Exact C4 invariance.**

   - ``hybrid``: by rotation averaging, as the paper model. The 4 rotations of the batch go
     through trunk, attention and ``fc1`` as one batch of 4N, and the 64-unit outputs are
     averaged. The free positional table makes a single pass position-aware but not
     invariant; the mean over the 4 rotations makes the model invariant by construction.
   - ``hybrid_p4``: in a single pass. Rotating the stamp by 90 degrees rotates the G-CNN's
     maps spatially and shifts their orientation axis cyclically
     (``group_conv``). The orientation mean removes the shift, so the 36 token features are
     the same up to the rotation of the grid. The positional embedding is tied across C4:
     the 4 cells of each orbit of the 6x6 grid share one vector (9 orbits; :func:`c4_orbits`),
     so the embedded tokens are a permutation of the unrotated ones. Self-attention, the
     MLPs and LayerNorm are permutation-equivariant; the mean and the CLS readout are
     permutation-invariant. The logits are invariant up to rounding (the order of the sums).
     The 4-rotation mean would be redundant: it averages 4 equal representations, at 4
     times the cost. The alternative (no orientation pooling, 144 tokens of one orientation
     each, positions tied over the joint C4 orbits of cell and orientation) keeps more
     orientation information, at 16 times the attention cost. It is not built here.
5. ``fc1`` (``dim`` -> 64), ReLU and the paper's dropout layer (per rotation, before the
   mean), then **the paper's head** (:class:`~astro_simclr.paper_variants.PaperHead`): the
   frozen feature BN on the features that ``feature_set`` selects, fc2, fc3 and the logits.

========================  ==========  =============  ===============
arch                      Parameters  MACs per view  MACs per stamp
========================  ==========  =============  ===============
``rot4`` (the paper's)    441,075     52.6 M         210.6 M (4 views)
``p4`` (B7, 8/16 fields)  228,531     53.1 M         53.1 M (1 view)
``hybrid``                371,251     55.3 M         221.4 M (4 views)
``hybrid`` + ``cls``      371,315     55.4 M         221.7 M (4 views)
``hybrid`` dim 128, 4     844,915     73.0 M         292.0 M (4 views)
``hybrid_p4``             153,907     55.2 M         55.2 M (1 view)
========================  ==========  =============  ===============

(with the 23 features; ``dim`` 64, ``depth`` 2, ``heads`` 4 and the mean readout unless
stated; :func:`~astro_simclr.paper_variants.count_macs`, convolutions and matrix products
only, LayerNorm, softmax and GELU not counted; ``tests/test_hybrid_model.py`` pins the
defaults.) The hybrid has fewer parameters than the paper model: its 2304 -> 64 fc1 (147k
parameters) becomes the token embedding (4k), two attention blocks (66k) and a 64 -> 64
fc1. Attention adds about 2.7 M MACs per view (5 %) to the trunk's 52.5 M. ``hybrid_p4``
is a single pass, so it costs about a quarter of ``hybrid`` per stamp, about as much as
the p4 G-CNN; its trunk is that G-CNN's.

**Initialisation**, drawn from the global torch generator at construction, so
``seed_everything`` fixes it: the trunk as its model (Glorot-uniform kernels, zero biases;
``group_conv``'s fans for p4); the token embedding, the attention and MLP layers, the
positional table and the CLS token from a normal of std 0.02 truncated at 2 sigma (the ViT
convention), zero biases, LayerNorm at (1, 0); ``fc1`` and the head as the paper's. No
dropout in the blocks and no stochastic depth: the paper's dropout flags are the only
dropout.

Attention is written out (two batched matrix products and a softmax), not
``F.scaled_dot_product_attention``: with 36 or 37 tokens the fused kernels buy nothing,
the computation is the same on every backend, and ``count_macs`` sees its products. The
tied positional table is expanded by a one-hot matrix product, not by indexing, so its
backward is a matrix product too (the backward of ``index_select`` is an accumulating
scatter, which has no deterministic MPS kernel; ``group_conv``). Nothing here imports
Lightning (ADR 0014): the module is engine-free.
"""

from __future__ import annotations

import math
from typing import Final, cast

import torch
import torch.nn.functional as F
from torch import nn

from astro_simclr.config import (
    DEFAULT_HYBRID_DEPTH,
    DEFAULT_HYBRID_DIM,
    DEFAULT_HYBRID_HEADS,
    DEFAULT_HYBRID_READOUT,
    HYBRID_ARCHS,
    HYBRID_READOUTS,
)
from astro_simclr.feature_sets import feature_columns
from astro_simclr.group_conv import Group, GroupConv, LiftingConv, orientation_mean
from astro_simclr.models import N_CLASSES, check_n_classes
from astro_simclr.paper_model import (
    PAPER_DROP_RATE,
    PAPER_IN_CHANNELS,
    PAPER_PAD,
    check_paper_images,
)
from astro_simclr.paper_variants import PaperHead, default_group_fields

__all__ = [
    "GRID_SIDE",
    "HYBRID_ARCHS",
    "HYBRID_READOUTS",
    "MLP_RATIO",
    "EncoderBlock",
    "GridPositions",
    "HybridStampClassifier",
    "c4_orbits",
    "check_hybrid",
]

#: Side of the trunks' last feature grid (27 -> 24 -> 12 -> 6): 36 tokens.
GRID_SIDE: Final = 6
#: Hidden width of the blocks' MLP, in units of ``dim`` (CCT's small models use 1-3).
MLP_RATIO: Final = 2
#: Width of fc1 (the paper's 64 units): the image representation the head reads.
_HIDDEN: Final = 64
#: The 4 rotations of the input (0, 90, 180 and 270 degrees; no flips), as the paper model.
_N_ROTATIONS: Final = 4
#: Standard deviation of the ViT-style initialisation (truncated at 2 sigma).
_INIT_STD: Final = 0.02
#: The paper trunk's filters: conv1-conv2, then conv3-conv5.
_PAPER_WIDTHS: Final = (32, 64)


def check_hybrid(*, dim: int, depth: int, heads: int, readout: str) -> None:
    """Check the attention part's shape flags.

    Raises:
        ValueError: if ``dim``, ``depth`` or ``heads`` is below 1, ``dim`` is not a
            multiple of ``heads``, or ``readout`` is not one of :data:`HYBRID_READOUTS`.
    """
    for name, value in (("hybrid_dim", dim), ("hybrid_depth", depth), ("hybrid_heads", heads)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if dim % heads:
        raise ValueError(f"hybrid_dim ({dim}) must be a multiple of hybrid_heads ({heads})")
    if readout not in HYBRID_READOUTS:
        valid = ", ".join(HYBRID_READOUTS)
        raise ValueError(f"hybrid_readout must be one of {valid}, got {readout!r}")


def c4_orbits(side: int) -> torch.Tensor:
    """The C4 orbit of every cell of a ``side`` x ``side`` grid, row-major: (side^2,) int64.

    Cells that ``torch.rot90`` maps onto each other (about the grid centre) share an id;
    the ids are 0 .. n_orbits - 1, numbered by their smallest cell. A 6x6 grid has 9 orbits
    of 4 cells; an odd side adds the centre as an orbit of its own.

    Raises:
        ValueError: if ``side`` is below 1.
    """
    if side < 1:
        raise ValueError(f"side must be >= 1, got {side}")
    cells = torch.arange(side * side).view(side, side)
    smallest = torch.stack([torch.rot90(cells, k) for k in range(_N_ROTATIONS)]).amin(dim=0)
    _, orbit = torch.unique(smallest.flatten(), return_inverse=True)
    out: torch.Tensor = orbit
    return out


def _trunc_normal_(tensor: torch.Tensor) -> None:
    nn.init.trunc_normal_(tensor, std=_INIT_STD, a=-2 * _INIT_STD, b=2 * _INIT_STD)


def _glorot_bias_(bias: torch.Tensor) -> None:
    """The paper's fc1-fc3 bias initialiser: U(+-sqrt(3 / units)) (``paper_model``)."""
    limit = math.sqrt(3.0 / bias.shape[0])
    nn.init.uniform_(bias, -limit, limit)


class GridPositions(nn.Module):
    """A learned 2-D positional embedding of a square grid, optionally tied across C4.

    ``forward()`` returns the (side^2, dim) embedding of the row-major cells. Untied
    (``tied_c4=False``) every cell has its own vector: ``table`` is (side^2, dim). Tied,
    the cells of one C4 orbit (:func:`c4_orbits`) share one: ``table`` is (n_orbits, dim),
    expanded by the constant one-hot matrix ``assign`` (side^2, n_orbits), so the embedding
    is unchanged by ``torch.rot90`` of the grid exactly (each entry is a copy).
    """

    #: (side^2, n_orbits) one-hot cell -> orbit matrix of the tied table; ``None`` untied.
    assign: torch.Tensor | None

    def __init__(self, side: int, dim: int, *, tied_c4: bool) -> None:
        super().__init__()
        self.side = side
        self.tied_c4 = tied_c4
        if tied_c4:
            orbit = c4_orbits(side)
            assign = F.one_hot(orbit).to(torch.get_default_dtype())
            self.table = nn.Parameter(torch.empty(assign.shape[1], dim))
        else:
            assign = None
            self.table = nn.Parameter(torch.empty(side * side, dim))
        # Not persistent: it follows from side and tied_c4, and .double() converts it.
        self.register_buffer("assign", assign, persistent=False)

    def reset_parameters(self) -> None:
        """A normal of std 0.02 truncated at 2 sigma (the ViT convention)."""
        _trunc_normal_(self.table)

    def forward(self) -> torch.Tensor:
        """The (side^2, dim) embedding of the grid's cells."""
        if self.assign is None:
            return self.table
        out: torch.Tensor = self.assign @ self.table
        return out


class EncoderBlock(nn.Module):
    """A pre-norm transformer encoder block: multi-head self-attention, then an MLP.

    ``forward(x)`` maps (B, T, dim) to (B, T, dim): ``x + proj(attention(norm1(x)))``, then
    ``x + fc_out(gelu(fc_in(norm2(x))))``. The attention is scaled dot-product attention
    over all T tokens with ``heads`` heads of ``dim // heads`` channels. Permutation-
    equivariant over the tokens (it has no positional information of its own). No dropout.

    Raises:
        ValueError: at construction, if ``dim`` or ``heads`` is below 1 or ``dim`` is not
            a multiple of ``heads``.
    """

    def __init__(self, dim: int, heads: int, *, mlp_ratio: int = MLP_RATIO) -> None:
        super().__init__()
        check_hybrid(dim=dim, depth=1, heads=heads, readout=HYBRID_READOUTS[0])
        self.heads = heads
        self.scale = float((dim // heads) ** -0.5)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc_in = nn.Linear(dim, mlp_ratio * dim)
        self.fc_out = nn.Linear(mlp_ratio * dim, dim)

    def reset_parameters(self) -> None:
        """Linear weights from the truncated normal, zero biases, LayerNorm at (1, 0)."""
        for linear in (self.qkv, self.proj, self.fc_in, self.fc_out):
            _trunc_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) -> (B, T, dim)."""
        b, t, dim = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, T, dim // heads)
        weights = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(b, t, dim)
        x = x + self.proj(attended)
        out: torch.Tensor = x + self.fc_out(F.gelu(self.fc_in(self.norm2(x))))
        return out


class _PaperTrunk(nn.Module):
    """The paper model's convolutions: (B, C, 21, 21) -> (B, 64, 6, 6)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        narrow, wide = _PAPER_WIDTHS
        self.channels = wide
        self.pad = nn.ZeroPad2d(PAPER_PAD)
        self.conv1 = nn.Conv2d(in_channels, narrow, kernel_size=4)
        self.conv2 = nn.Conv2d(narrow, narrow, kernel_size=5, padding="same")
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(narrow, wide, kernel_size=5, padding="same")
        self.conv4 = nn.Conv2d(wide, wide, kernel_size=5, padding="same")
        self.conv5 = nn.Conv2d(wide, wide, kernel_size=5, padding="same")
        self.pool2 = nn.MaxPool2d(2)

    def reset_parameters(self) -> None:
        """Glorot-uniform kernels, zero biases (as ``PaperStampClassifier``)."""
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:  # always: built with bias=True
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(self.pad(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        out: torch.Tensor = self.pool2(F.relu(self.conv5(x)))
        return out


class _P4Trunk(nn.Module):
    """The p4 G-CNN trunk, orientation-pooled: (B, C, 21, 21) -> (B, 2F, 6, 6).

    ``GroupStampClassifier``'s layers up to its second pool (F/2F fields), then the mean
    over the 4 orientations of each field: the token features are C4-invariant per cell.
    """

    def __init__(self, in_channels: int, fields: int) -> None:
        super().__init__()
        group = self.group = Group("p4")
        narrow, wide = fields, 2 * fields
        self.channels = wide
        self.pad = nn.ZeroPad2d(PAPER_PAD)
        self.conv1 = LiftingConv(group, in_channels, narrow, kernel_size=4)
        self.conv2 = GroupConv(group, narrow, narrow, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = GroupConv(group, narrow, wide, kernel_size=5, padding=2)
        self.conv4 = GroupConv(group, wide, wide, kernel_size=5, padding=2)
        self.conv5 = GroupConv(group, wide, wide, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2)

    def reset_parameters(self) -> None:
        """``group_conv``'s Glorot-uniform kernels, zero biases (as ``GroupStampClassifier``)."""
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(self.pad(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(F.relu(self.conv5(x)))
        return orientation_mean(x, self.group.order)


class HybridStampClassifier(nn.Module):
    """A conv trunk, self-attention over its 6x6 grid, and the paper's head (module docstring).

    ``forward(x_img, x_feat)`` takes (N, ``in_channels``, 21, 21) crops and, with features,
    the (N, 23) features; it returns (N, ``n_classes``) logits that are invariant to the
    rotations of ``x_img`` by 90 degrees, up to rounding. ``views`` is the number of
    rotations a forward runs (4 for ``hybrid``, 1 for ``hybrid_p4``): ``count_macs``
    divided by it gives the MACs per view.

    Raises:
        ValueError: at construction, for an ``arch`` not in :data:`HYBRID_ARCHS`, invalid
            shape flags (:func:`check_hybrid`), ``group_fields`` with ``hybrid`` or below 1,
            ``drop_rate`` outside [0, 1), ``in_channels < 1``, ``n_classes < 2`` or an
            invalid ``feature_set``; in ``forward``, for images that are not
            (N, ``in_channels``, 21, 21), or with features, a missing or misshapen
            ``x_feat``.
    """

    #: The learned CLS token (1, 1, dim) with ``readout="cls"``; ``None`` with the mean.
    cls_token: nn.Parameter | None

    def __init__(
        self,
        arch: str = "hybrid",
        *,
        dim: int = DEFAULT_HYBRID_DIM,
        depth: int = DEFAULT_HYBRID_DEPTH,
        heads: int = DEFAULT_HYBRID_HEADS,
        readout: str = DEFAULT_HYBRID_READOUT,
        group_fields: int | None = None,
        with_features: bool = True,
        dropout_active: bool = False,
        drop_rate: float = PAPER_DROP_RATE,
        in_channels: int = PAPER_IN_CHANNELS,
        n_classes: int = N_CLASSES,
        feature_set: str = "all",
    ) -> None:
        super().__init__()
        if arch not in HYBRID_ARCHS:
            raise ValueError(f"arch must be one of {', '.join(HYBRID_ARCHS)}, got {arch!r}")
        check_hybrid(dim=dim, depth=depth, heads=heads, readout=readout)
        if not 0 <= drop_rate < 1:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        check_n_classes(n_classes)
        with_features = bool(feature_columns(feature_set, with_features=with_features))
        self.arch = arch
        self.readout = readout
        self.with_features = with_features
        self.dropout_active = dropout_active
        self.in_channels = in_channels
        p4 = arch == "hybrid_p4"
        self.views = 1 if p4 else _N_ROTATIONS
        self.trunk: _PaperTrunk | _P4Trunk
        if p4:
            fields = default_group_fields("p4") if group_fields is None else group_fields
            if fields < 1:
                raise ValueError(f"group_fields must be >= 1, got {fields}")
            self.trunk = _P4Trunk(in_channels, fields)
        else:
            if group_fields is not None:
                raise ValueError("group_fields applies to hybrid_p4, not to hybrid")
            self.trunk = _PaperTrunk(in_channels)
        self.embed = nn.Linear(self.trunk.channels, dim)
        self.positions = GridPositions(GRID_SIDE, dim, tied_c4=p4)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim)) if readout == "cls" else None
        self.blocks = nn.ModuleList(EncoderBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, _HIDDEN)
        self.drop1 = nn.Dropout(drop_rate)
        self.head = PaperHead(
            with_features=with_features,
            dropout_active=dropout_active,
            drop_rate=drop_rate,
            n_classes=n_classes,
            feature_set=feature_set,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise every parameter (module docstring), in network order."""
        self.trunk.reset_parameters()
        _trunc_normal_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)
        self.positions.reset_parameters()
        if self.cls_token is not None:
            _trunc_normal_(self.cls_token)
        for block in self.blocks:
            cast("EncoderBlock", block).reset_parameters()
        self.norm.reset_parameters()
        nn.init.xavier_uniform_(self.fc1.weight)
        _glorot_bias_(self.fc1.bias)
        self.head.reset_parameters()

    def grid(self, x_img: torch.Tensor) -> torch.Tensor:
        """(N, C, 21, 21) crops -> (N, channels, 6, 6): the trunk's last grid (one view)."""
        check_paper_images(x_img, self.in_channels, type(self).__name__)
        out: torch.Tensor = self.trunk(x_img)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, 21, 21) -> (B, dim) for one view: tokens, blocks, final norm, readout.

        No rotation and no input check (:meth:`representation` does both): for ``hybrid``
        this is not invariant on its own, for ``hybrid_p4`` it is.
        """
        tokens = self.embed(self.trunk(x).flatten(2).transpose(1, 2)) + self.positions()
        if self.cls_token is not None:
            tokens = torch.cat((self.cls_token.expand(len(tokens), -1, -1), tokens), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        out: torch.Tensor = tokens[:, 0] if self.cls_token is not None else tokens.mean(dim=1)
        return out

    def representation(self, x_img: torch.Tensor) -> torch.Tensor:
        """(N, C, 21, 21) crops -> (N, 64): fc1's ReLU outputs, averaged over the views."""
        check_paper_images(x_img, self.in_channels, type(self).__name__)
        n = x_img.shape[0]
        views = x_img
        if self.views > 1:
            views = torch.cat([torch.rot90(x_img, k, dims=(2, 3)) for k in range(self.views)])
        h = F.relu(self.fc1(self.encode(views)))
        if self.dropout_active:
            h = self.drop1(h)  # per rotation, before the mean (as the paper model)
        return h.view(self.views, n, _HIDDEN).mean(dim=0)

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Crops (N, C, 21, 21) and, with features, (N, 23) -> (N, n_classes) logits."""
        out: torch.Tensor = self.head(self.representation(x_img), x_feat)
        return out
