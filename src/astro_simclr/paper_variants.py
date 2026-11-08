"""Efficient group-equivariant variants of the paper model (task 2.5-models: B7, B2', B1').

The faithful model (:class:`~astro_simclr.paper_model.PaperStampClassifier`, ``rot4``)
gets its rotation invariance by running the trunk on the 4 rotations of every stamp and
averaging fc1's outputs: 4 passes, 211 M MACs per stamp. A group-equivariant CNN (G-CNN;
:mod:`astro_simclr.group_conv`) is invariant by construction in a single pass. The
variants here keep everything else of the paper model, so each one is a single flag
against the replication (``docs/phases/02-stamp-classifier.md``, task 2.5):

==========  =====================================================  ==========  ========
arch        Trunk                                                  Parameters  MACs
==========  =====================================================  ==========  ========
``rot4``    the paper's (4 rotations, mean after fc1; faithful)    441,075     210.6 M
``single``  the paper's trunk, one pass, no mean (not invariant)   441,075     52.7 M
``p4``      p4 G-CNN, 8/16 fields (32/64 maps): **B7**             228,531     53.1 M
``p4m``     p4m G-CNN, 4/8 fields (32/64 maps): **B2'**            193,107     53.7 M
==========  =====================================================  ==========  ========

(with features; MACs per stamp from :func:`count_macs`.) :class:`GroupStampClassifier` is
the G-CNN: the paper's geometry, 21 zero-padded to 27, a 4x4 valid lifting convolution
(-> 24), a 5x5 "same" group convolution, 2x2 max-pool (-> 12), three 5x5 group
convolutions, 2x2 max-pool (-> 6); fc1 becomes a 6x6 group convolution to 64 fields
(:class:`~astro_simclr.group_conv.GroupReadout`), ReLU and the dropout layer per
orientation, then the mean over the orientations (the analogue of the paper's mean over
the 4 rotations after fc1). The head is the paper's, unchanged (:class:`PaperHead`: the
frozen feature BN, fc2, fc3 and the logits, with their dropout layers and
initialisation). ``group_fields`` is the number of fields of conv1-conv2; conv3-conv5 have
twice as many, like the paper's 32/64 filters. The default, ``32 // |G|``, gives the
paper's 32/64 *maps* (8/16 fields for p4, 4/8 for p4m), so the expanded convolutions have
the paper's shapes and about a quarter of its MACs; ``group_fields=16`` with p4 (16/32
fields) costs about as much as ``rot4`` (the capacity control arm).

**Context (B1').** ``context="avgpool63"`` adds the full 63x63 stamp, average-pooled to
21x21 (``data.float_input.context_split``), as 3 more input channels, after the crop's.
Only conv1 widens (+1.7 % MACs for p4). The pooled blocks tile the stamp symmetrically
(63 = 3 x 21), so rotating the stamp rotates the context like the crop, and the invariant
models stay invariant.

:data:`VARIANTS` names the rows of the plan (the models above, plus the control arms) as
``Settings`` overrides; ``scripts/bench_paper_models.py`` measures them. The ``Settings``
fields and their checks live in :mod:`astro_simclr.config` (ADR 0005).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from astro_simclr.config import CONTEXTS, PAPER_ARCHS
from astro_simclr.group_conv import (
    GROUPS,
    Group,
    GroupConv,
    GroupReadout,
    LiftingConv,
    orientation_mean,
)
from astro_simclr.models import N_CLASSES, N_FEATURES, check_n_classes
from astro_simclr.paper_model import (
    PAPER_CROP,
    PAPER_DROP_RATE,
    PAPER_IN_CHANNELS,
    PAPER_PAD,
    PaperStampClassifier,
    check_paper_images,
)

__all__ = [
    "CONTEXTS",
    "PAPER_ARCHS",
    "VARIANTS",
    "GroupStampClassifier",
    "PaperHead",
    "Variant",
    "build_paper_model",
    "count_macs",
    "count_parameters",
    "default_group_fields",
    "paper_in_channels",
]

#: Width of fc1-fc3 (the paper's 64 units); also the readout's number of fields.
_HIDDEN: Final = 64
#: Keras ``BatchNormalization`` epsilon of the feature BN (as ``paper_model``).
_BN_EPS: Final = 1e-3
#: Side of the last feature map (27 -> 24 -> 12 -> 6), the readout's kernel.
_LAST_SIDE: Final = 6
#: Maps of conv1-conv2 in the paper model (``rot4``); the default G-CNN width matches them.
_PAPER_MAPS: Final = 32


def _glorot_bias_(bias: torch.Tensor) -> None:
    """Keras' fallback for fc1-fc3's biases: U(+-sqrt(3 / units)) (``paper_model``)."""
    limit = math.sqrt(3.0 / bias.shape[0])
    nn.init.uniform_(bias, -limit, limit)


def paper_in_channels(context: str) -> int:
    """The model's input channels for ``context``: 3 (the crop), 6 with the 63x63 context.

    Raises:
        ValueError: for a context not in :data:`CONTEXTS`.
    """
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {', '.join(CONTEXTS)}, got {context!r}")
    return PAPER_IN_CHANNELS * (1 if context == "none" else 2)


def default_group_fields(arch: str) -> int:
    """``group_fields`` when unset: ``32 // |G|``, the paper's 32/64 maps (p4 8, p4m 4).

    Raises:
        ValueError: if ``arch`` is not a G-CNN (``p4`` or ``p4m``).
    """
    if arch not in GROUPS:
        raise ValueError(f"only the G-CNNs ({', '.join(GROUPS)}) have fields, not {arch!r}")
    return _PAPER_MAPS // Group(arch).order


class PaperHead(nn.Module):
    """The paper model's head on a 64-d representation: features BN, fc2, fc3, logits.

    The same layers, dropout and initialisation as ``PaperStampClassifier``'s head
    (``paper_model``; ``tests/test_paper_variants.py`` checks the equivalence): with
    features, ``bn_features`` (``BatchNorm1d(23, eps=1e-3, momentum=0.0)``, running
    statistics frozen at (0, 1)) is concatenated (64 + 23 = 87); fc2 and fc3 to 64 with
    ReLU and a dropout layer each (applied only when ``dropout_active``, in train mode);
    then the ``n_classes`` logits (default 5). Glorot-uniform weights, Glorot-uniform
    fc2/fc3 biases, zero logits bias.

    Raises:
        ValueError: in ``forward``, with features, if ``x_feat`` is missing or not (N, 23).
    """

    def __init__(
        self,
        *,
        with_features: bool,
        dropout_active: bool,
        drop_rate: float = PAPER_DROP_RATE,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        check_n_classes(n_classes)
        self.dropout_active = dropout_active
        self.bn_features = (
            nn.BatchNorm1d(N_FEATURES, eps=_BN_EPS, momentum=0.0) if with_features else None
        )
        self.fc2 = nn.Linear(_HIDDEN + (N_FEATURES if with_features else 0), _HIDDEN)
        self.drop2 = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(_HIDDEN, _HIDDEN)
        self.drop3 = nn.Dropout(drop_rate)
        self.logits = nn.Linear(_HIDDEN, n_classes)

    def reset_parameters(self) -> None:
        """The paper's initialisation, in network order (fc2, fc3, logits, BN)."""
        for dense in (self.fc2, self.fc3):
            nn.init.xavier_uniform_(dense.weight)
            _glorot_bias_(dense.bias)
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        if self.bn_features is not None:
            self.bn_features.reset_parameters()

    def _dropout(self, layer: nn.Dropout, x: torch.Tensor) -> torch.Tensor:
        return layer(x) if self.dropout_active else x

    def forward(self, h: torch.Tensor, x_feat: torch.Tensor | None) -> torch.Tensor:
        """(N, 64) and, with features, (N, 23) -> (N, n_classes) logits."""
        n = h.shape[0]
        if self.bn_features is not None:
            if x_feat is None or tuple(x_feat.shape) != (n, N_FEATURES):
                got = None if x_feat is None else tuple(x_feat.shape)
                msg = f"with_features=True needs x_feat of shape (N, {N_FEATURES}); got {got}"
                raise ValueError(msg)
            h = torch.cat((h, self.bn_features(x_feat)), dim=1)
        h = self._dropout(self.drop2, F.relu(self.fc2(h)))
        h = self._dropout(self.drop3, F.relu(self.fc3(h)))
        out: torch.Tensor = self.logits(h)
        return out


class GroupStampClassifier(nn.Module):
    """The paper model with a single-pass p4 or p4m G-CNN trunk (B7, B2'; module docstring).

    ``forward(x_img, x_feat)`` takes (N, ``in_channels``, 21, 21) crops and, with features,
    (N, 23) features; it returns (N, 5) logits that are exactly invariant to the rotations
    of ``x_img`` by 90 degrees (p4), and also to its mirror flips (p4m), up to rounding.

    Initialisation (:meth:`reset_parameters`, in network order, from the global torch
    generator): the Glorot-uniform kernels of :mod:`astro_simclr.group_conv` (fans of the
    expanded linear map; the readout as a dense layer), zero convolution biases, the
    readout's 64 field biases U(+-sqrt(3/64)) like the paper's fc1, and :class:`PaperHead`.

    Raises:
        ValueError: at construction, for an unknown group, ``group_fields < 1``,
            ``in_channels < 1``, ``n_classes < 2`` or ``drop_rate`` outside [0, 1); in
            ``forward``, for images
            that are not (N, ``in_channels``, 21, 21), or with features, a missing or
            misshapen ``x_feat``.
    """

    def __init__(
        self,
        group: str = "p4",
        *,
        group_fields: int | None = None,
        with_features: bool = True,
        dropout_active: bool = False,
        drop_rate: float = PAPER_DROP_RATE,
        in_channels: int = PAPER_IN_CHANNELS,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        if not 0 <= drop_rate < 1:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        fields = default_group_fields(group) if group_fields is None else group_fields
        if fields < 1:
            raise ValueError(f"group_fields must be >= 1, got {fields}")
        g = self.group = Group(group)
        self.with_features = with_features
        self.dropout_active = dropout_active
        self.in_channels = in_channels
        self.fields = (fields, 2 * fields)
        narrow, wide = self.fields
        self.pad = nn.ZeroPad2d(PAPER_PAD)
        self.conv1 = LiftingConv(g, in_channels, narrow, kernel_size=4)
        self.conv2 = GroupConv(g, narrow, narrow, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = GroupConv(g, narrow, wide, kernel_size=5, padding=2)
        self.conv4 = GroupConv(g, wide, wide, kernel_size=5, padding=2)
        self.conv5 = GroupConv(g, wide, wide, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = GroupReadout(g, wide, _HIDDEN, kernel_size=_LAST_SIDE)
        self.drop1 = nn.Dropout(drop_rate)
        self.head = PaperHead(
            with_features=with_features,
            dropout_active=dropout_active,
            drop_rate=drop_rate,
            n_classes=n_classes,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise every parameter (class docstring), conv1 ... fc1, then the head."""
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            conv.reset_parameters()
        self.fc1.reset_parameters()
        _glorot_bias_(self.fc1.bias)
        self.head.reset_parameters()

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Crops and, with features, ``x_feat`` -> (N, n_classes) logits; no features: ignored."""
        check_paper_images(x_img, self.in_channels, type(self).__name__)
        x = self.pad(x_img)
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(F.relu(self.conv5(x)))
        h = F.relu(self.fc1(x)).flatten(1)  # (N, 64 * |G|): every orientation of each field
        if self.dropout_active:
            h = self.drop1(h)  # per orientation, before the mean (the paper: per rotation)
        out: torch.Tensor = self.head(orientation_mean(h, self.group.order), x_feat)
        return out


def build_paper_model(
    arch: str = "rot4",
    *,
    group_fields: int | None = None,
    context: str = "none",
    with_features: bool = True,
    dropout_active: bool = False,
    drop_rate: float = PAPER_DROP_RATE,
    n_classes: int = N_CLASSES,
) -> nn.Module:
    """The paper model or one of its 2.5 variants (module docstring), with ``n_classes`` logits.

    The defaults build exactly the faithful ``PaperStampClassifier`` (the same layers and
    the same draws from the global torch generator). ``n_classes`` only resizes the last
    layer; 5 (the default) is the phase-2 model.

    Raises:
        ValueError: for an ``arch`` not in :data:`PAPER_ARCHS`, a ``context`` not in
            :data:`CONTEXTS`, ``group_fields`` set for an ``arch`` that is not a G-CNN, or
            the constructors' errors.
    """
    if arch not in PAPER_ARCHS:
        raise ValueError(f"paper_arch must be one of {', '.join(PAPER_ARCHS)}, got {arch!r}")
    in_channels = paper_in_channels(context)
    if arch in GROUPS:
        return GroupStampClassifier(
            arch,
            group_fields=group_fields,
            with_features=with_features,
            dropout_active=dropout_active,
            drop_rate=drop_rate,
            in_channels=in_channels,
            n_classes=n_classes,
        )
    if group_fields is not None:
        raise ValueError(f"group_fields applies to {' and '.join(GROUPS)} only, not {arch!r}")
    return PaperStampClassifier(
        with_features=with_features,
        dropout_active=dropout_active,
        drop_rate=drop_rate,
        in_channels=in_channels,
        rotation_mean=arch == "rot4",
        n_classes=n_classes,
    )


def count_parameters(model: nn.Module) -> int:
    """The number of trainable parameters (``requires_grad``) of ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, *, in_channels: int = PAPER_IN_CHANNELS) -> int:
    """Multiply-accumulates of one eval-mode forward pass per stamp.

    Counted by ``torch.utils.flop_counter.FlopCounterMode`` (convolutions and matrix
    products; 2 FLOPs per MAC) on a CPU batch of zeros, (2, ``in_channels``, 21, 21) with
    (2, 23) features, divided by the batch. Element-wise operations, pooling, batch norm and
    the weight expansion of the group layers (independent of the batch) are not counted.
    The model must be on the CPU; its train/eval mode is restored afterwards.

    Raises:
        ValueError: if the counted FLOPs are not a whole number of MACs per stamp.
    """
    batch = 2
    images = torch.zeros(batch, in_channels, PAPER_CROP, PAPER_CROP)
    features = torch.zeros(batch, N_FEATURES)
    training = model.training
    model.eval()
    try:
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            model(images, features)
    finally:
        model.train(training)
    flops = counter.get_total_flops()
    if flops % (2 * batch):
        raise ValueError(f"{flops} FLOPs are not a whole number of MACs per stamp")
    return int(flops // (2 * batch))


@dataclass(frozen=True, slots=True)
class Variant:
    """One row of the plan's 2.5-arch table as ``Settings`` overrides.

    Attributes:
        row: the plan's row id (``A`` is the replication).
        overrides: the ``Settings`` fields that turn it on, on top of the ``carrasco2021``
            preset.
        description: what it tests.
    """

    row: str
    overrides: Mapping[str, object]
    description: str


#: The 2.5-models rows and their control arms, by name (``scripts/bench_paper_models.py``;
#: train one with ``--set key=value`` for each override).
VARIANTS: Final[Mapping[str, Variant]] = {
    "rot4": Variant("A", {}, "the replication: 4 rotations, mean after fc1"),
    "single-aug": Variant(
        "control",
        {"paper_arch": "single", "rot90_augment": True},
        "single pass with rot90 augmentation: is exact invariance needed?",
    ),
    "p4": Variant("B7", {"paper_arch": "p4"}, "single-pass p4 G-CNN, 8/16 fields"),
    "p4-16": Variant(
        "control",
        {"paper_arch": "p4", "group_fields": 16},
        "p4 at 16/32 fields, about rot4's cost: inductive bias vs capacity",
    ),
    "p4m": Variant("B2'", {"paper_arch": "p4m"}, "single-pass p4m G-CNN, 4/8 fields"),
    "p4-context": Variant(
        "B1'",
        {"paper_arch": "p4", "context": "avgpool63"},
        "p4 with the 63x63 stamp average-pooled to 21x21 as 3 more channels",
    ),
}
