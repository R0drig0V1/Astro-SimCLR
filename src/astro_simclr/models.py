"""Networks: stamp encoders and classifiers, the SimCLR wrapper, heads and the encoder factory.

Ported from the legacy ``models/models.py`` (task 1.4 of phase 1). Every class keeps the
attribute names of its legacy counterpart, so ``state_dict`` keys are identical and legacy
weights load with ``load_state_dict(strict=True)``. Legacy names:

=========================  ===============================================================
This module                Legacy
=========================  ===============================================================
``StampClassifier``        ``Stamp_classifier`` (Carrasco-Davis et al. 2021)
``StampEncoder``           ``Stamp_encoder`` (encoder names ``stamp<k>``)
``PStampsEncoder``         ``P_stamps_net_SimCLR_v2`` from ``b357b43^`` (name ``pstamps``)
``SimCLRNet``              ``CLR``
``LinearClassifier``       ``Linear_classifier`` (the ``nn.Module``, not the Lightning one)
``ThreeLayerClassifier``   ``Three_layers_classifier``
``Identity``               ``Identity`` (now ``torch.nn.Identity``)
``build_encoder``          the ``if/elif`` chains in ``utils/training.py`` (``SimCLR``)
=========================  ===============================================================

The stamp networks only accept 27x27 inputs: ``fc1`` is sized for that crop, as in the
legacy code. Any other size raises ``ValueError`` before the first convolution instead of
a shape mismatch deep inside ``fc1``. The ResNets accept any size >= 27 (27 and 63 are the
crops the experiments use).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from math import floor
from typing import Final

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

__all__ = [
    "ENCODER_NAMES",
    "N_CLASSES",
    "N_FEATURES",
    "STAMP_SIZE",
    "Encoder",
    "Identity",
    "LinearClassifier",
    "PStampsEncoder",
    "SimCLRNet",
    "StampClassifier",
    "StampEncoder",
    "ThreeLayerClassifier",
    "build_encoder",
]

#: Side of the only crop the stamp networks accept (their ``fc1`` is sized for it).
STAMP_SIZE: Final = 27
#: Number of tabular features the classifiers concatenate (26 in the pickles, 3 dropped).
N_FEATURES: Final = 23
#: Number of classes (AGN, SN, VS, Asteroid, Bogus).
N_CLASSES: Final = 5

#: The legacy ``Identity`` returned its input unchanged; ``torch.nn.Identity`` does the same
#: and also has no parameters, so state dicts are unaffected.
Identity = nn.Identity

_RESNETS: Final = {
    "resnet18": torchvision.models.resnet18,
    "resnet34": torchvision.models.resnet34,
    "resnet50": torchvision.models.resnet50,
    "resnet152": torchvision.models.resnet152,
}
#: The names :func:`build_encoder` accepts (``stamp<k>`` stands for any positive ``k``).
ENCODER_NAMES: Final = ("stamp<k>", "pstamps", *_RESNETS)
_STAMP_NAME: Final = re.compile(r"stamp(\d+(?:\.\d*)?|\.\d+)")


def _conv_output_shape(size: int, kernel: int, pad: int = 0, stride: int = 1) -> int:
    """Output side of a square convolution (legacy ``conv_output_shape``, dilation 1)."""
    return floor((size + 2 * pad - (kernel - 1) - 1) / stride + 1)


def _maxpool_output_shape(size: int, kernel: int = 2, stride: int = 2) -> int:
    """Output side of a square max-pool (legacy ``maxpool_output_shape``)."""
    return floor((size - (kernel - 1) - 1) / stride + 1)


def _check_stamp_size(x: torch.Tensor) -> None:
    """Raise ``ValueError`` unless ``x`` is a batch of 3x27x27 images."""
    if x.dim() != 4 or tuple(x.shape[1:]) != (3, STAMP_SIZE, STAMP_SIZE):
        raise ValueError(
            f"stamp networks need input of shape (N, 3, {STAMP_SIZE}, {STAMP_SIZE}) "
            f"(fc1 is sized for a {STAMP_SIZE}x{STAMP_SIZE} crop); got {tuple(x.shape)}. "
            f"Set image_size to {STAMP_SIZE} or use a ResNet encoder."
        )


def _rotation_average(
    conv: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """Average ``conv`` over the four 90-degree rotations of ``x`` (rotation invariance).

    The sum is taken in the legacy order ``(r0 + r1 + r2 + r3) / 4`` so that results match
    the legacy networks to floating-point rounding.
    """
    r1 = conv(torch.rot90(x, 0, [2, 3]))
    r2 = conv(torch.rot90(x, 1, [2, 3]))
    r3 = conv(torch.rot90(x, 2, [2, 3]))
    r4 = conv(torch.rot90(x, 3, [2, 3]))
    out: torch.Tensor = (r1 + r2 + r3 + r4) / 4
    return out


class _StampConvNet(nn.Module):
    """The shared convolutional trunk of the stamp networks (conv1-conv5, two pools, fc1).

    ``conv1`` is a 4x4 valid convolution; ``conv2``-``conv5`` keep the size (``same``
    padding); each pool halves it. For a 27x27 crop: 24 -> 12 -> 6, so ``fc1`` sees
    ``c2 * 6 * 6`` inputs.
    """

    def __init__(
        self, c1: int, c2: int, linear_output: int, kernel: int, *, fc1_bias: bool = True
    ) -> None:
        super().__init__()
        padding = kernel // 2
        side = _conv_output_shape(STAMP_SIZE, kernel=4)
        side = _maxpool_output_shape(_conv_output_shape(side, kernel, padding))
        for _ in range(3):
            side = _conv_output_shape(side, kernel, padding)
        side = _maxpool_output_shape(side)

        self.conv1 = nn.Conv2d(3, c1, 4)
        self.conv2 = nn.Conv2d(c1, c1, kernel, padding=padding)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(c1, c2, kernel, padding=padding)
        self.conv4 = nn.Conv2d(c2, c2, kernel, padding=padding)
        self.conv5 = nn.Conv2d(c2, c2, kernel, padding=padding)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.fl1 = nn.Flatten()
        self.fc1 = nn.Linear(c2 * side * side, linear_output, bias=fc1_bias)

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        """One rotation through the trunk: (N, 3, 27, 27) -> (N, ``fc1.out_features``)."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(x)
        x = self.fl1(x)
        out: torch.Tensor = F.relu(self.fc1(x))
        return out


class StampEncoder(_StampConvNet):
    """Rotation-invariant stamp encoder for SimCLR (legacy ``Stamp_encoder``, ``stamp<k>``).

    ``k`` scales the width: ``c1 = int(32 k)`` channels, then ``2 c1``, and
    ``int(128 k)`` output features (``out_features``). 5x5 kernels. The output is the mean
    of the four rotations, after the final ReLU.

    Raises:
        ValueError: if ``k`` gives zero channels or features, or (in ``forward``) if the
            input is not (N, 3, 27, 27).
    """

    def __init__(self, k: float = 1.0) -> None:
        c1, linear_output = int(k * 32), int(k * 128)
        if c1 < 1 or linear_output < 1:
            raise ValueError(f"stamp encoder width k must be >= 1/32, got {k}")
        super().__init__(c1, 2 * c1, linear_output, kernel=5)
        self.k = k
        self.c1 = c1
        self.c2 = 2 * c1
        self.linear_output = linear_output
        #: Dimension of the representation ``h`` (the input size of the projector).
        self.out_features = linear_output

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """(N, 3, 27, 27) images -> (N, ``out_features``) representations."""
        _check_stamp_size(x_img)
        return _rotation_average(self.conv, x_img)


class PStampsEncoder(_StampConvNet):
    """The historical ``pstamps`` encoder (legacy ``P_stamps_net_SimCLR_v2``, ``k=1.0``).

    Provenance: ``models/models.py`` at ``b357b43^`` (``c5c7e6f``), removed in ``b357b43``.
    It is the encoder of the only recorded 300k-alert SimCLR run (``31b10c9``,
    ``iter_clr_stamps_300.py``, ``encoder_name: pstamps``), whose settings reference run R3
    reuses. It differs from :class:`StampEncoder` in three ways: 3x3 kernels, ``fc1`` has no
    bias, and the rotation average goes through ``bn1`` (``BatchNorm1d``), so the output is
    not clipped at zero.

    Raises:
        ValueError: as :class:`StampEncoder`.
    """

    def __init__(self, k: float = 1.0) -> None:
        c1, linear_output = int(k * 32), int(k * 128)
        if c1 < 1 or linear_output < 1:
            raise ValueError(f"pstamps width k must be >= 1/32, got {k}")
        super().__init__(c1, 2 * c1, linear_output, kernel=3, fc1_bias=False)
        self.k = k
        self.c1 = c1
        self.c2 = 2 * c1
        self.linear_output = linear_output
        self.bn1 = nn.BatchNorm1d(linear_output)
        #: Dimension of the representation ``h`` (the input size of the projector).
        self.out_features = linear_output

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """(N, 3, 27, 27) images -> (N, ``out_features``) batch-normalised representations."""
        _check_stamp_size(x_img)
        out: torch.Tensor = self.bn1(_rotation_average(self.conv, x_img))
        return out


class StampClassifier(_StampConvNet):
    """Supervised stamp classifier (legacy ``Stamp_classifier``; Carrasco-Davis et al. 2021).

    The trunk (32/64 channels, 5x5 kernels, ``fc1`` to 64) is averaged over the four
    rotations; with ``with_features`` the 23 tabular features go through ``bn1`` and are
    concatenated. Then ``fc2`` (ReLU, dropout), ``fc3`` (ReLU) and ``fc4`` give 5 logits.

    Raises:
        ValueError: in ``forward``, if the images are not (N, 3, 27, 27), or if
            ``with_features`` is set and ``x_feat`` is missing or not (N, 23).
    """

    def __init__(self, drop_rate: float, with_features: bool) -> None:
        super().__init__(32, 64, 64, kernel=5)
        self.with_features = with_features
        if with_features:
            self.bn1 = nn.BatchNorm1d(N_FEATURES)
            self.fc2 = nn.Linear(64 + N_FEATURES, 64)
        else:
            self.fc2 = nn.Linear(64, 64)
        self.drop = nn.Dropout(p=drop_rate)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, N_CLASSES)

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Images (N, 3, 27, 27) and, with features, ``x_feat`` (N, 23) -> (N, 5) logits.

        Without ``with_features``, ``x_feat`` is ignored (the legacy signature required it).
        """
        _check_stamp_size(x_img)
        x = _rotation_average(self.conv, x_img)
        if self.with_features:
            if x_feat is None or tuple(x_feat.shape) != (x_img.shape[0], N_FEATURES):
                got = None if x_feat is None else tuple(x_feat.shape)
                raise ValueError(
                    f"with_features=True needs x_feat of shape (N, {N_FEATURES}); got {got}"
                )
            x = torch.cat((x, self.bn1(x_feat)), dim=1)
        x = self.drop(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        out: torch.Tensor = self.fc4(x)
        return out


class LinearClassifier(nn.Module):
    """A single linear layer (legacy ``Linear_classifier``; the key prefix ``model.`` is kept)."""

    def __init__(self, input_size: int, n_classes: int) -> None:
        super().__init__()
        self.model = nn.Linear(input_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, ``input_size``) -> (N, ``n_classes``) logits."""
        out: torch.Tensor = self.model(x)
        return out


class ThreeLayerClassifier(nn.Module):
    """MLP head ``fc1``-ReLU-dropout-``fc2``-ReLU-``fc3`` (legacy ``Three_layers_classifier``)."""

    def __init__(
        self,
        input_size_f1: int,
        input_size_f2: int,
        input_size_f3: int,
        n_classes: int,
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size_f1, input_size_f2)
        self.drop = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(input_size_f2, input_size_f3)
        self.fc3 = nn.Linear(input_size_f3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, ``input_size_f1``) -> (N, ``n_classes``) logits."""
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        out: torch.Tensor = self.fc3(x)
        return out


class SimCLRNet(nn.Module):
    """Encoder plus projection head (legacy ``CLR``, after Spijkervet's SimCLR).

    The projector is ``Linear-ReLU-Linear-ReLU-Linear`` (three layers, not the paper's two,
    as in the legacy code). The representation ``h`` is divided by its L2 norm before the
    projector (``h * (1 / ||h||)``, kept literally: an all-zero ``h`` gives NaN, as before).
    """

    def __init__(self, encoder: nn.Module, input_size: int, projection_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(input_size, projection_size, bias=True),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size, bias=True),
            nn.ReLU(),
            nn.Linear(projection_size, projection_size, bias=True),
        )

    def forward(
        self, x_img_i: torch.Tensor, x_img_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Two views -> ``(h_i, h_j, z_i, z_j)``: representations and projections."""
        h_i = self.encoder(x_img_i)
        h_j = self.encoder(x_img_j)
        z_i = self.projector(h_i * (1 / torch.norm(h_i, dim=1, keepdim=True)))
        z_j = self.projector(h_j * (1 / torch.norm(h_j, dim=1, keepdim=True)))
        return h_i, h_j, z_i, z_j


@dataclass(frozen=True, slots=True)
class Encoder:
    """An encoder network and the dimension of the representation it outputs."""

    #: Maps (N, 3, H, W) images to (N, ``out_features``) representations.
    module: nn.Module
    out_features: int


def build_encoder(name: str) -> Encoder:
    """Build the encoder called ``name`` (randomly initialised, as in the legacy code).

    Names: ``stamp<k>`` (:class:`StampEncoder` with width ``k``, e.g. ``stamp1``,
    ``stamp0.5``), ``pstamps`` (:class:`PStampsEncoder`, ``k=1.0``) and ``resnet18``,
    ``resnet34``, ``resnet50``, ``resnet152`` (torchvision, no pretrained weights, with
    ``fc`` replaced by :data:`Identity`).

    ``out_features`` comes from what the encoder declares: the stamp encoders'
    ``out_features`` and the ResNets' ``fc.in_features``. (The legacy code read
    ``encoder.bn1``, which ``Stamp_encoder`` lacks, so every ``stamp<k>`` failed, and
    ``'stamp' in name`` sent ``pstamps`` to ``float('s')``.)

    Raises:
        ValueError: for an unknown name (the message lists the valid ones) or a width ``k``
            too small to give one channel.
    """
    if name == "pstamps":
        pstamps = PStampsEncoder(k=1.0)
        return Encoder(pstamps, pstamps.out_features)
    if match := _STAMP_NAME.fullmatch(name):
        stamp = StampEncoder(k=float(match.group(1)))
        return Encoder(stamp, stamp.out_features)
    if name in _RESNETS:
        resnet = _RESNETS[name]()
        out_features = resnet.fc.in_features
        resnet.fc = Identity()
        return Encoder(resnet, out_features)
    raise ValueError(
        f"unknown encoder {name!r}; valid names: {', '.join(ENCODER_NAMES)} "
        "(stamp<k> with a positive width k, e.g. stamp1 or stamp0.5)"
    )
