"""The SimCLR network of phase 5: ResNet-18 with a CIFAR stem and a 2-layer projection head.

- **Encoder** (:func:`resnet18_cifar`): torchvision's ``resnet18(weights=None)`` with the
  ImageNet stem (7x7 stride-2 convolution, then a 3x3 stride-2 max-pool, a stride of 4)
  replaced by a 3x3 stride-1 convolution and no max-pool, the stem of SimCLR's CIFAR-10
  experiments (Chen et al. 2020, App. B.9). On 57-pixel views the feature maps are 57 ->
  57 -> 29 -> 15 -> 8, against 16 -> 2 with the ImageNet stem: a ZTF PSF is about 2 pixels
  wide (FWHM), and a stride of 4 in the stem discards it. ``fc`` is the identity, so the
  representation ``h`` is the 512-d global-average-pooled output of ``layer4``.
  ``zero_init_residual=True`` (default): the last BatchNorm of every residual block starts
  at gamma 0, so each block starts as the identity, as in SimCLR's ResNet (``init_zero``
  on the block's last batch norm). The new stem convolution gets torchvision's own
  convolution initialisation (Kaiming normal, ``fan_out``).
- **Head** (:class:`ProjectionHead`): Linear(512, 512) - BatchNorm1d - ReLU - Linear(512,
  128), the 2-layer head of SimCLR v1 with v2's BatchNorm on the hidden layer. The first
  linear layer has no bias (the BatchNorm right after it would cancel it). ``z`` is not
  normalised here: the loss normalises it (cosine similarity).
- :class:`SimCLRNetwork` returns ``(h, z)``; only ``h`` (before the head) is used
  downstream (probes, kNN, fine-tuning), as in SimCLR.

Built on the CPU from the global torch generator (the caller seeds it), so the initial
weights do not depend on the device (ADR 0014, "Seeding").
"""

from __future__ import annotations

from typing import Final

import torch
import torchvision
from torch import nn

__all__ = [
    "PROJECTION_DIM",
    "REPRESENTATION_DIM",
    "ProjectionHead",
    "SimCLRNetwork",
    "resnet18_cifar",
]

#: ``h``: ResNet-18's pooled ``layer4`` output.
REPRESENTATION_DIM: Final = 512
#: ``z``: the projection head's output, the input of the loss.
PROJECTION_DIM: Final = 128


def resnet18_cifar(*, in_channels: int = 3, zero_init_residual: bool = True) -> nn.Module:
    """ResNet-18 with the CIFAR stem (3x3 stride 1, no max-pool) and ``fc = Identity``.

    Maps (N, ``in_channels``, H, W) to (N, 512). Any H, W >= 8 works (global pooling).

    Raises:
        ValueError: if ``in_channels < 1``.
    """
    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {in_channels}")
    net = torchvision.models.resnet18(weights=None, zero_init_residual=zero_init_residual)
    stem = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # torchvision initialises its convolutions this way in ResNet.__init__.
    nn.init.kaiming_normal_(stem.weight, mode="fan_out", nonlinearity="relu")
    net.conv1 = stem
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    module: nn.Module = net
    return module


class ProjectionHead(nn.Module):
    """Linear(d, d, bias=False) - BatchNorm1d(d) - ReLU - Linear(d, out): ``h -> z``."""

    def __init__(
        self, in_features: int = REPRESENTATION_DIM, out_features: int = PROJECTION_DIM
    ) -> None:
        """Create the head (``in_features`` wide hidden layer, ``out_features`` output)."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, in_features, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(N, in_features) -> (N, out_features)."""
        z: torch.Tensor = self.layers(h)
        return z


class SimCLRNetwork(nn.Module):
    """The encoder and the projection head; ``forward`` returns ``(h, z)``."""

    def __init__(self, *, in_channels: int = 3, zero_init_residual: bool = True) -> None:
        """Build :func:`resnet18_cifar` and :class:`ProjectionHead` (512 -> 128)."""
        super().__init__()
        self.encoder = resnet18_cifar(
            in_channels=in_channels, zero_init_residual=zero_init_residual
        )
        self.head = ProjectionHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, C, H, W) views -> ``(h, z)``, (N, 512) and (N, 128)."""
        h: torch.Tensor = self.encoder(x)
        return h, self.head(h)
