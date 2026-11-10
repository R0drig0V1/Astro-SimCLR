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
- **Activation checkpointing** (``activation_checkpointing=True``, off by default; ADR 0025,
  amendment "activation checkpointing"): in train mode with gradients on, the encoder runs
  as nine segments (the stem, then each of the eight residual blocks) under
  :func:`torch.utils.checkpoint.checkpoint` (``use_reentrant=False``,
  ``preserve_rng_state=True``): only each segment's input is kept for the backward, and the
  segment's forward is run again there (:func:`checkpointed_encoder`). The recompute is in
  train mode, so its BatchNorm layers would update their running statistics a second time:
  every buffer of the segment is saved when the recompute starts and copied back when it
  ends (:func:`_restore_buffers`), so the buffers after the step are the forward's. The
  outputs, gradients and buffers equal those of the plain forward bit for bit on the CPU
  (a test); the network has no dropout, so the RNG state is preserved only to keep that
  true if one is ever added. The flag is a plain attribute, not state: the state dict, and
  so the checkpoints, are the same with it on and off. Eval mode or ``no_grad`` runs the
  plain forward.

Built on the CPU from the global torch generator (the caller seeds it), so the initial
weights do not depend on the device (ADR 0014, "Seeding").
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Final, cast

import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    "PROJECTION_DIM",
    "REPRESENTATION_DIM",
    "ProjectionHead",
    "SimCLRNetwork",
    "checkpointed_encoder",
    "encoder_segments",
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


def encoder_segments(encoder: nn.Module) -> list[tuple[nn.Module, Callable[..., torch.Tensor]]]:
    """The checkpointed segments of a :func:`resnet18_cifar` encoder, in forward order.

    Each is ``(module, function)``: the stem (``conv1``, ``bn1``, ``relu``, ``maxpool``),
    then every residual block of ``layer1`` to ``layer4``. Their composition, followed by
    ``avgpool``, ``flatten`` and ``fc``, is torchvision's ``ResNet._forward_impl``.

    Raises:
        TypeError: if ``encoder`` is not a torchvision ``ResNet``.
    """
    if not isinstance(encoder, torchvision.models.ResNet):
        raise TypeError(f"expected a torchvision ResNet, got {type(encoder).__name__}")
    net = encoder

    def stem(x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        return out

    stem_modules = nn.ModuleList([net.conv1, net.bn1, net.relu, net.maxpool])
    segments: list[tuple[nn.Module, Callable[..., torch.Tensor]]] = [(stem_modules, stem)]
    for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
        segments.extend((block, block) for block in cast(nn.Sequential, layer))
    return segments


@contextmanager
def _restore_buffers(module: nn.Module) -> Iterator[None]:
    """Leave every buffer of ``module`` as it was when the block was entered.

    The recompute of a checkpointed segment runs in train mode, so BatchNorm would update
    ``running_mean``, ``running_var`` and ``num_batches_tracked`` a second time per step.
    ``finally``: the checkpoint's early stop ends a recompute by raising inside the block.
    """
    saved = [(buffer, buffer.clone()) for buffer in module.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in saved:
                buffer.copy_(value)


def _contexts(
    module: nn.Module,
) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
    """``context_fn`` of a segment: nothing around the forward, buffers kept on recompute."""
    return nullcontext(), _restore_buffers(module)


def checkpointed_encoder(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The encoder's forward with each segment of :func:`encoder_segments` checkpointed.

    The same operations in the same order as ``encoder(x)``; only the tensors kept for the
    backward differ (each segment's input instead of its intermediate activations). The
    recompute leaves the segment's buffers as the forward left them (module docstring).

    Raises:
        TypeError: as :func:`encoder_segments`.
    """
    net = cast(torchvision.models.ResNet, encoder)
    for module, function in encoder_segments(encoder):
        x = checkpoint(
            function,
            x,
            use_reentrant=False,
            preserve_rng_state=True,
            context_fn=partial(_contexts, module),
        )
    out: torch.Tensor = net.fc(torch.flatten(net.avgpool(x), 1))
    return out


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
    """The encoder and the projection head; ``forward`` returns ``(h, z)``.

    ``activation_checkpointing`` (a plain attribute, settable after construction) runs the
    encoder through :func:`checkpointed_encoder` in train mode with gradients on (module
    docstring); it is not part of the state dict.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        zero_init_residual: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        """Build :func:`resnet18_cifar` and :class:`ProjectionHead` (512 -> 128)."""
        super().__init__()
        self.encoder = resnet18_cifar(
            in_channels=in_channels, zero_init_residual=zero_init_residual
        )
        self.head = ProjectionHead()
        self.activation_checkpointing = activation_checkpointing

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, C, H, W) views -> ``(h, z)``, (N, 512) and (N, 128)."""
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            h = checkpointed_encoder(self.encoder, x)
        else:
            h = self.encoder(x)
        return h, self.head(h)
