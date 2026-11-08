"""The stamp classifiers of Reyes-Jainaga et al. 2023 (RJ23), engine-free (phase 3, task 3.4).

RJ23 (ApJL 952, L43, arXiv:2304.13080, Appendix A and Table 2) train one convolutional
classifier per input scenario: the full 63x63 stamp, the 16x16 crop, the 16x16 low-resolution
stamp and the 8x8x24 multi-scale stamp, all built by :mod:`astro_simclr.data.rj23_input`
(channels last there; :func:`~astro_simclr.data.rj23_input.channels_first` gives the
``(N, C, H, W)`` tensors this module takes). Their code is public (MIT,
``github.com/alercebroker/multiscale_stamps``) but its ``models.py`` is missing, so this is a
reimplementation from the paper's text and table; nothing is copied. The details the paper
leaves open are pinned in ADR 0012, each with its source.

The modules are plain ``nn.Module`` builders with no Lightning (the phase-4 rule: training
engines wire them through thin adapters). They import torch and the numpy-only
``rj23_input`` constants, nothing else of the package.

**The network** (:class:`RJ23Classifier`), for each scenario's row of :data:`RJ23_TABLE2`:

1. the 8 views of the input, the 4 counter-clockwise rotations (``torch.rot90``, as
   ``tf.image.rot90``) and the up-down flip of each (:func:`dihedral_views`), go through the
   same trunk as one batch of 8N;
2. the trunk: Table 2's stack of 3x3 convolutions with ``'same'`` padding, each followed by
   a ReLU, and 2x2 max-poolings with TF ``'same'`` padding (:class:`TFSameMaxPool2d`),
   ending on 8x8 maps in every scenario (full 63 -> 32 -> 16 -> 8; cropped and low-res
   16 -> 8; multi-scale no pooling);
3. flatten, then the mean over the 8 views: :meth:`RJ23Classifier.representation`, exactly
   invariant under the dihedral group D4 acting on the input;
4. the head: dropout, concatenation of the position ``(x, y, z)`` on the unit sphere
   (:func:`~astro_simclr.data.rj23_input.unit_sphere`), Keras-default batch normalisation
   (:class:`KerasBatchNorm1d`), FC to Table 2's dense size, ReLU, FC to ``n_classes``
   logits (5 for R5; RJ23 had 6). ``bn_position="before_concat"`` is the pre-registered
   sensitivity arm: the BN normalises the image representation only, before the position
   is concatenated.

**TF-default parity** (RJ23 trained with ``tf.keras`` defaults; each has a hand-computed
golden test in ``tests/test_rj23_model.py``, with no TF dependency):

- convolution and dense kernels Glorot-uniform, biases zero (:func:`init_keras_defaults`);
- batch normalisation with Keras' momentum 0.99 (torch's ``momentum=0.01``), epsilon 1e-3,
  and the moving variance updated with the *biased* batch variance, as Keras does on 2-D
  inputs (torch uses the unbiased one);
- max-pooling with TF ``'same'`` padding (torch has none).

The Adam epsilon of 1e-7 (Keras' default) is part of the training regime, task 3.4b, not of
this module; so are Table 2's learning rates, which :data:`RJ23_TABLE2` records.

**Parameter counts** (:func:`rj23_parameter_count`, tested against ``sum(p.numel())``): with
``c_0`` input channels, the convolution widths ``c_1 ... c_L``, the first kernel ``k_1`` and
the others ``k = 3``, the representation width ``D = 64 c_L``, the dense size ``F`` and
``K`` classes::

    sum_l (k_l^2 c_{l-1} + 1) c_l          convolutions (weights and biases)
    + 2 (D + 3)                            BN gamma and beta (2 D before the concat)
    + (D + 3 + 1) F + (F + 1) K            the two FC layers

which gives 99,778 (full), 1,202,808 (cropped), 86,648 (low-res) and 179,124
(multi-scale) parameters for K = 5, six fewer each with the BN before the concat.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import torch
import torch.nn.functional as F
from torch import nn

from astro_simclr.data.rj23_input import VIEW_NAMES, VIEW_SHAPES, ViewName

__all__ = [
    "KERAS_BN_EPS",
    "KERAS_BN_MOMENTUM",
    "MAXPOOL",
    "N_POSITION",
    "N_VIEWS",
    "R5_CLASSES",
    "RJ23_TABLE2",
    "BNPosition",
    "KerasBatchNorm1d",
    "RJ23Architecture",
    "RJ23Classifier",
    "TFSameMaxPool2d",
    "build_rj23_classifier",
    "dihedral_views",
    "init_keras_defaults",
    "rj23_parameter_count",
    "tf_same_padding",
]

#: Marker of a 2x2 max-pooling layer in :attr:`RJ23Architecture.layers`.
MAXPOOL: Final = "maxpool"
#: The 4 rotations times the up-down flip (RJ23 App. A).
N_VIEWS: Final = 8
#: The position input: ``(x, y, z)`` on the unit sphere.
N_POSITION: Final = 3
#: The classes of R5 (task 3.5): RJ23's six without the satellites.
R5_CLASSES: Final = 5
#: Keras ``BatchNormalization`` defaults: ``momentum=0.99`` (the weight of the old moving
#: value), in torch's convention the weight of the new batch statistic.
KERAS_BN_MOMENTUM: Final = 0.01
#: Keras ``BatchNormalization`` default ``epsilon``.
KERAS_BN_EPS: Final = 1e-3
#: Side of the trunk's last feature maps in every scenario (RJ23 App. A).
_LAST_SIDE: Final = 8
#: Kernel size of every convolution but the first (Table 2: ``conv3-*``).
_KERNEL: Final = 3

BNPosition = Literal["after_concat", "before_concat"]
_BN_POSITIONS: Final[tuple[BNPosition, ...]] = ("after_concat", "before_concat")

Layer = int | Literal["maxpool"]


@dataclass(frozen=True)
class RJ23Architecture:
    """One column of RJ23's Table 2.

    Attributes:
        scenario: the input view (:data:`~astro_simclr.data.rj23_input.VIEW_NAMES`).
        layers: the convolution-pooling section in order: an ``int`` is a convolution with
            that many filters, :data:`MAXPOOL` a 2x2 max-pooling with stride 2.
        dense: units of the first FC layer.
        dropout: the fraction of units dropped before the position is concatenated.
        learning_rate: Table 2's Adam learning rate (used by the training regime, 3.4b).
        first_kernel_size: kernel of the first convolution (RJ23 searched 3 and 5; the
            final models use 3, ``first_kernel_size = 3`` in their training script); the
            other convolutions are 3x3.
    """

    scenario: ViewName
    layers: tuple[Layer, ...]
    dense: int
    dropout: float
    learning_rate: float
    first_kernel_size: int = _KERNEL

    @property
    def input_shape(self) -> tuple[int, int, int]:
        """``(C, H, W)`` of the scenario's view (channels first)."""
        height, width, channels = VIEW_SHAPES[self.scenario]
        return channels, height, width

    @property
    def conv_widths(self) -> tuple[int, ...]:
        """The filters of the convolutions, in order."""
        return tuple(layer for layer in self.layers if isinstance(layer, int))

    @property
    def representation_size(self) -> int:
        """Width of the flattened, view-averaged representation: ``8 * 8 * c_L``."""
        return _LAST_SIDE * _LAST_SIDE * self.conv_widths[-1]


#: RJ23's Table 2 (Reyes-Jainaga et al. 2023, ApJL 952, L43, Appendix A; arXiv:2304.13080v2):
#: the architectures and learning rates of the four scenarios as published. The filter
#: counts, dense sizes, dropout rates and learning rates also match
#: ``hyperparameters_per_dataset`` in RJ23's ``src/train_and_save_best_models.py`` (task 3.0
#: research), which fixes ``first_kernel_size = 3`` and ``with_batchnorm = True``.
RJ23_TABLE2: Final[Mapping[ViewName, RJ23Architecture]] = MappingProxyType(
    {
        "full": RJ23Architecture(
            scenario="full",
            layers=(37, MAXPOOL, 60, MAXPOOL, 29, MAXPOOL, 26),
            dense=31,
            dropout=0.71726,
            learning_rate=7.589e-4,
        ),
        "cropped": RJ23Architecture(
            scenario="cropped",
            layers=(112, 27, MAXPOOL, 92, 121),
            dense=133,
            dropout=0.866799,
            learning_rate=3.4627e-4,
        ),
        "lowres": RJ23Architecture(
            scenario="lowres",
            layers=(63, 31, MAXPOOL, 108, 14),
            dense=22,
            dropout=0.73065,
            learning_rate=1.125e-3,
        ),
        "multiscale": RJ23Architecture(
            scenario="multiscale",
            layers=(76, 18, 54, 28),
            dense=69,
            dropout=0.8478,
            learning_rate=7.445e-4,
        ),
    }
)


def tf_same_padding(size: int, kernel: int, stride: int) -> tuple[int, int, int]:
    """TF ``'same'`` padding along one axis: ``(output size, pad before, pad after)``.

    The output is ``ceil(size / stride)``; TF pads ``max((out - 1) * stride + kernel - size,
    0)`` pixels, ``total // 2`` before and the rest after (so an odd total puts the extra
    pixel at the end). For a 2x2 pooling with stride 2: 63 -> 32 with one pixel after, 32 ->
    16 and 16 -> 8 with none.

    Raises:
        ValueError: if ``size``, ``kernel`` or ``stride`` is below 1.
    """
    if size < 1 or kernel < 1 or stride < 1:
        raise ValueError(f"size, kernel and stride must be >= 1, got {size}, {kernel}, {stride}")
    out = -(-size // stride)
    total = max((out - 1) * stride + kernel - size, 0)
    return out, total // 2, total - total // 2


class TFSameMaxPool2d(nn.Module):
    """TF's ``MaxPool2D(kernel_size, strides=stride, padding='same')`` on ``(N, C, H, W)``.

    The input is padded as :func:`tf_same_padding` says with ``-inf``, so a window that
    overlaps the padding takes the maximum of its valid pixels only (TF ignores the
    padding; zero padding would differ on negative inputs). The output is ``ceil(H /
    stride) x ceil(W / stride)``. torch's ``MaxPool2d(ceil_mode=True)`` agrees only when TF
    pads nothing before, as for a 2x2 pooling with stride 2; this layer is exact in general.

    Raises:
        ValueError: at construction, if ``kernel_size`` or ``stride`` is below 1; in
            ``forward``, if the input is not 4-D.
    """

    def __init__(self, kernel_size: int = 2, stride: int | None = None) -> None:
        super().__init__()
        stride = kernel_size if stride is None else stride
        if kernel_size < 1 or stride < 1:
            raise ValueError(f"kernel_size and stride must be >= 1, got {kernel_size}, {stride}")
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Max-pool ``x`` (N, C, H, W) with TF 'same' padding."""
        if x.ndim != 4:
            raise ValueError(f"expected (N, C, H, W) input, got shape {tuple(x.shape)}")
        _, top, bottom = tf_same_padding(x.shape[2], self.kernel_size, self.stride)
        _, left, right = tf_same_padding(x.shape[3], self.kernel_size, self.stride)
        if top or bottom or left or right:
            x = F.pad(x, (left, right, top, bottom), value=-math.inf)
        return F.max_pool2d(x, self.kernel_size, self.stride)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding='same'"


class KerasBatchNorm1d(nn.BatchNorm1d):
    """``tf.keras.layers.BatchNormalization()`` with its defaults, on ``(N, features)``.

    - ``eps=1e-3`` and Keras' ``momentum=0.99``, which in torch's convention is
      ``momentum=0.01``: ``running = 0.99 * running + 0.01 * batch``;
    - gamma 1, beta 0, moving mean 0, moving variance 1 (the same as torch);
    - training normalises with the batch mean and the biased batch variance (as torch),
      but the **moving variance is updated with the biased variance** too: Keras' non-fused
      path (every 2-D input) takes ``tf.nn.moments`` for both, while torch's
      ``BatchNorm1d`` updates its running variance with the unbiased one (a factor
      ``N / (N - 1)``, 256/255 at RJ23's batch size);
    - evaluation normalises with the moving statistics.

    Raises:
        ValueError: in training mode, on a batch of one sample (as torch's ``BatchNorm1d``).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__(num_features, eps=KERAS_BN_EPS, momentum=KERAS_BN_MOMENTUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` (N, features) and, in training, update the moving statistics."""
        if not self.training:
            return super().forward(x)
        # The output from the batch statistics, without touching the running buffers ...
        out = F.batch_norm(x, None, None, self.weight, self.bias, True, 0.0, self.eps)
        # ... which then follow Keras: the biased variance, momentum 0.99.
        assert self.running_mean is not None and self.running_var is not None  # noqa: S101
        assert self.num_batches_tracked is not None  # noqa: S101
        with torch.no_grad():
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean.lerp_(mean, self.momentum or 0.0)
            self.running_var.lerp_(var, self.momentum or 0.0)
            self.num_batches_tracked.add_(1)
        return out


def init_keras_defaults(module: nn.Module) -> None:
    """Keras' default initialisation of every convolution, dense and batch-norm layer.

    Kernels Glorot-uniform, ``U(-l, l)`` with ``l = sqrt(6 / (fan_in + fan_out))``; for a
    convolution ``fan_in = k^2 c_in`` and ``fan_out = k^2 c_out`` (Keras' and
    ``nn.init.xavier_uniform_``'s fans agree), for a dense layer the input and output
    widths. Biases zero. Batch norm gamma 1, beta 0, moving statistics (0, 1). Layers are
    visited in ``module.modules()`` order, kernel then bias, drawing from the global torch
    generator, so ``torch.manual_seed`` fixes the weights.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d | nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm1d):
            layer.reset_parameters()


def dihedral_views(x: torch.Tensor) -> torch.Tensor:
    """The 8 views of RJ23's App. A stacked on the batch axis: ``(N, C, H, W) -> (8N, C, H, W)``.

    Views ``k = 0..3`` are ``rot90(x, k)`` (counter-clockwise, as ``tf.image.rot90``),
    views ``4 + k`` their up-down flips (rows reversed, as ``tf.image.flip_up_down``); view
    ``v`` of sample ``i`` is row ``v * N + i``. The 8 transforms are the dihedral group D4
    of the square, so a D4 transform of ``x`` only permutes the views of each sample.

    Raises:
        ValueError: if ``x`` is not 4-D with square images.
    """
    if x.ndim != 4 or x.shape[2] != x.shape[3]:
        raise ValueError(f"expected square (N, C, H, H) images, got shape {tuple(x.shape)}")
    rotations = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)]
    flips = [torch.flip(r, dims=(2,)) for r in rotations]
    return torch.cat(rotations + flips)


def _check_architecture(architecture: RJ23Architecture, n_classes: int) -> None:
    if architecture.scenario not in VIEW_NAMES:
        raise ValueError(f"scenario must be one of {VIEW_NAMES}, got {architecture.scenario!r}")
    widths = architecture.conv_widths
    if not widths or not isinstance(architecture.layers[0], int):
        raise ValueError("the convolution-pooling section must start with a convolution")
    if any(w < 1 for w in widths) or architecture.dense < 1:
        raise ValueError("convolution widths and the dense size must be >= 1")
    if not 0 <= architecture.dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {architecture.dropout}")
    if architecture.first_kernel_size < 1 or architecture.first_kernel_size % 2 == 0:
        # TF and torch place the extra 'same' pixel alike for even kernels, but RJ23 only
        # searched 3 and 5: keep the symmetric case.
        raise ValueError(f"first_kernel_size must be odd, got {architecture.first_kernel_size}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    side = architecture.input_shape[1]
    for layer in architecture.layers:
        if layer == MAXPOOL:
            side = tf_same_padding(side, 2, 2)[0]
    if side != _LAST_SIDE:
        msg = (
            f"the {architecture.scenario} trunk must end on {_LAST_SIDE}x{_LAST_SIDE} maps "
            f"(RJ23 App. A), but its poolings give {side}x{side}"
        )
        raise ValueError(msg)


class RJ23Classifier(nn.Module):
    """RJ23's stamp classifier for one scenario (module docstring; ADR 0012).

    ``forward(x_img, x_pos)`` takes the scenario's view ``(N, C, H, W)`` (channels first;
    :data:`~astro_simclr.data.rj23_input.VIEW_SHAPES` gives ``(H, W, C)``) and the position
    ``(N, 3)``, and returns ``(N, n_classes)`` logits. Submodules: ``trunk`` (the
    convolution-pooling section, ReLU after every convolution), ``dropout``, ``bn``
    (:class:`KerasBatchNorm1d` of ``D + 3`` features after the concatenation, or ``D``
    before it), ``fc`` (to ``architecture.dense``, then ReLU) and ``logits``. All
    parameters are initialised by :func:`init_keras_defaults`.

    Raises:
        ValueError: at construction, on an architecture whose trunk does not end on 8x8
            maps, a dropout outside [0, 1), an even first kernel, ``n_classes`` below 2 or
            an unknown ``bn_position``; in ``forward``, on images or positions of the wrong
            shape.
    """

    def __init__(
        self,
        architecture: RJ23Architecture,
        *,
        n_classes: int = R5_CLASSES,
        bn_position: BNPosition = "after_concat",
    ) -> None:
        super().__init__()
        _check_architecture(architecture, n_classes)
        if bn_position not in _BN_POSITIONS:
            raise ValueError(f"bn_position must be one of {_BN_POSITIONS}, got {bn_position!r}")
        self.architecture = architecture
        self.n_classes = n_classes
        self.bn_position: BNPosition = bn_position
        layers: list[nn.Module] = []
        channels = architecture.input_shape[0]
        kernel = architecture.first_kernel_size
        for layer in architecture.layers:
            if layer == MAXPOOL:
                layers.append(TFSameMaxPool2d(2))
                continue
            assert isinstance(layer, int)  # noqa: S101 (narrowing for mypy)
            layers += [nn.Conv2d(channels, layer, kernel, padding="same"), nn.ReLU()]
            channels, kernel = layer, _KERNEL
        self.trunk = nn.Sequential(*layers)
        width = architecture.representation_size
        self.dropout = nn.Dropout(architecture.dropout)
        self.bn = KerasBatchNorm1d(width + N_POSITION if bn_position == "after_concat" else width)
        self.fc = nn.Linear(width + N_POSITION, architecture.dense)
        self.logits = nn.Linear(architecture.dense, n_classes)
        init_keras_defaults(self)

    def _check_images(self, x_img: torch.Tensor) -> None:
        expected = self.architecture.input_shape
        if x_img.ndim != 4 or tuple(x_img.shape[1:]) != expected:
            msg = (
                f"the {self.architecture.scenario} model needs images of shape (N, "
                f"{', '.join(map(str, expected))}), got {tuple(x_img.shape)}"
            )
            raise ValueError(msg)

    def representation(self, x_img: torch.Tensor) -> torch.Tensor:
        """The trunk's flattened output averaged over the 8 views, ``(N, 64 c_L)``.

        Exactly invariant (up to the float summation order) under every rotation and flip
        of ``x_img``: those only permute the views (:func:`dihedral_views`).

        Raises:
            ValueError: if ``x_img`` is not ``(N, C, H, W)`` of the scenario.
        """
        self._check_images(x_img)
        n = x_img.shape[0]
        features = self.trunk(dihedral_views(x_img))
        out: torch.Tensor = features.flatten(start_dim=1).view(N_VIEWS, n, -1).mean(dim=0)
        return out

    def head(self, representation: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
        """Dropout, the position, BN, FC + ReLU and the logits: ``(N, D), (N, 3) -> (N, K)``.

        Raises:
            ValueError: if ``x_pos`` is not ``(N, 3)`` for the ``N`` of the representation.
        """
        n = representation.shape[0]
        if x_pos.ndim != 2 or tuple(x_pos.shape) != (n, N_POSITION):
            msg = f"x_pos must be (N, {N_POSITION}) with N = {n}, got {tuple(x_pos.shape)}"
            raise ValueError(msg)
        h = self.dropout(representation)
        if self.bn_position == "after_concat":
            h = self.bn(torch.cat((h, x_pos), dim=1))
        else:
            h = torch.cat((self.bn(h), x_pos), dim=1)
        out: torch.Tensor = self.logits(F.relu(self.fc(h)))
        return out

    def forward(self, x_img: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
        """Views ``(N, C, H, W)`` and positions ``(N, 3)`` -> ``(N, n_classes)`` logits."""
        return self.head(self.representation(x_img), x_pos)


def build_rj23_classifier(
    scenario: ViewName,
    *,
    n_classes: int = R5_CLASSES,
    bn_position: BNPosition = "after_concat",
) -> RJ23Classifier:
    """RJ23's classifier for ``scenario`` with Table 2's hyperparameters (:data:`RJ23_TABLE2`).

    ``bn_position="before_concat"`` builds the pre-registered sensitivity arm (ADR 0012).
    The weights are drawn from the global torch generator.

    Raises:
        ValueError: on an unknown scenario or ``bn_position``, or ``n_classes`` below 2.
    """
    if scenario not in RJ23_TABLE2:
        raise ValueError(f"scenario must be one of {VIEW_NAMES}, got {scenario!r}")
    return RJ23Classifier(RJ23_TABLE2[scenario], n_classes=n_classes, bn_position=bn_position)


def rj23_parameter_count(
    architecture: RJ23Architecture,
    *,
    n_classes: int = R5_CLASSES,
    bn_position: BNPosition = "after_concat",
) -> int:
    """The trainable parameters of :class:`RJ23Classifier`, by the module docstring's formula.

    Computed from the architecture alone (no model is built), so the tests can check it
    against ``sum(p.numel())``.
    """
    total = 0
    channels = architecture.input_shape[0]
    kernel = architecture.first_kernel_size
    for width in architecture.conv_widths:
        total += (kernel * kernel * channels + 1) * width
        channels, kernel = width, _KERNEL
    rep = architecture.representation_size
    total += 2 * (rep + N_POSITION if bn_position == "after_concat" else rep)
    total += (rep + N_POSITION + 1) * architecture.dense
    total += (architecture.dense + 1) * n_classes
    return total
