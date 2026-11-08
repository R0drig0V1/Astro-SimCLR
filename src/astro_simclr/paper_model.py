"""The paper-faithful stamp classifier of Carrasco-Davis et al. 2021 (task 2.2a of phase 2).

:class:`PaperStampClassifier` is ALeRCE's ``DeepHitsWithFeatures`` (and, without features,
``DeepHitsExtraLayer``) from ``alercebroker/pipeline@a3705ac``,
``stamp_classifier_step/model/modules/networks/``, re-implemented in PyTorch. The code that
produced the published model (checkpoint M1) is the authority where it and the paper's text
disagree (``docs/phases/02-stamp-classifier.md``, "Sources of truth"); no ALeRCE code is
copied here.

It is a new class, not a change of the phase-1 ``models.StampClassifier``, whose port of the
2022 legacy network stays untouched. The differences with that network are the input (a
21x21 crop, zero-padded to 27), a 4x4 valid first convolution, five convolutions instead of
four, the dropout layers, the frozen feature BatchNorm and the initialisation.

Layer names and the TF variable scopes of M1 (what the 2.2b weight mapping reads):

=================  =====================================  ============================
This module        M1 (TF1, under ``network/``)           Shape here
=================  =====================================  ============================
``conv1``          ``conv_1_1/conv2d``                    3 -> 32, 4x4, valid
``conv2``          ``conv_2_2/conv2d``                    32 -> 32, 5x5, same
``conv3``          ``conv_3_4/conv2d``                    32 -> 64, 5x5, same
``conv4``          ``conv_4_5/conv2d``                    64 -> 64, 5x5, same
``conv5``          ``conv_5_6/conv2d``                    64 -> 64, 5x5, same
``fc1``            ``dense_1_10/dense_1_10``              2304 -> 64
``bn_features``    ``batchnorm_features``                 23 (with features only)
``fc2``            ``dense_2_12/dense_2_12``              87 -> 64 (64 -> 64 without)
``fc3``            ``dense_3_13/dense_3_13``              64 -> 64
``logits``         ``output_logits``                      64 -> 5
=================  =====================================  ============================

TF stores convolution kernels as HWIO and dense kernels as (in, out), and flattens the last
feature map in (h, w, c) order; this module flattens NCHW, in (c, h, w) order. The 2.2b
mapping therefore transposes the kernels and permutes the rows of ``fc1``.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

from astro_simclr.models import N_CLASSES, N_FEATURES, check_n_classes

__all__ = [
    "PAPER_CROP",
    "PAPER_DROP_RATE",
    "PAPER_IN_CHANNELS",
    "PAPER_PAD",
    "PaperStampClassifier",
    "check_paper_images",
]

#: Side of the centre crop the paper model takes (ALeRCE ``CROP_SIZE`` 21).
PAPER_CROP: Final = 21
#: Zero padding on every side of the crop (ALeRCE ``tf.pad(..., [[3, 3], [3, 3]])``): 21 -> 27.
PAPER_PAD: Final = 3
#: The dropout rate of M1's dropout layers (``droprate0.5000``), used only when they are active.
PAPER_DROP_RATE: Final = 0.5
#: Input channels of the paper model: science, template, difference.
PAPER_IN_CHANNELS: Final = 3
#: Keras ``BatchNormalization`` epsilon (the default, which M1's feature BN kept).
_BN_EPS: Final = 1e-3
#: Width of fc1-fc3 (ALeRCE ``units=64``).
_HIDDEN: Final = 64
#: Side of the last feature map: (21 + 6 - 3) = 24, then two 2x2 max-pools.
_LAST_SIDE: Final = 6
#: The 4 rotations of the input (0, 90, 180 and 270 degrees; no flips).
_N_ROTATIONS: Final = 4


def _glorot_bias_(bias: torch.Tensor) -> None:
    """Keras' fallback initialiser for a 1-D variable created without one.

    ``tf.keras.layers.Dense(bias_initializer=None)`` (ALeRCE ``layers.dense`` passes
    ``None``) leaves ``initializers.get(None) = None``, and ``Layer.add_weight`` then uses
    ``glorot_uniform`` (TF 1.14 and 1.15). For a shape ``(units,)`` its fans are
    ``fan_in = fan_out = units``, so the limit is ``sqrt(6 / (2 units)) = sqrt(3 / units)``:
    0.2165 for 64 units, the ``random_uniform/max`` constant of M1's meta graph.
    """
    limit = math.sqrt(3.0 / bias.shape[0])
    nn.init.uniform_(bias, -limit, limit)


def check_paper_images(x_img: torch.Tensor, channels: int, model: str) -> None:
    """Check that ``x_img`` is a batch of 21x21 crops with ``channels`` channels.

    Raises:
        ValueError: if ``x_img`` is not (N, ``channels``, 21, 21).
    """
    if x_img.ndim != 4 or tuple(x_img.shape[1:]) != (channels, PAPER_CROP, PAPER_CROP):
        msg = (
            f"{model} needs images of shape (N, {channels}, {PAPER_CROP}, {PAPER_CROP}), "
            f"got {tuple(x_img.shape)}"
        )
        raise ValueError(msg)


class PaperStampClassifier(nn.Module):
    """ALeRCE's stamp classifier: ``DeepHitsWithFeatures``, ``DeepHitsExtraLayer`` without features.

    ``forward(x_img, x_feat)`` takes 21x21 crops (N, 3, 21, 21) and, with features, the 23
    alert features (N, 23); it returns (N, ``n_classes``) logits (5, CD21's classes, by
    default; an N-class dataset passes its own count, task 3.1b):

    1. the 4 rotations of the batch (``torch.rot90``, counter-clockwise like
       ``tf.image.rot90``) go through the trunk as one batch of 4N:
       ``ZeroPad2d(3)`` (21 -> 27), conv1 4x4 valid (-> 24), conv2 5x5 same, 2x2 max-pool
       (-> 12), conv3-conv5 5x5 same, 2x2 max-pool (-> 6), flatten (2304), fc1 to 64, ReLU and
       a dropout layer (one per rotation, before the mean);
    2. the mean over the 4 rotations;
    3. with features: ``bn_features`` on the features, concatenated (64 + 23 = 87);
    4. fc2 to 64, ReLU, dropout; fc3 to 64, ReLU, dropout; ``logits`` to ``n_classes``.

    Every convolution and fc1-fc3 is followed by a ReLU. Facts of M1 this reproduces
    (verified against its meta graph and checkpoint; ``docs/phases/02-stamp-classifier.md``):

    - **Dropout was inactive in training.** ALeRCE calls Keras ``Dropout`` without a
      ``training`` argument, and the learning phase it then follows was never fed, so the
      dropout branches never ran. ``dropout_active=False`` (the default, faithful) keeps the
      layers but never applies them, in train or eval mode. ``dropout_active=True`` applies
      ``nn.Dropout(drop_rate)`` in train mode (diagnostic 1 of task 2.4, ``drop_rate=0.5``).
    - **The feature BatchNorm's running statistics were never updated** (M1's
      ``moving_mean`` is 0 and ``moving_variance`` 1). Here it is
      ``BatchNorm1d(23, eps=1e-3, momentum=0.0)``: training normalises with the batch
      statistics, and the running buffers stay at (0, 1), so evaluation computes
      ``gamma * x / sqrt(1 + 1e-3) + beta``.
    - **Initialisation:** Glorot-uniform kernels (Keras' default; the same fans as
      ``nn.init.xavier_uniform_``); zero biases on the convolutions and ``logits``; but
      Glorot-uniform biases on fc1-fc3, U(-0.2165, 0.2165) (see :func:`_glorot_bias_`: ALeRCE
      passes ``bias_initializer=None``, and M1's graph initialises those biases with
      ``random_uniform``). BN: gamma 1, beta 0. Weights are drawn from the global torch
      generator, so ``seed_everything`` fixes them.
    - No flips: only the 4 rotations.

    Trainable parameters: 441,075 with features, 439,557 without (M1's counts).

    Two variants of task 2.5-models (``paper_variants``; the defaults are the faithful model,
    whose initialisation draws and outputs they leave unchanged):

    - ``in_channels=6`` takes the 63x63 context (``data.float_input.context_split``) as 3
      more channels after the crop's: only conv1 widens (+1,536 parameters).
    - ``rotation_mean=False`` is the single-pass model: the trunk runs once on the
      unrotated batch, with no mean (a quarter of the MACs, not invariant; the control arm
      trained with rot90 augmentation).

    Raises:
        ValueError: at construction, if ``drop_rate`` is not in [0, 1), ``in_channels``
            is not positive or ``n_classes`` is not an integer >= 2; in ``forward``, if the
            images are not (N, ``in_channels``, 21, 21), or with features if ``x_feat`` is
            missing or not (N, 23).
    """

    def __init__(
        self,
        *,
        with_features: bool = True,
        dropout_active: bool = False,
        drop_rate: float = PAPER_DROP_RATE,
        in_channels: int = PAPER_IN_CHANNELS,
        rotation_mean: bool = True,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        if not 0 <= drop_rate < 1:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        check_n_classes(n_classes)
        self.with_features = with_features
        self.dropout_active = dropout_active
        self.in_channels = in_channels
        self.rotation_mean = rotation_mean
        # The layers are built (and reset_parameters draws them) in M1's order: the
        # faithful model's seeded initialisation must not change.
        self.pad = nn.ZeroPad2d(PAPER_PAD)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding="same")
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding="same")
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding="same")
        self.conv5 = nn.Conv2d(64, 64, kernel_size=5, padding="same")
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * _LAST_SIDE * _LAST_SIDE, _HIDDEN)
        self.drop1 = nn.Dropout(drop_rate)
        self.bn_features = (
            nn.BatchNorm1d(N_FEATURES, eps=_BN_EPS, momentum=0.0) if with_features else None
        )
        self.fc2 = nn.Linear(_HIDDEN + (N_FEATURES if with_features else 0), _HIDDEN)
        self.drop2 = nn.Dropout(drop_rate)
        self.fc3 = nn.Linear(_HIDDEN, _HIDDEN)
        self.drop3 = nn.Dropout(drop_rate)
        self.logits = nn.Linear(_HIDDEN, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise every parameter as M1's graph did (see the class docstring).

        Layers are initialised in network order (conv1 ... logits), kernel then bias.
        """
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            nn.init.xavier_uniform_(conv.weight)
            if conv.bias is not None:  # always: the convolutions are built with bias=True
                nn.init.zeros_(conv.bias)
        for dense in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(dense.weight)
            _glorot_bias_(dense.bias)
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        if self.bn_features is not None:
            self.bn_features.reset_parameters()  # gamma 1, beta 0, running (0, 1)

    def _dropout(self, layer: nn.Dropout, x: torch.Tensor) -> torch.Tensor:
        return layer(x) if self.dropout_active else x

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 21, 21) -> (B, 64): padding, the convolutions, fc1, ReLU and dropout."""
        x = self.pad(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(F.relu(self.conv5(x)))
        x = F.relu(self.fc1(torch.flatten(x, start_dim=1)))
        return self._dropout(self.drop1, x)

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Crops (N, ``in_channels``, 21, 21) and, with features, ``x_feat`` (N, 23) -> logits.

        Without features, ``x_feat`` is ignored.
        """
        check_paper_images(x_img, self.in_channels, type(self).__name__)
        n = x_img.shape[0]
        if self.rotation_mean:
            rotations = torch.cat([torch.rot90(x_img, k, dims=(2, 3)) for k in range(_N_ROTATIONS)])
            h = self._trunk(rotations).view(_N_ROTATIONS, n, _HIDDEN).mean(dim=0)
        else:
            h = self._trunk(x_img)
        return self._head(h, x_feat)

    def _head(self, h: torch.Tensor, x_feat: torch.Tensor | None) -> torch.Tensor:
        """The 64-d image representation and the features -> (N, n_classes) logits."""
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
