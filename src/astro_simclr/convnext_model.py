"""ConvNeXt-atto and ConvNeXt-pico stamp classifiers at the native 63 px (plan row B8; ADR 0022).

The ZTF groups that moved past custom CNNs deploy ConvNeXt-pico (Rehemtulla et al. 2025,
arXiv:2512.11957; ORACLE-2, arXiv:2607.00228), so it is the "modern CNN" control that any
transformer row of task 2.5 must beat. :class:`ConvNeXtStampClassifier` puts a timm ConvNeXt
backbone in the paper model's frame, so that it is one ``paper_arch`` flag against the
replication (``paper_variants.build_paper_model``):

1. **Input: the whole 63x63 stamp**, not the paper's 21x21 crop. The data path is the
   paper's float path at the full side (``data.float_input.crop_split(images, 63)``: the
   per-channel min-max on the stamp without epsilon, NaN -> 0, the NaN approximation), 3
   channels (science, template, difference). The model zero-pads it to 64 at the bottom and
   the right, after rotating it (point 3), so the padding does not break the invariance.
2. **Stage maps 32-16-8-4.** timm's ConvNeXt patchifies with a 4x4, stride-4 stem, which at
   64 px leaves 16-8-4-2 maps. timm's ``patch_size=2`` does not fix it: timm then adds a
   stride-2 downsample to stage 0 to keep the total stride at 32 (maps 32 at the stem, then
   16-8-4-2). Here the backbone keeps timm's default stem *shape* (4x4 kernel, so that the
   Zoobot weights load unchanged; stage 0 then has no downsample) and the stem runs at
   stride 2 with padding 1: an overlapping 4x4 / 2 stem, 64 -> 32, then stages at 32, 16, 8
   and 4 (:data:`STAGE_SIDES`, checked by the tests). The scratch and pretrained models are
   the same architecture, so the two rows differ only in their initial weights.
3. **Exact C4 invariance by rotation averaging**, as the paper model: the 4 rotations of the
   batch go through the backbone as one batch of 4N; the global-average-pooled, normalised
   features (320 for atto, 512 for pico) go through ``fc1`` to 64 units, ReLU and a dropout
   layer per rotation; then the mean over the 4 rotations. The logits are invariant to the
   rotations of the input by 90 degrees up to rounding (the order of the 4-term mean).
4. **The paper's head** (:class:`~astro_simclr.paper_variants.PaperHead`): the frozen
   feature BN on the features that ``feature_set`` selects, fc2, fc3 and the logits.

=====================  ==========  ========  ==============  ===============
arch                   Parameters  Backbone  MACs per view   MACs per stamp
=====================  ==========  ========  ==============  ===============
``convnext_atto``      3,405,227   3,374,520  178.6 M        714.6 M
``convnext_pico``      8,575,667   8,532,672  446.3 M        1,785.0 M
=====================  ==========  ========  ==============  ===============

(with the 23 features; :func:`~astro_simclr.paper_variants.count_macs` over the 4 rotations,
divided by 4 for a view; convolutions and matrix products only, layer norms and GELUs not
counted; ``tests/test_convnext_model.py`` pins the exact values.) The paper
model is 441,075 parameters and 52.7 M MACs per view (210.6 M per stamp).

**Initialisation.** timm's (truncated normal, std 0.02, zero biases; LayerScale 1e-6), drawn
from the global torch generator at construction, so ``seed_everything`` fixes it; the
replaced stem is drawn the same way, after the backbone. ``fc1`` and the head follow the
paper (Glorot-uniform weights; Glorot-uniform biases on fc1-fc3, zero on the logits). No
stochastic depth and no timm head dropout: the paper's dropout flags are the only dropout.

**Pretrained weights** (``Settings.backbone_weights: zoobot``): :data:`BACKBONE_WEIGHTS`
pins the Zoobot ConvNeXt-pico encoder (Walmsley et al. 2024, arXiv:2404.02973) on Hugging
Face by commit and SHA-256. :func:`fetch_backbone_weights` downloads it at run time (never
committed; the Hugging Face cache keeps it) and refuses a file whose size or SHA-256
differ, or a weights entry whose licence is not recorded. :func:`load_backbone_weights`
loads it into the backbone with every key and shape checked. Nothing here touches the
network unless those functions are called: a from-scratch model needs no network access
(and ``timm.create_model(pretrained=False)`` downloads nothing).

The 6-channel context (``context: avgpool63``) does not apply: the native input already is
the whole stamp. Nothing here imports Lightning (ADR 0014): the module is engine-free.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import torch
import torch.nn.functional as F
from torch import nn

from astro_simclr.config import BACKBONE_WEIGHTS_CHOICES, CONVNEXT_ARCHS
from astro_simclr.data.schema import STAMP_SHAPE, DataError
from astro_simclr.feature_sets import feature_columns
from astro_simclr.models import N_CLASSES, check_n_classes
from astro_simclr.paper_model import PAPER_DROP_RATE, PAPER_IN_CHANNELS
from astro_simclr.paper_variants import PaperHead

if TYPE_CHECKING:
    from timm.models.convnext import ConvNeXt

__all__ = [
    "BACKBONE_WEIGHTS",
    "CONVNEXT_ARCHS",
    "NATIVE_SIDE",
    "PADDED_SIDE",
    "STAGE_SIDES",
    "ZOOBOT_CONVNEXT_PICO",
    "BackboneWeights",
    "ConvNeXtStampClassifier",
    "WeightsError",
    "backbone_weights_for",
    "check_native_images",
    "fetch_backbone_weights",
    "load_backbone_weights",
]

#: Side of the input: the whole ZTF stamp.
NATIVE_SIDE: Final = STAMP_SHAPE[0]
#: Side after the zero padding (one row at the bottom, one column at the right).
PADDED_SIDE: Final = 64
#: Sides of the stem's output and of the 4 stages' outputs at :data:`PADDED_SIDE`.
STAGE_SIDES: Final = (32, 32, 16, 8, 4)
#: Width of fc1 (the paper's 64 units): the image representation the head reads.
_HIDDEN: Final = 64
#: The 4 rotations of the input (0, 90, 180 and 270 degrees; no flips), as the paper model.
_N_ROTATIONS: Final = 4
#: The stem's kernel (timm's default, the Zoobot weights' shape), stride and padding.
_STEM_KERNEL, _STEM_STRIDE, _STEM_PADDING = 4, 2, 1
#: timm's initialisation of convolutions and linear layers.
_TIMM_INIT_STD: Final = 0.02


class WeightsError(DataError):
    """Pretrained weights are unusable: unknown, unlicensed, corrupted or mismatched."""


@dataclass(frozen=True, slots=True)
class BackboneWeights:
    """One pinned set of pretrained backbone weights on the Hugging Face Hub.

    Attributes:
        name: the ``Settings.backbone_weights`` value that selects it.
        arch: the ``paper_arch`` it fits.
        repo_id: the Hub model repository.
        revision: the commit of ``repo_id`` the file is read at (a full SHA-1, never a
            branch: a pushed update cannot change a run).
        filename: the weights file in the repository (a ``torch.save`` state dict).
        size: its size in bytes.
        sha256: its SHA-256 (hex), checked after every download.
        licence: the SPDX identifier the source declares for the weights, or ``None`` when
            no licence is recorded: then :func:`fetch_backbone_weights` refuses it.
        licence_source: where the licence was read, and when.
    """

    name: str
    arch: str
    repo_id: str
    revision: str
    filename: str
    size: int
    sha256: str
    licence: str | None
    licence_source: str

    def record(self) -> dict[str, object]:
        """The fields a run records in its metrics (JSON-ready)."""
        return {
            "name": self.name,
            "arch": self.arch,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "licence": self.licence,
        }


#: The Zoobot ConvNeXt-pico encoder (Galaxy Zoo supervised pretraining, 224 px, 3 channels;
#: timm ``convnext_pico`` with ``num_classes=0``). The licence is the model card's front
#: matter (``license: apache-2.0``, also the Hub tag ``license:apache-2.0``); the card has no
#: other text. The Zoobot code repository (github.com/mwalmsley/zoobot) is GPL-3.0, but no
#: Zoobot code is used here, only the weights file. No Zoobot ConvNeXt-atto exists.
ZOOBOT_CONVNEXT_PICO: Final = BackboneWeights(
    name="zoobot",
    arch="convnext_pico",
    repo_id="mwalmsley/zoobot-encoder-convnext_pico",
    revision="ca4dc5278aa675d7982e690a1ec10de8e945963e",
    filename="pytorch_model.bin",
    size=34_178_370,
    sha256="b0311878fd133aea9c330097d8be2f66c835580145539f6dfe4d48fd5f8edf6c",
    licence="Apache-2.0",
    licence_source=(
        "huggingface.co/mwalmsley/zoobot-encoder-convnext_pico README.md front matter and "
        "Hub API tags at the pinned revision, read 2026-11-08"
    ),
)

#: ``Settings.backbone_weights`` value -> ``paper_arch`` -> the pinned weights.
BACKBONE_WEIGHTS: Final[Mapping[str, Mapping[str, BackboneWeights]]] = {
    "zoobot": {ZOOBOT_CONVNEXT_PICO.arch: ZOOBOT_CONVNEXT_PICO},
}

#: ``download(weights) -> local path``: :func:`_hub_download` by default; tests pass a fake.
Downloader = Callable[[BackboneWeights], Path]


def backbone_weights_for(name: str, arch: str) -> BackboneWeights | None:
    """The pinned weights ``name`` for ``arch``, or ``None`` for ``name == "none"``.

    Raises:
        WeightsError: if ``name`` is not a :data:`BACKBONE_WEIGHTS_CHOICES` value, or no
            weights of that name exist for ``arch``.
    """
    if name not in BACKBONE_WEIGHTS_CHOICES:
        valid = ", ".join(BACKBONE_WEIGHTS_CHOICES)
        raise WeightsError(f"backbone_weights must be one of {valid}, got {name!r}")
    if name == "none":
        return None
    by_arch = BACKBONE_WEIGHTS[name]
    if arch not in by_arch:
        msg = f"no {name} weights exist for {arch!r}; available: {', '.join(sorted(by_arch))}"
        raise WeightsError(msg)
    return by_arch[arch]


def check_native_images(x_img: torch.Tensor, channels: int, model: str) -> None:
    """Check that ``x_img`` is a batch of whole stamps, (N, ``channels``, 63, 63).

    Raises:
        ValueError: if it is not.
    """
    expected = (channels, NATIVE_SIDE, NATIVE_SIDE)
    if x_img.ndim != 4 or tuple(x_img.shape[1:]) != expected:
        msg = (
            f"{model} needs images of shape (N, {channels}, {NATIVE_SIDE}, {NATIVE_SIDE}), "
            f"got {tuple(x_img.shape)}"
        )
        raise ValueError(msg)


def _glorot_bias_(bias: torch.Tensor) -> None:
    """The paper's fc1-fc3 bias initialiser: U(+-sqrt(3 / units)) (``paper_model``)."""
    limit = math.sqrt(3.0 / bias.shape[0])
    nn.init.uniform_(bias, -limit, limit)


class ConvNeXtStampClassifier(nn.Module):
    """A timm ConvNeXt on the whole stamp, 4-rotation averaged, with the paper's head.

    See the module docstring. ``forward(x_img, x_feat)`` takes (N, ``in_channels``, 63, 63)
    stamps and, with features, the (N, 23) features; it returns (N, ``n_classes``) logits.
    ``backbone`` is the timm ``ConvNeXt`` (``num_classes=0``: its forward returns the
    pooled, normalised features), whose state dict is what :func:`load_backbone_weights`
    fills. ``input_side`` (63) tells ``count_macs`` and the data path the input size.

    Raises:
        ValueError: at construction, for an ``arch`` not in :data:`CONVNEXT_ARCHS`,
            ``drop_rate`` outside [0, 1), ``in_channels < 1``, ``n_classes < 2`` or an
            invalid ``feature_set``; in ``forward``, for images that are not
            (N, ``in_channels``, 63, 63), or with features, a missing or misshapen
            ``x_feat``.
        RuntimeError: if the installed timm builds a stage 0 with a downsample (its stage
            layout changed; the 32-16-8-4 maps would silently become 16-8-4-2).
    """

    #: The side of the stamps ``forward`` takes.
    input_side: Final = NATIVE_SIDE

    def __init__(
        self,
        arch: str = "convnext_pico",
        *,
        with_features: bool = True,
        dropout_active: bool = False,
        drop_rate: float = PAPER_DROP_RATE,
        in_channels: int = PAPER_IN_CHANNELS,
        n_classes: int = N_CLASSES,
        feature_set: str = "all",
    ) -> None:
        super().__init__()
        if arch not in CONVNEXT_ARCHS:
            valid = ", ".join(CONVNEXT_ARCHS)
            raise ValueError(f"arch must be one of {valid}, got {arch!r}")
        if not 0 <= drop_rate < 1:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        check_n_classes(n_classes)
        with_features = bool(feature_columns(feature_set, with_features=with_features))
        self.arch = arch
        self.with_features = with_features
        self.dropout_active = dropout_active
        self.in_channels = in_channels
        self.backbone = _convnext_backbone(arch, in_channels)
        self.fc1 = nn.Linear(self.backbone.num_features, _HIDDEN)
        self.drop1 = nn.Dropout(drop_rate)
        self.head = PaperHead(
            with_features=with_features,
            dropout_active=dropout_active,
            drop_rate=drop_rate,
            n_classes=n_classes,
            feature_set=feature_set,
        )
        nn.init.xavier_uniform_(self.fc1.weight)
        _glorot_bias_(self.fc1.bias)
        self.head.reset_parameters()

    def stage_sides(self, x_img: torch.Tensor) -> tuple[int, ...]:
        """The map sides of the stem and the 4 stages on ``x_img`` (a diagnostic).

        ``x_img`` is (N, ``in_channels``, 63, 63); it is padded as in :meth:`forward` (no
        rotation). Returns :data:`STAGE_SIDES` for this model.
        """
        check_native_images(x_img, self.in_channels, type(self).__name__)
        x = self.backbone.stem(_pad(x_img))
        sides = [int(x.shape[-1])]
        for stage in self.backbone.stages:
            x = stage(x)
            sides.append(int(x.shape[-1]))
        return tuple(sides)

    def representation(self, x_img: torch.Tensor) -> torch.Tensor:
        """(N, C, 63, 63) stamps -> (N, 64): the rotation mean of fc1's ReLU outputs."""
        check_native_images(x_img, self.in_channels, type(self).__name__)
        n = x_img.shape[0]
        views = torch.cat([torch.rot90(x_img, k, dims=(2, 3)) for k in range(_N_ROTATIONS)])
        h = F.relu(self.fc1(self.backbone(_pad(views))))
        if self.dropout_active:
            h = self.drop1(h)  # per rotation, before the mean (as the paper model)
        return h.view(_N_ROTATIONS, n, _HIDDEN).mean(dim=0)

    def forward(self, x_img: torch.Tensor, x_feat: torch.Tensor | None = None) -> torch.Tensor:
        """Stamps (N, C, 63, 63) and, with features, (N, 23) -> (N, n_classes) logits."""
        out: torch.Tensor = self.head(self.representation(x_img), x_feat)
        return out


def _pad(x: torch.Tensor) -> torch.Tensor:
    """63 -> 64: one zero row at the bottom and one zero column at the right."""
    extra = PADDED_SIDE - NATIVE_SIDE
    return F.pad(x, (0, extra, 0, extra))


def _convnext_backbone(arch: str, in_channels: int) -> ConvNeXt:
    """timm's ``arch`` without classifier, its stem at stride 2 (module docstring, point 2).

    Raises:
        RuntimeError: if timm's stage 0 has a downsample (class docstring).
    """
    # timm is imported here, not at module import: it is slow to import, and the paper
    # model and the CLI's other subcommands never need it.
    import timm
    from timm.layers.weight_init import trunc_normal_
    from timm.models.convnext import ConvNeXt

    backbone = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=in_channels)
    old = backbone.stem[0] if isinstance(backbone, ConvNeXt) else None
    if (
        not isinstance(backbone, ConvNeXt)
        or not isinstance(old, nn.Conv2d)
        or not isinstance(backbone.stages[0].downsample, nn.Identity)
    ):
        raise RuntimeError(
            f"timm's {arch} has an unexpected layout (a patch stem, and no downsample in "
            "stage 0, expected)"
        )
    stem = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=_STEM_KERNEL,
        stride=_STEM_STRIDE,
        padding=_STEM_PADDING,
    )
    trunc_normal_(stem.weight, std=_TIMM_INIT_STD)
    if stem.bias is not None:  # always: built with bias=True, as timm's stem
        nn.init.zeros_(stem.bias)
    backbone.stem[0] = stem
    return backbone


def fetch_backbone_weights(weights: BackboneWeights, *, download: Downloader | None = None) -> Path:
    """The local path of ``weights``, downloaded at run time and verified.

    ``download`` fetches the file at the pinned revision (default: the Hugging Face Hub
    client, which caches it under ``HF_HOME``; ``HF_HUB_OFFLINE=1`` reads the cache only).
    The file is accepted only if its size and SHA-256 are the pinned ones.

    Raises:
        WeightsError: if ``weights.licence`` is ``None`` (no licence recorded: the weights
            are refused before any download), the download fails, or the size or SHA-256
            differ.
    """
    if weights.licence is None:
        msg = (
            f"refusing the {weights.name} weights for {weights.arch}: no licence is recorded "
            f"for {weights.repo_id}; record it in convnext_model.BACKBONE_WEIGHTS first"
        )
        raise WeightsError(msg)
    fetch = _hub_download if download is None else download
    try:
        path = fetch(weights)
    except OSError as err:  # huggingface_hub's HTTP and offline errors are OSErrors
        raise WeightsError(f"cannot download {weights.repo_id}@{weights.revision}: {err}") from err
    size = path.stat().st_size
    if size != weights.size:
        raise WeightsError(f"{path}: {size} bytes, expected {weights.size} ({weights.repo_id})")
    digest = _sha256(path)
    if digest != weights.sha256:
        msg = f"{path}: SHA-256 {digest}, expected {weights.sha256} ({weights.repo_id})"
        raise WeightsError(msg)
    return path


def _hub_download(weights: BackboneWeights) -> Path:  # pragma: no cover - network
    """``huggingface_hub.hf_hub_download`` of ``weights`` at its pinned revision."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=weights.repo_id, filename=weights.filename, revision=weights.revision
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone_weights(
    model: ConvNeXtStampClassifier, weights: BackboneWeights, path: Path
) -> None:
    """Load the verified file ``path`` of ``weights`` into ``model.backbone``, in place.

    The file is read with ``torch.load(weights_only=True)`` (tensors only, no code), and
    loaded strictly: every key of the backbone, and nothing else, with the same shapes.
    ``fc1`` and the head keep their initialisation.

    Raises:
        WeightsError: if ``weights`` is not for ``model.arch``, the model does not take the
            3 stamp channels the weights were trained on, the file is not a state dict of
            tensors, or its keys or shapes differ from the backbone's.
    """
    if weights.arch != model.arch:
        raise WeightsError(f"the {weights.name} weights are for {weights.arch}, not {model.arch}")
    if model.in_channels != PAPER_IN_CHANNELS:
        msg = f"the {weights.name} weights take 3 channels, the model {model.in_channels}"
        raise WeightsError(msg)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise WeightsError(f"{path} is not a state dict of tensors")
    own = model.backbone.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    shapes = sorted(
        key for key in set(own) & set(state) if tuple(own[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shapes:
        msg = (
            f"{path} does not fit the {model.arch} backbone: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}, other shapes {shapes[:5]}"
        )
        raise WeightsError(msg)
    model.backbone.load_state_dict(state, strict=True)
