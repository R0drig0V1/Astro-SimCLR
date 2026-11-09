"""Test-time augmentation (TTA) of the paper classifiers: views and logit averaging (plan row D4).

Pure, engine-free and deterministic: no randomness, no random stream, no Lightning. The
consumer is ``ensemble-eval`` (:mod:`astro_simclr.ensemble_eval`), which evaluates the
stored weights of finished runs; nothing here changes how a run trains or evaluates
itself.

**Where the averaging happens.** The paper model (CD21, ``paper_arch: rot4``,
:class:`~astro_simclr.paper_model.PaperStampClassifier`) is *cyclic-pooling*: its forward
runs the trunk on the 4 rotations of the crop and averages the 64-d **features** after fc1,
before the head (fc2, fc3, logits); the head sees one pooled vector. The ConvNeXt rows (B8)
pool their 4 rotations the same way, and the group CNNs pool over the orientations of
their last group convolution (``p4``: the 4 rotations; ``p4m``: the 8 elements of D4). A
TTA view is a transformed input that goes through the whole model, so its extra views are
averaged at the **logit** level, outside the model (:func:`combine_view_logits`: the
arithmetic mean of the views' logits, then the caller's softmax). Feeding a C4-pooled model
the 4 rotations again would only repeat its own feature mean (the 4 rotations of a rotated
stamp are the same 4 stamps), so the views are the cosets of the model's own symmetry
group (:func:`tta_views`):

=================  ===========  =========================  ===========================
mode               ``rot4``     ``d4``                     ``d4_shift``
=================  ===========  =========================  ===========================
C4 pooling         1 view       identity, flip (2)         2 x 9 offsets (18)
(rot4, p4,
ConvNeXt,
hybrid,
hybrid_p4)
D4 pooling (p4m)   1 view       1 view (already D4)        1 x 9 offsets (9)
none (single)      1 view       the 8 dihedral views       8 x 9 offsets (72)
=================  ===========  =========================  ===========================

- ``rot4`` (the default) is the model's own forward, unchanged: for the paper model the
  CD21 rotation mean, for ``single`` one pass. It is one identity view, whose logits are
  returned as they are (no arithmetic), so its probabilities are bit for bit the run's
  own evaluation.
- ``d4`` adds the reflections: for a C4-pooled model the flipped stamp, whose 4 internal
  rotations are the 4 reflections of the stamp, so the 2 views cover the 8 elements of
  D4 (feature-averaged within each view, logit-averaged across the two). For
  ``p4m``, which is D4-invariant already, it is the identity; for ``single``, all 8.
  The result is invariant under any dihedral transform of the input (up to float
  rounding of the model's own mean).
- ``d4_shift`` is ``d4`` x the 9 crop offsets ``(dy, dx)``, ``dy, dx in {-1, 0, 1}``: the
  central 21x21 crop re-taken 1 px off centre from the 63x63 stamp (no padding: the stamp
  has a 21-px margin), each crop normalised on its own pixels like the paper's
  (``data.float_input.crop_then_normalise(offset=...)``). The offset set is symmetric
  under D4, so the result stays D4-invariant. It needs the float stamps and a 21-px
  crop model without the context channels (:func:`check_tta`).

A view is ``rot90(flip(x), k)`` (:func:`apply_dihedral`): first the optional flip of the
columns, then ``k`` counter-clockwise quarter turns (``torch.rot90``, as the model's own
rotations), applied to the whole input, the context channels included (the 63x63
context rotates with the stamp, ``data.float_input``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import torch

from astro_simclr.config import CONVNEXT_ARCHS, PAPER_ARCHS

__all__ = [
    "ARCH_SYMMETRY",
    "SHIFT_OFFSETS",
    "TTA_MODES",
    "Symmetry",
    "TtaMode",
    "View",
    "apply_dihedral",
    "check_tta",
    "combine_view_logits",
    "model_symmetry",
    "tta_logits",
    "tta_views",
]

#: The TTA modes (module docstring); ``rot4`` is the default and the run's own evaluation.
type TtaMode = Literal["rot4", "d4", "d4_shift"]
TTA_MODES: Final[tuple[TtaMode, ...]] = get_args(TtaMode.__value__)

#: The input symmetry a model pools over internally: none, the 4 rotations, or D4.
type Symmetry = Literal["none", "c4", "d4"]

#: The symmetry of each ``paper_arch`` (``paper_variants``): what its forward averages.
ARCH_SYMMETRY: Final[Mapping[str, Symmetry]] = {
    "rot4": "c4",
    "single": "none",
    "p4": "c4",
    "p4m": "d4",
    **dict.fromkeys(CONVNEXT_ARCHS, "c4"),
    # The conv-attention hybrids (plan row B9; hybrid_model) are exactly C4-invariant, by
    # two routes. ``hybrid`` runs its paper trunk and attention on the 4 rotations and
    # averages the 64-d features after fc1, as rot4 does. ``hybrid_p4`` makes one pass: the
    # p4 trunk is pooled over its 4 orientations before the attention and the positional
    # embeddings are tied across the C4 orbits of the 6x6 grid, so a rotated stamp only
    # permutes the tokens, which the attention and the mean / CLS readout do not see.
    # Neither is reflection-invariant (the tied positions are C4, not D4, orbits).
    # tests/test_tta.py proves both: d4 TTA with these 2 views is D4-invariant only if the
    # model is C4-invariant.
    "hybrid": "c4",
    "hybrid_p4": "c4",
}

#: The crop offsets ``(dy, dx)`` of ``d4_shift``, row-major (the centre is the 5th).
SHIFT_OFFSETS: Final[tuple[tuple[int, int], ...]] = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
)

_CENTRE: Final = (0, 0)
_QUARTER_TURNS: Final = 4


@dataclass(frozen=True, slots=True)
class View:
    """One TTA view: ``rotation`` quarter turns after an optional ``flip``, at a crop offset.

    ``offset`` ``(dy, dx)`` is where the crop is taken in the 63x63 stamp (the caller
    builds that crop); ``rotation`` and ``flip`` are applied to it by :func:`apply_dihedral`.

    Raises:
        ValueError: if ``rotation`` is not in 0-3.
    """

    rotation: int = 0
    flip: bool = False
    offset: tuple[int, int] = _CENTRE

    def __post_init__(self) -> None:
        if isinstance(self.rotation, bool) or self.rotation not in range(_QUARTER_TURNS):
            raise ValueError(f"rotation must be 0-3 quarter turns, got {self.rotation!r}")

    @property
    def is_identity(self) -> bool:
        """Whether the view is the input itself (no flip, no turn, the centre crop)."""
        return self.rotation == 0 and not self.flip and self.offset == _CENTRE

    def describe(self) -> str:
        """A short name: ``identity``, ``flip+rot90``, ``rot180@(-1,0)``, ..."""
        parts = (["flip"] if self.flip else []) + (
            [f"rot{90 * self.rotation}"] if self.rotation else []
        )
        name = "+".join(parts) or "identity"
        return name if self.offset == _CENTRE else f"{name}@({self.offset[0]},{self.offset[1]})"


def model_symmetry(arch: str) -> Symmetry:
    """The symmetry ``paper_arch`` ``arch`` pools over internally (:data:`ARCH_SYMMETRY`).

    Raises:
        ValueError: for an ``arch`` not in ``config.PAPER_ARCHS``.
    """
    if arch not in PAPER_ARCHS or arch not in ARCH_SYMMETRY:
        raise ValueError(f"paper_arch must be one of {', '.join(PAPER_ARCHS)}, got {arch!r}")
    return ARCH_SYMMETRY[arch]


def _dihedral_cosets(symmetry: Symmetry) -> tuple[View, ...]:
    """Representatives of D4 modulo the model's own symmetry (identity first)."""
    if symmetry == "d4":
        return (View(),)
    if symmetry == "c4":
        return (View(), View(flip=True))
    return tuple(View(k, flip) for flip in (False, True) for k in range(_QUARTER_TURNS))


def tta_views(mode: str, symmetry: Symmetry) -> tuple[View, ...]:
    """The views of TTA ``mode`` for a model with internal ``symmetry`` (module docstring).

    ``rot4`` is always the single identity view. ``d4`` is the dihedral views the model
    does not already pool over; ``d4_shift`` repeats them at each of :data:`SHIFT_OFFSETS`
    (offsets outer, dihedral views inner).

    Raises:
        ValueError: for a mode not in :data:`TTA_MODES` or an unknown symmetry.
    """
    if mode not in TTA_MODES:
        raise ValueError(f"tta must be one of {', '.join(TTA_MODES)}, got {mode!r}")
    if symmetry not in get_args(Symmetry.__value__):
        raise ValueError(f"unknown symmetry {symmetry!r}")
    if mode == "rot4":
        return (View(),)
    cosets = _dihedral_cosets(symmetry)
    if mode == "d4":
        return cosets
    return tuple(
        View(view.rotation, view.flip, offset) for offset in SHIFT_OFFSETS for view in cosets
    )


def apply_dihedral(images: torch.Tensor, view: View) -> torch.Tensor:
    """``images`` (N, C, H, W) under the view's flip then rotation (its offset is ignored).

    The flip mirrors the columns (``W``); the rotation is ``torch.rot90(k, dims=(2, 3))``,
    counter-clockwise like the model's own. The identity returns ``images`` itself.

    Raises:
        ValueError: if ``images`` is not 4-dimensional with square images.
    """
    if images.ndim != 4 or images.shape[2] != images.shape[3]:
        raise ValueError(f"images must be (N, C, S, S), got {tuple(images.shape)}")
    out = torch.flip(images, dims=(3,)) if view.flip else images
    return torch.rot90(out, view.rotation, dims=(2, 3)) if view.rotation else out


def combine_view_logits(logits: Sequence[torch.Tensor]) -> torch.Tensor:
    """The TTA logits: the mean over the views of their (N, C) logits.

    One view returns its logits unchanged (the ``rot4`` path stays bit-identical); more
    are stacked and averaged in their own dtype, in the given order.

    Raises:
        ValueError: if no view is given or the views' shapes differ.
    """
    if not logits:
        raise ValueError("no view logits to combine")
    shapes = {tuple(item.shape) for item in logits}
    if len(shapes) > 1 or logits[0].ndim != 2:
        raise ValueError(f"the views' logits must all be (N, C), got {sorted(shapes)}")
    if len(logits) == 1:
        return logits[0]
    return torch.stack(tuple(logits)).mean(dim=0)


def tta_logits(
    forward: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    views: Sequence[View],
) -> torch.Tensor:
    """The TTA logits of a batch whose views need no re-cropping (centre offsets only).

    ``forward`` maps (N, C, S, S) inputs to (N, n_classes) logits (the model, with its
    features bound). Each view's transformed batch goes through it once.

    Raises:
        ValueError: if a view has an offset (``d4_shift`` crops come from the stamps:
            the caller builds them), or as :func:`apply_dihedral` and
            :func:`combine_view_logits`.
    """
    shifted = [view.describe() for view in views if view.offset != _CENTRE]
    if shifted:
        raise ValueError(f"views {shifted} need crops re-taken from the stamps")
    return combine_view_logits([forward(apply_dihedral(images, view)) for view in views])


def check_tta(mode: str, *, paper_arch: str, context: str, paper_input: str) -> None:
    """Refuse a TTA mode that a run's input cannot support.

    ``d4_shift`` re-takes the 21x21 crop from the 63x63 float stamps, so it needs
    ``paper_input: float32``, a 21-px model (not a ConvNeXt, which sees the whole stamp:
    shifting it would need padding) and no context channels (the context is the whole
    stamp, which a crop shift does not move).

    Raises:
        ValueError: for an unknown mode or ``paper_arch``, or ``d4_shift`` on such a run.
    """
    if mode not in TTA_MODES:
        raise ValueError(f"tta must be one of {', '.join(TTA_MODES)}, got {mode!r}")
    model_symmetry(paper_arch)
    if mode != "d4_shift":
        return
    if paper_input != "float32":
        raise ValueError("tta d4_shift re-crops the float32 stamps: needs paper_input float32")
    if paper_arch in CONVNEXT_ARCHS:
        raise ValueError(f"tta d4_shift shifts the 21x21 crop; {paper_arch} sees the whole stamp")
    if context != "none":
        raise ValueError(
            f"tta d4_shift shifts the crop but not the context ({context}): use d4 for it"
        )
