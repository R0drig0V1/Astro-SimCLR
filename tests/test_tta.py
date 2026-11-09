"""Tests of the test-time augmentation views and their averaging (plan row D4, ``tta``).

The pure functions directly: the view sets of each mode and model symmetry, the dihedral
transforms, the logit mean (one view returned untouched: ``rot4`` is the model's own
forward), the invariance of ``d4`` and ``d4_shift`` under every dihedral transform of the
input for each paper architecture, the shifted crops of ``data.float_input`` and the
refusals of ``check_tta``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astro_simclr.cli import TTA_CHOICES
from astro_simclr.config import PAPER_ARCHS
from astro_simclr.data.float_input import crop_split, crop_then_normalise
from astro_simclr.paper_model import PaperStampClassifier
from astro_simclr.paper_variants import build_paper_model
from astro_simclr.tta import (
    ARCH_SYMMETRY,
    SHIFT_OFFSETS,
    TTA_MODES,
    View,
    apply_dihedral,
    check_tta,
    combine_view_logits,
    model_symmetry,
    tta_logits,
    tta_views,
)

#: The 8 elements of D4 as views (identity first).
_D4 = [View(k, flip) for flip in (False, True) for k in range(4)]


def test_modes_and_symmetries_cover_the_cli_and_every_arch() -> None:
    assert TTA_CHOICES == TTA_MODES == ("rot4", "d4", "d4_shift")
    assert set(ARCH_SYMMETRY) == set(PAPER_ARCHS)
    assert [model_symmetry(a) for a in ("rot4", "single", "p4", "p4m")] == [
        "c4",
        "none",
        "c4",
        "d4",
    ]
    with pytest.raises(ValueError, match="paper_arch"):
        model_symmetry("resnet")


@pytest.mark.parametrize(
    ("symmetry", "d4_views"), [("c4", 2), ("d4", 1), ("none", 8)], ids=["c4", "d4", "none"]
)
def test_view_counts(symmetry: str, d4_views: int) -> None:
    rot4 = tta_views("rot4", symmetry)  # type: ignore[arg-type]
    d4 = tta_views("d4", symmetry)  # type: ignore[arg-type]
    shifted = tta_views("d4_shift", symmetry)  # type: ignore[arg-type]
    assert rot4 == (View(),)
    assert rot4[0].is_identity
    assert len(d4) == d4_views
    assert d4[0] == View()
    assert len(shifted) == 9 * d4_views
    assert len(set(shifted)) == len(shifted)
    assert {view.offset for view in shifted} == set(SHIFT_OFFSETS)
    assert all(view.offset == (0, 0) for view in d4)
    with pytest.raises(ValueError, match="tta must be"):
        tta_views("d8", symmetry)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="symmetry"):
        tta_views("d4", "c8")  # type: ignore[arg-type]


def test_c4_cosets_cover_d4() -> None:
    """identity and flip, each with the model's 4 internal rotations, are the 8 elements."""
    x = torch.arange(9.0).reshape(1, 1, 3, 3)
    covered = {
        tuple(torch.rot90(apply_dihedral(x, view), k, dims=(2, 3)).flatten().tolist())
        for view in tta_views("d4", "c4")
        for k in range(4)
    }
    every = {tuple(apply_dihedral(x, g).flatten().tolist()) for g in _D4}
    assert covered == every
    assert len(every) == 8


def test_view_validation_and_names() -> None:
    assert View().describe() == "identity"
    assert View(1, True).describe() == "flip+rot90"
    assert View(2, False, (-1, 0)).describe() == "rot180@(-1,0)"
    assert not View(offset=(0, 1)).is_identity
    for bad in (4, -1, True):
        with pytest.raises(ValueError, match="rotation"):
            View(bad)


def test_apply_dihedral() -> None:
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    assert apply_dihedral(x, View()) is x
    expected = torch.rot90(torch.flip(x, dims=(3,)), 3, dims=(2, 3))
    assert torch.equal(apply_dihedral(x, View(3, True)), expected)
    assert torch.equal(apply_dihedral(x, View(0, True)), torch.flip(x, dims=(3,)))
    images = {tuple(apply_dihedral(x, g).flatten().tolist()) for g in _D4}
    assert len(images) == 8
    with pytest.raises(ValueError, match="images"):
        apply_dihedral(x[0], View())


def test_combine_view_logits() -> None:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[3.0, 0.0], [1.0, 0.0]])
    assert combine_view_logits([a]) is a
    assert torch.equal(combine_view_logits([a, b]), torch.tensor([[2.0, 1.0], [2.0, 2.0]]))
    with pytest.raises(ValueError, match="no view"):
        combine_view_logits([])
    with pytest.raises(ValueError, match="must all be"):
        combine_view_logits([a, b[:1]])
    with pytest.raises(ValueError, match="must all be"):
        combine_view_logits([a.flatten()])


def _model(arch: str) -> torch.nn.Module:
    torch.manual_seed(0)
    model = build_paper_model(arch, group_fields=None, context="none", with_features=True)
    return model.eval()


def _inputs(side: int = 21) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    images = torch.rand(3, 3, side, side, generator=generator)
    return images, torch.randn(3, 23, generator=generator)


def test_rot4_is_the_models_own_forward() -> None:
    """``rot4`` returns the model's logits themselves: today's evaluation, bit for bit."""
    model = _model("rot4")
    assert isinstance(model, PaperStampClassifier)
    images, features = _inputs()
    with torch.inference_mode():
        expected = model(images, features)
        got = tta_logits(lambda x: model(x, features), images, tta_views("rot4", "c4"))
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    "arch", ["rot4", "single", "p4", "p4m", "convnext_atto", "hybrid", "hybrid_p4"]
)
def test_d4_is_invariant_under_every_dihedral_transform(arch: str) -> None:
    model = _model(arch)
    side = 63 if arch.startswith("convnext") else 21
    images, features = _inputs(side)
    views = tta_views("d4", model_symmetry(arch))
    with torch.inference_mode():

        def forward(x: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = model(x, features)
            return out

        reference = tta_logits(forward, images, views)
        for g in _D4[1:]:
            moved = tta_logits(forward, apply_dihedral(images, g), views)
            torch.testing.assert_close(moved, reference, rtol=1e-5, atol=1e-5)


def test_rot4_of_the_single_pass_model_is_not_invariant() -> None:
    """The control: without TTA, the single-pass model changes under a flip."""
    model = _model("single")
    images, features = _inputs()
    with torch.inference_mode():
        plain = model(images, features)
        flipped = model(apply_dihedral(images, View(0, True)), features)
    assert not torch.allclose(plain, flipped, atol=1e-4)


def _stamps(n: int = 3) -> np.ndarray:
    return np.random.default_rng(2).random((n, 63, 63, 3), dtype=np.float32) + np.float32(0.01)


def _shift_tta(model: torch.nn.Module, stamps: np.ndarray, features: torch.Tensor) -> torch.Tensor:
    """``d4_shift`` from 63x63 stamps (NHWC): each view's crop, then its dihedral transform."""
    logits = []
    for view in tta_views("d4_shift", "c4"):
        crops = crop_then_normalise(stamps, offset=view.offset).transpose(0, 3, 1, 2)
        batch = apply_dihedral(torch.from_numpy(np.ascontiguousarray(crops)), view)
        logits.append(model(batch, features))
    return combine_view_logits(logits)


def test_d4_shift_is_invariant_under_dihedral_transforms_of_the_stamp() -> None:
    model = _model("rot4")
    stamps = _stamps()
    features = _inputs()[1]
    with torch.inference_mode():
        reference = _shift_tta(model, stamps, features)
        for flip in (False, True):
            for k in range(4):
                if not flip and k == 0:
                    continue
                moved = stamps[:, :, ::-1, :] if flip else stamps
                moved = np.ascontiguousarray(np.rot90(moved, k, axes=(1, 2)))
                got = _shift_tta(model, moved, features)
                torch.testing.assert_close(got, reference, rtol=1e-5, atol=1e-5)


def test_shifted_crops() -> None:
    stamps = _stamps()
    centre = crop_then_normalise(stamps)
    assert np.array_equal(crop_then_normalise(stamps, offset=(0, 0)), centre)
    # A shift by (dy, dx) is the centre crop of the stamp rolled by (-dy, -dx).
    for dy, dx in SHIFT_OFFSETS:
        rolled = np.roll(stamps, (-dy, -dx), axis=(1, 2))
        shifted = crop_then_normalise(stamps, offset=(dy, dx))
        assert np.array_equal(shifted, crop_then_normalise(rolled))
        assert np.array_equal(crop_split(stamps, offset=(dy, dx), chunk=2), shifted)
    # The largest shifts reach the stamp's corners: no padding is ever needed.
    corner = np.ascontiguousarray(stamps[:, 42:, :21, :])
    assert np.array_equal(
        crop_then_normalise(stamps, offset=(21, -21)), crop_then_normalise(corner)
    )
    for bad in ((22, 0), (0, -22), (True, 0), (0.5, 0)):
        with pytest.raises(ValueError, match="out of the stamp"):
            crop_then_normalise(stamps, offset=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of the stamp"):
        crop_split(stamps, offset=(0, 30))


def test_shift_direction_against_an_independent_slice() -> None:
    """``offset=(1, -1)`` is rows 22-42 and columns 20-40 of the 63x63 stamp, min-maxed."""
    stamps = _stamps()
    window = stamps[:, 22:43, 20:41, :].copy()
    window -= window.min(axis=(1, 2), keepdims=True)
    window /= window.max(axis=(1, 2), keepdims=True)
    shifted = crop_then_normalise(stamps, offset=(1, -1), nan_approximation=False)
    assert np.array_equal(shifted, window)


def test_check_tta() -> None:
    for mode in TTA_MODES:
        check_tta(mode, paper_arch="rot4", context="none", paper_input="float32")
    check_tta("d4", paper_arch="convnext_atto", context="none", paper_input="float32")
    check_tta("d4", paper_arch="rot4", context="avgpool63", paper_input="uint8")
    refusals = [
        ({"paper_arch": "rot4", "context": "none", "paper_input": "uint8"}, "float32"),
        ({"paper_arch": "convnext_pico", "context": "none", "paper_input": "float32"}, "whole"),
        ({"paper_arch": "p4", "context": "avgpool63", "paper_input": "float32"}, "context"),
    ]
    for kwargs, match in refusals:
        with pytest.raises(ValueError, match=match):
            check_tta("d4_shift", **kwargs)
    with pytest.raises(ValueError, match="tta must be"):
        check_tta("d8", paper_arch="rot4", context="none", paper_input="float32")
    with pytest.raises(ValueError, match="paper_arch"):
        check_tta("d4", paper_arch="vit", context="none", paper_input="float32")
    with pytest.raises(ValueError, match="crops re-taken"):
        tta_logits(lambda x: x, torch.zeros(1, 3, 21, 21), tta_views("d4_shift", "d4"))
