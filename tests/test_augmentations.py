"""Tests for astro_simclr.augmentations: the protocol, the registry and the ported transforms.

The characterization test replays the seeds of scripts/characterize/augmentations/ on the
port and compares with the views the legacy classes produced (tests/fixtures/augmentations/).
"""

from __future__ import annotations

import multiprocessing
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from _pair_dataset import PairDataset
from lightning import seed_everything
from lightning.fabric.utilities.seed import pl_worker_init_function
from PIL import Image
from torch.utils.data import DataLoader

from astro_simclr.augmentations import (
    AUGMENTATIONS,
    DEFERRED_AUGMENTATIONS,
    LEGACY_ALIASES,
    Augmentation,
    DeferredAugmentationError,
    GaussianBlur,
    PipelineAugmentation,
    RandomResizedCenterCrop,
    Stamp,
    UnknownAugmentationError,
    get_augmentation,
    resolve_augmentation_name,
)

FIXTURE = Path(__file__).parent / "fixtures" / "augmentations" / "legacy_views.npz"
NAMES = sorted(AUGMENTATIONS)


class _ToTensor:
    """Minimal implementation: HWC uint8 to CHW float in [0, 1], no randomness."""

    def view(self, img: Stamp) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255

    def pair(self, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
        return self.view(img), self.view(img)


def _consume(aug: Augmentation, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
    # Static check: a structural implementation is accepted where Augmentation is expected.
    return aug.pair(img)


def _stamps(n: int, size: int = 63, seed: int = 0) -> Stamp:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


def _seed_all(seed: int) -> None:
    # What seed_everything does to the three global generators, without its log line.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_structural_implementation_satisfies_protocol() -> None:
    aug = _ToTensor()
    assert isinstance(aug, Augmentation)
    img = np.full((27, 27, 3), 255, dtype=np.uint8)
    first, second = _consume(aug, img)
    assert first.shape == second.shape == (3, 27, 27)
    assert torch.equal(aug.view(img), torch.ones(3, 27, 27))


def test_object_without_pair_is_not_an_augmentation() -> None:
    class _ViewOnly:
        def view(self, img: Stamp) -> torch.Tensor:
            return torch.zeros(1)

    assert not isinstance(_ViewOnly(), Augmentation)


# --- registry -----------------------------------------------------------------------------


def test_registry_holds_exactly_the_phase1_names() -> None:
    assert set(AUGMENTATIONS) == {
        "simclr",
        "simclr_centered",
        "astro_perspective",
        "astro_perspective_gray",
        "rotate_flip",
        "center_crop",
    }
    assert set(LEGACY_ALIASES) == {
        "simclr",
        "simclr2",
        "astro7",
        "astro8",
        "rotation_v3",
        "without_aug",
    }
    assert set(LEGACY_ALIASES.values()) == set(AUGMENTATIONS)
    # No new name may shadow a different legacy transform.
    assert not (set(AUGMENTATIONS) | set(LEGACY_ALIASES)) & DEFERRED_AUGMENTATIONS


@pytest.mark.parametrize(("legacy_name", "name"), sorted(LEGACY_ALIASES.items()))
def test_legacy_alias_builds_the_same_augmentation(legacy_name: str, name: str) -> None:
    assert resolve_augmentation_name(legacy_name) == name
    assert resolve_augmentation_name(name) == name
    aug = get_augmentation(legacy_name, 27)
    assert aug.name == name
    img = _stamps(1)[0]
    seed_everything(5, verbose=False)
    via_alias = aug.pair(img)
    seed_everything(5, verbose=False)
    via_name = get_augmentation(name, 27).pair(img)
    assert torch.equal(via_alias[0], via_name[0])
    assert torch.equal(via_alias[1], via_name[1])


@pytest.mark.parametrize("name", ["astro9", "rot_elastic", "simclr3", "gray_scale"])
def test_deferred_legacy_name_says_deferred(name: str) -> None:
    with pytest.raises(DeferredAugmentationError, match="deferred"):
        get_augmentation(name, 27)


def test_unknown_name_lists_valid_names() -> None:
    with pytest.raises(UnknownAugmentationError, match="valid names") as info:
        get_augmentation("simclr_v9", 27)
    assert not isinstance(info.value, DeferredAugmentationError)
    assert "rotate_flip" in str(info.value)
    assert "rotation_v3" in str(info.value)


@pytest.mark.parametrize("size", [0, -3, True, 27.0])
def test_bad_size_is_rejected(size: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        get_augmentation("simclr", size)  # type: ignore[arg-type]


def test_augmentation_satisfies_protocol_and_has_repr() -> None:
    aug = get_augmentation("astro_perspective_gray", 27)
    assert isinstance(aug, Augmentation)
    assert isinstance(aug, PipelineAugmentation)
    assert repr(aug) == "PipelineAugmentation(name='astro_perspective_gray')"


# --- shapes, ranges, randomness -----------------------------------------------------------


@pytest.mark.parametrize("size", [21, 27, 63])
@pytest.mark.parametrize("name", NAMES)
def test_views_have_shape_dtype_and_range(name: str, size: int) -> None:
    seed_everything(0, verbose=False)
    aug = get_augmentation(name, size)
    for img in _stamps(3):
        for view in (aug.view(img), *aug.pair(img)):
            assert view.shape == (3, size, size)
            assert view.dtype == torch.float32
            assert float(view.min()) >= 0.0
            assert float(view.max()) <= 1.0
            # Every pipeline ends in ToTensor on a PIL image: values are exactly k / 255.
            scaled = view * 255
            assert torch.allclose(scaled, scaled.round(), atol=1e-3)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "center_crop"])
def test_random_augmentations_give_two_different_views(name: str) -> None:
    seed_everything(0, verbose=False)
    aug = get_augmentation(name, 27)
    img = _stamps(1)[0]
    assert any(not torch.equal(*aug.pair(img)) for _ in range(5))


def test_center_crop_is_deterministic() -> None:
    img = _stamps(1)[0]
    first, second = get_augmentation("center_crop", 27).pair(img)
    expected = torch.from_numpy(img[18:45, 18:45]).permute(2, 0, 1).float() / 255
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_same_views(name: str) -> None:
    img = _stamps(1)[0]
    runs = []
    for _ in range(2):
        seed_everything(7, verbose=False)
        runs.append(get_augmentation(name, 27).pair(img))
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])


# --- characterization against the legacy classes -------------------------------------------


@pytest.fixture(scope="module")
def legacy() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def test_fixture_records_provenance(legacy: dict[str, np.ndarray]) -> None:
    assert len(str(legacy["legacy_sha"])) == 40
    assert str(legacy["torch_version"]).startswith("2.")
    assert str(legacy["torchvision_version"])


@pytest.mark.parametrize("size", [27, 63])
@pytest.mark.parametrize("legacy_name", sorted(LEGACY_ALIASES))
def test_port_matches_legacy_views(
    legacy: dict[str, np.ndarray], legacy_name: str, size: int
) -> None:
    # The fixture is keyed by the legacy names; compare under the registry name.
    name = LEGACY_ALIASES[legacy_name]
    seed = int(legacy["seed"])
    aug = get_augmentation(name, size)
    for n, img in enumerate(legacy["stamps"]):
        _seed_all(seed + n)
        first, second = aug.pair(img)
        for got, key in ((first, "first"), (second, "second")):
            want = legacy[f"{legacy_name}_{size}_{key}"][n].astype(np.int16)
            diff = np.abs(np.rint(got.numpy() * 255).astype(np.int16) - want)
            # The only float step that is not exact is the Gaussian blur's conv2d, whose
            # last bit can differ between CPUs (fixtures are made on arm64, CI is x86),
            # and ToPILImage truncates to uint8: allow a rare off-by-one pixel.
            assert diff.max() <= 1, f"{name}/{size}/{key}/{n}: max diff {diff.max()}"
            assert (diff > 0).mean() < 1e-3, f"{name}/{size}/{key}/{n}: too many diffs"


# --- building blocks -----------------------------------------------------------------------


def test_center_crop_rejects_reversed_ranges() -> None:
    with pytest.raises(ValueError, match="min, max"):
        RandomResizedCenterCrop(27, scale=(1.0, 0.5))
    with pytest.raises(ValueError, match="min, max"):
        RandomResizedCenterCrop(27, ratio=(2.0, 1.0))


@pytest.mark.parametrize(
    ("width", "height", "ratio", "expected"),
    [
        # No draw can fit (scale > 1), so the fallback picks the largest centred crop.
        (40, 10, (0.75, 1.3333), (0, 13, 10, 13)),  # too wide: clamp to max ratio
        (10, 40, (0.75, 1.3333), (13, 0, 13, 10)),  # too tall: clamp to min ratio
        (30, 30, (0.75, 1.3333), (0, 0, 30, 30)),  # within ratio: whole image
    ],
)
def test_center_crop_fallback(
    width: int,
    height: int,
    ratio: tuple[float, float],
    expected: tuple[int, int, int, int],
) -> None:
    img = Image.new("RGB", (width, height))
    assert RandomResizedCenterCrop.get_params(img, (5.0, 6.0), ratio) == expected


def test_center_crop_is_centred() -> None:
    random.seed(3)
    img = Image.new("RGB", (63, 63))
    i, j, h, w = RandomResizedCenterCrop.get_params(img, (0.18, 0.5), (0.75, 4 / 3))
    assert (i, j) == ((63 - h) // 2, (63 - w) // 2)


def test_gaussian_blur_preserves_a_constant_image() -> None:
    np.random.seed(0)
    img = Image.new("RGB", (27, 27), color=(100, 50, 200))
    out = np.asarray(GaussianBlur(kernel_size=6)(img))
    assert out.shape == (27, 27, 3)
    # ToPILImage truncates, so a constant may lose one level to float rounding.
    assert np.all(np.abs(out.astype(int) - [100, 50, 200]) <= 1)


# --- DataLoader reproducibility (plan 1.5 acceptance) --------------------------------------


# Workers start by fork where it is safe. A spawned worker (the macOS and Windows default)
# re-imports torch, torchvision and Lightning: about 5 s of CPU per worker, 26 s of the suite
# for this test alone. The start method does not change what the test checks: torch's worker
# loop seeds torch, numpy and `random` from the base seed in every worker whatever the start
# method, then the worker init function reseeds them. macOS does not fork here because fork
# after the Objective-C runtime or Metal has started (an MPS test in the same process) can
# crash the child.
_FORK = sys.platform != "darwin" and "fork" in multiprocessing.get_all_start_methods()


def _first_two_batches(name: str, num_workers: int, seed: int) -> list[list[torch.Tensor]]:
    seed_everything(seed, workers=True, verbose=False)
    # The worker init function is what the Lightning Trainer injects into its DataLoaders
    # after seed_everything(workers=True): it seeds torch, numpy and `random` per worker.
    loader = DataLoader(
        PairDataset(name),
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=pl_worker_init_function,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    batches = []
    for batch in loader:
        batches.append(list(batch))
        if len(batches) == 2:
            break
    return batches


# astro_perspective_gray uses all three generators: torch (torchvision), `random` (centre crop) and
# np.random (blur).
@pytest.mark.parametrize(
    "num_workers",
    [
        0,
        pytest.param(
            2,
            marks=pytest.mark.skipif(
                not _FORK,
                reason="DataLoader workers do not fork on this platform (spawn costs ~5 s per "
                "worker); the case runs on the Linux CI job",
            ),
        ),
    ],
)
def test_same_seed_gives_identical_first_two_batches(num_workers: int) -> None:
    first = _first_two_batches("astro_perspective_gray", num_workers, seed=11)
    second = _first_two_batches("astro_perspective_gray", num_workers, seed=11)
    assert len(first) == len(second) == 2
    for batch_a, batch_b in zip(first, second, strict=True):
        for a, b in zip(batch_a, batch_b, strict=True):
            assert torch.equal(a, b)
    # The seed matters.
    other = _first_two_batches("astro_perspective_gray", num_workers, seed=12)
    assert not torch.equal(first[0][0], other[0][0])
