"""Tests for the Augmentation protocol (astro_simclr.augmentations)."""

import numpy as np
import torch

from astro_simclr.augmentations import Augmentation, Stamp


class _ToTensor:
    """Minimal implementation: HWC uint8 to CHW float in [0, 1], no randomness."""

    def view(self, img: Stamp) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255

    def pair(self, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
        return self.view(img), self.view(img)


def _consume(aug: Augmentation, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
    # Static check: a structural implementation is accepted where Augmentation is expected.
    return aug.pair(img)


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
