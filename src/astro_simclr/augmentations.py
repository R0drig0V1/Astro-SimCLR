"""The ``Augmentation`` protocol shared by the datasets (task 1.2) and the transforms (1.5).

A dataset holds uint8 stamps (HxWx3, the output of the legacy ``img_float2int``) and asks
its augmentation for either one view (classifier datasets) or two independent views of the
same stamp (contrastive datasets). Keeping that contract here lets the two tasks be built in
parallel: the datasets depend only on this protocol, never on a concrete transform.

Randomness is not passed in: implementations draw from the global torch / numpy / Python
generators, which ``lightning.seed_everything(seed, workers=True)`` seeds per worker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch

__all__ = ["Augmentation", "Stamp"]

#: A stamp as stored by the datasets: uint8, shape (H, W, 3) (science, template, difference).
type Stamp = npt.NDArray[np.uint8]


@runtime_checkable
class Augmentation(Protocol):
    """Turns a stamp into one augmented view, or into a pair of views.

    Both methods return float tensors of shape (3, H', W'), where H'xW' is the configured
    crop size.
    """

    def view(self, img: Stamp) -> torch.Tensor:
        """Return one augmented view of ``img``."""

    def pair(self, img: Stamp) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independently augmented views of ``img`` (the SimCLR positive pair)."""
