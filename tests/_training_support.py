"""Constants shared by the training tests.

Training tests run on the CPU everywhere, and on MPS or CUDA only where one is available
(the portability rule of the plan): parametrize them with ``ACCELERATORS``.
"""

from __future__ import annotations

import pytest
import torch

#: Stamps per split of the synthetic files: small, so training tests stay fast.
N_PER_SPLIT = 10
#: ``classifier_f32_dir``: Train stamp ``LEAKED_TRAIN`` is a copy of Validation stamp
#: ``LEAKED_VALIDATION`` (the leaked stamp the paper model's Train drops).
LEAKED_TRAIN = 3
LEAKED_VALIDATION = 7

#: CPU always; MPS and CUDA only when present.
ACCELERATORS = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "mps",
        id="mps",
        marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"),
    ),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    ),
]
