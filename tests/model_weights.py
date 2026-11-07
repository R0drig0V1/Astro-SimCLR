"""Deterministic weights for the model characterization fixtures (task 1.4).

``scripts/characterize/models/generate.py`` fills the *legacy* networks with these values
and records their outputs; ``tests/test_models.py`` fills the ported networks the same way
and compares. The values depend only on parameter names and shapes, never on a random
generator, so the fixtures survive torch upgrades that change the default initialisation.
The ported classes keep the legacy attribute names, so both sides see the same names.

This file must stay importable with torch alone: the generator loads it by path from an
isolated environment.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def fill_deterministic(module: nn.Module) -> None:
    """Overwrite every parameter and floating buffer of ``module`` in place.

    Weights get ``sin`` waves scaled by ``1/sqrt(fan_in)`` (so activations neither vanish
    nor explode through a ResNet); biases and BatchNorm shifts get small values;
    BatchNorm ``running_var`` and ``weight`` stay near 1. Entries are visited in sorted
    name order and the phase depends on that index.
    """
    state = module.state_dict()
    with torch.no_grad():
        for index, name in enumerate(sorted(state)):
            tensor = state[name]
            if not tensor.is_floating_point():
                continue
            wave = torch.sin(
                0.37 * torch.arange(tensor.numel(), dtype=torch.float64) + 1.3 * index
            ).reshape(tensor.shape)
            leaf = name.rsplit(".", 1)[-1]
            if leaf == "running_var":
                values = 1.0 + 0.25 * wave.abs()
            elif leaf == "running_mean":
                values = 0.1 * wave
            elif tensor.dim() == 1 and leaf == "weight":  # BatchNorm scale
                values = 1.0 + 0.1 * wave
            elif tensor.dim() == 1:  # biases, BatchNorm shift
                values = 0.05 * wave
            else:
                fan_in = tensor.numel() // tensor.shape[0]
                values = math.sqrt(3.0 / fan_in) * wave
            tensor.copy_(values.to(tensor.dtype))
