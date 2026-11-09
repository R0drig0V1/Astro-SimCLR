"""Tests of the optimizer factory (``training.optimizers``): the legacy hyperparameters.

Every optimizer the configs name is built with the legacy code's fixed hyperparameters, and
an unknown name fails before any training with the list of valid names. (The Lightning
Trainer, callback and data-module helpers this file also tested were removed in task 4.8.)
"""

from __future__ import annotations

import pytest
import torch

from astro_simclr.optim import LARS
from astro_simclr.training.optimizers import (
    OPTIMIZER_NAMES,
    UnknownOptimizerError,
    build_optimizer,
    check_optimizer_name,
)


def _params() -> list[torch.nn.Parameter]:
    return list(torch.nn.Linear(3, 2).parameters())


def test_optimizers_keep_the_legacy_hyperparameters() -> None:
    adamw = build_optimizer("AdamW", _params(), 0.01)
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["betas"] == (0.5, 0.9)
    adam = build_optimizer("Adam", _params(), 0.01)
    assert type(adam) is torch.optim.Adam
    assert adam.defaults["betas"] == (0.5, 0.9)
    sgd = build_optimizer("SGD", _params(), 0.01)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.defaults["momentum"] == 0
    lars = build_optimizer("LARS", _params(), 0.258, lars_legacy_trust_ratio=False)
    assert isinstance(lars, LARS)
    assert lars.defaults["use_nesterov"] is True
    assert lars.defaults["momentum"] == 0.9
    assert lars.defaults["weight_decay"] == 0.0
    assert lars.defaults["legacy_trust_ratio"] is False
    assert lars.defaults["lr"] == 0.258


def test_unknown_optimizer_lists_the_valid_names() -> None:
    with pytest.raises(UnknownOptimizerError, match="valid names: AdamW, Adam, SGD, LARS"):
        build_optimizer("RMSprop", _params(), 0.01)
    # The names are case-sensitive (legacy: an UnboundLocalError when training started).
    for name in ("adamw", "Lars"):
        with pytest.raises(UnknownOptimizerError):
            check_optimizer_name(name)
    for name in OPTIMIZER_NAMES:
        check_optimizer_name(name)
    assert set(OPTIMIZER_NAMES) == {"AdamW", "Adam", "SGD", "LARS"}
