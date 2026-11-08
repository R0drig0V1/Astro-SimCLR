"""Optimizer factory shared by every training module.

The legacy modules each had an ``if/elif`` chain over the optimizer name with no ``else``:
the classifiers knew ``AdamW``, ``Adam`` and ``SGD``, and ``SimCLR`` knew only ``LARS``, so
any other name ended in ``UnboundLocalError`` (a known defect of the plan). Here one factory
knows all four names, every module accepts any of them, and an unknown name raises
:class:`UnknownOptimizerError` listing the valid ones.

The hyperparameters are the legacy ones: ``AdamW`` and ``Adam`` with ``betas=(0.5, 0.9)``
(and torch's default weight decay), ``SGD`` without momentum, and ``LARS`` with
``weight_decay=0``, ``momentum=0.9`` and Nesterov (the legacy SimCLR call).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import torch
from torch.optim import Optimizer

from astro_simclr.optim import LARS

__all__ = [
    "ADAM_BETAS",
    "OPTIMIZER_NAMES",
    "UnknownOptimizerError",
    "build_optimizer",
    "check_optimizer_name",
]

#: The names :func:`build_optimizer` accepts (the legacy spellings).
OPTIMIZER_NAMES: Final = ("AdamW", "Adam", "SGD", "LARS")
#: The legacy ``betas`` of Adam and AdamW.
ADAM_BETAS: Final = (0.5, 0.9)


class UnknownOptimizerError(ValueError):
    """The optimizer name is not one of :data:`OPTIMIZER_NAMES`."""


def check_optimizer_name(name: str) -> None:
    """Fail early, at module construction, on an optimizer name the factory does not know.

    Raises:
        UnknownOptimizerError: if ``name`` is not in :data:`OPTIMIZER_NAMES`.
    """
    if name not in OPTIMIZER_NAMES:
        msg = f"unknown optimizer {name!r}; valid names: {', '.join(OPTIMIZER_NAMES)}"
        raise UnknownOptimizerError(msg)


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    *,
    lars_legacy_trust_ratio: bool = True,
) -> Optimizer:
    """Build the optimizer ``name`` over ``params`` with learning rate ``lr``.

    ``lars_legacy_trust_ratio`` is passed to :class:`~astro_simclr.optim.LARS` as
    ``legacy_trust_ratio`` (ignored by the other optimizers).

    Raises:
        UnknownOptimizerError: if ``name`` is not in :data:`OPTIMIZER_NAMES`.
        ValueError: if torch or LARS rejects ``lr`` (for example a negative value).
    """
    check_optimizer_name(name)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, betas=ADAM_BETAS)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, betas=ADAM_BETAS)
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr)
    return LARS(
        params,
        lr=lr,
        weight_decay=0.0,
        momentum=0.9,
        use_nesterov=True,
        legacy_trust_ratio=lars_legacy_trust_ratio,
    )
