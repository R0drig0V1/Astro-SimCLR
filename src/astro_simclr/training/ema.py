"""Weight EMA of the paper classifier (plan row D3): a shadow copy averaged after each step.

:class:`EmaWeights` keeps an exponential moving average of a model's parameters, the
"shadow" model, updated after every optimizer step::

    d_s     = ema_decay_at(decay, s, warmup=warmup)
    shadow <- shadow + (1 - d_s) * (theta - shadow)      (= d_s * shadow + (1 - d_s) * theta)

where ``s`` is the number of optimizer steps done *before* the step that produced
``theta`` (0 after the first step; the engines' ``step``, ADR 0014). With ``warmup`` the
decay is TF's ``ExponentialMovingAverage(num_updates=s)`` ramp, also the warmup of timm's
``ModelEmaV2``/``V3``: ``min(decay, (1 + s) / (10 + s))``, so the shadow forgets the
random initialisation within the first few hundred steps instead of the ~``1/(1 - decay)``
steps a fixed decay needs.

**Buffers are copied, not averaged.** The BatchNorm running statistics (and
``num_batches_tracked``) of the shadow are the training model's, copied at each update.
They are already a running average (of the batch statistics, momentum 0.1), and an
average of them would describe no model's activations: neither the shadow's nor the
training model's. Copying keeps the shadow's normalisation that of the most recent
training batches, which is what timm's ``ModelEmaV3(exclude_buffers=True)`` and most EMA
recipes for BN networks do (the alternative, recomputing the statistics of the shadow
with a pass over Train, would draw data outside the training stream). The paper model's
BN is the one on the 23 features (``paper_model``); the ConvNeXt rows use LayerNorm and
have no buffers.

**Where it lives.** The shadow is a deep copy of the model, a submodule of the module that
trains (``tasks.paper.PaperNet.ema``; ``PaperClassifierModule.ema`` on the Lightning
path until task 4.8): it moves with the
model to its device, its tensors are part of ``state_dict()`` (so ``last`` checkpoints
save it and a resume restores it exactly, and the best weights kept by the
``best_val_loss`` selection are the shadow's too), and its parameters have
``requires_grad=False`` and are left out of the optimizer (:func:`trainable_parameters`).
A module with an EMA evaluates with the shadow: its forward in ``eval()`` mode
(validation, test, the final evaluation) runs the shadow, in ``train()`` mode the
training model.

Nothing here imports Lightning (CLAUDE.md, "New training flags are engine-free"): the
plain task calls :meth:`EmaWeights.update` after the optimizer step with the step
number (as the Lightning module did), and nothing in it draws a random number.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator

import torch
from torch import nn

__all__ = ["EMA_PREFIX", "EmaWeights", "check_ema_decay", "ema_decay_at", "trainable_parameters"]

#: The attribute (and ``state_dict`` prefix) of the shadow in the modules that train it.
EMA_PREFIX = "ema"


def check_ema_decay(decay: float) -> float:
    """``decay`` if it is a valid EMA decay, in (0, 1).

    Raises:
        ValueError: if it is not (1 would never move, 0 is no average).
    """
    if not 0 < decay < 1:
        raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
    return decay


def ema_decay_at(decay: float, step: int, *, warmup: bool) -> float:
    """The decay of the update after optimizer step ``step`` (0-based; module docstring).

    Without ``warmup`` it is ``decay``; with it ``min(decay, (1 + step) / (10 + step))``
    (0.1 after the first step, 0.5 after the ninth, ``decay`` once the ramp passes it).

    Raises:
        ValueError: if ``step`` is negative.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if not warmup:
        return decay
    return min(decay, (1 + step) / (10 + step))


def trainable_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """``module.parameters()`` without the shadow's (the ``ema.`` submodule), in order.

    Without an EMA it is exactly ``module.parameters()``: the optimizer of a run without
    the flag is built over the same parameters in the same order as before it existed.
    """
    prefix = f"{EMA_PREFIX}."
    for name, parameter in module.named_parameters():
        if not name.startswith(prefix):
            yield parameter


class EmaWeights(nn.Module):
    """The EMA shadow of ``model`` (module docstring), initialised to a copy of it.

    ``shadow`` is a deep copy of ``model`` (its parameters frozen); the copy draws no
    random number, so building it does not move the global generator. Call
    :meth:`reset` after loading other weights into ``model`` (the ConvNeXt backbone,
    ADR 0022) so the shadow starts from them.

    Raises:
        ValueError: if ``decay`` is not in (0, 1).
    """

    def __init__(self, model: nn.Module, decay: float, *, warmup: bool = False) -> None:
        super().__init__()
        self.decay = check_ema_decay(decay)
        self.warmup = warmup
        self.shadow = copy.deepcopy(model)
        self.shadow.requires_grad_(False)

    def decay_at(self, step: int) -> float:
        """:func:`ema_decay_at` with this shadow's decay and warmup."""
        return ema_decay_at(self.decay, step, warmup=self.warmup)

    def _pairs(
        self, model: nn.Module
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
        """(shadow, model) parameter pairs and buffer pairs, by name.

        Raises:
            ValueError: if ``model`` does not have the shadow's parameter and buffer names.
        """
        mine: dict[str, torch.Tensor] = dict(self.shadow.named_parameters())
        theirs: dict[str, torch.Tensor] = dict(model.named_parameters())
        my_buffers = dict(self.shadow.named_buffers())
        their_buffers = dict(model.named_buffers())
        if mine.keys() != theirs.keys() or my_buffers.keys() != their_buffers.keys():
            raise ValueError("the model does not have the EMA shadow's parameters and buffers")
        params = [(mine[name], theirs[name]) for name in mine]
        buffers = [(my_buffers[name], their_buffers[name]) for name in my_buffers]
        return params, buffers

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Average ``model``'s parameters into the shadow after optimizer step ``step``.

        ``shadow.lerp_(theta, 1 - d)`` per parameter with ``d = decay_at(step)``; every
        buffer is copied (module docstring). ``model`` must be the model the shadow was
        copied from (or one with its names), on the shadow's device.

        Raises:
            ValueError: if ``step`` is negative or ``model``'s names differ.
        """
        weight = 1.0 - self.decay_at(step)
        params, buffers = self._pairs(model)
        for shadow, theta in params:
            shadow.lerp_(theta.detach(), weight)
        for shadow, buffer in buffers:
            shadow.copy_(buffer)

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        """Copy every parameter and buffer of ``model`` into the shadow.

        Raises:
            ValueError: if ``model``'s names differ from the shadow's.
        """
        params, buffers = self._pairs(model)
        for shadow, value in [*params, *buffers]:
            shadow.copy_(value)

    def forward(self, *args: object, **kwargs: object) -> object:
        """The shadow model's forward."""
        return self.shadow(*args, **kwargs)
