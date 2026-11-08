"""The SimCLR objective of phase 5: NT-Xent in the paper's form, always in float32.

This reuses the package loss :class:`astro_simclr.losses.NT_Xent` with
``legacy_denominator=False`` (ADR 0002: the paper's denominator, ``exp(s_pos) +
sum_neg exp(s_neg)``; the legacy default is close to DCL and is not SimCLR). Phase 5
reports SimCLR, so the paper form is the only one here; the legacy/DCL form is an
ablation of a later phase.

The projections are cast to float32 and autocast is turned off around the loss, so the
similarities, the ``logsumexp`` and the temperature division are float32 even when the
network runs under bf16/fp16 autocast (plan 5.3 decides the pretraining precision; the
loss stays float32 whatever it decides).
"""

from __future__ import annotations

import torch

from astro_simclr.losses import NT_Xent

__all__ = ["SimCLRLoss"]


class SimCLRLoss:
    """NT-Xent (paper mode) of a batch of positive pairs, in float32."""

    def __init__(self, temperature: float) -> None:
        """Create the loss at ``temperature`` (> 0).

        Raises:
            ValueError: if ``temperature`` is not positive.
        """
        self.loss = NT_Xent(temperature, legacy_denominator=False)

    @property
    def temperature(self) -> float:
        """The softmax temperature."""
        return self.loss.temperature

    def __call__(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(loss, mean positive cosine, mean negative cosine)`` of (B, D) projections.

        ``z_j[k]`` is the positive of ``z_i[k]``. See :meth:`NT_Xent.forward`.

        Raises:
            ValueError: if the inputs are not two (B, D) tensors of the same shape.
        """
        with torch.autocast(device_type=z_i.device.type, enabled=False):
            result: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.loss(
                z_i.float(), z_j.float()
            )
        return result
