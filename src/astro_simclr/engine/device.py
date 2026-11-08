"""The device policy of the plain engine: one device, determinism flags, float32 precision.

What Lightning's ``Trainer(accelerator, devices, deterministic, benchmark)`` decided
implicitly is one frozen record here (ADR 0014, "Determinism"):

- **one device**: ``world_size > 1`` is refused (multi-GPU is out of scope);
- ``cudnn.benchmark = not deterministic`` (``training.runs.build_trainer``);
- ``torch.use_deterministic_algorithms(deterministic)``, always called (False included);
- ``CUBLAS_WORKSPACE_CONFIG=":4096:8"`` set only when deterministic, never unset, as
  Lightning's ``_set_torch_flags`` does;
- the float32 precision of ADR 0010 (``ieee`` | ``tf32``), applied by
  ``training.runs.apply_fp32_precision`` before any module is built and recorded.

:meth:`DevicePolicy.torch_flags` states what the loop applies, so a test can compare it with
Lightning's; :meth:`DevicePolicy.apply` applies it (task 4.1), with the precision, before any
module is built and before any CUDA work (so the cuBLAS setting takes effect).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal

import torch

from astro_simclr.config import ConfigError
from astro_simclr.engine.precision import apply_fp32_precision

__all__ = ["CUBLAS_WORKSPACE_CONFIG", "DevicePolicy", "Fp32Precision"]

#: The cuBLAS workspace setting that makes CUDA matmuls deterministic (Lightning's value).
CUBLAS_WORKSPACE_CONFIG: Final = ":4096:8"
#: ADR 0010's float32 precision of convolutions and matmuls.
Fp32Precision = Literal["ieee", "tf32"]


@dataclass(frozen=True, slots=True)
class DevicePolicy:
    """The device a run trains on and the torch flags it sets.

    Attributes:
        accelerator: ``cpu``, ``cuda`` or ``mps`` (``Settings.accelerator``; ``auto`` is
            resolved before this record is built).
        deterministic: ``Settings.deterministic``.
        fp32_precision: ``Settings.fp32_precision`` (ADR 0010).
        world_size: processes in the run; only 1 is supported.

    Raises:
        ConfigError: for ``world_size != 1`` (multi-GPU is out of scope, ADR 0014), an
            unknown accelerator or an unknown precision.
    """

    accelerator: Literal["cpu", "cuda", "mps"]
    deterministic: bool = False
    fp32_precision: Fp32Precision = "ieee"
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size != 1:
            msg = (
                f"world_size {self.world_size}: the plain engine trains on one device only "
                "(multi-GPU is out of scope, ADR 0014)"
            )
            raise ConfigError(msg)
        if self.accelerator not in ("cpu", "cuda", "mps"):
            raise ConfigError(f"unknown accelerator {self.accelerator!r}")
        if self.fp32_precision not in ("ieee", "tf32"):
            raise ConfigError(
                f"fp32_precision must be 'ieee' or 'tf32', got {self.fp32_precision!r}"
            )

    @property
    def cudnn_benchmark(self) -> bool:
        """``torch.backends.cudnn.benchmark``: on unless the run is deterministic."""
        return not self.deterministic

    def torch_flags(self) -> dict[str, object]:
        """The process-global settings the loop applies, by name.

        ``CUBLAS_WORKSPACE_CONFIG`` is ``None`` when the environment is left as it is.
        """
        return {
            "cudnn.benchmark": self.cudnn_benchmark,
            "use_deterministic_algorithms": self.deterministic,
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG if self.deterministic else None,
        }

    @property
    def device(self) -> torch.device:
        """The training device (the current CUDA device for ``cuda``)."""
        return torch.device(self.accelerator)

    def apply(self) -> None:
        """Set the precision (ADR 0010) and the determinism flags of :meth:`torch_flags`.

        Process-global, as Lightning's ``Trainer`` sets them: ``cudnn.benchmark``,
        ``torch.use_deterministic_algorithms`` (``False`` too) and, when deterministic,
        ``CUBLAS_WORKSPACE_CONFIG``, never unset. Nothing here touches the device: an
        unavailable accelerator is the caller's to refuse (``experiments.check_accelerator``).
        """
        apply_fp32_precision(self.fp32_precision)
        flags = self.torch_flags()
        torch.backends.cudnn.benchmark = bool(flags["cudnn.benchmark"])
        torch.use_deterministic_algorithms(self.deterministic)
        workspace = flags["CUBLAS_WORKSPACE_CONFIG"]
        if isinstance(workspace, str):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
