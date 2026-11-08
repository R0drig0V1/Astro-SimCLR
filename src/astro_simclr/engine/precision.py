"""ADR 0010's float32 precision policy, engine-free: set it, and read back what a run used.

The policy (``ieee`` | ``tf32``) covers cuDNN convolutions and matmuls together. It is
process-global, so every run sets it before any module is built, also when an earlier run
in the same process set another one (the preflight, the tests, the 4.7(a) differential
test that runs both engines in one process).

These functions lived in ``training.runs`` (which imports Lightning, so the plain engine
could not use them); ``training.runs`` now delegates here, with the same behaviour, so the
two engines set and record the policy through one implementation.
"""

from __future__ import annotations

from typing import Any, Final, cast

import torch

from astro_simclr.config import ConfigError

__all__ = ["FP32_MATMUL_PRECISION", "apply_fp32_precision", "fp32_precision_record"]

#: ``torch.set_float32_matmul_precision`` value of each ``fp32_precision`` (ADR 0010, rule 3).
FP32_MATMUL_PRECISION: Final[dict[str, str]] = {"ieee": "highest", "tf32": "high"}
#: The first CUDA compute capability (major) with TF32 tensor cores: Ampere.
_TF32_MIN_MAJOR: Final = 8


def apply_fp32_precision(precision: str) -> None:
    """Set the float32 precision of cuDNN convolutions and of matmuls to ``precision``.

    torch 2.14 runs cuDNN convolutions in TF32 by default
    (``torch.backends.cudnn.conv.fp32_precision == "tf32"``), so without this a run on an
    Ampere or newer GPU would silently use 10-bit-mantissa convolutions while the GTX 1050
    (Pascal, no TF32) uses float32 (ADR 0010). Call it before any module is built.

    Convolutions use the per-operator API of the 2.14 line. Matmuls go through
    ``torch.set_float32_matmul_precision`` instead of
    ``torch.backends.cuda.matmul.fp32_precision``: after the latter is set to ``tf32``,
    ``torch.get_float32_matmul_precision()`` raises (torch reports mixed legacy and new
    APIs, measured on 2.14.1), and Lightning's CUDA accelerator calls it on Ampere GPUs.
    ``high`` also lets oneDNN use TF32 for CPU matmuls on CPUs that support it.

    Raises:
        ConfigError: for a precision other than ``ieee`` / ``tf32``.
    """
    if precision not in FP32_MATMUL_PRECISION:
        raise ConfigError(f"fp32_precision must be 'ieee' or 'tf32', got {precision!r}")
    # torch.backends.cudnn is untyped; the cast states the documented attribute.
    conv = cast(Any, torch.backends.cudnn).conv
    conv.fp32_precision = precision
    torch.set_float32_matmul_precision(FP32_MATMUL_PRECISION[precision])


def fp32_precision_record(device: torch.device) -> dict[str, Any]:
    """What float32 precision a run on ``device`` used, for the run's machine record.

    ``fp32_precision`` is the effective policy, ``tf32`` only when ``device`` is a CUDA GPU
    with TF32 (compute capability 8.0 or newer) and the convolutions or the matmuls allow
    it, else ``ieee``: this is what a comparison must not mix (``reference.machine_signature``).
    ``fp32_backends`` holds the torch settings read back (cuDNN convolutions, CUDA and oneDNN
    matmuls) and ``compute_capability`` the CUDA device's ``"major.minor"`` (``None`` on
    other devices).
    """
    backends = cast(Any, torch.backends)
    settings = {
        "cudnn_conv": backends.cudnn.conv.fp32_precision,
        "cuda_matmul": backends.cuda.matmul.fp32_precision,
        "mkldnn_matmul": backends.mkldnn.matmul.fp32_precision,
    }
    capability = None
    tf32 = False
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        capability = f"{major}.{minor}"
        uses_tf32 = "tf32" in (settings["cudnn_conv"], settings["cuda_matmul"])
        tf32 = major >= _TF32_MIN_MAJOR and uses_tf32
    return {
        "fp32_precision": "tf32" if tf32 else "ieee",
        "fp32_backends": settings,
        "compute_capability": capability,
    }
