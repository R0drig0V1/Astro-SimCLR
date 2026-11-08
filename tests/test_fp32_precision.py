"""The float32 precision policy of a run (ADR 0008): set, recorded and never mixed.

torch 2.14 runs cuDNN convolutions in TF32 by default, so ``Settings.fp32_precision``
(default ``ieee``) is applied at the start of every run, recorded with its effect in
``metrics.json["machine"]`` and is part of the machine signature that a comparison must not
mix. Everything here runs on the CPU: a CUDA device is faked where its compute capability
matters.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from _protocol_support import MACHINE, fake_run, predictions

from astro_simclr import experiments
from astro_simclr.config import FP32_PRECISIONS, ConfigError, Settings, settings_from_mapping
from astro_simclr.presets import CARRASCO2021_PRESET, PRESETS, preset_report
from astro_simclr.protocol.compare import compare_variants
from astro_simclr.reference import machine_signature
from astro_simclr.training import apply_fp32_precision
from astro_simclr.training.runs import fp32_precision_record

LABELS = [i // 20 for i in range(100)]


def _conv_precision() -> str:
    return str(cast(Any, torch.backends.cudnn).conv.fp32_precision)


@pytest.fixture(autouse=True)
def _restore_precision() -> Iterator[None]:
    """The policy is process-global: put back what the test found (torch's own defaults)."""
    conv, matmul = _conv_precision(), torch.get_float32_matmul_precision()
    yield
    cast(Any, torch.backends.cudnn).conv.fp32_precision = conv
    torch.set_float32_matmul_precision(matmul)


def test_settings_default_to_ieee_and_validate() -> None:
    assert Settings().fp32_precision == "ieee"
    assert FP32_PRECISIONS == ("ieee", "tf32")
    with pytest.raises(ConfigError, match="fp32_precision 'bf16' is not one of: ieee, tf32"):
        Settings(fp32_precision="bf16")


@pytest.mark.parametrize(("policy", "matmul"), [("ieee", "highest"), ("tf32", "high")])
def test_apply_sets_convolutions_and_matmuls(policy: str, matmul: str) -> None:
    # Start from the opposite policy, so that the call is what changes it.
    cast(Any, torch.backends.cudnn).conv.fp32_precision = "tf32" if policy == "ieee" else "ieee"
    apply_fp32_precision(Settings(fp32_precision=policy))
    assert _conv_precision() == policy
    assert torch.backends.cuda.matmul.fp32_precision == policy
    # Lightning's CUDA accelerator reads this on Ampere GPUs: it must not raise.
    assert torch.get_float32_matmul_precision() == matmul


def test_record_reads_back_the_backends_on_the_cpu() -> None:
    apply_fp32_precision(Settings(fp32_precision="tf32"))
    record = fp32_precision_record(torch.device("cpu"))
    # No TF32 on the CPU (for convolutions and CUDA matmuls): the effective policy is ieee.
    assert record["fp32_precision"] == "ieee"
    assert record["compute_capability"] is None
    assert record["fp32_backends"]["cudnn_conv"] == "tf32"
    assert record["fp32_backends"]["cuda_matmul"] == "tf32"
    assert set(record["fp32_backends"]) == {"cudnn_conv", "cuda_matmul", "mkldnn_matmul"}


@pytest.mark.parametrize(
    ("capability", "policy", "effective"),
    [
        ((6, 1), "tf32", "ieee"),  # the GTX 1050 (Pascal): no TF32 whatever the setting
        ((8, 6), "ieee", "ieee"),  # Ampere with the faithful policy
        ((8, 6), "tf32", "tf32"),
        ((12, 0), "tf32", "tf32"),  # the RTX 5090 (Blackwell)
    ],
)
def test_record_on_a_cuda_device(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], policy: str, effective: str
) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    apply_fp32_precision(Settings(fp32_precision=policy))
    record = fp32_precision_record(torch.device("cuda", 0))
    assert record["fp32_precision"] == effective
    assert record["compute_capability"] == f"{capability[0]}.{capability[1]}"


def test_machine_records_the_policy() -> None:
    apply_fp32_precision(Settings())
    trainer = SimpleNamespace(strategy=SimpleNamespace(root_device=torch.device("cpu")))
    machine = experiments._machine(cast(Any, trainer))
    assert machine["fp32_precision"] == "ieee"
    assert machine["fp32_backends"]["cudnn_conv"] == "ieee"
    assert machine["compute_capability"] is None


class _Stop(Exception):
    pass


def test_run_command_applies_the_policy_before_the_data_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The policy is set before the dataset is opened, so before any module is built."""
    seen: list[str] = []

    def dataset_path(settings: Settings) -> Path:
        seen.append(_conv_precision())
        raise _Stop

    monkeypatch.setattr(experiments, "dataset_path", dataset_path)
    cast(Any, torch.backends.cudnn).conv.fp32_precision = "tf32"  # torch's default
    with pytest.raises(_Stop):
        experiments.run_command(
            "train-supervised", Settings(accelerator="cpu", output_dir=tmp_path)
        )
    assert seen == ["ieee"]


def test_tf32_is_a_preset_deviation() -> None:
    faithful = settings_from_mapping(PRESETS[CARRASCO2021_PRESET])
    assert faithful.fp32_precision == "ieee"
    report = preset_report(faithful.replace(fp32_precision="tf32"))
    assert report is not None and report["faithful"] is False
    assert report["deviations"] == {"fp32_precision": {"value": "tf32", "preset": "ieee"}}


def test_machine_signature_holds_the_effective_policy() -> None:
    assert machine_signature({"machine": {**MACHINE, "fp32_precision": "tf32"}}) == (
        "cpu",
        None,
        "2.14.0",
        None,
        "tf32",
    )
    # A run written before ADR 0008 ran without TF32: it reads as ieee.
    old = machine_signature({"machine": MACHINE})
    assert old == machine_signature({"machine": {**MACHINE, "fp32_precision": "ieee"}})


def _run(root: Path, name: str, seed: int, machine: dict[str, Any]) -> Path:
    preds = {"test": predictions(LABELS, LABELS), "val": predictions(LABELS, LABELS)}
    return fake_run(root, name, preds, seed=seed, machine=machine)


def test_compare_variants_refuses_mixed_policies(tmp_path: Path) -> None:
    gpu = {**MACHINE, "device": "cuda", "gpu": "NVIDIA GeForce RTX 5090", "cuda": "12.8"}
    base = [_run(tmp_path, "base", 0, {**gpu, "fp32_precision": "ieee"})]
    tf32 = [_run(tmp_path, "tf32", 0, {**gpu, "fp32_precision": "tf32"})]
    with pytest.raises(ConfigError, match="float32 precisions"):
        compare_variants(base, tf32, n_boot=10)
    allowed = compare_variants(base, tf32, n_boot=10, allow_mixed_machines=True)
    assert any("WARN: mixed machines" in line for line in allowed.lines)
    # A pre-ADR run (no field) and an ieee run on the same machine are one policy.
    old = [_run(tmp_path, "old", 0, gpu)]
    lines = compare_variants(base, old, n_boot=10).lines
    assert not any("mixed" in line for line in lines)
