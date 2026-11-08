"""Tests of ``astro-simclr preflight``, with a mocked ``nvidia-smi`` and mocked ``torch.cuda``.

No test needs a GPU: step 1 runs against fake driver/capability/build combinations (the
GTX 1050 of the plan, a Turing T4, an RTX 5090), and the full preflight runs on the CPU with
synthetic data.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from _training_support import N_PER_SPLIT

from astro_simclr import preflight
from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.preflight import (
    GpuInfo,
    Timed,
    arch_supported,
    check_cuda,
    check_data,
    check_manifest,
    check_resources,
    choose_torch_build,
    estimate,
    query_gpus,
    run_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs" / "reference"
#: torch 2.14 cu126 and default (CUDA 13) architecture lists.
CU126_ARCHS = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
CU13_ARCHS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
GTX_1050 = GpuInfo("NVIDIA GeForce GTX 1050", "582.53", (6, 1))
FULL = "td_ztf_stamp_17_06_20_red-002"


# --- Step 1: choosing and checking the build -----------------------------------------------


@pytest.mark.parametrize(
    ("driver", "capability", "extra", "supported"),
    [
        (582, (6, 1), "cu126", True),  # GTX 1050 (Pascal), new driver: still cu126
        (550, (6, 1), "cu126", True),
        (582, (7, 0), "cu126", True),  # V100
        (582, (7, 5), None, True),  # T4 (Turing), new driver
        (570, (7, 5), "cu126", True),  # T4 on an older driver (Colab/Kaggle)
        (580, (8, 6), None, True),
        (579, (8, 6), "cu126", True),
        (580, (12, 0), None, True),  # RTX 5090 (Blackwell)
        (570, (12, 0), None, False),  # Blackwell needs CUDA 13: update the driver
        (500, (8, 6), "cu126", False),  # too old even for cu126
    ],
)
def test_choose_torch_build(
    driver: int, capability: tuple[int, int], extra: str | None, supported: bool
) -> None:
    build = choose_torch_build(driver, capability)
    assert (build.extra, build.supported) == (extra, supported)
    assert build.command == "uv sync --locked" + (f" --extra {extra}" if extra else "")


@pytest.mark.parametrize(
    ("capability", "archs", "expected"),
    [
        ((6, 1), CU126_ARCHS, True),  # sm_60 code runs on 6.1
        ((6, 1), CU13_ARCHS, False),
        ((8, 9), ["sm_86"], True),
        ((8, 6), ["sm_89"], False),
        ((12, 0), CU126_ARCHS, False),
        ((12, 0), CU13_ARCHS, True),
        ((9, 0), ["sm_90a", "compute_90"], True),
        ((9, 1), ["sm_90a"], False),  # arch-specific code runs only on 9.0
        ((7, 5), ["sm_70"], True),  # same major, lower minor: binary compatible
        ((9, 0), ["compute_90", "sm_x"], False),
    ],
)
def test_arch_supported(capability: tuple[int, int], archs: list[str], expected: bool) -> None:
    assert arch_supported(capability, archs) is expected


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


def test_query_gpus_parses_nvidia_smi() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return _completed("NVIDIA GeForce GTX 1050, 582.53, 6.1\n")

    assert query_gpus(runner) == [GTX_1050]
    assert calls[0][1] == "--query-gpu=name,driver_version,compute_cap"
    assert GTX_1050.driver_major == 582


def test_query_gpus_falls_back_to_the_wsl_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # An ssh shell on WSL2: /usr/lib/wsl/lib is not on PATH (the GTX 1050 machine).
    monkeypatch.setattr(shutil, "which", lambda name: None)
    tried: list[str] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        tried.append(command[0])
        if command[0] == "nvidia-smi":
            raise FileNotFoundError(command[0])
        return _completed("NVIDIA GeForce GTX 1050, 582.53, 6.1\n")

    assert query_gpus(runner) == [GTX_1050]
    assert tried == ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"]


def test_query_gpus_uses_the_binary_found_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/bin/nvidia-smi")
    tried: list[str] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        tried.append(command[0])
        return _completed("Tesla T4, 570.1, 7.5\n")

    assert query_gpus(runner) == [GpuInfo("Tesla T4", "570.1", (7, 5))]
    assert tried == ["/opt/bin/nvidia-smi"]


def test_query_gpus_without_compute_cap_or_driver() -> None:
    answers = iter([_completed("", 2), _completed("Tesla T4, 535.104.05\n")])
    assert query_gpus(lambda _: next(answers)) == [GpuInfo("Tesla T4", "535.104.05", None)]

    def missing(_: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    assert query_gpus(missing) is None
    assert query_gpus(lambda _: _completed("No devices were found\n")) is None
    assert query_gpus(lambda _: _completed("", 9)) is None


def _fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: str | None,
    available: bool = True,
    archs: list[str] | None = None,
    capability: tuple[int, int] = (6, 1),
) -> None:
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: archs or [])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)


def test_gtx_1050_with_cu126_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, cuda="12.6", archs=CU126_ARCHS)
    step = check_cuda([GTX_1050])
    assert step.status == "PASS", step.lines
    assert "needed: uv sync --locked --extra cu126" in "\n".join(step.lines)


def test_gtx_1050_with_the_default_build_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, cuda="13.0", archs=CU13_ARCHS)
    step = check_cuda([GTX_1050])
    assert step.status == "FAIL"
    assert "no kernels for sm_61" in step.lines[-1]
    assert (
        "fix: uv sync --locked --extra cu126, then run every command with UV_NO_SYNC=1"
        in (step.lines[-1])
    )


@pytest.mark.parametrize(
    ("gpu", "cuda", "available", "message"),
    [
        (GTX_1050, None, True, "CPU build; fix: uv sync --locked --extra cu126"),
        (GTX_1050, "12.6", False, "is_available() is False"),
        (GpuInfo("A10", "570.1", (8, 6)), "13.0", True, "CUDA 13.0 needs a driver >= 580"),
        (GpuInfo("RTX 5090", "570.1", (12, 0)), "13.0", True, "cu126 cannot run it"),
        (GpuInfo("old", "470.1", None), "12.6", False, "cannot read"),
    ],
)
def test_check_cuda_failures(
    gpu: GpuInfo, cuda: str | None, available: bool, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_torch(monkeypatch, cuda=cuda, available=available, archs=CU13_ARCHS)
    step = check_cuda([gpu])
    assert step.status == "FAIL"
    assert message in "\n".join(step.lines)


def test_check_cuda_capability_from_torch_and_a_working_non_recommended_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_torch(monkeypatch, cuda="12.6", archs=CU126_ARCHS, capability=(7, 5))
    step = check_cuda([GpuInfo("Tesla T4", "582.10", None)])
    assert step.status == "PASS"
    assert "note: works, but the recommended build is `uv sync --locked`" in step.lines[-1]


def test_no_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, cuda=None, available=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(preflight, "_is_wsl", lambda: False)
    step = check_cuda(None)
    assert step.status == "FAIL"
    assert "looked on PATH and at /usr/lib/wsl/lib/nvidia-smi" in step.lines[0]
    assert "run this project inside WSL2" in step.lines[-1]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(preflight, "_is_wsl", lambda: True)
    assert "install or update the driver on Windows" in check_cuda(None).lines[-1]


def test_no_nvidia_smi_but_cuda_in_torch_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, cuda="12.6", archs=CU126_ARCHS, capability=(6, 1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "GTX 1050")
    monkeypatch.setattr(preflight, "_is_wsl", lambda: True)
    step = check_cuda(None)
    text = "\n".join(step.lines)
    assert step.status == "WARN", text
    assert "the driver version is unknown, so its check is skipped" in text
    assert "add /usr/lib/wsl/lib to PATH" in text
    assert "GPU 0: GTX 1050, driver unknown, compute capability (6, 1)" in text
    assert "needed: uv sync --locked --extra cu126" in text
    # The arch check still runs: the default build cannot serve sm_61.
    _fake_torch(monkeypatch, cuda="13.0", archs=CU13_ARCHS, capability=(6, 1))
    assert check_cuda(None).status == "FAIL"


@pytest.mark.parametrize(
    ("capability", "extra"), [((6, 1), "cu126"), ((8, 6), None), ((12, 0), None)]
)
def test_choose_torch_build_without_a_driver(
    capability: tuple[int, int], extra: str | None
) -> None:
    build = choose_torch_build(None, capability)
    assert (build.extra, build.supported) == (extra, True)


# --- Steps 2-4: resources, data, manifest --------------------------------------------------


@pytest.fixture
def data_copy(synthetic_prepared: dict[str, Path], tmp_path: Path) -> Path:
    """A writable copy of the four prepared synthetic datasets."""
    data = tmp_path / "data"
    for name, path in synthetic_prepared.items():
        shutil.copytree(path, data / name)
    return data


def test_resources(data_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    step = check_resources(data_copy, tmp_path / "out")
    assert step.status == "PASS", step.lines
    (data_copy / "big.pkl").write_bytes(b"0" * 2000)
    monkeypatch.setattr(preflight, "_available_memory", lambda: (1000, "available"))
    step = check_resources(data_copy, tmp_path / "out")
    assert step.status == "FAIL"
    assert "big.pkl" in step.lines[1]
    monkeypatch.setattr(preflight, "_available_memory", lambda: (None, "unknown"))
    assert check_resources(data_copy, tmp_path / "out").status == "WARN"


def test_data_check(data_copy: Path, tmp_path: Path) -> None:
    assert check_data(data_copy).status == "PASS"
    labels = data_copy / FULL / "Test" / "labels.npy"
    labels.write_bytes(labels.read_bytes()[:-1] + b"\x01")
    step = check_data(data_copy)
    assert step.status == "FAIL"
    assert FULL in "\n".join(step.lines)
    assert check_data(tmp_path / "empty").status == "FAIL"


def test_manifest_written_then_verified(data_copy: Path) -> None:
    first = check_manifest(data_copy)
    assert first.status == "PASS"
    assert first.lines[0].startswith("wrote")
    manifest = data_copy / "data-manifest.json"
    assert set(json.loads(manifest.read_text())["datasets"]) == {
        FULL,
        "td_ztf_stamp_17_06_20_sup_1",
        "td_ztf_stamp_17_06_20_sup_10",
        "td_ztf_stamp_simclr_300",
    }
    assert check_manifest(data_copy).status == "PASS"

    recorded = json.loads(manifest.read_text())
    recorded["datasets"][FULL]["source"]["sha256"] = "0" * 64
    recorded["datasets"]["gone"] = {}
    del recorded["datasets"]["td_ztf_stamp_17_06_20_sup_1"]
    manifest.write_text(json.dumps(recorded))
    step = check_manifest(data_copy)
    text = "\n".join(step.lines)
    assert step.status == "FAIL"
    assert f"{FULL}: differs from the manifest" in text
    assert "gone: in the manifest but missing" in text
    assert "WARN: td_ztf_stamp_17_06_20_sup_1 is not in the manifest" in text

    manifest.write_text("{")
    assert check_manifest(data_copy).status == "FAIL"
    manifest.write_text("[]")
    assert check_manifest(data_copy).status == "FAIL"


def test_manifest_migration_to_float32(data_copy: Path, tmp_path: Path) -> None:
    # Task 2.1: re-preparing with --images float32 adds images_f32.npy to every split. The
    # recorded manifest then fails, naming the new files and the command that re-creates it;
    # a uint8-only re-prepare gives the same entry and still matches.
    from astro_simclr.data import prepare
    from astro_simclr.data.prepared import read_meta
    from astro_simclr.data.synthetic import make_synthetic

    assert check_manifest(data_copy).status == "PASS"  # written
    raw = tmp_path / "raw"
    make_synthetic(raw, n_per_split=N_PER_SPLIT, seed=0)  # the seed of synthetic_prepared
    prepare(raw / f"{FULL}.pkl", data_copy, overwrite=True)
    assert check_manifest(data_copy).status == "PASS"

    prepare(raw / f"{FULL}.pkl", data_copy, overwrite=True, images="float32")
    step = check_manifest(data_copy)
    text = "\n".join(step.lines)
    assert step.status == "FAIL"
    assert f"{FULL}: differs from the manifest" in text
    assert "  Train/images_f32.npy: new (prepared with --images float32)" in text
    manifest = data_copy / "data-manifest.json"
    assert f"rm {manifest} && astro-simclr preflight --data-dir {data_copy}" in text
    manifest.unlink()
    assert check_manifest(data_copy).status == "PASS"
    assert check_manifest(data_copy).status == "PASS"

    # The RAM estimate counts float32 as 4 bytes per value.
    meta = read_meta(data_copy / FULL)
    expected = sum(
        int(np.prod(info["shape"])) * np.dtype(info["dtype"]).itemsize
        for split in meta["splits"].values()
        for info in split["files"].values()
    )
    assert preflight._array_bytes(meta) == expected


def test_manifest_differences_are_listed() -> None:
    current = {
        "kind": "classifier",
        "source": {"sha256": "a"},
        "splits": {
            "Train": {"n": 2, "class_counts": [1, 1], "files": {"images.npy": "x", "new.npy": "y"}},
            "Test": {"n": 1, "class_counts": [1], "files": {}},
        },
    }
    recorded = {
        "kind": "pretraining",
        "source": {"sha256": "a"},
        "splits": {
            "Train": {"n": 3, "class_counts": [1, 1], "files": {"images.npy": "z", "old.npy": "w"}},
            "Validation": {},
        },
    }
    assert preflight._manifest_differences(recorded, current) == [
        "kind changed",
        "Test: split added",
        "Train: n changed",
        "Train/images.npy: SHA-256 changed",
        "Train/new.npy: new",
        "Train/old.npy: removed",
        "Validation: split removed",
    ]
    assert preflight._manifest_differences([], current) == ["the recorded entry is not an object"]
    assert preflight._manifest_differences(
        {"kind": "classifier", "source": {"sha256": "a"}}, current
    ) == ["the recorded splits are not an object"]
    assert preflight._manifest_differences(current, current) == ["the entry differs"]


# --- Steps 5-6 and the whole preflight -----------------------------------------------------


def test_estimate() -> None:
    timed = [
        Timed("R1", "full", 45, 220, 45, 10.0, 20),
        Timed("R3-linear-eval", "sup_1", 45, 220, 30, 10.0, 3),
        Timed("R3-fine-tune", "sup_1", 45, 220, 45, 10.0, 20),
        Timed("R4", "missing", 45, 220, 45, 10.0, 20),
    ]
    counts = {
        "full": 45_000,
        "td_ztf_stamp_17_06_20_sup_1": 450,
        "td_ztf_stamp_17_06_20_sup_10": 4_500,
        FULL: 45_000,
    }
    step = estimate(timed, counts.get)
    text = "\n".join(step.lines)
    # R1: 1000 steps/epoch * 220 epochs / 10 steps/s = 22000 s = 6.1 h per seed, 5 seeds.
    assert "R1: <= 30.6 GPU-h full set, 6.1 minimal" in text
    assert "measured at batch 30, not 45" in text
    assert "only 3 timed steps" in text
    assert "R4: no estimate for missing" in text
    assert "R3-fine-tune" not in text
    assert step.status == "WARN"


def test_synthetic_preflight_on_cpu(tmp_path: Path) -> None:
    stream = io.StringIO()
    passed = run_preflight(
        output_dir=tmp_path,
        synthetic=True,
        accelerator="cpu",
        reference_dir=REFERENCE,
        runs=["R3", "R4"],
        overrides=["num_workers=0"],
        n_per_split=20,
        steps=2,
        stream=stream,
    )
    text = stream.getvalue()
    assert passed, text
    for line in (
        "[SKIP] cuda",
        "[PASS] data",
        "[PASS] manifest",
        "[PASS] dry-chain",
        "preflight: PASS",
    ):
        assert line in text
    for name in ("R3-pretrain", "R3-linear-eval", "R3-fine-tune", "R4"):
        assert (tmp_path / "dry-chain" / name / "metrics.json").is_file()
    assert "batch 20 (capped from 550)" in text
    assert "R1" not in text.split("[PASS] dry-chain")[1].split("[")[0]
    assert not (tmp_path / "synthetic" / "raw").exists()


def test_preflight_stops_when_cuda_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No nvidia-smi and no CUDA in torch, whatever the host has: on a GPU machine torch
    # would describe the GPU and step 1 would pass, so the dry chain would run.
    _fake_torch(monkeypatch, cuda=None, available=False)
    stream = io.StringIO()
    assert not run_preflight(
        output_dir=tmp_path,
        synthetic=True,
        reference_dir=REFERENCE,
        gpu_query=lambda: None,
        stream=stream,
    )
    assert "preflight: FAIL (fix the torch build first)" in stream.getvalue()
    assert not (tmp_path / "synthetic").exists()


def test_preflight_on_a_data_dir_without_the_reference_datasets(
    data_copy: Path, tmp_path: Path
) -> None:
    for name in ("td_ztf_stamp_17_06_20_sup_1",):
        shutil.rmtree(data_copy / name)
    stream = io.StringIO()
    passed = run_preflight(
        output_dir=tmp_path / "out",
        data_dir=data_copy,
        accelerator="cpu",
        reference_dir=REFERENCE,
        runs=["R4"],
        stream=stream,
    )
    # R4 was requested explicitly: it must run, so a missing dataset fails.
    assert not passed
    assert "FAIL R4: dataset td_ztf_stamp_17_06_20_sup_1 is not prepared" in stream.getvalue()
    assert "(requested with --runs)" in stream.getvalue()


def test_default_runs_skip_missing_data_with_a_warning(data_copy: Path, tmp_path: Path) -> None:
    for name in ("td_ztf_stamp_simclr_300", FULL):  # R3 pretrain, R1 and R2 cannot run
        shutil.rmtree(data_copy / name)
    stream = io.StringIO()
    passed = run_preflight(
        output_dir=tmp_path / "out",
        data_dir=data_copy,
        accelerator="cpu",
        reference_dir=REFERENCE,
        overrides=["num_workers=0"],
        steps=1,
        stream=stream,
    )
    text = stream.getvalue()
    assert passed, text
    assert "[WARN] dry-chain" in text
    assert "SKIP R1: dataset td_ztf_stamp_17_06_20_red-002 is not prepared" in text
    assert "SKIP R3-linear-eval: no pretrain checkpoint" in text


def test_resources_are_checked_before_the_data_is_read(
    data_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "_available_memory", lambda: (1000, "available"))
    read: list[Path] = []
    monkeypatch.setattr(preflight, "check_data", lambda *args: read.append(args[0]))
    stream = io.StringIO()
    assert not run_preflight(
        output_dir=tmp_path,
        data_dir=data_copy,
        accelerator="cpu",
        reference_dir=REFERENCE,
        stream=stream,
    )
    assert "[FAIL] resources" in stream.getvalue()
    assert read == []  # nothing was hashed
    assert not (data_copy / "data-manifest.json").exists()


def test_preflight_fails_on_bad_data(data_copy: Path, tmp_path: Path) -> None:
    (data_copy / FULL / "meta.json").write_text("{")
    stream = io.StringIO()
    assert not run_preflight(
        output_dir=tmp_path,
        data_dir=data_copy,
        accelerator="cpu",
        reference_dir=REFERENCE,
        stream=stream,
    )
    assert "the dry chain did not run" in stream.getvalue()
    assert not (data_copy / "data-manifest.json").exists()  # never recorded


def test_preflight_argument_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ConfigError, match="exactly one"):
        run_preflight(output_dir=tmp_path, reference_dir=REFERENCE)
    with pytest.raises(ConfigError, match="unknown reference run"):
        run_preflight(output_dir=tmp_path, synthetic=True, runs=["R9"], reference_dir=REFERENCE)
    with pytest.raises(ConfigError, match="no reference configs"):
        run_preflight(output_dir=tmp_path, synthetic=True, reference_dir=tmp_path)
    with pytest.raises(SystemExit):
        main(["preflight"])  # --synthetic or --data-dir is required
    code = main(["preflight", "--synthetic", "--runs", "R9", "--output-dir", str(tmp_path)])
    assert code == 2
    assert "unknown reference run(s) R9" in capsys.readouterr().err


def test_preflight_cli_with_an_unavailable_accelerator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    argv = [
        "preflight",
        "--synthetic",
        "--accelerator",
        "mps",
        "--output-dir",
        str(tmp_path),
        "--reference-dir",
        str(REFERENCE),
    ]
    assert main(argv) == 1
    assert "accelerator 'mps' was requested" in capsys.readouterr().out


def test_wrapper_script_is_thin() -> None:
    script = ROOT / "scripts" / "gpu-preflight.sh"
    assert script.stat().st_mode & 0o111
    assert "exec uv run --no-sync astro-simclr preflight" in script.read_text()
