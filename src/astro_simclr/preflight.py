"""``astro-simclr preflight``: check that a machine can run the reference runs, before it does.

``scripts/gpu-preflight.sh`` is a thin wrapper around this subcommand. The checks run in
order, each prints PASS, FAIL, WARN or SKIP, and the exit code is the result (0: no FAIL):

1. **CUDA** (:func:`check_cuda`): ``nvidia-smi`` gives the driver and the compute
   capability; :func:`choose_torch_build` derives the torch build this machine needs
   (ADR 0001, rule 6) and the installed build is checked against it: CUDA visible to torch,
   ``torch.version.cuda`` supported by the driver, and the GPU's architecture in
   ``torch.cuda.get_arch_list()``. On a mismatch it prints the exact ``uv sync`` command
   that fixes it. Skipped (but the accelerator still checked) with ``--accelerator`` other
   than ``cuda``.
2. **Resources** (:func:`check_resources`): free RAM and disk against the data sizes, from
   ``meta.json`` and file sizes only (nothing is loaded yet).
3. **Data** (:func:`check_data`): ``inspect-data`` on every prepared directory (schema,
   ``meta.json`` and the SHA-256 of every array, memory-mapped).
4. **Manifest** (:func:`check_manifest`): the first run writes ``data-manifest.json`` in
   the data directory; later runs verify the data against it.
5. **Dry chain** (:func:`dry_chain`): the reference configs (``configs/reference/``) in dry
   mode (1 epoch, a few batches): pretrain, linear-eval (on that checkpoint), fine-tune,
   then the supervised runs, on the requested accelerator and at the reference batch sizes
   (so an out-of-memory GPU fails here, not after hours).
6. **Throughput** (:func:`estimate`): steps/s measured during step 5 with the real
   DataLoader (augmentations, ``num_workers``), and the GPU-hour estimate of R1-R4 from the
   Train counts, batch sizes and ``max_epochs`` (an upper bound: EarlyStopping usually stops
   sooner; validation time is not included).

``--synthetic`` first writes the synthetic stand-ins of the four files (``make-synthetic``,
``prepare-data``) under the output directory, so a CUDA machine can be checked without the
real data (level 2 of the plan); its estimate then uses the real Train counts of
``docs/data.md``. ``--data-dir DIR`` checks the prepared real data instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TextIO

import torch
from lightning.pytorch.callbacks import Callback

from astro_simclr.config import ConfigError, load_settings
from astro_simclr.data.prepared import is_prepared, read_meta
from astro_simclr.data.schema import DataError

__all__ = [
    "MANIFEST_NAME",
    "REAL_TRAIN_COUNTS",
    "REFERENCE_RUNS",
    "GpuInfo",
    "StepResult",
    "ThroughputTimer",
    "TorchBuild",
    "arch_supported",
    "build_manifest",
    "check_cuda",
    "check_data",
    "check_manifest",
    "check_resources",
    "choose_torch_build",
    "dry_chain",
    "estimate",
    "prepared_dirs",
    "query_gpus",
    "run_preflight",
]

Status = Literal["PASS", "FAIL", "WARN", "SKIP"]

#: The data manifest, in the data directory.
MANIFEST_NAME: Final = "data-manifest.json"
_MANIFEST_FORMAT: Final = "astro-simclr-data-manifest"
#: Driver needed by the CUDA 13 builds (the default PyPI torch on Linux).
CUDA13_MIN_DRIVER: Final = 580
#: Driver needed by the CUDA 12.x builds (``cu126``), through minor-version compatibility.
CUDA12_MIN_DRIVER: Final = 525
#: Disk the outputs of the reference runs need (checkpoints, logs), with margin.
_DISK_NEEDED: Final = 2 * 1024**3
#: RAM for the process itself (torch, the CUDA context, loaders), on top of the data.
_RAM_HEADROOM: Final = 2 * 1024**3

#: The reference runs as the dry chain runs them: (name, subcommand, config file). The R3
#: entries run in this order (linear-eval and fine-tune load the pretrain checkpoint);
#: ``R3-fine-tune`` checks the fine-tune subcommand and is not a reference run itself.
REFERENCE_RUNS: Final = (
    ("R3-pretrain", "pretrain", "R3-pretrain.yaml"),
    ("R3-linear-eval", "linear-eval", "R3-linear-eval.yaml"),
    ("R3-fine-tune", "fine-tune", "R3-linear-eval.yaml"),
    ("R1", "train-supervised", "R1.yaml"),
    ("R2", "train-supervised", "R2.yaml"),
    ("R4", "train-supervised", "R4.yaml"),
)
#: Seeds of the full set per reference run (the minimal set is 1 seed each).
_FULL_SEEDS: Final = {"R1": 5, "R2": 5, "R3-pretrain": 1, "R3-linear-eval": 1, "R4": 1}
#: R3's linear evaluation runs at these label fractions (one run each).
_R3_FRACTIONS: Final = (
    "td_ztf_stamp_17_06_20_sup_1",
    "td_ztf_stamp_17_06_20_sup_10",
    "td_ztf_stamp_17_06_20",
)
#: Train-split sizes of the real files (``docs/data.md``), for the synthetic estimate. The
#: 100 % entry is ``red`` (the deduplicated original); the pretraining set is approximate.
REAL_TRAIN_COUNTS: Final = {
    "td_ztf_stamp_17_06_20": 50_582,
    "td_ztf_stamp_17_06_20_sup_10": 7_480,
    "td_ztf_stamp_17_06_20_sup_1": 810,
    "td_ztf_stamp_simclr_300": 300_000,
}


@dataclass(slots=True)
class StepResult:
    """One preflight step: its status and the lines it printed."""

    name: str
    status: Status
    lines: list[str] = field(default_factory=list)

    def warn(self, line: str | None = None) -> None:
        """Add ``line`` (if any) and downgrade PASS to WARN (a FAIL stays a FAIL)."""
        if line is not None:
            self.lines.append(line)
        if self.status == "PASS":
            self.status = "WARN"


# --- Step 1: CUDA --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """One GPU as ``nvidia-smi`` reports it. ``capability`` is None if it cannot say."""

    name: str
    driver: str
    capability: tuple[int, int] | None

    @property
    def driver_major(self) -> int:
        """The driver's major version (``582.53`` -> 582)."""
        return int(self.driver.split(".")[0])


Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    # A fixed command line, no shell; nvidia-smi is looked up on PATH on purpose.
    return subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, timeout=60, check=False
    )


def _parse_capability(text: str) -> tuple[int, int] | None:
    major, sep, minor = text.strip().partition(".")
    if not sep or not major.isdigit() or not minor.isdigit():
        return None
    return int(major), int(minor)


def query_gpus(runner: Runner = _run) -> list[GpuInfo] | None:
    """The GPUs ``nvidia-smi`` lists, or None if it is missing or fails.

    Older drivers do not know the ``compute_cap`` field; the query is then repeated without
    it and the capability is left to torch.
    """
    fields = "name,driver_version,compute_cap"
    for query in (fields, "name,driver_version"):
        try:
            done = runner(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
        except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
            return None
        if done.returncode == 0:
            gpus = []
            for line in done.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2 or not parts[1][:1].isdigit():
                    continue
                capability = _parse_capability(parts[2]) if len(parts) > 2 else None
                gpus.append(GpuInfo(parts[0], parts[1], capability))
            return gpus or None
    return None


@dataclass(frozen=True, slots=True)
class TorchBuild:
    """The torch build a GPU needs: ``extra`` None is the default PyPI build (CUDA 13)."""

    extra: str | None
    reason: str
    #: False if no build of this project can run the GPU with this driver.
    supported: bool = True

    @property
    def command(self) -> str:
        """The ``uv sync`` command that installs this build."""
        return "uv sync --locked" + (f" --extra {self.extra}" if self.extra else "")


def choose_torch_build(driver_major: int, capability: tuple[int, int]) -> TorchBuild:
    """The torch build for an NVIDIA driver and a GPU compute capability (ADR 0001, rule 6).

    - compute capability >= 10 (Blackwell, e.g. sm_120): the default build (CUDA 13) with a
      driver >= 580; ``cu126`` has no Blackwell kernels, so an older driver must be updated;
    - compute capability < 7.5 (Maxwell, Pascal, Volta): ``cu126``, because CUDA 13 dropped
      these architectures (whatever the driver);
    - otherwise: the default build with a driver >= 580, ``cu126`` below it;
    - ``cu126`` itself needs a driver >= 525.
    """
    sm = f"sm_{capability[0]}{capability[1]}"
    if capability >= (10, 0):
        if driver_major >= CUDA13_MIN_DRIVER:
            return TorchBuild(None, f"{sm} (Blackwell) needs the CUDA 13 build")
        return TorchBuild(
            None,
            f"{sm} (Blackwell) needs the CUDA 13 build and a driver >= {CUDA13_MIN_DRIVER} "
            f"(this one is {driver_major}); cu126 cannot run it: update the driver",
            supported=False,
        )
    if capability < (7, 5):
        reason = f"{sm} is older than Turing (7.5), which CUDA 13 dropped"
        extra = "cu126"
    elif driver_major >= CUDA13_MIN_DRIVER:
        return TorchBuild(None, f"driver {driver_major} >= {CUDA13_MIN_DRIVER} runs CUDA 13")
    else:
        reason = f"driver {driver_major} < {CUDA13_MIN_DRIVER} cannot run CUDA 13"
        extra = "cu126"
    if driver_major < CUDA12_MIN_DRIVER:
        return TorchBuild(
            extra,
            f"{reason}, and cu126 needs a driver >= {CUDA12_MIN_DRIVER}: update the driver",
            supported=False,
        )
    return TorchBuild(extra, reason)


def arch_supported(capability: tuple[int, int], arch_list: Sequence[str]) -> bool:
    """True if a build with the architectures ``arch_list`` runs a GPU of ``capability``.

    CUDA binaries run on GPUs of the same major version and an equal or higher minor
    (``sm_60`` code runs on a 6.1 GPU), so any ``sm_<major><m>`` with ``m <= minor`` counts.
    """
    major, minor = capability
    for arch in arch_list:
        if not arch.startswith("sm_"):
            continue
        digits = arch[3:].rstrip("abf")
        if (
            len(digits) >= 2
            and digits.isdigit()
            and int(digits[:-1]) == major
            and int(digits[-1]) <= minor
        ):
            return True
    return False


def _installed_build() -> str:
    return f"torch {torch.__version__} (CUDA {torch.version.cuda or 'none: a CPU build'})"


def _fix(build: TorchBuild) -> str:
    return f"fix: {build.command}, then run every command with UV_NO_SYNC=1 (ADR 0001)"


def _build_problem(gpu: GpuInfo, capability: tuple[int, int], build: TorchBuild) -> str | None:
    """Why the installed torch cannot run this GPU, or None if it can."""
    cuda = torch.version.cuda
    if cuda is None:
        return f"the installed torch is a CPU build; {_fix(build)}"
    needed_driver = CUDA13_MIN_DRIVER if int(cuda.split(".")[0]) >= 13 else CUDA12_MIN_DRIVER
    if gpu.driver_major < needed_driver:
        return f"CUDA {cuda} needs a driver >= {needed_driver}; {_fix(build)}"
    if not torch.cuda.is_available():
        return f"torch.cuda.is_available() is False; {_fix(build)}"
    archs = list(torch.cuda.get_arch_list())
    if not arch_supported(capability, archs):
        sm = f"sm_{capability[0]}{capability[1]}"
        return f"this build has no kernels for {sm} (it has {', '.join(archs)}); {_fix(build)}"
    return None


def check_cuda(gpus: list[GpuInfo] | None) -> StepResult:
    """Step 1: the driver, the GPU and the installed torch build agree (see the module doc)."""
    step = StepResult("cuda", "FAIL")
    out = step.lines
    if gpus is None:
        out.append("nvidia-smi not found or failed: no NVIDIA driver is visible")
        if sys.platform == "win32":
            out.append("Windows-native PyPI torch is CPU-only: run this project inside WSL2")
        return step
    gpu = gpus[0]
    capability = gpu.capability
    if capability is None and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
    out.append(f"GPU 0: {gpu.name}, driver {gpu.driver}, compute capability {capability}")
    out.append(f"installed: {_installed_build()}")
    if capability is None:
        out.append("cannot read the compute capability (old nvidia-smi and no CUDA in torch)")
        return step
    build = choose_torch_build(gpu.driver_major, capability)
    out.append(f"needed: {build.command} ({build.reason})")
    if not build.supported:
        return step
    problem = _build_problem(gpu, capability, build)
    if problem is not None:
        out.append(problem)
        return step
    cuda = torch.version.cuda or ""
    if (build.extra == "cu126") != cuda.startswith("12."):
        out.append(f"note: works, but the recommended build is `{build.command}`")
    step.status = "PASS"
    return step


# --- Steps 2-4: resources, data, manifest --------------------------------------------------


def _available_memory() -> tuple[int | None, str]:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024, "available"
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "total"
    except (ValueError, OSError):  # pragma: no cover - platform without sysconf
        return None, "unknown"


def _gib(n: float) -> str:
    return f"{n / 1024**3:.1f} GiB"


def _array_bytes(meta: dict[str, Any]) -> int:
    total = 0
    for split in meta["splits"].values():
        for info in split["files"].values():
            n = 1
            for dim in info["shape"]:
                n *= int(dim)
            total += n * (1 if info["dtype"] == "uint8" else 8)
    return total


def _dataset_bytes(path: Path) -> int:
    # An unreadable meta.json counts as 0 here: step 3 reports it.
    try:
        return _array_bytes(read_meta(path))
    except (DataError, KeyError, TypeError, ValueError):
        return 0


def prepared_dirs(data_dir: Path) -> list[Path]:
    """The prepared layouts directly under ``data_dir``, sorted by name."""
    if not data_dir.is_dir():
        return []
    return sorted(path for path in data_dir.iterdir() if is_prepared(path))


def check_resources(data_dir: Path, output_dir: Path) -> StepResult:
    """Step 2: RAM for the largest prepared dataset plus headroom, and disk for the outputs.

    Pickles found in ``data_dir`` only warn: training never reads them, but ``prepare-data``
    needs about their size in RAM.
    """
    step = StepResult("resources", "PASS")
    ram, kind = _available_memory()
    largest = max((_dataset_bytes(path) for path in prepared_dirs(data_dir)), default=0)
    needed = largest + _RAM_HEADROOM
    if ram is None:
        step.lines.append("RAM: cannot be measured on this platform")
        step.status = "WARN"
    else:
        ok = ram >= needed
        step.lines.append(
            f"RAM: {_gib(ram)} {kind}, needed {_gib(needed)} (largest dataset {_gib(largest)} + "
            f"{_gib(_RAM_HEADROOM)} headroom)"
        )
        if not ok:
            step.status = "FAIL"
        for pickle in sorted(data_dir.glob("*.pkl")):
            if pickle.stat().st_size > ram:
                step.warn(
                    f"WARN: {pickle.name} ({_gib(pickle.stat().st_size)}) does not fit in RAM; "
                    "prepare-data needs about its size"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    step.lines.append(f"disk: {_gib(free)} free under {output_dir}, needed {_gib(_DISK_NEEDED)}")
    if free < _DISK_NEEDED:
        step.status = "FAIL"
    return step


def check_data(data_dir: Path) -> StepResult:
    """Step 3: ``inspect-data`` on every prepared directory of ``data_dir``."""
    from astro_simclr.data.inspection import summarize

    step = StepResult("data", "PASS")
    dirs = prepared_dirs(data_dir)
    if not dirs:
        step.lines.append(f"no prepared dataset in {data_dir} (run prepare-data)")
        step.status = "FAIL"
        return step
    for path in dirs:
        try:
            summary = summarize(path)
        except DataError as err:
            step.lines.append(f"{path.name}: {err}")
            step.status = "FAIL"
            continue
        counts = {split: info["n"] for split, info in summary["splits"].items()}
        step.lines.append(f"{path.name}: {summary['kind']}, {counts}, hashes verified")
    return step


def build_manifest(data_dir: Path) -> dict[str, Any]:
    """The manifest of the prepared datasets in ``data_dir`` (from their ``meta.json``)."""
    datasets: dict[str, Any] = {}
    for path in prepared_dirs(data_dir):
        meta = read_meta(path)
        datasets[path.name] = {
            "kind": meta["kind"],
            "source": meta["source"],
            "splits": {
                split: {
                    "n": info["n"],
                    "class_counts": info["class_counts"],
                    "files": {name: file["sha256"] for name, file in info["files"].items()},
                }
                for split, info in meta["splits"].items()
            },
        }
    return {"format": _MANIFEST_FORMAT, "format_version": 1, "datasets": datasets}


def check_manifest(data_dir: Path) -> StepResult:
    """Step 4: write ``data-manifest.json`` on the first run, verify against it afterwards.

    A dataset of the manifest that is missing or differs fails; a dataset not in the
    manifest only warns (delete the manifest to record the new set).
    """
    step = StepResult("manifest", "PASS")
    path = data_dir / MANIFEST_NAME
    current = build_manifest(data_dir)
    if not path.is_file():
        path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        step.lines.append(f"wrote {path} ({len(current['datasets'])} datasets)")
        return step
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        step.lines.append(f"{path}: invalid JSON ({err})")
        step.status = "FAIL"
        return step
    if not isinstance(recorded, dict) or recorded.get("format") != _MANIFEST_FORMAT:
        step.lines.append(f"{path}: not a data manifest")
        step.status = "FAIL"
        return step
    for name, entry in recorded.get("datasets", {}).items():
        if name not in current["datasets"]:
            step.lines.append(f"{name}: in the manifest but missing from {data_dir}")
            step.status = "FAIL"
        elif current["datasets"][name] != entry:
            step.lines.append(f"{name}: differs from the manifest (changed or re-prepared)")
            step.status = "FAIL"
        else:
            step.lines.append(f"{name}: matches the manifest")
    for name in sorted(set(current["datasets"]) - set(recorded.get("datasets", {}))):
        step.warn(f"WARN: {name} is not in the manifest (delete {path.name} to re-create)")
    return step


# --- Steps 5-6: dry chain and throughput ---------------------------------------------------


def _synchronize(device: torch.device) -> None:
    # Kernels run asynchronously: without a sync the clock would measure queueing, not work.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class ThroughputTimer(Callback):
    """Training steps per second, between consecutive batches of the same epoch.

    The first batch of each epoch is not timed (worker start-up, cuDNN benchmarking); each
    interval includes the DataLoader's wait, so the rate is what a real run gets.
    """

    def __init__(self) -> None:
        self.intervals: list[float] = []
        self._last: float | None = None

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Forget the previous epoch's clock."""
        self._last = None

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Record the time since the previous batch of this epoch."""
        _synchronize(pl_module.device)
        now = time.perf_counter()
        if self._last is not None:
            self.intervals.append(now - self._last)
        self._last = now

    @property
    def steps_per_second(self) -> float | None:
        """The measured rate, or None if no interval was timed."""
        total = sum(self.intervals)
        return len(self.intervals) / total if self.intervals and total > 0 else None


@dataclass(slots=True)
class Timed:
    """What step 6 needs from one dry run."""

    name: str
    dataset: str
    batch_size: int
    max_epochs: int
    measured_batch_size: int
    steps_per_second: float | None
    n_steps: int


def _train_count(data_dir: Path, dataset: str) -> int | None:
    path = data_dir / dataset
    if not is_prepared(path):
        return None
    return int(read_meta(path)["splits"]["Train"]["n"])


def _run_one(
    name: str,
    command: str,
    config: Path,
    overrides: list[str],
    forced: list[str],
    data_dir: Path,
    out: list[str],
) -> Timed | None:
    """One dry run. ``reference`` (config + ``--set``) gives the batch size and epochs the
    estimate assumes; ``forced`` (dry mode, paths, accelerator) applies on top of it."""
    from astro_simclr.experiments import run_command
    from astro_simclr.hparams import COMMAND_DEFAULTS

    defaults = COMMAND_DEFAULTS[command]
    reference = load_settings(config, overrides, defaults=defaults)
    settings = load_settings(config, [*overrides, *forced], defaults=defaults)
    n_train = _train_count(data_dir, settings.dataset)
    if n_train is None:
        out.append(f"SKIP {name}: dataset {settings.dataset} is not prepared in {data_dir}")
        return None
    batch = min(reference.batch_size, n_train)
    settings = settings.replace(batch_size=batch)
    timer = ThroughputTimer()
    started = time.perf_counter()
    run_command(command, settings, callbacks=[timer], progress_bar=False)
    seconds = time.perf_counter() - started
    rate = timer.steps_per_second
    rate_text = "not measured" if rate is None else f"{rate:.2f} steps/s"
    capped = "" if batch == reference.batch_size else f" (capped from {reference.batch_size})"
    out.append(
        f"{name} ({command}, {reference.dataset}, batch {batch}{capped}): ok in {seconds:.1f} s, "
        f"{rate_text} over {len(timer.intervals)} steps"
    )
    return Timed(
        name,
        reference.dataset,
        reference.batch_size,
        reference.max_epochs,
        batch,
        rate,
        len(timer.intervals),
    )


def _dry_overrides(
    command: str, name: str, accelerator: str, data_dir: Path, chain: Path, steps: int
) -> list[str]:
    forced = [
        f"accelerator={accelerator}",
        f"data_dir={data_dir}",
        f"output_dir={chain / name}",
        "max_epochs=1",
        f"limit_train_batches={steps + 1}",
        "limit_val_batches=2",
    ]
    if command in ("linear-eval", "fine-tune"):
        forced.append(f"pretrained={chain / 'R3-pretrain'}")
    return forced


def dry_chain(
    *,
    reference_dir: Path,
    data_dir: Path,
    output_dir: Path,
    accelerator: str,
    runs: Sequence[str],
    overrides: Sequence[str],
    steps: int,
) -> tuple[StepResult, list[Timed]]:
    """Step 5: every selected reference config, in dry mode, on ``accelerator``.

    Each run gets ``max_epochs=1``, ``limit_train_batches=steps + 1`` (one untimed batch),
    ``limit_val_batches=2`` and its own output directory under ``output_dir/dry-chain``;
    ``overrides`` (``--set``) come before these. A batch size larger than the Train split
    is capped to it. A dataset that is not prepared is skipped (WARN); any error fails.

    Raises:
        AcceleratorError: if the accelerator is not available (step 1 should have failed).
    """
    from astro_simclr.experiments import AcceleratorError

    step = StepResult("dry-chain", "PASS")
    timed: list[Timed] = []
    chain = output_dir / "dry-chain"
    pretrained = False
    for name, command, config_name in REFERENCE_RUNS:
        if name.split("-")[0] not in runs:
            continue
        downstream = command in ("linear-eval", "fine-tune")
        if downstream and not pretrained:
            step.warn(f"SKIP {name}: no pretrain checkpoint from R3-pretrain")
            continue
        forced = _dry_overrides(command, name, accelerator, data_dir, chain, steps)
        try:
            result = _run_one(
                name,
                command,
                reference_dir / config_name,
                list(overrides),
                forced,
                data_dir,
                step.lines,
            )
        except AcceleratorError:
            raise
        except (ConfigError, DataError, RuntimeError) as err:
            # RuntimeError covers torch.OutOfMemoryError: the batch does not fit the GPU.
            step.lines.append(f"FAIL {name} ({command}): {type(err).__name__}: {err}")
            step.status = "FAIL"
            continue
        if result is None:
            step.warn()
            continue
        pretrained = pretrained or command == "pretrain"
        timed.append(result)
    if not timed and step.status != "FAIL":
        step.lines.append("no reference run could run (no prepared dataset?)")
        step.status = "FAIL"
    return step, timed


def _hours(n_train: int, timed: Timed) -> float | None:
    if timed.steps_per_second is None:
        return None
    steps = (n_train // timed.batch_size) * timed.max_epochs
    return steps / timed.steps_per_second / 3600


def estimate(timed: Sequence[Timed], train_count: Callable[[str], int | None]) -> StepResult:
    """Step 6: GPU-hours of each reference run, full and minimal set.

    ``train_count(dataset)`` gives the Train size the estimate assumes (the prepared real
    data, or :data:`REAL_TRAIN_COUNTS` for a synthetic preflight).
    """
    step = StepResult("throughput", "PASS")
    total_full = total_minimal = 0.0
    for run in timed:
        if run.name not in _FULL_SEEDS:
            continue
        datasets = _R3_FRACTIONS if run.name == "R3-linear-eval" else (run.dataset,)
        hours = 0.0
        missing = []
        for dataset in datasets:
            n_train = train_count(dataset)
            per_seed = None if n_train is None else _hours(n_train, run)
            if per_seed is None:
                missing.append(dataset)
            else:
                hours += per_seed
        if missing:
            step.warn(f"{run.name}: no estimate for {', '.join(missing)}")
            continue
        full, minimal = hours * _FULL_SEEDS[run.name], hours
        total_full += full
        total_minimal += minimal
        note = ""
        if run.measured_batch_size != run.batch_size:
            note = f" (measured at batch {run.measured_batch_size}, not {run.batch_size})"
        if run.n_steps < 5:
            note += f" (only {run.n_steps} timed steps: rough)"
        step.lines.append(
            f"{run.name}: <= {full:.1f} GPU-h full set, {minimal:.1f} minimal "
            f"({run.max_epochs} epochs max, batch {run.batch_size}){note}"
        )
    step.lines.append(
        f"total: <= {total_full:.1f} GPU-h full set, {total_minimal:.1f} GPU-h minimal set "
        "(upper bound: EarlyStopping usually stops sooner; validation not included)"
    )
    return step


# --- The whole preflight -------------------------------------------------------------------


def _synthetic_data(output_dir: Path, n_per_split: int, seed: int) -> Path:
    from astro_simclr.data.prepared import prepare
    from astro_simclr.data.synthetic import make_synthetic

    raw = output_dir / "synthetic" / "raw"
    data = output_dir / "synthetic" / "data"
    for path in make_synthetic(raw, n_per_split, seed):
        prepare(path, data, overwrite=True)
    shutil.rmtree(raw)  # the prepared copies are all that training reads
    return data


def _print(step: StepResult, stream: TextIO) -> None:
    stream.write(f"[{step.status}] {step.name}\n")
    for line in step.lines:
        stream.write(f"    {line}\n")
    stream.flush()


def _check_arguments(
    synthetic: bool, data_dir: Path | None, runs: Sequence[str], reference_dir: Path
) -> None:
    if synthetic == (data_dir is not None):
        raise ConfigError("preflight needs exactly one of --synthetic and --data-dir DIR")
    unknown = sorted(set(runs) - {"R1", "R2", "R3", "R4"})
    if unknown:
        raise ConfigError(f"unknown reference run(s) {', '.join(unknown)}; valid: R1, R2, R3, R4")
    if not (reference_dir / "R1.yaml").is_file():
        raise ConfigError(f"{reference_dir} has no reference configs (R1.yaml, ...)")


def _accelerator_step(
    accelerator: str, gpu_query: Callable[[], list[GpuInfo] | None]
) -> StepResult:
    from astro_simclr.experiments import check_accelerator

    if accelerator == "cuda":
        return check_cuda(gpu_query())
    step = StepResult("cuda", "SKIP", [f"accelerator {accelerator}: CUDA not checked"])
    try:
        check_accelerator(accelerator)
    except ConfigError as err:
        step.status, step.lines = "FAIL", [str(err)]
    return step


def run_preflight(
    *,
    output_dir: Path,
    data_dir: Path | None = None,
    synthetic: bool = False,
    accelerator: str = "cuda",
    reference_dir: Path = Path("configs/reference"),
    runs: Sequence[str] = ("R1", "R2", "R3", "R4"),
    overrides: Sequence[str] = (),
    n_per_split: int = 2400,
    seed: int = 0,
    steps: int = 20,
    gpu_query: Callable[[], list[GpuInfo] | None] = query_gpus,
    stream: TextIO | None = None,
) -> bool:
    """Run the six steps (module docstring) and print them; True if none failed.

    Exactly one of ``synthetic`` and ``data_dir`` is required. ``gpu_query`` is injectable
    for tests (a mocked ``nvidia-smi``); ``stream`` defaults to the current ``sys.stdout``.
    A failed step 1 stops the preflight, and so does a
    failed step 2-4 (the dry chain would only fail again).

    Raises:
        ConfigError: if both or neither data source is given, ``reference_dir`` has no
            reference configs, or a run name is unknown.
    """
    _check_arguments(synthetic, data_dir, runs, reference_dir)
    stream = sys.stdout if stream is None else stream
    done: list[StepResult] = []

    def record(step: StepResult) -> bool:
        done.append(step)
        _print(step, stream)
        return step.status != "FAIL"

    if not record(_accelerator_step(accelerator, gpu_query)):
        stream.write("preflight: FAIL (fix the torch build first)\n")
        return False
    if data_dir is None:
        data_dir = _synthetic_data(output_dir, n_per_split, seed)
        stream.write(f"synthetic data ({n_per_split} stamps per split) in {data_dir}\n")
    checks = [check_resources(data_dir, output_dir), check_data(data_dir)]
    if checks[-1].status != "FAIL":  # never record a manifest of data that failed its check
        checks.append(check_manifest(data_dir))
    if not all([record(step) for step in checks]) or len(checks) < 3:
        stream.write("preflight: FAIL (the data is not usable; the dry chain did not run)\n")
        return False
    chain, timed = dry_chain(
        reference_dir=reference_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        accelerator=accelerator,
        runs=runs,
        overrides=list(overrides),
        steps=steps,
    )
    record(chain)
    real_dir = data_dir

    def train_count(dataset: str) -> int | None:
        if synthetic:
            return REAL_TRAIN_COUNTS.get(dataset)
        return _train_count(real_dir, dataset)

    record(estimate(timed, train_count))
    failed = any(step.status == "FAIL" for step in done)
    stream.write(f"preflight: {'FAIL' if failed else 'PASS'}\n")
    return not failed
