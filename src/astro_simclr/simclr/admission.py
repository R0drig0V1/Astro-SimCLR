"""``simclr-view-admission``: plan 5.2's admission test of a SimCLR view (task 5.2; ADR 0025).

Plan 5.2: "a view whose single-view accuracy drop on Validation (replication checkpoint)
exceeds 1 point is not admitted". The test asks whether a view destroys what a supervised
classifier of the same stamps needs: the replication checkpoint (task 2.4's
``carrasco2021`` paper classifier, a ``train-supervised`` run of the plain engine or a
stored Lightning run) is evaluated twice on the **screen's selection half** of
Validation, on the stamps as stored and on the stamps under the view, and the drop of the
accuracy decides.

**Selection only** (plan 5.0c). Admission decides which views, and so which SSL arms,
exist: it is selection, and the arms are later scored on the screen's decision half. So it
reads only the selection half of a stored splits file (``--splits``, required, verified
against the layout as the pilot does: ``protocol.splits.verify_splits``), never the
decision half, the CV parts or Test. In CD21 Validation is exactly the screen (250 + 250
stamps), so the plan's "on Validation" (written before the screen) means its selection
half: 250 stamps, one stamp 0.4 points of accuracy, so ``--repeats`` averages several
draws of the view. The report records the splits file's SHA-256 and the rows used.

``astro-simclr simclr-view-admission RUN_DIR... --splits FILE --out FILE.json [--data DIR]
[--seed N] [--repeats R] [--device D] [view flags]`` (:func:`view_admission`). Several ``RUN_DIR``
(seeds of the replication) are each evaluated on the same viewed stamps; the verdict reads
the mean of their drops.

**The view** (:class:`AdmissionView`, :func:`view_stamps`) is the pilot's own arithmetic
(:mod:`astro_simclr.simclr.views`), applied **once per selection stamp** with the
parameters of ``(seed, r)`` for repeat ``r`` (the pilot's draw functions and streams at
"step" ``r``, the first of their two views): the noise (C5, with the fits of the
selection stamps), the blur, the gamma, a dihedral element (``--dihedral``), and the
window's shift (``--shift``), zoom and rotation. Every option is off unless given, so the
test is of exactly the views asked for (one for a single-view test). The view arithmetic
runs on the CPU, so the viewed stamps do not depend on the evaluation device.

**How a 57-pixel training view meets a classifier of 63-pixel stamps.** The paper model
reads the centre 21 x 21 pixels of the 63-pixel stamp, re-normalised (``data.float_input``),
not a 57-pixel window. So the view is applied **at 63 pixels, keeping the size**, and the
run's own input pipeline follows (``crop_split`` at ``paper_input_side``, with the run's
``nan_approximation``): the photometric views act on the whole stamp as in training; the
geometric ones are the training window's map (:func:`~.views.window_transform`, the same
zoom, angle and shift, the shift giving way to the rotation) resampled over a 63-pixel
output centred where the training window's centre would be. The classifier's 21-pixel crop
therefore sees what the centre of the training view shows. The outer ring beyond the
57-pixel window, which the crop never reads, is clamped to the stamp's edge
(:func:`~.views.resample`); its only reader is the NaN approximation's count of exact
zeros over the stamp. Without any view the input is the run's own on those rows, bit for
bit (a test).

**What it measures.** The 21-pixel crop sees the view's effect at the centre, where the
source is; a with-features checkpoint also reads the alert features, unchanged by any
view, so its drop understates the image's sensitivity: run the without-features
replication as well. The run selected its weights on Validation (the paper's protocol), so
the clean accuracy is optimistic; the drop is a difference on the same stamps. A rotation
or zoom is a bilinear resampling, which also smooths the stamp (:mod:`.views`, "The
interpolation confound"): its drop measures both.

**Verdict:** ``admitted`` if the mean drop of the accuracy (clean minus viewed, over the
runs and the repeats) is at most :data:`MAX_DROP` (1 point), ``not admitted`` otherwise.
The balanced accuracy is reported too (the selection half is balanced, so they agree
there). The report (``--out``, JSON, written atomically, never over a file without
``--overwrite``) holds every run's clean and viewed scores, the view and its draws' seed.
Exit 0 whatever the verdict.

**Refusals** (``ConfigError``): no view asked for, ``repeats < 1``, a run that is not a
finished ``train-supervised`` run of the paper classifier, an input the test cannot
reproduce (``paper_input`` other than ``float32``, or the context channels), a layout
whose data hash is not the run's, an existing ``--out`` without ``--overwrite``; a splits
file that is unreadable or made for other data is a ``DataError`` (``SplitsError``).
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.config import ConfigError, Settings
from astro_simclr.data.files import atomic_write_bytes, sha256_file
from astro_simclr.data.float_input import crop_split
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.engine import Dataset, DevicePolicy, TierChoice, open_eval
from astro_simclr.engine.record import machine_record
from astro_simclr.ensemble_eval import Member, member_net, member_settings, read_member
from astro_simclr.paper_variants import paper_input_side
from astro_simclr.protocol.splits import read_splits, verify_splits
from astro_simclr.runs.identity import data_sha256
from astro_simclr.runs.manifest import is_finished
from astro_simclr.simclr.views import (
    CENTRE_OFFSET,
    STAMP_SIDE,
    ViewConfig,
    background_noise,
    dihedral,
    draw_extra_params,
    draw_view_params,
    gamma_remap,
    gaussian_blur,
    noise_generator,
    resample,
    window_transform,
)
from astro_simclr.stamp_relation import stamp_fits
from astro_simclr.tasks.common import PAPER_STAMP_CLASSIFIER, check_accelerator, dataset_path
from astro_simclr.tasks.paper import PaperNet, paper_crops
from astro_simclr.tasks.plain import resolve_accelerator
from astro_simclr.tasks.results import StageOutput, evaluation_entry, finite
from astro_simclr.training.tensor_splits import TensorSplit

__all__ = [
    "FORMAT",
    "MAX_DROP",
    "AdmissionReport",
    "AdmissionView",
    "model_input",
    "view_admission",
    "view_stamps",
]

#: ``--out``'s ``format``; its version is :data:`FORMAT_VERSION`.
FORMAT: Final = "astro-simclr-view-admission"
FORMAT_VERSION: Final = 1
#: Plan 5.2's threshold: a mean accuracy drop above 1 point is not admitted.
MAX_DROP: Final = 0.01
_COMMAND: Final = "train-supervised"
_SPLIT: Final = "Validation"
#: What is evaluated: the selection half of the stored splits file's screen.
_PART: Final = "screen selection half"
_CPU: Final = torch.device("cpu")


@dataclass(frozen=True, slots=True)
class AdmissionView:
    """The view under test (module docstring, "The view"); everything is off by default.

    Attributes:
        views: the noise and the optional views, with their strengths.
        dihedral: apply one of the 8 dihedral elements.
        shift: shift the window by its integer offset (at most 3 pixels).
    """

    views: ViewConfig = field(default_factory=ViewConfig)
    dihedral: bool = False
    shift: bool = False

    @property
    def any(self) -> bool:
        """Whether anything is applied."""
        return self.views.noise or self.views.extra or self.dihedral or self.shift

    def describe(self) -> dict[str, Any]:
        """The view as the report records it (only what is on, with its strength)."""
        config = self.views
        strengths = {
            "noise_scale": config.noise_scale,
            "scale_jitter": config.scale_jitter,
            "blur_sigma": config.blur_sigma,
            "intensity_jitter": config.intensity_jitter,
            "rotation_degrees": config.rotation_degrees,
        }
        out: dict[str, Any] = {k: v for k, v in strengths.items() if v > 0}
        if config.noise:
            out["noise_min_r2"] = config.noise_min_r2
        if self.dihedral:
            out["dihedral"] = True
        if self.shift:
            out["shift"] = True
        return out


def view_stamps(
    stamps: torch.Tensor,
    view: AdmissionView,
    *,
    seed: int,
    repeat: int,
    fits: torch.Tensor | None = None,
) -> torch.Tensor:
    """The (N, C, 63, 63) stamps under ``view``, once each, with the draws of ``(seed, repeat)``.

    ``stamps`` are float32 (N, C >= 3, 63, 63) on the CPU, as stored. The order and the
    arithmetic are the pilot's (:func:`~.views.make_views`), each parameter the first view's
    of :func:`~.views.draw_view_params` and :func:`~.views.draw_extra_params` at step
    ``repeat``; the geometric part is resampled to 63 pixels (module docstring). ``fits``
    (N, 6) are needed by the noise. Returns a new tensor.

    Raises:
        ValueError: if the noise is on without ``fits``, or as the view functions.
    """
    n = stamps.shape[0]
    config = view.views
    params = draw_view_params(seed, repeat, n)
    extra = draw_extra_params(seed, repeat, n, config)
    x = stamps
    if config.noise:
        if fits is None:
            raise ValueError("the noise view needs the stamp fits")
        x = background_noise(
            x, fits, params.noise_u[0], noise_generator(seed, repeat, _CPU), config
        )
    if extra.blur_sigma is not None:
        x = gaussian_blur(x, extra.blur_sigma[0], radius=config.blur_radius)
    if extra.gamma is not None:
        x = gamma_remap(x, extra.gamma[0])
    if view.dihedral:
        x = dihedral(x, params.dihedral[0])
    if view.shift or config.geometric:
        offsets = params.offsets[0] if view.shift else np.full((n, 2), CENTRE_OFFSET)
        matrix, shift = window_transform(
            offsets,
            None if extra.scale is None else extra.scale[0],
            None if extra.angle is None else extra.angle[0],
        )
        x = resample(x, matrix, shift, side=STAMP_SIDE)
    return x


def model_input(stamps: torch.Tensor, settings: Settings) -> npt.NDArray[np.float32]:
    """The run's model input of (N, 3, 63, 63) stamps: its crop, re-normalised, channels first.

    ``crop_split`` at the model's ``paper_input_side`` with the run's ``nan_approximation``:
    ``training.tensor_splits.float_crops`` on stamps given in memory.
    """
    nhwc = np.ascontiguousarray(stamps.permute(0, 2, 3, 1).numpy(), dtype=np.float32)
    side = paper_input_side(settings.paper_arch)
    crops = crop_split(nhwc, side, nan_approximation=settings.nan_approximation)
    return np.ascontiguousarray(crops.transpose(0, 3, 1, 2))


def _check_run(member: Member, settings: Settings) -> None:
    """A finished paper-classifier run whose input the test reproduces (module docstring)."""
    manifest, where = member.manifest, member.run_dir
    if not is_finished(manifest.status):
        raise ConfigError(f"{where}: the run is {manifest.status}, not finished")
    model = manifest.config.get("supervised_model")
    if manifest.command != _COMMAND or model != PAPER_STAMP_CLASSIFIER:
        raise ConfigError(
            f"{where}: the admission test reads {_COMMAND} runs of {PAPER_STAMP_CLASSIFIER}, "
            f"not {manifest.command} of {model}"
        )
    if settings.paper_input != "float32" or settings.context != "none":
        raise ConfigError(
            f"{where}: the admission test views the float32 stamps without the context "
            f"channels (paper_input={settings.paper_input}, context={settings.context})"
        )


def _probabilities(
    net: PaperNet,
    images: npt.NDArray[np.float32],
    split: TensorSplit,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """The softmax (N, C) of ``net`` on ``images`` with the split's features, on the CPU.

    The engine's evaluation batches in order, ``eval()`` and ``inference_mode``: the
    computation of the run's own evaluation (``PaperTask.eval_step``).
    """
    dataset = Dataset(
        images=images, features=split.features, labels=split.labels, n_classes=split.n_classes
    )
    out = []
    with torch.inference_mode():
        for x_img, x_feat, _ in open_eval(dataset, TierChoice(), device, batch_size=batch_size):
            out.append(net(x_img, x_feat).softmax(dim=1).cpu())
    return torch.cat(out)


def _scores(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    entry = evaluation_entry(StageOutput.of(probabilities, labels, _CPU))
    return {"accuracy": entry["accuracy"], "balanced_accuracy": entry["balanced_accuracy"]}


@dataclass(slots=True)
class AdmissionReport:
    """The outcome of :func:`view_admission`: the ``--out`` document and the lines to print."""

    payload: dict[str, Any]
    lines: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        """Whether the view passed (mean accuracy drop at most :data:`MAX_DROP`)."""
        return bool(self.payload["admitted"])


def _device(member: Member, name: str | None) -> torch.device:
    trained = member.metrics.get("machine", {}).get("device", "cpu")
    chosen = name if name is not None else str(trained)
    check_accelerator(chosen)
    return torch.device(resolve_accelerator(chosen))


def _read_runs(run_dirs: Sequence[Path]) -> list[tuple[Member, Settings]]:
    """Each run with its ``Settings``, checked (module docstring, "Refusals")."""
    if not run_dirs:
        raise ConfigError("the admission test needs at least one RUN_DIR")
    runs = []
    for path in run_dirs:
        member = read_member(path)
        settings = member_settings(member)
        _check_run(member, settings)
        runs.append((member, settings))
    return runs


def _selection_rows(splits_file: Path, prepared: Path) -> npt.NDArray[np.int64]:
    """The screen's selection half (sorted Validation indices) of a verified splits file.

    Raises:
        DataError: if the file is unreadable or not made for ``prepared`` (``SplitsError``).
    """
    splits = read_splits(splits_file)
    verify_splits(splits, prepared)
    return np.asarray(splits.screen.selection, dtype=np.int64)


def _selection_stamps(prepared: Path, rows: npt.NDArray[np.int64]) -> torch.Tensor:
    """The stored float32 Validation stamps ``rows`` of ``prepared``, (n, 3, 63, 63), CPU."""
    stored = load_prepared(prepared).splits[_SPLIT].images_f32
    if stored is None:
        raise DataError(f"{prepared}: {_SPLIT} has no float32 stamps (images_f32.npy)")
    chosen = np.asarray(stored[rows])  # only these rows are read from the memory map
    return torch.from_numpy(np.ascontiguousarray(chosen.transpose(0, 3, 1, 2)))


def _score_run(
    member: Member,
    settings: Settings,
    prepared: Path,
    rows: npt.NDArray[np.int64],
    viewed: Sequence[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """One run's clean and viewed scores on ``rows`` and its drops (a ``runs`` row)."""
    whole = paper_crops(settings, prepared, (_SPLIT,))[_SPLIT]
    split = TensorSplit(
        images=np.ascontiguousarray(whole.images[rows]),
        features=np.ascontiguousarray(whole.features[rows]),
        labels=np.ascontiguousarray(whole.labels[rows]),
        n_classes=whole.n_classes,
    )
    labels = torch.from_numpy(split.labels)
    net = member_net(member, settings).to(device)
    batch = settings.eval_batch_size
    clean = _scores(_probabilities(net, split.images, split, device, batch), labels)
    scored = [
        _scores(_probabilities(net, model_input(x, settings), split, device, batch), labels)
        for x in viewed
    ]
    mean = {
        key: statistics.mean(s[key] for s in scored) for key in ("accuracy", "balanced_accuracy")
    }
    return {
        "run_dir": str(member.run_dir),
        "run_id": member.manifest.run_id,
        "seed": member.seed,
        "clean": clean,
        "viewed": scored,
        "viewed_mean": mean,
        "drop": clean["accuracy"] - mean["accuracy"],
        "drop_balanced": clean["balanced_accuracy"] - mean["balanced_accuracy"],
    }


def view_admission(
    run_dirs: Sequence[Path],
    view: AdmissionView,
    *,
    splits: Path,
    out: Path,
    data: Path | None = None,
    seed: int = 0,
    repeats: int = 1,
    device: str | None = None,
    overwrite: bool = False,
) -> AdmissionReport:
    """Plan 5.2's admission test of ``view`` on the screen's selection half (module docstring).

    ``splits`` is the stored splits file whose screen's selection half is evaluated (never
    its decision half). ``data`` defaults to the first run's ``data_dir / dataset``;
    ``device`` to the device the first run trained on. Writes ``out`` (JSON, atomically)
    and returns the report.

    Raises:
        ConfigError: on a refusal (module docstring), or an unavailable device.
        DataError: if the layout is unreadable or has no float32 Validation stamps, or the
            splits file is unreadable or made for other data.
    """
    if not view.any:
        raise ConfigError("no view to test: give at least one view flag")
    if repeats < 1 or seed < 0:
        raise ConfigError(f"need repeats >= 1 and seed >= 0, got {repeats} and {seed}")
    if out.suffix != ".json":
        raise ConfigError(f"--out must be a .json file, got {out}")
    if out.exists() and not overwrite:
        raise ConfigError(f"{out} exists; pass --overwrite to replace it")
    runs = _read_runs(run_dirs)
    first = runs[0][1]
    prepared = dataset_path(first) if data is None else data
    digest = data_sha256(prepared)
    for member, _ in runs:
        if member.manifest.identity.data_sha256 != digest:
            raise ConfigError(f"{prepared} is not the data {member.run_dir} trained on")
    rows = _selection_rows(splits, prepared)
    stamps = _selection_stamps(prepared, rows)
    fits = torch.from_numpy(stamp_fits(stamps.numpy())) if view.views.noise else None
    target = _device(runs[0][0], device)
    DevicePolicy(
        cast(Literal["cpu", "cuda", "mps"], target.type),
        first.deterministic,
        cast(Literal["ieee", "tf32"], first.fp32_precision),
    ).apply()
    viewed = [view_stamps(stamps, view, seed=seed, repeat=r, fits=fits) for r in range(repeats)]
    scored = [
        _score_run(member, settings, prepared, rows, viewed, target) for member, settings in runs
    ]
    drop = statistics.mean(row["drop"] for row in scored)
    admitted = drop <= MAX_DROP
    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "command": "simclr-view-admission",
        "view": view.describe(),
        "applied": "once per stamp at 63 px (the pilot's arithmetic), then the run's own crop",
        "seed": seed,
        "repeats": repeats,
        "split": _SPLIT,
        "part": _PART,
        "splits": {"path": str(splits), "sha256": sha256_file(splits)},
        "rows": rows.tolist(),
        "n_stamps": int(stamps.shape[0]),
        "data": {"path": str(prepared), "sha256": digest},
        "metric": "accuracy",
        "max_drop": MAX_DROP,
        "drop": drop,
        "drop_balanced": statistics.mean(row["drop_balanced"] for row in scored),
        "admitted": admitted,
        "verdict": "admitted" if admitted else "not admitted",
        "runs": scored,
        "evaluation_machine": {k: v for k, v in machine_record(target).items() if k != "engine"},
        "versions": {"astro_simclr": version("astro-simclr"), "torch": str(torch.__version__)},
    }
    cleaned = cast(dict[str, Any], finite(payload))
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(out, (json.dumps(cleaned, indent=2, allow_nan=False) + "\n").encode())
    lines = [
        f"simclr-view-admission: {json.dumps(view.describe())} on the {_PART} of {prepared} "
        f"({stamps.shape[0]} stamps, {repeats} repeat(s), seed {seed})"
    ]
    lines += [
        f"  seed {row['seed']}: accuracy {row['clean']['accuracy']:.4f} -> "
        f"{row['viewed_mean']['accuracy']:.4f} (drop {100 * row['drop']:+.2f} points)"
        for row in scored
    ]
    lines.append(
        f"  mean drop {100 * drop:+.2f} points (max {100 * MAX_DROP:.0f}): "
        f"{payload['verdict']}; wrote {out}"
    )
    return AdmissionReport(cleaned, lines)
