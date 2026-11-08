"""``ensemble-eval``: seed ensembles and test-time augmentation of stored split runs (plan row D4).

``astro-simclr ensemble-eval RUN_DIR... --splits FILE --fold K --tta MODE --out JSON``
evaluates, without retraining, the selected weights of several finished runs of one arm
(the same config, engine, data, machine class and part of the same splits file; one run
per seed) on the part's evaluation stamps, with the test-time augmentation ``MODE``
(:mod:`astro_simclr.tta`), and reports:

- each member's balanced accuracy (with its run's own stored value next to it);
- the mean of the members' balanced accuracies;
- the **ensemble**: the mean over the members of their softmax probabilities, stamp by
  stamp (``protocol.predictions.ensemble``, the seed ensemble McNemar compares), and its
  gain over the members' mean.

The pre-registered D4 hypothesis reads these numbers: a 5-seed ensemble beats the mean of
its own members by at least 0.4 points of balanced accuracy, and ``d4_shift`` alone adds at
least 0.1 (one run per invocation with ``--tta d4_shift`` against the same with ``rot4``,
or against the runs themselves, with ``compare-variants --splits``).

**The ensemble as an arm.** ``--out FILE.json`` is shaped like a split run's
``metrics.json`` for what ``compare-variants`` reads (``seed``, ``machine``,
``data.source``, ``protocol``, ``predictions``), and the predictions are written next to it
(``<stem>.predictions_evaluation.npz``, the format of ``protocol.predictions``), so the JSON
file itself is a ``RUN_DIR`` argument of ``compare-variants --splits FILE`` (which accepts a
metrics file for a directory). Its ``seed`` is the member's seed for one member (so a TTA'd
run pairs with its own run), the sorted list of the members' seeds for an ensemble (so two
ensembles of the same seeds pair cell by cell). ``machine`` is the members' training
machine record (they must agree); where ``ensemble-eval`` ran is ``evaluation_machine``.

**Test.** The evaluation part of the fold (or the screen's decision half) is the default
and the only decision input. ``--test`` adds the real Test split (500 stamps for ``red``),
which the protocol reports once per pre-registered comparison and never decides on
(``docs/evaluation-protocol.md``); without the flag, Test is never read.

**Refusals** (:func:`check_members`, before any data is read): a run that is not
``finished``, not a ``train-supervised`` run of the paper classifier, or of another part
(fold or screen) or splits file than asked; runs that differ in config hash, engine, data,
machine class, numerics epoch or pretrained weights, or in the machine they trained on
(``reference.machine_signature``); a repeated run (the same seed twice: an attempt of
the same run, or the same directory). All are ``ConfigError``
(exit 2), with the run and the field named.

**Exactness.** Each member is evaluated as its run evaluated itself: the weights of
``ckpt/best.pt`` (plain) or ``checkpoints/best.ckpt`` (Lightning) in the model
``build_paper_model`` builds from the manifest's config, the input of
``tasks.paper.paper_crops``, the engine's evaluation batches (``engine.open_eval``) of the
config's ``eval_batch_size``, in ``eval()`` and ``inference_mode``, with the config's
float32 precision and determinism (``engine.DevicePolicy``). With ``--tta rot4`` (the
default) the probabilities are therefore bit for bit the run's own ``predictions_*.npz``
on the device it trained on (``tests/test_ensemble_eval.py``); on another device (the
default ``--accelerator`` is the members' own) they differ by float rounding, and the
report says so.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import torch

from astro_simclr.config import ConfigError, Settings, settings_from_mapping
from astro_simclr.data.float_input import crop_split
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.engine import Dataset, DevicePolicy, TierChoice, load_best, open_eval
from astro_simclr.engine.record import machine_record
from astro_simclr.feature_sets import feature_columns
from astro_simclr.losses import P_stamps_loss
from astro_simclr.paper_variants import build_paper_model
from astro_simclr.protocol.predictions import Predictions, ensemble, save_predictions
from astro_simclr.protocol.split_runs import SplitRequest, SplitRun, file_sha256, open_split_run
from astro_simclr.reference import load_run, machine_signature
from astro_simclr.runs.identity import SCREEN, SplitPart, data_sha256
from astro_simclr.runs.layout import BEST_CHECKPOINT, LIGHTNING_BEST
from astro_simclr.runs.manifest import Manifest, ManifestError, is_finished, read_manifest
from astro_simclr.tasks.common import (
    PAPER_STAMP_CLASSIFIER,
    check_accelerator,
    check_schedule,
    dataset_path,
)
from astro_simclr.tasks.paper import PaperNet, paper_crops
from astro_simclr.tasks.plain import resolve_accelerator
from astro_simclr.tasks.results import StageOutput, evaluation_entry, finite
from astro_simclr.training.tensor_splits import TensorSplit, protocol_splits
from astro_simclr.tta import (
    View,
    apply_dihedral,
    check_tta,
    combine_view_logits,
    model_symmetry,
    tta_views,
)

__all__ = [
    "FORMAT",
    "EnsembleReport",
    "EvalPart",
    "Member",
    "check_members",
    "ensemble_eval",
    "evaluation_parts",
    "member_net",
    "member_probabilities",
    "read_member",
]

#: ``--out``'s ``format`` (the file is also shaped like a split run's ``metrics.json``).
FORMAT: Final = "astro-simclr-ensemble-eval"
#: Its version; bumped on any change of the keys or their meaning.
FORMAT_VERSION: Final = 1
#: The only training command whose runs ``ensemble-eval`` reads.
_COMMAND: Final = "train-supervised"
#: Stamps per chunk when re-cropping the stamps for ``d4_shift`` (about 190 MB of float32).
_CHUNK: Final = 4096


@dataclass(frozen=True, slots=True)
class Member:
    """One run of the ensemble: its directory, validated manifest and ``metrics.json``."""

    run_dir: Path
    manifest: Manifest
    metrics: dict[str, Any]

    @property
    def seed(self) -> int:
        """The run's seed (its identity's)."""
        return self.manifest.identity.seed


def read_member(run_dir: Path) -> Member:
    """Read and validate ``run_dir/manifest.json`` and read its ``metrics.json``.

    Raises:
        ConfigError: if the manifest is missing or invalid, or the metrics are unreadable.
    """
    try:
        manifest = read_manifest(run_dir)
    except ManifestError as err:
        raise ConfigError(f"{run_dir}: {err}") from None
    return Member(run_dir, manifest, load_run(run_dir))


def _part_name(part: SplitPart) -> str:
    return "the screen" if part == SCREEN else f"fold {part}"


def _check_member(member: Member, splits_sha256: str, part: SplitPart) -> None:
    """One member alone: finished, the paper classifier, on ``part`` of the splits file.

    Raises:
        ConfigError: otherwise, naming the run.
    """
    manifest, where = member.manifest, member.run_dir
    if not is_finished(manifest.status):
        raise ConfigError(f"{where}: the run is {manifest.status}, not finished")
    model = manifest.config.get("supervised_model")
    if manifest.command != _COMMAND or model != PAPER_STAMP_CLASSIFIER:
        msg = (
            f"{where}: ensemble-eval reads {_COMMAND} runs of {PAPER_STAMP_CLASSIFIER}, "
            f"not {manifest.command} of {model}"
        )
        raise ConfigError(msg)
    identity = manifest.identity
    if identity.split != part:
        got = "no split" if identity.split is None else _part_name(identity.split)
        raise ConfigError(f"{where}: a run of {got}, not of {_part_name(part)}")
    if identity.splits_sha256 != splits_sha256:
        msg = (
            f"{where}: trained on another splits file (SHA-256 {identity.splits_sha256}, "
            f"not {splits_sha256})"
        )
        raise ConfigError(msg)


def _arm(member: Member) -> dict[str, object]:
    """What every member of one ensemble shares (module docstring, "Refusals")."""
    identity = member.manifest.identity
    return {
        "config": identity.config_sha256,
        "engine": member.manifest.engine,
        "data": identity.data_sha256,
        "machine class": identity.machine_class,
        "numerics epoch": identity.numerics_epoch,
        "pretrained weights": identity.pretrained_sha256,
        "training machine": machine_signature(member.metrics),
    }


def check_members(members: Sequence[Member], *, splits_sha256: str, part: SplitPart) -> None:
    """Refuse members that are not runs of one arm on ``part`` of one splits file.

    ``part`` is the fold number or ``"screen"`` (``runs.identity.SplitPart``) and
    ``splits_sha256`` the SHA-256 of the splits file asked for (module docstring,
    "Refusals").

    Raises:
        ConfigError: on the first refusal, naming the run and the field.
    """
    if not members:
        raise ConfigError("ensemble-eval needs at least one RUN_DIR")
    for member in members:
        _check_member(member, splits_sha256, part)
    first, expected = members[0], _arm(members[0])
    for member in members[1:]:
        for name, value in _arm(member).items():
            if value != expected[name]:
                msg = (
                    f"{member.run_dir} and {first.run_dir} differ in {name} ({value!r} and "
                    f"{expected[name]!r}): an ensemble is one arm's seeds"
                )
                raise ConfigError(msg)
    # Every other identity field is equal now, so a repeated run id is a repeated seed.
    run_ids = [member.manifest.run_id for member in members]
    if len(set(run_ids)) != len(run_ids):
        seeds = [member.seed for member in members]
        raise ConfigError(f"the runs repeat a run (seeds {seeds}); one run per seed")


def member_settings(member: Member) -> Settings:
    """The run's ``Settings`` from its manifest's config, checked as the paper model's.

    Raises:
        ConfigError: if the config no longer reads, or is not the paper model's schedule.
    """
    settings = settings_from_mapping(dict(member.manifest.config))
    check_schedule(_COMMAND, settings)
    return settings


def _best_state(member: Member) -> dict[str, torch.Tensor]:
    """The selected weights: ``ckpt/best.pt`` (plain, its run id checked) or Lightning's.

    Raises:
        ConfigError: if the file is missing (a finished classifier's ``best`` may have been
            dropped by the retention policy) or belongs to another run.
    """
    engine = member.manifest.engine
    path = member.run_dir / (BEST_CHECKPOINT if engine == "plain" else LIGHTNING_BEST)
    if not path.is_file():
        msg = f"{path}: the run's selected weights are missing (ensemble-eval needs best)"
        raise ConfigError(msg)
    if engine == "plain":
        best = load_best(path)
        if best["identity"]["run_id"] != member.manifest.run_id:
            raise ConfigError(f"{path} holds the weights of another run")
        return best["model"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ConfigError(f"{path} is not a Lightning checkpoint")
    return cast(dict[str, torch.Tensor], checkpoint["state_dict"])


def member_net(member: Member, settings: Settings) -> PaperNet:
    """The run's model with its selected weights, on the CPU, in ``eval()``.

    Built as ``tasks.paper.build_paper_task`` builds it, without the ConvNeXt backbone
    download (every weight comes from the checkpoint).

    Raises:
        ConfigError: as :func:`_best_state`, or if the weights do not fit the model.
    """
    columns = feature_columns(settings.feature_set, with_features=settings.with_features)
    model = build_paper_model(
        settings.paper_arch,
        group_fields=settings.group_fields,
        context=settings.context,
        with_features=settings.with_features,
        dropout_active=settings.dropout_active,
        drop_rate=settings.drop_rate,
        feature_set=settings.feature_set,
    )
    net = PaperNet(model, P_stamps_loss(settings.beta_loss), reads_features=bool(columns))
    try:
        net.load_state_dict(_best_state(member))
    except RuntimeError as err:
        raise ConfigError(f"{member.run_dir}: the weights do not fit its config ({err})") from None
    return net.eval()


#: Where a part's stamps are: (split name, rows of that split), in the part's order.
Segments = tuple[tuple[str, npt.NDArray[np.int64]], ...]


@dataclass(frozen=True, slots=True)
class EvalPart:
    """One evaluated set: its crops (``split``) and where its stamps come from.

    ``stage`` names it in the output (``evaluation`` or ``test``), ``name`` is the
    ``metrics.json`` description of the set, ``index`` what the predictions file stores
    (pool index for a fold, Validation index for the screen, file order for Test), and
    ``segments`` the rows of the prepared splits, in order, for re-cropping.
    """

    stage: Literal["evaluation", "test"]
    name: str
    split: TensorSplit
    index: npt.NDArray[np.int64]
    segments: Segments


def evaluation_parts(
    settings: Settings, data: Path, split_run: SplitRun, *, test: bool
) -> list[EvalPart]:
    """The part's evaluation stamps, and Test with ``test`` (module docstring, "Test").

    Raises:
        DataError: as ``tasks.paper.paper_crops``, or if the splits do not fit the data.
    """
    parts = split_run.parts()
    n_train = split_run.splits.n_train
    pool = paper_crops(settings, data, ("Train", "Validation"))
    roles = protocol_splits(pool, parts, n_train)
    rows = parts.evaluation
    in_train = rows < n_train
    segments: Segments = (("Train", rows[in_train]), ("Validation", rows[~in_train] - n_train))
    out = [
        EvalPart(
            "evaluation", parts.evaluation_name, roles["Test"], parts.evaluation_index, segments
        )
    ]
    if test:
        split = paper_crops(settings, data, ("Test",))["Test"]
        index = np.arange(split.labels.size, dtype=np.int64)
        out.append(EvalPart("test", "Test", split, index, (("Test", index),)))
    return out


def shifted_images(
    settings: Settings, data: Path, part: EvalPart, offset: tuple[int, int]
) -> npt.NDArray[np.float32]:
    """The part's crops re-taken at ``offset`` from the float32 stamps, (N, 3, 21, 21).

    ``offset`` ``(0, 0)`` gives exactly ``part.split.images`` (the same
    ``crop_then_normalise``); the stamps are read ``_CHUNK`` at a time.

    Raises:
        DataError: if the data have no float32 stamps.
    """
    prepared = load_prepared(data)
    blocks = []
    for name, rows in part.segments:
        stamps = prepared.splits[name].images_f32
        if stamps is None:
            raise DataError(f"{data}: {name} has no float32 stamps, which d4_shift re-crops")
        for start in range(0, rows.size, _CHUNK):
            chunk = np.asarray(stamps[rows[start : start + _CHUNK]])
            crops = crop_split(chunk, nan_approximation=settings.nan_approximation, offset=offset)
            blocks.append(crops.transpose(0, 3, 1, 2))
    return np.ascontiguousarray(np.concatenate(blocks))


#: ``images_at(offset)``: the part's (N, C, S, S) model input with the crop at ``offset``.
ImagesAt = Callable[[tuple[int, int]], npt.NDArray[np.float32]]


def member_probabilities(
    net: PaperNet,
    part: EvalPart,
    views: Sequence[View],
    images_at: ImagesAt,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The member's TTA probabilities (N, C) and labels (N,) on ``part``, on the CPU.

    For each offset of ``views``, the part's crops at that offset go through the engine's
    evaluation batches (``open_eval``, ``batch_size`` rows, in order); each batch goes
    through ``net`` once per dihedral view of that offset. Then, batch by batch, the views'
    logits are combined (:func:`astro_simclr.tta.combine_view_logits`) and soft-maxed, as
    ``PaperTask.eval_step`` does: with the single identity view of ``rot4`` this is that
    computation exactly.
    """
    offsets = list(dict.fromkeys(view.offset for view in views))
    per_view: dict[View, list[torch.Tensor]] = {view: [] for view in views}
    targets: list[torch.Tensor] = []
    split = part.split
    with torch.inference_mode():
        for offset in offsets:
            images = split.images if offset == (0, 0) else images_at(offset)
            dataset = Dataset(
                images=images,
                features=split.features,
                labels=split.labels,
                n_classes=split.n_classes,
            )
            batches = open_eval(dataset, TierChoice(), device, batch_size=batch_size)
            for x_img, x_feat, y in batches:
                for view in views:
                    if view.offset == offset:
                        per_view[view].append(net(apply_dihedral(x_img, view), x_feat))
                if offset == offsets[0]:
                    targets.append(y.argmax(dim=1))
    probabilities = [
        combine_view_logits([per_view[view][b] for view in views]).softmax(dim=1).cpu()
        for b in range(len(targets))
    ]
    return torch.cat(probabilities), torch.cat(targets).cpu()


@dataclass(slots=True)
class EnsembleReport:
    """The outcome of :func:`ensemble_eval`: the ``--out`` document and the lines to print."""

    payload: dict[str, Any]
    lines: list[str] = field(default_factory=list)


def _scores(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return evaluation_entry(StageOutput.of(probabilities, labels, torch.device("cpu")))


def _summary(values: Sequence[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def _stored_balanced_accuracy(member: Member, stage: str) -> float | None:
    value = member.metrics.get("evaluation_balanced_accuracy") if stage == "evaluation" else None
    return float(value) if isinstance(value, int | float) else None


def _device(members: Sequence[Member], accelerator: str | None) -> torch.device:
    """The evaluation device: ``accelerator``, else the one the members trained on."""
    trained = members[0].metrics.get("machine", {}).get("device", "cpu")
    name = accelerator if accelerator is not None else str(trained)
    check_accelerator(name)
    return torch.device(resolve_accelerator(name))


def ensemble_eval(
    run_dirs: Sequence[Path],
    splits_file: Path,
    *,
    fold: int | None,
    tta: str = "rot4",
    test: bool = False,
    out: Path,
    data: Path | None = None,
    accelerator: str | None = None,
) -> EnsembleReport:
    """Evaluate the runs' selected weights with ``tta``, alone and as an ensemble (module doc).

    ``fold`` is the fold (``None``: the screen); ``data`` the prepared layout (default: the
    members' ``data_dir / dataset``, whose data hash must be theirs); ``accelerator`` the
    device (default: the members' training device). Writes ``out`` and its predictions.

    Raises:
        ConfigError: on a refusal (:func:`check_members`), an unknown or unsupported TTA
            mode, data of another hash, missing weights, an unavailable accelerator, or an
            ``out`` that is not a ``.json`` path.
        DataError: if the data or the splits file are unreadable or do not match.
    """
    if out.suffix != ".json":
        raise ConfigError(f"--out must be a .json file, got {out}")
    try:
        splits_sha256 = file_sha256(splits_file)
    except OSError as err:
        raise ConfigError(f"--splits {splits_file}: cannot read it ({err})") from None
    part_id: SplitPart = SCREEN if fold is None else fold
    members = [read_member(path) for path in run_dirs]
    check_members(members, splits_sha256=splits_sha256, part=part_id)
    settings = member_settings(members[0])
    try:
        check_tta(
            tta,
            paper_arch=settings.paper_arch,
            context=settings.context,
            paper_input=settings.paper_input,
        )
    except ValueError as err:
        raise ConfigError(str(err)) from None
    views = tta_views(tta, model_symmetry(settings.paper_arch))
    prepared = dataset_path(settings) if data is None else data
    expected = members[0].manifest.identity.data_sha256
    if data_sha256(prepared) != expected:
        raise ConfigError(f"{prepared} is not the data the runs trained on (SHA-256 {expected})")
    split_run = open_split_run(SplitRequest(splits_file, fold), prepared)
    device = _device(members, accelerator)
    policy = DevicePolicy(
        cast(Literal["cpu", "cuda", "mps"], device.type),
        settings.deterministic,
        cast(Literal["ieee", "tf32"], settings.fp32_precision),
    )
    policy.apply()
    parts = evaluation_parts(settings, prepared, split_run, test=test)
    nets = [member_net(member, member_settings(member)).to(device) for member in members]
    lines = [
        f"ensemble-eval: {_part_name(part_id)} of {splits_file}, {len(members)} member(s), "
        f"tta {tta} ({len(views)} view(s): {', '.join(v.describe() for v in views)})"
    ]
    trained = members[0].metrics.get("machine", {}).get("device")
    if trained != device.type:
        lines.append(
            f"  WARN: evaluated on {device.type}, trained on {trained}: the probabilities "
            "differ from the runs' own by float rounding"
        )
    results: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    member_rows: list[dict[str, Any]] = [
        {
            "run_dir": str(m.run_dir),
            "run_id": m.manifest.run_id,
            "attempt": m.manifest.attempt,
            "seed": m.seed,
        }
        for m in members
    ]
    for part in parts:

        def images_at(offset: tuple[int, int], part: EvalPart = part) -> npt.NDArray[np.float32]:
            return shifted_images(settings, prepared, part, offset)

        outputs = [
            member_probabilities(net, part, views, images_at, device, settings.eval_batch_size)
            for net in nets
        ]
        member_preds = [
            Predictions(probs.float().numpy(), labels.numpy().astype(np.int64), part.index)
            for probs, labels in outputs
        ]
        combined = ensemble(member_preds)
        ensemble_entry = _scores(
            torch.from_numpy(combined.probabilities), torch.from_numpy(combined.labels)
        )
        balanced = []
        for row, member, (probs, labels) in zip(member_rows, members, outputs, strict=True):
            entry = _scores(probs, labels)
            balanced.append(entry["balanced_accuracy"])
            row[part.stage] = {
                "accuracy": entry["accuracy"],
                "balanced_accuracy": entry["balanced_accuracy"],
                "stored_balanced_accuracy": _stored_balanced_accuracy(member, part.stage),
            }
        mean = statistics.mean(balanced)
        gain = ensemble_entry["balanced_accuracy"] - mean
        results[part.stage] = {
            "member_mean_balanced_accuracy": mean,
            "member_std_balanced_accuracy": statistics.stdev(balanced)
            if len(balanced) > 1
            else None,
            "ensemble": ensemble_entry,
            "ensemble_minus_member_mean": gain,
        }
        file = f"{out.stem}.predictions_{part.stage}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_predictions(combined, out.parent / file)
        predictions[part.stage] = {"file": file, "split": part.name, "n": int(part.index.size)}
        lines.append(f"  {part.stage} ({part.name}, {part.index.size} stamps):")
        for row in member_rows:
            stored = row[part.stage]["stored_balanced_accuracy"]
            note = "" if stored is None else f" (run's own {stored:.4f})"
            lines.append(
                f"    seed {row['seed']}: balanced accuracy "
                f"{row[part.stage]['balanced_accuracy']:.4f}{note}"
            )
        lines.append(f"    mean of the members: {_summary(balanced)}")
        lines.append(
            f"    ensemble (mean of probabilities): {ensemble_entry['balanced_accuracy']:.4f}, "
            f"ensemble - mean {gain:+.4f}"
        )
    seeds = sorted(member.seed for member in members)
    evaluation_machine = {k: v for k, v in machine_record(device).items() if k != "engine"}
    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "command": "ensemble-eval",
        "tta": {
            "mode": tta,
            "symmetry": model_symmetry(settings.paper_arch),
            "views": [view.describe() for view in views],
            "combine": "mean of the views' logits, then softmax",
        },
        "ensemble": "mean of the members' softmax probabilities",
        "seed": seeds[0] if len(seeds) == 1 else seeds,
        "members": member_rows,
        "config_sha256": members[0].manifest.identity.config_sha256,
        "engine": members[0].manifest.engine,
        "machine": members[0].metrics.get("machine"),
        "evaluation_machine": evaluation_machine,
        "versions": {"astro_simclr": version("astro-simclr"), "torch": str(torch.__version__)},
        "data": members[0].metrics.get("data"),
        "protocol": split_run.describe(),
        "evaluation_balanced_accuracy": results["evaluation"]["ensemble"]["balanced_accuracy"],
        "results": results,
        "predictions": predictions,
    }
    cleaned = cast(dict[str, Any], finite(payload))
    out.write_text(json.dumps(cleaned, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines.append(f"ensemble-eval: wrote {out}")
    return EnsembleReport(cleaned, lines)
