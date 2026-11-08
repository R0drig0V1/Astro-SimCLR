"""One :class:`RunRecord` per run directory: from its manifest, or backfilled (legacy).

A run directory holds ``manifest.json`` (every run since task A1), or only
``metrics.json`` (the archived runs: ``results/phase2/**``, ``~/runs`` on the GPU
machine). :func:`read_run_dir` reads either into the same record:

- **Manifest runs** are validated by :func:`~astro_simclr.runs.manifest.read_manifest`
  (hashes recomputed) and carry their full provenance. Their ``metrics.json``, when the run
  has finished, adds the metrics summary.
- **Legacy runs** (``legacy: true``) are backfilled from ``metrics.json``. What it does not
  record is stored as null: the run id and attempt, the data SHA-256 of ADR 0015 (the file
  records the source pickle's SHA-256 only, kept as ``data_source_sha256``), the
  pretrained checkpoint's SHA-256, the git state, ``uv.lock`` and ``numerics_epoch`` (the
  runs predate it). Three fields are derived instead, each from a fact of the code at the
  time: ``status`` is ``finished`` (``run_command`` writes ``metrics.json`` only when the fit
  has ended), ``engine`` is ``lightning`` (the only engine before task A1, and every run
  since writes a manifest), and ``machine_class`` is
  :func:`~astro_simclr.runs.identity.default_machine_class` of the recorded machine.

**Config formats.** The recorded config is read by the reader of its ``config_format``
(:data:`CONFIG_FORMATS`). A ``metrics.json`` without a ``config_format`` key is
``flat-1``, the only format so far: the flat ``Settings`` view
(:func:`~astro_simclr.config.settings_to_plain`). The ``flat-1`` reader lowers a config to
today's flat view before hashing it, so an archived run gets the config hash today's code
gives the same experiment: a field added to ``Settings`` after the run was recorded is
filled with the value that reproduces the code before the field existed
(:data:`FLAT_1_FILLS`, with the reason for each). A field that the table does not fill, or
a key that today's ``Settings`` does not have, makes the config unreadable, and the run is
skipped with that reason rather than given a hash that could match the wrong runs. The
fields added after task A0 need no entry: they follow ADR 0015 rule 9
(``canonical.HASH_NEUTRAL_DEFAULTS``). The nested config of task 4.2 will be a new format
with its own reader, lowering to the same flat view.

A manifest run is ``flat-1`` too (manifest format 1 records the flat view); its config
hash is the one the manifest records, which its reader has already verified.

Two directories can hold the same attempt: ``results/phase2/replication`` is a byte-exact
copy of ``~/runs/replication``. :attr:`RunRecord.key` is the same for both (the run id
and attempt; for a legacy run, the ``metrics.json`` SHA-256), so a comparison counts one
attempt, not two.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from astro_simclr.config import ConfigError, Settings
from astro_simclr.runs.canonical import CanonicalError, config_sha256_of, semantic_config_from_plain
from astro_simclr.runs.identity import SCREEN, SplitPart, default_machine_class
from astro_simclr.runs.layout import METRICS_FILE
from astro_simclr.runs.manifest import MANIFEST_FILE, ManifestError, read_manifest

__all__ = [
    "CONFIG_FORMATS",
    "DEFAULT_CONFIG_FORMAT",
    "FLAT_1_FILLS",
    "RecordError",
    "RunRecord",
    "Source",
    "is_run_dir",
    "lower_flat_1",
    "metrics_summary",
    "read_run_dir",
]

#: Where a record came from: a manifest (task A1 on), or a backfilled ``metrics.json``.
type Source = Literal["manifest", "metrics"]

#: ``config_format`` of a ``metrics.json`` that names none (every file written so far).
DEFAULT_CONFIG_FORMAT: Final = "flat-1"

#: ``flat-1``: the fields added to ``Settings`` after the phase-2 replication runs were
#: recorded and before task A0, each with the value that reproduces the code from before
#: the field existed, and why. A test pins each value to today's default; a default
#: changed later must not change this table.
FLAT_1_FILLS: Final[Mapping[str, tuple[object, str]]] = MappingProxyType(
    {
        "fp32_precision": (
            "ieee",
            "ADR 0010 rule 5: every run before it ran on hardware without TF32",
        ),
        "progress_bar_refresh_rate": (50, "display only (non-semantic)"),
        "paper_arch": ("rot4", "the paper's rotation-tied model, the only one before p4/p4m"),
        "group_fields": (None, "no group-convolution width before p4/p4m"),
        "context": ("none", "no context input before the 63 px context"),
        "rot90_augment": (False, "no random 90-degree rotation before the flag"),
        "aug_noise": (0.0, "C5 off: no class-aware augmentation before ADR 0006"),
        "aug_feature_jitter": (0.0, "C5 off"),
        "synthetic_bogus": (0.0, "C6 off"),
        "aug_min_r2": (0.5, "read only when C5 or C6 is on"),
        "flux_rescale": (0.0, "C7 off: no flux rescaling before ADR 0007"),
        "flux_k_min": (0.5, "read only when C7 is on"),
        "flux_k_max": (1.5, "read only when C7 is on"),
    }
)

#: The metrics a record keeps (:func:`metrics_summary`): per evaluation stage, these keys.
_STAGE_METRICS: Final = ("accuracy", "balanced_accuracy", "macro_f1")


class RecordError(ValueError):
    """A run directory cannot be read into a record (the reason is the message)."""


def _settings_fields() -> frozenset[str]:
    return frozenset(field.name for field in dataclasses.fields(Settings))


def lower_flat_1(config: Mapping[str, object]) -> dict[str, object]:
    """A ``flat-1`` config lowered to today's flat view (module docstring).

    Raises:
        RecordError: if a key is unknown to today's ``Settings``, or a missing field has
            no entry in :data:`FLAT_1_FILLS`.
    """
    fields = _settings_fields()
    unknown = sorted(set(config) - fields)
    missing = sorted(fields - set(config))
    unfilled = [key for key in missing if key not in FLAT_1_FILLS]
    if unknown or unfilled:
        msg = (
            f"config_format flat-1: keys unknown to today's Settings {unknown or 'none'}, "
            f"missing fields no reader fills {unfilled or 'none'}"
        )
        raise RecordError(msg)
    lowered = dict(config)
    for key in missing:
        lowered[key] = FLAT_1_FILLS[key][0]
    return lowered


#: The config readers, by ``config_format``: each lowers a recorded config to today's
#: flat view.
CONFIG_FORMATS: Final[Mapping[str, Callable[[Mapping[str, object]], dict[str, object]]]] = (
    MappingProxyType({"flat-1": lower_flat_1})
)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def metrics_summary(metrics: Mapping[str, Any]) -> dict[str, float]:
    """The numbers a record keeps from ``metrics.json``.

    ``<stage>.accuracy``, ``<stage>.balanced_accuracy`` and ``<stage>.macro_f1`` for each
    stage of ``evaluation`` (``val``, ``test``; a split run's evaluation part), then
    ``monitor_best``, ``best_step``, ``last_step`` and ``seconds_per_step`` (from
    ``timing``). Missing or non-finite values are left out.
    """
    out: dict[str, float] = {}
    evaluation = metrics.get("evaluation")
    if isinstance(evaluation, Mapping):
        for stage, result in sorted(evaluation.items()):
            if not isinstance(result, Mapping):
                continue
            for key in _STAGE_METRICS:
                value = _finite(result.get(key))
                if value is not None:
                    out[f"{stage}.{key}"] = value
    timing = metrics.get("timing")
    top = {
        "monitor_best": metrics.get("monitor_best"),
        "best_step": metrics.get("best_step"),
        "last_step": metrics.get("last_step"),
        "seconds_per_step": timing.get("seconds_per_step") if isinstance(timing, Mapping) else None,
    }
    for key, raw in top.items():
        value = _finite(raw)
        if value is not None:
            out[key] = value
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRecord:
    """One run directory as the registry indexes it (module docstring).

    ``run_dir`` is absolute. ``config`` is the config as recorded (not lowered);
    ``machine`` is the recorded machine record (ADR 0010), ``None`` if absent;
    ``metrics`` is :func:`metrics_summary`, empty before ``metrics.json`` exists.
    """

    run_dir: Path
    source: Source
    legacy: bool
    config_format: str
    run_id: str | None
    attempt: int | None
    command: str
    config_sha256: str
    seed: int
    split: SplitPart
    data_sha256: str | None
    splits_sha256: str | None
    pretrained_sha256: str | None
    data_source_sha256: str | None
    machine_class: str | None
    engine: str
    numerics_epoch: int | None
    git_sha: str | None
    git_dirty: bool | None
    uv_lock_sha256: str | None
    status: str
    created_at: str | None
    started_at: str | None
    ended_at: str | None
    preset: str | None
    metrics_sha256: str | None
    machine: Mapping[str, object] | None
    metrics: Mapping[str, float]
    config: Mapping[str, object]

    @property
    def key(self) -> str:
        """The attempt's key: ``<run id[:16]>-a<attempt>``, or ``legacy-<metrics SHA[:16]>``.

        Copies of one attempt in two directories share it (module docstring).
        """
        if self.run_id is not None and self.attempt is not None:
            return f"{self.run_id[:16]}-a{self.attempt}"
        return f"legacy-{(self.metrics_sha256 or '')[:16]}"


def is_run_dir(path: Path) -> bool:
    """Whether ``path`` holds a ``manifest.json`` or a ``metrics.json``."""
    return (path / MANIFEST_FILE).is_file() or (path / METRICS_FILE).is_file()


def _read_metrics(path: Path) -> tuple[dict[str, Any], str]:
    """``metrics.json`` as a mapping, and the file's SHA-256."""
    try:
        raw = path.read_bytes()
    except OSError as err:
        raise RecordError(f"cannot read {path.name} ({err})") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as err:
        raise RecordError(f"{path.name} is not valid JSON ({err})") from None
    if not isinstance(data, dict):
        raise RecordError(f"{path.name} is not a JSON object")
    return data, hashlib.sha256(raw).hexdigest()


def _preset(metrics: Mapping[str, Any]) -> str | None:
    preset = metrics.get("preset")
    if isinstance(preset, Mapping) and isinstance(preset.get("name"), str):
        return str(preset["name"])
    return None


def _from_manifest(run_dir: Path) -> RunRecord:
    try:
        manifest = read_manifest(run_dir / MANIFEST_FILE)
    except ManifestError as err:
        raise RecordError(str(err)) from None
    metrics: dict[str, Any] = {}
    metrics_sha = None
    if (run_dir / METRICS_FILE).is_file():
        metrics, metrics_sha = _read_metrics(run_dir / METRICS_FILE)
    identity = manifest.identity
    return RunRecord(
        run_dir=run_dir,
        source="manifest",
        legacy=False,
        config_format=DEFAULT_CONFIG_FORMAT,
        run_id=manifest.run_id,
        attempt=manifest.attempt,
        command=manifest.command,
        config_sha256=identity.config_sha256,
        seed=identity.seed,
        split=identity.split,
        data_sha256=identity.data_sha256,
        splits_sha256=identity.splits_sha256,
        pretrained_sha256=identity.pretrained_sha256,
        data_source_sha256=_data_source_sha256(metrics),
        machine_class=identity.machine_class,
        engine=manifest.engine,
        numerics_epoch=identity.numerics_epoch,
        git_sha=manifest.git.sha,
        git_dirty=manifest.git.dirty,
        uv_lock_sha256=manifest.uv_lock_sha256,
        status=manifest.status,
        created_at=manifest.created_at,
        started_at=manifest.started_at,
        ended_at=manifest.ended_at,
        preset=_preset(metrics),
        metrics_sha256=metrics_sha,
        machine=manifest.machine,
        metrics=MappingProxyType(metrics_summary(metrics)),
        config=manifest.config,
    )


def _data_source_sha256(metrics: Mapping[str, Any]) -> str | None:
    data = metrics.get("data")
    source = data.get("source") if isinstance(data, Mapping) else None
    sha = source.get("sha256") if isinstance(source, Mapping) else None
    return sha if isinstance(sha, str) else None


def _legacy_split(metrics: Mapping[str, Any]) -> tuple[SplitPart, str | None]:
    """The split part and splits SHA-256 of ``metrics["protocol"]`` (a split run), or none."""
    protocol = metrics.get("protocol")
    if protocol is None:
        return None, None
    if not isinstance(protocol, Mapping):
        raise RecordError("protocol is not a mapping")
    sha, mode, fold = protocol.get("splits_sha256"), protocol.get("mode"), protocol.get("fold")
    if not isinstance(sha, str):
        raise RecordError("protocol has no splits_sha256")
    if mode == SCREEN and fold is None:
        return SCREEN, sha
    if mode == "fold" and type(fold) is int and fold >= 0:
        return fold, sha
    raise RecordError(f"protocol mode {mode!r} with fold {fold!r} is not a split run")


def _legacy_machine_class(machine: Mapping[str, Any] | None) -> str | None:
    if machine is None:
        return None
    device, gpu, processor = machine.get("device"), machine.get("gpu"), machine.get("processor")
    if not isinstance(device, str) or not isinstance(processor, str):
        return None
    if gpu is not None and not isinstance(gpu, str):
        return None
    return default_machine_class(device, gpu, processor)


def _from_metrics(run_dir: Path) -> RunRecord:
    metrics, metrics_sha = _read_metrics(run_dir / METRICS_FILE)
    command, seed, config = metrics.get("command"), metrics.get("seed"), metrics.get("config")
    if not isinstance(command, str) or not isinstance(config, dict):
        raise RecordError(f"{METRICS_FILE} has no command and config: not a training run")
    if type(seed) is not int or seed < 0 or config.get("seed") != seed:
        raise RecordError(f"{METRICS_FILE}: seed {seed!r} is not the config's seed")
    config_format = metrics.get("config_format", DEFAULT_CONFIG_FORMAT)
    reader = CONFIG_FORMATS.get(config_format) if isinstance(config_format, str) else None
    if reader is None:
        raise RecordError(f"config_format {config_format!r} has no reader in this version")
    lowered = reader(config)
    try:
        config_sha = config_sha256_of(semantic_config_from_plain(command, lowered))
    except (ConfigError, CanonicalError) as err:
        raise RecordError(f"config: {err}") from None
    split, splits_sha = _legacy_split(metrics)
    machine = metrics.get("machine")
    machine = machine if isinstance(machine, dict) else None
    return RunRecord(
        run_dir=run_dir,
        source="metrics",
        legacy=True,
        config_format=config_format,
        run_id=None,
        attempt=None,
        command=command,
        config_sha256=config_sha,
        seed=seed,
        split=split,
        data_sha256=None,
        splits_sha256=splits_sha,
        pretrained_sha256=None,
        data_source_sha256=_data_source_sha256(metrics),
        machine_class=_legacy_machine_class(machine),
        engine="lightning",
        numerics_epoch=None,
        git_sha=None,
        git_dirty=None,
        uv_lock_sha256=None,
        status="finished",
        created_at=None,
        started_at=None,
        ended_at=None,
        preset=_preset(metrics),
        metrics_sha256=metrics_sha,
        machine=None if machine is None else MappingProxyType(machine),
        metrics=MappingProxyType(metrics_summary(metrics)),
        config=MappingProxyType(config),
    )


def read_run_dir(run_dir: Path) -> RunRecord:
    """The record of the run directory ``run_dir`` (made absolute; module docstring).

    Raises:
        RecordError: if it holds neither file, its manifest is invalid, or its
            ``metrics.json`` is not a readable training run (no command or config, a seed
            that is not the config's, a config format without a reader, a config the
            reader cannot lower, a malformed ``protocol``).
    """
    run_dir = run_dir.resolve()
    if (run_dir / MANIFEST_FILE).is_file():
        return _from_manifest(run_dir)
    if (run_dir / METRICS_FILE).is_file():
        return _from_metrics(run_dir)
    raise RecordError(f"{run_dir} has neither {MANIFEST_FILE} nor {METRICS_FILE}")
