"""The versioned ``manifest.json`` of a run directory: schema, writer, validating reader.

Task A0 (ADR 0013) freezes the schema; A1 makes every run write it. Format
``astro-simclr-run-manifest``, version 1 (:data:`FORMAT`, :data:`FORMAT_VERSION`)::

    format, format_version     this schema
    run_id                     RunIdentity.run_id (runs.identity)
    attempt                    1, 2, ...: identical-identity reruns are new attempts
    command                    the training subcommand
    config                     the resolved settings as run (settings_to_plain), every key
    config_sha256              the config hash (runs.canonical), = identity.config_sha256
    canonical_version          the canonical form the hash was made with
    identity                   RunIdentity.to_plain(): seed, split, data / splits /
                               pretrained SHA-256, machine class, engine, numerics epoch
    git                        {"sha", "dirty"}: recorded, not hashed (null outside git)
    uv_lock_sha256             SHA-256 of uv.lock (null if absent)
    machine                    {"device", "gpu", "torch", "cuda", "fp32_precision", ...}:
                               metrics.json["machine"]'s record (ADR 0010); extra keys
                               (cudnn, platform, ...) kept as written. Null until started
    engine                     "lightning" or "plain" (= identity.engine)
    seeds                      named random streams -> seed; "seed" is identity.seed
    concurrent_jobs            jobs sharing the device while it ran (1: alone)
    status                     queued | running | finished | failed | cancelled
    timestamps                 {"created_at", "started_at", "ended_at"}: ISO 8601 with a
                               UTC offset, null until reached

**Statuses.** :func:`transition` allows ``queued -> running | cancelled`` and
``running -> finished | failed | cancelled | queued`` (``queued`` again: an expired lease,
the attempt resumes from ``last``). ``finished``, ``failed`` and ``cancelled`` are
terminal, and only ``finished`` is finished (:func:`is_finished`): a failed or cancelled
attempt never stands for a result.

**Reading** (:func:`read_manifest`) checks everything that can be checked from the file
alone:

- the format and version, the exact key set (no key repeated) and every type (``true``
  and ``1.0`` are not the integer 1);
- the consistency of the status and timestamps;
- that ``config.pretrained`` and ``identity.pretrained_sha256`` are both set or both null,
  and that ``engine`` is ``identity.engine``;
- that ``config_sha256`` and ``run_id`` are what ``config`` and ``identity`` hash to.

Every failure, a garbled file included, is a :class:`ManifestError`, so a manifest edited
by hand or written by a buggy writer fails there rather than in a comparison. The config
hash is recomputed from the recorded ``config`` mapping
(``canonical.semantic_config_from_plain(complete=False)``), never through today's
``Settings``, so a manifest stays readable after ``Settings`` changes. A validated
manifest is read-only (its mappings are ``MappingProxyType`` copies).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast, get_args

from astro_simclr.config import FP32_PRECISIONS, ConfigError, Settings, settings_to_plain
from astro_simclr.data.files import atomic_write_bytes
from astro_simclr.runs.canonical import (
    CANONICAL_VERSION,
    CanonicalError,
    canonical_json,
    config_sha256_of,
    semantic_config_from_plain,
)
from astro_simclr.runs.identity import (
    ENGINES,
    Engine,
    GitState,
    IdentityError,
    RunIdentity,
    check_sha256,
)

__all__ = [
    "ENGINES",
    "FORMAT",
    "FORMAT_VERSION",
    "MACHINE_KEYS",
    "MANIFEST_FILE",
    "STATUSES",
    "TERMINAL_STATUSES",
    "Engine",
    "Manifest",
    "ManifestError",
    "Status",
    "is_finished",
    "new_manifest",
    "read_manifest",
    "transition",
    "utc_now",
    "write_manifest",
]

#: ``manifest.json["format"]``.
FORMAT: Final = "astro-simclr-run-manifest"
#: The version the writer records and the reader accepts; bumped on any schema change.
FORMAT_VERSION: Final = 1
#: The manifest's file name in a run directory.
MANIFEST_FILE: Final = "manifest.json"

#: The life cycle of one attempt (module docstring, "Statuses").
type Status = Literal["queued", "running", "finished", "failed", "cancelled"]
STATUSES: Final[tuple[Status, ...]] = get_args(Status.__value__)
#: The statuses an attempt never leaves.
TERMINAL_STATUSES: Final[frozenset[Status]] = frozenset({"finished", "failed", "cancelled"})

_TRANSITIONS: Final[Mapping[Status, frozenset[Status]]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"finished", "failed", "cancelled", "queued"}),
    "finished": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}

#: The keys every ``machine`` record has (``experiments._machine``; ADR 0010).
MACHINE_KEYS: Final = ("device", "gpu", "torch", "cuda", "fp32_precision")

_KEYS: Final = (
    "format",
    "format_version",
    "run_id",
    "attempt",
    "command",
    "config",
    "config_sha256",
    "canonical_version",
    "identity",
    "git",
    "uv_lock_sha256",
    "machine",
    "engine",
    "seeds",
    "concurrent_jobs",
    "status",
    "timestamps",
)
_TIMESTAMP_KEYS: Final = ("created_at", "started_at", "ended_at")
_GIT_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class ManifestError(ValueError):
    """A manifest is invalid: malformed, inconsistent, or its hashes do not match."""


def utc_now() -> str:
    """The current time as the manifest records it (ISO 8601, seconds, ``+00:00``)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ManifestError(f"timestamps.{name} must be an ISO 8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ManifestError(f"timestamps.{name}: {value!r} is not ISO 8601") from None
    if parsed.tzinfo is None:
        raise ManifestError(f"timestamps.{name}: {value!r} has no UTC offset")
    return parsed


def _same_int(value: object, expected: int) -> bool:
    # `true` and `1.0` equal 1 in Python, but they are not the integer the schema means.
    return type(value) is int and value == expected


def _check_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManifestError(f"{name} must be an integer >= 1, got {value!r}")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class Manifest:
    """One attempt's ``manifest.json`` (module docstring). Validated on construction.

    ``config`` is the full resolved settings as plain values (non-semantic keys included);
    ``identity.config_sha256`` must be its config hash.

    Raises:
        ManifestError: on any invalid or inconsistent field (see the module docstring).
    """

    identity: RunIdentity
    attempt: int
    command: str
    config: Mapping[str, object]
    git: GitState
    uv_lock_sha256: str | None
    machine: Mapping[str, object] | None
    engine: Engine
    seeds: Mapping[str, int]
    concurrent_jobs: int
    status: Status
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None

    def __post_init__(self) -> None:
        # Read-only copies: a validated manifest must not be mutated into an inconsistent one.
        for name in ("config", "seeds", "machine"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        _check_count(self.attempt, "attempt")
        _check_count(self.concurrent_jobs, "concurrent_jobs")
        if self.engine not in ENGINES:
            raise ManifestError(f"engine {self.engine!r} is not one of: {', '.join(ENGINES)}")
        if self.engine != self.identity.engine:
            msg = f"engine {self.engine} is not identity.engine {self.identity.engine}"
            raise ManifestError(msg)
        self._check_config()
        self._check_provenance()
        self._check_seeds()
        if self.machine is not None:
            _check_machine(self.machine)
        self._check_status()

    def _check_config(self) -> None:
        try:
            semantic = semantic_config_from_plain(self.command, self.config, complete=False)
            recomputed = config_sha256_of(semantic)
        except (ConfigError, CanonicalError) as err:
            raise ManifestError(f"config: {err}") from None
        if recomputed != self.identity.config_sha256:
            msg = (
                f"config hashes to {recomputed}, not to identity.config_sha256 "
                f"{self.identity.config_sha256}"
            )
            raise ManifestError(msg)
        if "seed" in self.config and not _same_int(self.config["seed"], self.identity.seed):
            msg = f"config seed {self.config['seed']!r} is not identity.seed {self.identity.seed}"
            raise ManifestError(msg)
        # `pretrained` is an identity key (not hashed): the identity must carry its content.
        if (self.config.get("pretrained") is None) != (self.identity.pretrained_sha256 is None):
            msg = (
                "config pretrained and identity.pretrained_sha256 go together: pretrained "
                f"{self.config.get('pretrained')!r}, pretrained_sha256 "
                f"{self.identity.pretrained_sha256!r}"
            )
            raise ManifestError(msg)

    def _check_provenance(self) -> None:
        sha, dirty = self.git.sha, self.git.dirty
        if sha is not None and (not isinstance(sha, str) or not _GIT_SHA.fullmatch(sha)):
            raise ManifestError(f"git.sha must be a hex commit id or null, got {sha!r}")
        if dirty is not None and not isinstance(dirty, bool):
            raise ManifestError(f"git.dirty must be true, false or null, got {dirty!r}")
        if sha is None and dirty is not None:
            raise ManifestError("git.dirty needs git.sha: outside git both are null")
        if self.uv_lock_sha256 is not None:
            try:
                check_sha256(self.uv_lock_sha256, "uv_lock_sha256")
            except IdentityError as err:
                raise ManifestError(str(err)) from None

    def _check_seeds(self) -> None:
        if not isinstance(self.seeds, Mapping) or not _same_int(
            self.seeds.get("seed"), self.identity.seed
        ):
            msg = f"seeds must map 'seed' to identity.seed {self.identity.seed}, got {self.seeds!r}"
            raise ManifestError(msg)
        for name, value in self.seeds.items():
            if not isinstance(name, str) or type(value) is not int or value < 0:
                raise ManifestError(f"seeds: {name!r} -> {value!r} is not a name -> integer >= 0")

    def _check_status(self) -> None:
        if self.status not in STATUSES:
            raise ManifestError(f"status {self.status!r} is not one of: {', '.join(STATUSES)}")
        created = _parse_time(self.created_at, "created_at")
        started = None if self.started_at is None else _parse_time(self.started_at, "started_at")
        ended = None if self.ended_at is None else _parse_time(self.ended_at, "ended_at")
        terminal = self.status in TERMINAL_STATUSES
        if terminal != (ended is not None):
            raise ManifestError(f"status {self.status}: ended_at must be set exactly when ended")
        if self.status in ("running", "finished") and (started is None or self.machine is None):
            raise ManifestError(f"status {self.status} needs started_at and machine")
        if self.machine is not None and started is None:
            raise ManifestError("machine is recorded when the attempt starts: needs started_at")
        times = [time for time in (created, started, ended) if time is not None]
        if times != sorted(times):
            raise ManifestError("timestamps must satisfy created_at <= started_at <= ended_at")

    @property
    def run_id(self) -> str:
        """The run id (:attr:`RunIdentity.run_id`)."""
        return self.identity.run_id

    def to_plain(self) -> dict[str, object]:
        """The manifest as the JSON document of the module docstring."""
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "command": self.command,
            "config": dict(self.config),
            "config_sha256": self.identity.config_sha256,
            "canonical_version": CANONICAL_VERSION,
            "identity": self.identity.to_plain(),
            "git": {"sha": self.git.sha, "dirty": self.git.dirty},
            "uv_lock_sha256": self.uv_lock_sha256,
            "machine": None if self.machine is None else dict(self.machine),
            "engine": self.engine,
            "seeds": dict(self.seeds),
            "concurrent_jobs": self.concurrent_jobs,
            "status": self.status,
            "timestamps": {
                "created_at": self.created_at,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
            },
        }


def _check_machine(machine: Mapping[str, object]) -> None:
    if not isinstance(machine, Mapping):
        raise ManifestError(f"machine must be a mapping, got {machine!r}")
    missing = [key for key in MACHINE_KEYS if key not in machine]
    if missing:
        raise ManifestError(f"machine lacks {', '.join(missing)}")
    for key in ("device", "torch"):
        if not isinstance(machine[key], str) or not machine[key]:
            raise ManifestError(f"machine.{key} must be a non-empty string")
    for key in ("gpu", "cuda"):
        if machine[key] is not None and not isinstance(machine[key], str):
            raise ManifestError(f"machine.{key} must be a string or null")
    if machine["fp32_precision"] not in FP32_PRECISIONS:
        valid = ", ".join(FP32_PRECISIONS)
        raise ManifestError(f"machine.fp32_precision must be one of: {valid}")
    try:
        canonical_json(machine)
    except CanonicalError as err:
        raise ManifestError(f"machine: {err}") from None


def is_finished(status: Status) -> bool:
    """Whether an attempt with ``status`` stands for a result: ``finished`` only."""
    return status == "finished"


def new_manifest(
    command: str,
    settings: Settings,
    identity: RunIdentity,
    *,
    git: GitState,
    uv_lock_sha256: str | None,
    attempt: int = 1,
    concurrent_jobs: int = 1,
    seeds: Mapping[str, int] | None = None,
    created_at: str | None = None,
) -> Manifest:
    """A ``queued`` manifest for a run of ``command`` with ``settings`` and ``identity``.

    The engine is ``identity.engine``. ``seeds`` defaults to ``{"seed": settings.seed}``;
    the plain engine will name more streams. ``created_at`` defaults to :func:`utc_now`.

    Raises:
        ManifestError: if ``identity`` is not the identity of these settings (another
            config hash or seed, or a ``pretrained_sha256`` without ``settings.pretrained``
            or the reverse), or any field is invalid.
    """
    return Manifest(
        identity=identity,
        attempt=attempt,
        command=command,
        config=settings_to_plain(settings),
        git=git,
        uv_lock_sha256=uv_lock_sha256,
        machine=None,
        engine=identity.engine,
        seeds={"seed": settings.seed} if seeds is None else dict(seeds),
        concurrent_jobs=concurrent_jobs,
        status="queued",
        created_at=utc_now() if created_at is None else created_at,
    )


def transition(
    manifest: Manifest,
    status: Status,
    *,
    at: str | None = None,
    machine: Mapping[str, object] | None = None,
) -> Manifest:
    """``manifest`` moved to ``status`` at time ``at`` (default :func:`utc_now`).

    ``running`` records ``machine`` (required) and ``started_at`` (the first start only:
    a requeued attempt that runs again keeps it). A terminal status records ``ended_at``.

    Raises:
        ManifestError: if the move is not allowed (module docstring), ``running`` has no
            ``machine``, or the result is invalid (a time before an earlier one).
    """
    allowed = _TRANSITIONS[manifest.status]
    if status not in allowed:
        valid = ", ".join(sorted(allowed)) or "none: it is terminal"
        raise ManifestError(f"status {manifest.status} cannot become {status} (allowed: {valid})")
    now = utc_now() if at is None else at
    if status == "running":
        if machine is None:
            raise ManifestError("status running needs the machine record")
        started = manifest.started_at or now
        return replace(manifest, status=status, machine=dict(machine), started_at=started)
    if status in TERMINAL_STATUSES:
        return replace(manifest, status=status, ended_at=now)
    return replace(manifest, status=status)


def write_manifest(run_dir: Path, manifest: Manifest) -> Path:
    """Write ``run_dir/manifest.json`` atomically and return its path.

    A reader never sees a half-written manifest (``data.files.atomic_write_bytes``).

    Raises:
        OSError: if the directory is not writable.
    """
    text = json.dumps(manifest.to_plain(), indent=2, allow_nan=False) + "\n"
    path = run_dir / MANIFEST_FILE
    atomic_write_bytes(path, text.encode("utf-8"))
    return path


def read_manifest(path: Path) -> Manifest:
    """Read and validate a manifest: ``path`` is the file or its run directory.

    Raises:
        ManifestError: if the file cannot be read, is not JSON, is of another format or
            version, has missing or unknown keys, or fails any check of :class:`Manifest`,
            or its ``run_id`` / ``config_sha256`` are not what its content hashes to. The
            message starts with the path.
    """
    file = path / MANIFEST_FILE if path.is_dir() else path
    try:
        raw = file.read_bytes()
    except OSError as err:
        raise ManifestError(f"{file}: cannot read the manifest ({err})") from None
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ManifestError) as err:
        # A truncated or garbled file must be a ManifestError: a registry scan skips those.
        raise ManifestError(f"{file}: invalid JSON ({err})") from None
    try:
        return _parse(data)
    except (ManifestError, IdentityError) as err:
        raise ManifestError(f"{file}: {err}") from None


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json.loads``' object hook: a repeated key is an error, never "the last one wins"."""
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ManifestError(f"key {key!r} appears twice")
        out[key] = value
    return out


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data[key]
    if not isinstance(value, Mapping):
        raise ManifestError(f"{key} must be a mapping, got {value!r}")
    return cast(Mapping[str, Any], value)


def _parse(data: object) -> Manifest:
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ManifestError(f"not an {FORMAT} file")
    if not _same_int(data.get("format_version"), FORMAT_VERSION):
        msg = (
            f"format version {data.get('format_version')!r}; this version of astro-simclr "
            f"reads version {FORMAT_VERSION}"
        )
        raise ManifestError(msg)
    missing = [key for key in _KEYS if key not in data]
    unknown = sorted(set(data) - set(_KEYS))
    if missing or unknown:
        raise ManifestError(f"missing keys {missing or 'none'}, unknown {unknown or 'none'}")
    if not _same_int(data["canonical_version"], CANONICAL_VERSION):
        version = data["canonical_version"]
        msg = f"canonical_version {version!r}; this version reads {CANONICAL_VERSION}"
        raise ManifestError(msg)
    git = _mapping(data, "git")
    timestamps = _mapping(data, "timestamps")
    if set(git) != {"sha", "dirty"} or set(timestamps) != set(_TIMESTAMP_KEYS):
        raise ManifestError(
            "git needs exactly sha and dirty; timestamps exactly " + ", ".join(_TIMESTAMP_KEYS)
        )
    if not isinstance(data["command"], str):
        raise ManifestError(f"command must be a string, got {data['command']!r}")
    machine = data["machine"]
    manifest = Manifest(
        identity=RunIdentity.from_plain(_mapping(data, "identity")),
        attempt=data["attempt"],
        command=data["command"],
        config=_mapping(data, "config"),
        git=GitState(git["sha"], git["dirty"]),
        uv_lock_sha256=data["uv_lock_sha256"],
        machine=None if machine is None else _mapping(data, "machine"),
        engine=data["engine"],
        seeds=_mapping(data, "seeds"),
        concurrent_jobs=data["concurrent_jobs"],
        status=data["status"],
        created_at=timestamps["created_at"],
        started_at=timestamps["started_at"],
        ended_at=timestamps["ended_at"],
    )
    for key, expected in (
        ("run_id", manifest.run_id),
        ("config_sha256", manifest.identity.config_sha256),
    ):
        if data[key] != expected:
            raise ManifestError(
                f"{key} {data[key]!r} is not what the content hashes to ({expected})"
            )
    return manifest
