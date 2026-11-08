"""The run id, ``NUMERICS_EPOCH`` and the provenance helpers (task A0; ADR 0012).

**Run id** = SHA-256 of the canonical JSON of :class:`RunIdentity`:

- ``config_sha256``: the config hash (:mod:`astro_simclr.runs.canonical`), which already
  holds the training subcommand;
- ``seed``;
- ``split``: the fold (an int), ``"screen"``, or ``None`` for a run on the file's own
  Train / Validation / Test;
- ``data_sha256``: :func:`prepared_data_sha256` of the prepared layout;
- ``splits_sha256``: the SHA-256 of the stored splits file, ``None`` without a split;
- ``pretrained_sha256``: the SHA-256 of the pretrained checkpoint (``linear-eval``,
  ``fine-tune``), ``None`` without one. Not in the plan's list: without it two
  evaluations of different encoders would share a run id;
- ``machine_class``: a slug naming a class of machines that compute alike
  (:func:`default_machine_class`: ``cuda-nvidia-geforce-gtx-1050``, ``cpu-arm64``, ...);
- ``engine``: ``lightning`` or ``plain`` (:data:`ENGINES`). Not in the plan's list either:
  a plain-engine run must never pass for an attempt of a Lightning run (plan A1: a
  new-engine variant is compared only with a baseline rerun on the new engine);
- ``numerics_epoch``: :data:`NUMERICS_EPOCH`.

The git SHA is recorded in the manifest, not hashed: a commit that changes no numerics must
not make every run new. Identical-identity runs are allowed as new **attempts** (the
confirmatory rerun of protocol 2.3, GPU non-determinism); the attempt counter is recorded
next to the run id and names the run directory (``runs.layout.run_dir_name``), never the
id.

**When to bump** :data:`NUMERICS_EPOCH`: in any PR that changes what an existing config
computes on an existing machine class, so that its runs never pass for attempts of the
older ones. For example: a fix of an augmentation (the C7 rotation-order fix would have
bumped it), a different operation order or random-number consumption, a changed default
precision, a torch / torchvision / CUDA wheel change in ``uv.lock`` (ADR 0001 pins them),
a data-pipeline change that moves a pixel. Not for: a new flag whose default reproduces
the old behaviour, a bit-exact refactor, logging or documentation. The pull request
template asks every PR.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, get_args

from astro_simclr.config import Settings
from astro_simclr.data.files import sha256_file
from astro_simclr.data.prepared import read_meta
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.runs.canonical import canonical_json, config_sha256, sha256_text

__all__ = [
    "ENGINES",
    "NUMERICS_EPOCH",
    "RUN_ID_VERSION",
    "SCREEN",
    "Engine",
    "GitState",
    "IdentityError",
    "RunIdentity",
    "SplitPart",
    "check_sha256",
    "data_sha256",
    "default_machine_class",
    "git_state",
    "prepared_data_sha256",
    "run_identity",
    "split_part",
    "uv_lock_sha256",
]

#: The numerics generation of the package: bumped by any PR that changes what an existing
#: config computes (module docstring, "When to bump"). Part of every run id.
NUMERICS_EPOCH: Final = 1

#: Version of the run-id document; bumped only if the fields of :class:`RunIdentity` or
#: their encoding change (every run id changes then).
RUN_ID_VERSION: Final = 1

#: :attr:`RunIdentity.split` of a screen run (the 2.5 screen of ``--splits FILE --screen``).
SCREEN: Final = "screen"

#: A fold number, :data:`SCREEN`, or ``None`` (no split run).
type SplitPart = int | Literal["screen"] | None

#: The training engine of a run: today's Lightning loop, or the plain loop of phase 4.
type Engine = Literal["lightning", "plain"]
ENGINES: Final[tuple[Engine, ...]] = get_args(Engine.__value__)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MACHINE_CLASS = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


class IdentityError(ValueError):
    """A run identity is invalid: a bad hash, seed, split, machine class or epoch."""


def check_sha256(value: object, name: str) -> str:
    """``value`` if it is a lowercase hex SHA-256 digest.

    Raises:
        IdentityError: otherwise.
    """
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise IdentityError(f"{name} must be a lowercase hex SHA-256, got {value!r}")
    return value


def _check_int(value: object, name: str, minimum: int) -> int:
    # bool is a subclass of int: `seed: true` is a typo, not 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise IdentityError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RunIdentity:
    """What a run id is the hash of (module docstring). Validated on construction.

    Raises:
        IdentityError: on a malformed field, a ``split`` without ``splits_sha256`` (or the
            reverse), or a machine class that is not a slug.
    """

    config_sha256: str
    seed: int
    split: SplitPart
    data_sha256: str
    splits_sha256: str | None
    pretrained_sha256: str | None
    machine_class: str
    engine: Engine
    numerics_epoch: int = NUMERICS_EPOCH

    def __post_init__(self) -> None:
        check_sha256(self.config_sha256, "config_sha256")
        check_sha256(self.data_sha256, "data_sha256")
        _check_int(self.seed, "seed", 0)
        _check_int(self.numerics_epoch, "numerics_epoch", 1)
        if self.split is not None and self.split != SCREEN:
            _check_int(self.split, "split (a fold)", 0)
        if (self.split is None) != (self.splits_sha256 is None):
            msg = (
                "split and splits_sha256 go together: a split run needs the splits file's "
                f"SHA-256 and a normal run has neither (split {self.split!r}, splits_sha256 "
                f"{self.splits_sha256!r})"
            )
            raise IdentityError(msg)
        if self.splits_sha256 is not None:
            check_sha256(self.splits_sha256, "splits_sha256")
        if self.pretrained_sha256 is not None:
            check_sha256(self.pretrained_sha256, "pretrained_sha256")
        if not isinstance(self.machine_class, str) or not _MACHINE_CLASS.fullmatch(
            self.machine_class
        ):
            msg = (
                "machine_class must be a slug (lowercase letters, digits, '.', '_', '-'; "
                f"at most 64), got {self.machine_class!r}"
            )
            raise IdentityError(msg)
        if self.engine not in ENGINES:
            raise IdentityError(f"engine {self.engine!r} is not one of: {', '.join(ENGINES)}")

    def to_plain(self) -> dict[str, object]:
        """The identity as plain JSON values (the manifest's ``identity``)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_plain(cls, data: Mapping[str, object]) -> RunIdentity:
        """The identity recorded as :meth:`to_plain`.

        Raises:
            IdentityError: on a missing or unknown key, or an invalid value.
        """
        fields = {field.name for field in dataclasses.fields(cls)}
        missing, unknown = sorted(fields - set(data)), sorted(set(data) - fields)
        if missing or unknown:
            msg = f"identity: missing keys {missing or 'none'}, unknown {unknown or 'none'}"
            raise IdentityError(msg)
        split = data["split"]
        if split is not None and split != SCREEN:
            split = _check_int(split, "split (a fold)", 0)
        return cls(
            config_sha256=_as_str(data["config_sha256"], "config_sha256"),
            seed=_check_int(data["seed"], "seed", 0),
            split=split,  # None, SCREEN or an int, checked above
            data_sha256=_as_str(data["data_sha256"], "data_sha256"),
            splits_sha256=_as_optional_str(data["splits_sha256"], "splits_sha256"),
            pretrained_sha256=_as_optional_str(data["pretrained_sha256"], "pretrained_sha256"),
            machine_class=_as_str(data["machine_class"], "machine_class"),
            engine=data["engine"],  # type: ignore[arg-type]  # checked by __post_init__
            numerics_epoch=_check_int(data["numerics_epoch"], "numerics_epoch", 1),
        )

    @property
    def run_id(self) -> str:
        """The run id: the hex SHA-256 of the canonical identity document."""
        document = {"run_id_version": RUN_ID_VERSION, **self.to_plain()}
        return sha256_text(canonical_json(document))


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise IdentityError(f"{name} must be a string, got {value!r}")
    return value


def _as_optional_str(value: object, name: str) -> str | None:
    return None if value is None else _as_str(value, name)


def run_identity(
    command: str,
    settings: Settings,
    *,
    data_sha256: str,
    machine_class: str,
    engine: Engine,
    split: SplitPart = None,
    splits_sha256: str | None = None,
    pretrained_sha256: str | None = None,
    numerics_epoch: int = NUMERICS_EPOCH,
) -> RunIdentity:
    """The identity of a run of ``command`` with ``settings`` (seed and config hash from them).

    ``pretrained_sha256`` is required exactly when ``settings.pretrained`` is set: the path
    is not hashed (``canonical.IDENTITY_KEYS``), the checkpoint's content is.

    Raises:
        ConfigError: if ``command`` is not a training subcommand.
        IdentityError: on an invalid field, or ``pretrained_sha256`` without
            ``settings.pretrained`` (or the reverse).
    """
    if (settings.pretrained is None) != (pretrained_sha256 is None):
        msg = (
            "pretrained_sha256 goes with settings.pretrained: "
            f"pretrained {settings.pretrained!r}, pretrained_sha256 {pretrained_sha256!r}"
        )
        raise IdentityError(msg)
    return RunIdentity(
        config_sha256=config_sha256(command, settings),
        seed=settings.seed,
        split=split,
        data_sha256=data_sha256,
        splits_sha256=splits_sha256,
        pretrained_sha256=pretrained_sha256,
        machine_class=machine_class,
        engine=engine,
        numerics_epoch=numerics_epoch,
    )


def split_part(request: SplitRequest | None) -> SplitPart:
    """:attr:`RunIdentity.split` of a split request: its fold, :data:`SCREEN` or ``None``."""
    if request is None:
        return None
    return SCREEN if request.fold is None else request.fold


def prepared_data_sha256(meta: Mapping[str, object]) -> str:
    """The data SHA-256 of a prepared layout, from its ``meta.json`` (already read).

    The hash of the canonical JSON of the content the run reads: the ``kind``, the source
    pickle's SHA-256, and per split its size and every array file's SHA-256 (recorded by
    ``prepare-data``). Not the raw ``meta.json`` bytes: those hold ``created_by`` (the
    package version) and the format version, while a version-1 layout and a uint8-only
    version-2 layout of one pickle have the same arrays byte for byte. A layout with
    ``images_f32.npy`` hashes differently from its uint8-only form: it has more content.

    Raises:
        IdentityError: if the meta lacks the source hash, the splits or a file's hash.
    """
    source = meta.get("source")
    splits = meta.get("splits")
    if not isinstance(source, Mapping) or not isinstance(splits, Mapping):
        raise IdentityError("prepared meta: needs 'source' and 'splits' mappings")
    content: dict[str, object] = {}
    for split, info in splits.items():
        if not isinstance(info, Mapping) or not isinstance(info.get("files"), Mapping):
            raise IdentityError(f"prepared meta: split {split!r} has no 'files' mapping")
        hashes: dict[str, str] = {}
        for name, file in info["files"].items():
            recorded = file.get("sha256") if isinstance(file, Mapping) else None
            hashes[str(name)] = check_sha256(recorded, f"meta splits.{split}.files.{name}.sha256")
        content[str(split)] = {"n": info.get("n"), "files": hashes}
    document = {
        "kind": meta.get("kind"),
        "source_sha256": check_sha256(source.get("sha256"), "meta source.sha256"),
        "splits": content,
    }
    return sha256_text(canonical_json(document))


def data_sha256(path: Path) -> str:
    """:func:`prepared_data_sha256` of the prepared layout at ``path``.

    Reads ``meta.json`` only (the array hashes were computed by ``prepare-data``;
    ``verify_prepared`` checks them against the files).

    Raises:
        DataError: if ``path`` is not a readable prepared layout.
        IdentityError: if its meta lacks a hash.
    """
    return prepared_data_sha256(read_meta(path))


def uv_lock_sha256(root: Path) -> str | None:
    """The hex SHA-256 of ``root/uv.lock``, or ``None`` if there is no such file.

    Raises:
        OSError: if the file exists but cannot be read.
    """
    lock = root / "uv.lock"
    return sha256_file(lock) if lock.is_file() else None


@dataclass(frozen=True, slots=True)
class GitState:
    """The commit a run was made from, and whether the tree had changes.

    Both are ``None`` outside a git checkout (or without a ``git`` binary): an installed
    wheel has no commit to record.
    """

    sha: str | None
    dirty: bool | None


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        # A fixed argv, no shell; `git` from PATH, as every developer tool here.
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(root), *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return None


def git_state(root: Path) -> GitState:
    """The HEAD commit and dirty flag of the git checkout containing ``root``.

    Dirty means ``git status --porcelain`` lists anything: a modified, staged or untracked
    (not ignored) file. Run outputs are ignored (``outputs/`` in ``.gitignore``), so a run
    writing into the checkout does not make the next one dirty.
    """
    head = _git(root, "rev-parse", "--verify", "HEAD")
    if head is None or head.returncode != 0:
        return GitState(None, None)
    status = _git(root, "status", "--porcelain")
    dirty = None if status is None or status.returncode != 0 else bool(status.stdout.strip())
    return GitState(head.stdout.strip(), dirty)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._]+", "-", text.lower()).strip("-") or "unknown"


def default_machine_class(device: str, gpu: str | None, processor: str) -> str:
    """The machine class of a device, when no spec names one.

    CUDA devices are classed by GPU name (``cuda-nvidia-geforce-gtx-1050``): one model
    computes alike across hosts. CPU and MPS are classed by processor architecture
    (``cpu-x86_64``, ``mps-arm64``), since their kernels differ by ISA. The result is cut
    to the 64 characters a slug may have.
    """
    tail = gpu if device == "cuda" else processor
    return f"{_slug(device)}-{_slug(tail or 'unknown')}"[:64]
