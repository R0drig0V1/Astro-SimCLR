"""The runner's side of a submission: config, per-SHA worktrees, ``queue enqueue`` (task A3).

**Runner home** (:class:`RunnerConfig`, ``$ASTRO_SIMCLR_HOME``, default ``~/.astro-simclr``,
on the GPU machine)::

    runner.yaml              repo, data_dir, runs_root, uv, machine_classes (`queue init`)
    queue/queue.sqlite       the queue (queue.store); never copied
    queue/logs/<job>.log     each job's stdout and stderr
    specs/<spec id>.yaml     every accepted spec, the bytes as submitted (synced back)
    worktrees/<sha>/         one detached git worktree per submitted commit, with its .venv
    runs/<spec name>/<run id[:16]>-a<attempt>/   run directories (default runs_root)

The runner's own code (the clone in ``repo``, its ``.venv``) is only the queue: it never
computes a run's settings or hashes. :func:`enqueue` does, in order:

1. parses the spec (its own reading, for ``scratch``, ``extra``, the name and the machine
   class, which must be one this runner serves when ``runner.yaml`` lists them) and checks
   the SHA (40 hex digits);
2. :func:`ensure_worktree`: ``worktrees/<sha>``, created with ``git worktree add --detach``
   from ``repo`` (after ``git fetch origin`` if the commit is not there yet);
3. refuses the worktree if ``git status --porcelain`` lists anything and the spec is not
   ``scratch`` (someone edited it on the GPU machine);
4. ``uv sync --locked [--extra <machine.extra>]`` in the worktree (ADR 0001's extras);
5. writes ``specs/<spec id>.yaml`` and runs the *worktree's* ``astro-simclr spec expand``
   on it (:mod:`~astro_simclr.queue.job`): the cells, run ids and config hashes of that
   commit;
6. adds the spec and one job per cell to the queue (:meth:`QueueStore.add_job`), each with
   the argv ``<worktree>/.venv/bin/astro-simclr spec run-cell <spec file> --cell <key>
   --attempt <n> --run-dir <dir> --data-dir <data_dir> --repo-root <worktree>``.

Every external command is an argv list through a :data:`Commands` callable (no shell), so
the tests replace ``uv`` and keep real git.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import yaml

from astro_simclr.data.files import atomic_write_bytes
from astro_simclr.queue.job import EXPANSION_FORMAT
from astro_simclr.queue.store import Job, NewJob, QueueStore
from astro_simclr.runs.layout import run_dir_name
from astro_simclr.runs.spec import parse_spec

__all__ = [
    "HOME_ENV",
    "RUNNER_FILE",
    "Commands",
    "EnqueueResult",
    "RunnerConfig",
    "RunnerError",
    "default_home",
    "enqueue",
    "ensure_worktree",
    "run_command",
]

#: The environment variable of the home directory (shared with the registry, A2).
HOME_ENV: Final = "ASTRO_SIMCLR_HOME"
#: The runner's config file in the home directory.
RUNNER_FILE: Final = "runner.yaml"
_SHA = re.compile(r"[0-9a-f]{40}")

#: Runs an argv (no shell) in ``cwd`` with ``stdin``; returns the completed process.
type Commands = Callable[..., subprocess.CompletedProcess[bytes]]


class RunnerError(RuntimeError):
    """A submission cannot be accepted (bad SHA, dirty worktree, failed command, ...)."""


def run_command(
    argv: Sequence[str], *, cwd: Path | None = None, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    """The default :data:`Commands`: ``subprocess.run(argv)``, output captured, no shell."""
    return subprocess.run(  # noqa: S603 - an argv list, never a shell string
        list(argv), cwd=cwd, input=stdin, capture_output=True, check=False
    )


def default_home() -> Path:
    """``$ASTRO_SIMCLR_HOME``, or ``~/.astro-simclr``."""
    home = os.environ.get(HOME_ENV)
    return Path(home).expanduser() if home else Path.home() / ".astro-simclr"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunnerConfig:
    """The runner's settings (module docstring), stored in ``<home>/runner.yaml``.

    ``machine_classes`` are the classes this runner's workers serve (``queue init
    --machine-class``); when set, a spec of another class is refused at enqueue time
    instead of waiting forever for a worker. ``entry_point`` is the argv prefix of the
    per-SHA CLI, relative to the worktree's
    ``.venv`` by default; the tests set it to this interpreter's ``-m astro_simclr.cli``.
    """

    home: Path
    repo: Path
    data_dir: Path
    runs_root: Path
    uv: str = "uv"
    entry_point: tuple[str, ...] | None = None
    machine_classes: tuple[str, ...] = ()

    @property
    def queue_file(self) -> Path:
        """``<home>/queue/queue.sqlite``."""
        return self.home / "queue" / "queue.sqlite"

    @property
    def log_dir(self) -> Path:
        """``<home>/queue/logs``."""
        return self.home / "queue" / "logs"

    @property
    def specs_dir(self) -> Path:
        """``<home>/specs``: every accepted spec (shared with the registry, A2)."""
        return self.home / "specs"

    @property
    def worktrees(self) -> Path:
        """``<home>/worktrees``."""
        return self.home / "worktrees"

    def cli(self, worktree: Path) -> list[str]:
        """The argv prefix of ``astro-simclr`` of the commit checked out at ``worktree``."""
        if self.entry_point is not None:
            return list(self.entry_point)
        return [str(worktree / ".venv" / "bin" / "astro-simclr")]

    def to_plain(self) -> dict[str, object]:
        """The YAML document of ``runner.yaml``."""
        return {
            "repo": str(self.repo),
            "data_dir": str(self.data_dir),
            "runs_root": str(self.runs_root),
            "uv": self.uv,
            "entry_point": None if self.entry_point is None else list(self.entry_point),
            "machine_classes": list(self.machine_classes),
        }

    def save(self) -> Path:
        """Write ``<home>/runner.yaml`` and return its path."""
        self.home.mkdir(parents=True, exist_ok=True)
        path = self.home / RUNNER_FILE
        atomic_write_bytes(path, yaml.safe_dump(self.to_plain(), sort_keys=False).encode())
        return path

    @classmethod
    def load(cls, home: Path | None = None) -> RunnerConfig:
        """Read ``<home>/runner.yaml`` (home: :func:`default_home`).

        Raises:
            RunnerError: if the file is missing or malformed (``queue init`` writes it).
        """
        root = default_home() if home is None else home
        path = root / RUNNER_FILE
        try:
            data: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError:
            raise RunnerError(f"{path} not found: run `astro-simclr queue init` first") from None
        except yaml.YAMLError as err:
            raise RunnerError(f"{path}: invalid YAML ({err})") from None
        if not isinstance(data, dict) or not {"repo", "data_dir", "runs_root"} <= set(data):
            raise RunnerError(f"{path} needs repo, data_dir and runs_root")
        entry = data.get("entry_point")
        return cls(
            home=root,
            repo=Path(str(data["repo"])).expanduser(),
            data_dir=Path(str(data["data_dir"])).expanduser(),
            runs_root=Path(str(data["runs_root"])).expanduser(),
            uv=str(data.get("uv") or "uv"),
            entry_point=None if entry is None else tuple(str(arg) for arg in entry),
            machine_classes=tuple(str(name) for name in data.get("machine_classes") or ()),
        )


def _check(result: subprocess.CompletedProcess[bytes], what: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", "replace").strip()
        raise RunnerError(f"{what} failed (exit {result.returncode}): {detail[-2000:]}")
    return (result.stdout or b"").decode("utf-8", "replace")


def ensure_worktree(config: RunnerConfig, sha: str, *, commands: Commands = run_command) -> Path:
    """The detached worktree of commit ``sha`` under ``<home>/worktrees`` (created if absent).

    Raises:
        RunnerError: if ``sha`` is not a full commit id, the commit is not in ``repo`` even
            after ``git fetch origin``, or an existing worktree is at another commit.
    """
    if not _SHA.fullmatch(sha):
        raise RunnerError(f"{sha!r} is not a full 40-digit commit id")
    dest = config.worktrees / sha
    git = ["git", "-C"]
    if dest.exists():
        head = _check(commands([*git, str(dest), "rev-parse", "HEAD"]), "git rev-parse").strip()
        if head != sha:
            raise RunnerError(f"{dest} is at {head}, not {sha}: remove it and resubmit")
        return dest
    have = commands([*git, str(config.repo), "cat-file", "-e", f"{sha}^{{commit}}"])
    if have.returncode != 0:
        _check(commands([*git, str(config.repo), "fetch", "--quiet", "origin"]), "git fetch")
        have = commands([*git, str(config.repo), "cat-file", "-e", f"{sha}^{{commit}}"])
        if have.returncode != 0:
            raise RunnerError(f"commit {sha} is not in {config.repo} nor on origin: push it")
    config.worktrees.mkdir(parents=True, exist_ok=True)
    _check(
        commands([*git, str(config.repo), "worktree", "add", "--detach", str(dest), sha]),
        "git worktree add",
    )
    return dest


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """What :func:`enqueue` did: the spec id, the jobs added and those already queued."""

    spec_id: str
    worktree: Path
    added: tuple[Job, ...]
    existing: tuple[Job, ...]

    def to_plain(self) -> dict[str, object]:
        """The JSON report ``queue enqueue`` prints (and ``queue submit`` relays)."""
        return {
            "spec_id": self.spec_id,
            "worktree": str(self.worktree),
            "added": [job.to_plain() for job in self.added],
            "existing": [job.to_plain() for job in self.existing],
        }


def _expansion(text: str) -> dict[str, object]:
    try:
        data = json.loads(text)
    except ValueError:
        raise RunnerError(f"spec expand printed no JSON: {text[-500:]!r}") from None
    if not isinstance(data, dict) or data.get("format") != EXPANSION_FORMAT:
        raise RunnerError("spec expand printed another format")
    return cast(dict[str, object], data)


def enqueue(
    config: RunnerConfig,
    store: QueueStore,
    spec_text: bytes,
    *,
    sha: str,
    now: float,
    new_attempt: bool = False,
    commands: Commands = run_command,
) -> EnqueueResult:
    """Accept a submitted spec at commit ``sha`` and queue its cells (module docstring).

    Raises:
        SpecError: if the spec is invalid.
        RunnerError: on a bad SHA, a dirty worktree for a non-scratch spec, or a failed
            git, ``uv sync`` or ``spec expand``.
    """
    spec = parse_spec(spec_text.decode("utf-8"))
    if config.machine_classes and spec.machine_class not in config.machine_classes:
        served = ", ".join(config.machine_classes)
        msg = (
            f"spec {spec.name} is for machine class {spec.machine_class}; this runner serves "
            f"{served}: submit it to that machine's host"
        )
        raise RunnerError(msg)
    worktree = ensure_worktree(config, sha, commands=commands)
    status = _check(commands(["git", "-C", str(worktree), "status", "--porcelain"]), "git status")
    if status.strip() and not spec.scratch:
        raise RunnerError(
            f"{worktree} has local changes and spec {spec.name} is not scratch:\n{status}"
        )
    sync = [config.uv, "sync", "--locked"]
    if spec.extra is not None:
        sync += ["--extra", spec.extra]
    _check(commands(sync, cwd=worktree), "uv sync")
    config.specs_dir.mkdir(parents=True, exist_ok=True)
    spec_file = config.specs_dir / f"{spec.spec_id}.yaml"
    if not spec_file.exists():
        atomic_write_bytes(spec_file, spec_text)
    expand = [
        *config.cli(worktree),
        "spec",
        "expand",
        str(spec_file),
        "--data-dir",
        str(config.data_dir),
        "--repo-root",
        str(worktree),
    ]
    expansion = _expansion(_check(commands(expand, cwd=worktree), "spec expand"))
    spec_id = str(expansion["spec_id"])
    if spec_id != spec.spec_id:
        # The commit's code reads the spec differently from the runner's: trust the commit.
        target = config.specs_dir / f"{spec_id}.yaml"
        if not target.exists():
            atomic_write_bytes(target, spec_text)
        spec_file = target
    store.add_spec(spec_id, spec.name, spec_file, scratch=spec.scratch, now=now)
    added: list[Job] = []
    existing: list[Job] = []
    cells = expansion.get("cells")
    for entry in cells if isinstance(cells, list) else []:
        cell, run_id = str(entry["cell"]), str(entry["run_id"])

        def command_for(
            attempt: int, cell: str = cell, run_id: str = run_id
        ) -> tuple[Path, list[str]]:
            run_dir = config.runs_root / spec.name / run_dir_name(run_id, attempt)
            argv = [
                *config.cli(worktree),
                "spec",
                "run-cell",
                str(spec_file),
                "--cell",
                cell,
                "--attempt",
                str(attempt),
                "--run-dir",
                str(run_dir),
                "--data-dir",
                str(config.data_dir),
                "--repo-root",
                str(worktree),
            ]
            return run_dir, argv

        job, new = store.add_job(
            NewJob(
                spec_id=spec_id,
                cell=cell,
                run_id=run_id,
                config_sha256=str(entry["config_sha256"]),
                git_sha=sha,
                machine_class=spec.machine_class,
                cwd=worktree,
            ),
            command_for,
            now=now,
            new_attempt=new_attempt,
        )
        (added if new else existing).append(job)
    return EnqueueResult(spec_id, worktree, tuple(added), tuple(existing))
