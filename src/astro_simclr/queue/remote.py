"""The laptop's side: ``queue submit``, ``queue status --host`` and ``queue sync`` (task A3).

**Transport.** Everything that reaches the runner goes through a :class:`SshTransport`:

- :meth:`SshTransport.run` sends one argv. ssh joins its command arguments with spaces
  and hands the result to the remote login shell, so the argv is sent as one
  ``shlex.join`` string: each element reaches the remote program unchanged, whatever it
  holds. The spec itself travels on stdin, never in the command line;
- :meth:`SshTransport.pull` rsyncs a remote directory into a local one over the same ssh.
  Remote paths are checked against :data:`SAFE_PATH`, since older rsyncs let the remote
  shell parse them.

The tests use the same class with a fake ``ssh`` program that runs the command through
``sh -c`` on the local machine, as sshd would (``tests/fake_ssh.py``), so quoting, stdin,
exit codes and dropped connections are exercised for real.

**Submission** (:func:`submit`):

1. the spec is read and validated locally;
2. a non-``scratch`` spec needs a clean tree (``git status --porcelain`` empty, so the spec
   file is committed too) and a commit that a remote branch contains (the runner fetches
   it from ``origin``). A ``scratch`` spec only warns: its runs record ``dirty`` and are
   never a result;
3. the argv ``<remote astro-simclr> queue enqueue --sha <HEAD> -`` runs on the runner
   with the spec's bytes on stdin; its JSON report comes back.

**Sync** (:func:`sync`): asks the runner where its runs and specs are (``queue paths``),
then

1. rsyncs the run directories, without checkpoints, locks or any SQLite file;
2. for each synced attempt that is ``finished``, rsyncs the checkpoints that the retention
   policy keeps (``runs.layout.retention``: ``best`` of a ``pretrain`` run);
3. rsyncs the accepted specs into the local ``$ASTRO_SIMCLR_HOME/specs`` (the registry
   resolves spec ids there);
4. runs ``astro-simclr runs reindex <dest>`` (the registry, A2) if this version has it.

rsync with ``--partial`` is idempotent: after a dropped connection, run it again.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from astro_simclr.queue.runner import Commands, run_command
from astro_simclr.runs.identity import git_state
from astro_simclr.runs.layout import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    LIGHTNING_BEST,
    LIGHTNING_LAST,
    retention,
)
from astro_simclr.runs.manifest import MANIFEST_FILE, ManifestError, is_finished, read_manifest
from astro_simclr.runs.spec import check_trainable, load_spec

__all__ = [
    "DEFAULT_REMOTE_CLI",
    "DEFAULT_SSH",
    "REINDEX",
    "RUN_EXCLUDES",
    "SAFE_PATH",
    "SshTransport",
    "SubmitError",
    "SyncReport",
    "remote_json",
    "remote_paths",
    "submit",
    "sync",
]

#: The runner's ``astro-simclr``, relative to the remote home (ssh starts there).
DEFAULT_REMOTE_CLI: Final = "Astro-SimCLR/.venv/bin/astro-simclr"
#: ssh options: never prompt, notice a dead connection within a minute.
DEFAULT_SSH: Final = (
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
)
#: What the first rsync pass leaves out: checkpoints, locks, and SQLite files of any kind.
RUN_EXCLUDES: Final = (
    "--exclude=checkpoints/",
    "--exclude=ckpt/",
    "--exclude=.lock",
    "--exclude=*.sqlite",
    "--exclude=*.sqlite-*",
    "--exclude=*.db",
)
#: Remote paths :meth:`SshTransport.pull` accepts (no spaces, quotes or shell syntax).
SAFE_PATH: Final = re.compile(r"[A-Za-z0-9._/~+-]+")
#: The argv of the registry's reindex (task A2), followed by the synced root.
REINDEX: Final = (sys.executable, "-m", "astro_simclr.cli", "runs", "reindex")
_HOST = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._@-]*")
_LIGHTNING: Final = (LIGHTNING_BEST, LIGHTNING_LAST)
_PLAIN: Final = (BEST_CHECKPOINT, LAST_CHECKPOINT)


class SubmitError(RuntimeError):
    """A submission was refused locally, or the runner refused or could not be reached."""


@dataclass(frozen=True, slots=True)
class SshTransport:
    """Runs argvs and rsyncs over ``ssh`` to ``host`` (module docstring).

    Raises:
        SubmitError: if ``host`` is not a plain host name (``user@host`` allowed): it must
            not start with ``-`` or hold shell syntax.
    """

    host: str
    ssh: tuple[str, ...] = DEFAULT_SSH
    rsync: str = "rsync"
    commands: Commands = run_command

    def __post_init__(self) -> None:
        if not _HOST.fullmatch(self.host):
            raise SubmitError(f"host {self.host!r} is not a host name")

    def run(
        self, argv: Sequence[str], *, stdin: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        """Run ``argv`` on the host; ``stdin`` is sent as its standard input."""
        # `--` ends ssh's options; the remote command is one exactly quoted string.
        return self.commands([*self.ssh, "--", self.host, shlex.join(argv)], stdin=stdin)

    def pull(
        self, remote: str, local: Path, filters: Sequence[str] = ()
    ) -> subprocess.CompletedProcess[bytes]:
        """``rsync -a --partial`` the remote directory ``remote`` into ``local``.

        Raises:
            SubmitError: if ``remote`` is not a :data:`SAFE_PATH`.
        """
        if not SAFE_PATH.fullmatch(remote):
            raise SubmitError(f"remote path {remote!r} has characters rsync would expand")
        local.mkdir(parents=True, exist_ok=True)
        argv = [
            self.rsync,
            "-a",
            "--partial",
            "-e",
            shlex.join(self.ssh),
            *filters,
            f"{self.host}:{remote.rstrip('/')}/",
            f"{local}/",
        ]
        return self.commands(argv)


def _decode(data: bytes | None) -> str:
    return (data or b"").decode("utf-8", "replace")


def remote_json(
    transport: SshTransport, argv: Sequence[str], *, stdin: bytes | None = None
) -> object:
    """Run ``argv`` on the runner and parse the JSON it prints.

    Raises:
        SubmitError: if ssh fails (exit 255), the command fails, or it prints no JSON.
    """
    result = transport.run(argv, stdin=stdin)
    if result.returncode == 255:
        raise SubmitError(f"ssh to {transport.host} failed: {_decode(result.stderr).strip()}")
    if result.returncode != 0:
        detail = _decode(result.stderr).strip() or _decode(result.stdout).strip()
        raise SubmitError(f"the runner refused (exit {result.returncode}): {detail}")
    try:
        return json.loads(_decode(result.stdout))
    except ValueError:
        raise SubmitError(
            f"the runner printed no JSON: {_decode(result.stdout)[-500:]!r}"
        ) from None


def _git(root: Path, *args: str, commands: Commands) -> str:
    result = commands(["git", "-C", str(root), *args])
    if result.returncode != 0:
        raise SubmitError(f"git {' '.join(args)} failed: {_decode(result.stderr).strip()}")
    return _decode(result.stdout).strip()


def submit(
    spec_path: Path,
    transport: SshTransport,
    *,
    repo_root: Path,
    remote_cli: Sequence[str] = (DEFAULT_REMOTE_CLI,),
    new_attempt: bool = False,
    commands: Commands = run_command,
) -> object:
    """Submit the spec at ``spec_path`` at the commit of ``repo_root`` (module docstring).

    Returns the runner's JSON report (``queue enqueue``).

    Raises:
        SpecError: if the spec is invalid or its engine is ``lightning`` (task 4.8).
        SubmitError: for a dirty tree or an unpushed commit with a non-scratch spec, a
            spec outside the tree, ssh failing, or the runner refusing.
    """
    spec = load_spec(spec_path)
    check_trainable(spec)
    state = git_state(repo_root)
    if state.sha is None:
        raise SubmitError(f"{repo_root} is not a git checkout")
    if not spec.scratch:
        if state.dirty:
            raise SubmitError(
                f"{repo_root} has uncommitted changes and spec {spec.name} is not scratch: "
                "commit them (the runner checks out the commit, not your tree)"
            )
        tracked = commands(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", str(spec_path.resolve())]
        )
        if tracked.returncode != 0:
            raise SubmitError(f"{spec_path} is not committed in {repo_root}")
        if not _git(repo_root, "branch", "-r", "--contains", state.sha, commands=commands):
            raise SubmitError(f"commit {state.sha[:12]} is on no remote branch: push it first")
    elif state.dirty:
        sys.stderr.write(
            f"astro-simclr: warning: scratch spec {spec.name}: the runner gets commit "
            f"{state.sha[:12]}, not your uncommitted changes\n"
        )
    argv = [*remote_cli, "queue", "enqueue", "--sha", state.sha, "-"]
    if new_attempt:
        argv.append("--new-attempt")
    return remote_json(transport, argv, stdin=spec_path.read_bytes())


def remote_paths(transport: SshTransport, *, remote_cli: Sequence[str]) -> dict[str, str]:
    """The runner's ``runs_root`` and ``specs_dir`` (``queue paths``).

    Raises:
        SubmitError: if ssh fails or the answer is malformed.
    """
    data = remote_json(transport, [*remote_cli, "queue", "paths"])
    if not isinstance(data, dict) or not {"runs_root", "specs_dir"} <= set(data):
        raise SubmitError("queue paths: malformed answer")
    return {key: str(data[key]) for key in ("runs_root", "specs_dir")}


@dataclass(frozen=True, slots=True)
class SyncReport:
    """What :func:`sync` did."""

    runs: int
    checkpoints: tuple[str, ...]
    reindexed: bool | None


def _retained_checkpoints(dest: Path) -> list[str]:
    """Paths (relative to ``dest``) of the checkpoints that finished attempts keep."""
    keep: list[str] = []
    for manifest_file in sorted(dest.glob(f"*/*/{MANIFEST_FILE}")):
        try:
            manifest = read_manifest(manifest_file)
        except ManifestError:
            continue
        if not is_finished(manifest.status):
            continue
        rel = manifest_file.parent.relative_to(dest).as_posix()
        names = _LIGHTNING if manifest.engine == "lightning" else _PLAIN
        decisions = retention(manifest, names)
        keep += [
            f"{rel}/{path}" for path, decision in decisions.items() if decision.action == "keep"
        ]
    return keep


def sync(
    transport: SshTransport,
    dest: Path,
    *,
    specs_dest: Path,
    remote_cli: Sequence[str] = (DEFAULT_REMOTE_CLI,),
    reindex: bool = True,
    commands: Commands = run_command,
) -> SyncReport:
    """Bring the runner's run directories and specs to ``dest`` (module docstring).

    Raises:
        SubmitError: if ssh or rsync fail, or ``runs reindex`` fails.
    """
    paths = remote_paths(transport, remote_cli=remote_cli)
    for remote, local, filters, what in (
        (paths["runs_root"], dest, RUN_EXCLUDES, "run directories"),
        (paths["specs_dir"], specs_dest, ("--include=*.yaml", "--exclude=*"), "specs"),
    ):
        result = transport.pull(remote, local, filters)
        if result.returncode != 0:
            raise SubmitError(f"rsync of the {what} failed: {_decode(result.stderr).strip()}")
    keep = _retained_checkpoints(dest)
    if keep:
        only = ["-m", "--include=*/", *[f"--include=/{path}" for path in keep], "--exclude=*"]
        result = transport.pull(paths["runs_root"], dest, only)
        if result.returncode != 0:
            raise SubmitError(f"rsync of the checkpoints failed: {_decode(result.stderr).strip()}")
    runs = len(list(dest.glob(f"*/*/{MANIFEST_FILE}")))
    reindexed: bool | None = None
    if reindex:
        result = commands([*REINDEX, str(dest)])
        stderr = _decode(result.stderr)
        if result.returncode == 2 and "invalid choice: 'runs'" in stderr:
            # argparse's answer before the registry (task A2) is merged.
            sys.stderr.write("astro-simclr: note: no `runs reindex` in this version; synced only\n")
            reindexed = False
        elif result.returncode != 0:
            raise SubmitError(f"runs reindex failed: {stderr.strip()}")
        else:
            reindexed = True
    return SyncReport(runs, tuple(keep), reindexed)
