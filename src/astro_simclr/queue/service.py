"""Hosting the worker: a systemd user service, or a tmux session found by its exact name.

**systemd** (preferred; WSL2 with ``systemd=true`` in ``/etc/wsl.conf``): :func:`systemd_unit`
writes ``~/.config/systemd/user/astro-simclr-worker.service``:

- ``Restart=always``: a worker killed by anything (``kill -9`` included) is back after
  ``RestartSec``; the jobs it left resume when their leases expire;
- ``KillMode=mixed``: ``systemctl --user stop`` sends SIGTERM to the worker only, which
  stops its jobs and requeues them (:meth:`Worker.shutdown`); whatever is left after
  ``TimeoutStopSec`` is SIGKILLed with the whole cgroup, so no orphan outlives the service;
- with ``loginctl enable-linger <user>`` the user manager, and so the worker, runs without
  a login session and starts at boot.

**tmux** (no systemd): :func:`tmux_has_session_argv` and :func:`tmux_start_argv` name the
session as ``=<name>``. tmux matches a bare ``-t name`` by prefix and then by pattern, so
``has-session -t screen25`` was true while only ``screen25c`` ran, and a queue waited 25
minutes for a session that did not exist (2026-11-08). The ``=`` prefix asks for exactly
that name. A tmux session dies with ``wsl --shutdown`` or a reboot and nothing restarts
it: the survival checklist (``docs/queue.md``) says how to bring it back.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Final

__all__ = [
    "SERVICE_NAME",
    "service_commands",
    "systemd_unit",
    "tmux_has_session_argv",
    "tmux_kill_session_argv",
    "tmux_start_argv",
    "tmux_target",
    "unit_path",
]

#: The systemd user unit's name.
SERVICE_NAME: Final = "astro-simclr-worker.service"
_SESSION = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _systemd_quote(arg: str) -> str:
    """One ``ExecStart`` word: systemd splits on spaces and reads C-style quotes and ``%``."""
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    escaped = escaped.replace("$", "$$")
    return f'"{escaped}"'


def _environment(assignment: str) -> str:
    """One ``Environment=`` value: quoted, ``%`` escaped; ``$`` is literal there."""
    escaped = assignment.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'"{escaped}"'


def systemd_unit(worker_argv: Sequence[str], *, home: Path, working_dir: Path) -> str:
    """The text of the worker's systemd user unit (module docstring).

    ``worker_argv`` is the full ``astro-simclr queue worker ...`` command, each element
    quoted for ``ExecStart``; ``home`` becomes ``ASTRO_SIMCLR_HOME``.
    """
    exec_start = " ".join(_systemd_quote(arg) for arg in worker_argv)
    return "\n".join(
        [
            "[Unit]",
            "Description=astro-simclr queue worker (docs/queue.md)",
            "After=default.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={str(working_dir).replace('%', '%%')}",
            f"Environment={_environment(f'ASTRO_SIMCLR_HOME={home}')}",
            f"ExecStart={exec_start}",
            "Restart=always",
            "RestartSec=10",
            "KillMode=mixed",
            "KillSignal=SIGTERM",
            "TimeoutStopSec=60",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def unit_path(config_home: Path) -> Path:
    """Where the unit goes: ``<XDG config home>/systemd/user/astro-simclr-worker.service``."""
    return config_home / "systemd" / "user" / SERVICE_NAME


def service_commands(user: str) -> list[list[str]]:
    """The argvs that enable the unit and keep it running without a login (lingering)."""
    return [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", SERVICE_NAME],
        ["loginctl", "enable-linger", user],
    ]


def tmux_target(session: str) -> str:
    """``=<session>``: tmux's exact-name target (module docstring).

    Raises:
        ValueError: if ``session`` is not a plain name (letters, digits, ``_``, ``-``).
    """
    if not _SESSION.fullmatch(session):
        raise ValueError(f"tmux session {session!r} must be letters, digits, '_' or '-'")
    return f"={session}"


def _tmux(socket: str | None) -> list[str]:
    return ["tmux"] if socket is None else ["tmux", "-L", socket]


def tmux_has_session_argv(session: str, *, socket: str | None = None) -> list[str]:
    """``tmux has-session -t =<session>``: exit 0 only if that exact session exists.

    ``socket`` selects a tmux server (``-L``; the tests use their own).
    """
    return [*_tmux(socket), "has-session", "-t", tmux_target(session)]


def tmux_kill_session_argv(session: str, *, socket: str | None = None) -> list[str]:
    """``tmux kill-session -t =<session>``: never another session sharing its prefix."""
    return [*_tmux(socket), "kill-session", "-t", tmux_target(session)]


def tmux_start_argv(
    session: str, worker_argv: Sequence[str], *, home: Path, socket: str | None = None
) -> list[str]:
    """``tmux new-session -d -s <session> <shell command>`` running the worker.

    tmux hands the command to the shell, so it is one ``shlex.join`` string, with
    ``ASTRO_SIMCLR_HOME`` set by ``env``.
    """
    tmux_target(session)  # validates the name
    command = shlex.join(["env", f"ASTRO_SIMCLR_HOME={home}", *worker_argv])
    return [*_tmux(socket), "new-session", "-d", "-s", session, command]
