"""Where the registry lives: ``$ASTRO_SIMCLR_HOME`` (default ``~/.astro-simclr``).

The registry is per user and per machine, never per checkout: every worktree of the
multi-agent workflow sees one index of one machine's runs. A registry inside a git worktree
would be committed, copied with the checkout or deleted with it, so :func:`registry_path`
refuses one. The experiment specs the queue accepts (task A3) are kept next to it, in
``specs/`` (:func:`specs_dir`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

__all__ = [
    "HOME_ENV",
    "REGISTRY_FILE",
    "SPECS_DIR",
    "RegistryError",
    "astro_simclr_home",
    "check_outside_worktrees",
    "registry_path",
    "specs_dir",
]

#: The environment variable that moves the per-user state directory.
HOME_ENV: Final = "ASTRO_SIMCLR_HOME"
#: The registry's file name in the state directory.
REGISTRY_FILE: Final = "registry.sqlite"
#: The directory of accepted experiment specs (``<spec_id>.yaml``, written by the queue).
SPECS_DIR: Final = "specs"


class RegistryError(Exception):
    """The registry cannot be used: a bad location, a newer schema, an unknown run, ..."""


def astro_simclr_home(environ: Mapping[str, str] | None = None) -> Path:
    """The per-user state directory: ``$ASTRO_SIMCLR_HOME``, else ``~/.astro-simclr``.

    ``environ`` defaults to ``os.environ``. An empty variable counts as unset.
    """
    env = os.environ if environ is None else environ
    value = env.get(HOME_ENV, "")
    return Path(value).expanduser() if value else Path.home() / ".astro-simclr"


def check_outside_worktrees(path: Path) -> Path:
    """``path`` made absolute, if no directory above it is a git working tree.

    A directory holding ``.git`` (a directory in a clone, a file in a linked worktree) is
    a working tree.

    Raises:
        RegistryError: if ``path`` lies inside one.
    """
    resolved = path.expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            msg = (
                f"{resolved} is inside the git working tree {parent}: the registry must "
                f"live outside every worktree (set {HOME_ENV})"
            )
            raise RegistryError(msg)
    return resolved


def registry_path(explicit: Path | None = None, environ: Mapping[str, str] | None = None) -> Path:
    """The registry file: ``explicit``, else ``astro_simclr_home()/registry.sqlite``.

    Raises:
        RegistryError: if it lies inside a git working tree.
    """
    path = astro_simclr_home(environ) / REGISTRY_FILE if explicit is None else explicit
    return check_outside_worktrees(path)


def specs_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The directory of accepted experiment specs: ``astro_simclr_home()/specs``."""
    return astro_simclr_home(environ) / SPECS_DIR
