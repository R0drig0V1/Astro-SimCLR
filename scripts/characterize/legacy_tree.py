"""Read the retired legacy research code from the ``legacy-final`` git tag (task 1.8).

The legacy code (``utils/``, ``models/``, ``losses/``, the root scripts and ``env.yml``) was
deleted from the working tree; it stays readable at the tag ``legacy-final``, which points
at the last commit of ``main`` that still had it. The characterization generators in this
directory import it from there, never from the checkout, so their oracle does not depend on
what is checked out:

- :func:`show` returns one legacy file as text (``git show legacy-final:<path>``);
- :func:`put_on_path` copies whole legacy packages into a temporary directory and puts that
  directory first on ``sys.path``, so ``from utils.lars import LARS`` imports the tagged code;
- :func:`commit` is the full SHA the tag points at, which the generators record in the
  fixtures (``legacy_sha`` / ``legacy_commit``).

``LEGACY_REF`` is the only place that names the tag. The environment variable
``ASTRO_SIMCLR_LEGACY_REF`` overrides it (any commit-ish), for checking a generator against
another commit, or before the tag has been fetched.

To read a legacy file by hand: ``git show legacy-final:utils/training.py``. If the tag is
missing locally, ``git fetch origin tag legacy-final``.

Excluded from ruff, mypy and pytest (``scripts/characterize/``).
"""

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: The git tag that keeps the retired legacy code (created on main by the integrator).
LEGACY_REF = "legacy-final"


def ref():
    """The legacy commit-ish: ``ASTRO_SIMCLR_LEGACY_REF`` if set, else ``LEGACY_REF``."""
    return os.environ.get("ASTRO_SIMCLR_LEGACY_REF", LEGACY_REF)


def _git(*args, text=True):
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=text, check=True
        ).stdout
    except subprocess.CalledProcessError as err:
        stderr = err.stderr if text else err.stderr.decode(errors="replace")
        sys.exit(
            f"git {' '.join(args)} failed: {stderr.strip()}\n"
            f"Is the tag {ref()!r} present? Try: git fetch origin tag {LEGACY_REF}"
        )


def commit():
    """Full SHA of the commit the legacy ref points at (exits if the ref is missing)."""
    return _git("rev-parse", "--verify", f"{ref()}^{{commit}}").strip()


def show(path):
    """Text of the legacy file ``path`` (relative to the repository root) at the ref."""
    return _git("show", f"{ref()}:{path}")


def put_on_path(*packages):
    """Copy the legacy ``packages`` (directories such as ``"utils"``) into a temporary
    directory, put it first on ``sys.path`` and return the legacy commit.

    The directory is removed at interpreter exit. Exits if a package is not in the tree
    at the ref.
    """
    sha = commit()
    tmp = Path(tempfile.mkdtemp(prefix="legacy-final-"))
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    for package in packages:
        files = _git("ls-tree", "-r", "--name-only", sha, "--", package).split()
        if not files:
            sys.exit(f"{package}/ is not in the tree at {ref()} ({sha[:7]})")
        for name in files:
            dest = tmp / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_git("show", f"{sha}:{name}", text=False))
    sys.path.insert(0, str(tmp))
    return sha
