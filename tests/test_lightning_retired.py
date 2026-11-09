"""Task 4.8 retired Lightning: nothing imports it, and the plain runs keep their identity.

- No file of ``src/`` or ``tests/`` imports ``lightning``, ``pytorch_lightning`` or
  ``torchmetrics`` (a grep of the import statements), and the environment no longer
  installs them (``pyproject.toml`` dropped them).
- The run id and the config hash of a plain run do not change with the switch:
  ``tests/fixtures/plain_run/manifest.json`` is a plain run of
  ``configs/runs/carrasco2021.yaml`` recorded on main before task 4.8 (its ``request.json``
  has the overrides). Resolving the same config today gives the same config hash (through
  the nested config and through the flat view), and the identity recorded in the manifest
  gives the same run id, with the engine named explicitly or left to the new default.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from astro_simclr.config_tree import config_sha256 as nested_config_sha256
from astro_simclr.config_tree import resolve_config
from astro_simclr.runs.canonical import config_sha256
from astro_simclr.runs.identity import run_identity
from astro_simclr.runs.manifest import read_manifest

ROOT = Path(__file__).resolve().parents[1]
PLAIN_RUN = Path(__file__).parent / "fixtures" / "plain_run"
#: An import statement of a retired package (``import x``, ``from x import``, ``from x.y``).
_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:lightning|pytorch_lightning|torchmetrics)\b", re.MULTILINE
)


def test_no_file_imports_lightning_or_torchmetrics() -> None:
    files = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py")])
    assert len(files) > 100
    offenders = [str(p.relative_to(ROOT)) for p in files if _IMPORT.search(p.read_text("utf-8"))]
    assert offenders == []


def test_the_import_grep_catches_every_form() -> None:
    for line in (
        "import lightning as L",
        "from lightning.pytorch import Trainer",
        "    import torchmetrics",
        "from pytorch_lightning import x",
    ):
        assert _IMPORT.search(line), line
    for line in ("# the lightning-final tag", "x = 'import lightning'", "import lightningx"):
        assert not _IMPORT.search(line), line


@pytest.mark.parametrize("package", ["lightning", "pytorch_lightning", "torchmetrics"])
def test_the_retired_packages_are_not_installed(package: str) -> None:
    assert importlib.util.find_spec(package) is None


@pytest.mark.parametrize("explicit_engine", [True, False], ids=["engine-plain", "default"])
def test_a_stored_plain_run_keeps_its_config_hash_and_run_id(explicit_engine: bool) -> None:
    manifest = read_manifest(PLAIN_RUN)
    request = json.loads((PLAIN_RUN / "request.json").read_text(encoding="utf-8"))
    overrides = list(request["overrides"])
    if not explicit_engine:  # plain is the default since task 4.8
        overrides.remove("runtime.engine=plain")
    config = resolve_config(ROOT / request["config_file"], overrides, command=None)
    assert config.runtime.engine == manifest.engine == "plain"
    # The config hash, through the nested config and through the flat view (ADR 0015).
    assert nested_config_sha256(config) == manifest.identity.config_sha256
    settings = config.to_settings()
    assert config_sha256(manifest.command, settings) == manifest.identity.config_sha256
    # The run id, recomputed from the run's identity fields.
    identity = run_identity(
        manifest.command,
        settings,
        data_sha256=manifest.identity.data_sha256,
        machine_class=manifest.identity.machine_class,
        engine="plain",
    )
    assert identity == manifest.identity
    assert identity.run_id == manifest.run_id
