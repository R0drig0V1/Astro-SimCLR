"""Write ``tests/fixtures/lightning_run/``: a small split run trained by the Lightning engine.

Task 4.8 removed Lightning from the package, but the stored Lightning runs (phase 2's
replication, screen and CV) must stay readable: ``ensemble-eval``, ``compare-variants``,
``report-phase2`` and the run registry read their ``manifest.json``, ``metrics.json``,
``predictions_*.npz`` and ``checkpoints/best.ckpt`` (a torch pickle, loaded with
``torch.load(weights_only=True)``, never through Lightning). ``tests/test_lightning_runs.py``
checks that on this fixture.

The run: ``train-supervised`` of the paper classifier on ``carrasco2021``, fold 2 of the
splits of ``tests/_engine_cases.make_split_data``'s data (25 synthetic stamps per split,
seed 1, Train stamp 3 a copy of Validation stamp 7), ``paper_arch: hybrid`` with
non-default shapes (``hybrid_dim`` 8, ``hybrid_depth`` 1, ``hybrid_heads`` 1; the
smallest model, and the B9 case ``ensemble-eval`` once failed to load), 6 steps with a
validation every 3, on the CPU with deterministic kernels.

Run it from a checkout of the ``lightning-final`` tag (the last commit with Lightning)::

    git worktree add /tmp/lf lightning-final && cd /tmp/lf && uv sync --locked
    uv run python scripts/characterize/lightning_run/generate.py OUT_DIR

then copy ``OUT_DIR/lightning_run`` to ``tests/fixtures/``. The run is trained in a
temporary directory with relative paths, and every absolute path the manifest records
(the source checkout) is replaced by ``<repo>``, so the fixture names no machine.
``last.ckpt`` (the optimizer state, 3x the weights) is not kept: only ``best.ckpt`` is read.

Excluded from ruff, mypy and pytest (``scripts/characterize/``).
"""

import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

from astro_simclr.config import Settings
from astro_simclr.data import prepare
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.experiments import run_command
from astro_simclr.protocol.split_runs import SplitRequest
from astro_simclr.protocol.splits import make_splits, write_splits

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {
    "schedule": "carrasco2021",
    "supervised_model": "paper_stamp_classifier",
    "image_size": 21,
    "eval_batch_size": 64,
    "early_stopping": False,
    "monitor": "loss_val",
    "optimizer": "Adam",
    "lr": 3e-2,
    "beta_loss": 0.5,
    "batch_size": 4,
    "max_steps": 6,
    "val_every_n_steps": 3,
    "horizon_increment": 0,
    "checkpoint_every_n_steps": 3,
    "paper_arch": "hybrid",
    "hybrid_dim": 8,
    "hybrid_depth": 1,
    "hybrid_heads": 1,
    "accelerator": "cpu",
    "deterministic": True,
    "progress_bar_refresh_rate": 0,
}
FOLD = 2
KEPT = ("manifest.json", "metrics.json", "predictions_evaluation.npz", "checkpoints/best.ckpt")


def _data(work: Path) -> tuple[Path, Path]:
    """``_engine_cases.make_split_data``, under ``work`` (relative paths)."""
    raw = work / "raw"
    make_synthetic(raw, n_per_split=25, seed=1)
    source = raw / CLASSIFIER_FILES[0]
    content = pickle.loads(source.read_bytes())  # noqa: S301 (written just above)
    for key in ("images", "features"):
        content["Train"][key][3] = content["Validation"][key][7]
    source.write_bytes(pickle.dumps(content))
    prepared = prepare(source, Path("data"), images="float32")
    path = Path("splits.json")
    write_splits(make_splits(prepared, selection_per_class=1, screen_per_class=1), path)
    return prepared, path


def _scrub(path: Path) -> None:
    """Replace the checkout's absolute path (and the home directory) in a JSON file."""
    text = path.read_text(encoding="utf-8")
    text = text.replace(str(ROOT), "<repo>").replace(str(Path.home()), "<home>")
    path.write_text(text, encoding="utf-8")


def main(out: Path) -> None:
    target = out.resolve() / "lightning_run"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        os.chdir(work)
        prepared, splits = _data(work)
        settings = Settings(
            data_dir=prepared.parent, dataset=prepared.name, output_dir=Path("run"), **SETTINGS
        )
        run_command("train-supervised", settings, progress_bar=False,
                    split=SplitRequest(splits, FOLD))
        if target.exists():
            shutil.rmtree(target)
        for name in KEPT:
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(work / "run" / name, target / name)
        shutil.copy(splits, target / "splits.json")
    for name in ("manifest.json", "metrics.json"):
        _scrub(target / name)
    json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    print(f"wrote {target}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
