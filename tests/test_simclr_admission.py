"""Tests of plan 5.2's admission test (``simclr.admission``, ``simclr-view-admission``).

A module fixture trains, on the CPU, a tiny plain ``train-supervised`` run of the paper
classifier on synthetic prepared data (25 stamps per split, float32), the shape of the
replication checkpoint, and stores a splits file of the data (a screen of 2 + 2 stamps per
class). Then: only the screen's selection half is evaluated, and without a view its input
is the run's own on those rows, bit for bit; each view changes the stamps,
deterministically for a seed and differently across repeats; the report's arithmetic
(drop = clean - mean viewed, the verdict against 1 point); the CLI; and the refusals (a
splits file of other data included).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from astro_simclr.cli import main
from astro_simclr.config import ConfigError, Settings
from astro_simclr.data import prepare
from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import DataError
from astro_simclr.data.synthetic import CLASSIFIER_FILES, make_synthetic
from astro_simclr.ensemble_eval import member_net, member_settings, read_member
from astro_simclr.protocol.splits import make_splits, read_splits, write_splits
from astro_simclr.simclr.admission import (
    MAX_DROP,
    AdmissionView,
    model_input,
    view_admission,
    view_stamps,
)
from astro_simclr.simclr.views import STAMP_SIDE, ViewConfig, rotation_bound
from astro_simclr.stamp_relation import stamp_fits
from astro_simclr.tasks.paper import paper_crops
from astro_simclr.tasks.plain import run_plain

_SCHEDULE: dict[str, Any] = {
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
}

VIEWS = {
    "noise": AdmissionView(ViewConfig(noise_scale=3.0, noise_min_r2=-1.0)),
    "scale": AdmissionView(ViewConfig(scale_jitter=0.3)),
    "blur": AdmissionView(ViewConfig(blur_sigma=1.5)),
    "intensity": AdmissionView(ViewConfig(intensity_jitter=0.4)),
    "rotation": AdmissionView(ViewConfig(rotation_degrees=rotation_bound())),
    "dihedral": AdmissionView(dihedral=True),
    "shift": AdmissionView(shift=True),
}


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The prepared data and a finished paper-classifier run directory on it."""
    raw = tmp_path_factory.mktemp("adm_raw")
    make_synthetic(raw, n_per_split=25, seed=1)
    source = raw / CLASSIFIER_FILES[0]
    # Our own synthetic file, written just above: unpickling it is safe.
    content = pickle.loads(source.read_bytes())  # noqa: S301
    for key in ("images", "features"):
        content["Train"][key][3] = content["Validation"][key][7]  # the leaked stamp
    source.write_bytes(pickle.dumps(content))
    data = prepare(source, tmp_path_factory.mktemp("adm_prepared"), images="float32")
    splits = tmp_path_factory.mktemp("adm_splits") / "splits.json"
    write_splits(make_splits(data, selection_per_class=1, screen_per_class=2), splits)
    root = tmp_path_factory.mktemp("adm_runs")
    settings = Settings(
        accelerator="cpu",
        deterministic=True,
        data_dir=data.parent,
        dataset=data.name,
        output_dir=root / "seed0",
        seed=0,
        **_SCHEDULE,
    )
    run_plain("train-supervised", settings)
    return {"data": data, "run": root / "seed0", "splits": splits}


def _selection(run: dict[str, Path]) -> np.ndarray:
    return np.asarray(read_splits(run["splits"]).screen.selection, dtype=np.int64)


def _stamps(data: Path) -> torch.Tensor:
    stored = np.asarray(load_prepared(data).splits["Validation"].images_f32)
    return torch.from_numpy(np.ascontiguousarray(stored.transpose(0, 3, 1, 2)))


def test_without_a_view_the_input_is_the_runs_own_on_the_selection_rows(
    run: dict[str, Path],
) -> None:
    settings = member_settings(read_member(run["run"]))
    rows = _selection(run)
    own = paper_crops(settings, run["data"], ("Validation",))["Validation"].images
    np.testing.assert_array_equal(model_input(_stamps(run["data"])[rows], settings), own[rows])


@pytest.mark.parametrize("name", list(VIEWS))
def test_each_view_changes_the_stamps_deterministically(name: str, run: dict[str, Path]) -> None:
    stamps = _stamps(run["data"])
    fits = torch.from_numpy(stamp_fits(stamps.numpy()))
    view = VIEWS[name]
    first = view_stamps(stamps, view, seed=0, repeat=0, fits=fits)
    assert first.shape == stamps.shape == (stamps.shape[0], 3, STAMP_SIDE, STAMP_SIDE)
    assert torch.isfinite(first).all() and not torch.equal(first, stamps)
    assert torch.equal(first, view_stamps(stamps, view, seed=0, repeat=0, fits=fits))
    assert not torch.equal(first, view_stamps(stamps, view, seed=0, repeat=1, fits=fits))
    assert view.any and view.describe()


def test_noise_needs_the_fits(run: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="fits"):
        view_stamps(_stamps(run["data"]), VIEWS["noise"], seed=0, repeat=0)


def test_the_report_and_its_verdict(run: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "rotation.json"
    view = AdmissionView(ViewConfig(rotation_degrees=4.0, intensity_jitter=0.2))
    report = view_admission([run["run"]], view, splits=run["splits"], out=out, repeats=2, seed=3)
    payload = json.loads(out.read_text())
    assert payload == report.payload
    assert payload["format"] == "astro-simclr-view-admission" and payload["repeats"] == 2
    assert payload["view"] == {"intensity_jitter": 0.2, "rotation_degrees": 4.0}
    rows = _selection(run)
    decision = read_splits(run["splits"]).screen.decision
    # Only the screen's selection half: never a decision row (plan 5.0c).
    assert payload["rows"] == rows.tolist() and not set(payload["rows"]) & set(decision)
    assert payload["n_stamps"] == rows.size == 10 and payload["part"] == "screen selection half"
    assert payload["splits"]["sha256"] and payload["splits"]["path"] == str(run["splits"])
    (row,) = payload["runs"]
    assert len(row["viewed"]) == 2
    mean = (row["viewed"][0]["accuracy"] + row["viewed"][1]["accuracy"]) / 2
    assert row["drop"] == pytest.approx(row["clean"]["accuracy"] - mean)
    assert payload["drop"] == row["drop"]
    assert payload["admitted"] == (payload["drop"] <= MAX_DROP) == report.admitted
    assert payload["verdict"] in ("admitted", "not admitted")
    # The clean accuracy is the run's net on its own input of the selection rows.
    member = read_member(run["run"])
    settings = member_settings(member)
    split = paper_crops(settings, run["data"], ("Validation",))["Validation"]
    with torch.inference_mode():
        logits = member_net(member, settings)(
            torch.from_numpy(split.images[rows]), torch.from_numpy(split.features[rows])
        )
    expected = float((logits.argmax(dim=1).numpy() == split.labels[rows]).mean())
    assert row["clean"]["accuracy"] == pytest.approx(expected)
    with pytest.raises(ConfigError, match="exists"):
        view_admission([run["run"]], view, splits=run["splits"], out=out)
    again = view_admission(
        [run["run"]], view, splits=run["splits"], out=out, repeats=2, seed=3, overwrite=True
    )
    assert again.payload["runs"] == payload["runs"]


def test_cli(run: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli.json"
    code = main(
        [
            "simclr-view-admission",
            str(run["run"]),
            "--out",
            str(out),
            "--data",
            str(run["data"]),
            "--splits",
            str(run["splits"]),
            "--device",
            "cpu",
            "--blur-sigma",
            "1.0",
            "--shift",
        ]
    )
    assert code == 0
    assert "mean drop" in capsys.readouterr().out
    assert json.loads(out.read_text())["view"] == {"blur_sigma": 1.0, "shift": True}
    assert main(["simclr-view-admission", str(run["run"]), "--out", str(tmp_path / "x.json"),
                 "--splits", str(run["splits"]), "--rotation-degrees", "9"]) == 2  # fmt: skip


def test_refusals(run: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    splits = run["splits"]
    with pytest.raises(ConfigError, match="no view"):
        view_admission([run["run"]], AdmissionView(), splits=splits, out=out)
    with pytest.raises(ConfigError, match="repeats"):
        view_admission([run["run"]], VIEWS["blur"], splits=splits, out=out, repeats=0)
    with pytest.raises(ConfigError, match=r"\.json"):
        view_admission([run["run"]], VIEWS["blur"], splits=splits, out=tmp_path / "r.txt")
    with pytest.raises(ConfigError, match="at least one"):
        view_admission([], VIEWS["blur"], splits=splits, out=out)
    other = make_other_data(tmp_path)
    with pytest.raises(ConfigError, match="not the data"):
        view_admission([run["run"]], VIEWS["blur"], splits=splits, out=out, data=other)
    # A splits file made for other data is refused, as the pilot refuses it.
    foreign = tmp_path / "foreign.json"
    write_splits(make_splits(other, selection_per_class=1, screen_per_class=2), foreign)
    with pytest.raises(DataError):
        view_admission([run["run"]], VIEWS["blur"], splits=foreign, out=out)
    assert not out.exists()


def make_other_data(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    make_synthetic(raw, n_per_split=25, seed=2)
    return prepare(raw / CLASSIFIER_FILES[0], tmp_path / "other", images="float32")
