"""Tests of the corpus fits sidecar and of the pilot's noise view on a corpus index.

Offline and CPU only, on synthetic data made here (``_corpus_support``): an archive corpus
of ten raw stamps over two nights (two per cutout file, one with NaN pixels), its index
against a CD21-like layout, and a **mirror** layout whose Train stamps are the index's
stamps normalised as stored, in index order. Covered:

- the fits through the corpus path (``build-corpus-fits``) equal ``stamp_fits`` on the
  mirror layout's Train stamps (the layout path of the pilot) bit for bit, whatever the
  chunking;
- a corpus run with noise trains the same model on ``memmap`` and ``host`` as a layout
  run with noise on the mirror (same batches, views, noise and fits), is deterministic for
  a seed, differs from the run without noise, and resumes exactly;
- the sidecar's round trip, its binding to the index's SHA-256 and the pilot's refusals
  (no fits, fits of another index or rebuilt fits on resume, fits without noise or on a
  layout); the CLI.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _corpus_support import (
    N_CLASSES,
    archive_row,
    cd21_split,
    oid_name,
    sky_stamps,
    write_archive,
    write_layout,
)

from astro_simclr.archive.corpus_fits import (
    FITS_FORMAT,
    CorpusFits,
    compute_corpus_fits,
    read_corpus_fits,
    verify_corpus_fits,
    write_corpus_fits,
)
from astro_simclr.archive.corpus_index import (
    build_corpus_index,
    read_corpus_index,
    write_corpus_index,
    write_npz_atomically,
)
from astro_simclr.archive.reader import normalise_stamps
from astro_simclr.cli import main
from astro_simclr.config import ConfigError
from astro_simclr.data.files import sha256_file
from astro_simclr.data.prepared import write_prepared
from astro_simclr.data.schema import DataError, SplitData, StampData
from astro_simclr.simclr.pilot import (
    PilotConfig,
    load_checkpoint,
    load_pilot_data,
    train_pilot,
)
from astro_simclr.stamp_relation import DEFAULT_MIN_R2, FIT_COLUMNS, stamp_fits

N_STAMPS = 2 * N_CLASSES  # the mirror layout's Train: two stamps per class


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The archive, its index and fits, the CD21 layout it excludes, and the mirror layout."""
    root = tmp_path_factory.mktemp("fits")
    cd21 = write_layout(root / "cd21")
    rng = np.random.default_rng(7)
    raw = sky_stamps(rng, N_STAMPS).astype(np.float32)
    raw[2, :4, :5, 0] = np.nan  # NaN pixels: masked in the fits' background sigma
    nights: dict[int, list[Any]] = {20200101: [], 20200102: []}
    for i in range(N_STAMPS):
        night = 20200101 + i % 2
        row = archive_row(oid_name(100 + i), 1000 + i, 2458850.0 + i, 20.0 + 9 * i, -40.0, night)
        nights[night].append((row, raw[i]))
    archive = write_archive(root / "archive", nights)
    index = root / "index.npz"
    write_corpus_index(build_corpus_index(archive, layout=cd21), index)
    rows = read_corpus_index(index).rows
    assert len(rows) == N_STAMPS  # nothing excluded: every stamp is in the corpus
    # The mirror: the index's stamps, in index order, normalised as stored.
    stamps = normalise_stamps(raw[rows["candid"] - 1000])
    split = cd21_split(rng, 2, {})
    train = SplitData(
        np.round(stamps * 255).astype(np.uint8),
        split.labels,
        split.features,
        images_f32=stamps,
    )
    splits = {"Train": train, "Validation": cd21_split(rng, 1, {}), "Test": cd21_split(rng, 1, {})}
    source = {"name": "mirror.pkl", "size_bytes": 0, "sha256": "7" * 64}
    mirror = write_prepared(StampData("classifier", splits), root / "mirror", source)
    fits = root / "index.fits.npz"
    write_corpus_fits(compute_corpus_fits(index), fits)
    return {"root": root, "cd21": cd21, "archive": archive, "index": index, "mirror": mirror}


def _pilot(world: dict[str, Path], **changes: Any) -> PilotConfig:
    base = PilotConfig(
        data=str(world["index"]),
        epochs=None,
        steps=4,
        batch_size=2,
        warmup_epochs=0.2,
        noise_scale=3.0,
        corpus_fits=str(world["root"] / "index.fits.npz"),
        seed=3,
        device="cpu",
        tier="memmap",
        log_every=1,
        checkpoint_every=2,
        diagnostics_every=4,
        keep_every_epochs=None,
        diag_samples=4,
        knn_bank_per_class=1,
        embed_batch=8,
    )
    return replace(base, **changes)


def _model(out: Path) -> dict[str, torch.Tensor]:
    return dict(load_checkpoint(out / "ckpt" / "final.pt")["model"])


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_corpus_fits_equal_the_layout_path_bit_for_bit(world: dict[str, Path]) -> None:
    layout = load_pilot_data(_pilot(world, data=str(world["mirror"]), corpus_fits=None))
    expected = stamp_fits(layout.train.numpy())  # what a layout run computes
    for chunk_rows in (4096, 3, 1):
        fits = compute_corpus_fits(world["index"], chunk_rows=chunk_rows)
        assert fits.fits.dtype == np.float32 and fits.fits.shape == (N_STAMPS, len(FIT_COLUMNS))
        np.testing.assert_array_equal(fits.fits.view(np.uint32), expected.view(np.uint32))
    # The noise acts on these stamps: most fits pass the R^2 cut, the sigmas are positive.
    assert (expected[:, 3] >= DEFAULT_MIN_R2).sum() >= N_STAMPS // 2
    assert (expected[:, 4:6] > 0).all()
    stored = read_corpus_fits(world["root"] / "index.fits.npz")
    np.testing.assert_array_equal(stored.fits, expected)
    meta = stored.meta
    assert meta["format"] == FITS_FORMAT and meta["columns"] == list(FIT_COLUMNS)
    assert meta["index"]["sha256"] == sha256_file(world["index"]) == stored.index_sha256
    assert meta["index"]["n"] == meta["n"] == len(stored) == N_STAMPS
    assert meta["archive"]["path"] == str(world["archive"])


def test_corpus_run_with_noise_matches_the_layout_run(
    world: dict[str, Path], tmp_path: Path
) -> None:
    memmap = tmp_path / "memmap"
    train_pilot(_pilot(world), memmap)
    train_pilot(_pilot(world, tier="host"), tmp_path / "host")
    train_pilot(_pilot(world), tmp_path / "again")
    layout = _pilot(world, data=str(world["mirror"]), corpus_fits=None, tier="host")
    train_pilot(layout, tmp_path / "layout")
    model = _model(memmap)
    assert _same(model, _model(tmp_path / "host"))  # the tier is not part of the numbers
    assert _same(model, _model(tmp_path / "again"))  # deterministic for a seed
    assert _same(model, _model(tmp_path / "layout"))  # the same fits, views and noise
    train_pilot(_pilot(world, noise_scale=0.0, corpus_fits=None), tmp_path / "plain")
    assert not _same(model, _model(tmp_path / "plain"))  # the noise acts
    train_pilot(_pilot(world, seed=4), tmp_path / "seed4")
    assert not _same(model, _model(tmp_path / "seed4"))
    record = json.loads((memmap / "pilot.json").read_text())
    fits = record["data"]["pretrain"]["fits"]
    path = world["root"] / "index.fits.npz"
    assert fits["path"] == str(path) and fits["sha256"] == sha256_file(path)
    assert fits["index_sha256"] == record["data"]["pretrain"]["index_sha256"]
    assert record["config"]["corpus_fits"] == str(path)
    assert record["noise_fields"] == "device-dependent"
    plain = json.loads((tmp_path / "plain" / "pilot.json").read_text())
    assert "corpus_fits" not in plain["config"] and "fits" not in plain["data"]["pretrain"]


def test_corpus_run_with_noise_resumes_exactly(world: dict[str, Path], tmp_path: Path) -> None:
    whole, parts = tmp_path / "whole", tmp_path / "parts"
    train_pilot(_pilot(world), whole)
    assert train_pilot(_pilot(world), parts, stop_at=1).steps_done == 1
    assert train_pilot(_pilot(world), parts, resume=True, stop_at=3).steps_done == 3
    assert train_pilot(_pilot(world), parts, resume=True).steps_done == 4
    assert _same(_model(whole), _model(parts))
    for name in ("metrics.jsonl", "diagnostics.jsonl"):
        assert (whole / name).read_text() == (parts / name).read_text()


def test_pilot_refuses_missing_or_foreign_fits(world: dict[str, Path], tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="build-corpus-fits INDEX_FILE"):
        load_pilot_data(_pilot(world, corpus_fits=None))
    with pytest.raises(ConfigError, match="set noise_scale > 0"):
        _pilot(world, noise_scale=0.0)
    with pytest.raises(ConfigError, match="a layout run fits its own"):
        load_pilot_data(_pilot(world, data=str(world["mirror"])))
    plain = load_pilot_data(_pilot(world, noise_scale=0.0, corpus_fits=None))  # no fits read
    with pytest.raises(ConfigError, match="needs its stamp fits"):
        train_pilot(_pilot(world), tmp_path / "given", data=plain)
    # Fits of another index: one built without the near-duplicate check (fewer rules).
    other = tmp_path / "other.npz"
    write_corpus_index(build_corpus_index(world["archive"], layout=world["cd21"], near=None), other)
    with pytest.raises(DataError, match="another index"):
        load_pilot_data(_pilot(world, data=str(other), allow_weak_index=True))
    sidecar = read_corpus_fits(world["root"] / "index.fits.npz")
    with pytest.raises(DataError, match="another index"):
        verify_corpus_fits(sidecar, index_sha256=sidecar.index_sha256, n=3, source=other)


def test_pilot_refuses_to_resume_on_rebuilt_fits(world: dict[str, Path], tmp_path: Path) -> None:
    fits = tmp_path / "fits.npz"
    write_corpus_fits(compute_corpus_fits(world["index"]), fits)
    config = _pilot(world, corpus_fits=str(fits))
    train_pilot(config, tmp_path / "run", stop_at=1)
    rebuilt = compute_corpus_fits(world["index"], chunk_rows=3)  # same values, another record
    write_corpus_fits(rebuilt, fits, overwrite=True)
    with pytest.raises(ConfigError, match="other corpus fits"):
        train_pilot(config, tmp_path / "run", resume=True)


def test_fits_file_round_trip_and_refusals(world: dict[str, Path], tmp_path: Path) -> None:
    fits = compute_corpus_fits(world["index"], archive=world["archive"])
    path = tmp_path / "fits.npz"
    write_corpus_fits(fits, path)
    again = read_corpus_fits(path)
    np.testing.assert_array_equal(again.fits, fits.fits)
    assert again.meta == fits.meta
    with pytest.raises(DataError, match="already exists"):
        write_corpus_fits(fits, path)
    with pytest.raises(ValueError, match="chunk_rows"):
        compute_corpus_fits(world["index"], chunk_rows=0)
    with pytest.raises(DataError, match="not a corpus fits file"):
        read_corpus_fits(world["root"] / "missing.npz")
    with pytest.raises(DataError, match="not a corpus fits file"):
        read_corpus_fits(world["index"])  # an index: no "fits" array
    bad: list[tuple[dict[str, Any], np.ndarray[Any, Any]]] = [
        ({**fits.meta, "format_version": 99}, fits.fits),
        ({**fits.meta, "format": "other"}, fits.fits),
        (fits.meta, fits.fits[:-1]),
        (fits.meta, fits.fits.astype(np.float64)),
        (fits.meta, np.where(np.arange(6) == 0, np.nan, fits.fits).astype(np.float32)),
    ]
    for meta, values in bad:
        write_npz_atomically(path, {"fits": values}, meta, overwrite=True)
        with pytest.raises(DataError):
            read_corpus_fits(path)
    assert len(CorpusFits(fits.fits[:2], fits.meta)) == 2


def test_cli_builds_fits_and_pilots_with_noise(
    world: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "fits.npz"
    build = ("build-corpus-fits", world["index"], "--out", out)
    assert main([str(a) for a in (*build, "--quiet")]) == 0
    printed = capsys.readouterr()
    assert f"{N_STAMPS} stamps" in printed.out and printed.err == ""
    assert main([str(a) for a in build]) == 2  # exists
    assert main([str(a) for a in (*build, "--overwrite", "--chunk-rows", 0)]) == 2
    archive = ("--archive", world["archive"], "--chunk-rows", 4)
    assert main([str(a) for a in (*build, "--overwrite", "--json", *archive)]) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out)["chunk_rows"] == 4 and "fits: 4/10 stamps" in printed.err
    run = tmp_path / "run"
    pilot = ("simclr-pilot", world["index"], "--output-dir", run, "--device", "cpu")
    options = ("--batch-size", 2, "--steps", 2, "--noise-scale", 3)
    assert main([str(a) for a in (*pilot, *options)]) == 2  # no fits
    assert main([str(a) for a in (*pilot, *options, "--corpus-fits", out)]) == 0
    record = json.loads((run / "pilot.json").read_text())
    assert record["config"]["corpus_fits"] == str(out)
    assert record["data"]["pretrain"]["fits"]["sha256"] == sha256_file(out)
