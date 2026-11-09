"""Synthetic CD21 layouts and archive corpora for the corpus tests (offline, made here).

Shared by ``tests/test_corpus_index.py`` (the index, the reader, the pilot on an index) and
``tests/test_corpus_fits.py`` (the fits sidecar and the noise view on a corpus): raw-like
stamps (:func:`sky_stamps`), a small CD21-like prepared layout with float32 stamps and set
positions (:func:`write_layout`), and an archive corpus written with the archive's own
``NightWriter`` (:func:`write_archive`; no tarballs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from _protocol_support import standardised

from astro_simclr.archive.layout import NightWriter, check_corpus
from astro_simclr.data.prepared import write_prepared
from astro_simclr.data.schema import (
    FEATURE_NAMES,
    LABEL_NAMES,
    STAMP_SHAPE,
    SplitData,
    StampData,
)

N_CLASSES = len(LABEL_NAMES)
RA_COLUMN, DEC_COLUMN = FEATURE_NAMES.index("ra"), FEATURE_NAMES.index("dec")
ARCSEC = 1 / 3600


def min_max(stamps: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
    """Per-channel min-max of finite stamps, float32 (what the stored stamps are)."""
    low = stamps.min(axis=(-3, -2), keepdims=True)
    high = stamps.max(axis=(-3, -2), keepdims=True)
    return ((stamps - low) / (high - low)).astype(np.float32)


def sky_stamps(rng: np.random.Generator, n: int) -> npt.NDArray[np.float64]:
    """``n`` raw-like stamps: noise, a Gaussian source of random flux and width."""
    yy, xx = np.mgrid[:63, :63] - 31.0
    out = rng.normal(100.0, 5.0, size=(n, *STAMP_SHAPE))
    for i in range(n):
        width = rng.uniform(1.5, 3.0)
        source = np.exp(-(xx**2 + yy**2) / (2 * width**2))
        out[i] += rng.uniform(50, 500) * source[..., None]
    return out


def cd21_split(
    rng: np.random.Generator, per_class: int, positions: dict[int, tuple[float, float]]
) -> SplitData:
    """A labelled split of ``per_class`` stamps per class at random positions (some set)."""
    n = per_class * N_CLASSES
    stamps = min_max(sky_stamps(rng, n))
    features = rng.normal(size=(n, len(FEATURE_NAMES)))
    # Random positions on the sky, far from each other and from the archive's.
    features[:, RA_COLUMN] = standardised("ra", rng.uniform(0, 360, n))
    features[:, DEC_COLUMN] = standardised("dec", rng.uniform(-30, 80, n))
    for row, (ra, dec) in positions.items():
        features[row, RA_COLUMN] = standardised("ra", ra)
        features[row, DEC_COLUMN] = standardised("dec", dec)
    labels = np.repeat(np.arange(N_CLASSES, dtype=np.int64), per_class)
    images = np.round(stamps * 255).astype(np.uint8)
    return SplitData(images, labels, features, images_f32=stamps)


#: Positions of some CD21 rows (degrees): Validation 0 and Test 1 (near RA 0 = 360).
CD21_POSITIONS = {
    "Train": {},
    "Validation": {0: (150.0, 20.0)},
    "Test": {1: (359.9999, 10.0)},
}


def write_layout(dest: Path, *, per_class: int = 4, seed: int = 0) -> Path:
    """A CD21-like prepared layout (float32 stamps, :data:`CD21_POSITIONS`) at ``dest``."""
    rng = np.random.default_rng(seed)
    splits = {name: cd21_split(rng, per_class, CD21_POSITIONS[name]) for name in CD21_POSITIONS}
    source = {"name": f"{dest.name}.pkl", "size_bytes": 0, "sha256": f"{seed:064d}"}
    return write_prepared(StampData("classifier", splits), dest, source)


def archive_row(
    oid: str, candid: int, jd: float, ra: float, dec: float, night: int
) -> tuple[Any, ...]:
    """An archive index row (``NightWriter.add``'s ``row``) with the fields the tests set."""
    return (oid, candid, jd, 1, ra, dec, night, 1, True, 0.5, np.nan, candid)


def oid_name(i: int) -> str:
    """A distinct ZTF-like ``oid`` for ``i``."""
    letters = "".join(chr(ord("a") + (i // 26**k) % 26) for k in range(7))
    return f"ZTF18{letters}"


def write_archive(
    archive: Path, nights: dict[int, list[tuple[tuple[Any, ...], npt.NDArray[np.float32]]]]
) -> Path:
    """An archive corpus at ``archive``: per night, its ``(row, cutout)`` pairs, 2 per file."""
    check_corpus(archive, {"selection": {"test": True}, "exclusions": {"count": 0}})
    for night, rows in nights.items():
        writer = NightWriter(archive, night, rows_per_file=2)
        for row, cutout in rows:
            writer.add(row, cutout)
        writer.commit({"source": {"tarball": f"ztf_public_{night}.tar.gz"}})
    return archive
