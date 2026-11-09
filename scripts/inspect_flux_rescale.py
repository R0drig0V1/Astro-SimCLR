"""Visual and numeric checks of the transient flux rescaling (plan row C7; ADR 0007).

On the Train split of a prepared layout with float32 stamps, through the paper's input path
(``training.tensor_splits.float_crops``), this fits the central PSF of every difference crop
(``training.flux_rescale.fit_central_psf``) and writes to ``OUT``:

- ``C7_<class>.png`` for AGN, SN, VS and Asteroid (C7 never rescales Bogus): ``--rows``
  stamps with a good PSF fit, real (science / template / difference) next to the same stamp
  with ``k = 0.6`` and ``k = 1.4``. A ``k`` below the stamp's alert floor
  (``flux_rescale.k_floor``: ``magpsf' <= diffmaglim``) is raised to the floor, as training
  does, and the panel title shows the ``k`` used and the new ``magpsf``;
- ``report.json`` (also printed): per class, the share of PSF fits that failed (never
  rescaled), the median headroom ``diffmaglim - magpsf`` (mag), the share whose floor is
  above 0.6 and above ``flux_k_max`` 1.5, the median fitted FWHM, and the fit time.

Panels are stretched like ``inspect_class_augment.py`` (1st percentile to maximum, square
root); the model sees the [0, 1] crops. CPU only, about 3 minutes for ``red``::

    uv run python scripts/inspect_flux_rescale.py DATA_DIR/td_ztf_stamp_17_06_20_red-002 OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

from astro_simclr.data.m1_features import M1_FEATURE_NORM_STATS
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.training.class_augment import (
    KEPT_FEATURE_NAMES,
    masked_pixels,
    renormalise,
    stamp_fits,
)
from astro_simclr.training.flux_rescale import (
    fit_central_psf,
    k_floor,
    psf_exclusions,
    rescale_transient,
    rescaled_features,
)
from astro_simclr.training.tensor_splits import float_crops

_CHANNELS = ("sci", "tmpl", "diff")
_KS = (0.6, 1.4)
_K_MAX = 1.5
_MAGPSF = KEPT_FEATURE_NAMES.index("magpsf")


def _stretch(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, 1), image.max()
    high = high if high > low else low + 1e-9
    return np.sqrt(np.clip((image - low) / (high - low), 0, 1))


def _magpsf(features: torch.Tensor) -> np.ndarray:
    mean, std = M1_FEATURE_NORM_STATS["magpsf"]
    return (features[:, _MAGPSF].double() * std + mean).numpy()


def _grid(path: Path, title: str, rows: list[list[tuple[np.ndarray, str]]]) -> None:
    headers = ["real", f"k={_KS[0]}", f"k={_KS[1]}"]
    n_rows, n_cols = len(rows), 9
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.15 * n_cols, 1.25 * n_rows + 0.4))
    axes = np.atleast_2d(axes)
    for r, groups in enumerate(rows):
        for g, (crop, label) in enumerate(groups):
            for c in range(3):
                ax = axes[r, 3 * g + c]
                ax.imshow(_stretch(crop[c]), cmap="gray", origin="lower")
                head = f"{headers[g]} {_CHANNELS[c]}" if r == 0 else _CHANNELS[c]
                ax.set_title(f"{head}\n{label}" if c == 0 else head, fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _class_grid(
    out: Path,
    name: str,
    rows: np.ndarray,
    images: np.ndarray,
    features: np.ndarray,
    fits: np.ndarray,
) -> None:
    crops = torch.from_numpy(images[rows]).double()
    feats = torch.from_numpy(features[rows])
    floor = k_floor(feats.double()).numpy()
    mag = _magpsf(feats)
    masked = masked_pixels(crops)
    grid: list[list[tuple[np.ndarray, str]]] = [
        [(images[i], f"#{i} m={mag[j]:.2f}")] for j, i in enumerate(rows)
    ]
    for k_wanted in _KS:
        k = torch.from_numpy(np.maximum(k_wanted, floor))
        changed = rescale_transient(
            crops,
            torch.from_numpy(fits[rows, 6:10]).double(),
            torch.from_numpy(fits[rows, 0]).double(),
            k,
        )
        built = renormalise(changed, masked).float().numpy()
        new_mag = _magpsf(rescaled_features(feats, k.float()))
        for j in range(len(rows)):
            grid[j].append((built[j], f"k={float(k[j]):.2f} m={new_mag[j]:.2f}"))
    title = f"C7 {name} (Train): real | k={_KS[0]} | k={_KS[1]} (raised to the alert floor)"
    _grid(out / f"C7_{name}.png", title, grid)


def main() -> None:
    """Parse the arguments, fit the PSFs, write the grids and the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="a prepared layout with float32 stamps")
    parser.add_argument("out", type=Path, help="output directory (created)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8, help="stamps per grid")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = float_crops(args.data, names=("Train",))["Train"]
    images, features, labels = train.images, train.features, train.labels
    start = time.perf_counter()
    psf = fit_central_psf(images)
    seconds = time.perf_counter() - start
    fits = np.concatenate([stamp_fits(images), psf], axis=1)
    ok = psf[:, 4] > 0
    floor = k_floor(torch.from_numpy(features).double()).numpy()
    headroom = -2.5 * np.log10(floor)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * psf[:, 3]
    rng = np.random.default_rng(args.seed)
    per_class: dict[str, Any] = {}
    for label, name in enumerate(LABEL_NAMES):
        rows = labels == label
        per_class[name] = {
            "median_headroom_mag": round(float(np.median(headroom[rows])), 2),
            "floor_above_0.6": round(float((floor[rows] > _KS[0]).mean()), 3),
            "floor_above_k_max": round(float((floor[rows] > _K_MAX).mean()), 3),
            "median_fwhm_px": round(float(np.median(fwhm[rows & ok])), 2),
        }
        if name != "Bogus":
            pool = np.flatnonzero(rows & ok & (fits[:, 0] != 0) & (floor <= _K_MAX))
            chosen = np.sort(rng.choice(pool, min(args.rows, pool.size), replace=False))
            _class_grid(args.out, name, chosen, images, features, fits)
    report = {
        "dataset": args.data.name,
        "psf_fit_seconds_per_stamp": round(seconds / len(labels), 5),
        "psf_skipped": {k: round(v, 3) for k, v in psf_exclusions(fits, labels)["skipped"].items()},
        "per_class": per_class,
    }
    text = json.dumps(report, indent=2)
    (args.out / "report.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
