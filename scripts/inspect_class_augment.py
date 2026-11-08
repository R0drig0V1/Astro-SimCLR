"""Visual and numeric checks of the class-aware augmentation (plan rows C5 and C6; ADR 0006).

On the Train split of a prepared layout with float32 stamps (``prepare-data --images
float32``), through the paper's input path (``device_data.float_crops``: 21x21 crops,
per-channel min-max, NaN -> 0), this writes to ``OUT``:

- ``C5_<class>.png``: per class, ``--rows`` real crops (science / template / difference)
  next to two C5 noise draws of each (``class_augment.noise_augment`` at ``--aug-noise``,
  among the stamps with ``R^2 >= --min-r2``);
- ``C6_synthetic.png``: ``--rows`` synthetic hot-pixel Bogus (``class_augment.
  synthetic_bogus``, each from a random Train stamp of its batch), next to a real Bogus and
  a real Asteroid;
- ``report.json`` (also printed): per class, the share of stamps excluded by
  ``R^2 < --min-r2`` and the share usable as a C6 source; the PSF FWHM measurement that
  sets ``class_augment.MIN_PSF_FWHM_PX``; the widest spike's FWHM.

**PSF measurement.** A pixel-integrated circular Gaussian plus a constant is fitted
(Levenberg-Marquardt per stamp, float64) to the central 11x11 pixels of the stored 63x63
stamps of every Asteroid and SN, in the science and the difference; fits with S/N > 15
(the Gaussian's peak over the residual RMS) and the centre within 1.5 px are kept. Its
FWHM is what ``class_augment.moment_fwhm`` returns for a pixel-integrated Gaussian, so the
spike and the PSF are measured alike. ``min_fwhm_px`` is the smallest FWHM of the difference
fits: there the static sources are subtracted, while in the science a static neighbour
inside the window can pull the fit onto a noise pixel (on ``red``, the 10 science fits under
1 px, inspected, are such failures).

Every panel is stretched from its own 1st percentile to its maximum, square root, so faint
structure shows and a one-pixel spike stays the brightest pixel; the model sees the [0, 1]
crops. CPU only::

    uv run python scripts/inspect_class_augment.py DATA_DIR/td_ztf_stamp_17_06_20_red-002 OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

from astro_simclr.data.prepared import load_prepared
from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.training.class_augment import (
    DEFAULT_MIN_R2,
    MIN_PSF_FWHM_PX,
    NOISE_STREAM,
    SPIKE_LINE_RATIO,
    SYNTHETIC_STREAM,
    augment_rng,
    fit_exclusions,
    masked_pixels,
    moment_fwhm,
    noise_augment,
    stamp_fits,
    synthetic_bogus,
)
from astro_simclr.training.device_data import float_crops

_CHANNELS = ("sci", "tmpl", "diff")
_WINDOW = 11
_MIN_SNR = 15.0
_FIT_CHUNK = 2048


def _stretch(image: np.ndarray) -> np.ndarray:
    """From the 1st percentile to the maximum, square root (faint structure shows).

    Not the 1-99.5 % of the integrator's grids: there a one-pixel spike saturates like the
    second and third brightest noise pixels and cannot be told from them.
    """
    low, high = np.percentile(image, 1), image.max()
    high = high if high > low else low + 1e-9
    return np.sqrt(np.clip((image - low) / (high - low), 0, 1))


def _show(ax: Any, image: np.ndarray, title: str) -> None:
    ax.imshow(_stretch(image), cmap="gray", origin="lower")
    ax.set_title(title, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])


def _grid(
    path: Path, title: str, rows: list[list[tuple[np.ndarray, str]]], headers: list[str]
) -> None:
    """One figure: ``rows`` of (crop (3, S, S), label) groups, 3 panels per group."""
    n_rows, n_cols = len(rows), 3 * len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.15 * n_cols, 1.25 * n_rows + 0.4))
    axes = np.atleast_2d(axes)
    for r, groups in enumerate(rows):
        for g, (crop, label) in enumerate(groups):
            for k in range(3):
                head = f"{headers[g]} {_CHANNELS[k]}" if r == 0 else _CHANNELS[k]
                _show(axes[r, 3 * g + k], crop[k], f"{head}\n{label}" if k == 0 else head)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _c5_grids(
    out: Path, images: np.ndarray, labels: np.ndarray, fits: np.ndarray, args: argparse.Namespace
) -> None:
    rng = np.random.default_rng(args.seed)
    for label, name in enumerate(LABEL_NAMES):
        pool = np.flatnonzero((labels == label) & (fits[:, 3] >= args.min_r2))
        rows = np.sort(rng.choice(pool, min(args.rows, pool.size), replace=False))
        batch, batch_fits = torch.from_numpy(images[rows]), torch.from_numpy(fits[rows])
        draws = [
            noise_augment(
                batch, batch_fits, args.aug_noise, augment_rng(args.seed, step, NOISE_STREAM)
            )[0].numpy()
            for step in (0, 1)
        ]
        grid = [
            [
                (images[i], f"#{i} R2={fits[i, 3]:.2f}"),
                (draws[0][j], f"aug_noise={args.aug_noise}"),
                (draws[1][j], "another draw"),
            ]
            for j, i in enumerate(rows)
        ]
        title = f"C5 {name} (Train): real | two noise draws; per-panel sqrt stretch p1-max"
        _grid(out / f"C5_{name}.png", title, grid, ["real", "C5", "C5"])


def _c6_grid(
    out: Path,
    images: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    fits: np.ndarray,
    args: argparse.Namespace,
) -> None:
    rng = np.random.default_rng(args.seed + 1)
    bogus, asteroid = (
        np.flatnonzero(labels == LABEL_NAMES.index(c)) for c in ("Bogus", "Asteroid")
    )
    eligible = fits[:, 3] >= args.min_r2
    synthetic: list[tuple[np.ndarray, str]] = []
    step = 0
    while len(synthetic) < args.rows:
        rows = rng.choice(len(labels), 64, replace=False)
        targets = torch.nn.functional.one_hot(torch.from_numpy(labels[rows]), len(LABEL_NAMES))
        built, _, _, replaced = synthetic_bogus(
            torch.from_numpy(images[rows]),
            torch.from_numpy(features[rows]),
            targets.float(),
            torch.from_numpy(fits[rows]),
            0.25,
            augment_rng(args.seed, step, SYNTHETIC_STREAM),
            min_r2=args.min_r2,
        )
        for j in np.flatnonzero(replaced):
            source = rows[j]
            text = f"from {LABEL_NAMES[labels[source]]} #{source} a={fits[source, 0]:+.2f}"
            synthetic.append((built[j].numpy(), text))
        step += 1
    real_bogus = rng.choice(bogus[eligible[bogus]], args.rows, replace=False)
    real_asteroid = rng.choice(asteroid, args.rows, replace=False)
    grid = [
        [synthetic[r], (images[b], f"Bogus #{b}"), (images[a], f"Asteroid #{a}")]
        for r, (b, a) in enumerate(zip(real_bogus, real_asteroid, strict=True))
    ]
    title = "C6: synthetic hot-pixel Bogus | real Bogus | real Asteroid (Train); sqrt p1-max"
    _grid(out / "C6_synthetic.png", title, grid[: args.rows], ["synthetic", "real", "real"])


def _gaussian_model(p: torch.Tensor) -> torch.Tensor:
    """A pixel-integrated circular Gaussian + constant on the W x W window, for one stamp.

    ``p`` is (log flux, x0, y0, log sigma, constant), the centre relative to the window's.
    """
    grid = torch.arange(_WINDOW, dtype=torch.float64) - _WINDOW // 2
    scale = torch.exp(p[3]) * np.sqrt(2)

    def profile(centre: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            torch.erf((grid + 0.5 - centre) / scale) - torch.erf((grid - 0.5 - centre) / scale)
        )

    return torch.exp(p[0]) * torch.outer(profile(p[2]), profile(p[1])) + p[4]


def _gaussian_fwhm(cuts: np.ndarray, iterations: int = 60) -> tuple[np.ndarray, ...]:
    """FWHM, S/N and centre offset of :func:`_gaussian_model` fitted to each of (N, W, W).

    Levenberg-Marquardt, independent per stamp (a batched 5x5 solve per iteration, the
    Jacobians by ``torch.func``), so a fit does not depend on the other stamps.
    """
    x = torch.tensor(cuts, dtype=torch.float64).flatten(1)
    n, half = x.shape[0], _WINDOW // 2
    window = x.view(n, _WINDOW, _WINDOW)
    background = x.median(dim=1).values
    peak = window[:, half - 1 : half + 2, half - 1 : half + 2].amax(dim=(1, 2)) - background
    params = torch.stack(
        [
            torch.log(peak.clamp_min(1e-6) * 2 * np.pi * 0.8),  # flux of a sigma-0.9 peak
            torch.zeros(n, dtype=torch.float64),
            torch.zeros(n, dtype=torch.float64),
            torch.full((n,), np.log(0.9), dtype=torch.float64),
            background,
        ],
        dim=1,
    )

    def residual(p: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return _gaussian_model(p).flatten() - data

    residuals = torch.func.vmap(residual)
    jacobians = torch.func.vmap(torch.func.jacrev(residual))
    damping = torch.full((n,), 1e-3, dtype=torch.float64)
    cost = (residuals(params, x) ** 2).sum(dim=1)
    eye = torch.eye(5, dtype=torch.float64)
    for _ in range(iterations):
        r, jac = residuals(params, x), jacobians(params, x)
        jtj = jac.transpose(1, 2) @ jac
        step = torch.linalg.solve(
            jtj
            + damping[:, None, None] * torch.diag_embed(torch.diagonal(jtj, dim1=1, dim2=2))
            + 1e-12 * eye,
            -(jac.transpose(1, 2) @ r[..., None])[..., 0],
        )
        trial = params + step
        trial_cost = (residuals(trial, x) ** 2).sum(dim=1)
        better = torch.isfinite(trial_cost) & (trial_cost < cost)
        params = torch.where(better[:, None], trial, params)
        cost = torch.where(better, trial_cost, cost)
        damping = torch.where(better, damping / 3, damping * 4).clamp(1e-9, 1e9)
    sigma = torch.exp(params[:, 3])
    rms = (cost / x.shape[1]).sqrt()
    snr = torch.exp(params[:, 0]) / (2 * np.pi * sigma**2) / rms
    offset = params[:, 1:3].abs().amax(dim=1)
    return (2 * np.sqrt(2 * np.log(2)) * sigma).numpy(), snr.numpy(), offset.numpy()


def _psf_report(stamps: Any, labels: np.ndarray) -> dict[str, Any]:
    centre, half = stamps.shape[1] // 2, _WINDOW // 2
    report: dict[str, Any] = {}
    for name in ("Asteroid", "SN"):
        rows = np.flatnonzero(labels == LABEL_NAMES.index(name))
        for channel, tag in ((0, "science"), (2, "difference")):
            widths = []
            for start in range(0, rows.size, _FIT_CHUNK):
                chunk = rows[start : start + _FIT_CHUNK]
                cuts = np.asarray(
                    stamps[chunk, centre - half : centre + half + 1, :, channel][
                        :, :, centre - half : centre + half + 1
                    ]
                )
                fwhm, snr, offset = _gaussian_fwhm(cuts)
                good = (snr > _MIN_SNR) & (offset < 1.5) & (fwhm > 0.3) & (fwhm < 8)
                widths.append(fwhm[good])
            values = np.concatenate(widths)
            quantiles = np.percentile(values, [0, 0.1, 1, 5, 50])
            report[f"{name} {tag}"] = {
                "fits_kept": int(values.size),
                "stamps": int(rows.size),
                "fwhm_px_min_p0.1_p1_p5_median": [round(float(q), 3) for q in quantiles],
            }
    # The difference only: in the science, a static neighbour inside the window can pull the
    # fit onto a noise pixel (the few science fits under 1 px are such failures).
    report["min_fwhm_px"] = min(
        value["fwhm_px_min_p0.1_p1_p5_median"][0]
        for key, value in report.items()
        if key.endswith("difference")
    )
    return report


def main() -> None:
    """Parse the arguments, write the grids and the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", type=Path, help="a prepared layout with float32 stamps")
    parser.add_argument("out", type=Path, help="output directory (created)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8, help="stamps per grid")
    parser.add_argument("--aug-noise", type=float, default=1.0)
    parser.add_argument("--min-r2", type=float, default=DEFAULT_MIN_R2)
    parser.add_argument("--skip-psf", action="store_true", help="skip the PSF fits (slow)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = float_crops(args.data, names=("Train",))["Train"]
    images, features, labels = train.images, train.features, train.labels
    fits = stamp_fits(images)
    _c5_grids(args.out, images, labels, fits, args)
    _c6_grid(args.out, images, features, labels, fits, args)

    a, b = fits[:, 0], fits[:, 1]
    unmasked = ~masked_pixels(torch.from_numpy(images)).flatten(1).any(dim=1).numpy()
    source = (fits[:, 3] >= args.min_r2) & (a * b < 0) & (fits[:, 4] > 0) & unmasked
    worst = np.zeros((21, 21))
    worst[10, 10], worst[10, 11] = 1.0, SPIKE_LINE_RATIO[1]
    report: dict[str, Any] = {
        "dataset": args.data.name,
        "r2_exclusions": fit_exclusions(fits, labels, args.min_r2),
        "c6_source_share": {
            name: round(float(source[labels == k].mean()), 3) for k, name in enumerate(LABEL_NAMES)
        },
        "widest_spike_fwhm_px": round(moment_fwhm(worst), 3),
        "MIN_PSF_FWHM_PX": MIN_PSF_FWHM_PX,
    }
    if not args.skip_psf:
        stored = load_prepared(args.data).splits["Train"].images_f32
        if stored is None:
            raise SystemExit(f"{args.data} has no float32 stamps")
        report["psf"] = _psf_report(stored, labels)
    text = json.dumps(report, indent=2)
    (args.out / "report.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
