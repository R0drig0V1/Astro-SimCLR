# Class-aware augmentation: sample grids and measurements (plan rows C5 and C6)

Output of `scripts/inspect_class_augment.py` on the Train split of
`td_ztf_stamp_17_06_20_red-002` (float32 stamps, the paper's 21x21 input), seed 0,
`--aug-noise 1.0`, `--min-r2 0.5` (ADR 0006, `docs/architecture.md`, "Class-aware
augmentation"). CPU only; regenerate with:

```bash
uv run python scripts/inspect_class_augment.py DATA_DIR/td_ztf_stamp_17_06_20_red-002 \
    results/phase2/class-augment
```

| File | What it shows |
|---|---|
| `C5_<class>.png` | 8 real crops per class (science / template / difference) next to two C5 noise draws of each |
| `C6_synthetic.png` | 8 synthetic hot-pixel Bogus (the source class and its `a` in the title) next to a real Bogus and a real Asteroid |
| `report.json` | per class: the share excluded by `R^2 < 0.5` and the share usable as a C6 source; the PSF FWHM fits; the widest spike's FWHM |

Every panel is stretched from its 1st percentile to its maximum, square root (a 1-99.5 %
stretch would hide a one-pixel spike among the brightest noise pixels).

Measurements (2026-11-08):

- **Excluded by R^2 < 0.5:** AGN 10.6 %, SN 6.9 %, VS 13.8 %, Asteroid 4.8 %, Bogus 15.2 %.
- **Usable as a C6 source:** AGN 87 %, SN 91 %, VS 68 %, Asteroid 92 %, Bogus 70 %.
- **Smallest PSF FWHM** (differences, fits with S/N > 15): 1.19 px (SN; Asteroids 1.21 px).
  `class_augment.MIN_PSF_FWHM_PX` is 1.19. The widest synthetic spike is 0.48 px.
