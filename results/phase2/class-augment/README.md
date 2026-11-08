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
| `C6_synthetic.png` | 8 synthetic hot-pixel Bogus (the source class and its `a` in the title) next to a real positive-alert Bogus and a real Asteroid |
| `report.json` | per class: the share excluded by `R^2 < 0.5` and the share usable as a C6 source; the synthetic rows' source classes; the share of templates with a source at the centre, synthetic and real; the difference's centre z, synthetic and real; the PSF FWHM fits; the widest spike's FWHM |

Every panel is stretched from its 1st percentile to its maximum, square root (a 1-99.5 %
stretch would hide a one-pixel spike among the brightest noise pixels).

Measurements (2026-11-08):

- **Excluded by R^2 < 0.5:** AGN 10.6 %, SN 6.9 %, VS 13.8 %, Asteroid 4.8 %, Bogus 15.2 %.
- **Usable as a C6 source** (positive alerts of Asteroid and Bogus only): Asteroid 92 %,
  Bogus 57 %; the synthetic rows come 62 % from Asteroids and 38 % from Bogus.
- **Templates with a source at the centre** (`template_centre_z` > 5): synthetic 6.1 %,
  real positive-alert Bogus 29.5 % (check: at most + 0.05, passed); real per class AGN
  99 %, SN 61 %, VS 98 %, Asteroid 2 %, Bogus 33 %.
- **Centre z of the difference** (against the border median and MAD): synthetic median
  +29.1, minimum +8.3 (1,000 rows); real medians +5.4 to +43.6 for every class and sign of
  `a` (an alert is always a central peak).
- **Smallest PSF FWHM** (differences, fits with S/N > 15): 1.19 px (SN; Asteroids 1.21 px).
  `class_augment.MIN_PSF_FWHM_PX` is 1.19. The widest synthetic spike is 0.48 px.
