# Transient flux rescaling: sample grids and measurements (plan row C7)

Output of `scripts/inspect_flux_rescale.py` on the Train split of
`td_ztf_stamp_17_06_20_red-002` (float32 stamps, the paper's 21x21 input), seed 0 (ADR 0007,
`docs/architecture.md`, "Transient flux rescaling"). CPU only, about 3 minutes; regenerate
with:

```bash
uv run python scripts/inspect_flux_rescale.py DATA_DIR/td_ztf_stamp_17_06_20_red-002 \
    results/phase2/flux-rescale
```

| File | What it shows |
|---|---|
| `C7_<class>.png` | AGN, SN, VS, Asteroid: 8 stamps with a good PSF fit, real next to `k = 0.6` and `k = 1.4` (raised to the stamp's alert floor when needed; titles show the `k` used and `magpsf`) |
| `report.json` | per class: PSF fits skipped, median headroom `diffmaglim - magpsf`, the share whose alert floor is above 0.6 and above 1.5, the median fitted FWHM; the fit time per stamp |

Panels are stretched from the 1st percentile to the maximum, square root.

Measurements (2026-11-08):

- **PSF fits skipped:** AGN 10.9 %, SN 3.9 %, VS 9.9 %, Asteroid 4.9 % (Bogus 30.6 %, never
  rescaled).
- **Median headroom:** AGN 0.35 mag, SN 0.96, VS 2.18, Asteroid 0.92. The alert floor is
  above `k = 0.6` for 74 % of the AGN, 26 % of the SN, 9 % of the VS and 30 % of the
  Asteroids; above 1.5 for none of them.
- **Median fitted FWHM:** 1.98 (AGN), 2.07 (SN), 2.11 (VS), 2.16 (Asteroid) px.
- **Fit time:** 3.6 ms per stamp on the CPU.
