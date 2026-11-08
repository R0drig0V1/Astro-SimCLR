<!-- ADR 0007: how the transient flux rescaling of plan row C7 is built, and why. -->
# 0007: Transient flux rescaling of the central PSF (C7)

- **Status:** accepted (2026-11-08, phase 2, task 2.5-input/training, plan row C7)
- **Implemented by:** `src/astro_simclr/training/flux_rescale.py`,
  `Settings.flux_rescale` / `flux_k_min` / `flux_k_max` (`config.py`), the 11-column
  `TensorSplit.fits` (`training/device_data.py`), `PaperClassifierModule.class_augment`
  (`training/modules.py`), `experiments._with_train_fits`
- **Checked by:** `tests/test_flux_rescale.py`, `tests/test_split_runs.py`
  (`p4-augment-fold`, `test_augmented_resume_is_exact`), the golden fixture (each new flag
  at its default); `scripts/inspect_flux_rescale.py` (grids and measurements in
  `results/phase2/flux-rescale/`)
- **Builds on:** ADR 0006 (the cross-channel fit, the masked min-max, the seeding)

## Context

Real SN, AGN, VS and Asteroids are seen at many brightnesses, but each Train stamp shows its
transient at one, and SN has only 1,396 Train stamps (the tiling repeats each about 10
times). The plan's row C7 asks for training stamps whose transient is brighter or fainter.
The input is the 21x21 crop with each channel min-max normalised on its own, plus 23
standardised features, some of which (`magpsf`, `sigmapsf`, `diffmaglim`) describe the
transient's flux.

## Decision

1. **Rescale a model of the central source, never the whole difference.** The per-channel
   min-max is scale-invariant: `k * diff` normalises to exactly the same input, so scaling
   the difference changes nothing. What must change is the transient *relative to the
   noise*. A pixel-integrated circular Gaussian plus a constant is fitted to the central
   11x11 pixels of the normalised difference crop (Levenberg-Marquardt per stamp, float64,
   masked pixels with weight 0); the difference gains `(k - 1) * PSF` (the Gaussian without
   its constant) and the science `(k - 1) * PSF / a`, with `a` the science coefficient of
   the stamp's cross-channel fit `diff ~ a*sci + b*tmpl + c` (ADR 0006): the science change
   that produces that difference change, so the relation still holds exactly. For a negative
   alert (`a < 0`, difference stored flipped) a brighter alert is a fainter science, as it
   should be. The template, the noise, the host and the neighbours are untouched; then the
   per-channel min-max with the masked pixels kept out (`class_augment.renormalise`).
2. **The fit is precomputed once per Train stamp**, on the CPU, next to `stamp_fits` (the
   11-column `AUGMENT_COLUMNS` table in `TensorSplit.fits`, the fourth element of the
   training batches), on the run's own Train role (a fold's training part in a split run).
   About 3.6 ms per stamp, so about 3 minutes for `red` at the start of a C7 run.
3. **Only good fits are rescaled**: finite, positive flux, centroid within 2 px of the
   centre per axis, FWHM in [1, 6] px, S/N (the Gaussian's peak over the residual RMS) at
   least 5. Skipped on `red` Train: AGN 10.9 %, SN 3.9 %, VS 9.9 %, Asteroid 4.9 % (and
   Bogus 30.6 %, which C7 never rescales anyway), reported per class in
   `metrics.json["paper_data"]["flux_rescale"]`. The median fitted FWHM is 2.0-2.2 px for
   the four classes, consistent with the PSF measurement of ADR 0006.
4. **The features follow the image**, de-standardised with M1's statistics
   (`data.m1_features`), changed, re-standardised: `magpsf' = magpsf - 2.5 log10 k` and
   `sigmapsf' = sigmapsf / k`. The second is an approximation: the magnitude error is
   `1.0857 sigma_flux / flux`, and `sigma_flux` is dominated by the background, which C7
   keeps; the source's own Poisson noise, which grows with the flux, is ignored. It is small
   for faint transients near the limit, where most of the headroom is. The other features
   (`fwhm`, `classtar`, `sharpnr`, ...) are left alone.
5. **The result is still an alert**: `magpsf' <= diffmaglim` (the 5-sigma limit, so
   S/N >= 5), that is `k >= 10^(-0.4 (diffmaglim - magpsf))`, the stamp's floor. `k` is drawn
   log-uniformly (symmetric in magnitudes) in `[max(flux_k_min, floor), flux_k_max]`; a row
   whose floor exceeds `flux_k_max` is left as it is (0.0 % of AGN, SN, VS and Asteroids on
   `red` at 1.5). Median headroom on `red` Train: AGN 0.35 mag, SN 0.96, VS 2.18, Asteroid
   0.92: 74 % of the AGN cannot go down to `k = 0.6`, so AGN are mostly brightened.
6. **Never Bogus**: an artefact has no transient flux to rescale, and its "PSF" fit is often
   an artefact's shape. Bogus rows and the synthetic Bogus of C6 are never rescaled.
7. **Flags and order**: `flux_rescale` (the probability that a row is rescaled; 0 = off),
   `flux_k_min` (0.5) and `flux_k_max` (1.5), listed in both presets and
   `configs/phase2/carrasco2021*.yaml` at their defaults (a C7 run is a recorded
   deviation), rejected for other models, needing the float input path and `context: none`
   (the pooled context would keep the old flux). In `training_step`
   (`PaperClassifierModule.augment_plan` / `apply_plan`): C6, C7, C5's noise, then the rotation of
   `rot90_augment`, then the feature jitter. Generator stream 4 of
   `class_augment.augment_rng` (`[seed, global_step, 4]`), two fixed-size draws per batch
   (which rows, and `k`).
8. **C7 runs before the rotation.** The PSF centroid is fitted once on the unrotated crop
   and may be up to 2 px off centre; adding `(k - 1) * PSF` to an already rotated crop
   would put the extra flux where the transient was, not where it is: a fake companion
   about 2 px away for `k > 1`, a dipole-like residual for `k < 1` (found in the
   integrator's review of PR #22, with `single` + `rot90_augment` + C7 accepted by the
   settings). Rotating the augmented crop instead is exact, and C5 / C6, which are
   rotation-agnostic (the cross-channel fit, the border sigmas and the centre spike are
   invariant), move with it. A run with `rot90_augment` and C5 / C6 now draws the same
   augmentations but applies the rotation after them, so it differs from the same run made
   before this change; runs without `rot90_augment` are unchanged.
9. **Prefetch** (ADR 0006, item 10): C7's host part (`flux_rescale.plan_flux`: the draws,
   the alert floors and the rescaled crops, computed in float64 on the host as before) is
   planned by the training loader ahead of the step; the step only assigns the built rows
   and updates `magpsf` / `sigmapsf` on the device (`apply_flux`). Bit for bit the in-step
   result.

## Consequences

- A C7 run is `train-supervised --preset carrasco2021 --set flux_rescale=0.5 [...]`.
- The fit table makes the training batches' fourth element 11 columns wide when C7 is on;
  C5 and C6 read only the first 6.
- `scripts/inspect_class_augment.py` keeps its own copy of the Levenberg-Marquardt fit (it
  fits the stored 63x63 stamps' science and difference to measure the PSF minimum); the
  training fit lives in `flux_rescale`.
- Later idea, not built: rescaling with a per-stamp PSF from the science's stars, or
  injecting a fake SN on a real host (the surveys' standard technique).
