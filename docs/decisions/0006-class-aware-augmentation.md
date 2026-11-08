<!-- ADR 0006: how the class-aware training augmentation of plan rows C5 and C6 is built, and why. -->
# 0006: Class-aware training augmentation with a computed difference (C5, C6)

- **Status:** accepted (2026-11-08, phase 2, task 2.5-input/training, plan rows C5 and C6)
- **Implemented by:** `src/astro_simclr/training/class_augment.py`,
  `Settings.aug_noise` / `aug_feature_jitter` / `synthetic_bogus` / `aug_min_r2`
  (`config.py`), `TensorSplit.fits` (`training/device_data.py`),
  `PaperClassifierModule.training_step` (`training/modules.py`), `experiments._build_paper`
- **Checked by:** `tests/test_class_augment.py`, `tests/test_split_runs.py`
  (`p4-augment-fold`, `test_augmented_resume_is_exact`), the golden fixture
  `tests/fixtures/paper_model/faithful.npz` (each new flag at its default);
  `scripts/inspect_class_augment.py` (grids and measurements in
  `results/phase2/class-augment/`)

## Context

A visual inspection of the Train stamps per class and channel and the replication's
validation confusion matrix (2026-11-08) gave two rows. C5: SN has 1,396 Train stamps
against about 14,500 for AGN and VS, so the tiling repeats each SN about 10 times; noise
should make the copies differ. C6: about half of the inspected Bogus are hot pixels or
cosmic rays, a 1-2 pixel spike at the centre of the science and the difference on an
unremarkable template; synthetic ones can be made from any stamp's template. The model input
is the 21x21 crop with each channel min-max normalised on its own (NaN -> 0), plus 23
standardised features. The faithful preset must stay bit-for-bit what produced the CD21
result, and the user asked whether the difference would be computed or augmented on its own.

## Decision

1. **The difference is computed from the science and the template, never noised on its own.**
   Per stamp, a least-squares fit `diff ~ a*sci + b*tmpl + c` over the crop's 441 pixels,
   in the stored normalised units, explains the difference well (median R^2 0.77-0.95 per
   class on `red` Train), and `sign(a)` agrees with `isdiffpos` for 98-99.6 % of the stamps
   (a negative alert's difference is stored flipped). The per-channel min-max does not
   destroy the cross-channel relation, and the model can use it; independent noise in the
   difference would break it. Any change `dsci`, `dtmpl` moves the difference by
   `a*dsci + b*dtmpl`. No independent term for the difference's own noise: in the raw
   images that noise *is* the science's minus the convolved template's.
2. **The fits are precomputed once, on the CPU in float64, and travel with the Train
   crops** (`TensorSplit.fits`, a fourth element of the training batches), rather than
   being fitted on the device per batch: the R^2 threshold is a discrete decision, and a
   float32 fit on another device could flip it for a borderline stamp. They are computed
   on the run's own Train role, after `protocol_splits`, so a split run uses its fold's
   training part. Stamps below `aug_min_r2` (default 0.5; 5-15 % per class on `red`, the
   most for Bogus and VS) are neither noised nor used as C6 sources, and the share is
   recorded per class in `metrics.json["paper_data"]["class_augment"]`.
3. **C5 noise is scaled to each crop channel's own background sigma** (1.4826 x the MAD of
   the two-pixel border, masked pixels left out), times `u ~ U(0, 1)` per channel times
   `aug_noise`, then the three channels are min-max normalised again without epsilon, NaN
   -> 0, so that training inputs keep the [0, 1], minimum 0, maximum 1 statistics of the
   evaluation inputs. The former-NaN pixels (the exact zeros of a channel with several)
   stay 0 and out of the min/max, as in the float input path.
4. **The feature jitter skips the discrete features**: `isdiffpos` (binary) and the integer
   counts `ndethist`, `ncovhist`, `non_detections` (exact integer lattices in the stored
   values; `docs/data.md`). The clipped `sgscore1-3` / `distpsnr1-3` floors are point
   masses but continuous features; a small jitter moves them slightly, which is accepted.
5. **C6 builds a synthetic Bogus from the replaced row's own stamp with its own relation**:
   the template is unchanged; the science is the static sky the relation implies,
   `-(b/a) * tmpl`, plus Gaussian noise of the source science's background sigma, plus a
   spike at the centre; the difference is the relation applied to them,
   `a * (noise + spike) + c` (the source's transient, which lives in its residual, is
   gone); then the min-max. **Only positive alerts are sources** (`a > 0`, `b < 0`: the
   template enters with a positive scale), with no masked pixel. An alert is a peak at the
   centre of its stored difference, negative alerts included (their difference is stored
   flipped: on `red` Train the median centre z of the real differences, against the border
   median and MAD, is +5.4 to +43.6 for every class and sign), and a hot pixel in the
   science can only trigger a positive alert. A first version also built Bogus from
   `a < 0` sources, which gave the spike negative in the difference: a central dip
   (z about -30) that no real stamp has, in about a third of the synthetic rows (found by
   the integrator's review of PR #20). Putting the spike in the template for `a < 0` was
   rejected: a hot pixel surviving a co-added template is rarer, and real negative Bogus are
   mostly bad subtractions. Now the synthetic differences have a median centre z of +30.7
   (minimum +8.6 over 1,000 rows; +29.1 and +8.3 with the source classes below).

   **Only Asteroid and Bogus rows are sources.** A synthetic Bogus keeps its source's
   template, so the source class decides what lies under the spike. With
   `class_augment.template_centre_z` (the template's central 3x3 mean as z against its
   border; a source at the centre when z > 5), on `red` Train a source sits at the template
   centre in 99 % of the AGN, 98 % of the VS, 61 % of the SN, 2 % of the Asteroids, 33 % of
   the Bogus and 30 % of the positive-alert Bogus. A second version (after the fix above)
   still took AGN and VS sources, so 49 % of its synthetic templates had a source at the
   centre (the integrator's measure on 6,000 stamps, 2026-11-08), with two failures: a
   galaxy nucleus or a star under the spike while the donor Bogus' features say there is
   no PS1 source nearby (`distpsnr1 < 1"` for 100 % of the AGN, 92 % of the VS, 5 % of the
   positive Bogus), and a hot pixel on a galaxy nucleus looks like a faint nuclear SN, a
   case the model already confuses with Bogus (SN -> Bogus 25 of 2,500 on validation).
   Now 6.1 % of the synthetic templates have a source at the centre, against 29.5 % of the
   real positive-alert Bogus; `scripts/inspect_class_augment.py` checks that the synthetic
   share is at most the real one + 0.05 (and exits with 1 otherwise), and a unit test
   checks it on a synthetic batch. **SN rows are never replaced** either: SN is the weakest
   class (1,396 Train stamps).

   **Features: a Bogus source keeps its own; an Asteroid source takes a donor's**, those of
   a real positive-alert Bogus row (stored `isdiffpos` 0) **of the same batch**, instead of
   a separate device-resident bank: the batch is a shuffled stream of the run's own
   training part (also for split runs), it needs no extra state, and a batch has about
   20 % Bogus after the tiling (an Asteroid source without a donor is skipped). **A real
   stamp of another class is never relabelled**: an Asteroid row's stamp is only raw
   material, and its own label is replaced together with its image; a Bogus row stays a
   Bogus. A replaced row is not noised again by C5. On `red` Train the share of stamps
   usable as a source is Asteroid 92 % and Bogus 57 %, and the synthetic rows come 62 %
   from Asteroids and 38 % from Bogus.
6. **The spike is narrower than any PSF in the data, by an explicit, tested measure.**
   `moment_fwhm` is the circular-Gaussian FWHM from the pixel second moments less a pixel's
   own 1/12; for a pixel-integrated Gaussian it equals the Gaussian's FWHM, which is what
   the PSF measurement fits. The smallest PSF FWHM, `MIN_PSF_FWHM_PX = 1.19` px, is the
   minimum over the differences of every Asteroid (1.21 px) and SN (1.19 px) of `red` Train
   (a pixel-integrated circular Gaussian + constant, Levenberg-Marquardt per stamp, fits
   with S/N > 15). The difference is used because its static sources are subtracted; in
   the science a static neighbour inside the window spoiled 10 fits (all under 1 px,
   inspected), while the science's first percentile (1.22-1.25 px) agrees. The alert
   feature `fwhm` is not usable: mapped to pixels through these fits, its smallest values
   are about 0 px. The spikes are one pixel (0 px) or a horizontal or vertical 2-pixel line
   (at most 0.48 px; 0.96 px even along the line); diagonal lines are excluded (1.52 px
   along their axis). Spike peak: 10-50 sigma of the source science.
7. **Seeding as `rot90_augment`, one stream per augmentation**: `default_rng([seed,
   global_step, stream])`, streams 1 (noise), 2 (jitter), 3 (synthetic). Device-independent,
   exact on resume, without consuming the global generator; turning one augmentation on
   does not change another's draws (common random numbers across ablations), and
   `rot90_augment` keeps its `[seed, step]` draws. Each draw has a fixed size per batch.
8. **Flags and presets as in ADR 0005**: off by default, listed in both presets and
   `configs/phase2/carrasco2021*.yaml` at their defaults, so an augmented run is a recorded
   deviation; rejected for other models. `aug_noise` and `synthetic_bogus` need the float
   input path (they re-normalise like it); `synthetic_bogus` needs `context: none` (the
   context channels would need a synthetic context, deferred); C5 with context noises the
   crop channels only; `aug_feature_jitter` needs `with_features`.
9. **`compare-reference` reads a missing config key as its default.** Adding preset keys
   would otherwise add WARN lines for every run older than the key (the defect recorded in
   the plan for the 2.5-models keys). The defaults of new flags are the behaviour the old
   runs had, by construction.
10. **Prefetch: plan on the host, apply on the device** (2026-11-08, audit item 5). The
   first version made every draw on the host *inside the step* and copied the results to
   the device with blocking, pageable copies, after `.cpu()` reads of the batch: about 8-10
   CPU-GPU synchronisations per augmented step, each draining the GPU queue. Every input
   of the decisions is known on the host before the step (the stream's batch indices are a
   pure function of `(seed, b)`, the Train arrays and fits are NumPy, the generators are
   `[seed, step, stream]`), so the work is split: `PaperClassifierModule.augment_plan`
   (host) makes every draw and decision of C6, C7, C5, the rotation and the jitter, with the
   same NumPy calls and float64 arithmetic (`class_augment.plan_noise` / `plan_synthetic` /
   `plan_jitter`, `flux_rescale.plan_flux`); `apply_plan` (device) runs exactly the torch
   operations the step ran before (`apply_noise`, `apply_synthetic`, `apply_flux`, the
   rotation's gather, the jitter's addition). The training loader
   (`DeviceTensorDataModule(planner=module.augment_plan)`, wired by `experiments` for every
   augmented run) gathers the step's rows from a CPU copy of the Train split, plans, and
   uploads the plan (`class_augment.upload_plan`: pinned memory and `non_blocking=True` on
   CUDA) with the batch; the step makes no host round trip. **The results are bit for bit
   those of the in-step path on every device** (the same values reach the same device
   operations), so the reproducibility contract is unchanged: the draws are
   device-independent and exact on resume, while the float32 device operations (C5's
   min-max, C6's min-max of the built rows, C7's feature update) are, as before, the
   device's. A seeded torch generator on the device was rejected: CUDA's Philox stream
   differs from the CPU's, so the draws would depend on the device. The in-step path
   (`augmented_batch`) remains for direct calls and as the reference of the equivalence
   tests (`tests/test_augment_prefetch.py`: the loader's planned batches equal it over 200
   steps with every flag on, and a short augmented run gives the same `metrics.json`,
   timing aside, and the same predictions on CPU and MPS). Checked against the
   pre-prefetch code as well: 200 CPU steps with every flag on, `torch.equal` on every
   tensor, and the same short run's `metrics.json` on CPU and MPS.

## Consequences

- A C5 / C6 run is `train-supervised --preset carrasco2021 --set aug_noise=1.0
  --set aug_feature_jitter=0.05 --set synthetic_bogus=0.05 [...]`; the values are choices of
  the 2.5 runs, not fixed here.
- An augmented step costs a host-device round trip (the selection of rows and the draws are
  on the CPU) and C6's construction on the CPU for the few replaced rows.
- C6 shifts the class balance of each batch towards Bogus by up to `synthetic_bogus`, taken
  from Asteroid and Bogus rows only; the tiling is unchanged.
- Old checkpoints of `PaperClassifierModule` load unchanged (the new hyperparameters have
  defaults); old runs' metrics no longer warn about the 2.5 flags in `compare-reference`.
- Later ideas, not built: fake SN injected on real host galaxies (aimed at SN -> Asteroid);
  synthetic context channels for C6.
