<!-- ADR 0019: the SimCLR pilot of phase 5 (task 5.0b): engine-free views, network, loss, LARS and schedule, probe, diagnostics, and a minimal plain-PyTorch loop. -->
# 0019: The 5.0b SimCLR pilot: engine-free functions and a minimal plain loop

- **Status:** accepted (2026-11-08, phase 5, task 5.0b; plan review
  `docs/plan-reviews/2026-11-08-phase-5.md`, findings F7, F8, F10, F11; the SimCLR review
  of the package, findings A1-A6 and B1-B8)
- **Implemented by:** `src/astro_simclr/simclr/` (`views.py`, `network.py`,
  `objective.py`, `optim.py`, `probe.py`, `diagnostics.py`, `pilot.py`, `commands.py`);
  `src/astro_simclr/stamp_relation.py`; the per-group exclusion of `optim.LARS`
  (`EXCLUDE_KEY`); four new `engine.randomness.RESERVED_STREAMS`
- **Checked by:** `tests/test_simclr_views.py`, `tests/test_simclr_model.py`,
  `tests/test_simclr_probe.py`, `tests/test_simclr_pilot.py` (CPU only: a few steps)

## Context

Phase 5 asks whether SimCLR pretraining on ZTF stamps helps classification with few
labels. Before the user is asked to approve a corpus download (about 20 TB of archive),
task 5.0b runs a pilot on the 50.6k CD21 Train stamps, with no download and without
waiting for the plain engine's loop (task 4.1). The pilot is judged by a probe on the CD21
screen halves only (CV is out: its pool's Train stamps are pretrained on).

The package's SimCLR (`training.modules.SimCLRModule`, Lightning) cannot be the pilot.
A review of it found: a cosine schedule that ignores the run's length (A1), a LARS that
freezes every zero-initialised parameter, BatchNorm shifts included (A2), a linear
evaluation with a double softmax (A3), colour jitter and greyscale that mix the science,
template and difference channels (A4), a blur as wide as the PSF (A5), crops that can drop
the source (A6), and the ImageNet stem (stride 4 before the first block). Phase 4's rule
(ADR 0014, CLAUDE.md) also requires new training code to be engine-free.

## Decision

### 1. A separate, engine-free package with its own minimal loop

`astro_simclr.simclr` imports no Lightning (a fresh-interpreter test checks it). Every
piece is a pure function or a small object that takes `step` (optimizer steps done before
the step), so task 5.3 can wrap the same functions in an engine `Task` without changing
them. The loop of the pilot (`pilot.train_pilot`) is deliberately minimal: one device,
the stamps resident on it (`tier="device"`, tier 1) or on the host (`"host"`), a step
grid for logs, diagnostics and checkpoints, and a final embedding dump. Its checkpoint is
a pilot-only format (`astro-simclr-pilot-checkpoint`, version 1: config, data record,
steps done, model, optimizer; read with `weights_only=True`), not the engine's
`Checkpoint`, whose policy and identity fields have no meaning without the engine. Global
RNG state is not stored because the loop never draws from it: every draw comes from the
per-step generators, so the resume is exact (checked on CPU against an uninterrupted
run). `stop_at` ends a call early for time-boxed jobs; the schedule stays the whole run's.

### 2. The views

Input: the stored 63-pixel stamps (per-channel min-max, NaN -> 0), float32
(`images_f32.npy`; a uint8-only layout is read as `images / 255` and recorded as a
deviation). A training view is, in this order:

1. **noise** (off by default, `noise_scale`): C5's noise (ADR 0006) on the whole 63-pixel
   stamp: Gaussian noise on the science and the template scaled by `u * noise_scale *
   sigma_bg`, the difference moved through the stamp's fit, then the per-channel min-max
   again. It comes first because the fit describes the stored stamp, and because "before
   normalisation" (plan 5.2) means the min-max is redone after the noise;
2. one of the **8 dihedral** elements (`rot90` by `k % 4`, then a left-right flip if
   `k >= 4`): exact, no interpolation or fill;
3. a **57-pixel window** at an integer offset in 0-6 per axis, the same for every channel:
   a shift of at most 3 pixels, so the source cannot leave the view, and no pixel is filled.

**The window is not re-normalised.** The view keeps the 63-pixel stamp's min-max: a
per-window min-max would make the values of a pixel depend on where the window falls,
which is not a property of the sky. Evaluation uses the centred window (offset 3), the
centre of the training distribution. Hue, saturation, greyscale, colour jitter, blur,
scale jitter and resized crops are not in the module (plan 5.2: blur and scale jitter are
ablations only).

### 3. Randomness: four new streams

`RESERVED_STREAMS` gains `simclr_views` (6), `simclr_view_noise` (7),
`simclr_batch_order` (8) and `simclr_diagnostics` (9); 5 is left to task 4.3's
`block_shuffle` (PR #42). Each stream has its own third word, so none shares a generator
with another (unlike the two-word `[seed, j]` of `rot90_augment` and the `row` stream's
passes, the correlation phase 4 records). The per-sample parameters of both
views of step `s` come from the host generator `default_rng([seed, s, 6])`, drawn in full
every step (`k` (2, B), offsets (2, B, 2), noise strengths (2, B, 2)) so that turning the
noise on never changes the geometry. The noise's per-pixel fields come from the device
generator of `[seed, s, 7]` (device-dependent values; `pilot.json` records
`noise_fields: device-dependent`). The batch order is a permutation per pass, keyed
`[seed, pass, 8]` (the second word is the pass, not the step: one permutation serves all
of a pass's steps); the last partial batch of a pass is dropped. The diagnostics' fixed
sample and views use `[seed, 0, 9]` and `[seed, 1, 9]`.

### 4. Network, loss, optimizer, schedule

- **Encoder:** torchvision `resnet18(weights=None, zero_init_residual=True)` with a 3x3
  stride-1 stem and no max-pool (the CIFAR stem of Chen et al. 2020, App. B.9); 57 -> 57
  -> 29 -> 15 -> 8; `fc = Identity`; `h` is 512-d. `zero_init_residual` follows SimCLR's
  ResNet (`init_zero` on each block's last batch norm). The new stem convolution gets
  torchvision's own initialisation.
- **Head:** Linear(512, 512, no bias) - BatchNorm1d - ReLU - Linear(512, 128). The first
  bias is dropped because the BatchNorm cancels it.
- **Forward:** both views in one concatenated batch, so BatchNorm normalises over the 2B
  views (SimCLR's single-device equivalent of its global BN).
- **Loss:** `losses.NT_Xent(legacy_denominator=False)` (ADR 0002's paper mode), in float32
  with autocast off. The legacy/DCL form is not used: phase 5 reports SimCLR.
- **LARS:** the package's LARS with `legacy_trust_ratio=False` (the trust ratio only when
  both norms are > 0), no Nesterov, momentum 0.9, `eeta` 0.001, weight decay 1e-6. BatchNorm
  (and other normalisation) parameters and every bias go to a second param group with
  `exclude_from_adaptation: True` and no weight decay: plain SGD with momentum, as SimCLR's
  `exclude_from_weight_decay` / `exclude_from_layer_adaptation`. The exclusion is a param-
  group key read with `group.get` (`optim.EXCLUDE_KEY`), not a constructor default, so every
  existing LARS run and optimizer state is unchanged.
- **Rate:** `0.3 * B / 256`.
- **Schedule (per optimizer step):** linear warmup `base * (s + 1) / W` over `W = round(
  warmup_epochs * steps_per_pass)` steps (at most `total_steps - 1`), then a cosine from
  `base` that reaches `lr_min` (default 0) **exactly** at the run's final step `total_steps
  - 1`. `total_steps` is the run's length (`epochs * steps_per_pass`, or `steps`), so the
  schedule can neither outlive nor undershoot it (A1).

### 5. The probe on frozen `h`

- **Embeddings:** `h` of every Train and Validation stamp on the centred view, in eval mode
  (BatchNorm running statistics), dumped once per checkpoint of interest
  (`simclr-embed` also dumps the random-init network: the go/no-go's random-init arm).
- **Label subsets:** `per_class` stamps per class, stored in a JSON file with the SHA-256 of
  the candidate indices and of the Train labels. Candidates are the Train stamps a screened
  run may use (the splits file's `screen_training_excluded` left out). Subset `r` takes the
  first `per_class` of a permutation of each class's candidates drawn from
  `default_rng([seed, r, class])`, so budgets are nested (10 per class inside 100 per
  class, subset by subset) and every arm trains on the same subsets.
- **Probe:** multinomial logistic regression on standardised `h` (the subset's mean and
  std), plain cross-entropy on the logits (not A3's double softmax), an L2 penalty on the
  weights, fitted by full-batch L-BFGS in float64 on the CPU from zero weights. The problem
  is convex, so the fit is deterministic and has no learning rate or epoch count to tune.
- **Selection:** the penalty is chosen from a declared grid (1e-4, 1e-3, 1e-2, 1e-1) on
  the screen's **selection** half; only the chosen probe is scored on the **decision**
  half. Balanced accuracy (the halves are balanced, so it is also the accuracy).

### 6. Diagnostics

Every `diagnostics_every` steps and at the end, on a fixed sample (512 pretraining
stamps) with fixed noiseless views, in eval mode: the effective rank of `cov(h)` (Roy and
Vetterli), the per-dimension std of normalised `z` (mean and minimum), alignment and
uniformity (Wang and Isola, `t = 2`), and a weighted kNN monitor (`k = 20`, `T = 0.07`,
Wu et al. 2018; a one-hot vote sum, deterministic on every device) with a bank of up to
1000 pretraining stamps per class, queried by the screen's selection half. The decision
half is never read by the loop.

### 7. Pretraining data

The pilot pretrains on Train without `screen_training_excluded` (the leaked stamp and the
Train stamps grouped with Validation), so the screen halves are not pretrained on, not even
through a ghost of the same object. Validation and Test are never pretrained on.

### 8. The stamp relation leaves `training`

`stamp_fits`, `masked_pixels`, `renormalise`, `perturb` and their constants move,
unchanged, from `training/class_augment.py` to `astro_simclr/stamp_relation.py`, because
importing anything from the `training` package imports Lightning (its `__init__`).
`class_augment` re-exports every name, so no caller changes.

## Consequences

- No existing config changes its numbers: LARS's default behaviour, `NT_Xent`'s default,
  the Lightning SimCLR and the paper model are untouched; the moved functions are the same
  code.
- The pilot is not engine-equivalent: it has no validation grid, no horizon, no manifest
  (A1's `manifest.json`), and its checkpoint format is its own. Task 5.3 runs the same
  functions on the engine's loop (tiers 1 and 3, exact resume, retention, manifest with the
  corpus and exclusion SHA-256) and retires `pilot.py`'s loop.
- The pilot's precision is ADR 0010's float32 (`ieee`) by default; `--fp32-precision tf32`
  is available and recorded. 5.3 decides the pretraining precision.
- The go/no-go also needs the scratch supervised ResNet-18 trained with the same views; it
  is not part of this package yet.
- The noise is off by default: it is an option of 5.2 whose admission test (a single-view
  accuracy drop of at most 1 point) has not been run.
