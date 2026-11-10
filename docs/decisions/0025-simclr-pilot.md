<!-- ADR 0025: the SimCLR pilot of phase 5 (task 5.0b): engine-free views, network, loss, LARS and schedule, probe, diagnostics, and a minimal plain-PyTorch loop. -->
# 0025: The 5.0b SimCLR pilot: engine-free functions and a minimal plain loop

- **Status:** accepted (2026-11-08, phase 5, task 5.0b; plan review
  `docs/plan-reviews/2026-11-08-phase-5.md`, findings F7, F8, F10, F11; the SimCLR review
  of the package, findings A1-A6 and B1-B8)
- **Implemented by:** `src/astro_simclr/simclr/` (`views.py`, `network.py`,
  `objective.py`, `optim.py`, `probe.py`, `diagnostics.py`, `pilot.py`, `commands.py`;
  the amendments' `scratch.py`, `compare.py`, `finetune.py`);
  `src/astro_simclr/stamp_relation.py`; the per-group exclusion of `optim.LARS`
  (`EXCLUDE_KEY`); four new `engine.randomness.RESERVED_STREAMS`
- **Checked by:** `tests/test_simclr_views.py`, `tests/test_simclr_model.py`,
  `tests/test_simclr_probe.py`, `tests/test_simclr_pilot.py` (CPU only: a few steps);
  the scratch baseline (amendment below): `tests/test_simclr_scratch.py` and the golden
  case `simclr-scratch-noise` of `tests/test_engine_regression.py`; the fine-tuning arm:
  `tests/test_simclr_finetune.py` and the golden case `simclr-finetune-noise`; the corpus
  amendments:
  `tests/test_corpus_index.py`, `tests/test_corpus_fits.py`

## Context

Phase 5 asks whether SimCLR pretraining on ZTF stamps helps classification with few
labels. Before the user is asked to approve a corpus download (about 20 TB of archive),
task 5.0b runs a pilot on the 50.6k CD21 Train stamps, with no download and without
waiting for the plain engine's loop (task 4.1; it landed with PR #52 after this pilot was
written, and 5.3 moves the pilot onto it). The pilot is judged by a probe on the CD21
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
  is not part of this package yet (added by the amendment "the scratch baseline and the
  declared margin" below: `simclr.scratch`, `simclr.compare`).
- The noise is off by default: it is an option of 5.2 whose admission test (a single-view
  accuracy drop of at most 1 point) has not been run.

## Amendment (2026-11-08, task 5.1): pretraining on the corpus index

`simclr-pilot` also accepts a corpus index file (ADR 0019, amendment) as DATA. The
pretraining stamps are then the index's archive rows, normalised exactly as the stored
stamps (the 63-px input of point 2). Evaluation stays on CD21:
- the layout the index excluded (`--eval-data`, whose source SHA-256 must match) gives the
  kNN bank, the screen's selection half and the embedding dump;
- `--splits` keeps its meaning.

The pretraining stamps have three tiers:

| `--tier` | Use |
|---|---|
| `memmap` | The default for an index. Each batch is read sorted by file offset, normalised on the host into a pinned tensor (CUDA) and uploaded without blocking. The next step's batch is prefetched in one worker thread |
| `host` | The corpus normalised once in host memory, refused above `--host-cap-gb` (default 24) |
| `device` | Refused for a corpus: about 18.6 GB for 390k stamps, too much beside other jobs (ADR 0018) |

The tiers give identical batches; a CPU test checks that `memmap` and `host` give the same
model.

The prefetch matters because LARS synchronises with the device at every step, so an
unhidden read adds to each step. An interleaved A/B on the shared 5090, with batch 256 and
the GPU busy with another job, measured 0.624 s per step without the prefetch and 0.550 s
with it, about the 66-ms read.

The noise view needs per-stamp fits, which are not computed for a corpus, so a corpus run
refuses `noise_scale > 0`. (Superseded by the amendment of 2026-11-09 below: with the fits
sidecar of `build-corpus-fits`, a corpus run takes the noise view.)

`pilot.json` records the index (path and SHA-256), the archive, the `exclusion_sha256`,
the counts and the rules.

The pilot refuses an index weaker than the plan's exclusion: a radius below 1.5″, or no
near-duplicate check. `--allow-weak-index` (the config's `allow_weak_index`, recorded)
overrides it for tests.

A resume compares the checkpoint's index SHA-256, `exclusion_sha256` and row count with
the current ones. An index rebuilt at the same path is therefore refused, not mixed in.

`PilotConfig` gains `archive`, `eval_data`, `host_cap_gb` and `allow_weak_index`. They do
not change a layout run's numbers, so a checkpoint written before them resumes with them at
their defaults.

## Amendment (2026-11-09, tasks 5.0b/5.0c/5.3): probe monitors during pretraining

**Context.** The kNN monitor of section 6 can mislead. In the first pilot it rose while the
linear probe on frozen `h` scored below the random-init network (0.586 against 0.618). The
user wants the downstream metric visible while the run trains, and the 5.3 abort rule
stated on it. Two monitors are added in `simclr.monitor`, both off by default. Off, a run's
files are those of before: `pilot.json`, the checkpoints, `metrics.jsonl`,
`diagnostics.jsonl` and the embeddings were compared with `main`'s and are identical but
for the paths. The record and the checkpoint config leave the new fields out at their
defaults.

**Inputs (user decision, 2026-11-08).** Images only: the probe input is the encoder's `h`
of the centred 57-pixel crop, in eval mode. No alert feature, position or anything else
of the candidate, and no augmentation (no noise, dihedral view or shift), for both
monitors. A test checks that the probe input equals the encoder's output on
`centre_crop(x)`.

**Role (5.0c).** Both curves are diagnostics and an abort input, **never a checkpoint
selector**: the epoch count is fixed and the final checkpoint is kept.

### The periodic budget probe

- **When:** `--probe-every K` passes, at the abort check, and at the final step.
- **The probe:** 5.4's (`protocol.probe.fit_probe`), not a new one. Standardised `h`, plain
  cross-entropy, L2 `DEFAULT_WEIGHT_DECAY = 1e-3`, full-batch L-BFGS in float64 on the CPU:
  deterministic, with nothing to tune. The class balance comes from the subsets (`b`
  objects of every class), not from a weighted loss, so the monitor's number is the one 5.4
  computes on the same rows.
- **Labels:** the stored label-budget file (`make-label-budgets`). It must be drawn from
  the run's splits file (`verify_label_budgets`). The probe uses repeat 0 of part `fold0`
  at 10 and 100 objects per class. Nothing is drawn, so no stream is added.
- **Train stamps only.** A CV fold's training part is a part of the pool, Train plus
  Validation, so a stored subset can hold Validation stamps, even of the screen's decision
  half. On CD21 `fold0`:

  | Budget | Validation stamps in the subset | Of them in the decision half | Train stamps kept |
  |---|---|---|---|
  | 10 | 1 | 0 | 49 |
  | 100 | 11 | 6 | 492 |

  Training on them would read the decision half, and evaluating on the selection half
  would score stamps the probe had trained on. So the monitor keeps only the subset's Train
  stamps that the screened run may use, and records what it dropped and the per-class
  counts left. No Train stamp of the subset was screen-excluded on CD21.
- **Evaluation:** the screen's **selection** half only, the kNN monitor's data. Its `h`
  is embedded once per point and shared with the kNN. The decision half, the CV evaluation
  parts (the training part excludes its fold's evaluation part, and no Validation stamp is
  trained on) and Test (never loaded by the pilot) are not read. A test poisons every
  Validation stamp outside the selection half with NaN, records every row the monitors
  embed, and checks both.
- **Keys:** `probe_b10_selection_bacc` and `probe_b100_selection_bacc` in
  `diagnostics.jsonl`, `null` on the lines that are not probe points.
- **Cost on CPU:** about 10 s per point on a loaded laptop (9-12 s over three points, 4
  threads). That is the 250 selection stamps and the 492 Train stamps embedded, plus two
  fits. The whole Train set is not embedded.

### The abort rule (5.3)

- **Opt-in:** `--abort-if-probe-below-random EPOCH`.
- **The baseline:** the same probe on the network as initialised. It is computed once,
  before the first step, and kept in the checkpoint, so a resume does not recompute it.
- **The check:** at the end of pass EPOCH, the run stops unless the largest budget's probe
  beats the baseline strictly (a tie aborts).
- **A clean stop:**
  - `ckpt/last.pt` keeps the state;
  - `pilot.json` records `status: aborted` and the check's values;
  - neither `final.pt` nor the embeddings are written;
  - the CLI exits 0 with a message;
  - a resume is refused.

The plan's 5.3 rule states the kNN; the probe replaces it as the abort input because the
kNN is the metric that misled.

### The online probe

- **Opt-in:** `--online-probe`.
- **The head:** `Linear(512, 5)` on `h.detach()`, initialised at zero so that it draws
  nothing from the global generator (it is also built under `fork_rng`).
- **The optimizer:** its own Adam, lr 1e-3, no weight decay, on class-balanced
  cross-entropy with weights `N / (C n_c)` of the usable Train stamps.
- **The side-batch:** each step, before the training forward, 256 of the usable Train stamps
  (`PilotData.rows`; the evaluation layout's with a corpus as DATA), drawn from
  `host_generator(seed, step, simclr_online_probe)`. That is the **new stream 12**,
  assigned by the integrator.
- **The forward:** the eval view, with the encoder in eval mode under `no_grad`, so its
  BatchNorm statistics do not move. The encoder's weights, gradients and losses are
  bit-identical with the probe on and off (tests).
- **Keys:** `online_probe_train_acc` (the running accuracy on the side-batches since the
  previous diagnostics line) and `online_probe_selection_bacc`, at each diagnostics point.
- **What it is:** a 100 %-label trend monitor. It lags the final probe, since it is trained
  online on a moving encoder.
- **Cost:** about 4.6 s per step on the laptop CPU, against about 35 s for the training
  step at B = 256.

### State and resume

With a monitor on, the checkpoint gains `monitor`:

- the baseline;
- the abort outcome;
- the head, its optimizer and the running counts.

The format version stays 1: the entry is absent without a monitor, and the config
equality already ties it to the run. A resume continues exactly, and the existing
truncation of the log lines keeps `diagnostics.jsonl` free of duplicates (a test).
`pilot.json` gains `status` (`running`, `stopped`, `aborted`, `complete`) and `monitors`.

### Engine-free

Both monitors are pure functions or small objects outside every engine. There is no
Lightning import, and the only draw is keyed by `(seed, step)`. 5.3 moves them onto the
plain engine with the rest of the pilot.

## Amendment (2026-11-09, task 5.0b): the scratch baseline and the declared margin

**Context.** The pilot's go/no-go (plan 5.0b) needs the SSL probe to beat "the scratch
supervised model at the lowest budget by a pre-registered margin". The user decided
(2026-11-09) that the scratch model is "the same ResNet architecture or very close", and
the integrator declared the margin before any baseline result. The probe reports measured
on the RTX 5090 (main `827c41a`) were 0.571 +- 0.052 at 10 per class and 0.734 +- 0.021 at
100 (random-init probe 0.417 and 0.515); no scratch number existed when this recipe was
fixed.

### The baseline (`simclr.scratch`, `simclr-scratch`)

- **Network:** the pilot's encoder, `resnet18_cifar()` (CIFAR stem, `zero_init_residual`,
  512-d `h`), then `Linear(512, 5)`; random init, built on the CPU after the engine seeds
  the global streams (encoder first, so it starts as the pilot's would from that seed).
- **Data:** one fit per label subset of the `make-label-subsets` file the probe used, on
  that subset's Train stamps only: the training stream holds nothing else, so a Train stamp
  outside the subset, Validation, the decision half and Test cannot be trained on. Each
  step is **class-stratified**: `k = min(10, per_class)` stamps of each class (a batch of
  50), the first `k` of a permutation of the class's subset stamps. Every batch is exactly
  balanced, so the mean cross-entropy is the class-balanced one.
- **View:** the pilot's training view (C5 noise at the pilot's scale 3, one of the 8
  dihedral elements, a 57-pixel integer window), one per stamp. Evaluation is the probe's
  input: the centred 57-pixel window, eval mode, no augmentation.
- **Recipe:** AdamW (betas 0.9, 0.999), decoupled weight decay on convolution and linear
  weights only (normalisation parameters and biases excluded, as the pilot's LARS groups
  them), a linear warmup over 5 % of the steps then a cosine to 0 exactly at the last step
  (`optim.warmup_cosine_lr`), float32 (`ieee` by default). AdamW because it needs no
  per-budget rate tuning beyond a small grid and is the usual from-scratch choice at these
  sizes; the cosine because it ends the fixed run at rate 0, so the final weights (kept,
  5.0c) are settled.
- **Length:** fixed per budget and the same for every subset and trial: 1,000 steps at 10
  per class (one pass of the 50 stamps per step; 8 x 49 geometric views and the noise make
  about 20k distinct inputs, and the training loss is near 0 well before the end), 2,000 at
  100 per class (200 passes, the usual supervised budget at CIFAR scale). Another budget
  needs `--steps`. No early stopping and no checkpoint selection (5.0c).
- **Grid and selection (5.0c):** `lr` in (5e-4, 2e-3) x `weight_decay` in (0.05, 0.5):
  **four trials per subset, the probe's trial count** (its four L2 values), so both arms
  have the same trial budget. Each trial's final weights are scored on the screen's
  **selection** half (balanced accuracy, the fit's only validation); the first best in grid
  order is kept, the probe's rule; only it is scored on the **decision** half. The grid
  and the trial counts (`trials_per_subset`, `n_trials`) are in the report.
- **Randomness:** stream 13, `simclr_scratch` (assigned by the integrator). Every draw of
  step `s` of subset `r` comes from one host generator `default_rng([seed_r, s, 13])`, in
  order: the batch, the dihedral elements, the offsets, the noise strengths and the
  noise's standard-normal fields (float32). Drawing the fields on the host (the pilot draws
  them on the device, stream 7) costs a few milliseconds per step at B = 50 and makes every
  draw device-independent. `seed_r` is the first 32-bit word of `SeedSequence([seed, r,
  13])`: each subset has its own initial weights and augmentations, so the spread over
  subsets includes the initialisation's, as the hierarchical bootstrap's resampling of
  subsets assumes. The trials of one subset share `seed_r` and differ only in their
  hyperparameters.
- **On the plain engine:** each trial is `engine.loop.fit` with a `ScratchTask`, the
  subset's `ScratchStream` (the batch of step `s` a pure function of `s`), a fixed horizon
  (`Horizon(steps, 0)`), one validation after the last step (`ValidationGrid(steps)`,
  monitor `selection_bacc`, `max`, through `score`), `keep=("stop",)` and no checkpoint
  (`run_dir=None`). The engine fitted without any change. A trial is short, so the command
  resumes at subset granularity: it rewrites its report after each subset, and `--resume`
  keeps the finished subsets of a report of the same configuration (identity keys
  compared). The trial joins the golden regression set (`simclr-scratch-noise`).
- **Cost:** 20 subsets x 4 trials x (1,000 + 2,000) steps of 50 stamps at 57 pixels, about
  240k steps: a few hours on the RTX 5090.

### Reports and the comparison (`simclr.compare`, `simclr-compare`)

- **Version 2 of the subset report** (both `simclr-probe` and `simclr-scratch`):
  `report_version: 2`, `subsets_file` and `splits_file` (path and SHA-256),
  `decision_rows` and `decision_labels`, and per subset `decision_predictions` (the kept
  model's predicted class of each decision-half stamp, in its order). A version-1 probe
  report stays readable as a table; the comparison refuses it and asks to re-run
  `simclr-probe` (a CPU minute on the stored embeddings).
- **The interval:** `protocol.stats.hierarchical_paired_bootstrap` of balanced accuracy,
  variant - baseline, paired by subset (the same stored subsets file, checked by SHA-256)
  and by stamp (the same decision rows and labels): the subsets resampled with
  replacement, then the decision half's objects within class (the splits file's groups;
  the alerts of one object move together), 10,000 resamples, seed 0, 95 %.
- **The declared margin (H1's reading):** at **10 per class** the SSL probe (variant) must
  beat the scratch model (baseline) by **more than +2 points**: `go` iff the lower bound of
  the paired 95 % interval is above +0.02 (rounded to 12 digits, so a bound of exactly
  0.02 is `no-go`). 100 per class is secondary: its interval is reported with no verdict.
  The plan's other condition, the SSL probe beating the random-init probe, is read from
  the same command with the random-init report as the baseline.
- **Fairness:** both arms train on the same subsets, are chosen on the selection half by
  the same rule with four trials each, and are scored on the same decision-half stamps from
  the same input (the centred view, eval mode). Neither reads the decision half or Test for
  any choice.

## Amendment (2026-11-09, tasks 5.1/5.3): the noise view on a corpus index

The GO of the 5.0b pilot leads to pretraining on the corpus index with the recipe that won
on CD21: batch 256, `lr_per_256` 0.3, temperature 0.2 and the C5 noise at scale 3. The
amendment of 2026-11-08 refused the noise on a corpus; without it the views collapse
(LARS grid v2: positive similarity 0.998, a probe below random init).

**Decision.**
- `PilotConfig` gains `corpus_fits`: the sidecar of `build-corpus-fits` (ADR 0019,
  amendment of 2026-11-09), `--corpus-fits FITS_FILE` on `simclr-pilot`. A corpus run with
  `noise_scale > 0` needs it, and the refusal without it names the command that builds it.
  The field is refused without noise and with a layout as DATA, and `to_plain` leaves it
  out unset, so a run without it writes the record and checkpoint config it wrote before.
- `load_pilot_data` reads the sidecar and refuses it unless it names the index's SHA-256
  and row count (`verify_corpus_fits`). `pilot.json` records `data.pretrain.fits`: the
  sidecar's path, SHA-256, format and the index SHA-256 it names. A resume also compares
  the fits' SHA-256, so fits rebuilt at the same path are refused.
- The views, their randomness and the noise arithmetic are the pilot's: no new stream. The
  fits are indexed by the same stored positions as the stamps (a corpus's index positions;
  `memmap` and `host` alike), on the training device (24 bytes a row). A layout run still
  fits its Train stamps when it starts.

**Checks** (CPU tests, `tests/test_corpus_fits.py`): on an archive whose stamps are also
written as a layout (the "mirror"), the corpus fits equal the layout path's `stamp_fits`
bit for bit, and the corpus run with noise trains the same model on `memmap` and `host` as
the layout run with noise on the mirror. It is deterministic for a seed and resumes
exactly. The fits lookup costs about 3 µs per batch of 256 on the CPU, against about 66 ms
for the read of the batch's stamps on the GPU machine.

## Amendment (2026-11-09, tasks 5.0c/5.3): activation checkpointing and the CUDA peak

The batch-size grid {256, 1024, 2048} (user, 2026-11-09: batch 1024 first, "more is
better", Chen et al. 2020) does not fit the pilot's memory as it is. On the RTX 5090
(32 GB) the batch-256 pilot uses about 10 GB; at batch 1024 the run filled 31.6 GB and
logged no step in 10 minutes (the WSL driver spilled to system memory). The pilot has no
autocast; the encoder's activations dominate: about 20 MiB per 57-pixel view kept for the
backward (measured on the CPU by the bytes of the tensors autograd saves), so 40 GiB for
the 2048 views of a batch of 1024.

**Decision.**
- `PilotConfig.activation_checkpointing` (default off; `--activation-checkpointing` on
  `simclr-pilot`): `SimCLRNetwork(activation_checkpointing=True)` runs the encoder, in
  train mode with gradients on, as nine segments under `torch.utils.checkpoint.checkpoint`
  (`use_reentrant=False`, `preserve_rng_state=True`): the stem (conv1, bn1, ReLU, the
  identity max-pool), then each of the eight residual blocks
  (`network.encoder_segments`, `network.checkpointed_encoder`). The head (512-d, 2 BN
  layers) is not checkpointed: its activations are 3 KB per view. Eval mode and `no_grad`
  (diagnostics, probes, embeddings, the online probe's encoder pass) run the plain forward.
- **Granularity: one residual block per segment.** Checkpointing keeps each segment's
  input and, during the backward, the whole saved set of one segment at a time. Measured
  per view (CPU, bytes autograd saves): off 20.1 MiB; per block 3.8 MiB of segment inputs
  plus at most 4.0 MiB for the block being recomputed (a `layer1` block, 57 x 57 x 64);
  per stage (`layer1` to `layer4` whole) 2.25 MiB of inputs but 8.0 MiB for `layer1`
  recomputed. Per block has the lower peak (about 7.8 against 10.3 MiB a view, 39 % of the
  plain run's); both recompute every segment's forward exactly once, so their arithmetic
  cost is the same: about one more encoder forward a step, roughly a third more encoder
  time (a backward costs about two forwards). Finer segments (per convolution) would cut
  the transient further but add a recompute boundary at every BatchNorm; not needed for
  batch 1024. Expected at batch 1024 on the 5090: about 16 GiB of activations plus the
  2.4 GB of stamps, the gradients and the allocator's slack, under 32 GB; batch 2048
  (about 32 GiB of activations) does not fit with this alone (it would need bf16, plan
  5.3's precision decision).
- **BatchNorm buffers.** The recompute runs in train mode, so every BatchNorm of the
  segment would update `running_mean`, `running_var` and `num_batches_tracked` a second
  time per step. The segment's `context_fn` saves every buffer of the segment when the
  recompute starts and copies it back when the recompute ends (also when the
  checkpoint's early stop ends it by raising): the buffers after the step are the
  forward's. Keeping BN outside the segments was not possible at block granularity
  (torchvision's `BasicBlock` holds its BN layers), and setting the momentum to 0 still
  counts `num_batches_tracked`.
- **Numerics.** The recompute performs the same operations on the same inputs, and the
  non-reentrant checkpoint keeps the autograd graph (only the saved tensors are produced
  again), so gradients accumulate in the same order. On the CPU the losses, every
  gradient, the weights, the BatchNorm buffers and the LARS momenta after N steps equal
  the run without the flag **bit for bit**, and so do the pilot's metrics, diagnostics and
  embeddings (tests). On CUDA the same holds whenever the kernels are deterministic
  (`--deterministic`); without it, a CUDA run is not bitwise reproducible from one run
  to the next with or without the flag (`cudnn.benchmark` and the atomics of some
  backward kernels), and the flag adds nothing beyond that run-to-run noise. The RNG state is preserved explicitly although the network has no dropout,
  so a stochastic layer added later stays exact.
- **Record and resume.** `to_plain` leaves the field out while off, so a run without it
  writes the record and the checkpoint config of before; on, `pilot.json`'s `config` and
  the checkpoints' carry `activation_checkpointing: true`. The flag changes memory, not
  numbers, so **a resume may turn it on or off** (`_EXECUTION_FIELDS`: the config
  comparison of `restore` leaves it out): a run started without it resumes with it after
  an out-of-memory, and the result equals the uninterrupted run (a test). The record and
  the checkpoints written after the resume hold the new value; the state dict is the same
  either way (the flag is an attribute, not a buffer).
- **Peak memory.** On CUDA each `metrics.jsonl` line adds `cuda_max_memory_allocated` and
  `cuda_max_memory_reserved` (bytes; the allocator's peaks since the process started, one
  `simclr-pilot` call), nothing on the CPU or MPS. A `--steps` dry run with `--log-every 1`
  sizes a batch before the real run; the reserved peak is closer to what `nvidia-smi`
  shows.
- **Golden set.** The CLAUDE.md checklist for a new training flag covers the plain
  engine's tasks (`tests/_engine_cases.py`, `Settings` fields); the pilot runs its own
  loop outside the engine (this ADR, section 1) and has no `Settings` field or config-tree
  home. The flag draws nothing (no stream) and changes no number, so a golden would be
  the flag-off golden again; its regression check is instead the bit-for-bit equality with
  the flag off, in the same test process (`tests/test_simclr_model.py`,
  `tests/test_simclr_pilot.py`). When 5.3 moves the pilot onto the engine, the flag joins
  the golden set with it.

**Cost** (CPU, a 4-thread step of the network and LARS on random views, interleaved
runs on a loaded laptop; GPU timings come from the 5090 dry run):

| Mode | Peak RSS growth, B = 32 / 64 (median of 3) | Step time vs off (paired ratios, 6 pairs) |
|---|---|---|
| off | 1.23 / 2.30 GiB | 1 (3.5-8.5 s at B = 32, 7.0-15.1 s at B = 64) |
| per block (chosen) | 0.84 / 1.56 GiB | median 1.4x (1.09x-1.90x) |
| per stage | 1.12 / 2.00 GiB | median 1.75x (0.96x-2.9x) |

The process's resident memory includes the weights, gradients, LARS momenta and the
allocator's slack, so it falls less than the activations alone (-32 % at B = 64 for per
block, against -61 % of the saved bytes), but the order is the analytic one: per block
below per stage below off. The time ratios scatter with the laptop's load (load average
10-15 during the runs); the arithmetic says about 1.3x for the encoder (one extra
forward on top of a forward and a backward), and the 5090 dry run gives the real figure.

## Amendment (2026-11-10, plan 5.4): the fine-tuning arm

**Context.** Neither more pretraining data (the corpus run, 0.568 / 0.721) nor batch 1024
(0.574 / 0.735) moved the SSL linear probe (0.571 / 0.734 at 10 / 100 per class on the
decision half; the scratch ResNet-18 0.434 / 0.680). The user decided (2026-11-10) to
pursue fine-tuning and the views; plan 5.4 reads H1 and H2 in fine-tuning mode. This
amendment adds the **exploratory** fine-tuning arm on the CD21 screen halves (selection /
decision), with the probe's and the scratch arm's label subsets; H1's pre-registered test
set is not touched.

### The arm (`simclr.finetune`, `simclr-finetune`)

- **Exactly the scratch arm, but the encoder's start.** `simclr-finetune DATA --checkpoint
  CKPT ...` runs `simclr-scratch`'s handler, recipe, grid (lr {5e-4, 2e-3} x weight decay
  {0.05, 0.5}: four trials, the probe's and scratch's trial budget), fixed step counts
  (1,000 / 2,000), final weights, selection rule and report shape. `ScratchTask` builds
  `ScratchNetwork` from the seeded generator as before (encoder, then head) and then
  copies the checkpoint's encoder state into the encoder: the encoder's random init is
  drawn and discarded, so the `Linear(512, 5)` head starts with the scratch arm's weights.
  The paired difference fine-tune - scratch therefore isolates the initialisation.
- **A command, not `simclr-scratch --init`.** The arm is a different experiment: its report
  says so (`report: simclr-finetune`, which `simclr-compare` prints), the resume identity
  of both commands now includes `report` (a scratch report is not resumed as a fine-tune,
  nor the reverse) and the fine-tune's adds `init_sha256` (another checkpoint is refused).
  Both commands share one handler and one argument set, so they cannot drift apart.
- **What is loaded.** Only the `encoder.*` entries of the checkpoint's `model`: the
  convolution and BatchNorm weights and the BatchNorm buffers. The projection head, the
  LARS state and the monitors are ignored. A file that is not a pilot checkpoint
  (`format`), of a version outside `finetune.KNOWN_CKPT_VERSIONS` (today 1), without
  `model`, `data` or `steps_done`, whose encoder keys, shapes or dtypes are not
  `resnet18_cifar()`'s, or with non-finite values is refused with the reason. The pilot
  must have been pretrained against the same layout (source SHA-256) **and the same
  splits file** (its SHA-256): a checkpoint pretrained without the splits file, or with
  another, did not leave out this screen's Train exclusions, and is refused.
- **What is recorded** (the report's `init`): the checkpoint's path, SHA-256, format
  version, `steps_done`, `total_steps`, `pretraining_complete`, its config, and the
  pretraining data: `kind` (`cd21-layout`, the pilot: **CD21 Train was in pretraining**,
  unlabelled, less the splits file's exclusions; or `archive-corpus`, the corpus run: it was
  not), the layout and splits SHA-256, `n_pretrain`, and for a corpus the index and
  exclusion SHA-256; `cd21_train_in_pretraining` says it in one field.
- **Randomness: stream 13 reused, deliberately** (an exception to "a new stream per
  flag"). The fine-tune draws its batches, views and noise from `draw_step` on stream 13
  with `subset_seed(seed, r)`, exactly as `simclr-scratch` does: at the same `--seed`,
  subset `r` sees the same batches, dihedral elements, offsets and noise at every step in
  both arms (**common random numbers**). The rule exists so that two random processes of
  one run are not correlated by sharing a stream; here no process is added, and the
  correlation is between two arms, where it is wanted: it removes the augmentation's and
  the batch order's noise from the paired difference, and the hierarchical paired
  bootstrap resamples whole subsets of paired outcomes, so the interval accounts for it.
  The pairing holds only at the same `--seed` (recorded in both reports); against the
  probe report, which draws nothing, the arms share only the subsets.
- **Golden set.** The trial runs on the plain engine, so it joins the regression set:
  `simclr-finetune-noise` (`tests/test_engine_regression.py`) is the scratch case's trial
  with the encoder from a tiny pilot checkpoint built deterministically in the test
  (`tests/_engine_cases.pilot_checkpoint_payload`: a seeded `SimCLRNetwork`, its
  parameters moved by seeded normals and its BatchNorm buffers by one train-mode forward).
  No `Settings` field: the arm is a command of the `simclr` package, as the scratch arm.
- **Checked by** `tests/test_simclr_finetune.py`: with a "checkpoint" equal to the scratch
  init, the fine-tune equals the scratch arm bit for bit (rows and final weights); with a
  pilot checkpoint the encoder at step 0 equals the checkpoint's and the head the scratch
  arm's; every refusal; determinism; the subset and the selection half alone reach
  training and selection (the rest poisoned with NaN); the command end to end from a real
  two-step `simclr-pilot` checkpoint, its resume identity, and its pairing in
  `simclr-compare` with the scratch and the probe reports, unchanged.
- **Cost:** the scratch arm's plus a checkpoint read: about 1.2 min per subset at 10 per
  class and 2.5 min at 100 on the RTX 5090, so about 25 and 50 minutes for 20 subsets.
- **Known limit.** The recipe is the scratch arm's on purpose, including its learning rates;
  fine-tuning often prefers a lower rate for the encoder than for the head. If the arm
  under-performs the probe, a fine-tuning grid (a lower encoder rate, a frozen-then-unfrozen
  schedule) is a separate, declared change, with its own trial budget.

## Amendment (2026-11-10, task 5.2): stronger views and their admission test

Neither 6x more unlabelled stamps (the corpus) nor batch 1024 moved the probe at 10 per
class (0.571 / 0.568 / 0.574); the user decided (2026-11-10) to pursue the views and
fine-tuning. In SimCLR the views are usually the strongest lever, and the pilot's are only
the 8 dihedral elements, a 57-pixel integer crop of the 63-pixel stamp and C5's noise.
This amendment adds four optional views of plan 5.2, each off by default with a strength
to sweep, and the plan's admission test. It takes no new ADR number: the views are section
2's pipeline extended, under the same rules.

### The views (`simclr.views`; `PilotConfig` and `simclr-pilot` flags)

| View | Flag (off at 0) | Per-sample parameter | Plan 5.2 |
|---|---|---|---|
| scale jitter | `--scale-jitter J` | zoom `s` log-uniform in `[1, 1 + J]` | ablation only |
| blur | `--blur-sigma S` (at most 10) | Gaussian `sigma = u S` pixels | ablation only |
| intensity | `--intensity-jitter G` | gamma, `ln gamma` uniform in `[-G, G]` | a candidate |
| rotation | `--rotation-degrees D` (at most 6.52) | `theta` uniform in `[-D, D]` | a candidate |

**Order** (section 2's list, extended): noise, blur, gamma at 63 pixels; the dihedral
element; the window (crop, or resampling with scale or rotation).

- **Noise first**, as before: its fit and `sigma_bg` describe the stored stamp. A blur
  before it would make them stale (a blurred stamp has a smaller background sigma).
- **Blur** next: linear and the same kernel on the three channels, so the difference's fit
  relation survives it (up to the min-max). It is a **normalised convolution**: the
  former-NaN pixels (several exact zeros in a channel, `stamp_relation.masked_pixels`) get
  weight 0, so they neither spread their 0 into the source nor take a value; the stamp is
  reflected at its edges; then the per-channel min-max (masked pixels out). After the
  noise, it smooths the noise too: a worse seeing of the observed stamp.
- **Gamma** last of the photometric views, because it needs [0, 1] input: `x ** gamma`,
  **the same gamma on the three channels**. A per-channel remap would be the hue or
  saturation the plan forbids and would break the science / template / difference
  relation. The remap must fix 0 and 1 so that the view keeps the stored stamps' min-max
  statistics: an affine contrast change is either undone by a min-max or moves the minimum
  and maximum off 0 and 1, which no evaluation stamp has; the gamma is the simplest
  one-parameter family of monotone maps that fix them. It is not linear, so it bends the
  fit relation: the admission test prices that.
- **Dihedral** element, exact, as before.
- **Scale and rotation** with the shift, as **one bilinear resampling** of the 63-pixel
  stamp (`window_transform`, `resample`): view pixel `q` (from the window's centre) reads
  `c + t + R(theta) q / s`. One interpolation, not two. The zoom is **magnification only**
  (as SimCLR's random resized crop, which only magnifies): a zoom out would need pixels
  beyond the stamp, a fill. The source stays at the centre for both.

**The rotation bound and the shift.** The centred 57-pixel window's corner `(28, 28)`
rotated by `theta` reaches `28 (cos theta + sin theta)` from the centre on one axis, and
must stay within 31 (pixel centres): `sqrt(2) 28 sin(theta + 45 deg) <= 31`, so `theta* =
asin(31 / (28 sqrt 2)) - 45 deg = 6.524 deg` (`rotation_bound`). But the window also
shifts by up to 3 pixels, which uses all of that margin: a fill-free rotation of a
window at offset 0 or 6 has `theta = 0` (and 2.08 / 4.25 degrees at shifts 2 / 1). Two
ways out: shrink the rotation per sample to its window's margin (then half of the samples,
those with a 3-pixel shift on an axis, never rotate, and the mean bound is about 1.5
degrees: a view too weak to test), or **shrink the shift to the rotation's margin**,
chosen: `t = (offset - 3) * min(1, (31 - e) / 3)` with `e = 28 (|cos theta| + |sin theta|)
/ s`. Every view pixel reads inside the stamp (no fill, tested on random draws and at the
bound); without rotation (`s >= 1`) the shift is the integer one, unchanged; at the bound
it is 0; in between it is fractional (bilinear). The integer offsets are drawn as before,
so turning the rotation on changes no other draw.

**Never** (plan 5.2): hue, saturation, greyscale, colour jitter, per-channel remaps, and
source-dropping crops (also random erasing / cutout: they can delete the source). Not
added: a zoom out (a fill), a contrast that moves the extremes (an input no evaluation
stamp has).

**Randomness.** Each view draws `u` (2, B) from its own host stream, `default_rng([seed,
step, n])` with `simclr_view_scale` 14, `simclr_view_blur` 15, `simclr_view_intensity` 16
and `simclr_view_rotation` 17 (numbers assigned by the integrator), **only when it is
on** (`draw_extra_params`). `simclr_views` (6) is still drawn in full every step. So one
view on or off changes no other view's numbers, and with all four off nothing new is
drawn: the pilot's draws and numbers are those of main before this amendment, bit for bit
(`tests/test_simclr_views_unchanged.py`: the views of a fixed batch with and without noise
and a 4-step CPU run with noise, its losses, weights and LARS state, against a reference
recorded from main 40b7470 on `darwin-arm64-default`; skipped on other platforms, as the
engine goldens). The parameters are host draws (device-independent); the arithmetic runs
on the training device and draws no per-pixel field.

**Record and resume.** `to_plain` leaves each strength out at 0, so a run without the
views writes the record and the checkpoint config of before; on, `pilot.json`'s `config`
and the checkpoints carry it. A `--resume` compares them (they change the numbers, unlike
`activation_checkpointing`): any change, on, off or of strength, is refused.

**Engine-free; the golden set.** The views reach no engine task: the scratch baseline
(`simclr.scratch`, the only one with a golden) and the fine-tuning arm that reuses its
draws (`simclr.finetune`) build their own `ViewConfig` from the noise alone and are
unchanged; the pilot runs its own loop (section 1). So the CLAUDE.md
checklist's golden step does not apply (as for activation checkpointing); the regression
checks are the bit-for-bit reference with the views off and, on, an exact resume and
determinism on the CPU (`tests/test_simclr_views_pilot.py`). When 5.3 moves the pilot
onto the engine, the views join the golden set with it.

### The admission test (`simclr.admission`, `simclr-view-admission`)

Plan 5.2: "a view whose single-view accuracy drop on Validation (replication checkpoint)
exceeds 1 point is not admitted". `simclr-view-admission RUN_DIR... --out FILE.json` reads
finished `train-supervised` runs of the paper classifier (task 2.4's replication, plain
`ckpt/best.pt` or a stored Lightning `best.ckpt`, through `ensemble_eval.member_net`),
checks that the layout is the data they trained on (data SHA-256), and evaluates each on
the layout's whole Validation split, as stored and with the view.

- **Once per stamp, seeded.** The view's parameters are the pilot's draws at "step" `r`
  (`--seed`, repeat `r` of `--repeats`, default 1; the first of the two views); the noise
  uses the Validation stamps' fits and a CPU generator of `simclr_view_noise`. The view
  arithmetic runs on the CPU, so the viewed stamps do not depend on the evaluation device.
  `--dihedral` and `--shift` test the pilot's fixed views too; flags combine (a combined
  view is tested as one).
- **A 57-pixel view against a classifier of 63-pixel stamps.** The paper model reads the
  centre 21 x 21 pixels of the 63-pixel stamp, re-normalised, not a 57-pixel window. The
  view is applied **at 63 pixels, keeping the size**: the photometric part as in training,
  the geometric part as the training window's map (the same zoom, angle and shift, the
  shift giving way to the rotation) resampled to a 63-pixel output centred where the
  training window's centre would be; then the run's own `crop_split` (its side and
  `nan_approximation`). The 21-pixel crop sees what the centre of a training view shows.
  The outer ring beyond the 57-pixel window, which the crop never reads, is clamped to the
  stamp's edge; its only reader is the NaN approximation's count of exact zeros over the
  stamp. Without a view the input is the run's own bit for bit, and the clean accuracy is
  its `val_acc_best` (tests).
- **Verdict:** `admitted` when the mean accuracy drop (clean minus viewed, over runs and
  repeats) is at most 1 point; the balanced accuracy is reported too (CD21's Validation is
  balanced, so they agree there). Exit 0 whatever the verdict.
- **Caveats, recorded.** A with-features checkpoint also reads the 23 alert features,
  which no view touches, so its drop understates the image's sensitivity: run the
  without-features replication as well (task 2.4 has both). The replication selected its
  weights on Validation, so the clean accuracy is optimistic; the drop compares the same
  stamps. The 21-pixel crop sees only the centre: a view that degrades the periphery
  passes more easily here than it would for the 57-pixel encoder.
- Refused: no view, `repeats < 1`, a run that is not a finished paper-classifier
  `train-supervised` run, `paper_input` other than `float32` or the context channels (an
  input the test does not rebuild), other data, an existing `--out` without `--overwrite`.

**Checks** (CPU): `tests/test_simclr_views_optional.py` (the draws, their streams and
ranges, the rotation bound, the no-fill rule on random draws and at the bound, the
identity at strength 0 and the resampling equal to the crop bit for bit, the same
transform on the three channels for the blur and the gamma, masked pixels kept out of the
blur), `tests/test_simclr_views_pilot.py` (each view: deterministic, an exact resume,
recorded, a changed strength refused; the CLI), `tests/test_simclr_views_unchanged.py`,
`tests/test_simclr_admission.py` (a tiny synthetic plain run: the input without a view,
each view deterministic per seed and repeat, the report's arithmetic and verdict, the CLI,
the refusals).

**Cost** (CPU, `make_views` of both views of a batch of 256, 4 threads, median of 10
interleaved repeats on a laptop at load average 37, so a rough figure): without options
19 ms; noise +94 ms, scale +20 ms, blur sigma 1 +139 ms and sigma 2 +230 ms (7 and 13
taps, twice for the normalised convolution's weights), gamma +32 ms, rotation +30 ms, the
four together +191 ms. The network's CPU step is several seconds at batch 32 already
(the activation-checkpointing amendment); on the GPU the views are elementwise work on
2 x 256 x 3 x 63 x 63 values, which the 5090 dry run of each flag will time.
