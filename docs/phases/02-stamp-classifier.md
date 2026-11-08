<!-- Phase 2 of the roadmap (docs/phases/README.md); owned by the integrator. Draft: not yet
reviewed by /multi-agent-herdr:improve-plan. -->
# Phase 2: replicate and improve the stamp classifier

## Goal

The user's research direction (2026-11-07) has two steps:

1. **Replicate** the ALeRCE stamp classifier of Carrasco-Davis et al. 2021 (AJ 162, 231;
   arXiv:2008.03309). The target is its published **test** accuracy: 0.941 ± 0.004 with
   the 23 alert features and 0.883 ± 0.006 without them, as the mean ± std of 5 runs on
   the balanced 100-per-class test set.
2. **Improve** it through controlled ablations. Each change is one flag, compared with the
   replication under the same protocol.

The replication is the baseline any future paper needs, whether supervised or
self-supervised. The legacy port of phase 1 stays untouched: the paper-faithful model and
every improvement are new, named variants or flags, never silent changes.

Out of scope: self-supervised pretraining (a later phase, which uses this phase's
evaluation protocol), Rubin stamps, and new data.

## Starting point (audit of 2026-11-07)

These facts come from a line-by-line audit of our code against the paper and against the
original ALeRCE TF1 code, which is vendored in `alercebroker/pipeline` @ `a3705ac`, under
`stamp_classifier_step/model/`, together with the M1 checkpoint of the paper. The
integrator spot-checked them (Adam call, padding, dropout, pooling).

- **Data match.** Our `red` file is almost certainly the paper's exact split. Its Train
  counts equal the paper's post-cleaning counts minus 100 Validation and 100 Test per
  class, except 6 de-duplicated VS and Bogus stamps. Validation and Test are balanced at
  100 per class (500 each). Features are already clipped per Table A1 and standardised on
  Train, and the 23 kept features match ALeRCE's list and order.
- **Already faithful:** the conv trunk (conv1 4×4 valid 32; conv2–5 5×5 same 32/64; two
  max-pools; fc1 2304→64), the 4-rotation mean after fc1 + ReLU, and `CE − β·H` with the
  gradient through both factors.
- **Differences from the paper (and from ALeRCE's code):**

  | Item | Paper / ALeRCE code | Ours (phase-1 port) |
  |---|---|---|
  | Input | float32; centre-crop 21×21, then per-channel min-max **on the crop**, NaN → 0, zero-pad to 27 | min-max over the full 63×63, quantised to **uint8**, crop 27×27 of real pixels |
  | Dropout | 0.5 after fc1 (per rotation, before the mean), fc2 and fc3 | 0.2 after fc2 only |
  | Class balance | Train replicated per class to the largest class | none (SN is 2.8 % of Train; the evaluation sets are balanced) |
  | Optimiser | Adam, lr 1e-3, constant, no WD. The paper text says β = (0.5, 0.9); **the code uses (0.9, 0.999)** | AdamW / Adam (ambiguous), lr 0.0025 |
  | Batch / β | 64 / 0.5 | 45 / 0.016 |
  | Selection | best **validation loss** (CE − βH), checked every 10 iterations; 30k iterations plus 10k extensions | best validation **accuracy** per epoch, patience 40–70 |
  | Reported | **test**, mean of 5 runs | **validation**, best epoch (0.9496 is optimistic and not comparable) |
  | Init / feature BN | Glorot-uniform, zero bias; Keras BN (eps 1e-3, momentum 0.99) | torch defaults |

- **Confirmed flaw F1 (uint8).** Normalising over 63×63 and then rounding to 256 levels
  erases the background of the central crop whenever a bright source sits elsewhere in
  the stamp. Measured on 1,000 Validation + Test stamps, the template background has σ < 1
  level in 48 % of stamps (VS 83 %, Bogus 66 %); with the paper's float crop-then-normalise
  it is 18.8 %. The difference channel is mostly unaffected.
- **Confirmed flaw F10 (optimistic evaluation).** Every legacy number is the max over
  epochs on the same 500-stamp validation set that chose the hyperparameters. There is
  no legacy test number.
- **Shared with the paper:** per-stamp min-max erases relative flux between science and
  template. It can be changed (task 2.5), not fixed.
- **Data caveats** (task 1.2): `sup_1` and `sup_10` are oversampled draws (`sup_1` has
  14 distinct SN stamps). One `red` Train stamp also appears in Validation or Test.

## Dependencies

- **Phase 1:** tasks 1.6 (Lightning modules) and 1.7 (CLI `train-supervised`, metrics
  JSON, `compare-reference`, preflight). Phase 2 extends those. It does not fork them.
- **A CUDA GPU** for the runs: Kaggle, Colab, the lab server or a rented card (plan 1,
  open question "GPU for 1.9b"). The phase-1 runbook (task 1.9a) applies.

## Tasks

Each task is one PR, except the manual runs. The portability rule of phase 1 applies to
all of them.

### 2.1 Float input path (after 1.7)

- `prepare-data --images float32` also writes the stored float32 stamps, untouched, as
  `images_f32.npy` (63×63×3; about 2.4 GB for `red`). The uint8 layout and the legacy
  path stay as they are.
- A `crop_then_normalise(stamp, crop=21)` transform: centre crop, per-channel min-max on
  the crop, NaN → 0.
- **NaN approximation.** The pickles already hold `nan_val = 0`, which cannot be told
  apart from the minimum. Zero pixels in a channel with more than one exact zero (5.5 % of
  stamps) are treated as former NaNs. This is a documented approximation, with a test on
  synthetic NaN stamps.
- **Acceptance:**
  - Characterization tests against ALeRCE's `ztf_preprocessor` crop and normalisation on
    synthetic stamps, reimplemented in the test from the vendored code.
  - The float path and the uint8 path give the same crop when quantised.

### 2.2 Paper-faithful model and preset `carrasco2021` (after 2.1)

- `StampClassifier` variant (or `PaperStampClassifier`):
  - `ZeroPad2d(3)` (21 → 27);
  - Dropout(0.5) after fc1 per rotation before the mean, after fc2 and after fc3;
  - feature BN with eps 1e-3 and momentum 0.01;
  - Glorot-uniform weights with zero biases;
  - no flips.
- Training preset:
  - `P_stamps_loss(beta=0.5)`;
  - Adam, lr 1e-3, constant, no weight decay;
  - `adam_betas: code | paper`, default `code` = (0.9, 0.999), the published model; the
    discrepancy is recorded in an ADR;
  - batch 64;
  - Train balanced by **deterministic tiling** to the largest class, then shuffled.
- **Iteration-based schedule:**
  - 30,000-iteration horizon;
  - validation loss (CE − 0.5·H, eval mode) every 10 iterations, best checkpoint kept;
  - a new best at iteration *i* sets horizon = max(horizon, *i* + 10,000);
  - the best checkpoint is restored at the end.
- **Acceptance:**
  - Parameter count equals the TF model.
  - Forward-shape tests.
  - The tiling sampler yields exact class balance.
  - Schedule unit tests: the horizon extends, and the best checkpoint is restored.
  - A CPU smoke run on synthetic data.

### 2.3 Evaluation protocol (after 1.7; can run in parallel with 2.1–2.2)

- Every phase-2 run reports **test** accuracy of the checkpoint chosen on validation,
  plus validation accuracy, macro-F1, per-class recall and the confusion matrix, as the
  mean ± std (ddof = 1) over seeds.
- `compare-reference` gains the reference `CD21`:
  - with features: 0.941 ± 0.004;
  - without features: 0.883 ± 0.006;
  - 5 seeds each;
  - gate: `|mean − ref| ≤ max(2·sqrt(σ_ref²/5 + σ_new²/5), 0.01)`.
- Paired comparisons for the ablations: the same seeds, the difference of means with its
  std, and McNemar on pooled test predictions.
- Power note in the docs: 500 test stamps and seed std ≈ 0.004 give a minimum detectable
  effect of about 0.7–1 point with 3 seeds. Effects below 1 point need 5 seeds.
- Leakage: the one `red` Train stamp found in Validation or Test is removed for phase-2
  runs, and the removal is reported.

### 2.4 Replication runs (after 2.2 and 2.3; manual, GPU)

- `carrasco2021` with and without features, 5 seeds each (10 runs). Gate with `CD21`.
- If the gate fails, the planned diagnostics, in order:
  1. `adam_betas: paper`;
  2. without the NaN approximation;
  3. the legacy 27-px crop with float input;
  4. selection on validation accuracy.

  Each diagnostic is reported, never silently adopted.
- Results go to `results/phase2/replication/` (metrics JSON and figures, no
  checkpoints).

### 2.5 Ablations (after 2.4 passes; each a flag, one PR per group, runs manual)

Every row is compared with the replication, using paired seeds (3, or 5 when the effect
is below 1 point). The order is the audit's ranking.

| Id | Flag / change | Hypothesis | Estimate (to confirm) |
|---|---|---|---|
| A1 | Post-hoc logit adjustment `logits + log(π_bal/π_train)` on the **unbalanced** training variant | Fixes the prior shift without tiling | SN recall +5–10 pts vs unbalanced |
| A2 | Robust normalisation: asinh((x − median)/(a·MAD)), or RJ23 min/p99 with a ±2 clip, plus a NaN-mask channel | Keeps faint structure and relative scale | +0.3–1 pt |
| A3 | Field of view 21+pad / 27 / 41 / 63 and multi-scale 4×16² (Reyes-Jainaga 2023); `fc1` sized from the input | More context helps SN (host galaxy) | ±0.5–1.5 pts, sign unknown |
| A4 | D4 invariance: 8 copies (rotations × flips); flips as test-time augmentation | Exact sky symmetry | +0–0.5 pt |
| A5 | Seed ensemble (average the probabilities of the 5 seeds) | Variance reduction, zero extra training | +0.3–0.8 pt |
| A6 | Restore the `magnr` and `distnr` features (columns 14–15) | Brightness context lost by min-max | +0–1 pt |
| A7 | Dropout and β tuning; report the calibration error (ECE) | Regularisation, calibration | accuracy ±0.3; calibration large |
| A8 | Pre-activation rotation averaging | A different inductive bias | ~0 ± 0.3 pt |

FFT input (Carvajal et al. 2025) is left out: its gains are on satellites, which this
5-class set does not have.

### 2.6 Combined model (after 2.5)

- Combine the ablations that helped, as long as they are compatible. Run 5 seeds and
  report against the replication with the paired tests.
- `docs/results-phase2.md` holds the table, the confusion matrices and a short discussion
  that can be reused in a paper.

## Risks

- **The replication misses the target.** The NaN approximation, TF vs PyTorch numerics
  and the Adam-betas ambiguity could cost a point. The planned diagnostics in 2.4 cover
  them, and a documented 0.93x with the reasons is still a usable baseline.
- **Effects below the noise.** 500 test stamps cannot resolve effects under about
  0.7 pt. Use 5 seeds and paired tests, and do not over-claim.
- **Compute.** About 10 replication runs plus about 8 ablations × 3–5 seeds, so roughly
  35–50 runs of a small CNN. The tiling makes an epoch about 72k stamps.
- **Scope creep into self-supervised learning:** it is kept for a later phase.

## Open questions for the user

- Default `adam_betas`: `code` (0.9, 0.999, what produced the published model) or `paper`
  (0.5, 0.9, what the text says)? The proposal is `code`, and an ADR records the choice.
- Should the replication run on 21+pad, as the paper does, even though the legacy
  experiments used 27? The proposal is 21+pad for the replication, and 27 as ablation A3.
- Which GPU runs 2.4–2.6 (the same question as phase 1, task 1.9b)?
