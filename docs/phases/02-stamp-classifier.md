<!-- Phase 2 of the roadmap (docs/phases/README.md); owned by the integrator. Reviewed with
/multi-agent-herdr:improve-plan (docs/plan-reviews/2026-11-08-phase-2.md). -->
# Phase 2: replicate and improve the stamp classifier

## Goal

The user's goal (2026-11-07) is to replicate the paper, not the 2022 legacy numbers, and then
improve on it. That gives two steps:

1. **Replicate** the ALeRCE stamp classifier of Carrasco-Davis et al. 2021 (AJ 162, 231;
   arXiv:2008.03309). The target is its published **test** accuracy, 0.941 ± 0.004 with
   the 23 alert features, mean ± std of 5 runs on the balanced 100-per-class test set.
   The without-features figure (0.883 ± 0.006) is advisory only: see task 2.3.
2. **Improve** it, starting with the architecture, through controlled comparisons. Each
   change is one flag, compared with the replication under the protocol of task 2.3.

The legacy port of phase 1 stays untouched. The paper-faithful model and every improvement
are new, named variants or flags. Out of scope: self-supervised pretraining (a later phase,
which reuses this phase's evaluation protocol), Rubin stamps and new data.

## Starting point (audit and plan review of 2026-11-07/08)

### Sources of truth

| Source | What it gives |
|---|---|
| The paper (arXiv:2008.03309v2) | The text. Where text and code disagree, **the code is the authority**, because it produced the published model |
| ALeRCE's TF1 code, vendored in `alercebroker/pipeline` @ `a3705ac`, `stamp_classifier_step/model/` | The implementation |
| **The M1 checkpoint** at `results/staging_model/DeepHits_EntropyRegBeta0.5000_batch64_lr0.00100_droprate0.5000_inputsize21_filtersize5_0_20200708-160759/` | The published model's weights, plus its validation event log (3,000 evaluations, every 10 iterations up to 30,000) |

Licence and safety: the pipeline repository has no LICENSE file. ALeRCE code and weights
are **never copied into this repository**. They are downloaded at use time, pinned by
commit and SHA-256. Downloaded `.pkl` files are never unpickled; read them, if at all,
with `pickletools`.

### Facts verified against the code and the checkpoint

- **Data.**
  - `red` is the paper's split: the post-cleaning counts 14,742 / 1,596 / 14,723 / 9,799 /
    10,734, minus 100 Validation and 100 Test per class, give 50,594. `red` has 50,582,
    so 12 stamps were removed as duplicates (6 VS, 6 Bogus).
  - The features are clipped per Table A1 and standardised on Train. For `isdiffpos`,
    M1's normalisation statistics give the same values (scaled only); task 2.1 checks the
    rest by de-standardising with M1's mean and std (integer `ndethist`, `ncovhist` and
    `non_detections`, a floor of −1 on the clipped `sgscore`/`distpsnr`, and {0, 1} for
    `isdiffpos`).
  - The 23 kept features and their order match ALeRCE's list.
- **Input.**
  - The pickles were written with `CROP_SIZE 63`, then a per-channel nanmin/nanmax
    normalisation (no epsilon), then NaN → 0.
  - The paper's model crops 21×21 at the centre, re-normalises per channel on the crop
    (no epsilon; a constant crop gives 0/0 → NaN → 0), sets NaN → 0, and zero-pads to
    27.
  - Min-max is affine-invariant, so re-normalising the stored float32 stamps is exact,
    except for the former NaN pixels.
- **Network.** M1 has 441,075 trainable parameters with features and 439,557 without.
  - conv1 4×4 valid, 32 filters;
  - conv2–5 5×5 "same" convolutions with 32 / 64 / 64 / 64 filters;
  - two 2×2 max-pools;
  - fc1 2304→64, ReLU, a dropout layer (rate 0.5), then the mean over the 4 rotations (no
    flips);
  - features through BN, then concatenated: 64 + 23 = 87;
  - fc2 87→64, ReLU, dropout; fc3 64→64, ReLU, dropout; 64→5 logits;
  - Glorot-uniform weights, zero biases.

- **Dropout was inactive in training** (plan review round 2; verified by the integrator in
  the meta graph). Keras `Dropout` is called without a `training` argument
  (`modules/layers.py:95-100`), so it follows `keras_learning_phase`, a
  `PlaceholderWithDefault` that defaults to False and is never fed. The dropout branches
  (`dense_{1,2,3}/dp/cond/Switch`) therefore never ran: **the effective dropout is 0 in
  training and in evaluation**. Consistent with this, the train-mode batch accuracy reaches
  0.986 late in training, and the dropout rate was inert in the code (graph), consistent with Table A2.
  Paper Table 2 shows dropout only on fc2 and "90×64"; both are paper errors, and the
  code wins.
- **Feature BatchNorm.** Its running statistics were **never updated**. In M1,
  `moving_mean` is all 0 and `moving_variance` all 1, while `gamma` and `beta` were trained:
  the Keras BN's update ops never reached TF1's `UPDATE_OPS`. So training used batch
  statistics, and evaluation and selection used `gamma·x/sqrt(1 + 1e-3) + beta`.
  Verified by the integrator.
- **Optimiser.**
  - Adam with **β = (0.9, 0.999)** and ε = 1e-8 (meta graph), and a constant lr of 1e-3,
    no weight decay. The paper text's (0.5, 0.9) are unused defaults. M1's
    `beta2_power = 0.999^4782` confirms β2 (verified by the integrator).
  - The loss is `CE − 0.5·H(softmax)`, with the gradient through both factors.
  - Batch 64.
  - Train balanced by deterministic tiling: full copies plus a prefix, up to the
    largest class.
- **Schedule and selection.**
  - The validation loss is computed every 10 iterations, and the best checkpoint is
    kept.
  - Every new best at iteration i sets horizon = max(horizon, i + 10,000), starting from
    30,000.
- **M1's actual training curve:**
  - best validation loss −0.0616 at **iteration 4,780**, with validation accuracy
    **0.950**;
  - validation accuracy moves by up to 1.4 points between consecutive evaluations, peaks
    at 0.954 (iteration 3,120), and drifts to about 0.92 after 10k;
  - the run ended at 30,000 iterations without any extension.

  So the outcome is set by the first ~8k iterations and by which noisy checkpoint wins.
- **Reference statistics.** ALeRCE's tables use `np.std` (ddof = 0). σ_ref = 0.004 is
  therefore about 0.0045 at ddof = 1.
- **The without-features reference** (0.883 ± 0.006) has no stated hyperparameters. Its
  search script uses a different split (60 per class) and the `DeepHitsExtraLayer`
  network, which is the same model without the BN and concat. Assumption: M1's
  hyperparameters on the 100-per-class split.

### Differences from our phase-1 port

| Item | Paper code (authority) | Phase-1 port |
|---|---|---|
| Input | float32, crop 21 + per-crop min-max + pad to 27 | uint8, min-max over 63×63, crop of 27 real pixels |
| Dropout | layers present, **inactive** (effective rate 0) | 0.2 after fc2, active |
| Feature BN | running stats frozen at (0, 1), eps 1e-3 | torch default (momentum 0.1, eps 1e-5) |
| Balance | tiling | none |
| Adam | (0.9, 0.999), lr 1e-3 | (0.5, 0.9) |
| Selection | validation loss every 10 iterations, horizon rule | per epoch |
| Init | Glorot-uniform, zero bias | torch default |

The phase-1 flaw F1 (uint8 quantisation erases the template background) and the optimistic
legacy evaluation are documented in the audit. The phase-1 data caveats (oversampled
`sup_*`; one `red` Train stamp also in Validation or Test) still apply.

### Already running (interim, 2026-11-08)

A "paper hyperparameters on the phase-1 port" run (5 seeds, GTX 1050, `red` tiled in a
derived prepared copy) gives an early data point. It is not the replication and does not
gate anything. It also uses active dropout, which M1 did not have, so it is not
informative about dropout.

## Machines

- **GTX 1050** (4 GB, WSL2, `cu126`). The phase-1 preflight passes there.
- **RTX 5090** from 2026-11-08 (default CUDA 13 build).
- **Mac M4** (MPS).

Every run's metrics JSON records the GPU, torch and CUDA versions. Runs in one comparison
use the same machine and build.

## Tasks

The critical path to the first replication number is:

{2.1 ∥ 2.2a ∥ 2.3} → 2.2b (M1 oracle, no GPU) → 2.2c → 1-seed pilot → 2.4.

Phase-1 task 1.9a (cloud runbook) is needed only for cloud GPUs. Each task is one PR,
except the manual runs, and the phase-1 portability rule applies to all of them.

### 2.1 Float input path (now) — done (PR #12, merged as 4059070, 2026-11-08)

- `prepare-data --images float32` also writes the stored float32 stamps as
  `images_f32.npy`. `meta.json` moves to format version 2; readers accept versions 1 and
  2, and the file may be absent, so the uint8-only path still works.
- The manifest changes. The preflight names the command that regenerates it.
- `crop_then_normalise(stamp, crop=21)`: centre crop, per-channel min-max on the crop
  **without epsilon** (a constant crop gives 0), NaN → 0.
- The NaN approximation. The stored NaNs are 0, the same as the minimum. Zero pixels count
  as former NaNs when the channel, **counted over the 63×63 stamp**, has more than one exact
  zero. It is a flag, on by default, and documented.
- **Acceptance:**
  - Golden `.npz` fixtures generated **once by running ALeRCE's own
    `ztf_preprocessor` functions** (`image_crop_at_center`, `image_normalize_by_image`,
    `image_nan_to_num`, all plain numpy) on synthetic stamps that include NaNs, constant
    channels and all-NaN crops. Commit the fixtures and the generator script, not the
    ALeRCE code.
  - An "alignment check" that the uint8 and float crops cover the same pixels.

### 2.2a Paper-faithful model, schedule and sampler (now; synthetic data only)

- `PaperStampClassifier`, a new class:
  - `ZeroPad2d(3)`;
  - dropout layers after fc1 (per rotation, before the mean), fc2 and fc3, with
    `dropout_active=False` as the faithful default (M1 trained with dropout inactive);
    `dropout_active=True, rate=0.5` is diagnostic 1 of 2.4;
  - feature `BatchNorm1d(23, eps=1e-3, momentum=0.0)`, so the running statistics stay at
    (0, 1);
  - Glorot-uniform weights and zero biases;
  - no flips.

  A `with_features=False` variant matches ALeRCE's `DeepHitsExtraLayer`.
- A deterministic tiling sampler: full copies plus a prefix, exactly as ALeRCE's
  `_oversampling_array`.
- The iteration schedule:
  - `max_steps` starts at 30,000;
  - validation every 10 iterations on the GPU; the selection loss is TF's **mean of
    batch-64 means in `red` file order** (the same definition in 2.2a and 2.2b);
  - the best `state_dict` is kept in memory and written once at the end;
  - a new best at iteration i raises the horizon to i + 10,000;
  - the best is restored at the end.
- Speed: the 21×21 float crops are precomputed once per run (about 270 MB) and kept on the
  device, with batches drawn by index. No per-item DataLoader on the hot path.
- **Integration with the CLI (`experiments.run_command`).**
  - A `DeviceTensorDataModule` yields batches from an index sampler, already on the
    device, with `num_workers=0` enforced.
  - The Trainer runs `max_steps=30000, val_check_interval=10, check_val_every_n_epoch=None`.
  - There is **no EarlyStopping** under `schedule: carrasco2021`.
  - A `HorizonCallback` raises `fit_loop.epoch_loop.max_steps`, keeps the best weights and
    the horizon, and saves both in its `state_dict()`, so `--resume` restores them.
  - At the end it writes `best.ckpt` in Lightning format, so the metrics path works
    unchanged.
  - New metrics `best_step` and `last_step`, read by `compare-reference`.
  - New validated `Settings` fields: `max_steps`, `val_every_n_steps`,
    `horizon_increment`, `early_stopping`.
  - The `HorizonCallback` exposes `best_model_path`, `best_model_score` and `best_step`,
    and `run_command` branches on the schedule.
  - An "epoch" is one pass over the tiled indices (1,136 steps), so the epoch-end hooks and
    metric buffers fire.
  - `last.ckpt` is written with `every_n_train_steps` (for example 1,000).
  - The preset pins `eval_batch_size=64`, needed for TF's mean of batch means, and an
    override is an error.
- **Acceptance:**
  - A resume from a `last.ckpt` written mid-epoch restores the horizon and the best weights.
  - `run_command` produces `metrics.json` for the preset.
  - In train mode with `dropout_active=False`, the outputs are deterministic for a fixed
    input.
  - **441,075** trainable parameters with features and **439,557** without.
  - The BN running buffers stay at (0, 1) after training steps.
  - The sampler gives exact class balance and the documented ordering.
  - Schedule unit tests: the horizon extends, the best is restored, and validation runs
    every 10 iterations.
  - A CPU smoke run on synthetic data.

### 2.2b M1 oracle (after 2.1 and 2.2a; no GPU)

- A script downloads M1 at the pinned commit, checks its SHA-256 and converts it to a
  local `.npz` that is never committed. It uses a pure-Python TF-checkpoint reader or a
  throwaway TF environment.
- The weight mapping: HWIO kernels become OIHW, dense kernels are transposed, and fc1's rows
  are permuted from TF's (h, w, c) flatten order to PyTorch's (c, h, w).
- **Two separate checks:**
  1. **Port equivalence.**
     - An independent NumPy forward pass in NHWC on the raw TF-layout weights: no
       transposes, symmetric "same" padding.
     - It runs on the same inputs as the torch model.
     - Acceptance: max |Δlogit| ≤ 1e-4, and 100 % argmax agreement on all 500 Validation
       stamps. A miss here is a mapping or architecture bug.
  2. **Data path.**
     - M1 is run through `PaperStampClassifier` on `red` Validation, using the float path.
     - Acceptance: accuracy = 475/500 ("exact"), or within ±2 stamps; validation loss
       −0.0616 ± 0.005 (mean of batch-64 means, `red` file order).
     - The stamps flagged by the NaN approximation (about 27) are reported separately.
- Reported:
  - M1's test accuracy;
  - the accuracy change with the NaN approximation off and with the uint8 path (these price
    diagnostics without training);
  - the M1 validation curve, saved as `results/phase2/replication/m1_val_curve.csv`
    (derived numbers only).

### 2.2c Preset `carrasco2021` (after 2.2b)

- CLI wiring of `train-supervised --preset carrasco2021`:
  - the float path and the tiling;
  - Adam (0.9, 0.999), ε = 1e-8, lr 1e-3;
  - β = 0.5, batch 64;
  - the schedule of 2.2a;
  - test on the selected checkpoint.
- `adam_betas: paper` (0.5, 0.9) is available only as a diagnostic.
- ADR `0003-carrasco2021-preset.md` records:
  - the code-is-authority rule and the frozen BN;
  - the betas and the Adam ε placement (TF1 vs torch);
  - the without-features assumption;
  - the NaN approximation.
- **Acceptance:**
  - A **1-seed pilot in under 60 minutes on the GTX 1050** (or the 5090's time recorded),
    with seconds per iteration
    logged.
  - The best validation loss occurs before iteration 10,000, with validation accuracy
    ≥ 0.94 near the best.

### 2.3 Evaluation protocol (now) — done (PR #13, merged as 71980de, 2026-11-08). Real-data finding: `red` Train 2995 (labelled Bogus) is the same stamp as Validation 115 (labelled SN); excluded from Train only.

- Every run reports:
  - **test** accuracy of the checkpoint selected on validation;
  - validation accuracy, macro-F1 and per-class recall;
  - the confusion matrix;
  - the machine, torch and CUDA versions;
  - mean ± std (ddof = 1) over **5 seeds**.
- **Decision protocol (no selection bias).** Decisions never use the metric of the stamps
  that chose the checkpoint.
  - **Default for 2.5: stratified 5-fold CV on Train ∪ Validation** (seed 0, stored).
    - Each fold holds out an evaluation part.
    - Each fold carves its own checkpoint-selection set (100 per class) from its
      training part.
    - Tiling happens only within the training part.
    - The metric is balanced accuracy on the held-out part, using 2 seeds per fold.
  - **Adoption rule:** a flag is adopted only if the paired-bootstrap 95 % CI of the CV
    difference against the baseline excludes 0. Section 2.6 combines only adopted flags.
  - **Screen:** a fixed 250/250 split of Validation (selection / decision, 50 per class,
    seeded). It can only **drop** a variant, when that variant is worse by 4 points or more.
    It never adopts one: its MDE is about 4 points.
  - The replication itself (2.4) keeps the paper protocol (all 500 Validation stamps for
    selection), so it stays comparable with 0.941.
- **Rule: test metrics are never read to choose a variant, a diagnostic or a combination.**
  Every keep/drop decision uses validation (mean over 5 seeds), the M1 oracle, or a
  cross-validation run. Test is computed, archived, and reported once per pre-registered
  comparison.
- `compare-reference --ref CD21`, a two-tier gate, with features:
  - **faithful:** |mean − 0.941| ≤ 0.005, **and** the M1 oracle passed. Note: with σ ≈
    0.0045 on both 5-run means, a perfect replication fails this tier about 8 % of the
    time, so one confirmatory 5-seed rerun is allowed;
  - **acceptable:** |mean − 0.941| ≤ 0.01.

  Both tiers also require:
  - SN recall within ±4 points of 87 %;
  - mean validation accuracy within ±0.01 of 0.950.

  The without-features reference (0.883) is advisory: reported, not gating.
- Comparisons of two variants, both reported:
  - **McNemar on seed-ensembled predictions per stamp** (average the 5 seeds' probabilities
    first). Pooling seed × stamp pairs is not allowed.
  - A **paired bootstrap over stamps**: resample the 500 test stamps, averaging over seeds.
- Power note in the docs:
  - with σ_seed ≈ 0.004, the minimum detectable effect of the seed noise is about 1.2
    points with 3 seeds and about 0.8 with 5;
  - the test-sampling noise on 500 stamps is about 1 point of standard error for the
    difference, so effects under about 2 points are "on this test set" only;
  - claims below 2 points need k-fold CV on Train ∪ Validation with balanced accuracy,
    about 5× the compute (the 5090).
- Leakage: the one `red` Train stamp also found in Validation or Test is removed **from
  Train only**. Validation and Test stay at 500, as in the paper.

### 2.4 Replication runs (after the 2.2c pilot; manual)

- `carrasco2021` with features, 5 seeds, plus without features, 5 seeds (advisory). Run on
  one machine: the 5090 if available, otherwise the 1050 overnight (about 45 minutes per
  run).
- If the gate fails, the diagnostics are priced first with the M1 oracle (no training),
  then screened with the 250/250 split. A diagnostic is adopted only if a re-check of the
  gate passes, which counts as the confirmatory rerun. The candidates, in order:
  1. dropout active (0.5);
  2. `adam_betas: paper`;
  3. without the NaN approximation;
  4. BN momentum 0.01;
  5. selection on validation accuracy.
- Results go to `results/phase2/replication/`: metrics JSON, figures and the M1 curve. No
  checkpoints.

### 2.5 Improvements (after 2.4 reaches at least "acceptable")

Decided: architecture first (see Decisions). The tables below separate the two lines.

Every row is one flag with 5 seeds against the replication, decided by the protocol of
2.3, with test archived. If 2.5 runs on another machine than 2.4, the 5-seed baseline is
re-run there first (budgeted).

**2.5-arch: architectural changes (the user's main interest).**

| Id | Change | Hypothesis |
|---|---|---|
| B1 | Field of view: 21 + pad / 27 / 41 / 63 and multi-scale 4×16² (Reyes-Jainaga 2023); fc1 sized from the input. 41/63 px only on the 5090, or with a per-batch path (the crops do not fit next to the model on 4 GB) | More context helps SN (the host galaxy) |
| B2 | D4 invariance: 8 copies (rotations × flips), or p4m group-equivariant convolutions | An exact sky symmetry |
| B3 | Batch norm in the conv trunk | Optimisation stability |
| B4 | Width / depth (filters ×1.5–2; one more conv block) | Capacity |
| B5 | Residual blocks; global-average pooling instead of fc1 | A modern inductive bias, fewer parameters |
| B6 | Rotation averaging before the activation | A different invariance point |

**2.5-input/training: data and training changes.**

| Id | Change | Hypothesis |
|---|---|---|
| C1 | Robust normalisation (asinh/MAD, or min/p99 with a ±2 clip) plus a NaN-mask channel | Keeps faint structure and relative scale |
| C2 | Restore the `magnr` and `distnr` features | Brightness context lost by min-max |
| C3 | Active dropout (a genuine change: M1 had none) and β tuning; the calibration error (ECE) | Regularisation, calibration |
| C4 | Unbalanced training with post-hoc logit adjustment | Prior shift without tiling; needs its own unbalanced 5-seed baseline |

**Reference line, not an ablation:** a seed ensemble of 5 runs (no extra training).

FFT input (Carvajal et al. 2025) is left out: its gains are on satellites, which this
5-class set does not have.

### 2.6 Combined model (after 2.5)

- The combination is **pre-registered from validation results**: the flags that improved
  mean validation accuracy and are compatible. It is run with 5 seeds, and test is
  reported **once** with the paired tests of 2.3.
- `docs/results-phase2.md` holds the tables, the confusion matrices and a discussion that
  can be reused in a paper.

## Risks

- **The replication misses.** The M1 oracle separates input or model errors from training
  dynamics, so the cause is found without extra GPU runs.
- **Effects below the noise.** 500 test stamps cannot support claims under about
  2 points. The rule is 5 seeds, the correct tests, and CV for small effects.
- **Compute.**
  - About 45 minutes per run on the 1050, so the 10 replication runs take about 7–8 h.
  - The architectural line under the CV protocol costs about (6 flags + baseline) × 5
    folds × 2 seeds ≈ 70 runs, so ~53 h on the 1050. It runs **on the 5090**, where the
    GPU-resident pipeline of 2.2a should cut it several times. The 250/250 screen runs
    first, to drop clear losers cheaply.
  - The input/training line (2.5-input/training) runs only if needed.
  - B1 at 63 px costs about 5× the pixels, and B2 2× the copies.
- **Mixed machines.** The 1050 (`cu126`) and the 5090 (CUDA 13) differ numerically, so a
  comparison never mixes them.

## Decisions (2026-11-08; the user delegated them to the integrator)

- Scope of 2.5: **architecture first** (2.5-arch), as the user asked. 2.5-input/training
  runs only if the architectural line plateaus below the target.
- The without-features gate is **advisory**.
