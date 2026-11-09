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
  - Glorot-uniform weights; zero biases on the convolutions and the logits, but
    **Glorot-uniform biases on fc1–fc3**, U(±0.2165): ALeRCE passes
    `bias_initializer=None` and Keras falls back to `glorot_uniform` (M1 graph:
    `dense_*/bias/Initializer/random_uniform`; found in task 2.2a).

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

### 2.2a Paper-faithful model, schedule and sampler (now; synthetic data only) — done (PR #14, merged as 0ce8c77, 2026-11-08)

- `PaperStampClassifier`, a new class:
  - `ZeroPad2d(3)`;
  - dropout layers after fc1 (per rotation, before the mean), fc2 and fc3, with
    `dropout_active=False` as the faithful default (M1 trained with dropout inactive);
    `dropout_active=True, rate=0.5` is diagnostic 1 of 2.4;
  - feature `BatchNorm1d(23, eps=1e-3, momentum=0.0)`, so the running statistics stay at
    (0, 1);
  - Glorot-uniform weights; zero conv/logits biases, Glorot-uniform fc1–fc3 biases;
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

### 2.2b M1 oracle (after 2.1 and 2.2a; no GPU) — done, **PASSED** (PR #16, merged as 15f6427, 2026-11-08)

Results (`results/phase2/replication/m1_oracle.json`):
- **Port equivalence:** max |Δlogit| 1.6e-6 and 100 % argmax agreement.
- **Data path:** 475/500 exact, loss −0.06162. The 41 NaN-approximated stamps are all
  correct.
- **M1 test accuracy: 0.936** (SN recall 0.86). The paper's 0.941 is a 5-run mean.
- **Pricing of the input paths on M1, validation:**

  | Input path | Change |
  |---|---|
  | no NaN approximation | −0.2 pt |
  | uint8 21 + pad | −2.2 pts |
  | **uint8 27 px, the phase-1 path** | **−4.0 pts** |

  This confirms the phase-1 input flaw F1.

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

### 2.2c Preset `carrasco2021` (after 2.2b) — done (PR #15, merged as da3e3ee, 2026-11-08)

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
  - **Pilot result (GTX 1050, seed 0, initial horizon 40 by mistake, so no 30k minimum):**
    - 25.7 min (0.050 s/step);
    - best validation loss −0.0688 at iteration 20,770, validation accuracy 0.942, test
      0.926.
  - **"Best before 10k" was a wrong criterion.** The training-dynamics diagnostic
    (2026-11-08) found no systematic difference from M1:
    - data pipeline, loss, optimiser, BN, dropout and precision all match TF1, item by
      item;
    - 3 MPS seeds track M1's curve through 6k iterations and reach −0.063 to −0.068 by
      ~5k;
    - M1's "best at 4,780" is a noise spike on 500 validation stamps.
  - Replication seed 0 with the full preset: **test 0.942**, validation 0.946.

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

### 2.4 Replication runs (after the 2.2c pilot; manual) — with features done, **CD21 PASS (faithful)** (2026-11-08); without features done (advisory: 1.06 points low)

**Result (with features, GTX 1050, `cu126`, 5 seeds, code at 510395e, PR #15's first commit; `src/` differs from its merge da3e3ee only by the M1 oracle):**

| Seed | best_step | val_acc_best | test (at best) |
|---|---|---|---|
| 0 | 18,930 | 0.946 | 0.942 |
| 1 | 4,210 | 0.940 | 0.946 |
| 2 | 4,540 | 0.940 | 0.940 |
| 3 | 4,430 | 0.944 | 0.938 |
| 4 | 6,330 | 0.944 | 0.948 |
| **mean ± std (ddof=1)** | | **0.9428 ± 0.0027** | **0.9428 ± 0.0041** |

`astro-simclr compare-reference ... --ref CD21 --oracle-report results/phase2/replication/m1_oracle.json`
exited 0 with `CD21: PASS (faithful)`: |Δ| = 0.0018 against the paper's 0.941 ± 0.004;
SN test recall 0.872 ± 0.015 (needs 0.87 ± 0.04); validation 0.9428 (needs 0.950 ± 0.01);
M1 oracle passed. About 26 minutes per run (0.050 s per step). The best steps (4.2k–6.3k,
one outlier at 18.9k) match M1's best near 4.8k. No diagnostic was needed, so the list
below stays unused. Archived in `results/phase2/replication/` (PR #19).

**Fixed defect (PR #20):** `compare-reference` warned `paper_arch=None (expected 'rot4')` and the same for `context` and `rot90_augment` on runs made before task 2.5-models. A missing key should read as its faithful default. The verdict is not affected.

**Result (without features, advisory; same machine, main @ 695ed93):** test 0.870 / 0.870 /
0.872 / 0.888 / 0.862, so **0.8724 ± 0.0095** against the paper's 0.883 ± 0.006
(|Δ| = 0.0106, just outside "acceptable"); validation 0.882 / 0.880 / 0.896 / 0.886 /
0.888; best steps 2.1k–6.9k. `compare-reference` prints `CD21: REPORT` (advisory, exit 0).
The likely cause is the preset's documented assumption: `carrasco2021-no-features` reuses
M1's hyperparameters, while the paper's without-features model was tuned separately and
there is no checkpoint to check it against. The gap is about 2.5 standard errors of the
5-seed mean. It does not block 2.5, which is with features. Open, low priority: tune β
and the learning rate for the without-features model on the screen, if the paper's
without-features number is ever needed.


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

**Status:** unblocked by the 2.4 gate. Two prerequisites, in parallel:

- **2.5-infra (CV training path)** — done (PR #17, merged as 689346b, 2026-11-08; resumes across folds are refused). `make-splits` writes the screen and the 5 CV folds, but
  nothing trains or evaluates on them yet. Needed: `train-supervised --splits FILE
  --fold K | --screen` (train on the fold's training part, select on its selection part,
  predict its evaluation part; test untouched and not computed), and a `compare-variants`
  mode that aggregates the folds and applies the superiority rule of 2.3 and the
  non-inferiority rule below (balanced accuracy, paired bootstrap 95 % CI).
- **2.5-models (B7, B2′, B1′)** — done (PR #18, merged as 6b5aba9, 2026-11-08; flags `paper_arch`, `group_fields`, `context`, `rot90_augment`, ADR 0005). The p4 / p4m single-pass models and the 63″ context
  channels as flags of `paper_stamp_classifier`, with invariance unit tests, parameter
  counts, and MAC counts.

Decided: architecture first (see Decisions). The tables below separate the two lines.

Every row is one flag with 5 seeds against the replication, decided by the protocol of
2.3, with test archived. If 2.5 runs on another machine than 2.4, the 5-seed baseline is
re-run there first (budgeted).

**Object-level leakage (2026-11-08, phase-3 review; verified).** Some Validation and Test rows share a position (within 1.5″) with a Train alert. These are different alerts of the same object or artefact:
- Validation: 24 of 500 (Bogus 20 of 100);
- Test: 27 of 500 (Bogus 21 of 100).

The stored splits are stratified but not grouped, so a CV fold can train on one alert of an artefact and evaluate on another, which favours flags that memorise. Consequences:
- **The 2.5 CV decisions use grouped splits.** Task "grouped decision splits", agent `grouped-splits`, writes `configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json`. The CV waits for it.
- The architecture screen already running used the ungrouped file. It is drop-only, and all its arms see the same leak, so its drops stand.
- The replication (2.4) keeps CD21's protocol, with this caveat.

**2.5-arch: architectural changes (the user's main interest).**

| Id | Change | Hypothesis |
|---|---|---|
| B1 | Field of view: 21 + pad / 27 / 41 / 63 and multi-scale 4×8² (Reyes-Jainaga 2023: 1″/2″/4″/8″ per pixel, 256 values); fc1 sized from the input. 41/63 px only on the 5090, or with a per-batch path (the crops do not fit next to the model on 4 GB) | More context helps SN (the host galaxy) |
| B2 | D4 invariance: 8 copies (rotations × flips), or p4m group-equivariant convolutions | An exact sky symmetry |
| B3 | Batch norm in the conv trunk | Optimisation stability |
| B4 | Width / depth (filters ×1.5–2; one more conv block) | Capacity |
| B5 | Residual blocks; global-average pooling instead of fc1 | A modern inductive bias, fewer parameters |
| B6 | Rotation averaging before the activation | A different invariance point |
| B7 | **Single-pass p4 group-equivariant CNN** (same geometry 27→24→12→6; 8/16 p4 fields = 32/64 maps; fc1 as a 6×6 p4 conv to 64 fields, then the mean over orientations) | Replaces the 4-pass rotation mean: exact invariance, **4× fewer MACs** (53 M vs 211 M), 228,531 parameters (vs 441,075), ~3–4× the throughput |
| B2′ | **Single-pass p4m** (4/8 fields): exact D4 invariance | The cheap version of B2 (the 8-copy version costs ~2× A) |
| B1′ | **p4 trunk + 63″ context:** the 63×63 stamp average-pooled to 21×21 as 3 more channels (+1.7 % MACs) | More context for SN hosts; fits the 4 GB GTX 1050 |

**Transformer-family rows (research, 2026-11-08).** Full report:
scratchpad `transformers.md`, with every claim marked verified or inferred.

What the evidence says:
- The ZTF groups deploy **ConvNeXt**, not ViTs (Rehemtulla et al., arXiv:2512.11957).
  - Their custom CNN wins below about 100k training alerts; we have about 50k.
  - ConvNeXt from scratch is about 0.02 AUC below the CNN; Galaxy-Zoo-pretrained MaxViT is
    about +0.025.
- Plain ViTs need large pretraining. On CIFAR-10 (50k images, 32 px), a ViT from scratch
  reaches 93.6 % against 95–96 % for a ResNet.
- Rotation-equivariant attention is immature.
- No ViT on ZTF, ATLAS, Pan-STARRS or Rubin stamps was found.

Rows, ranked by expected value per cost:

| Id | Change | Hypothesis |
|---|---|---|
| B8 | **ConvNeXt-atto/pico at native 63 px** (padded to 64; in timm, also replace `stages[0].downsample` with Identity for true 32-16-8-4 maps), from scratch and with Zoobot weights, plus 4-rotation averaging; about 3.4–8.5M parameters, 45–112M MACs per view | The modern-CNN control that any transformer must beat. Expected: a tie to −1 point from scratch, a tie to +1 with Zoobot |
| B9 | **Conv-attention hybrid:** our (or p4) conv stages down to an 8×8 or 16×16 grid, then 2–4 attention blocks (or CoAtNet- or MaxViT-nano at 64 px) | Attention over a small grid without the ViT data hunger; a natural place to fuse metadata. Expected: a tie to +1 over B8 |

A ViT-Ti with patch 7, pretrained on 1–5M unlabelled stamps, belongs to phase 5.5 (SSL),
not to 2.5. timm is a new dependency, so B8 and B9 need an ADR and a licence check of the
Zoobot weights first.

**Phase 2.5 architecture and augmentation CV: results (2026-11-08, RTX 5090, Lightning engine).**
- Protocol: grouped 5-fold CV (`configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json`), 2 seeds per fold, 51,081 pool stamps.
- Statistics: balanced accuracy, a paired class-stratified bootstrap with 10,000 resamples, seed 0. Margin 0.005 for non-inferiority.
- Baseline: rot4 (CD21 replication) at 0.9345.

| Row | Arm | Pooled | Δ (95 % CI) | Superiority | Non-inferiority | Decision |
|---|---|---|---|---|---|---|
| C5 | aug_noise 1.0 + feature jitter 0.05 | 0.9365 | +0.21 [−0.01, +0.43] pts | inconclusive | adopt | **not adopted** (pre-registered superiority rule; the CI touches 0). The user decided on 2026-11-08 to keep the original standard and run no confirmatory replication. Its noise stays an input to the phase-5 views and the C5+position diagnostic |
| B7 | p4 | 0.9358 | +0.13 [−0.10, +0.35] | inconclusive | **adopt** | **adopted as the efficiency arm** (non-inferior at about half the compute) |
| B1′ | p4 + avgpool63 context | 0.9325 | −0.20 [−0.45, +0.05] | inconclusive | adopt | dropped (its hypothesis was a gain; more compute) |
| B2′ | p4m | 0.9313 | −0.32 [−0.56, −0.08] | **worse** | inconclusive | dropped |
| C6 | synthetic hot-pixel Bogus 0.05 | 0.9349 | +0.04 [−0.17, +0.26] | inconclusive | adopt | dropped (no gain, about 2x slower) |

Raw logs: `~/runs/p25/cv/` on the 5090. The test set was not used.

**Position and C5 diagnostic CV: results (2026-11-08/09, RTX 5090, plain engine).**
- Protocol: images with no alert features except as stated; grouped 5-fold CV × 2 seeds; 51,081 pool stamps; paired bootstrap with 10,000 resamples; superiority rule.

| Comparison | Pooled | Δ (95 % CI) | Decision |
|---|---|---|---|
| images + position (`feature_set=position`) vs images only (`none`) | 0.8849 → 0.9150 | **+3.01 [+2.72, +3.31] pts** | **adopted**: sky position alone recovers about 60 % of the 23 features' gain |
| + aug_noise 1.0 vs position | 0.9162 | +0.12 [−0.11, +0.36] | not adopted (inconclusive) |
| + C5 (noise + feature jitter) vs position | 0.9171 | +0.21 [−0.03, +0.46] | **not adopted** (inconclusive). The same +0.2 pattern as with all features; the user kept the pre-registered standard |

**D4 results (2026-11-09): TTA and seed ensembles on the stored rot4 CV runs (Lightning; `ensemble-eval`, PR #58).**

| Inference | Pooled | Δ vs rot4 (95 % CI) | Decision |
|---|---|---|---|
| rot4 (CD21's own evaluation) | 0.9345 | — | — |
| **d4** (rot4 model: identity + flip) | **0.9374** | **+0.30 [+0.17, +0.43] pts** | **adopted**; meets the pre-registered "TTA alone ≥ +0.1" |
| d4_shift (d4 × ±1 px crop shifts) | 0.9366 | +0.22 [+0.06, +0.38] | passes, but below d4 alone, so the shifts cost; d4 is the TTA of record |

- **2-seed ensembles (exploratory; the pre-registered arm is 5 seeds):** the ensemble beats its members' mean by +0.57 pts (rot4), +0.51 (d4) and +0.45 (d4_shift), averaged over 5 folds, with no CI.
- The 5-seed ensemble test needs 3 more seeds per fold.

**B8 results (2026-11-09): ConvNeXt-atto from scratch vs the paper model (rot4), both on the plain engine (RTX 5090).**
- Protocol: images and all alert features as in CD21; grouped 5-fold CV × 2 seeds; 51,081 pool stamps; paired bootstrap with 10,000 resamples; superiority rule. ConvNeXt-atto at 63 px (`model.arch=convnext_atto data.image_size=63`), same schedule and augmentations. The baseline is a new plain-engine rot4 CV (`rot4-plain`), so both arms share engine, machine and build.

| Comparison | Pooled | Δ (95 % CI) | Decision |
|---|---|---|---|
| convnext-atto vs rot4-plain | 0.9365 → 0.9191 | **−1.74 [−2.04, −1.44] pts** | **not adopted** (worse on every fold, −1.5 to −2.4; McNemar p = 8.5e-57). Within the "tie to −1 point from scratch" expectation's worse end, past it. Zoobot weights are not tried here (they need timm, ADR) |

- rot4-plain's pooled 0.9365 matches the Lightning rot4 CV (0.9345) within its fold-to-fold std (0.0040).

**D3 declaration (2026-11-09, before any D3 run; the user chose D3 next).**
- **Recipe:** `schedule.selection=final`, `schedule.lr_schedule=cosine`, `schedule.lr_warmup_steps=0`, `schedule.ema_decay=0.999`, `schedule.ema_warmup=true`, `schedule.horizon_increment=0`, `schedule.max_steps=15000`; everything else as `carrasco2021` (images and all features, rot4 training, Adam 1e-3, batch 64).
- **Why 15,000 steps:** in the rot4-plain CV every run went to 30,000 steps and its best validation loss came at a median step of about 6,400 (3,370–13,840): later steps overfit at a constant rate. A cosine over T steps moves the weights about as much as T/2 steps at the full rate, so T = 15,000 matches the median best step with room for the late folds, and halves the cost.
- **Why 0.999:** an averaging window of about 1,000 steps (1.6 passes over a fold's training part); the warmup ramp keeps the early shadow from holding the random initial weights.
- **Arms and test:** 5 folds × seeds {0, 1}, plain engine, RTX 5090, against the stored rot4-plain CV (same engine, machine and build). The registered hypothesis: `compare-variants --rule non-inferiority --margin 0.001` (mean at least baseline − 0.1 pts), plus a fold-to-fold std at least 20 % smaller and a lower mean ECE (secondary, reported). One trial: no other decay or length is tried before this result is recorded.

**D3 results (2026-11-09, the declared single trial).**

| Comparison | Pooled | Δ (95 % CI) | Decision |
|---|---|---|---|
| D3 (EMA 0.999, cosine 15k, final weights) vs rot4-plain | 0.9365 → 0.9235 | **−1.30 [−1.60, −1.01] pts** | **not adopted** (worse on every fold, −0.8 to −2.1; fails non-inferiority at −0.1) |

- Secondary: fold-to-fold std 0.0081 against 0.0040 (larger, not 20 % smaller); ECE 0.2230 ± 0.0030 against 0.2288 ± 0.0100 (about the same mean, much less spread across runs).
- Unresolved, exploratory: McNemar on the 2-seed **ensembles** favours D3 (790 stamps right only with D3 against 537 only with rot4), while the decision statistic, the balanced accuracy of the **seed mean**, drops. They differ in two ways (ensemble against mean, plain accuracy against balanced accuracy), so this does not say which one moved; per-class recalls and the ensembles' balanced accuracy (`ensemble-eval`) would. Not pursued under this row.
- D3 stays closed: the protocol declared one trial.

**Improvement rows from the literature (research, 2026-11-08).** The full ranked report
(18 ideas, every claim marked verified or inferred) is in the scratchpad `improvements.md`.

Facts that constrain the choice:
- **Leakage:** CD21's Asteroid labels require the `ssdistnr` field, so any `ss*` field as a
  feature leaks the label. `rb` and `drb` are a likely shortcut for Bogus.
- **Validation noise:** about 1.1 points of standard error on 500 stamps, so checkpoint
  selection on it adds variance.
- **Imbalance:** balanced tiling already makes the training prior uniform. Logit adjustment,
  focal loss and class-balanced loss are largely redundant.
- **SN repetition:** each of the 1.4k SNe is seen about 270 times. SN needs more distinct
  stamps, not more weight.
- **Normalisation:** per-channel min-max makes brightness and contrast augmentations no-ops.

Pre-registered rows (5 grouped folds × seeds, paired, hierarchical bootstrap):

| Id | Change | Pre-registered hypothesis | Needs |
|---|---|---|---|
| D1 | **Non-detection history** from the first alert's `prv_candidates` (time since the last upper limit, rise above it, number of limits) | SN→Asteroid confusions −25 %, balanced accuracy +0.3 pts (CI > 0) | raw alerts for all CD21 rows (fetch-alerts) |
| D2 | **More candidate fields:** `fid`, PS1 colours, `magpsf − PS1 mag`, `scorr`, `chipsf`, `elong`, `sumrat`, `magdiff`, `nbad`, `jd − jdstarthist` (never `ss*`, `rb` or `drb`) | AGN↔VS + SN→Bogus confusions −15 %, balanced accuracy +0.3 pts | raw alerts |
| D3 | **Weight EMA with a fixed cosine schedule**, scoring the final weights (no checkpoint selection on the noisy validation set) | mean ≥ baseline − 0.1 pts, fold-to-fold std −20 %, lower ECE | none |
| D4 | **5-seed ensemble plus extended test-time augmentation** (flips, ±1 px shifts) | ensemble beats its members' mean by ≥ 0.4 pts; TTA alone ≥ +0.1 | none |
| D5 | **Training on later alerts of training objects** (evaluation stays on first alerts) | SN recall +2 pts, no class −0.3, balanced accuracy +0.3 | raw alerts plus detection lists |

D3 and D4 are dispatched now. D1, D2 and D5 wait for the raw alerts of CD21 Train: the
GTX 1050 machine's fetch has done Validation and Test, and Train is about 7 % done.

**Efficiency rows (B7, B2′, B1′) need a non-inferiority rule**, because the superiority rule
of 2.3 cannot adopt a change that is only "as good". Adopt when the lower bound of the
paired-bootstrap 95 % CI of the CV balanced-accuracy difference is above −0.5 points, and
report MACs and throughput (on the deployment CPU: x86 with ONNX Runtime) next to it.

Control arms:
- **single-pass + rot90 augmentation** (is exact invariance needed?);
- **p4 16/32** at about A's cost (separates the inductive bias from capacity).

Every invariant configuration gets a unit test: max |Δlogit| < 1e-5 under rot90, and under
flips for p4m. Geometry (padding, pooling parity) can silently break G-CNN invariance.

These rows come from the architecture exploration of 2026-11-08 (prototypes and a
validation-only screen on the uint8 path; all candidates reached 0.915–0.925 there,
within noise). The prototypes are hand-written p4/p4m layers that run as plain `conv2d`
(MPS-compatible, no escnn dependency).

**2.5-input/training: data and training changes.**

| Id | Change | Hypothesis |
|---|---|---|
| C1 | Robust normalisation (asinh/MAD, or min/p99 with a ±2 clip) plus a NaN-mask channel | Keeps faint structure and relative scale |
| C2 | Restore the `magnr` and `distnr` features | Brightness context lost by min-max |
| C3 | Active dropout (a genuine change: M1 had none) and β tuning; the calibration error (ECE) | Regularisation, calibration |
| C4 | Unbalanced training with post-hoc logit adjustment | Prior shift without tiling; needs its own unbalanced 5-seed baseline |

| C5 | **Class-preserving augmentation** (training only): per-channel Gaussian noise scaled to each crop's own background σ, then the per-channel min-max again; small jitter on the continuous features | SN has 1,396 Train stamps against ~14.5k for AGN and VS, so tiling repeats each SN about 10×. Noise makes the copies differ. Validation SN recall is 0.876, the weakest class |
| C6 | **Synthetic hot-pixel Bogus**: the template of a real stamp; science = template + noise + a 1–2 px spike at the centre (sharper than the PSF); difference = noise + the same spike; the features of a real Bogus plus jitter | Half the inspected Bogus are this type. The spike must stay sharper than the PSF, or the synthetic Bogus look like Asteroids (same empty template) |

| C7 | **Transient flux rescaling** (all classes but Bogus; training only): fit a Gaussian PSF to the central source of the difference; add `(k−1)·PSF` to the difference and `(k−1)·PSF/a` to the science (the noise is untouched, so the S/N really changes); update `magpsf` and `sigmapsf` to match; keep it an alert (`magpsf' ≤ diffmaglim`, S/N ≥ 5) | Real SN, AGN, VS and Asteroids vary in brightness. Scaling the whole difference would do nothing, because the min-max is scale-invariant. Headroom to the detection limit (median): AGN 0.35 mag, SN 0.96, VS 2.18, Asteroid 0.92 |

C5 and C6 come from a visual inspection of the Train stamps per class and channel and the
replication's **validation** confusion matrix (2026-11-08):
- SN → Asteroid 37 and SN → Bogus 25 of 2,500;
- AGN → VS 26;
- the Bogus errors are spread over the other classes.

Stamps of other classes are **never relabelled** as Bogus by adding an artefact: an SN with a
cosmic ray is still an SN. Later idea: fake SN injected on real host galaxies (the surveys'
standard technique), aimed at SN → Asteroid.

**C5/C6 implemented** (PR #20, ADR 0006): flags `aug_noise`, `aug_feature_jitter`, `synthetic_bogus` and `aug_min_r2`; the difference is recomputed from the per-stamp fit `diff ≈ a·sci + b·tmpl + c`; C6 sources are positive alerts, never SN.

**C7 origin (2026-11-08).** The replication's errors on the screen's *selection* half (the decision half stays untouched): SN → Asteroid are SN with no host in the 21-px crop, one with its host just outside it (the B1′ context targets that case), one on the ecliptic. 27 % of the Train SN (380) are already hostless, with median |ecl_lat| 28° against 6.6° for Asteroids, so synthetic hostless SN are **not** planned: they would mostly trade Asteroid recall for SN recall.

**User decision (2026-11-08):** C5 and C6 run alongside the architecture line, not only if
it plateaus.

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
