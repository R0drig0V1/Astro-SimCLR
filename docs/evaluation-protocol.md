<!-- The phase-2 evaluation protocol (task 2.3 of docs/phases/02-stamp-classifier.md; the CV
     training path of task 2.5-infra): the rules, the power note and how to use the tools.
     Code: src/astro_simclr/protocol/ (split_runs, compare), the --splits path of
     src/astro_simclr/experiments.py and training/device_data.py,
     src/astro_simclr/reference.py (CD21), the metrics JSON of src/astro_simclr/experiments.py.
     Design choices: docs/decisions/0004-evaluation-protocol.md. -->
# Evaluation protocol (phase 2)

Every phase-2 result (the replication of Carrasco-Davis et al. 2021, then every improvement)
is measured and compared the same way. This page states the rules and how the code applies
them. The plan is the authority: `docs/phases/02-stamp-classifier.md`, task 2.3.

## The rule: test is never a decision input

**Test metrics are never read to choose a variant, a diagnostic or a combination.**
Every keep/drop decision uses one of:

- validation, as the mean over 5 seeds, and only on stamps that did **not** choose the
  checkpoint (the screen's decision half, below);
- the M1 oracle (task 2.2b), which needs no training;
- a cross-validation run on Train + Validation.

Test is computed for every run, archived, and **reported once per pre-registered
comparison**. The code helps with this:

- `compare-variants` on normal runs always ends in `REPORT` and says why: on `--split test`
  because of this rule, and on `--split val` because a normal run's Validation stamps chose
  its checkpoint.
- Decisions come only from *split runs* (`train-supervised --splits FILE --fold K |
  --screen`, below), which never load Test, and `compare-variants --splits FILE` only
  accepts them. The decision helpers (`screen_comparison`, `cv_comparison` in
  `astro_simclr.protocol.compare`) only accept predictions on the screen's decision half or
  on a CV fold's evaluation part. They check the stamp indices, so test or full-Validation
  predictions are refused.

The same principle applies to validation: **a decision never uses the metric of the stamps
that chose the checkpoint.** The replication itself (task 2.4) is the exception. It keeps
the paper's protocol, which selects on all 500 Validation stamps, so that its number stays
comparable with 0.941. Its validation accuracy is a gate condition, not a decision between
variants.

## What every run reports

`metrics.json` of every classifier run (`train-supervised`, `linear-eval`, `fine-tune`)
holds, for the checkpoint selected on validation:

| Key | Content |
|---|---|
| `test_acc_at_best` | test accuracy (micro; on the balanced test set it equals balanced accuracy) |
| `val_acc_best`, `val_macro_f1`, `test_macro_f1` | validation accuracy, macro-F1 |
| `evaluation.{val,test}` | `accuracy`, `balanced_accuracy`, `macro_f1`, `recall_per_class` (by class name; `null` for a class with no stamp), the normalised and count confusion matrices |
| `predictions.{val,test}` | the per-stamp predictions file (`predictions_val.npz`, `predictions_test.npz`), the evaluated set and its size |
| `machine` | `device` (the one Lightning chose), `gpu` (the CUDA device name; `null` elsewhere), `torch`, `cuda`, `cudnn`, `platform`, `processor`, `python` |

The predictions files hold `probabilities` (float32 softmax, N × 5, columns in label order),
`labels` and `index` (the stamp's position in the evaluated set). They are plain `.npz`
files read with `allow_pickle=False`. Comparisons pair runs stamp by stamp through them.

Results are summarised as mean ± std (ddof = 1) over **5 seeds**. Runs in one comparison
use the same machine and build (`machine.device`, `gpu`, `torch`, `cuda`). The 1050
(`cu126`) and the 5090 (CUDA 13) differ numerically. `compare-variants` refuses mixed runs,
and `compare-reference` prints a WARN line for them.

## The replication gate: `compare-reference --ref CD21`

```bash
astro-simclr compare-reference runs/cd21/seed{0..4} runs/cd21-nofeat/seed{0..4} \
    --ref CD21 --oracle-report results/phase2/replication/m1_oracle.json
```

The gate judges the with-features runs (`config.with_features: true`) on the mean test
accuracy against the paper's 0.941 ± 0.004:

| Tier | Condition |
|---|---|
| **faithful** | \|mean − 0.941\| ≤ 0.005 **and** the M1 oracle passed |
| **acceptable** | \|mean − 0.941\| ≤ 0.01 |

Both tiers also require:

- the mean SN test recall within ±4 points of 87 %;
- the mean `val_acc_best` within ±0.01 of 0.950.

Other details of the gate:

- The oracle report is the JSON written by task 2.2b. Its top-level `passed` must be
  `true`. Without the report, the best possible result is "acceptable".
- The gate is defined on **exactly 5 seeds**. Another count gives `REPORT`, with the tier
  the runs would reach, never `PASS`.
- The paper's σ = 0.004 is ALeRCE's `np.std`, with **ddof = 0** (about 0.0045 at ddof = 1).
  The runs' std is printed both ways.
- With σ ≈ 0.0045 on both 5-run means, a perfect replication misses the faithful tier about
  8 % of the time. The SE of the difference is 0.0045·√(2/5) ≈ 0.0028, and
  P(|Z| > 0.005/0.0028) ≈ 0.08. So one confirmatory 5-seed rerun is allowed, and it is
  judged as its own set.
- The without-features runs are compared with 0.883 ± 0.006 and **only reported**
  (advisory). That reference has no stated hyperparameters (plan, "Starting point").

Exit codes: 0 for PASS (either tier) or REPORT, 1 for FAIL.

## Comparing two variants: `compare-variants` (report)

```bash
astro-simclr compare-variants --baseline runs/base/seed{0..4} --variant runs/b3/seed{0..4} \
    [--split test|val] [--n-boot 10000] [--seed 0]
```

On normal runs (one per seed) it reports both tests the plan asks for, and always exits 0.
Decisions use `--splits FILE` on split runs instead (see "Running a decision end to end").
The difference is always variant − baseline.

- **McNemar on seed-ensembled predictions.** For each stamp and variant, the 5 seeds'
  probabilities are averaged and the argmax is taken. Then the exact two-sided McNemar
  test runs on the two "correct" vectors. Pooling seed × stamp pairs is not allowed: the
  pairs of one stamp are correlated, so pooling would overstate the evidence.
- **A paired bootstrap over stamps.** Each stamp's score is the fraction of seeds that
  classify it correctly. The mean of these scores is the seed-mean accuracy. The stamps are
  resampled with replacement, the same draw for both variants, and **stratified by class**:
  each class keeps its count, because the evaluation sets are balanced by design. The
  output is the 95 % percentile interval of the difference.

Both sides must come from the same data (`metrics.json` `data.source.sha256`) and evaluate
the same stamps, in the same order, with the same labels. The seeds must be distinct within
a side.

## The decision protocol (2.5 and later)

Decisions never use the metric of the stamps that chose the checkpoint. The splits below
are **seeded and stored**: `configs/splits/td_ztf_stamp_17_06_20_red-002.json`, generated on
the real data with

```bash
astro-simclr make-splits dataset/prepared/td_ztf_stamp_17_06_20_red-002 \
    --out configs/splits/td_ztf_stamp_17_06_20_red-002.json      # seed 0
```

The file records the dataset's source SHA-256 and the SHA-256 of Train's and Validation's
labels. `verify_splits(read_splits(file), prepared_dir)` does several checks:

- it refuses another dataset, but accepts a renamed or moved copy;
- with the real labels, it checks the class balance of the screen halves and of the
  selection sets, and the stratification of the folds;
- it re-runs the leakage check against `excluded_train`. **The stored file is the source of truth**, not
the generator: numpy does not promise the same seeded permutation in every version.

### Cross-validation (the default for 2.5)

The split is a stratified 5-fold CV on the *pool* Train + Validation: 51,081 stamps, after
the leakage removal. Pool index `i < 50,582` is Train stamp `i`, and pool index
`50,582 + j` is Validation stamp `j` (`ProtocolSplits.locate`). For each fold:

- its **evaluation** part (about 10,215 stamps) is held out and gives the decision metric:
  **balanced accuracy**;
- its **selection** set (100 per class, drawn from the rest of the pool) chooses the
  checkpoint;
- its **training** part (about 40,365 stamps) is the rest of the pool. It is the only part
  that is tiled (balanced by repetition) and trained on.

Each fold uses 2 seeds. `cv_comparison` concatenates every fold's per-stamp seed-mean
scores. Every pool stamp is held out exactly once, so this covers the whole pool. It then
bootstraps the balanced-accuracy difference, stratified by class. The bootstrap is **paired
by pool index**: both sides must cover each fold's evaluation part in its stored order, so
the same position is the same stamp on both sides, and each resample draws the same stamps
for both.

**The adoption rule (superiority, task 2.3):** a flag is adopted only if the 95 % interval
of its CV difference against the baseline lies **entirely above 0** (`adoption_decision`:
`adopt`, `worse` or `inconclusive`). Task 2.6 combines only adopted flags.

**The non-inferiority rule (efficiency rows of 2.5: B7, B2′, B1′).** A change whose point is
to be cheaper (fewer MACs or parameters) only needs to be "as good", which the superiority
rule cannot adopt. It is adopted when the lower bound of the same interval is **above
−0.5 points** (−0.005; `non_inferiority_decision`, `--rule non-inferiority`, `--margin`).
An interval entirely below −margin is `worse`. The margin is a fraction of balanced
accuracy; values above 0.05 are refused as a likely unit mistake (0.5 typed for half a
point). Report MACs and throughput next to it (plan, 2.5). The rule is chosen before the
runs, per row: it is not a fallback when superiority fails.

### The 250/250 screen (drop only)

Validation is split into a selection half and a decision half, 50 stamps per class each. A
screened run chooses its checkpoint on the selection half and is evaluated on the decision
half. `screen_comparison` **drops** a variant whose seed-mean accuracy on the decision half
is 4 points or more below the baseline's. It never adopts one: its minimum detectable
effect is about 4 points. It runs first, to drop clear losers cheaply.

## Running a decision end to end (task 2.5-infra)

### Split runs: `train-supervised --splits FILE --fold K | --screen`

A split run is a normal `carrasco2021` run (any preset, diagnostic or 2.5 flag) whose data
are one part of the stored splits:

| | `--fold K` (0-4) | `--screen` |
|---|---|---|
| trains on (tiled to the largest class, as in the preset) | fold K's training part | Train minus `excluded_train` |
| selects the checkpoint on (same monitor, horizon rule and batch-64 means as the preset) | fold K's selection set (500) | the screen's selection half (250) |
| predicts | fold K's evaluation part (~10,215) | the screen's decision half (250) |
| `index` in the predictions file | pool index | Validation index |

- **Test is never loaded**, so it can be neither measured nor predicted. Only Train and
  Validation are read and cropped.
- Before anything is built, the splits file is **verified against the prepared data**
  (`verify_splits`: the source and label SHA-256 and sizes, the balance, the
  stratification, a fresh leakage check). A file made for other data fails with exit 1 and
  says to regenerate it with `make-splits`. A fold that the file does not have, another
  schedule than `carrasco2021`, or `exclude_leaked_train: false` fail with exit 2.
- **Resume** (`--resume PATH|auto`) continues only the same run. The run's identity (the
  splits SHA-256, the mode and the fold) is in every checkpoint. A checkpoint of another
  fold, of the screen, or of a normal run is refused with exit 2 before any step, and a
  normal run refuses a split run's checkpoint. This matters because two folds can have the
  same tiled size, so the stream check alone would accept the checkpoint, and fold K's
  weights were trained on fold K′'s evaluation part.
- **Feature statistics.** The 23 features were standardised once, with the statistics of
  the whole of Train, before any split (`prepare-data`). So the evaluation stamps of a fold
  that come from Train contributed to the means and standard deviations its training part
  uses. This is unsupervised (no label is used) and the same for every variant, so it does
  not bias a comparison. Still, "no evaluation stamp influences training" holds for the
  labels and the gradient updates, not literally for the feature scaling.
- The outputs: `metrics.json` with `protocol` (`mode`, `fold`, `splits_file`,
  `splits_sha256`, the part sizes, `excluded_train`), `selection_acc_best`,
  `evaluation_accuracy`, `evaluation_balanced_accuracy` (the decision metric of the fold),
  `evaluation.{selection,evaluation}` (the usual per-part block),
  `predictions.evaluation` → `predictions_evaluation.npz`, and
  `confusion_matrix_{selection,evaluation}.png`. There is no `val_acc_best` or
  `test_acc_at_best`, so a split run can never enter the CD21 gate. The Lightning logs call
  the evaluation part `test` (it is the data module's `Test` role).
- Measured on the Mac M4 CPU (fold 0 of `red`, `--accelerator cpu --set max_steps=200
  horizon_increment=0`; `horizon_increment=0` keeps a short run short): 261 s of wall time
  in total. Of these, 217 s are the fit loop, 1.08 s per step with a 500-stamp selection
  pass every 10 steps. The other ~45 s are the fixed cost of the run: verifying the splits
  (the leakage check), cropping Train and Validation, and predicting the 10,219 evaluation
  stamps. The fold's training part has 40,362 stamps, 58,065 after tiling (907 steps per
  epoch). A full fold (30,000+ steps) is therefore a GPU job: about 9 h on this CPU.

### The commands

Run every arm on the same machine and build (the 5090 for the full CV), with the same seeds
on both sides. The baseline arms are run once per machine and shared by every row. `--set`
takes several `KEY=VALUE` items, so it goes last.

```bash
SPLITS=configs/splits/td_ztf_stamp_17_06_20_red-002.json
TRAIN="astro-simclr train-supervised --preset carrasco2021 --data-dir dataset/prepared --splits $SPLITS"

# 1. The screen (cheap; it can only drop): one run per seed and arm.
for s in 0 1; do
  $TRAIN --screen --seed $s --output-dir runs/screen/base/seed$s
  $TRAIN --screen --seed $s --output-dir runs/screen/b7/seed$s --set KEY=VALUE
done
astro-simclr compare-variants --splits $SPLITS \
    --baseline runs/screen/base/seed* --variant runs/screen/b7/seed*
# exit 1 (DROP): stop here. exit 0 (KEEP): not dropped, go on to the CV.

# 2. The CV: 5 folds x 2 seeds per arm.
for k in 0 1 2 3 4; do
  for s in 0 1; do
    $TRAIN --fold $k --seed $s --output-dir runs/cv/base/fold$k-seed$s
    $TRAIN --fold $k --seed $s --output-dir runs/cv/b7/fold$k-seed$s --set KEY=VALUE
  done
done
astro-simclr compare-variants --splits $SPLITS \
    --baseline runs/cv/base/fold* --variant runs/cv/b7/fold*            # superiority
astro-simclr compare-variants --splits $SPLITS \
    --baseline runs/cv/base/fold* --variant runs/cv/b7/fold* \
    --rule non-inferiority [--margin 0.005]                             # efficiency rows
```

### What `compare-variants --splits FILE` checks and prints

- Every run is a split run trained on **this** file (its SHA-256 is in the run's
  `protocol`), all in one mode (folds or screen), on the same data (`data.source.sha256`)
  and machine and build (unless `--allow-mixed-machines`).
- Both sides cover **the same (fold, seed) cells**, once each, and the CV runs cover
  **every fold** of the file. A missing fold, a seed on one side only or a repeated cell is
  an error that names it (exit 2). Fewer than 2 seeds in a fold gives a WARN line.
- Each run's predictions cover exactly its part, in order (otherwise exit 2).
- **CV:** the balanced accuracy of each fold (seed means) for both arms and their
  difference, the pooled balanced accuracies, McNemar on the seed ensembles over all folds
  (reported, not a rule), the paired bootstrap 95 % CI, and the rule's outcome. Last line
  `compare-variants: ADOPT` (exit 0) or `compare-variants: NOT ADOPTED (worse|inconclusive)`
  (exit 1).
- **Screen:** the seed-mean accuracies on the decision half, McNemar, and
  `compare-variants: DROP` (exit 1) or `KEEP` (exit 0: not dropped; the screen never
  adopts). `--rule` is refused for the screen.

Exit codes follow `compare-reference`: 0 for a pass (ADOPT, KEEP), 1 for a fail (NOT
ADOPTED, DROP) and for unusable data, 2 for invalid arguments or runs that cannot be
compared.

## Leakage

The deduplication of `red` removed duplicates within each split only. `make-splits`
(`astro_simclr.protocol.leakage.find_duplicates`) compares the uint8 stamps of every pair of
splits. On `td_ztf_stamp_17_06_20_red-002` it finds exactly one duplicate: **Train 2995,
labelled Bogus, is Validation 115, labelled SN**, with identical features. The same alert
therefore carries two different labels.

The stamp is removed **from Train only** (`excluded_train`), and it is in no CV fold. The
`carrasco2021` runs drop it (`exclude_leaked_train`; tasks 2.2a and 2.2c), and split runs
take it from the splits file, checked against a fresh leakage check. Validation and Test
keep their 500 stamps, as in the paper, so Validation 115 stays, labelled SN.

## Power note

The noise sources that limit a claim:

- **Seed noise.** With σ_seed ≈ 0.004, the minimum detectable effect between two n-seed
  means is (t₀.₉₇₅ + t₀.₈₀)·σ·√(2/n), with 2n − 2 degrees of freedom:
  - about **1.2 points with 3 seeds**: (2.776 + 0.941) × 0.004 × 0.816;
  - about **0.8 points with 5 seeds**: (2.306 + 0.889) × 0.004 × 0.632.
- **Test sampling.** On 500 test stamps, the standard error of a paired difference is about
  √(p_discordant / 500) ≈ 1 point (with about 5 % discordant stamps). Effects under about
  **2 points** hold "on this test set" only.
- **The screen.** On 250 decision stamps the same estimate gives about 1.4 points of
  standard error, so its MDE is about 4 points.
- **Claims below 2 points** need the k-fold CV on Train + Validation with balanced accuracy.
  That costs about 5× the compute: (6 flags + baseline) × 5 folds × 2 seeds ≈ 70 runs, on
  the 5090.
