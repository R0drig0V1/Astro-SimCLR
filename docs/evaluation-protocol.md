<!-- The phase-2 evaluation protocol (task 2.3 of docs/phases/02-stamp-classifier.md): the
     rules, the power note and how to use the tools. Code: src/astro_simclr/protocol/,
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

- `compare-variants` always ends in `REPORT` and says why: on `--split test` because of this
  rule, and on `--split val` because a normal run's Validation stamps chose its checkpoint.
- The decision helpers (`screen_comparison`, `cv_comparison` in
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

## Comparing two variants: `compare-variants`

```bash
astro-simclr compare-variants --baseline runs/base/seed{0..4} --variant runs/b3/seed{0..4} \
    [--split test|val] [--n-boot 10000] [--seed 0]
```

It reports both tests the plan asks for. The difference is always variant − baseline.

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
bootstraps the balanced-accuracy difference, stratified by class.

**The adoption rule:** a flag is adopted only if the 95 % interval of its CV difference
against the baseline lies **entirely above 0** (`adoption_decision`: `adopt`, `worse` or
`inconclusive`). Task 2.6 combines only adopted flags.

### The 250/250 screen (drop only)

Validation is split into a selection half and a decision half, 50 stamps per class each. A
screened run chooses its checkpoint on the selection half and is evaluated on the decision
half. `screen_comparison` **drops** a variant whose seed-mean accuracy on the decision half
is 4 points or more below the baseline's. It never adopts one: its minimum detectable
effect is about 4 points. It runs first, to drop clear losers cheaply.

> **Not wired yet.** Training on a fold or on a screen half needs the data module to take
> index subsets, and to write predictions whose `index` is the Validation index (screen) or
> the pool index (fold). Today every run writes `index = arange(N)` for the whole of
> Validation and Test. That wiring belongs with the GPU-resident data module of task 2.2a.
> The split definitions, this predictions contract and the decision helpers are ready for
> it.

## Leakage

The deduplication of `red` removed duplicates within each split only. `make-splits`
(`astro_simclr.protocol.leakage.find_duplicates`) compares the uint8 stamps of every pair of
splits. On `td_ztf_stamp_17_06_20_red-002` it finds exactly one duplicate: **Train 2995,
labelled Bogus, is Validation 115, labelled SN**, with identical features. The same alert
therefore carries two different labels.

The stamp is removed **from Train only** (`excluded_train`), and it is in no CV fold.
Training runs do **not** drop it yet. The replication's data path (tasks 2.2a and 2.2c) has
to apply `train_keep_indices(n_train, excluded_train)`. Validation and Test keep their 500
stamps, as in the paper, so Validation 115 stays, labelled SN.

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
