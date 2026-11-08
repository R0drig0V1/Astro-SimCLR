<!-- The phase-2 evaluation protocol (task 2.3 of docs/phases/02-stamp-classifier.md; the CV
     training path of task 2.5-infra): the rules, the power note and how to use the tools.
     Code: src/astro_simclr/protocol/ (groups, splits, audit, split_runs, compare), the --splits path of
     src/astro_simclr/experiments.py and training/device_data.py,
     src/astro_simclr/reference.py (CD21), the metrics JSON of src/astro_simclr/experiments.py.
     Design choices: docs/decisions/0004-evaluation-protocol.md. The last section, "Phase 3
     benchmark metrics" (task 3.2), documents src/astro_simclr/protocol/{alerts,grouping,
     object_splits,benchmark_metrics,bench_report}.py; its choices are in ADR 0008. -->
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
variants. It also keeps the paper's splits, which share objects across Train, Validation and
Test (about 20 % of the Validation and Test Bogus; see "Leakage" below).

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

The task 2.4 replication runs (5 seeds with features, 5 without) are archived in
`results/phase2/replication/`: each run's `metrics.json`, confusion matrices and per-stamp
predictions (the baseline for later paired comparisons), and the gate's output. Its
`README.md` has the provenance, the per-seed table and the commands. Runs made before task
2.5-models lack `paper_arch`, `context` and `rot90_augment` (and the C5 / C6 flags) in their
config. The archived gate output printed a WARN line for each; since task 2.5 C5/C6 a
missing key reads as its `Settings` default (the faithful value), so the gate no longer
warns about them, and the verdict is unchanged (PASS, faithful).

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

Decisions never use the metric of the stamps that chose the checkpoint, and never evaluate
on an object that a run trained on. The splits below are **seeded, grouped by object and
stored**. **2.5 decides on the grouped file**,
`configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json` (format version 2), generated on
the real data with

```bash
astro-simclr make-splits dataset/prepared/td_ztf_stamp_17_06_20_red-002 \
    --out configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json   # seed 0, 1.5"
```

The first file, `configs/splits/td_ztf_stamp_17_06_20_red-002.json` (format version 1,
stratified but **not grouped**), is kept unchanged and still readable, so the runs made on
it stay reproducible; it is not used for new decisions (see "Leakage").

The file records the dataset's source SHA-256, the SHA-256 of Train's and Validation's
labels, and the grouping (method, radius, the groups of two or more stamps).
`verify_splits(read_splits(file), prepared_dir)` does several checks:

- it refuses another dataset, but accepts a renamed or moved copy;
- no group spans two parts (two folds, a fold's selection set and its training part, the
  two screen halves), and the stored groups are what a fresh grouping of the data finds;
- with the real labels, it checks the class balance of the screen halves and of the
  selection sets, and the stratification of the folds;
- it re-runs the leakage check against `excluded_train`.

**The stored file is the source of truth**, not the generator: numpy does not promise the
same seeded permutation in every version.

### Cross-validation (the default for 2.5)

The split is a class-stratified **group** 5-fold CV on the *pool* Train + Validation: 51,081
stamps, after the leakage removal, in 49,717 position groups (663 of two or more stamps).
Pool index `i < 50,582` is Train stamp `i`, and pool index `50,582 + j` is Validation stamp
`j` (`ProtocolSplits.locate`). Every group is evaluated by exactly one fold. Groups are
placed largest first into the fold where they least raise the per-class fill, so per class
two folds differ by at most the largest group's stamps of that class (on `red`: by at most
1). For each fold:

- its **evaluation** part (about 10,216 stamps) is held out and gives the decision metric:
  **balanced accuracy**;
- its **selection** set (100 per class) chooses the checkpoint. It is drawn from whole
  groups outside the fold, in seeded random order; a group whose class is already full is
  skipped, and a group that fills a class gives only the stamps needed. The rest of such a
  *selection group* is **unused** in that fold (neither selection nor training), so no
  group spans the selection set and the training part. On `red` no fold has unused stamps;
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

Validation is split into a selection half and a decision half, 50 stamps per class each, and
no group spans the two halves (Validation's groups are placed largest first into the half
with room for them). A screened run trains on Train without `excluded_train` and without
`screen_excluded_train`: the **107 Train alerts whose group has a Validation member**, as
the leaked stamp leaves Train. It chooses its checkpoint on the selection half and is
evaluated on the decision half.

*Fallback.* If the groups make 50 + 50 per class impossible (a group bigger than what is
left in either half is left out of the screen), both halves are trimmed at random to the
largest per-class count they share, which the file stores as the screen's `per_class`, with
the reason in `grouping.screen_fallback`; `make-splits` prints it. On `red` it is not needed.
The CV selection sets need no fallback: they are always exact. `screen_comparison` **drops** a variant whose seed-mean accuracy on the decision half
is 4 points or more below the baseline's. It never adopts one: its minimum detectable
effect is about 4 points. It runs first, to drop clear losers cheaply.

## Running a decision end to end (task 2.5-infra)

### Split runs: `train-supervised --splits FILE --fold K | --screen`

A split run is a normal `carrasco2021` run (any preset, diagnostic or 2.5 flag) whose data
are one part of the stored splits:

| | `--fold K` (0-4) | `--screen` |
|---|---|---|
| trains on (tiled to the largest class, as in the preset) | fold K's training part | Train minus `excluded_train` and, on a grouped file, `screen_excluded_train` |
| selects the checkpoint on (same monitor, horizon rule and batch-64 means as the preset) | fold K's selection set (500) | the screen's selection half (250) |
| predicts | fold K's evaluation part (~10,215) | the screen's decision half (250) |
| `index` in the predictions file | pool index | Validation index |

- **Test is never loaded**, so it can be neither measured nor predicted. Only Train and
  Validation are read and cropped.
- Before anything is built, the splits file is **verified against the prepared data**
  (`verify_splits`: the source and label SHA-256 and sizes, the group integrity and a
  fresh grouping, the balance, the stratification, a fresh leakage check). A file made for other data fails with exit 1 and
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
  `splits_sha256`, the part sizes, `excluded_train`, and `grouping`: method, radius, group
  count and the screen's grouped exclusions, `null` for a version-1 file), `selection_acc_best`,
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
SPLITS=configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json
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
  and machine, build and effective float32 precision (ADR 0010; unless
  `--allow-mixed-machines`).
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

Two kinds of leakage connect the splits of `red`. `astro-simclr leakage-audit PREPARED_DIR`
(`astro_simclr.protocol.audit`) reports both.

**The same stamp.** The deduplication of `red` removed duplicates within each split only.
`make-splits` (`astro_simclr.protocol.leakage.find_duplicates`) compares the uint8 stamps of
every pair of splits. On `td_ztf_stamp_17_06_20_red-002` it finds exactly one duplicate:
**Train 2995, labelled Bogus, is Validation 115, labelled SN**, with identical features. The
same alert therefore carries two different labels. The stamp is removed **from Train only**
(`excluded_train`), and it is in no CV fold. The `carrasco2021` runs drop it
(`exclude_leaked_train`; tasks 2.2a and 2.2c), and split runs take it from the splits file,
checked against a fresh leakage check. Validation and Test keep their 500 stamps, as in the
paper, so Validation 115 stays, labelled SN.

**The same object.** `red` removed exact duplicates only, so one object (or one artefact)
can have several *distinct* alerts. Grouping the alerts by position (`ra`, `dec`
de-standardised with M1's statistics, linked within 1.5", connected components;
`astro_simclr.protocol.groups`) gives, on `red` (from `leakage-audit`):

| | rows sharing a position with a Train alert | by class |
|---|---|---|
| Validation | 24 of 500 | Bogus 20, AGN 2, SN 1, VS 1 |
| Test | 27 of 500 | Bogus 21, SN 3, AGN 2, VS 1 |

Train itself holds 656 groups of 2 to 16 alerts (1,997 alerts: Bogus 1,418, VS 290,
AGN 253, SN 36), mostly the same artefact seen again, and 107 Train alerts share a group
with Validation. The version-1 splits were stratified but not grouped, so a CV fold could
train on one alert of an artefact and evaluate on another. That rewards whatever memorises
more, and the adoption rule would read it as superiority. Hence:

- **2.5 decisions use the grouped file** (format version 2, above): no group spans two
  folds, a fold's selection set and its training part, or the two screen halves, and the
  screen's runs leave out the 107 Train alerts grouped with Validation. `verify_splits`
  checks the groups at every split run.
- **The paper replication (2.4) keeps CD21's protocol**: the paper's Train, Validation and
  Test, so that its number stays comparable with 0.941. Caveat: about 20 % of the
  Validation and Test Bogus (20 and 21 of 100) share a position with a Train alert, so its
  Bogus recall, and the test accuracy through it, may be optimistic for unseen objects.
- The grouping is a parameter (`--radius-arcsec`, 1.5" by default) and can come from an
  external id instead (`--group-ids FILE`: a `.npz` with one array per split, for example
  the `oid`s of task 3.1's `fetch-alerts`). Position matching is transitive, so a chain of
  close alerts is one group even when its ends are farther apart than the radius.

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

## Phase 3 benchmark metrics

Task 3.2 of `docs/phases/03-newer-benchmarks.md`: the evaluation machinery of R5, the
object-level benchmark under RJ23's protocol (5 imbalanced classes, macro-F1). The code
takes any number of classes, so RJ23's six (with Satellite) are a `--labels` away. The
phase-2 rules above still hold: Test is never a decision input, and comparisons pair runs
item by item. The code is in
`src/astro_simclr/protocol/` (`alerts`, `grouping`, `object_splits`, `benchmark_metrics`,
`bench_report`, and `stats.paired_bootstrap_macro_f1`). The design choices are in ADR 0008.
Everything is built and tested on synthetic tables. The real `oid`s come from task 3.1.

### The input: an alert table

The input is a CSV with one row per labelled alert. Task 3.1's index can feed it directly.

| Column | Content |
|---|---|
| `oid` | the ZTF object id: letters, digits, `_`, `-`, `.`; 1-64 characters |
| `candid` | the alert id: a positive integer below 2⁶³, unique |
| `jd` | the alert's Julian date: finite and > 0 |
| `label` | a class name; by default R5's `AGN, SN, VS, Asteroid, Bogus` (the phase-2 five, in label order); `--labels` sets others, such as RJ23's six with `Satellite` |
| `group` | optional: the split's group key, when it is not the `oid`: the plan's group id is the `oid` where matched, otherwise a 1.5″ position group built elsewhere |

Other columns (`fid`, `split`, `row`) are ignored.

The file is downloaded data, so it is **untrusted**:
- it is parsed with this schema by the `csv` module, and no field is ever evaluated;
- a bad row is an error that names its line.

The table's fingerprint (SHA-256 of the canonical rows, sorted by `candid`) does not
depend on the row order. Per-alert predictions refer to alerts by `candid`.

### Metrics and the zero-division rule

`benchmark_metrics(labels, predicted, label_names)` gives, for any number of classes:

- per class: precision, recall, F1, support and predicted count;
- macro precision, macro recall and macro-F1 (unweighted means over the classes);
- micro accuracy;
- the confusion matrix of counts, and the same matrix with each row normalised to 100 (the
  percentage of true class *i* predicted as *j*, as RJ23 shows).

F1 is `2 TP / (2 TP + FP + FN)`. The zero-division rule:

| Case | Precision | Recall | F1 | In the macro means? |
|---|---|---|---|---|
| true samples, never predicted | 0 (0/0 → 0) | 0 | 0 | **yes**: never predicting a hard class must not pay |
| predicted, no true sample | 0 | 0 (0/0 → 0) | 0 | **yes**: its false positives are real errors |
| absent from labels and predictions | `None` | `None` | `None` | no |

This is the rule of the phase-2 macro-F1 (`training.metrics.macro_f1`, torchmetrics), and a
test checks that the two agree. A confusion row with no true sample is all zeros.

**Prior-reweighted macro-F1.** Precision depends on the class prevalence: a rare class's
false positives come from the large classes. So the report also gives macro precision and
macro-F1 with each confusion row rescaled to a reference count (`counts[i] / support_i ×
reference_i`; recall is unchanged).
- Scaling the reference changes nothing, so only the class proportions matter. The default
  reference is RJ23's class totals (`RJ23_CLASS_COUNTS`: AGN 9,774, SN 3,615, VS 10,211,
  Asteroid 9,180, Bogus 15,950, Satellite 580), taken in the order of the label names. A
  stratified 60/20/20 split keeps these proportions in its test set, up to rounding.
- This reweighted macro-F1 is **R5's headline metric**, with the raw one next to it (plan,
  task 3.5).
- `--reference-counts` sets another reference. With a class that is not one of RJ23's
  there is no default, and the reweighting is left out.

### Object-grouped, class-stratified splits: `make-object-splits`

```bash
astro-simclr make-object-splits dataset/raw/alerts.csv --out configs/splits/rj23-rebuild.json \
    [--seed 0] [--fractions 0.6 0.2 0.2] [--labels AGN,SN,VS,Asteroid,Bogus] \
    [--drop-conflicts]
```

Whole **groups** (the `oid`, or the `group` column) go to Train, Validation or Test, so no
group is in two splits. The draw, per class:
1. Shuffle the class's groups with `default_rng([seed, stream, class])`.
2. Give each group, in that order, to the split whose alert count is furthest below its
   target (`fraction × class alerts`).

What the draw guarantees:
- **Bound:** within every class, each split's count ends at most `g` above its target and
  at most `(k − 1)·g` below it, where `g` is the class's largest group and `k` the number
  of splits. `verify_object_splits` checks this bound.
- **Tolerance:** with groups of 1-3 alerts, each class is within 1 point of 60/20/20 (a
  test checks this).
- **Order:** the order is random, not largest-first. A largest-first order sends the large
  objects to Train, so Test would have fewer alerts per object, and its first-stamp view
  would differ.

**An object has one label.** A group whose alerts carry two labels is refused, unless
`--drop-conflicts` leaves it out of every split. The dropped groups are recorded in the
file.

Validation is also split into a **selection half and a decision half**: 50/50 of each
class's alerts, grouped and stratified the same way, with their own generator stream. So
phase-2-style decisions (select the checkpoint on one half, decide on the other) are
possible on the imbalanced 6-class set. `grouping.grouped_halves` does the same on any
per-row arrays.

**Constraints** (task 3.5, ADR 0016). `make_object_splits(..., allowed={group: splits})`
restricts named groups to some splits. This is how R5 applies the matrix below: a group that
holds a CD21 Train row goes only to R5 Train, and one with a CD21 Validation or Test row only
to Validation or Test.
- Within a class, the constrained groups are placed first, each in the allowed split
  furthest below its target, then the free ones as above.
- The file (format version 2) lists the constraints, and verification checks each one. The
  bound is checked only for the classes without a constrained group, because constraints
  can push a class past it. R5 reports the achieved fractions in `split_counts.csv`.
- Without constraints the draw is unchanged. Version 1 files are still read.

**The stored file is the source of truth**, as for the phase-2 splits.
- It holds the table's fingerprint, the group key, the label names, the seed, the
  fractions, the conflicts, the constraints, the sorted group ids per split and per half,
  and the class counts per part.
- `verify_object_splits(splits, table)` refuses another table or group key. It also
  checks that every group is in exactly one split, that the conflicts and halves match,
  the bound, and the counts.
- The same seed gives the same file.

**The cross-benchmark rule** (`grouping.assert_no_shared_groups`) takes the per-row group
ids and split membership of two benchmarks, in one id space (for example `oid`, or a
position group built across both). It counts the groups that each of the 3 × 3 split pairs
shares, and refuses the forbidden pairs. The default is the plan's matrix
(`CROSS_BENCHMARK_ALLOWED`):

| A \ B | Train | Validation | Test |
|---|---|---|---|
| **Train** | allowed | forbidden | forbidden |
| **Validation** | forbidden | allowed | allowed |
| **Test** | forbidden | allowed | allowed |

So a CD21 Train object may go only to R5 Train, and CD21 Validation and Test objects never
to R5 Train. `allowed=()` refuses all nine pairs. The returned count matrix is what the
pretraining exclusion lists are built from. This module never builds position groups: it
accepts any per-row group-id array (the position groups come from a separate task).

### The first-stamp view

`first_stamps(table, rows)` returns, for each object among `rows`, its **earliest alert in
the table**: the smallest `jd`, with ties broken by the smaller `candid`. This is RJ23's
first-stamp view only if the table holds each object's first detection. Task 3.1/3.5 must
make sure it does, or must document the difference.

### The report: `bench-report`

```bash
astro-simclr bench-report runs/rj23-full/seed{0..4} --splits FILE --alerts alerts.csv \
    [--baseline runs/rj23-ms/seed{0..4}] [--split test|val] [--reference-counts N ...] \
    [--n-boot 10000] [--seed 0] [--allow-mixed-machines]
```

A **benchmark run** is a run directory that follows this contract (tasks 3.4/3.6 write it):
- `metrics.json` holds `seed`, `machine` and `benchmark.splits_sha256` (the SHA-256 of the
  object-splits file the run was trained on), and `predictions.<stage>.file`;
- the predictions file is the usual `.npz` with one column per class, and its `index` is
  the **`candid`**.

The report checks, before reporting:
- the splits verify against the table;
- every run names this splits file;
- the predictions cover exactly the split's alerts, once each, with the table's labels;
- seeds are distinct within a side;
- the machines are the same (or `--allow-mixed-machines`).

It reports on two views, **all alerts** and **first stamp**. For each side:

- macro-F1, macro precision, macro recall and accuracy, as mean ± std (ddof = 1) over the
  seeds, and the macro-F1 of the seed ensemble;
- the prior-reweighted macro-F1 and macro precision;
- per class: alerts and objects, precision, recall and F1 (mean ± std), and the **minimum
  detectable effect** of a recall difference (below);
- the confusion matrix in % of each true class (mean over seeds).

With `--baseline` (the positional runs are the variant), the comparison is on the **seed
ensembles**: the mean probabilities per alert, then the argmax.
- **Paired bootstrap of the macro-F1 difference** (`stats.paired_bootstrap_macro_f1`).
  Each resample draws the alerts within every class with replacement (stratified). The
  same draw applies to both sides (paired). Both confusion matrices are recomputed from the
  resample: macro-F1 is not a mean of per-alert scores, because precision depends on the
  false positives of the other classes. The output is the 95 % percentile interval.
  Identical sides give exactly [0, 0], and a golden test checks the implementation against
  a plain one-resample-at-a-time version.
- **McNemar**: the exact test on the ensembles' "correct" vectors.

The report always ends in `REPORT` (exit 0). On Test it is a pre-registered report and never
a decision. Decisions use the selection and decision halves, or a CV, as in phase 2.

**The MDE per class** (`stats.minimum_detectable_effect`) is the smallest recall difference
that a two-sided test at α = 0.05 detects with 80 % power: `(z₀.₉₇₅ + z₀.₈)·SE` = 2.80·SE.
- With a baseline it is paired, with `SE = √(discordant fraction / n)`.
- Without one it is unpaired, with `SE = √(2p(1 − p)/n)`.
- `n` is the number of the class's **objects**, not alerts: the alerts of one object are
  correlated, so using alerts would understate the MDE.
- Example: about 116 test objects at 72 % recall (RJ23's Satellite) give an unpaired MDE
  of about 16.4 points. R5's test set of about 8–9k objects gives far smaller MDEs on its
  large classes.
