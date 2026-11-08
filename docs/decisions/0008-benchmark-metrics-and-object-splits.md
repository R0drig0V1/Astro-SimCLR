<!-- ADR 0008: design choices of the phase-3 benchmark metrics and object splits (task 3.2). -->
# 0008: Benchmark metrics and object-grouped splits (phase 3)

- **Status:** accepted (2026-11-08, phase 3, task 3.2; with the amendments of the phase-3
  plan review)
- **Code:** `src/astro_simclr/protocol/{alerts,grouping,object_splits,benchmark_metrics,
  bench_report}.py`, `stats.paired_bootstrap_macro_f1`, `stats.minimum_detectable_effect`
- **Rules:** `docs/evaluation-protocol.md`, "Phase 3 benchmark metrics"

## Context

Phase 3 (`docs/phases/03-newer-benchmarks.md`) builds R5, an object-level benchmark under
RJ23's protocol from public sources. It needs:
- 5 imbalanced classes (RJ23's six without Satellite, out of scope), with N classes kept
  general for a later 6-class set;
- macro-F1, with the macro-F1 reweighted to RJ23's class counts as the headline;
- a 60/20/20 stratified split grouped by object;
- a first-stamp view.

Several alerts of one object look alike, so a split by alert leaks. The phase-2 tools assume
5 balanced classes and a per-stamp mean metric. Once the replication runs (task 3.6) are
made against these choices, changing them would invalidate those runs.

## Decision

1. **Zero-division rule of macro-F1.**
   - 0/0 is taken as 0.
   - A class with true samples that is never predicted counts in the macro means with
     F1 = 0, and so does a class that is predicted but absent from the truth.
   - Only a class absent from both the labels and the predictions is left out (`None`).
   - This is torchmetrics' rule, and so phase 2's. The alternative, leaving out any class
     with an undefined precision, would reward a model for never predicting a hard class.
2. **Prior-reweighted macro-F1 is reported next to the raw one.**
   - Each confusion row is rescaled to a reference count, which changes precision only.
   - Scaling every reference count by one factor changes nothing, so only the
     proportions matter. The default reference is RJ23's class totals
     (`RJ23_CLASS_COUNTS`, `rj23_reference(label_names)`): a stratified 60/20/20 split
     keeps them in its test set up to rounding, so no test counts are needed.
   - With a class that is not one of RJ23's there is no default.
3. **The macro-F1 bootstrap recounts the confusion matrices.**
   - Macro-F1 is not a mean of per-item scores, because precision depends on the false
     positives of the other classes. So `stats.paired_bootstrap` (per-stamp seed-mean
     scores) cannot be reused.
   - `paired_bootstrap_macro_f1` keeps its sampling design: stratified by class, the same
     draw for both sides, its own generator and memory-bounded blocks. It recounts every
     side's confusion matrix per resample, as one matrix product.
   - The report bootstraps the **seed ensembles** (mean probabilities, then argmax), as
     McNemar does.
   - A golden test checks the implementation against a one-resample-at-a-time version.
4. **Groups are assigned whole, per class, in random order, by the largest deficit.**
   - The bound: no split ends more than `g` above its target or more than `(k − 1)·g`
     below it (`g` is the class's largest group). `verify_object_splits` checks it.
   - The order is random, not largest-first. On a synthetic RJ23-like table, largest-first
     put 45 % of the objects but 60 % of the alerts in Train, so Test held smaller objects
     and a different first-stamp view.
   - A class that cannot give every split a group is refused.
5. **The group key is any per-row id.**
   - It is the `oid` by default, or a `group` column, such as a position-group id built
     elsewhere (the "grouped-splits" task). This code never builds groups.
   - The group is part of the table's fingerprint when it is given.
   - A group with two labels is refused, or dropped and recorded with `--drop-conflicts`.
6. **The splits are stored by group id, with the table's fingerprint.**
   - The fingerprint is the SHA-256 of the canonical rows sorted by `candid`, so it does
     not depend on the row order.
   - As in ADR 0004, the stored file is the source of truth, not the generator.
   - Validation's grouped, stratified **selection and decision halves** are stored too, so
     phase-2-style decisions are possible on this set.
7. **Per-alert predictions are indexed by `candid`.** It is the alert's own id, so it
   survives any reordering of the table. A benchmark run records the SHA-256 of the splits
   file it was trained on (`benchmark.splits_sha256`), and the report refuses runs of
   another file.
8. **The first-stamp view is the earliest alert in the table** (smallest `jd`, then
   smaller `candid`). It is RJ23's view only if the table holds each object's first
   detection: tasks 3.1/3.5 must make sure it does, or document the difference.
9. **The cross-benchmark rule follows the plan's matrix** (task 3.2,
   `CROSS_BENCHMARK_ALLOWED`).
   - Allowed: Train/Train, and any pair of Validation and Test.
   - Forbidden: one benchmark's Train with the other's Validation or Test, either way.
   - `allowed=()` refuses all nine pairs.
   - The function returns the 3 × 3 matrix of shared counts, from which the pretraining
     exclusion lists are built. The group id is the `oid` where matched, otherwise a 1.5″
     position group built across both benchmarks, by another task.
10. **The per-class MDE is computed on objects, not alerts.** The alerts of one object are
    correlated, so counting them would understate the MDE. With a baseline it is paired
    (from the ensembles' discordance); without one it is unpaired (from the recall).

## Consequences

- `Predictions` takes `n_classes` (default 5), so every phase-2 file, check and output is
  unchanged. R5's label names are the phase-2 five (`BENCHMARK_LABEL_NAMES`); RJ23's six
  (`RJ23_LABEL_NAMES`) are a `--labels` away. The 5-class modules tied to the prepared layout (`splits`, `device_data`,
  `prepared`) stay 5-class: they describe that dataset.
- Tasks 3.4/3.6 must write benchmark runs to the contract of `bench_report` (`seed`,
  `machine`, `benchmark.splits_sha256`, `predictions.<stage>.file` with one column per
  class and `index` = `candid`).
- No object-splits file is committed until task 3.1 gives the real alert table. A
  regenerated file needs new runs.
- A very small class has a large MDE: about 116 test objects at 72 % recall (RJ23's
  Satellite) give an unpaired recall MDE of about 16 points, so per-class claims on small
  classes need large effects.
