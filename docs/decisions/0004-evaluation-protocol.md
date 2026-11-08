<!-- ADR 0004: design choices of the phase-2 evaluation protocol (task 2.3). -->
# 0004: Evaluation protocol: stored splits, paired tests and the leaked stamp

## Context

Task 2.3 of `docs/phases/02-stamp-classifier.md` defines how every phase-2 result is
measured and compared: the CD21 replication gate, McNemar and a paired bootstrap for two
variants, a 250/250 screen of Validation, a stratified 5-fold CV on Train + Validation with
a bootstrap-CI adoption rule, and the removal of the Train stamp that leaks into
Validation/Test. The plan fixes the rules but leaves several implementation choices open.
Once 2.5 has run against them, changing these choices would invalidate its results, and
future readers will ask why they were made. The rules themselves are in
`docs/evaluation-protocol.md`.

## Decision

1. **The splits are stored, and the stored file is the source of truth.**
   `make-splits` draws them, and `configs/splits/td_ztf_stamp_17_06_20_red-002.json` keeps
   them.
   - The file holds the source SHA-256 and the label SHA-256 of Train and Validation, and
     `verify_splits` refuses another dataset.
   - numpy does not promise that a seeded `Generator` gives the same permutation in every
     version, so regenerating is a check, not the definition.
   - Each part has its own generator, `default_rng([seed, part, ...])`. These are
     independent of the global generators that `seed_run` sets, so a training seed never
     moves a split.
2. **Pool indices.**
   - The pool is Train then Validation: Validation stamp `j` is pool index `n_train + j`.
   - The CV file stores one character per pool stamp: its evaluation fold, or `-` for an
     excluded stamp. Each fold's selection set is stored as a sorted list.
3. **The bootstrapped quantity is the seed-mean accuracy.**
   - Each stamp's score is the fraction of seeds that classify it correctly, so the
     observed difference is the difference of the 5-seed mean accuracies, which is what the
     tables report. McNemar instead uses the seed ensemble, as the plan says.
   - The CV variant concatenates every fold's held-out scores (each pool stamp is held out
     once) and bootstraps the **balanced accuracy**.
4. **The bootstrap is stratified by class**, a percentile interval with 10,000 resamples
   by default.
   - The evaluation sets are balanced by design (100 or 50 per class; the CV folds are
     stratified), so the class mix is not a source of sampling noise.
   - Stratifying also makes accuracy equal balanced accuracy on balanced sets.
5. **The adoption rule is one-sided in effect.** `adopt` needs the whole 95 % interval
   above 0. An interval below 0 is `worse`, and anything else is `inconclusive`. Only
   `adopt` adopts.
6. **McNemar is exact.** It is the binomial two-sided test, computed with exact integers,
   so there is no continuity-correction choice and no overflow for large discordant counts.
7. **Duplicates are exact uint8 matches.**
   - The prepared uint8 stamps are hashed (BLAKE2b-128) and compared across every pair of
     splits. Identical float stamps are always identical in uint8.
   - On `red` this finds one duplicate: Train 2995 (Bogus) = Validation 115 (SN), with the
     same features. It is removed from Train only, as the plan says. Validation keeps it,
     with its SN label, so that Validation stays the paper's 500 stamps.
8. **The CD21 gate is defined on exactly 5 seeds.** Another count gives `REPORT`, never
   `PASS`. A confirmatory rerun is another 5-seed set, judged on its own. The faithful tier
   reads the M1 oracle from a JSON report whose top-level `passed` is a boolean (the
   contract for task 2.2b).
9. **Every run writes per-stamp probabilities** (`predictions_{val,test}.npz`), not only
   summary metrics. Paired tests need them, and they are small: 500 × 5 floats per split.

## Consequences

- 2.5 results are tied to the committed splits file. A new dataset, or a deliberate change
  of the splits, needs a new file and new baselines.
- The training data modules must take index subsets (a fold's training and selection
  parts, the screen's halves) and drop `excluded_train`. They must also write predictions
  whose `index` is the Validation index (screen) or the pool index (CV), which the decision
  helpers check. Done in task 2.5-infra (see the amendment below).
- Runs made before task 2.3 have no predictions, per-class recall or machine block. The
  CD21 gate fails them clearly, and `compare-variants` asks for a re-run.
- The Validation stamp 115 label conflict (SN vs Bogus for the same alert) stays in the
  evaluation set, as in the paper. It costs at most 1 of 500 stamps (0.2 points).

## Amendment (task 2.5-infra, 2026-11-08): split runs and the decision command

10. **Split runs are separate runs with their own outputs.** `train-supervised --splits
    FILE --fold K | --screen` (the `carrasco2021` schedule only) maps the run's parts onto
    the device data module's three roles (training → `Train`, selection → `Validation`,
    evaluation → `Test`) instead of adding a new data module, so the faithful path stays
    bit-for-bit the same. Test is never read. The metrics have no `val_acc_best` or
    `test_acc_at_best`, so a split run cannot enter the CD21 gate, and a normal run cannot
    enter a decision. The flags are CLI arguments, not `Settings` fields: the part a run
    used is recorded in `metrics.json["protocol"]`, and the preset report stays about the
    training settings.
11. **The splits file is verified at training time, and its SHA-256 pairs the runs.** A run
    refuses a file whose fingerprint (source and label SHA-256, sizes) or leakage check
    does not match the prepared data; it records the file's SHA-256, and
    `compare-variants --splits FILE` refuses runs trained on another file. The screen
    keeps the Validation index in its predictions (decision 2 and the consequence above),
    the folds the pool index.
12. **Non-inferiority for the efficiency rows.** The superiority rule (5) cannot adopt a
    change that is only "as good". For the rows whose point is a lower cost (B7, B2′, B1′),
    the plan adopts when the lower bound of the same 95 % interval is above −0.5 points.
    The margin is a fraction (0.005); values above 0.05 are refused, because the likely
    mistake is a margin typed in points. The rule is chosen per row before the runs, not as
    a fallback.
13. **Exit codes of a decision** follow `compare-reference`: 0 for ADOPT or KEEP, 1 for NOT
    ADOPTED or DROP (and unusable data), 2 for invalid arguments or runs that cannot be
    compared (another splits file, a missing fold or seed, mixed modes or machines).
14. **A resume continues only the same split run** (found in the review of PR #17). The
    stream check (item count, batch size, seed) cannot tell two folds apart, because their
    tiled sizes can be equal. So the run's identity (splits SHA-256, mode, fold) goes into
    the training loader's checkpoint state, and a resume from another fold, the screen or a
    normal run, or a normal run resumed from a split run's checkpoint, is refused before
    any step. Normal runs store no identity, so their checkpoints are unchanged, and
    checkpoints from before this change read as normal runs.

## Amendment (grouped splits, 2026-11-08): no object in two parts

The integrator's audit found that `red` removed exact duplicates only: 24 of the 500
Validation stamps and 27 of the 500 Test stamps have a distinct Train alert within 1.5"
(mostly Bogus), and Train holds groups of up to 16 alerts. The version-1 splits were
stratified but not grouped, so a CV fold could train and evaluate on the same artefact,
which favours flags that memorise more. This amendment supersedes decision 7's scope (exact
duplicates only) for the 2.5 decisions; decision 7 itself still holds.

15. **Group by sky position by default.** `ra`/`dec` de-standardised with M1's statistics,
    linked within **1.5"** (about one ZTF pixel plus astrometric scatter; a parameter), and
    taken as connected components (union-find). Transitive chaining is accepted: an object
    that drifts or a dense artefact is one group even if its ends are farther apart than the
    radius, and over-grouping only costs balance, never leakage. The search is an exact
    dec-sorted window (the separation is never below the `dec` difference, so the window is
    exact at the poles and across the RA wrap) with the haversine formula: numpy only, no
    scipy dependency for a k-d tree that would save under a second on `red`. An external id
    (`--group-ids`, a `.npz` per split; the `oid`s of task 3.1) can replace positions.
16. **The pool is grouped over Train ∪ Validation only.** Test never enters a decision, so a
    Test alert must not merge two pool groups; `leakage-audit` groups all three splits for
    the report.
17. **A new file, format version 2, beside the old one.**
    `configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json` holds the grouping (method,
    radius, the groups of two or more pool stamps, `screen_excluded_train`, the screen
    fallback) and is the file of every 2.5 decision. The version-1 file stays unchanged and
    readable (`read_splits` accepts both, `write_splits` writes an object back in its own
    version), so phase-2 results made on it stay reproducible. `verify_splits` re-groups the
    data and refuses a file whose groups differ or that splits a group across two parts.
18. **Group k-fold by greedy placement.** Groups are placed largest first (seeded random
    ties) into the fold that least raises `sum_c n_gc * count_fc / total_c`. Per class, two
    folds differ by at most the most stamps of that class in one group, which
    `verify_splits` uses as its tolerance (1 on `red`, as in version 1).
19. **Selection sets from whole groups, exact sizes.** A fold draws groups outside its
    evaluation part in random order; a group that would overfill a class gives only the
    stamps needed, and its other stamps are *unused* in that fold (out of the training part
    as well). This keeps 100 per class exactly and draws groups, not stamps, uniformly; the
    alternative (only whole groups, sizes approximate) would change the selection metric's
    noise from fold to fold.
20. **The screen leaves Train's group-mates of Validation out of training.** The screen
    trains on Train and evaluates on Validation, so the 107 Train alerts whose group has a
    Validation member leave its training set, as the leaked stamp does in decision 7.
    Validation keeps its 500 stamps. The halves keep groups whole. If that makes 50 + 50
    per class impossible, both halves are trimmed to the largest common per-class count
    (recorded in the file, printed by `make-splits`) rather than given unequal classes; on
    `red` it is not needed.
21. **The paper replication (2.4) keeps CD21's splits.** Its number must stay comparable
    with 0.941, so it is reported with the caveat that about 20 % of the Validation and
    Test Bogus share a position with Train.
