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
  helpers check. This wiring belongs with the GPU-resident data module of task 2.2a.
- Runs made before task 2.3 have no predictions, per-class recall or machine block. The
  CD21 gate fails them clearly, and `compare-variants` asks for a re-run.
- The Validation stamp 115 label conflict (SN vs Bogus for the same alert) stays in the
  evaluation set, as in the paper. It costs at most 1 of 500 stamps (0.2 points).
