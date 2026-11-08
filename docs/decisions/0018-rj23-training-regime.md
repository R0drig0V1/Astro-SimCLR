<!-- ADR 0018: the training regime of RJ23's classifiers (task 3.4b): the schedule policy,
     the class-uniform sampler, Keras' Adam, the two protocols and the CLI path, each
     choice with its source, as engine-free objects with thin Lightning adapters. -->
# 0018: The RJ23 training regime: macro-F1 every 500 steps from step 0, strict patience 5, a 100k cap, stop and best kept, per-sample class-uniform batches, Keras Adam with Keras' ε placement, `--preset rj23-<scenario>`

- **Status:** accepted (2026-11-08, phase 3, task 3.4b)
- **Implemented by:** `src/astro_simclr/rj23_regime.py`, `src/astro_simclr/rj23_data.py`
  (engine-free); `src/astro_simclr/training/rj23.py` (the Lightning adapters);
  `src/astro_simclr/presets.py` (`rj23-<scenario>`), `configs/phase3/rj23-*.yaml`
- **Checked by:** `tests/test_rj23_regime.py`, `tests/test_rj23_data.py`,
  `tests/test_rj23_training.py`, `tests/test_presets.py`

## Context

Reyes-Jainaga et al. 2023 (RJ23, ApJL 952, L43, arXiv:2304.13080) train each scenario's
classifier (ADR 0013) with Adam, unweighted cross-entropy and class-balanced batches of
256, validate the macro-F1 every 500 iterations, stop with patience 5 and save **the model at
the stopping iteration**, not the best one (task 3.0's research of the paper and of their
MIT training script; nothing is copied). Their code uses `tf.keras` defaults, and the class
balance comes from `tf.data.Dataset.sample_from_datasets` over one dataset per class with
uniform weights. Left open: whether iteration 0 is validated, what "patience" counts, a
maximum length, the within-class order of the per-class datasets, and Keras' Adam in torch
(torch's default ε is 1e-8 and placed differently). The plan (3.4b) asks for both kept
weights from one run, a hard cap reported when hit, and the two protocols of phase 3. The
phase-4 rule (ADR 0014) asks for engine-free policy objects with thin adapters, and draws
from `engine.randomness.host_generator` with a reserved stream.

## Decision

1. **Validation grid from step 0.** `step` is the number of optimizer steps done (ADR 0014's
   convention). The loop validates after steps 0, 500, 1000, ...: validation `k` is step
   `500 k`, and step 0 is the initial weights. This is "iteration 0 included"; validating
   after the first update instead would shift the grid by one step out of 500, which no
   number can see. Lightning's validation loop cannot validate at step 0, so the adapter's
   callback runs the evaluation itself (`limit_val_batches=0`).
2. **Patience 5 on a strict improvement**, counted in validations: a validation whose
   macro-F1 is not strictly above the best so far adds one to the wait, a strict
   improvement resets it, and the run stops at the validation that brings the wait to 5
   (Keras' and Lightning's `EarlyStopping` with `min_delta=0`; ADR 0014's `EarlyStop`). An
   equal value is no improvement; NaN never is. The monitored value is the macro-F1 of the
   argmax predictions on the selection part (ADR 0008's zero-division rule).
3. **A hard cap of 100,000 steps** (the plan's; RJ23 state none), required to be a multiple
   of the validation interval so that the cap step is validated. A stop there that patience
   did not cause is `stopped_by: cap` with `capped: true` in `metrics.json`, so the runs that
   hit it are reported. Patience reaching 5 at the cap step is a patience stop.
4. **Both kept weights from one run.** The policy's decision names the weights to snapshot:
   `best` at each new best, `stop` at the stop (both when the cap step is a new best). The
   run writes both (`checkpoints/stop.ckpt`, `best.ckpt`), evaluates both on the evaluated
   parts and writes both sets of predictions. **`stop` is primary** (`rj23_keep: stop`, as
   RJ23 saved): its predictions take the plain names that `bench-report` reads
   (`predictions_<stage>.npz`), the other keep's are `predictions_<stage>_<keep>.npz`.
5. **Per-sample class-uniform batches with replacement.** Each item of a batch draws its
   class uniformly, then an item of that class uniformly (`ClassUniformStream`). This has
   the marginals of `sample_from_datasets` with uniform weights (every class 1/K of every
   batch in expectation, every item of a class equally likely); it does not reproduce the
   within-class order of RJ23's per-class `tf.data` pipelines (their shuffle buffer and
   repeat are not recorded), which only correlates draws within a class. Unlike CD21's
   tiling (`training.tiling`), the batch proportions are random and no class is copied to
   the largest's size. Batch `b` is drawn from `host_generator(seed, b, 5)` only (stream 5,
   `rj23_class_uniform`, reserved in `engine.randomness.RESERVED_STREAMS`), so a batch is a
   pure function of the seed and the step: the stored position of a resume is its step
   count (ADR 0014's `BatchSource` contract), and a resumed run sees the batches of an
   uninterrupted one.
6. **Keras Adam with Keras' ε placement**: β = (0.9, 0.999), ε = 1e-7, and the update
   `θ −= lr·sqrt(1 − β2^t)/(1 − β1^t) · m / (sqrt(v) + ε)` (`KerasAdam`; `keras_adam` with
   Table 2's learning rate), no weight decay. torch's `Adam` adds ε to the bias-corrected
   `sqrt(v̂)`, which equals Keras' only with the step-dependent `ε / sqrt(1 − β2^t)`. ADR 0003
   kept torch's placement for CD21 (ε = 1e-8, where the two agree within 1 % after about
   4,000 steps). RJ23's ε is ten times larger and their runs are shorter (patience 5 at 500
   steps can stop after 2,500 steps), the parity is the task's stated goal, and the exact
   update is a 40-line optimizer with a hand-computed golden test, so we implement Keras'
   (the two differ by 6e-7 on the first step of the golden test). `optimizer: Adam` in the
   presets names it; the `rj23` path never builds torch's.
7. **Unweighted cross-entropy, no augmentation** beyond the model's 8 rotation and flip views
   (ADR 0013), as RJ23. The evaluation batch size is free: BN runs on its moving statistics.
8. **Two protocols** (`rj23_data.MODE_PARTS`; plan 3.4b, "Protocol"). The replication stops
   and selects on the **whole Validation** and evaluates Validation and Test (the paper
   protocol, as 2.4 did for CD21). A phase-3 decision (`--screen`) stops on the
   **selection** half of Validation and evaluates the selection and decision halves; Test
   is never read. Both train on Train. The object splits are verified against the alert
   table (the layout's own, grouped by `oid`, or `--alerts CSV`) and every alert against the
   layout before anything trains.
9. **The CLI path: `train-supervised --preset rj23-<scenario> --splits OBJECT_SPLITS`.**
   `train-supervised` is where supervised stamp classifiers train, and `--preset` is how a
   paper-faithful setting is named (`carrasco2021`): the preset is recorded in
   `metrics.json["preset"]` with every changed key as a deviation, so a run can never pass
   for the replication by accident. One preset per scenario (`rj23-full`, `rj23-cropped`,
   `rj23-lowres`, `rj23-multiscale`, the `VIEW_NAMES` spellings) because Table 2's learning
   rate is per scenario: a single `rj23` preset plus `--set rj23_scenario=...` would make
   the scenario's learning rate a hidden coupling or a deviation. `--splits` takes the
   object-splits file (required), `--screen` keeps its phase-2 meaning (train on Train,
   select on the selection half, predict the decision half), and `--fold` is refused (object
   splits have no folds). The schedule's definitions are settings rules (`monitor:
   macro_f1_val`, `with_features: false`, no EarlyStopping, no workers, the cap and
   checkpoint interval on the grid), errors to change, not defaults.
10. **Engine-free objects, thin adapters.** `RJ23Schedule.decide(step, history)` is a pure
    function (the history is the whole state; a resume recomputes the decision), the
    sampler, the optimizer and `predict_probabilities` import no Lightning (a test checks
    it in a fresh interpreter), and the parts are built by `rj23_data` with numpy only. The
    Lightning adapter (`training.rj23`) holds every part on the training device, validates
    in a callback, and records the run in `manifest.json` (ADR 0015) with the object splits
    hashed as its splits: a decision run is a `screen`, a replication run takes the new
    split part `whole` (ADR 0015's identity requires a part with a splits hash; no existing
    run id changes); the plain engine (phase 4) adapts the same objects.

## Consequences

- One run reports the stop and the best weights; 3.6's gate uses the stop (primary), and the
  best is a reported sensitivity, not a second chance.
- A resumed Lightning run continues the stream, the optimizer, the history and the kept
  weights exactly, but not the dropout masks (Lightning does not restore the global torch
  generator; the plain engine's checkpoint does). A run interrupted mid-fit ends with an
  error that says to resume it.
- The Keras ε placement makes `rj23` runs non-comparable, at the 1e-7 level, with a torch
  Adam run of the same settings; nothing compares them.
- `bench-report` reads the primary keep; reporting the other keep needs a flag there (not
  needed by 3.6).
- The numbers of no existing config change (`numerics_epoch` unchanged): the regime adds new
  settings, a new schedule and a new reserved stream. No existing config hash changes
  either: `rj23_scenario`, `rj23_bn_position` and `rj23_large_values` are hash-neutral at
  their defaults (`runs.canonical.HASH_NEUTRAL_DEFAULTS`), and `rj23_keep` is non-semantic
  (both keeps are computed; it only names the primary).
