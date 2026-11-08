<!-- ADR 0003: what the carrasco2021 preset reproduces of Carrasco-Davis et al. 2021, and why (task 2.2c). -->
# 0003: The `carrasco2021` preset: ALeRCE's code as the authority

- **Status:** accepted (2026-11-08, phase 2, task 2.2c)
- **Implemented by:** `src/astro_simclr/presets.py` (`train-supervised --preset
  carrasco2021`), `configs/phase2/carrasco2021.yaml` and `carrasco2021-no-features.yaml`,
  `training.device_data.float_crops`, `training.optimizers.PAPER_ADAM_BETAS`
- **Checked by:** `tests/test_presets.py`, `tests/test_carrasco2021.py`,
  `tests/test_paper_model.py`; the M1 oracle of task 2.2b

## Context

Phase 2 replicates the ALeRCE stamp classifier of Carrasco-Davis et al. 2021 (AJ 162, 231;
arXiv:2008.03309), the published model "M1", whose test accuracy is 0.941 ± 0.004 with
features (`docs/phases/02-stamp-classifier.md`). Three sources describe it: the paper text,
ALeRCE's TF1 code (`alercebroker/pipeline@a3705ac`, `stamp_classifier_step/model/`) and the
M1 checkpoint with its meta graph and event log. They disagree in several places, and a
replication must pick one answer for each. The answers are hard to reverse (every 2.4 and
2.5 result is measured against this baseline) and several look wrong to a reader of the
paper alone, so they are recorded here. The facts behind them were verified in the plan
review (`docs/plan-reviews/2026-11-08-phase-2.md`) and in tasks 2.1 and 2.2a.

## Decision

1. **The code is the authority over the paper text.** The code (and the checkpoint it
   wrote) produced the published number; the text is a description of it. Where they
   disagree, the preset follows the code, and the paper's value becomes at most a
   diagnostic (decisions 3, 5).

2. **The feature BatchNorm is frozen at (0, 1).** In M1, the feature BN's `moving_mean` is
   all 0 and its `moving_variance` all 1, while `gamma` and `beta` were trained: the Keras
   BN's update ops never reached TF1's `UPDATE_OPS`. So training normalised the features
   with batch statistics, and evaluation and selection used `gamma · x / sqrt(1 + 1e-3) +
   beta`. `PaperStampClassifier` uses `BatchNorm1d(23, eps=1e-3, momentum=0.0)`: the same
   arithmetic in both modes (task 2.2a, tested).

3. **Dropout is inactive** (`dropout_active: false`). ALeRCE calls Keras `Dropout` without
   a `training` argument, so it follows `keras_learning_phase`, a `PlaceholderWithDefault`
   that defaults to False and is never fed: the dropout branches never ran, in training or
   evaluation. The layers are kept (after fc1 per rotation, fc2 and fc3) with the code's
   rate 0.5 (`drop_rate`), used only by the `dropout_active: true` diagnostic. The paper's
   Table 2 (dropout on fc2 only, a "90×64" layer) is a description error.

4. **The fc1–fc3 biases are Glorot-uniform**, not zero. ALeRCE passes
   `bias_initializer=None`, and Keras falls back to `glorot_uniform` (M1's graph:
   `dense_*/bias/Initializer/random_uniform`). The convolution and logit biases are zero.
   Only the distribution is reproduced, not TF's random numbers.

5. **Adam: β = (0.9, 0.999), ε = 1e-8, lr 1e-3, no weight decay** (`adam_betas: code`).
   ALeRCE's `modules/optimizers.adam(loss, lr, beta1=0.9, beta2=0.999)` never receives the
   `BETA1: 0.5, BETA2: 0.9` of its parameter defaults (the paper text's values), and M1's
   `beta2_power = 0.999^4782` confirms β2. `adam_betas: paper` gives (0.5, 0.9), a
   diagnostic only (and the phase-1 modules keep their legacy (0.5, 0.9) whatever the
   setting says).

   **The ε placement differs, and is kept as torch's.** TF1's `AdamOptimizer` updates
   `θ -= lr_t · m / (sqrt(v) + ε)` with `lr_t = lr · sqrt(1 - β2^t) / (1 - β1^t)`: ε is
   added to the *uncorrected* `sqrt(v)`. torch updates `θ -= lr · m̂ / (sqrt(v̂) + ε)`.
   TF1 is therefore torch with `ε_t = ε / sqrt(1 - β2^t)`: 3.2e-7 at step 1, 1.3e-8 at
   step 1,000, within 1 % of 1e-8 after about 4,000 steps. The two differ only where
   `sqrt(v̂)` is of the order of 1e-7, that is for parameters whose gradients are about
   that small, early in training; we judged that negligible next to the seed noise
   (σ ≈ 0.004) and did not write a custom optimizer. If the gate fails and the M1 oracle
   rules out the input and the model, a TF1-placement Adam is a cheap candidate to add.

6. **Without features, M1's hyperparameters are assumed** (`carrasco2021-no-features`).
   The paper gives 0.883 ± 0.006 without stating how it was obtained. ALeRCE's search
   script for it uses another split (60 per class) and `DeepHitsExtraLayer`, which is
   `PaperStampClassifier(with_features=False)` (439,557 parameters). We train that network
   with every other setting of the preset on the 100-per-class split. Because of the
   assumption, the CD21 gate only reports this line (advisory).

7. **The NaN approximation is on** (`nan_approximation: true`). The pickles hold the stamps
   after a 63×63 min-max and NaN → 0, so ALeRCE's crop normalisation, which skipped NaNs,
   cannot be repeated exactly. The approximation (task 2.1, `data.float_input`) re-marks a
   channel's zeros as former NaNs when the channel has more than one exact zero over the
   full stamp. It is exact when the channel's minimum lies outside the crop. On `red` it
   touches 41 of the 500 Validation stamps (`data.float_input.approximated_stamps`; task
   2.2b reports their accuracy separately). `nan_approximation: false` treats
   every stored zero as a pixel, a diagnostic. The input is the stored **float32** stamps
   (`paper_input: float32`; `prepare-data --images float32`); the interim uint8 crops of
   task 2.2a stay as the `paper_input: uint8` diagnostic.

8. **The leaked Train stamp is removed from Train only** (`exclude_leaked_train: true`).
   `red` Train 2995 (Bogus) is byte-identical to Validation 115 (SN) (ADR 0004). The run
   finds the cross-split duplicates with `protocol.leakage` (the check `make-splits` stores
   and `verify_splits` repeats), drops them from Train **before** the tiling, and records
   them in `metrics.json["paper_data"]["excluded_train"]`. Validation and Test keep their
   500 stamps, so the numbers stay comparable with the paper's. The tiling then balances
   the 50,581 remaining stamps (Bogus loses one stamp, which changes neither the tiled size
   nor the largest class).

9. **A preset is a settings layer, and every deviation is recorded.** `--preset NAME`
   applies `presets.PRESETS[NAME]` after the legacy protocol and before `--hparams`,
   `--config`, `--set` and the flags, so any value can still be changed. The run's
   `metrics.json["preset"]` lists every preset key whose value differs (`faithful: false`)
   and the diagnostic flags turned on, and `compare-reference --ref CD21` expects the
   preset's values, so a diagnostic run prints a WARN line there. The config files hold
   the same values for `--config` users; a test keeps them equal.

## Consequences

- The replication is defined by `presets.PRESETS`, not by the paper text. A reader who
  compares the preset with the paper sees (0.9, 0.999), no dropout and Glorot biases; this
  ADR is the reference for why.
- Diagnostics are explicit `--set` changes on top of the preset, recorded in the metrics,
  and never mistaken for the replication. The order in which 2.4 tries them is the plan's.
- TF's random numbers (initial weights, the shuffle stream), its convolution numerics and
  the validation grid's one-step shift (task 2.2a) are not reproduced. The M1 oracle (task
  2.2b) checks the network and the input without training; what remains is training
  dynamics, measured as seed variance.
- The preset keeps `deterministic: false` (the `Settings` default, as the legacy runs): a
  run is seeded (weights, batch stream) but not bit-identical across GPUs or reruns.
- A run needs the float32 layout (2.4 GB for `red`); without it the run fails with a
  `DataError` naming the `prepare-data --images float32` command.
- The leakage check reads every uint8 stamp once per run (a few seconds on `red`).
