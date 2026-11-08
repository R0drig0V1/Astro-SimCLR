<!-- ADR 0016: the feature_set diagnostic of the paper model (all | position | none), and why the selection is in the model (task 2.5). -->
# 0016: The paper model's `feature_set` diagnostic: a column selection inside the model

- **Status:** accepted (2026-11-08, phase 2, task 2.5 diagnostic)
- **Implemented by:** `src/astro_simclr/feature_sets.py`, `Settings.feature_set`
  (`config.py`), `paper_model.PaperStampClassifier`, `paper_variants.PaperHead` /
  `GroupStampClassifier` / `build_paper_model`, `training.modules.PaperClassifierModule`,
  `presets` (`feature_set: all`), `runs.canonical.HASH_NEUTRAL_DEFAULTS`,
  `reference._with_features`
- **Checked by:** `tests/test_feature_sets.py`, `tests/test_paper_variants.py` (the golden
  fixture `tests/fixtures/paper_model/faithful.npz`), `tests/test_split_runs.py`,
  `tests/test_runs_canonical.py` (`tests/fixtures/runs/canonical_configs.json`)

## Context

The user asked how much a model loses with only the images, or with the images and the sky
position. A feature-importance analysis of the replication found 0.943 Validation accuracy
with the 23 features and 0.886 retrained without them, and that the position (`gal_lat`,
`ecl_lat`) carries the VS-vs-AGN and Asteroid-vs-SN cues that the pixels lack: VS cluster
near the Galactic plane, Asteroids near the ecliptic. The images-only model already exists
(`with_features: false`, the `carrasco2021-no-features` preset); images plus position does
not. The diagnostic must be one flag against the replication, must leave the faithful run
bit-for-bit unchanged (including its config hash and its checkpoints), and must work with
the split runs and the other 2.5 flags.

## Decision

1. **One field, `feature_set: all | position | none`, next to `with_features`.** `all` is
   faithful (the default); `position` keeps the 6 sky coordinates of the 23 kept features,
   `ra`, `dec`, `ecl_lat`, `ecl_long`, `gal_lat`, `gal_long` (kept columns 10, 11, 16-19,
   checked by name against the schema and `class_augment.KEPT_FEATURE_NAMES`); `none` reads
   no feature. `with_features` stays: it is the field of every phase-1 module, of the
   presets and of `compare-reference`, and replacing it would move every recorded config.
   `position` with `with_features: false` is a contradiction and an error;
   `aug_feature_jitter` with `none` is an error too (a jitter drawn and then ignored), as it
   already was with `with_features: false`. Only `paper_stamp_classifier` accepts a value
   other than `all`, like the other 2.5 flags.
2. **The selection is part of the model, not of the data.** Every batch still carries the 23
   columns; the model picks its columns (`index_select` with a non-persistent index buffer)
   just before the features BatchNorm, which, with fc2's feature inputs, is sized by the
   selected count (6 for `position`: 439,953 parameters). Selecting in the data would have
   meant a second feature layout through `device_data`, the split runs' pool gathering, C5's
   jitter mask, C7's `magpsf` / `sigmapsf` update and the M1 oracle, all of which index the
   23 columns by name. In the model, they are untouched: C5 still jitters (and C7 still
   updates) all 23, and the head ignores the columns it does not read. The rule of ADR 0014
   holds: `feature_sets.feature_columns` is a pure function, there is no randomness and no
   engine import, and the plain engine will build the same model.
3. **`all` does no indexing, and `none` is `with_features: false`.** With `all` the index
   buffer is `None`, so the faithful forward pass is the same code as before (the golden
   fixture reproduces, once more, with `feature_set: all` explicit). With `none` every
   constructor builds the without-features model: the same layers, the same draws from the
   global generator, the same `state_dict` keys, and `PaperClassifierModule.with_features`
   reads false. A test trains both through `run_command` on the CPU and compares the
   checkpoint tensors and the prediction files byte for byte.
4. **The faithful records do not move.** `feature_set` is recorded in a module's
   hyperparameters only when it is not `all` (as `n_classes` only when not 5), so the
   faithful `hparams.yaml` and checkpoints keep their keys and old checkpoints load with the
   default. It is the first field added after canonical version 1 to
   `HASH_NEUTRAL_DEFAULTS` at `all`, so every existing config hash and run id is unchanged
   (the fixture only gains the `position` and `none` cases).
5. **A preset deviation, never the no-features preset.** `PRESETS` and
   `configs/phase2/carrasco2021*.yaml` list `feature_set: all`, so a `position` or `none`
   run is `faithful: false` with `feature_set` in `metrics.json["preset"]["deviations"]`.
   `preset_for` still chooses the preset by `with_features`, so a `feature_set: none` run
   (with `with_features: true`) is a deviation of `carrasco2021`, not a faithful
   `carrasco2021-no-features` run: the two differ in their recorded config and hash (the
   safe direction of ADR 0015) though they compute the same numbers.
6. **The CD21 gate groups by what the model read.** `compare-reference --ref CD21` counts a
   `feature_set: none` run as a without-features run (the advisory line, not the gate) and
   a `position` run as a with-features run; both get the WARN line of a deviation.

## Consequences

- A run is the preset plus one flag: `train-supervised --preset carrasco2021 --set
  feature_set=position` (or `none`), with `--splits FILE --fold K | --screen` for the CV,
  and combinable with `paper_arch`, `context`, `rot90_augment` and the C5-C7 flags.
- The 23 kept columns stay the contract of the data path; a future subset is one more entry
  of `feature_sets._COLUMNS` (and `Settings.feature_set`'s choices), with no data change.
- `position` changes the numbers by design (a smaller head, other draws for fc2); the
  defaults change nothing, so `NUMERICS_EPOCH` does not move.
