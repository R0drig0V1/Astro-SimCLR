<!-- Architecture of src/astro_simclr/: one section per module. Each phase-1 task edits only
     its own section, so parallel PRs do not conflict. Plan: docs/phases/01-runnable-package.md. -->
# Architecture

`src/astro_simclr/` is the typed, tested package that replaced the legacy research code
(`utils/`, `models/`, `losses/`, the root scripts and `env.yml`). Phase 1 ported the legacy
modules one task at a time, each against a characterization fixture generated from the
legacy code; task 1.8 then deleted the legacy code from the tree.

**Legacy code.** Every legacy path cited on this page and in the docstrings (`utils/training.py`,
`iter_clr.py`, ...) is readable at the git tag `legacy-final`, the last commit of `main` that
had it: `git show legacy-final:utils/training.py` (or `git fetch origin tag legacy-final`
first). Paths cited with another commit (`b357b43^`, `31b10c9`) are read at that commit. The
generators in `scripts/characterize/` import the legacy code through
`scripts/characterize/legacy_tree.py`, which names the tag once (`LEGACY_REF`; the
environment variable `ASTRO_SIMCLR_LEGACY_REF` overrides it) and copies the legacy packages
into a temporary directory. Their recorded `legacy_sha`/`legacy_commit` is the commit the
tag points at; fixtures generated before task 1.8 recorded the `HEAD` they ran at, which had
the same legacy files. The notebooks in `jupyter_examples/` import the legacy code too: run
them from a checkout of the tag.

```
cli ──► config, hparams (subcommand defaults, legacy key mapping), presets (carrasco2021)
 │
 ├──► data ──────► augmentations (protocol)      registry of transforms (1.5)
 │                                    ▲
 ├──► experiments (one training run) ──► training (Lightning 2) ──► models, losses, optim, metrics
 │                                          datamodules ──► data (prepared layout) + augmentations
 ├──► reference (compare-reference: the gates of R1-R4 and CD21)
 ├──► protocol (make-splits, compare-variants: the phase-2 evaluation protocol) ──► data
 ├──► oracle (fetch-m1, m1-oracle: the published M1 vs our input path) ──► paper_model, data
 ├──► alerts (fetch-alerts, alerts-summary: raw alerts from ALeRCE) ──► data (prepared, m1_features)
 ├──► r5 (r5-sources, build-r5: the R5 benchmark) ──► alerts (by_oid, store), protocol, data.raw_alerts
 │    (training ──► paper_variants: rot4 / single / p4 / p4m ──► paper_model, group_conv;
 │     paper_model ──► feature_sets: all / position / none)
 ├──► preflight ──► experiments (R1-R4), tasks.plain (CD21), data (inspection, manifest)
 ├──► tasks (phase 4, 4.6: the paper classifier on the plain engine; no Lightning) ──► engine,
 │      training (Lightning-free submodules: paper_augment, tensor_splits, class_augment, ...)
 │    (experiments ──► tasks.common, tasks.paper, tasks.results: shared with Lightning)
 └──► engine (phase 4: the contracts, the loop, the data tiers; imports no Lightning) ──► config, data, runs
 └──► runs (phase 4, A0: run identity, manifest.json, layout, retention) ──► config, data
 └──► registry (phase 4, A2: runs reindex | ls | show | compare; SQLite) ──► runs, reference
 └──► simclr (phase 5, 5.0b: the SimCLR pilot; imports no Lightning) ──► engine.randomness,
      optim (LARS), losses (NT_Xent), stamp_relation, protocol (splits, stats), data

rj23_model (engine-free nn.Modules, not wired yet: task 3.4b) ──► data.rj23_input
```

Conventions shared by every module:

- **Settings** come from `config.Settings`, never from module-level constants.
- **Randomness:** the global streams come from `lightning.seed_everything(seed,
  workers=True)`; per-step draws (augmentations) come from generators keyed by
  `(seed, step, stream)` (ADR 0006, ADR 0014: `engine.randomness`), never from an unseeded
  or global generator.
- **Devices:** no module hard-codes CUDA. The accelerator comes from the settings, and tensors
  follow their module's device.
- **Behaviour changes** against the legacy code are listed in the section of the module that
  makes them, with the legacy flag that restores the old behaviour, if any.
- **Typing:** mypy strict; `# type: ignore[override]` is allowed only on Lightning hook
  overrides (whose signatures Lightning types loosely).

## Environment and torch wheels

Runtime dependencies: torch, torchvision, lightning 2.x, torchmetrics, numpy, pyyaml,
matplotlib, tensorboard, fastavro, and timm (the ConvNeXt rows; it brings huggingface-hub and
safetensors, and resolves against each fork's torch: [ADR 0022](decisions/0022-convnext-native-stamp.md)). torch and torchvision come from PyPI by default (MPS on macOS,
CUDA 13 on Linux), or from the PyTorch indexes through the conflicting extras `cpu` and
`cu126`. After `uv sync --locked --extra X`, run everything with `UV_NO_SYNC=1` or
`uv run --no-sync`. Design and rationale: [ADR 0001](decisions/0001-torch-wheels-and-accelerators.md);
`scripts/check-torch-wheels.sh` asserts it in CI. CI runs on pushes to `main` (ubuntu),
weekly on macOS and on demand, never on pull requests: the integrator's `verify-merge.sh` is
the PR gate ([ADR 0021](decisions/0021-ci-on-main-pushes-only.md)).

## `config` (task 1.1)

`Settings`: a frozen, keyword-only dataclass with the hardware (`accelerator`, `devices`,
`num_workers`), paths (`data_dir`, `output_dir`), reproducibility (`seed`, `deterministic`)
and training protocol (`max_epochs`, `patience`, `min_delta`, `monitor`, `eval_batch_size`)
of one run.

- `load_settings(path, overrides)`: defaults, then the YAML file, then `key=value`
  overrides (later wins). YAML is read with `yaml.safe_load` only.
- Unknown keys, wrong types (`seed: "1"`, `devices: true`) and out-of-range values raise
  `ConfigError` (a `ValueError`); integers are accepted for float fields and strings for
  path fields, nothing else is converted.
- Defaults: `accelerator: auto`; `devices: 1` (`NT_Xent` has no all-gather, so DDP would
  change the loss); `deterministic: false` (the legacy runs used `benchmark=True`); the
  protocol defaults are the legacy classifier values (220 epochs, patience 40,
  `min_delta` 0.001, `accuracy_val`, eval batch 100).
- `fp32_precision: ieee | tf32` (ADR 0010; default `ieee`): the float32 precision of cuDNN
  convolutions and of matmuls. `ieee` is full float32 on every device; `tf32` lets Ampere+
  GPUs use TensorFloat-32 (torch 2.14's own default for convolutions). Set at the start of
  every run (`training.runs.apply_fp32_precision`) and recorded, with its effect, in
  `metrics.json["machine"]`.
- Later tasks add their fields here (for example the NT-Xent denominator flag of 1.3).
- **Task 1.7 fields** (flat, so every one can be `--set`): EarlyStopping `check_finite`
  (default true, Lightning's default) and `divergence_threshold` (null: never; R1 uses
  0.1); `limit_train_batches` / `limit_val_batches` (null: every batch; the dry mode sets
  2); `progress_bar_refresh_rate` (added by audit item 2; default 50: redraw Lightning's
  progress bar every 50 batches; 0 hides it; no effect on the numbers, not a preset key);
  and the hyperparameters of the training subcommands. Each subcommand reads only its
  own; the others are recorded in the run's metrics but unused:

  | Field | Default | Read by |
  |---|---|---|
  | `dataset` | `td_ztf_stamp_17_06_20_red-002`, the received 100 % file (the run reads `data_dir/<dataset>`) | all four |
  | `image_size`, `batch_size`, `lr`, `optimizer`, `augmentation` | 27, 45, 0.002495…, `AdamW`, `center_crop` | all four (`pretrain`: encoder lr and optimizer) |
  | `beta_loss`, `drop_rate`, `with_features` | 0.01606…, 0.2, true | the classifiers (`drop_rate`: stamp classifier, fine-tuning head) |
  | `supervised_model` | `stamp_classifier` (or `resnet18/34/50/152`) | `train-supervised` |
  | `encoder_name`, `projection_dim`, `temperature`, `lars_legacy_trust_ratio` | `resnet18`, 300, 0.1791…, true | `pretrain` |
  | `pretrained` | null (a checkpoint or a pretrain output directory) | `linear-eval`, `fine-tune` |
  | `fine_tune_lr_factor` | 0.2 (legacy `fine_tuning.py`: `lr=0.2*hparams.lr`) | `fine-tune` |

  The classifier defaults are `results/hparams_best_ce.yaml` with `AdamW` (the optimizer of
  every tuned trial behind it; the plan's known defects). Optional fields accept `null`.
  Float fields also accept numeric strings such as `1e-3`, which YAML 1.1 (PyYAML) does not
  read as a float. `hparams.LABEL_FRACTION_DATASETS` names the 1, 10 and 100 % datasets by
  the received files; the legacy 100 % `td_ztf_stamp_17_06_20.pkl` was never received, and
  `red-002` (the same file without duplicate stamps) stands in for it.
- **Task 2.2a fields** (the `carrasco2021` schedule; see `training`, "The carrasco2021
  schedule"):

  | Field | Default | Meaning |
  |---|---|---|
  | `schedule` | `epochs` | `epochs` (the legacy protocol) or `carrasco2021` (ALeRCE's iteration schedule) |
  | `max_steps` | 30000 | `carrasco2021`: initial horizon in optimizer steps (`Trainer(max_steps)`) |
  | `val_every_n_steps` | 10 | `carrasco2021`: validation and selection interval |
  | `horizon_increment` | 10000 | `carrasco2021`: a new best at step `s` raises the horizon to `s + increment` |
  | `checkpoint_every_n_steps` | 1000 | `carrasco2021`: `last.ckpt` interval, a multiple of `val_every_n_steps` |
  | `early_stopping` | true | `epochs`: false drops EarlyStopping; `carrasco2021` requires false |
  | `dropout_active` | false | `paper_stamp_classifier`: apply its dropout layers (rate `drop_rate`) |

  Under `carrasco2021` the schedule's definitions are errors to change, not defaults:
  `eval_batch_size` must be 64 (the selection loss is TF's mean of batch-64 means),
  `early_stopping` false and `num_workers` 0 (`ConfigError` otherwise). The
  `carrasco2021` preset (task 2.2c, see `presets`) sets them.
- **Task 2.2c fields** (read by `paper_stamp_classifier` only; ADR 0003):

  | Field | Default | Meaning |
  |---|---|---|
  | `paper_input` | `float32` | `float32`: `images_f32.npy` through `crop_split(21)` (the paper's input); `uint8`: the interim uint8 crops (a diagnostic) |
  | `nan_approximation` | true | `float32` input: `crop_split(nan_approximation=...)`; false is a diagnostic |
  | `exclude_leaked_train` | true | drop the Train stamps that duplicate a Validation/Test stamp (`protocol.leakage`), from Train only |
  | `adam_betas` | `code` | Adam/AdamW betas: `code` (0.9, 0.999), `paper` (0.5, 0.9; a diagnostic). The phase-1 modules keep (0.5, 0.9) |
- **Task 2.5-models fields** (the paper model's architecture variants; `paper_variants`,
  ADR 0005). The defaults are the faithful model, and the `carrasco2021` presets list them,
  so a variant run is a recorded deviation:

  | Field | Default | Meaning |
  |---|---|---|
  | `paper_arch` | `rot4` | `rot4` (faithful: 4 rotations, mean after fc1), `single` (one pass, not invariant), `p4` (B7), `p4m` (B2') |
  | `group_fields` | null | `p4` / `p4m`: fields of conv1-conv2 (conv3-conv5 twice); null is `32 // \|G\|` (8 / 4). An error with `rot4` / `single` |
  | `context` | `none` | `avgpool63`: the 63×63 stamp average-pooled to 21×21 as 3 more channels (B1'); needs `paper_input: float32` |
  | `rot90_augment` | false | rotate each training stamp by a random multiple of 90° (seeded by `seed` and the step) |
  | `feature_set` | `all` | the features the head reads (a diagnostic; ADR 0016): `all` 23 (faithful), `position` the 6 sky coordinates (`ra`, `dec`, `ecl_lat`, `ecl_long`, `gal_lat`, `gal_long`), `none` the images only (bit-identical to `with_features: false`). `position` needs `with_features` |
- **Plan rows C5 and C6** (the paper model's class-aware training augmentation;
  `training.class_augment`, ADR 0006). Off by default, listed in the presets like the
  2.5-models flags:

  | Field | Default | Meaning |
  |---|---|---|
  | `aug_noise` | 0.0 | C5: the largest background-noise scale (in units of each crop channel's background sigma) added to the science and template; the difference follows. ≥ 0; needs `paper_input: float32` |
  | `aug_feature_jitter` | 0.0 | C5: Gaussian jitter sigma (standardised units) on the continuous features; the discrete ones never change. ≥ 0; needs `with_features` and a `feature_set` other than `none` |
  | `synthetic_bogus` | 0.0 | C6: fraction of each training batch replaced by synthetic hot-pixel Bogus, in [0, 1); needs `paper_input: float32` and `context: none` |
  | `aug_min_r2` | 0.5 | C5 / C6: stamps whose `diff ~ a·sci + b·tmpl + c` fit has a lower R² are not noised nor used as C6 sources; in [0, 1] |
- **Plan row C7** (transient flux rescaling; `training.flux_rescale`, ADR 0007). Off by
  default, listed in the presets:

  | Field | Default | Meaning |
  |---|---|---|
  | `flux_rescale` | 0.0 | probability that a training row (never Bogus) has its central transient rescaled by `k`; in [0, 1]; needs `paper_input: float32` and `context: none` |
  | `flux_k_min` | 0.5 | smallest `k` (raised per stamp to keep it an alert); `0 < flux_k_min <= flux_k_max` |
  | `flux_k_max` | 1.5 | largest `k` |

  `experiments` rejects a changed one with another `supervised_model` (`ConfigError`).
- `load_settings(path, overrides, defaults=...)`: `defaults` (a subcommand's legacy
  protocol) sits between the dataclass defaults and the file. `read_settings_file` reads a
  YAML mapping without checking keys; `settings_to_plain` gives the plain (YAML/JSON) form
  that `show-config` prints and the metrics JSON records.

## `config_tree` (phase 4, task 4.2; ADR 0017)

`src/astro_simclr/config_tree/`: the nested, typed run configuration that `astro-simclr run`
reads. It lowers to the flat `Settings` (what today's `run_command` executes, and the view
the config hash is defined on, ADR 0015 rule 10), so it changes no numbers.

- **`schema`**: `Config`, a frozen dataclass of eight sections. `run` (`seed`,
  `output_dir`), `runtime` (`engine`, `accelerator`, `devices`, `num_workers`,
  `deterministic`, `fp32_precision`, `progress_bar_refresh_rate`), `data` (`dir`,
  `dataset`, `image_size`, `batch_size`, `with_features`) and `eval` (`batch_size`,
  `limit_batches`) are plain; four are tagged unions on `kind`:

  | Section | `kind` | Fields (flat name when it differs) |
  |---|---|---|
  | `model` | `stamp_classifier` | `drop_rate`, `beta_loss` |
  | | `resnet` | `name` (`supervised_model`: `resnet18/34/50/152`), `beta_loss` |
  | | `paper_stamp_classifier` | `arch` (`paper_arch`), `backbone_weights`, `group_fields`, `context`, `feature_set`, `input` (`paper_input`), `nan_approximation`, `exclude_leaked_train`, `dropout_active`, `drop_rate`, `beta_loss` |
  | | `simclr` | `encoder` (`encoder_name`), `projection_dim`, `temperature`, `legacy_denominator` (`nt_xent_legacy_denominator`) |
  | | `linear_eval` | `pretrained`, `beta_loss` |
  | | `fine_tune` | `pretrained`, `beta_loss`, `drop_rate`, `lr_factor` (`fine_tune_lr_factor`) |
  | `optim` | `Adam`, `AdamW` | `lr`, `betas` (`adam_betas`) |
  | | `SGD` | `lr` |
  | | `LARS` | `lr`, `legacy_trust_ratio` (`lars_legacy_trust_ratio`) |
  | `schedule` | `epochs` | `max_epochs`, `monitor`, `early_stopping`, `patience`, `min_delta`, `check_finite`, `divergence_threshold`, `limit_train_batches` |
  | | `carrasco2021` | `max_steps`, `val_every_n_steps`, `horizon_increment`, `checkpoint_every_n_steps`, `monitor`, `limit_train_batches` (lowers `early_stopping: false`) |
  | `augment` | `transforms` | `name` (`augmentation`) |
  | | `paper` | `rot90` (`rot90_augment`), `noise` (`aug_noise`), `feature_jitter`, `synthetic_bogus`, `min_r2` (`aug_min_r2`), `flux_rescale`, `flux_k_min`, `flux_k_max` |

  - The model decides the command (`Config.command`). Every default is the default of the
    flat field (`Config()` lowers to `Settings()`). `runtime.engine` is `lightning` (the
    default until 4.7/4.8) or `plain` (task 4.6: `astro-simclr run` then trains with
    `tasks.plain.run_plain`, the paper classifier on `carrasco2021` only); it is the run
    id's engine, not hashed.
  - A `Config` is valid by construction: the tags, `paper_stamp_classifier` ⇔
    `carrasco2021` schedule ⇔ `paper` augmentation, then the lowered `Settings`' checks,
    with the flat names in their message mapped to nested paths
    (`eval_batch_size is eval.batch_size`).
  - `flat_view(config)` (every `Settings` field), `Config.to_settings()`,
    `config_sha256(config)` (the flat settings' hash, through
    `semantic_config_from_plain(complete=True)`), `config_to_plain` /
    `config_from_mapping` (plain nested values, unknown keys and types refused with their
    path) and `config_json` (the nested canonical JSON).
  - `config_from_settings(command, settings)` raises a flat run to the nested form (the
    shim). It refuses, naming the keys, when a flat field the chosen variants do not hold
    is not at its default.
- **`presets`**: `PRESETS`, written out as `Config` objects: `carrasco2021`,
  `carrasco2021-no-features` and `pretrain-legacy`, `linear-eval-legacy`,
  `fine-tune-legacy`, `train-supervised-legacy` (`hparams.COMMAND_DEFAULTS`).
  `LEGACY_PRESETS` maps each command to its legacy preset.
- **`resolve`**: `resolve_config(path, overrides, command=None)`. A nested file (section
  keys, each a mapping, and optionally `preset: NAME`) starts from its preset, or from the
  legacy preset of its `model.kind`'s command; then the file; then the dotted `--set
  section.key=value` patch; then the flat `--set key=value` items on the flat view. A
  differing `kind` replaces its section (the tag rule); the dotted items are merged first,
  so their order does not matter. A flat file needs `command` and resolves as
  `<command> --config FILE` does.
- The hash migration test (`tests/test_config_tree_presets.py`): each nested preset has the
  flat preset's flat view and hash; the nested forms of the A0 cases have the pinned
  hashes; every committed flat config goes through the shim with its old hash; every
  `Settings` field has exactly one nested home.
- `configs/runs/` holds nested configs (`carrasco2021.yaml`: the preset on `cuda`).

## `cli` (task 1.1; subcommands from 1.2, 1.7, 3.1, 3.5 and 5.1a; `run` from 4.2)

The `astro-simclr` console script (`[project.scripts]`). `main(argv)` returns the exit code,
so tests call it in-process. Every subcommand registers in `build_parser()` and takes the
shared `--config FILE` and repeatable `--set KEY=VALUE` options through
`add_settings_arguments()` / `settings_from_args()`. An invalid setting prints one
`astro-simclr: error: ...` line and exits with 2.

- `show-config` (1.1): prints the resolved settings as YAML (itself a valid settings file).
- `inspect-data PATH [--json]`, `make-synthetic --out DIR [--n-per-split N] [--seed S]`
  and `prepare-data PKL [--out DIR] [--overwrite]` (1.2; see the `data` section). Data
  errors print one `astro-simclr: error: ...` line and exit with 1.
- `--set` takes one or more `KEY=VALUE` items and may be repeated (`--set max_epochs=1
  limit_train_batches=2`); put it after the positional arguments.
- **`run CONFIG [--set ...] [--command CMD] [--show] [--resume PATH|auto] [--splits FILE
  --fold K | --screen]`** (task 4.2, see `config_tree`): resolve a nested config (or a flat
  one with `--command`), lower it and train it with `run_command` (the Lightning engine).
  `--set` takes `section.key=value` (an unknown path exits 2) and flat `key=value` items.
  `--show` prints `# command:`, `# config hash:` and the resolved config as YAML, and
  does not train. The four training subcommands below are its aliases until 4.8.
- **Training (1.7):** `pretrain`, `linear-eval`, `fine-tune`, `train-supervised` (see
  `experiments`). Each takes `--config`, `--set`, `--hparams FILE` (a legacy
  `results/hparams_best_*.yaml`, see `hparams`), `--data-dir`, `--output-dir`, `--seed`,
  `--accelerator`, `--resume PATH|auto`; `linear-eval` and `fine-tune` also `--pretrained
  PATH`; `train-supervised` also `--preset carrasco2021|carrasco2021-no-features` (task
  2.2c, see `presets`) and `--splits FILE --fold K | --screen` (task 2.5-infra, a split
  run; see `protocol`; `--fold`/`--screen` without `--splits`, or the reverse, exit 2).
  Settings resolve as: `Settings` defaults → the subcommand's legacy
  protocol (`hparams.COMMAND_DEFAULTS`) → `--preset` → `--hparams` → `--config` → `--set` →
  the explicit flags
  (later wins; `cli.training_settings`). The legacy file comes before the config so that a
  config can correct it (R1's `AdamW`). Dry mode: `--set max_epochs=1 limit_train_batches=2
  limit_val_batches=2` (not `fast_dev_run`, which writes no checkpoint). The final line
  prints `best_epoch`, `best_step`, `monitor_best`, `val_acc_best`, `test_acc_at_best` and
  `seconds_per_step`.
- `compare-reference RUN_DIR... --ref R1|R2|R3|R4|CD21 [--oracle-report FILE]` (see
  `reference`): exit 0 for PASS or REPORT, 1 for FAIL.
- `make-splits PREPARED_DIR --out FILE [--seed 0] [--overwrite]` and `compare-variants
  --baseline RUN... --variant RUN... [--split test|val] [--n-boot N] [--seed S]
  [--allow-mixed-machines]` (task 2.3, see `protocol`); `compare-variants` always exits 0
  (a report). With `--splits FILE [--rule superiority|non-inferiority] [--margin F]` (task
  2.5-infra) it decides on split runs: 0 for ADOPT / KEEP, 1 for NOT ADOPTED / DROP;
  `--rule`/`--margin` without `--splits`, `--split` with it, or `--margin` without
  `--rule non-inferiority` exit 2.
- `make-object-splits ALERTS_CSV --out FILE [--seed 0] [--fractions 0.6 0.2 0.2]
  [--labels NAMES] [--drop-conflicts] [--overwrite]` and `bench-report RUN... --splits FILE
  --alerts ALERTS_CSV [--baseline RUN...] [--split test|val] [--reference-counts N...]
  [--n-boot N] [--seed S] [--allow-mixed-machines]` (task 3.2, see `protocol`):
  `bench-report` always exits 0 (a report); runs of another splits file, of other alerts or
  of mixed machines exit 2, an unusable table, splits file or predictions file exits 1.
- `report-phase2 [--replication DIR [--oracle-report FILE]] [--screen DIR --screen-splits
  FILE] [--cv DIR --cv-splits FILE] [--baseline-arm rot4] [--non-inferiority ARM...]
  [--margin F] [--registry [FILE]] [--n-boot N] [--seed S] [--allow-mixed-machines]
  [--out FILE]` (task 2.6, see `protocol`, module `phase2_report`): writes the phase-2
  results report (`docs/results-phase2.md`) as markdown, to stdout or atomically to
  `--out`; always exit 0 (a report). Inconsistent inputs or runs that `compare-reference`
  or `compare-variants --splits` would refuse exit 2, an unreadable predictions or splits
  file exits 1.
- `fetch-m1` and `m1-oracle --data-dir DIR [--out DIR]` (task 2.2b, see `oracle`): exit
  0 if the oracle passed, else 1.
- `fetch-alerts --dataset NAME --split S [--data-dir DIR] [--out DIR] [--rate 1.5]
  [--limit N] [--resume] [--batch-size 100] [--cooldown 300] [--from-snapshot RAW_DIR]
  [--quiet]` and `alerts-summary RAW_DIR [--split S]... [--json]` (task 3.1, see
  `alerts`). `fetch-alerts` exits 0 when the run ends. It turns SIGTERM, SIGHUP and Ctrl-C
  into a stop flag polled between rows (exit 128 + the signal; a second signal forces the
  exit), and restores the previous handlers. A store that exists without `--resume` is a
  data error (1); an out-of-range option is 2. `alerts-summary` exits 0 if every split
  shown meets the acceptance, else 1.
- `fetch-alerts --oids FILE --out DIR [--resume] [--rate] [--batch-size] [--cooldown]
  [--quiet]` (task 3.5, see `alerts.by_oid`): the by-oid mode; `--dataset`, `--split`,
  `--limit` and `--from-snapshot` are refused with it (2). `r5-sources --out DIR [--only
  zenodo|bts] [--keep-archive]` and `build-r5 --sources DIR --cd21-raw RAW_DIR --fetched
  DIR (--out DIR | --dry-run) [--from-snapshot] [--expect BUILD_JSON] [--seed 0]
  [--max-rounds 20] [--rate 1.5] [...]` (task 3.5, see `r5`; handlers in `r5/cli.py`):
  `build-r5` exits 0 when written (and matching `--expect`), 1 when incomplete, refused
  offline, or different from `--expect`; a dry run exits 0 and prints the next round's
  request count. The same signal handling as `fetch-alerts`.
- `ingest-archive TARBALL... --out DIR (--exclude FILE | --no-exclusions) [--night N]
  [--no-oid-year] [--max-ndethist N] [--rows-per-file 4096] [--overwrite] [--quiet]`,
  `archive-summary CORPUS_DIR [--verify] [--json]` and `plan-archive-nights [--seed 0]
  [--labelled-per-year 24] [--later-per-year 8] [--spares 2] [--first D] [--last D]
  [--stamps-per-night N] [--tarball-gb GB] [--download-hours H] [--measured DIR] [--out
  FILE] [--json]` (task 5.1a, see `archive`).
  - `-` as the only TARBALL reads one night from stdin and needs `--night`; `--night`
    without `-`, or `-` beside files, exits 2.
  - `ingest-archive` exits 0 when every night is done. A bad tarball name, an unreadable
    tarball, other rules than `corpus.json`'s, or another file for a done night exit 1.
  - `archive-summary --verify` loads and re-hashes every night: a bad one exits 1.
- `preflight (--synthetic | --data-dir DIR) [--accelerator cuda] [--runs R1,R2,R3,R4]
  [--set ...]` (see `preflight`): exit 0 if no step failed, else 1.
  `scripts/gpu-preflight.sh` is a thin `uv run --no-sync` wrapper.
- `runs reindex [ROOT ...] [--rebuild]`, `runs ls`, `runs show KEY`, `runs compare SPEC`
  (task A2, see `registry`): the local run registry. Exit 0 on success (`reindex` also when
  it skipped directories), 1 when the registry cannot be used or a comparison does not
  resolve.
- **SimCLR pilot (task 5.0b, see `simclr`):** `simclr-pilot DATA --output-dir OUT
  [--splits FILE] [--epochs N | --steps N] [--batch-size B] [--temperature T] [--lr-per-256
  X] [--warmup-epochs E] [--noise-scale S] [--seed N] [--device auto|cpu|cuda|mps]
  [--fp32-precision ieee|tf32] [--deterministic] [--tier device|host] [--resume]
  [--stop-at N]`; `simclr-embed DATA --output-dir OUT (--checkpoint FILE | --random-init)
  [--seed N]`; `make-label-subsets DATA --splits FILE --per-class K [--n-subsets R]
  [--seed S] --out FILE [--overwrite]`; `simclr-probe EMBEDDINGS --subsets FILE --splits
  FILE --out FILE [--weight-decay X...]`. Registered by `simclr.commands.add_simclr_commands`;
  each exits 0 when it ends (an existing run without `--resume`, an existing subsets file
  or a budget larger than a class: 2).
- Exit codes: 0 success; 1 unusable data (`DataError`) or a failed gate/preflight; 2 invalid
  settings or arguments, including an unavailable accelerator (`AcceleratorError`).

## `augmentations` (protocol in 1.1; transforms in 1.5)

`Augmentation`, a runtime-checkable `Protocol` with `view(img)` (one augmented view) and
`pair(img)` (two independent views, the SimCLR positive pair). Input: a `Stamp`, a uint8
array of shape (H, W, 3) (the output of the legacy `img_float2int`). Output: float32
tensors of shape (3, size, size) with values k/255 in [0, 1]. The datasets (1.2) depend
only on this protocol, never on a concrete transform.

- `get_augmentation(name, size)` builds a `PipelineAugmentation` from the registry
  `AUGMENTATIONS`. The registry names say what each pipeline does; the legacy names are
  accepted as aliases (`LEGACY_ALIASES`, resolved by `resolve_augmentation_name`), because
  `results/hparams_best_*.yaml` and the legacy entry points use them. The built object
  carries the registry name.

  | Name | Legacy alias (class) | Pipeline |
  |---|---|---|
  | `simclr` | `simclr` (`SimCLR_augmentation`) | random resized crop, colour jitter, grey, blur |
  | `simclr_centered` | `simclr2` (`SimCLR_augmentation_v2`) | centred crop, flip, jitter, grey, blur, random 90° rotation |
  | `astro_perspective` | `astro7` (`Astro_augmentation_v7`) | rotation, centred crop, perspective, flip, jitter, blur |
  | `astro_perspective_gray` | `astro8` (`Astro_augmentation_v8`) | `astro_perspective` plus random grey |
  | `rotate_flip` | `rotation_v3` (`Rotation_v3`) | rotation, centre crop, flip |
  | `center_crop` | `without_aug` (`Resize_img`) | centre crop only: the validation/test transform |

  Each converts the stamp to a PIL image and runs the legacy torchvision pipeline in the
  legacy order; `pair` calls `view` twice, first view first. No new name equals a legacy
  name of a different transform (for example `rotation` and `astro` are deferred legacy
  variants).
- The 33 other legacy names (`DEFERRED_AUGMENTATIONS`, mostly albumentations/OpenCV
  variants) raise `DeferredAugmentationError`; any other name raises
  `UnknownAugmentationError` listing the valid names (both are `ValueError`s).
- Ported building blocks: `RandomResizedCenterCrop` (draws from Python `random`),
  `GaussianBlur` (sigma from `np.random`, `F.conv2d` with the legacy float32 kernel),
  `RandomRot90` (torch).
- **Randomness:** the transforms use the global torch, numpy and Python generators. Seed
  them with `lightning.seed_everything(seed, workers=True)`; DataLoader workers are seeded
  by Lightning's `pl_worker_init_function`, which the Trainer injects (a DataLoader built
  outside a Trainer must pass it as `worker_init_fn`). Same seed → identical batches, with
  0 or more workers, and a different seed → different batches (tested with 0 workers
  everywhere and with 2 workers where they fork). The 2-worker case forks its workers
  explicitly and is skipped, with its reason, on macOS: a spawned worker re-imports torch,
  torchvision and Lightning (about 5 s of CPU each; 26 s of the suite), and fork is unsafe
  there once Metal has started. The start method does not change what is checked: torch's
  worker loop seeds torch, numpy and `random` from the base seed in every worker under any
  start method, then the init function reseeds them. The Linux CI job runs it.
- **Naming change:** registry names are descriptive; the legacy names keep working as
  aliases, so no config needs to change.
- **Characterization:** `tests/fixtures/augmentations/legacy_views.npz`, from
  `scripts/characterize/augmentations/generate.py` (legacy classes, sizes 27 and 63). The
  port must match it pixel for pixel, except for rare off-by-one levels after the blur's
  float conv (CPU-dependent last bit, then uint8 truncation).
- **Behaviour changes:** `RandomResizedCenterCrop` raises `ValueError` on a reversed
  `scale`/`ratio` (the legacy class only warned); the legacy `np.random.seed(0)` at import
  of `utils/gaussian_blur.py` is gone; `GaussianBlur` no longer builds `nn.Conv2d` modules,
  whose weight init consumed torch random numbers at construction (the transforms' outputs
  for a given generator state are unchanged).

## `data` (task 1.2; float input path in 2.1)

`src/astro_simclr/data/`. File schemas, provenance and the workflow are in
[`docs/data.md`](data.md).

```
pickle ──read_pickle──► validate ──► StampData ──write_prepared──► OUT/<stem>/*.npy + meta.json
 (once, trusted machine: prepare-data)                                        │
                                       StampData ◄──load_prepared (mmap, no pickle)┘
                                           │
                       StampDataset / ContrastiveDataset (+ an Augmentation)
```

- `schema`: the constants (`LABEL_NAMES`, the 26 `FEATURE_NAMES`, `DISCARDED_FEATURES`
  13–15, `SPLITS`, `STAMP_SHAPE`), `SplitData` (uint8 images, int64 labels, float64
  features; `None` for the unlabelled pretraining Train split), `StampData` (kind
  `classifier` or `pretraining`, three splits), and the errors `DataError` ⊃
  `SchemaError` (names file, split and key), `TruncatedFileError`.
- `preprocessing`: the three legacy functions with the same arithmetic (fixtures, `atol`
  1e-6), plus `stamp_to_uint8`, the checked `img_float2int` that the loader uses.
- `pickle_loader`: `read_pickle` is the only unpickling call of the package. It checks the
  final STOP opcode first, so a truncated multi-GB file fails in milliseconds, and its
  unpickler resolves only numpy's array, dtype and scalar globals (numpy 1.x and 2.x
  names). `validate` is lenient about containers (list or ndarray, object feature arrays,
  numpy or Python ints) and strict about content. Each stamp is converted to uint8 in its
  *stored* dtype, one at a time, so peak memory is the unpickled object plus the uint8 copy.
- `prepared`: `prepare` / `write_prepared` write `OUT/<pickle stem>/{Train,Validation,Test}/
  {images,labels,features}.npy` and `meta.json` (source name, size, SHA-256; kind; schema;
  dtype, shape and SHA-256 of every file; class counts). The directory is built as
  `.<stem>.partial` and renamed at the end; `--overwrite` replaces only a directory that is
  itself a prepared layout. No timestamps, so the output is deterministic. `load_prepared`
  uses `np.load(allow_pickle=False, mmap_mode="r")` and checks dtypes, shapes, label range
  and feature finiteness against the schema and `meta.json`; `verify_prepared` re-hashes.
- `datasets` (the only torch module): `StampDataset` returns `(augmentation.view(stamp),
  features, label)`. Features are float32, with the discarded columns removed (23 by
  default). The label is one-hot int64 by default, or an int64 scalar with `one_hot=False`.
  `ContrastiveDataset` returns `augmentation.pair(stamp)`. Each item gets a writable copy
  of its stamp, never a view of the memory map.
  The augmentation normally comes from `get_augmentation(name, size)` (1.5); with
  `center_crop` (legacy `without_aug`) an item equals the legacy `img_float2int` → `ToPILImage` → `Resize_img`
  path, and every registered name is tested with `ContrastiveDataset`.
- `synthetic`: `make_synthetic` writes the four pickles with the real containers and
  dtypes, and a rotation-invariant class pattern inside the central 27×27 crop (a
  nearest-centroid classifier gets ≥ 0.95). Its numpy `Generator` is seeded explicitly,
  because it builds files rather than training randomness. The same seed gives the same
  bytes on the same machine and numpy version (no fixture compares them across machines).
- `inspection`: `summarize` / `format_summary`, behind `inspect-data [--json]`. Inspecting
  is also a full check (schema validation, or `meta.json` plus hashes). The preflight's
  data manifest (1.7) is built from these summaries.

Task 2.1 adds the paper's float input path (`docs/data.md`, "Float input path"):

- `SplitData.images_f32` (float32, `None` by default): the stored stamps without the uint8
  rounding. `load_pickle(path, float32=True)` fills it (values checked to lie in [0, 1]),
  and `prepare(..., images="float32")` / `prepare-data --images float32` writes it as
  `images_f32.npy`. `meta.json` is format version 2 (adds `images` and the optional file);
  `read_meta` accepts 1 and 2, so older layouts load as uint8-only.
- `float_input`: `crop_then_normalise(stamps, crop=21, *, nan_approximation=True)` is
  ALeRCE's crop → per-channel min-max (no epsilon) → NaN → 0, on `(..., H, W, C)` in the
  input dtype; `former_nan_mask` (zeros of channels with more than one exact zero over the
  full stamp), `approximated_stamps` (stamps where a masked zero is inside the crop),
  `crop_split` (a whole float32 split, chunked, for 2.2a's precomputed crops) and
  `crop_bounds` (ALeRCE's crop arithmetic: `(21, 42)` for 21 on 63). Golden fixture:
  `tests/fixtures/data/alerce_crop.npz`, from ALeRCE's own functions
  (`scripts/characterize/data/alerce_crop.py`).
- `float_input`, task 2.5-models (plan row B1'): `context_then_normalise(stamps, factor=3,
  *, nan_approximation=True)` is the whole stamp average-pooled by 3 (63 → 21; block means
  accumulated in float64, ignoring NaNs and, with the approximation, the former NaNs), then
  the crop's per-channel min-max without epsilon and NaN → 0; `context_split` runs it over
  a split. It commutes exactly with the rotations and flips of the stamp (tested). Ours,
  not ALeRCE's; the choice and its NaN handling are in `docs/data.md` and ADR 0005.
- `m1_features`: `M1_FEATURE_NORM_STATS` (M1's 23 means and stds, recorded as constants
  with provenance; `scripts/characterize/data/m1_feature_stats.py` re-derives them with
  `pickletools`), `destandardise`, and `check_destandardisation`, behind
  `check-features DIR [--json]` (exit 1 on a failed test).

Neither module imports torch. The uint8 path (`images.npy`, `StampDataset`) is unchanged.

Task 3.3 adds RJ23's input path on raw cutouts (`docs/data.md`, "RJ23 inputs"; ADR 0009):

- `rj23_input`: `normalise(stamps, *, large_values="paper", chunk_size=1000)` takes
  `(N, 63, 63, 3)` raw floats with NaN/inf and returns `(N, 63, 63, 6)` float32 (the
  |x|-min-p99 normalisation clipped to ±2, then the 3 mask channels); `invalid_mask`,
  `channel_statistics` (`min|x|`, `p99|x|` per stamp and channel, one sort per chunk, equal
  to `np.nanpercentile`). `large_values="code"` is RJ23's code behaviour for `|x| > 1e10`.
- The views: `full_view`, `cropped_view` (24:40), `lowres_view` (one 4×4 TF-'same'
  pooling), `multiscale_view` (8×8×24, scale-major 1″/2″/4″/8″), and `rj23_views`, which
  runs the normalisation and the requested views chunk by chunk into preallocated outputs.
  `tf_same_avg_pool` is TF's `AveragePooling2D(k, padding='same')` (valid-pixel means,
  float64 sums); `crop_offset` is RJ23's `⌈(n − size) / 2⌉`.
- `unit_sphere(ra, dec)` (degrees → `(N, 3)` float32) and `channels_first`.
- numpy only (a test checks that importing it does not import torch); not wired into a
  data module yet: that waits for task 3.1b's `raw-alerts` kind.

Task 3.1b adds the `raw-alerts` kind (`docs/data.md`, "Raw-alert layout"; ADR 0012):

```
fetcher (3.1): cutouts + per-row table [+ features] ──write_raw_alerts──► DIR/<name>/*.npy + meta.json
                                                                                  │
                                  RawAlerts ◄──load_raw_alerts (mmap, no pickle)───┘
   (cutouts, index, label_names, features; n_classes)
       ├──► protocol.alerts.alert_table_from_raw ──► object splits, benchmark report (3.2)
       └──► rows_of(split candids) ──► 3.3 input path ──► N-class model
```

- `raw_alerts`: `INDEX_DTYPE` (the per-row structured dtype), `make_index` (columns → the
  table, with kind and width checks, never a silent cast or truncation), `write_raw_alerts`
  (the builder; atomic and deterministic like `write_prepared`; optional `features`, the 23
  `CD21_FEATURES` not standardised), `load_raw_alerts` → `RawAlerts` (cutouts and features
  memory-mapped, the index read and checked: dtypes, shapes, label range, unique `candid`,
  field ranges, counts; `rows_of(candids)`), `verify_raw_alerts` (SHA-256),
  `label_names_of` (the label names from `meta.json` alone) and `check_label_names`. The
  `_*_problem` functions hold the rules, and both the writer and the loader call them, so
  the writer never writes what the loader refuses. No split and no mask is stored: the
  splits are task 3.2's (by `oid` group, ADR 0008), the mask a function of the cutouts.
- `protocol.alerts.alert_table_from_raw(data, group=None)`: task 3.2's `AlertTable` from a
  loaded layout. The index's `oid` character set, `candid` and `jd` rules are the
  table's, so every stored row is accepted (tested).
- Task 3.3's `rj23_input.rj23_views(data.cutouts[rows])` and
  `unit_sphere(data.index["ra"][rows], data.index["dec"][rows])` read a loaded layout as
  it is (tested); a data module that feeds them to the RJ23 models is task 3.4's.
- `prepared`: `read_meta_file` reads `meta.json` of either kind (format check only),
  `layout_kind` returns its kind without raising, and `read_meta` / `load_prepared` refuse
  a `raw-alerts` layout with a message naming `load_raw_alerts`. Versions 1 and 2 are
  unchanged, and the writer still writes version 2.
- `inspection`: `summarize` dispatches on the kind; a `raw-alerts` summary has the row
  count, both files, the counts per class (under the layout's own names) and per
  `label_source`. `preflight.prepared_dirs` leaves `raw-alerts` layouts out (its steps and
  manifest read splits).
- `synthetic.synthetic_raw_alerts(label_names, n_per_class, seed)`: raw-count cutouts with
  NaN corners and inf pixels and a valid per-row table, for tests and for the fetcher's.

The CLI imports these modules inside its handlers, so `show-config` does not import them;
`data/__init__.py` does not re-export the torch datasets.

Behaviour changes against the legacy code (no legacy flag: none of them changes the
values loaded from a valid file):

- The datasets no longer unpickle in `__init__` or convert per item. The conversion happens
  once, at load time, with the same `img_float2int` arithmetic, in the stored dtype. This
  is verified bit-exact on all 61,872 stamps of the three complete real files.
- The loader rejects what the legacy code silently accepted: stamps outside [0, 1] or with
  NaN/inf (`np.uint8` of such values is undefined), non-63×63×3 stamps, non-integer or
  out-of-range labels, non-finite features, missing or unexpected keys, and pickles that
  reference non-numpy globals.
- The legacy `Dataset_simclr` crashed without an augmentation (`self.transformation` was
  never set). `ContrastiveDataset` always requires one.
- `repair_non_squared_stamp` raises a clear `ValueError` for stamps larger than 63 or not
  (H, W, 3). The legacy code raised numpy's "negative dimensions" error or a concatenate
  error.

## `losses` (task 1.3)

`src/astro_simclr/losses.py`, ported from the legacy `losses/losses.py`.

- `NT_Xent(temperature, *, legacy_denominator=True)`: the SimCLR loss on a batch of B
  positive pairs `(z_i, z_j)`, each (B, D), B ≥ 1. Returns `(loss, pos, neg)` as the legacy
  code did: the mean loss over the 2B views, and (without gradient) the mean cosine
  similarity of the positive and of the negative pairs (`neg` is NaN at B = 1, where there
  are no negatives). The default denominator is the legacy `1 + Σ_neg exp(s_neg)`, close to
  DCL; `legacy_denominator=False` gives the paper's `exp(s_pos) + Σ_neg exp(s_neg)`
  ([ADR 0002](decisions/0002-nt-xent-denominator.md)). The training modules read the flag
  from the setting `nt_xent_legacy_denominator` (default `true`). No all-gather: one device.
- `P_stamps_loss(beta)`: `CE(logits, argmax(targets)) − β·H(softmax(logits))`, both batch
  means; `targets` are one-hot (B, C). The entropy's gradient flows through both factors,
  as in the legacy code.
- Both raise `ValueError` on mismatched shapes; `NT_Xent` also on `temperature ≤ 0`.
- Characterization: `tests/fixtures/losses/{nt_xent,p_stamps_loss}.npz` (loss, `pos`,
  `neg` and input gradients of the legacy classes, B ∈ {1, 2, 7, 100, 150}, T ∈ {0.1791,
  0.5}), generated by `scripts/characterize/losses/generate.py`; matched at `atol=1e-6`.

Behaviour changes against the legacy code:

- **Mask from the actual batch.** The legacy class built negative masks only for its
  constructor `batch_size` and for a hard-coded 100, so any other batch size failed (for
  example a last batch with `drop_last=False`). The mask is now built per call, and
  `NT_Xent` no longer takes `batch_size`. No legacy flag: every batch the legacy code
  accepted gives the same result.
- **Similarity by one matrix product** (with the `nn.CosineSimilarity` formula and `eps`)
  instead of the broadcast `(N, N, D)` tensor; equal to float32 rounding, much less memory.
- **`P_stamps_loss` takes no `batch_size`** (the legacy class stored it and never used it).
- **Input checks:** mismatched shapes and a non-positive temperature raise `ValueError`
  instead of failing inside torch (or, for the temperature, producing inf/NaN).

## `models` (task 1.4)

`src/astro_simclr/models.py`, ported from the legacy `models/models.py`. Every class keeps
its legacy attribute names, so `state_dict` keys are identical and legacy weights load with
`strict=True`; modules are also created in the legacy order, so a given seed gives the
legacy initial weights.

| Class | Legacy name | Notes |
|---|---|---|
| `StampEncoder(k)` | `Stamp_encoder` (`stamp<k>`) | widths `int(32k)`/`2·int(32k)`, 5×5 kernels, `out_features = int(128k)`, mean of the 4 rotations after ReLU |
| `PStampsEncoder(k=1.0)` | `P_stamps_net_SimCLR_v2` from `b357b43^` (`pstamps`) | 3×3 kernels, `fc1` without bias, rotation mean through `bn1` |
| `StampClassifier(drop_rate, with_features, n_classes=5)` | `Stamp_classifier` | Carrasco-Davis et al. 2021; 23 features through `bn1`, `n_classes` logits |
| `SimCLRNet(encoder, input_size, projection_size)` | `CLR` | 3-layer projector on `h / ‖h‖`; returns `(h_i, h_j, z_i, z_j)` |
| `LinearClassifier`, `ThreeLayerClassifier` | `Linear_classifier`, `Three_layers_classifier` | heads; `LinearClassifier` keeps the `model.` key prefix |
| `Identity` | `Identity` | alias of `torch.nn.Identity` |

**`pstamps` provenance.** `P_stamps_net_SimCLR_v2` was removed in `b357b43`; the port comes
from its parent (`c5c7e6f`). It is the encoder of the only recorded 300k-alert SimCLR run
(`31b10c9:iter_clr_stamps_300.py`, `encoder_name: pstamps`), the settings of reference run R3.

**`build_encoder(name) -> Encoder(module, out_features)`** replaces the `if/elif` chains of
the legacy `SimCLR`. Names: `stamp<k>` (any positive `k`, e.g. `stamp1`, `stamp0.5`),
`pstamps`, `resnet18`, `resnet34`, `resnet50`, `resnet152` (torchvision, no pretrained
weights, `fc` replaced by `Identity`). `out_features` is what the encoder declares: the
stamp encoders' `out_features`, the ResNets' `fc.in_features`.

**N classes (task 3.1b).** `N_CLASSES = 5` (CD21's AGN, SN, VS, Asteroid, Bogus) is now the
*default* class count, not a constant of the networks. Every classifier takes
`n_classes`, checked by `check_n_classes` (an integer ≥ 2), and only its last layer
depends on it:

- the networks: `StampClassifier`, `paper_model.PaperStampClassifier`,
  `paper_variants.PaperHead` / `GroupStampClassifier` / `build_paper_model`;
- the Lightning modules: `ClassifierModule` (it sizes the metrics and the confusion
  matrices), `SupervisedCEModule`, `PaperClassifierModule`, `SupervisedCEResNetModule`,
  `LinearEvalModule`, `FineTuneModule`;
- the data: `device_data.TensorSplit.n_classes` (the one-hot width and the label range;
  `DeviceTensorDataModule` refuses splits that disagree), `protocol.predictions`
  (`Predictions` takes any K ≥ 2 columns, `load_predictions(path, n_classes=5)` checks the
  count, `check_paired` refuses runs with different counts).

An N-class run reads the count from its dataset, `len(data.raw_alerts.label_names_of(path))`,
and passes it down. **The phase-2 path is byte-identical:** with the default, the layers
are built in the same order and draw the same weights, and the modules record `n_classes`
in their hyperparameters only when it is not 5, so `hparams.yaml` and checkpoints are what
they were (`tests/test_n_classes.py` pins the seeded weights, the hyperparameter names and
the bytes of a predictions file, computed before the parameter existed). CD21-specific code
stays 5-class: `schema.LABEL_NAMES`, the pickle loader and the split layouts, the protocol's
splits and leakage, the M1 oracle, the CLI's metrics, and C6 (`synthetic_bogus` needs
`n_classes=5`, it draws from CD21's Bogus and Asteroid). The training CLI does not build
N-class runs yet: RJ23's input path and models are tasks 3.3 and 3.4.

**Input size.** The stamp networks accept only (N, 3, 27, 27): `fc1` is sized for that
crop. Anything else raises `ValueError` naming the expected shape. The ResNets accept 27
and 63 (any size ≥ 27).

**Characterization.** `scripts/characterize/models/generate.py` runs the legacy classes
(read with `git show` at `b357b43` and `c5c7e6f`) in an isolated torch 2.14.1 environment
and writes `tests/fixtures/models/*.npz`: parameter count, a fingerprint of the seeded
initialisation, inputs and eval-mode outputs with the deterministic weights of
`tests/model_weights.py` (independent of the RNG). `tests/test_models.py` checks them with
`rtol=1e-4, atol=1e-5`. The seeded-init fingerprint is checked only for the stamp networks
and heads, whose default init uses `uniform_` and is portable across CPUs. The ResNets'
`kaiming_normal_` goes through torch's vectorised CPU `normal_`, which differs between x86
(AVX) and arm64 (NEON) for the same seed, so their fixtures are checked by parameter count
and fixed-weight forward pass only.

**Behaviour changes against the legacy code** (none changes a forward pass or a parameter
count):

- The factory reads `out_features` instead of `encoder.bn1.num_features`, so every
  `stamp<k>` encoder works (legacy: `AttributeError`), and `pstamps` is matched before
  `stamp<k>` (legacy: `'stamp' in name` sent it to `float('s')`). Unknown names raise
  `ValueError` listing the valid ones (legacy: `AttributeError` on an unset encoder).
  `k` too small for one channel raises `ValueError`.
- `resnet34` is new (the legacy chain had 18, 50 and 152).
- Stamp networks reject non-27×27 input with `ValueError` (legacy: a matmul shape error
  in `fc1`).
- `StampClassifier.forward(x_img, x_feat=None)`: `x_feat` is optional without features
  (legacy: required and ignored) and its shape is checked with features.
- `SimCLRNet.forward(x_i, x_j)` drops the unused `*x_feat` of `CLR.forward`.
- Dropped: the dead `kernel == 3/7` branches and the public `conv_output_shape` /
  `maxpool_output_shape` helpers (now private).

## `paper_model` (task 2.2a)

`src/astro_simclr/paper_model.py`: `PaperStampClassifier(with_features=True,
dropout_active=False, drop_rate=0.5)`, ALeRCE's `DeepHitsWithFeatures` (without features,
`DeepHitsExtraLayer`) of Carrasco-Davis et al. 2021, re-implemented from
`alercebroker/pipeline@a3705ac` (no code copied). A new class: the phase-1
`StampClassifier` (the legacy 2022 network) is unchanged.

- Input (N, 3, 21, 21) crops and (N, 23) features; output (N, 5) logits. The 4 rotations
  (`torch.rot90`, counter-clockwise as `tf.image.rot90`; no flips) go through the trunk as
  one batch of 4N: `ZeroPad2d(3)`, `conv1` 4×4 valid (32), `conv2` 5×5 same (32), max-pool,
  `conv3`-`conv5` 5×5 same (64), max-pool, flatten (2304, in (c, h, w) order), `fc1` (64),
  ReLU, `drop1`; then the mean over the rotations; `bn_features` and the concatenation
  (87); `fc2`, ReLU, `drop2`; `fc3`, ReLU, `drop3`; `logits`.
- **441,075** trainable parameters with features, **439,557** without (M1's counts;
  tested).
- **Dropout inactive by default.** The layers exist, but `dropout_active=False` never applies
  them (M1's `keras_learning_phase` was never fed); `True` applies `nn.Dropout(drop_rate)` in
  train mode (diagnostic 1 of 2.4).
- **Frozen feature BN:** `BatchNorm1d(23, eps=1e-3, momentum=0.0)`: batch statistics in
  training, running buffers fixed at (0, 1), so evaluation is
  `gamma * x / sqrt(1 + 1e-3) + beta` (tested after training steps).
- **Initialisation (code is the authority):** Glorot-uniform kernels; zero biases on the
  convolutions and `logits`; **Glorot-uniform biases on `fc1`-`fc3`**, U(±0.2165).
  ALeRCE's `layers.dense` passes `bias_initializer=None` to `tf.keras.layers.Dense`, and
  Keras' `add_weight` (TF 1.14 and 1.15) then falls back to `glorot_uniform`, whose fans
  for a shape `(64,)` are 64 and 64. M1's meta graph confirms it:
  `dense_{1_10,2_12,3_13}/bias/Initializer/random_uniform` with max 0.21650635, while
  `conv_*/bias` and `output_logits/bias` use `Initializer/zeros`. The plan's "zero biases"
  holds only for those. BN gamma 1, beta 0.
- Layer names map to M1's variable scopes (table in the module docstring) for the 2.2b
  weight mapping: HWIO → OIHW, dense kernels transposed, `fc1` rows permuted from TF's
  (h, w, c) flatten to (c, h, w).
- Task 2.5-models adds two keyword arguments whose defaults keep the faithful model (same
  layers, same draws from the global generator, same outputs; tested against
  `tests/fixtures/paper_model/faithful.npz`, generated from the commit before them by
  `scripts/characterize/paper_model/generate.py`): `in_channels=3` (6 with the 63×63
  context; only `conv1` widens) and `rotation_mean=True` (`False`: the `single` arch, one
  pass, no mean). `check_paper_images` is the shared input check.
- The `feature_set` diagnostic (ADR 0016; `feature_sets`) adds `feature_set="all"`: the
  head reads the 23 features, only the 6 sky coordinates (`position`: `bn_features` and
  fc2's feature inputs shrink to 6, **439,953** parameters) or none (`none`: exactly the
  `with_features=False` model). `forward` still takes (N, 23) features and selects the
  columns itself with `index_select` through a non-persistent `feature_index` buffer
  (`None` for all 23 or none: the faithful forward pass does no indexing, and the
  `state_dict` keeps M1's keys). `feature_index(columns)` builds it.

## `feature_sets` (task 2.5 diagnostic; ADR 0016)

`src/astro_simclr/feature_sets.py`: the columns of the 23 kept features (`KEPT_FEATURES`,
the stored 26 without `rb`, `distnr`, `magnr`; equal to `class_augment.KEPT_FEATURE_NAMES`,
tested) that the paper model's head reads, per `Settings.feature_set`.

| `feature_set` | Columns | Kept indices |
|---|---|---|
| `all` | the 23 (faithful) | 0-22 |
| `position` | `POSITION_FEATURES`: `ra`, `dec`, `ecl_lat`, `ecl_long`, `gal_lat`, `gal_long` | 10, 11, 16, 17, 18, 19 |
| `none` | none (images only; the `with_features: false` model) | — |

- `feature_columns(feature_set, *, with_features=True)` and `feature_names(...)`: pure
  functions, no randomness, no engine import (ADR 0014). `with_features=False` reads
  nothing; with `position` it is a `ValueError`, as an unknown set is.
- The selection is in the model, not in the data: batches keep the 23 columns, so the split
  runs, C5's jitter and C7's `magpsf` / `sigmapsf` update are unchanged (ADR 0016,
  decision 2). A run: `train-supervised --preset carrasco2021 --set feature_set=position`.
- **Tested** (`tests/test_feature_sets.py`): the columns by name against the schema; `none`
  equal to `with_features: false` for every `paper_arch` (initial tensors, logits) and
  after a CPU training run (checkpoint tensors and prediction-file bytes); `position`
  ignores the other 17 columns and reloads from its checkpoint; the settings checks, the
  preset deviation and the CD21 grouping. `all` reproduces the golden fixture
  (`tests/test_paper_variants.py`), and the split runs take the flag with the 2.5 flags
  (`tests/test_split_runs.py`).

## `group_conv` (task 2.5-models)

`src/astro_simclr/group_conv.py`: hand-written p4 (C4) and p4m (D4) group convolutions
(Cohen & Welling 2016), no new dependency (escnn needs a Fortran compiler; the portability
rule). Each layer expands its parameters into a plain `conv2d` kernel on every call and
runs one `F.conv2d`, so it works on CPU, CUDA and MPS, and costs what a plain CNN with
`fields × |G|` channels costs.

- `Group("p4" | "p4m")`: elements `(m, r)` acting as `rot90(flip^m(x), r)`, the Cayley table
  `mul` (derived from the transforms) and `inv`. Regular maps have channels in
  `(field, g)` order.
- `LiftingConv(group, in_channels, out_fields, k, padding)`: image → regular map;
  `GroupConv(group, in_fields, out_fields, k, padding)`: regular → regular, with the input
  orientations permuted by `h → g⁻¹h` for output `g`; `GroupReadout(group,
  in_fields, out_fields, k)`: a valid `GroupConv` covering the whole k×k map (the paper's
  fc1); `orientation_mean(x, order)`: the invariant mean over the orientations of each
  field. Biases: one per field.
- **Expansion = one gather** (ADR 0005, decision 8): each layer computes once, from the
  transforms applied to an `arange` of its weight's shape, the flat source index of every
  expanded entry and keeps it as a non-persistent buffer (it follows `.to()`, stays out of
  `state_dict`). `expanded()` is one `index_select` of the flattened weight through
  `_KernelGather`, whose backward gathers each entry's `|G|` copies with an inverse index
  and adds them from the last output orientation to the first. Not `index_select`'s own
  backward: that is an `index_add_` scatter, with no deterministic MPS kernel, and it sums
  in another order. Values and gradients are bit-equal to the transform-by-transform
  expansion it replaced (~200 `select`/`rot90`/`stack` ops per p4 training step).
- **Glorot fans** (the initialisation of the faithful model, defined for group layers):
  those of the expanded linear map. Convolutions use Keras' convention on the expanded
  kernel, `fan_in = in_channels·k²`, `fan_out = out_fields·|G|·k²` (= `xavier_uniform_`'s
  fans of the expanded kernel; for p4 8/16 the expanded kernels have the paper's shapes, so
  the same bounds). The readout is a dense layer (1×1 output): `fan_in = in_fields·|G|·k²`,
  `fan_out = out_fields·|G|` (2304 and 256 for B7, against the paper fc1's 2304 and 64).
- **Geometry:** exact equivariance needs every op to map its grid onto itself under the
  rotations about the centre: valid convolutions, odd "same" convolutions with symmetric
  padding, 2×2 pooling of even sides. The paper's 27 → 24 → 12 → 6 does.
- **Tested** (`tests/test_group_conv.py`): the group axioms and the table against the
  transforms; the equivariance law `layer(T_g x) = T_g(layer(x))` with the orientation
  permutation, for every element, kernel parity and padding, in float64 with random
  biases; the readout's invariance after the orientation mean; the fans and bounds; the
  gathered kernel and its gradient bit-equal to the transform expansion (kept in the test
  as the reference) for every group layer of the p4 and p4m paper models and a few odd
  shapes, on CPU and on MPS / CUDA when present, also under
  `torch.use_deterministic_algorithms(True)`; the buffers out of `state_dict`.

## `paper_variants` (task 2.5-models)

`src/astro_simclr/paper_variants.py`: the plan's rows B7, B2' and B1' and their control arms
as flags of `paper_stamp_classifier` (`Settings.paper_arch`, `group_fields`, `context`,
`rot90_augment`; ADR 0005).

| `paper_arch` | Trunk | Parameters (features) | MACs per stamp |
|---|---|---|---|
| `rot4` | the paper's, 4 rotations, mean after fc1 (faithful) | 441,075 | 210,577,152 |
| `single` | the paper's trunk, one pass (not invariant) | 441,075 | 52,651,776 |
| `p4` | p4 G-CNN, 8/16 fields (B7) | 228,531 | 53,094,144 |
| `p4`, `group_fields: 16` | p4 G-CNN, 16/32 fields (capacity control) | 587,635 | 209,397,504 |
| `p4m` | p4m G-CNN, 4/8 fields (B2') | 193,107 | 53,683,968 |
| `p4`, `context: avgpool63` | B7 with the 63×63 context (B1') | 228,915 | 53,978,880 (+1.7 %) |
| `convnext_atto`, `image_size: 63` | timm ConvNeXt-atto on the whole stamp, 4 rotations (B8; `convnext_model`) | 3,405,227 | 714,557,184 |
| `convnext_pico`, `image_size: 63` | timm ConvNeXt-pico on the whole stamp, 4 rotations (B8) | 8,575,667 | 1,785,014,016 |

- `GroupStampClassifier(group, *, group_fields=None, with_features, dropout_active,
  drop_rate, in_channels)`: the paper's geometry with group layers: `ZeroPad2d(3)`,
  `conv1` lifting 4×4 valid (24), `conv2` 5×5 same, max-pool (12), `conv3`-`conv5` 5×5
  same, max-pool (6), `fc1` a 6×6 `GroupReadout` to 64 fields, ReLU, `drop1` per
  orientation, the orientation mean (64), then `PaperHead`. `group_fields` (default
  `32 // |G|`) are conv1-conv2's fields; conv3-conv5 have twice as many. Initialisation:
  `group_conv`'s Glorot kernels, zero conv biases, the readout's field biases
  U(±sqrt(3/64)) like the paper's fc1, then the head.
- `PaperHead`: the paper model's head (frozen feature BN, fc2, fc3, logits, their dropout
  layers and initialisation); tested equal to `PaperStampClassifier`'s head with the same
  weights.
- `build_paper_model(arch, *, group_fields, context, with_features, dropout_active,
  drop_rate, n_classes, feature_set)`: the factory `PaperClassifierModule` uses; the defaults build the faithful
  model. `paper_in_channels(context)`, `default_group_fields(arch)`, `paper_input_side(arch)`
  (21, or 63 for the ConvNeXts: what `image_size` must be).
- `count_parameters(model)`; `count_macs(model, *, in_channels)`: MACs per stamp of an
  eval forward, from `torch.utils.flop_counter` (convolutions and matrix products, not the
  kernel expansion, pooling or BN), tested against the analytic counts. The input side is
  the model's `input_side` (63 for the ConvNeXts) or 21; a 4-rotation model counts its 4
  passes (divide by 4 for a view).
- `VARIANTS`: name → `Variant(row, overrides, description)`: `rot4` (A), `single-aug`
  (`single` + `rot90_augment`), `p4` (B7), `p4-16`, `p4m` (B2'), `p4-context` (B1'),
  `convnext-atto`, `convnext-pico` and `convnext-pico-zoobot` (B8). A run
  is the `carrasco2021` preset plus `--set key=value` for each override, for example
  `train-supervised --preset carrasco2021 --set paper_arch=p4 --set context=avgpool63`.
  Split runs (`--splits FILE --fold K | --screen`, 2.5-infra) take the same flags: one crop
  reader in `experiments._build_paper` serves both paths, and `device_data.protocol_splits`
  gathers the 6-channel images by pool index like the 3-channel ones (tested in
  `tests/test_split_runs.py`).
- `scripts/bench_paper_models.py`: parameters, MACs, CPU latency (batch 1) and
  throughput (batch 512, default threads and 1 thread), a batch-64 training step (forward
  and backward of the summed logits, train mode, no optimiser step) and, on the
  accelerator torch reports, batch 1 / 512 / 4096 and the training step, for every variant
  (`--out FILE` writes JSON). The group layers re-expand their kernels on every call,
  which shows in the batch-1 latency and the training step more than in the throughput; an
  exported, constant-folded model does not pay it.
- **Tested** (`tests/test_paper_variants.py`): once per new flag, the faithful preset with
  that flag at its default reproduces the golden fixture (parameter count, every initial
  tensor's sum, float64 logits); every invariant configuration (`rot4`, `p4`, `p4` 16/32,
  `p4m`, with and without context and features) has max |Δlogit| < 1e-5 under the
  rotations by 90° in float32 (< 1e-10 in float64), and p4m also under the flips, with
  random biases, on the whole model; `single` is not invariant and p4 not flip-invariant
  (the check can fail); invariance through the input path (rotate the 63×63 stamp, then
  crop and pool); the counts; the settings checks; CPU (and MPS/CUDA when present) training
  smoke runs through `run_command`.

## `convnext_model` (task 2.5, plan row B8; ADR 0022)

`src/astro_simclr/convnext_model.py`: ConvNeXt-atto and ConvNeXt-pico (timm) as
`paper_arch` values of `paper_stamp_classifier`, on the whole 63×63 stamp.

- `ConvNeXtStampClassifier(arch, *, with_features, dropout_active, drop_rate, in_channels,
  n_classes, feature_set)`: (N, 3, 63, 63) stamps → the 4 rotations as one batch → zero
  padding to 64 (bottom, right) → the timm backbone (`num_classes=0`; its stem a 4×4
  convolution at stride 2, padding 1; stage maps 32-16-8-4, `STAGE_SIDES`) → pooled,
  normalised features (320 / 512) → `fc1` to 64, ReLU, dropout per rotation → the rotation
  mean → `PaperHead`. Exactly invariant to 90° rotations up to rounding. timm's
  initialisation for the backbone (the replaced stem drawn the same way), the paper's for
  fc1 and the head, all from the seeded global generator. `stage_sides(x)` and
  `representation(x)` are diagnostics. timm is imported only here, when a model is built.
- | arch | Parameters | Backbone | MACs per view | MACs per stamp |
  |---|---|---|---|---|
  | `convnext_atto` | 3,405,227 | 3,374,520 | 178.6 M | 714.6 M |
  | `convnext_pico` | 8,575,667 | 8,532,672 | 446.3 M | 1,785.0 M |

  (with the 23 features; the paper model is 441,075 and 52.7 M per view.)
- **Why this stem:** timm's `patch_size=2` alone is compensated by a stride-2 downsample in
  stage 0 (maps 16-8-4-2 again); a 2×2 stem would not fit the Zoobot 4×4 stem kernel. The
  constructor refuses a timm whose stage 0 downsamples.
- **Pretrained weights:** `BackboneWeights` (repo, commit, file, size, SHA-256, licence);
  `ZOOBOT_CONVNEXT_PICO` (Apache-2.0 per its model card; Walmsley et al. 2024,
  arXiv:2404.02973); `BACKBONE_WEIGHTS` (`Settings.backbone_weights` → arch → weights);
  `backbone_weights_for(name, arch)`; `fetch_backbone_weights(weights, *, download)` (the
  Hugging Face Hub client by default, cached under `HF_HOME`; refuses an entry without a
  recorded licence before any download, and a file of another size or SHA-256);
  `load_backbone_weights(model, weights, path)` (`torch.load(weights_only=True)`, strict
  keys and shapes, backbone only). `WeightsError` is a `DataError` (one CLI error line).
  `experiments._load_backbone_weights` calls them after the seeded build and records the
  pin in `metrics.json["paper_data"]["backbone_weights"]`.
- The data path at 63: `device_data.float_crops(..., side=63)` (and `interim_uint8_crops`):
  `crop_split(images, 63)`, the paper's normalisation on the whole stamp; about 2.4 GB of
  device-resident splits for `red` (9 times the crops). The 21×21-crop flags (`context`,
  `aug_noise`, `synthetic_bogus`, `flux_rescale`) are refused; `rot90_augment` and
  `aug_feature_jitter` work.
- **Tested** (`tests/test_convnext_model.py`): stage maps, the timm trap, shapes, float64
  invariance with every parameter perturbed (and a flip visible), counts, `feature_set`,
  the seeded initialisation, input and layout checks, the weights loader with a fake
  downloader (licence gate, size, SHA-256, keys, shapes, channels), from scratch with sockets
  disabled, the settings, the run wiring; a CPU split run of `convnext_atto` in
  `tests/test_split_runs.py`; the golden fixture with `backbone_weights: none`.

## `rj23_model` (task 3.4)

`src/astro_simclr/rj23_model.py`: the stamp classifiers of Reyes-Jainaga et al. 2023 (RJ23)
for the four scenarios of `data.rj23_input` (full 63², cropped 16², low-resolution 16²,
multi-scale 8×8×24), rebuilt from the paper's Table 2 and Appendix A because RJ23's
`models.py` is not public. ADR 0013 pins every detail the paper leaves open, with its
source. **Engine-free** (the phase-4 rule): plain `nn.Module` builders that import torch
and `rj23_input`'s constants only, no Lightning (a test checks it); task 3.4b wires them
into training.

- `RJ23_TABLE2`: scenario → `RJ23Architecture(scenario, layers, dense, dropout,
  learning_rate, first_kernel_size=3)`, Table 2 as published (cited in code); `layers`
  lists the convolution widths and `MAXPOOL` markers in order. `input_shape` is the
  channels-first view shape, `representation_size` `8·8·c_L`.

| Scenario | Trunk (3×3 'same' convs, 2×2 TF-'same' max-pools) | Dropout | FC | Parameters (K = 5) |
|---|---|---|---|---|
| `full` | 37, pool, 60, pool, 29, pool, 26 (63 → 32 → 16 → 8) | 0.71726 | 31 | 99,778 |
| `cropped` | 112, 27, pool, 92, 121 (16 → 8) | 0.866799 | 133 | 1,202,808 |
| `lowres` | 63, 31, pool, 108, 14 (16 → 8) | 0.73065 | 22 | 86,648 |
| `multiscale` | 76, 18, 54, 28 (8) | 0.8478 | 69 | 179,124 |

- `RJ23Classifier(architecture, *, n_classes=5, bn_position="after_concat")` and
  `build_rj23_classifier(scenario, ...)`: `forward(x_img, x_pos)` takes the view
  `(N, C, H, W)` (`rj23_input.channels_first`) and the unit-sphere position `(N, 3)`
  (`rj23_input.unit_sphere`) and returns `(N, K)` logits. `dihedral_views` stacks the 8
  views (4 counter-clockwise rotations, then the up-down flip of each) as one batch of
  8N through `trunk` (ReLU after every convolution); `representation` flattens and
  averages over the views (`(N, 64·c_L)`, exactly D4-invariant on the view); `head`:
  `dropout`, concat(x, y, z), `bn`, `fc`, ReLU, `logits`. `bn_position="before_concat"`
  is the pre-registered sensitivity arm (BN on the representation only).
- TF-default parity: `TFSameMaxPool2d(kernel_size, stride)` (TF 'same' padding from
  `tf_same_padding`, filled with −∞); `KerasBatchNorm1d` (momentum 0.99 as torch's 0.01,
  eps 1e-3, moving variance from the biased batch variance as Keras' non-fused path);
  `init_keras_defaults` (Glorot-uniform kernels, zero biases, BN gamma 1 and beta 0,
  drawn from the global generator after the constructor's own default draws). The Adam
  epsilon 1e-7 and Table 2's learning rates belong to the training regime (3.4b).
- `rj23_parameter_count(architecture, *, n_classes, bn_position)`: the formula of the
  module docstring, `Σ (k_l² c_{l−1} + 1) c_l + 2(D + 3) + (D + 4) F + (F + 1) K`.
- **Tested** (`tests/test_rj23_model.py`): Table 2's values; output and trunk shapes per
  scenario on `rj23_views` outputs (5 and 6 classes); the parameter counts, hand-computed
  and by formula, for both BN positions; golden tests for TF 'same' padding and max-pooling
  (negative inputs, padding before, the gradient), Keras BN (one hand-computed step,
  against torch's unbiased update, output and gradient equal to torch's) and the Glorot
  bounds; the representation's invariance under all 8 D4 elements on the view (float64,
  random biases), on the raw stamp for the full scenario (all 8) and every scenario (the
  transpose), the cropped scenario seeing the mirrored 23:39 window of a rotated stamp
  (ADR 0009) and the pooled scenarios not invariant to a raw 90° rotation; the head order
  and the arm; seeded determinism of builds, losses and gradients under
  `use_deterministic_algorithms`; eval logits and a training step on MPS and CUDA where
  present against the CPU.

## `oracle` (task 2.2b)

`src/astro_simclr/oracle/`: the published model M1 of Carrasco-Davis et al. 2021 as a
no-GPU check that our input path and `PaperStampClassifier` reproduce it. No TensorFlow,
and no ALeRCE code or weights in the repository (the pipeline repository has no LICENSE).

- **`tf_files`**: pure-Python readers of the two TF1 formats M1 uses. `read_bundle(index,
  data, prefix=...)` reads a tensor-bundle V2 checkpoint (`model.index`, a LevelDB-style
  table, plus its single data shard) into float32 arrays in TF's layout; it checks the
  table magic, that blocks are uncompressed, one little-endian shard, `DT_FLOAT`, `size ==
  4 * prod(shape)` and every block's and tensor's masked CRC32C, and raises `TFFormatError`
  otherwise. `read_scalar_events(buf)` reads the `simple_value` scalars of a TF event file
  (TFRecord, both CRCs checked), by tag, with their steps.
- **`m1`**: the pinned files (`M1_COMMIT` = `alercebroker/pipeline@a3705ac`, `M1_DIR`, and
  `M1_FILES`: path, size and SHA-256 of `model.index`, `model.data-00000-of-00001` and the
  validation event log). `fetch_m1(cache_dir, source_dir=None)` downloads them (only
  `https://` or `file://`), or copies them from `source_dir`, verifies size and SHA-256,
  and writes `CACHE/m1_weights.npz`: the 22 `network/` variables (the Adam slots are
  dropped) under their TF names, plus a `provenance` JSON string; it loads with
  `allow_pickle=False`. The cache defaults to `$XDG_CACHE_HOME/astro-simclr/m1/a3705ac`
  (`~/.cache/...`): outside the repository, never committed. A verified cached file is not
  downloaded again.
- **The weight mapping**, `to_state_dict(weights)` and `load_m1_model(weights)` (strict
  `load_state_dict`, eval mode): HWIO → OIHW; dense `(in, out)` → `(out, in)`; fc1's input
  rows permuted from TF's `(h, w, c)` flatten of the 6×6×64 map to torch's `(c, h, w)`
  (`kernel.reshape(6, 6, 64, 64).transpose(3, 2, 0, 1)`); the feature BN's gamma, beta and
  moving statistics (M1's are 0 and 1) with `num_batches_tracked = 0`; biases unchanged.
  `tf_shapes(with_features)` lists the expected names and shapes (441,075 trainable
  parameters with features, 439,557 without; tested).
- **`numpy_forward(weights, images, features, pad=3)`**: the independent reference of the
  port check: M1's forward pass in float64 NumPy, NHWC, on the raw TF-layout weights (no
  transposes): the 4 rotations `np.rot90(k)` stacked on the batch axis, zero padding,
  convolutions as sums of shifted matrix products with TF's symmetric "same" padding (odd
  kernels only), 2×2 max-pools, the `(h, w, c)` flatten, the mean over rotations, the BN in
  inference form, fc2, fc3, logits. `pad=0` takes 27×27 crops (the uint8 diagnostic).
- **`run`**: `run_m1_oracle(data_dir, weights, curve=None)` returns the report; `write_report`,
  `read_val_curve` and `write_curve` (CSV `step,val_loss,val_accuracy`, float32 values in
  their shortest form). The two checks, on the 500 `red` Validation stamps:
  1. *port equivalence*: torch vs `numpy_forward` on the same float-path crops,
     `max |Δlogit| <= 1e-4` and 100 % argmax agreement;
  2. *data path*: M1 through `crop_split` (NaN approximation on): correct stamps within ±2
     of 475 **and** the selection loss within ±0.005 of −0.0616. `selection_loss` is the
     training loss `P_stamps_loss(0.5)` averaged as TF did: the mean of batch-64 means in
     file order (the last batch, of 52, counts as one), the definition of 2.2a.

  The report's top-level `passed` is both checks; `compare-reference --ref CD21
  --oracle-report FILE` reads it (`reference.read_oracle_report`). It also holds the
  stamps the NaN approximation touches (indices and accuracy) apart from the others; M1's
  **test** accuracy, loss and per-class recall (archived, never a decision input); the
  diagnostics **on Validation only**: NaN approximation off, uint8 21×21 crops padded to
  27 (2.2a's interim input) and uint8 27×27 crops of real pixels (phase 1's input; through
  `numpy_forward`, since the torch model takes 21×21), each with its accuracy change; the
  summary of M1's validation curve; data and package versions.
- **CLI:** `fetch-m1 [--cache-dir DIR] [--source-dir DIR]` and `m1-oracle --data-dir DIR
  [--out DIR] [--cache-dir DIR] [--source-dir DIR]`, which writes `OUT/m1_oracle.json` and
  `OUT/m1_val_curve.csv` and exits 0 if both checks passed, 1 otherwise (or on a fetch or
  data error). The committed results are in `results/phase2/replication/` (derived numbers
  only), from:

  ```bash
  uv run astro-simclr m1-oracle --data-dir dataset/prepared/td_ztf_stamp_17_06_20_red-002 \
      --out results/phase2/replication
  ```

- **Result on `red` (2026-11-08, Mac M4, CPU, torch 2.14.1):** passed. Port: max |Δlogit|
  1.6e-6, argmax 500/500. Data path: **475/500 (exact)**, loss −0.06162 (M1's log:
  −0.06157 at iteration 4,780). 41 Validation stamps are touched by the NaN approximation,
  all 41 correct (434/459 for the others); 34 in Test. M1's test accuracy is **0.936**
  (468/500; SN recall 0.86). Validation diagnostics: NaN approximation off 0.948 (−0.002),
  uint8 21 padded 0.928 (−0.022), uint8 27 0.910 (−0.040).
- Tests (`tests/test_oracle_*.py`) use synthetic checkpoints and event files written by
  `tests/_tf_writer.py` (from the format description, with knobs that break one property at
  a time), random M1-shaped weights, `file://` downloads and the synthetic prepared data.

## `alerts` (task 3.1)

`src/astro_simclr/alerts/`: recovers the raw ZTF alerts behind a prepared split from
ALeRCE's public services, verifies them against the stored stamps and stores them with
their provenance. Method, layout, statuses and the acceptance are in
[`docs/data.md`](data.md), "Raw alerts"; the format and dependency choices in ADR 0011.
Nothing in the package imports torch.

```
prepared split ──targets (destandardise)──► fetch_row ──► AlerceClient ──► Recorder ──► Fetcher ──► urllib
                                              │  match_detections, ties   (cache, snapshot)  (rate limit,
                                              │  parse_alert (fastavro, FITS)                 retries)
                                              │  verify_cutouts (alerce_normalise)
                                              ▼
                         batch ──_commit──► shard-NNNNN.npz (ok cutouts + raw responses), then index.csv lines
                                                    ▲
                     Snapshot (offline RawFetch) ───┘  fetch-alerts --from-snapshot
```

- `transport`: `RateLimiter` (request starts at least `1/rate` apart; one limiter covers
  both hosts), and `Fetcher.get(url, max_bytes)`.
  - Retries timeouts, `OSError`, `http.client.HTTPException`, 5xx and 429: 5 attempts,
    backoff `2·2^(k-1)` s capped at 60 s, but at least a numeric `Retry-After` (up to
    600 s), no jitter (a single sequential client).
  - Other statuses raise `HttpError` at once.
  - `urllib_opener` opens only `https://` and follows redirects only to `https://`
    (`_HttpsRedirects`). It reads at most `max_bytes` in chunks within a 300 s total
    deadline (`TimeoutError`, retried), and sends the generic `USER_AGENT`.
- `snapshot`: `Recorder` is the `Getter` between the client and a `RawFetch`
  (`online_fetch(fetcher)` or `store.Snapshot.fetch`).
  - It caches cone searches and detection lists by URL (an LRU bounded at 256 MB of
    bodies), and keeps every new response (url, kind, UTC time, SHA-256, bytes) as
    `pending()` until a commit has made it durable (`drop_pending(n)`).
  - It lists the responses each row used (`row_refs`, as `kind:sha256`).
- `alerce`: `AlerceClient.objects_near`, `detections` and `alert_avro` (the URLs of the
  module docstring), and `make_client(rate)`, the production stack (the CLI's seam for
  tests).
  - The JSON is validated: `items` of ZTF-shaped `oid`s, and detections with an integer
    `candid` (string or int) and numeric `magpsf`/`sigmapsf`. Others are skipped, and a
    wrong shape raises `ResponseError`.
  - An `oid` is checked against `ZTF\d{2}[a-z]{7}` before it goes into a URL. Size caps:
    16 MB of JSON, 1 MB of avro.
- `parse`: `check_container` walks the avro container before fastavro reads it.
  - It accepts the `null`, `deflate` and `snappy` codecs, and bounds the decompressed
    size to 4 MB: the snappy preamble length, or deflate inflated with an output limit.
  - `check_schema` refuses a writer schema that is not a record, or that has an array or
    map of zero-size items (`null`, empty records, `fixed` of size 0). Every other item
    takes at least one byte, so the decoding work is bounded by the data.
  - `parse_alert` then reads exactly one record with the embedded writer schema (ALeRCE's
    record names differ from ZTF's). It checks `objectId`, `candid` and the required
    candidate fields and types, and keeps the scalar fields, non-finite floats as `null`.
  - The three cutouts are parsed by `fits.read_fits_image`, and CCD-edge ones repaired
    with `repair_non_squared_stamp` (or refused with `CutoutShapeError` under
    `repair=False`, the archive corpus; see `archive`).
- `fits`: `gunzip` (a gzip stream with an output cap), and `read_fits_image`, a strict
  reader of one 2D `BITPIX -32/-64` image, at most 256 px per side, without scaling. It
  returns native float32 `(NAXIS2, NAXIS1)`, NaN kept, with no flip. It replaces a FITS
  library for the one layout ZTF uses.
- `fetch`:
  - `targets`, `interleaved_order` (round-robin over the classes) and `match_detections`
    (distinct candids within 1e-4 on both magnitudes);
  - `alerce_normalise` (`crop_then_normalise` at 63 without the NaN approximation) and
    `verify_cutouts`;
  - `fetch_row` (never raises for a miss);
  - `fetch_alerts`: resume (an `ok` row whose shard is missing is fetched again), and
    batches of `batch_size` rows. A full batch is detached before `_commit`, so a failed
    commit is never retried by the `finally`. The run stops between rows when `stop()`
    turns true. The cooldown follows 3 consecutive *transient* failures
    (`RowResult.transient`). `fetch.json`'s data identity and every status-deciding
    setting (radius, tolerances, `max_candidates`) must match on `--resume`.
- `by_oid` (task 3.5): `fetch-alerts --oids FILE --out DIR`. `FetchItem(kind, oid,
  candid)` (`detections` or `avro`), `read_items` (`OID` / `OID CANDID` lines),
  `fetch_by_oid(items, out, client=...)`: the same `Recorder`, `Fetcher`, shard format
  (`write_shard` with no cutouts: the avro bodies are the alerts), batch commits, stop,
  cooldown and resume as `fetch_alerts`. Its own `index.csv` (`ItemLine`: kind, oid, candid,
  status, n_detections, raw_shape, shard, responses, requests, seconds, note) and
  `fetch.json` (`astro-simclr-oid-fetch` v1). Statuses: `ok`, `not_found` (404),
  `refused` (other 4xx), `bad_alert` (final), `http_error` (transient, retried).
  `final_statuses(out)`; `Snapshot(out)` serves its responses. `alerce` gained
  `DetectionRecord` / `parse_detection_records` (`candid`, `jd = mjd + 2400000.5`,
  `has_stamp`, `fid`, `ra`, `dec`), `detections_url`, `avro_url`; `store.Snapshot`
  gained `__contains__`, `iter_many` (bulk, shard by shard) and `read_many`; `transport`
  gained `https_opener` (for the streamed Zenodo download).
- `store`: the layout and its readers. `IndexRow` and `COLUMNS`; `IndexWriter`
  (append, fsync, repairs a truncated last line); `read_index` and `latest_rows`;
  `write_shard` / `read_shard` (atomic; `allow_pickle=False`); `load_recovered` (the `ok`
  alerts, by row); `Snapshot` (URL → stored body, SHA-256 checked); `summarize` /
  `Summary` / `format_summary` (counts, groups, ghosts, the recovery of non-ghost rows,
  CCD-edge counts, costs, the 98 % acceptance); `duplicates` / `format_duplicates` (a
  candid stored for several rows, within or across splits).

Tests (`tests/test_alerts_*.py`, offline): `_alerts_support` writes FITS and avro files
(fastavro, all three codecs) and runs a fake ALeRCE on an in-memory sky with one Validation
row per status. They cover:
- ties decided by the stamp check;
- two objects in the cone, and cache hits on a shared object;
- resume, and the retry of `http_error` only;
- orphan shards, a truncated index, and an interrupt mid-batch;
- the cooldown, the offline rebuild, and duplicates across splits;
- backoff and `Retry-After` on a fake clock;
- the size caps and malformed inputs.

## `r5` (task 3.5)

`src/astro_simclr/r5/`: builds R5, the phase-3 5-class object-level benchmark, under RJ23's
recipe from public sources. The recipe, sources, commands and outputs are in
[`docs/data.md`](data.md), "R5"; the decisions in ADR 0020. Nothing imports torch.

```
r5-sources ──► sources (Zenodo archive streamed, MD5/size checked, CSV read unextracted;
                        BTS CSV via Fetcher; JSON records with SHA-256)
build-r5 ──► build.build_r5 ──loop──► select.evaluate(Inputs, Responses) ──missing──► alerts.by_oid.fetch_by_oid
                 │                         │  recipe: draw_alert, ndet, JD_MAX     (online rounds only)
                 │                         ▼
                 │                  Selection (decisions per walk, drops, missing)
                 ▼
          build.assemble: components (oid + 1.5" across R5 and CD21) ─► constrained make_object_splits
                          ─► assert_no_shared_groups ─► first stamps of Test
          build.write_outputs: r5/, r5-first/ (write_raw_alerts), alerts.csv, splits.json,
                          index.csv, deviations/drops/eras/split_counts.csv, build.json
```

- `recipe`: `JD_MAX`, `SEED`, `SUBCLASSES` (the plan's table: label, subclass, source,
  source classes, RJ23's target and cap), `SN_TARGET`, `BOGUS_MAX_NDET`; `eligible`
  (stamped, `jd ≤ JD_MAX`, by candid), `draw_alert` (`default_rng([seed, oid_hash])`),
  `ndet`, `first_stamp_order`, `bts_subclass`, `stable_hash` / `oid_hash`, `year_of_jd`.
- `sources`: `download_zenodo` (injectable stream; partial file removed on failure),
  `download_bts` (injectable `get`), `load_source` (SHA-256 against the record),
  `read_zenodo`, `read_bts` (sexagesimal RA/Dec), `SourceRow`.
- `select`: `read_cd21` (task 3.1's index, latest line per row, all three splits),
  `Inputs`, `Responses` (several snapshots by URL, the first serving; the by-oid store's
  final failures), the candidate pools per subclass (`_add_zenodo`, `_add_bts`,
  `_add_cd21`, the label claims), `_decide` (the cut order of the module docstring),
  `_walk` (stops at accepted + pending = target), `evaluate` → `Selection`. Parsed
  responses are cached across rounds; bodies stream shard by shard, so memory holds one
  shard and the compact detection summaries.
- `build`: `load_inputs`, `assemble` (groups, the dropped cross-split and mixed-label
  groups, the constraints, the splits, the matrix, the first stamps), `write_outputs`
  (cutouts re-parsed from the snapshot into a memory map, then `write_raw_alerts`),
  `build_r5` (the round loop; dry run, offline refusal, `.partial` directory renamed at
  the end), `check_expected`.
- `cli`: `fetch_by_oid_command`, `r5-sources`, `build-r5`, `stop_on_signals`.
- `protocol.grouping.assign_groups(..., allowed=mask)` and
  `protocol.object_splits.make_object_splits(..., allowed={group: splits})`: the
  constraints of the cross-benchmark rule (format version 2 of the object-splits file);
  unchanged draws without them.

Tests (offline; `tests/_r5_support.py` builds the world: a tiny Zenodo archive and BTS CSV
through the real download functions, task 3.1's store of CD21 with detection JSON and avro
in its shards, and a fake service for the by-oid fetch). They cover:
- every drop cause;
- the seeded draw before `JD_MAX` and ndet against the world's detections;
- the first stamps;
- the cross-benchmark matrix and the pinned groups;
- the deviation table;
- the walk's stop at the target;
- the seeded selection order;
- an online build in rounds, then an offline rebuild that is byte-identical;
- the refusal of an incomplete snapshot;
- `--expect`;
- the CLI commands.

## `archive` (phase 5, task 5.1a; ADR 0019)

`src/astro_simclr/archive/`: the unlabelled pretraining corpus from the ZTF public alert
archive. The rules, layout, commands and budget are in [`docs/data.md`](data.md),
"Archive corpus"; the decisions in ADR 0019. Nothing in the package imports torch or opens
a network connection.

```
tarball file or stdin ──► TarballStream ──Member──► parse_alert(repair=False) ──Alert──► _decide
  (SHA-256, gzip cap,       (regular .avro            (alerts.parse: container,         │ invalid_values
   tar stream mode,          members only,             schema and size checks,          │ first_detection_failure
   member and count caps)    nothing extracted)        strict FITS)                     │ exclusions
                                                                                         ▼
           corpus.json (rules) ◄── check_corpus      NightWriter ──► nights/.N.partial ──rename──► nights/N
                                                                                         │
                                         corpus_index (one alert per oid, read-time exclusions) ◄──┘
```

- `stream`:
  - `TarballStream(source, name=, max_member_bytes=1 MiB, max_decompressed_bytes=64 GiB,
    max_members=5M)` takes a `Path` or a binary stream.
  - The raw bytes pass through `_HashingReader` (SHA-256, byte count; drained to the end
    so the hash covers the whole file), then `gzip.GzipFile`, then `_LimitedReader` (the
    decompressed cap), then `tarfile.open(mode="r|")`.
  - It yields `Member(position, name, data)` for regular `.avro` files under the member
    cap. Everything else is counted in `StreamStats`.
  - `_BoundedTarInfo._proc_member` refuses pax and GNU long-name headers over 1 MiB
    (tarfile reads their body whole) and GNU sparse members, before any body is read.
  - After the last member the gzip stream is read to its end, which checks its CRC-32
    and length trailer.
  - Corrupt or truncated input raises `TarballError` (a `DataError`); real I/O errors
    propagate.
  - `night_of` and `tarball_name` map `ztf_public_YYYYMMDD.tar.gz` to and from the night,
    with a date check.
- `selection`:
  - `Selection(oid_year=True, max_ndethist=None)`.
  - `first_detection_failure(alert, selection)` returns `previous`, `oid_year`,
    `ndethist` or `None`, in that order. `oid_year` and `jd_year` are its helpers.
  - `Exclusions` holds the oids, the file's SHA-256 and name. `load_exclusions(path)`
    reads UTF-8 text, at most 64 MB, one oid per line, `#` comments, each line checked
    against `raw_alerts.OID_PATTERN`.
- `layout`:
  - `INDEX_DTYPE` and `NightWriter(out, night, rows_per_file)`: `add` buffers one cutout
    file; `commit(meta)` writes `index.npy` and `meta.json`, fsyncs them, renames the
    `.partial` directory and fsyncs the parent.
  - `check_corpus(out, record)` creates or compares `corpus.json` (written atomically).
    `corpus_lock(out)` is an exclusive `flock` on `OUT/.lock`. `recover_night` puts back a
    night that a crash left at `.<night>.replaced` during an overwrite.
  - `load_night` checks `meta.json`, the files' dtypes and shapes, the row totals and the
    index values. `verify_night` re-hashes. `Night.cutout(row)` reads one stamp.
  - `list_nights` and `iter_nights` see complete nights only.
  - `corpus_index(out, exclude=)` keeps the earliest alert of each `oid` (`jd`, then
    `candid`), drops the excluded oids, and sorts by night and row. `excluded_rows` gives
    the mask.
- `ingest`:
  - `Rules(selection, exclusions, rows_per_file, overwrite, limits)`.
  - `ingest_archive(tarballs, out, rules, progress=)` refuses two tarballs for one night.
    `ingest_tarball` handles one file, and `ingest_stream(handle, night, out, rules)` one
    piped night.
  - `_done` checks `corpus.json`, then skips a complete night (same name and, for a file,
    the same size) without opening its input.
  - `_decide` and `_index_row` apply the rules of `REASONS` in order. The counts go to
    `meta.json`.
- `plan`:
  - `plan_nights(seed, labelled_per_year, later_per_year, spares, first, last)` returns
    `NightPlan`: `PlannedNight(night, stratum, rank, pilot)` with `use` (`h1` or
    `ablation`).
  - The strata are drawn from `default_rng([seed, year])`, and the pilot stratum after
    them.
  - `estimate(plan, stamps_per_night, tarball_gb, download_hours, measured)` gives the
    stamps, GB and hours per year, per use and in total. The basis is assumed or measured
    on a corpus. `format_estimate` prints it.
- `alerts.parse` changes for the archive:
  - `oid` and `candid` are optional (the alert's own are taken).
  - `repair=False` raises `CutoutShapeError` for non-63×63 cutouts.
  - `Alert.previous_detections` counts the `prv_candidates` entries with a `candid`.

Tests (`tests/test_archive_*.py`, offline): `_archive_support` writes ZTF-like avro alerts
(three schema eras: 2018 without `drb`, 2020, and 2024 with `fp_hists`) and tarballs with
hostile members. They cover:
- streaming, and the hash of the whole file;
- links, devices, and `..` and absolute names, with nothing extracted;
- the member, decompressed-size (gzip bomb) and member-count caps;
- truncated, non-gzip and corrupt files;
- every reason of the ingest on one night, with the NaN of the cutouts kept bit for bit;
- the optional rules, and a piped night;
- resume without reading a complete night, and a night cut short and redone;
- another file for a complete night, and a corpus refusing other rules;
- a deterministic writer, `corpus_index`, the loader's checks and the CLI;
- the plan's strata, seeds, spares, pilot and uses, and the assumed and measured estimates.

## `optim` (task 1.5)

`src/astro_simclr/optim.py`: the SimCLR pretraining optimizer and LR schedule.

- `LARS(params, *, lr, momentum=0.9, use_nesterov=False, weight_decay=0.0, eeta=0.001,
  legacy_trust_ratio=True)`: port of `utils/lars.py` (classic momentum). The legacy SimCLR
  call is `LARS(params, lr=lr, weight_decay=0.0, momentum=0.9, use_nesterov=True)`.
- `cosine_warmup_lr(epoch, base_lr, ...)` and `cosine_warmup_scheduler(optimizer,
  base_lr, ...)` (a `LambdaLR`, stepped once per epoch) reimplement the legacy timm 0.6.x
  `CosineLRScheduler` (warmup 5 epochs from `lr/3`, no warmup prefix, `t_initial` 350, one
  cycle, `lr_min` 1e-4 afterwards). timm is not a dependency.
- **Characterization:** `tests/fixtures/optim/legacy_optim.npz`, from
  `scripts/characterize/optim/generate.py` (legacy LARS, two steps, with/without Nesterov
  and weight decay; the optimizer's `param_groups[0]["lr"]` for epochs 0–400 under timm
  0.6.13, lr 0.258 and 1.258). Tolerance `atol=1e-6`.
- **Behaviour changes:**
  - device-agnostic (the legacy `get_device()` returned -1 on CPU and crashed, and would
    have mapped MPS to `cuda:N`);
  - `legacy_trust_ratio=False` uses `.gt(0)` instead of the always-true `.ge(0)`: a zero
    gradient (legacy: NaN parameter) and a zero weight (legacy: never updated) get trust
    ratio 1. The default keeps the legacy behaviour, because the fix changes ResNet
    results;
  - `add_(tensor, alpha=)` replaces the deprecated `add_(scalar, tensor)` overload;
  - hyperparameters come from each param group (legacy: partly from instance attributes;
    identical with one group); `p.grad` is no longer modified in place by weight decay;
  - dropped: the unused `epoch` argument/counter of `step`, the unimplemented
    `classic_momentum=False` and the never-consulted `exclude_from_*` lists; the closure
    runs under `torch.enable_grad()` and the step under `torch.no_grad()`.
- **Per-group exclusion (task 5.0b, ADR 0025):** a param group with
  `exclude_from_adaptation: True` (`EXCLUDE_KEY`) gets neither the weight decay nor the trust
  ratio (plain SGD with momentum): SimCLR's treatment of BatchNorm parameters and biases,
  used by `simclr.optim`. The key is read with `group.get`, so groups without it (every
  existing run) and optimizer states saved before it are unchanged.
- **Performance note (kept for equivalence):** the trust ratio is moved to the host with
  `.item()` once per parameter per step, as in the legacy code. On CUDA (and MPS) that is a
  device-to-host synchronisation per parameter tensor, about 60 per step for a ResNet-18,
  which stalls the GPU queue. It is kept because the scalar arithmetic then matches the
  legacy fixtures; a fused, sync-free LARS would be a later-phase optimisation with its own
  equivalence check.

## `training` (task 1.6)

`src/astro_simclr/training/`: the Lightning 2 port of the legacy `utils/training.py`
(PyTorch Lightning 1.4), plus the data modules and run helpers the CLI (1.7) builds on. A run
is:

```python
seed_run(settings)  # L.seed_everything(seed, workers=True)
module = SimCLRModule(
    encoder_name="pstamps",
    image_size=27,
    projection_dim=300,
    temperature=0.1791,
    lr_encoder=0.258,
    nt_xent_legacy_denominator=settings.nt_xent_legacy_denominator,
)
data = ContrastiveDataModule(
    path, image_size=27, batch_size=550, augmentation="astro7", num_workers=settings.num_workers
)
callbacks = build_callbacks(settings, SimCLRModule.MONITORABLE, dirpath=out / "checkpoints")
build_trainer(settings, callbacks=callbacks).fit(module, datamodule=data)
```

`seed_run` must come first: building a module draws its initial weights from the global
generators. Nothing that draws random numbers runs before the network in any module, so a
seed gives the same initial weights as the bare `models` classes (tested), hence as the
legacy modules.

**Modules** (`modules.py`):

| Class | Legacy class | Network | Trained parameters |
|---|---|---|---|
| `SimCLRModule` | `SimCLR` | `SimCLRNet(build_encoder(encoder_name), ...)` | all, `NT_Xent` |
| `LinearEvalModule(pretrained)` | `Linear_classifier` | frozen copy of the encoder, `h/‖h‖` ⊕ raw features, `LinearClassifier` | the head only |
| `FineTuneModule(pretrained)` | `Fine_tuning` | copy of the encoder ⊕ `BatchNorm1d(23)` features; ResNets: `LinearClassifier`, stamp encoders: `ThreeLayerClassifier(·, 64, 64)` | all |
| `SupervisedCEModule` | `Supervised_Cross_Entropy` | `StampClassifier(drop_rate, with_features)` | all |
| `SupervisedCEResNetModule` | `Supervised_Cross_Entropy_Resnet` | ResNet ⊕ `BatchNorm1d(23)` features, `LinearClassifier` | all |

- The four classifiers share `ClassifierModule`: `P_stamps_loss(beta_loss)` on one-hot
  targets; each train/validation/test step appends its detached `argmax(softmax(logits))`,
  true labels and loss to an **explicit per-stage buffer**, which `on_*_epoch_end` reads and
  clears (the Lightning 2 replacement of `*_epoch_end(outputs)`). The epoch loss is the mean
  of the batch losses, as before.
- Logging is `self.log` (any logger), under the legacy names: `Loss/<Stage>`,
  `Accuracy/<Stage>`, `Precision/<Stage>`, `Recall/<Stage>` with `<Stage>` in `Train`,
  `Validation`, `Test`; plus `accuracy_val`, `loss_val` (and `accuracy_test`, `loss_test`)
  on the progress bar. SimCLR logs `Loss_encoder/Train`, `loss_train_enc`, `lr`,
  `positive_mean`, `negative_mean`. The x-axis is now Lightning's global step, not the epoch.
- Validation and test store the row-normalised confusion matrix (on the CPU) in
  `module.confusion_matrices["val"|"test"]`, the labels `(y_pred, y_true)` in
  `module.predictions["val"|"test"]`, for the figures and metrics JSON of 1.7, and the
  softmax probabilities (N, 5) in `module.probabilities["val"|"test"]`, for the per-stamp
  predictions files of task 2.3. The validations inside `trainer.fit` skip all three
  (`trainer.state.fn == FITTING`; audit item 2): they only feed the monitor, and every
  output of a run comes from `trainer.validate` / `trainer.test(ckpt_path=best)` after the
  fit, so a 30,000-step `carrasco2021` run no longer builds 3,000 confusion matrices nor
  copies them, the labels and the probabilities to the host, only to throw them away. The
  epoch metrics pass
  `validate_args=False` (the labels are argmaxes, always in range), which skips
  torchmetrics' value checks and their host sync on a GPU. Neither changes a number:
  `tests/test_fit_validation.py` runs a short `carrasco2021` schedule with the old
  behaviour patched back in and compares `metrics.json` (except `timing`), the predictions
  files and the best weights.
- `MONITORABLE` (class attribute) lists what `Settings.monitor` may select:
  `accuracy_val`, `loss_val` for the classifiers, `loss_train_enc` for SimCLR (which has no
  validation loop). The protocol default `accuracy_val` therefore raises `ConfigError` for
  SimCLR: `pretrain` (1.7) must set `monitor: loss_train_enc`.
- `LinearEvalModule` and `FineTuneModule` take the pretrained `SimCLRModule` or its
  hyperparameters. They store `hparams["pretrained"] = dict(pretrained.hparams)` (plain
  values), so `load_from_checkpoint(path)` rebuilds the encoder architecture and the
  checkpoint's `state_dict` restores its weights (tested: same logits after a reload). In
  `setup` they reject a data module whose `image_size` differs from the encoder's.
- **Optimizers** (`optimizers.py`): `build_optimizer(name, params, lr)` knows `AdamW` and
  `Adam` (`betas=(0.5, 0.9)`), `SGD` and `LARS` (`weight_decay=0`, momentum 0.9, Nesterov;
  `legacy_trust_ratio` from the module hparam `lars_legacy_trust_ratio`, default `True`).
  Every module accepts every name; an unknown name raises `UnknownOptimizerError` at
  construction, listing the valid ones. SimCLR adds `cosine_warmup_scheduler` with
  `interval: "epoch"`, `frequency: 1` (tested: `last_epoch` equals the epoch and the lr
  follows `cosine_warmup_lr` while each epoch has several optimizer steps).
- `NT_Xent` takes the module hparam `nt_xent_legacy_denominator`, which the caller sets from
  `Settings.nt_xent_legacy_denominator` (ADR 0002). Both legacy flags are saved in the
  checkpoint hparams.

**Metrics** (`metrics.py`), the one implementation of the four legacy copies, on
torchmetrics ≥ 1 with `task="multiclass"`: `classification_metrics(y_pred, y_true)` gives
**micro** accuracy (explicit `average="micro"`: `MulticlassAccuracy` defaults to macro, and
every legacy accuracy is micro; tested on fixed, imbalanced predictions, also through a
module's logged `accuracy_val`), macro precision and macro recall (classes absent from
predictions and targets left out, as in torchmetrics 0.6); `confusion_matrix(y_pred,
y_true)` uses `normalize="true"` (an empty row is zeros). Added in 1.7 for the metrics JSON:
`confusion_counts` (not normalised) and `macro_f1` (classes absent from predictions and
targets left out, like precision and recall). `classification_metrics(...,
validate_args=False)` skips torchmetrics' dtype and range checks of the label values (same
numbers; for labels known to be in range, like the modules' argmaxes); its own shape check
always runs.

**Data modules** (`datamodules.py`) build their datasets in `setup()` (legacy:
`prepare_data()`, which runs on one process only), from a prepared layout directory,
memory-mapped:

- `ClassifierDataModule(path, image_size, batch_size, eval_batch_size=100,
  augmentation="center_crop", num_workers=0)`: `StampDataset` batches `(image, 23 features,
  one-hot label)`; Train uses `get_augmentation(augmentation, image_size)` (registry names or
  legacy aliases), Validation and Test always `center_crop`. Rejects a pretraining file.
- `ContrastiveDataModule(path, image_size, batch_size, augmentation, num_workers=0)`:
  `ContrastiveDataset` view pairs of the Train split of any file; no validation loader.
- Loaders keep the legacy settings: training shuffled, `drop_last=True`,
  `pin_memory=False`; evaluation unshuffled. A Train split smaller than one batch raises
  `DataError` in `setup("fit")` (with `drop_last` it would train on nothing); `validate`
  and `test` do not check it. Invalid names and sizes fail at construction.
- Hyperparameters are saved into checkpoints with the path as a string, so checkpoints load
  with `torch.load(weights_only=True)` (tested).
- With `num_workers > 0` and the `spawn` start method (macOS), each worker gets a pickled
  copy of the dataset, and a pickled memory map is a full in-memory copy of the split;
  under `fork` (Linux) the map is shared. To revisit before R3 runs on macOS. `fork` is
  the Linux default only up to Python 3.13 (the project's version): 3.14 switches it to
  `forkserver`, which pickles like `spawn`, so R1-R4 (`num_workers: 4`) would copy the
  split into every worker. A move to 3.14 sets `multiprocessing_context` explicitly first.

**Run helpers** (`runs.py`):

- `seed_run(settings)`: `L.seed_everything(settings.seed, workers=True)`.
- `apply_fp32_precision(settings)` (ADR 0010): `torch.backends.cudnn.conv.fp32_precision`
  and `torch.set_float32_matmul_precision("highest" | "high")` from `fp32_precision`
  (the matmul call, not `torch.backends.cuda.matmul.fp32_precision`, because after the
  latter is set to `tf32` `torch.get_float32_matmul_precision()` raises, and Lightning calls
  it on Ampere+ GPUs). Process-global; `run_command` calls it before anything is built.
  `fp32_precision_record(device)`: the effective policy (`tf32` only on a CUDA device of
  compute capability 8.0+ that allows it), the torch settings read back
  (`fp32_backends`) and `compute_capability`, for `metrics.json["machine"]`. Since task
  4.1 both are `engine.precision`'s (the plain engine shares them; same behaviour).
- `build_callbacks(settings, monitorable, dirpath, filename="best")`: `EarlyStopping` and
  `ModelCheckpoint(save_top_k=1, enable_version_counter=False)` (the legacy
  `ModelCheckpoint_V2` overrode a private method to the same effect), both on
  `settings.monitor` with `patience` and `min_delta` from the settings. `mode` comes from
  `MONITOR_MODES` (`accuracy_val` max; `loss_val`, `loss_train_enc` min); a monitor the
  module does not log raises `ConfigError`. Nothing hard-codes `accuracy_val`.
- `build_last_checkpoint(dirpath)` (1.7): a second, monitor-less `ModelCheckpoint` that
  writes `last.ckpt` every epoch, what `--resume` continues from. Not `save_last=True`:
  since Lightning 2.6, `save_last` writes only in steps where the top-k checkpoint was also
  written, so `last.ckpt` went stale after an epoch without improvement (found by the
  resume test of 1.7). `build_callbacks` also passes `check_finite` and
  `divergence_threshold` from the settings to EarlyStopping (1.7).
- `build_trainer(settings, callbacks=..., logger=True, **overrides)`: `accelerator`,
  `devices`, `max_epochs`, `deterministic`, and (when set) `limit_train_batches` /
  `limit_val_batches` from the settings, `benchmark=not deterministic`
  (legacy `True`), `default_root_dir=output_dir`. This is the only place a device is chosen.
  The progress bar is a `TQDMProgressBar(refresh_rate=progress_bar_refresh_rate)` (default
  50 batches; Lightning's 1 cost ~0.4 ms a step of the loop's ~1.3 ms on the CPU, audit
  item 2); a rate of 0, or `enable_progress_bar=False` in the overrides, hides it.

**Tests** run on the CPU everywhere and on MPS/CUDA when available
(`tests/_training_support.py: ACCELERATORS`): `fast_dev_run` for every module and variant
(with/without features; stamp, `pstamps` and ResNet encoders) on the synthetic data
(`make_synthetic`, then `prepare`; `tests/conftest.py`), followed by a test pass; a seeded
2-epoch SimCLR run gives the same step losses and epoch loss twice (and a different seed
does not); the scheduler steps once per epoch; selection on `loss_train_enc` and `loss_val`;
EarlyStopping stops a flat run; every epoch and every validation pass starts from an empty
buffer (a module whose predictions flip between passes); downstream checkpoints reload.
Worker seeding (`num_workers=2`) is covered by the augmentation tests (on Linux, where the
workers fork), not repeated here because spawned workers cost about 25 s on macOS.

**Behaviour changes against the legacy code** (no result of a valid legacy run changes):

- Loading and loaders moved from the modules to the data modules; the modules' unused
  arguments are gone (`method`, `batch_size*`, `augmentation`, data paths; `drop_rate` of
  the linear and ResNet modules).
- `LinearEvalModule` and `FineTuneModule` copy the pretrained encoder, so training never
  modifies the module passed in (legacy: the linear module held the caller's SimCLR module,
  fine-tuning trained the caller's encoder in place). The frozen encoder stays in eval mode
  through Lightning's `train()` calls (legacy: `eval()` inside `forward`).
- `FineTuneModule` accepts `stamp<k>` encoders, with the `pstamps` head (legacy: no branch,
  so the module failed). `SupervisedCEResNetModule` accepts `resnet34`; a non-ResNet name
  raises `ValueError`.
- Every optimizer name works in every module; unknown names raise (legacy:
  `UnboundLocalError`).
- Every classifier has a test loop (legacy: only linear eval and fine-tuning), and logs
  `loss_val` (legacy: only those two).
- SimCLR's logged `lr` is the rate the epoch trained with; the legacy value was read after
  the epoch's scheduler step, i.e. the next epoch's rate.
- **Kept, although odd:** `LinearEvalModule` passes `softmax(logits)` to `P_stamps_loss`,
  which applies its own softmax (legacy `Linear_classifier.prediction_target_loss`). Fixing
  it changes linear-evaluation results, so it is left to a later phase.

### The `carrasco2021` schedule (task 2.2a)

ALeRCE's training procedure as Lightning pieces, all under `src/astro_simclr/training/`.

**Tiling** (`tiling.py`, a re-export of `engine/stream.py` since task 4.3, so the plain
engine shares the stream without importing Lightning). `tile_indices(labels)` is ALeRCE's
`balance_data_by_replication` on indices: classes in sorted label order; a smaller class
becomes `max // n` full copies of its indices (file order) plus their first `max % n`; the
largest class is kept. Every class then has `max` entries (`red`: 5 × 14,542 = 72,710).
`PermutationStream(n, batch_size, seed)` reproduces `tf.data`
`shuffle(100000).repeat().batch(64)` (the buffer holds the whole tiled set, so each pass is a
uniform permutation, and batches run across pass boundaries with no drop): batch `b` is the
stream positions `[64b, 64b + 64)`, pass `j` the permutation of
`numpy.random.default_rng([seed, j])`. A batch is a pure function of `(seed, b)`: the
documented exception to "randomness comes from `seed_everything`", and what makes a
mid-epoch resume exact. TF's random numbers are not reproduced.

**Device data** (`device_data.py`; since task 4.6 the Lightning-free half, `TensorSplit`,
the crop builders and `check_split_identity`, lives in `tensor_splits.py` and is
re-exported, so the plain engine builds the same arrays without Lightning).
`TensorSplit(images, features, labels, fits=None)`:
float32 (N, 3, 21, 21) (or (N, 6, 21, 21) with the 63×63 context of task 2.5-models, or
the whole stamps (N, 3, 63, 63) of the ConvNeXt rows, plan row B8; `IMAGE_SIDES`),
float32 (N, 23), int64 (N,), and optionally the float32 (N, 6) `class_augment.stamp_fits`
of the crops (C5 / C6), or (N, 11) with `flux_rescale.fit_central_psf` after them (C7),
validated. `without_train_indices` keeps Train's fits with its
stamps; `protocol_splits` builds roles without fits. When the Train split has fits, its
training batches carry them as a fourth element (validation and test never do).
`DeviceTensorDataModule(source, batch_size=64, eval_batch_size=64, seed=0, balance=True,
planner=None)` takes the three splits (a mapping,
or a callable resolved once at the first `setup`), moves them once to
`trainer.strategy.root_device` (CPU without a Trainer) and yields `(crops, features,
one-hot labels)`:

- Train: an epoch is `len(tiled) // batch_size` steps (1,136 for `red`); the loader starts
  at `trainer.global_step` and yields stream batches up to the next epoch boundary, as
  device slices of each pass's permutation and a device gather. No DataLoader, no workers.
  It implements `state_dict` / `load_state_dict`, so Lightning treats it as resumable; the
  state is the stream definition, and a resume with other data, batch size or seed raises
  `DataError`. A Train split smaller than one batch raises `DataError` in `setup("fit")`.
- Validation / Test: consecutive `eval_batch_size` batches in file order, the last partial.
  With 64, `ClassifierModule`'s epoch loss (the mean of the batch losses) is TF's
  selection loss, the mean of batch-64 means in file order.

The splits come from a prepared layout (task 2.2c):

- `float_crops(path, nan_approximation=True, context=False, side=21)`, **the paper's input**: each
  split's `images_f32.npy` through `data.float_input.crop_split(21)` (per-crop min-max
  without epsilon, NaN → 0, the NaN approximation), channels first, and the 23 kept
  features; `context=True` (`Settings.context: avgpool63`) appends the 3 channels of
  `context_split`. `side=63` (the ConvNeXt rows; ADR 0022) is `crop_split(63)`, the same
  normalisation on the whole stamp (about 2.4 GB for `red`), without context. A layout
  without float32 stamps raises `DataError` naming `prepare-data --images float32`.
- `interim_uint8_crops(path, side=21)`, the stand-in of 2.2a (centre 21×21, or the whole
  63×63, of the uint8 stamps / 255, no re-normalisation), kept as the `paper_input: uint8`
  diagnostic.
- `without_train_indices(splits, excluded)` drops Train stamps (the leaked one) in file
  order, before the tiling; Validation and Test are unchanged.
- Split runs (task 2.5-infra; see `protocol`): both builders take `names=("Train",
  "Validation")` so that Test is never read, and `protocol_splits(splits, parts, n_train)`
  gathers a run's parts by pool index into the module's three roles (`Train` = the part
  trained on and tiled, `Validation` = the selection part, `Test` = the evaluation part).
  The data module is unchanged except for `run_identity` (`SplitRun.identity()`), which
  the training loader adds to its checkpoint state as `split_run` (absent for a normal
  run, so normal checkpoints are unchanged). `check_split_identity(stored, current)` refuses
  a resume across split runs or between a split run and a normal run (`ConfigError`).

**Horizon** (`schedule.py`). `HorizonCallback(dirpath, filename="best", monitor="loss_val",
mode="min", initial_horizon=30000, increment=10000)` on a Trainer with
`max_steps=initial_horizon, val_check_interval=10, check_val_every_n_epoch=None`:

- after each fitting validation (not the sanity check, not `trainer.validate`), a strict
  improvement of `monitor` (NaN never improves) keeps a copy of `state_dict()` on its
  device and sets `horizon = max(horizon, step + increment)`, written to
  `trainer.fit_loop.epoch_loop.max_steps`;
- `on_train_end` loads the best weights into the module and writes `best.ckpt` once
  (`trainer.save_checkpoint(weights_only=True)`, then the best's `epoch` and `global_step`
  in place of the last ones), so `validate/test(ckpt_path=best)` and the metrics work
  unchanged; nothing is written if no validation ran;
- `state_dict()` saves the monitor, mode, horizon, best score, step, epoch and weights (CPU)
  in every `last.ckpt`; `load_state_dict` restores them (the horizon is the larger of the
  saved one and the new run's `max_steps`) and refuses another monitor or mode;
- with `last_every_n_steps` it writes `last.ckpt` itself, **at the end of the validation**
  of every such step (after the update above) and once more when training ends. Not a
  `ModelCheckpoint(every_n_train_steps=...)`: that saves in `on_train_batch_end`, before the
  step's validation, and on resume Lightning marks the owed validation as done without
  running it, so a new best at a checkpoint step was lost (found in review; tested). Hence
  `checkpoint_every_n_steps` must be a multiple of `val_every_n_steps`;
- a resume at or past the horizon (from the final `last.ckpt`) skips Lightning's fit loop,
  so `on_train_end` never runs; `on_fit_end` then writes `best.ckpt` from the restored
  state (tested);
- it exposes `best_model_path`, `best_model_score` and `best_step` (what the metrics read
  from a `ModelCheckpoint`).

Step numbering: TF validated after iteration `i` ∈ {0, 10, ...}, i.e. after `i + 1` steps;
Lightning after 10, 20, ... steps. `best_step` counts steps done, so M1's "iteration 4,780"
is step 4,781 here; the one-step shift of the validation grid is not corrected.

**Module.** `modules.PaperClassifierModule(beta_loss, lr, optimizer, with_features=True,
dropout_active=False, drop_rate=0.5, adam_betas="code")`: `PaperStampClassifier` under
`ClassifierModule` (`P_stamps_loss`, the same logging and buffers). Its optimizer is the
factory's with `adam_betas=optimizers.PAPER_ADAM_BETAS[adam_betas]`: `code` (0.9, 0.999),
ALeRCE's, or `paper` (0.5, 0.9), a diagnostic (task 2.2c). `build_optimizer` takes
`adam_betas` (default the legacy (0.5, 0.9)); Adam and AdamW use `eps=1e-8`, and Adam has no
weight decay (torch's defaults, now explicit). TF1 places ε differently; ADR 0003 explains
why torch's placement is kept. Task 2.5-models adds `paper_arch="rot4"`,
`group_fields=None`, `context="none"` (the model from `paper_variants.build_paper_model`)
and `rot90_augment=False`, `augment_seed=0`: with augmentation, `training_step` rotates
each stamp by `k·90°`, `k` from `numpy.random.default_rng([augment_seed, global_step])`
(device-independent, exact on resume); validation and test are not augmented. Plan rows
C5 and C6 add `aug_noise`, `aug_feature_jitter`, `synthetic_bogus` and `aug_min_r2` (all
off by default; see "Class-aware augmentation" below), and row C7 `flux_rescale`,
`flux_k_min`, `flux_k_max` (see "Transient flux rescaling"). The `feature_set` diagnostic
(ADR 0016) adds `feature_set="all"`, passed to the model; with `none` the module's
`with_features` reads false. It is saved in the hyperparameters only when it is not `all`,
so the faithful `hparams.yaml` and checkpoints are unchanged.

### Class-aware augmentation (plan rows C5 and C6)

`src/astro_simclr/training/class_augment.py`; decisions in ADR 0006. Training only, in
`PaperClassifierModule.augment_plan` (host, planned ahead by the training loader; see
"Prefetch" below) and `apply_plan` (in `training_step`, on the device-resident batch): C6,
C7 (below), then C5's noise on the rows C6 did not replace, **then** the rotation
of `rot90_augment`, then the feature jitter on every row. The class augmentation runs before
the rotation because the stamp fits describe the unrotated crop (C7's PSF centroid in
particular; C5 / C6 are rotation-agnostic). Validation and test are never augmented.

The cross-channel relation (`stamp_fits`, `FIT_COLUMNS`, `masked_pixels`, `renormalise`,
`perturb` and their constants) lives in `src/astro_simclr/stamp_relation.py`, outside the
`training` package (whose `__init__` imports Lightning), so that the SimCLR views can use it
engine-free; `class_augment` re-exports it unchanged (ADR 0025).

- **The difference is computed, not augmented.** `stamp_fits(images)` fits, per crop and in
  float64, `diff ~ a·sci + b·tmpl + c` (least squares over the 441 pixels, a
  pseudo-inverse), its R², and the background sigmas of the science and the template
  (1.4826 × the MAD of the two-pixel border, masked pixels left out): float32 (N, 6),
  `FIT_COLUMNS`. `experiments._build_paper` computes them once on the run's Train role
  (the fold's training part in a split run) when `aug_noise` or `synthetic_bogus` is on, and
  records `fit_exclusions` (the share of each class below `aug_min_r2`) in
  `metrics.json["paper_data"]["class_augment"]`. On `red` Train the median R² is 0.77–0.95
  per class and `sign(a)` agrees with `isdiffpos` for 98–99.6 % of the stamps. Any change
  `dsci`, `dtmpl` moves the difference by `a·dsci + b·dtmpl` (`perturb`), with no
  independent term.
- **C5, noise** (`noise_augment`): for the rows with R² ≥ `aug_min_r2`, `dsci` and `dtmpl`
  are N(0, (u·`aug_noise`·σ)²) per pixel, with `u ~ U(0, 1)` per channel and σ that crop
  channel's background sigma; then `renormalise`: ALeRCE's per-channel min-max without
  epsilon over the unmasked pixels, NaN → 0. Masked pixels (`masked_pixels`: the exact zeros
  of a channel with several, the former NaNs) stay 0 and out of the min/max. Other rows and
  the context channels are returned bit for bit.
- **C5, feature jitter** (`jitter_features`): N(0, `aug_feature_jitter`²) on the 19
  continuous kept features; `DISCRETE_FEATURES` (`isdiffpos`; the integer counts
  `ndethist`, `ncovhist`, `non_detections`) never change.
- **C6, synthetic hot-pixel Bogus** (`synthetic_bogus`): `round(fraction·B)` rows, taken in
  a random order among the possible sources: positive alerts only (`a > 0`, `b < 0`; a hot
  pixel in the science cannot trigger a negative alert, and an `a < 0` source would give a
  central dip that no real difference has) of Asteroid or Bogus rows only (their templates
  are mostly empty at the centre, like the real positive Bogus; an AGN or VS template puts
  a nucleus or a star under the spike), R² ≥ `aug_min_r2`, σ_sci > 0, no masked pixel. A
  Bogus source keeps its features; an Asteroid source takes those of a real positive-alert
  Bogus row of the batch (none: the Asteroid is skipped). Each becomes: template unchanged;
  science `−(b/a)·tmpl + n + spike`
  (the static sky the relation implies, `n` Gaussian of the source science's σ, the spike
  10–50 σ); difference `a·(n + spike) + c`; then the min-max; label Bogus.
  `template_centre_z` (the template's central 3x3 mean as z against its border, a source
  when above `CENTRE_SOURCE_Z` = 5) measures the share of synthetic templates with a source
  at the centre: 6.1 % on `red` Train, against 29.5 % for the real positive-alert Bogus
  (`scripts/inspect_class_augment.py` checks at most + 0.05). `spike_patterns`: one pixel at the centre, or (probability 0.5) a 2-pixel
  horizontal or vertical line, second pixel 0.25–1 of the peak.
- **The spike is sharper than any PSF**: `moment_fwhm` (the pixel second moments less 1/12,
  as a circular Gaussian FWHM) gives at most 0.48 px for a spike (0.96 px along a line),
  against `MIN_PSF_FWHM_PX` = 1.19 px, the smallest FWHM of a pixel-integrated Gaussian
  fitted to the difference of every Asteroid and SN of `red` Train
  (`scripts/inspect_class_augment.py`; the report and the sample grids are in
  `results/phase2/class-augment/`).
- **Randomness**: one NumPy generator per augmentation, `augment_rng(seed, step, stream)` =
  `default_rng([seed, global_step, stream])`, streams 1 (noise), 2 (jitter), 3 (synthetic);
  every draw has a fixed size per batch. Device-independent, exact on resume, and
  `rot90_augment`'s `[seed, step]` draws are unchanged.
- **Prefetch** (ADR 0006, item 10): each augmentation is a host *plan* and a device *apply*
  (`plan_noise` / `apply_noise`, `plan_synthetic` / `apply_synthetic`, `plan_jitter`;
  C7's `plan_flux` / `apply_flux`; the public `noise_augment`, `synthetic_bogus`,
  `jitter_features`, `flux_rescale` are apply(plan(...))). `PaperClassifierModule.
  augment_plan(host_batch, step)` plans a whole step (C6, C7, C5, rotation, jitter: a dict
  of CPU tensors); `apply_plan(batch, plan)` applies it with the same torch operations.
  For an augmented run (`module.augmenting`), `experiments` gives the data module
  `planner=module.augment_plan`: the training loader keeps a CPU copy of the Train split,
  gathers the step's rows there, plans, and yields `(*batch, plan)` with the plan uploaded
  by `upload_plan` (pinned + `non_blocking` on CUDA); `training_step` recognises the plan
  (a trailing dict) and only applies it. No `.cpu()` and no blocking copy in the step;
  bit for bit the in-step path (`augmented_batch`), which stays for direct calls
  (`tests/test_augment_prefetch.py`).

**Tested** (`tests/test_class_augment.py`, `tests/test_split_runs.py`): the fit recovers
known `a, b, c`; the computed difference equals `a·sci + b·tmpl + c` before the min-max;
`renormalise` equals the float path's min-max; augmented crops have per-channel min 0 and
max 1 with masked pixels at 0; low-R² rows, skipped rows and the context are untouched; the
discrete features are untouched; synthetic rows are Bogus, with the spike as the science's
maximum, at most 2 pixels above half of it, the same spike as a central peak of the
difference (positive centre z for every replaced row), a Bogus keeping its features and an
Asteroid taking a positive-alert donor's, only positive Asteroid or Bogus sources, and the
synthetic templates no fuller at the centre than the real positive Bogus' (+ 0.05); no sources or no donors replace nothing; every spike is
below `MIN_PSF_FWHM_PX`; defaults off are bit-identical with or without fits; evaluation is
not augmented; the settings' checks; the preset deviations; a p4 run on every accelerator;
an augmented fold run (fits equal to those of the fold's training part) and its exact
resume.

**Tested:** the tiling example and exact balance; the stream against the concatenated
permutations; batches against the tiled stream, on the CPU; eval file order; validation at
10, 20, 30 across 7-step epochs; the horizon extending (bests at 10, 20, 40 → stop at 65)
and the best restored in the module and in `best.ckpt`; `mode=max`, NaN, missing monitor;
`run_command` writing `metrics.json`; a resume from a `last.ckpt` written after the step-9
validation of 2-step epochs (validation every 3) restoring the horizon and best step,
validating at exactly the remaining steps, and ending with bit-identical best weights and
metrics to the uninterrupted run; a resume at the horizon still writing `best.ckpt`; a
smoke run on every available accelerator.

The full replication settings are the `carrasco2021` preset (see `presets`):
`train-supervised --preset carrasco2021`, or `--config configs/phase2/carrasco2021.yaml`.

### Transient flux rescaling (plan row C7)

`src/astro_simclr/training/flux_rescale.py`; decisions in ADR 0007. Training only, in
`PaperClassifierModule.class_augment`, after C6 and before C5's noise; never Bogus (nor C6's
synthetic rows). Scaling the whole difference would do nothing: the per-channel min-max is
scale-invariant. So:

- `fit_central_psf(images)`: once per Train stamp (`experiments._with_train_fits`, appended
  to `stamp_fits` as the 11-column `AUGMENT_COLUMNS` table), a pixel-integrated circular
  Gaussian plus a constant fitted by Levenberg-Marquardt (per stamp, float64,
  `torch.func` Jacobians) to the central 11×11 of the normalised difference, masked pixels
  with weight 0: `psf_flux`, `psf_x0`, `psf_y0`, `psf_sigma`, `psf_ok` (finite, flux > 0,
  centroid within 2 px, FWHM in [1, 6] px, S/N ≥ 5). About 3.6 ms per stamp. `psf_exclusions`
  (the failed share per class) goes to `metrics.json["paper_data"]["flux_rescale"]`.
- `rescale_transient(crops, params, a, k)`: difference `+ (k − 1)·PSF`, science
  `+ (k − 1)·PSF / a` (`a` from the cross-channel fit: a negative alert's science gets
  fainter when its alert gets brighter), template unchanged; then `renormalise` with the
  masked pixels.
- `rescaled_features(features, k)`: through M1's statistics, `magpsf − 2.5 log10 k` and
  `sigmapsf / k` (a background-dominated error; ADR 0007).
- `flux_rescale(...)`: a row is rescaled with probability `flux_rescale` if not Bogus, its
  fit is ok and `k_floor` (`10^(−0.4 (diffmaglim − magpsf))`, the smallest `k` that keeps
  `magpsf' ≤ diffmaglim`) is at most `flux_k_max`; `k` is log-uniform in
  `[max(flux_k_min, floor), flux_k_max]`. Stream 4 of `class_augment.augment_rng`.

On `red` Train: fits skipped AGN 10.9 %, SN 3.9 %, VS 9.9 %, Asteroid 4.9 %; median headroom
AGN 0.35, SN 0.96, VS 2.18, Asteroid 0.92 mag; median fitted FWHM 2.0–2.2 px
(`scripts/inspect_flux_rescale.py`, `results/phase2/flux-rescale/`).

**Tested** (`tests/test_flux_rescale.py`): the fit recovers a known Gaussian and rejects an
off-centre source and pure noise; masked pixels are ignored; before the min-max only the
central source changes (the residual after removing `k·PSF` equals the original's, and the
cross-channel relation still holds); a refit of the rescaled difference has `k` times the
flux, and `magpsf` / `sigmapsf` move by `−2.5 log10 k` and `1/k`; the alert constraint over
many draws, a stamp beyond its limit left alone; Bogus never rescaled; seeded draws; the
module (training only), the settings and the preset deviation; a p4 run on every
accelerator; and in `tests/test_split_runs.py` an augmented fold run (the fits are the
fold's training part's) and its exact resume.

## `engine` (phase 4, task B0; ADR 0014)

`src/astro_simclr/engine/`: the contracts of the plain-PyTorch training loop that replaces
Lightning (phase 4). B0 freezes them; the loop (4.1), the typed config (4.2) and the data
tiers (4.3) are built against them, and the 4.7 gate compares the two engines. **Nothing in
the package imports Lightning** (checked in a fresh interpreter), so a flag written against
it works on both engines until 4.8. ADR 0014 lists, with file and line, every hidden
Lightning behaviour the loop reproduces (C1-C11) and the test that pins each.

| Module | Contract |
|---|---|
| `task` | `Task`: `module`, `make_optimizer()` (after the device move), `train_step(batch, step)` (the loss with its graph; `step` = optimizer steps done before it), `eval_step(batch) -> EvalOutput`, `state()` / `load_state()`. `mean_of_batch_losses`: the monitored value, the float32 mean of the batch means (not the per-sample mean). |
| `policies` | `Monitor.improves` (strict, NaN never, the first finite value always: `HorizonCallback`'s rule); `ValidationGrid.due(steps_done)` (multiples of the interval only: no step 0, no epoch-end, no forced final validation); `Horizon.extended` / `resumed`; `EarlyStop` (Lightning's semantics, for `rj23`); `Keep = stop \| best`; `Schedule`, whose defaults are the `carrasco2021` settings. |
| `checkpoint` | `Checkpoint` (a `TypedDict` of plain values, read with `weights_only=True`) and its frozen key set `CHECKPOINT_KEYS`: format, engine, identity (A0's run id and attempt, the split run), `steps_done` (also the stream position), model, optimizer, task, policy, stream definition, every RNG stream, numerics (ADR 0010) and versions. `check_resumable` refuses Lightning checkpoints, other engines, formats and incomplete ones. |
| `device` | `DevicePolicy`: one device (`world_size != 1` refused), `cudnn.benchmark = not deterministic`, `use_deterministic_algorithms(deterministic)`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"` only when deterministic (`torch_flags()`, checked equal to Lightning's `_set_torch_flags`), ADR 0010's precision. |
| `data` | `Batch` / `TrainBatch` (equal to `device_data`'s); `DataTier` (`device`, `pinned`, `memmap`, all exact); `TierChoice` (non-float32 storage is a recorded deviation); the `BatchSource` (`batch(step)`, a pure function of the stream and the step) and `EvalSource` protocols. |
| `randomness` | `host_generator(seed, step, stream)` (= `class_augment.augment_rng`: per-sample parameters, device-independent); `device_generator(seed, step, stream, device)` (per-pixel fields: the seed is device-independent, the values are not, so runs record the device); `RESERVED_STREAMS`, the stream numbers in use (1-4 the paper model's augmentations, 5 the `block` stream's passes (4.3, ADR 0018), 6-9 the SimCLR pilot's views, noise fields, batch order and diagnostics; ADR 0025). |

`tests/test_engine_contracts.py` also pins the Lightning 2.6.6 facts the checklist relies on
(the validation grid across 7-step epochs, `global_step` inside `training_step`, the
gradient `None` at validation, `no_grad` in the fit's validations and `inference_mode` in
`trainer.validate`, the epoch numbering of `metrics.json`), so an upgrade that changes one
fails here.

### The engine core (task 4.1; ADR 0014, amendment of task 4.1)

The loop itself, against the contracts above. The CLI is not wired to it yet (task 4.6);
the Lightning path is unchanged.

| Module | What it does |
|---|---|
| `loop` | `fit(build, train, validation, schedule=, device=, seed=, run_dir=, identity=, resume=, on_validation=, progress=) -> (task, FitResult)`: precision and flags (`DevicePolicy.apply`), `seed_everything`, `build()` on the CPU, the move, `make_optimizer()`; per step forward, `zero_grad(set_to_none=True)`, backward, `step()`; after each step the grid, then the validation (`zero_grad`, every submodule's mode captured, `eval()`, `no_grad`, the modes restored), the policies (`Monitor`, `Horizon`, `EarlyStop.update`), `last` if due; the loop ends when `steps_done >= horizon` after that. `evaluate(task, sources)` (`inference_mode`) and `evaluate_kept(task, result, sources, keep)` (`stop` first, then the restored best). `FitResult`: `last_step`, `horizon`, `best_step`, `best_score`, the derived `best_epoch` / `last_epoch`, this fit's `validations`, `steps`, `fit_seconds`, the best weights. Only `loss_val` is monitored for now. |
| `checkpoint` (4.1 part) | `capture_rng` / `restore_rng` (Python, NumPy's global, torch CPU, CUDA or MPS); `numerics_record`; `save_checkpoint` (atomic: a temporary file, then a rename); `load_checkpoint` (`weights_only`, `check_resumable`); `check_identity` (run id and split run; the attempt may differ) and `check_numerics` (precision and determinism); `BestWeights`, `ckpt/best.pt`'s record (weights of the best validation, not resumable), read by `load_best`. |
| `policies` (4.1 part) | `EarlyStop.update(value, best, wait, mode)`: Lightning's `_evaluate_stopping_criteria` in its order and float32 arithmetic. |
| `precision` | ADR 0010's `apply_fp32_precision(precision)` and `fp32_precision_record(device)`, moved from `training.runs` (which delegates). |
| `randomness` (4.1 part) | `seed_everything(seed)`: `lightning.seed_everything`'s three calls, without its environment variables. |
| `scalars` | `ScalarLog`: `scalars.jsonl`, one flushed line per validation (step, epoch, `loss_val`, mean `loss_train` since the previous validation, `improved`, horizon, wait count, seconds); a resume keeps the lines of steps up to the checkpoint's; NaN is `null`. `read_scalars`. |
| `record` | `machine_record(device)` (`experiments._machine`'s keys, `engine: plain`) and `recorded(record)`, which ends a `runs.record.start_run(..., engine="plain")` record `finished`, `failed` or `cancelled`. |

Run outputs under `run_dir` (ADR 0015's layout): `scalars.jsonl`, `ckpt/last.pt` (after the
validation of every `checkpoint_every_n_steps` steps and at the end of training, if a step
ran), `ckpt/best.pt` (at the end, if `best` is kept). **No progress bar**: a `progress`
stream gets one line per validation when given.

**Resume order** (ADR 0014, C10): `load_checkpoint`, `check_identity`, `check_numerics`;
precision and seed; `build()` and `model` loaded on the CPU; the move; the optimizer and its
state; `task.load_state`; the policy (the monitor must match; `Horizon.resumed`); the
stream (`TrainSource.resume`, refused on another stream); the RNG streams last.

**Tested.** `tests/test_engine_loop.py` (a toy task with an active dropout and a frozen
submodule): the grid and the horizon, the step keys, the modes, gradients and no-grad /
inference contexts, `keep`, early stopping against Lightning's `EarlyStopping` over value
sequences, the checkpoint contents, exact resume through the global RNG streams (and its
control: without `restore_rng` it is not exact), every resume refusal, atomic writes, the
seeding and flags, the scalars, and the `engine: plain` manifest.
`tests/test_engine_differential.py` is **the 4.7(a) CPU differential test**, a CI gate.
Since task 4.6 it runs `run_command`'s `carrasco2021` path and the real plain runner
(`tasks.plain.run_plain`, section `tasks`) in one process on the synthetic data, with a
horizon extension, and they agree bit for bit: the initial weights; the steps, `loss_val`
and every `state_dict` tensor after each validation; the whole `metrics.json` but for the
entries that name the engine or the run (`machine.engine`, `checkpoints`, `versions`,
`accelerator`, `run`, `timing`, `config.output_dir`); the `predictions_*.npz` arrays; both
manifests `finished`. Cases: the 2.5 flags (`single`, `p4`, `p4m`, `convnext_atto`,
`context`, `rot90`, C5, C6, C7, both `feature_set` diagnostics, `fp32_precision: tf32`,
all of C5-C7 with `p4`), a fold and the screen, a mid-run plain resume, and each
accelerator present (deterministic). It also ports the three Lightning-bound resume tests
to the plain runner.

### The data layer (task 4.3; ADR 0018)

One dataset API and three tiers that serve identical bytes, so the tier is a memory versus
bandwidth choice that cannot change a result.

| Module | What it does |
|---|---|
| `dataset` | `Dataset`: float32 `images` of any per-row shape (possibly a memory map), optional in-memory `features` / `labels` (+ `n_classes`) / `fits`, and an optional `rows` selection (`select` composes; `materialised` copies). Adapters: `prepared_split(path, split)` (`classifier`; `unlabelled` for a pretraining `Train`; refuses layouts without `images_f32.npy`), `raw_alerts_dataset(path, candids=...)` (raw cutouts, labels from the index, the 23 CD21 features as float32), or the constructor (`arrays`). |
| `stream` | `tile_indices` and `PermutationStream` (moved from `training/tiling.py`); `BlockShuffleStream`; `TrainOrder` (tiling, then the stream over its positions; per-pass row orders cached; `state()` = the stream keys plus `rows_sha256`). |
| `tiers` | `open_train(dataset, TierChoice, StreamSpec, device) -> TrainSource` (`batch(step)`, the frozen `TrainBatch`; `columns(step)` by name, for unlabelled kinds; `state()`, `resume(state, steps_done)`, `describe()` for the manifest) and `open_eval(...) -> EvalBatches` (consecutive batches, the last partial; `iter_columns()`). |
| `bench` | `python -m astro_simclr.engine.bench`: batches/s per tier and device, interleaved rounds, median. |

- **`device`:** the materialised dataset on the device; each pass's row order is uploaded
  once and sliced, so a step makes no host-to-device copy. Its batches are `torch.equal` to
  `DeviceTensorDataModule`'s (tested), and the stream keys are the Lightning loader's.
- **`pinned`:** the dataset in pageable host memory; the step's rows are gathered into one
  of two pinned staging buffers and uploaded `non_blocking` on the current stream, one step
  ahead; a buffer is refilled only after the event behind its last upload completes. On a
  CPU device nothing is staged or prefetched.
- **`memmap`:** the images stay in their file; a batch is read sorted by file offset and
  scattered into the staging buffer, then uploaded as in `pinned`. float32 only.
- **Shuffle granularity** (`StreamSpec.shuffle`): `row` (default; a uniform permutation per
  pass, the paper's, the only one with `balance`) or `block` (`block_rows`-row blocks
  permuted, then positions shuffled within windows of `window_blocks` blocks; a batch reads
  at most two windows of contiguous ranges, but file neighbours share batches far more
  often). It is part of the stream definition, served identically by every tier.
- **The stream definition** (`TrainSource.state()`, the checkpoint's `stream`): `n`,
  `batch_size`, `seed`, the block parameters when used, `rows_sha256` and `storage_dtype`.
  Not the tier: a resume may move to another tier and stays exact (tested). The only
  mutable state is the position (`steps_done`).
- **Storage dtype:** `float16` / `uint8` images (`TierChoice.storage_dtype`) are quantised
  once and dequantised on the device after the gather; `describe()["deviation"]` is true,
  a resume under another dtype is refused, and the `memmap` tier refuses them. `uint8`
  dequantises through a 256-entry table of `k / 255` computed on the host (numpy, correctly
  rounded), not by a division on the device: CUDA divides by a Python scalar as a
  multiplication by the reciprocal, one ulp off for 126 of the 256 values, so the same
  batch would differ between CPU and CUDA.

**Tested** (`tests/test_engine_tiers.py`, on CPU and on MPS or CUDA when present): `pinned`
and `memmap` `torch.equal` to `device` for 40 steps across pass boundaries, balanced and
not, with fits and a scrambled selection, and in evaluation (partial last batch, two
passes); `block` identical on every tier; `device` equal to `DeviceTensorDataModule`; exact
resume mid-pass on every tier and onto another tier; out-of-order steps; resume refusals
(seed, batch size, balance, storage dtype); the quantised values and their record;
unlabelled columns. `tests/test_engine_stream.py` and `tests/test_engine_dataset.py` test
the stream and the adapters on the synthetic prepared and raw-alerts layouts.

**Throughput** (`python -m astro_simclr.engine.bench`, Apple M4, torch 2.14.1, 2026-11-08,
batch 64, 300 timed batches × 7 interleaved rounds, median; the Mac was shared with other
agents, load average about 40-50, so the rounds spread by 2-3×; the memmap file is in the
page cache). Results: `results/phase4/data-tiers-bench-{crop21,stamp63}.json`.

| Rows | Device | `device` | `pinned` | `memmap` |
|---|---|---|---|---|
| 3×21×21 (the paper's crops), n = 20,000 | CPU | 6,180 batches/s | 7,611 | 5,128 |
| | MPS | 7,460 | 1,440 | 1,451 |
| 3×63×63 (phase 5's stamps), n = 10,000 | CPU | 1,398 (4.3 GB/s) | 1,615 | 929 |
| | MPS | 2,873 (8.8 GB/s) | 666 (2.0 GB/s) | 589 (1.8 GB/s) |

Every tier serves more than 580 batches/s of 63-px stamps (under 1.7 ms per batch, a
fraction of a SimCLR step). On MPS the host tiers cost about 0.7 ms per batch against
0.13 ms for `device`: with unified memory there is nothing to gain from them, so `device` is
the choice whenever the data fit; `pinned` and `memmap` exist for CUDA machines and the
phase-5 corpus. Their CUDA and cold-disk numbers are measured on the GPU machine.

## `tasks` (phase 4, task 4.6; ADR 0014, amendment of task 4.6)

`src/astro_simclr/tasks/`: the paper classifier on the plain engine, and what both engines'
runners share. **Nothing in it imports Lightning** (tested in a fresh interpreter, with
the CLI and the preflight). `astro-simclr run CONFIG` with `runtime.engine: plain` trains
through it; Lightning stays the default until 4.7/4.8.

| Module | What it does |
|---|---|
| `common` | The run checks of `run_command`, moved from `experiments` (which re-exports them): `check_accelerator` (`AcceleratorError`), `dataset_path`, `check_names`, `check_schedule` / `check_paper_variant`, `check_split`. |
| `paper` | `paper_inputs(settings, data, split_run)`: the splits as `TensorSplit`s (float path or uint8 diagnostic, leaked Train stamps removed, a split run's roles, the C5-C7 fits) and the `paper_data` entry; `experiments._build_paper` calls it too. `load_backbone` (ConvNeXt weights, ADR 0022). `PaperNet` (the model under `model`, the loss as `criterion`: `PaperClassifierModule`'s `state_dict` keys). `PaperTask`, the engine `Task`: a step plans the augmentation of step `step` on a host copy (`training.paper_augment.PaperAugment`, the object the Lightning module delegates to), uploads and applies it, then the loss; `build_optimizer` with the configured betas. `build_paper_task(settings, info)` builds it on the CPU (after the seed); `check_paper_task` runs its checks before the manifest. |
| `results` | `StageOutput` and `classifier_results`: the Validation / Test (or selection / evaluation) entries of `metrics.json`, `predictions_*.npz` and the confusion-matrix figures, moved from `experiments` and used by both engines. |
| `plain` | `run_plain(command, settings, resume=, split=, on_step=, on_validation=, progress=)`: `run_command` for `train-supervised` of the paper classifier on `carrasco2021` (every 2.5 flag, split runs, `fp32_precision`). The same checks, the data on the `device` tier, `start_run(..., engine="plain")`, `engine.fit`, the final evaluation of the restored best, `metrics.json` (the Lightning keys, with `machine.engine: plain`, `checkpoints` `ckpt/{best,last}.pt`, no Lightning version, `accelerator` the device type), the manifest `finished`. `resume="auto"`: `OUTPUT_DIR/ckpt/last.pt`. Refuses other commands, models and schedules (use `lightning`), `devices != 1`, and `limit_train_batches` / `limit_val_batches`. |

`training/__init__` loads its names lazily (PEP 562), so its Lightning-free submodules
(`paper_augment`, `tensor_splits`, `class_augment`, `flux_rescale`, `metrics`,
`optimizers`, `tiling`) import without Lightning.

**Tested.** `tests/test_engine_differential.py` (the 4.7(a) gate, section `engine`) and
`tests/test_tasks_plain.py`: a plain run's outputs, every refusal (before the manifest, or
a failed attempt after it), the accelerator and resume resolution, `astro-simclr run` with
`runtime.engine: plain`, the Lightning-free imports, the timer and the preflight's CD21 run.

## `presets` (task 2.2c)

`src/astro_simclr/presets.py`: named settings for `train-supervised --preset NAME`, applied
after the legacy protocol and before `--hparams`, `--config`, `--set` and the flags.
Decisions and provenance: ADR 0003.

- `PRESETS["carrasco2021"]`: the paper-faithful replication of Carrasco-Davis et al. 2021
  (model M1). `paper_stamp_classifier`, `image_size` 21, `paper_input: float32`,
  `nan_approximation: true`, `with_features: true`, `dropout_active: false` (`drop_rate`
  0.5 for the diagnostic), `exclude_leaked_train: true`, batch 64, eval batch 64,
  `num_workers` 0; `Adam`, `adam_betas: code` (0.9, 0.999), lr 1e-3, `beta_loss` 0.5;
  `schedule: carrasco2021`, `monitor: loss_val`, `max_steps` 30000, validation every 10
  steps, horizon +10000, no EarlyStopping; dataset `td_ztf_stamp_17_06_20_red-002`.
  `PRESETS["carrasco2021-no-features"]` is the same with `with_features: false`.
- `configs/phase2/carrasco2021.yaml` and `carrasco2021-no-features.yaml` hold the same
  values with their provenance (`--config`); `tests/test_presets.py` keeps them equal.
- `DIAGNOSTICS`: the documented diagnostic flags of 2.4 (`dropout_active: true`,
  `adam_betas: paper`, `nan_approximation: false`, `paper_input: uint8`).
- Task 2.5-models adds `paper_arch: rot4`, `group_fields: null`, `context: none` and
  `rot90_augment: false` to both presets (the defaults, so the faithful settings are
  unchanged), so that a variant run is reported as a deviation, not as faithful. Plan rows
  C5 and C6 add `aug_noise: 0.0`, `aug_feature_jitter: 0.0`, `synthetic_bogus: 0.0` and
  `aug_min_r2: 0.5` the same way (ADR 0006), row C7 `flux_rescale: 0.0`,
  `flux_k_min: 0.5`, `flux_k_max: 1.5` (ADR 0007), ADR 0010 `fp32_precision: ieee`, so
  that a TF32 run is a deviation, and ADR 0016 `feature_set: all`, so that a `position` or
  `none` run is a deviation of `carrasco2021` (`preset_for` still goes by
  `with_features`).
- `preset_for(settings)`: the preset of a `carrasco2021`-schedule run (by `with_features`),
  else `None`. `preset_report(settings)`: `name`, `faithful` (no preset key changed),
  `deviations` (key → `{value, preset}`) and `diagnostics` (the flags turned on), written
  to `metrics.json["preset"]`. Seed, paths, accelerator and `checkpoint_every_n_steps` are
  not compared.

The pilot (task 2.2c, one seed on the GPU machine):

```bash
uv run --no-sync astro-simclr prepare-data dataset/td_ztf_stamp_17_06_20_red-002.pkl \
    --images float32 --out dataset --overwrite      # once: adds images_f32.npy
uv run --no-sync astro-simclr train-supervised --preset carrasco2021 --accelerator cuda \
    --data-dir dataset --seed 0 --output-dir outputs/phase2/carrasco2021/seed-0
uv run --no-sync astro-simclr compare-reference outputs/phase2/carrasco2021/seed-0 --ref CD21 \
    --oracle-report results/phase2/replication/m1_oracle.json   # the faithful tier needs it
```

## `hparams` (task 1.7)

`src/astro_simclr/hparams.py`: what the training subcommands take from the legacy code.

- `COMMAND_DEFAULTS`: per subcommand, the settings that differ from the `Settings`
  defaults, so each subcommand's protocol is its legacy `iter_*.py` one. Only `pretrain`
  differs (`iter_clr.py`: 400 epochs, patience 100, `min_delta` 0.002 on `loss_train_enc`,
  `resnet18`, `simclr`, batch 450, LARS lr 1.258, dataset `td_ztf_stamp_simclr_300`); the
  classifiers keep the `Settings` defaults (220 epochs, patience 40, `min_delta` 0.001 on
  `accuracy_val`).
- `load_legacy_hparams(path, command)` maps a `results/hparams_best_*.yaml` file to
  settings through documented tables. A **classifier file** (`hparams_best_ce*.yaml`;
  `CLASSIFIER_KEYS`, identity names) configures `train-supervised`, `linear-eval` and
  `fine-tune`. A **SimCLR file** (`hparams_best_simclr.yaml`; `SIMCLR_KEYS`) is split:
  `pretrain` takes `*_encoder`, `encoder_name`, `projection_dim`, `temperature`,
  `augmentation`, `image_size`; `linear-eval` / `fine-tune` take `*_classifier`,
  `beta_loss`, `with_features`, `image_size` (its `augmentation` is the *pretraining* one).
  The keys of the other group are skipped and reported on stderr, never applied.
- Errors (`ConfigError`): a key outside the table (the message lists the valid keys), a
  file of the wrong kind for the subcommand, `method` other than `simclr` (SupCon is in the
  backlog) and `balanced_batch: true` (no such argument in the current legacy module).
  Augmentation names may be legacy aliases.

## `experiments` (task 1.7)

`src/astro_simclr/experiments.py`: `run_command(command, settings, resume=None,
callbacks=(), progress_bar=True)` runs one training subcommand, for the CLI and the
preflight alike:

1. `check_accelerator`: `cuda` without `torch.cuda.is_available()` (or `mps`, `gpu`
   without one) raises `AcceleratorError`, naming the torch build and the preflight, before
   anything is built;
2. `apply_fp32_precision` (ADR 0010), then `dataset_path`: `data_dir/<dataset>` must be a
   prepared layout (`DataError` says how to
   prepare it);
3. `seed_run` before any module is built (downstream: after loading the pretrained
   checkpoint, so the head's initial weights do not depend on the loading path);
4. the module and data module from the settings: `SimCLRModule` + `ContrastiveDataModule`,
   or a classifier module + `ClassifierDataModule`. `linear-eval` / `fine-tune` load the
   pretrain checkpoint (`load_pretrained`: a `.ckpt` or a pretrain output directory; its
   hyperparameters must be SimCLR's, and `image_size` must equal the encoder's);
   `fine-tune` trains with `lr * fine_tune_lr_factor`;
5. `trainer.fit(ckpt_path=resume)` with EarlyStopping, `best.ckpt` (by monitor) and
   `last.ckpt` (every epoch); `--resume auto` means `OUT/checkpoints/last.ckpt` if present;
6. classifiers: `validate` and `test` on `best.ckpt`, then the figures and the metrics.

Outputs, all under `output_dir`: `manifest.json` (task A1, section `runs`: `queued` and
`running` before the fit, then `finished`, `failed` or `cancelled`),
`checkpoints/{best,last}.ckpt`, `lightning_logs/version_N/` (TensorBoard), `metrics.json`
(with `run`: run id, attempt and config hash), and for the classifiers
`confusion_matrix_{val,test}.png`. `metrics.json`: command, seed, `best_epoch` (0-based),
`last_epoch`, monitor and `monitor_best`, versions, accelerator, `machine` (task 2.3:
`device` from `trainer.strategy.root_device`, `gpu` = the CUDA device name or `null`,
`torch`, `cuda`, `cudnn`, `platform`, `processor`, `python`; ADR 0010: `fp32_precision`,
the effective `ieee` or `tf32`, `fp32_backends` and `compute_capability`; task A1:
`engine`, `lightning`), the data (prepared path,
source SHA-256, class counts per split from `meta.json`), the resolved `config`; the
classifiers add `val_acc_best` (micro accuracy on Validation at the best checkpoint: with
the monitor `accuracy_val`, the best validation accuracy over epochs, the metric of the
legacy tables), `test_acc_at_best`, `val_macro_f1`, `test_macro_f1` and, per split
(`evaluation.val|test`), the balanced accuracy, `recall_per_class` (by class name, `null`
for an absent class) and the normalised and count confusion matrices. They also write
`predictions_{val,test}.npz` (softmax probabilities, labels and stamp index of the selected
checkpoint, `protocol.predictions`), listed under `predictions` with the evaluated split.
Non-finite numbers are written as `null`.

**Resume** restores the epoch counter, optimizer, LR schedule and the EarlyStopping /
checkpoint state (tested: 2 + 2 epochs equal 4 in epoch, global step, lr and `wait_count`).
It does not restore the global random generators, so the later epochs' augmentations and
shuffling differ from an uninterrupted run.

**Schedules (task 2.2a).** `run_command` branches on `Settings.schedule`. `epochs` is the
above (EarlyStopping only if `early_stopping`). `carrasco2021` requires `train-supervised`
with `supervised_model: paper_stamp_classifier` and `image_size: 21` (and the paper model
requires the schedule; `ConfigError` otherwise). It builds `PaperClassifierModule` and a
`DeviceTensorDataModule` over `float_crops` (or `interim_uint8_crops` with
`paper_input: uint8`; the module from `paper_module(settings)`, which also passes the
2.5-models flags), with Train minus the stamps `protocol.leakage` finds in Validation or
Test (`exclude_leaked_train`, found on the uint8 stamps before training), and a Trainer
with `max_epochs=-1`, `max_steps`, `val_check_interval=val_every_n_steps`,
`check_val_every_n_epoch=None`, no sanity validation, a `HorizonCallback` on `monitor`
(mode from `MONITOR_MODES`) instead of EarlyStopping and the `ModelCheckpoint`s; it writes
`best.ckpt`, and `last.ckpt` after the validation of every `checkpoint_every_n_steps` steps
and at the end. The outputs and metrics are the same files. Every run's
metrics add `best_step` (the `global_step` stored in `best.ckpt`) and `last_step`;
`carrasco2021` adds `horizon` (the final one), `paper_data` (`input`, `crop`,
`nan_approximation`, `excluded_train`, `n_train`, `n_tiled`, and with C5 / C6
`class_augment`: `min_r2` and each class's share below it; with C7 `flux_rescale`: each
class's share of failed PSF fits) and `preset`
(`presets.preset_report`: `name`, `faithful`, `deviations`, `diagnostics`). A resumed
`carrasco2021` run is exact: the data stream is a function of the step, and the horizon
and best weights come back from `last.ckpt`.

**Timing (task 2.2c).** Every run's metrics hold `timing`: `fit_seconds` (wall clock from
the first step to the end of the fit loop, validation included), `steps` (the optimizer
steps of this fit, fewer than `last_step` after a resume) and `seconds_per_step`; `null`
when no step ran (a resume at the horizon).

## `reference` (task 1.7)

`src/astro_simclr/reference.py`, behind `compare-reference`. `REFERENCES` holds R1-R4
(plan, "Reference runs"): the accepted commands, the gate and, for the pass rule, the legacy
mean, sample std (ddof=1) and number of repeats. Each run directory's `metrics.json` is
checked (command, `supervised_model`, no duplicate seed) and `val_acc_best` aggregated
(mean, std with ddof=1, any n).

- **R1, R2, pass rule:** one run: `|x - m_old| <= max(2 s_old sqrt(1 + 1/n_old), 0.01)`;
  n >= 2: `|mean - m_old| <= max(2 sqrt(s_old²/n_old + s_new²/n), 0.01)`. The plan writes
  `/5`; the code uses the real repeat counts: R1 0.9496 ± 0.002191 (n=5,
  `summary_tuning_ce.csv` row 10), R2 0.902667 ± 0.009866 (**n=3**: the ResNet tuning ran 3
  trials per configuration).
- **R3, baseline:** the pretrain run and a linear-eval run at each of 1, 10 and 100 % are
  all required (otherwise FAIL, "incomplete"); each linear-eval run must name one of the
  given pretrain runs in its `pretrained` setting. The pretrain loss must be finite; every
  linear-eval run needs finite accuracies and a validation accuracy above the
  majority-class baseline of its own Validation split (class counts copied from
  `meta.json` into `metrics.json`).
- Every run's config is compared with the distinguishing settings of its reference config
  (`Reference.expected`, tested against `configs/reference/`); differences print WARN
  lines, so an R4 run given as R1 is flagged. A key missing from a run's config (written
  before the setting existed) reads as its `Settings` default, so the task 2.4 runs, which
  predate the 2.5 flags, do not warn about them.
- **R4:** report only. R1 also prints the Carrasco-Davis et al. 2021 figures as context.
- **CD21 (task 2.3), two-tier gate of the paper replication**, on the with-features runs'
  `test_acc_at_best` (the without-features runs, `with_features: false` or
  `feature_set: none`, are compared with 0.883 ± 0.006 and only reported; a
  `feature_set: position` run counts as with features and warns as a deviation). The runs must be `supervised_model: paper_stamp_classifier` (task 2.2a); a run
  whose config differs from the `carrasco2021` preset (`Reference.expected` is
  `presets.PRESETS["carrasco2021"]` without `with_features`; task 2.2c) gets a WARN line,
  so a diagnostic or shortened run is flagged. *Faithful*: `|mean - 0.941| <= 0.005` and the M1 oracle passed
  (`--oracle-report FILE`, a JSON whose top-level `passed` is a boolean; the contract for
  2.2b). *Acceptable*: `|mean - 0.941| <= 0.01`. Both need the mean SN test recall
  (`evaluation.test.recall_per_class.SN`) within ±0.04 of 0.87 and the mean `val_acc_best`
  within ±0.01 of 0.950. Defined on exactly 5 seeds: another count gives REPORT with the
  tier it would reach. The paper's σ is ddof=0; the runs' std is printed with ddof=1 and 0.
  Runs with and without features may share seeds (`Reference.split_by_features`); the
  result line is `CD21: PASS (faithful|acceptable)`. Constants: `CD21_*`.
- Every gate prints a WARN line when the runs' `machine_signature` differs. The signature
  is the device, GPU, torch, CUDA, the effective float32 precision (ADR 0010; a run
  without it reads as `ieee`) and the engine (task A1; a run without it reads as
  `lightning`). `compare-variants` and `bench-report` refuse such a mix unless
  `--allow-mixed-machines` is given, so a new-engine variant is compared only with a
  baseline rerun on the new engine.

## `protocol` (task 2.3; split runs and decisions in 2.5-infra; grouped splits; benchmarks in 3.2; label budgets in 5.4)

`src/astro_simclr/protocol/`: the code of the phase-2 evaluation protocol, and of the
phase-3 benchmarks (task 3.2); the rules and the power note are in
`docs/evaluation-protocol.md`, the design choices in ADR 0004 (phase 2) and ADR 0008
(phase 3). No torch import (except `probe`, which no command imports), so `make-splits`,
`leakage-audit`, `compare-variants`, `make-object-splits`, `bench-report`, `report-phase2`
and `make-label-budgets` start fast.

| Module | What it does |
|---|---|
| `leakage` | `find_duplicates(StampData)`: byte-identical uint8 stamps across every pair of splits (BLAKE2b-128), with both labels and whether the features match; `train_exclusions` (the Train indices to drop, from Train only), `train_keep_indices(n_train, excluded)` |
| `groups` | Phase 2: builds object groups from the prepared data (unlike task 3.2's `grouping`, which assigns given ids): `neighbour_pairs(ra, dec, radius_arcsec)` (every pair within the radius, inclusive; an exact dec-sorted window search with the haversine separation, numpy only, correct at the poles and across the RA wrap; refuses non-finite coordinates and `|dec| > 90`), `union_groups` (union-find, canonical ids: groups numbered by their smallest row), `position_groups` / `external_groups` (equal ids, e.g. `oid`) → `Grouping(method, radius_arcsec, group_of)` (`n_groups`, `multi()`), `split_positions(data, names)` (ra/dec de-standardised with M1's statistics), `grouping_for(data, names, radius_arcsec=1.5, group_ids=None)`, `load_group_ids(npz)` (one array per split, integers or strings, no pickles; returns the file's name and SHA-256) |
| `splits` | Format version 2 (grouped; version 1 still read). `make_splits(prepared_dir, seed=0, radius_arcsec=1.5, group_ids_file=None)`: groups the pool (Train ∪ Validation), then the grouped 250/250 screen of Validation (groups placed largest first into the half with room; a documented fallback trims both halves to a common per-class count), `screen_excluded_train` (the Train alerts grouped with Validation, out of the screen's training), and the class-stratified group 5-fold CV (greedy placement, largest group first, per-class spread at most the largest group's stamps of the class), each fold with an exact 100-per-class selection set drawn from whole groups outside the fold (the other stamps of a selection group are the fold's `unused`); leaked Train stamps are in no fold. At most 10 folds (`fold_of` is one digit per stamp). `write_splits` / `read_splits` (structure checks: sizes, disjointness, ranges, positive integers, the grouping block, and `check_group_integrity`: no group in two folds, in a fold's selection and training, or in both screen halves; `screen_excluded_train` consistent) / `verify_splits(splits, prepared_dir, group_ids_file=None)` (dataset identity by source and label SHA-256 and sizes, not the directory name; group integrity; the stored groups equal a fresh grouping; class balance; fold stratification within the grouped tolerance; `excluded_train` equal to a fresh leakage check). `ProtocolSplits.locate` maps pool indices to Train and Validation indices; `ProtocolSplits.screen_training_excluded()` |
| `audit` | `leakage_audit(prepared_dir, radius_arcsec=1.5, group_ids_file=None)` (CLI `leakage-audit`): groups all three splits together and reports the groups within each split and per split pair by class, the Validation and Test rows with a Train alert within the radius (or the same id), and the byte duplicates, as a JSON-ready dict; `format_audit` prints it |
| `predictions` | `Predictions(probabilities, labels, index, n_classes=5)` (task 3.2: `n_classes` generalises the column count; 5 by default, so phase 2 is unchanged; the benchmarks pass 6 and index by `candid`), `save_predictions` / `load_predictions` (`.npz`, `allow_pickle=False`), `check_paired`, `ensemble` (mean probabilities per stamp), `seed_mean_correct` |
| `stats` | `mcnemar` (exact, integer arithmetic), `paired_bootstrap` (percentile CI of variant − baseline, stratified by class, metric `accuracy` or `balanced_accuracy`, own `default_rng(seed)`, memory-bounded blocks), `paired_bootstrap_macro_f1(pred_baseline, pred_variant, labels, n_classes)` (task 3.2: the same paired, class-stratified resampling for macro-F1; each resample recounts every run's confusion matrix as one matrix product of item multiplicities with (run, true, predicted) indicator columns; observed = mean-over-runs macro-F1 difference, the report passes one row per side, the seed-ensemble argmax; identical sides give exactly [0, 0]; `BootstrapCI.metric` is now `IntervalMetric`), `hierarchical_paired_bootstrap(pred_baseline, pred_variant, labels, n_classes, metric="balanced_accuracy"\|"macro_f1", groups=None, reference_counts=None)` (task 5.4: `(R, N)` predictions paired by label subset; each resample draws the subsets with replacement, then the objects within class (an object's items together, one class per object), shared by all subsets and both arms; statistic = mean over drawn subsets of the paired difference; optional prior-reweighted macro-F1; `_hierarchical_draws` fixes the draw order, which the golden tests replay; they also enumerate a tiny case exactly), `minimum_detectable_effect(n, discordant= \| rate=)` (2.80·SE, paired or unpaired), `balanced_accuracy`, `adoption_decision` (superiority: `adopt` only if the CI is above 0), `non_inferiority_decision(interval, margin=0.005)` (`adopt` if the CI lower bound is above −margin, `worse` if the CI is below it; `check_margin` refuses margins outside (0, 0.05], a likely points-for-fraction mistake), `screen_decision` (`drop` at 4 points or more, never adopt) |
| `split_runs` | Task 2.5-infra. `SplitRequest(file, fold)` (`fold=None`: the screen); `open_split_run(request, prepared_dir)` reads the file, refuses a fold it does not have (`ConfigError`) and runs `verify_splits` on the prepared data (a wrong fingerprint is a `SplitsError`); `SplitRun.parts()` gives `RunParts` (training, selection, evaluation as sorted pool indices, and `evaluation_index`: pool indices for a fold, Validation indices for the screen) and checks that they are pairwise disjoint and hold no excluded stamp; `SplitRun.describe()` is `metrics.json["protocol"]` (mode, fold, splits file and SHA-256, part sizes, `grouping`); the screen trains on Train minus `ProtocolSplits.screen_training_excluded()`; `SplitRun.identity()` (splits SHA-256, mode, fold) is stored in every checkpoint and checked on resume. `file_sha256` |
| `compare` | `compare_variants(baseline_dirs, variant_dirs, stage)`: loads each run's `metrics.json` and predictions file, refuses repeated seeds, runs on different data (`data.source.sha256`: a normal run's `index` is `arange(N)`, so the data identity and the labels are what show two runs saw the same stamps), unpaired runs and mixed machines or float32 precisions (unless allowed), reports McNemar on the seed ensembles and the bootstrap, always as REPORT; it points split runs to `--splits`. `screen_comparison` / `cv_comparison(..., rule="superiority"\|"non-inferiority", margin)`: the decision helpers, which check that the predictions cover exactly the screen's decision half or each fold's evaluation part (so the CV bootstrap is paired by pool index); `CVResult` holds the per-fold and pooled balanced accuracies, the CI, McNemar on the seed ensembles and the decision. `compare_decision(baseline_dirs, variant_dirs, splits_file, rule, margin)` (CLI `compare-variants --splits`): loads split runs (`load_split_run`), checks the splits file's SHA-256, one mode, one data source and machine, the same (fold, seed) cells on both sides and every fold, then applies `cv_comparison` (verdict `ADOPT` / `NOT ADOPTED`) or `screen_comparison` (`KEEP` / `DROP`; `--rule` refused); `DecisionComparison.mcnemar` is the screen's McNemar on the seed ensembles (the CV's is `cv.mcnemar`) |
| `phase2_report` | Task 2.6. `build_report(ReportInputs)` → the markdown of `docs/results-phase2.md` (CLI `report-phase2`). `find_arms(dir)`: one arm per subdirectory, its runs at any depth (`DIR/<arm>/.../metrics.json`; a run directly under `DIR` is refused). Sections: the replication (per-seed table, mean ± std with ddof = 1, and `reference.compare_reference("CD21", ...)` verbatim with the M1 oracle report, default `DIR/m1_oracle.json`); the screen (`compare_decision` per arm against `--baseline-arm`, rot4 by default: accuracies, McNemar, KEEP/DROP and the verbatim output); the CV (`compare_decision` with each arm's pre-registered rule, superiority or `--non-inferiority`; balanced accuracy per run, per fold and pooled from `CVResult`, the CI, McNemar, both verdicts from `stats.adoption_decision` / `stats.non_inferiority_decision` on the same interval, the verbatim output; `pooled_class_scores` gives the per-class recall and confusion matrix of every arm from the per-stamp seed-mean scores, so the mean recall is the pooled balanced accuracy); then throughput (`read_run_facts`: s/step, GPU, the manifest's `concurrent_jobs`, flagged when > 1, "not recorded" without a manifest) and provenance (git commits and config hashes of manifest runs; with `--registry`, indexed and stale runs by `metrics.json` SHA-256). Deterministic: sorted arms and runs, a seeded bootstrap, paths as given, no timestamps. No statistics of its own |
| `alerts` | Task 3.2. `AlertTable` (sorted by `candid`: `oid`, `candid`, `jd`, `label`, `label_names`, `group`, `group_key` `"oid"` or `"group"`), `AlertTable.from_rows` (checks: valid ids, unique positive `candid`, finite positive `jd`, labels in range), `sha256` (canonical rows by `candid`, the group only with a `group` column), `rows_of(candids)`; `read_alert_table(csv, label_names=BENCHMARK_LABEL_NAMES)` (untrusted input: `csv` module and a schema, errors name the line; extra columns ignored), `write_alert_table`; `label_conflicts` (groups with two labels); `first_stamps(table, rows)` (earliest `jd`, then smaller `candid`, per `oid`). `BENCHMARK_LABEL_NAMES` = R5's classes (the phase-2 five), `RJ23_LABEL_NAMES` = the five + `Satellite` |
| `grouping` | Task 3.2, on per-row arrays of any group ids (never builds groups): `group_rows` (ids, label, size, conflicts), `assign_groups` (per class, `default_rng([seed, stream, class])` shuffle, then largest-deficit assignment; refuses a class that cannot give every part a group; task 3.5: an optional `allowed` mask, the most constrained groups placed first, each to the allowed part furthest below target), `grouped_split` (refuse or drop conflicting groups), `grouped_halves` (50/50 selection/decision, own stream), `check_stratified` (no group in two parts; per class and part, within `+g` / `−(k−1)g` of the target), `stratification_bound`, `shared_groups` / `assert_no_shared_groups(..., allowed=CROSS_BENCHMARK_ALLOWED)` (the cross-benchmark rule: by default the plan's matrix, Train/Train and Validation/Test pairs allowed, Train with the other's Validation or Test refused; `allowed=()` refuses all nine; returns the shared-count matrix) |
| `object_splits` | Task 3.2. `make_object_splits(table, seed, fractions=(0.6, 0.2, 0.2), on_conflict="refuse"\|"drop", allowed=None)`: `grouped_split` by `table.group` (task 3.5: `allowed` maps groups to the splits they may be in, stored as `constraints`, format version 2; version 1 still read; the bound is verified only for classes without constraints), then `grouped_halves` of Validation; `ObjectSplits` (table identity, group key, label names, seed, fractions, conflicts, sorted group ids per split and half, class counts per part), `split_of` / `half_of` / `rows(table, part)`; `write_object_splits` / `read_object_splits` (structure: valid sorted ids, disjoint splits and halves, halves inside Validation) / `verify_object_splits(splits, table)` (fingerprint, group key, label names, conflicts, coverage, the bound, the counts). CLI `make-object-splits` |
| `benchmark_metrics` | Task 3.2, numpy only: `confusion_counts`, `confusion_percent` (rows to 100; an empty row stays zeros), `macro_f1`, `macro_f1_from_counts` (stacked `(..., C, C)`), `reweighted_counts(counts, reference)` (prior reweighting), `benchmark_metrics(labels, predicted, label_names, reference_counts=None)` → `BenchmarkMetrics` (accuracy, macro P/R/F1, `ClassScores` per class, counts, percent, reweighted macro P and F1; `as_dict`). Zero-division rule: 0/0 → 0, and classes absent from labels and predictions are left out of the macro means (`None` per class), as torchmetrics. `RJ23_CLASS_COUNTS`, `rj23_reference(label_names)` (the default reference: only proportions matter) |
| `bench_report` | Task 3.2. `load_benchmark_run` (the run contract: `seed`, `machine`, `benchmark.splits_sha256`, `predictions.<stage>.file` with `index` = `candid`; aligned to the split's rows, which it must cover exactly with the table's labels), `benchmark_report(run_dirs, splits_file, alerts_file, baseline_dirs=(), stage="test", reference_counts=None)` → lines: per view (all alerts, first stamp) and side, macro-F1/P/R and accuracy mean ± std over seeds, the seed-ensemble macro-F1, the prior-reweighted macro-F1, per-class scores with alerts, objects and the recall MDE (on objects), the % confusion matrix; with a baseline, the ensembles' macro-F1 bootstrap CI and McNemar; always `REPORT`. CLI `bench-report` |
| `label_budgets` | Task 5.4. `BudgetPart(name, rows, objects, labels, selection)` (one training part as rows of a larger index space); `cv_parts(splits, pool_labels)` (a fold's training part; the object is the position group named by its smallest pool index in the part) and `object_split_parts(splits, table)` (R5 Train; the object is the table's group key); `cv_source` / `object_split_source` (parent name, SHA-256 and data fingerprint); `cv_parts_from_files` / `object_split_parts_from_files` (read and **verify** the parent on the data first) → `PartsSource`. `make_label_budgets(parts, source, label_names, object_id, budgets=(10, 100, 1000), n_repeats=20, selection_fraction=0.2, include_full=True, seed=0)`: per part and repeat, one seeded object order (`default_rng([seed, 5, part, repeat])`); per class, budget `b` = the first `b` objects having the class (nested budgets, exactly `b` per class); the selection set = the next objects of the class outside the subset, `floor(fraction × b)` capped to the smallest remainder over classes, parts and repeats (CD21: 2/20/83). `LabelBudgets`, `write_label_budgets` / `read_label_budgets` (structure) / `verify_label_budgets(budgets, parts, source)` (parent kind, SHA-256 and fingerprint, part names, per-class sizes and classes, selection outside the subset); `resolve(budgets, part)` → `BudgetSubset(part, repeat, budget, labelled, selection)` rows, 100 % last (the whole part and its own selection set). `BudgetError` is a `SplitsError` |
| `budget_parser` | Task 5.4: argparse only, handlers load on use. `make-label-budgets --out FILE (--splits FILE --data DIR \| --object-splits FILE --alerts CSV) [--budgets ...] [--repeats R] [--selection-fraction F] [--no-full] [--seed N] [--overwrite]` and `verify-label-budgets FILE` with the same parent options (exit 0 / 1) |
| `probe` | Task 5.4 (imports torch). `fit_probe(h, labels, n_classes, weight_decay=1e-3)` → `LinearProbe` (standardised features; mean CE on logits + `wd/2·‖W‖²`; full-batch L-BFGS, strong Wolfe, from zeros, on `h`'s device and dtype; deterministic); `run_probes(embeddings, labels, subsets, evaluation, n_classes, weight_decay=(1e-3,), reference_counts=None, device="cpu", dtype=float64)` → one `ProbeResult` per subset (chosen penalty, selection BA when a grid is searched on the subset's selection set, BA / macro-F1 / reweighted macro-F1 on the part's evaluation rows, the predictions); `paired_predictions(baseline, variant, part, budget)` (the `(R, N)` input of `hierarchical_paired_bootstrap`, paired by repeat); `summarise` (mean ± std over subsets). Same objective as the 5.0b pilot's probe |

- **Phase 3** (task 3.2): the benchmark modules above (`alerts`, `grouping`,
  `object_splits`, `benchmark_metrics`, `bench_report`) are built and tested on synthetic
  alert tables; the real ones come from task 3.1. No object-splits file is committed
  yet. Rules: `docs/evaluation-protocol.md`, "Phase 3 benchmark metrics"; choices: ADR 0008.
- **Stored splits:** `configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json` (format
  version 2, seed 0, 1.5", made with `make-splits` on the real data; the file of every 2.5
  decision) and `configs/splits/td_ztf_stamp_17_06_20_red-002.json` (version 1, not
  grouped; kept for the runs made on it). On `red`: one duplicate, Train 2995 (Bogus) =
  Validation 115 (SN); 49,717 pool groups (663 of two or more stamps); 107 Train alerts
  grouped with Validation leave the screen's training; folds of about 40,365 training /
  500 selection / 10,216 evaluation stamps, no unused stamps.
- **Stored label budgets** (task 5.4): `configs/label-budgets/td_ztf_stamp_17_06_20_red-002-grouped.json`
  (from the grouped CD21 splits on the real data, seed 0; 5 folds × 20 repeats × 10/100/1000
  objects per class plus 100 %; selection 2/20/83 per class; about 3.7 MB). Rules:
  `docs/evaluation-protocol.md`, "Phase 5 label budgets".
- Randomness: the splits and the bootstrap use their own seeded generators, independent of
  `seed_run`, so a training seed never moves a split and a comparison is reproducible on
  its own (an exception to the "Randomness" convention above, for data definitions and
  statistics only).
- **Split runs** (task 2.5-infra): `experiments.run_command(..., split=SplitRequest)`
  (`train-supervised --splits FILE --fold K | --screen`, `carrasco2021` schedule only)
  verifies the file first, then builds the device data from Train and Validation only
  (`device_data.protocol_splits`), trains on the training part, selects on the selection
  part, predicts the evaluation part as `predictions_evaluation.npz` and records
  `protocol`, `selection_acc_best`, `evaluation_accuracy` and
  `evaluation_balanced_accuracy` (no `val`/`test` entries). Before any step, a
  `carrasco2021` resume compares the identity stored in the checkpoint
  (`experiments._resume_identity`) with the run's own, so a fold never continues from
  another fold's weights. The default path, and so the
  faithful preset, is unchanged. The end-to-end commands are in
  `docs/evaluation-protocol.md`.

## `runs` (phase 4, task A0; ADR 0015)

`src/astro_simclr/runs/` defines the interfaces of run identity that the registry (A2), the
queue (A3) and the new engine build on. Nothing in it imports torch. `experiments.run_command`
calls `record.start_run` and `RunRecord.end` around the fit (task A1).

- **`canonical`**: `canonical_json(value)` writes sorted keys with no whitespace, ASCII
  escapes, `repr` floats, `-0.0` as `0.0` and ints distinct from floats, and refuses
  NaN and infinity. `semantic_config(command, settings)` returns
  `{"canonical_version": 1, "command", "settings"}`: the resolved settings (defaults
  materialised) without `NON_SEMANTIC_KEYS` (`accelerator`, `data_dir`, `output_dir`,
  `progress_bar_refresh_rate`, `checkpoint_every_n_steps`, each with its reason in the code)
  and `IDENTITY_KEYS` (`seed`, `pretrained`, which are run-id components). `config_sha256`
  is its SHA-256.
  - `HASH_NEUTRAL_DEFAULTS` leaves semantic fields out at one value: `num_workers: 0`
    (with workers, per-worker seeding changes the augmentation draws, so any other count is
    hashed), and fields added later at the value that keeps old hashes: `feature_set: all`
    (ADR 0016).
  - `semantic_config_from_plain(command, view, complete=True)` and `semantic_diff` are the
    hook of the nested config (4.2): `config_tree.config_sha256` lowers to the flat view
    and hashes it there (ADR 0017).
  - The pinned hashes of the `carrasco2021` presets, of ten 2.5 variants (`feature_set:
    position` and `none` among them) and of the two
    legacy protocols are in `tests/fixtures/runs/canonical_configs.json`; regenerate them with
    `uv run python tests/_canonical_cases.py`, and only on purpose.
- **`identity`**: `RunIdentity` holds the config hash, `seed`, `split` (a fold, `"screen"`,
  or `None`), `data_sha256`, `splits_sha256`, `pretrained_sha256`, `machine_class`, `engine`
  (`lightning` or `plain`) and
  `numerics_epoch`. `run_id` is the SHA-256 of its canonical JSON.
  - `run_identity(command, settings, ...)` builds it from a run's settings.
  - `NUMERICS_EPOCH` is the integer that a PR bumps when it changes what an existing config
    computes. The module docstring lists when; the PR template asks.
  - Helpers: `prepared_data_sha256` / `data_sha256` (from `meta.json`'s recorded array
    hashes, split layouts and `raw-alerts` pools alike; independent of `created_by` and of
    the format version), `uv_lock_sha256`,
    `git_state` (HEAD and `git status --porcelain`; `None` outside git), `split_part` (from
    a `SplitRequest`) and `default_machine_class` (`cuda-<gpu model>`, `cpu-<isa>`,
    `mps-<isa>`).
- **`manifest`**: `Manifest` is the versioned `manifest.json` (format
  `astro-simclr-run-manifest`, version 1). Its fields are:
  - run id and attempt;
  - command, the full config, the config hash and `canonical_version`;
  - the identity;
  - git `{sha, dirty}` and the `uv.lock` SHA-256;
  - `machine`: the `metrics.json` record, with `fp32_precision` (ADR 0010);
  - `engine` (the identity's), named `seeds` and `concurrent_jobs`;
  - `status` and the UTC timestamps.

  `new_manifest` starts it `queued`. `transition` allows `queued → running | cancelled`
  and `running → finished | failed | cancelled | queued` (an expired lease); only
  `finished` counts as finished (`is_finished`). `write_manifest` writes the file
  atomically. `read_manifest` checks the format, the exact keys (none repeated) and the
  types, the consistency of the status and timestamps and of `pretrained` with its
  SHA-256, and that the recorded `config_sha256` and `run_id` are what the content hashes
  to; any failure, a garbled file included, is a `ManifestError`.
- **`layout`**: `run_dir_name(run_id, attempt)` is `<id[:16]>-a<attempt>`. The files of an
  attempt are `manifest.json`, `metrics.json`, `scalars.jsonl`, `tb/`,
  `predictions_*.npz` and `ckpt/{last,best}.pt`; Lightning runs keep
  `checkpoints/{last,best}.ckpt` and `lightning_logs/`. `classify_artefact(path)` gives a
  file's kind. `retention(manifest, paths)` is pure and decides keep or delete, with a
  reason, for each path:
  - an unfinished attempt keeps everything;
  - a finished one deletes `last`, and keeps `best` only for `pretrain`;
  - records, results, logs, figures and unknown files are kept.
- **`record`** (task A1): `start_run(command, settings, data=..., machine=..., split=...,
  pretrained_file=..., resumed=...)` computes the identity, with the machine class from
  the Trainer's device and `engine: lightning`, and writes `manifest.json` as `queued` then
  `running`. `RunRecord.end(status)` writes `finished`, `failed` or `cancelled`, once.
  Attempts, since `--output-dir` keeps only the latest attempt:
  - none, or a manifest of another run id: attempt 1;
  - `--resume` of an attempt left `running` (killed): the same attempt, through `queued`
    (like an expired lease);
  - the same run id otherwise: the next attempt.

  An unreadable manifest is a `ConfigError`, never overwritten. The git state and the
  `uv.lock` SHA-256 come from the source checkout (`source_root()`, `None` in an installed
  wheel). `concurrent_jobs` comes from `ASTRO_SIMCLR_CONCURRENT_JOBS` (default 1).

## `registry` (phase 4, task A2; ADR 0024)

`src/astro_simclr/registry/` is the local run registry: a derived SQLite index of run
directories at `$ASTRO_SIMCLR_HOME/registry.sqlite` (default `~/.astro-simclr`), rebuildable
from them. Nothing in it imports torch; `parser.py` imports only argparse, so `--help` stays
fast and the handlers (`commands.py`) load on use.

- **`home`**: `astro_simclr_home()`, `registry_path()` (refuses a path inside a git working
  tree), `specs_dir()` (`$ASTRO_SIMCLR_HOME/specs`, where A3's queue writes accepted specs).
  `RegistryError` is the package's error (exit 1).
- **`readers`**: `read_run_dir(path)` returns an `IndexedRun`:
  - from `manifest.json` (validated by `runs.manifest.read_manifest`), plus the metrics of
    `metrics.json` once finished;
  - or, backfilled from `metrics.json` alone, `legacy: true`, with null run id, attempt,
    data / pretrained SHA-256, git, `uv.lock` and `numerics_epoch`; `status: finished`,
    `engine: lightning` and the machine class derived (ADR 0024 rule 4).

  The config is read by the reader of its `config_format` (`CONFIG_FORMATS`; `flat-1` when
  the file names none): `lower_flat_1` fills the fields added since with `FLAT_1_FILLS`
  (each the pre-field behaviour, with its reason; fields added after A0 from
  `canonical.HASH_NEUTRAL_DEFAULTS`) and refuses unknown or unfillable keys, so
  the archived runs hash like today's presets. `metrics_summary` keeps
  `<stage>.{accuracy,balanced_accuracy,macro_f1}`, `monitor_best`, `best_step`,
  `last_step` and `seconds_per_step`. `IndexedRun.key` is `<run id[:16]>-a<attempt>` or
  `legacy-<metrics SHA[:16]>`; copies of an attempt share it.
- **`store`**: `open_registry(path)` creates and migrates the file (`SCHEMA_VERSION` in
  `PRAGMA user_version`; `MIGRATIONS` append-only; a newer file is refused). Tables:
  `roots`, `runs` (one row per run directory), `artefacts` (path, kind, size, mtime,
  SHA-256; deleted with their run) and `specs` (id, name, file).
- **`index`**: `reindex(conn, roots, specs=..., rebuild=False)` walks the roots (no
  symlinks, no hidden directories, not into a run directory), replaces each root's rows in
  one transaction, reports skipped directories and missing remembered roots, rehashes only
  changed files, then indexes the spec files of `specs_dir()` through A3's `load_spec`.
- **`spec_view`**: `SpecLike`, the `Protocol` of A3's `runs.spec.ExperimentSpec` that the
  registry reads; `load_spec_file` (a lazy import of `runs.spec`: `SpecUnavailableError`
  without A3); `ComparisonPlan` and `PlannedCell`, the minimal frozen dataclasses a
  comparison runs on (`plan_from_spec` builds one and hashes the spec's splits file).
- **`compare`**: `resolve(plan, records, use=..., attempts="error"|"latest",
  allow_mixed=...)` maps each cell to one finished attempt, or raises `ComparisonError`
  with every problem: a missing cell, several attempts without a choice, copies that
  differ, mixed machine signatures / engines / numerics epochs (allowed per kind), or
  different data (never allowed). Matching runs outside the cells are counted.
  `format_report` / `resolution_to_plain` print the cells and the mean ± std (ddof=1) per
  arm; a report, never a decision (`compare-variants` decides).
- **`parser`** / **`commands`**: `astro-simclr runs reindex [ROOT ...] [--rebuild]`,
  `runs ls [filters] [--json]`, `runs show KEY [--json]`, `runs compare SPEC [--metric M]
  [--use KEY] [--attempts latest] [--allow-mixed KIND] [--json]`; `--registry FILE` on
  each.

## `simclr` (phase 5, task 5.0b; ADR 0025)

`src/astro_simclr/simclr/`: SimCLR for the pilot that gates the corpus download (plan
5.0b). **Engine-free**: nothing imports Lightning (checked in a fresh interpreter), every
draw is keyed by `step`, so 5.3 can run the same functions on the plain engine.

| Module | Contents |
|---|---|
| `views` | `make_views(images, params, config, *, fits, generator)`: two training views (B, C, 57, 57) of (B, C, 63, 63) stamps: optional C5 noise on the 63-pixel stamp (`background_noise`: science and template noised, the difference through the fit, the min-max again), then `dihedral(images, k)` (8 exact elements: `rot90` by `k % 4`, flip if `k >= 4`), then `crop(images, offsets)` (an integer 57-pixel window, offsets 0-6, the same for every channel, no fill). `centre_crop` is the evaluation view (offset 3). `draw_view_params(seed, step, B)`: `k`, offsets and noise strengths of both views from `default_rng([seed, step, simclr_views])`, always drawn in full; `noise_generator(seed, step, device)`: the noise fields' device generator. |
| `network` | `resnet18_cifar()`: torchvision ResNet-18, 3x3 stride-1 stem, no max-pool, `fc = Identity`, `zero_init_residual=True` (57 -> 57 -> 29 -> 15 -> 8, `h` 512-d); `ProjectionHead`: Linear(512, 512, no bias)-BN-ReLU-Linear(512, 128); `SimCLRNetwork(x) -> (h, z)`. |
| `objective` | `SimCLRLoss(T)(z_i, z_j)`: `losses.NT_Xent` in paper mode (ADR 0002), in float32 with autocast off; returns `(loss, pos, neg)`. |
| `optim` | `lars_param_groups(model)`: convolution/linear weights (weight decay 1e-6, trust ratio) and an excluded group (normalisation parameters and biases: no decay, no trust ratio, `optim.EXCLUDE_KEY`); `build_lars(model, lr=...)`: fixed trust ratio, no Nesterov, momentum 0.9; `base_learning_rate(B) = 0.3 * B / 256`; `warmup_cosine_lr(step, base_lr, total_steps, warmup_steps, lr_min=0)`: linear warmup to `base_lr` at step `W - 1`, then a cosine that is exactly `lr_min` at `total_steps - 1`; `set_learning_rate`. |
| `diagnostics` | `effective_rank(h)` (exp of the entropy of the normalised eigenvalues of `cov(h)`), `dimension_std(z)`, `alignment(z_i, z_j)`, `uniformity(z, t=2)`, `knn_predict` / `knn_balanced_accuracy` (cosine, `k = 20`, vote weights `exp(sim / 0.07)`). |
| `probe` | `draw_label_subsets(labels, candidates, per_class, n_subsets, seed)` (nested across budgets: subset `r` is the head of a per-class permutation keyed `[seed, r, class]`), `write_label_subsets` / `read_label_subsets` (format `astro-simclr-label-subsets` v1, with the SHA-256 of the candidates and the labels); `fit_probe(h, labels, weight_decay)`: logistic regression, standardised `h`, plain cross-entropy on logits, L2 on the weights, full-batch L-BFGS in float64 on the CPU; `probe_screen(...)`: the penalty chosen on the screen's selection half from `WEIGHT_DECAY_GRID`, the score on the decision half, per subset. |
| `pilot` | `PilotConfig` (the run's whole definition; `epochs` or `steps`), `load_pilot_data` (Train and Validation as float32 (N, 3, 63, 63); with a splits file, verified, pretraining leaves out `screen_training_excluded`), `BatchOrder` (a permutation per pass keyed `[seed, pass, simclr_batch_order]`, last partial batch dropped), `PilotRun` (model built on the CPU from the seed, LARS, the step: views, one forward of both views concatenated, loss, rate, backward, step), `diagnose`, `embed`, `dump_embeddings`, `train_pilot(config, out_dir, resume=, stop_at=)`. |
| `commands` | The `simclr-pilot`, `simclr-embed`, `make-label-subsets` and `simclr-probe` subcommands (see `cli`). |

A pilot run directory holds `pilot.json` (config, data record with the source and splits
SHA-256, schedule, device, precision and determinism flags, versions), `metrics.jsonl`
(loss, mean positive and negative cosine, rate), `diagnostics.jsonl`, `ckpt/last.pt`,
`ckpt/step-NNNNNNN.pt` every `keep_every_epochs` passes, `ckpt/final.pt` (format
`astro-simclr-pilot-checkpoint` v1, `weights_only=True`) and `embeddings/` (`train_h.npy`,
`validation_h.npy`, the labels and `meta.json`). `--resume` restores the weights, the
optimizer and the steps done from `last.pt` (same config only) and drops the log lines past
it; the continued run takes the same batches and views (exact on CPU, and on CUDA with
`--deterministic`).

The pilot's protocol, end to end:

```bash
D=dataset/prepared/td_ztf_stamp_17_06_20_red-002; S=configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json
astro-simclr simclr-pilot $D --splits $S --output-dir runs/pilot/ssl --epochs 100 --device cuda
astro-simclr simclr-embed $D --random-init --seed 0 --output-dir runs/pilot/random-init
astro-simclr make-label-subsets $D --splits $S --per-class 10 --n-subsets 20 --out runs/pilot/subsets-10.json
astro-simclr simclr-probe runs/pilot/ssl/embeddings --subsets runs/pilot/subsets-10.json --splits $S --out runs/pilot/probe-ssl-10.json
astro-simclr simclr-probe runs/pilot/random-init --subsets runs/pilot/subsets-10.json --splits $S --out runs/pilot/probe-random-10.json
```

## `preflight` (task 1.7)

`src/astro_simclr/preflight.py`, behind `astro-simclr preflight` and
`scripts/gpu-preflight.sh`. Each step prints PASS, FAIL, WARN or SKIP; a failed step 1, or
a failed step 2-4, stops it.

1. **CUDA:** `nvidia-smi` (driver, compute capability; injectable for tests), looked up on
   PATH and then at `/usr/lib/wsl/lib/nvidia-smi` (WSL2 shells that are not login shells,
   such as ssh, lack it on PATH). Without it but with CUDA in torch, the GPU comes from
   `torch.cuda.get_device_name/get_device_capability`, the driver checks are skipped and
   the step is a WARN (with a WSL2 hint) →
   `choose_torch_build`, the table of ADR 0001 rule 6 (capability < 7.5 → `cu126`;
   Blackwell → default build with driver >= 580; else driver >= 580 → default, below →
   `cu126`; `cu126` needs >= 525; with the driver unknown, by capability only). Then the
   installed build: `torch.version.cuda` set, a
   driver new enough for it, `torch.cuda.is_available()`, and `arch_supported(capability,
   torch.cuda.get_arch_list())` (same major and a lower or equal minor; `sm_XYa`
   arch-specific code only on that exact GPU). A mismatch prints the exact `uv sync`
   command. With
   `--accelerator` other than `cuda` the step is a SKIP that still checks the accelerator.
2. **Resources** (first, from `meta.json` and file sizes only, before any array is read):
   RAM (`MemAvailable`, or total memory where that is all the OS reports)
   >= the largest prepared dataset's arrays (from `meta.json`) + 2 GiB; 2 GiB free disk
   under the output directory. Pickles in the data directory only warn. Steps 2, 3 and 4
   run in order and stop at the first FAIL.
3. **Data:** `inspect-data` (`summarize`) on every prepared directory: schema, `meta.json`,
   hashes.
4. **Manifest:** `data-manifest.json` in the data directory (per dataset: kind, source,
   per-split counts and file SHA-256s). Written on the first run; later runs fail on a
   missing or changed dataset and warn on a new one. Never written if step 3 failed. A
   changed dataset lists what changed (for example `Train/images_f32.npy: new (prepared
   with --images float32)`, task 2.1), and the step prints the command that re-creates the
   manifest (`rm DIR/data-manifest.json && astro-simclr preflight --data-dir DIR`). Step 2
   sizes each array by its dtype (float32: 4 bytes).
5. **Dry chain:** the configs of `configs/reference/` (R3 pretrain → R3 linear-eval and
   fine-tune on that checkpoint → R1, R2, R4; `--runs` selects) through `run_command`, with
   `max_epochs=1`, `limit_train_batches=steps+1`, `limit_val_batches=2`, at the reference
   batch sizes (capped to the Train size), on the chosen accelerator, so a batch that does
   not fit the GPU fails here. Then **CD21** (task 4.6): the `carrasco2021` preset on the
   **plain engine** (`tasks.plain.run_plain`), `steps+1` steps with one validation after the
   last and no horizon extension, on `red-002` with float32 stamps (`--synthetic` prepares
   that file with them). R1-R4 stay on Lightning (their modules are frozen at
   `lightning-final`); `--runs CD21` alone never imports Lightning (tested in a fresh
   interpreter). Without `--runs`, a run whose dataset is not prepared is a SKIP (WARN); a
   run named in `--runs` must run, or the step FAILs.
6. **Throughput:** `ThroughputTimer` (engine-free: `tick(device)` after each step,
   `reset()`; `lightning_timer(timer)` adapts it to a Lightning callback that resets it at
   each epoch start, the plain engine ticks it through `fit(on_step=...)`) measures steps/s
   between consecutive steps (the first untimed; the device is synchronised so the clock
   measures work), with the real data path. The estimate per run is
   `(n_train // batch) * max_epochs / rate` (CD21: the initial horizon, 30,000 steps, a
   lower bound), times 5 seeds (R1, R2, CD21) for the full set, with R3's linear evaluation summed over the 1,
   10 and 100 % files; an upper bound, without validation time. With `--synthetic` (the
   four synthetic files, named as the received ones, `--n-per-split` 2400 by default,
   under the output directory) it
   uses the real Train counts of `docs/data.md` (`REAL_TRAIN_COUNTS`).
