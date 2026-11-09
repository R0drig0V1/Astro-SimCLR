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

**Lightning.** Until task 4.8 the package trained on PyTorch Lightning 2 (`experiments`,
the `training` modules, data modules and callbacks, the PIL `augmentations` and the
`data.datasets` they fed). Task 4.8 removed it once the 4.7 gate had held the plain engine
bit for bit equal to it: nothing in `src/` imports Lightning, torchmetrics or TensorBoard,
and none of them is a dependency. The last commit with Lightning is the git tag
`lightning-final`. The legacy commands (`pretrain`, `linear-eval`, `fine-tune`), the stamp
and ResNet baselines, the R1-R4 dry runs and every Lightning run are re-runnable from a
checkout of it (`git worktree add ../astro-simclr-lf lightning-final`, then `uv sync
--locked` there), and this page's sections on the removed modules are readable there
(`git show lightning-final:docs/architecture.md`). Runs that Lightning wrote stay readable
here (sections `runs`, `registry`, `tta and ensemble_eval`). The one generator that needs
Lightning, `scripts/characterize/lightning_run/generate.py` (the stored Lightning run of
`tests/fixtures/lightning_run/`), runs from a checkout of `lightning-final`; the
characterization fixture of the removed `augmentations` module and its generator went
with it.

```
cli ──► config, config_tree, hparams (legacy key mapping), presets (carrasco2021, rj23-*)
 │
 ├──► data (pickle loader, prepared and raw-alert layouts, float and RJ23 input paths)
 ├──► tasks.plain (run, train-supervised, spec run-cell: the training runner) ──► engine,
 │      tasks.paper (the paper classifier), tasks.rj23 (RJ23's), tasks.results (metrics.json)
 ├──► reference (compare-reference: the gates of R1-R4 and CD21)
 ├──► protocol (make-splits, compare-variants: the phase-2 evaluation protocol) ──► data
 ├──► ensemble_eval (ensemble-eval: seed ensembles and TTA of stored split runs; D4) ──► tta,
 │      tasks.paper (PaperNet, paper_crops), engine (open_eval, DevicePolicy), runs, protocol
 ├──► oracle (fetch-m1, m1-oracle: the published M1 vs our input path) ──► paper_model, data
 ├──► alerts (fetch-alerts, alerts-summary: raw alerts from ALeRCE) ──► data (prepared, m1_features)
 ├──► r5 (r5-sources, build-r5: the R5 benchmark) ──► alerts (by_oid, store), protocol, data.raw_alerts
 │    (tasks.paper ──► paper_variants: rot4 / single / p4 / p4m ──► paper_model, group_conv;
 │     paper_variants ──► convnext_model (B8), hybrid_model (B9) ──► paper_variants.PaperHead;
 │     paper_model ──► feature_sets: all / position / none;
 │     tasks.paper ──► training: paper_augment, class_augment, flux_rescale, ema,
 │     lr_schedule, tensor_splits, optimizers, metrics)
 ├──► preflight ──► tasks.plain (the CD21 dry run), data (inspection, manifest)
 └──► engine (phase 4: the contracts, the loop, the data tiers) ──► config, data, runs
 └──► runs (phase 4, A0: run identity, manifest.json, layout, retention) ──► config, data
 └──► queue (phase 4, A3: specs, the durable queue, ssh) ──► runs.spec, tasks.plain
 └──► registry (phase 4, A2: runs reindex | ls | show | compare; SQLite) ──► runs, reference
 └──► simclr (phase 5, 5.0b: the SimCLR pilot) ──► engine.randomness,
      optim (LARS), losses (NT_Xent), stamp_relation, protocol (splits, stats), data

cli train-supervised --preset rj23-* (or run with an rj23 config) ──► tasks.plain ──► tasks.rj23
   ──► rj23_regime (schedule policy, class-uniform sampler, Keras Adam: engine-free)
   ──► rj23_data (parts from a raw-alerts layout + object splits) ──► data.rj23_input, protocol
   ──► rj23_model (engine-free nn.Modules) ──► data.rj23_input
```

Conventions shared by every module:

- **Settings** come from `config.Settings`, never from module-level constants.
- **Randomness:** the global streams come from `engine.randomness.seed_everything(seed)`
  (the three calls of `lightning.seed_everything`, so a seed builds the initial weights a
  Lightning run of that seed built); per-step draws (augmentations) come from generators
  keyed by `(seed, step, stream)` (ADR 0006, ADR 0014: `engine.randomness`), never from an
  unseeded or global generator.
- **Devices:** no module hard-codes CUDA. The accelerator comes from the settings, and tensors
  follow their module's device.
- **Behaviour changes** against the legacy code are listed in the section of the module that
  makes them, with the legacy flag that restores the old behaviour, if any.
- **Typing:** mypy strict; `# type: ignore[override]` is allowed only on Lightning hook
  overrides (CLAUDE.md). No Lightning hook remains since task 4.8; the one such ignore left
  in `src/` is `rj23_regime.KerasAdam.step`.

## Environment and torch wheels

Runtime dependencies: torch, torchvision, numpy, pyyaml, matplotlib, fastavro, and timm (the
ConvNeXt rows; it brings huggingface-hub and safetensors, and resolves against each fork's
torch: [ADR 0022](decisions/0022-convnext-native-stamp.md)). lightning, torchmetrics and
tensorboard were dropped in task 4.8 (the relock only removed packages; the torch pins and
extras are unchanged). torch and torchvision come from PyPI by default (MPS on macOS,
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
  every run (`engine.precision.apply_fp32_precision`) and recorded, with its effect, in
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

  **Since task 4.8** the commands and models that read most of these fields train only at
  the git tag `lightning-final`: `pretrain`, `linear-eval`, `fine-tune` and the
  `stamp_classifier` / ResNet baselines of `train-supervised`, with the `epochs` schedule.
  The fields stay (they parse, resolve and hash as before, so stored runs keep their config
  hash), but the plain runner trains only the paper classifier on `carrasco2021` and RJ23's
  on `rj23`. It refuses `devices` other than 1 and `limit_train_batches` /
  `limit_val_batches` (Lightning's dry-run limits; a short plain run sets `max_steps`,
  `val_every_n_steps` and `horizon_increment`), and reads neither `check_finite`,
  `divergence_threshold`, `progress_bar_refresh_rate` nor `augmentation`.
- **Task 2.2a fields** (the `carrasco2021` schedule; see `training`, "The carrasco2021
  schedule"):

  | Field | Default | Meaning |
  |---|---|---|
  | `schedule` | `epochs` | `epochs` (the legacy protocol; trains only at `lightning-final`) or `carrasco2021` (ALeRCE's iteration schedule) |
  | `max_steps` | 30000 | `carrasco2021`: initial horizon in optimizer steps (`engine.policies.Horizon`) |
  | `val_every_n_steps` | 10 | `carrasco2021`: validation and selection interval |
  | `horizon_increment` | 10000 | `carrasco2021`: a new best at step `s` raises the horizon to `s + increment` |
  | `checkpoint_every_n_steps` | 1000 | `carrasco2021`: `last` checkpoint interval (`ckpt/last.pt`), a multiple of `val_every_n_steps` |
  | `early_stopping` | true | `epochs`: false drops EarlyStopping; `carrasco2021` requires false |
  | `dropout_active` | false | `paper_stamp_classifier`: apply its dropout layers (rate `drop_rate`) |

  Under `carrasco2021` the schedule's definitions are errors to change, not defaults:
  `eval_batch_size` must be 64 (the selection loss is TF's mean of batch-64 means),
  `early_stopping` false and `num_workers` 0 (`ConfigError` otherwise). The
  `carrasco2021` preset (task 2.2c, see `presets`) sets them.
- **Task 3.4b fields** (the `rj23` schedule; see `training`, "The rj23 schedule"; ADR 0023).
  `schedule: rj23` reuses `max_steps` (the hard cap), `val_every_n_steps` (the validation
  interval, step 0 included) and `patience` (in validations), and adds:

  | Field | Default | Meaning |
  |---|---|---|
  | `rj23_scenario` | `full` | the input and its Table 2 classifier: `full`, `cropped`, `lowres`, `multiscale` (`RJ23_SCENARIOS`) |
  | `rj23_bn_position` | `after_concat` | ADR 0013's BN placement; `before_concat` is the sensitivity arm |
  | `rj23_keep` | `stop` | whose predictions are primary: the weights at the stop (RJ23's) or the best validation's; both are written |
  | `rj23_large_values` | `paper` | `rj23_input.normalise(large_values=...)` (ADR 0009) |

  Under `rj23`, `supervised_model: rj23`, `with_features: false`, `monitor: macro_f1_val`,
  `early_stopping: false`, `num_workers: 0`, `patience >= 1`, and `max_steps` and
  `checkpoint_every_n_steps` multiples of `val_every_n_steps` are required; the `rj23_*`
  fields changed, or `supervised_model: rj23`, under another schedule are errors
  (`ConfigError`). The `rj23-<scenario>` presets set them.
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
  | `paper_arch` | `rot4` | `rot4` (faithful: 4 rotations, mean after fc1), `single` (one pass, not invariant), `p4` (B7), `p4m` (B2'), `convnext_atto` / `convnext_pico` (B8), `hybrid` / `hybrid_p4` (B9) |
  | `group_fields` | null | `p4` / `p4m` / `hybrid_p4`: fields of conv1-conv2 (conv3-conv5 twice); null is `32 // \|G\|` (8 / 4). An error with the other archs |
  | `hybrid_dim` | 64 | `hybrid` / `hybrid_p4` (B9): token width of the attention blocks, a multiple of `hybrid_heads`. Another value is an error with the other archs (all four `hybrid_*` fields) |
  | `hybrid_depth` | 2 | the hybrids: pre-norm transformer encoder blocks (≥ 1) |
  | `hybrid_heads` | 4 | the hybrids: attention heads per block (≥ 1) |
  | `hybrid_readout` | `mean` | the hybrids: `mean` of the 36 grid tokens, or a learned `cls` token |
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

  `tasks.common.check_paper_variant` rejects a changed one with another `supervised_model`
  (`ConfigError`).
- Plan row D3 (`training.ema`, `training.lr_schedule`; section `training`, "Weight EMA,
  learning-rate schedule and the final weights"), all off by default and hash-neutral
  there (`runs.canonical.HASH_NEUTRAL_DEFAULTS`), so every existing config keeps its
  numbers and its config hash. They need `schedule: carrasco2021` (`ConfigError`
  otherwise):

  | Field | Default | Meaning |
  |---|---|---|
  | `checkpoint_selection` | `best_val_loss` | the scored weights: the best validation loss's (CD21) or `final`, the last step's (the EMA's with `ema_decay`); `final` needs `horizon_increment: 0` |
  | `lr_schedule` | `constant` | CD21's constant rate, or `cosine` to 0 over exactly `max_steps` (needs `horizon_increment: 0`) |
  | `lr_warmup_steps` | 0 | linear warmup steps, fewer than `max_steps` (needs `horizon_increment: 0`) |
  | `ema_decay` | `null` | the weight EMA's decay in (0, 1), e.g. 0.999; validation, selection and scoring then use the EMA weights |
  | `ema_warmup` | false | ramp the decay as `min(ema_decay, (1 + step) / (10 + step))`; needs `ema_decay` |
- `load_settings(path, overrides, defaults=...)`: `defaults` (a subcommand's legacy
  protocol) sits between the dataclass defaults and the file. `read_settings_file` reads a
  YAML mapping without checking keys; `settings_to_plain` gives the plain (YAML/JSON) form
  that `show-config` prints and the metrics JSON records.

## `config_tree` (phase 4, task 4.2; ADR 0017)

`src/astro_simclr/config_tree/`: the nested, typed run configuration that `astro-simclr run`
reads. It lowers to the flat `Settings` (what the runner, `tasks.plain.run_plain`, executes,
and the view the config hash is defined on, ADR 0015 rule 10), so it changes no numbers.

- **`schema`**: `Config`, a frozen dataclass of eight sections. `run` (`seed`,
  `output_dir`), `runtime` (`engine`, `accelerator`, `devices`, `num_workers`,
  `deterministic`, `fp32_precision`, `progress_bar_refresh_rate`), `data` (`dir`,
  `dataset`, `image_size`, `batch_size`, `with_features`) and `eval` (`batch_size`,
  `limit_batches`) are plain; four are tagged unions on `kind`:

  | Section | `kind` | Fields (flat name when it differs) |
  |---|---|---|
  | `model` | `stamp_classifier` | `drop_rate`, `beta_loss` |
  | | `resnet` | `name` (`supervised_model`: `resnet18/34/50/152`), `beta_loss` |
  | | `paper_stamp_classifier` | `arch` (`paper_arch`), `backbone_weights`, `group_fields`, `hybrid_dim`, `hybrid_depth`, `hybrid_heads`, `hybrid_readout`, `context`, `feature_set`, `input` (`paper_input`), `nan_approximation`, `exclude_leaked_train`, `dropout_active`, `drop_rate`, `beta_loss` |
  | | `rj23` | `scenario` (`rj23_scenario`), `bn_position` (`rj23_bn_position`), `large_values` (`rj23_large_values`) (task 3.4b) |
  | | `simclr` | `encoder` (`encoder_name`), `projection_dim`, `temperature`, `legacy_denominator` (`nt_xent_legacy_denominator`) |
  | | `linear_eval` | `pretrained`, `beta_loss` |
  | | `fine_tune` | `pretrained`, `beta_loss`, `drop_rate`, `lr_factor` (`fine_tune_lr_factor`) |
  | `optim` | `Adam`, `AdamW` | `lr`, `betas` (`adam_betas`) |
  | | `SGD` | `lr` |
  | | `LARS` | `lr`, `legacy_trust_ratio` (`lars_legacy_trust_ratio`) |
  | `schedule` | `epochs` | `max_epochs`, `monitor`, `early_stopping`, `patience`, `min_delta`, `check_finite`, `divergence_threshold`, `limit_train_batches` |
  | | `carrasco2021` | `max_steps`, `val_every_n_steps`, `horizon_increment`, `checkpoint_every_n_steps`, `monitor`, `limit_train_batches`, and plan row D3's `selection` (`checkpoint_selection`), `lr_schedule`, `lr_warmup_steps`, `ema_decay`, `ema_warmup` (lowers `early_stopping: false`) |
  | | `rj23` | `max_steps` (the cap), `val_every_n_steps`, `patience`, `checkpoint_every_n_steps`, `monitor`, `keep` (`rj23_keep`) (lowers `early_stopping: false`; task 3.4b) |
  | `augment` | `transforms` | `name` (`augmentation`) |
  | | `paper` | `rot90` (`rot90_augment`), `noise` (`aug_noise`), `feature_jitter`, `synthetic_bogus`, `min_r2` (`aug_min_r2`), `flux_rescale`, `flux_k_min`, `flux_k_max` |
  | | `none` | no fields: RJ23's model averages its own 8 rotation/flip views (task 3.4b) |

  - The model decides the command (`Config.command`). Every default is the default of the
    flat field (`Config()` lowers to `Settings()`). `runtime.engine` is `plain` (`ENGINES`;
    the default and the only value since task 4.8): `astro-simclr run` trains with
    `tasks.plain.run_plain`, the paper classifier on `carrasco2021` and RJ23's on `rj23`.
    `lightning` (the default until then) is refused with a `ConfigError` naming the git tag
    `lightning-final`. It is the run id's engine, not hashed. The other model variants
    (`stamp_classifier`, `resnet`, `simclr`, `linear_eval`, `fine_tune`) and the `epochs` /
    `transforms` sections still resolve, lower and hash (`--show` works), but they train
    only at the tag.
  - A `Config` is valid by construction: the tags, `paper_stamp_classifier` ⇔
    `carrasco2021` schedule ⇔ `paper` augmentation, `rj23` model ⇔ `rj23` schedule ⇔
    `none` augmentation, then the lowered `Settings`' checks,
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
  `carrasco2021-no-features`, `rj23-full|cropped|lowres|multiscale` (task 3.4b) and
  `pretrain-legacy`, `linear-eval-legacy`,
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

## `cli` (task 1.1; subcommands from 1.2, 1.7, 3.1, 3.5, 5.1a and A3; `run` from 4.2)

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
  one with `--command`), lower it and train it on the plain engine (`tasks.plain.run_plain`;
  `runtime.engine: lightning` exits 2 naming the git tag `lightning-final`).
  `--set` takes `section.key=value` (an unknown path exits 2) and flat `key=value` items.
  `--show` prints `# command:`, `# config hash:` and the resolved config as YAML, and
  does not train.
- **Training (1.7; plain only since 4.8):** `train-supervised`, the flat alias of `run`. It
  takes `--config`, `--set`, `--hparams FILE` (a legacy `results/hparams_best_*.yaml`, see
  `hparams`), `--data-dir`, `--output-dir`, `--seed`, `--accelerator`, `--resume
  PATH|auto` (`auto`: `OUTPUT_DIR/ckpt/last.pt` if any), `--preset
  carrasco2021|carrasco2021-no-features|rj23-<scenario>` (tasks 2.2c and 3.4b, see
  `presets`) and `--splits FILE --fold K | --screen` (task 2.5-infra, a split run; see
  `protocol`; `--fold`/`--screen` without `--splits`, or the reverse, exit 2). It trains
  the paper classifier on `carrasco2021` (`tasks.plain.run_plain`) or, with `--preset
  rj23-<scenario>` (or any `schedule: rj23`; task 3.4b), RJ23's classifier
  (`tasks.rj23.run_rj23_plain` through `run_plain`): `--splits` is then the object-splits
  file of the `raw-alerts` layout (required), `--screen` makes it a decision run, `--fold`
  is refused, and `--alerts CSV` names the alert table the splits were drawn from (default:
  the layout's own); `--alerts` outside `rj23` exits 2. Any other model or schedule (the
  `stamp_classifier` and ResNet baselines, the `epochs` schedule) exits 2 naming
  `lightning-final`.
  Settings resolve as: `Settings` defaults → the subcommand's legacy
  protocol (`hparams.COMMAND_DEFAULTS`) → `--preset` → `--hparams` → `--config` → `--set` →
  the explicit flags
  (later wins; `cli.training_settings`). The legacy file comes before the config so that a
  config can correct it. A short run sets `max_steps`, `val_every_n_steps` and
  `horizon_increment` (`limit_train_batches` / `limit_val_batches`, Lightning's dry mode,
  exit 2). The final line prints those of `best_epoch`, `best_step`, `monitor_best`,
  `val_acc_best`, `test_acc_at_best`, `selection_acc_best`,
  `evaluation_balanced_accuracy` and `seconds_per_step` that the run has (an `rj23` run:
  the kept weights, the stop and best steps, the stop reason and each stage's macro-F1).
- **Removed (task 4.8):** `pretrain`, `linear-eval` and `fine-tune` (`cli.REMOVED_COMMANDS`)
  trained on Lightning. Naming one prints where it is frozen (the git tag
  `lightning-final`, with the `git worktree add` command) and exits 2; phase 5 writes SimCLR
  pretraining and its evaluation fresh (`simclr-pilot`, `simclr-probe`).
- `compare-reference RUN_DIR... --ref R1|R2|R3|R4|CD21 [--oracle-report FILE]` (see
  `reference`): exit 0 for PASS or REPORT, 1 for FAIL.
- `make-splits PREPARED_DIR --out FILE [--seed 0] [--overwrite]` and `compare-variants
  --baseline RUN... --variant RUN... [--split test|val] [--n-boot N] [--seed S]
  [--allow-mixed-machines]` (task 2.3, see `protocol`); `compare-variants` always exits 0
  (a report). With `--splits FILE [--rule superiority|non-inferiority] [--margin F]` (task
  2.5-infra) it decides on split runs: 0 for ADOPT / KEEP, 1 for NOT ADOPTED / DROP;
  `--rule`/`--margin` without `--splits`, `--split` with it, or `--margin` without
  `--rule non-inferiority` exit 2.
- `ensemble-eval RUN_DIR... --splits FILE --fold K | --screen [--tta rot4|d4|d4_shift]
  [--test] --out JSON [--overwrite] [--data DIR] [--accelerator ...]` (plan row D4, see `tta and
  ensemble_eval`): exit 0 when written, 2 for runs that cannot form an ensemble or a TTA
  mode their input cannot support, 1 for unusable data.
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
- `preflight (--synthetic | --data-dir DIR) [--accelerator cuda] [--runs CD21]
  [--steps N] [--n-per-split N] [--set ...]` (see `preflight`): exit 0 if no step failed,
  else 1; R1-R4 in `--runs` exit 2 naming the git tag `lightning-final` (task 4.8 also
  removed `--reference-dir`).
  `scripts/gpu-preflight.sh` is a thin `uv run --no-sync` wrapper.
- `runs reindex [ROOT ...] [--rebuild]`, `runs ls`, `runs show KEY`, `runs compare SPEC`
  (task A2, see `registry`): the local run registry. Exit 0 on success (`reindex` also when
  it skipped directories), 1 when the registry cannot be used or a comparison does not
  resolve.
- `spec check|expand|run-cell` and `queue init|enqueue|paths|worker|status|cancel|service|
  tmux|submit|sync` (task A3, see `queue` and `docs/queue.md`): `spec run-cell` exits with
  the job's code (0 finished, 1 failed, 2 refused, 75 busy, 128 + a signal); the queue's
  own errors (location, transport, runner, schema) print one line and exit 1, an invalid
  spec exits 2.
- **SimCLR pilot (task 5.0b, see `simclr`):** `simclr-pilot DATA --output-dir OUT
  [--splits FILE] [--epochs N | --steps N] [--batch-size B] [--temperature T] [--lr-per-256
  X] [--warmup-epochs E] [--noise-scale S] [--seed N] [--device auto|cpu|cuda|mps]
  [--fp32-precision ieee|tf32] [--deterministic] [--tier device|host] [--probe-every K
  --label-budgets FILE [--probe-budgets B...] [--probe-part P] [--probe-repeat R]
  [--abort-if-probe-below-random EPOCH]] [--online-probe [--online-probe-batch N]
  [--online-probe-lr X]] [--resume] [--stop-at N]`; `simclr-embed DATA --output-dir OUT (--checkpoint FILE | --random-init)
  [--seed N]`; `make-label-subsets DATA --splits FILE --per-class K [--n-subsets R]
  [--seed S] --out FILE [--overwrite]`; `simclr-probe EMBEDDINGS --subsets FILE --splits
  FILE --out FILE [--weight-decay X...]`. Registered by `simclr.commands.add_simclr_commands`;
  each exits 0 when it ends (an existing run without `--resume`, an existing subsets file
  or a budget larger than a class: 2).
- Exit codes: 0 success; 1 unusable data (`DataError`) or a failed gate/preflight; 2 invalid
  settings or arguments, including an unavailable accelerator (`AcceleratorError`).

## `data` (task 1.2; float input path in 2.1)

`src/astro_simclr/data/`. File schemas, provenance and the workflow are in
[`docs/data.md`](data.md).

```
pickle ──read_pickle──► validate ──► StampData ──write_prepared──► OUT/<stem>/*.npy + meta.json
 (once, trusted machine: prepare-data)                                        │
                                       StampData ◄──load_prepared (mmap, no pickle)┘
                                           │
              training.tensor_splits (the paper's crops), engine.dataset (prepared_split)
```

**Removed in task 4.8:** `data.datasets` (`StampDataset`, `ContrastiveDataset`) and the
`augmentations` module they took (the `Augmentation` protocol and the registry of the
legacy PIL / torchvision pipelines, with the legacy aliases, characterized against
`tests/fixtures/augmentations/legacy_views.npz`, which went with them). Only the Lightning
data modules of the legacy commands used them; they are at the git tag `lightning-final`.
The paper model's training augmentation is `training.paper_augment` (section `training`),
SimCLR's views `simclr.views`.

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
  `crop_bounds` (ALeRCE's crop arithmetic: `(21, 42)` for 21 on 63). Plan row D4 adds
  `offset=(dy, dx)` to `crop_then_normalise` and `crop_split`: the window moves by `dy`
  rows and `dx` columns (refused if it leaves the stamp), then the same normalisation;
  `(0, 0)`, the default, is the paper's crop through the same code. Golden fixture:
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

Neither module imports torch. The uint8 path (`images.npy`) is unchanged.

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

The CLI imports these modules inside its handlers, so `show-config` does not import them.

Behaviour changes against the legacy code (no legacy flag: none of them changes the
values loaded from a valid file):

- Nothing unpickles in a dataset's `__init__` or converts per item (the legacy datasets
  did). The conversion happens once, at load time, with the same `img_float2int` arithmetic, in the stored dtype. This
  is verified bit-exact on all 61,872 stamps of the three complete real files.
- The loader rejects what the legacy code silently accepted: stamps outside [0, 1] or with
  NaN/inf (`np.uint8` of such values is undefined), non-63×63×3 stamps, non-integer or
  out-of-range labels, non-finite features, missing or unexpected keys, and pickles that
  reference non-numpy globals.
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
- the training metrics (`training.metrics`, `num_classes`) and, until task 4.8, the
  Lightning modules (`ClassifierModule`, which sized the metrics and the confusion
  matrices, and its subclasses);
- the data: `training.tensor_splits.TensorSplit.n_classes` (the one-hot width and the
  label range), `protocol.predictions`
  (`Predictions` takes any K ≥ 2 columns, `load_predictions(path, n_classes=5)` checks the
  count, `check_paired` refuses runs with different counts).

An N-class run reads the count from its dataset, `len(data.raw_alerts.label_names_of(path))`,
and passes it down. **The phase-2 path is byte-identical:** with the default, the layers
are built in the same order and draw the same weights (the Lightning modules recorded
`n_classes` in their hyperparameters only when it was not 5, so their `hparams.yaml` and
checkpoints were what they had been). `tests/test_n_classes.py` pins the seeded weights
(the paper task's `PaperNet` has the former `PaperClassifierModule`'s keys and draws) and
the bytes of a predictions file, computed before the parameter existed. CD21-specific code
stays 5-class: `schema.LABEL_NAMES`, the pickle loader and the split layouts, the protocol's
splits and leakage, the M1 oracle, the CLI's metrics, and C6 (`synthetic_bogus` needs
`n_classes=5`, it draws from CD21's Bogus and Asteroid). The N-class runs are RJ23's
(`schedule: rj23`, task 3.4b), whose task takes the count from the layout.

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
| `hybrid` | the paper's convolutions, 2 attention blocks over the 6×6 grid, 4 rotations (B9; `hybrid_model`) | 371,251 | 221,357,824 |
| `hybrid_p4` | the p4 trunk, orientation-pooled, 2 attention blocks with C4-tied positions, 1 pass (B9) | 153,907 | 55,246,720 |

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
  drop_rate, n_classes, feature_set, hybrid_dim, hybrid_depth, hybrid_heads,
  hybrid_readout)`: the factory `tasks.paper.paper_model` calls (the former
  `PaperClassifierModule`'s); the defaults build the faithful
  model. `paper_in_channels(context)`, `default_group_fields(arch)`, `paper_input_side(arch)`
  (21, or 63 for the ConvNeXts: what `image_size` must be).
- `count_parameters(model)`; `count_macs(model, *, in_channels)`: MACs per stamp of an
  eval forward, from `torch.utils.flop_counter` (convolutions and matrix products, not the
  kernel expansion, pooling or BN), tested against the analytic counts. The input side is
  the model's `input_side` (63 for the ConvNeXts) or 21; a 4-rotation model counts its 4
  passes (divide by 4 for a view).
- `VARIANTS`: name → `Variant(row, overrides, description)`: `rot4` (A), `single-aug`
  (`single` + `rot90_augment`), `p4` (B7), `p4-16`, `p4m` (B2'), `p4-context` (B1'),
  `convnext-atto`, `convnext-pico` and `convnext-pico-zoobot` (B8), `hybrid`,
  `hybrid-cls`, `hybrid-wide` (dim 128, 4 blocks) and `hybrid-p4` (B9). A run
  is the `carrasco2021` preset plus `--set key=value` for each override, for example
  `train-supervised --preset carrasco2021 --set paper_arch=p4 --set context=avgpool63`.
  Split runs (`--splits FILE --fold K | --screen`, 2.5-infra) take the same flags: one crop
  reader, `tasks.paper.paper_crops`, serves both paths, and `tensor_splits.protocol_splits`
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
  smoke runs. Every variant's exact training numbers are the golden set's (section
  `engine`).

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
  `tasks.paper.load_backbone` calls them after the seeded build and records the
  pin in `metrics.json["paper_data"]["backbone_weights"]`.
- The data path at 63: `tensor_splits.float_crops(..., side=63)` (and `interim_uint8_crops`):
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

## `hybrid_model` (task 2.5, plan row B9)

`src/astro_simclr/hybrid_model.py`: conv-attention hybrids as `paper_arch` values of
`paper_stamp_classifier`, on the paper's 21×21 crop (and the 6-channel context). Engine-free
(no Lightning import; tested in a fresh interpreter).

- `HybridStampClassifier(arch, *, dim, depth, heads, readout, group_fields, with_features,
  dropout_active, drop_rate, in_channels, n_classes, feature_set)`: a conv trunk down to its
  last 6×6 grid (36 tokens) → `embed` (linear to `dim`) + `positions` (learned 2-D table)
  [+ a CLS token] → `depth` × `EncoderBlock` → LayerNorm → mean of the grid tokens or the
  CLS token → `fc1` to 64, ReLU, dropout per view → the mean over the views → `PaperHead`
  (`feature_set` honoured).
  - `hybrid`: the paper's five convolutions (64 channels per token); one positional vector
    per cell; **4 rotations as one batch, mean after fc1**: exactly invariant by
    averaging (one view alone is not: the free table sees where a pattern is).
  - `hybrid_p4`: the p4 G-CNN trunk (`group_fields` F, default 8: F/2F fields) then the
    **mean over the 4 orientations** of each field (2F channels per token); the positional
    table is **tied across C4** (one vector per orbit of the 6×6 grid, 9 orbits;
    `c4_orbits`). A 90° rotation of the stamp then permutes the tokens, which attention,
    the MLPs, LayerNorm and both readouts do not see: exactly invariant **in one pass**,
    so no 4-rotation mean (it would average 4 equal representations at 4× the cost). Not
    built: 144 tokens of one orientation each with positions tied over the joint
    (cell, orientation) orbits, which keeps orientation information at 16× the attention
    cost.
- `EncoderBlock(dim, heads)`: pre-norm; LayerNorm → `qkv` → scaled dot-product attention
  written as two batched products and a softmax (counted by `count_macs`, the same on every
  backend) → `proj` → residual; LayerNorm → `fc_in` (2·dim, `MLP_RATIO`) → GELU → `fc_out`
  → residual. No dropout, no stochastic depth. Tested equal to `nn.MultiheadAttention`
  with the same weights, and permutation-equivariant.
- `GridPositions(side, dim, *, tied_c4)`: the tied table is expanded by a constant one-hot
  matrix product (each cell gets an exact copy of its orbit's row), so its backward is a
  matrix product, not `index_select`'s scatter (no deterministic MPS kernel).
- Initialisation (seeded global generator): trunks as their models (Glorot, zero biases;
  `group_conv` fans for p4); embedding, attention, MLP, positions and CLS from a normal of
  std 0.02 truncated at 2σ, zero biases, LayerNorm (1, 0); fc1 and the head as the paper's.
- | arch | Parameters | MACs per view | MACs per stamp |
  |---|---|---|---|
  | `rot4` (the paper's) | 441,075 | 52.6 M | 210.6 M (4 views) |
  | `hybrid` | 371,251 | 55.3 M | 221.4 M (4 views) |
  | `hybrid`, `hybrid_readout: cls` | 371,315 | 55.4 M | 221.7 M |
  | `hybrid`, `hybrid_dim: 128`, `hybrid_depth: 4` | 844,915 | 73.0 M | 292.0 M |
  | `hybrid_p4` | 153,907 | 55.2 M | 55.2 M (1 view) |

  (with the 23 features; defaults dim 64, depth 2, heads 4, mean.) The paper's 2304 → 64
  fc1 (147k parameters) becomes the embedding (4k), two blocks (66k) and a 64 → 64 fc1;
  attention adds about 2.7 M MACs per view (5 %).
- Settings: `hybrid_dim`, `hybrid_depth`, `hybrid_heads`, `hybrid_readout` (defaults in
  `config.HYBRID_FLAGS`); another value with a non-hybrid arch is a `ConfigError`. They are
  listed in the `carrasco2021` presets (a changed shape is a deviation), hash-neutral at their
  defaults (`runs.canonical.HASH_NEUTRAL_DEFAULTS`: no existing hash moves) and recorded in
  the module's hyperparameters only when changed (the faithful `hparams.yaml` is unchanged).
- **Tested** (`tests/test_hybrid_model.py`): shapes; float64 invariance under the 3
  rotations with every parameter perturbed, both archs, both readouts, with and without the
  context (a flip visible); one view not invariant for `hybrid` and invariant for
  `hybrid_p4`; the orbits, the tied table, the block; the pinned and the analytic counts;
  initialisation, seed, dropout, `feature_set`; the checks; the settings, deviations,
  hyperparameters and a reload. The golden fixture with the four flags at their defaults
  (`tests/test_paper_variants.py`); CPU split runs of `hybrid` (CLS, C5-C7 augmentation) and
  `hybrid_p4` (context) in `tests/test_split_runs.py`.

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

## `rj23_regime` and `rj23_data` (task 3.4b)

The training regime of RJ23 as **engine-free** objects (the phase-4 rule from B0 on: no
Lightning). The plain engine's adapter is `tasks.rj23`, which uses them as they are; until
task 4.8 the Lightning adapter `training.rj23` did too, and the 4.7(a) test held the two
bit for bit equal (section `engine`). ADR 0023 pins what the sources leave open.

`src/astro_simclr/rj23_regime.py`:

- `RJ23Schedule(eval_every=500, patience=5, max_steps=100_000, mode="max")`, the schedule
  policy. `decide(step, history) -> RJ23Decision` is a pure function of the optimizer steps
  done and the monitored values so far (`step // eval_every + 1` of them: validation `k`
  is step `k·eval_every`, step 0 the initial weights). The decision: `validate` (the step
  is on the grid), `stop`, `keep` (`{"best"}` at a new strict best, `"stop"` at the stop),
  `reason` (`patience` or `cap`; `capped`), `best_step`, `best_value`, `wait`. Patience is
  Keras' / Lightning's `EarlyStopping` with `min_delta=0`, counted in validations; NaN never
  improves; `max_steps` must be a multiple of `eval_every`, so the cap step is validated,
  and a stop there that patience did not cause is `cap`. A history that goes on past a
  stop is an error.
- `ClassUniformStream(labels, batch_size, seed, *, n_classes=None)`, RJ23's
  `sample_from_datasets` with uniform weights: each item draws its class uniformly, then an
  item of that class uniformly (with replacement). `batch(b)` uses
  `engine.randomness.host_generator(seed, b, STREAM)` only (`STREAM` = 10,
  `RESERVED_STREAMS["rj23_class_uniform"]`), so a batch is a pure function of `(seed, b)`
  and a resume
  at step `s` (the stored position) continues the stream exactly; `iterate(start)`,
  `state()` (the definition a resume checks), `batches_per_pass` (`ceil(n / batch)`, the
  logging epoch). Not CD21's tiling (`training.tiling`).
- `KerasAdam(params, lr, betas=(0.9, 0.999), eps=1e-7)` and `keras_adam(params, lr)`:
  Keras' update, `θ −= lr·sqrt(1 − β2^t)/(1 − β1^t) · m / (sqrt(v) + ε)`, ε added to the
  uncorrected `sqrt(v)` (torch's `Adam` adds it to `sqrt(v̂)`).
- `predict_probabilities(model, images, positions, batch_size)`: softmax in eval mode under
  `no_grad`, in fixed batches, mode restored; float32 on the CPU.

`src/astro_simclr/rj23_data.py`: `rj23_parts(data, splits, scenario, *, mode, table,
large_values)` verifies the object splits (`protocol.object_splits`) against the alert table
(default `protocol.alerts.alert_table_from_raw(data)`, grouped by `oid`; a CSV table for
another group key), checks that every alert of a part is in the layout with the same label,
and builds the parts of the protocol (`MODE_PARTS`):

| Mode | Train on | Stop and select on | Evaluated (stage names) |
|---|---|---|---|
| `replication` (the paper protocol) | `Train` | the whole `Validation` | `Validation` (`val`), `Test` (`test`) |
| `decision` (a phase-3 decision; `--screen`) | `Train` | the `selection` half | `selection`, `decision`; `Test` never read |

Each `RJ23Part` holds the scenario's view `(N, C, H, W)` float32
(`rj23_input.rj23_views` → `channels_first`), the unit-sphere position `(N, 3)`, the labels
and the `candid`s (ascending).

**Tested** (`tests/test_rj23_regime.py`, `tests/test_rj23_data.py`): the grid from step 0;
the stop at patience on a strict improvement (an equal value is no improvement), the wait
reset, NaN, `min` mode, purity, the cap flag (and patience at the cap step), refusals of an
inconsistent history and bad values; the sampler's class shares (1/3 each on a 600/30/3
split, within 5 sd of 51,200 draws), within-class uniformity, determinism and a resume from
a stored position; Keras Adam against two hand-computed steps, against torch's placement
(they differ by 6e-7 at step 1) and equal to torch's Adam with ε = 0; the parts of both
modes for every scenario (shapes, counts, `candid` order), a CSV table, a label mismatch,
an alert missing from the layout, splits of another table or of other classes.

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

## `archive` (phase 5, tasks 5.1a and 5.1; ADR 0019)

`src/astro_simclr/archive/`: the unlabelled pretraining corpus from the ZTF public alert
archive, and its index. The rules, layout, commands and budget are in
[`docs/data.md`](data.md), "Archive corpus" and "Corpus index"; the decisions in ADR 0019
and its amendment. Nothing in the package imports torch or opens a network connection.

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
- `corpus_index` (task 5.1):
  - `build_corpus_index(archive, layout=, oid_files=, candid_files=, radius_arcsec=1.5,
    duplicates="first", seed=0, near=DEFAULT_NEAR, chunk_rows=4096, progress=)` lists the
    complete nights once, reads every row's identity and position, and applies `RULES` in
    order: `candid_list`, `oid_list`, `position` (`protocol.groups.match_any_within`
    against every CD21 alert, positions from `split_positions`; a match drops its whole
    `oid`), `near_duplicate` (on
    the rows left; a match drops its whole `oid`), then `duplicate_oid` (`_one_per_oid`:
    the earliest, or a draw from `host_generator(seed, 0, corpus_duplicates)`). It returns
    `CorpusIndex(rows, meta)`.
  - `_reference_layout` re-hashes each `features.npy` and `images_f32.npy` it reads
    against the layout's `meta.json`, and refuses a layout without float32 stamps when the
    near-duplicate check runs.
  - `load_candid_exclusions` is the candid counterpart of `selection.load_exclusions`.
  - `write_corpus_index` / `read_corpus_index` handle the `.npz` (`rows` and `meta` as a
    JSON string; atomic write; format, version, dtype, `n` and sort order checked).
    `check_archive(index, archive)` compares `corpus.json`'s SHA-256 and the `meta.json`
    SHA-256 of every night the index uses.
- `near_duplicates`: `descriptors(stamps)` (mean-pool 7 to 9×9×3, centred per channel,
  unit norm, float32; zeros for a stamp with no structure), `NearDuplicateRules(cosine=0.99,
  max_abs_diff=1e-3, candidates=4)`, `ReferenceStamps(names, images)` (the CD21 splits,
  memory-mapped; `locate`, `stamp`, `descriptors`), and `find_near_duplicates(chunks,
  reference, rules)` → `NearDuplicates(matched, matches, candidates, checked)`. Stage 1
  thresholds float32 cosines (numpy) in blocks of 1024 query rows and keeps the best
  `candidates` per row (ties by the lower reference row); stage 2 checks the full-resolution
  pixel difference.
- `reader`: `normalise_stamps(raw)` (`alerce_normalise`, then float32) and
  `CorpusStamps(archive, rows)`, which memory-maps the indexed nights (`load_night`),
  checks the index's `oid`/`candid` against them, and maps each index position to a
  cutout file and offset. `read(positions, out=)` reads sorted by `(file, offset)` and
  scatters to the requested order; `normalised(positions)`; `chunks(positions, n)` yields
  `(positions, normalised)` blocks; `nbytes`.
- `commands`: the `build-corpus-index` subcommand (`add_corpus_index_command`, registered by
  `cli.build_parser`).
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

`tests/test_corpus_index.py` (offline: a CD21-like layout with float32 stamps and set
positions, and an archive written with `NightWriter`) covers:
- the cross-match inside and outside 1.5″, across the RA wrap and at the pole, and its
  ties;
- every rule and its count, with an object on two nights, and `first` and seeded `random`;
- a renormalised copy of a CD21 stamp dropped with its `oid`, and a noisy look-alike only
  counted;
- the SHA-256 of every input, the index file's round trip and refusals, and
  `check_archive`;
- the reader across cutout files and in any order, and its refusals of another archive;
- the pilot from an index: the same model on `memmap` and `host`, the CD21 evaluation and
  record, the refusals (`device`, the host cap, noise, another evaluation layout, a weak
  index, a resume on a rebuilt index), the one-batch prefetch, and a resume from a
  checkpoint written before the corpus fields;
- the CLI.

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

## `training` (task 1.6; engine-free since task 4.8)

`src/astro_simclr/training/`: the engine-free pieces that the plain engine's tasks
(`tasks.paper`, section `tasks`) train with. **Nothing here imports Lightning** (ADR 0014).

Until task 4.8 the package was the Lightning 2 port of the legacy `utils/training.py`
(PyTorch Lightning 1.4): the modules `SimCLRModule`, `LinearEvalModule`, `FineTuneModule`,
`SupervisedCEModule`, `SupervisedCEResNetModule`, `PaperClassifierModule` and `RJ23Module`;
the data modules `ClassifierDataModule`, `ContrastiveDataModule`, `DeviceTensorDataModule`
and `RJ23DataModule`; the callbacks `HorizonCallback` and `RJ23ScheduleCallback`; and the
Trainer helpers of `training.runs` (`seed_run`, `build_callbacks`, `build_trainer`). They
are frozen at the git tag `lightning-final`, with their design notes and their behaviour
changes against the legacy code (`git show lightning-final:docs/architecture.md`, section
`training`). What they computed for the paper classifier and RJ23 is what the plain engine
computes: the 4.7(a) differential test held the two engines bit for bit equal, and the
golden regression set now holds those numbers fixed (section `engine`).

| Module | What it holds |
|---|---|
| `metrics` | `confusion_counts` (int64), `confusion_matrix` (normalised over the true labels; a class without true samples gives a row of zeros) and `macro_f1` (the unweighted mean of the per-class F1 over the classes present in the predictions or the targets), for `tasks.results`. Plain torch since task 4.8: torchmetrics' operations (`multiclass_confusion_matrix`, `multiclass_f1_score`) in their order and dtypes, so every recorded number is bit for bit what torchmetrics gave (checked equal on 20,000 random cases before torchmetrics went; the golden set holds the evaluation entries). A label outside `[0, num_classes)` or a float label is a `ValueError`. `classification_metrics` (the Lightning modules' logged accuracy, precision and recall) went with them. |
| `optimizers` | `build_optimizer(name, params, lr, *, lars_legacy_trust_ratio=True, adam_betas=ADAM_BETAS)`: `AdamW` (torch's weight decay 0.01) and `Adam` (none), both with `eps` 1e-8 and `adam_betas` (default the legacy (0.5, 0.9)), `SGD` and `LARS` (`weight_decay` 0, momentum 0.9, Nesterov); an unknown name raises `UnknownOptimizerError` listing the valid ones (the legacy chains ended in `UnboundLocalError`). `PAPER_ADAM_BETAS`: `code` (0.9, 0.999), ALeRCE's, or `paper` (0.5, 0.9), a diagnostic (task 2.2c; ADR 0003 explains why torch's ε placement is kept). |
| `tiling` | A re-export of `engine/stream.py` (`tile_indices`, `PermutationStream`; below). |
| `tensor_splits` | The paper model's splits as arrays (below): `TensorSplit`, `float_crops`, `interim_uint8_crops`, `without_train_indices`, `protocol_splits`, `check_split_identity`, and the batch formats `Batch` / `TrainBatch`. |
| `class_augment`, `flux_rescale`, `paper_augment` | The paper model's training augmentation (plan rows C5-C7, subsections below); `paper_augment.PaperAugment` composes them per step. |
| `ema`, `lr_schedule` | Plan row D3 (subsection below). |

**Tested:** `tests/test_training_metrics.py` (the metrics by hand, the label-range error, a
class present in the predictions only), `tests/test_training_optimizers.py` (the factory),
`tests/test_tensor_splits.py` (the arrays and the split identity).

### The `carrasco2021` schedule (task 2.2a)

ALeRCE's training procedure, on the plain engine: the stream (`engine.stream`, re-exported
by `training.tiling`), the splits (`training.tensor_splits`), the horizon and the selection
(`engine.policies`, built from the settings by `tasks.plain.schedule_of`) and the model
(`tasks.paper`).

**Tiling** (`engine/stream.py`, re-exported by `tiling.py`). `tile_indices(labels)` is ALeRCE's
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

**Splits** (`tensor_splits.py`). `TensorSplit(images, features, labels, fits=None,
n_classes=5)`: float32 (N, 3, 21, 21) (or (N, 6, 21, 21) with the 63×63 context of task
2.5-models, or the whole stamps (N, 3, 63, 63) of the ConvNeXt rows, plan row B8;
`IMAGE_SIDES`), float32 (N, 23), int64 (N,), and optionally the float32 (N, 6)
`class_augment.stamp_fits` of the crops (C5 / C6), or (N, 11) with
`flux_rescale.fit_central_psf` after them (C7), validated. The plain engine serves them on
the `device` tier (`engine.tiers`, section `engine`): Train as the tiled stream, an epoch of
`len(tiled) // batch_size` steps (1,136 for `red`; the epoch only numbers `best_epoch` /
`last_epoch`), with the fits as a fourth element when Train has them; Validation and Test
in consecutive `eval_batch_size` batches in file order, the last partial. With 64 the
monitored loss (the mean of the batch losses, `engine.task.mean_of_batch_losses`) is TF's
selection loss, the mean of batch-64 means in file order. A Train split smaller than one
batch is refused. The builders (task 2.2c):

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
  order, before the tiling, keeping Train's fits with its stamps; Validation and Test are
  unchanged.
- Split runs (task 2.5-infra; see `protocol`): both builders take `names=("Train",
  "Validation")` so that Test is never read, and `protocol_splits(splits, parts, n_train)`
  gathers a run's parts by pool index into three roles (`Train` = the part trained on and
  tiled, `Validation` = the selection part, `Test` = the evaluation part), without fits.
  The run's identity (`SplitRun.identity()`) is in every checkpoint (`engine.checkpoint`);
  `check_split_identity(stored, current)` refuses a resume across split runs or between a
  split run and a normal run (`ConfigError`).

**Horizon and selection** (`engine.policies`; `tasks.plain.schedule_of(settings)`): a
validation every `val_every_n_steps` steps (`ValidationGrid`; no epoch-end, no forced final
validation); a strict improvement of `loss_val` (`Monitor`: NaN never improves, the first
finite value always does) keeps a copy of the weights and raises the horizon to
`max(horizon, step + horizon_increment)` (`Horizon.extended`); the run ends when the steps
done reach the horizon, checked after the step's validation, so a new best at the horizon
step extends the run. `ckpt/last.pt` is written after the validation of every
`checkpoint_every_n_steps` steps (after the policy update, so a resume owes no validation;
hence the multiple of `val_every_n_steps`) and at the end; `ckpt/best.pt` holds the best
weights at the end. A resume restores the horizon (the larger of the saved one and the new
`max_steps`, `Horizon.resumed`), the best weights and score, and refuses another monitor.
With `checkpoint_selection: final` (plan row D3) no best weights are kept.

Step numbering: TF validated after iteration `i` ∈ {0, 10, ...}, i.e. after `i + 1` steps;
here after 10, 20, ... steps (as Lightning did). `best_step` counts steps done, so M1's
"iteration 4,780" is step 4,781 here; the one-step shift of the validation grid is not
corrected.

**Model** (`tasks.paper`, section `tasks`): `PaperNet` is `PaperStampClassifier` (or the
2.5 variant `paper_variants.build_paper_model` builds, through `tasks.paper.paper_model`)
under `model`, with `P_stamps_loss(beta_loss)` as `criterion`, the `state_dict` keys of the
former `PaperClassifierModule`. Its optimizer is `build_optimizer` with
`adam_betas=PAPER_ADAM_BETAS[adam_betas]` over every parameter but the EMA's. The
training augmentation is `paper_augment.PaperAugment`: `rot90_augment` rotates each stamp
by `k·90°`, `k` from `numpy.random.default_rng([seed, step])` (device-independent, exact on
resume), then C5-C7 (below); validation and test are never augmented. With `feature_set:
none` the model ignores the features.

**Tested** (`tests/test_tiling.py`, `tests/test_engine_stream.py`,
`tests/test_tensor_splits.py`, `tests/test_carrasco2021.py`): the tiling example and exact balance; the stream against the
concatenated permutations; `run_plain` on the synthetic data with 2-step epochs, so the
validation grid, the horizon and mid-epoch checkpoints cross epoch boundaries, writing
`metrics.json` (the horizon extension, the best and last steps and epochs, the
checkpoints); a smoke run on every available accelerator; the settings' checks. The exact
numbers are the golden set's, the resumes `tests/test_engine_resume.py`'s (a mid-epoch
resume, a resume at the horizon, an augmented resume; section `engine`).

The full replication settings are the `carrasco2021` preset (see `presets`):
`train-supervised --preset carrasco2021`, or `--config configs/phase2/carrasco2021.yaml`.

### Class-aware augmentation (plan rows C5 and C6)

`src/astro_simclr/training/class_augment.py`; decisions in ADR 0006. Training only, in
`paper_augment.PaperAugment.plan` (host) and `apply` (on the device-resident batch), called
by `tasks.paper.PaperTask.train_step` (see "Prefetch" below): C6, C7 (below), then C5's
noise on the rows C6 did not replace, **then** the rotation
of `rot90_augment`, then the feature jitter on every row. The class augmentation runs before
the rotation because the stamp fits describe the unrotated crop (C7's PSF centroid in
particular; C5 / C6 are rotation-agnostic). Validation and test are never augmented.

The cross-channel relation (`stamp_fits`, `FIT_COLUMNS`, `masked_pixels`, `renormalise`,
`perturb` and their constants) lives in `src/astro_simclr/stamp_relation.py`, outside the
`training` package (whose `__init__` imported Lightning until task 4.8), so that the SimCLR
views can use it engine-free; `class_augment` re-exports it unchanged (ADR 0025).

- **The difference is computed, not augmented.** `stamp_fits(images)` fits, per crop and in
  float64, `diff ~ a·sci + b·tmpl + c` (least squares over the 441 pixels, a
  pseudo-inverse), its R², and the background sigmas of the science and the template
  (1.4826 × the MAD of the two-pixel border, masked pixels left out): float32 (N, 6),
  `FIT_COLUMNS`. `tasks.paper.with_train_fits` computes them once on the run's Train role
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
  `default_rng([seed, step, stream])` (`step`: the optimizer steps done before the step,
  Lightning's `global_step`), streams 1 (noise), 2 (jitter), 3 (synthetic); every draw has
  a fixed size per batch. Device-independent, exact on resume, and `rot90_augment`'s
  `[seed, step]` draws are unchanged.
- **Plan and apply** (ADR 0006, item 10): each augmentation is a host *plan* and a device
  *apply* (`plan_noise` / `apply_noise`, `plan_synthetic` / `apply_synthetic`, `plan_jitter`;
  C7's `plan_flux` / `apply_flux`; the public `noise_augment`, `synthetic_bogus`,
  `jitter_features`, `flux_rescale` are apply(plan(...))). `PaperAugment.plan(host_batch,
  step)` plans a whole step (C6, C7, C5, rotation, jitter: a dict of CPU tensors);
  `apply(batch, plan)` applies it with the same torch operations, so the result is bit for
  bit the in-step one on every device. `PaperTask.train_step` copies the step's batch to
  the host, plans, uploads the plan with `upload_plan` (pinned + `non_blocking` on CUDA)
  and applies it. The Lightning path planned one step ahead from a CPU copy of the Train
  split, with no blocking copy in the step (the data module's `planner`); that prefetch
  went with it in task 4.8, and the plans, hence the numbers, are the same (the golden
  set).

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

### Transient flux rescaling (plan row C7)

`src/astro_simclr/training/flux_rescale.py`; decisions in ADR 0007. Training only, in
`PaperAugment`'s class augmentation, after C6 and before C5's noise; never Bogus (nor C6's
synthetic rows). Scaling the whole difference would do nothing: the per-channel min-max is
scale-invariant. So:

- `fit_central_psf(images)`: once per Train stamp (`tasks.paper.with_train_fits`, appended
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
task (training only), the settings and the preset deviation; a p4 run on every
accelerator; and in `tests/test_split_runs.py` an augmented fold run (the fits are the
fold's training part's) and its exact resume.

### The `rj23` schedule (task 3.4b)

RJ23's regime runs on the plain engine through `tasks.rj23` (section `tasks`): `train-supervised
--preset rj23-<scenario> --splits OBJECT_SPLITS [--screen] [--alerts CSV]`, or `astro-simclr
run` of an `rj23` config, goes to `tasks.plain.run_plain` with an `RJ23Request(splits, mode,
alerts)`, which runs `tasks.rj23.run_rj23_plain`. The policy objects are `rj23_regime`'s and
the parts `rj23_data`'s (ADR 0023), used as they are. Until task 4.8 the Lightning adapter
`training.rj23` (`RJ23Module`, `RJ23DataModule`, `RJ23ScheduleCallback`, `run_rj23`) ran them
too and wrote the same files with the same values; it is at the git tag `lightning-final`.

- **The task:** `RJ23Task`, `rj23_model.build_rj23_classifier` built on the CPU after the
  seed, unweighted cross-entropy on int64 labels (`loss_train`), `keras_adam`; its
  `eval_step` is `predict_probabilities` of the batch.
- **The data:** every part on the training device; `RJ23TrainSource` gives
  `ClassUniformStream` batch `s` at step `s` (an epoch is `batches_per_pass` steps, for
  logging only), and its checkpoint state is the stream definition plus the run identity
  (splits SHA-256, mode, scenario): a resume of another run is a `ConfigError`, of another
  stream (data, batch size, seed) a `DataError` (`check_stream_state`).
- **The schedule** (`rj23_schedule`): the selection part's macro-F1 (`macro_f1_score`,
  `benchmark_metrics.macro_f1`) as `macro_f1_val`/`max`, validated at step 0 and every
  `val_every_n_steps` steps (`ValidationGrid(at_start=True)`), a fixed horizon at the cap
  (`Horizon(max_steps, 0)`), `RJ23Schedule` as the stop rule on the history, and both
  keeps (`stop`, `best`). `ckpt/last.pt` is written after the validations of every
  `checkpoint_every_n_steps` steps and at the stop.
- **Outputs:** `ckpt/last.pt` (the whole training state; its weights are the `stop` keep),
  `ckpt/best.pt` (the `best` keep), `scalars.jsonl` (from step 0), `predictions_<stage>.npz`
  of the primary keep (`rj23_keep`) and `predictions_<stage>_<other keep>.npz`, index =
  `candid`, and `metrics.json`: the `bench-report` contract (`seed`, `machine`,
  `benchmark.splits_sha256`, `predictions.<stage>.file`), `benchmark` (splits, mode, alerts,
  per-part class counts), `schedule` (`history` of `[step, macro-F1]`, `stopped_by`,
  `capped`, `stop_step`, `best_step`), `keep` (both keeps' step, checkpoint, evaluation and
  predictions), `evaluation` (the primary's: accuracy, balanced accuracy, macro-F1,
  recalls, confusion counts), `preset`, `data`, `config`, `versions`, `timing`, and `run`
  (run id, attempt, config hash). `manifest.json` (`engine: plain`) is written around the
  fit (`runs.record.start_run`, ADR 0015): a replication run hashes its object splits
  through `object_splits=` (split part `whole`), a decision run passes them as a screen.

**Resume** restores the weights, the optimizer, the step, the history, the kept weights and
every RNG stream: the batch stream and the dropout masks continue exactly (Lightning's
checkpoints held no global RNG state, so its resumes were not exact). A resumed run that
had stopped trains nothing.

**Tested** (`tests/test_rj23_training.py`, `tests/test_tasks_rj23.py`, CPU, a synthetic
5-class `raw-alerts` layout): an end-to-end run (the history agrees with the policy, the
`bench-report` contract read by `benchmark_report`, both keeps written, the preset
deviations); with scripted validations, the stop at patience keeping the stop and the best
weights, `rj23_keep: best`, the cap; a decision run; a CSV alert table; resumes; the
refusals; the presets against `RJ23_TABLE2` and the input constants; the settings rules;
the CLI. The exact numbers of each scenario, protocol and stop reason are the golden set's,
the resumes before a patience stop `tests/test_engine_resume.py`'s (section `engine`).

### Weight EMA, learning-rate schedule and the final weights (plan row D3)

The CD21 protocol picks the checkpoint of the best validation loss on 500 Validation stamps,
whose accuracy has a standard error of about 1.1 points: the choice is noisy. Row D3 scores
instead the **final** weights of a run whose learning rate anneals, averaged by a weight
EMA. Three engine-free pieces (CLAUDE.md's checklist), each off by default:

- **`training.ema`** (`EmaWeights`, `ema_decay_at`, `trainable_parameters`): a deep copy of
  the model (`shadow`, frozen), updated after every optimizer step as
  `shadow.lerp_(theta, 1 − d_s)`, `d_s = ema_decay` or, with `ema_warmup`, TF's / timm's
  ramp `min(ema_decay, (1 + s) / (10 + s))`, where `s` is the steps done before the step.
  **Buffers are copied, not averaged**: the BatchNorm running statistics (the paper model's
  BN on the 23 features) are already a running average of batch statistics, and an average
  of them describes no model's activations; the shadow keeps the most recent training
  statistics (timm's `exclude_buffers` choice). The shadow is the submodule `ema` of the
  training module (`PaperNet.ema`, registered after `model`, so its `state_dict` keys
  `ema.shadow.*` are those of the former `PaperClassifierModule.ema`): it moves with the model
  to its device, `last` checkpoints carry it and a resume restores it exactly, and the
  best weights kept by `best_val_loss` include it. Its parameters are left out of the
  optimizer (`trainable_parameters`: exactly `parameters()` without an EMA). **In
  `eval()` mode the module's forward runs the shadow** (every validation, test and final
  evaluation), in `train()` mode the model. With the ConvNeXt backbone weights (ADR 0022),
  the shadow is copied after they are loaded (`reset_ema` / built after `load_backbone`).
- **`training.lr_schedule`** (`LrSchedule`): the factor of `lr` at step `s`: warmup
  `(s + 1) / w` for `s < w`, then 1 (`constant`) or `0.5 (1 + cos(π (s − w) / (T − w)))`
  (`cosine`, `T = max_steps`, clamped to 0 past `T`). Applied by a `LambdaLR` stepped after
  every optimizer step (`PaperTask.after_step`; its state in the task state of the
  checkpoint; Lightning stepped it with `interval: step`). A constant rate without warmup
  builds no scheduler.
- **`checkpoint_selection`**: `best_val_loss` (CD21) or `final`. With `final` the
  `Schedule(keep=("stop",))` still tracks the best validation (`best_step`, `monitor_best`,
  the scalars) but keeps no weights and writes no best checkpoint; the run evaluates its
  last weights (`evaluate_kept(..., ("stop",))`), and `metrics.json["checkpoints"]["best"]` is
  `null`. `final`, `cosine` and a warmup need `horizon_increment: 0`, so the run is exactly
  `max_steps` long and the validation decides nothing. The EMA alone keeps CD21's
  selection and horizon rule (it then selects among EMA weights).

Every paper run's `metrics.json` has `selection` (`tasks.results.selection_entry`):
`checkpoint_selection`, `scored_step` (the best validation's step, or the last step),
`ema` and `ece_bins`; `val_acc_best` / `test_acc_at_best` keep their names and are the
scored weights' values. The faithful preset lists the five fields (a D3 run is a preset
deviation); with them off a run is bit-identical to one before the flags (no submodule, no
scheduler, the same optimizer parameters, an empty task state).

**Tested** (`tests/test_weight_ema.py`): the EMA update and its warmup against hand
computations, the BN buffers copied, the shadow frozen and out of the optimizer, on the
model's device on every accelerator, the eval-mode forward; the schedule's factors and the
`LambdaLR` rates by hand; ECE by hand; the `Settings` rules; the EMA and scheduler state
in the checkpoints; `final` scoring the last weights (re-evaluated from `last.pt`) and
giving the same results under arbitrary validation values. The golden set holds the
numbers of each flag and of all of them together, and `tests/test_engine_resume.py` a
mid-run resume with them (section `engine`).

## `engine` (phase 4, task B0; ADR 0014)

`src/astro_simclr/engine/`: the contracts of the plain-PyTorch training loop that replaced
Lightning (phase 4), and the only training engine since task 4.8. B0 froze them; the loop
(4.1), the typed config (4.2) and the data tiers (4.3) are built against them, and the 4.7
gate held the two engines bit for bit equal before Lightning was removed. **Engine-free
rule** (ADR 0014, CLAUDE.md): nothing in `src/` imports Lightning (`tests/test_engine_contracts.py`
checks the engine in a fresh interpreter, `tests/test_lightning_retired.py` greps every
import statement); a new training flag is written against these contracts and joins the
golden regression set (below). ADR 0014 lists, with file and line, every hidden Lightning
behaviour the loop reproduces (C1-C11).

| Module | Contract |
|---|---|
| `task` | `Task`: `module`, `make_optimizer()` (after the device move), `train_step(batch, step)` (the loss with its graph; `step` = optimizer steps done before it), `eval_step(batch) -> EvalOutput`, `state()` / `load_state()`. `mean_of_batch_losses`: the monitored value, the float32 mean of the batch means (not the per-sample mean). |
| `policies` | `Monitor.improves` (strict, NaN never, the first finite value always: Lightning's `HorizonCallback`'s rule); `ValidationGrid.due(steps_done)` (multiples of the interval only: no epoch-end, no forced final validation), and `at_start` (default off: the initial weights too, step 0, `rj23`); `Horizon.extended` / `resumed`; `EarlyStop` (Lightning's semantics; unused); `StopRule` (`decide(step, history) -> .stop`, an engine-free policy on the whole history: `rj23_regime.RJ23Schedule`, as it is); `Keep = stop \| best`; `Schedule` (`stop_rule` or `early_stop`, not both), whose defaults are the `carrasco2021` settings. |
| `checkpoint` | `Checkpoint` (a `TypedDict` of plain values, read with `weights_only=True`) and its frozen key set `CHECKPOINT_KEYS`: format, engine, identity (A0's run id and attempt, the split run), `steps_done` (also the stream position), model, optimizer, task, policy (with a `stop_rule`, its `history` of monitored values: absent otherwise, so the `carrasco2021` checkpoints keep their keys), stream definition, every RNG stream, numerics (ADR 0010) and versions. `check_resumable` refuses Lightning checkpoints, other engines, formats and incomplete ones. |
| `device` | `DevicePolicy`: one device (`world_size != 1` refused), `cudnn.benchmark = not deterministic`, `use_deterministic_algorithms(deterministic)`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"` only when deterministic (`torch_flags()`, the flags of Lightning's `_set_torch_flags`, written out in the test since 4.8), ADR 0010's precision. |
| `data` | `Batch` / `TrainBatch` (equal to `training.tensor_splits`'s, the format of the former Lightning loader); `DataTier` (`device`, `pinned`, `memmap`, all exact); `TierChoice` (non-float32 storage is a recorded deviation); the `BatchSource` (`batch(step)`, a pure function of the stream and the step) and `EvalSource` protocols. |
| `randomness` | `host_generator(seed, step, stream)` (= `class_augment.augment_rng`: per-sample parameters, device-independent); `device_generator(seed, step, stream, device)` (per-pixel fields: the seed is device-independent, the values are not, so runs record the device); `RESERVED_STREAMS`, the stream numbers in use (1-4 the paper model's augmentations, 5 the `block` stream's passes (4.3, ADR 0018), 6-9 the SimCLR pilot's views, noise fields, batch order and diagnostics; ADR 0025; 10 RJ23's class-uniform stream, ADR 0023; 11 the corpus index's `--duplicates random` draw, task 5.1; 12 the SimCLR pilot's online-probe side-batch, `simclr.monitor`). |

Until task 4.8 `tests/test_engine_contracts.py` also pinned the Lightning 2.6.6 facts the
checklist relied on (the validation grid across 7-step epochs, `global_step` inside
`training_step`, the gradient `None` at validation, the epoch numbering of `metrics.json`).
Since then the rules taken from Lightning (the monitor's strict comparison, the device
flags, early stopping, `seed_everything`, the device tier's tiled stream) are written out
as explicit values in `tests/test_engine_contracts.py`, `tests/test_engine_loop.py` (early
stopping against a table recorded from Lightning 2.6's `EarlyStopping`) and
`tests/test_engine_tiers.py`.

### The engine core (task 4.1; ADR 0014, amendment of task 4.1)

The loop itself, against the contracts above. Task 4.6 wired the CLI to it
(`tasks.plain`); since task 4.8 it is the only loop.

| Module | What it does |
|---|---|
| `loop` | `fit(build, train, validation, schedule=, device=, seed=, run_dir=, identity=, resume=, on_validation=, progress=, score=) -> (task, FitResult)`: precision and flags (`DevicePolicy.apply`), `seed_everything`, `build()` on the CPU, the move, `make_optimizer()`; per step forward, `zero_grad(set_to_none=True)`, backward, `step()`, `Task.after_step(step)` (a scheduler step, a weight EMA update; plan row D3); a fresh fit validates first when the grid is `at_start` (that validation's epoch is 0, as Lightning's `current_epoch`, not the `-1` of `(steps - 1) // batches_per_pass`); after each step the grid, then the validation (`zero_grad`, every submodule's mode captured, `eval()`, `no_grad`, the modes restored), the policies (`Monitor`, `Horizon`, `EarlyStop.update` or the `StopRule` on the history), `last` if due (never at step 0); the loop ends when `steps_done >= horizon` or the stop rule stopped, after that (a resumed run whose rule had stopped trains nothing). `evaluate(task, sources)` (`inference_mode`) and `evaluate_kept(task, result, sources, keep)` (`stop` first, then the restored best). `FitResult`: `last_step`, `horizon`, `best_step`, `best_score`, the derived `best_epoch` / `last_epoch`, this fit's `validations`, `steps`, `fit_seconds`, the best weights, the run's `history` (with a stop rule). The monitored value is `loss_val` (the mean of the batch losses) unless a `score` of the validation's outputs is given, under the monitor's name (task 3.4b: `macro_f1_val`); `scalars.jsonl` and the progress line use that name. |
| `checkpoint` (4.1 part) | `capture_rng` / `restore_rng` (Python, NumPy's global, torch CPU, CUDA or MPS); `numerics_record`; `save_checkpoint` (atomic: a temporary file, then a rename); `load_checkpoint` (`weights_only`, `check_resumable`); `check_identity` (run id and split run; the attempt may differ) and `check_numerics` (precision and determinism); `BestWeights`, `ckpt/best.pt`'s record (weights of the best validation, not resumable), read by `load_best`. |
| `policies` (4.1 part) | `EarlyStop.update(value, best, wait, mode)`: Lightning's `_evaluate_stopping_criteria` in its order and float32 arithmetic. |
| `precision` | ADR 0010's `apply_fp32_precision(precision)` and `fp32_precision_record(device)`, moved from `training.runs` (removed in task 4.8). `apply_fp32_precision` sets `torch.backends.cudnn.conv.fp32_precision` and `torch.set_float32_matmul_precision("highest" \| "high")` (the matmul call, not `torch.backends.cuda.matmul.fp32_precision`, after which `torch.get_float32_matmul_precision()` raises); `fp32_precision_record`: the effective policy (`tf32` only on a CUDA device of compute capability 8.0+), the settings read back and the capability, for `metrics.json["machine"]`. |
| `randomness` (4.1 part) | `seed_everything(seed)`: `lightning.seed_everything`'s three calls, without its environment variables. |
| `scalars` | `ScalarLog`: `scalars.jsonl`, one flushed line per validation (step, epoch, `loss_val`, mean `loss_train` since the previous validation, `improved`, horizon, wait count, seconds); a resume keeps the lines of steps up to the checkpoint's; NaN is `null`. `read_scalars`. |
| `record` | `machine_record(device)` (the keys of the Lightning runner's `machine_record`, `engine: plain`) and `recorded(record)`, which ends a `runs.record.start_run(..., engine="plain")` record `finished`, `failed` or `cancelled`. |

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
inference contexts, `keep`, early stopping against a table recorded from Lightning 2.6's
`EarlyStopping` over value sequences, the step-0 validation, a `score` and a `StopRule`
(the history in the checkpoint, a resumed stopped run training nothing, a mid-run resume
keeping the history, none in a `carrasco2021` checkpoint), the checkpoint contents, exact
resume through the global RNG streams (and its control: without `restore_rng` it is not
exact), every resume refusal, atomic writes, the seeding and flags, the scalars, and the
`engine: plain` manifest.

**The golden regression set** (task 4.8; `tests/test_engine_regression.py`, cases in
`tests/_engine_cases.py`, goldens in `tests/fixtures/engine_golden/<platform>/<case>.json`)
replaced the 4.7(a) CPU differential test, which ran Lightning's `run_command` and
`tasks.plain.run_plain` in one process and held them bit for bit equal. The goldens are the
plain engine's outputs, recorded on main before Lightning was removed, while that test
passed, so they are what both engines computed. For every case, on the synthetic data, it
asserts bit for bit: the initial weights (a SHA-256 of every tensor); every fit validation
(its step, the monitored value, a SHA-256 of the weights then); `metrics.json` but for what
names the engine, the machine, the run or a temporary path (`_engine_cases.computed`: the
steps, horizon, epochs and `monitor_best`, `paper_data`, `preset`, `protocol`, every
evaluation entry and the `rj23` schedule record); every array of every `predictions_*.npz`;
the manifest `finished` with `engine: plain`. The cases are the differential test's: every
2.5 and D3 flag (`_engine_cases.FLAGS`: the faithful preset, `single`, `p4`, `p4m`,
`convnext_atto`, `hybrid_p4`, a `cls` hybrid, `context`, `rot90`, C5, C6, C7, both
`feature_set` diagnostics, `fp32_precision: tf32`, the EMA, `cosine`, a constant rate with
warmup, `final`, D3 together and with C5-C7 on `p4m`, C5-C7 on `p4`), a fold (augmented)
and the screen, and the `rj23` scenarios, protocols, sensitivity flags and both stop
reasons (`RJ23_CASES`).

- **Platforms.** CPU kernels round differently between instruction sets and BLAS builds,
  so a golden holds on the platform it was recorded on, `<sys.platform>-<machine>-<torch
  CPU capability>` (`_engine_cases.platform_key`; `darwin-arm64-default` today). On another
  platform every case is skipped, naming it. A torch upgrade that changes these numbers is
  a numerics change: it fails here until the goldens are recorded again (bump
  `NUMERICS_EPOCH`).
- **Recording** (a new flag, a new platform, or a deliberate numerics change, said in the
  PR): `ASTRO_SIMCLR_RECORD_GOLDEN=1 uv run pytest tests/test_engine_regression.py` writes
  this platform's goldens instead of checking them; review the diff of
  `tests/fixtures/engine_golden/` before committing it. A new training flag adds its case
  to `FLAGS` (or `RJ23_CASES`) and records its golden (CLAUDE.md's checklist names the
  differential test's `_FLAGS`, which this replaces).
- **Resumes** (`tests/test_engine_resume.py`): each resume is compared with an
  uninterrupted plain run of the same config, which the golden set holds to what Lightning
  computed, bit for bit (the validations after the resume point, `metrics.json`, the
  predictions, the manifests): the paper classifier with C5-C7, and with C5-C7 and D3, from
  a mid-run `last.pt`; the three Lightning-bound resume tests of task 2.2a, ported (a
  mid-epoch `last` checkpoint, a resume at the horizon that still writes `best`, an
  augmented resume); the `rj23` schedule mid-run, after it finished (training nothing),
  and two validations, one validation and zero steps before a patience stop.
- **Stored runs** (`tests/test_lightning_retired.py`): `tests/fixtures/plain_run/` is a
  plain run of `configs/runs/carrasco2021.yaml` recorded before task 4.8; resolving the same
  config today gives the same config hash and run id, with the engine named or left to the
  new default. Stored Lightning runs: section `tta and ensemble_eval`.

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
  once and sliced, so a step makes no host-to-device copy. Its batches were `torch.equal`
  to Lightning's `DeviceTensorDataModule`'s until task 4.8 removed it (now tested against
  the tiled stream's rows, gathered by hand), and the stream keys are that loader's.
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
passes); `block` identical on every tier; `device` serving the tiled stream; exact
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

`src/astro_simclr/tasks/`: what the plain engine trains, and the runner. Task 4.6 wrote it
next to Lightning's `experiments.run_command`, which it replaced bit for bit (the 4.7 gate);
since task 4.8 it is the only runner: `astro-simclr run`, `train-supervised`, the queue's
`spec run-cell` and the preflight's CD21 dry run call `tasks.plain.run_plain`. **Nothing in
it imports Lightning** (tested in a fresh interpreter, with the CLI and the preflight).

| Module | What it does |
|---|---|
| `common` | The run checks, moved from `experiments` with their messages: `check_accelerator` (`AcceleratorError`, naming the torch build and the preflight, before anything is built), `dataset_path` (`data_dir/<dataset>` must be a prepared layout; the `DataError` says how to prepare it), `check_schedule` / `check_paper_variant` (the `carrasco2021` schedule, the paper model and its 2.5 flags go together; `image_size` is `paper_arch`'s side) and `check_split` (a split run is `train-supervised` on `carrasco2021`). `Settings.augmentation` is no longer checked: no runner reads it. |
| `paper` | `paper_crops(settings, data, names)`: the splits as the run's model sees them (the float path with its context channels, or the uint8 diagnostic, at `paper_arch`'s side), shared by `paper_inputs` and `ensemble-eval`. `paper_inputs(settings, data, split_run)`: the splits as `TensorSplit`s (float path or uint8 diagnostic, leaked Train stamps removed, a split run's roles, the C5-C7 fits from `with_train_fits`) and the `paper_data` entry. `load_backbone` (ConvNeXt weights, ADR 0022). `paper_model(settings)`: `build_paper_model` with every model field, the B9 `hybrid_*` shapes included, the one mapping that training and `ensemble_eval.member_net` share. `PaperNet` (the model under `model`, the loss as `criterion`, the D3 weight EMA as `ema`, run in `eval()` mode: the former `PaperClassifierModule`'s `state_dict` keys). `PaperTask`, the engine `Task`: a step plans the augmentation of step `step` on a host copy (`training.paper_augment.PaperAugment`), uploads and applies it, then the loss; `build_optimizer` with the configured betas over every parameter but the EMA's, the D3 `LambdaLR` if any; `after_step` steps it and updates the EMA; `state()` holds the scheduler's state (empty without one). `build_paper_task(settings, info)` builds it on the CPU (after the seed); `check_paper_task` runs its checks before the manifest. |
| `results` | `StageOutput` and `classifier_results`: the Validation / Test (or selection / evaluation) entries of `metrics.json` (with each stage's `ece`, plan row D3), `predictions_*.npz` and the confusion-matrix figures, moved from `experiments`, so a plain run writes the results a Lightning run wrote; `selection_entry` (`metrics.json["selection"]`). |
| `plain` | `run_plain(command, settings, resume=, split=, rj23=, on_step=, on_validation=, progress=, queued=)` (`queued`: task A3's `spec run-cell`, on both schedules): `train-supervised` of the paper classifier on `carrasco2021` (every 2.5 and D3 flag, split runs, `fp32_precision`). The checks, the data on the `device` tier, `start_run(..., engine="plain")`, `engine.fit`, the final evaluation of the scored weights (the restored best, or the last with `checkpoint_selection: final`), `metrics.json`, the manifest `finished`. `resume="auto"`: `OUTPUT_DIR/ckpt/last.pt`. With `schedule: rj23` and `rj23=RJ23Request(...)` (required there, refused elsewhere; `split` refused with it) it returns `tasks.rj23.run_rj23_plain`'s run. Refuses other commands, models and schedules (the legacy commands and the `stamp_classifier` / ResNet baselines: naming the git tag `lightning-final`), `devices != 1` and `limit_train_batches` / `limit_val_batches` (`check_plain_runtime`, both paths), and a Lightning checkpoint as `--resume` (naming the tag, which resumes it). |
| `rj23` | The `rj23` schedule (task 3.4b; ADR 0023; section `training`, "The rj23 schedule"): `RJ23Request(splits, mode, alerts)`, `rj23_inputs(settings, request)` (the layout, the alert table, the verified object splits, the protocol's parts, the run identity), `check_stream_state`, `keep_record`, `schedule_record`, `rj23_metrics`, and the plain adapter over `rj23_regime`'s objects as they are: `RJ23Task`, `RJ23TrainSource` (stream 10), `RJ23EvalSource`, `macro_f1_score`, `rj23_schedule`. `run_rj23_plain`: the inputs, the parts on the device, `start_run(..., engine="plain")` (a replication's splits as `whole`, a decision's as a screen), `engine.fit`, both keeps evaluated (`stop`: the last weights, `ckpt/last.pt`; `best`: `ckpt/best.pt`), `metrics.json` and the manifest. |

**Order of a run** (`run_plain`): the checks (accelerator, schedule, split, resume), the data
(`tasks.paper.paper_inputs`, on the `device` tier), the manifest (`running`), then
`engine.fit` (precision and flags, seed, the task built on the CPU, the move, the
optimizer), the final evaluation of the scored weights, `metrics.json` and the manifest's
`finished`. A run refused before the manifest exists never started and leaves none (task
A1).

**Outputs**, all under `output_dir`: `manifest.json` (task A1, section `runs`; `engine:
plain`), `scalars.jsonl`, `ckpt/{last,best}.pt` (section `engine`), `metrics.json` (with
`run`: run id, attempt and config hash), `predictions_{val,test}.npz` (or
`predictions_evaluation.npz` for a split run) and `confusion_matrix_<stage>.png`.
`metrics.json` keeps the keys of the Lightning runs, so the stored runs of both engines are
read and compared alike: command, seed, `best_epoch` / `last_epoch` (derived from the
steps, ADR 0014, C5), `best_step`, `last_step`, monitor and `monitor_best`, `checkpoints`
(`ckpt/best.pt`, `ckpt/last.pt`; `best` `null` under `checkpoint_selection: final`),
`versions` (`astro_simclr`, `torch`; no Lightning version), `accelerator` (the device type),
`machine` (`engine.record.machine_record`: `device`, `gpu` = the CUDA device name or
`null`, `torch`, `cuda`, `cudnn`, `platform`, `processor`, `python`; ADR 0010's
`fp32_precision`, `fp32_backends`, `compute_capability`; `engine: plain`), the data
(prepared path, source SHA-256, class counts per split from `meta.json`), the resolved
`config`, `resumed_from`, `timing` (`fit_seconds`, wall clock from the first step to the
end of the fit loop, validation included; `steps`, this fit's optimizer steps, fewer than
`last_step` after a resume; `seconds_per_step`; `null` when no step ran), `horizon` (the
final one), `paper_data` (`input`, `crop`, `nan_approximation`, `excluded_train`,
`n_train`, `n_tiled`, and with C5 / C6 `class_augment`: `min_r2` and each class's share
below it; with C7 `flux_rescale`: each class's share of failed PSF fits; with a backbone,
`backbone_weights`), `preset` (`presets.preset_report`: `name`, `faithful`, `deviations`,
`diagnostics`), `selection` (plan row D3) and the classifier results: `val_acc_best` (micro
accuracy on Validation at the scored weights), `test_acc_at_best`, `val_macro_f1`,
`test_macro_f1` and, per split (`evaluation.val|test`), the balanced accuracy,
`recall_per_class` (by class name, `null` for an absent class), the normalised and count
confusion matrices and the ECE, with `predictions` naming each predictions file
(softmax probabilities, labels and stamp index of the scored weights,
`protocol.predictions`). A split run has `protocol`, `selection_acc_best`,
`evaluation_accuracy` and `evaluation_balanced_accuracy` instead of the `val` / `test`
entries (section `protocol`). Non-finite numbers are written as `null`.

**Resume** (`--resume PATH|auto`) is exact: the stream is a function of the step, and
`ckpt/last.pt` holds the weights, the optimizer, the task state, the policy (horizon, best
weights and score) and every RNG stream (section `engine`). The engine refuses a Lightning
checkpoint, another run id (another config, data, machine class or split), another
precision or determinism policy and another stream; because the run id holds the config, a
resume cannot extend a run by raising `max_steps`: that is a new run.

**Tested.** The golden set and the resume tests (section `engine`) and
`tests/test_tasks_plain.py`: a plain run's outputs, every refusal (before the manifest, or
a failed attempt after it; a Lightning checkpoint, with the tag that resumes it), the
accelerator and resume resolution, `astro-simclr run` with `runtime.engine: plain`, the
Lightning-free imports, the timer and the preflight's CD21 run.
`tests/test_tasks_rj23.py`: a plain RJ23 run's outputs (checkpoints, both keeps'
predictions, `scalars.jsonl` from step 0, the manifest), `run_plain`'s refusals of a bad
request, a resume of another run, the sources and the stream check, Lightning-free imports;
`tests/test_rj23_training.py`: section `training`, "The rj23 schedule".

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
- `PRESETS["rj23-<scenario>"]` (`RJ23_PRESETS`, task 3.4b; ADR 0023), one per scenario
  of RJ23 (`full`, `cropped`, `lowres`, `multiscale`): `supervised_model: rj23`, the
  scenario, `rj23_bn_position: after_concat`, `rj23_large_values: paper`,
  `with_features: false`; `optimizer: Adam` (built as Keras' Adam) with Table 2's
  learning rate (`RJ23_LEARNING_RATES`: 7.589e-4, 3.4627e-4, 1.125e-3, 7.445e-4; a test
  keeps them equal to `rj23_model.RJ23_TABLE2`), batch 256, `num_workers` 0;
  `schedule: rj23`, `monitor: macro_f1_val`, validation every 500 steps, patience 5,
  `max_steps` 100000, no EarlyStopping, `rj23_keep: stop`; `fp32_precision: ieee`. The
  dataset and the splits are not part of a preset. One preset per scenario so that the
  learning rate is the preset's and another one is a recorded deviation.
  `configs/phase3/rj23-<scenario>.yaml` hold the same values with their provenance.
- `preset_for(settings)`: the preset of a `carrasco2021`-schedule run (by `with_features`),
  `rj23-<rj23_scenario>` for an `rj23` run, else `None`. `preset_report(settings)`:
  `name`, `faithful` (no preset key changed), `deviations` (key → `{value, preset}`) and
  `diagnostics` (the flags turned on; `carrasco2021` only), written to
  `metrics.json["preset"]`. Seed, paths, accelerator and `checkpoint_every_n_steps` are
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
  Augmentation names may be legacy aliases; they are passed through unchecked (the
  Lightning data modules that resolved them are at the git tag `lightning-final`).
- Since task 4.8 only `train-supervised` takes `--hparams`, and the files were tuned for
  models that train only at that tag (the `stamp_classifier` and ResNet baselines, the
  SimCLR commands); `COMMAND_DEFAULTS` and the mapping stay so that the legacy settings
  still resolve as they did.

## `reference` (task 1.7)

`src/astro_simclr/reference.py`, behind `compare-reference`. `REFERENCES` holds R1-R4
(plan, "Reference runs"): the accepted commands, the gate and, for the pass rule, the legacy
mean, sample std (ddof=1) and number of repeats. Each run directory's `metrics.json` is
checked (command, `supervised_model`, no duplicate seed) and `val_acc_best` aggregated
(mean, std with ddof=1, any n). Since task 4.8 the R1-R4 runs train only at the git tag
`lightning-final` (their commands and models went with Lightning); the gates still read
their stored `metrics.json`.

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
| `groups` | Phase 2: builds object groups from the prepared data (unlike task 3.2's `grouping`, which assigns given ids): `neighbour_pairs(ra, dec, radius_arcsec)` (every pair within the radius, inclusive; an exact dec-sorted window search with the haversine separation, numpy only, correct at the poles and across the RA wrap; refuses non-finite coordinates and `|dec| > 90`), `union_groups` (union-find, canonical ids: groups numbered by their smallest row), `position_groups` / `external_groups` (equal ids, e.g. `oid`) → `Grouping(method, radius_arcsec, group_of)` (`n_groups`, `multi()`), `split_positions(data, names)` (ra/dec de-standardised with M1's statistics), `grouping_for(data, names, radius_arcsec=1.5, group_ids=None)`, `load_group_ids(npz)` (one array per split, integers or strings, no pickles; returns the file's name and SHA-256); `match_any_within(ra, dec, ref_ra, ref_dec, radius_arcsec)` (task 5.1: which rows lie within the radius of any reference row, and the nearest one; the same dec-window search over the sorted reference, `O(N w)`) |
| `splits` | Format version 2 (grouped; version 1 still read). `make_splits(prepared_dir, seed=0, radius_arcsec=1.5, group_ids_file=None)`: groups the pool (Train ∪ Validation), then the grouped 250/250 screen of Validation (groups placed largest first into the half with room; a documented fallback trims both halves to a common per-class count), `screen_excluded_train` (the Train alerts grouped with Validation, out of the screen's training), and the class-stratified group 5-fold CV (greedy placement, largest group first, per-class spread at most the largest group's stamps of the class), each fold with an exact 100-per-class selection set drawn from whole groups outside the fold (the other stamps of a selection group are the fold's `unused`); leaked Train stamps are in no fold. At most 10 folds (`fold_of` is one digit per stamp). `write_splits` / `read_splits` (structure checks: sizes, disjointness, ranges, positive integers, the grouping block, and `check_group_integrity`: no group in two folds, in a fold's selection and training, or in both screen halves; `screen_excluded_train` consistent) / `verify_splits(splits, prepared_dir, group_ids_file=None)` (dataset identity by source and label SHA-256 and sizes, not the directory name; group integrity; the stored groups equal a fresh grouping; class balance; fold stratification within the grouped tolerance; `excluded_train` equal to a fresh leakage check). `ProtocolSplits.locate` maps pool indices to Train and Validation indices; `ProtocolSplits.screen_training_excluded()` |
| `audit` | `leakage_audit(prepared_dir, radius_arcsec=1.5, group_ids_file=None)` (CLI `leakage-audit`): groups all three splits together and reports the groups within each split and per split pair by class, the Validation and Test rows with a Train alert within the radius (or the same id), and the byte duplicates, as a JSON-ready dict; `format_audit` prints it |
| `calibration` | Plan row D3, numpy only: `expected_calibration_error(probabilities, labels, n_bins=15)`, the top-label ECE of Guo et al. 2017 (right-closed equal-width bins, float64), a secondary metric: `evaluation_entry`'s `ece`, `compare-variants` and `report-phase2` |
| `predictions` | `Predictions(probabilities, labels, index, n_classes=5)` (task 3.2: `n_classes` generalises the column count; 5 by default, so phase 2 is unchanged; the benchmarks pass 6 and index by `candid`), `save_predictions` / `load_predictions` (`.npz`, `allow_pickle=False`), `check_paired`, `ensemble` (mean probabilities per stamp), `seed_mean_correct` |
| `stats` | `mcnemar` (exact, integer arithmetic), `paired_bootstrap` (percentile CI of variant − baseline, stratified by class, metric `accuracy` or `balanced_accuracy`, own `default_rng(seed)`, memory-bounded blocks), `paired_bootstrap_macro_f1(pred_baseline, pred_variant, labels, n_classes)` (task 3.2: the same paired, class-stratified resampling for macro-F1; each resample recounts every run's confusion matrix as one matrix product of item multiplicities with (run, true, predicted) indicator columns; observed = mean-over-runs macro-F1 difference, the report passes one row per side, the seed-ensemble argmax; identical sides give exactly [0, 0]; `BootstrapCI.metric` is now `IntervalMetric`), `hierarchical_paired_bootstrap(pred_baseline, pred_variant, labels, n_classes, metric="balanced_accuracy"\|"macro_f1", groups=None, reference_counts=None)` (task 5.4: `(R, N)` predictions paired by label subset; each resample draws the subsets with replacement, then the objects within class (an object's items together, one class per object), shared by all subsets and both arms; statistic = mean over drawn subsets of the paired difference; optional prior-reweighted macro-F1; `_hierarchical_draws` fixes the draw order, which the golden tests replay; they also enumerate a tiny case exactly), `minimum_detectable_effect(n, discordant= \| rate=)` (2.80·SE, paired or unpaired), `balanced_accuracy`, `adoption_decision` (superiority: `adopt` only if the CI is above 0), `non_inferiority_decision(interval, margin=0.005)` (`adopt` if the CI lower bound is above −margin, `worse` if the CI is below it; `check_margin` refuses margins outside (0, 0.05], a likely points-for-fraction mistake), `screen_decision` (`drop` at 4 points or more, never adopt) |
| `split_runs` | Task 2.5-infra. `SplitRequest(file, fold)` (`fold=None`: the screen); `open_split_run(request, prepared_dir)` reads the file, refuses a fold it does not have (`ConfigError`) and runs `verify_splits` on the prepared data (a wrong fingerprint is a `SplitsError`); `SplitRun.parts()` gives `RunParts` (training, selection, evaluation as sorted pool indices, and `evaluation_index`: pool indices for a fold, Validation indices for the screen) and checks that they are pairwise disjoint and hold no excluded stamp; `SplitRun.describe()` is `metrics.json["protocol"]` (mode, fold, splits file and SHA-256, part sizes, `grouping`); the screen trains on Train minus `ProtocolSplits.screen_training_excluded()`; `SplitRun.identity()` (splits SHA-256, mode, fold) is stored in every checkpoint and checked on resume. `file_sha256` |
| `compare` | `compare_variants(baseline_dirs, variant_dirs, stage)`: loads each run's `metrics.json` and predictions file, refuses repeated seeds, runs on different data (`data.source.sha256`: a normal run's `index` is `arange(N)`, so the data identity and the labels are what show two runs saw the same stamps), unpaired runs and mixed machines or float32 precisions (unless allowed), reports McNemar on the seed ensembles, the bootstrap and each side's ECE per run (`run_ece`; secondary), always as REPORT; it points split runs to `--splits`. `screen_comparison` / `cv_comparison(..., rule="superiority"\|"non-inferiority", margin)`: the decision helpers, which check that the predictions cover exactly the screen's decision half or each fold's evaluation part (so the CV bootstrap is paired by pool index); `CVResult` holds the per-fold and pooled balanced accuracies, the CI, McNemar on the seed ensembles and the decision; the CV output adds the fold-to-fold std (ddof = 1) and the ECE per run (secondary, plan row D3). `compare_decision(baseline_dirs, variant_dirs, splits_file, rule, margin)` (CLI `compare-variants --splits`): loads split runs (`load_split_run`), checks the splits file's SHA-256, one mode, one data source and machine, the same (fold, seed) cells on both sides and every fold, then applies `cv_comparison` (verdict `ADOPT` / `NOT ADOPTED`) or `screen_comparison` (`KEEP` / `DROP`; `--rule` refused); `DecisionComparison.mcnemar` is the screen's McNemar on the seed ensembles (the CV's is `cv.mcnemar`) |
| `phase2_report` | Task 2.6. `build_report(ReportInputs)` → the markdown of `docs/results-phase2.md` (CLI `report-phase2`). `find_arms(dir)`: one arm per subdirectory, its runs at any depth (`DIR/<arm>/.../metrics.json`; a run directly under `DIR` is refused). Sections: the replication (per-seed table: the scored step, `metrics.json["selection"]["scored_step"]` or, before plan row D3, `best_step`, the scored weights' accuracies and the test ECE, from `metrics.json` or the predictions file, mean ± std with ddof = 1, and `reference.compare_reference("CD21", ...)` verbatim with the M1 oracle report, default `DIR/m1_oracle.json`); the screen (`compare_decision` per arm against `--baseline-arm`, rot4 by default: accuracies, McNemar, KEEP/DROP and the verbatim output); the CV (`compare_decision` with each arm's pre-registered rule, superiority or `--non-inferiority`; balanced accuracy per run, per fold and pooled from `CVResult`, the CI, McNemar, both verdicts from `stats.adoption_decision` / `stats.non_inferiority_decision` on the same interval, the std over folds and a calibration table (ECE per run; secondary), the verbatim output; `pooled_class_scores` gives the per-class recall and confusion matrix of every arm from the per-stamp seed-mean scores, so the mean recall is the pooled balanced accuracy); then throughput (`read_run_facts`: s/step, GPU, the manifest's `concurrent_jobs`, flagged when > 1, "not recorded" without a manifest) and provenance (git commits and config hashes of manifest runs; with `--registry`, indexed and stale runs by `metrics.json` SHA-256). Deterministic: sorted arms and runs, a seeded bootstrap, paths as given, no timestamps. No statistics of its own |
| `alerts` | Task 3.2. `AlertTable` (sorted by `candid`: `oid`, `candid`, `jd`, `label`, `label_names`, `group`, `group_key` `"oid"` or `"group"`), `AlertTable.from_rows` (checks: valid ids, unique positive `candid`, finite positive `jd`, labels in range), `sha256` (canonical rows by `candid`, the group only with a `group` column), `rows_of(candids)`; `read_alert_table(csv, label_names=BENCHMARK_LABEL_NAMES)` (untrusted input: `csv` module and a schema, errors name the line; extra columns ignored), `write_alert_table`; `label_conflicts` (groups with two labels); `first_stamps(table, rows)` (earliest `jd`, then smaller `candid`, per `oid`). `BENCHMARK_LABEL_NAMES` = R5's classes (the phase-2 five), `RJ23_LABEL_NAMES` = the five + `Satellite` |
| `grouping` | Task 3.2, on per-row arrays of any group ids (never builds groups): `group_rows` (ids, label, size, conflicts), `assign_groups` (per class, `default_rng([seed, stream, class])` shuffle, then largest-deficit assignment; refuses a class that cannot give every part a group; task 3.5: an optional `allowed` mask, the most constrained groups placed first, each to the allowed part furthest below target), `grouped_split` (refuse or drop conflicting groups), `grouped_halves` (50/50 selection/decision, own stream), `check_stratified` (no group in two parts; per class and part, within `+g` / `−(k−1)g` of the target), `stratification_bound`, `shared_groups` / `assert_no_shared_groups(..., allowed=CROSS_BENCHMARK_ALLOWED)` (the cross-benchmark rule: by default the plan's matrix, Train/Train and Validation/Test pairs allowed, Train with the other's Validation or Test refused; `allowed=()` refuses all nine; returns the shared-count matrix) |
| `object_splits` | Task 3.2. `make_object_splits(table, seed, fractions=(0.6, 0.2, 0.2), on_conflict="refuse"\|"drop", allowed=None)`: `grouped_split` by `table.group` (task 3.5: `allowed` maps groups to the splits they may be in, stored as `constraints`, format version 2; version 1 still read; the bound is verified only for classes without constraints), then `grouped_halves` of Validation; `ObjectSplits` (table identity, group key, label names, seed, fractions, conflicts, sorted group ids per split and half, class counts per part), `split_of` / `half_of` / `rows(table, part)`; `write_object_splits` / `read_object_splits` (structure: valid sorted ids, disjoint splits and halves, halves inside Validation) / `verify_object_splits(splits, table)` (fingerprint, group key, label names, conflicts, coverage, the bound, the counts). CLI `make-object-splits` |
| `benchmark_metrics` | Task 3.2, numpy only: `confusion_counts`, `confusion_percent` (rows to 100; an empty row stays zeros), `macro_f1`, `macro_f1_from_counts` (stacked `(..., C, C)`), `reweighted_counts(counts, reference)` (prior reweighting), `benchmark_metrics(labels, predicted, label_names, reference_counts=None)` → `BenchmarkMetrics` (accuracy, macro P/R/F1, `ClassScores` per class, counts, percent, reweighted macro P and F1; `as_dict`). Zero-division rule: 0/0 → 0, and classes absent from labels and predictions are left out of the macro means (`None` per class), as `training.metrics` (and torchmetrics
before it). `RJ23_CLASS_COUNTS`, `rj23_reference(label_names)` (the default reference: only proportions matter) |
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
  the run's seed, so a training seed never moves a split and a comparison is reproducible on
  its own (an exception to the "Randomness" convention above, for data definitions and
  statistics only).
- **Split runs** (task 2.5-infra): `tasks.plain.run_plain(..., split=SplitRequest)`
  (`train-supervised --splits FILE --fold K | --screen`, `carrasco2021` schedule only)
  verifies the file first, then builds the device data from Train and Validation only
  (`tensor_splits.protocol_splits`), trains on the training part, selects on the selection
  part, predicts the evaluation part as `predictions_evaluation.npz` and records
  `protocol`, `selection_acc_best`, `evaluation_accuracy` and
  `evaluation_balanced_accuracy` (no `val`/`test` entries). Before any step, a resume
  compares the identity stored in the checkpoint (the run id, which holds the split part,
  and the split run's identity: `engine.checkpoint.check_identity`) with the run's own, so
  a fold never continues from another fold's weights. The default path, and so the
  faithful preset, is unchanged. The end-to-end commands are in
  `docs/evaluation-protocol.md`.

## `tta` and `ensemble_eval` (plan row D4)

Seed ensembles and extended test-time augmentation (TTA), evaluated on the stored weights
of finished split runs, without retraining. Nothing here changes how a run trains or
evaluates itself: there is no `Settings` field, so every config computes what it computed
(no new config hash, no `NUMERICS_EPOCH` bump).

**`tta`** (`src/astro_simclr/tta.py`): pure, engine-free and deterministic (no
randomness, so no random stream). Modes `TTA_MODES = ("rot4", "d4", "d4_shift")`.

- *Where the averaging happens.* The paper model (CD21, `paper_arch: rot4`) is
  cyclic-pooling: its forward averages the 64-d **features** of the 4 rotations after fc1,
  before the head. The ConvNeXt rows pool their 4 rotations the same way, `p4` its 4
  orientations, `p4m` the 8 of D4 (`ARCH_SYMMETRY`: `c4`, `d4` or `none` for `single`).
  The B9 hybrids are `c4`: `hybrid` averages the features of its 4 rotations, and
  `hybrid_p4` pools its p4 orientations before the attention, with positional embeddings
  tied across C4, so a rotation only permutes its tokens (neither is reflection-invariant).
  A TTA view goes through the whole model, so its extra views are averaged at the
  **logit** level, outside it (`combine_view_logits`: the mean of the views' logits; the
  caller applies the softmax). Re-feeding the rotations to a C4-pooled model would only
  repeat its own feature mean, so `tta_views(mode, symmetry)` gives the cosets of the
  model's symmetry in D4:

  | Model symmetry | `rot4` | `d4` | `d4_shift` |
  |---|---|---|---|
  | `c4` (rot4, p4, ConvNeXt, hybrid, hybrid_p4) | 1 view | identity, flip (2) | 2 × 9 offsets (18) |
  | `d4` (p4m) | 1 view | 1 view | 1 × 9 offsets (9) |
  | `none` (single) | 1 view | the 8 dihedral views | 8 × 9 offsets (72) |

  The 4 internal rotations of the flipped stamp are its 4 reflections, so for the paper
  model `d4` covers the 8 elements of D4: feature-averaged within each view,
  logit-averaged across the two (not the same as a feature mean over all 8, which the
  head never saw).
- `rot4` (the default) is one identity view whose logits are returned untouched: the
  run's own evaluation, bit for bit (for `single`, one pass).
- `d4_shift`: `d4` at the 9 crop offsets `(dy, dx)`, `dy, dx ∈ {-1, 0, 1}`
  (`SHIFT_OFFSETS`): the 21×21 crop re-taken 1 px off centre from the 63×63 float stamp
  (`crop_then_normalise(offset=...)`, each crop normalised on its own pixels; no padding,
  the stamp has a 21-px margin). The offset set is D4-symmetric, so the result stays
  D4-invariant. `check_tta` refuses it for `paper_input: uint8`, the ConvNeXt rows (they
  see the whole stamp) and `context: avgpool63` (the context would not move with the
  crop; use `d4`).
- `View(rotation, flip, offset)`, `apply_dihedral(images, view)` (flip the columns, then
  `rotation` counter-clockwise quarter turns, `torch.rot90` as the model's own; on all
  channels, the context included), `tta_logits(forward, images, views)` (centre views
  only).

**`ensemble_eval`** (`src/astro_simclr/ensemble_eval.py`, CLI `ensemble-eval`):

- `read_member(run_dir)` reads the validated `manifest.json` and `metrics.json`;
  `check_members(members, splits_sha256, part)` refuses, before any data is read, a run
  that is not `finished`, not `train-supervised` of `paper_stamp_classifier`, of another
  fold / the screen or splits file, and members that differ in config hash, engine, data
  hash, machine class, numerics epoch, pretrained weights or training machine
  (`reference.machine_signature`), or repeat a run (`ConfigError`, exit 2).
- `member_net` rebuilds the model from the manifest's config (`tasks.paper.paper_model`,
  the training's own mapping of the settings to the model, the B9 `hybrid_*` shapes
  included: before the PR #60 review fix it dropped them and a B9 run with non-default
  shapes did not load; no backbone download) and loads the selected weights: `ckpt/best.pt`
  (plain, its run id checked) or `checkpoints/best.ckpt` (Lightning, read with
  `torch.load(weights_only=True)`, never through Lightning; the same `state_dict` keys as
  `PaperNet`). A Lightning checkpoint holds no run id, so what it holds is checked: its
  `global_step` is the run's `best_step`, its `augment_seed` the seed, and each saved
  hyperparameter the config's value (a copy from another run of the same config, seed and
  best step would pass). A plan row D3 run with `checkpoint_selection: final` is scored on
  its last weights: `ckpt/last.pt` (run id and `last_step` checked) or
  `checkpoints/last.ckpt` (its `global_step` checked against `last_step`); a run with
  `ema_decay` gets its `PaperNet.ema`, so the `eval()` forward runs the EMA weights, as
  the run did.
- `evaluation_parts`: the part's evaluation stamps from `paper_crops` and
  `protocol_splits` (pool index for a fold, Validation index for the screen), and Test
  only with `--test`; `shifted_images` re-crops them for `d4_shift` from the memory-mapped
  float32 stamps, 4096 at a time.
- `ensemble_probabilities` evaluates every member as its run did, offsets outer (the
  `d4_shift` crops are built once per offset for all members): `engine.open_eval` batches of the
  config's `eval_batch_size`, `eval()` and `inference_mode`, the config's float32
  precision and determinism (`DevicePolicy`), on the members' training device unless
  `--accelerator`; per batch, the views' logits are combined and soft-maxed as
  `PaperTask.eval_step`. With `rot4` the probabilities equal the run's own
  `predictions_evaluation.npz` (tested for plain weights, fold and screen, and on the
  stored Lightning run).
- The **ensemble** is `protocol.predictions.ensemble` (the float64 mean of the members'
  probabilities). The report has each member's balanced accuracy (and the run's stored
  one), their mean and std, the ensemble's evaluation entry and `ensemble − mean`.
- `--out FILE.json` is shaped like a split run's `metrics.json` for what
  `compare-variants --splits` reads (`seed`: the member's for one member, the sorted
  list for an ensemble; `machine`: the members' training machine; `data`; `protocol`;
  `predictions.evaluation` → `<stem>.predictions_evaluation.npz`), plus `format:
  astro-simclr-ensemble-eval` version 1, `tta`, `members`, `results`,
  `evaluation_machine`. The JSON file itself is a `RUN_DIR` of `compare-variants`.
  `_check_out` refuses, before any evaluation, an `out` inside a member's run directory
  (always: those are the runs' records) and an existing `out` or predictions file unless
  `--overwrite`; once everything is computed, the predictions are written first and the
  JSON last, atomically (`data.files.atomic_write_bytes`).

**Tested.** `tests/test_tta.py` (view sets, D4 invariance of `d4` for rot4 / single / p4 /
p4m / ConvNeXt-atto and of `d4_shift` from the stamps, the shifted crops, the refusals) and
`tests/test_ensemble_eval.py` (six tiny plain CPU split runs: `rot4` reproduces their
predictions bit for bit, the ensemble arithmetic, `d4` / `d4_shift` / `--test`, the output
read by `compare-variants --splits`, every refusal, the `--out` refusals and that a failed
evaluation writes nothing).

**Stored Lightning runs** (task 4.8; `tests/test_lightning_runs.py`):
`tests/fixtures/lightning_run/` is a small split run (fold 2, a B9 hybrid with non-default
shapes) trained by the Lightning engine before its removal: its `manifest.json`,
`metrics.json`, `predictions_evaluation.npz`, `checkpoints/best.ckpt` and the splits file
(`last.ckpt` is not kept; a test renames `best.ckpt` to it for the `final` case).
`scripts/characterize/lightning_run/generate.py` writes it, run from a checkout of the git
tag `lightning-final`. Without Lightning, `ensemble-eval`
rebuilds it (the hybrid shapes included) and reproduces its predictions (bit for bit on
the platform that trained it), checks the Lightning weights against the run, reads
`last.ckpt` for `checkpoint_selection: final` and refuses to mix it with a plain run;
`compare-variants --splits` loads it as a split run whose machine signature names its
engine; `report-phase2` reads its facts and provenance; the registry indexes it as
`engine: lightning`.

## `runs` (phase 4, task A0; ADR 0015)

`src/astro_simclr/runs/` defines the interfaces of run identity that the registry (A2), the
queue (A3) and the plain engine build on. Nothing in it imports torch. The runners
(`tasks.plain.run_plain`, `tasks.rj23.run_rj23_plain`; Lightning's `run_command` until task
4.8) call `record.start_run` and `RunRecord.end` around the fit (task A1). Runs that
Lightning wrote (`engine: lightning`) stay readable: their manifests, layout and
retention are unchanged.

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
    (ADR 0016), `backbone_weights: none` (ADR 0022) and, task 3.4b, `rj23_scenario: full`,
    `rj23_bn_position: after_concat`, `rj23_large_values: paper`, `rj23_keep: stop`.
  - `semantic_config_from_plain(command, view, complete=True)` and `semantic_diff` are the
    hook of the nested config (4.2): `config_tree.config_sha256` lowers to the flat view
    and hashes it there (ADR 0017).
  - The pinned hashes of the `carrasco2021` presets, of ten 2.5 variants (`feature_set:
    position` and `none` among them) and of the two
    legacy protocols are in `tests/fixtures/runs/canonical_configs.json`; regenerate them with
    `uv run python tests/_canonical_cases.py`, and only on purpose.
- **`identity`**: `RunIdentity` holds the config hash, `seed`, `split` (a fold, `"screen"`,
  or `None`), `data_sha256`, `splits_sha256`, `pretrained_sha256`, `machine_class`, `engine`
  (`lightning` for the runs written until task 4.8, or `plain`) and
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
  `predictions_*.npz` and `ckpt/{last,best}.pt`; the Lightning runs (written until task
  4.8) have `checkpoints/{last,best}.ckpt` and `lightning_logs/`. `classify_artefact(path)` gives a
  file's kind. `retention(manifest, paths)` is pure and decides keep or delete, with a
  reason, for each path:
  - an unfinished attempt keeps everything;
  - a finished one deletes `last`, and keeps `best` only for `pretrain`;
  - records, results, logs, figures and unknown files are kept.
- **`record`** (task A1): `start_run(command, settings, data=..., machine=..., split=...,
  object_splits=..., pretrained_file=..., resumed=...)` computes the identity
  (`object_splits`, task 3.4b: an `rj23` replication run's object-splits file, hashed as
  its splits with the split part `whole`; at most one of `split` and `object_splits`), with the machine class from
  the run's machine record (`engine.record.machine_record`) and `engine: plain` (the default
  since task 4.8), and writes `manifest.json` as `queued` then
  `running`. `RunRecord.end(status)` writes `finished`, `failed` or `cancelled`, once.
  Attempts, since `--output-dir` keeps only the latest attempt:
  - none, or a manifest of another run id: attempt 1;
  - `--resume` of an attempt left `running` (killed): the same attempt, through `queued`
    (like an expired lease);
  - the same run id otherwise: the next attempt.

  An unreadable manifest is a `ConfigError`, never overwritten. The git state and the
  `uv.lock` SHA-256 come from the source checkout (`source_root()`, `None` in an installed
  wheel). `concurrent_jobs` comes from `ASTRO_SIMCLR_CONCURRENT_JOBS` (default 1).
  `queued=QueueAttempt(attempt, machine_class, run_id)` (task A3) makes it a queued run:
  a run id other than `run_id` is refused before training; the
  attempt and the machine class are the queue's, a fresh directory starts that attempt,
  an unfinished manifest of the same run id and attempt continues it (with or without a
  `last` checkpoint), and anything else in the directory is a `ConfigError`.
- **`spec`** (task A3; ADR 0026; `docs/queue.md`): the experiment-spec format shared by
  the queue and the registry (A2). `parse_spec` / `load_spec` validate a YAML file
  (`astro-simclr-experiment-spec`, version 1: name, `train-supervised`, optional preset
  and shared settings, arms as overrides, seeds, optional splits with folds or the screen,
  machine class and uv extra, `engine` (`plain`, the default since task 4.8, or
  `lightning`; a run-id component, not a setting), `scratch`) and resolve every arm once (an `rj23` arm needs `engine: plain`
  and its object splits as `screen: true` or `whole: true`, split parts `screen` / `whole`), so a bad setting
  fails before anything is queued. `seed`, `data_dir`, `output_dir` and `pretrained` are
  reserved. `ExperimentSpec.spec_id` is the SHA-256 of the canonical JSON of the parsed
  content; `cells()` are arms × seeds × split parts (`Cell(arm, seed, split)`, key
  `<arm>/s<seed>/<f<k>|screen|all>`); `arm_settings` resolves a cell like
  `train-supervised` on the command line; `arm_config_sha256(arm)` is one config hash per
  arm; `expected_run_ids` gives each cell's run id (on the spec's engine) from the data
  and splits SHA-256. `engine: lightning` still parses, so that `runs compare` resolves the
  run ids of a spec that trained on Lightning, but `check_trainable` (`queue submit`, `spec
  run-cell`) refuses it, naming the git tag `lightning-final`. `engine` is required since
  task 4.8 (before it, a missing key meant `lightning`): a spec without it is refused with
  what to write, never read as another engine with other run ids.

## `queue` (phase 4, task A3; ADR 0026)

`src/astro_simclr/queue/` replaces the hand-written tmux/bash queues with experiment specs
(`runs.spec`), a durable SQLite queue on the GPU machine and ssh submission. The user guide
and the survival checklist are in `docs/queue.md`. Nothing imports torch at module level.

- **`location`**: `check_queue_location(path)` refuses a queue file on a WSL Windows drive
  (`/mnt/<letter>/`), on a mount whose type is in `UNSAFE_FSTYPES` (`9p`, `drvfs`, NFS,
  CIFS, sshfs, ...; from `/proc/mounts`) or in a synced folder (`SYNCED_FOLDERS`: Dropbox,
  OneDrive, iCloud, ...).
- **`store`**: `QueueStore` over `$ASTRO_SIMCLR_HOME/queue/queue.sqlite` (WAL,
  `synchronous=FULL`, `user_version` 1; a newer schema is refused). Tables `specs`,
  `jobs` (one row per attempt of one cell: run id, attempt, run directory, cwd and argv,
  status in the manifest's vocabulary, lease token / owner / expiry, host, boot id, pid,
  slot, resumes, cancel flag, exit code) and `peak_memory` (the largest measured peak per
  config hash and machine class). `claim` leases the head of the queue under
  `BEGIN IMMEDIATE` if the admission callback gives it a slot (first in, first out);
  `heartbeat`, `finish` and `requeue` name the lease token, so a worker whose lease was
  taken over writes nothing; `requeue` counts resumes and fails a job past
  `max_resumes` (5); `add_job` gives a run id its next attempt and skips one that is
  queued, running or finished unless `new_attempt`.
- **`admission`**: `Slot(name, gpu, capacity_bytes, max_jobs=1)`; `admits` and
  `choose_slot` are pure: at most `max_jobs` per slot, an empty slot always admits, a job
  joins others only if all of them and it have a measured peak and the sum fits under
  `capacity × 0.9`.
- **`job`**: the fixed entry point of the submitted commit. `expand` (`spec expand`)
  prints the spec id and each cell's run id and config hash as JSON, without torch.
  `run_cell` (`spec run-cell`) holds `DIR/.lock` (`flock`; exit 75 if held), refuses a
  dirty tree for a non-scratch spec (exit 2), exits 0 for a finished attempt and refuses
  an ended or foreign one, then trains: `tasks.plain.run_plain(..., resume="auto",
  queued=QueueAttempt(...))` (an `rj23` arm: `rj23=RJ23Request(object splits, "decision" |
  "replication")` instead of `split`). A spec with `engine: lightning` is refused before
  anything (`runs.spec.check_trainable`, also called by `queue submit`; task 4.8); SIGTERM, SIGINT and SIGHUP end it at once without writing
  (the manifest stays `running`, resumable like after `kill -9`). `QueueAttempt` carries
  the cell's run id, which `start_run` checks before the first step; at the end it writes
  `job_result.json` (peak CUDA memory).
- **`worker`**: `Worker(store, machine_class, slots, log_dir, ...)`; each `step` renews its
  leases (a lost one: kill the child, write nothing), handles cancel requests (SIGTERM,
  then SIGKILL after the grace), collects exits (`outcome`: 0 finished; 75 retry after a
  lease, not counted; a signal or `128 + signal` a counted requeue; anything else
  failed), reaps expired leases of other workers (killing the orphan's process group first
  when it is on this host and boot and its command line names the run directory, so a
  recycled pid is never hit), and claims while a slot admits. Jobs start with
  `Popen(argv, cwd, start_new_session=True)`, stdin closed, output appended to
  `logs/<job>.log`, `CUDA_VISIBLE_DEVICES` and `ASTRO_SIMCLR_CONCURRENT_JOBS` set.
  `shutdown` (SIGTERM, SIGHUP) records the jobs that already ended, sends SIGTERM to all
  others, waits one grace period for them together and requeues them, uncounted. The
  reaper re-reads a lease before killing and requeues only if it is still expired
  (`requeue(expired_only=True)`); a requeued job whose cancel was asked for is cancelled.
  `queue cancel --spec` needs a prefix matching exactly one spec (`store.spec_ids`).
- **`runner`**: `RunnerConfig` (`$ASTRO_SIMCLR_HOME/runner.yaml`: repo, data_dir,
  runs_root, uv, the machine classes it serves; a spec of another class is refused). `ensure_worktree` checks a commit out into `worktrees/<sha>` (detached;
  `git fetch origin` if missing). `enqueue` refuses a dirty worktree for a non-scratch
  spec, runs `uv sync --locked [--extra X]`, writes `specs/<spec id>.yaml`, runs the
  worktree's `spec expand` and queues one job per cell with the argv `<worktree>/.venv/
  bin/astro-simclr spec run-cell ...` into `runs_root/<spec name>/<run id[:16]>-a<n>`.
  Every command is an argv list.
- **`remote`**: `SshTransport(host)` runs an argv as one `shlex.join` string after `--`
  (ssh hands it to the remote shell) with stdin, and `pull`s with `rsync -a --partial`;
  host names and remote paths are checked. `submit` refuses a dirty tree, an uncommitted
  spec or an unpushed commit for a non-scratch spec, then sends `queue enqueue --sha HEAD -`
  with the spec on stdin. `sync` asks `queue paths`, rsyncs the run directories without
  checkpoints, locks or SQLite files, then the checkpoints the retention policy keeps for
  finished attempts, then the specs, and runs `runs reindex <dest>` (skipped with a note
  while the registry is absent).
- **`service`**: the systemd user unit (`Restart=always`, `KillMode=mixed`,
  `TimeoutStopSec=60`, `ExecStart` words quoted for systemd) and the tmux argvs, whose
  targets are always `=<name>` (exact match: `-t screen25` matched `screen25c`).
- **`cli`**: `spec check|expand|run-cell` and `queue init|enqueue|paths|worker|status|
  cancel|service|tmux|submit|sync`; the queue's errors print one line and exit 1.

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
  `<stage>.{accuracy,balanced_accuracy,macro_f1,ece}`, `monitor_best`, `best_step`,
  `scored_step` (plan row D3, when the run records `selection`), `last_step` and
  `seconds_per_step`. `IndexedRun.key` is `<run id[:16]>-a<attempt>` or
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
| `pilot` | `PilotConfig` (the run's whole definition; `epochs` or `steps`; task 5.1 added `tier="memmap"`, `archive`, `eval_data` and `host_cap_gb`, and a checkpoint written before them resumes with them at their defaults; the probe monitors' fields, left out of `to_plain` at their defaults), `load_pilot_data` (a layout: Train and Validation as float32 (N, 3, 63, 63); with a splits file, verified, pretraining leaves out `screen_training_excluded`. A corpus index file: the CD21 layout it excluded, the same way, for the evaluation, plus `CorpusPretrain(CorpusStamps, record)` after `check_archive`), the pretraining stamps (`TensorStamps(images, rows)` for `device`/`host`, `MemmapStamps` for `memmap`: a sorted read, the normalisation and a pinned tensor per batch; `_pretraining_stamps` picks them and refuses `device` for a corpus and a corpus above `host_cap_gb` on `host`; `_weak_rules` refuses an index below the plan's exclusion unless `allow_weak_index`; `restore` also compares the checkpoint's index SHA-256, exclusion SHA-256 and `n`), `BatchOrder` (a permutation per pass keyed `[seed, pass, simclr_batch_order]`, last partial batch dropped), `PilotRun` (model built on the CPU from the seed, LARS, the step: views, one forward of both views concatenated, loss, rate, backward, step), `diagnose`, `embed`, `dump_embeddings`, `train_pilot(config, out_dir, resume=, stop_at=)`. |
| `monitor` | The probe monitors of pretraining (both opt-in; see below). `budget_train_rows(budgets, part, repeat, wanted, n_train, usable, train_labels)`: the Train stamps of a stored label-budget subset (Validation pool indices and Train stamps outside `usable` dropped, the counts recorded); `eval_h(model, images, rows, device, batch)`: `h` of the eval view (centre crop, eval mode, `inference_mode`); `probe_balanced_accuracy`: the 5.4 probe (`protocol.probe.fit_probe`, `DEFAULT_WEIGHT_DECAY`, float64 L-BFGS) scored by balanced accuracy; `probe_beats_baseline` (strictly greater); `side_batch(seed, step, rows, size)`: the online probe's side-batch from `[seed, step, simclr_online_probe]` (stream 12); `OnlineProbe`: a zero-initialised `Linear(512, 5)` with its own Adam, `update` (encoder in eval mode under `no_grad`, the head on detached `h`, class-balanced cross-entropy), `take_train_accuracy`, `balanced_accuracy`, `state` / `load`. |
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

**Probe monitors** (`simclr.monitor`; tasks 5.0b/5.0c/5.3, ADR 0025 amendment of
2026-11-09). Both are off by default, and then the run's files are byte for byte those of
before (the record and the checkpoint config leave the monitor fields out at their
defaults). Both feed the probe **images only**: `h` (512-d) of the centred 57-pixel crop,
the network in eval mode, no augmentation and no alert feature. Both are **diagnostics and
an abort input, never a checkpoint selector** (5.0c: a fixed epoch count, the final
checkpoint is kept).

- `--probe-every K --label-budgets FILE` (with `--splits`): every K passes, at the abort
  check and at the final step, a diagnostics line adds `probe_b10_selection_bacc` and
  `probe_b100_selection_bacc` (`--probe-budgets`; `null` on the other lines). The probe is
  5.4's, trained on repeat 0 (`--probe-repeat`) of part `fold0` (`--probe-part`) of the
  stored label-budget file, which must be drawn from the run's splits file
  (`verify_label_budgets`). A CV training part is a part of the pool, so its subsets can
  hold Validation stamps, even of the screen's decision half (CD21 `fold0` at 100: 11, 6 of
  them in the decision half): only the subset's usable Train stamps are kept (CD21: 49 at
  10, 492 at 100; recorded in `pilot.json`). It is scored on the screen's selection half
  only, embedded once per point and shared with the kNN monitor. A fresh run first scores
  the random-init network the same way (the baseline, kept in the checkpoint). Cost on a
  loaded laptop CPU: about 10 s per point (the 250 selection stamps and about 500 Train
  stamps embedded, two L-BFGS fits).
- `--abort-if-probe-below-random EPOCH`: at the end of pass EPOCH the run stops unless the
  largest budget's probe beats the random-init one (strictly). `ckpt/last.pt` keeps the
  state, `pilot.json` records `status: aborted` and the check's values, no `final.pt` or
  embeddings are written, the CLI exits 0, and a resume is refused.
- `--online-probe`: a zero-initialised `Linear(512, 5)` head trained each step, before the
  training forward, on a side-batch (`--online-probe-batch`, 256) of the usable Train
  stamps drawn from stream 12, with its own Adam (`--online-probe-lr`, 1e-3) on
  class-balanced cross-entropy, on `h.detach()` from the encoder in eval mode under
  `no_grad`. The encoder's weights and gradients are bit-identical with it on and off (a
  test). Each diagnostics line adds `online_probe_train_acc` (the running accuracy on the
  side-batches since the previous line) and `online_probe_selection_bacc`. It is a
  100 %-label trend monitor that lags the final probe. With a corpus as DATA, its labels
  (and the periodic probe's) come from the evaluation layout. About 4.6 s per step on the
  laptop CPU against about 35 s for the training step at B = 256.
- With a monitor on, `pilot.json` adds `status` (`running`, `stopped`, `aborted`,
  `complete`; rewritten when the call ends) and `monitors` (inputs, the probe's rows and
  baseline, the abort outcome, the online head's settings), and the checkpoint a `monitor`
  entry (the baseline, the abort outcome, the head, its optimizer and running counts), so a
  resume continues exactly and writes no duplicate diagnostics line.

The pilot's protocol, end to end:

```bash
D=dataset/prepared/td_ztf_stamp_17_06_20_red-002; S=configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json
astro-simclr simclr-pilot $D --splits $S --output-dir runs/pilot/ssl --epochs 100 --device cuda \
  --probe-every 5 --label-budgets configs/label-budgets/td_ztf_stamp_17_06_20_red-002-grouped.json \
  --abort-if-probe-below-random 20 --online-probe
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
5. **Dry chain:** **CD21** (task 4.6), the only dry run since task 4.8: the `carrasco2021`
   preset on the plain engine (`tasks.plain.run_plain`), `steps+1` optimizer steps (`--steps`,
   20 by default) with one validation after the last and no horizon extension, at the
   preset's batch size (capped to the Train size) on the chosen accelerator, so a batch
   that does not fit the GPU fails here; on `red-002` with float32 stamps (`--synthetic`
   prepares that file with them). `--set` overrides come before the dry run's own. Without
   `--runs`, a dataset that is not prepared with float32 stamps is a SKIP (WARN); named in
   `--runs CD21`, it must run, or the step FAILs. The R1-R4 dry runs (the configs of
   `configs/reference/`: R3 pretrain, linear-eval and fine-tune, R1, R2, R4) trained on
   Lightning; task 4.8 removed them with `--reference-dir`, and naming one in `--runs` is
   refused (exit 2) with the git tag that keeps them, `lightning-final`. The preflight never
   imports Lightning (tested in a fresh interpreter).
6. **Throughput:** `ThroughputTimer` (engine-free: `tick(device)` after each step, through
   `fit(on_step=...)`; `reset()`) measures steps/s between consecutive steps (the first
   untimed; the device is synchronised so the clock measures work), with the real data
   path. The estimate is the initial horizon over the rate (30,000 steps: a lower bound,
   since a new best extends it), times 5 seeds for the full set; validation time is not
   included. With `--synthetic` (the four synthetic files, named as the received ones,
   `--n-per-split` 2400 by default, under the output directory) it uses the real Train
   counts of `docs/data.md` (`REAL_TRAIN_COUNTS`) to know the dataset.
