<!-- Architecture of src/astro_simclr/: one section per module. Each phase-1 task edits only
     its own section, so parallel PRs do not conflict. Plan: docs/phases/01-runnable-package.md. -->
# Architecture

`src/astro_simclr/` is the typed, tested package that replaces the legacy research code at
the repository root (`utils/`, `models/`, `losses/`, the root scripts). Phase 1 ports the
legacy modules one task at a time; until a module is ported, its legacy file is the
characterization oracle and stays excluded from ruff, mypy and coverage.

```
cli ──► config, hparams (subcommand defaults, legacy key mapping)
 │
 ├──► data ──────► augmentations (protocol)      registry of transforms (1.5)
 │                                    ▲
 ├──► experiments (one training run) ──► training (Lightning 2) ──► models, losses, optim, metrics
 │                                          datamodules ──► data (prepared layout) + augmentations
 ├──► reference (compare-reference: the gates of R1-R4)
 └──► preflight ──► experiments, data (inspection, manifest)
```

Conventions shared by every module:

- **Settings** come from `config.Settings`, never from module-level constants.
- **Randomness** comes from `lightning.seed_everything(seed, workers=True)`; no module
  creates its own generators.
- **Devices:** no module hard-codes CUDA. The accelerator comes from the settings, and tensors
  follow their module's device.
- **Behaviour changes** against the legacy code are listed in the section of the module that
  makes them, with the legacy flag that restores the old behaviour, if any.
- **Typing:** mypy strict; `# type: ignore[override]` is allowed only on Lightning hook
  overrides (whose signatures Lightning types loosely).

## Environment and torch wheels

Runtime dependencies: torch, torchvision, lightning 2.x, torchmetrics, numpy, pyyaml,
matplotlib, tensorboard. torch and torchvision come from PyPI by default (MPS on macOS,
CUDA 13 on Linux), or from the PyTorch indexes through the conflicting extras `cpu` and
`cu126`. After `uv sync --locked --extra X`, run everything with `UV_NO_SYNC=1` or
`uv run --no-sync`. Design and rationale: [ADR 0001](decisions/0001-torch-wheels-and-accelerators.md);
`scripts/check-torch-wheels.sh` asserts it in CI.

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
- Later tasks add their fields here (for example the NT-Xent denominator flag of 1.3).
- **Task 1.7 fields** (flat, so every one can be `--set`): EarlyStopping `check_finite`
  (default true, Lightning's default) and `divergence_threshold` (null: never; R1 uses
  0.1); `limit_train_batches` / `limit_val_batches` (null: every batch; the dry mode sets
  2); and the hyperparameters of the training subcommands. Each subcommand reads only its
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
- `load_settings(path, overrides, defaults=...)`: `defaults` (a subcommand's legacy
  protocol) sits between the dataclass defaults and the file. `read_settings_file` reads a
  YAML mapping without checking keys; `settings_to_plain` gives the plain (YAML/JSON) form
  that `show-config` prints and the metrics JSON records.

## `cli` (task 1.1; subcommands from 1.2 and 1.7)

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
- **Training (1.7):** `pretrain`, `linear-eval`, `fine-tune`, `train-supervised` (see
  `experiments`). Each takes `--config`, `--set`, `--hparams FILE` (a legacy
  `results/hparams_best_*.yaml`, see `hparams`), `--data-dir`, `--output-dir`, `--seed`,
  `--accelerator`, `--resume PATH|auto`; `linear-eval` and `fine-tune` also `--pretrained
  PATH`. Settings resolve as: `Settings` defaults → the subcommand's legacy protocol
  (`hparams.COMMAND_DEFAULTS`) → `--hparams` → `--config` → `--set` → the explicit flags
  (later wins; `cli.training_settings`). The legacy file comes before the config so that a
  config can correct it (R1's `AdamW`). Dry mode: `--set max_epochs=1 limit_train_batches=2
  limit_val_batches=2` (not `fast_dev_run`, which writes no checkpoint).
- `compare-reference RUN_DIR... --ref R1|R2|R3|R4` (see `reference`): exit 0 for PASS or
  REPORT, 1 for FAIL.
- `preflight (--synthetic | --data-dir DIR) [--accelerator cuda] [--runs R1,R2,R3,R4]
  [--set ...]` (see `preflight`): exit 0 if no step failed, else 1.
  `scripts/gpu-preflight.sh` is a thin `uv run --no-sync` wrapper.
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
  0 or more workers (tested with 0 and 2).
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
- `m1_features`: `M1_FEATURE_NORM_STATS` (M1's 23 means and stds, recorded as constants
  with provenance; `scripts/characterize/data/m1_feature_stats.py` re-derives them with
  `pickletools`), `destandardise`, and `check_destandardisation`, behind
  `check-features DIR [--json]` (exit 1 on a failed test).

Neither module imports torch. The uint8 path (`images.npy`, `StampDataset`) is unchanged.

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
| `StampClassifier(drop_rate, with_features)` | `Stamp_classifier` | Carrasco-Davis et al. 2021; 23 features through `bn1`, 5 logits |
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
  `module.confusion_matrices["val"|"test"]`, and the labels `(y_pred, y_true)` in
  `module.predictions["val"|"test"]`, for the figures and metrics JSON of 1.7.
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
targets left out, like precision and recall).

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
  under `fork` (Linux) the map is shared. To revisit before R3 runs on macOS.

**Run helpers** (`runs.py`):

- `seed_run(settings)`: `L.seed_everything(settings.seed, workers=True)`.
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

**Tests** run on the CPU everywhere and on MPS/CUDA when available
(`tests/_training_support.py: ACCELERATORS`): `fast_dev_run` for every module and variant
(with/without features; stamp, `pstamps` and ResNet encoders) on the synthetic data
(`make_synthetic`, then `prepare`; `tests/conftest.py`), followed by a test pass; a seeded
2-epoch SimCLR run gives the same step losses and epoch loss twice (and a different seed
does not); the scheduler steps once per epoch; selection on `loss_train_enc` and `loss_val`;
EarlyStopping stops a flat run; every epoch and every validation pass starts from an empty
buffer (a module whose predictions flip between passes); downstream checkpoints reload.
Worker seeding (`num_workers=2`) is covered by the augmentation tests, not repeated here
because spawned workers cost about 25 s on macOS.

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
2. `dataset_path`: `data_dir/<dataset>` must be a prepared layout (`DataError` says how to
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

Outputs, all under `output_dir`: `checkpoints/{best,last}.ckpt`,
`lightning_logs/version_N/` (TensorBoard), `metrics.json`, and for the classifiers
`confusion_matrix_{val,test}.png`. `metrics.json`: command, seed, `best_epoch` (0-based),
`last_epoch`, monitor and `monitor_best`, versions, accelerator, the data (prepared path,
source SHA-256, class counts per split from `meta.json`), the resolved `config`; the
classifiers add `val_acc_best` (micro accuracy on Validation at the best checkpoint: with
the monitor `accuracy_val`, the best validation accuracy over epochs, the metric of the
legacy tables), `test_acc_at_best`, `val_macro_f1`, `test_macro_f1` and, per split, the
normalised and count confusion matrices. Non-finite numbers are written as `null`.

**Resume** restores the epoch counter, optimizer, LR schedule and the EarlyStopping /
checkpoint state (tested: 2 + 2 epochs equal 4 in epoch, global step, lr and `wait_count`).
It does not restore the global random generators, so the later epochs' augmentations and
shuffling differ from an uninterrupted run.

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
  lines, so an R4 run given as R1 is flagged.
- **R4:** report only. R1 also prints the Carrasco-Davis et al. 2021 figures as context.

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
   not fit the GPU fails here. Without `--runs`, a run whose dataset is not prepared is a
   SKIP (WARN); a run named in `--runs` must run, or the step FAILs.
6. **Throughput:** `ThroughputTimer` measures steps/s between consecutive batches of an
   epoch (first batch untimed; the device is synchronised so the clock measures work), with
   the real DataLoader. The estimate per run is `(n_train // batch) * max_epochs / rate`,
   times 5 seeds (R1, R2) for the full set, with R3's linear evaluation summed over the 1,
   10 and 100 % files; an upper bound, without validation time. With `--synthetic` (the
   four synthetic files, named as the received ones, `--n-per-split` 2400 by default,
   under the output directory) it
   uses the real Train counts of `docs/data.md` (`REAL_TRAIN_COUNTS`).
