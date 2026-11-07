<!-- Architecture of src/astro_simclr/: one section per module. Each phase-1 task edits only
     its own section, so parallel PRs do not conflict. Plan: docs/phases/01-runnable-package.md. -->
# Architecture

`src/astro_simclr/` is the typed, tested package that replaces the legacy research code at
the repository root (`utils/`, `models/`, `losses/`, the root scripts). Phase 1 ports the
legacy modules one task at a time; until a module is ported, its legacy file is the
characterization oracle and stays excluded from ruff, mypy and coverage.

```
cli ──► config
 │
 ├──► data ──────► augmentations (protocol)      registry of transforms (1.5)
 │                                    ▲
 └──► training (Lightning) ──► models, losses, optim
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
- Planned: `pretrain`,
  `linear-eval`, `fine-tune`, `train-supervised`, `compare-reference`, `preflight` (1.7).

## `augmentations` (protocol in 1.1; transforms in 1.5)

`Augmentation`, a runtime-checkable `Protocol` with `view(img)` (one augmented view) and
`pair(img)` (two independent views, the SimCLR positive pair). Input: a `Stamp`, a uint8
array of shape (H, W, 3) (the output of the legacy `img_float2int`). Output: float tensors
of shape (3, H', W'). The datasets (1.2) depend only on this protocol, never on a concrete
transform.

Task 1.5 adds the registry keyed by the legacy names (`simclr`, `simclr2`, `astro7`,
`astro8`, `rotation_v3`, `without_aug`); any other legacy name fails with an error that
says it is deferred.

## `data` (task 1.2)

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
- `synthetic`: `make_synthetic` writes the four pickles with the real containers and
  dtypes, and a rotation-invariant class pattern inside the central 27×27 crop (a
  nearest-centroid classifier gets ≥ 0.95). Its numpy `Generator` is seeded explicitly,
  because it builds files rather than training randomness. The same seed gives the same
  bytes on the same machine and numpy version (no fixture compares them across machines).
- `inspection`: `summarize` / `format_summary`, behind `inspect-data [--json]`. Inspecting
  is also a full check (schema validation, or `meta.json` plus hashes). The preflight's
  data manifest (1.7) is built from these summaries.

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

Planned: `src/astro_simclr/optim.py`. LARS (device-agnostic; legacy `.ge(0)` by default,
`.gt(0)` behind a flag) and the cosine-with-warmup schedule that replaces timm.

## Lightning modules (task 1.6)

Planned (the module name is chosen by 1.6). `SimCLR`, `Linear_classifier`, `Fine_tuning`
and the supervised cross-entropy modules on Lightning 2, the `LightningDataModule`, and one
shared implementation of the metrics and the confusion matrix.
