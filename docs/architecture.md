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
- Planned: `inspect-data`, `make-synthetic`, `prepare-data` (1.2); `pretrain`,
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

Planned: `src/astro_simclr/data/`. The legacy preprocessing (`repair_non_squared_stamp`,
`preprocess_stamps`, `img_float2int`), the one-view `StampDataset` and the paired
`ContrastiveDataset` on the `Augmentation` protocol, the lenient pickle loader (the only
place that unpickles), and the prepared `.npy` layout that training reads. Schemas:
`docs/data.md`.

## `losses` (task 1.3)

Planned: `src/astro_simclr/losses.py`. `NT_Xent` (mask from the actual batch; legacy
denominator by default, the paper's behind a flag, ADR 0002) and `P_stamps_loss`.

## `models` (task 1.4)

Planned: `src/astro_simclr/models.py`. The stamp encoders and classifiers, `CLR`, the
linear and three-layer heads, the historical `pstamps` network, and the encoder factory
(`stamp<k>`, `pstamps`, `resnet18/34/50/152`).

## `optim` (task 1.5)

Planned: `src/astro_simclr/optim.py`. LARS (device-agnostic; legacy `.ge(0)` by default,
`.gt(0)` behind a flag) and the cosine-with-warmup schedule that replaces timm.

## Lightning modules (task 1.6)

Planned (the module name is chosen by 1.6). `SimCLR`, `Linear_classifier`, `Fine_tuning`
and the supervised cross-entropy modules on Lightning 2, the `LightningDataModule`, and one
shared implementation of the metrics and the confusion matrix.
