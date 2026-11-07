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
