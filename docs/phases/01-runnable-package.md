<!-- Phase 1 of the roadmap (docs/phases/README.md); owned by the integrator. -->
# Phase 1: a runnable, tested package

## Goal

Make the legacy research code runnable, and prove it, on today's stack (Python 3.12,
Lightning 2.x, current PyTorch). There are three levels of proof:

1. **Smoke run, on any machine.** On a laptop without a GPU (macOS arm64, CPU or MPS) and in
   CI, the full chain runs on a small synthetic dataset: SimCLR pretraining, then linear
   evaluation, then fine-tuning, plus the supervised cross-entropy baseline. It runs from
   the `astro-simclr` CLI and is covered by `scripts/check.sh`.
2. **CUDA run without the real data.** `scripts/gpu-preflight.sh --synthetic` passes on any
   CUDA machine. A free Colab or Kaggle T4 is enough, and the user can do it as soon as task
   1.9a (the runbook with the Colab/Kaggle recipe) is merged. This separates "CUDA works" from "the data works".
3. **Real run.** On a CUDA machine with the real ZTF pickles, the
   [reference runs](#reference-runs) are started with `uv sync --locked [--extra cu126]`,
   followed by `--no-sync` commands and a data path. Phase 1 makes this possible and
   documents it. Running it to completion is a manual acceptance step (task 1.9b), because
   the development machine has no GPU.

Out of scope, recorded in the [backlog](#backlog-for-later-phases): Ray Tune searches,
embedding visualization, the augmentation variants no phase-1 configuration uses, the
notebooks and new data sources.

## Starting point (facts checked in the code, 2026-11-07)

### Stack and code

- `src/astro_simclr/` holds only the template's `greet` example. All the real code is in
  the legacy tree (`utils/`, `models/`, `losses/` and root scripts, 7,290 lines), which is
  excluded from ruff, mypy and coverage.
- `pyproject.toml` declares no runtime dependencies. The legacy environment (`env.yml`) pins
  Python 3.8, `pytorch-lightning==1.4.5`, `torchmetrics==0.6.0`, timm 0.6.x and Ray 1.x
  (`ray.tune.suggest`). These APIs were removed in Lightning 2 (`*_epoch_end` hooks,
  `gpus=`, the private method behind `ModelCheckpoint_V2`), in torchmetrics ≥ 0.11 (`task=`
  is now required), in timm 1.x (`CosineLRScheduler.get_epoch_values`) and in Ray 2
  (`ray.tune.suggest` became `ray.tune.search`). This is a port, not a copy.
- Every module logs with `self.logger.experiment.add_scalar` (41 calls in
  `utils/training.py`). That works only with TensorBoardLogger.
- `utils/config.py` hard-codes `gpus: 1` and `model_path: ../weights`. (`utils/metrics.py`
  defaults to CUDA, but nothing imports it.)
- All the augmentations phase 1 needs are pure torchvision. `albumentations` and `cv2` are
  used only by the deferred variants (`utils/transformations.py:344` onwards).
  `RandomResizedCenterCrop` draws from Python `random`, and `GaussianBlur` draws from
  `np.random`.
- `image_size` is a centre crop (`Resize_img` = `CenterCrop` + `ToTensor`).
  `Stamp_classifier` and `Stamp_encoder` only work at 27×27: `fc1` is hard-coded
  (`models/models.py:100` and `:179`). The ResNets also accept 63.

### Data

The data is in `dataset/`, which is not versioned and not on this machine. Nobody has
inspected the real files: the schema below is inferred from the code.

- **Classifier pickles** `td_ztf_stamp_17_06_20{,_sup_1,_sup_10}.pkl` (100 %, 1 % and 10 %
  of the labels) have the structure
  `{"Train"|"Validation"|"Test": {"images", "labels", "features"}}`:
  - stamps are 63×63×3 floats in [0, 1] (science, template and difference);
  - 5 classes (`AGN, SN, VS, Asteroid, Bogus`);
  - 26 features, of which 13–15 are dropped, leaving 23.

  The label-fraction splits are frozen in the files.
- **Pretraining pickle** `td_ztf_stamp_simclr_300.pkl`: `Train` contains *only*
  `images`, a Python list of float64 arrays. `Validation` and `Test` are copied from the
  classifier pickle (`utils/simclr_dataset.py:23`). It was built from
  `/home/shared/pickles/` on a lab server, and that script is its only record. It holds
  about 300k alerts (`31b10c9`), so 14–29 GB as float32/float64. `pickle.load` needs all
  of that in RAM; free Colab has about 12.7 GB. Both legacy datasets convert stamps to
  uint8 (`img_float2int`) as their first transform.

### Reproducibility

- The only seed is `np.random.seed(0)`, set at import time in `utils/gaussian_blur.py:7`.
  Runs are repeated through Ray's `Repeater`. The legacy results are therefore not
  bit-reproducible.
- The settings that define the reference experiments are hard-coded in the entry points,
  not in the YAML files: max_epochs, EarlyStopping patience and min_delta, the checkpoint
  monitor (`accuracy_val`), the eval batch size (100), `drop_last=True` and
  `benchmark=True`. The values differ between tuning (`hyperparams_*.py`) and repetition
  (`iter_*.py`) runs.

### Known defects

Each one needs a test. "Legacy flag" means that the fix changes results, so the legacy
behaviour stays the default.

| Defect | Where | Phase-1 handling |
|---|---|---|
| `NT_Xent` builds its negative mask only for `batch_size` and 100; any other batch size fails | `losses/losses.py:53-54` | fix (mask from the actual batch) |
| `NT_Xent` uses `exp(0)` instead of `exp(s_pos)` in the denominator | `losses/losses.py:121-124` | legacy default, paper formula behind a flag (user decision, 2026-11-07) |
| The SimCLR stamp branch reads `encoder.bn1`, which `Stamp_encoder` lacks, so *every* stamp encoder fails; `pstamps` (the name in `hparams_best_simclr.yaml`) also fails to parse | `utils/training.py:1267-1273` | port `pstamps` = `P_stamps_net_SimCLR_v2(k=1.0)` from `b357b43^:models/models.py` |
| LARS calls `g_norm.get_device()`, which returns -1 on CPU, so it crashes; it would also map MPS to `cuda:N` | `utils/lars.py:118` | fix (device-agnostic) |
| LARS tests `w_norm.ge(0)` / `g_norm.ge(0)`, which are always true: a zero gradient gives NaN/inf, and a zero weight gives trust ratio 0, so zero-initialised BN biases never update | `utils/lars.py:120-122` | legacy flag (`.gt` changes ResNet results) |
| `configure_optimizers` only handles `LARS`; any other name ends in `UnboundLocalError` | `utils/training.py:1632` | fix (error that lists the valid names) |
| The legacy linear evaluation passes `softmax(logits)` to `P_stamps_loss`, which applies its own softmax (a double softmax, which flattens the gradients) | `utils/training.py` (`Linear_classifier`) | kept for equivalence in task 1.6 (`LinearEvalModule`); a correct-loss flag belongs to the phase that reports SSL linear-eval numbers |
| `SimCLR` stores `method` but ignores it; SupCon existed and was removed in `a187828` | `utils/training.py:1236` | backlog |
| `hparams_best_ce.yaml` says `Adam`, but every tuned trial behind it used `AdamW` (hand edit in `468344f`; the search space is `['AdamW', 'SGD']`) | `results/hparams_best_ce.yaml` | open question; the reference run uses AdamW |
| `hparams_best_simclr.yaml` does not come from the current `hyperparams_clr.py` search space, and it carries classifier keys that `SimCLR.__init__` does not accept. Its values look like a hand edit of the only recorded 300k-alert run (`31b10c9:iter_clr_stamps_300.py`: batch 550, `lr_encoder` 0.258, `lr_classifier` 0.0196, 850 epochs, patience 100), with the decimal points shifted (12.258 and 0.001956, batch 150). `iter_clr.py` has a third variant (1.258, batch 450) | `results/`, `iter_clr.py:110-125` | R3 uses the `31b10c9` settings; open question; key mapping per subcommand |
| The CE numbers come from a code version that accepted `balanced_batch`, which the current `Supervised_Cross_Entropy` does not. The ResNet tuning call (`max_epochs=220`) is commented out | `hyperparams_ce.py`, `hyperparams_ce_resnet.py:272-277` | R1/R2 configs state the assumptions (`balanced_batch=False` has no effect; R2 max_epochs = 220) |
| Datasets and checkpoints are pickles, and YAML is read with `FullLoader` | `utils/dataset.py`, `linear_classifier.py:127` | trust note; `yaml.safe_load` |

### Reference numbers in `results/`

- **CE stamp classifier:** 0.9496 ± 0.0022 validation accuracy (`summary_tuning_ce.csv`
  row 10: AdamW, batch 45, lr 0.002495, β 0.0161, dropout 0.2, features, 5 repeats). It was
  measured under the tuning protocol of `hyperparams_ce.py` (max 220 epochs, patience 70,
  best validation epoch).
- **CE ResNet-18:** 0.9027 ± 0.0099 (`summary_tuning_ce_resnet.csv` row 6, which matches
  `hparams_best_ce_resnet.yaml`: rotation_v3, 63 px, no features).
- **External reference (context, not a gate):** Carrasco-Davis et al. 2021 (AJ 162, 231,
  arXiv:2008.03309) is the ALeRCE stamp classifier on the same ZTF 5-class data (51,594
  alerts; validation and test balanced at 100 per class; 21×21 crop zero-padded to 27; 23 of the 26 alert features). It
  reports validation accuracy 0.950 ± 0.005 and **test accuracy 0.941 ± 0.004** (0.883
  without features). R1 reports its test accuracy next to that figure. Later ALeRCE work
  (multi-scale stamps, Reyes-Jainaga et al. 2023) uses another class set and split, and no
  public SSL result on these stamps was found. The labelled dataset is not publicly
  downloadable (the original repo is gone).
- **SimCLR / linear evaluation:** there is no number in the current tree. Older tables, from
  another code version, are in git history (`5dc07f5^:results/summary_tuning_simclr.csv`).

## Definition of "equivalent" to the legacy code

1. **Unit level (CI).** For fixed inputs, the migrated preprocessing, losses, LARS step,
   LR schedule and model forward passes match characterization fixtures.
   - **Fixtures:** small `.npz` files under `tests/fixtures/`, each recording the legacy
     commit SHA and the torch version.
   - **Generators:** `scripts/characterize/`, which is excluded from ruff, mypy and pytest.
     They run legacy code in an isolated environment, with timm 0.6.x where needed and
     `get_device` monkeypatched for LARS.
   - **Tolerances:** `atol=1e-6` for losses, preprocessing and LARS on small tensors;
     `rtol=1e-4, atol=1e-5` for conv-net forward passes, because fixtures generated on
     arm64 are checked on x86 CI.
   - **Behaviour changes:** listed in the PR and in `docs/architecture.md`.
2. **Metric level (manual, GPU, task 1.9b).** The metric is the best validation accuracy
   over epochs (micro), as in the Ray Tune tables. σ is the sample standard deviation
   (ddof=1), as in the legacy tables.
   - **Full set (5 seeds):**
     `|mean_new − mean_old| ≤ max(2·sqrt(σ_old²/5 + σ_new²/5), 0.01)`.
   - **Minimal set (1 seed):** `|x − mean_old| ≤ max(2·σ_old·sqrt(1 + 1/5), 0.01)`.
   - Test accuracy at the best epoch is reported too.
   - `astro-simclr compare-reference` (task 1.7) computes all of this. SimCLR has no reference, so it gets
   a sanity bound (see [Reference runs](#reference-runs)) and its result becomes the first
   recorded reference.

## Reference runs

These runs are defined here, so that task 1.9b can be checked. Each config lists the
protocol: split, max_epochs, patience, min_delta, monitor and checkpoint selection.

| Id | Run | Labels | Seeds (full / minimal) | Gate |
|---|---|---|---|---|
| R1 | CE stamp classifier, row-10 config (AdamW), tuning protocol (max 220 epochs, patience 70, `check_finite`, `divergence_threshold=0.1`) | 100 % | 5 / 1 | pass rule above vs 0.9496 ± 0.0022 |
| R2 | CE ResNet-18, `hparams_best_ce_resnet.yaml`, max 220 epochs (assumed) | 100 % | 5 / 1 | pass rule above vs 0.9027 ± 0.0099 |
| R3 | SimCLR pretrain with the `31b10c9` settings (`pstamps`, batch 550, LARS lr 0.258, temperature 0.1791, 850 epochs, patience 100 on the train loss, `check_finite`), then linear eval with R1's classifier hyperparameters (AdamW), `with_features: false`, `augmentation: without_aug`, max 220 epochs, patience 40 | 1, 10 and 100 % | 1 | finishes without NaN; accuracy above the majority-class baseline at every fraction; reported next to R4 |
| R4 | CE stamp classifier, R1 config with `with_features: false` (same feature setting as R3) | 1 % | 1 | reported (the comparison point for R3) |

All reference configs live in `configs/reference/` with a provenance comment. They set:
- `accelerator: cuda`, so a missing GPU fails fast instead of silently training on CPU;
- `devices: 1`, because `NT_Xent` has no all-gather and DDP would change the loss.

`docs/running.md` estimates GPU-hours for the full and the minimal sets, using the
throughput the preflight measures (steps/s × dataset counts × epochs). R3 needs the
pretraining set prepared as `.npy` (task 1.2). `docs/running.md` states its RAM and disk
needs.

## Tasks

Each task is one PR off `main`, small enough for one development agent. 1.2–1.5 run in
parallel once 1.1 is merged.

**Portability rule (user requirement, 2026-11-07; applies to every task):** the same code
runs on CPU, Apple MPS and NVIDIA CUDA, on arm64 and x86_64, with no code change. Only the
`accelerator` setting changes (`auto` by default).
- No device names (`"cuda"`, `.cuda()`, `get_device()`) outside the Lightning `Trainer`
  configuration. Tensors follow their inputs (`device=x.device`), and models are moved by
  Lightning.
- No numerical result may depend on the CPU architecture:
  - fixtures compare values with the plan's tolerances;
  - seeded randomness that is not portable (torch's vectorised CPU `normal_` differs
    between x86 and arm64, as PR #3 showed) is never compared against a fixture.
- CI runs `scripts/check.sh` on `ubuntu-latest` (x86_64) **and** `macos-latest` (arm64),
  both with the `cpu` extra. A test that touches devices runs on CPU everywhere and on MPS
  or CUDA when available (`pytest.mark.skipif`), never only on one of them.
- The CI matrix is added by the next task that touches `.github/workflows/ci.yml`, or as a
  small integrator-dispatched chore before task 1.6, whichever comes first.

**Exception to the CLAUDE.md refactor rule (stated here, and to be repeated in each PR):**
the legacy modules stay in the tree and in `extend-exclude` until 1.8, as the
characterization oracle. Before 1.8, `main` is tagged `legacy-final`.

### 1.1 Foundations (blocks all others) — done (PR #1, merged as e8b9249, 2026-11-07)

**Dependencies**

Add with `uv add`: `torch`, `torchvision`, `lightning` (2.x), `torchmetrics`, `numpy`,
`pyyaml`, `matplotlib` and `tensorboard`. Nothing else until a phase-1 module imports it:
no timm, albumentations, opencv, pandas, scikit-learn, seaborn or python-box.

**ADR `0001-torch-wheels-and-accelerators.md`**

This design was verified with uv 0.11 on toy projects (plan review, round 2).

- The default is PyPI torch: MPS on the Mac, and CUDA 13 on Linux, which needs NVIDIA
  driver ≥ 580.
- There are **two** extras, `cpu` and `cu126`, declared in `[tool.uv] conflicts` and
  mapped through `[tool.uv.sources]` for both `torch` and `torchvision`. One extra alone
  cannot be locked, and without the conflict the default install silently resolves to CPU
  wheels.
- `torch` and `torchvision` are pinned to the newest minor version that has CPU, CUDA-12
  and CUDA-13 builds (today `torch>=2.14,<2.15`, `torchvision==0.29.*`). Every fork runs
  torch 2.14.1 (verified); torchvision may differ by a patch (0.29.1+cu126 vs 0.29.2),
  which is accepted.
  `dependabot.yml` ignores torch/torchvision minor bumps; they are raised by hand with an
  ADR note.
- Markers-only is rejected, because CI and the GPU machine are both linux x86_64.
- **Sync discipline:**
  - `uv run` re-syncs the default extras, which undoes `--extra cpu` and `--extra cu126`.
  - So everything after the first `uv sync --locked --extra X` runs with
    `UV_NO_SYNC=1` or `uv run --no-sync`.
  - The machine's driver picks the extra: driver < 580 → `--extra cu126`.

**CI**

- `ci.yml` runs `uv sync --locked --extra cpu`, then `scripts/check.sh` with
  `UV_NO_SYNC: "1"`.
- Assertions on exact strings:
  - The default `uv export --locked` contains `torch==X` with no local `+` label, so
    pip-audit audits it.
  - `uv sync --locked --dry-run --python-platform x86_64-manylinux_2_28` lists `nvidia-`
    packages.
  - The same command with `--extra cu126` lists `+cu126`.

  These dry-runs download nothing.

**Package skeleton**

- `src/astro_simclr/config.py`: typed settings (a dataclass, loaded from YAML with
  `yaml.safe_load`, overridable from the CLI). It holds `accelerator` (`auto`), `devices`,
  `num_workers`, `data_dir`, `output_dir`, `seed`, `deterministic`, and the training
  protocol: max_epochs, patience, min_delta, monitor, eval batch size.
- `src/astro_simclr/cli.py`: a stub of `[project.scripts] astro-simclr` with a `main(argv)`
  entry point. Later tasks add subcommands.
- `docs/architecture.md`: one section per module, so that parallel PRs do not conflict.
- The `Augmentation` protocol (`view(img)` and `pair(img)`), which 1.2 and 1.5 both build
  on.
- Remove the `greet` example.

**Repository**

- `.gitignore` gains `outputs/`, `lightning_logs/`, `tb_logs/` and `*.ckpt`.
- Typing policy: `# type: ignore[override]` is allowed on Lightning hook overrides only.

**Acceptance**

- `uv sync --locked` succeeds on macOS arm64, and `uv sync --locked --extra cpu` succeeds
  on `ubuntu-latest`.
- CI logs show no `nvidia-` download.
- A Dependabot-style `uv lock --upgrade` succeeds, shown in the PR.
- `scripts/check.sh` is green.
- CI wall time, measured with the current `enable-cache: false`, is recorded in the PR and
  stays under 10 minutes.
- pip-audit is green and covers torch.

### 1.2 Data: schemas, loader, inspection, synthetic data (after 1.1) — done (PR #5, merged as 5523547, 2026-11-07)

- `docs/data.md`: per-file schemas (classifier vs pretraining), file names, provenance and
  size, and the manifest workflow.
- `src/astro_simclr/data/`:
  - Move in `repair_non_squared_stamp`, `preprocess_stamps` and `img_float2int`.
  - Port `StampDataset` (one view) and `ContrastiveDataset` (pair) on the `Augmentation`
    protocol.
  - A lenient pickle loader that accepts lists or ndarrays and fails with named errors
    (file, split, key). It applies `img_float2int` to the *stored* dtype before any
    coercion: converting float64 to float32 first changes the rounding of some pixels
    (59 out of 10M uniform values, verified).
  - Pickle loading is confined to one function. Its docstring states the trust assumption,
    and it carries a scoped `# noqa: S301`.
- CLI subcommands:
  - `inspect-data PATH` prints keys, shapes, dtypes, counts per class, file size and
    SHA-256.
  - `make-synthetic --out DIR --n-per-split N --seed S` writes all four pickles with the
    real per-file schemas. The stamps carry a strong class-dependent pattern.
  - `prepare-data PKL --out DIR` runs once, on a trusted machine that holds the pickles
    and has the RAM to unpickle them (the preflight's RAM check guards it too). It writes
    the **prepared layout**, the only format the training subcommands read from
    `--data-dir`:
    - one directory per source file, with one subdirectory per split;
    - in each split: `images.npy` (uint8, N×63×63×3), `labels.npy` and `features.npy`
      (where the schema has them);
    - a `meta.json` with the source SHA-256 and the schema.

    The prepared files are loaded with `np.load(allow_pickle=False, mmap_mode="r")`, so
    repeated runs never touch pickle.
  - `inspect-data` and the manifest accept both the pickles and the prepared layout.
- **Acceptance:**
  - Characterization tests for the three preprocessing functions, including non-square
    stamps on each side and NaN/inf inputs.
  - Schema-violation tests.
  - `inspect-data` runs on the synthetic files.
  - `prepare-data` gives the same uint8 arrays as the legacy path (float64 input, legacy
    `img_float2int`), not merely the same as the new loader.
  - `make-synthetic` is deterministic: the same seed gives the same bytes.

### 1.3 Losses (after 1.1) — done (PR #2, merged as 49e9831, 2026-11-07)

- Port `NT_Xent` and `P_stamps_loss` to `src/astro_simclr/losses.py`.
- The mask is built from the actual batch size.
- **Denominator (user decision, 2026-11-07):** the legacy formula
  `−s_pos + log(1 + Σ_neg exp(s_neg))` is the default, so that runs stay comparable with
  `results/`. The paper's formula `−s_pos + log(exp(s_pos) + Σ_neg exp(s_neg))` is behind
  `legacy_denominator=False`, exposed in the config and the CLI.
- ADR `0002-nt-xent-denominator.md` records the choice and notes that the legacy variant
  is close to DCL (Yeh et al., 2021).
- **Acceptance:**
  - Both formulas are checked against independent reference implementations written in
    the test, for batch sizes 1, 2, 7 and 150.
  - The default formula matches the legacy fixtures.
  - `P_stamps_loss` equals CE − β·H(softmax) on hand-computed cases.

### 1.4 Models (after 1.1) — done (PR #3, merged as 0c880fe, 2026-11-07)

- Port `Stamp_encoder`, `Stamp_classifier`, `CLR`, `Linear_classifier`,
  `Three_layers_classifier` and `Identity` to `src/astro_simclr/models.py`.
- Port `pstamps` = `P_stamps_net_SimCLR_v2` from `b357b43^:models/models.py`.
- The encoder factory covers `stamp<k>`, `pstamps` and `resnet18/34/50/152`. It reads the
  feature dimension from the encoder's declared output, not from `bn1`. Unknown names fail
  with an error that lists the valid ones.
- **Acceptance:**
  - Stamp models: forward-shape tests at 27×27, with and without features. Any other
    size fails with a clear error.
  - ResNets: forward-shape tests at 27 and 63.
  - Parameter counts match the legacy classes, including the historical `pstamps`.
  - Forward outputs with fixed weights match the fixtures.

### 1.5 Augmentations, LARS and the LR schedule (after 1.1) — done (PR #4, merged as 404bb00, 2026-11-07)

- Port, on the `Augmentation` protocol, exactly `simclr`, `simclr2`, `astro7`, `astro8`,
  `rotation_v3` and `without_aug`. These are the names the active entry points and R1–R4
  use, and all of them are pure torchvision. They go behind a registry keyed by the legacy
  names. Every other name fails with an error that says it is deferred.
  (Update 2026-11-07, PR #6, requested by the user: the registry uses descriptive names
  `simclr`, `simclr_centered`, `astro_perspective`, `astro_perspective_gray`,
  `rotate_flip` and `center_crop`, and the legacy names stay valid as aliases.)
- Randomness comes from `seed_everything(seed, workers=True)`. No generator plumbing in
  the transforms.
- Port LARS to `src/astro_simclr/optim.py`:
  - It works on any device.
  - `.ge(0)` stays the default; `.gt(0)` is behind a flag.
- Reimplement the cosine-with-warmup schedule and drop timm. The fixture comes from timm
  0.6.x `get_epoch_values` (warmup 5, `t_initial` 350, `lr_min` 1e-4).
- **Acceptance:**
  - Two DataLoaders with the same seed yield identical first two batches, with
    `num_workers` 0 and 2.
  - Augmentation shapes and value ranges are checked.
  - LARS: a single-step fixture test, plus explicit zero-gradient and zero-weight cases,
    under both flags.
  - LR schedule: the test checks the *optimizer's* `param_groups[0]["lr"]` against the
    fixture for epochs 0–400, covering warmup, the cosine and the `lr_min` plateau.

### 1.6 Lightning modules (after 1.2–1.5) — done (PR #9, merged as 57dae04, 2026-11-07)

- Port `SimCLR`, `Linear_classifier`, `Fine_tuning` and both supervised CE modules to
  Lightning 2:
  - `on_*_epoch_end` with explicit output buffers;
  - `self.log` instead of `logger.experiment.add_scalar`;
  - torchmetrics with `task="multiclass"` and an explicit `average="micro"` for accuracy,
    because `MulticlassAccuracy` defaults to macro and the legacy value is micro;
  - confusion matrix with `normalize="true"`;
  - `accelerator`/`devices` instead of `gpus`;
  - `ModelCheckpoint(enable_version_counter=False)` instead of `ModelCheckpoint_V2`.
- Datasets are built in `LightningDataModule.setup()`, not in `prepare_data()`, so
  multi-GPU runs work.
- One shared implementation of the metrics and the confusion matrix replaces the four
  copies.
- `L.seed_everything(seed, workers=True)` runs at the start of every run.
- Unknown optimizer names raise an error.
- **Acceptance:**
  - `Trainer(fast_dev_run=True, accelerator="cpu")` passes for every module on the
    synthetic data.
  - A seeded 2-epoch CPU run of `SimCLR` gives the same loss twice.
  - The LR scheduler steps once per epoch.
  - On fixed, imbalanced predictions, accuracy equals the legacy micro value.

### 1.7 CLI, end-to-end smoke test, synthetic preflight (after 1.6)

- Subcommands `pretrain`, `linear-eval`, `fine-tune` and `train-supervised`. Each takes
  `--config`, `--data-dir`, `--output-dir`, `--seed`, `--accelerator` and generic
  `--set key=value` overrides (for example `--set devices=1`), and `--resume PATH|auto`,
  which is passed to `trainer.fit(ckpt_path=...)`.
  - `--accelerator cuda` without CUDA fails with a clear error.
  - The **dry mode** for chained commands is
    `--set max_epochs=1 limit_train_batches=2 limit_val_batches=2` plus small batch sizes,
    *not* `fast_dev_run`. `fast_dev_run` disables checkpoints, EarlyStopping and loggers,
    so `pretrain` would leave nothing for `linear-eval` to load.
  - `ModelCheckpoint(save_last=True)` runs alongside the best-by-monitor checkpoint.
  - Reference configs set `num_workers` explicitly.
  - `results/hparams_best_*.yaml` load through a documented key mapping per subcommand;
    unknown keys are an error.
  - Each subcommand's protocol defaults to the legacy `iter_*.py` values.
  - The configs of R1–R4 are in `configs/reference/`.
- Outputs go only under `--output-dir`: checkpoints, the confusion-matrix figure and a
  metrics JSON with `val_acc_best`, `best_epoch`, `test_acc_at_best`, macro-F1, the
  confusion matrix, the seed and the resolved config.
- `compare-reference RUN_DIRS --ref R1|R2|R3|R4` aggregates the seeds (mean, std with
  ddof=1, any n) and applies that run's gate:
  - R1 and R2: the pass rule;
  - R3: the majority-class baseline, taken from the class counts in `meta.json`;
  - R4: report only.

  It prints PASS, FAIL or REPORT. Its exit code is the result.
- `tests/test_smoke.py` calls `cli.main([...])` in-process, on CPU, on a tiny synthetic
  dataset in `tmp_path`. The chain is make-synthetic, then prepare-data, then pretrain,
  linear-eval (which loads the pretrain checkpoint), fine-tune and train-supervised, with
  1 epoch, small batches and the `stamp1` encoder. It asserts that:
  - every output exists and its metrics are finite;
  - `--resume`: 2 epochs, stop, then 2 more give the same epoch counter, optimizer lr and
    EarlyStopping `wait_count` as an uninterrupted 4-epoch run;
  - train-supervised validation accuracy is ≥ 0.5 with seed 0.
- `astro-simclr preflight [--synthetic | --data-dir DIR]` holds the logic and is tested
  with a mocked `nvidia-smi`. `scripts/gpu-preflight.sh` is a thin wrapper that calls it
  with `uv run --no-sync`. The exit code is the result. In order, it:
  1. checks that CUDA is visible to torch and that `torch.version.cuda` suits the driver;
     if not, it names the right extra;
  2. checks free RAM and disk against the file sizes, *before* anything is loaded;
  3. runs `inspect-data` on the prepared layout (memory-mapped);
  4. on the first real run, writes `data-manifest.json`; later runs verify against it;
  5. runs the dry-mode chain pretrain → linear-eval → fine-tune, plus train-supervised;
  6. measures throughput with the real DataLoader (augmentations, `num_workers`) at each
     reference batch size, and prints the GPU-hour estimate for R1–R4.
- **Acceptance:**
  - The smoke test passes inside `scripts/check.sh` and takes under 60 s on CI. There is
    no marker that skips it.
  - The same commands work with `--accelerator mps` on the development Mac. This is
    checked by hand and stated in the PR.

### 1.8 Retire the legacy code (after 1.7; not required for the goal)

- Tag `legacy-final`.
- Delete all legacy Python and remove every `extend-exclude` entry for it. The backlog
  points to the tag for the Ray Tune scripts, visualization, `simclr_dataset.py` and the
  deferred augmentations.
- README: replace the conda instructions with uv, and link `docs/running.md`. Delete
  `env.yml`, which stays readable at the tag.
- **Acceptance:**
  - `git grep` finds no import of a deleted module.
  - `extend-exclude` holds only `jupyter_examples/`, `results/` and `scripts/characterize/`.
    The generators there run against a `git worktree` of `legacy-final` (and of `b357b43^`
    for `pstamps`), in a throwaway `uv run --isolated --with timm==0.6.5` environment.
  - The fixtures still pass.

### 1.9a GPU runbook (after 1.7; development agent)

- `docs/running.md` covers:
  - **Level 2 recipes:** Colab and Kaggle, step by step: `uv` install, sync with the
    extra that suits the platform's driver, then `preflight --synthetic`.
    - Check `nvidia-smi` first. The default extra needs driver ≥ 580 and a Turing or newer
      GPU (T4 works). P100 and V100 need `cu126`.
    - Copy data from Drive to local disk before training.
    - Write results back after each seed, because the VMs are wiped.
    - Colab free has 2 vCPUs and about 12.7 GB of RAM, so R3 needs the prepared `.npy`.
      Kaggle T4×2 has about 30 GB of RAM and can run two seeds at once.
  - Preparing a CUDA machine: driver check, choice of extra, the sync discipline from
    ADR 0001.
  - Placing the data: `prepare-data`, then creating or verifying the manifest.
  - `gpu-preflight.sh`.
  - The exact commands for R1–R4, with their RAM, disk and GPU-hour needs.
  - Resuming from a checkpoint.
  - Where the results go. Only the metrics JSON and the figures are committed, under
    `results/phase1/`, never checkpoints.
- **Acceptance:** a test parses the runbook's `astro-simclr` command blocks and runs each
  one on CPU with the synthetic data, in dry mode with batch sizes overridden to fit the
  synthetic splits. Each block is tagged in the markdown, so that `nvidia-smi`, `uv sync`
  and `gpu-preflight.sh` lines are excluded.

### 1.9b Reference runs (after 1.9a; manual)

- **Acceptance (needs a GPU and the data, done by the user or on a rented machine):**
  - The preflight passes on the real data.
  - R1–R4 finish (at least the minimal set).
  - The results are recorded in `results/phase1/` and judged by the gates in
    [Reference runs](#reference-runs).
  - Until then, the phase status is "done except 1.9b".

## Risks

- **Torch wheels.** A wrong extra set silently swaps the torch build, and an old driver
  rejects new CUDA wheels. ADR 0001, the CI checks and the preflight driver check cover
  this.
- **Lightning 2 semantics.** Epoch-end aggregation, checkpoint keys and LR-scheduler
  stepping differ from 1.4. The tests in 1.6 assert them.
- **Smoke-test runtime** could push `scripts/check.sh` past what agents tolerate. Shrink
  the synthetic data first. Skipping the test is not an option.
- **The data may be gone** (it lived on a lab server). Phase 1 still delivers levels 1 and
  2 without it.
- **Legacy checkpoints** (`../weights`) will not load into the ported modules. That is
  accepted.

## Open questions for the user

- ~~**NT-Xent denominator**~~: decided 2026-11-07. The legacy formula is the default and
  the paper's formula is behind a flag (task 1.3).
- **Data received 2026-11-07** (Google Drive export, kept untracked in `dataset/`, with a
  local `SHA256SUMS`). Checked statically with `pickletools` and the final STOP opcode;
  nothing was unpickled.

  | File | Size | State |
  |---|---|---|
  | `td_ztf_stamp_17_06_20_sup_1.pkl` | 69 MB | complete |
  | `td_ztf_stamp_17_06_20_sup_10.pkl` | 279 MB | complete |
  | `td_ztf_stamp_17_06_20_red-002.pkl` | 2.3 GB | complete; keys `Train`/`Validation`/`Test` × `images`/`labels`/`features`; float32 stamps. Not named in the plan: probably the "red" (reduced) set behind the `*_red_frac_*` figures |
  | `td_ztf_stamp_17_06_20_sup_100-001.pkl` | 3.2 GB | **truncated**: no STOP opcode, and only `Train` (images, features) before the end of the file. Needs a fresh download |
  | `td_ztf_stamp_simclr_300.pkl` | — | **missing** (the 300k-alert pretraining set R3 needs) |

  Still open: the full 100 % file and the 300k pretraining set. Where do they live
  (`deepspace`/`deepsleep`)?

  Task 1.2 (PR #5) inspected the complete files (details in `docs/data.md`):
  - **`red` is the full labelled set:** it is the original `td_ztf_stamp_17_06_20.pkl`
    with duplicate stamps removed. Train has 50,582 stamps (14,542 / 1,396 / 14,517 /
    9,599 / 10,528), and Validation and Test have 100 per class.
  - **`sup_1` and `sup_10` are random 1 % / 10 % draws of `red`'s Train, oversampled to
    balance the classes.** `sup_1` has only 14 distinct SN stamps, so the label fraction
    is about *stamps*, not alerts, and low-label results carry large variance.
  - **`sup_100` is `red`'s Train oversampled to 72,710**, not a different dataset.
    `red` can stand in for the 100 % runs once the protocol says whether to balance it.
  - **Features are already standardized on Train** (`isdiffpos` is scaled only).
  - **One `red` Train stamp also appears in Validation or Test:** a leakage of one item,
    to be removed before any reported number (phase-1 runs keep the legacy files as-is).
- **Data governance:** may the ZTF/ALeRCE pickles be uploaded to third-party clouds (Colab,
  Kaggle, RunPod)? Only the synthetic level-2 run is free of this question.
- ~~**CE reference optimizer**~~: resolved 2026-11-07. The Drive copy of
  `hparams_best_ce.yaml` says `optimizer: AdamW` and `balanced_batch: false`, so the repo's
  `Adam` was a later hand edit. R1 uses AdamW.
- **The SimCLR config:** is `hparams_best_simclr.yaml` (lr 12.26, batch 150)
  intentional, or a mistyped edit of the 300k run (lr 0.258, batch 550)? The plan assumes
  the latter.
- **SimCLR numbers or checkpoints:** partly answered 2026-11-07. The Drive export holds 574
  confusion-matrix figures from 2022, with mean ± std accuracy in their titles. They cover
  linear eval and fine-tuning for the 300k and "red" pretraining runs, stamp/ResNet encoders
  and 1/10/100 % labels. Example: 300k, ResNet-50, astro-aug-v8, 1 % labels, with features:
  test 0.839 ± 0.003. Turning them into a table would give R3 a real reference (backlog
  candidate for 1.9a). The checkpoints in `../weights` are still not found.
- **SupCon:** port it from history (`a187828`) in phase 1, or in the backlog? The default
  is the backlog.
- **GPU for 1.9b, decided 2026-11-07 (user):**
  - The user's Windows PC, through WSL2 (Ubuntu 26.04, reachable over Tailscale), runs
    R1, R2, R4 and the phase-2 replication. It has a GTX 1050 with 4 GB (Pascal, compute
    capability 6.1), Windows driver 582.53, 12 GB RAM plus 8 GB swap in WSL, and 4 CPUs.
    Verified by agent `foundations`: `uv sync --locked --extra cu126` gives torch
    2.14.1+cu126, CUDA available, and `check.sh` green. R2 at 63 px still has to pass the
    preflight (4 GB VRAM).
  - R3 (SimCLR) needs a larger GPU. An RTX 5090 (Blackwell, sm_120) may be available from
    2026-11-08; it must use the default build (`uv sync --locked`, CUDA 13, driver ≥ 580),
    because cu126 has no sm_120.
  - **ADR 0001 gap, to fix in task 1.7.** Rule 6 picks the extra by driver only, which
    is wrong for pre-Turing GPUs with a new driver (this PC). The rule must also check
    compute capability: < 7.5 → `cu126`. Blackwell needs the default build and a driver
    ≥ 580. Windows-native torch from PyPI is CPU-only; use WSL2. The preflight (1.7, step
    1) checks the same thing through `torch.cuda.get_device_capability()`.
- (Superseded by the entry above.) **GPU for 1.9b:** which machine, and who runs it? Research (2026-11-07) estimates R1–R4 at
  about 8–30 T4-hours, because the work is bound by per-step overhead, so an A100 is barely
  faster. The cheapest options, in order:
  1. the old lab servers;
  2. Kaggle T4×2 (free, about 30 GPU-h/week);
  3. a rented RTX 3090/4090 with ≥ 8 vCPUs (about USD 3–10 in total).

  NLHPC's V100s need `cu126`, and its AMD GPUs would need a ROCm extra (backlog).

## Backlog for later phases

- Ray Tune searches (`hyperparams_*.py`) on Ray 2, from tag `legacy-final`.
- SupCon (Khosla et al., 2020) from `a187828`, and the `method` switch.
- Embedding visualization (t-SNE / UMAP).
- Deferred augmentations. They first need a decision on albumentations (archived; its
  successor is AGPL) versus reimplementation.
- `build-pretrain-set` (from `utils/simclr_dataset.py`) and the regeneration of the
  1 % / 10 % splits (`separacion_dataset.ipynb`).
- **Rubin/LSST alert stamps** (research, 2026-11-07):
  - **Access:** alerts are world-public since 2026-02-24, and the stamps can be fetched in
    bulk from the Pitt-Google Avro bucket or through Fink Data Transfer. ALeRCE serves them
    per object (`get_stamps(..., survey="lsst")`). An RSP account (eligible through a
    Chilean institution) unlocks the Alerts API and, later, DP2 and prompt images.
  - **Labels:** no public labelled Rubin stamp set exists. Labels must come from
    crossmatches (SSO, Milliquas/AllWISE, Gaia/VSX, TNS) or from the ALeRCE team.
  - **Pipeline gaps:**
    - stamps are 30–102 px at 0.2″/px, mostly 30×30;
    - they are in nJy, with variance and mask planes, and not north-up;
    - there are 6 bands;
    - several ZTF features have no equivalent (PS1 sgscore/distpsnr, distnr/magnr,
      chinr/sharpnr, classtar, diffmaglim);
    - bogus alerts are pre-filtered at `reliability` < 0.5.
  - **Licensing:** proprietary products cannot be redistributed.
- A ROCm extra (NLHPC AMD MI100/MI210) if that cluster is used.
- **Research directions** (literature survey, 2026-11-07). No published SSL work on
  ZTF/Rubin alert triplets was found, so a careful study would be close to novel.
  - **Baselines,** to attribute any gain to SSL:
    - SupCon;
    - fine-tuned Zoobot (ConvNeXt-nano, ViT-nano p4/p8) and ImageNet ConvNeXt, following
      Rehemtulla et al. 2025 (arXiv:2512.11957);
    - BTSbot ZTF-pretrained weights;
    - channel ablations (no difference image; Acero-Cuellar et al. 2023).
  - **Methods:** BYOL or VICReg (no negatives, less sensitive to batch size), a DINO-style
    small-patch ViT, and the SimCLRv2 low-label recipe.
  - **Augmentations:** physically consistent ones only: identical flips, rotations and
    shifts on all three channels; noise matched to the stamp; channel dropout. No colour
    jitter, no large-scale random-resized crops.
  - **Leakage check:** the 300k pretraining alerts may include the objects of the
    validation and test sets. The pickles carry no object ids, so this cannot be checked
    in phase 1.
  - **Label fractions:** use repeated, stratified 1 % / 10 % subsets over several seeds,
    instead of the single frozen files.
  - **Closest related work from the ALeRCE group** (Estévez / Förster, 2024–2026):
    - Neira, Estévez & Förster 2024, Temporal Stamp Classifier (IJCNN, arXiv:2405.15073):
      sequences of 2–5 ZTF triplets, about 98 % on AGN/SN/VS. Multiple detections of one
      object are a natural source of contrastive positive pairs.
    - Carvajal et al. 2025 (A&A 697, A119): FFT of the difference stamp for
      satellite/bogus detection, on top of the stamp classifier.
    - De la Fuente Pérez 2025, M.Sc. thesis advised by Estévez (repositorio.uchile.cl
      2250/217754): self-supervised multimodal pretraining for ATAT (light curves plus
      tabular data, not stamps).
    - ATAT (A&A 689, A289; github.com/alercebroker/ATAT).
  - **Training-free adaptation block** (for the SSL / Rubin phase; user request
  2026-11-07). Adapt a pretrained encoder to a new survey or condition without
  retraining, in this order:
  1. AdaBN: recompute the BatchNorm statistics on target stamps (Li et al. 2016,
     arXiv:1603.04779; Schneider et al. 2020, arXiv:2006.16971).
  2. TENT (arXiv:2006.10726).
  3. Directional ablation of a domain or nuisance direction in the embedding, folded into
     the last linear layer as W' = (I − uuᵀ)W.
  4. Surgical fine-tuning of the first layers (arXiv:2210.11466).
  5. A model soup or WiSE-FT over the 1/10/100 % fine-tunes of one shared encoder
     (arXiv:2203.05482, arXiv:2109.01903).

  Step 3 is inspired by MIT CSAIL's "Sparks of Weight-Space Programming" (Takehi, Teoh,
  Zhang and Khattab, 2026-11-02, a blog report, not peer reviewed;
  rikiyat.github.io/blog/weight-space-programming). There, GPT-6 Astra repaired
  Qwen3.5-0.8B's repetition loops with a rank-one projection edit on 0.016 % of the
  weights (MATH-500 7.9 → 48.1). It fixes a localised failure; it does not add ability. A
  cheap test: domain-probe accuracy and 5-class linear-probe accuracy before and after
  removing 1–k directions, against random directions. Seed ensembles (phase 2, ablation
  A5) cannot be weight-averaged without a shared initialisation.
- **Small LSST test set (unverified, no licence):** an ALeRCE Hackathon 2026 repository
    with 1,694 human-labelled LSST alerts (github.com/jurados/ALeRCE-ZTFLSST-HubbleConstant).
- Notebooks under `jupyter_examples/`.
