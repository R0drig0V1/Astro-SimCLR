<!-- ADR 0013: the plain-PyTorch training engine (phase 4, task B0): what the loop reproduces from Lightning, its frozen contracts, the data tiers and the equivalence gate. -->
# 0013: A plain-PyTorch training loop, held to Lightning's behaviour by a checklist and a gate

- **Status:** accepted (2026-11-08, phase 4, task B0; plan review
  `docs/plan-reviews/2026-11-08-phase-4.md`, findings F1, F5, F6, F12, F14)
- **Implemented by:** the contracts in `src/astro_simclr/engine/` (`task.py`, `policies.py`,
  `checkpoint.py`, `device.py`, `data.py`, `randomness.py`); the loop itself in task 4.1,
  the config in 4.2, the data tiers in 4.3, the tasks in 4.6
- **Checked by:** `tests/test_engine_contracts.py` (the contracts, and the Lightning
  reference facts of the checklist); the existing tests named in the checklist; the 4.7
  equivalence gate once it exists

## Context

The user decided (2026-11-08) to replace Lightning by a small training loop of our own
(phase 4). Lightning is not only scaffolding: the replication's numbers depend on what it
does implicitly, and the archived CD21 runs cannot be matched bit for bit (they ran with
`benchmark=True`, non-deterministic). So the loop must reproduce a known list of hidden
behaviours, each verified in the sources rather than remembered, and the replacement must
pass a gate before Lightning goes (4.7, 4.8).

Every item below was read in the installed **Lightning 2.6.6** and in this repository at
`813935c` (`origin/main` when this ADR was written). Paths: `PL/` is
`.venv/lib/python3.12/site-packages/lightning/pytorch/`, `FAB/` is `.../lightning/fabric/`,
`torch/` is the installed torch 2.14.1; ours are relative to `src/astro_simclr/`. Lightning
facts that were not obvious from the code are also pinned by tests now
(`tests/test_engine_contracts.py`), so a Lightning upgrade that changes one fails before
the 4.7(a) test is written against it.

## Decision

### 1. The checklist: what the loop reproduces

The scope is the `carrasco2021` path (the paper classifier, its 2.5 flags, split runs),
phase 3's `rj23` schedule and phase 5. The legacy modules (ResNet and stamp
`SupervisedCE*`, the old linear evaluation and fine-tuning) are frozen at `lightning-final`
(plan 4.6) and are not reproduced.

**C1. Seeding, then build on the CPU, then move to the device.**
- Today: `apply_fp32_precision` first (`experiments.py:1003`, ADR 0010); `seed_run` calls
  `L.seed_everything(seed, workers=True)` (`training/runs.py:118`), which seeds `random`,
  NumPy's global generator and torch (CPU and every CUDA device) (`FAB/utilities/seed.py:59-65`);
  the module is built right after, on the CPU (`experiments.py:508-509`), so its initial
  weights do not depend on the device. The data are built and moved at the `setup` hook
  (`PL/trainer/trainer.py:1039`, `training/device_data.py:196-198`); the model is moved by
  `strategy.setup` (`trainer.py:1053`, `PL/strategies/strategy.py:155`) and the optimizer is
  built after the move (`strategy.py:159`).
- Nothing else on the path draws from the global streams: the augmentations use per-step
  generators (C6) and the dropout of the faithful model is inactive. `dropout_active=True`
  would draw from the device's global generator, which is device-dependent.
- Plain loop: precision → seed → build the task on the CPU → data to the device → model to
  the device → `Task.make_optimizer()`.
- Pinned by: `test_seeded_initialisation_is_reproducible` (`tests/test_paper_model.py:81`);
  4.7(a): `torch.equal` on the parameters before the first step.

**C2. `zero_grad(set_to_none=True)` at every step, and before each validation.**
- Per step: Lightning calls `optimizer.step(closure)` (`PL/core/module.py:1368`), and the
  closure runs the forward, then `zero_grad`, then the backward (`PL/loops/optimization/
  automatic.py:130-142`). `LightningModule.optimizer_zero_grad` calls
  `optimizer.zero_grad()` (`core/module.py:1390`), whose default is `set_to_none=True`
  (`torch/optim/optimizer.py:1048`).
- Before each fit validation: `on_validation_model_zero_grad` (`PL/loops/
  training_epoch_loop.py:402-404`) calls `self.zero_grad()` (`PL/core/hooks.py:159-161`),
  `nn.Module.zero_grad(set_to_none=True)` (`torch/nn/modules/module.py:2957`): every
  parameter's gradient is `None` during the validation.
- Plain loop: forward, `optimizer.zero_grad(set_to_none=True)`, backward, `step()`;
  `module.zero_grad(set_to_none=True)` before each validation. (`None` rather than zero
  matters: Adam skips parameters whose gradient is `None`, `torch/optim/adam.py:151`.)
- Pinned by: `test_lightning_fit_validation_mode` (the gradient is `None` at validation).

**C3. `eval()` during validation, then each submodule's mode restored; `no_grad` in the
fit, `inference_mode` in the final evaluation.**
- Lightning captures every submodule's `training` flag and calls `eval()`
  (`PL/loops/evaluation_loop.py:338-343`, `core/hooks.py:170`), then restores the captured
  flags, not a blanket `train()` (`evaluation_loop.py:345-352`,
  `PL/utilities/model_helpers.py:70-83`). It never calls `train()` at the start of a fit
  (the module is built in train mode; at the start of training `PL/loops/fit_loop.py:429`,
  `:532-540` only warns about submodules in eval).
- **Correction to the plan's wording:** the validations *inside* `fit` run under
  `torch.no_grad`, not `inference_mode`: the epoch loop builds its validation loop with
  `inference_mode=False` (`training_epoch_loop.py:88-90`, `PL/loops/utilities.py:174-179`).
  Only `trainer.validate` / `trainer.test`, the final evaluation, use `inference_mode`
  (`trainer.py:438-440`, default `True`). The numbers are the same; the difference is that
  tensors kept from a fit validation (the best weights `HorizonCallback` clones) are normal
  tensors, while an inference tensor could never enter autograd again.
- Plain loop: capture the modes, `eval()`, `torch.no_grad()` for fit validations, restore
  the modes; `torch.inference_mode()` for the final evaluation.
- Pinned by: `test_lightning_fit_validation_mode`, `test_lightning_validate_uses_inference_mode`.

**C4. The validation trigger.**
- With `check_val_every_n_epoch=None` and an integer `val_check_interval`, Lightning
  validates when `(total_batch_idx + 1) % val_check_batch == 0` (`training_epoch_loop.py:556-560`;
  `val_check_batch = val_check_interval`, `fit_loop.py:293-294`); `total_batch_idx + 1` is
  the batches seen across epochs (`training_epoch_loop.py:99-103`), which is the optimizer
  steps done (one step per batch, no accumulation).
- That branch overrides `is_val_check_batch = is_last_batch` (`training_epoch_loop.py:551`),
  so the 1,136-step epoch boundary of `red` triggers nothing.
- **Correction to the plan's wording:** there is no forced validation after the final step.
  The last step is validated only when it is on the grid (`max_steps=25` validates at 10
  and 20 only). The presets always are: 30,000, and a new best on the grid at `s` gives
  `s + 10,000`. No sanity validation runs before training (`num_sanity_val_steps=0`,
  `experiments.py:937`).
- The horizon extension: the loop's `done` is checked after `on_advance_end`
  (`training_epoch_loop.py:120-133`), so a new best at the horizon step has already raised
  `max_steps` (`training/schedule.py:144`, `173`) and the run continues.
- Plain loop: after each step, `ValidationGrid.due(steps_done)` → validate → update the
  policies → write `last` if due → stop if `steps_done >= horizon` (or early stop).
- Pinned by: `test_validates_every_10_steps_across_epochs` (`tests/test_schedule.py:95`),
  `test_horizon_extends_and_best_is_restored` (`:102`),
  `test_lightning_validation_grid_and_epochs`, `test_grid_validates_on_multiples_only`.

**C5. `HorizonCallback`'s `global_step`.**
- `trainer.global_step` is the optimizer steps *completed* (`training_epoch_loop.py:113-117`),
  incremented after the `optimizer_step` hook (`automatic.py:280`). In `on_validation_end`,
  `step = trainer.global_step` (`schedule.py:165`) counts the step just taken: `best_step`
  is that count, and the horizon becomes `max(horizon, step + increment)` (`schedule.py:173`).
  M1's "best at iteration 4,780" is `best_step` 4,781.
- `last.ckpt` is written after the validation of every `checkpoint_every_n_steps` steps
  (`schedule.py:175-177`) and when training ends (`:185`); the best weights are loaded at
  the end and written with the best's epoch and step (`:193-207`). A resume at or past the
  horizon skips the fit loop (`fit_loop.py:202-206`), and `on_fit_end` writes `best`
  (`schedule.py:188-191`).
- `metrics.json` epochs: `best_epoch = (best_step - 1) // batches_per_pass` (the epoch
  running at that validation) and `last_epoch = current_epoch - 1 = (last_step - 1) //
  batches_per_pass` (`experiments.py:696-697`).
- Plain loop: the same numbering, named `steps_done`; the epoch is derived, never counted.
- Pinned by: `test_lightning_validation_grid_and_epochs` (the epoch formula),
  `test_resume_at_the_horizon_still_writes_best` (`tests/test_carrasco2021.py:170`).

**C6. Augmentation seeds keyed by the optimizer steps done before the step.**
- Inside `training_step`, `self.global_step` has not been incremented yet (C5), so it is
  the steps done before this one: `rotation_steps(n, global_step)` (`training/modules.py:518`,
  `:480`) and `augment_rng(seed, global_step, stream)` (`modules.py:522`, `:534`;
  `training/class_augment.py:178-184`). The batch of that step is stream batch
  `global_step` (`device_data.py:250-254`, `:420-422`; `training/tiling.py:94-98`).
- Plain loop: `Task.train_step(batch, step=steps_done)` and `BatchSource.batch(steps_done)`.
- Pinned by: `test_lightning_validation_grid_and_epochs` (`global_step` in the steps is
  0, 1, 2, ...), `test_rot90_augmentation_is_seeded_by_step_and_rotates_training_batches`
  (`tests/test_paper_variants.py:428`), `test_train_loader_starts_at_the_global_step`
  (`tests/test_device_data.py:81`), `test_augmented_resume_is_exact`
  (`tests/test_split_runs.py:638`), `test_host_generator_is_augment_rng`.

**C7. The monitor value: the float32 mean of the batch losses, compared with a strict `<`.**
- `ClassifierModule._log_epoch` logs `loss_val = torch.stack(losses).mean()` (`modules.py:243`,
  `:256`), the batch means weighing equally (TF's "mean of batch-64 means"; the last batch
  has 52 stamps). Lightning passes it on unchanged: the epoch-end value is accumulated as
  `0 + v·1` and divided by 1 (`PL/trainer/connectors/logger_connector/result.py:234`, `:251`),
  the batch size being 1 because the batch reference is cleared after every batch
  (`logger_connector.py:220-221`, `result.py:341-352`).
- `HorizonCallback` compares `float(metric)` strictly; NaN never improves; the first finite
  value always does (`schedule.py:136-140`).
- Plain loop: `mean_of_batch_losses` and `Monitor.improves`.
- Pinned by: `test_monitor_rule_is_horizon_callback_s`,
  `test_monitor_value_is_the_mean_of_batch_means`, `test_mode_max_and_nan_never_improves`
  (`tests/test_schedule.py:120`).

**C8. Determinism flags.**
- `build_trainer` passes `deterministic` and `benchmark=not deterministic`
  (`training/runs.py:221-222`); the Trainer's constructor applies them
  (`PL/trainer/connectors/accelerator_connector.py:111`, `:571-593`):
  `cudnn.benchmark = not deterministic`; `torch.use_deterministic_algorithms(deterministic)`,
  called with `False` too; `CUBLAS_WORKSPACE_CONFIG=":4096:8"` set only when deterministic
  and never unset. This happens after the module is built and before any data reach the
  device. ADR 0010's float32 precision is set before both.
- Plain loop: `DevicePolicy.torch_flags()`, applied at the same point (before any CUDA
  work, so the cuBLAS setting takes effect).
- Pinned by: `test_device_flags_are_lightning_s` (both values of `deterministic`).

**C9. The final evaluation runs on the restored best weights.**
- `HorizonCallback._finish` loads the best weights into the module and writes `best.ckpt`
  (`schedule.py:193-207`); `_classifier_results` then calls `trainer.validate` and
  `trainer.test` with `ckpt_path=best` (`experiments.py:863-866`; split runs: `:816-818`),
  so Validation and Test (or the selection and evaluation parts) are predicted with the best
  weights, in `eval()` and `inference_mode`.
- Plain loop: `keep="best"` loads the best weights and evaluates; `keep="stop"` evaluates
  the last weights; `("stop", "best")` evaluates both (RJ23, task 3.4b).
- Pinned by: `test_validate_after_fit_does_not_change_the_best` (`tests/test_schedule.py:130`),
  `test_run_command_writes_metrics` (`tests/test_carrasco2021.py:104`); 4.7(a): identical
  final metrics.

**C10. Checkpoint contents and the resume order.**
- Lightning's checkpoint (`PL/trainer/connectors/checkpoint_connector.py:415-500`): `epoch`,
  `global_step`, `pytorch-lightning_version`, `state_dict`, `loops` (the progress counters
  and the training loader's state: the stream definition and a split run's identity,
  `device_data.py:256-261`), `callbacks` (`HorizonCallback.state_dict`: monitor, mode,
  horizon, best score, step and epoch, the best weights on the CPU, `schedule.py:209-225`),
  `optimizer_states`, `lr_schedulers` and the hyperparameters. **It holds no RNG state**:
  the `carrasco2021` resume is exact only because nothing draws from the global streams
  (C1); the `epochs` schedule's resume is documented as not exact (`experiments.py:49-52`).
- Lightning's resume order (`trainer.py:1039-1070`): the data `setup`; the weights, loaded
  on the CPU before the move, the data module and the callbacks (`HorizonCallback.
  load_state_dict`: the monitor must match, the horizon is `max(initial, saved)`,
  `schedule.py:235-249`); `strategy.setup` (model to the device, optimizer built and moved);
  `on_fit_start` (the horizon reaches `max_steps`); the loops (progress, the loader's
  stream checked) and the optimizer state. `run_command` checks the split-run identity before
  any of this (`experiments.py:1011-1016`).
- Plain loop: the `Checkpoint` of section 2, with every RNG stream; the resume order
  in section 2.
- Pinned by: `test_resume_from_mid_epoch_last_checkpoint` (`tests/test_carrasco2021.py:127`),
  `test_resume_at_the_horizon_still_writes_best` (`:170`), `test_augmented_resume_is_exact`
  (`tests/test_split_runs.py:638`), all ported to the plain engine in 4.1;
  `test_checkpoint_is_plain_values_and_resumable`.

**C11. What Lightning does not do here, so the loop does not either:** no gradient
clipping, no accumulation, one optimizer and no LR scheduler (`ClassifierModule.
configure_optimizers`), precision `32-true` (no autocast; ADR 0010 sets the float32 policy),
no sanity validation, and no batch transfer (the batches are on the device already;
`strategy.py:263-279` is a no-op for them).

### 2. The interfaces (`src/astro_simclr/engine/`)

Frozen now; 4.1-4.3 implement against them. A change after B0 is an amendment of this ADR.

- **`Task`** (`task.py`): `module` (what the loop moves, switches between modes and
  checkpoints), `make_optimizer()` (once, after the move), `train_step(batch, step)` (the
  0-d loss with its graph; `step` = steps done before), `eval_step(batch) -> EvalOutput`
  (detached batch mean loss, probabilities, labels), `state()` / `load_state()` (task state
  outside `module.state_dict()`). The loop owns the modes, `zero_grad`, the backward, the
  step and the no-grad contexts (C2, C3). `mean_of_batch_losses` is the monitored value.
- **Loop** (4.1): one function over a `Task`, a `BatchSource`, the evaluation sources, a
  `Schedule` and a `DevicePolicy`, in the order of C1 and C4. It reads no global step
  counter: `steps_done` is the only clock.
- **Policies** (`policies.py`): `Monitor` (strict improvement, NaN never), `ValidationGrid`
  (`due(steps_done)`; the `ValidationTrigger` protocol for others), `Horizon`
  (`extended`, `resumed`; increment 0 is a fixed cap), `EarlyStop` (Lightning's
  `EarlyStopping` semantics, counted in validations, `PL/callbacks/early_stopping.py:148`,
  `:249-276`; first user `rj23`), `Keep = "stop" | "best"`, and `Schedule`, whose defaults
  are the `carrasco2021` settings.
- **Checkpoint** (`checkpoint.py`): the keys `format_version`, `engine` (`"plain"`),
  `identity` (A0's `run_id` and `attempt`, the split-run identity), `steps_done` (also the
  stream position), `model`, `optimizer`, `task`, `policy` (horizon, best score and step,
  wait count, best weights), `stream` (the definition to check), `rng` (Python, NumPy's
  global, torch CPU, every CUDA device, MPS), `numerics` (ADR 0010's record and the
  determinism flags; a resume under another policy is refused) and `versions`. Plain values
  only, so `torch.load(weights_only=True)` reads it. **Resume order:** `check_resumable`
  and the identity check → precision and seed → build the task on the CPU → load `model`
  → move → build the optimizer, load `optimizer` → `task` → `policy` → check `stream` →
  restore `rng` last, just before the first step (so nothing in between consumes it).
- **Precision policy:** ADR 0010 unchanged (`fp32_precision: ieee | tf32`, applied before
  any module is built, recorded and part of the machine signature); `DevicePolicy` carries
  it, `numerics` records it.
- **Device** (`device.py`): `DevicePolicy(accelerator, deterministic, fp32_precision,
  world_size)`, with `world_size != 1` refused and `torch_flags()` as in C8.
- **Batch format** (`data.py`): `Batch` / `TrainBatch`, equal to `device_data`'s (checked).

### 3. The data tiers (task 4.3)

Each tier is a `BatchSource` whose batch of step `s` is a pure function of the stream
definition and `s`, so all three give the same batches and the same resume.

| Tier | Where the data live | Exactness |
|---|---|---|
| 1. `device` | every split on the training device, gathered there by index (today's `DeviceTensorDataModule`) | exact |
| 2. `pinned` | float32 in pinned host memory; a per-step gather on the host and a non-blocking upload, one step ahead | exact: the same float32 values reach the same ops |
| 3. `memmap` | float32 in a disk memmap, read in shards (phase 5, data larger than host memory) | exact, for the same reason |

The tier is chosen by the config, never by the engine. Storing `uint8` or `float16` to make
data fit is a `TierChoice` with `is_deviation` true: a recorded preset deviation, **never an
automatic fallback** (plan finding F6: the uint8 path quantises the stamps).

### 4. SimCLR view randomness (phase 5)

- **Per-sample parameters** (crop offsets, flips, rotations, jitter strengths, noise
  scales) are drawn on the host from `host_generator(seed, step, stream)`
  (`SeedSequence([seed, step, stream])`, the rule of `class_augment.augment_rng`): the same
  on every device, reproducible across machines.
- **Per-pixel fields** (noise images) are drawn on the device from
  `device_generator(seed, step, stream, device)`, a `torch.Generator` seeded with 64 bits of
  the same `SeedSequence`; a host draw and upload would cost more than the step (F12). The
  seed is device-independent, the values are not (CPU and CUDA generators differ), so a
  run that uses them records the device type, and comparisons stay within it.
- Interpolating ops (resized crops, rotations by arbitrary angles) are rounding-class:
  their outputs may differ in the last bits between devices and kernels, and their tests
  use a fixed tolerance, not `torch.equal`.
- Each augmentation has its own stream number (`RESERVED_STREAMS`); a new one takes a new
  number.

### 5. Out of scope

- **Multi-GPU:** the engine refuses `world_size > 1` (`DevicePolicy`). NT-Xent's batch is
  per device, and no phase needs more than one GPU.
- **Cross-engine resume:** `check_resumable` refuses a Lightning checkpoint; in-flight
  Lightning runs finish on Lightning. The two formats share no keys by design.

### 6. The 4.7 equivalence gate

- **(a) CPU differential test, a CI gate on Linux and macOS until 4.8.** Both engines run in
  one process on synthetic data, on a schedule with a horizon extension; bit-exact:
  `torch.equal` on the parameters after each validation, identical `loss_val` sequences,
  the same `best_step`, horizon and `last_step`, identical final metrics. Parametrised over
  every preset key, the 2.5 flags, split runs and a mid-run resume. It lives in `tests/`, so
  `scripts/check.sh` runs it: a flag added to one engine only fails CI.
- **(b) GPU differential on the GTX 1050:** both engines with `deterministic=True` for
  about 3,000 steps, bit-exact (a manual run, recorded under `results/`).
- **(c) Distributional check:** a 5-seed CD21 rerun with `deterministic=False` on the plain
  engine, reported next to a Lightning-versus-Lightning same-seed rerun as the noise floor
  (the archived runs were non-deterministic, so they cannot be matched exactly). The
  pass rule is fixed in the 4.7 PR before the runs; the proposal is that every plain-versus-
  Lightning seed difference of the seed-mean test metrics lies within the range of the
  Lightning-versus-Lightning ones.

### 7. The engine-free rule for new flags

From B0 on, a new training flag (2.5, phase 3's `rj23`, phase 5) is a pure function or a
policy object that neither engine owns, wired into each engine by a thin adapter, until
4.8. The author's checklist is in CLAUDE.md ("Rules"). A cut-over date is set at 4.7,
after which nothing new lands on Lightning.

## Consequences

- The loop's behaviour is a list that can be checked line by line, and three items of the
  plan's wording are corrected by reading the sources: fit validations run under `no_grad`
  (C3), the final step is validated only on the grid (C4), and Lightning checkpoints hold
  no RNG state (C10).
- The plain checkpoint stores every RNG stream, so a resume of a task that draws from the
  global generators (`dropout_active=True`) becomes exact, which it is not today.
- Lightning's checkpoints and the plain engine's are disjoint; the A0 manifest's `engine`
  field and the same-engine comparison rule (A1) keep their results apart.
- Two engines coexist until 4.8, and every new flag costs an adapter on each; the
  differential test is what keeps them from drifting.
- `tests/test_engine_contracts.py` imports Lightning to pin the reference facts; at 4.8
  those tests go with Lightning, and the contract tests stay.
- Revisit if the 4.7 gate cannot pass on CPU (an op whose CPU kernel is not deterministic
  across calls), or when a phase needs multi-GPU.
