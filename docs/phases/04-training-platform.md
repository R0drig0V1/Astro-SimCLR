<!-- Phase 4 of the roadmap (docs/phases/README.md); owned by the integrator. Reviewed (round 1,
docs/plan-reviews/2026-11-08-phase-4.md); amendments applied below. -->
# Phase 4: a modern training platform (plain PyTorch, local registry)

## Goal

The user's request (2026-11-08): replace the old training scaffolding, which "is very old and
archaic", with a modern one. **User decisions:**
- the engine is **plain PyTorch**, a small loop of our own, with no Lightning;
- experiment tracking is **local only**: a run registry plus local scalars, with no
  external service.

Phase 5 (SimCLR) is built on this platform. Phases 2 and 3 move to it once equivalence is
shown.

What is wrong today (audit and SimCLR review, 2026-11-08):
- **Hidden behaviour.** Lightning hid a schedule bug in SimCLR (the cosine is fixed at 350
  epochs whatever `max_epochs` is). Monitored metrics depend on an invisible callback order,
  and the frozen encoder stays in `eval()` only through a `train()` override.
- **Monoliths.** `experiments.py` has 1,023 lines and dispatches every command. `Settings` is
  flat, with about 70 fields, and `cli.py` has 828 lines.
- **Two data paths.** The legacy PIL/uint8 path with CPU workers repeatedly quantises the
  stamps. The GPU-resident path is used by the paper model only. The augmentations make
  CPU↔GPU round trips.
- **No orchestration.** Runs are launched with hand-written bash queues in tmux over ssh,
  watched by `grep` loops. Nothing declares "these arms × folds × seeds".
- **No registry.** Results lie in `~/runs` on the GPU machine. Nothing queries them by
  config, commit or machine.
- **No sweep tool.** Ray Tune was legacy only.
- **Performance.** On a 5090, Lightning and logging would take an estimated 25–35 % of a
  step. CUDA graphs and torch.compile need a loop we control.

## Principles

- **Explicit over implicit:**
  - one visible training loop;
  - schedules computed from the run's actual length;
  - every RNG stream named and seeded;
  - the precision policy (fp32, TF32 or bf16) set and recorded.
- **Equivalence before replacement.** The new engine must reproduce the replication before
  Lightning is removed (task 4.7). Phase-2 results stay valid and re-runnable.
- **Portability rule (phase 1):** the same code runs on CPU, CUDA (sm_61 to Blackwell) and
  MPS. Performance features are flags with an equivalence test.
- **Small, typed, tested.** mypy strict, docstrings, and tests on CPU. Lightning's
  conveniences are reimplemented only where we need them.

## Tasks (two tracks, after review round 1)

**Track A, the daily pain, starts now on today's CLI** (no engine dependency): A0 → {A1, A2,
A3}.

**Track B, the engine:** B0 → {4.1, 4.2, 4.3} → 4.6 → 4.7 → 4.8.

**Interfaces frozen first, all in A0 and B0:**
- the canonical config JSON and its hash;
- the manifest schema;
- the batch tuple format (`device_data.TrainBatch`);
- the checkpoint key set.

**Rule from B0 on:** new flags (2.5, phase 3's `rj23` schedule) are engine-free pure
functions (model builders, augmentation functions, samplers, schedule policy objects)
wired by thin adapters. The CPU differential test (4.7a) stays a CI gate until 4.8, so a
flag added to one engine only fails CI. A cut-over date is set at 4.7, after which nothing
new lands on Lightning.

### A0 Run identity, manifest, registry, specs and queue: ADR and implemented schemas
- **Run id** = hash(canonical semantic config, seed, fold or screen, data SHA, splits SHA,
  machine class, `numerics_epoch`), plus an `attempt` counter.
  - `numerics_epoch` is an integer in the package, bumped by any PR that changes numerics.
    The PR template asks for it; for example, the C7 rotation-order fix would have bumped
    it.
  - The git SHA is recorded but not hashed.
  - Non-semantic keys (`output_dir`, `checkpoint_every_n_steps`, `accelerator`, …) are
    listed in code and excluded from the hash.
  - `failed` and `cancelled` are never "finished".
  - Identical-config reruns are allowed as new attempts: protocol 2.3's confirmatory rerun,
    and GPU non-determinism.
- **Canonical JSON:** sorted keys, `repr` floats, defaults materialised. Implemented and
  tested with fixtures. A migration test proves that the hash of each flat preset equals
  the hash of its nested equivalent (4.2).
- **Manifest** (`manifest.json` in every run directory):
  - config and hash; git SHA and dirty flag;
  - `uv.lock`, data and splits SHA-256;
  - machine, device and fp32 precision (ADR 0010);
  - `engine` (`lightning` or `plain`);
  - seeds, `concurrent_jobs`, status and timings.
- **Run-directory layout:** `manifest.json`, `metrics.json`, `scalars.jsonl`, `tb/`,
  `predictions_*.npz`, `ckpt/{last,best}.pt`.
- **Retention:** delete `last` on success; keep `best` only for pretraining runs; keep
  metrics and predictions forever. The registry records each artefact's SHA and presence.

### A1 Every run writes `manifest.json` (today's `run_command`)
`engine: lightning`. `machine_signature` gains `engine`, and comparisons refuse mixed
engines unless allowed, as for machines and precision. Rule: a new-engine variant is
compared only with a baseline rerun on the new engine.

### A2 Local registry and backfill
- A derived, rebuildable SQLite index at `$ASTRO_SIMCLR_HOME/registry.sqlite`, outside
  every worktree, with `schema_version` and migrations.
- `runs reindex|ls|show|compare`.
- A backfill of `results/phase2/**` and `~/runs`, with missing provenance as nulls and
  `legacy: true`.
- Readers versioned on `config_format`, tested on the archived metrics.
- **Comparisons take an experiment-spec id.** They fail if the resolved set differs from the
  spec's arms × seeds × folds, or holds more than one attempt per cell without an explicit
  choice.

### A3 Experiment specs, the durable queue and runners (drives today's `train-supervised`)
- **Specs** (YAML: arms as overrides, seeds, folds or screen, machine) expand into run
  ids.
- **Queue:** SQLite on the ext4 side, never on `/mnt/c` or synced.
- **Workers:** a systemd user service, or hosted in tmux with lingering.
  - Jobs hold heartbeat leases; an expired lease is re-queued and resumes from `last`.
  - Concurrency 1 on the GTX 1050, which is compute-bound (audit). On the 5090, set by
    measurement.
  - Admission by measured peak memory.
- **Remote:**
  - the runner checks out the submitted SHA into a per-SHA worktree;
  - it runs `uv sync --locked --extra <machine>`;
  - it refuses dirty trees for non-scratch specs;
  - commands are argv arrays running a fixed entry point that reads the spec file.
- **Sync:** an rsync of run directories (no checkpoints unless retained), then
  `runs reindex`. Never copy the SQLite file.
- **Acceptance:**
  - CPU crash-injection tests with fake jobs: `kill -9`, then lease expiry, then resume;
  - a fake ssh transport;
  - a manual survival checklist: ssh drop, `wsl --shutdown`, reboot.
- This replaces the hand-written tmux/bash queues. One of those queues stalled for
  25 minutes on 2026-11-08, because `tmux has-session -t screen25` matched `screen25c` by
  prefix.

### B0 Engine ADR (checklist of Lightning's hidden behaviour) — PR #35 (ADR 0014, pending renumber)
The ADR corrects three items of the original checklist, verified in the Lightning 2.6.6
sources:
- fit validations run under `no_grad`, not `inference_mode`;
- there is no forced validation after the final step when it is off the validation grid;
- Lightning checkpoints hold no RNG state.

The checklist as first written:
The ADR lists every behaviour the loop must reproduce, each pinned by a test in 4.7(a):
- **Seeding:** `seed_everything`, then the model built on CPU, then moved to the device.
- **Gradients:** `zero_grad(set_to_none=True)`, also before each validation.
- **Validation mode:** `eval()` with `inference_mode`, then back to `train()`.
- **Validation trigger:** `(total_batch_idx + 1) % 10 == 0`, with no validation at the
  1,136-step epoch boundary and one after the final step. The horizon extension that a
  validation can trigger.
- **Augmentation seeds** keyed by the optimizer steps done before the step.
- **Monitor value:** the float32 `torch.stack(losses).mean()`, with a strict `<`.
- **Determinism:** `cudnn.benchmark = not deterministic`, `use_deterministic_algorithms`,
  and `CUBLAS_WORKSPACE_CONFIG`.
- **Final evaluation** on the restored best weights.

Plus:
- **Data tiers (4.3):**
  1. device-resident;
  2. host-pinned float32 with a per-step gather and async upload (exact);
  3. a disk memmap with sharded reads (phase 5).

  uint8 or fp16 on the device is a recorded preset deviation, never an automatic fallback.
- **SimCLR views:** per-sample parameters are host-drawn (device-independent); per-pixel
  fields are drawn from a seeded per-step device generator (device-dependent, recorded);
  interpolating ops are rounding-class.
- **Config:** nested frozen dataclasses with tagged unions, reusing today's coercers.
  `--set a.b=c` errors on an unknown path.
- **Out of scope:** multi-GPU. The engine refuses `world_size > 1`.
- **No cross-engine resume:** in-flight Lightning runs finish on Lightning.

**Known issue (data-tiers finding, PR #42).** The tiling row stream's pass *j* and
`rot90_augment` at step *j* both use `default_rng([seed, j])`, so their draws are
correlated. This affects only runs with `rot90_augment`. The engine must give every stream
a distinct stream id. Fixing it on the Lightning path changes numerics, so it waits for
the engine.

### 4.1 Engine core (after B0)
- The loop, the horizon and early-stop policies, checkpoints and exact resume, the
  precision policy (ADR 0010), local scalars.
- Ports of the semantics of today's Lightning-bound resume tests:
  - `test_resume_from_mid_epoch_last_checkpoint`;
  - `test_resume_at_the_horizon_still_writes_best`;
  - `test_augmented_resume_is_exact`.

### 4.2 Typed config and CLI (after A0 and B0)
- The nested config and presets.
- A shim for the flat YAML and `--set`.
- The hash migration test (A0).
- `astro-simclr run CONFIG`, with the old commands as aliases until 4.8.

### 4.3 Data layer (after B0)
- The three tiers.
- Batched tensor augmentations.
- C5/C6/C7 replayed exactly: `torch.equal` against today's `augmented_batch` for
  200 steps on CPU with every flag on, plus a GPU check.
- SimCLR pair views (phase 5).

### 4.5 Sweeps — deferred to phase 5
No current phase needs a search: 2.5 decides pre-registered flags, and 3.4 runs no
search. When it comes back, the guard is structural:
- trials run in a mode that never loads the decision or test roles;
- the registry tags sweep runs and refuses to return their evaluation metrics;
- the trial count is recorded;
- pruning is off by default, because this validation signal is noise-dominated.

### 4.6 Tasks on the engine (after 4.1–4.3)
- The paper classifier with all its 2.5 flags and split runs.
- Phase 3's `rj23` schedule.
- `preflight` (`ThroughputTimer` and `run_command`).

Cut:
- the ResNet and stamp `SupervisedCE*` baselines and the legacy linear evaluation and
  fine-tuning are frozen at the tag `lightning-final` (the way to reproduce phase-1 runs);
- phase 5 writes linear evaluation and fine-tuning fresh.

### 4.7 Equivalence gate (after 4.6)
- **(a) CPU differential test,** in CI on Linux and macOS while Lightning exists.
  - Both engines run in one process on synthetic data, on a schedule with horizon
    extension.
  - Bit-exact: `torch.equal` on the parameters after each validation, identical `loss_val`
    sequences, the same `best_step`, the same horizon and `last_step`, identical final
    metrics.
  - Parametrised over every preset key, the 2.5 flags, split runs, and a mid-run resume.
- **(b) GPU differential on the 1050:** both engines with `deterministic=True` for about
  3,000 steps, bit-exact.
- **(c) Distributional check:** a 5-seed CD21 rerun with `deterministic=False`, reported
  next to a Lightning-versus-Lightning same-seed rerun as the noise floor. The archived runs
  were non-deterministic (`benchmark=True`), so they cannot be matched exactly.

### 4.8 Switch and retire (after 4.7)
- Remove Lightning, the legacy data modules and the PIL path once nothing uses them.
- Tag `lightning-final` first.
- Update CLAUDE.md, the docs and the runbook.

### 4.9 Performance flags (after the 5090 bring-up)
- **5090 bring-up first:**
  - preflight;
  - the precision policy;
  - Lightning-versus-plain s/step for rot4 and p4.
- **Then CUDA graphs, torch.compile and bf16,** each with a rounding-class equivalence test
  of fixed tolerance and a numeric acceptance: for example ≥ 1.5× steps/s on the 5090, or
  the feature is cut.
- **Note:** CUDA graphs need static shapes, but the last validation batch has 52 stamps and
  C6 replaces a variable number of rows.

## Risks
- **Equivalence drift:** a different op order or RNG consumption changes the results. The
  answer is the 4.7 gate and keeping Lightning until it passes.
- **Scope creep:** a framework of our own grows without bound. Keep only what the phases
  use, and review each PR against the 4.0 ADR.
- **Contention:** phase-2 runs and the 2.5 PRs keep landing on the old path during the
  migration. 4.6 ports the flags as they exist at its start, and later 2.5 flags land on
  both paths until 4.8.
- **The GPU machine:** WSL2 and ssh remote execution must survive disconnects (the queue is
  durable, and runs resume).
