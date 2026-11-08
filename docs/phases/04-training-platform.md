<!-- Phase 4 of the roadmap (docs/phases/README.md); owned by the integrator. DRAFT: to be
reviewed with the improve-plan procedure before dispatch. -->
# Phase 4: a modern training platform (plain PyTorch, local registry) — draft

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

## Tasks

The dependency order is: 4.0 → {4.1, 4.2, 4.3, 4.4} in parallel → 4.5, 4.6 → 4.7 → 4.8.

### 4.0 Design ADR and interfaces (one agent, short)
An ADR that fixes the platform's shape, plus typed skeletons with no logic:
- **`engine/`:** `Task`, a protocol with `train_step`, `eval_step`, `state_dict` and
  `load_state_dict`; `Loop` (step-based, with evaluation every N steps, hooks as plain
  function lists, and early-stop/horizon policies as objects); `Checkpoint` (model,
  optimisers, schedulers, every RNG stream, the data-stream position, and the run identity);
  `Precision` (fp32, TF32 or bf16 policy); `Device`.
- **`config/`:** nested typed configs (`run`, `data`, `model`, `optim`, `schedule`,
  `augment`, `eval`, `runtime`) with presets, `--set a.b=c` overrides, validation, and a
  canonical serialisation with a content hash. ADR question: dataclasses or pydantic v2.
- **`runs/`:** the run manifest:
  - config and hash;
  - git SHA and dirty flag;
  - `uv.lock` SHA, data SHA, splits SHA;
  - machine, device and precision;
  - seeds;
  - status and timings.
- **`registry/`:** a local SQLite index of runs, and the query API.
- **`orchestrate/`:**
  - an experiment spec: arms × seeds × folds × machine, which expands into run ids;
  - a queue;
  - runners (local, and remote over ssh).
- **The deprecation plan** for `Settings`, `experiments.run_command` and the Lightning
  modules.

### 4.1 Engine core (after 4.0)
- **The loop:**
  - step-based training with gradient accumulation off by default;
  - evaluation every N steps (the paper's 10) with a fused, batch-64-means option;
  - the horizon and early-stop policies;
  - checkpoints `last` and `best`, with exact resume (RNG, stream position, scheduler).
- **Precision:** the policy from task `tf32-policy` (fp32 default), and an optional bf16
  autocast.
- **Optional performance features, each behind a flag with an equivalence test:**
  torch.compile, and CUDA graphs on a static-shape path.
- **Logging:** local scalars (CSV or JSONL, plus TensorBoard event files), with no progress
  bar by default.
- **Tests:** exact resume, the schedule length, hook order, and determinism on CPU.

### 4.2 Typed config and CLI (after 4.0)
- Nested configs with presets (carrasco2021 and its variants, migrated) and overrides.
- A loader that reads today's flat YAML and `--set` (a compatibility shim).
- One CLI built from the config tree (`astro-simclr run CONFIG [--set ...]`), with the old
  commands kept as thin aliases until 4.8.

### 4.3 Data layer (after 4.0)
- **One dataset API** over the prepared kinds (classifier, unlabelled, raw-alerts once
  phase 3's 3.1b lands), with device-resident tensors and a seeded, resumable stream.
- **Batched augmentations as tensor ops**, with no PIL:
  - the C5/C6/C7 and rot90 policy kept bit-identical (host-precomputed draws, ADR
    0006/0007);
  - the SimCLR pair views (dihedral, small shifts, per-channel affine) for phase 5;
  - prefetch with pinned memory, and no blocking syncs.
- **Memory planning:** 63-px tensors for 50k stamps are about 2.4 GB in float32, too big
  for the 1050 at 63 px, so a chunked or uint8-on-device fallback.

### 4.4 Run registry, orchestration and queue (after 4.0)
- **Manifest and registry:**
  - every run writes its manifest and final metrics, and registers in SQLite;
  - `astro-simclr runs ls|show|compare`;
  - `compare-reference` and `compare-variants` read run sets from registry queries.
- **Experiment specs.** A YAML file lists the arms (config overrides), seeds and folds or
  the screen; it expands into content-addressed run ids. A finished run is never repeated.
- **The queue:**
  - a durable queue per machine (SQLite), with `queue add|run|status|cancel`;
  - a GPU-aware concurrency limit (it can run 2 small jobs on the 1050 if a measurement
    shows a throughput gain);
  - resume after interruption;
  - remote submission over ssh to the GPU machine, with results synced back to the
    registry.

  This replaces the hand-written tmux/bash queues.
- **Notifications:** a completion summary file, plus an optional local hook.

### 4.5 Sweeps (after 4.1, 4.2, 4.4)
- Optuna with SQLite storage and pruning, driven by experiment specs.
- **Protocol guard:** a sweep may only optimise on selection data, never the decision half
  or test (task 2.3's rules). The guard is enforced in code.

### 4.6 Tasks on the engine (after 4.1–4.3)
Port, one at a time:
- the paper classifier, with all its 2.5 flags: p4/p4m, context, C5/C6/C7, rot90, split
  runs;
- the ResNet and stamp supervised baselines;
- linear evaluation and fine-tuning, **fixing the known defects**: the double softmax
  becomes plain cross-entropy behind a recorded switch, and the schedule follows the run's
  length.

The SimCLR task itself is phase 5's, built on this.

### 4.7 Equivalence gate (after 4.6)
- **On CPU:**
  - the carrasco2021 task on the new engine reproduces the Lightning run's loss trajectory,
    selected step and final metrics over a short schedule, exactly (or within a tolerance
    fixed in the ADR, if the op order must change);
  - the golden fixtures and the M1 oracle pass unchanged.
- **On the GTX 1050:** one seed of the replication rerun on the new engine matches the
  archived seed's selected step and metrics (exactly, or within the documented tolerance),
  and a 5-seed rerun passes CD21 again.
- **Throughput:** reported before and after.

### 4.8 Switch and retire (after 4.7)
- Phases 2 and 3 run on the new platform.
- Remove Lightning, the legacy data modules and the PIL path, and the old `Settings` and
  `run_command` once nothing uses them.
- Update CLAUDE.md, docs/architecture.md, and the runbook for the GPU machine.

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
