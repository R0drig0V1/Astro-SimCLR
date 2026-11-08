<!-- ADR 0018: experiment specs, the durable SQLite queue with heartbeat leases, workers,
     per-SHA worktrees and ssh submission, and sync by rsync then reindex (task A3). -->
# 0018: Experiment specs and the durable run queue

- **Status:** accepted (2026-11-08, phase 4, task A3)
- **Implemented by:** `src/astro_simclr/runs/spec.py`, `src/astro_simclr/queue/`
  (`store`, `location`, `admission`, `job`, `worker`, `runner`, `remote`, `service`,
  `cli`), `runs.record.QueueAttempt` and `run_command(queued=...)`
- **Checked by:** `tests/test_runs_spec.py`, `tests/test_queue_store.py`,
  `tests/test_queue_worker.py` (crash injection with real processes),
  `tests/test_queue_job.py`, `tests/test_queue_remote.py` (a fake ssh),
  `tests/test_queue_cli.py`, `tests/test_runs_record.py`; the manual survival checklist in
  `docs/queue.md`

## Context

Runs on the GPU machine (a GTX 1050 under WSL2) were launched by hand-written bash queues in
tmux over ssh and watched with `grep` loops. Nothing declared "these arms × seeds × folds";
a disconnect, a `wsl --shutdown` or a crash lost the queue's position; and one queue stalled
25 minutes on 2026-11-08 because `tmux has-session -t screen25` matched the session
`screen25c` by prefix. Plan task A3 asks for specs that expand into ADR 0015 run ids, a
durable queue, workers with leases and resume, admission by memory, remote submission of
a git SHA, and sync for the registry (A2), which shares the spec format.

## Decision

1. **Spec format** (`runs.spec`, version 1). YAML with `name`, `command`
   (`train-supervised` only for now), optional `preset` and shared `settings`, `arms` as a
   mapping name → overrides, `seeds`, optional `splits` (`file` inside the checkout, and
   `folds` or `screen: true`), `machine` (`class`, the run id's machine class, ADR 0015
   rule 7; `extra`, the uv extra) and `scratch`. `seed`, `data_dir`, `output_dir` and
   `pretrained` are reserved. Every arm is resolved at parse time. The **spec id** is the
   SHA-256 of the canonical JSON of the parsed content (comments and key order do not
   count; `scratch` does). A cell is `(arm name, seed, split part)`; the registry reads
   `name`, `spec_id`, `command`, `machine_class`, `splits.file`, `arms[].name`, `cells()`
   and `arm_config_sha256(arm)`, which are therefore stable.
2. **The queue is SQLite on the runner's local disk**, `$ASTRO_SIMCLR_HOME/queue/
   queue.sqlite`, WAL with `synchronous=FULL`. Opening refuses WSL Windows drives
   (`/mnt/<letter>`), `9p`/`drvfs`, network and FUSE mounts and synced folders: SQLite's
   locks and `fsync` are unreliable there. The file is never copied; the laptop learns
   about runs only from their run directories. *Rejected:* a queue of files in a
   directory (no atomic claim); a server (Redis, a broker) for one machine.
3. **Leases with fencing tokens.** A claim (`BEGIN IMMEDIATE`) sets a random token and an
   expiry (120 s); the worker renews it every poll (5 s). Every later write names the
   token, so a worker whose lease was taken over cannot overwrite the new holder. An
   expired lease is requeued and resumes: same attempt, same run directory,
   `run_command(resume="auto")` continues `checkpoints/last.ckpt`. Five requeues make a
   job `failed`. Statuses are the manifest's (ADR 0015 rule 11).
4. **Workers own processes, not the other way round.** A job is
   `Popen(argv, start_new_session=True)`: its own process group, killed as a whole. The
   reaper kills an expired job's orphan first, but only when it runs on the same host and
   boot (`/proc/sys/kernel/random/boot_id`) *and* the pid's command line names the run
   directory, so a recycled pid is never killed. As a second guard, `spec run-cell` holds an
   exclusive `flock` on `DIR/.lock`: a second process on one directory exits 75 and is
   retried later. Exit codes: 0 finished; 75 retry; a signal (negative, or `128 + n`) a
   counted requeue; anything else `failed`, not retried (an exception would fail again).
   Shutdown (SIGTERM) sends SIGTERM to every job, waits one grace period for all, and
   requeues them, uncounted. The reaper re-reads the lease just before it kills, and its
   requeue writes only if the lease is still expired. A requeue of a job whose cancel was
   asked for cancels it instead.
5. **SIGTERM of a job writes nothing.** `spec run-cell` exits at once on SIGTERM, SIGINT and
   SIGHUP, leaving the manifest `running`, as after `kill -9`. Letting `run_command` handle
   the signal would record `cancelled`, a terminal status, and the attempt could not
   resume. A user's cancel is recorded by the worker after the job is gone (`cancelled` in
   the queue and the manifest).
6. **The queue fixes the attempt and machine class; `run_command` writes the manifest.**
   Task A1's `runs.record.start_run` gains `queued=QueueAttempt(attempt, machine_class)`:
   the attempt is the one the directory is named after (`run_dir_name`), the machine class
   the spec's; a fresh directory starts it, an unfinished manifest of the same run id and
   attempt continues it even before any checkpoint (A1 alone would open a new attempt
   then), and any other manifest is an error. `QueueAttempt` also carries the run id the
   spec expanded to, and `start_run` refuses any other before the first step (a check
   after training would waste hours).
7. **Concurrency and admission.** One job per GPU by default (the 1050 is compute-bound).
   With `--slots-per-gpu N` and the GPU's memory, a job joins running ones only if all have
   a measured peak (`max_memory_reserved`, recorded per config hash and machine class from
   finished runs) and the sum fits under 90 % of the memory; an unmeasured config runs
   alone, which is its measurement. The head of the queue is never overtaken (FIFO).
8. **Remote submission runs the submitted commit's code, in its own worktree.** `queue
   submit` refuses a dirty tree, an uncommitted spec or an unpushed commit unless the spec
   is `scratch`, then sends the fixed argv `<remote cli> queue enqueue --sha <HEAD> -` over
   ssh, the spec on stdin. ssh hands its command to the remote shell as one string, so the
   argv is sent as `shlex.join(argv)` after `--`; host names and rsync paths are checked
   against strict patterns. The runner checks the SHA out into `worktrees/<sha>` (detached;
   `git fetch origin` if needed), refuses it if edited (non-scratch), runs `uv sync --locked
   --extra <extra>`, and expands the spec with that worktree's `astro-simclr spec expand`,
   so settings, presets and hashes are the commit's. Each job's argv is stored at enqueue
   time and run as a list: `<worktree>/.venv/bin/astro-simclr spec run-cell <spec file>
   --cell ... --attempt ... --run-dir ... --data-dir ... --repo-root ...`. *Rejected:*
   running the runner's own checkout (the code would not be the submitted commit's);
   interpolated shell strings.
9. **Sync is rsync of run directories, then reindex.** No checkpoints, locks or SQLite
   files on the first pass; a second pass brings the checkpoints that ADR 0015's
   retention keeps for finished attempts (`best` of `pretrain`); the accepted specs
   (`$ASTRO_SIMCLR_HOME/specs/<spec id>.yaml`, the bytes as submitted) go to the local
   home, where the registry resolves spec ids. Then `astro-simclr runs reindex <dest>`
   (A2's stable argv).
10. **Hosting.** A systemd user service (`Restart=always`, `KillMode=mixed`, lingering) is
    the default; tmux is the fallback, and every tmux target is `=<name>`, tmux's exact
    match. The manual survival checklist (`docs/queue.md`) covers an ssh drop, `kill -9`,
    `wsl --shutdown` and a reboot (WSL needs a logon task to start).

## Consequences

- A queued run differs from a hand-launched one only in the attempt and machine class it is
  given; its manifest and `metrics.json` are those of `run_command`.
- A spec whose `machine.class` differs from the device's default class gives run ids that a
  hand-launched run on that device would not match: name the default class
  (`cuda-nvidia-geforce-gtx-1050`) unless a different class is meant.
- `scratch` does not change a run id (the git state is recorded, not hashed): a scratch
  submission of an already finished cell is skipped like any other. Use `--new-attempt` to
  rerun.
- A resume recomputes up to `checkpoint_every_n_steps` steps; an `epochs`-schedule resume
  is not bit-identical (ADR 0014). A crash while `last.ckpt` is written can truncate it, and
  that job then needs a new attempt.
- Commits from before this change have no `spec` subcommand, so they cannot be submitted.
- The worker imports no torch: a GPU worker needs `--machine-class`, and `--gpu-memory-mib`
  for more than one slot per GPU.
