<!-- The run queue (phase 4, task A3; ADR 0018): experiment specs, the runner, the worker,
     submission and sync over ssh, and the manual survival checklist. Code:
     src/astro_simclr/runs/spec.py and src/astro_simclr/queue/. -->
# The run queue

The queue replaces the hand-written tmux/bash queues on the GPU machine. You write an
**experiment spec** (arms × seeds × folds), submit it from the laptop over ssh, a **worker**
on the GPU machine runs one job per cell and survives disconnects, crashes and reboots,
and `queue sync` brings the run directories back for the registry (`runs reindex`, task
A2). Design and reasons: ADR 0018. Module reference: `docs/architecture.md`, sections
`runs` (`spec`) and `queue`.

## Experiment specs

A spec is a YAML file, committed in the repository (for example under `configs/specs/`):

```yaml
format: astro-simclr-experiment-spec
version: 1
name: c5-flux-screen            # a slug: names the spec's run directories
command: train-supervised       # the only command of version 1
preset: carrasco2021            # optional (presets.PRESETS)
settings:                       # optional: overrides shared by every arm
  max_steps: 30000
arms:                           # arm name -> overrides of the shared settings
  baseline: {}
  flux: {flux_rescale: 0.5}
seeds: [0, 1, 2, 3, 4]
splits:                         # optional; without it, the file's own Train/Validation/Test
  file: configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json
  screen: true                  # or `folds: [0, 1, 2, 3, 4]`
machine:
  class: cuda-nvidia-geforce-gtx-1050   # the run id's machine class
  extra: cu126                  # `uv sync --extra` on that machine (ADR 0001)
scratch: false                  # true: a dirty tree is allowed (never a result)
```

- Each arm's settings resolve as on the command line: the `Settings` defaults, the legacy
  protocol of `train-supervised`, the preset, `settings`, the arm's overrides, then the
  cell's seed. `seed`, `data_dir`, `output_dir` and `pretrained` cannot be set.
- `astro-simclr spec check FILE` validates the spec (every arm is resolved) and prints its
  **spec id** and cells. The spec id is the SHA-256 of the parsed content: comments and key
  order do not change it. The registry's `runs compare` takes it (A2).
- Each cell (`<arm>/s<seed>/<f<fold>|screen|all>`) is one run id (ADR 0015). The seeds of
  one arm share its config hash.

## The GPU machine (runner), once

```bash
cd ~/Astro-SimCLR && git pull && uv sync --locked --extra cu126
.venv/bin/astro-simclr queue init --repo ~/Astro-SimCLR --data-dir ~/Astro-SimCLR/dataset/prepared
```

This writes `~/.astro-simclr/runner.yaml` (`$ASTRO_SIMCLR_HOME` overrides the home) and
creates the queue, `~/.astro-simclr/queue/queue.sqlite`. The queue must be on the Linux side
(ext4): `queue init` refuses `/mnt/c`, network mounts and synced folders. Run directories go
to `~/.astro-simclr/runs/<spec name>/<run id[:16]>-a<attempt>/` (`--runs-root` to change it).

Then host the worker, preferably as a systemd user service (WSL2 needs `systemd=true` under
`[boot]` in `/etc/wsl.conf`):

```bash
.venv/bin/astro-simclr queue service --gpu 0 --machine-class cuda-nvidia-geforce-gtx-1050 --write
systemctl --user daemon-reload
systemctl --user enable --now astro-simclr-worker.service
loginctl enable-linger "$USER"      # runs without a login session, starts at boot
journalctl --user -u astro-simclr-worker -f
```

Without systemd, use tmux; the session is found by its **exact** name (tmux matches a bare
`-t name` by prefix: `has-session -t screen25` was true while only `screen25c` existed):

```bash
.venv/bin/astro-simclr queue tmux --session astro-simclr-worker --gpu 0 \
    --machine-class cuda-nvidia-geforce-gtx-1050
tmux attach -t =astro-simclr-worker
```

Worker options (`queue worker --help`): `--gpu N` (repeat for several GPUs; none means a
CPU worker), `--machine-class` (required with a GPU: the jobs of other classes are never
claimed), `--slots-per-gpu N` (default 1) with `--gpu-memory-mib M` (admission by measured
peak memory; a config's first run always runs alone), `--lease-seconds` (120),
`--poll-seconds` (5), `--grace-seconds` (30).

## The laptop

```bash
git push                                   # the runner fetches the commit from origin
uv run astro-simclr queue submit configs/specs/c5.yaml --host rvidal@desktop-qj6b97d
uv run astro-simclr queue status --host rvidal@desktop-qj6b97d
uv run astro-simclr queue sync --host rvidal@desktop-qj6b97d --dest ~/runs/gpu
```

- `submit` refuses a dirty tree, an uncommitted spec or a commit on no remote branch,
  unless the spec is `scratch` (then it warns: the runner gets the commit, not your
  changes). The runner checks the commit out into `~/.astro-simclr/worktrees/<sha>/`, runs
  `uv sync --locked --extra <machine.extra>` there, refuses that worktree if someone
  edited it (non-scratch), and queues one job per cell, expanded by that commit's code.
- Resubmitting the same spec queues nothing new: a cell whose run id is queued, running or
  finished is skipped. `--new-attempt` reruns every cell as a new attempt (protocol 2.3's
  confirmatory rerun). A failed or cancelled cell is queued again as its next attempt.
- `--remote-cli` is the runner's `astro-simclr` relative to the remote home (default
  `Astro-SimCLR/.venv/bin/astro-simclr`).
- `sync` copies the run directories without checkpoints, then the checkpoints the
  retention policy keeps (`best` of a finished `pretrain`), then the accepted specs into
  `~/.astro-simclr/specs/`, and runs `astro-simclr runs reindex <dest>`. It never copies a
  SQLite file. Rerun it after a dropped connection: rsync resumes (`--partial`).

On the runner: `queue status [--spec ID] [--json]`, `queue cancel JOB... | --spec ID` (`ID`
an id or a prefix that matches exactly one spec; a running job is stopped by its worker,
then marked `cancelled` in the queue and its manifest). Each job's output is in `~/.astro-simclr/queue/logs/<job>.log`.

## What happens on failures

| Event | What the queue does |
|-------|---------------------|
| A job exits non-zero (an exception) | `failed`, not retried; resubmit for a new attempt |
| A job killed by a signal (`kill -9`, the OOM killer) | requeued, resumes from `checkpoints/last.ckpt`; `failed` after 5 requeues |
| The worker killed (`kill -9`) | under systemd the unit's other processes die with it (the job too); in tmux the job keeps running as an orphan. When the lease (120 s) expires, the next worker (systemd restarts it after 10 s) kills the orphan if there is one, requeues the job and resumes it |
| The worker stopped (`systemctl --user stop`, SIGTERM) | its jobs get SIGTERM, are requeued at once (not counted) and resume when a worker starts |
| Two processes on one run directory | impossible: `spec run-cell` holds `DIR/.lock`; the second exits 75 and is retried later |
| ssh drops during `submit` or `sync` | the command fails with the ssh error; nothing half-done remains (the enqueue is one transaction; rsync resumes); run it again |
| `wsl --shutdown`, a reboot | the queue file is durable (`synchronous=FULL`); running jobs come back as expired leases from another boot and resume |

A resumed Lightning run continues from its last checkpoint (every
`checkpoint_every_n_steps`, 1000 by default): up to that many steps are recomputed. A
`carrasco2021` resume is exact; an `epochs` resume is not bit-identical (Lightning
checkpoints hold no RNG state; ADR 0014). A crash *while* `last.ckpt` is being written can
leave it truncated; the resumed job then fails and needs a new attempt (rare: the write
takes well under a second).

## Manual survival checklist

Run it once on the GPU machine after installing the service, with a small scratch spec
(for example the dry spec below), and again after any change to the queue. No step may
lose a job or run one twice. Record the date and the outcome in the PR or the plan.

```yaml
format: astro-simclr-experiment-spec
version: 1
name: survival
command: train-supervised
settings: {max_epochs: 30, checkpoint_every_n_steps: 50}
arms: {baseline: {}}
seeds: [0, 1]
machine: {class: cuda-nvidia-geforce-gtx-1050, extra: cu126}
scratch: true
```

1. **Submit.** From the laptop, `queue submit survival.yaml --host ...`. Check
   `queue status --host ...`: two jobs, one `running` (one slot), one `queued`.
2. **ssh drop.** Start `queue sync` and cut the network (or `kill` the ssh process) half-way.
   Expected: the job keeps running on the runner (`queue status` from a new ssh session);
   a second `queue sync` completes; nothing in the destination is a SQLite file.
3. **Worker `kill -9`.** On the runner, kill the worker's main process only:
   `systemctl --user kill --kill-whom=main -s KILL astro-simclr-worker` (`--kill-who` on
   systemd older than 252), or `kill -9` the worker's pid in tmux. Expected: systemd
   restarts the worker within 10 s (and stops the unit's leftover job process); in tmux
   the job keeps running as an orphan. After the lease (≤ 120 s) the new worker kills the
   orphan if it still runs, the job goes `queued` then `running` with `resumes 1`, and its
   log shows a resume from `checkpoints/last.ckpt`. In tmux, start the new worker with
   `queue tmux` (exact session name).
4. **Job `kill -9`.** `kill -9` the `spec run-cell` process. Expected: requeued at once
   (`resumes` +1), resumed from `last.ckpt`.
5. **`wsl --shutdown`.** From Windows, `wsl --shutdown` while a job runs, then start the
   distribution again (open a WSL terminal, or the scheduled task below). Expected: with
   systemd and lingering, the worker starts by itself; the job, an expired lease of
   another boot, is requeued without a kill and resumes. With tmux: start `queue tmux`
   again, same result.
6. **Reboot.** Reboot Windows. WSL does not start by itself: add a Task Scheduler task "At
   log on" (or "At startup" with the user's credentials) running
   `wsl.exe -d <distro> --exec /bin/true`, so the distribution, its systemd and the worker
   come up. Expected: as in step 5, and the queue file (`queue status`) is intact.
7. **Finish and sync.** Let both jobs finish: `finished`, `resumes` as counted above, each
   run directory with one `manifest.json` (`finished`, the queue's attempt) and
   `metrics.json`. `queue sync` brings them back without checkpoints; `runs reindex` lists
   them under the spec id.
8. **Cancel.** Submit again with `--new-attempt`, then `queue cancel --spec <id>` on the
   runner. Expected: the queued attempt is `cancelled` at once, the running one within the
   grace period (30 s), both manifests `cancelled`.
