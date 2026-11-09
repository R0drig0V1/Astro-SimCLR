"""The durable run queue: experiment specs in, run directories out (phase 4, task A3; ADR 0026).

It replaces the hand-written tmux/bash queues. The pieces, from the laptop to the GPU:

- :mod:`~astro_simclr.queue.remote`: ``queue submit`` checks the local tree (clean unless
  the spec is ``scratch``; the commit on a remote branch) and sends the spec over a
  :class:`~astro_simclr.queue.remote.SshTransport` (ssh; a fake ssh in the tests) to the
  runner, as a fixed argv with the spec on stdin; ``queue sync`` rsyncs the run directories
  and the specs back and runs ``runs reindex`` (the registry, A2). SQLite is never copied.
- :mod:`~astro_simclr.queue.runner`: on the GPU machine, ``queue enqueue`` checks the
  submitted SHA out into a per-SHA worktree, runs ``uv sync --locked --extra <machine>``,
  refuses a dirty worktree for a non-scratch spec, and expands the spec with *that
  commit's* code (``spec expand``) into one job per cell.
- :mod:`~astro_simclr.queue.store`: the queue itself, a SQLite file on a local disk
  (:mod:`~astro_simclr.queue.location` refuses ``/mnt/c``, network and synced folders),
  with heartbeat leases.
- :mod:`~astro_simclr.queue.worker`: claims jobs (:mod:`~astro_simclr.queue.admission`:
  one per GPU by default, more only by measured peak memory), runs each as its own process
  group, renews its leases, requeues an expired lease (killing the orphan first) so that it
  resumes from ``last``.
- :mod:`~astro_simclr.queue.job`: the fixed entry point a job runs, ``spec run-cell``,
  inside the per-SHA worktree: one cell of a spec under a run-directory lock, through
  ``tasks.plain.run_plain`` (which writes its ``manifest.json``, ADR 0015) with the queue's attempt
  and machine class (``runs.record.QueueAttempt``).
- :mod:`~astro_simclr.queue.service`: a systemd user unit, or a tmux session found by its
  exact name, to host the worker.

The spec format itself is :mod:`astro_simclr.runs.spec`, shared with the registry. Nothing
here imports torch at module level.
"""
