"""The local run registry (phase 4, task A2; ADR 0018).

A derived, rebuildable SQLite index of run directories, kept outside every worktree at
``$ASTRO_SIMCLR_HOME/registry.sqlite`` (default ``~/.astro-simclr``). The run directories
are the source of truth: deleting the file loses nothing that ``astro-simclr runs reindex``
cannot rebuild from them.

- :mod:`~astro_simclr.registry.home`: where the registry lives, and the refusal to put it
  inside a git worktree;
- :mod:`~astro_simclr.registry.readers`: one :class:`~astro_simclr.registry.readers.RunRecord`
  per run directory, read from its ``manifest.json`` (task A1) or, for an archived run
  without one, backfilled from its ``metrics.json`` (``legacy``); the readers of the
  recorded config are versioned on ``config_format``;
- :mod:`~astro_simclr.registry.store`: the SQLite schema, its ``schema_version`` and the
  migrations;
- :mod:`~astro_simclr.registry.index`: ``reindex``, the scan of the run roots;
- :mod:`~astro_simclr.registry.compare`: comparisons by experiment-spec id (the spec format
  is task A3's ``runs.spec``);
- :mod:`~astro_simclr.registry.commands`: ``astro-simclr runs reindex | ls | show | compare``.

Nothing here imports torch or Lightning.
"""
