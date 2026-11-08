"""Run identity, manifest and run-directory layout (phase 4, task A0; ADR 0015).

The frozen interfaces that the registry (A2), the queue (A3) and the new engine build on:

- :mod:`~astro_simclr.runs.canonical`: the canonical JSON, the semantic config of a run
  (its :class:`~astro_simclr.config.Settings` without the non-semantic and identity keys)
  and its SHA-256, the config hash;
- :mod:`~astro_simclr.runs.identity`: :data:`~astro_simclr.runs.identity.NUMERICS_EPOCH`,
  :class:`~astro_simclr.runs.identity.RunIdentity` and the run id, and the provenance
  helpers (data, splits and ``uv.lock`` SHA-256, git state, machine class);
- :mod:`~astro_simclr.runs.manifest`: the versioned ``manifest.json`` schema, its writer
  and its validating reader;
- :mod:`~astro_simclr.runs.layout`: the files of a run directory and the retention policy,
  a pure function (nothing deletes files yet).

Nothing here imports torch or Lightning, so the registry and the queue stay light.
"""
