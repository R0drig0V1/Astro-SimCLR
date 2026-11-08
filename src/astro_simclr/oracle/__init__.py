"""The M1 oracle of phase 2 (task 2.2b): the published model as a no-GPU check of the replication.

- :mod:`.tf_files`: pure-Python readers of a TF1 checkpoint (tensor bundle) and of a TF
  event log, so that no TensorFlow install is needed;
- :mod:`.m1`: M1's pinned files (commit and SHA-256), :func:`~.m1.fetch_m1` (download to a
  local cache and convert to ``.npz``, never committed), the weight mapping onto
  :class:`~astro_simclr.paper_model.PaperStampClassifier`, and the independent NumPy pass;
- :mod:`.run`: :func:`~.run.run_m1_oracle`, the port-equivalence and data-path checks and
  the report that ``compare-reference --ref CD21 --oracle-report`` reads.

The CLI commands are ``fetch-m1`` and ``m1-oracle`` (:mod:`astro_simclr.cli`);
``docs/architecture.md`` ("oracle") documents them.
"""
