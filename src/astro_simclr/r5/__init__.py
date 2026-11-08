"""R5: the phase-3, 5-class object-level benchmark under RJ23's recipe, from public sources.

Task 3.5 of ``docs/phases/03-newer-benchmarks.md``; ADR 0016 has the decisions and
``docs/data.md`` ("R5") the layout and the commands. Submodules:

- :mod:`~astro_simclr.r5.recipe`: the pre-registered constants (``JD_MAX``, the targets and
  caps per subclass) and the seeded per-object alert draw;
- :mod:`~astro_simclr.r5.sources`: dated snapshots of the Zenodo labelled set and the BTS
  SN table, and their parsers (untrusted CSV);
- :mod:`~astro_simclr.r5.select`: the candidates per subclass, RJ23's cuts and the walk to
  the targets, on the response snapshots;
- :mod:`~astro_simclr.r5.build`: the grouping with CD21, the constrained splits, the
  first-stamp view, the outputs and the online/offline driver;
- :mod:`~astro_simclr.r5.cli`: the ``r5-sources``, ``build-r5`` and ``fetch-alerts --oids``
  commands.

Nothing here imports torch.
"""
