"""The phase-2 evaluation protocol (task 2.3 of ``docs/phases/02-stamp-classifier.md``).

The rules themselves (what may decide what, the power note) are in
``docs/evaluation-protocol.md``; this package is the code behind them:

- :mod:`~astro_simclr.protocol.leakage`: stamps duplicated across splits, and the Train
  stamps to drop because of them (from Train only);
- :mod:`~astro_simclr.protocol.splits`: the stored decision splits (the 250/250 screen of
  Validation and the stratified 5-fold CV on Train + Validation), ``make-splits``;
- :mod:`~astro_simclr.protocol.predictions`: the per-stamp probabilities every classifier
  run writes, and seed ensembling;
- :mod:`~astro_simclr.protocol.stats`: McNemar, the paired stratified bootstrap, balanced
  accuracy, the adoption (superiority) and non-inferiority rules and the screen rule;
- :mod:`~astro_simclr.protocol.split_runs`: runs on one part of the stored splits
  (``train-supervised --splits FILE --fold K | --screen``, task 2.5-infra): the verified
  file, the parts as pool indices, the ``protocol`` metrics entry;
- :mod:`~astro_simclr.protocol.compare`: ``compare-variants`` (the report on normal runs,
  and the decision on split runs with ``--splits``) and the decision helpers (screen,
  cross-validation: superiority and non-inferiority) that combine the above.

Nothing here imports torch, so the CLI commands that use it stay fast.
"""
