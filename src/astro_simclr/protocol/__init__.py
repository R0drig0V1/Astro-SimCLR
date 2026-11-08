"""The evaluation protocol: phase 2 (task 2.3 of ``docs/phases/02-stamp-classifier.md``), phase 3.

The rules themselves (what may decide what, the power note) are in
``docs/evaluation-protocol.md``; this package is the code behind them:

- :mod:`~astro_simclr.protocol.leakage`: stamps duplicated across splits, and the Train
  stamps to drop because of them (from Train only);
- :mod:`~astro_simclr.protocol.groups`: object groups (alerts within 1.5" of each other,
  or with the same external id), an exact numpy neighbour search and union-find;
- :mod:`~astro_simclr.protocol.splits`: the stored decision splits (the 250/250 screen of
  Validation and the stratified 5-fold CV on Train + Validation, grouped by object since
  format version 2), ``make-splits``;
- :mod:`~astro_simclr.protocol.audit`: ``leakage-audit``, the objects shared across splits;
- :mod:`~astro_simclr.protocol.predictions`: the per-stamp probabilities every classifier
  run writes, and seed ensembling;
- :mod:`~astro_simclr.protocol.stats`: McNemar, the paired stratified bootstrap, balanced
  accuracy, the adoption (superiority) and non-inferiority rules and the screen rule;
- :mod:`~astro_simclr.protocol.split_runs`: runs on one part of the stored splits
  (``train-supervised --splits FILE --fold K | --screen``, task 2.5-infra): the verified
  file, the parts as pool indices, the ``protocol`` metrics entry;
- :mod:`~astro_simclr.protocol.compare`: ``compare-variants`` (the report on normal runs,
  and the decision on split runs with ``--splits``) and the decision helpers (screen,
  cross-validation: superiority and non-inferiority) that combine the above;
- :mod:`~astro_simclr.protocol.phase2_report`: ``report-phase2``, the phase-2 results
  report (``docs/results-phase2.md``, task 2.6) built from run directories with
  ``compare-reference`` and ``compare-variants --splits``.

The phase-3 benchmarks (task 3.2 of ``docs/phases/03-newer-benchmarks.md``; the section
"Phase 3 benchmark metrics" of ``docs/evaluation-protocol.md``):

- :mod:`~astro_simclr.protocol.alerts`: the per-alert table (``oid``, ``candid``, ``jd``,
  ``label``, optional ``group``), parsed as untrusted data, its fingerprint, label
  conflicts and the first-stamp view;
- :mod:`~astro_simclr.protocol.grouping`: grouped, class-stratified assignment of any
  per-row group ids (splits and selection/decision halves) and the cross-benchmark rule;
- :mod:`~astro_simclr.protocol.object_splits`: the stored object-grouped Train/Validation/
  Test splits of an alert table (``make-object-splits``) and their verification;
- :mod:`~astro_simclr.protocol.benchmark_metrics`: macro-F1, macro precision and recall,
  per class, the confusion matrix in percent, the zero-division rule, prior reweighting;
- :mod:`~astro_simclr.protocol.bench_report`: ``bench-report`` on runs with per-alert
  predictions (``stats.paired_bootstrap_macro_f1`` and ``stats.minimum_detectable_effect``
  are its statistics).

Nothing here imports torch, so the CLI commands that use it stay fast.
"""
