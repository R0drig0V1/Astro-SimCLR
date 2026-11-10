"""SimCLR on the plain-PyTorch platform (phase 5; task 5.0b's pilot; ADR 0025).

Engine-free (ADR 0014, CLAUDE.md): nothing in this package imports Lightning
(``tests/test_simclr_pilot.py`` checks it), so every function here can run on the plain
engine of phase 4 once its loop exists (task 5.3). Modules:

- :mod:`.views`: the training views (8 dihedral elements, a 57-pixel integer shift, C5's
  noise, and task 5.2's scale jitter, blur, gamma and small-angle rotation as options) and
  the centred evaluation view;
- :mod:`.admission`: plan 5.2's admission test of a view (``simclr-view-admission``);
- :mod:`.network`: ResNet-18 with a CIFAR stem (``h``, 512-d) and the projection head
  (``z``, 128-d);
- :mod:`.objective`: NT-Xent in paper mode, in float32;
- :mod:`.optim`: LARS with BatchNorm and biases excluded, ``0.3 * B / 256``, the
  per-step warmup and cosine over exactly the run's length;
- :mod:`.diagnostics`: effective rank, per-dimension std, alignment, uniformity, kNN;
- :mod:`.probe`: stored label subsets per class and the logistic-regression probe on
  frozen ``h``;
- :mod:`.monitor`: the probe monitors of pretraining (a periodic budget probe with the
  abort rule, and a stop-gradient online probe), images only, on the selection half;
- :mod:`.pilot`: the pilot's minimal pretraining loop, checkpoints and embedding dump;
- :mod:`.scratch`: the go/no-go's supervised scratch baseline (the pilot's encoder plus a
  linear head, trained per label subset on the plain engine);
- :mod:`.compare`: the paired comparison of two arms' subset reports and the verdict;
- :mod:`.commands`: the ``simclr-*`` subcommands of the CLI.
"""
