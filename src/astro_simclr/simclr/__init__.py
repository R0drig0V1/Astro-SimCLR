"""SimCLR on the plain-PyTorch platform (phase 5; task 5.0b's pilot; ADR 0016).

Engine-free (ADR 0014, CLAUDE.md): nothing in this package imports Lightning
(``tests/test_simclr_pilot.py`` checks it), so every function here can run on the plain
engine of phase 4 once its loop exists (task 5.3). Modules:

- :mod:`.views`: the training views (8 dihedral elements, a 57-pixel integer shift, C5's
  noise as an option) and the centred evaluation view;
- :mod:`.network`: ResNet-18 with a CIFAR stem (``h``, 512-d) and the projection head
  (``z``, 128-d);
- :mod:`.objective`: NT-Xent in paper mode, in float32;
- :mod:`.optim`: LARS with BatchNorm and biases excluded, ``0.3 * B / 256``, the
  per-step warmup and cosine over exactly the run's length;
- :mod:`.diagnostics`: effective rank, per-dimension std, alignment, uniformity, kNN;
- :mod:`.probe`: stored label subsets per class and the logistic-regression probe on
  frozen ``h``;
- :mod:`.pilot`: the pilot's minimal pretraining loop, checkpoints and embedding dump;
- :mod:`.commands`: the ``simclr-*`` subcommands of the CLI.
"""
