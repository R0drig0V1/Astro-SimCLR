"""Training tasks on the plain engine (task 4.6): the run checks, the paper classifier
task, the RJ23 task and the runner, all Lightning-free.

- :mod:`.common`: the run checks of the runners (once Lightning's ``run_command``'s);
- :mod:`.paper`: the paper classifier's data and its engine ``Task``;
- :mod:`.rj23`: the ``rj23`` schedule's inputs, records, task and run;
- :mod:`.results`: the classifier results of ``metrics.json`` (as a Lightning run wrote
  them);
- :mod:`.plain`: :func:`~.plain.run_plain`, the training runner.
"""
