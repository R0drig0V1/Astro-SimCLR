"""Training tasks on the engines (task 4.6): the shared checks, the paper classifier task
and the plain engine's runner, all Lightning-free.

- :mod:`.common`: run checks shared with ``experiments.run_command``;
- :mod:`.paper`: the paper classifier's data and its engine ``Task``;
- :mod:`.results`: the classifier results of ``metrics.json`` (shared with Lightning);
- :mod:`.plain`: :func:`~.plain.run_plain`, ``run_command`` on the plain engine.
"""
