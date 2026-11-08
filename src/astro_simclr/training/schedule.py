"""ALeRCE's iteration schedule as a Lightning callback (``schedule: carrasco2021``, task 2.2a).

Carrasco-Davis et al. 2021 train by iterations, not epochs. ALeRCE's ``fit`` /
``_validate`` (``models/classifiers/deepHits_with_features_entropy_reg.py`` in
``alercebroker/pipeline@a3705ac``; never copied here):

- train until the iteration counter reaches the **horizon**, initially 30,000;
- every 10 iterations, compute the validation loss (the mean of batch-64 means);
- a strictly better value than the best so far saves the model ("best") **and** raises the
  horizon to ``max(horizon, iteration + 10,000)`` (the horizon criterion uses the same
  comparison, with ``CRITERION_PERCENTAGE`` 0);
- at the end, restore the best model.

:class:`HorizonCallback` implements this on a Trainer configured with
``max_steps=<initial horizon>``, ``val_check_interval=10`` and
``check_val_every_n_epoch=None`` (validation every 10 global steps):

- after each fitting validation it reads the monitored metric from
  ``trainer.callback_metrics``; a strict improvement (``<`` for ``min``, ``>`` for ``max``;
  NaN never improves) keeps a copy of the weights **in memory** (on their device) and sets
  the horizon, which it writes to ``trainer.fit_loop.epoch_loop.max_steps``;
- at the end of training it loads the best weights into the module and writes them **once**
  as ``best.ckpt``, a Lightning checkpoint (weights only, with the ``epoch`` and
  ``global_step`` of the best validation), so ``trainer.validate/test(ckpt_path=...)`` and
  the metrics of ``experiments`` work unchanged;
- ``state_dict()`` holds the best score, step, epoch, horizon and the best weights (on the
  CPU), so ``fit(ckpt_path=last.ckpt)`` resumes with both;
- with ``last_every_n_steps`` it also writes ``last.ckpt`` (the full training state, what
  ``--resume`` continues from) **at the end of a validation**, after updating the best and
  the horizon, every that many steps (a multiple of the validation interval), and once more
  when training ends. A ``ModelCheckpoint(every_n_train_steps=...)`` would save in
  ``on_train_batch_end``, before that step's validation, and Lightning's resume marks that
  owed validation as done without running it: a new best there would be lost.
- If a resumed run starts at or past its horizon, Lightning skips the fit loop and
  ``on_train_end`` never runs; ``on_fit_end`` then writes ``best.ckpt`` from the restored
  state instead.

It exposes ``best_model_path``, ``best_model_score`` and ``best_step``, the attributes
``experiments`` reads from a ``ModelCheckpoint``.

**Step numbering.** TF validated after the training step of iteration ``i`` for ``i`` in
0, 10, 20, ..., i.e. after ``i + 1`` optimizer steps, and trained ``horizon`` steps in all;
Lightning validates after 10, 20, 30, ... steps. ``best_step`` is Lightning's global step
(steps done), so M1's "best at iteration 4,780" is 4,781 steps in this numbering, and a new
best at step ``s`` sets the horizon to ``s + increment`` (TF: ``s + increment - 1``). The
one-step shift of the validation grid is not corrected.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

from astro_simclr.config import ConfigError

__all__ = ["HorizonCallback"]


class HorizonCallback(Callback):
    """Keep the best weights on ``monitor`` and extend the training horizon (ALeRCE rule).

    Args:
        dirpath: directory of ``best.ckpt``.
        filename: its name without ``.ckpt``.
        monitor: the validation metric (``loss_val`` in the paper).
        mode: ``min`` or ``max``.
        initial_horizon: the starting number of optimizer steps (the Trainer's
            ``max_steps``; 30,000 in the paper).
        increment: a new best at step ``s`` sets the horizon to
            ``max(horizon, s + increment)`` (10,000 in the paper; 0 never extends).
        last_filename: name of the latest-state checkpoint without ``.ckpt``.
        last_every_n_steps: write it at the end of the validations of the steps that are
            multiples of this (and when training ends); None: never.

    Raises:
        ValueError: at construction, for a ``mode`` other than ``min``/``max``, a
            non-positive horizon or a negative increment.
        ConfigError: after a validation, if the module did not log ``monitor``.
    """

    def __init__(
        self,
        *,
        dirpath: Path,
        filename: str = "best",
        monitor: str = "loss_val",
        mode: Literal["min", "max"] = "min",
        initial_horizon: int = 30_000,
        increment: int = 10_000,
        last_filename: str = "last",
        last_every_n_steps: int | None = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if last_every_n_steps is not None and last_every_n_steps < 1:
            raise ValueError(f"last_every_n_steps must be >= 1, got {last_every_n_steps}")
        if initial_horizon < 1 or increment < 0:
            msg = f"need initial_horizon >= 1, increment >= 0; got {initial_horizon}, {increment}"
            raise ValueError(msg)
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.increment = increment
        self.last_filename = last_filename
        self.last_every_n_steps = last_every_n_steps
        #: The current horizon (the Trainer's ``max_steps``).
        self.horizon = initial_horizon
        #: The best monitored value so far (a 0-d CPU tensor), or None before any validation.
        self.best_model_score: torch.Tensor | None = None
        #: Global step (optimizer steps done) of the best validation, or None.
        self.best_step: int | None = None
        #: Epoch (0-based) of the best validation, or None.
        self.best_epoch: int | None = None
        #: Path of ``best.ckpt`` once written at the end of training; empty before.
        self.best_model_path = ""
        self._best_state: dict[str, torch.Tensor] | None = None
        self._trainer: L.Trainer | None = None

    @property
    def last_checkpoint(self) -> Path:
        """Where ``last.ckpt`` is written: ``dirpath / (last_filename + ".ckpt")``."""
        return self.dirpath / f"{self.last_filename}.ckpt"

    @property
    def best_checkpoint(self) -> Path:
        """Where ``best.ckpt`` is written: ``dirpath / (filename + ".ckpt")``."""
        return self.dirpath / f"{self.filename}.ckpt"

    def _improves(self, value: float) -> bool:
        if self.best_model_score is None:
            return not math.isnan(value)
        best = float(self.best_model_score)
        return value < best if self.mode == "min" else value > best

    def _apply_horizon(self) -> None:
        if self._trainer is not None:
            self._trainer.fit_loop.epoch_loop.max_steps = self.horizon

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Remember the Trainer, so that a restored horizon reaches it before training."""
        self._trainer = trainer

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Set ``max_steps`` to the (possibly restored) horizon."""
        self._trainer = trainer
        self._apply_horizon()

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """After a fitting validation: on a strict improvement, keep the weights and extend."""
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            logged = ", ".join(sorted(trainer.callback_metrics)) or "nothing"
            msg = f"HorizonCallback monitors {self.monitor!r}, but the module logged {logged}"
            raise ConfigError(msg)
        value = float(metric)
        step = trainer.global_step
        if self._improves(value):
            self.best_model_score = torch.tensor(value)
            self.best_step = step
            self.best_epoch = trainer.current_epoch
            self._best_state = {
                key: tensor.detach().clone() for key, tensor in pl_module.state_dict().items()
            }
            self.horizon = max(self.horizon, step + self.increment)
            self._apply_horizon()
        if self.last_every_n_steps is not None and step % self.last_every_n_steps == 0:
            # After this step's validation and the update above, so a resume owes nothing.
            trainer.save_checkpoint(self.last_checkpoint)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the final ``last.ckpt``, then restore the best weights and write ``best.ckpt``.

        Nothing is written for the best if no validation ran.
        """
        if self.last_every_n_steps is not None:
            trainer.save_checkpoint(self.last_checkpoint)  # the last weights, not the best
        self._finish(trainer, pl_module)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write ``best.ckpt`` if training was skipped (a resume at or past the horizon)."""
        if not self.best_model_path:
            self._finish(trainer, pl_module)

    def _finish(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Load the best weights into the module and write them as ``best.ckpt`` (once)."""
        if self._best_state is None or self.best_model_path:
            return
        pl_module.load_state_dict(self._best_state)
        path = self.best_checkpoint
        trainer.save_checkpoint(path, weights_only=True)
        if trainer.is_global_zero:
            # Lightning stamps the current (last) epoch and step; record the best's instead,
            # as a ModelCheckpoint written at that validation would have.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["epoch"] = self.best_epoch
            checkpoint["global_step"] = self.best_step
            torch.save(checkpoint, path)
        self.best_model_path = str(path)

    def state_dict(self) -> dict[str, Any]:
        """Best score, step, epoch, horizon and best weights (CPU), for ``last.ckpt``."""
        best_state = (
            None
            if self._best_state is None
            else {key: tensor.cpu() for key, tensor in self._best_state.items()}
        )
        score = None if self.best_model_score is None else float(self.best_model_score)
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "horizon": self.horizon,
            "best_model_score": score,
            "best_step": self.best_step,
            "best_epoch": self.best_epoch,
            "best_state": best_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a resumed run's best weights and horizon.

        The horizon is the larger of the saved one and this run's initial horizon.

        Raises:
            ConfigError: if the checkpoint selected on another monitor or mode.
        """
        saved = (state_dict["monitor"], state_dict["mode"])
        if saved != (self.monitor, self.mode):
            msg = (
                f"the checkpoint selected on {saved[0]} ({saved[1]}), this run on "
                f"{self.monitor} ({self.mode}); resume with the same monitor"
            )
            raise ConfigError(msg)
        # A larger max_steps in the resumed run's settings may extend it; nothing shortens it.
        self.horizon = max(self.horizon, int(state_dict["horizon"]))
        score = state_dict["best_model_score"]
        self.best_model_score = None if score is None else torch.tensor(score)
        self.best_step = state_dict["best_step"]
        self.best_epoch = state_dict["best_epoch"]
        self._best_state = state_dict["best_state"]
        self._apply_horizon()
