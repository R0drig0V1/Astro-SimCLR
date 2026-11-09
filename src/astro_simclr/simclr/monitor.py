"""Linear-probe monitors of SimCLR pretraining (phase 5, tasks 5.0b/5.0c/5.3; ADR 0025).

The kNN monitor (:func:`~astro_simclr.simclr.diagnostics.knn_balanced_accuracy`) can
mislead: in the first pilot it rose while the linear probe on frozen ``h`` scored below the
random-init network. Two opt-in monitors put the downstream metric in
``diagnostics.jsonl`` while the run trains. Both are **diagnostics and an abort input, never
a checkpoint selector** (task 5.0c: a fixed epoch count, the final checkpoint is kept).

**Inputs (both monitors): images only, no augmentation.** The probe's input is the
encoder's ``h`` (512-d) of the centred 57-pixel crop (:func:`~.views.centre_crop`), with the
network in eval mode (BatchNorm on its running statistics): no noise, no dihedral view, no
shift, and no alert feature, position or anything else of the candidate.

**The periodic budget probe** (``probe_every`` passes, and at the final step):

- *The probe* is the 5.4 one (:func:`astro_simclr.protocol.probe.fit_probe`): multinomial
  logistic regression on standardised ``h``, plain cross-entropy, L2
  :data:`~astro_simclr.protocol.probe.DEFAULT_WEIGHT_DECAY`, full-batch L-BFGS from zero
  weights in float64 on the CPU: deterministic, no seed, no learning rate. The class
  balance comes from the subsets (``b`` objects of every class), not from a weighted loss,
  so the monitor's number is the 5.4 probe's.
- *The labels* are the stored label-budget file's (``make-label-budgets``), which must be
  drawn from the run's splits file: the subsets of one declared part (``fold0``) and repeat
  (0) at the budgets 10 and 100 objects per class. No new draw, so no new stream.
- *Train rows only* (:func:`budget_train_rows`): a CV fold's training part is a part of the
  pool (Train + Validation), so a stored subset can hold Validation stamps, even of the
  screen's decision half (CD21 ``fold0`` at 100: 11 Validation stamps, 6 of them in the
  decision half). The monitor keeps only the subset's Train stamps that the screened run may
  use (``PilotData.rows``) and records what it dropped and the per-class counts left.
- *Evaluation* on the screen's **selection** half only: the kNN monitor's data. The decision
  half, every CV evaluation part (the training part excludes its fold's evaluation part, and
  no Validation stamp is trained on) and Test (never loaded by the pilot) are not read.
- *Random-init baseline*: the same probe on the network as initialised, computed once
  before the first step and kept in the checkpoint.
- *Abort* (``abort_if_probe_below_random = E``): at the end of pass ``E`` (a forced probe
  point), the run stops cleanly unless the largest budget's probe beats the baseline
  (:func:`probe_beats_baseline`).

**The online probe** (:class:`OnlineProbe`): a ``Linear(512 -> 5)`` head trained beside the
encoder, a 100 %-label **trend** monitor that lags the final probe. Each step, before the
training forward, it takes a labelled side-batch of the Train stamps the run may use
(:func:`side_batch`: ``host_generator(seed, step, simclr_online_probe)``), embeds it in the
eval view under ``no_grad`` and trains the head on the **detached** ``h`` with its own Adam
(:data:`ONLINE_LR`, no weight decay) on a class-balanced cross-entropy (class weights
``N / (C n_c)`` of those Train stamps). The head starts at zero (no draw from the global
generator) and the encoder never sees its loss: with it on or off, the encoder's weights
are bit-identical. At each diagnostics point it logs its running train accuracy (since the
previous point) and its balanced accuracy on the screen's selection half.

Nothing here imports Lightning; randomness is a pure function of ``(seed, step)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from astro_simclr.data.schema import LABEL_NAMES
from astro_simclr.engine.randomness import RESERVED_STREAMS, host_generator
from astro_simclr.protocol.label_budgets import BudgetPart, LabelBudgets, resolve
from astro_simclr.protocol.probe import DEFAULT_WEIGHT_DECAY, fit_probe
from astro_simclr.protocol.stats import balanced_accuracy
from astro_simclr.simclr.network import REPRESENTATION_DIM, SimCLRNetwork
from astro_simclr.simclr.views import centre_crop

__all__ = [
    "DEFAULT_PART",
    "DEFAULT_REPEAT",
    "ONLINE_BATCH",
    "ONLINE_LR",
    "PROBE_BUDGETS",
    "BudgetRows",
    "OnlineProbe",
    "budget_train_rows",
    "eval_h",
    "probe_balanced_accuracy",
    "probe_beats_baseline",
    "probe_key",
    "side_batch",
]

#: The budgets (objects per class) of the periodic probe.
PROBE_BUDGETS: Final = (10, 100)
#: The label-budget part and repeat the periodic probe trains on (declared, fixed).
DEFAULT_PART: Final = "fold0"
DEFAULT_REPEAT: Final = 0
#: The online probe's side-batch size and Adam learning rate (declared).
ONLINE_BATCH: Final = 256
ONLINE_LR: Final = 1e-3


def probe_key(budget: int) -> str:
    """The ``diagnostics.jsonl`` key of the periodic probe at ``budget`` objects per class."""
    return f"probe_b{budget}_selection_bacc"


@dataclass(frozen=True, slots=True)
class BudgetRows:
    """The Train stamps the periodic probe trains on, per budget, and what was dropped.

    Attributes:
        rows: per budget, sorted Train indices (all in the run's usable Train rows).
        record: per budget, ``n`` (stamps kept), ``per_class`` (their class counts),
            ``dropped_validation`` (Validation stamps of the stored subset) and
            ``dropped_excluded`` (Train stamps the screened run may not use).
    """

    rows: dict[int, npt.NDArray[np.int64]]
    record: dict[str, Any]


def budget_train_rows(
    budgets: LabelBudgets,
    part: BudgetPart,
    *,
    repeat: int,
    wanted: Sequence[int],
    n_train: int,
    usable: npt.NDArray[np.int64],
    train_labels: npt.NDArray[np.int64],
) -> BudgetRows:
    """The Train rows of the stored subsets of ``part``, ``repeat``, at the ``wanted`` budgets.

    ``part`` is rebuilt from the verified splits (:func:`~astro_simclr.protocol.
    label_budgets.cv_parts`; its rows are pool indices, Train first). A pool index ``>=
    n_train`` is a Validation stamp and is dropped; so is a Train stamp outside ``usable``
    (the screened run's Train rows). Nothing is drawn.

    Raises:
        ValueError: if a wanted budget is not in the file, ``repeat`` is out of range, or a
            budget keeps no stamp.
        KeyError: if the file has no part named ``part.name``.
    """
    missing = sorted(set(wanted) - set(budgets.budgets))
    if missing:
        raise ValueError(f"budgets {missing} are not in the label-budget file {budgets.budgets}")
    if not 0 <= repeat < budgets.n_repeats:
        raise ValueError(f"repeat {repeat} is not in range({budgets.n_repeats})")
    stored = {
        subset.budget: subset.labelled
        for subset in resolve(budgets, part)
        if subset.repeat == repeat and subset.budget in wanted
    }
    rows: dict[int, npt.NDArray[np.int64]] = {}
    record: dict[str, Any] = {}
    for budget in sorted(wanted):
        labelled = stored[budget]
        train = labelled[labelled < n_train]
        kept = np.sort(train[np.isin(train, usable)]).astype(np.int64)
        if kept.size == 0:
            raise ValueError(f"budget {budget}: no usable Train stamp in the stored subset")
        rows[budget] = kept
        record[str(budget)] = {
            "n": int(kept.size),
            "per_class": np.bincount(train_labels[kept], minlength=len(LABEL_NAMES)).tolist(),
            "dropped_validation": int(labelled.size - train.size),
            "dropped_excluded": int(train.size - kept.size),
        }
    return BudgetRows(rows, record)


def eval_h(
    model: SimCLRNetwork,
    images: torch.Tensor,
    rows: npt.NDArray[np.int64],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """``h`` (N, 512) float32 on the CPU of ``images[rows]`` in the eval view (no augmentation).

    The centred 57-pixel crop, the model in eval mode, under ``torch.inference_mode``; the
    model's mode is restored. The result is a normal (not inference) tensor, so a probe can
    be fitted on it.
    """
    was_training = model.training
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, rows.size, batch_size):
            chunk = torch.from_numpy(rows[start : start + batch_size]).to(images.device)
            h, _ = model(centre_crop(images[chunk]).to(device))
            parts.append(h.float().cpu())
    model.train(was_training)
    if not parts:
        return torch.empty(0, REPRESENTATION_DIM)
    return torch.cat(parts).clone()  # outside inference mode: an ordinary tensor


def probe_balanced_accuracy(
    train_h: torch.Tensor,
    train_labels: npt.NDArray[np.int64],
    eval_h_: torch.Tensor,
    eval_labels: npt.NDArray[np.int64],
    *,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> float:
    """Fit the 5.4 probe on ``(train_h, train_labels)``, score it on the evaluation rows.

    Float64 on the CPU; deterministic for given inputs.

    Raises:
        ValueError: as :func:`astro_simclr.protocol.probe.fit_probe`.
    """
    probe = fit_probe(
        train_h.to("cpu", torch.float64),
        torch.from_numpy(np.asarray(train_labels, dtype=np.int64)),
        n_classes=len(LABEL_NAMES),
        weight_decay=weight_decay,
    )
    predicted = probe.predict(eval_h_.to("cpu", torch.float64))
    truth = np.asarray(eval_labels, dtype=np.int64)
    return balanced_accuracy((predicted == truth).astype(np.float64), truth)


def probe_beats_baseline(value: float, baseline: float) -> bool:
    """The abort rule's test: the probe beats the random-init probe strictly."""
    return value > baseline


def side_batch(
    seed: int, step: int, rows: npt.NDArray[np.int64], size: int
) -> npt.NDArray[np.int64]:
    """The online probe's labelled side-batch of step ``step``: ``size`` of ``rows``.

    Drawn without replacement (with replacement if ``rows`` holds fewer than ``size``) from
    ``host_generator(seed, step, simclr_online_probe)``: a pure function of ``(seed,
    step)``, independent of every training stream.
    """
    generator = host_generator(seed, step, RESERVED_STREAMS["simclr_online_probe"])
    chosen = generator.choice(rows, size, replace=size > rows.size)
    return np.asarray(chosen, dtype=np.int64)


class OnlineProbe:
    """The online linear probe (module docstring): a zero-initialised head and its Adam."""

    def __init__(
        self,
        train_labels: npt.NDArray[np.int64],
        rows: npt.NDArray[np.int64],
        device: torch.device,
        *,
        lr: float = ONLINE_LR,
    ) -> None:
        """Build the head on ``device``; class weights from the labels of ``rows``.

        Raises:
            ValueError: if ``rows`` is empty or ``lr`` is not positive.
        """
        if rows.size == 0 or not lr > 0:
            raise ValueError(f"need rows and lr > 0, got {rows.size} rows and lr {lr}")
        n_classes = len(LABEL_NAMES)
        self.head = nn.Linear(REPRESENTATION_DIM, n_classes).to(device)
        with torch.no_grad():  # zero init: no draw from the global generator
            self.head.weight.zero_()
            self.head.bias.zero_()
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        counts = np.bincount(train_labels[rows], minlength=n_classes).astype(np.float64)
        weights = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        self.labels = torch.tensor(np.asarray(train_labels, dtype=np.int64), device=device)
        self.correct = torch.zeros((), dtype=torch.int64, device=device)
        self.seen = 0

    def update(
        self,
        model: SimCLRNetwork,
        images: torch.Tensor,
        rows: npt.NDArray[np.int64],
        device: torch.device,
    ) -> None:
        """One step of the head on the side-batch ``images[rows]`` (eval view, detached ``h``).

        The encoder runs in eval mode under ``no_grad`` (its BatchNorm statistics do not
        move) and is put back in its mode; only the head's parameters get gradients.
        """
        was_training = model.training
        model.eval()
        with torch.no_grad():
            chunk = torch.from_numpy(rows).to(images.device)
            h, _ = model(centre_crop(images[chunk]).to(device))
        model.train(was_training)
        y = self.labels[torch.from_numpy(rows).to(self.labels.device)]
        logits = self.head(h.detach().float())
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        self.optimizer.step()
        self.correct += (logits.detach().argmax(dim=1) == y).sum()
        self.seen += int(rows.size)

    def take_train_accuracy(self) -> float | None:
        """The running train accuracy since the last call (``None`` if nothing was seen)."""
        value = None if self.seen == 0 else int(self.correct) / self.seen
        self.correct.zero_()
        self.seen = 0
        return value

    def balanced_accuracy(self, h: torch.Tensor, labels: npt.NDArray[np.int64]) -> float:
        """The head's balanced accuracy on (N, 512) ``h`` with ``labels``."""
        with torch.no_grad():
            weight = self.head.weight.detach().cpu()
            bias = self.head.bias.detach().cpu()
            predicted = (h.float().cpu() @ weight.T + bias).argmax(dim=1).numpy()
        truth = np.asarray(labels, dtype=np.int64)
        return balanced_accuracy((predicted == truth).astype(np.float64), truth)

    def state(self) -> dict[str, Any]:
        """The head, the optimizer and the running counts (plain values; checkpointed)."""
        return {
            "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "correct": int(self.correct),
            "seen": self.seen,
        }

    def load(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state` (an exact resume)."""
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.correct.fill_(int(state["correct"]))
        self.seen = int(state["seen"])
