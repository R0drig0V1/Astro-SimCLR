"""The go/no-go's supervised scratch baseline (phase 5, task 5.0b; ADR 0025, amendment 5.0b).

The pilot's go/no-go compares the SSL probe with a **supervised ResNet-18 trained from
scratch on the same few labels** (user decision, 2026-11-09: "the same ResNet architecture
or very close"). This module trains it, on the plain engine (``engine.loop.fit`` with a
:class:`ScratchTask`), one label subset at a time.

**The network** (:class:`ScratchNetwork`): the pilot's encoder, :func:`~.network.
resnet18_cifar` (CIFAR stem, ``zero_init_residual``, 512-d ``h``), then ``Linear(512, 5)``.
No projection head: it is a classifier. Built on the CPU right after the engine seeds the
global streams, encoder first, so its encoder starts as the pilot's would from that seed.

**The data of a subset.** Only the subset's Train stamps reach the training stream
(:class:`ScratchStream`): a Train stamp outside the subset, a Validation stamp, the
decision half and Test cannot be trained on, because they are not in it. Each step takes a
**class-stratified** batch: ``k = min(batch_per_class, per_class)`` stamps of every class
(10 by default, so 50 stamps), a permutation of the class's subset stamps cut to ``k``.
Every batch is exactly balanced, so the plain mean cross-entropy is the class-balanced
cross-entropy (the subsets themselves hold ``per_class`` of each class).

**The view** (:func:`scratch_view`): one view per stamp, the pilot's training view
(:mod:`.views`): C5's noise at the pilot's scale (3 background sigmas by default), one of
the 8 dihedral elements, a 57-pixel window at an integer offset in 0-6. Evaluation is the
pilot's eval view: the centred 57-pixel window (:func:`~.views.centre_crop`), the network in
eval mode, no augmentation: exactly the input the SSL probe sees.

**Randomness** (ADR 0014; stream 13, ``simclr_scratch``, assigned by the integrator). Every
draw of step ``s`` of subset ``r`` comes from **one host generator**,
``host_generator(seed_r, s, 13)`` (:func:`draw_step`), in this order: the batch (per class,
in class order), the dihedral elements (B,), the offsets (B, 2), the noise strengths (B, 2)
and, when the noise is on, the noise's standard-normal fields (B, 2, 63, 63) in float32.
The fields are drawn on the host (not on the device, as the pilot's are): a batch of 50
needs 0.4M normals per step, a few milliseconds, and then every value of a scratch run is
**device-independent**. ``seed_r`` (:func:`subset_seed`) is the first 32-bit word of
``SeedSequence([seed, r, 13])``: each subset has its own initial weights and augmentations,
so the spread over subsets includes the initialisation's (the hierarchical bootstrap
resamples subsets as independent draws). The grid's trials of one subset share ``seed_r``:
they differ only in their hyperparameters.

**Recipe and selection** (plan 5.0c; :class:`ScratchRecipe`). AdamW (betas 0.9, 0.999),
decoupled weight decay on the convolution and linear weights only (normalisation
parameters and biases excluded, as SimCLR's LARS groups them: :func:`~.optim.
lars_param_groups`), a linear warmup over 5 % of the steps then a cosine to 0 at the run's
last step (:func:`~.optim.warmup_cosine_lr`), float32. A **fixed step count per budget**
(1,000 at 10 per class, 2,000 at 100 per class, the same for every subset and trial) and
the **final weights** are kept: there is no early stopping and no checkpoint selection.
The declared grid is ``lr`` in (5e-4, 2e-3) x ``weight_decay`` in (0.05, 0.5): four trials
per subset, the probe's trial count (its four L2 penalties), so both arms have the same
trial budget (5.0c, "Bookkeeping"). Each trial's final weights are scored on the screen's
**selection** half (balanced accuracy: the fit's single validation, after its last step);
the first best in grid order is kept, the probe's rule; only the kept trial is scored on
the **decision** half.

Steps: at 10 per class a step sees the whole subset (one pass per step), so 1,000 steps
are 1,000 passes over 50 stamps, whose 8 x 49 geometric views (and the noise) make about
20k distinct inputs; the training loss is near 0 well before the end. At 100 per class a
pass is 10 steps, so 2,000 steps are 200 passes, the usual supervised budget at CIFAR
scale. The cost is 20 subsets x 4 trials x (1,000 + 2,000) steps, about 240k steps of 50
stamps at 57 pixels (a few hours on the RTX 5090).

**Why the engine.** The engine's ``fit`` already runs a fixed-length fit (a ``Horizon``
with no increment), validates once after the last step on a monitor of the task's outputs
(``score``) and keeps the last weights (``keep=("stop",)``); the subset's stream serves the
batch of each step as a pure function of ``step``, and the task is a plain classifier. No
checkpoint is written (``run_dir=None``): a trial is short, and the command resumes at
subset granularity instead (``simclr-scratch --resume``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from astro_simclr.config import ConfigError
from astro_simclr.data.schema import LABEL_NAMES, DataError
from astro_simclr.engine.data import Batch, TrainBatch
from astro_simclr.engine.device import DevicePolicy
from astro_simclr.engine.loop import evaluate, fit
from astro_simclr.engine.policies import Horizon, Monitor, Schedule, ValidationGrid
from astro_simclr.engine.randomness import RESERVED_STREAMS, host_generator
from astro_simclr.engine.task import EvalOutput
from astro_simclr.protocol.stats import balanced_accuracy
from astro_simclr.simclr.network import REPRESENTATION_DIM, resnet18_cifar
from astro_simclr.simclr.optim import lars_param_groups, set_learning_rate, warmup_cosine_lr
from astro_simclr.simclr.views import (
    MAX_OFFSET,
    N_DIHEDRAL,
    STAMP_SIDE,
    ViewConfig,
    centre_crop,
    crop,
    dihedral,
    noise_with_fields,
)
from astro_simclr.stamp_relation import DEFAULT_MIN_R2, stamp_fits

__all__ = [
    "BATCH_PER_CLASS",
    "DEFAULT_STEPS",
    "EVAL_BATCH",
    "LR_GRID",
    "MONITOR",
    "NOISE_SCALE",
    "SCRATCH_STREAM",
    "WARMUP_FRACTION",
    "WEIGHT_DECAY_GRID",
    "EvalStamps",
    "ScratchNetwork",
    "ScratchRecipe",
    "ScratchStream",
    "ScratchTask",
    "StepDraw",
    "default_steps",
    "draw_step",
    "fit_trial",
    "scratch_screen",
    "scratch_view",
    "selection_score",
    "subset_seed",
]

#: The stream of every draw of the baseline (assigned by the integrator).
SCRATCH_STREAM: Final = RESERVED_STREAMS["simclr_scratch"]
#: The declared learning-rate grid (AdamW).
LR_GRID: Final = (5e-4, 2e-3)
#: The declared decoupled weight-decay grid (AdamW), on conv and linear weights only.
WEIGHT_DECAY_GRID: Final = (0.05, 0.5)
#: The fixed step count of each declared budget (stamps per class -> optimizer steps).
DEFAULT_STEPS: Final = MappingProxyType({10: 1000, 100: 2000})
#: Stamps of each class in a training batch (capped at the subset's ``per_class``).
BATCH_PER_CLASS: Final = 10
#: The linear warmup's share of the steps.
WARMUP_FRACTION: Final = 0.05
#: C5's noise scale of the pilot's run (background sigmas).
NOISE_SCALE: Final = 3.0
#: Evaluation batch (eval mode: the batch size does not change the outputs' values).
EVAL_BATCH: Final = 256
#: The engine monitor's name: the final weights' balanced accuracy on the selection half.
MONITOR: Final = "selection_bacc"


def default_steps(per_class: int) -> int:
    """The declared step count of a budget (:data:`DEFAULT_STEPS`).

    Raises:
        ConfigError: for a budget without a declared count (give the steps explicitly).
    """
    if per_class not in DEFAULT_STEPS:
        declared = ", ".join(str(b) for b in DEFAULT_STEPS)
        raise ConfigError(
            f"no declared step count at {per_class} per class (declared: {declared}); give --steps"
        )
    return int(DEFAULT_STEPS[per_class])


@dataclass(frozen=True, slots=True)
class ScratchRecipe:
    """Everything a scratch fit is, but the subset, its seed and the trial's lr and decay.

    Attributes:
        steps: optimizer steps of every trial (fixed; the final weights are kept).
        batch_per_class: stamps of each class per batch (capped at the subset's size).
        warmup_fraction: the linear warmup's share of ``steps`` (then a cosine to 0).
        noise_scale: C5's noise scale of the view (0 turns it off).
        noise_min_r2: stamps whose fit has ``R^2`` below this are not noised.
        lr_grid: the declared AdamW learning rates.
        weight_decay_grid: the declared decoupled weight decays.
        eval_batch: the evaluation batch size.

    Raises:
        ConfigError: for ``steps`` or a batch size below 1, a warmup fraction outside
            [0, 1), a negative noise scale, an empty or repeated grid, or a non-positive
            learning rate or negative weight decay.
    """

    steps: int
    batch_per_class: int = BATCH_PER_CLASS
    warmup_fraction: float = WARMUP_FRACTION
    noise_scale: float = NOISE_SCALE
    noise_min_r2: float = DEFAULT_MIN_R2
    lr_grid: tuple[float, ...] = LR_GRID
    weight_decay_grid: tuple[float, ...] = WEIGHT_DECAY_GRID
    eval_batch: int = EVAL_BATCH

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_per_class < 1 or self.eval_batch < 1:
            raise ConfigError(
                "steps, batch_per_class and eval_batch must be >= 1, got "
                f"{self.steps}, {self.batch_per_class}, {self.eval_batch}"
            )
        if not 0 <= self.warmup_fraction < 1:
            raise ConfigError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")
        if self.noise_scale < 0:
            raise ConfigError(f"noise_scale must be >= 0, got {self.noise_scale}")
        for name, grid in (("lr", self.lr_grid), ("weight_decay", self.weight_decay_grid)):
            if not grid or len(set(grid)) != len(grid):
                raise ConfigError(f"the {name} grid must be non-empty without repeats: {grid}")
        if min(self.lr_grid) <= 0 or min(self.weight_decay_grid) < 0:
            raise ConfigError("learning rates must be > 0 and weight decays >= 0")

    @property
    def warmup_steps(self) -> int:
        """``round(warmup_fraction * steps)``, at most ``steps - 1``."""
        return min(self.steps - 1, round(self.warmup_fraction * self.steps))

    @property
    def views(self) -> ViewConfig:
        """The view's noise settings."""
        return ViewConfig(noise_scale=self.noise_scale, noise_min_r2=self.noise_min_r2)

    @property
    def trials(self) -> tuple[tuple[float, float], ...]:
        """``(lr, weight_decay)`` of every trial, in grid order (lr outer)."""
        return tuple((lr, wd) for lr in self.lr_grid for wd in self.weight_decay_grid)

    def to_plain(self) -> dict[str, Any]:
        """The recipe as the report records it (the fixed choices included)."""
        return {
            "network": "resnet18_cifar + Linear(512, C)",
            "optimizer": "AdamW(betas=(0.9, 0.999), eps=1e-8); no decay on norms and biases",
            "schedule": "linear warmup, then cosine to 0 at the last step",
            "loss": "cross-entropy on class-stratified batches",
            "kept_weights": "final",
            "steps": self.steps,
            "warmup_steps": self.warmup_steps,
            "warmup_fraction": self.warmup_fraction,
            "batch_per_class": self.batch_per_class,
            "noise_scale": self.noise_scale,
            "noise_min_r2": self.noise_min_r2,
            "lr_grid": list(self.lr_grid),
            "weight_decay_grid": list(self.weight_decay_grid),
            "eval_batch": self.eval_batch,
            "stream": SCRATCH_STREAM,
        }


def subset_seed(seed: int, subset: int) -> int:
    """The seed of subset ``subset``: the first 32-bit word of ``SeedSequence([seed, r, 13])``.

    Raises:
        ValueError: for a negative ``seed`` or ``subset``.
    """
    if seed < 0 or subset < 0:
        raise ValueError(f"seed and subset must be >= 0, got {seed}, {subset}")
    words = np.random.SeedSequence([seed, subset, SCRATCH_STREAM]).generate_state(1, np.uint32)
    return int(words[0])


@dataclass(frozen=True, slots=True)
class StepDraw:
    """Every random number of one training step (host arrays).

    Attributes:
        positions: int64 (B,), the batch's stamps as positions in the subset, class by class.
        dihedral: int64 (B,), each view's dihedral element (0-7).
        offsets: int64 (B, 2), each view's window offset ``(dy, dx)`` (0-6).
        noise_u: float64 (B, 2), the noise strengths of the science and the template.
        noise_z: float32 (B, 2, 63, 63) standard normals, or ``None`` without noise.
    """

    positions: npt.NDArray[np.int64]
    dihedral: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    noise_u: npt.NDArray[np.float64]
    noise_z: npt.NDArray[np.float32] | None


def draw_step(
    seed: int,
    step: int,
    by_class: Sequence[npt.NDArray[np.int64]],
    per_class: int,
    *,
    noise: bool,
    side: int = STAMP_SIDE,
) -> StepDraw:
    """The draws of step ``step`` from ``host_generator(seed, step, 13)`` (module docstring).

    ``by_class[c]`` holds the subset positions of class ``c``; each class gives the first
    ``per_class`` of a permutation of them.

    Raises:
        ValueError: if a class has fewer than ``per_class`` positions, ``per_class < 1``,
            or as :func:`~astro_simclr.engine.randomness.host_generator`.
    """
    if per_class < 1 or any(rows.size < per_class for rows in by_class):
        raise ValueError(f"every class needs at least per_class = {per_class} >= 1 stamps")
    rng = host_generator(seed, step, SCRATCH_STREAM)
    positions = np.concatenate([rng.permutation(rows)[:per_class] for rows in by_class])
    n = positions.size
    k = rng.integers(0, N_DIHEDRAL, size=n, dtype=np.int64)
    offsets = rng.integers(0, MAX_OFFSET + 1, size=(n, 2), dtype=np.int64)
    u = rng.random((n, 2))
    z = rng.standard_normal((n, 2, side, side), dtype=np.float32) if noise else None
    return StepDraw(positions.astype(np.int64), k, offsets, u, z)


def scratch_view(
    stamps: torch.Tensor, fits: torch.Tensor | None, draw: StepDraw, config: ViewConfig
) -> torch.Tensor:
    """The training views (B, C, 57, 57) of the drawn stamps of (N, C, 63, 63) ``stamps``.

    Noise (with ``draw.noise_z``, when ``config.noise``), then the dihedral element, then
    the window: the pilot's view (:func:`~.views.make_views`), one per stamp.

    Raises:
        ValueError: if the noise is on without ``fits`` or drawn fields.
    """
    index = torch.from_numpy(draw.positions).to(stamps.device)
    x = stamps[index]
    if config.noise:
        if fits is None or draw.noise_z is None:
            raise ValueError("noised views need the stamp fits and drawn noise fields")
        z = torch.from_numpy(draw.noise_z)
        x = noise_with_fields(x, fits[index.to(fits.device)], draw.noise_u, z, config)
    return crop(dihedral(x, draw.dihedral), draw.offsets)


class ScratchNetwork(nn.Module):
    """:func:`~.network.resnet18_cifar` then ``Linear(512, n_classes)``: logits."""

    def __init__(self, n_classes: int = len(LABEL_NAMES)) -> None:
        """Build the encoder, then the linear head (both from the global generator)."""
        super().__init__()
        self.encoder = resnet18_cifar()
        self.head = nn.Linear(REPRESENTATION_DIM, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) views -> (N, n_classes) logits."""
        logits: torch.Tensor = self.head(self.encoder(x))
        return logits


def _one_hot(labels: npt.NDArray[np.int64], n_classes: int, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)).to(device)
    return F.one_hot(tensor, n_classes).to(torch.float32)


class ScratchStream:
    """The training stream of one subset: a class-stratified, augmented batch per step.

    ``stamps`` (N, 3, 63, 63) float32 are the subset's Train stamps on the training device
    (nothing else), ``labels`` (N,) their classes, ``fits`` (N, 6) their stamp fits on the
    device (needed with noise). The batch of step ``s`` is a pure function of the stream and
    ``s`` (:func:`draw_step`, :func:`scratch_view`): ``(views, (B, 0) features, one-hot)``.

    Raises:
        DataError: if a class of ``range(n_classes)`` is missing from the subset, or the
            noise is on without ``fits``.
    """

    def __init__(
        self,
        stamps: torch.Tensor,
        labels: npt.NDArray[np.int64],
        fits: torch.Tensor | None,
        *,
        seed: int,
        recipe: ScratchRecipe,
        n_classes: int = len(LABEL_NAMES),
    ) -> None:
        """Index the subset by class (draws nothing)."""
        self.stamps = stamps
        self.labels = np.asarray(labels, dtype=np.int64)
        self.fits = fits
        self.seed = seed
        self.config = recipe.views
        self.n_classes = n_classes
        self.by_class = [
            np.flatnonzero(self.labels == c).astype(np.int64) for c in range(n_classes)
        ]
        smallest = min(rows.size for rows in self.by_class)
        if smallest == 0:
            raise DataError("the subset misses a class: a class-stratified batch is impossible")
        if self.config.noise and fits is None:
            raise DataError("the noised view needs the subset's stamp fits")
        self.per_class = min(recipe.batch_per_class, smallest)
        self._passes = max(1, smallest // self.per_class)

    @property
    def batches_per_pass(self) -> int:
        """Steps of one pass over the smallest class (1 when a batch holds it all)."""
        return self._passes

    def batch(self, step: int) -> TrainBatch:
        """The views, empty features and one-hot labels of step ``step``."""
        draw = draw_step(self.seed, step, self.by_class, self.per_class, noise=self.config.noise)
        views = scratch_view(self.stamps, self.fits, draw, self.config)
        features = torch.zeros((views.shape[0], 0), device=views.device)
        return views, features, _one_hot(self.labels[draw.positions], self.n_classes, views.device)

    def state(self) -> dict[str, Any]:
        """The stream's definition (checked by :meth:`resume`)."""
        return {
            "kind": "simclr-scratch",
            "n": int(self.labels.size),
            "per_class": self.per_class,
            "seed": self.seed,
            "noise_scale": self.config.noise_scale,
            "noise_min_r2": self.config.noise_min_r2,
        }

    def resume(self, state: dict[str, Any], steps_done: int) -> None:
        """Check that ``state`` is this stream's (the batches are pure functions of ``step``).

        Raises:
            DataError: for another stream's state.
        """
        del steps_done  # the batch of every step is recomputed from the step
        if state != self.state():
            raise DataError(f"the checkpoint's stream {state} is not this one {self.state()}")


@dataclass(slots=True)
class EvalStamps:
    """An evaluation source: the eval views (centre crop) of labelled stamps, in order.

    ``stamps`` (N, 3, 63, 63) are cropped once and moved to ``device``; the batches are
    ``(views, (B, 0) features, one-hot)`` of ``batch_size`` stamps, the last one partial.
    """

    stamps: torch.Tensor
    labels: npt.NDArray[np.int64]
    device: torch.device
    batch_size: int = EVAL_BATCH
    n_classes: int = len(LABEL_NAMES)
    _views: torch.Tensor = field(init=False)
    _onehot: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.stamps.shape[0] != np.asarray(self.labels).shape[0] or self.stamps.shape[0] < 1:
            raise ValueError("need one label per stamp and at least one stamp")
        self._views = centre_crop(self.stamps).contiguous().to(self.device)
        self._onehot = _one_hot(np.asarray(self.labels), self.n_classes, self.device)

    def __iter__(self) -> Iterator[Batch]:
        """The batches of one pass."""
        n = self._views.shape[0]
        for start in range(0, n, self.batch_size):
            views = self._views[start : start + self.batch_size]
            features = torch.zeros((views.shape[0], 0), device=views.device)
            yield views, features, self._onehot[start : start + self.batch_size]

    def __len__(self) -> int:
        """The number of batches."""
        return -(-self._views.shape[0] // self.batch_size)


class ScratchTask:
    """The engine task of one trial: :class:`ScratchNetwork`, AdamW and the lr schedule.

    The rate of step ``s`` (:func:`~.optim.warmup_cosine_lr`) is set on the optimizer in
    :meth:`train_step`, before the loop's ``optimizer.step()``. The loss is the mean
    cross-entropy of the logits (the batches are class-stratified, so it is class-balanced).
    """

    def __init__(self, recipe: ScratchRecipe, lr: float, weight_decay: float) -> None:
        """Build the network on the CPU (from the global generator the engine seeded)."""
        self.recipe = recipe
        self.lr = lr
        self.weight_decay = weight_decay
        self._module = ScratchNetwork()
        self._optimizer: torch.optim.Optimizer | None = None

    @property
    def module(self) -> nn.Module:
        """The network."""
        return self._module

    def make_optimizer(self) -> torch.optim.Optimizer:
        """AdamW: decay on conv/linear weights, none on normalisation parameters and biases."""
        groups = [
            {"params": g["params"], "weight_decay": g["weight_decay"]}
            for g in lars_param_groups(self._module, self.weight_decay)
        ]
        self._optimizer = torch.optim.AdamW(groups, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        return self._optimizer

    def train_step(self, batch: TrainBatch, step: int) -> torch.Tensor:
        """Set the step's rate, then the batch's mean cross-entropy (graph attached)."""
        if self._optimizer is None:
            raise RuntimeError("make_optimizer() was not called")  # pragma: no cover
        rate = warmup_cosine_lr(
            step,
            base_lr=self.lr,
            total_steps=self.recipe.steps,
            warmup_steps=self.recipe.warmup_steps,
        )
        set_learning_rate(self._optimizer, rate)
        logits = self._module(batch[0])
        return F.cross_entropy(logits, batch[2].argmax(dim=1))

    def after_step(self, step: int) -> None:
        """Nothing: the rate is set before each step."""

    def eval_step(self, batch: Batch) -> EvalOutput:
        """The batch's loss, softmax and targets (under the loop's eval and no-grad)."""
        logits = self._module(batch[0])
        targets = batch[2].argmax(dim=1)
        loss = F.cross_entropy(logits, targets).detach().to(torch.float32)
        return EvalOutput(loss, torch.softmax(logits, dim=1).detach(), targets)

    def state(self) -> dict[str, Any]:
        """No state outside the weights and the optimizer."""
        return {}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Nothing to restore."""


def _predictions(
    outputs: Sequence[EvalOutput],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    predicted = torch.cat([o.probabilities.argmax(dim=1) for o in outputs]).cpu().numpy()
    targets = torch.cat([o.targets for o in outputs]).cpu().numpy()
    return predicted.astype(np.int64), targets.astype(np.int64)


def selection_score(outputs: Sequence[EvalOutput]) -> float:
    """Balanced accuracy of a validation's outputs (the engine's ``score``)."""
    predicted, targets = _predictions(outputs)
    return balanced_accuracy((predicted == targets).astype(np.float64), targets)


def _schedule(recipe: ScratchRecipe) -> Schedule:
    """A fixed-length fit validated once, after its last step; the last weights kept."""
    return Schedule(
        monitor=Monitor(MONITOR, "max"),
        validation=ValidationGrid(every_n_steps=recipe.steps),
        horizon=Horizon(initial=recipe.steps, increment=0),
        keep=("stop",),
        checkpoint_every_n_steps=None,
    )


def fit_trial(
    stream: ScratchStream,
    selection: EvalStamps,
    recipe: ScratchRecipe,
    *,
    lr: float,
    weight_decay: float,
    seed: int,
    device: DevicePolicy,
) -> tuple[ScratchTask, float]:
    """One trial on the engine: its task (final weights) and its selection-half score.

    ``seed`` seeds the global streams before the network is built (the engine's order).
    """
    task, result = fit(
        lambda: ScratchTask(recipe, lr, weight_decay),
        stream,
        selection,
        schedule=_schedule(recipe),
        device=device,
        seed=seed,
        score=selection_score,
    )
    if not isinstance(task, ScratchTask):  # pragma: no cover - fit returns build()'s task
        raise TypeError(f"expected a ScratchTask, got {type(task).__name__}")
    return task, result.validations[-1].value


def scratch_screen(
    train_images: torch.Tensor,
    train_labels: npt.NDArray[np.int64],
    subsets: Sequence[npt.NDArray[np.int64]],
    selection: tuple[torch.Tensor, npt.NDArray[np.int64]],
    decision: tuple[torch.Tensor, npt.NDArray[np.int64]],
    recipe: ScratchRecipe,
    *,
    seed: int,
    device: DevicePolicy,
    done: Mapping[int, dict[str, Any]] | None = None,
    on_subset: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Train, select and score the baseline on every subset (module docstring).

    ``train_images`` (N_train, 3, 63, 63) float32 on the host are every Train stamp in file
    order, ``train_labels`` their classes, ``subsets`` the Train indices of each subset;
    ``selection`` and ``decision`` are ``(stamps, labels)`` of the screen halves (host,
    63-pixel). ``done`` maps subset numbers to rows already computed (a resume: they are
    returned unchanged), ``on_subset`` gets each new row. Returns one row per subset:
    ``{"subset", "seed", "lr", "weight_decay", "selection_bacc", "decision_bacc",
    "trials": [{"lr", "weight_decay", "selection_bacc"}], "decision_predictions"}``.

    Raises:
        DataError: if a subset misses a class (:class:`ScratchStream`).
    """
    finished = dict(done or {})
    target = device.device
    selection_source = EvalStamps(selection[0], selection[1], target, recipe.eval_batch)
    decision_source = EvalStamps(decision[0], decision[1], target, recipe.eval_batch)
    rows: list[dict[str, Any]] = []
    for r, subset in enumerate(subsets):
        if r in finished:
            rows.append(finished[r])
            continue
        index = torch.from_numpy(np.asarray(subset, dtype=np.int64))
        host_stamps = train_images[index].contiguous()
        stamps = host_stamps.to(target)
        fits = None
        if recipe.views.noise:
            fits = torch.from_numpy(stamp_fits(host_stamps.numpy())).to(target)
        seed_r = subset_seed(seed, r)
        trials: list[dict[str, Any]] = []
        best: tuple[float, ScratchTask] | None = None
        for lr, wd in recipe.trials:
            stream = ScratchStream(
                stamps, train_labels[np.asarray(subset)], fits, seed=seed_r, recipe=recipe
            )
            task, score = fit_trial(
                stream, selection_source, recipe, lr=lr, weight_decay=wd, seed=seed_r, device=device
            )
            trials.append({"lr": lr, "weight_decay": wd, "selection_bacc": score})
            if best is None or score > best[0]:
                best = (score, task)
        assert best is not None  # noqa: S101 - the recipe's grid is not empty
        kept = best[1]
        predicted, targets = _predictions_of(kept, decision_source)
        row = {
            "subset": r,
            "seed": seed_r,
            "lr": kept.lr,
            "weight_decay": kept.weight_decay,
            "selection_bacc": best[0],
            "decision_bacc": balanced_accuracy((predicted == targets).astype(np.float64), targets),
            "trials": trials,
            "decision_predictions": predicted.tolist(),
        }
        rows.append(row)
        if on_subset is not None:
            on_subset(row)
    return rows


def _predictions_of(
    task: ScratchTask, source: EvalStamps
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The kept trial's predictions on ``source`` (``engine.loop.evaluate``: eval, inference)."""
    result = evaluate(task, {"eval": source})["eval"]
    predicted = result.probabilities.argmax(dim=1).numpy().astype(np.int64)
    return predicted, result.targets.numpy().astype(np.int64)
