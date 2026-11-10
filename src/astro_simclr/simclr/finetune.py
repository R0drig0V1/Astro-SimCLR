"""The fine-tuning arm of the label-budget screen: the scratch arm from a pilot's encoder.

Phase 5 (plan 5.4, "Mode: fine-tuning only for H1 and H2"; ADR 0025, amendment "the
fine-tuning arm"). ``simclr-finetune`` is :mod:`.scratch`'s arm with one difference: the
encoder of :class:`~.scratch.ScratchNetwork` starts from a ``simclr-pilot`` checkpoint
(``ckpt/final.pt``, ``ckpt/last.pt`` or a kept ``ckpt/step-N.pt``) instead of its random
init. Everything else is the scratch arm's, on purpose, so that the paired difference
fine-tune - scratch isolates the initialisation:

- the same recipe, grid, step counts, final weights and selection rule
  (:class:`~.scratch.ScratchRecipe`, :func:`~.scratch.scratch_screen`);
- the **same draws** (common random numbers): stream 13 and :func:`~.scratch.draw_step`,
  so subset ``r`` sees the same batches, views and noise at every step as in the scratch
  run of the same ``--seed``; and the same ``Linear(512, 5)`` head init, because the
  network is still built encoder first from the seeded global generator (the encoder's
  random init is drawn, then overwritten), so the head's weights are the scratch arm's.

**What is loaded** (:func:`load_pilot_init`): only the ``encoder.*`` entries of the
checkpoint's ``model`` state dict, the convolution and BatchNorm weights and the BatchNorm
buffers (``running_mean``, ``running_var``, ``num_batches_tracked``; the momentum is 0.1,
so the count changes nothing). The projection head, the LARS state and the monitors are
ignored. The checkpoint must be a pilot checkpoint of a version in
:data:`KNOWN_CKPT_VERSIONS` whose encoder has exactly :func:`~.network.resnet18_cifar`'s
keys, shapes and dtypes, with finite values; anything else is refused with the reason.

**What is recorded** (:attr:`PilotInit.record`, the report's ``init``): the checkpoint's
path, SHA-256, format version, step and total steps, its config, and the pretraining data
it holds: a CD21 layout run (the pilot's: CD21 Train, less the splits file's exclusions,
was in pretraining, unlabelled) or a corpus index run (the corpus run's: it was not;
``cd21_train_in_pretraining``). :func:`check_init_data` refuses a checkpoint pretrained
against another layout or another splits file than the fine-tune's (the screen halves'
objects would not be the ones the pretraining left out).

Nothing here trains; :mod:`.commands` passes :attr:`PilotInit.encoder` to
:func:`~.scratch.scratch_screen`.
"""

from __future__ import annotations

import pickle  # only its exception type: torch.load reads with weights_only
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import torch

from astro_simclr.config import ConfigError
from astro_simclr.data.files import sha256_file
from astro_simclr.data.schema import DataError
from astro_simclr.simclr.network import resnet18_cifar
from astro_simclr.simclr.pilot import load_checkpoint

__all__ = [
    "ENCODER_PREFIX",
    "KNOWN_CKPT_VERSIONS",
    "PilotInit",
    "check_init_data",
    "encoder_state",
    "load_pilot_init",
]

#: The pilot checkpoint versions whose ``model`` this module knows how to read.
KNOWN_CKPT_VERSIONS: Final = (1,)
#: The prefix of the encoder's entries in a pilot's ``SimCLRNetwork`` state dict.
ENCODER_PREFIX: Final = "encoder."


@dataclass(frozen=True, slots=True)
class PilotInit:
    """A pilot checkpoint's encoder state (CPU tensors, read-only mapping) and its record."""

    encoder: Mapping[str, torch.Tensor]
    record: dict[str, Any]

    @property
    def sha256(self) -> str:
        """The checkpoint file's SHA-256 (the fine-tune report's resume identity)."""
        return str(self.record["sha256"])


def _reference_encoder() -> dict[str, torch.Tensor]:
    """``resnet18_cifar()``'s state dict, built without moving the global generator."""
    with torch.random.fork_rng(devices=[]):
        return dict(resnet18_cifar().state_dict())


def encoder_state(model: Mapping[str, Any], source: str = "the checkpoint") -> dict[str, Any]:
    """The ``encoder.*`` entries of a ``SimCLRNetwork`` state dict, prefix removed, checked.

    Raises:
        DataError: if an encoder key is missing or unexpected, or an entry is not a tensor
            of the reference encoder's shape and dtype, or a floating entry is not finite.
    """
    reference = _reference_encoder()
    state = {
        key[len(ENCODER_PREFIX) :]: value
        for key, value in model.items()
        if key.startswith(ENCODER_PREFIX)
    }
    missing = sorted(set(reference) - set(state))
    unexpected = sorted(set(state) - set(reference))
    if missing or unexpected:
        raise DataError(
            f"{source}: its encoder is not resnet18_cifar's (missing {missing[:3]}, "
            f"unexpected {unexpected[:3]})"
        )
    for key, ref in reference.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise DataError(f"{source}: encoder entry {key} is not a tensor")
        if value.shape != ref.shape or value.dtype != ref.dtype:
            raise DataError(
                f"{source}: encoder entry {key} is {tuple(value.shape)} {value.dtype}, "
                f"expected {tuple(ref.shape)} {ref.dtype}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise DataError(f"{source}: encoder entry {key} has non-finite values")
    return {key: state[key].detach().cpu().clone() for key in reference}


def _pretraining(data: Mapping[str, Any]) -> dict[str, Any]:
    """The pretraining data a checkpoint's ``data`` record describes (module docstring)."""
    pretrain = data.get("pretrain")
    common = {
        "eval_layout": data.get("path"),
        "source_sha256": data.get("source_sha256"),
        "splits_sha256": data.get("splits_sha256"),
        "n_pretrain": data.get("n_pretrain"),
    }
    if pretrain is None:
        return {
            "kind": "cd21-layout",
            **common,
            "n_excluded_train": len(data.get("excluded_train", [])),
            "cd21_train_in_pretraining": True,
        }
    return {
        "kind": str(pretrain.get("kind")),
        **common,
        "index": pretrain.get("index"),
        "index_sha256": pretrain.get("index_sha256"),
        "exclusion_sha256": pretrain.get("exclusion_sha256"),
        "cd21_train_in_pretraining": False,
    }


def load_pilot_init(path: Path) -> PilotInit:
    """Read the encoder of a ``simclr-pilot`` checkpoint and its record (module docstring).

    Raises:
        DataError: if the file is unreadable or not a pilot checkpoint
            (:func:`~.pilot.load_checkpoint`), of an unknown version, without a ``model``,
            ``data`` or ``steps_done``, or its encoder does not fit (:func:`encoder_state`).
    """
    try:
        payload = load_checkpoint(path)
    except (pickle.UnpicklingError, EOFError) as err:  # not a torch.save file at all
        raise DataError(f"{path}: cannot read the checkpoint: {err}") from err
    version = payload.get("format_version")
    if version not in KNOWN_CKPT_VERSIONS:
        known = ", ".join(str(v) for v in KNOWN_CKPT_VERSIONS)
        raise DataError(
            f"{path}: a pilot checkpoint of version {version!r}; this command reads version {known}"
        )
    for key in ("model", "data", "steps_done"):
        if key not in payload:
            raise DataError(f"{path}: the pilot checkpoint has no {key!r}")
    model, data = payload["model"], payload["data"]
    if not isinstance(model, Mapping) or not isinstance(data, Mapping):
        raise DataError(f"{path}: the pilot checkpoint's model or data is not a mapping")
    encoder = encoder_state(model, str(path))
    total = payload.get("total_steps")
    steps = int(payload["steps_done"])
    record = {
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "format_version": version,
        "steps_done": steps,
        "total_steps": total,
        "pretraining_complete": total is not None and steps == int(total),
        "loaded": "encoder weights and BatchNorm buffers (encoder.*); head, optimizer ignored",
        "pretraining": _pretraining(data),
        "config": payload.get("config"),
    }
    return PilotInit(MappingProxyType(encoder), record)


def check_init_data(init: PilotInit, data_record: Mapping[str, Any]) -> None:
    """Refuse a checkpoint pretrained against another layout or splits file than ``data``'s.

    ``data_record`` is :func:`~.pilot.load_pilot_data`'s record of the fine-tune's layout
    (loaded with its splits file).

    Raises:
        ConfigError: if the source SHA-256 or the splits file SHA-256 differ.
    """
    pretraining = init.record["pretraining"]
    if pretraining["source_sha256"] != data_record.get("source_sha256"):
        raise ConfigError(
            f"{init.record['checkpoint']} was pretrained against another layout than "
            f"{data_record.get('path')}"
        )
    if pretraining["splits_sha256"] != data_record.get("splits_sha256"):
        raise ConfigError(
            f"{init.record['checkpoint']} was pretrained with another splits file (or none): "
            "its pretraining did not leave out this screen's objects"
        )
