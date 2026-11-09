"""The ``simclr-*`` subcommands of the CLI (phase 5, task 5.0b; ADR 0025).

Registered by :func:`add_simclr_commands` from :func:`astro_simclr.cli.build_parser`; the
handlers import torch and the pilot lazily, so ``--help`` stays fast.

- ``simclr-pilot DATA --output-dir OUT [--splits FILE] (--epochs N | --steps N) ...``: the
  pilot's pretraining (:func:`~astro_simclr.simclr.pilot.train_pilot`); ``--resume``
  continues ``OUT/ckpt/last.pt``; ``--stop-at N`` ends the call after N steps (a
  time-boxed job); ``--probe-every K --label-budgets FILE``, ``--abort-if-probe-below-random
  EPOCH`` and ``--online-probe`` turn on the probe monitors (:mod:`~astro_simclr.simclr.
  monitor`). DATA is a prepared layout, or a corpus index (``build-corpus-index``,
  task 5.1) with ``--archive DIR`` and ``--eval-data LAYOUT`` defaulting to the ones it
  records; ``--tier`` defaults to ``device`` for a layout and ``memmap`` for an index.
  Exit 0 when the call ends.
- ``simclr-embed DATA --output-dir OUT (--checkpoint FILE | --random-init) [--seed N]``:
  the embedding dump of a checkpoint, or of the randomly initialised network (the
  random-init probe of the go/no-go), without training.
- ``make-label-subsets DATA --splits FILE --per-class K --n-subsets R --out FILE``: stored
  label subsets of the Train stamps a screened run may use
  (:func:`~astro_simclr.simclr.probe.draw_label_subsets`).
- ``simclr-probe EMBEDDINGS --subsets FILE --splits FILE --out FILE``: the probe on the
  dumped ``h`` (:func:`~astro_simclr.simclr.probe.probe_screen`): the penalty chosen on
  the screen's selection half, the score on its decision half; prints the decision-half
  mean and writes the per-subset table as JSON. Exit 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from astro_simclr.config import ConfigError

__all__ = ["add_simclr_commands"]

# The monitors' defaults, as in ``simclr.monitor`` (kept here so that building the parser
# does not import torch; a test checks that they agree).
PROBE_BUDGETS = (10, 100)
DEFAULT_PART = "fold0"
DEFAULT_REPEAT = 0
ONLINE_BATCH = 256
ONLINE_LR = 1e-3


def _pilot(args: argparse.Namespace) -> int:
    from astro_simclr.simclr.pilot import PilotConfig, train_pilot

    config = PilotConfig(
        data=str(args.data),
        splits=None if args.splits is None else str(args.splits),
        epochs=None if args.steps is not None else args.epochs,
        steps=args.steps,
        batch_size=args.batch_size,
        temperature=args.temperature,
        lr_per_256=args.lr_per_256,
        warmup_epochs=args.warmup_epochs,
        noise_scale=args.noise_scale,
        seed=args.seed,
        device=args.device,
        fp32_precision=args.fp32_precision,
        deterministic=args.deterministic,
        tier=args.tier or ("memmap" if args.data.is_file() else "device"),
        archive=None if args.archive is None else str(args.archive),
        eval_data=None if args.eval_data is None else str(args.eval_data),
        host_cap_gb=args.host_cap_gb,
        allow_weak_index=args.allow_weak_index,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        diagnostics_every=args.diagnostics_every,
        knn_bank_per_class=args.knn_bank_per_class,
        probe_every=args.probe_every,
        label_budgets=None if args.label_budgets is None else str(args.label_budgets),
        probe_budgets=tuple(args.probe_budgets),
        probe_part=args.probe_part,
        probe_repeat=args.probe_repeat,
        abort_if_probe_below_random=args.abort_if_probe_below_random,
        online_probe=args.online_probe,
        online_probe_batch=args.online_probe_batch,
        online_probe_lr=args.online_probe_lr,
    )
    run = train_pilot(config, args.output_dir, resume=args.resume, stop_at=args.stop_at)
    if run.status == "aborted":
        assert run.abort is not None  # noqa: S101 - set by the abort rule
        sys.stdout.write(
            f"simclr-pilot: aborted at step {run.steps_done}: {run.abort['key']} "
            f"{run.abort['value']:.4f} <= random init {run.abort['random_init']:.4f} "
            f"({args.output_dir}/pilot.json)\n"
        )
        return 0
    sys.stdout.write(f"simclr-pilot: {run.steps_done} steps done in {args.output_dir}\n")
    return 0


def _embed(args: argparse.Namespace) -> int:
    import torch

    from astro_simclr.simclr.network import SimCLRNetwork
    from astro_simclr.simclr.pilot import (
        PilotConfig,
        dump_embeddings,
        load_checkpoint,
        load_pilot_data,
        resolve_device,
    )

    device = resolve_device(args.device)
    data = load_pilot_data(PilotConfig(data=str(args.data), batch_size=2, epochs=1))
    torch.manual_seed(args.seed)
    model = SimCLRNetwork()
    source: dict[str, Any]
    step = 0
    if args.checkpoint is not None:
        payload = load_checkpoint(args.checkpoint)
        if payload["data"]["source_sha256"] != data.record["source_sha256"]:
            raise ConfigError(f"{args.checkpoint} was trained on other data than {args.data}")
        model.load_state_dict(payload["model"])
        step = int(payload["steps_done"])
        source = {"checkpoint": str(args.checkpoint), "config": payload["config"]}
    else:
        source = {"random_init": True, "seed": args.seed}
    dump_embeddings(
        model.to(device),
        data,
        args.output_dir,
        device,
        batch_size=args.embed_batch,
        step=step,
        source=source,
    )
    sys.stdout.write(f"simclr-embed: wrote {args.output_dir}\n")
    return 0


def _make_label_subsets(args: argparse.Namespace) -> int:
    import numpy as np

    from astro_simclr.data.prepared import load_prepared
    from astro_simclr.protocol.splits import read_splits, verify_splits
    from astro_simclr.simclr.probe import draw_label_subsets, write_label_subsets

    if args.out.exists() and not args.overwrite:
        raise ConfigError(f"{args.out} already exists; pass --overwrite to replace it")
    labels = load_prepared(args.data).splits["Train"].labels
    if labels is None:
        raise ConfigError(f"{args.data} has no Train labels")
    splits = read_splits(args.splits)
    verify_splits(splits, args.data)
    candidates = np.setdiff1d(
        np.arange(labels.size), np.asarray(splits.screen_training_excluded(), dtype=np.int64)
    )
    try:
        subsets = draw_label_subsets(
            np.asarray(labels),
            candidates,
            per_class=args.per_class,
            n_subsets=args.n_subsets,
            seed=args.seed,
        )
    except ValueError as err:
        raise ConfigError(str(err)) from err
    write_label_subsets(subsets, args.out)
    sys.stdout.write(
        f"make-label-subsets: {args.n_subsets} subsets of {args.per_class} per class "
        f"-> {args.out}\n"
    )
    return 0


def _probe(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from astro_simclr.data.schema import DataError
    from astro_simclr.protocol.splits import read_splits
    from astro_simclr.simclr.probe import WEIGHT_DECAY_GRID, probe_screen, read_label_subsets

    folder: Path = args.embeddings
    meta = json.loads((folder / "meta.json").read_text())
    files = meta["files"]
    if "train_labels" not in files or "validation_labels" not in files:
        raise DataError(f"{folder}: the embeddings have no labels")
    train_h = torch.from_numpy(np.load(folder / files["train_h"]))
    train_y = np.load(folder / files["train_labels"])
    val_h = torch.from_numpy(np.load(folder / files["validation_h"]))
    val_y = np.load(folder / files["validation_labels"])
    subsets = read_label_subsets(args.subsets, train_y)
    splits = read_splits(args.splits)
    if splits.dataset["source_sha256"] != meta["data"]["source_sha256"]:
        raise DataError(f"{args.splits} was made for other data than the embeddings")
    screen = splits.screen
    report = probe_screen(
        train_h,
        train_y,
        subsets,
        (val_h[torch.from_numpy(screen.selection)], val_y[screen.selection]),
        (val_h[torch.from_numpy(screen.decision)], val_y[screen.decision]),
        grid=tuple(args.weight_decay) if args.weight_decay else WEIGHT_DECAY_GRID,
    )
    report["embeddings"] = {"path": str(folder), "step": meta["step"], "source": meta["source"]}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.write(
        f"simclr-probe: {report['per_class']} per class, decision-half balanced accuracy "
        f"{report['decision_bacc_mean']:.4f} +- {report['decision_bacc_std']:.4f} "
        f"over {len(report['subsets'])} subsets -> {args.out}\n"
    )
    return 0


def _add_pilot(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    pilot = commands.add_parser(
        "simclr-pilot",
        help="the 5.0b SimCLR pilot: pretrain ResNet-18 on a prepared layout's Train stamps",
        description="Pretrain SimCLR (ResNet-18 with a CIFAR stem, NT-Xent in paper mode, "
        "LARS) on the Train stamps of a prepared layout, or on the archive rows of a corpus "
        "index (evaluated on CD21), with checkpoints, diagnostics and a final embedding dump "
        "(ADR 0025). With --splits, the stamps a screened run may not use are left out and "
        "the kNN monitor queries the screen's selection half.",
    )
    pilot.add_argument(
        "data",
        type=Path,
        metavar="DATA",
        help="a prepared layout directory, or a corpus index file (build-corpus-index)",
    )
    pilot.add_argument("--output-dir", type=Path, required=True, help="the run directory")
    pilot.add_argument("--splits", type=Path, help="a stored splits file of DATA")
    length = pilot.add_mutually_exclusive_group()
    length.add_argument("--epochs", type=int, default=100, help="passes (default 100)")
    length.add_argument("--steps", type=int, help="optimizer steps instead of --epochs")
    pilot.add_argument("--batch-size", type=int, default=256)
    pilot.add_argument("--temperature", type=float, default=0.5)
    pilot.add_argument("--lr-per-256", type=float, default=0.3, help="lr = this * B / 256")
    pilot.add_argument("--warmup-epochs", type=float, default=10.0)
    pilot.add_argument("--noise-scale", type=float, default=0.0, help="C5 noise; 0 = off")
    pilot.add_argument("--seed", type=int, default=0)
    pilot.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    pilot.add_argument("--fp32-precision", default="ieee", choices=("ieee", "tf32"))
    pilot.add_argument("--deterministic", action="store_true")
    pilot.add_argument(
        "--tier",
        choices=("device", "host", "memmap"),
        help="where the pretraining stamps live (default: device for a layout, memmap for a "
        "corpus index; a corpus is never kept on the device)",
    )
    pilot.add_argument(
        "--archive", type=Path, help="the archive corpus of an index (default: the recorded one)"
    )
    pilot.add_argument(
        "--eval-data",
        type=Path,
        help="the CD21 layout evaluating a corpus run (default: the one the index excluded)",
    )
    pilot.add_argument(
        "--allow-weak-index",
        action="store_true",
        help='accept an index built with a radius below 1.5" or without the near-duplicate '
        "check (recorded; never for a result)",
    )
    pilot.add_argument(
        "--host-cap-gb",
        type=float,
        default=24.0,
        help="largest corpus held in host memory by --tier host (default 24)",
    )
    pilot.add_argument("--log-every", type=int, default=50)
    pilot.add_argument("--checkpoint-every", type=int, default=1000)
    pilot.add_argument("--diagnostics-every", type=int, default=1000)
    pilot.add_argument("--knn-bank-per-class", type=int, default=1000)
    _add_monitor_arguments(pilot)
    pilot.add_argument("--resume", action="store_true", help="continue OUT/ckpt/last.pt")
    pilot.add_argument(
        "--stop-at", type=int, help="end this call after this many steps (resume later)"
    )
    pilot.set_defaults(handler=_pilot)


def _add_monitor_arguments(pilot: argparse.ArgumentParser) -> None:
    """The probe monitors' options (``simclr.monitor``; all off by default)."""
    monitors = pilot.add_argument_group(
        "probe monitors",
        "Linear probes on h of the eval view (centre crop, eval mode, no augmentation, no "
        "features), scored on the screen's selection half only: diagnostics and an abort "
        "input, never a checkpoint selector.",
    )
    monitors.add_argument(
        "--probe-every",
        type=int,
        metavar="K",
        help="the periodic budget probe every K passes and at the final step (needs "
        "--splits and --label-budgets)",
    )
    monitors.add_argument(
        "--label-budgets",
        type=Path,
        help="a stored label-budget file drawn from --splits (make-label-budgets)",
    )
    monitors.add_argument(
        "--probe-budgets",
        type=int,
        nargs="+",
        default=list(PROBE_BUDGETS),
        help=f"objects per class (default {' '.join(map(str, PROBE_BUDGETS))})",
    )
    monitors.add_argument(
        "--probe-part", default=DEFAULT_PART, help=f"the file's part (default {DEFAULT_PART})"
    )
    monitors.add_argument(
        "--probe-repeat", type=int, default=DEFAULT_REPEAT, help="the file's repeat (default 0)"
    )
    monitors.add_argument(
        "--abort-if-probe-below-random",
        type=int,
        metavar="EPOCH",
        help="stop cleanly after pass EPOCH unless the largest budget's probe beats the "
        "random-init probe",
    )
    monitors.add_argument(
        "--online-probe",
        action="store_true",
        help="train a stop-gradient Linear(512 -> 5) head on labelled Train side-batches",
    )
    monitors.add_argument("--online-probe-batch", type=int, default=ONLINE_BATCH)
    monitors.add_argument("--online-probe-lr", type=float, default=ONLINE_LR, help="Adam rate")


def add_simclr_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``simclr-*`` and ``make-label-subsets`` subcommands."""
    _add_pilot(commands)

    embed = commands.add_parser(
        "simclr-embed",
        help="dump h of a pilot checkpoint, or of the random-init network",
        description="Write h (centred 57-pixel view, eval mode) of every Train and "
        "Validation stamp of DATA, from a pilot checkpoint or from the network initialised "
        "with --seed (the random-init baseline).",
    )
    embed.add_argument("data", type=Path, metavar="DATA", help="a prepared layout directory")
    embed.add_argument("--output-dir", type=Path, required=True)
    source = embed.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="a pilot checkpoint (ckpt/*.pt)")
    source.add_argument("--random-init", action="store_true", help="no training")
    embed.add_argument("--seed", type=int, default=0, help="the random init's seed")
    embed.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    embed.add_argument("--embed-batch", type=int, default=1024)
    embed.set_defaults(handler=_embed)

    subsets = commands.add_parser(
        "make-label-subsets",
        help="store label subsets of K Train stamps per class for the probe",
        description="Draw R nested label subsets of K stamps per class from the Train "
        "stamps a screened run may use (the splits file's exclusions left out).",
    )
    subsets.add_argument("data", type=Path, metavar="DATA", help="a prepared layout directory")
    subsets.add_argument("--splits", type=Path, required=True)
    subsets.add_argument("--per-class", type=int, required=True)
    subsets.add_argument("--n-subsets", type=int, default=20)
    subsets.add_argument("--seed", type=int, default=0)
    subsets.add_argument("--out", type=Path, required=True)
    subsets.add_argument("--overwrite", action="store_true")
    subsets.set_defaults(handler=_make_label_subsets)

    probe = commands.add_parser(
        "simclr-probe",
        help="the linear probe on dumped h, scored on the screen halves",
        description="Fit the logistic-regression probe on each label subset, choose its L2 "
        "penalty on the screen's selection half and score it on the decision half.",
    )
    probe.add_argument("embeddings", type=Path, metavar="EMBEDDINGS", help="an embeddings dir")
    probe.add_argument("--subsets", type=Path, required=True)
    probe.add_argument("--splits", type=Path, required=True)
    probe.add_argument("--out", type=Path, required=True, help="the JSON report")
    probe.add_argument(
        "--weight-decay", type=float, nargs="+", help="the L2 grid (default: the declared one)"
    )
    probe.set_defaults(handler=_probe)
