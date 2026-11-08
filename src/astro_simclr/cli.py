"""The ``astro-simclr`` command line (``[project.scripts]``).

Each subcommand registers itself in :func:`build_parser` and reads its settings through
:func:`add_settings_arguments` / :func:`settings_from_args`, so that every command accepts
the same ``--config FILE`` and ``--set key=value [key=value ...]`` options. Subcommands:

- ``show-config``: the resolved settings;
- data (task 1.2, ``docs/data.md``): ``inspect-data``, ``make-synthetic``, ``prepare-data``
  (``--images float32``, task 2.1), and ``check-features`` (task 2.1: were the stored
  features standardised with M1's statistics? Exit 1 if not);
- training (task 1.7, :mod:`astro_simclr.experiments`): ``pretrain``, ``linear-eval``,
  ``fine-tune``, ``train-supervised``. Their settings resolve as: the ``Settings``
  defaults, the subcommand's legacy protocol (``hparams.COMMAND_DEFAULTS``), ``--hparams``
  (a legacy ``results/hparams_best_*.yaml``, through the mapping of
  :mod:`astro_simclr.hparams`), ``--config``, ``--set``, then the explicit flags
  (``--data-dir``, ``--output-dir``, ``--seed``, ``--accelerator``, ``--pretrained``);
  later wins. The legacy file comes first so that a config can correct it (R1's AdamW);
- ``compare-reference RUN_DIR... --ref R1|R2|R3|R4|CD21 [--oracle-report FILE]``
  (:mod:`astro_simclr.reference`): exit 0 for PASS or REPORT, 1 for FAIL;
- protocol (task 2.3, :mod:`astro_simclr.protocol`): ``make-splits`` writes the stored
  decision splits of a prepared dataset; ``compare-variants --baseline RUN... --variant
  RUN...`` reports McNemar on the seed ensembles and the paired bootstrap (always exit 0:
  a report, never a decision);
- ``preflight`` (:mod:`astro_simclr.preflight`): exit 0 if no step failed, else 1.
- M1 oracle (task 2.2b, :mod:`astro_simclr.oracle`): ``fetch-m1`` downloads M1 at the
  pinned commit (SHA-256 checked) into a local cache and converts it to ``.npz``;
  ``m1-oracle --data-dir DIR --out DIR`` runs the port-equivalence and data-path checks
  and writes ``m1_oracle.json`` and ``m1_val_curve.csv``: exit 0 if both passed, else 1.

The heavy modules (torch, Lightning) are imported inside the handlers, so ``show-config``
and ``--help`` stay fast. A :class:`~astro_simclr.data.DataError` (bad, truncated or missing
data) prints one ``astro-simclr: error: ...`` line and exits with 1; invalid settings,
arguments or an unavailable accelerator exit with 2.

``--set`` takes one or more ``key=value`` items (and may be repeated), so put it after the
positional arguments of a command, or end it with ``--``.

:func:`main` takes ``argv`` explicitly so tests can call it in-process; it returns the exit
code instead of calling ``sys.exit``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from astro_simclr.config import (
    ACCELERATORS,
    ConfigError,
    Settings,
    load_settings,
    parse_override,
    read_settings_file,
    settings_from_mapping,
    settings_to_plain,
)
from astro_simclr.hparams import COMMAND_DEFAULTS, COMMANDS, load_legacy_hparams

if TYPE_CHECKING:
    from astro_simclr.oracle.m1 import M1Files

__all__ = [
    "add_settings_arguments",
    "build_parser",
    "main",
    "settings_from_args",
    "training_settings",
]

#: Exit code for invalid settings or arguments (argparse also uses 2 for usage errors).
EXIT_USAGE = 2
#: Exit code for unusable data: missing, truncated, corrupt or of the wrong schema.
EXIT_DATA = 1


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--config`` and ``--set`` options to a subcommand parser."""
    parser.add_argument(
        "--config", type=Path, metavar="FILE", help="YAML settings file (see config.Settings)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="override settings (one or more KEY=VALUE; repeatable); VALUE is parsed as YAML",
    )


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Resolve the settings of a parsed command line (``--config``, then ``--set``).

    Raises:
        ConfigError: if the file or any override is invalid.
    """
    config: Path | None = args.config
    overrides: list[str] = args.overrides
    return load_settings(config, overrides)


def _show_config(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    # Paths are printed as strings so the output is itself a valid settings file.
    sys.stdout.write(yaml.safe_dump(settings_to_plain(settings), sort_keys=False))
    return 0


def _inspect_data(args: argparse.Namespace) -> int:
    from astro_simclr.data.inspection import format_summary, summarize

    settings_from_args(args)  # validates --config/--set, which inspect-data does not use
    path: Path = args.path
    summary = summarize(path)
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    else:
        sys.stdout.write(format_summary(summary))
    return 0


def _make_synthetic(args: argparse.Namespace) -> int:
    from astro_simclr.data.synthetic import make_synthetic

    settings = settings_from_args(args)
    seed: int = settings.seed if args.seed is None else args.seed
    out: Path = args.out
    try:
        paths = make_synthetic(out, args.n_per_split, seed)
    except ValueError as err:
        raise ConfigError(str(err)) from None
    for path in paths:
        sys.stdout.write(f"wrote {path}\n")
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    from astro_simclr.data.prepared import prepare

    settings = settings_from_args(args)
    out: Path = settings.data_dir if args.out is None else args.out
    source: Path = args.pickle
    dest = prepare(source, out, overwrite=args.overwrite, images=args.images)
    sys.stdout.write(f"wrote {dest}\n")
    return 0


def _check_features(args: argparse.Namespace) -> int:
    from astro_simclr.data.m1_features import check_destandardisation
    from astro_simclr.data.prepared import load_prepared

    settings_from_args(args)  # validates --config/--set, which check-features does not use
    path: Path = args.path
    data = load_prepared(path)
    report: dict[str, list[dict[str, object]]] = {}
    for split, content in data.splits.items():
        if content.features is None:
            continue
        report[split] = [
            {"feature": c.feature, "test": c.test, "passed": c.passed, "detail": c.detail}
            for c in check_destandardisation(content.features)
        ]
    passed = all(check["passed"] for checks in report.values() for check in checks)
    if args.json:
        sys.stdout.write(json.dumps({"path": str(path), "passed": passed, "splits": report}))
        sys.stdout.write("\n")
    else:
        lines = [f"{path}: features de-standardised with M1's statistics"]
        for split, checks in report.items():
            lines.append(f"  {split}:")
            lines.extend(
                f"    {'PASS' if c['passed'] else 'FAIL'}  {c['feature']:<15} {c['test']:<9} "
                f"{c['detail']}"
                for c in checks
            )
        lines.append("PASS" if passed else "FAIL")
        sys.stdout.write("\n".join(lines) + "\n")
    return 0 if passed else 1


def _add_data_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    inspect = commands.add_parser(
        "inspect-data",
        help="describe and check a stamp pickle or a prepared directory",
        description="Print keys, dtypes, shapes, counts per class, size and SHA-256 of a "
        "stamp pickle or a prepared directory, and check it against its schema. Exits "
        "with 1 if the data is truncated, corrupt or does not match the schema.",
    )
    inspect.add_argument("path", type=Path, metavar="PATH", help="a .pkl or a prepared dir")
    inspect.add_argument("--json", action="store_true", help="print the summary as JSON")
    add_settings_arguments(inspect)
    inspect.set_defaults(handler=_inspect_data)

    synthetic = commands.add_parser(
        "make-synthetic",
        help="write the four synthetic stand-ins of the ZTF pickles",
        description="Write synthetic versions of the three classifier pickles and the "
        "pretraining pickle, with the real schemas and a strong class pattern. The same "
        "seed gives byte-identical files.",
    )
    synthetic.add_argument("--out", type=Path, required=True, metavar="DIR")
    synthetic.add_argument(
        "--n-per-split",
        type=int,
        default=50,
        metavar="N",
        help="stamps per split of every file (default: 50; >= 5)",
    )
    synthetic.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="random seed (default: the seed setting)",
    )
    add_settings_arguments(synthetic)
    synthetic.set_defaults(handler=_make_synthetic)

    prepare = commands.add_parser(
        "prepare-data",
        help="convert a stamp pickle into the prepared .npy layout",
        description="Unpickle a trusted stamp pickle (it needs about its size in RAM), "
        "check it, and write OUT/<pickle stem>/ with uint8 images, labels, features and "
        "meta.json. Training reads only this layout. With --images float32 it also writes "
        "the stored float32 stamps (images_f32.npy, 4x the size of images.npy), which the "
        "paper's float input path needs.",
    )
    prepare.add_argument("pickle", type=Path, metavar="PKL")
    prepare.add_argument(
        "--images",
        choices=("uint8", "float32"),
        default="uint8",
        help="uint8: images.npy only (default); float32: also images_f32.npy",
    )
    prepare.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="parent directory (default: the data_dir setting)",
    )
    prepare.add_argument(
        "--overwrite", action="store_true", help="replace an existing prepared directory"
    )
    add_settings_arguments(prepare)
    prepare.set_defaults(handler=_prepare_data)

    features = commands.add_parser(
        "check-features",
        help="check that the stored features were standardised with M1's statistics",
        description="De-standardise the features of a prepared directory with the "
        "normalisation statistics of the paper's model (M1) and check them: integer "
        "ndethist, ncovhist and non_detections; a floor of -1 on the clipped sgscore and "
        "distpsnr features; isdiffpos in {0, 1}. Exits with 1 if a check fails.",
    )
    features.add_argument("path", type=Path, metavar="DIR", help="a prepared directory")
    features.add_argument("--json", action="store_true", help="print the report as JSON")
    add_settings_arguments(features)
    features.set_defaults(handler=_check_features)


def training_settings(command: str, args: argparse.Namespace) -> Settings:
    """Resolve the settings of a training subcommand (the order is in the module docstring).

    Keys of ``--hparams`` that belong to another subcommand are reported on stderr.

    Raises:
        ConfigError: if any source is invalid, or the legacy file does not map.
    """
    data: dict[str, object] = dict(COMMAND_DEFAULTS[command])
    if args.hparams is not None:
        legacy = load_legacy_hparams(args.hparams, command)
        if legacy.skipped:
            sys.stderr.write(
                f"astro-simclr: note: {args.hparams}: not used by {command}: "
                f"{', '.join(legacy.skipped)}\n"
            )
        data.update(legacy)
    if args.config is not None:
        data.update(read_settings_file(args.config))
    for text in args.overrides:
        key, value = parse_override(text)
        data[key] = value
    flags = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "accelerator": args.accelerator,
        "pretrained": getattr(args, "pretrained", None),
    }
    data.update({key: value for key, value in flags.items() if value is not None})
    return settings_from_mapping(data)


def _train(args: argparse.Namespace) -> int:
    from astro_simclr.experiments import METRICS_FILE, run_command

    command: str = args.command
    settings = training_settings(command, args)
    metrics = run_command(command, settings, resume=args.resume)
    summary = ", ".join(
        f"{key} {metrics[key]}"
        for key in ("best_epoch", "monitor_best", "val_acc_best", "test_acc_at_best")
        if key in metrics
    )
    sys.stdout.write(f"{command}: {summary}; wrote {settings.output_dir / METRICS_FILE}\n")
    return 0


_TRAINING_HELP = {
    "pretrain": "SimCLR pretraining of an encoder (legacy iter_clr.py)",
    "linear-eval": "linear evaluation of a pretrained encoder (legacy linear_classifier.py)",
    "fine-tune": "fine-tuning of a pretrained encoder (legacy fine_tuning.py)",
    "train-supervised": "supervised CE baseline: stamp classifier or ResNet (iter_ce_*.py)",
}


def _add_training_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    for command in COMMANDS:
        sub = commands.add_parser(
            command,
            help=_TRAINING_HELP[command],
            description=f"{_TRAINING_HELP[command]}. Writes checkpoints, metrics.json and "
            "(classifiers) confusion-matrix figures under the output directory. Dry mode: "
            "--set max_epochs=1 limit_train_batches=2 limit_val_batches=2.",
        )
        add_settings_arguments(sub)
        sub.add_argument(
            "--hparams",
            type=Path,
            metavar="FILE",
            help="legacy results/hparams_best_*.yaml, mapped per subcommand (applied before "
            "--config)",
        )
        sub.add_argument("--data-dir", metavar="DIR", help="parent of the prepared datasets")
        sub.add_argument("--output-dir", metavar="DIR", help="where every output goes")
        sub.add_argument("--seed", type=int, metavar="N", help="random seed")
        sub.add_argument(
            "--accelerator", choices=sorted(ACCELERATORS), help="cpu, cuda, mps, gpu or auto"
        )
        sub.add_argument(
            "--resume",
            metavar="PATH|auto",
            help="continue from a checkpoint; auto: OUTPUT_DIR/checkpoints/last.ckpt if any",
        )
        if command in ("linear-eval", "fine-tune"):
            sub.add_argument(
                "--pretrained",
                metavar="PATH",
                help="the pretrain checkpoint, or a pretrain output directory",
            )
        sub.set_defaults(handler=_train)


def _compare_reference(args: argparse.Namespace) -> int:
    from astro_simclr.reference import compare_reference, exit_code

    comparison = compare_reference(args.ref, args.run_dirs, oracle_report=args.oracle_report)
    sys.stdout.write("\n".join(comparison.lines) + "\n")
    return exit_code(comparison.result)


def _compare_variants(args: argparse.Namespace) -> int:
    from astro_simclr.protocol.compare import compare_variants

    comparison = compare_variants(
        args.baseline,
        args.variant,
        stage=args.split,
        n_boot=args.n_boot,
        seed=args.seed,
        allow_mixed_machines=args.allow_mixed_machines,
    )
    sys.stdout.write("\n".join(comparison.lines) + "\n")
    return 0


def _make_splits(args: argparse.Namespace) -> int:
    from astro_simclr.protocol.splits import make_splits, write_splits

    if args.out.exists() and not args.overwrite:
        raise ConfigError(f"{args.out} already exists; pass --overwrite to replace it")
    splits = make_splits(args.data, seed=args.seed)
    write_splits(splits, args.out)
    for dup in splits.duplicates:
        sys.stdout.write(
            f"duplicate: {dup['split_a']} {dup['index_a']} ({dup['label_a']}) = "
            f"{dup['split_b']} {dup['index_b']} ({dup['label_b']}), features equal: "
            f"{dup['features_equal']}\n"
        )
    sizes = ", ".join(
        f"fold {f.fold}: {f.training.size}/{f.selection.size}/{f.evaluation.size}"
        for f in splits.folds
    )
    sys.stdout.write(
        f"excluded from Train: {list(splits.excluded_train)}\n"
        f"screen: {splits.screen.selection.size} selection / {splits.screen.decision.size} "
        f"decision Validation stamps\n"
        f"cv (training/selection/evaluation): {sizes}\n"
        f"wrote {args.out}\n"
    )
    return 0


def _preflight(args: argparse.Namespace) -> int:
    from astro_simclr.preflight import run_preflight

    runs = None if args.runs is None else [n.strip() for n in args.runs.split(",") if n.strip()]
    passed = run_preflight(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        accelerator=args.accelerator,
        reference_dir=args.reference_dir,
        runs=runs,
        overrides=args.overrides,
        n_per_split=args.n_per_split,
        seed=args.seed,
        steps=args.steps,
    )
    return 0 if passed else EXIT_DATA


def _add_check_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    compare = commands.add_parser(
        "compare-reference",
        help="judge reference runs against their gate (PASS, FAIL or REPORT)",
        description="Aggregate the metrics.json of the given run directories (mean, std "
        "with ddof=1) and apply the gate of the reference run: the pass rule (R1, R2), the "
        "majority-class baseline (R3) or report only (R4). Exit 0 for PASS or REPORT, 1 "
        "for FAIL.",
    )
    compare.add_argument("run_dirs", type=Path, nargs="+", metavar="RUN_DIR")
    compare.add_argument("--ref", required=True, choices=["R1", "R2", "R3", "R4", "CD21"])
    compare.add_argument(
        "--oracle-report",
        type=Path,
        default=None,
        metavar="FILE",
        help="CD21 only: the M1 oracle report (JSON with a boolean 'passed'); the faithful "
        "tier needs it",
    )
    compare.set_defaults(handler=_compare_reference)
    _add_protocol_commands(commands)

    preflight = commands.add_parser(
        "preflight",
        help="check that this machine can run the reference runs",
        description="CUDA and torch build, RAM and disk, the data and its manifest, a dry "
        "run of every reference config and the GPU-hour estimate. Exit 0 if no step "
        "failed, else 1.",
    )
    source = preflight.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--synthetic", action="store_true", help="generate synthetic data and use it"
    )
    source.add_argument("--data-dir", type=Path, metavar="DIR", help="the prepared real data")
    preflight.add_argument(
        "--output-dir", type=Path, default=Path("outputs/preflight"), metavar="DIR"
    )
    preflight.add_argument(
        "--accelerator", default="cuda", choices=sorted(ACCELERATORS), help="default: cuda"
    )
    preflight.add_argument(
        "--reference-dir", type=Path, default=Path("configs/reference"), metavar="DIR"
    )
    preflight.add_argument(
        "--runs",
        default=None,
        metavar="LIST",
        help="comma-separated, e.g. R1,R2,R4; each must run (default: every run whose data "
        "is present)",
    )
    preflight.add_argument(
        "--n-per-split", type=int, default=2400, metavar="N", help="synthetic stamps per split"
    )
    preflight.add_argument("--steps", type=int, default=20, metavar="N", help="timed steps")
    preflight.add_argument("--seed", type=int, default=0, metavar="N")
    preflight.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="settings applied to every dry run (e.g. num_workers=0)",
    )
    preflight.set_defaults(handler=_preflight)


def _add_protocol_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    variants = commands.add_parser(
        "compare-variants",
        help="McNemar and paired bootstrap of two sets of runs (a report)",
        description="Pair the per-stamp predictions of a baseline and a variant (one run "
        "per seed each): McNemar on the seed-ensembled predictions and a class-stratified "
        "paired bootstrap over stamps of the seed-mean accuracy difference (variant - "
        "baseline). Always a report (exit 0): test is never a decision input "
        "(docs/evaluation-protocol.md).",
    )
    variants.add_argument("--baseline", type=Path, nargs="+", required=True, metavar="RUN_DIR")
    variants.add_argument("--variant", type=Path, nargs="+", required=True, metavar="RUN_DIR")
    variants.add_argument("--split", choices=["test", "val"], default="test")
    variants.add_argument("--n-boot", type=int, default=10_000, metavar="N")
    variants.add_argument("--seed", type=int, default=0, metavar="N", help="bootstrap seed")
    variants.add_argument(
        "--allow-mixed-machines",
        action="store_true",
        help="compare runs from different machines or builds (the plan forbids it)",
    )
    variants.set_defaults(handler=_compare_variants)

    splits = commands.add_parser(
        "make-splits",
        help="write the stored decision splits (screen, 5-fold CV) of a prepared dataset",
        description="Find the stamps duplicated across splits (removed from Train only), "
        "then draw the 250/250 screen of Validation and the stratified 5-fold CV on Train + "
        "Validation, each fold with its own 100-per-class selection set; write them as "
        "JSON (docs/evaluation-protocol.md).",
    )
    splits.add_argument("data", type=Path, metavar="PREPARED_DIR")
    splits.add_argument("--out", type=Path, required=True, metavar="FILE")
    splits.add_argument("--seed", type=int, default=0, metavar="N")
    splits.add_argument("--overwrite", action="store_true")
    splits.set_defaults(handler=_make_splits)


def _fetch(args: argparse.Namespace) -> M1Files:
    """Run :func:`~astro_simclr.oracle.m1.fetch_m1`; download or file errors are data errors."""
    from astro_simclr.data.schema import DataError
    from astro_simclr.oracle.m1 import fetch_m1

    try:
        return fetch_m1(args.cache_dir, source_dir=args.source_dir)
    except (OSError, ValueError) as err:
        raise DataError(f"cannot fetch M1: {err}") from None


def _fetch_m1(args: argparse.Namespace) -> int:
    files = _fetch(args)
    sys.stdout.write(f"wrote {files.weights}\nval log {files.val_events}\n")
    return 0


def _m1_oracle(args: argparse.Namespace) -> int:
    from astro_simclr.oracle.m1 import load_weights
    from astro_simclr.oracle.run import read_val_curve, run_m1_oracle, write_curve, write_report

    files = _fetch(args)
    curve = read_val_curve(files.val_events)
    report = run_m1_oracle(args.data_dir, load_weights(files.weights), curve=curve)
    out: Path = args.out
    write_curve(curve, out / "m1_val_curve.csv")
    write_report(report, out / "m1_oracle.json")
    port, data = report["checks"]["port_equivalence"], report["checks"]["data_path"]
    approx = report["nan_approximation"]["approximated"]
    lines = [
        f"port equivalence: max |delta logit| = {port['max_abs_logit_diff']:.3g} "
        f"(<= {port['tolerance']:g}), argmax agreement {port['argmax_agreement']:.3f} -> "
        f"{'passed' if port['passed'] else 'FAILED'}",
        f"data path: {data['correct']}/{data['n']} correct (M1 {data['expected_correct']} "
        f"± {data['correct_tolerance']}), loss {data['loss']:.4f} (M1 "
        f"{data['expected_loss']} ± {data['loss_tolerance']}) -> "
        f"{'passed' if data['passed'] else 'FAILED'}",
        f"NaN-approximated Validation stamps: {approx['correct']}/{approx['n']} correct",
        f"test: {report['test']['correct']}/{report['test']['n']} correct "
        f"(accuracy {report['test']['accuracy']:.3f})",
    ]
    lines += [
        f"diagnostic {name}: Validation accuracy {d['accuracy']:.3f} "
        f"(delta {d['delta_accuracy']:+.3f})"
        for name, d in report["diagnostics_validation"].items()
    ]
    lines += [f"wrote {out / 'm1_oracle.json'}", "PASS" if report["passed"] else "FAIL"]
    sys.stdout.write("\n".join(lines) + "\n")
    return 0 if report["passed"] else EXIT_DATA


def _add_oracle_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    def add_source(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--cache-dir",
            type=Path,
            default=None,
            metavar="DIR",
            help="where M1 is downloaded and converted (default: ~/.cache/astro-simclr/m1/"
            "<commit>; never inside the repository)",
        )
        parser.add_argument(
            "--source-dir",
            type=Path,
            default=None,
            metavar="DIR",
            help="copy M1's files from DIR (model.index, model.data-00000-of-00001, "
            "val.events) instead of downloading them; they are checked the same way",
        )

    fetch = commands.add_parser(
        "fetch-m1",
        help="download M1 (pinned commit, SHA-256) and convert it to a local .npz",
        description="Download the published checkpoint M1 and its validation log from "
        "alercebroker/pipeline at the pinned commit, check each file's size and SHA-256, "
        "and convert the network's variables to CACHE/m1_weights.npz (TF names and "
        "layout). Without TensorFlow; nothing is unpickled. Never commit the files.",
    )
    add_source(fetch)
    fetch.set_defaults(handler=_fetch_m1)

    oracle = commands.add_parser(
        "m1-oracle",
        help="check our input path and paper model against the published M1 (no GPU)",
        description="Load M1 into PaperStampClassifier and run the two checks of task 2.2b "
        "on red Validation: port equivalence (torch vs an independent NumPy pass, max "
        "|delta logit| <= 1e-4, all argmax equal) and data path (475 ± 2 of 500 correct, "
        "loss -0.0616 ± 0.005). Also reports the NaN-approximated stamps, M1's test "
        "accuracy and the input diagnostics. Writes OUT/m1_oracle.json (top-level "
        "'passed', read by compare-reference --ref CD21 --oracle-report) and "
        "OUT/m1_val_curve.csv. Exit 0 if both checks passed, else 1.",
    )
    oracle.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="the prepared red layout, with images_f32.npy (prepare-data --images float32)",
    )
    oracle.add_argument("--out", type=Path, default=Path("outputs/m1-oracle"), metavar="DIR")
    add_source(oracle)
    oracle.set_defaults(handler=_m1_oracle)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser with every subcommand."""
    parser = argparse.ArgumentParser(
        prog="astro-simclr",
        description="SimCLR and supervised contrastive learning on ZTF stamps.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('astro-simclr')}"
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    show = commands.add_parser("show-config", help="print the resolved settings as YAML")
    add_settings_arguments(show)
    show.set_defaults(handler=_show_config)
    _add_data_commands(commands)
    _add_training_commands(commands)
    _add_check_commands(commands)
    _add_oracle_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI on ``argv`` (default: ``sys.argv[1:]``) and return the exit code.

    Without a subcommand it prints the help and returns 2. Invalid settings print one
    ``error:`` line on stderr and return 2; argparse usage errors exit with 2 as well.
    Unusable data (a ``DataError``) prints one ``error:`` line and returns 1.
    """
    from astro_simclr.data.schema import DataError

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    try:
        code: int = args.handler(args)
    except ConfigError as err:
        sys.stderr.write(f"astro-simclr: error: {err}\n")
        return EXIT_USAGE
    except DataError as err:
        sys.stderr.write(f"astro-simclr: error: {err}\n")
        return EXIT_DATA
    return code


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    sys.exit(main())
