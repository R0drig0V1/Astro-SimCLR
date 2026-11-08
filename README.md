# Astro-SimCLR

A typed, tested research codebase for classifying ZTF alert stamps (science, template and
difference cutouts plus alert features into AGN, SN, VS, Asteroid or Bogus), built on PyTorch.

- **Replication of the ALeRCE stamp classifier** (Carrasco-Davis et al. 2021, "CD21"):
  5 seeds of the `carrasco2021` preset reach **test 0.9428 ± 0.0041** against the paper's
  **0.941 ± 0.004**, so the gate reads **`CD21: PASS (faithful)`**. An oracle loads the
  published M1 checkpoint into our model and checks the input path and the port (475/500
  Validation correct, as expected; max |Δlogit| 1.6e-6). Archive and provenance:
  [`results/phase2/replication/`](results/phase2/replication/README.md).
- **Architecture and augmentation research** on top of the replication, decided under a
  pre-registered [evaluation protocol](docs/evaluation-protocol.md): single-pass p4 / p4m
  group-equivariant CNNs, the 63″ context channels, class-aware augmentation (noise,
  synthetic hot-pixel Bogus) and transient flux rescaling.
- **Raw-alert infrastructure and benchmarks** (phase 3): verified retrieval of the raw
  alerts behind the labelled stamps (`fetch-alerts`), RJ23 input functions, macro-F1
  benchmark metrics, object-grouped splits, and an object-level 5-class benchmark (R5)
  being built from public sources only.
- **Self-supervised pretraining** (SimCLR, Chen et al. 2020; supervised contrastive, Khosla
  et al. 2020) is the next line, on a large unlabelled corpus of raw ZTF alerts.

History: the repository began as a 2021 course project on SimCLR for ZTF stamps.
That code was retired from the tree and stays readable at the git tag `legacy-final`.

## Status

| Phase | Plan | State |
|---|---|---|
| 1. Runnable, tested package | [01-runnable-package.md](docs/phases/01-runnable-package.md) | done; cloud GPU runbook (1.9a) left |
| 2. Replicate and improve the stamp classifier | [02-stamp-classifier.md](docs/phases/02-stamp-classifier.md) | replication passed; improvements (2.5) running |
| 3. Raw-alert infrastructure and the R5 benchmark | [03-newer-benchmarks.md](docs/phases/03-newer-benchmarks.md) | in progress |
| 4. Training platform (plain PyTorch, local run registry) | [04-training-platform.md](docs/phases/04-training-platform.md) | reviewed; first tasks dispatchable |
| Next: self-supervised pretraining | [roadmap](docs/phases/README.md) | not started |

The [roadmap](docs/phases/README.md) is the authority on what is done and what runs now.

## Install

Requires [uv](https://docs.astral.sh/uv/); it installs Python 3.13 and the exact versions of
`uv.lock`.

```bash
uv sync --locked                 # PyPI torch: MPS on macOS, CUDA 13 on Linux (driver >= 580, Turing or newer)
uv sync --locked --extra cpu     # no GPU (what CI uses)
uv sync --locked --extra cu126   # GPU older than Turing (e.g. GTX 1050), or driver < 580
uv run astro-simclr --help       # the command line
```

After installing an extra, skip uv's automatic re-sync, which would put the default build
back: set `UV_NO_SYNC=1` or use `uv run --no-sync`. The driver and compute-capability table
(Blackwell needs the default build) is in
[ADR 0001](docs/decisions/0001-torch-wheels-and-accelerators.md); on Windows use WSL2.

On a GPU machine, check the build and run a dry training before real work:

```bash
scripts/gpu-preflight.sh --synthetic        # CUDA and the torch build, no real data needed
scripts/gpu-preflight.sh --data-dir DIR     # also the prepared data and its manifest
```

The preflight names the right `uv sync` command when the installed build does not fit the GPU.

## Quickstart

### 1. Data

The ZTF stamp datasets (the CD21 pickles) are **not public** and are not in this
repository. [docs/data.md](docs/data.md) has the files, their SHA-256, schemas and
provenance. Training never reads the pickles: convert each one once.

```bash
uv run astro-simclr inspect-data dataset/td_ztf_stamp_17_06_20_red-002.pkl   # schema, counts, SHA-256
uv run astro-simclr prepare-data dataset/td_ztf_stamp_17_06_20_red-002.pkl \
    --images float32 --out dataset/prepared   # float32 stamps: the paper's input path needs them
```

Without the data, `astro-simclr make-synthetic --out DIR --n-per-split N --seed S` writes
synthetic stand-ins with the same layout.

### 2. Reproduce the replication

```bash
# The M1 oracle (no GPU): downloads M1 pinned by commit and SHA-256, outside the repository
uv run astro-simclr m1-oracle --data-dir dataset/prepared/td_ztf_stamp_17_06_20_red-002 --out OUT

# 5 seeds of the paper preset (about 25 min each on a GTX 1050)
for s in 0 1 2 3 4; do
  uv run --no-sync astro-simclr train-supervised --preset carrasco2021 --accelerator cuda \
      --data-dir dataset/prepared --seed $s --output-dir runs/replication/with-features/seed-$s
done

# The gate (CPU is enough; reads the metrics.json files). Exit 0 for PASS or REPORT, 1 for FAIL
uv run astro-simclr compare-reference runs/replication/with-features/seed-{0..4} \
    --ref CD21 --oracle-report OUT/m1_oracle.json
```

The archived runs reproduce the published verdict without a GPU:

```bash
R=results/phase2/replication
uv run astro-simclr compare-reference $R/with-features/seed-{0..4} \
    --ref CD21 --oracle-report $R/m1_oracle.json
```

`--preset carrasco2021-no-features` is the advisory without-features line (test
0.8724 ± 0.0095 against the paper's 0.883 ± 0.006; `CD21: REPORT`). Training is not bitwise
deterministic, so a rerun gives close but not identical numbers.

### 3. Compare a variant on the stored splits

Decisions use split runs, which never load Test. The stored, object-grouped splits are in
`configs/splits/`; one arm per flag, same seeds and machine on both sides:

```bash
SPLITS=configs/splits/td_ztf_stamp_17_06_20_red-002-grouped.json
TRAIN="uv run --no-sync astro-simclr train-supervised --preset carrasco2021 --data-dir dataset/prepared --splits $SPLITS"

for k in 0 1 2 3 4; do            # 5-fold CV x 2 seeds per arm
  for s in 0 1; do
    $TRAIN --fold $k --seed $s --output-dir runs/cv/base/fold$k-seed$s
    $TRAIN --fold $k --seed $s --output-dir runs/cv/p4/fold$k-seed$s --set paper_arch=p4
  done
done
uv run astro-simclr compare-variants --splits $SPLITS \
    --baseline runs/cv/base/fold* --variant runs/cv/p4/fold* \
    --rule non-inferiority        # efficiency rows; the default rule is superiority
```

`--screen` instead of `--fold K` runs the cheap 250/250 screen, which can only drop an arm.
Both are described in [docs/evaluation-protocol.md](docs/evaluation-protocol.md)
("Running a decision end to end").

The same split runs answer the feature diagnostic (ADR 0016): `--set feature_set=position`
keeps only the 6 sky coordinates of the 23 features, `--set feature_set=none` the images
only (the same model as `with_features: false`).

### 4. Raw alerts

`fetch-alerts` (task 3.1) recovers the raw alerts behind a prepared split from ALeRCE's
public services. It checks each one against the stored stamp and keeps the raw cutouts with
their real NaN masks, the candidate record and every raw response. Runs are resumable, and
the store can be rebuilt offline with `--from-snapshot`.

```bash
uv run astro-simclr fetch-alerts --data-dir dataset/prepared --dataset td_ztf_stamp_17_06_20_red-002 \
    --split Train --out dataset/raw/td_ztf_stamp_17_06_20_red-002 --rate 1.5 --quiet
uv run astro-simclr fetch-alerts ... --resume --quiet        # after any stop
uv run astro-simclr alerts-summary dataset/raw/td_ztf_stamp_17_06_20_red-002
```

The method, the timings and the store layout are in [docs/data.md](docs/data.md) ("Raw
alerts") and [ADR 0011](docs/decisions/0011-raw-alert-store.md). The other phase-3 tools are
`make-object-splits`, `bench-report` and `leakage-audit`.

R5 (task 3.5) is built from the Zenodo labelled set, the Bright Transient Survey and the
CD21 store above. One seeded alert is drawn per object before 2023. The build is resumable,
and it rebuilds offline byte for byte:

```bash
uv run astro-simclr r5-sources --out dataset/r5/sources
uv run astro-simclr build-r5 --sources dataset/r5/sources \
    --cd21-raw dataset/raw/td_ztf_stamp_17_06_20_red-002 --fetched dataset/r5/fetched \
    --out dataset/r5/R5 --quiet                 # add --dry-run first for the request count
uv run astro-simclr build-r5 ... --from-snapshot --out /tmp/R5 --expect dataset/r5/R5/build.json
```

The recipe and the outputs (deviation and era tables, frozen index) are in
[docs/data.md](docs/data.md) ("R5") and [ADR 0020](docs/decisions/0020-r5-reconstruction.md).

## Repository map

```
src/astro_simclr/   the package: data, models (paper model, p4/p4m variants), losses,
                    augmentations, training, protocol (splits, gates, comparisons), oracle,
                    alerts (raw-alert retrieval), r5 (the R5 build), CLI
tests/              pytest suite (branch coverage >= 80 %)
configs/reference/  R1-R4 legacy reference-run configs, with provenance
configs/splits/     stored phase-2 decision splits (make-splits)
results/            phase2/ (replication archive, augmentation inspections); 2021 tuning results
docs/               architecture.md (one section per module), data.md, evaluation-protocol.md,
                    phases/ (the plan), decisions/ (ADRs), plan-reviews/
scripts/            check.sh, gpu-preflight.sh, check-torch-wheels.sh, inspection scripts,
                    characterize/ (legacy fixture generators)
jupyter_examples/   2021 exploratory notebooks (they import the legacy code)
```

Module responsibilities: [docs/architecture.md](docs/architecture.md). Design decisions:
[docs/decisions/](docs/decisions/README.md). The 2021 code (root scripts, `utils/`,
`models/`, `losses/`, `env.yml`) is at the tag `legacy-final`:

```bash
git fetch origin tag legacy-final        # once, if the tag is not local
git show legacy-final:utils/training.py  # read one legacy file
git worktree add ../Astro-SimCLR-legacy legacy-final   # the whole tree (to run the notebooks)
```

## Evaluation rules

Test metrics are never a decision input: they are computed, archived and reported once per
pre-registered comparison. Every keep/drop decision comes from validation stamps that did
not select the checkpoint, from the M1 oracle, or from cross-validation on Train +
Validation, using the stored split files in `configs/splits/` (their SHA-256 is checked).
The decision splits are grouped by object, because some Validation and Test alerts share a position
with a Train alert (24 and 27 of 500). Comparisons pair predictions stamp by stamp
(McNemar, paired bootstrap) and never mix machines or builds. The full rules, the
replication gate and the power note: [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

## Citing and licences

This repository is under the [MIT licence](LICENSE). If you use it, cite the work it
replicates and builds on:

- Carrasco-Davis, R. et al. 2021, "Alert Classification for the ALeRCE Broker System: The
  Real-time Stamp Classifier", AJ 162, 231 (arXiv:2008.03309).
- Reyes-Jainaga, I. et al. 2023, ApJL (arXiv:2304.13080): the RJ23 protocol and inputs.
- Cohen, T. and Welling, M. 2016, "Group Equivariant Convolutional Networks", ICML
  (arXiv:1602.07576): the p4 / p4m layers.
- Chen, T. et al. 2020, "A Simple Framework for Contrastive Learning of Visual
  Representations", ICML (arXiv:2002.05709): SimCLR.
- Khosla, P. et al. 2020, "Supervised Contrastive Learning", NeurIPS (arXiv:2004.11362).

**No ALeRCE code or weights are included.** The model and input path are reimplemented.
`fetch-m1` downloads the M1 checkpoint at use time, pinned by commit and SHA-256, into a
cache outside the repository, and nothing downloaded is unpickled. The ZTF stamp datasets
are not redistributed.

## Development

```bash
scripts/check.sh                 # ruff, ruff format, mypy strict, pytest + coverage: "green"
UV_NO_SYNC=1 scripts/check.sh    # the same with a torch extra installed
uv run ruff format . && uv run ruff check --fix .
scripts/check-torch-wheels.sh    # after touching the torch pins, extras or indexes
```

`scripts/check.sh` is the single definition of green; CI runs exactly that script. It runs
the tests in parallel, one process per CPU (`PYTEST_XDIST_AUTO_NUM_WORKERS=N` caps it). GitHub
workflows also run zizmor, actionlint, gitleaks, pip-audit, CodeQL and Scorecard; security
reports go through [SECURITY.md](SECURITY.md). The scaffolding comes from
[python-template](https://github.com/R0drig0V1/python-template) (`uvx copier@9.19.1 update`
on a clean tree).

Work is planned in [docs/phases/](docs/phases/README.md) and done by several Claude agents in
parallel, each in its own git worktree, with one integrator merging verified pull requests
into `main` ([CLAUDE.md](CLAUDE.md), "Multi-agent workflow"). To contribute: branch from
`main`, put new code in `src/astro_simclr/` with tests and docstrings, update the matching
`docs/` page in the same change, add an ADR for decisions that are hard to reverse, keep
`scripts/check.sh` green, and open a pull request against `main` using Conventional Commits.
