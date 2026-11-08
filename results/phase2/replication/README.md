<!-- Archive of the phase-2 replication of Carrasco-Davis et al. 2021 (tasks 2.2b and 2.4 of
     docs/phases/02-stamp-classifier.md): the M1 oracle report and the 5-seed replication
     runs, with features (gating) and without features (advisory). Derived numbers only, no
     checkpoints. -->
# Phase-2 replication results (tasks 2.2b and 2.4)

This directory archives the replication of the ALeRCE stamp classifier (Carrasco-Davis et
al. 2021, AJ 162, 231) with the `carrasco2021` presets. The plan asks for "metrics JSON,
figures and the M1 curve. No checkpoints." (`docs/phases/02-stamp-classifier.md`, task 2.4).
The rules that judge these files are in `docs/evaluation-protocol.md`.

The per-stamp predictions are kept so that later variants (task 2.5) can be compared
against the replication stamp by stamp, as the protocol requires (McNemar and a paired
bootstrap).

## Files

| Path | Content |
|---|---|
| `m1_oracle.json` | The M1 oracle report (task 2.2b, PR #16). Its top-level `passed: true` feeds the faithful tier of the gate. |
| `m1_val_curve.csv` | M1's validation curve from its training log (task 2.2b). |
| `with-features/seed-K/` | One `carrasco2021` run (with features, gating), K = 0..4. |
| `no-features/seed-K/` | One `carrasco2021-no-features` run (advisory), K = 0..4. |
| `gate_with-features.txt` | `compare-reference --ref CD21` on the 5 with-features runs: `CD21: PASS (faithful)`. |
| `gate_no-features.txt` | The same on the 5 without-features runs: `CD21: REPORT` (advisory only). |
| `gate_combined.txt` | The same on all 10 runs, the form of the command in the plan: `CD21: PASS (faithful)`, with the advisory line. |

Each `seed-K/` directory holds the files `train-supervised` writes, minus `checkpoints/`
and `lightning_logs/`:

| File | Content |
|---|---|
| `metrics.json` | The run's metrics, config, machine and timing (keys in `docs/evaluation-protocol.md`, "What every run reports"). |
| `confusion_matrix_val.png`, `confusion_matrix_test.png` | Normalised confusion matrices of the checkpoint selected on validation. |
| `predictions_val.npz`, `predictions_test.npz` | Per-stamp `probabilities` (float32 softmax, 500 × 5, label order AGN, SN, VS, Asteroid, Bogus), `labels` and `index` (0..499). Read with `numpy.load(..., allow_pickle=False)`. |

The 50 run files total 1,568,295 bytes (about 1.5 MiB). They were copied unchanged from the
GPU machine; their SHA-256 matched the originals after the copy. The accuracy recomputed
from each predictions file equals the `evaluation.{val,test}.accuracy` of its
`metrics.json`.

`metrics.json` keeps the absolute paths of the GPU machine (`config.data_dir`,
`config.output_dir`, `data.path`, under `/home/rvidal/`) as provenance. They are not used
by `compare-reference`. The files hold no hostname.

## Provenance

Both lines ran on the same machine and build, one run after another (no other GPU job):

- **Machine:** NVIDIA GeForce GTX 1050 (4 GB), WSL2 (`Linux-6.18.40.1-microsoft-standard-WSL2`),
  Python 3.12.16, torch `2.14.1+cu126` (CUDA 12.6, cuDNN 91002), Lightning 2.6.6,
  `astro_simclr` 0.1.0.
- **Data:** `td_ztf_stamp_17_06_20_red-002.pkl` (SHA-256 `48f50ca5…beebb75`, in each
  `metrics.json`), prepared with `--images float32`. Train 50,581 stamps after the leakage
  exclusion (Train 2995), 72,710 after tiling; Validation and Test 500 each (100 per class).
- **With features:** commit **510395e** (the first commit of PR #15,
  `feat/preset-carrasco2021`), from a separate clone checked out at that commit. PR #15 was
  merged later as da3e3ee. Between 510395e and da3e3ee, `src/` differs only by the M1
  oracle (`src/astro_simclr/oracle/` and the `fetch-m1` / `m1-oracle` commands, PR #16); the
  training path, the preset, `pyproject.toml` and `uv.lock` are identical. Runs on
  2026-11-08, 05:41–07:48 CET (01:41–03:48 UTC−3), seeds 0..4 in order.
- **Without features:** main at **695ed93** (same `src/`, `pyproject.toml` and `uv.lock` as
  da3e3ee; 695ed93 only changes the plan). Runs on 2026-11-08, 07:51–09:57 CET
  (03:51–05:57 UTC−3).
- Every run trained the full 30,000 steps (`schedule: carrasco2021`, no early stopping) in
  about 25.1 minutes of fit time (0.050 s per step) and exited 0.

## Results

Test accuracy is of the checkpoint selected on validation loss (`test_acc_at_best`). Test
is archived and reported, never a decision input (protocol, "The rule").

**With features (`carrasco2021`, gating):**

| Seed | best_step | val loss (best) | val_acc_best | test (at best) | SN test recall |
|---|---|---|---|---|---|
| 0 | 18,930 | −0.0698 | 0.946 | 0.942 | 0.85 |
| 1 | 4,210 | −0.0607 | 0.940 | 0.946 | 0.89 |
| 2 | 4,540 | −0.0598 | 0.940 | 0.940 | 0.87 |
| 3 | 4,430 | −0.0593 | 0.944 | 0.938 | 0.88 |
| 4 | 6,330 | −0.0687 | 0.944 | 0.948 | 0.87 |
| **mean ± std (ddof=1)** | | | **0.9428 ± 0.0027** | **0.9428 ± 0.0041** | **0.872 ± 0.015** |

Paper: 0.941 ± 0.004 (ddof = 0). |Δ| = 0.0018, so **CD21: PASS (faithful)** (the M1 oracle
passed; SN recall and validation accuracy within their bands).

**Without features (`carrasco2021-no-features`, advisory):**

| Seed | best_step | val loss (best) | val_acc_best | test (at best) | SN test recall |
|---|---|---|---|---|---|
| 0 | 6,880 | 0.0221 | 0.882 | 0.870 | 0.82 |
| 1 | 2,100 | 0.0231 | 0.880 | 0.870 | 0.80 |
| 2 | 3,630 | 0.0230 | 0.896 | 0.872 | 0.83 |
| 3 | 3,400 | 0.0247 | 0.886 | 0.888 | 0.86 |
| 4 | 5,830 | 0.0199 | 0.888 | 0.862 | 0.73 |
| **mean ± std (ddof=1)** | | | **0.8864 ± 0.0062** | **0.8724 ± 0.0095** | **0.808 ± 0.049** |

Paper: 0.883 ± 0.006. |Δ| = 0.0106, just outside the "acceptable" width; the line is
advisory, so the gate prints **CD21: REPORT** (exit 0). The plan's 2.4 section discusses
the likely cause (the preset reuses M1's hyperparameters).

The validation loss is the preset's monitored loss (`monitor_best`): cross-entropy minus
β = 0.5 times the prediction entropy (`losses.py`), so it can be negative.

### Known WARN lines in the gate output

The gate files contain one `WARN: ... differs from the CD21 config: paper_arch=None
(expected 'rot4'), context=None (expected 'none'), rot90_augment=None (expected False)` per
run. These three settings were added to the preset later, by task 2.5-models (PR #18), so
the archived `metrics.json` files do not have them. Their defaults are exactly the expected
values (`paper_arch: rot4`, `context: none`, `rot90_augment: false` in `config.Settings`),
so the runs match the current preset. The WARN does not change the verdict.

## Reproducing

The runs (on the GPU machine; `--data-dir` is the parent of the prepared directory):

```bash
# once: the prepared data with the float32 stamps the paper's input path needs
uv run --no-sync astro-simclr prepare-data dataset/td_ztf_stamp_17_06_20_red-002.pkl \
    --images float32 --out dataset/prepared

# with features, at commit 510395e (or da3e3ee: same training code)
for s in 0 1 2 3 4; do
  uv run --no-sync astro-simclr train-supervised --preset carrasco2021 --accelerator cuda \
      --data-dir dataset/prepared --seed $s --output-dir runs/replication/with-features/seed-$s
done

# without features, at commit 695ed93
for s in 0 1 2 3 4; do
  uv run --no-sync astro-simclr train-supervised --preset carrasco2021-no-features \
      --accelerator cuda --data-dir dataset/prepared --seed $s \
      --output-dir runs/replication/no-features/seed-$s
done
```

The original runs called the virtual environment's `astro-simclr` directly
(`.venv/bin/astro-simclr`), with the same arguments. Training is not bitwise deterministic
(`deterministic: false`), so a rerun on the same machine gives close but not identical
numbers.

The gate, from the repository root on any machine (CPU is enough; it only reads the
JSON files):

```bash
R=results/phase2/replication
uv run astro-simclr compare-reference $R/with-features/seed-{0..4} \
    --ref CD21 --oracle-report $R/m1_oracle.json > $R/gate_with-features.txt
uv run astro-simclr compare-reference $R/no-features/seed-{0..4} \
    --ref CD21 --oracle-report $R/m1_oracle.json > $R/gate_no-features.txt
uv run astro-simclr compare-reference $R/with-features/seed-{0..4} $R/no-features/seed-{0..4} \
    --ref CD21 --oracle-report $R/m1_oracle.json > $R/gate_combined.txt
```

All three exit 0. `compare-reference` has no `--json` option; the text files are its
complete output. They were produced on 2026-11-08 on a Mac (CPU) at main 7058d4b.

The M1 oracle files come from `astro-simclr m1-oracle` (see `docs/architecture.md`, the
`oracle` section).
