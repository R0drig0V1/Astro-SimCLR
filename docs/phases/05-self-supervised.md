<!-- Phase 5 of the roadmap (docs/phases/README.md); owned by the integrator. Reviewed (round 1,
docs/plan-reviews/2026-11-08-phase-5.md); amendments applied. H1 decided under delegation (2026-11-08). -->
# Phase 5: self-supervised pretraining (SimCLR, ResNet first)

## Goal

The user's goal (2026-11-08): self-supervised pretraining with a standard ResNet encoder
first, then something custom, on the plain-PyTorch platform of phase 4.

**Pre-registered question** (task 5.0). H1 was decided by the integrator on 2026-11-08, under the user's explicit delegation ("take the best decision possible"). The user may still revise it before any pretraining run. Does
SimCLR pretraining on unlabelled ZTF alert stamps improve classification with few labels and
without the alert features, against the **strongest** baseline?

Distribution shift (later ZTF years, Rubin) is **not** a phase-5 claim: no labelled set
after 2022 exists. It is a later phase, possibly with an "R5-late" set.

Prior work: Rehemtulla et al. (arXiv:2512.11957) ran label fractions on ZTF triplets. Galaxy
Zoo (Zoobot) pretraining beat ImageNet and training from scratch, and their custom CNN won
below about 100k training alerts. A dated literature search replaces any novelty claim
(task 5.0).

## Tasks

### 5.0 Pre-registration (an ADR, merged before any pretraining run)

**H1 (primary), decided (delegated, 2026-11-08):**
- **Benchmark and metric:** R5 test, with the reweighted macro-F1. If R5 slips, the CD21
  test balanced accuracy.
- **Setting:** images only, the lowest label budget, fine-tuned.
- **Claim:** SSL-ResNet-18 beats the strongest baseline of 5.4.
- **Test:** the lower bound of the paired 95 % CI (a hierarchical bootstrap over label
  subsets, then objects) is above **+2 points**.

**H2 (secondary):** the middle budget, with a Holm
correction.

**Everything else is exploratory** and labelled as such.

**The null outcome** is stated in advance and reported. `docs/results-phase5.md` holds the
pre-registered tables whatever the result is. A dated literature search is recorded.

### 5.0b Pilot: now, with no download and no new engine (gates the download request)
- **Engine-free pure functions,** following phase 4's B0 rule:
  - the views of 5.2;
  - a ResNet-18 with a CIFAR stem (3×3 stride 1, no max-pool), `fc = Identity`, 512-d `h`;
  - a Linear–BN–ReLU–Linear head to 128;
  - NT-Xent in paper mode (`nt_xent_legacy_denominator: false`), in fp32;
  - LARS with the trust-ratio fix and BN/bias excluded, weight decay 1e-6;
  - lr 0.3·B/256, with a per-iteration warmup and a cosine over exactly the run's length;
  - a probe on precomputed, frozen `h`.
- **Pilot pretraining** on CD21 Train images (50.6k stamps), on the 1050 at a small batch or
  on the 5090.
- **Evaluation:** the CD21 **screen halves only**. CV is out, because its pool's Train
  stamps were pretrained on.
- **Go/no-go:** the SSL probe beats the random-init probe and the scratch supervised model
  at the lowest budget by a pre-registered margin. Only then is the user asked to approve
  the corpus download.
- **Margin, declared 2026-11-09 before any baseline result** (user decision: compare with a
  ResNet trained from scratch, the same architecture): the lower bound of the paired 95 %
  CI (hierarchical bootstrap: subsets, then objects) of SSL probe − scratch above **+2
  points** of balanced accuracy at 10 per class; 100 per class reported, no verdict.
- **Result (2026-11-09): GO.**
  - Pretraining: `simclr-pilot` on the screened CD21 Train stamps, ResNet-18 (CIFAR stem),
    batch 256, LARS lr 0.3·B/256, T 0.2, C5 noise 3, 100 epochs (19,700 steps, warmup
    10), seed 0, RTX 5090 (main 827c41a; 1.6 h shared with a CV). Monitors: the abort check
    at epoch 20 passed (probe at 100/class 0.58 against 0.46 random init); final kNN on the
    selection half 0.704, online probe 0.76.
  - Evaluation: the screen's decision half (250 stamps, 250 objects), 20 label subsets per
    budget (`make-label-subsets --seed 0`), 4 trials per subset for every arm
    (`simclr-probe` L2 grid; `simclr-scratch` lr × weight decay grid, PR #66), chosen on
    the selection half.

  | Arm, 10 per class | Balanced accuracy (mean ± sd over subsets) |
  |---|---|
  | SSL probe | **0.571 ± 0.052** |
  | Scratch ResNet-18 (supervised, same views) | 0.434 ± 0.050 |
  | Random-init probe | 0.417 ± 0.044 |

  | Paired comparison, 10 per class | Δ (95 % CI) | Verdict |
  |---|---|---|
  | SSL probe − scratch | **+13.7 [+10.2, +17.2] pts** | **go** (margin +2) |
  | SSL probe − random-init probe | +15.4 [+11.7, +19.2] pts | (secondary) |

  - 100 per class (secondary, no verdict): SSL probe 0.734 ± 0.021, scratch 0.680 ± 0.047,
    random-init probe 0.515 ± 0.015; SSL − scratch **+5.4 [+1.9, +9.0] pts**, SSL −
    random-init +21.9 [+17.7, +26.2]. The SSL advantage shrinks as labels grow (+13.7 at
    10 per class), as expected. (The scratch run at 100 per class was interrupted by a WSL
    restart after 13 subsets and resumed with `--resume`.)
  - **Corpus pretraining, launched 2026-11-09 (main df55f5a, PR #67):** the same recipe on
    the corpus index of the 8 archive nights (311,966 rows; 311,716 kept: 224 within
    1.5″ of CD21, 26 repeated `oid`s, 0 near-duplicates), with the `build-corpus-fits`
    sidecar for the noise view; 50 passes fixed (about 61k steps, 3× the pilot's), probe
    every 2 passes, abort at pass 10. Compared with the CD21-only pilot by the same probe
    protocol when it ends; the monitors give the curve meanwhile (diagnostics, never a
    selector). Batch size stays 256 for this comparison; batch {256, 1024, 2048} goes to
    the fine grid (user, 2026-11-09).
  - **Corpus result (2026-11-09): no gain over the CD21-only pilot.** The run ended at
    19:35 (60,850 steps, 50 passes, 3.6 h on the 5090; the abort check passed). Final
    monitors: kNN on the selection half 0.716 (pilot 0.704), online probe 0.772 (0.76),
    effective rank of `h` 53.7 (41.5). The same probe protocol as the pilot (the same 20
    subsets per budget, the decision half):

  | Paired comparison (corpus SSL probe − baseline) | 10 per class | 100 per class |
  |---|---|---|
  | Corpus SSL probe (mean ± sd) | 0.568 ± 0.044 | 0.721 ± 0.025 |
  | − CD21-only pilot SSL probe (0.571 / 0.734) | −0.3 [−2.8, +2.1] pts | −1.3 [−3.7, +1.1] pts |
  | − scratch ResNet-18 (0.434 / 0.680) | +13.4 [+10.0, +16.8] pts | +4.1 [+0.5, +7.6] pts |
  | − random-init probe (0.417 / 0.515) | +15.1 [+11.2, +19.0] pts | +20.6 [+16.2, +25.1] pts |

    - Reading: 6× more unlabelled stamps, never including CD21 Train, give the same probe
      as pretraining on CD21 Train itself, which the pilot also evaluates on (the
      transductive advantage of the pilot is not measurable at this size). Both still beat
      the scratch model by the declared margin at 10 per class. The decision half has 250
      stamps: differences under about ±2.5 points are not resolved.
    - Not separated by this run: the number of steps (61k vs 19.7k), the passes over each
      stamp (50 vs 100) and the corpus's class mix. Data volume is not the bottleneck of
      this recipe; the batch-size grid {256, 1024, 2048} and the views are the next levers
      (exploratory, 5.0c rules). Artifacts: the server's run directory `p5/corpus/`
      (`probe-corpus-{10,100}.json`, `corpus-vs-{pilot,scratch,random}-{10,100}.json`).
  - **Batch 1024 (2026-11-09, main 6d25d15; user decision: batch 1024 first, "more is
    better"): no gain over batch 256.** The pilot recipe on CD21 Train with only the batch
    changed (lr linear in the batch, 0.3·1024/256 = 1.2; 100 passes, 4,900 steps, seed 0).
    It needs `--activation-checkpointing` (PR #68): without it the run spilled out of the
    32 GB of VRAM and was killed; with it the CUDA peak is 21.6 GiB allocated in a dry run
    and about 28 GiB reserved with the monitors, 0.93 s/step, 76 min. Final monitors: kNN
    0.712 (batch 256: 0.704), online probe 0.74 (0.76), effective rank of `h` 55.6 (41.5).

  | Paired comparison (batch-1024 SSL probe − baseline) | 10 per class | 100 per class |
  |---|---|---|
  | Batch-1024 SSL probe (mean ± sd) | 0.574 ± 0.055 | 0.735 ± 0.022 |
  | − batch-256 pilot SSL probe (0.571 / 0.734) | +0.3 [−1.6, +2.1] pts | +0.1 [−1.8, +2.0] pts |
  | − scratch ResNet-18 (0.434 / 0.680) | +13.9 [+10.4, +17.6] pts | +5.4 [+1.8, +9.1] pts |
  | − random-init probe (0.417 / 0.515) | +15.7 [+11.7, +19.8] pts | +22.0 [+17.7, +26.4] pts |

    - Reading: at a fixed number of passes, 4× the negatives per step and 4× fewer steps
      give the same probe. The batch effect of Chen et al. (2020) is largest for short
      runs on ImageNet-scale data with 1,000 classes; with 5 classes, many negatives of a
      large batch share the anchor's class, which may cap the gain. Batch 2048 needs bf16
      (about 32 GiB of activations with checkpointing; plan 5.3's precision decision).
      Neither more data (the corpus) nor a larger batch moved the probe: the remaining
      levers are the views, the encoder and the evaluation itself (fine-tuning, H1).
      Artifacts: the server's run directory `p5/b1024/` (`probe-b1024-{10,100}.json`,
      `b1024-vs-{pilot,scratch,random}-{10,100}.json`).
  - Caveats: the scratch model barely beats the random-init probe at 10 per class, so it
    is a weak bar; it is the declared one for this gate only. H1 is decided against the
    strongest baseline of 5.4 (paper model, ImageNet and Zoobot ConvNeXt), fine-tuned, on
    a larger evaluation set. The decision half has 250 stamps: the CI covers it.
  - Next, per the plan: the user is asked to approve the corpus (5.1); 14 GB of archive
    nights are already on the 5090 under the user's 20 GB cap.

### 5.0c Hyperparameter and selection policy
- **Search:** a declared grid, for example T ∈ {0.2, 0.5} × 2 lrs.
- **Selection metric:** a kNN or probe metric on **selection-only** data. Never the decision
  halves, CV evaluation parts or test.
- **Checkpoints:** a fixed epoch count, and the final checkpoint is kept. No selection on
  the loss.
- **Bookkeeping:** the same trial budget for every arm, with trial counts recorded.
- **Guard:** phase 4.5's structural sweep guard is implemented here.

### 5.1 The pretraining corpus (after 5.0b's go)

**5.1a `ingest-archive` spec.** Measured on 2026-11-08: 3,087 nightly files, 758 of them
empty; about 6.5–9.8 GB per night (8.0 GB in 2020); about 20 TB in total; about 175k alerts
per night; whole-night tarballs only; about 30 min of download per night.

**Pilot first:** on 3 nights (2018, 2020 and 2024), measure the yield, the composition and
the download time on the GPU machine.

**The tool:**
- explicit cuts, checked so that no target class is removed (no `rb`/`drb` threshold that
  drops Bogus);
- "first detection" means the object's first alert by `oid`, with the field and readout
  caveat of `ndethist` documented;
- streaming, filtering and discarding: no tarball stored;
- resumable per night, with the SHA-256 recorded;
- schema fixtures per year;
- untrusted tarballs: no path extraction, member-size caps, the bomb checks of
  `alerts/parse.py`.

**The composition report** is by era, using pseudo-labels from **our** replicated
classifier, never ALeRCE's outputs.

**A disk and transfer budget is stated before the user is asked.**

**Corpus:**
- 2018–2022 nights only for H1; 2018–2026 is a single ablation.
- One alert per object.
- **Exclusion of every CD21 object** (Train included, since CD21 CV evaluates them) and of
  R5 Validation/Test. It is the union of `candid`, `oid` and the 1.5″ position groups
  (ghosts included), applied when the corpus index is built.
- A near-duplicate check against every evaluation set.
- The exclusion-list SHA is recorded in every pretraining manifest.

The `simclr_300` fallback is dropped: the file is missing, and probably transductive.

### 5.2 Views (input pinned by the ADR)
- **Input:** 63 px, per-channel min-max with NaN → 0, plus an optional mask channel.
- **Dihedral views:** all 8.
- **Shift:** a random **57-px crop** of the 63-px stamp, integer and identical across
  channels, with no fill. Evaluation uses the centred 57-px crop.
- **Noise:** C5-style, on the stamp before normalisation, with the difference recomputed.
- **Scale jitter and blur are ablations only;** they spread 1-px spikes into PSF-like blobs.
- **Never:** hue, saturation, greyscale, or source-dropping crops.
- **Admission test:** a view whose single-view accuracy drop on **the screen's selection
  half** of Validation (replication checkpoint) exceeds 1 point is not admitted. Amended
  2026-11-10 (PR #70 review): in CD21 Validation is exactly the screen (250 selection +
  250 decision stamps), and admission decides which arms exist, so it is selection (5.0c:
  never the decision half). One stamp is 0.4 points; `--repeats` averages the view's draws.
- **Done 2026-11-10 (PR #70, main 028a5e6):** opt-in `--scale-jitter`, `--blur-sigma`
  (both ablations only), `--intensity-jitter` (one gamma on the three channels) and
  `--rotation-degrees` (at most 6.524°, the shift giving way to the rotation, no fill) on
  `simclr-pilot`, streams 14–17, and `simclr-view-admission`. Caveat for every rotation or
  zoom result: the view is a bilinear resampling, which also smooths the stamp and weakens
  the C5 noise (ADR 0025, "the interpolation confound"); the eval view is an exact crop.
  Pending: the admission runs (they need a replication checkpoint with weights: the 2.4
  runs were archived without them) and one pilot per admitted view.

### 5.3 SimCLR on the engine
The 5.0b functions run on phase-4 tiers 1 and 3 with exact resume. Only the 1M corpus needs
tier 3. Phase 5 does not wait for 4.6–4.8.

- **Precision:** TF32 or bf16 for pretraining, decided and recorded. There is no legacy
  reference, so ADR 0010's argument does not apply.
- **Retention** (`kind: pretrain`): the final checkpoint plus epochs {25, 50, 100, …}, and
  cached embeddings.
- **Manifest:** the corpus SHA, the exclusion SHA and the loss mode.
- **Diagnostics every N epochs:**
  - the effective rank of cov(h) and the per-dimension std of z;
  - alignment and uniformity;
  - a kNN monitor on the selection split;
  - nuisance probes (fid, year, field) from h.
- **Abort rule:** stop on rank collapse, or if the kNN does not beat random init by
  epoch 20.

### 5.4 Evaluation
- **Budgets:** labelled **objects per class**, 10 / 100 / 1,000, plus 100 %.
  - The subsets are stored files, generated per fold from that fold's training part only.
  - At the low budgets: a fixed schedule, or a selection set of at most 20 % of the budget,
    the same for every arm.
- **Repeats:**
  - probe on frozen `h`: ≥ 20 subsets;
  - fine-tuning: ≥ 10 subsets × 1 seed at the lowest budget, 5 at the middle one, 1 subset ×
    5 seeds at 100 %.
  - Arms are paired by subset.
  - The CI is the hierarchical bootstrap (subsets, then objects), a new function in
    `protocol.stats` with a golden test.
- **Feature settings:** images only (no alert features) for every pre-registered cell and both
  pretraining monitors (the probe input is h of the eval view: centre crop, no augmentation).
  **H3 removed (user decision, 2026-11-08, before any result):** the "all 23 features" setting
  is not pre-registered. Once the linear-probe evaluation is done and an encoder is chosen, a
  separate, newly declared test adds the features (BN-then-concat, as CD21).
- **Baselines:**
  - the random-init probe;
  - a scratch ResNet-18 trained with the same views;
  - **SSL on the labelled pool's images only** (data versus objective);
  - the paper model, and RJ23 multi-scale on R5;
  - ImageNet ConvNeXt-pico and **Zoobot ConvNeXt-pico**, fine-tuned, with Rehemtulla's
    input handling. timm is added via an ADR, with the weights' licences checked.
- **Fairness:** the same test-time augmentation policy and the same trial budget for every
  arm.
- **Mode:** the probe for exploratory cells; fine-tuning only for H1 and H2.
  - **Exploratory fine-tuning arm, done 2026-11-10 (PR #69, main 8b8a7df; user decision:
    pursue fine-tuning and the views):** `simclr-finetune` is `simclr-scratch` with the
    encoder from a `simclr-pilot` checkpoint: the same recipe, grid (4 trials), step
    counts, selection rule and draws (stream 13 reused on purpose: common random numbers),
    so fine-tune − scratch isolates the initialisation. It runs on the screen halves; H1's
    test set is not touched until the recipe is frozen. Pending: the runs from the pilot,
    corpus and batch-1024 checkpoints at 10 and 100 per class (the 5090 was unreachable on
    2026-11-10 morning).
- **Before relying on R5,** compute its per-class test counts and MDEs. Its Bogus and
  Asteroid come from CD21 Validation/Test only (phase 3 to confirm).

### 5.5 Later, gated by H1 (a separate phase)
- Custom encoders: p4/p4m and multi-scale; a **ViT-Ti with patch 7** (a 9×9 grid at 63 px) pretrained on 1–5M stamps (SimCLR with MoCo-v3 stabilisation for a like-for-like comparison; DINO or iBOT as ViT-native alternatives; MAE only with centre-aware masking). This is a 5090 job.
- SupCon, as a 100 %-label arm.
- Ablations: debiased contrastive loss, and multi-epoch positives (two alerts of one object
  as a pair).

Cut: BYOL and VICReg, the "position only" setting, and the shift question.

## Compute (to be completed in 5.0)

A budget table per task with a hard cap and a stop rule. Estimates:
- ResNet-18 with the CIFAR stem at 63 px costs about 26 GFLOP per stamp per step (2 views,
  forward and backward). 1M stamps × 200 epochs is about 35–50 h on the 5090 in fp32.
- The GTX 1050 fits about 32 pairs per batch, so it is for smoke runs only.
- Ablations run at 300k stamps and 100 epochs.
- The downstream cells number about 1,000 fine-tunes; probes are cheap.

## Risks
- **Leakage:** the exclusion union plus the near-duplicate check (5.1).
- **Same-class negatives** in a Bogus/Asteroid-heavy corpus: report the collision rate;
  debiased-loss ablation.
- **A null result** (Rehemtulla): pre-registered and reported either way.
- **Compute contention** with 2.5 and 3.6.
