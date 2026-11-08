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

**H2 and H3 (secondary):** the middle budget, and the "all 23 features" setting, with a Holm
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
- **Admission test:** a view whose single-view accuracy drop on Validation (replication
  checkpoint) exceeds 1 point is not admitted.

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
- **Feature settings:** none, and all 23 (BN-then-concat, as CD21).
- **Baselines:**
  - the random-init probe;
  - a scratch ResNet-18 trained with the same views;
  - **SSL on the labelled pool's images only** (data versus objective);
  - the paper model, and RJ23 multi-scale on R5;
  - ImageNet ConvNeXt-pico and **Zoobot ConvNeXt-pico**, fine-tuned, with Rehemtulla's
    input handling. timm is added via an ADR, with the weights' licences checked.
- **Fairness:** the same test-time augmentation policy and the same trial budget for every
  arm.
- **Mode:** the probe for exploratory cells; fine-tuning only for H1–H3.
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
