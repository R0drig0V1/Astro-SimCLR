<!-- Phase 5 of the roadmap (docs/phases/README.md); owned by the integrator. DRAFT: to be
reviewed with the improve-plan procedure before dispatch. -->
# Phase 5: self-supervised pretraining (SimCLR, ResNet first) — draft

## Goal

The user's goal (2026-11-08) is to start self-supervised pretraining with a standard
ResNet encoder, then move to something custom. It is built on the plain-PyTorch platform
of phase 4, with no Lightning. The question it answers:

**Does contrastive pretraining on unlabelled ZTF alert stamps improve classification:**
- with few labels (1 % and 10 %);
- without the 23 alert features (images only, or images plus position);
- under distribution shift (later ZTF years; Rubin later)?

The comparison is against supervised baselines trained on the same label fractions under
the protocol of task 2.3. No published SSL result on ZTF or Rubin alert triplets was found
(phase-1 research), so a careful study would be close to novel.

## What exists, and what is wrong with it (SimCLR review, 2026-11-08)

**Correct:**
- the NT-Xent loss in paper mode, checked against an independent reference (2.798286);
- the pair batching;
- the representation taken before the head;
- the frozen encoder kept in `eval()`.

**Wrong:**
- A1: the cosine schedule ignores `max_epochs` (fixed at 350).
- A2: LARS freezes zero-initialised parameters (the legacy trust ratio), and does not exclude
  BN and bias.
- A3: the double softmax in linear evaluation.
- A4: RGB colour jitter and greyscale mix science, template and difference.
- A5: strong blur erases the sharpness cue.
- A6: random-resized crops can drop the central source.
- The ImageNet ResNet stem: on 63 px, layer 4 is 2×2.
- The legacy loss form is DCL plus a constant (ADR 0002), not NT-Xent.

**The legacy code** is retired: tag `legacy-final`. Phase 5 writes the SimCLR task fresh on
the phase-4 engine.

## Tasks (draft)

### 5.1 The pretraining corpus (needs the user's approval of the download)
- **Unlabelled raw stamps** from the ZTF public alert archive (`ingest-archive`, built in
  phase 3's parallel task), with:
  - a seeded night sample over 2018–2026;
  - first detections with ALeRCE's quality cuts;
  - **an exclusion of every object** in CD21 Validation/Test and R5 Validation/Test (phase-3
    cross-benchmark rule).
- **Size:** start at about 1M stamps (about 48 GB float32 on disk, phase 4's tier 3 memmap).
- **Alternative if the download is not approved:** today's `simclr_300` (about 300k stamps,
  normalised, no ids), which **cannot be checked for leakage**. Results on it are labelled
  accordingly.

### 5.2 Views (augmentations), physically consistent
- Dihedral rotations and flips (all 8).
- A small centred shift (±3 px).
- Mild scale jitter.
- Per-channel brightness and contrast of about ±0.15, on floats.
- Optional: background noise as C5 (cross-channel consistent); channel dropout as an
  ablation.
- **Never:** hue, saturation, greyscale, strong blur, or crops that drop the source.
- Every view parameter is drawn per sample on the host; per-pixel noise follows phase 4's
  rule.
- A test checks that each view keeps the central source in frame.

### 5.3 The SimCLR task on the engine (ResNet-18 first)
- **Encoder:** a CIFAR-style stem (3×3 stride-1 conv, no max-pool), `fc = Identity`,
  512-d `h`.
- **Head:** Linear–BN–ReLU–Linear to 128.
- **Loss:** NT-Xent in paper mode, T in [0.2, 0.5], in fp32.
- **Optimiser:** LARS with BN and bias excluded, weight decay 1e-6, lr 0.3·B/256, a 10-epoch
  warmup, and a cosine **over exactly the run's length**, stepped per iteration.
- **Batch:** about 256 on the 5090. The 1050's limit is measured.
- **SupCon** (Khosla 2020, the "out" variant) as a second loss, on labelled data.
- **Tests:**
  - the loss against a brute-force reference;
  - the schedule ends at lr_min exactly at the last step;
  - LARS leaves the excluded parameters with plain SGD;
  - the stem's output size.

### 5.4 Evaluation protocol for SSL
- **Linear probe:** plain cross-entropy on logits, frozen encoder in `eval()`, no
  augmentation.
- **Fine-tuning:** schedule over the run's length.
- **Label fractions 1 / 10 / 100 %:** stratified and grouped by object, 3–5 repeated
  subsets × seeds. The fractions replace the frozen `sup_1` and `sup_10` files.
- **Feature settings:** none, position only, all 23.
- **Benchmarks:**
  - CD21 grouped CV, for the decisions;
  - R5 test, for high-power reporting;
  - CD21 test, reported once.
- **Baselines** (attributing gains to SSL):
  - supervised from scratch on the same fraction;
  - the paper model;
  - ImageNet-pretrained ConvNeXt-nano or ResNet-18 fine-tuned (Rehemtulla et al.
    2025/26).

  Decisions use the protocol of task 2.3.

### 5.5 Then custom encoders
- p4/p4m equivariant encoders (phase 2.5's results inform this).
- The multi-scale input (RJ23).
- BYOL or VICReg, which need no negatives: an ablation.

## Dependencies

- **Phase 4:** engine core (4.1), config (4.2), data tiers (4.3), registry and queue (A2,
  A3).
- **Phase 3:** `raw-alerts` kind (merged), R5 (3.5), `ingest-archive`.
- **The RTX 5090:** for batch 256 at 63 px and for corpus throughput.

## Risks
- **Leakage** between the pretraining corpus and the evaluation objects: the exclusion
  lists, and a check by oid.
- **False negatives** in a batch: several alerts of the same object. Sample one alert per
  object per epoch, or mask same-oid pairs.
- **The usual negative result.** Rehemtulla et al. found that pretrained backbones only
  match a custom CNN on these stamps. Pre-register what counts as success: for example,
  a gain at 1 % labels, or without features.
- **Compute:** SimCLR needs many epochs. Budget on the 5090; the 1050 only for smoke runs.
