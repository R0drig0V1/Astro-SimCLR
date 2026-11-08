<!-- Phase 3 of the roadmap (docs/phases/README.md); owned by the integrator. DRAFT: to be
reviewed with /multi-agent-herdr:improve-plan before the dependent tasks start. -->
# Phase 3: the newer benchmarks and the raw-alert dataset (draft)

## Goal

The user's request (2026-11-08) is to prepare the jump to the post-2021 work while phase 2
runs: replicate the newer ALeRCE stamp classifiers and rebuild the data they need. The
targets are:

1. **RJ23**: Reyes-Jainaga et al. 2023 (ApJL, arXiv:2304.13080).
   - 6 classes (Satellite added);
   - an imbalanced, stratified 60/20/20 split;
   - **macro-F1** about 87 for full 63×63 stamps, matched by multi-scale 4×16² stamps;
   - also evaluated on each object's first stamp.
2. **C25**: Carvajal et al. 2025 (A&A 697, A119, arXiv:2411.03258).
   - the RJ23 dataset, plus a cutout of the 2D FFT of the difference image as an extra
     branch;
   - normalisation: the 99th percentile of the absolute values, then a clip at 2;
   - an integrity (NaN) mask as extra channels;
   - the satellite accuracy of the 16″ model rises from 72.0 % to 87.8 %.
3. **N24** (later): Neira, Estévez & Förster 2024 (arXiv:2405.15073), sequences of 2–5
   detections, 3 classes.

The phase-2 rules carry over: the evaluation protocol of task 2.3, no test-based decisions,
and the portability rule. Never copy ALeRCE code or weights into the repo.

## Facts so far (2026-11-08)

- **Raw stamps can be recovered.** De-standardise `ra`, `dec`, `magpsf` and `sigmapsf`
  with M1's statistics, match on ALeRCE's API to get `oid` and `candid`, then download from
  `avro.alerce.online`. In a probe, 9 of 10 stamps came back, and they equal our stored
  stamps to ≤ 1e-7 after ALeRCE's min-max (`docs/phases/README.md`).
- **The bulk unlabelled source** is the ZTF public alert archive
  (`ztf.uw.edu/alerts/public/`).
- **Neither RJ23's dataset nor C25's is known to be public.** To be checked (task 3.0).

## Tasks

Parallel from the start: 3.0, 3.1 and 3.2. After the plan review and 3.0: 3.3, 3.4, 3.5
and 3.6.

### 3.0 Research: availability and exact specifications (no repo change)
For RJ23, C25 and N24, establish:
- whether data, code or weights are public (GitHub, Zenodo, ALeRCE);
- the exact preprocessing: the scales and pooling of the multi-scale stamp, the
  normalisation, the mask, the FFT crop and padding;
- the architecture and training: optimiser, epochs, selection, augmentation, class
  weighting;
- the metric definitions;
- how the labels were built: TNS cross-match and its rules, the satellite tagging, the
  purity cuts.

Also: TNS access (public CSV or API key), and whether ALeRCE would share the RJ23
`oid`/`candid` list.

### 3.1 Raw alert retrieval (`fetch-alerts`)
A resumable, rate-limited downloader:
- **Input:** a prepared split (resolved through the de-standardised features), or a list
  of `oid`/`candid` pairs.
- **Output:** for each alert, the avro alert (three raw cutouts, the candidate record,
  previous candidates) in a sharded store under `dataset/raw/`, plus an index of `oid`,
  `candid`, `jd`, `fid` and the split row.
- **Acceptance:** on every recovered row, ALeRCE's min-max of the raw cutouts reproduces
  the stored float stamp to ≤ 1e-6. Mismatches and misses are reported per class. No
  unpickling, and downloaded files are treated as untrusted data.
- **Run** on the GPU machine for the whole `red` labelled set (about 52.6k alerts), at
  1–2 requests/s.

### 3.2 Benchmark metrics and object-level splits
- Macro-F1, macro precision and macro recall, plus per class.
- Paired bootstrap CIs for macro-F1, reusing `protocol.stats`.
- Splits stratified by class and **grouped by object** (`oid`): no object in two splits,
  because several alerts of one object would leak.
- A "first stamp" evaluation view.
- Built on synthetic data first; real data once 3.1 gives the `oid`s.

### 3.3 RJ23 / C25 input path (after 3.0)
From raw cutouts:
- the full 63×63 view, the cropped 16 view and the multi-scale 4×16² view;
- C25's normalisation (p99 of the absolute values, clip at 2) and the integrity mask
  channels;
- the FFT cutout of the difference image (C25).

Tested against the paper's figures where possible.

### 3.4 RJ23 / C25 models (after 3.0 and 3.3)
The adapted stamp classifier: rotations and flips averaged, plus position metadata (RJ23),
plus the FFT branch (C25). New model code in `src/astro_simclr/`; our p4/p4m variants are
candidates on top.

### 3.5 Dataset reconstruction (after 3.0 and 3.1)
A 6-class labelled set in RJ23's spirit:
- CD21's labelled alerts (recovered in 3.1);
- spectroscopic SN from TNS;
- satellites, by RJ23's rule, or by a documented substitute if the rule cannot be
  reproduced;
- RJ23's purity cuts.

Document every deviation. If ALeRCE shares the `oid` list, rebuild exactly instead.

### 3.6 Replication runs (after 3.2 to 3.5)
RJ23's scenarios (full, cropped, low-resolution, multi-scale) and C25's FFT models, 5 seeds,
macro-F1 on the rebuilt test set and on its first-stamp view. The result is reported as "on
our reconstruction", never as RJ23's own numbers unless the dataset is the same.

## Risks
- **The reconstructed dataset is not RJ23's**, so absolute numbers do not compare. Only
  the internal comparisons of scenarios are valid.
- **Satellite labels** (580 in RJ23) were partly visual. A substitute rule may change the
  class.
- **External services:** ALeRCE's API rate limits and outages; TNS terms of use. Keep the
  load polite and document the provenance.
- **Storage:** the raw labelled set is about 3 GB. The unlabelled corpus is much larger;
  that is the next phase.
