<!-- Data files of the project: schemas, provenance and workflow. Code: src/astro_simclr/data/
     (docs/architecture.md, section `data`). Checked against the real files on 2026-11-07;
     the float input and feature sections on 2026-11-08 (task 2.1); RJ23 inputs written
     2026-11-08 (task 3.3); the raw alerts on 2026-11-08 (task 3.1). -->
# Data

The ZTF stamp datasets are pickles, kept untracked in `dataset/` (`.gitignore`:
`dataset/*`). They are not public, and must never be committed or uploaded without a
decision on data governance (an open question in `docs/phases/01-runnable-package.md`).
The training commands never read the pickles: `prepare-data` converts each one once into
the [prepared layout](#prepared-layout), and everything else reads that.

## Files

| File | Kind | Size | Train | Validation | Test | State (2026-11-07) |
|---|---|---|---|---|---|---|
| `td_ztf_stamp_17_06_20.pkl` | classifier | ? | ? | 500? | 500? | **not received**. The original, before deduplication; the legacy code's "100 %" path |
| `td_ztf_stamp_17_06_20_red-002.pkl` | classifier | 2.49 GB | 50,582 | 500 | 500 | complete. The deduplicated original, `..._red.pkl`; `-002` is a download suffix |
| `td_ztf_stamp_17_06_20_sup_100-001.pkl` | classifier | 3.47 GB | (72,710) | — | — | **truncated** (no STOP opcode); needs a fresh download |
| `td_ztf_stamp_17_06_20_sup_10.pkl` | classifier | 293 MB | 7,480 | 500 | 500 | complete |
| `td_ztf_stamp_17_06_20_sup_1.pkl` | classifier | 73 MB | 810 | 500 | 500 | complete |
| `td_ztf_stamp_simclr_300.pkl` | pretraining | 14–29 GB (est.) | ~300k | 500 | 500 | **missing** |

SHA-256 of the received files (the user's `dataset/SHA256SUMS`, recomputed by
`inspect-data`):

```
48f50ca503b58ba9fe07ab4f923bfcf5d39f3d45326258ba28c8336fdbeebb75  td_ztf_stamp_17_06_20_red-002.pkl
15631b4f1c9ccbe2894211be65be127c856a78eb09aeca7f0853c864f8e82976  td_ztf_stamp_17_06_20_sup_1.pkl
60569c7b7bfa2e99f6769148980f59c65b54cb6ac64e41e147846355a191e518  td_ztf_stamp_17_06_20_sup_10.pkl
bd0260c3637099d3fb552b15a4c7341cc347090ce627f4d425452b1c0773499a  td_ztf_stamp_17_06_20_sup_100-001.pkl  (truncated)
```

### Provenance

From `jupyter_examples/separacion_dataset.ipynb`, and checked by hashing the stamps of the
received files:

1. `td_ztf_stamp_17_06_20.pkl` is the ALeRCE stamp-classifier dataset (Carrasco-Davis et
   al. 2021: 51,594 alerts, Validation and Test balanced at 100 per class).
2. `..._red.pkl` removes duplicate stamps (L2 distance < 1e-5) within each split. Its 50,582
   Train stamps are all distinct. One Train stamp also appears in Validation or Test.
   Distinct alerts of one object were kept: within 1.5", 24 Validation and 27 Test alerts
   share a position with a Train alert, and Train has groups of up to 16 alerts
   (`astro-simclr leakage-audit`; `docs/evaluation-protocol.md`, "Leakage").
3. `..._sup_100.pkl` is `red`'s Train split balanced by oversampling. Duplicates are drawn
   at random to bring each class up to the largest one: 5 × 14,542 = 72,710 stamps, which
   matches the size of the truncated file.
4. `..._sup_10.pkl` and `..._sup_1.pkl` are a random 10 % / 1 % of `red`'s Train, balanced
   the same way. `sup_10` has 5,058 distinct stamps, oversampled to 5 × 1,496. `sup_1` has
   505 distinct stamps, oversampled to 5 × 162 (SN: 14 distinct stamps). Every one of them
   is in `red`'s Train.
5. The Validation and Test stamps are identical in the three complete files (the stamps
   were compared, not the features or labels).
6. `td_ztf_stamp_simclr_300.pkl` was built by `utils/simclr_dataset.py` on a lab server,
   from `/home/shared/pickles/`. Its Train split holds about 300k unlabelled stamps,
   repaired and normalized with `nan_val=0`. Validation and Test are copied from
   `td_ztf_stamp_17_06_20.pkl`.

No random seed was set in the notebook, so the 1 % and 10 % splits cannot be regenerated
identically (backlog: regeneration with seeds).

## Schemas

All files are protocol-4 pickles written by numpy 1.x, and reference only
`numpy.core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype` and
`numpy.core.multiarray.scalar`. The top level is a dict with exactly the splits `Train`,
`Validation` and `Test`.

### Classifier files

Every split is a dict with the keys `images`, `features` and `labels` (in that order),
each a Python list of length N:

| Key | Item | Values |
|---|---|---|
| `images` | float32 ndarray (63, 63, 3) | [0, 1] (min 0, max 1 exactly); channels science, template, difference; no NaN/inf. Only ~1.8 % of pixels are exact multiples of 1/255, so the uint8 rounding matters |
| `features` | object ndarray (26,), items numpy float64 (and Python float in a few columns) | the 26 features below, already standardized on Train (mean 0, std 1), except `isdiffpos`, which is scaled but not centred (values 0 and 2.12); no NaN/inf |
| `labels` | numpy int64 | 0–4: `AGN`, `SN`, `VS`, `Asteroid`, `Bogus` |

Features, by column: 0 `sgscore1`, 1 `distpsnr1`, 2 `sgscore2`, 3 `distpsnr2`, 4 `sgscore3`,
5 `distpsnr3`, 6 `isdiffpos`, 7 `fwhm`, 8 `magpsf`, 9 `sigmapsf`, 10 `ra`, 11 `dec`,
12 `diffmaglim`, 13 `rb`, 14 `distnr`, 15 `magnr`, 16 `classtar`, 17 `ndethist`,
18 `ncovhist`, 19 `ecl_lat`, 20 `ecl_long`, 21 `gal_lat`, 22 `gal_long`,
23 `non_detections`, 24 `chinr`, 25 `sharpnr`. The legacy classifiers drop 13–15 (`rb`,
`distnr`, `magnr`) and use 23.

Counts per class (AGN, SN, VS, Asteroid, Bogus):

| File | Train | Validation | Test |
|---|---|---|---|
| `red-002` | 14,542 / 1,396 / 14,517 / 9,599 / 10,528 | 100 each | 100 each |
| `sup_10` | 1,496 each | 100 each | 100 each |
| `sup_1` | 162 each | 100 each | 100 each |

The plan's inferred schema (63×63×3 floats in [0, 1], 5 classes, 26 features of which 13–15
are dropped) is confirmed. Two things it did not say: the features are stored as object
arrays, already standardized, and the label-fraction files are oversampled.

### Pretraining file

As a classifier file, except that `Train` holds only `images`: a list of float64 (63, 63,
3) arrays (according to `utils/simclr_dataset.py` and the plan; not verifiable, since the
file is missing). The loader recognizes this kind by a `Train` split whose only key is
`images`.

### What the loader accepts

`load_pickle` (`src/astro_simclr/data/pickle_loader.py`) is lenient about containers:
lists, tuples or ndarrays for every key, object or numeric feature arrays, and numpy or
Python integers for the labels. It fails, with an error that names the file, split and key,
on:

- a missing or unexpected split or key, or an empty split;
- a stamp that is not (63, 63, 3) floating point, or with values that do not round into
  0–255 or that are NaN/inf;
- labels that are not integers, are out of 0–4, or do not match the number of stamps;
- features that are not 26 finite numbers per stamp;
- a truncated file (checked before unpickling) or a corrupt one;
- any global other than numpy's array, dtype and scalar.

**Trust:** unpickling can run arbitrary code. Load only these datasets, from a known
source, after checking their SHA-256. The numpy-only allowlist narrows the risk, but does
not make an untrusted pickle safe.

## Prepared layout

`astro-simclr prepare-data PKL --out DIR [--images uint8|float32]` (default `--out`: the
`data_dir` setting; default `--images uint8`) writes:

```
DIR/<pickle stem>/
    meta.json
    Train/images.npy        uint8   (N, 63, 63, 3)
    Train/images_f32.npy    float32 (N, 63, 63, 3)  only with --images float32
    Train/labels.npy        int64   (N,)        not in the pretraining Train split
    Train/features.npy      float64 (N, 26)     not in the pretraining Train split; all 26
    Validation/...  Test/...
```

`meta.json` records the format (`astro-simclr-prepared`, version 2), the kind, the image
dtypes (`images`: `["uint8"]` or `["uint8", "float32"]`), the source (name, size,
SHA-256), the schema (stamp shape, label and feature names), and per split the count, the
counts per class and each file's dtype, shape and SHA-256.

**Format versions.** Version 1 (task 1.2) had no `images_f32.npy` and no `images` key.
Version 2 (task 2.1) adds both. The writer always writes version 2, and the readers accept
1 and 2, so layouts prepared before 2.1 still load (as uint8-only). A uint8-only version-2
layout has the same array files, byte for byte, as version 1: re-preparing without
`--images float32` changes only `meta.json`.

**`images_f32.npy`** holds the stored stamps unchanged (float32, finite, in [0, 1]; a
float64 stamp of the pretraining file is rounded to float32), for the paper's float input
path ([below](#float-input-path)). It is 4× the size of `images.npy`: `red` takes 2.9 GB
prepared with it, and `prepare-data` then peaks at about 3.2 GB of RSS on the M4 (16 s).
On the three complete real files, `images.npy` is exactly `round(255 · images_f32.npy)`,
and its bytes are identical to the version-1 layouts. Stamps are converted
with the legacy `img_float2int` arithmetic in the *stored* dtype. The prepared arrays are
bit-identical to applying the legacy function to every stored stamp, which was checked on
all three complete real files. The features keep float64, and the datasets cast them to
float32 as the legacy code did.

Training opens it with `np.load(allow_pickle=False, mmap_mode="r")`, so no pickle is ever
read again and the stamps are paged in on demand: `red` takes 596 MB prepared, against
2.49 GB as a pickle. `prepare-data` needs about the pickle's size in RAM (the
pretraining set: 14–29 GB). The preflight (task 1.7) warns about pickles in the data
directory that do not fit in RAM, and checks the RAM for the prepared arrays.

## Float input path

The paper's model (Carrasco-Davis et al. 2021; M1 of phase 2) reads a 21×21 centre crop of
each stamp, re-normalised per channel on the crop. ALeRCE's code
(`alercebroker/pipeline@a3705ac`, `ztf_preprocessor.py`) does `image_crop_at_center` →
`image_normalize_by_image` → `image_nan_to_num`: rows and columns 21–41, `x -= nanmin`,
`x = x / nanmax`, **no epsilon** (a constant or all-NaN crop becomes 0), then NaN → 0.
`astro_simclr.data.float_input.crop_then_normalise(stamps, crop=21)` is that pipeline, and
`crop_split(images_f32)` applies it to a whole split in chunks.

Our pickles hold the stamps after the same normalisation over the full 63×63 stamp, with
NaN → 0. Min-max is affine-invariant, so re-normalising a stored crop gives ALeRCE's input
(to one float32 ulp), except where the raw stamp had NaNs: they are now 0, the same value as
the stamp's minimum, and would pull the crop's minimum down.

**The NaN approximation** (`nan_approximation=True`, the default): a zero pixel counts as a
former NaN when its channel, **counted over the 63×63 stamp**, has more than one exact zero
(a channel without NaNs has exactly one, its minimum). Those zeros are excluded from the
crop's minimum and maximum, and become 0 again. It is exact unless the channel's real
minimum also lies inside the crop (or is tied), in which case the minimum is masked too and
the crop's second-smallest value becomes its zero. `nan_approximation=False` treats every
stored zero as real (diagnostic 3 of task 2.4).

On `red`, every channel of every stamp has at least one exact zero. Of the 151,746 Train
channels, 133,141 have exactly one, 6 have two and 18,599 three or more (Validation and
Test: one, or four or more). The approximation touches (a masked zero inside the crop):

| Split | Stamps | AGN | SN | VS | Asteroid | Bogus |
|---|---|---|---|---|---|---|
| Train | 3,679 / 50,582 | 336 | 26 | 924 | 285 | 2,108 |
| Validation | 41 / 500 | 0 | 4 | 6 | 4 | 27 |
| Test | 34 / 500 | 1 | 2 | 6 | 2 | 23 |

(`float_input.approximated_stamps`; only these stamps' inputs differ between the two
settings.) The golden fixture `tests/fixtures/data/alerce_crop.npz` was produced once by
running ALeRCE's own three functions (`scripts/characterize/data/alerce_crop.py`, which
downloads the file pinned by commit and SHA-256 and never copies it) on synthetic raw
stamps with NaNs, constant channels and all-NaN crops: on raw stamps the port is
bit-identical, and from the stored stamps the approximation reproduces ALeRCE's input to
1.2e-7 on every case except the documented one (a NaN and the minimum both inside the
crop).

The 21×21 crop is concentric with the phase-1 uint8 crop of 27 (torchvision's
`CenterCrop`, rows 18–44): it is that crop minus 3 pixels per side, which the paper's model
zero-pads back to 27 (task 2.2a). `tests/test_data_float_input.py` checks the pixel
alignment.

### The 63×63 context (task 2.5-models, plan row B1')

`Settings.context: avgpool63` gives the paper model the whole stamp at the crop's size:
`float_input.context_then_normalise` average-pools the 63×63 stamp by 3 (each 21×21 pixel
is the mean of a 3×3 block; 63 = 3 × 21, so the blocks tile the stamp symmetrically and
the context rotates and flips with it), and the 3 resulting channels follow the crop's 3.
It is ours, not ALeRCE's. The decisions (ADR 0005):

- **Normalised like the crop, on its own pixels:** per channel, `x -= nanmin`,
  `x /= nanmax` without epsilon, NaN → 0, after the pooling. All 6 channels then lie in
  [0, 1] with minimum 0 and maximum 1 (0 if constant). Pooling is linear and min-max
  affine-invariant, so on the stored stamps this equals the same pipeline on the raw
  stamps, as for the crop.
- **NaN handling, the crop's:** with `nan_approximation` (one flag for both), the pixels of
  `former_nan_mask` become NaN before the pooling, and the block means ignore NaNs: a block
  with some former NaNs is the mean of its valid pixels; a block of only former NaNs stays
  NaN and becomes 0 (the minimum) after the normalisation, as a NaN pixel of the crop. One
  more pixel differs from the raw stamp in a channel that had NaNs: its minimum is 0 in the
  stored stamp, like the NaNs, and is left out of its block's mean with them (for the crop
  this happens only when the minimum is inside the crop; here it always is). Without the
  approximation, every stored zero is a real pixel.
- The block means are accumulated in float64 and cast back to float32, so the context does
  not depend on the order of the pixels and commutes exactly with the rotations (tested;
  in float32 the order changes the last bit, which would break exact invariance).

`context_split(images_f32)` applies it to a split in chunks; `training.device_data.
float_crops(..., context=True)` concatenates it to the crops (6 channels, about 270 MB more
for `red`). Only the float input path has it: `context` with `paper_input: uint8` is an
error.

## RJ23 inputs

Phase 3 (task 3.3) reproduces the input of Reyes-Jainaga et al. 2023 (RJ23,
arXiv:2304.13080) from **raw** cutouts: `(N, 63, 63, 3)` floats, channels science,
reference, difference, NaN and ±inf kept (the raw-alert kind of task 3.1b will store them;
the CD21 pickles cannot be used, their stamps are already min-max normalised).
`astro_simclr.data.rj23_input` is a numpy-only reimplementation, from the paper and the
description of RJ23's MIT code (`alercebroker/multiscale_stamps`, `src/data_loader.py`);
the open points are settled in ADR 0009.

**Normalisation and mask** (`normalise`, `invalid_mask`, `channel_statistics`), per stamp
and channel on the full 63×63, before any crop or pooling:

- `(x − min|x|) / (p99|x| − min|x| + 1e-8)`, clipped to [−2, 2]. The statistics skip NaN,
  ±inf and `|x| > 1e10`; p99 is numpy's linear percentile (`np.nanpercentile`). The centre
  is `min|x|`, not 0, so zero is preserved only when some pixel is 0, and a constant
  negative channel saturates at −2.
- The invalid pixels become 0; a channel with no pixel left for the statistics is all 0.
- The mask is 1.0 on invalid pixels. **`large_values`** decides about `|x| > 1e10`:
  `"paper"` (default) masks and zeroes them; `"code"` (RJ23's code) only leaves them out of
  the statistics, then normalises and clips them, unmasked.
- Output `(N, 63, 63, 6)` float32, channels `sci, ref, diff, m_sci, m_ref, m_diff`.

**Views** (`rj23_views(stamps, views, *, large_values, chunk_size)` from raw cutouts, or
`full_view` / `cropped_view` / `lowres_view` / `multiscale_view` from the normalised
stamps); all channels last, float32:

| View | Shape | Built as |
|---|---|---|
| `full` | 63×63×6 | the normalised stamp |
| `cropped` | 16×16×6 | rows and columns 24:40 |
| `lowres` | 16×16×6 | **one** 4×4, stride-4 average pooling, TF `'same'` padding |
| `multiscale` | 8×8×24 | 8×8 crops at 1″ (28:36 of 63), 2″ (12:20 of 32), 4″ (4:12 of 16), 8″ (all of 8) |

- TF `'same'` pooling (`tf_same_avg_pool`): output `⌈n / k⌉`, padding `⌊total / 2⌋`
  before and the rest after (on 63: one pixel at the end), each output the mean of the
  window's valid pixels. It equals torch's `avg_pool2d(..., ceil_mode=True)` when TF pads
  at most one pixel (tested), not otherwise (62 by 4).
- The multi-scale view pools iteratively by 2×2 after the normalisation (63 → 32 → 16 → 8),
  masks included (fractional at coarse scales). Channels are scale-major: `6s : 6s + 6`
  is scale 1″, 2″, 4″, 8″ for s = 0..3 (fields of view 8″, 16″, 32″, 63″).
- The crops start at `⌈(n − size) / 2⌉`, so they are centred on pixel 31.5 while the
  source is at 31: in every crop the source pixel sits at local (7, 7) of 16 or (3, 3) of 8.
  A rotation or flip of the stamp therefore moves the crop window by one pixel, and the
  pooled views change in their last row or column; the full view, the normalisation and
  the mask commute exactly with the 8 rotations and flips.

**Position** (`unit_sphere(ra, dec)`, degrees): `(cos ra cos dec, sin ra cos dec,
sin dec)`, `(N, 3)` float32; RJ23 use it instead of CD21's 23 features.

**Layout and memory.** `channels_first` gives `(N, C, H, W)` for torch
(`torch.from_numpy` takes it as is); the module never imports torch. Chunks of
`chunk_size` stamps (default 1000) bound the float64 intermediates (peak about 0.7 GB for
1000 stamps with all four views, measured); results do not depend on the chunk size. Per
stamp the float32 views take 95 KB (full) and 6 KB each (cropped, low-res, multi-scale).

C25 (Carvajal et al. 2025) use the same 6-channel stamps. Their FFT channel is not built:
C25's satellite labels cannot be rebuilt from public data (plan review round 2).

## Features and M1's normalisation

The stored features are standardised. `astro-simclr check-features DIR [--json]` checks
that they were standardised with **M1's own statistics**
(`astro_simclr.data.m1_features.M1_FEATURE_NORM_STATS`: the mean and std, ddof 0, of the 23
kept features, from M1's `feature_norm_stats.pkl` at `alercebroker/pipeline@a3705ac`,
SHA-256 `65a5611c…8064`, read with `pickletools` and never unpickled;
`scripts/characterize/data/m1_feature_stats.py` re-derives them). After `x · std + mean`:
`ndethist`, `ncovhist` and `non_detections` must be integers, the clipped `sgscore1–3` and
`distpsnr1–3` must have a floor of −1, and `isdiffpos` must be 0 or 1 (its M1 mean is 0: it
is scaled only). It exits with 1 if any test fails.

On 2026-11-08 `red-002`, `sup_10` and `sup_1` pass in all three splits: the counts are
integers to 5e-13, the six clipped features reach exactly −1 (in Train) and never go
below, and `isdiffpos` is exactly {0, 1}. A std wrong in the sixth digit would already break
`ncovhist` (std 425), so the files were standardised with exactly M1's statistics. Two
observations for later tasks: the Train means of the stored features (except `isdiffpos`)
are up to 3e-4 in absolute value, not 0 (M1's Train is the set before the 12 duplicates
were removed), and `non_detections` reaches −190 in Train (ALeRCE does not clip it at 0).

## Raw alerts (task 3.1)

The stored stamps are normalised: per channel, min-max over 63×63, then NaN → 0. The raw
alerts behind them are recovered from ALeRCE's public services by `astro-simclr
fetch-alerts` (code: `src/astro_simclr/alerts/`, ADR 0011). They keep the real NaN and
saturation masks, and the full candidate record. Phase 3 needs them for RJ23/C25 inputs,
the dataset reconstruction and the object-level splits. Phase 2 needs them for row C1 and
the fake-SN follow-ups. A pretraining corpus needs their `candid`s as its leakage
exclusion list.

### Method and provenance

The method comes from the integrator's probe of 2026-11-08: 9 of 10 probe stamps
recovered, with no flip or rotation. For each row of a prepared split:

1. De-standardise `ra`, `dec`, `magpsf` and `sigmapsf` with M1's statistics
   ([above](#features-and-m1s-normalisation)).
2. `GET https://api.alerce.online/ztf/v1/objects/?ra=..&dec=..&radius=1.5`. Keep the
   trailing slash: without it the server redirects.
   - No object at all: **`ghost_candidate`**. RJ23's `build_dataset.py` calls some CD21
     Bogus "ghost bogus": objects that are not in ALeRCE's database, whose stamps exist
     only in CD21's files.
3. `GET /ztf/v1/objects/{oid}/detections` for every object in the cone. Match a detection
   when `|Δmagpsf| ≤ 1e-4` and `|Δsigmapsf| ≤ 1e-4`. `candid` arrives as a string,
   `isdiffpos` as ±1, and there is no `jd` or `drb`: those come from the avro.
4. **Ties.** `jd` is not a stored feature, so an AGN or VS with hundreds of detections can
   tie. Every matching `candid` is downloaded (at most 16; more is `ambiguous` without
   downloading), and the stamp check decides.
5. `GET https://avro.alerce.online/get_avro?oid=..&candid=..` returns the whole alert:
   about 50 KB, snappy codec, the candidate record, the previous candidates, and the
   three gzipped FITS cutouts. It replaces three `get_stamp` FITS requests, and it gives
   `jd`, `drb`, `xpos` and `ypos`.
6. **Stamp check:** ALeRCE's min-max of the raw cutouts must equal the stored
   `images_f32` row, max |Δ| ≤ 1e-6.
   - The normalisation is per channel over 63×63, no epsilon, NaN → 0:
     `crop_then_normalise(raw, crop=63, nan_approximation=False)`.
   - A cutout cut at a CCD edge (smaller than 63×63) is padded with NaN by
     `repair_non_squared_stamp`, from the candidate's `xpos`/`ypos`, before the check.

Statuses: `ok` (one candidate passes; stored), `mismatch` (downloaded, none passes; the
closest is recorded, not stored), `ghost_candidate`, `no_object` (objects in the cone, none
with a detection), `no_detection_match`, `ambiguous` (several candidates pass, or more than
16 tie, or a miss when the cone search has an unread second page), `bad_alert` (no
candidate is a readable alert), and `http_error`. `http_error` is a timeout, 5xx or 429
still failing after the retries, or malformed JSON (these are transient), or a final 4xx. Only `http_error` is retried, by the next
`--resume`. Ghosts are final: no retry beyond the transport's backoff, and no wider
tolerance.

**Acceptance.**
- Every class recovers at least 98 % of its **non-ghost** rows as `ok`.
- The misses are reported in groups: not found, ambiguous, timeout, mismatch. Ghosts are
  counted per class apart.
- No stored row fails the stamp check: only `ok` rows are stored.
- Non-63×63 (CCD-edge) alerts are counted apart.
- `alerts-summary RAW_DIR` prints all of this, and exits 1 until the acceptance is met.

### Store layout

```
dataset/raw/<dataset>/<split>/
    fetch.json        format astro-simclr-raw-alerts v1; the prepared dataset and its
                      source SHA-256; split, size, class counts; endpoints; tolerances
    index.csv         one line per attempt, append-only; the last line of a row wins
    shard-00000.npz   one committed batch (100 rows by default), np.savez_compressed:
                        rows, candid   int64 (n,)              the ok rows only
                        cutouts        float32 (n, 63, 63, 3)  science, template,
                                                               difference; raw, NaN kept
                        candidates     uint8                   JSON list of the candidate
                                                               records (scalar fields)
                        responses      uint8                   JSON list: url, kind,
                                                               retrieved_at (UTC),
                                                               sha256, offset, size
                        bodies         uint8                   the raw response bytes
```

Each index line holds the row, its label and its status.
- **Search:** the de-standardised targets (`target_ra`, `target_dec`, `target_magpsf`,
  `target_sigmapsf`; for a ghost, the only position on record, for later position groups),
  the objects in the cone, and every matching `oid:candid`.
- **The chosen alert:** its candidate fields (`oid`, `candid`, `jd`, `fid`, `magpsf`,
  `sigmapsf`, `isdiffpos`, `rb`, `drb`, `ra`, `dec`), the raw shape, the NaN count and the
  check's max |Δ|.
- **Provenance and cost:** the shard of its batch, the responses it used
  (`kind:sha256`), and the requests and seconds it cost.

**Snapshot.** Every raw response is kept byte for byte: the cone-search JSON, the full
detection list of each object (needed later to pick a random or the first alert per
object), and every downloaded avro.
- So the index can be rebuilt offline: `fetch-alerts ... --from-snapshot RAW_DIR --out
  OTHER` re-decides every row from the stored bytes, with no network.
- Cone searches and detection lists are cached in memory by URL, because rows share
  objects (about 1,392 rows of `red` share a position).

**Why npz shards and a CSV index.**
- The run lasts 1–2 days and must survive interruptions. A batch is committed as one
  shard, written to a temporary file, fsynced and renamed; only then are its index lines
  appended and fsynced. An index line therefore never names a missing shard.
- The directory is fsynced after the rename, and `fetch.json` is written atomically.
- A crash in between leaves an orphan shard that no line references. A truncated last
  index line is dropped, and text fields never contain a newline.
- On `--resume`, an `ok` row whose shard is missing is fetched again.
- The shards load with `allow_pickle=False`, since the records are JSON text in uint8
  arrays.
- CSV needs no new dependency. One file per batch keeps the file count near 500 for `red`.

`dataset/` is untracked (`.gitignore`), and the raw alerts must not be committed either.
They are ZTF public alerts, but they are retrieved from ALeRCE's service and keyed to our
non-public labelled set.

**Size and cost**, from the smoke run below:
- about 94 KB per row, so about 4.9 GB for `red`'s 51,582 rows;
- about 3.1 requests per row, so about 160k requests.
- The run is sequential and **latency-bound**: 1.29 s per request on average at
  `--rate 1`, more than the 1 s interval, so the limiter rarely waited. At the smoke run's
  4.06 s per row, `red` takes about 58 h, and a higher `--rate` shortens that only where
  the latency is below the interval. The latency from the GPU machine may differ.

### Running it (GPU machine, WSL2, unattended)

```bash
astro-simclr fetch-alerts --data-dir DATA --dataset td_ztf_stamp_17_06_20_red-002 \
    --split Train --out dataset/raw/td_ztf_stamp_17_06_20_red-002 --rate 1.5 --quiet
# after any stop, or to retry the http_error rows:
astro-simclr fetch-alerts ... --resume --quiet
astro-simclr alerts-summary dataset/raw/td_ztf_stamp_17_06_20_red-002
```

- The prepared layout needs `images_f32.npy` (`prepare-data --images float32`).
- One rate limit covers both hosts; first attempts and retries go through it alike.
- Timeouts, 5xx and 429 are retried 5 times with backoff 2, 4, 8 and 16 s, capped at 60 s.
  A numeric `Retry-After` wins over the cap, up to 600 s. The API sends no rate-limit
  headers, so the throttling is ours.
- Each response must arrive in full within 300 s (the socket timeout of 60 s is per
  read). Redirects are followed only to `https://`.
- After 3 transient `http_error` rows in a row the run pauses for `--cooldown` seconds
  (default 300). A final 4xx does not count toward that streak.
- SIGTERM, SIGHUP (a closed ssh session) and Ctrl-C stop the run after the current row and
  commit the batch (exit 128 + the signal number), so a signal never cuts a commit. A
  second signal forces the exit. A hard kill loses at most one batch.
- A failed commit (a full disk, say) is not retried, and its responses stay pending.
  Resume after fixing the cause.
- Memory: the response cache is bounded at 256 MB.
- Rows are visited round-robin over the classes, so a partial run covers them evenly.
- The `User-Agent` is generic (`astro-simclr/<version> (research; raw alert retrieval)`).

### Smoke run (2026-11-08)

Run on `red-002` Validation with `--limit 50 --rate 1`, from the M4, at 50 rows (10 per
class):
- 47 `ok`, 3 `ghost_candidate` (all Bogus: rows 402, 405 and 408), and no mismatch,
  not-found, ambiguous or timeout;
- max |Δ| 1.2e-7 over the `ok` rows, re-checked from the store; 3 of them have NaN pixels;
- no tie and no CCD-edge cutout;
- 157 requests, 203 s: 4.06 s per alert;
- `--from-snapshot` rebuilt the same 50 statuses offline in 0.9 s.

## Workflow and manifest

1. **Receive** the pickles into `dataset/`, and record their SHA-256 (`SHA256SUMS`, kept
   with the data and out of git).
2. **Inspect** each one: `astro-simclr inspect-data dataset/X.pkl` prints keys, containers,
   dtypes, shapes, counts per class, size and SHA-256, and fails (exit 1) on a truncated,
   corrupt or off-schema file. `--json` gives the same summary as JSON. Compare the
   SHA-256 with the source's.
3. **Prepare** each one, once, on a trusted machine with enough RAM:
   `astro-simclr prepare-data dataset/X.pkl --out data/`. The source SHA-256 travels in
   `meta.json`, so a prepared directory can always be traced to the exact pickle.
4. **Check** the prepared directory: `astro-simclr inspect-data data/X` validates it
   against the schema and `meta.json`, and re-hashes every array file.
5. **Manifest** (task 1.7): on its first run on a data directory, `astro-simclr preflight
   --data-dir DIR` writes `DIR/data-manifest.json`: per prepared dataset, its kind, source
   (name, size, SHA-256) and, per split, the counts and the SHA-256 of every array file
   (from `meta.json`, after `inspect-data` has re-hashed them). Later runs verify against
   it: a missing or changed dataset fails, a new one warns (delete the manifest to record
   the new set). The chain
   source hash → `meta.json` → manifest means a run can be traced to its data, and a
   changed or re-prepared file is noticed, without hashing the multi-GB pickles again.

   **Migration to float32 (task 2.1).** Re-preparing a dataset with `--images float32`
   adds `images_f32.npy` to every split, so a manifest recorded before fails. The preflight
   lists what changed (`Train/images_f32.npy: new (prepared with --images float32)`) and
   names the command that re-creates the manifest:
   `rm DIR/data-manifest.json && astro-simclr preflight --data-dir DIR`. Re-preparing
   uint8-only gives the same manifest entry, so it still matches.

Without the real data, `astro-simclr make-synthetic --out DIR --n-per-split N --seed S`
writes all four files (`td_ztf_stamp_17_06_20{_red-002,_sup_10,_sup_1}.pkl` and
`td_ztf_stamp_simclr_300.pkl`). The 100 % file carries the name of the received stand-in
`red-002` (task 1.7; it was `td_ztf_stamp_17_06_20.pkl` before), so the reference configs
and the defaults name the same datasets on synthetic and on real data. They have the real containers and dtypes, and a strong
rotation-invariant class pattern in the central 27×27 crop. Every split of every file has
N stamps (N ≥ 5, classes balanced). The 1 % and 10 % files draw their Train split from the
full Train split with duplicates, and share its Validation and Test. The same seed gives
the same bytes.
