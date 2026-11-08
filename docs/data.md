<!-- Data files of the project: schemas, provenance and workflow. Code: src/astro_simclr/data/
     (docs/architecture.md, section `data`). Checked against the real files on 2026-11-07. -->
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

`astro-simclr prepare-data PKL --out DIR` (default `--out`: the `data_dir` setting) writes:

```
DIR/<pickle stem>/
    meta.json
    Train/images.npy        uint8   (N, 63, 63, 3)
    Train/labels.npy        int64   (N,)        not in the pretraining Train split
    Train/features.npy      float64 (N, 26)     not in the pretraining Train split; all 26
    Validation/...  Test/...
```

`meta.json` records the format (`astro-simclr-prepared`, version 1), the kind, the source
(name, size, SHA-256), the schema (stamp shape, label and feature names), and per split the
count, the counts per class and each file's dtype, shape and SHA-256. Stamps are converted
with the legacy `img_float2int` arithmetic in the *stored* dtype. The prepared arrays are
bit-identical to applying the legacy function to every stored stamp, which was checked on
all three complete real files. The features keep float64, and the datasets cast them to
float32 as the legacy code did.

Training opens it with `np.load(allow_pickle=False, mmap_mode="r")`, so no pickle is ever
read again and the stamps are paged in on demand: `red` takes 596 MB prepared, against
2.49 GB as a pickle. `prepare-data` needs about the pickle's size in RAM (the
pretraining set: 14–29 GB). The preflight (task 1.7) warns about pickles in the data
directory that do not fit in RAM, and checks the RAM for the prepared arrays.

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

Without the real data, `astro-simclr make-synthetic --out DIR --n-per-split N --seed S`
writes all four files (`td_ztf_stamp_17_06_20{,_sup_10,_sup_1}.pkl` and
`td_ztf_stamp_simclr_300.pkl`). They have the real containers and dtypes, and a strong
rotation-invariant class pattern in the central 27×27 crop. Every split of every file has
N stamps (N ≥ 5, classes balanced). The 1 % and 10 % files draw their Train split from the
full Train split with duplicates, and share its Validation and Test. The same seed gives
the same bytes.
