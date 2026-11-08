<!-- ADR 0008: how the RJ23 input path (task 3.3) settles the points where RJ23's paper and
     code differ or are silent. -->
# 0008: The RJ23 input path: the paper's `|x| > 1e10` rule by default, RJ23's code elsewhere

- **Status:** accepted (2026-11-08, phase 3, task 3.3)
- **Implemented by:** `src/astro_simclr/data/rj23_input.py`
- **Checked by:** `tests/test_rj23_input.py`

## Context

Reyes-Jainaga et al. 2023 (RJ23, arXiv:2304.13080) normalise each raw 63×63 cutout per
channel as `(x − min|x|) / (p99|x| − min|x| + 1e-8)`, clip to [−2, 2], append a 3-channel
invalid-pixel mask, and build four views (full 63², cropped 16², low-resolution 16²,
multi-scale 4 × 8²). Their code is public (`alercebroker/multiscale_stamps`, MIT; we
reimplement from its description in the task 3.0 research and cite it, nothing is copied).
The paper and the code disagree on one point, and leave a few others to the reader:

1. **Values with `|x| > 1e10`.** The paper (§II.1) says they are invalid: replaced with
   zeros and flagged. The code (`normalize_stamps`) only sets them to NaN in the copy of
   `|x|` used for the statistics; the pixel itself stays finite, is normalised and clipped
   (to ±2 in practice), and is **not** in the mask, which is `~np.isfinite(x)`.
2. **The low-resolution view.** The code uses one `AveragePooling2D(pool_size=(4, 4),
   padding='same')`, not two 2×2 poolings. The two agree except in the last, partial block
   (63 = 15 × 4 + 3): one 4×4 pooling averages rows 60–62 equally, two 2×2 poolings weigh
   row 62 twice as much (it is alone in its 2×2 block).
3. **TF 'same' average pooling** pads `out · k − n` pixels, `⌊total / 2⌋` before and the
   rest after, and divides by the number of *valid* pixels in each window. On 63 (k = 2 or
   4) the one padded pixel goes at the end.
4. **A channel with no pixel left for the statistics** (all NaN, or all `|x| > 1e10`): the
   code's statistics are NaN, every pixel becomes NaN and then 0 (`nan_to_num`).
5. **C25's order** (Carvajal et al. 2025, §2.1): NaN → 0 *before* the normalisation, which
   could pull `min|x|` to 0; RJ23's code skips them in the statistics and zeroes them after.

## Decision

1. **`large_values="paper"` is the default**: `|x| > 1e10` is invalid exactly like NaN and
   ±inf (out of the statistics, masked, zeroed). **`large_values="code"`** reproduces RJ23's
   code: out of the statistics only, normalised and clipped, not masked. Both are tested
   by hand-computed cases. The paper is the default because it is the documented method and
   a 1e10 pixel is a sentinel, not a flux: clipping it to ±2 makes it look like the
   brightest real pixel. The flag exists so that a replication can match the code if the
   difference turns out to matter; how many rows it touches is to be measured on the
   raw-alert data (task 3.1b), since the stored CD21 stamps no longer hold raw values.
2. **Everything else follows RJ23's code:** one 4×4 'same' pooling for the low-resolution
   view; iterated 2×2 'same' poolings (63 → 32 → 16 → 8) after the normalisation for the
   multi-scale view, masks pooled too (so fractional); crop starts `⌈(n − size) / 2⌉` (24
   for 16 on 63, 28 for 8 on 63, 12 and 4 on the pooled 32 and 16); scale-major channels
   1″/2″/4″/8″, each `sci, ref, diff, m_sci, m_ref, m_diff`; NaN and inf skipped in the
   statistics and zeroed afterwards (not C25's order); an empty channel is all zeros.
3. **The numerics:** the percentile is numpy's `linear` method (equal to
   `np.nanpercentile`, tested bit-for-bit), computed with one sort per chunk; the
   normalisation runs in float64 and is stored as float32; pooling sums in float64, so the
   result does not depend on the pixel order. Every function is numpy only, deterministic
   and chunked (results independent of the chunk size, tested).

## Consequences

- The defaults differ from RJ23's code only on pixels with `|x| > 1e10`; any run that wants
  the code's behaviour sets `large_values="code"`, and the reports name the mode.
- The crops are centred on pixel 31.5, not on the source at 31 (RJ23's own docstring notes
  it), and the pooled grids pad at the end. So a rotation or flip of the stamp moves each
  crop window by one pixel and changes the last row or column of the pooled views. RJ23
  average the network over the 8 rotations and flips of these views, which is not the
  same as rotating the stamp; the tests pin the exact relation (`crop(g(x)) = g(mirrored
  window of x)`) so that a later equivariant model (phase 3.4 or after) knows what it
  gets.
- C25's FFT branch is not built (plan review round 2: its satellite labels cannot be
  rebuilt from public data). If it is, its open choices (the median of what, the
  placement, `fftshift`, a log or a normalisation) need their own ADR.
