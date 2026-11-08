"""Calibration of the per-stamp probabilities: the expected calibration error (plan row D3).

:func:`expected_calibration_error` is the top-label ECE of Guo et al. 2017 ("On
Calibration of Modern Neural Networks") with ``M = 15`` equal-width confidence bins::

    ECE = sum_m |B_m| / n * |acc(B_m) - conf(B_m)|

where a stamp's confidence is its largest probability, its prediction the argmax, bin
``B_m`` holds the confidences in ``((m - 1) / M, m / M]`` (right-closed: a confidence of
exactly ``m / M`` is in bin ``m``, and 1.0 in the last), ``acc`` is the fraction of the
bin's stamps predicted correctly and ``conf`` their mean confidence; empty bins add
nothing. It is a **secondary** metric of the phase-2 comparisons (never a decision input,
``docs/evaluation-protocol.md``): ``metrics.json`` records it per evaluated stage
(``tasks.results.evaluation_entry``, both engines), and ``compare-variants`` and
``report-phase2`` report it per run from the predictions files.

Computed in float64 on the stored float32 probabilities. The right-closed rule is exact
in float64 only: it applies to the stored value, not to its decimal intent. A confidence
meant to be on an edge, such as 0.2 = 3/15, is stored as ``float32(0.2) =
0.2000000030...``, just above the float64 edge, so it lands in ``(3/15, 4/15]`` rather
than ``(2/15, 3/15]``. Nothing here imports torch.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["ECE_BINS", "expected_calibration_error"]

#: The number of equal-width confidence bins (Guo et al. 2017's 15).
ECE_BINS: Final = 15


def expected_calibration_error(
    probabilities: npt.ArrayLike, labels: npt.ArrayLike, *, n_bins: int = ECE_BINS
) -> float:
    """The top-label ECE of ``probabilities`` (N, C) against ``labels`` (N,) (module docstring).

    Raises:
        ValueError: if ``probabilities`` is not a non-empty (N, C) array with C >= 2,
            ``labels`` is not (N,), or ``n_bins`` is below 1.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError(f"probabilities must be a non-empty (N, C) array, got {probs.shape}")
    if truth.shape != (probs.shape[0],):
        raise ValueError(f"labels must be ({probs.shape[0]},), got {truth.shape}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == truth).astype(np.float64)
    # The inner edges 1/M .. (M-1)/M; side="left" puts a confidence equal to an edge in the
    # bin below it, the right-closed bins of the module docstring.
    edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    bins = np.searchsorted(edges, confidence, side="left")
    # |B_m|/n * |acc_m - conf_m| = |sum_{i in B_m} (correct_i - conf_i)| / n: the bin sizes
    # cancel, so only the per-bin sums of (correct - confidence) are needed.
    gap = np.bincount(bins, weights=correct - confidence, minlength=n_bins)
    return float(np.abs(gap).sum() / probs.shape[0])
