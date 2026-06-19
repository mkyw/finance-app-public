"""Effective sample size and confidence labels for kernel-weighted matches."""

from __future__ import annotations

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size.

    .. math:: N_{\\text{eff}} = \\frac{(\\sum W_i)^2}{\\sum W_i^2}

    Args:
        weights: Non-negative sample weights. Can include zeros; zeros
            contribute nothing to either numerator or denominator so they
            are effectively dropped.

    Returns:
        Effective sample size. Returns 0.0 if all weights are zero or the
        input is empty.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    sum_w = float(w.sum())
    if sum_w <= 0.0:
        return 0.0
    sum_w2 = float(np.square(w).sum())
    if sum_w2 <= 0.0:
        return 0.0
    return (sum_w * sum_w) / sum_w2


def confidence_label(n_effective: float) -> str:
    """Bucket an effective sample size into a four-level confidence label.

    Thresholds (calibrated on production Chicago profiles):
        N_eff >= 150 -> "high"
        N_eff >= 60  -> "moderate"
        N_eff >= 30  -> "low"
        otherwise    -> "very low"

    Earlier thresholds (200 / 100 / 50) were implicitly calibrated as if
    ACS sample weights were uniform. In practice ``n_eff`` is computed on
    ``kernel_w × sample_w``, and ACS replicate weights within a matched
    neighborhood routinely have CV > 0.8 — a single high-weight respondent
    pulls Kish ``n_eff`` well below the kernel-only ``n_eff``. The new
    bands reflect the observed operating range for well-matched Chicago
    profiles (kernel-only n_eff ~111, production n_eff 60-90).

    Match quality — independent of ACS weight concentration — is tracked
    separately via ``MatchResult.kernel_n_effective``; this label is a
    composite signal that is intentionally conservative.

    This is internal metadata only — services decide whether to surface the
    signal in the UI. The label itself is never shown to the end user verbatim.
    """
    if n_effective >= 150.0:
        return "high"
    if n_effective >= 60.0:
        return "moderate"
    if n_effective >= 30.0:
        return "low"
    return "very low"
