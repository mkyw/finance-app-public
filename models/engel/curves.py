"""QUAIDS-form Engel curves loaded from pipeline/artifacts/engel_coefficients/.

Budget share per category at equivalized income y_eq:

    w_c = alpha_c + beta_c * ln(y_eq) + gamma_c * (ln(y_eq))^2

Coefficient file shape (written by pipeline/export/export_coefficients.R):

    {
      "<cat>": {
        "alpha": float, "beta": float, "gamma": float,
        "mean_share": float, "is_necessity": bool,
        "r_squared": float, "is_residual": bool
      },
      ...
    }

Coefficients are cached at module scope keyed on artifacts_path so the
~2 kB JSON is only parsed once per process per path.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

from shared.types import SpendingDistribution

_LOG = logging.getLogger(__name__)

# Module-level cache, keyed by the resolved coefficients-file path so the
# disaggregated and aggregated coefficient files can both be held in one
# process (the Stage-2 comparison runner loads both). Each ~2 kB JSON is
# parsed once per distinct path per process.
_COEFFICIENTS_CACHE: dict[str, dict[str, Any]] = {}

# Shares are clamped into this range before being used anywhere — matters
# both for the elasticity denominator and for downstream dollar estimates.
_SHARE_MIN: float = 1e-6
_SHARE_MAX: float = 1.0

# Disposable-income sanity tolerance for engel_estimates_all.
_TOTAL_DEVIATION_TOLERANCE: float = 0.05


def _resolve_coefficients_path(
    artifacts_path: str, coefficients_path: str | None
) -> str:
    """Resolve the coefficients file path.

    ``coefficients_path`` (when given) is used verbatim — this is how the
    aggregated path points at ``coefficients_aggregated.json``. Otherwise we
    fall back to the canonical ``{artifacts_path}/engel_coefficients/
    coefficients.json`` (the disaggregated default).
    """
    if coefficients_path is not None:
        return coefficients_path
    return os.path.join(artifacts_path, "engel_coefficients", "coefficients.json")


def _load_coefficients(
    artifacts_path: str, coefficients_path: str | None = None
) -> dict[str, Any]:
    """Return the cached coefficients dict; load on first use of a path."""
    path = _resolve_coefficients_path(artifacts_path, coefficients_path)
    cached = _COEFFICIENTS_CACHE.get(path)
    if cached is None:
        with open(path) as f:
            cached = json.load(f)
        _COEFFICIENTS_CACHE[path] = cached
    return cached


def engel_share(
    category: str,
    equivalized_income: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
) -> float:
    """QUAIDS budget share for one category at a given equivalized income.

        w_c = alpha_c + beta_c * ln(y_eq) + gamma_c * (ln(y_eq))^2

    Output is clamped to ``[1e-6, 1.0]`` so the quadratic can't emit a
    negative or out-of-range share at income extremes and so the
    elasticity formula's ``w_c`` denominator never goes to zero.

    Args:
        category: Lowercase CEX category code (e.g. ``"rntval"``).
        equivalized_income: y_eq in dollars (positive). Callers should
            already have applied the sqrt equivalence scale.
        artifacts_path: Root artifacts directory (e.g. ``"pipeline/artifacts"``).

    Returns:
        Predicted budget share in ``[1e-6, 1.0]``.

    Raises:
        KeyError: if ``category`` is not in the coefficient file.
        ValueError: if ``equivalized_income`` is non-positive.
    """
    if equivalized_income <= 0:
        raise ValueError(
            f"equivalized_income must be positive, got {equivalized_income!r}"
        )
    coeffs = _load_coefficients(artifacts_path, coefficients_path)
    if category not in coeffs:
        raise KeyError(f"unknown category: {category!r}")

    c = coeffs[category]
    log_y = math.log(equivalized_income)
    raw = c["alpha"] + c["beta"] * log_y + c["gamma"] * log_y * log_y
    if raw < _SHARE_MIN:
        return _SHARE_MIN
    if raw > _SHARE_MAX:
        return _SHARE_MAX
    return float(raw)


def engel_estimates_all(
    equivalized_income: float,
    disposable_income: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
) -> dict[str, float]:
    """Dollar estimate per category = share * disposable_income.

    Logs a warning if the sum of dollar estimates deviates from
    ``disposable_income`` by more than 5%. Small deviations are
    expected because shares are clamped into ``[1e-6, 1.0]``; a large
    deviation typically means the polynomial has drifted far off at
    an extreme y_eq.

    Args:
        equivalized_income: y_eq used to evaluate every share.
        disposable_income: Dollar budget to allocate across categories.
        artifacts_path: Root artifacts directory.

    Returns:
        Mapping from category code to dollar estimate. Keys are the same
        55 codes as ``shared.constants.categories.CATEGORY_CODES``.
    """
    coeffs = _load_coefficients(artifacts_path, coefficients_path)
    estimates: dict[str, float] = {}
    for cat in coeffs:
        # Underscore-prefixed keys (``_training_range`` etc.) are
        # meta-entries, not categories; skip them.
        if cat.startswith("_"):
            continue
        share = engel_share(
            cat, equivalized_income, artifacts_path, coefficients_path
        )
        estimates[cat] = share * disposable_income

    total = sum(estimates.values())
    if disposable_income > 0:
        deviation = abs(total - disposable_income) / disposable_income
        if deviation > _TOTAL_DEVIATION_TOLERANCE:
            _LOG.warning(
                "engel_estimates_all sum=%.2f deviates %.1f%% from disposable_income=%.2f "
                "at y_eq=%.2f",
                total,
                deviation * 100,
                disposable_income,
                equivalized_income,
            )
    return estimates


def annotate_distributions(
    distributions: dict[str, SpendingDistribution],
    equivalized_income: float,
    disposable_income: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
) -> dict[str, SpendingDistribution]:
    """Fill ``engel_estimate`` and ``is_structural`` on each SpendingDistribution.

    Leaves ``p10..p90`` (from matching) and the feasibility / cohort /
    behavioral_gap fields (optimizer-owned) untouched.

    Args:
        distributions: Output of ``models.matching.algorithm.match_household``.
        equivalized_income: y_eq for the querying household.
        disposable_income: Dollar budget to allocate.
        artifacts_path: Root artifacts directory.

    Returns:
        New dict with the same keys as the input, each value a new
        ``SpendingDistribution`` with ``engel_estimate`` and
        ``is_structural`` populated. ``SpendingDistribution`` is frozen,
        so we return copies rather than mutating in place.
    """
    coeffs = _load_coefficients(artifacts_path, coefficients_path)
    estimates = engel_estimates_all(
        equivalized_income, disposable_income, artifacts_path, coefficients_path
    )

    out: dict[str, SpendingDistribution] = {}
    for cat, dist in distributions.items():
        if cat not in coeffs:
            # Category present in matching output but missing from
            # coefficient file — keep the distribution as-is.
            out[cat] = dist
            continue
        out[cat] = SpendingDistribution(
            p10=dist.p10,
            p25=dist.p25,
            p50=dist.p50,
            p75=dist.p75,
            p90=dist.p90,
            engel_estimate=estimates[cat],
            feasibility_adjusted=dist.feasibility_adjusted,
            cohort_position=dist.cohort_position,
            is_structural=bool(coeffs[cat]["is_necessity"]),
            behavioral_gap=dist.behavioral_gap,
            nonzero_rate=dist.nonzero_rate,
            conditional_p90=dist.conditional_p90,
            conditional_p10=dist.conditional_p10,
            conditional_p90_hi=dist.conditional_p90_hi,
            # Carry the cohort weighted mean + trim-mean through so the
            # allocator can anchor smooth/lumpy categories on them (would
            # reset to 0.0 otherwise).
            weighted_mean=dist.weighted_mean,
            trimmed_mean=dist.trimmed_mean,
        )
    return out
