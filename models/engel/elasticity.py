"""QUAIDS expenditure elasticities and feasibility-optimizer weights.

For category c at equivalized income y_eq:

    epsilon_c = 1 + (beta_c + 2 * gamma_c * ln(y_eq)) / w_c

where w_c = engel_share(c, y_eq) (already clamped to >= 1e-6, so the
denominator is safe).

``lambda_c = 1 / |epsilon_c - 1|`` is the *legacy* compression weight.

**This weighting was inverted and is no longer consumed by compression.**
``1/|eps-1|`` is largest when ε is near 1 (unit-elastic), so it protected
near-unit-elasticity categories and compressed *both* deep necessities
(ε≈0.2) and strong luxuries (ε≈1.8) hardest — the opposite of the intent.
The replacement is the unified soft-constraint optimizer
(``models/optimizer/soft_constraint_optimizer.py``, 2026-06), which drives
compression off each category's *empirical* expenditure elasticity (the
iso-elastic exponent), with conditional-p10 soft floors — not the legacy
``1/|eps-1|`` weight nor the interim Phase-A/Phase-B step machinery (both
retired). ``lambda_weight`` / ``lambda_weights_all`` are retained only for
call-site / test stability and as a QUAIDS-elasticity diagnostic; nothing in
the allocation path reads their output anymore.
(The earlier docstring claimed "high lambda -> necessity"; that was false
in both directions — see the investigation doc §1.)

Edge cases:

- ``|epsilon_c - 1| < 0.01``: cap lambda at 100.0 to avoid a near-division
  by zero dominating the feasibility solve.
- ``mean_share < 0.001`` (rare CEX categories like ``stdint``, ``jwlbg``,
  ``stock``): return lambda = 1.0 without computing elasticity. These
  categories are zero-median in most households (Decision 2), and the
  fitted polynomial is dominated by noise at shares this small — a
  neutral weight is more honest than an elasticity that looks sharp but
  is spurious.
"""

from __future__ import annotations

import math

from models.engel.curves import _load_coefficients, engel_share

# Lambda saturation when elasticity sits near unit (|eps - 1| < 0.01).
_LAMBDA_CAP: float = 100.0

# "Neutral" lambda for categories too rare to fit meaningfully.
_RARE_LAMBDA: float = 1.0

# Thresholds used in lambda_weight().
_EPS_UNIT_TOLERANCE: float = 0.01
_RARE_MEAN_SHARE_THRESHOLD: float = 0.001

# Elasticity floor below which we assume the fitted QUAIDS polynomial
# is pathological (typically indicates the quadratic term dominates
# near an extrapolation edge, flipping the good to inferior). When ε
# falls below this we fall back to a literature default rather than
# returning a near-zero or negative value that would cause the Engel
# income-gap correction to suppress spending on genuine necessities.
_ELASTICITY_FIT_FLOOR: float = 0.05

# Default fallback when a category is not in LITERATURE_ELASTICITY.
_ELASTICITY_FALLBACK_DEFAULT: float = 0.3

# Conservative published income elasticities for the categories whose
# fitted QUAIDS polynomials most reliably misbehave at high y_eq. Used
# only when the fitted ε on clamped y_eval falls below
# ``_ELASTICITY_FIT_FLOOR``. Values below 1.0 (necessities) tend to
# suppress less dramatically when extrapolated; the CEI 2024 fit
# flips these to large negatives, hence the safety net.
LITERATURE_ELASTICITY: dict[str, float] = {
    "eathome": 0.35,  # Food at home — Engel's law canonical
    "elec":    0.20,  # Electricity — near-inelastic
    "ngas":    0.20,  # Natural gas — similar
    "watrsh":  0.15,  # Water/sewer/trash — very inelastic
    "intphn":  0.40,  # Internet + phone
    "gas":     0.40,  # Motor fuel
    "health":  0.35,  # Out-of-pocket health care
    "vehins":  0.30,  # Vehicle insurance
}


def quaids_elasticity(
    category: str,
    equivalized_income: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
) -> float:
    """Expenditure elasticity under QUAIDS.

        epsilon_c = 1 + (beta_c + 2 * gamma_c * ln(y_eq)) / w_c

    ``w_c`` is taken from :func:`engel_share` which is already clamped to
    at least ``1e-6``, so the division is safe.

    Clamps polynomial evaluation to ``[p05, p95]`` of the QUAIDS training
    range (stored as ``_training_range`` in ``coefficients.json``). If
    the resulting elasticity is below ``_ELASTICITY_FIT_FLOOR`` (0.05) —
    indicating pathological fit, not just extrapolation — falls back to
    the literature value for that category from
    :data:`LITERATURE_ELASTICITY` (or ``_ELASTICITY_FALLBACK_DEFAULT``
    if the category is not listed). The clamp handles extrapolation
    outside the training range; the fallback handles fit failures
    where the polynomial dives through zero inside the range.

    Args:
        category: Lowercase CEX category code.
        equivalized_income: y_eq in dollars (positive). Clamped to
            ``[training_range.p05, training_range.p95]`` before use.
        artifacts_path: Root artifacts directory.

    Returns:
        The elasticity. Convention:
          - ``epsilon < 1`` -> necessity (share falls as income rises).
          - ``epsilon > 1`` -> luxury.

    Raises:
        KeyError: if the category isn't in the coefficient file.
        ValueError: via :func:`engel_share` for non-positive income.
    """
    coeffs = _load_coefficients(artifacts_path, coefficients_path)
    if category not in coeffs:
        raise KeyError(f"unknown category: {category!r}")
    c = coeffs[category]

    # Clamp y_eval to the QUAIDS training-range [p05, p95] so we never
    # extrapolate a quadratic polynomial outside its fitted support.
    y_eval = _clamp_to_training_range(equivalized_income, coeffs)

    share = engel_share(category, y_eval, artifacts_path, coefficients_path)
    log_y = math.log(y_eval)
    eps = 1.0 + (c["beta"] + 2.0 * c["gamma"] * log_y) / share

    # Fit-failure fallback. If even inside the clamped range the
    # polynomial produces a very low or negative elasticity, the fit
    # is not reliable at this income — swap in a published value.
    if not math.isfinite(eps) or eps < _ELASTICITY_FIT_FLOOR:
        return LITERATURE_ELASTICITY.get(category, _ELASTICITY_FALLBACK_DEFAULT)
    return eps


def _clamp_to_training_range(y_eq: float, coeffs: dict) -> float:
    """Clamp ``y_eq`` to ``[p05, p95]`` of the QUAIDS training range.

    If the coefficient file lacks a ``_training_range`` entry (e.g.
    legacy cache), returns ``y_eq`` unchanged so behavior is backwards
    compatible.
    """
    tr = coeffs.get("_training_range")
    if not isinstance(tr, dict):
        return float(y_eq)
    try:
        lo = float(tr.get("p05"))
        hi = float(tr.get("p95"))
    except (TypeError, ValueError):
        return float(y_eq)
    if lo <= 0 or hi <= 0 or lo >= hi:
        return float(y_eq)
    return max(lo, min(float(y_eq), hi))


def lambda_weight(
    category: str,
    equivalized_income: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
) -> float:
    """Legacy QUAIDS weight ``lambda_c = 1 / |epsilon_c - 1|``.

    No longer consumed by the optimizer (the compression redesign replaced
    the ``|eps-1|`` weighting — see module docstring). Retained as a
    diagnostic / for call-site stability. Never returns ``inf`` or ``NaN``.
    """
    coeffs = _load_coefficients(artifacts_path, coefficients_path)
    if category not in coeffs:
        raise KeyError(f"unknown category: {category!r}")

    mean_share = float(coeffs[category].get("mean_share", 0.0))
    if mean_share < _RARE_MEAN_SHARE_THRESHOLD:
        return _RARE_LAMBDA

    eps = quaids_elasticity(
        category, equivalized_income, artifacts_path, coefficients_path
    )
    gap = abs(eps - 1.0)
    if gap < _EPS_UNIT_TOLERANCE:
        return _LAMBDA_CAP

    lam = 1.0 / gap
    # gap >= 0.01 is guarded above; math.isfinite guards against any
    # pathological coefficient combination that slipped through.
    if not math.isfinite(lam):
        return _LAMBDA_CAP
    return min(lam, _LAMBDA_CAP)


def lambda_weights_all(
    equivalized_income: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
) -> dict[str, float]:
    """All lambda weights keyed by category code (55 disaggregated, or the
    aggregated set when ``coefficients_path`` points at the aggregated file).

    Args:
        equivalized_income: y_eq in dollars.
        artifacts_path: Root artifacts directory.
        coefficients_path: Explicit coefficients file (aggregated path).

    Returns:
        Mapping from category code to lambda weight.
    """
    coeffs = _load_coefficients(artifacts_path, coefficients_path)
    return {
        cat: lambda_weight(
            cat, equivalized_income, artifacts_path, coefficients_path
        )
        for cat in coeffs
        if not cat.startswith("_")
    }
