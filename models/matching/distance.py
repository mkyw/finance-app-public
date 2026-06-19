"""Gower distance between a HouseholdProfile and a candidate population.

Three matching features: log(gross_income), household_size, age, with fixed
weights 0.5 / 0.3 / 0.2 (sum to 1.0). Each feature is normalized by the
range observed in the candidate population, not a global constant — this
makes the distance adaptive to whatever pool the matching algorithm
assembled (user's PUMA plus top-M similar PUMAs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.types import HouseholdProfile

FEATURE_WEIGHTS: dict[str, float] = {
    "log_income": 0.5,
    "household_size": 0.3,
    "age": 0.2,
}

# Small floor to keep log finite for zero/negative incomes in the pool.
# ACS has some hincp = 0 households; using a flat floor avoids -inf
# propagating through the normalized distance.
_LOG_INCOME_FLOOR: float = 1.0


def gower_distance(
    profile: HouseholdProfile,
    population: pd.DataFrame,
) -> np.ndarray:
    """Return per-row Gower distance from ``profile`` to every row of ``population``.

    Args:
        profile: The querying household.
        population: DataFrame with at least columns ``gross_income``,
            ``household_size``, ``age``. Any number of rows, including 1.

    Returns:
        (N,) np.ndarray of non-negative distances. Values lie in [0, 1] in
        the normal case; they can exceed 1 when the profile sits outside
        the population's range (normalization is by population range, not
        by a clamped global range).

    Raises:
        ValueError: if ``population`` is empty or missing a required column.
    """
    required = ("gross_income", "household_size", "age")
    missing = [c for c in required if c not in population.columns]
    if missing:
        raise ValueError(f"population missing required columns: {missing}")
    if len(population) == 0:
        raise ValueError("population is empty")

    pop_income = population["gross_income"].to_numpy(dtype=np.float64)
    pop_size = population["household_size"].to_numpy(dtype=np.float64)
    pop_age = population["age"].to_numpy(dtype=np.float64)

    # Log-transform incomes (both sides) with a floor so log stays finite.
    pop_log_income = np.log(np.maximum(pop_income, _LOG_INCOME_FLOOR))
    profile_log_income = float(
        np.log(max(float(profile.gross_income), _LOG_INCOME_FLOOR))
    )
    profile_size = float(profile.household_size)
    profile_age = float(profile.age)

    # Feature ranges across the candidate pool. Guard against zero range
    # (all rows identical on a feature) by substituting 1.0 — the weighted
    # contribution of that feature then becomes |profile - pop|, which is
    # bounded by the feature's own scale; it can't dominate because its
    # weight is capped at 0.5.
    def _range(arr: np.ndarray) -> float:
        r = float(arr.max() - arr.min())
        return r if r > 0 else 1.0

    r_log_income = _range(pop_log_income)
    r_size = _range(pop_size)
    r_age = _range(pop_age)

    d_log_income = np.abs(pop_log_income - profile_log_income) / r_log_income
    d_size = np.abs(pop_size - profile_size) / r_size
    d_age = np.abs(pop_age - profile_age) / r_age

    return (
        FEATURE_WEIGHTS["log_income"] * d_log_income
        + FEATURE_WEIGHTS["household_size"] * d_size
        + FEATURE_WEIGHTS["age"] * d_age
    )
