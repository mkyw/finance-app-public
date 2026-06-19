"""Paycheck pace projection + display + d_variable computation.

``project_pace`` returns a raw float (positive, zero, or negative) and
leaves display logic to callers. ``pace_display`` implements the
"never show a negative number" rule by inverting the sign and
switching phrasing.

``compute_d_variable`` is the integration hook referenced from
``models.optimizer.feasibility``: annual take-home minus any
caller-supplied committed flows. Housing is NOT deducted here — it is
pinned as a fixed allocation inside the optimizer via
``compute_bounds`` so it shows up in the spending plan as the
user-reported housing line item instead of silently eating into
d_variable.
"""

from __future__ import annotations

from models.tax.state import take_home
from shared.types import HouseholdProfile


def project_pace(
    discretionary_available: float,
    current_spend: float,
    days_elapsed: int,
    days_remaining: int,
) -> float:
    """Projected end-of-period carry-forward.

    Formula:
        pace = discretionary_available
               - (current_spend / days_elapsed) * days_remaining

    Edge cases:
        days_elapsed == 0: no spend yet, whole amount is on pace.
        days_remaining == 0: period over; pace = discretionary - spend.
        pace < 0: returned as-is; caller handles display.
    """
    if days_elapsed <= 0:
        return float(discretionary_available)
    if days_remaining <= 0:
        return float(discretionary_available - current_spend)

    daily_rate = current_spend / days_elapsed
    projected_future_spend = daily_rate * days_remaining
    return float(discretionary_available - projected_future_spend)


def pace_display(pace: float) -> str:
    """Render ``pace`` as user-facing copy; never shows a negative number.

    Positive pace -> "On pace to carry forward $X this paycheck."
    Non-positive -> "This paycheck is running about $X short."

    Rounds to nearest dollar. Callers should feed exactly one instance
    of this into the primary display — the language contract is
    enforced here.
    """
    amount = int(round(abs(pace)))
    if pace > 0:
        return f"On pace to carry forward ${amount} this paycheck."
    return f"This paycheck is running about ${amount} short."


def compute_d_variable(
    profile: HouseholdProfile,
    filing_status: str = "single",
    additional_committed: float = 0.0,
    num_dependents: int = 0,
    *,
    detail=None,
) -> float:
    """Annual variable-discretionary budget.

    Housing is NOT deducted here. The user's ``housing_cost`` is
    pinned inside the allocator as a fixed line item (``rntval`` for
    renters, ``mrtgip``/``mrtgpp`` 70/30 for owners), so it shows up
    in the spending plan instead of being silently subtracted.

    Args:
        profile: Household profile (uses ``gross_income``, ``puma_code``,
            ``place_fips``, ``county_fips``).
        filing_status: Passed to :func:`take_home`.
        additional_committed: Any extra annual committed outflows
            (auto loans, childcare, etc.) the caller has already
            gathered. Housing is NOT passed here.

    Returns:
        ``take_home - additional_committed``. Floored at 0 if the
        subtraction overshoots (rare but possible).
    """
    th = take_home(
        profile.gross_income,
        profile.puma_code,
        filing_status=filing_status,
        place_fips=profile.place_fips,
        county_fips=profile.county_fips,
        num_dependents=num_dependents,
        detail=detail,
    )
    return max(0.0, th - additional_committed)
