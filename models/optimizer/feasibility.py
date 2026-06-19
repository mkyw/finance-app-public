"""Feasibility entry point — a thin wrapper around ``allocator.compute_allocations``.

The redesign moved the primary allocation logic into
``models/optimizer/allocator.py`` and the compression logic into
``models/optimizer/soft_constraint_optimizer.py``. ``solve_feasibility``
keeps its existing call-site contract (same positional args) so consumers
don't have to change, but delegates the arithmetic to the allocator.

Compression (when ``sum(primary_anchors) > d_variable``) is the unified
soft-constraint optimizer: a continuous iso-elastic water-filling
``x_i = clip(a_i·ν^(−ε_i), floor_i, a_i)`` that compresses every category
at its own empirical expenditure elasticity, ν found by 1-D bisection on
the budget. Floors are the composed conditional-p10 / necessity floors. No
step/recurring split — that two-phase machinery retired Phase 8 (2026-06).
When compression doesn't run, the solve is deterministic and microsecond-fast.

The previous CVXPY/OSQP quadratic program and the ``lambda = 1/|eps-1|``
weighting it consumed were removed in 2026-05; the two-phase Phase-A/Phase-B
mechanism that briefly replaced it was itself retired in 2026-06 (locked
DECISIONS.md LOCKED-DROP-ORDER-RETIRES / STEP-VS-RECURRING-DISTINCTION-RETIRES).

Solver status values in ``FeasibilityResult``:

    "primary"              — anchors fit within d_variable, no compression.
    "structural_deficit"   — Phase 1 tripped (lb_sum > d_variable).
    "soft_constrained"     — the soft-constraint optimizer met the budget
                             (renamed from "compressed", Phase 8 2026-06).
    "floor_infeasible"     — genuine deficit: composed floors still exceed
                             the budget (clean handoff for the deficit design).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from models.optimizer.allocator import compute_allocations
from shared.constants.categories import CATEGORY_CODES
from shared.types import (
    FinancialZone,
    HouseholdProfile,
    SpendingDistribution,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeasibilityResult:
    allocations: dict[str, float]
    financial_zone: FinancialZone
    structural_deficit: float
    feasibility_slack: float
    solver_status: str
    d_variable_adjusted: float
    debt_service: float
    # max(0, anchor_sum − d_variable_adjusted) on "soft_constrained",
    # else 0.0 — see AllocationResult.compression_gap.
    compression_gap: float = 0.0


def solve_feasibility(
    distributions: dict[str, SpendingDistribution],
    profile: HouseholdProfile,
    lambda_weights: dict[str, float],
    d_variable: float,
    *,
    cpi_scalars: dict[str, float] | None = None,
    cohort_median_income: float | None = None,
    artifacts_path: str | None = None,
    predicted_othdbt: float = 0.0,
    predicted_stddbt: float = 0.0,
    category_codes: list[str] | None = None,
    luxury_categories: frozenset[str] | None = None,
    coefficients_path: str | None = None,
    transport_finance_mean: float = 0.0,
) -> FeasibilityResult:
    """Run the full allocation (primary + optional compression).

    Args:
        distributions: Output of ``engel.annotate_distributions``.
        profile: Household profile — supplies tenure, gross_income
            (Engel correction numerator), and housing_cost (pinned
            into the housing line item by ``compute_bounds``).
        lambda_weights: Legacy ``engel.elasticity.lambda_weights_all``
            output. No longer consumed by the compression mechanism
            (which derives its own elasticity-driven barriers); accepted
            for call-site stability.
        d_variable: Dollar budget to allocate — full take-home;
            housing is pinned inside the allocator rather than
            pre-subtracted.
        cpi_scalars: Category-level CPI scalars vs cohort base year.
            When None, every scalar defaults to 1.0 (no temporal
            adjustment — acceptable for tests, not for production
            call sites).
        cohort_median_income: Weighted median of gross_income across
            the matched pool (from ``MatchResult``). When None, the
            Engel correction is skipped.
        artifacts_path: Root artifacts directory — needed for the
            QUAIDS elasticity lookup used by the correction and by the
            compression barriers.
        predicted_othdbt: Predicted credit-card balance in dollars
            (typically cohort p50). Drives CC minimum-payment
            subtraction from d_variable. 0.0 means no CC debt service.
        predicted_stddbt: Predicted student loan balance in dollars.
            Drives the 10-year amortization subtraction. 0.0 means no
            SL debt service.
    """
    codes = category_codes if category_codes is not None else CATEGORY_CODES
    scalars = cpi_scalars if cpi_scalars is not None else {c: 1.0 for c in codes}

    result = compute_allocations(
        distributions=distributions,
        profile=profile,
        d_variable=d_variable,
        cpi_scalars=scalars,
        lambda_weights=lambda_weights,
        cohort_median_income=cohort_median_income,
        artifacts_path=artifacts_path,
        predicted_othdbt=predicted_othdbt,
        predicted_stddbt=predicted_stddbt,
        category_codes=category_codes,
        luxury_categories=luxury_categories,
        coefficients_path=coefficients_path,
        transport_finance_mean=transport_finance_mean,
    )
    return FeasibilityResult(
        allocations=result.allocations,
        financial_zone=result.financial_zone,
        structural_deficit=result.structural_deficit,
        feasibility_slack=result.feasibility_slack,
        solver_status=result.solver_status,
        d_variable_adjusted=result.d_variable_adjusted,
        debt_service=result.debt_service,
        compression_gap=result.compression_gap,
    )
