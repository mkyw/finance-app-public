"""Primary allocation pipeline — CPI-scaled cohort p50 as the anchor.

The allocator is the single source of truth for how dollars map to
categories. It runs a cheap deterministic primary path for the common
case and only falls through to the unified soft-constraint optimizer in
``models/optimizer/soft_constraint_optimizer.py`` when anchors genuinely
exceed ``d_variable``.

Steps (per category c):

    A.  anchor_c = p50_c * cpi_scalar_c
        When matching returned p50_c == 0 (rare cats), seed from
        ``engel_estimate_c * cpi_scalar_c`` so there's still a dollar
        signal to carry through the remaining steps. Engel is the
        fallback-only anchor, not the primary one.

    B.  Engel income-gap correction — optional. When the user's income
        diverges from the matched cohort median, QUAIDS elasticities
        shift the anchor:

            correction_c = 1 + eps_c * (ln y_u - ln y_c) / ln y_c

        Clamped to [0.7, 1.4] so a sparse cohort can't whipsaw the
        recommendation. Skipped (correction = 1.0) when
        ``cohort_median_income`` is not provided.

    C.  UX bias — multiply by 1.05 (default) so projections err
        slightly high. Users who come in under the plan carry forward
        money, which is the positive-experience direction.

    D.  Clamp to [p10, p90]. Keeps the allocation inside the cohort's
        empirical envelope.

    E.  Feasibility check. If ``sum(anchor) <= d_variable``, done.
        Otherwise, defer to ``soft_constraint_optimize`` — a unified
        continuous iso-elastic water-filling that compresses EVERY category
        at its own empirical expenditure elasticity
        (``x_i = clip(a_i·ν^(−ε_i), floor_i, a_i)``, ν found by bisection).
        Soft floors are the composed conditional-p10 / necessity floors; a
        genuine deficit (floors exceed budget) surfaces ``floor_infeasible``.
        No step/recurring split — that machinery retired Phase 8 (2026-06).

Output contract: same shape as ``FeasibilityResult`` from
``feasibility.solve_feasibility`` so the callers in
``apps/api/profiles/services.py`` stay stable.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import numpy as np

from models.engel.elasticity import quaids_elasticity
from models.matching.cpi_scaler import resolve_scalar
from models.optimizer.bounds import apply_tenure_constraints, compute_bounds
from models.optimizer.debt_service import estimate_annual_debt_service
from shared.constants.categories import (
    CATEGORY_CODES,
    LUXURY_CATEGORIES,
    SMOOTH_MEAN_ANCHOR_CATEGORIES,
    TRIMMED_MEAN_ANCHOR_CATEGORIES,
)
from shared.types import (
    FinancialZone,
    HouseholdProfile,
    SpendingDistribution,
)

_LOG = logging.getLogger(__name__)

_IMPROVEMENT_SURPLUS_FRACTION: float = 0.10
_UX_BIAS_DEFAULT: float = 1.05

# Process-level cache of the compression-parameter artifact (per-category empirical
# elasticities), keyed by artifacts path. The artifact is small; loading it per
# household across the synthetic population would be wasteful.
_COMPRESSION_PARAMS_CACHE: dict[str, dict] = {}


def _compression_params(artifacts_path: str | None) -> dict:
    key = artifacts_path or ""
    if key not in _COMPRESSION_PARAMS_CACHE:
        from models.optimizer.soft_constraint_optimizer import (  # noqa: PLC0415
            _DEFAULT_PARAMS_PATH,
            load_compression_parameters,
        )
        path = (
            os.path.join(artifacts_path, "compression_parameters.json")
            if artifacts_path else _DEFAULT_PARAMS_PATH
        )
        _COMPRESSION_PARAMS_CACHE[key] = load_compression_parameters(path)
    return _COMPRESSION_PARAMS_CACHE[key]

# Engel correction floor. Prevents excessive suppression for
# low-income users even in high-elasticity luxury cats.
_CORRECTION_FLOOR: float = 0.3

# Engel correction ceilings — asymmetric by category type.
# Necessities (ε < 1) shouldn't scale unboundedly with income; cap at
# 2.0 so a high earner's grocery budget isn't pushed into absurdity.
# Luxuries (ε > 1) get room to scale meaningfully for very-high-income
# users (a $250k household legitimately eats out ~7× a median one)
# without permitting unbounded polynomial blow-up on rare cats.
_CORRECTION_CEILING_NECESSITY: float = 2.0
_CORRECTION_CEILING_LUXURY: float = 8.0

# Nonzero-rate threshold for switching the upper-bound ceiling from
# unconditional p90 to conditional_p90 (p90 over households reporting
# >0 spending). Categories with 30%+ reporting $0 (hotel, airshp,
# jwlbg, chrty, ...) have unconditional p90 heavily compressed by the
# zero mass; conditional_p90 is the 90th percentile among actual
# spenders, which better represents the legitimate luxury ceiling.
_ZERO_INFLATED_THRESHOLD: float = 0.7


@dataclass(frozen=True)
class AllocationResult:
    """Mirror of FeasibilityResult kept here so the allocator can stand
    alone. ``feasibility.solve_feasibility`` converts between the two.

    ``d_variable_adjusted`` is ``d_variable - debt_service`` (floored at
    0). The allocator uses this number for its Phase 1 structural
    check, primary-path feasibility comparison, and zone classification
    — not the raw ``d_variable`` that came in. Callers display both so
    users can see what debt service costs them.
    """

    allocations: dict[str, float]
    financial_zone: FinancialZone
    structural_deficit: float
    feasibility_slack: float
    solver_status: str
    d_variable_adjusted: float
    debt_service: float
    # The soft-deficit gap: max(0, anchor_sum − adjusted_d) when the
    # soft-constraint optimizer met the budget (solver_status ==
    # "soft_constrained") — the amount cohort-typical spending overshoots
    # the debt-adjusted budget. 0.0 on every other path, including
    # "floor_infeasible" (that regime routes to the deferred deficit/
    # benefits handoff, not the debt-accumulation annotation — Q6 scope
    # guard in debt_accumulation_prediction_scoping.md). Consumed by
    # models/optimizer/debt_accumulation.py; purely informational here —
    # no allocation arithmetic reads it.
    compression_gap: float = 0.0


def _engel_correction(
    category: str,
    user_equivalized_income: float,
    cohort_median_income: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
    luxury_categories: frozenset[str] = LUXURY_CATEGORIES,
) -> float:
    """Multiplicative income-gap correction via Marshallian demand scaling.

    Formula:

        correction = (user_income / cohort_median) ** ε_c

    where ``ε_c`` is the QUAIDS income elasticity for category c evaluated
    at the user's equivalized income. This is the standard
    constant-elasticity scaling used in applied demand analysis:

      ε_c < 1  (necessity, e.g. food at home, utilities)  -> spending
        scales slower than income; at 3.6× income ratio, factor ~2.0×.
      ε_c ≈ 1  (unit elastic) -> spending scales proportionally.
      ε_c > 1  (luxury, e.g. dining out, recreation) -> spending scales
        faster than income; at 3.6× income ratio, factor up to ~6.9×.

    Scale convention (MVP): user side is equivalized (sqrt scale) per
    Engel fitting convention; cohort side is the raw weighted median
    of gross_income from ``MatchResult``. The asymmetry is accepted
    for MVP because the Gower distance matching already partially
    controls for household size — the matched cohort's household-size
    distribution is close to the user's, so raw/raw ≈ equiv/equiv.
    This matters for multi-person users where the reported ratio
    may differ from the "pure" equivalized-both-sides ratio.

    Asymmetric clamping:
      floor = 0.3 (prevents total suppression for low earners)
      ceiling = 2.0 for necessities, 8.0 for luxuries

    Returns 1.0 (no correction) when either income is non-positive,
    when the coefficient lookup fails, or when the result is non-finite.
    """
    if user_equivalized_income <= 0 or cohort_median_income <= 0:
        return 1.0
    income_ratio = user_equivalized_income / cohort_median_income
    if income_ratio <= 0:
        return 1.0
    try:
        elasticity = quaids_elasticity(
            category, user_equivalized_income, artifacts_path, coefficients_path
        )
    except KeyError:
        return 1.0
    try:
        correction = income_ratio ** elasticity
    except (ValueError, OverflowError):
        return 1.0
    if not math.isfinite(correction):
        return 1.0
    # Cap selection uses the static necessity/luxury audit in
    # shared/constants/categories.py rather than the runtime
    # ``elasticity > 1`` check. The fitted QUAIDS polynomial's class
    # label can flip across incomes (e.g. taxis is luxury at $250k but
    # necessity at $70k) and sometimes contradicts standard demand
    # conventions (e.g. eatout ε=0.66 at $250k, but behaves as a
    # luxury). The static table anchors the cap to domain convention.
    is_luxury = category in luxury_categories
    ceiling = (
        _CORRECTION_CEILING_LUXURY
        if is_luxury
        else _CORRECTION_CEILING_NECESSITY
    )
    return max(_CORRECTION_FLOOR, min(correction, ceiling))


def _primary_anchors(
    distributions: dict[str, SpendingDistribution],
    cpi_scalars: dict[str, float],
    user_equivalized_income: float,
    cohort_median_income: float | None,
    artifacts_path: str | None,
    ux_bias: float,
    category_codes: list[str],
    coefficients_path: str | None,
    luxury_categories: frozenset[str],
) -> np.ndarray:
    """Build the per-category dollar anchor in ``category_codes`` order.

    Implements steps A-D from the module docstring. Clamping to
    ``[p10, p90]`` happens here; feasibility checking is left to the
    caller.
    """
    n = len(category_codes)
    anchors = np.zeros(n, dtype=np.float64)

    for i, cat in enumerate(category_codes):
        dist = distributions[cat]

        # Step A — CPI scaling already happened in match_household, so
        # ``dist.p50`` / ``dist.weighted_mean`` are current-price.
        #
        # Smooth, high-participation cats (SMOOTH_MEAN_ANCHOR_CATEGORIES)
        # anchor on the cohort weighted MEAN, which sits above the median
        # for these right-skewed-by-participation categories — a deliberate
        # high-side anchor (users correct down). Categories that stay
        # outlier-distorted post-de-lumping (TRIMMED_MEAN_ANCHOR_CATEGORIES,
        # e.g. household_goods) anchor on the trim95 mean — the undistorted
        # central tendency, so the anchor itself isn't pulled high by a few
        # big-spenders (an accuracy fix, separate from compression's budget
        # fix). All other cats keep the p50 anchor; zero-median cats fall back
        # to the Engel dollar estimate (computed from disposable_income, so it
        # still needs the category-specific inflation bump).
        if cat in SMOOTH_MEAN_ANCHOR_CATEGORIES and dist.weighted_mean > 0.0:
            anchor = float(dist.weighted_mean)
        elif cat in TRIMMED_MEAN_ANCHOR_CATEGORIES and dist.trimmed_mean > 0.0:
            anchor = float(dist.trimmed_mean)
        elif dist.p50 > 0.0:
            anchor = float(dist.p50)
        else:
            scalar = resolve_scalar(cpi_scalars, cat)
            anchor = float(dist.engel_estimate) * scalar

        # Step B — Engel income-gap correction. Skipped when cohort
        # median is missing or when the coefficients can't be loaded
        # (e.g. tenure-zeroed categories that the engel module
        # silently drops).
        if (
            cohort_median_income is not None
            and cohort_median_income > 0
            and artifacts_path is not None
        ):
            correction = _engel_correction(
                cat,
                user_equivalized_income,
                cohort_median_income,
                artifacts_path,
                coefficients_path,
                luxury_categories,
            )
            anchor *= correction

        # Step C — UX bias.
        anchor *= ux_bias

        # Step D — clamp to [p10, ub]. For zero-inflated cats, ub is
        # the conditional p90 (90th percentile among positive-spenders)
        # rather than the unconditional p90 that is suppressed by the
        # mass at zero. For tenure-/balance-zeroed cats both bounds
        # are 0 and the anchor collapses to 0 regardless.
        lb_c = max(0.0, float(dist.p10))
        if (
            dist.nonzero_rate < _ZERO_INFLATED_THRESHOLD
            and dist.conditional_p90 > 0.0
        ):
            ub_c = float(dist.conditional_p90)
        else:
            ub_c = float(dist.p90)
        if ub_c <= 0.0 and anchor > 0.0:
            # Zero-spread cohort but positive anchor from Engel —
            # cap at the Engel estimate to avoid an unbounded blow-up.
            ub_c = float(dist.engel_estimate) if dist.engel_estimate > 0 else anchor
        if ub_c < lb_c:
            ub_c = lb_c
        anchors[i] = max(lb_c, min(anchor, ub_c))

    return anchors


def _category_elasticities(
    codes: list[str],
    user_equivalized_income: float,
    artifacts_path: str | None,
    coefficients_path: str | None,
) -> np.ndarray:
    """Fitted QUAIDS income elasticity ε per category, in ``codes`` order.

    Consumed by the compression mechanism (Phase A drop order, Phase B
    floor depth + barrier steepness). Categories whose coefficients can't
    be looked up — or when ``artifacts_path`` is unavailable — default to a
    neutral ε = 1.0; the static luxury/necessity gate still drives the
    protect-vs-cut decision, so a missing ε only flattens the floor-depth
    modulation, never the protection itself.
    """
    n = len(codes)
    eps = np.ones(n, dtype=np.float64)
    if artifacts_path is None or user_equivalized_income <= 0:
        return eps
    for i, cat in enumerate(codes):
        try:
            eps[i] = quaids_elasticity(
                cat, user_equivalized_income, artifacts_path, coefficients_path
            )
        except (KeyError, ValueError):
            eps[i] = 1.0
    return eps


def _classify_zone(
    lb: np.ndarray,
    d_variable: float,
    slack: float,
) -> FinancialZone:
    """Zone classification post-allocation.

    SURVIVAL fires on structural deficit (lb_sum > d_variable). The
    allocator's caller has already checked this and dispatched, but
    the function is used from ``solve_feasibility`` where the check
    may also land here; keeping the triple-case rule inside one
    function avoids drift.
    """
    if float(lb.sum()) > d_variable:
        return FinancialZone.SURVIVAL
    if d_variable > 0 and (slack / d_variable) >= _IMPROVEMENT_SURPLUS_FRACTION:
        return FinancialZone.IMPROVEMENT
    return FinancialZone.STABILITY


def compute_allocations(
    distributions: dict[str, SpendingDistribution],
    profile: HouseholdProfile,
    d_variable: float,
    cpi_scalars: dict[str, float],
    lambda_weights: dict[str, float],
    cohort_median_income: float | None = None,
    artifacts_path: str | None = None,
    ux_bias: float = _UX_BIAS_DEFAULT,
    predicted_othdbt: float = 0.0,
    predicted_stddbt: float = 0.0,
    category_codes: list[str] | None = None,
    luxury_categories: frozenset[str] | None = None,
    coefficients_path: str | None = None,
    transport_finance_mean: float = 0.0,
) -> AllocationResult:
    """Allocate ``d_variable`` (= full take_home) across the spending categories.

    By default this runs the 55-category disaggregated set. The aggregated
    path passes ``category_codes=AGGREGATED_CATEGORY_CODES``,
    ``luxury_categories=AGGREGATED_LUXURY_CATEGORIES``, and
    ``coefficients_path`` pointing at ``coefficients_aggregated.json``.

    Args:
        distributions: From ``engel.annotate_distributions`` —
            ``p10..p90`` + ``engel_estimate`` + ``is_structural``
            populated on every category.
        profile: Household profile — supplies ``tenure``,
            ``gross_income`` (user_income for Engel gap correction),
            and ``housing_cost`` (pinned to the housing line item via
            ``compute_bounds``).
        d_variable: Annual dollar budget before debt service. Equals
            full take-home; housing is NOT pre-subtracted because it
            is pinned inside the allocator.
        cpi_scalars: BLS category-level CPI scalars vs the cohort base
            year (2024). Keys are CEX category codes.
        lambda_weights: Legacy QUAIDS ``1/|eps-1|`` weights. No longer
            consumed — the two-phase compression mechanism derives its
            own elasticity-driven barriers. Accepted for call-site
            stability; safe to pass an empty dict.
        cohort_median_income: Weighted median of gross_income across
            the matched pool. When None or non-positive, the Engel
            income-gap correction is skipped (factor = 1.0).
        artifacts_path: Root artifacts directory (for QUAIDS
            coefficients). Optional — when None, Engel correction is
            skipped.
        ux_bias: Multiplier applied after Engel correction. Default
            1.05 (5% above the cohort-typical estimate).
        predicted_othdbt: Predicted credit card balance in dollars
            (typically ``match.distributions["othdbt"].p50``). Drives
            the credit-card minimum-payment subtraction from
            ``d_variable``. 0.0 skips CC debt service.
        predicted_stddbt: Predicted student loan balance in dollars
            (typically ``match.distributions["stddbt"].p50``). Drives
            the 10-year amortization subtraction. 0.0 skips SL debt
            service.

    Returns:
        ``AllocationResult`` — dollar-per-category mapping plus zone,
        structural deficit, feasibility slack, solver status,
        ``d_variable_adjusted`` (= d_variable − debt_service floored at
        0) and the absolute ``debt_service`` in dollars. The status is
        ``"primary"`` when no compression was needed,
        ``"structural_deficit"`` when Phase 1 tripped, or whatever
        status the compression helper returned otherwise.
    """
    codes = category_codes if category_codes is not None else CATEGORY_CODES
    lux = luxury_categories if luxury_categories is not None else LUXURY_CATEGORIES

    distributions = apply_tenure_constraints(distributions, profile.tenure)
    lb, ub = compute_bounds(distributions, profile, category_codes=codes)
    lb_sum = float(lb.sum())

    # User-reported debt (when present) overrides the cohort-predicted
    # balances component-wise; absent inputs fall back to the cohort
    # prior. ``profile`` is the single source of truth — services.py's
    # display computation reads the same fields so the two agree.
    service = estimate_annual_debt_service(
        credit_card_balance=predicted_othdbt,
        student_loan_balance=predicted_stddbt,
        cc_carried_balance=profile.cc_carried_balance,
        student_loan_payment=profile.student_loan_payment,
        auto_loan_payment=profile.auto_loan_payment,
        other_debt_payment=profile.other_debt_payment,
    )
    debt_service = service["total_debt_service"]
    adjusted_d = max(0.0, float(d_variable) - debt_service)

    # Phase 1 — structural deficit check against post-debt-service
    # budget. Now that housing is pinned into ``lb`` via
    # compute_bounds, this correctly fires when the user's reported
    # housing alone overshoots take-home minus debt service.
    if lb_sum > adjusted_d:
        allocations = {c: float(lb[i]) for i, c in enumerate(codes)}
        return AllocationResult(
            allocations=allocations,
            financial_zone=FinancialZone.SURVIVAL,
            structural_deficit=lb_sum - adjusted_d,
            feasibility_slack=0.0,
            solver_status="structural_deficit",
            d_variable_adjusted=adjusted_d,
            debt_service=debt_service,
        )

    # Primary path — CPI-scaled p50 anchor, Engel gap correction, UX
    # bias, clamp to bounds. Post-clamp to lb/ub so the pinned
    # housing line item wins over any cohort-based anchor.
    anchors = _primary_anchors(
        distributions=distributions,
        cpi_scalars=cpi_scalars,
        user_equivalized_income=float(profile.equivalized_income),
        cohort_median_income=cohort_median_income,
        artifacts_path=artifacts_path,
        ux_bias=ux_bias,
        category_codes=codes,
        coefficients_path=coefficients_path,
        luxury_categories=lux,
    )
    anchors = np.minimum(np.maximum(anchors, lb), ub)

    # Auto-loan transport double-count offset (aggregated path only).
    # When auto_loan_payment is reported, debt service captures the full
    # payment. The transportation aggregate's anchor still embeds the cohort's
    # vehint+vehprn (vehicle financing) estimate — that's the double-count.
    # Subtract min(cohort_finance_estimate, reported_annual) from the
    # transportation anchor, bounded at the finance sub-component so operating
    # costs (gas, insurance, maintenance) are never reduced.
    if (
        profile.auto_loan_payment > 0.0
        and transport_finance_mean > 0.0
        and "transportation" in codes
    ):
        t_idx = codes.index("transportation")
        auto_annual = float(profile.auto_loan_payment) * 12.0
        offset = min(float(transport_finance_mean), auto_annual)
        anchors[t_idx] = max(anchors[t_idx] - offset, lb[t_idx])

    anchor_sum = float(anchors.sum())

    if anchor_sum <= adjusted_d:
        slack = adjusted_d - anchor_sum
        zone = _classify_zone(lb, adjusted_d, slack)
        allocations = {c: float(anchors[i]) for i, c in enumerate(codes)}
        return AllocationResult(
            allocations=allocations,
            financial_zone=zone,
            structural_deficit=0.0,
            feasibility_slack=slack,
            solver_status="primary",
            d_variable_adjusted=adjusted_d,
            debt_service=debt_service,
        )

    # Compression path — the unified soft-constraint optimizer: a continuous
    # iso-elastic water-filling x_i(ν) = clip(a_i·ν^(−ε_i), floor_i, a_i) driven by
    # per-category EMPIRICAL expenditure elasticities + conditional-p10 soft floors.
    # Every category compresses continuously at its own rate — no step/recurring
    # split, no Phase A all-or-nothing drop (retired Phase 8, 2026-06-02). Imported
    # lazily to keep the import graph flat; no circular dependency.
    from models.optimizer.soft_constraint_optimizer import (  # noqa: PLC0415
        category_elasticity,
        compose_floors,
        soft_constraint_optimize,
    )

    is_luxury = np.array([c in lux for c in codes], dtype=bool)
    params = _compression_params(artifacts_path)
    emp_eps = np.array(
        [category_elasticity(params, c) for c in codes], dtype=np.float64
    )
    # The φ necessity-protection floor uses the fitted QUAIDS ε (preserving the
    # existing necessity protection); the compression RATE uses the empirical ε.
    quaids_eps = _category_elasticities(
        codes, float(profile.equivalized_income), artifacts_path, coefficients_path,
    )
    p10 = np.array([float(distributions[c].p10) for c in codes], dtype=np.float64)
    cond_p10 = np.array(
        [float(distributions[c].conditional_p10) for c in codes], dtype=np.float64
    )
    nz = np.array(
        [float(distributions[c].nonzero_rate) for c in codes], dtype=np.float64
    )
    floors = compose_floors(anchors, p10, cond_p10, nz, is_luxury, quaids_eps)
    pinned = (ub - lb) <= 1e-6
    s_vec, status = soft_constraint_optimize(
        anchors=anchors,
        floors=floors,
        elasticities=emp_eps,
        budget=adjusted_d,
        pinned=pinned,
    )
    total = float(s_vec.sum())
    slack = max(0.0, adjusted_d - total)
    if status == "floor_infeasible":
        # Genuine deficit: necessities are at their protective floors and
        # luxuries at zero, yet the budget still doesn't cover them. Surface
        # it as a clean handoff state (SURVIVAL + nonzero structural_deficit)
        # for the separate deficit-handling design — do NOT compress through
        # a floor. See compression.py / investigation doc §7.
        zone = FinancialZone.SURVIVAL
        structural_deficit = max(0.0, total - adjusted_d)
    else:
        zone = _classify_zone(lb, adjusted_d, slack)
        structural_deficit = 0.0
    allocations = {c: float(s_vec[i]) for i, c in enumerate(codes)}
    return AllocationResult(
        allocations=allocations,
        financial_zone=zone,
        structural_deficit=structural_deficit,
        feasibility_slack=slack,
        solver_status=status,
        d_variable_adjusted=adjusted_d,
        debt_service=debt_service,
        # Surfaced only on the soft-constrained path; floor_infeasible is
        # the deficit-handoff regime and deliberately reports 0.0 here.
        compression_gap=(
            max(0.0, anchor_sum - adjusted_d)
            if status == "soft_constrained"
            else 0.0
        ),
    )
