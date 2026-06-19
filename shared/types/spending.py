"""Spending distribution, committed expense, and paycheck state types."""

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class SpendingDistribution:
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    engel_estimate: float
    feasibility_adjusted: float
    cohort_position: float
    is_structural: bool
    behavioral_gap: float = 0.0
    # Sample-weighted share of the matched cohort reporting strictly
    # positive spending on this category. Used by the allocator to
    # detect zero-inflated cats (hotel/airshp/jwlbg/...) where the
    # unconditional ``p90`` is dragged toward zero by the ~50%+ of
    # households reporting $0, clipping legitimate luxury anchors.
    nonzero_rate: float = 1.0
    # Weighted ``p90`` computed on the nonzero-only subset, CPI-scaled
    # by the same scalar as ``p10..p90``. The allocator uses this as
    # the upper-bound ceiling when ``nonzero_rate < 0.7``; for
    # saturated cats (eathome, elec, etc.) it equals ``p90``.
    conditional_p90: float = 0.0
    # Weighted p10 on the nonzero-only subset (the p10 among households that
    # actually spend on this category), scaled by the same CPI/spatial scalar as
    # ``p10..p90``. The soft-constraint optimizer uses it as the participation-aware
    # soft floor for high-participation categories whose unconditional p10 is a CEX
    # diary-recall-window zero (apparel, dining out, electronics): the realistic
    # minimum a participating household spends, vs the unconditional p10 = $0 that
    # would let compression drive the category to zero. 0.0 (unused) where the
    # cohort has no positive spenders or the stat isn't populated.
    conditional_p10: float = 0.0
    # Weighted ``p90`` on the nonzero AND high-earner subset of the cohort
    # (households with ``y_eq >= cohort median y_eq``), scaled by the same
    # CPI/spatial scalar as ``p10..p90``. The back-fill's high-income
    # discretionary ceiling stratification (2026-06-09, locked
    # HIGH-INCOME-DISCRETIONARY-CEILING-STRATIFICATION): the broad-cohort
    # ``conditional_p90`` under-represents high-earner discretionary spending
    # by 31-59% on the back-fill target cats, so for high-earner profiles the
    # cap reads this sub-distribution percentile instead (collapse-and-attribute
    # over the high-earner sub-population). Populated by ``match_household`` on
    # the plain-percentile and aggregate branches; 0.0 (unused → broad-cp90
    # fallback) for car-owner cats / vehreg / thin hi-earner subsets.
    conditional_p90_hi: float = 0.0
    # Sample×kernel-weighted MEAN of the cohort's category spend, scaled
    # by the same CPI/spatial scalar as ``p10..p90``. For right-skewed
    # categories the mean sits above ``p50`` (the participation/lumpiness
    # skew). Populated by ``match_household`` only for the plain-percentile
    # (non-car-owner, non-aggregate) categories — sufficient for the
    # smooth-category set the allocator anchors on the mean
    # (``SMOOTH_MEAN_ANCHOR_CATEGORIES``); 0.0 (unused) elsewhere.
    weighted_mean: float = 0.0
    # Sample×kernel-weighted TRIM95 mean (mean over values at/below the weighted
    # p95), scaled by the same scalar as ``p10..p90``. The outlier-robust central
    # tendency: where a few within-cohort big-spenders inflate ``weighted_mean``,
    # ``trimmed_mean`` sits below it and is the accurate anchor. The allocator
    # anchors on it for ``TRIMMED_MEAN_ANCHOR_CATEGORIES`` (lumpy anchor switch,
    # Build 2); 0.0 (unused) where not populated.
    trimmed_mean: float = 0.0
    # Residual back-fill (the reverse stage, Build 4 — models/optimizer/backfill.py).
    # ``feasibility_adjusted`` stays the MEASURED cohort-typical baseline (the
    # forward allocation, unchanged). ``backfill_inferred`` is the INFERRED
    # back-fill increment (≥ 0) redistributed into this slope-ceiling discretionary
    # category from an implausibly-large feasibility-slack residual; 0.0 for every
    # non-target category and whenever the back-fill does not fire. The displayed
    # value of the line is ``feasibility_adjusted + backfill_inferred``;
    # ``backfill_confidence`` marks which ("measured" vs "inferred-lifestyle").
    backfill_inferred: float = 0.0
    backfill_confidence: str = "measured"


@dataclass(frozen=True)
class CommittedExpense:
    name: str
    amount: float
    due_date: date


@dataclass(frozen=True)
class PaycheckState:
    gross_amount: float
    pay_date: date
    next_pay_date: date
    committed_expenses: list[CommittedExpense] = field(default_factory=list)
    discretionary_available: float = 0.0
    current_spend: float = 0.0
    pace_projection: float = 0.0
    buffer_balance: float = 0.0
