"""Residual back-fill — the post-allocation REVERSE stage (Build 4).

The forward pipeline (match → value-scale → Engel → clamp, in
``allocator.compute_allocations``) lands a per-category cohort-typical
prediction and reports the leftover ``feasibility_slack`` as the implied
savings. On the income-slope-ceilinged discretionary categories that leftover
is implausibly large (a $90k single reads ~31% saving vs an income-realistic
~13%). This stage runs AFTER the forward allocation lands and **redistributes
the implausible excess** of that residual into the slope-ceiling discretionary
categories, conditioned on an income-realistic savings benchmark.

This is the income-aware refinement of INCOME-SLOPE-CEILING (accept the ceiling
forward; redistribute its residual consequence reverse) — explicitly NOT
slope-ceiling factor-cranking, which stays refused. It is a distinct layer from
the per-category value-layer factors (CE-PCE / utilities-climate / recomposition):
those scale anchors multiplicatively at read time (forward); this redistributes
the *residual* conditioned on an external savings benchmark (reverse), running
after composition and all value-layer factors are fully settled
(COMPOSITION-BEFORE-FACTOR satisfied by construction).

Mechanism (spec ``agent-artifacts/investigations/residual_backfill_design.md``):

    residual_rate = feasibility_slack / d_variable_adjusted
    s*            = s_star(y_eq)                       # income-realistic rate
    FIRE iff solver_status == "primary" AND residual_rate > s* + buffer
    pool          = (residual_rate − s*) · d_variable_adjusted · g   # full excess to s*, gated
    weight_c      = max(0, measured_c) · ε_c           over the target set
    inferred_c    = pool · weight_c / Σweight, capped at min(2·measured_c, cp90_eff_c)
    new_slack     = feasibility_slack − Σ inferred_c   # residual absorbs the rest

where ``cp90_eff_c`` is the broad-cohort ``conditional_p90`` — or, for
high-earner profiles (``y_eq >= cohort_median_y_eq``),
``max(conditional_p90, conditional_p90_hi)`` (the MONOTONE ceiling
stratification, 2026-06-09 + the Build-1.1 max() floor: high-earner caps only
raise or hold the broad ceiling, never tighten; see
``STRATIFY_THRESHOLD_QUANTILE`` below).

The ``primary`` guard is load-bearing: low-income / compression / floor-infeasible
profiles never reach the back-fill (the solver is already compressing — there is
no excess residual to redistribute; the deficit/benefits branch owns that regime).

All [CALIBRATE] knobs carry placeholder values here (the curve, breakpoints,
``w(age)``, ``K``, ``buffer``); they are pinned against the Q9 grid in Build-4
Phase 2. ``expected_savings`` is already pinned from the Phase-1 SCF extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from models.engel.elasticity import quaids_elasticity
from shared.constants.categories import (
    BACKFILL_TARGET_CATEGORIES,
    ELASTIC_SINK_CATEGORIES,
)
from shared.types import HouseholdProfile, SpendingDistribution

# --------------------------------------------------------------------------- #
# [CALIBRATE] knobs — placeholder values for Phase 1; pinned in Phase 2.       #
# --------------------------------------------------------------------------- #

# Realistic-savings curve s*(y_eq): continuous piecewise-linear interpolation
# between band MIDPOINTS (NOT a step function — band cliffs are the
# transportation-A1 defect). Knots are (y_eq, s*) at the working-age quintile
# band midpoints; values are the Dynan SCF Δwealth age-30–59 mapping
# (0/9/13/18/26/37/49 %). Breakpoints are [CALIBRATE] against BEA
# equivalized-decile boundaries (knob 2); the $ midpoints below are the
# raw-household-quintile first pass.
_SSTAR_KNOTS: tuple[tuple[float, float], ...] = (
    (16_500.0, 0.00),   # Q1 midpoint  ≤$33k → 0%
    (47_500.0, 0.09),   # Q2 midpoint  $33–62k → 9%
    (81_500.0, 0.13),   # Q3 midpoint  $62–101k → 13%  (motivating-profile pin)
    (133_500.0, 0.18),  # Q4 midpoint  $101–166k → 18%
    (230_500.0, 0.26),  # Q5 midpoint  $166–295k → 26%
    (447_500.0, 0.37),  # top-5% midpoint $295–600k → 37%
    (600_000.0, 0.49),  # top-1%  ≥$600k → 49% (flat above)
)

# Soft-signal gate (Q3). w(age): age-confidence weight, DECREASING in age
# (older accumulated savings is lifecycle-noisy → trust it less, fall back to
# the cohort prior). Linear w(25)=1.0 → w(65)=0.3, clamped. [CALIBRATE] knob 4.
_W_AGE_YOUNG = (25.0, 1.0)
_W_AGE_OLD = (65.0, 0.3)
# K: suppression scale on above-expected savings. [CALIBRATE] knob 5.
_SUPPRESS_K = 1.0

# Trigger deadband above s* (Q5). [CALIBRATE] knob 6 (5–10 pp).
_BUFFER = 0.06

# Per-category cap multiple on the measured anchor (Q5): measured + inferred
# may not exceed min(_CAP_ANCHOR_MULT · measured, conditional_p90).
_CAP_ANCHOR_MULT = 2.0

# High-income discretionary ceiling stratification (2026-06-09, locked
# HIGH-INCOME-DISCRETIONARY-CEILING-STRATIFICATION). The broad-cohort
# ``conditional_p90`` under-represents high-earner discretionary spending by
# 31-59% across the back-fill targets (robust cross-city and cross-income —
# the stratification investigation's Q1). For profiles with
# ``y_eq >= cohort median y_eq`` the population ceiling reads the
# high-earner-subset ``conditional_p90_hi`` instead (collapse-and-attribute
# over the high-earner sub-distribution, COHORT-AVERAGE-RESPECTS-MUTUAL-
# EXCLUSION). The ``2×anchor`` clamp is UNCHANGED — it is the per-category
# INITIAL deployment target keeping the back-fill's distribution sensible,
# NOT a guard against over-predicting total spending (over-prediction of
# elastic consumption is the SAFE error direction —
# OVER-PREDICTION-IS-THE-SAFE-DIRECTION-FOR-SPENDING, 2026-06-10, which
# retired the old "over-prediction guard" framing; the down-direction
# residual sweep goes past these caps into the elastic sinks).
# The threshold quantile is the cohort MEDIAN: the gap holds robustly down to
# the top-tercile boundary and attenuates below, so median captures the
# population where the under-calibration is real without over-extending
# (the calibration decision; top-tercile stays the conservative fallback if
# cross-profile testing ever surfaces a reason). The threshold VALUE arrives
# per-match as ``cohort_median_y_eq`` (computed by ``match_household`` at
# this quantile); 0.0 → stratification off (broad cp90 everywhere).
STRATIFY_THRESHOLD_QUANTILE = 0.50

# expected_savings(age, y_eq) — LOCKED from the Phase-1 SCF extraction
# (`SCF Summary Extract Public Data (2022).csv`, median LIQ, WGT-as-is over all
# 22,975 implicate rows). Young (25–40) median liquid by income tercile, plus
# age-band multipliers vs the young (≤35) base. Phase 2 expands the grid from
# the same file/method; the values here are the locked first build.
_YOUNG_LIQ_BY_INCOME: tuple[tuple[float, float], ...] = (
    # (y_eq upper bound, median LIQ) — terciles; thresholds [CALIBRATE], values locked
    (40_000.0, 900.0),     # low income tercile
    (80_000.0, 6_300.0),   # mid income tercile
    (float("inf"), 22_000.0),  # high income tercile
)
# Age-band median LIQ (the age marginal): ≤35 $5,300 / 35–50 $8,000 /
# 50–65 $7,400 / 65+ $11,500. Used as a multiplier vs the ≤35 base so the
# income-tercile shape (the dominant signal) is age-scaled.
_AGE_BAND_LIQ: tuple[tuple[float, float], ...] = (
    (35.0, 5_300.0),
    (50.0, 8_000.0),
    (65.0, 7_400.0),
    (float("inf"), 11_500.0),
)
_AGE_LIQ_BASE = 5_300.0  # the ≤35 band, the income-tercile reference age

# --------------------------------------------------------------------------- #
# Savings-signal-weighting knobs (Build, 2026-05-29). Personalize the savings  #
# line by blending the cohort prior s*(y_eq) with the user-reported balance    #
# signal (`profile.savings`), credibility-weighted, soft-asymmetric. See       #
# `agent-artifacts/investigations/savings_signal_weighting_scoping.md`.        #
# --------------------------------------------------------------------------- #

# Working-start age for the balance→implied-rate conversion (balance is divided
# by years-since-working-start). [CALIBRATE] Population-weighted midpoint: ~60%
# of Americans pursue some college (working-start ~20-23), ~40% don't (~18) →
# a mean near 20-21. Start at 21. A one-year shift is a ~3-4% relative change in
# the implied rate, so the prediction is not knife-edge on this knob. Forward
# note: could be personalized via age × income × location → likelihood-of-
# higher-education (a 25yo at $90k more likely finished a 4-year degree and
# started at ~22 than a 25yo at $35k who started at ~18) — deferred.
WORKING_START_AGE: float = 21.0

# Credibility weight of a reported BALANCE signal, before the w_age age-noise
# multiplier. [CALIBRATE] ~0.6: a balance is a stock-proxy-for-flow with
# substantial conversion noise, so it should be meaningful but not dominate the
# cohort prior. w_user = SIGNAL_STRENGTH_BALANCE × w_age(age).
SIGNAL_STRENGTH_BALANCE: float = 0.6

# Nominal weight for a DIRECT savings-line override. The override path
# (`apply_savings_override` / `reconcile_four_way(savings_override=)`) PINS the
# user's value directly (bounded by slack), bypassing the blend entirely — so
# this knob is documentation-only for API symmetry; it is NOT consumed by the
# blend. The override carries the user's authority by pinning, not by weight.
SIGNAL_STRENGTH_OVERRIDE: float = 1.0

# --------------------------------------------------------------------------- #
# Down-direction coverage amplifier (piece-3 build, 2026-06-09; locked          #
# SAVINGS-SIGNAL-DOWN-DIRECTION-STRENGTHENING). When the balance signal points  #
# DOWN (implied rate below cohort prior), the credibility weight is amplified   #
# by how few months of the COHORT flow the balance covers:                      #
#                                                                               #
#   w_down = min(W_DOWN_MAX, SIGNAL_STRENGTH_BALANCE × w_age                    #
#                              × max(1, COVERAGE_REF_MONTHS / cov₀))            #
#   cov₀  = balance / (s_cohort × d_var_adj / 12)                               #
#                                                                               #
# Up-side and no-signal paths are byte-identical (direction-gated). Grounding   #
# is the Q1.4 SCF cross-tab's SPLIT-ARM finding (scoping addendum, d06c5bb):    #
# the high-income low-liquid population is a measured ~50/50 mixture of         #
# self-reported savers and non-savers (P(SAVED)=0.535, $100–200K, n=1,152) —    #
# unit weight (floor = ½ cohort) is the mixture-grounded prior — while the      #
# unmodeled savings-elsewhere channel is essentially absent below $200K         #
# (median nrlf $0; P>$10K = 4.1%), which retires the rationale for the old      #
# conservative 62.5% floor. The original W_DOWN_MAX≈2.0 (⅓-cohort floor) was    #
# ruled out by the flow-arm falsification (would over-correct the saver half).  #
# --------------------------------------------------------------------------- #

# Reference coverage (months of cohort-rate flow the balance covers) below
# which the amplifier engages. [CALIBRATE] 3.0 = the self-refutation bar
# (buffer-stock floor, scoping Q3); coverage ≥ 3 months → no amplification,
# byte-identical to the pre-build blend.
COVERAGE_REF_MONTHS: float = 3.0

# Hard cap on the amplified down-direction weight. [CALIBRATE — pinned against
# the Q1.4 cross-tab addendum, commit d06c5bb] 1.0 = unit weight = the
# ½-of-cohort blend floor; do NOT raise without a saver-vs-non-saver
# discriminant beyond the balance (HIGH-INCOME-LOW-LIQUID-IS-A-MIXTURE-NOT-A-TYPE).
W_DOWN_MAX: float = 1.0


# --------------------------------------------------------------------------- #
# Result type                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackfillResult:
    """Output of the reverse stage.

    ``inferred`` is the per-category back-fill increment (≥ 0), only for fired
    target categories; absent/zero elsewhere. ``new_slack`` is the residual
    after redistribution (the part the caps refused stays here — genuine
    high-savers retain it). The audit fields are surfaced in the dev view.
    """

    inferred: dict[str, float] = field(default_factory=dict)
    new_slack: float = 0.0
    fired: bool = False
    s_star: float = 0.0
    residual_rate: float = 0.0
    g: float = 1.0
    pool: float = 0.0
    # The PERSONALIZED (blended) benchmark actually used for the trigger +
    # pool (spend-arm build, 2026-06-09). Equals ``s_star`` when there is no
    # balance signal or the signal is at/above cohort (up-capped); below it
    # when a low balance pulled the blend down (the low-side arm). 0.0 on
    # the disabled/non-primary no-op path (mirrors ``s_star``).
    s_star_personalized: float = 0.0


# --------------------------------------------------------------------------- #
# Curve / gate / benchmark helpers                                             #
# --------------------------------------------------------------------------- #


def s_star(y_eq: float) -> float:
    """Income-realistic after-tax saving rate at equivalized income ``y_eq``.

    Continuous piecewise-linear interpolation between the band-midpoint knots;
    flat (clamped) below the first and above the last knot.
    """
    if y_eq <= _SSTAR_KNOTS[0][0]:
        return _SSTAR_KNOTS[0][1]
    if y_eq >= _SSTAR_KNOTS[-1][0]:
        return _SSTAR_KNOTS[-1][1]
    for (x0, y0), (x1, y1) in zip(_SSTAR_KNOTS, _SSTAR_KNOTS[1:]):
        if x0 <= y_eq <= x1:
            t = (y_eq - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return _SSTAR_KNOTS[-1][1]  # unreachable; defensive


def w_age(age: float) -> float:
    """Age-confidence weight on the savings signal, decreasing in age."""
    (a0, w0), (a1, w1) = _W_AGE_YOUNG, _W_AGE_OLD
    if age <= a0:
        return w0
    if age >= a1:
        return w1
    t = (age - a0) / (a1 - a0)
    return w0 + t * (w1 - w0)


def _suppress(savings_ratio: float) -> float:
    """suppress(r) = clip((r − 1) / K, 0, 1) — only above-expected suppresses."""
    return max(0.0, min(1.0, (savings_ratio - 1.0) / _SUPPRESS_K))


def expected_savings(age: float, y_eq: float) -> float:
    """Income- and age-conditioned median liquid-savings benchmark (SCF, locked)."""
    base = next(v for ub, v in _YOUNG_LIQ_BY_INCOME if y_eq <= ub)
    age_liq = next(v for ub, v in _AGE_BAND_LIQ if age <= ub)
    return base * (age_liq / _AGE_LIQ_BASE)


def soft_signal_gate(profile: HouseholdProfile) -> float:
    """g = 1 − w(age)·suppress(savings / expected_savings).

    Default g = 1.0 (full standard back-fill) when savings is unknown/0 — the
    over-prediction philosophy: err high, let the user adjust down. Only a
    balance clearly above the age/income expectation pulls g down.
    """
    if profile.savings <= 0.0:
        return 1.0
    exp = expected_savings(float(profile.age), float(profile.equivalized_income))
    if exp <= 0.0:
        return 1.0
    ratio = profile.savings / exp
    return 1.0 - w_age(float(profile.age)) * _suppress(ratio)


def balance_implied_rate(
    profile: HouseholdProfile,
    d_var_adj_annual: float,
    *,
    s_cohort: float | None = None,
) -> tuple[float | None, float]:
    """Convert a reported savings BALANCE into an implied saving rate + credibility weight.

    Returns ``(s_user_implied, w_user)``:

      - ``s_user_implied`` = (balance ÷ years-since-working-start) ÷ d_var_adj —
        the balance treated as a stock-proxy for cumulative flow, annualized.
      - ``w_user`` = ``SIGNAL_STRENGTH_BALANCE × w_age(age)`` — the balance signal
        is trusted less as age rises (older balances carry decades of
        accumulation / returns / life-event noise that swamp the recent-flow
        signal). Reuses the locked ``w_age`` curve — one source of truth for
        age-noise reasoning (the same curve the back-fill's soft signal gate
        uses); do not invent a parallel curve.

    When ``s_cohort`` is supplied AND the signal points down
    (``s_user_implied < s_cohort``), the **coverage amplifier** strengthens
    ``w_user`` up to ``W_DOWN_MAX`` (see the knob block — the piece-3
    mixture-grounded build). ``s_cohort=None`` (the legacy signature) and the
    up-direction are byte-identical to the pre-amplifier behavior.

    Returns ``(None, 0.0)`` for the uninformative cases (no balance reported,
    compressed/zero d_var_adj, or age at/below ``WORKING_START_AGE`` — a
    fresh-grad balance is mostly gifts/family, treat as no signal). The
    ``(_, 0.0)`` weight makes the blend collapse to the cohort prior, preserving
    the regression-safety property.
    """
    age = float(profile.age)
    balance = float(profile.savings)
    if balance <= 0.0 or d_var_adj_annual <= 0.0 or age <= WORKING_START_AGE:
        return (None, 0.0)
    years_accumulating = max(1.0, age - WORKING_START_AGE)
    implied_annual_saving = balance / years_accumulating
    s_user_implied = implied_annual_saving / d_var_adj_annual
    w_user = SIGNAL_STRENGTH_BALANCE * w_age(age)
    # Direction-gated coverage amplifier: only when the signal points DOWN
    # and the caller supplied the cohort prior (the personalized-rate path).
    # cov₀ = months of the COHORT-rate flow the balance covers — a balance a
    # cohort-typical saver would rebuild in under COVERAGE_REF_MONTHS is a
    # stock-flow inconsistency; the weight scales with its severity, capped
    # at W_DOWN_MAX (the ½-cohort mixture floor). Coverage ≥ the reference →
    # amp = 1 → byte-identical to the base weight.
    if s_cohort is not None and s_cohort > 0.0 and s_user_implied < s_cohort:
        cov_months = balance / (s_cohort * d_var_adj_annual / 12.0)
        amp = max(1.0, COVERAGE_REF_MONTHS / max(cov_months, 1e-9))
        w_user = min(W_DOWN_MAX, w_user * amp)
    return (s_user_implied, w_user)


def blend_savings_rate(
    s_cohort: float,
    s_user_implied: float,
    w_user: float,
    *,
    upward_confirmed: bool,
) -> float:
    """Soft-asymmetric credibility-weighted blend of the cohort prior + user signal.

    Symmetric form: ``s_blend = (w_user·s_user_implied + s_cohort) / (w_user + 1)``
    (the cohort prior carries unit weight; the user signal carries ``w_user``).

    Soft asymmetry (the locked OVER-PREDICTION direction): the user signal pulls
    the rate **down freely** (lower saving → larger remainder → safe), but
    pushing **up** (asserting more saving than the cohort) defers to explicit
    confirmation — so unless ``upward_confirmed`` (an explicit Q7 raise above the
    cohort prior), the result is capped at ``s_cohort``.

    Regression-safety: ``w_user <= 0`` (no balance signal) returns exactly
    ``s_cohort`` — the blend is a no-op for signal-less users.
    """
    if w_user <= 0.0:
        return s_cohort
    s_blend = (w_user * s_user_implied + s_cohort) / (w_user + 1.0)
    if upward_confirmed:
        return s_blend
    return min(s_blend, s_cohort)


def personalized_savings_rate(
    profile: HouseholdProfile,
    d_var_adj_annual: float,
    *,
    savings_override: float | None = None,
) -> tuple[float, float, float | None, float, bool]:
    """The personalized (blended) savings rate — single source of truth.

    Composes ``s_star`` + ``balance_implied_rate`` + ``blend_savings_rate``
    into the one rate both consumers read (spend-arm build, 2026-06-09):
    ``assign_residual_savings`` (the savings/remainder label split) and
    ``compute_backfill`` (the discretionary-lift benchmark — the low-side
    arm). Keeping them on one function guarantees the two stages agree on
    the same personalized benchmark for a given profile.

    Returns ``(s_personalized, s_cohort, s_user_implied, w_user,
    upward_confirmed)``. No balance signal → ``s_personalized ==
    s_cohort`` exactly (the regression-safe collapse, load-bearing).
    """
    s_cohort = s_star(float(profile.equivalized_income))
    # Passing s_cohort engages the down-direction coverage amplifier (the
    # personalized path is the ONLY amplified consumer; the bare legacy
    # signature stays byte-identical for direct callers/tests).
    s_user_implied, w_user = balance_implied_rate(
        profile, d_var_adj_annual, s_cohort=s_cohort
    )
    upward_confirmed = (
        savings_override is not None
        and float(savings_override) > s_cohort * d_var_adj_annual
    )
    s_personalized = blend_savings_rate(
        s_cohort=s_cohort,
        s_user_implied=s_user_implied if s_user_implied is not None else 0.0,
        w_user=w_user,
        upward_confirmed=upward_confirmed,
    )
    return s_personalized, s_cohort, s_user_implied, w_user, upward_confirmed


# --------------------------------------------------------------------------- #
# The reverse stage                                                            #
# --------------------------------------------------------------------------- #


def compute_backfill(
    *,
    measured: dict[str, float],
    cohort: dict[str, SpendingDistribution],
    profile: HouseholdProfile,
    feasibility_slack: float,
    solver_status: str,
    d_variable_adjusted: float,
    artifacts_path: str,
    coefficients_path: str | None = None,
    target_categories: frozenset[str] = BACKFILL_TARGET_CATEGORIES,
    enabled: bool = True,
    cohort_median_y_eq: float = 0.0,
    framing_d_variable_adjusted: float | None = None,
) -> BackfillResult:
    """Run the reverse stage; return per-category inferred increments + new slack.

    Args:
        measured: forward per-category allocation (``feasibility.allocations`` =
            the cohort-typical baseline). The measured value of each atom.
        cohort: the matched (value-scaled) distributions — source of
            ``conditional_p90`` (+ the high-earner ``conditional_p90_hi``)
            for the per-category cap.
        cohort_median_y_eq: the matched pool's weighted median equivalized
            income (``MatchResult.cohort_median_y_eq``) — the stratification
            threshold. When the profile's y_eq is at/above it, the population
            ceiling reads ``conditional_p90_hi`` (high-earner subset) instead
            of the broad ``conditional_p90``; 0.0 (default) disables the
            stratification (broad cp90 everywhere — the pre-build behavior).
        profile: supplies ``equivalized_income`` (curve + benchmark axis),
            ``age`` and ``savings`` (soft-signal gate).
        feasibility_slack, solver_status, d_variable_adjusted: forward result.
        target_categories: the slope-ceiling discretionary set (Q4).
        enabled: hard off-switch (tests / A-B); when False, a no-op result.
        framing_d_variable_adjusted: the loop-invariant committed-baseline
            disposable income used for the PERSONALIZED-RATE benchmark only
            (PRETAX-FRAMING-LOOP-INVARIANT — see ``assign_residual_savings``).
            The two stages must read the same benchmark for a given profile, so
            both pin it here. The trigger/pool dollars still use the live
            ``d_variable_adjusted``. ``None`` (default) → the live value
            (byte-identical for every direct caller / test).

    A no-op (``fired=False``, ``inferred`` empty, ``new_slack == feasibility_slack``)
    whenever the trigger gate (Q5) does not fire — which includes every non-primary
    solver status, so compression/deficit profiles are immune.
    """
    noop = BackfillResult(
        inferred={}, new_slack=feasibility_slack, fired=False,
        s_star=0.0, residual_rate=0.0, g=1.0, pool=0.0,
    )
    if not enabled or solver_status != "primary" or d_variable_adjusted <= 0.0:
        return noop

    residual_rate = feasibility_slack / d_variable_adjusted
    y_eq = float(profile.equivalized_income)
    sstar = s_star(y_eq)
    # Low-side spend arm (2026-06-09): the benchmark is the PERSONALIZED
    # (blended) savings rate, not the raw cohort s*. A low reported balance
    # pulls the blend below cohort → lower benchmark → earlier trigger +
    # larger pool → discretionary lifts toward cohort ceilings (the unsaved
    # income surfaces as predicted spending, not phantom remainder). The
    # symmetric counterpart of the high-balance ``soft_signal_gate`` g
    # suppression below, which is PRESERVED unchanged: on the high side the
    # blend is up-capped at cohort (``blend_savings_rate``), so
    # ``s_pers == sstar`` and the trigger/pool are byte-identical — the
    # suppression there is g's job, not the benchmark's. No balance →
    # ``s_pers == sstar`` exactly (regression-safe collapse). See
    # ``savings_balance_signal_diagnostic.md`` (the missed-asymmetry finding)
    # + DECISIONS-forward-note SAVINGS-SIGNAL-SPEND-ARM-AT-CURRENT-WEIGHTING.
    framing_dva = (
        d_variable_adjusted
        if framing_d_variable_adjusted is None
        else framing_d_variable_adjusted
    )
    s_pers, _, _, _, _ = personalized_savings_rate(profile, framing_dva)

    # Trigger: clearly above the personalized income-realistic rate, past
    # the deadband.
    if residual_rate <= s_pers + _BUFFER:
        return BackfillResult(
            inferred={}, new_slack=feasibility_slack, fired=False,
            s_star=sstar, residual_rate=residual_rate, g=1.0, pool=0.0,
            s_star_personalized=s_pers,
        )

    g = soft_signal_gate(profile)
    # Redistribute the FULL excess down to the personalized benchmark (not
    # benchmark+buffer — the buffer is the trigger deadband, not the
    # target), gated by the soft signal.
    pool = (residual_rate - s_pers) * d_variable_adjusted * g
    if pool <= 0.0:
        return BackfillResult(
            inferred={}, new_slack=feasibility_slack, fired=False,
            s_star=sstar, residual_rate=residual_rate, g=g, pool=0.0,
            s_star_personalized=s_pers,
        )

    # Distribution weights weight_c = max(0, measured_c) · ε_c over the target
    # set, with per-category room = cap − measured.
    #
    # High-earner stratification gate: at/above the cohort-median y_eq
    # threshold the population ceiling is ``max(conditional_p90,
    # conditional_p90_hi)`` — MONOTONE (Build 1.1, 2026-06-09): the
    # stratification only ever RAISES (or holds) the broad ceiling, never
    # tightens it. The high-earner-subset p90 is the signal for raising;
    # where it lands below broad (kernel-cohort upper-half p90 is a smaller,
    # noisier sample — observed live on eatout/airshp, implausible as a real
    # "high earners spend less" effect; see DECISIONS.md
    # KERNEL-COHORT-P90-BELOW-BROAD-IS-NOISE-NOT-SIGNAL) the broad cohort
    # ceiling is the floor. The max() also subsumes the unpopulated-cp90_hi
    # fallback (max(broad, 0) == broad — thin hi subset, car-owner cats,
    # legacy dists). Below the threshold (or threshold absent) the cap logic
    # is byte-identical to the pre-stratification build.
    stratify = cohort_median_y_eq > 0.0 and y_eq >= cohort_median_y_eq
    weights: dict[str, float] = {}
    room: dict[str, float] = {}
    for cat in sorted(target_categories):
        m = max(0.0, float(measured.get(cat, 0.0)))
        if m <= 0.0:
            continue
        try:
            eps = quaids_elasticity(cat, y_eq, artifacts_path, coefficients_path)
        except KeyError:
            continue
        if not math.isfinite(eps) or eps <= 0.0:
            continue
        if cat in cohort:
            cp90_broad = float(cohort[cat].conditional_p90)
            cp90_hi = float(cohort[cat].conditional_p90_hi)
        else:
            cp90_broad = cp90_hi = 0.0
        cp90 = max(cp90_broad, cp90_hi) if stratify else cp90_broad
        cap = _CAP_ANCHOR_MULT * m if cp90 <= 0.0 else min(_CAP_ANCHOR_MULT * m, cp90)
        r = max(0.0, cap - m)
        if r <= 0.0:
            continue
        weights[cat] = m * eps
        room[cat] = r

    if not weights:
        return BackfillResult(
            inferred={}, new_slack=feasibility_slack, fired=False,
            s_star=sstar, residual_rate=residual_rate, g=g, pool=pool,
            s_star_personalized=s_pers,
        )

    # Capped proportional water-filling: distribute the pool proportional to
    # weight, clip to each category's room, redistribute the remainder among
    # uncapped categories until the pool (or all room) is exhausted.
    inferred: dict[str, float] = {c: 0.0 for c in weights}
    remaining = pool
    active = set(weights)
    while remaining > 1e-6 and active:
        wsum = sum(weights[c] for c in active)
        if wsum <= 0.0:
            break
        newly_capped = set()
        distributed_this_round = 0.0
        for c in list(active):
            share = remaining * weights[c] / wsum
            free = room[c] - inferred[c]
            add = min(share, free)
            inferred[c] += add
            distributed_this_round += add
            if inferred[c] >= room[c] - 1e-9:
                newly_capped.add(c)
        remaining -= distributed_this_round
        if distributed_this_round <= 1e-9:
            break
        active -= newly_capped

    total = sum(inferred.values())
    # Residual absorbs what the caps refused (pool − total stays as slack);
    # new_slack ≥ s*·d_var ≥ 0 because total ≤ pool ≤ feasibility_slack.
    new_slack = feasibility_slack - total
    inferred = {c: v for c, v in inferred.items() if v > 0.0}
    return BackfillResult(
        inferred=inferred,
        new_slack=new_slack,
        fired=bool(inferred),
        s_star=sstar,
        residual_rate=residual_rate,
        g=g,
        pool=pool,
        s_star_personalized=s_pers,
    )


# --------------------------------------------------------------------------- #
# User-adjustment fixity (Q7)                                                  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Residual savings/investment assignment (Build 6, 2026-05-28).                 #
#                                                                              #
# The post-back-fill residual is the user's uncommitted dollars after          #
# committed outflows (Build 5) and discretionary spending (Build 4 back-fill) #
# have been accounted for. **Most of it is plausibly saving** — but the model  #
# had no flow category to express that, so it sat in `feasibility_slack`      #
# unassigned, failing the tool's complete-dollar-accounting premise            #
# ([[NEUTRAL-FRAMING-REFINEMENT]] 2026-05-28: predict-not-prescribe; the       #
# tool's job is to assign dollars to likely destinations, with the user        #
# correcting — NOT to decline-to-predict).                                      #
#                                                                              #
# This stage **labels** the post-back-fill residual: the part that is          #
# plausibly saving (bounded by the same income-realistic s* curve the          #
# back-fill uses) is assigned to a `savings_investment` flow; anything above   #
# that is the genuine remainder ("small, lumpy, truly-uncertain"). The user    #
# adjusts the savings_investment line if their actual saving differs; the      #
# remainder absorbs the delta 1:1 ([[USER-ADJUSTMENT-AUTHORITY]]).              #
#                                                                              #
# This closes the [[STOCK-FLOW-CORRESPONDENCE]] gap for `stock` — the          #
# taxable-brokerage-contribution flow is captured as part of                   #
# `savings_investment` (case 2/3 hybrid: the *aggregate* savings flow is       #
# predicted, individual sub-flows like 'taxable brokerage' vs 'emergency       #
# fund' vs 'extra retirement' remain the user's discretionary allocation       #
# within that aggregate).                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResidualAssignment:
    """Labels the post-back-fill residual as likely-savings + genuine remainder.

    ``savings_investment`` is the part of the residual the cohort at this
    income level realistically saves (bounded above by ``s*(y_eq) × d_var_adj``
    and below by 0). ``genuine_remainder`` is what's left — the truly
    uncertain "could go anywhere" chunk that absorbs user corrections.

    Invariant: ``savings_investment + genuine_remainder == post_backfill_slack``.
    """

    savings_investment: float          # annual $
    genuine_remainder: float           # annual $
    realistic_savings_rate: float      # the rate actually used (cohort prior, or blended)
    realistic_savings_dollars: float   # rate × d_var_adj (the cap)
    source: str = ""
    # Drives state-dependent display copy (Phase 3/4). One of:
    #   "signal_confirmed_cohort"        — no balance signal, or it landed ~cohort
    #   "signal_pulled_down"             — balance implies lower saving → pulled down
    #   "signal_would_pull_up_deferred"  — balance implies higher saving → cap held
    #   "user_pinned"                    — user overrode the savings line (Q7)
    framing_state: str = "signal_confirmed_cohort"


def assign_residual_savings(
    *,
    d_variable_adjusted: float,
    post_backfill_slack: float,
    profile: HouseholdProfile,
    savings_override: float | None = None,
    framing_d_variable_adjusted: float | None = None,
) -> ResidualAssignment:
    """Label the post-back-fill residual as savings-investment + genuine remainder.

    Mechanism: the part of the residual that the household at this income
    realistically saves (= `s*_personalized × d_variable_adjusted`) is labeled
    `savings_investment` (the residual's likely destination); anything above
    that is `genuine_remainder`. When the residual is below realistic savings,
    all of it is labeled savings (and the remainder is 0). Predict-not-prescribe:
    the model assigns the likely destination; the user is told "estimated
    saving" and can adjust.

    PERSONALIZATION (2026-05-29): the rate is the soft-asymmetric blend of the
    cohort prior `s*(y_eq)` with the user's reported-balance signal
    (`profile.savings` → `balance_implied_rate` → `blend_savings_rate`). The
    signal pulls the rate **down freely**; pushing **up** defers to an explicit
    Q7 raise (``savings_override`` above the cohort prior dollars). With no
    balance signal the blend collapses to exactly `s*(y_eq)` — the no-signal
    path is byte-for-byte the prior behavior (regression-safety, load-bearing).

    Bounded high-side (per [[OVER-PREDICTION-EXTENDED-TO-COMMITTED-OUTFLOWS]]):
    bound by the (personalized) realistic rate, not the full residual, so the
    "remainder" label stays honest about which portion is cohort-atypical. The
    ``savings_override`` here only sets ``upward_confirmed`` (releasing the
    upward cap of the default prediction); the user's pin is actually applied by
    ``apply_savings_override`` (the override path carries authority by pinning).

    FRAMING DECOUPLING (2026-06-18, PRETAX-FRAMING-LOOP-INVARIANT): the savings
    *rate* — and hence the framing direction, which selects the down-sweep /
    up-waterfall / taxable regime — is determined from
    ``framing_d_variable_adjusted`` when supplied, NOT the live
    ``d_variable_adjusted``. The pre-tax fixed point (services.py Stage 3b)
    iterates the tax wedge, and the up-waterfall's own 401(k)/HSA pre-tax top-up
    raises ``d_variable_adjusted`` on the next pass — which lowers
    ``s_user_implied`` (= balance ÷ years ÷ d_var_adj). For a borderline saver
    that feedback can flip ``s_user_implied`` across ``s_cohort`` and so toggle
    the waterfall on/off between passes, a discontinuity that admits NO fixed
    point (take_home 2-cycles to non-convergence). Pinning the rate to the
    committed-baseline disposable income (the topup-free first pass) makes the
    regime a stable profile property — the waterfall's own tax break can no
    longer retroactively disqualify the waterfall. The dollar split still scales
    by the live ``d_variable_adjusted`` so the four-way closes exactly each pass.
    ``None`` (the default) collapses to the live value — byte-identical for
    every direct caller / test.
    """
    framing_dva = (
        d_variable_adjusted
        if framing_d_variable_adjusted is None
        else framing_d_variable_adjusted
    )
    if d_variable_adjusted <= 0 or post_backfill_slack <= 0:
        return ResidualAssignment(
            savings_investment=0.0,
            genuine_remainder=max(0.0, post_backfill_slack),
            realistic_savings_rate=0.0,
            realistic_savings_dollars=0.0,
            framing_state="signal_confirmed_cohort",
        )
    # Shared helper (spend-arm build, 2026-06-09): the same personalized
    # rate compute_backfill uses as its lift benchmark — the two stages
    # cannot disagree on the benchmark for a given profile.
    s_personalized, s_cohort, s_user_implied, w_user, upward_confirmed = (
        personalized_savings_rate(
            profile, framing_dva, savings_override=savings_override
        )
    )
    realistic = s_personalized * d_variable_adjusted
    savings = min(post_backfill_slack, realistic)
    remainder = post_backfill_slack - savings

    # Framing direction is determined purely by the signal vs the cohort prior:
    # the blend is monotonic (s_blend ⋛ s_cohort iff s_user_implied ⋛ s_cohort).
    if w_user <= 0.0 or s_user_implied is None:
        framing_state = "signal_confirmed_cohort"
        source = "Dynan SCF Δwealth working-age curve (= back-fill s_star), bounded by post-back-fill residual"
    elif s_user_implied < s_cohort:
        framing_state = "signal_pulled_down"
        # Named-cause + savings-elsewhere hedge (mandatory with the amplifier,
        # USER-NEUTRAL-FRAMING / USER-ADJUSTMENT-AUTHORITY): descriptive,
        # correction pathway explicit, never prescriptive.
        source = (
            "Based on your reported balance + Dynan SCF cohort curve (your "
            "balance implies lower saving) — adjust if you hold savings "
            "elsewhere (brokerage, etc.); bounded by residual"
        )
    elif s_user_implied > s_cohort and not upward_confirmed:
        framing_state = "signal_would_pull_up_deferred"
        source = "Dynan SCF cohort curve (your reported balance suggests this could be higher — adjust if so), bounded by residual"
    else:
        framing_state = "signal_confirmed_cohort"
        source = "Dynan SCF Δwealth working-age curve (= back-fill s_star), bounded by post-back-fill residual"

    return ResidualAssignment(
        savings_investment=savings,
        genuine_remainder=remainder,
        realistic_savings_rate=s_personalized,
        realistic_savings_dollars=realistic,
        source=source,
        framing_state=framing_state,
    )


# --------------------------------------------------------------------------- #
# Down-direction residual sweep (2026-06-10; locked                             #
# REMAINDER-ZERO-INVARIANT-DOWN-DIRECTION).                                     #
#                                                                               #
# For the LOW-savings-contradiction case (the balance signal pulled the         #
# personalized savings rate below cohort — the non-saver), the post-savings     #
# `genuine_remainder` is a systematic UNDER-prediction of spending: the user    #
# demonstrably isn't saving those dollars, the back-fill stopped at the         #
# cohort-realistic ceilings, and the overflow got parked as "remainder."        #
# Under-prediction of spending is the HARMFUL error direction (the user must    #
# adjust up — "actually I spend more than that"); over-prediction is the SAFE   #
# direction for elastic consumption (the user comfortably adjusts down) —       #
# OVER-PREDICTION-IS-THE-SAFE-DIRECTION-FOR-SPENDING, which retires the old     #
# "over-prediction guard" framing of the p90/2×anchor caps. Those caps remain   #
# the back-fill's INITIAL deployment targets (sensible per-category             #
# distribution), but they are NOT a hard floor on total spending: the residual  #
# sweeps past them into the high-participation elastic sinks                    #
# (ELASTIC_SINK_CATEGORIES — nzr >= ~0.92, so the swept levels predict          #
# plausible monthly spending for the individual, not a lumpy cohort-mean        #
# artifact), distributed by marginal income response (measured × ε, the same    #
# kernel form the back-fill uses — the wide-absorbent-set regime the            #
# elasticity kernel was built for). No ceiling blocks the sweep, so             #
# `genuine_remainder` is IDENTICALLY 0 by construction for this case and the    #
# four-way closes as `committed + debt + spending + savings == take_home`       #
# (COMPLETE-DOLLAR-ACCOUNTING strengthened to an invariant).                    #
#                                                                               #
# Direction-gated: fires ONLY on the low contradiction                          #
# (`framing_state == "signal_pulled_down"`, i.e. w_user > 0 and                 #
# s_user_implied < s_cohort) on a primary solve. No-contradiction and           #
# high-contradiction (high-balance) profiles are byte-identical — the           #
# up-direction (residual → tax-advantaged savings waterfall) and the            #
# no-contradiction default are the banked follow-ups                            #
# (REMAINDER-ZERO-UP-DIRECTION-PENDING /                                        #
# NO-CONTRADICTION-RESIDUAL-DEFAULT-PENDING).                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResidualSweep:
    """Output of the down-direction residual sweep.

    ``swept`` is the per-sink increment (> 0 only for sinks that absorbed;
    Σ swept == ``total`` == the pre-sweep ``genuine_remainder`` exactly when
    fired). ``trigger`` names the firing condition (audit/display).
    """

    swept: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    fired: bool = False
    trigger: str = ""


def sweep_remainder_to_sinks(
    *,
    remainder: float,
    measured: dict[str, float],
    inferred: dict[str, float],
    y_eq: float,
    framing_state: str,
    solver_status: str,
    artifacts_path: str,
    coefficients_path: str | None = None,
    sink_categories: frozenset[str] = ELASTIC_SINK_CATEGORIES,
    enabled: bool = True,
) -> ResidualSweep:
    """Sweep the would-be genuine remainder into the elastic-sink categories.

    Runs AFTER ``compute_backfill`` (the capped deployment toward the initial
    ceilings — unchanged) and AFTER ``assign_residual_savings`` (the savings
    label — unchanged; the blend sets the savings line first, the sweep takes
    only what would have been parked as remainder).

    Distribution: ``w_c = max(0, measured_c + inferred_c) × ε_c`` over the
    sink set — proportional to each category's marginal response to income
    (dx ∝ x·ε), so high-ε sinks absorb proportionally more but all
    participate. NO per-category cap — the sinks absorb everything
    (over-prediction is the safe direction); the largest-weight sink absorbs
    the floating-point dust so ``total == remainder`` exactly.

    Fallbacks (defensive; the sinks are high-participation so ``measured`` is
    essentially never all-zero): missing/non-positive ε → the category drops
    from the weight set; all weights zero → ε-only weights; still empty →
    not fired (the remainder stays — surfaced, never silently dropped).

    A no-op (``fired=False``) unless ALL of: enabled (aggregated path only),
    ``solver_status == "primary"``, ``framing_state == "signal_pulled_down"``
    (the low-savings-contradiction trigger — the ONLY direction this build
    handles), and ``remainder > 0``.
    """
    noop = ResidualSweep()
    if (
        not enabled
        or solver_status != "primary"
        or framing_state != "signal_pulled_down"
        or remainder <= 0.0
    ):
        return noop

    # Marginal-response weights over the sink set.
    weights: dict[str, float] = {}
    eps_by_cat: dict[str, float] = {}
    for cat in sorted(sink_categories):
        try:
            eps = quaids_elasticity(cat, y_eq, artifacts_path, coefficients_path)
        except KeyError:
            continue
        if not math.isfinite(eps) or eps <= 0.0:
            continue
        eps_by_cat[cat] = eps
        level = max(0.0, float(measured.get(cat, 0.0))) + max(
            0.0, float(inferred.get(cat, 0.0))
        )
        if level > 0.0:
            weights[cat] = level * eps
    if not weights:
        weights = dict(eps_by_cat)  # ε-only fallback (all-zero measured levels)
    if not weights:
        return noop  # no usable sink — keep the remainder honest, don't drop it

    wsum = sum(weights.values())
    swept = {c: remainder * w / wsum for c, w in weights.items()}
    # Exact closure: the largest-weight sink absorbs the floating-point dust.
    dust = remainder - sum(swept.values())
    largest = max(weights, key=lambda c: weights[c])
    swept[largest] += dust
    return ResidualSweep(
        swept={c: v for c, v in swept.items() if v > 0.0},
        total=remainder,
        fired=True,
        trigger="low_savings_contradiction",
    )


def apply_remainder_sweep(
    base: ResidualAssignment, sweep: ResidualSweep
) -> ResidualAssignment:
    """Fold a fired sweep into the residual assignment: remainder ≡ 0.

    The savings line is untouched (the blend set it; the sweep only re-homes
    the would-be remainder into predicted spending). ``framing_state`` stays
    ``signal_pulled_down`` — the savings line keeps its named-cause framing.
    A non-fired sweep returns ``base`` unchanged.
    """
    if not sweep.fired:
        return base
    return ResidualAssignment(
        savings_investment=base.savings_investment,
        genuine_remainder=0.0,
        realistic_savings_rate=base.realistic_savings_rate,
        realistic_savings_dollars=base.realistic_savings_dollars,
        source=base.source,
        framing_state=base.framing_state,
    )


def reconcile_four_way(
    *,
    take_home: float,
    committed_total: float,
    debt_service: float,
    spending_total: float,
    s_star_rate: float,
    savings_override: float | None = None,
) -> tuple[float, float, float]:
    """The canonical shift-ready reconciler — returns ``(d_var_adj, savings, remainder)``.

    Given the post-override state of the four-way accounting (take_home,
    committed, debt_service, spending_total, optional savings_override) plus
    the income-realistic s* rate, return the three derived quantities.
    **The remainder is the single balancing term** — any combination of user
    overrides (one category, many categories simultaneously, savings line)
    leaves the remainder absorbing the net effect.

    Shift-ready property (Q7 + DYNAMIC-REALLOCATION, 2026-05-28):

      - *Adjust* (one category edit, total changes): the per-category change
        flows through to a changed ``spending_total`` → slack changes →
        remainder absorbs the difference (savings unchanged if its cap binds).

      - *Shift* (two offsetting category edits, total conserved, e.g.
        eatout +$200 and groceries −$200): ``spending_total`` unchanged
        → slack, savings, and remainder all unchanged. **Supported for free**
        by treating the remainder as the balance: any set of overrides whose
        deltas net to zero leave the balance untouched.

      - *Savings override*: pins savings; the remainder absorbs the change.

    The function is **label-neutral over categories** — it doesn't know which
    are "discretionary" vs "essential" (no virtue tiers; the planner is free
    to reallocate between any categories). Compression's drop-order
    (episodic-first) does NOT leak here — that's a separate solver concern.

    Args:
        take_home: gross annual take-home before any subtractions.
        committed_total: ``committed_outflows.total`` post any user overrides.
        debt_service: derived debt-service flow (CC + student-loan amortization).
        spending_total: Σ over all categories of ``measured + inferred`` post
            any user overrides (the back-fill targets and the
            forward-prediction categories — *all* of them, label-neutral).
        s_star_rate: ``s_star(y_eq)`` from the back-fill's curve.
        savings_override: user-pinned savings value, or None to use the
            cohort-prior baseline (s*-bounded).

    Returns:
        ``(d_var_adj, savings, remainder)`` all annual dollars.

    Invariants:
        committed_total + debt_service + spending_total + savings + remainder
            == take_home
        savings ∈ [0, min(slack, realistic_cap or override)]
        remainder ≥ 0
    """
    d_var_adj = max(0.0, float(take_home) - float(committed_total) - float(debt_service))
    slack = max(0.0, d_var_adj - float(spending_total))
    if savings_override is not None:
        savings = max(0.0, min(slack, float(savings_override)))
    else:
        realistic = float(s_star_rate) * d_var_adj
        savings = min(slack, realistic)
    remainder = slack - savings
    return d_var_adj, savings, remainder


def apply_savings_override(
    base: ResidualAssignment,
    user_savings: float,
) -> ResidualAssignment:
    """Q7 fixity for the savings_investment line.

    User overrides their actual savings amount; the genuine_remainder absorbs the
    delta 1:1. Adjusting savings *down* (you actually save less) → remainder
    *grows* (ego-win). Adjusting *up* → remainder shrinks. Honors
    [[USER-ADJUSTMENT-AUTHORITY]]: user's value is authoritative, residual
    absorbs, no sympathetic shift on discretionary or committed components.

    The total (savings + remainder) is preserved because the savings_override
    isn't subtracted from take-home — it's a label on the existing residual.
    """
    user_v = max(0.0, float(user_savings))
    total = base.savings_investment + base.genuine_remainder
    new_savings = min(user_v, total)
    new_remainder = total - new_savings
    return ResidualAssignment(
        savings_investment=new_savings,
        genuine_remainder=new_remainder,
        realistic_savings_rate=base.realistic_savings_rate,
        realistic_savings_dollars=base.realistic_savings_dollars,
        source=base.source + " (user-overridden savings_investment)",
        framing_state="user_pinned",
    )


def apply_category_overrides(
    measured: dict[str, float],
    inferred: dict[str, float],
    slack: float,
    overrides: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], float]:
    """Category-edit semantics: the user's value is authoritative; residual absorbs.

    A *category-level* edit (e.g. user sets entertainment to $180) pins that
    category to the user's value and lets the **residual absorb the difference
    1:1** — NO sympathetic shift on any other category, and the back-fill is NOT
    recomputed (Q7 fixity; the same pattern as the pinned housing line, locked
    #5, generalized to any user-edited category). The model does not try to be
    smarter than the user about the user's own spending.

    **Shift-ready property (Q7 + DYNAMIC-REALLOCATION, 2026-05-28):** ``overrides``
    accepts **arbitrary simultaneous values for any subset of categories**, with
    the residual as the single balancing term. Two consequences:

      - *Adjust* (one entry in ``overrides``, ``residual`` net-absorbs the delta).
      - *Shift* (two entries whose deltas sum to zero — e.g.
        ``{"eatout": prior+200, "groceries": prior-200}`` — ``residual``
        unchanged because Σ(deltas) = 0). **Supported for free** by the
        residual-as-balance semantics; no special-case "shift" path needed.

    See :func:`reconcile_four_way` for the canonical reconciler that handles
    the same property across all four buckets (committed / spending / savings /
    remainder) for the eventual dynamic planner.

    *Input* edits (income, location, family size, savings, car ownership,
    tenure, …) are a different path entirely — they change the conditioning, so
    they re-run the full forward pipeline + back-fill from scratch (just call
    ``run_profile_analysis`` again with the new inputs); they are NOT handled here.

    Returns the (effective per-category total, inferred, residual) after the
    overrides. For an overridden category the back-fill ``inferred`` is dropped
    to 0 (the user's number is now the whole value — measured-vs-inferred no
    longer applies to a user-set line); ``effective_total[c]`` is the override.
    """
    eff_measured = dict(measured)
    eff_inferred = dict(inferred)
    residual = slack
    for cat, user_value in overrides.items():
        prior_total = float(measured.get(cat, 0.0)) + float(inferred.get(cat, 0.0))
        delta = float(user_value) - prior_total
        residual -= delta                    # residual absorbs the difference 1:1
        eff_measured[cat] = float(user_value)  # user value is authoritative
        eff_inferred[cat] = 0.0                # no measured/inferred split on a user-set line
    effective_total = {
        c: eff_measured.get(c, 0.0) + eff_inferred.get(c, 0.0)
        for c in set(eff_measured) | set(eff_inferred)
    }
    return effective_total, eff_inferred, residual
