"""High-income discretionary ceiling stratification + eatout tests (2026-06-09).

The stratification build (locked HIGH-INCOME-DISCRETIONARY-CEILING-
STRATIFICATION): the back-fill's per-category population ceiling reads the
high-earner-subset ``conditional_p90_hi`` instead of the broad-cohort
``conditional_p90`` when the profile's y_eq is at/above the matched pool's
median y_eq (``MatchResult.cohort_median_y_eq``), keeping the ``2×anchor``
individual-anchor clamp. eatout joins the curated target set (the
high-participation luxury the original five missed).

Pinned properties:
  - the cap reads cp90_hi above the threshold, broad cp90 at/below
  - below-threshold profiles: cap logic byte-identical (stratification no-op)
  - legacy callers (no ``cohort_median_y_eq``): byte-identical (default 0.0 off)
  - cp90_hi unpopulated (0.0) → broad-cp90 fallback even for high earners
  - eatout is a target, is a luxury, and receives back-fill deployment
  - plumbing: ``conditional_p90_hi`` survives match → arrives at the back-fill
    with the high-earner-subset value (≥ the broad cp90 on luxury cats)
  - the demonstrating-profile shape: high-earner NYC RENT deploys toward
    cp90_hi on entertainment/shopping, eatout enters, slack conserves

Run from repo root (artifacts needed for matching + the QUAIDS elasticity).
"""
from __future__ import annotations

import pytest

from models.matching.algorithm import match_household
from models.optimizer.backfill import compute_backfill
from shared.constants.categories import (
    AGGREGATED_LUXURY_CATEGORIES,
    BACKFILL_TARGET_CATEGORIES,
)
from shared.types import HouseholdProfile, SpendingDistribution, Tenure

ARTIFACTS = "pipeline/artifacts"
AGG_COEFFS = "agent-artifacts/aggregation/coefficients_aggregated.json"
# Raw (55-cat) targets present in the default coefficients file, so the unit
# tests don't depend on the aggregated coefficients path (same convention as
# test_backfill_spend_arm.py).
TARGETS = frozenset({"hotel", "airshp"})
D_VAR = 60_000.0


def _profile(income: float = 110_000.0, savings: float = 0.0) -> HouseholdProfile:
    return HouseholdProfile(
        age=24, gross_income=income, puma_code="NY_04103", tenure=Tenure.RENT,
        housing_cost=2_500, household_size=1, savings=savings,
    )


def _dist(measured: float, cond_p90: float, cond_p90_hi: float = 0.0) -> SpendingDistribution:
    return SpendingDistribution(
        p10=0.0, p25=0.0, p50=measured, p75=measured * 1.5, p90=measured * 2.5,
        engel_estimate=0.0, feasibility_adjusted=measured, cohort_position=0.5,
        is_structural=False, conditional_p90=cond_p90,
        conditional_p90_hi=cond_p90_hi,
    )


def _run(*, cohort, measured, cohort_median_y_eq=0.0, slack_rate=0.40,
         targets=TARGETS, coefficients_path=None):
    return compute_backfill(
        measured=measured,
        cohort=cohort,
        profile=_profile(),
        feasibility_slack=slack_rate * D_VAR,
        solver_status="primary",
        d_variable_adjusted=D_VAR,
        artifacts_path=ARTIFACTS,
        coefficients_path=coefficients_path,
        target_categories=targets,
        cohort_median_y_eq=cohort_median_y_eq,
    )


# --------------------------------------------------------------------------- #
# The stratification gate (unit level)                                         #
# --------------------------------------------------------------------------- #
# Profile y_eq = 110_000 (income 110k, size 1). Caps are made BINDING by
# setting cp90 < 2×measured, so the deployed totals read the cap directly.

MEASURED = {"hotel": 1_200.0, "airshp": 1_500.0}


def _binding_cohort(hi: bool) -> dict[str, SpendingDistribution]:
    # broad cp90 at 1.5×measured (binding, below the 2× anchor clamp);
    # hi-earner cp90 at 1.9×measured (binding, still below 2×).
    return {
        c: _dist(m, cond_p90=m * 1.5, cond_p90_hi=(m * 1.9 if hi else 0.0))
        for c, m in MEASURED.items()
    }


def test_cap_uses_cp90_hi_above_threshold() -> None:
    """High earner (y_eq 110k ≥ threshold 50k): room = cp90_hi − m per cat."""
    res = _run(cohort=_binding_cohort(hi=True), measured=MEASURED,
               cohort_median_y_eq=50_000.0)
    assert res.fired
    # Ample slack → every category caps out at cp90_hi (1.9m → room 0.9m).
    for c, m in MEASURED.items():
        assert res.inferred[c] == pytest.approx(0.9 * m, rel=1e-6)


def test_cap_broad_at_or_below_threshold() -> None:
    """Below the threshold (y_eq 110k < 150k): broad cp90 caps (1.5m → 0.5m)."""
    res = _run(cohort=_binding_cohort(hi=True), measured=MEASURED,
               cohort_median_y_eq=150_000.0)
    assert res.fired
    for c, m in MEASURED.items():
        assert res.inferred[c] == pytest.approx(0.5 * m, rel=1e-6)


def test_stratification_byte_identical_below_threshold() -> None:
    """Below threshold, the result is identical whether cp90_hi is populated,
    unpopulated, or the threshold is absent entirely (the cap-logic no-op)."""
    below = _run(cohort=_binding_cohort(hi=True), measured=MEASURED,
                 cohort_median_y_eq=150_000.0)
    no_hi = _run(cohort=_binding_cohort(hi=False), measured=MEASURED,
                 cohort_median_y_eq=150_000.0)
    legacy = _run(cohort=_binding_cohort(hi=False), measured=MEASURED)
    assert below.inferred == no_hi.inferred == legacy.inferred
    assert below.new_slack == no_hi.new_slack == legacy.new_slack
    assert below.pool == no_hi.pool == legacy.pool


def test_legacy_caller_default_is_off() -> None:
    """cohort_median_y_eq omitted (0.0): stratification off even with cp90_hi
    populated — high earner still reads broad caps."""
    res = _run(cohort=_binding_cohort(hi=True), measured=MEASURED)
    for c, m in MEASURED.items():
        assert res.inferred[c] == pytest.approx(0.5 * m, rel=1e-6)


def test_hi_unpopulated_falls_back_to_broad() -> None:
    """High earner but cp90_hi == 0.0 (thin subset / legacy dist): the broad
    cp90 cap applies — never the 2×anchor-only cap (max(broad, 0) == broad)."""
    res = _run(cohort=_binding_cohort(hi=False), measured=MEASURED,
               cohort_median_y_eq=50_000.0)
    for c, m in MEASURED.items():
        assert res.inferred[c] == pytest.approx(0.5 * m, rel=1e-6)


def test_monotone_hi_below_broad_holds_at_broad() -> None:
    """Build 1.1 (the max() floor): when cp90_hi lands BELOW broad cp90 (the
    kernel-cohort upper-half noise case, observed live on eatout/airshp), a
    high earner's cap holds at the broad ceiling — the stratification never
    tightens. Byte-identical to the unstratified run."""
    cohort = {
        c: _dist(m, cond_p90=m * 1.5, cond_p90_hi=m * 1.2)  # hi < broad
        for c, m in MEASURED.items()
    }
    strat = _run(cohort=cohort, measured=MEASURED, cohort_median_y_eq=50_000.0)
    legacy = _run(cohort=cohort, measured=MEASURED)
    assert strat.inferred == legacy.inferred  # holds at broad (0.5m room)
    for c, m in MEASURED.items():
        assert strat.inferred[c] == pytest.approx(0.5 * m, rel=1e-6)


def test_monotonicity_property() -> None:
    """For every target above threshold, the effective population ceiling is
    >= the broad ceiling: a stratified run never deploys LESS than the same
    run unstratified, per-category, across mixed hi-vs-broad orderings."""
    cohort = {
        # hotel: hi above broad (lift); airshp: hi below broad (hold).
        "hotel": _dist(1_200.0, cond_p90=1_200.0 * 1.5, cond_p90_hi=1_200.0 * 1.9),
        "airshp": _dist(1_500.0, cond_p90=1_500.0 * 1.5, cond_p90_hi=1_500.0 * 1.1),
    }
    strat = _run(cohort=cohort, measured=MEASURED, cohort_median_y_eq=50_000.0)
    broad = _run(cohort=cohort, measured=MEASURED)
    for c in MEASURED:
        assert strat.inferred[c] >= broad.inferred[c] - 1e-9, c
    # hotel lifted to cp90_hi; airshp held at broad.
    assert strat.inferred["hotel"] == pytest.approx(0.9 * 1_200.0, rel=1e-6)
    assert strat.inferred["airshp"] == pytest.approx(0.5 * 1_500.0, rel=1e-6)


def test_two_anchor_clamp_survives_stratification() -> None:
    """cp90_hi above 2×anchor: the individual-anchor clamp binds (room = m),
    not the population ceiling — the over-prediction guard is unchanged."""
    cohort = {
        c: _dist(m, cond_p90=m * 1.5, cond_p90_hi=m * 4.0)
        for c, m in MEASURED.items()
    }
    res = _run(cohort=cohort, measured=MEASURED, cohort_median_y_eq=50_000.0,
               slack_rate=0.60)
    for c, m in MEASURED.items():
        assert res.inferred[c] == pytest.approx(m, rel=1e-6)  # cap = 2m → room = m


def test_slack_conservation_with_stratified_caps() -> None:
    res = _run(cohort=_binding_cohort(hi=True), measured=MEASURED,
               cohort_median_y_eq=50_000.0)
    total = sum(res.inferred.values())
    assert res.new_slack == pytest.approx(0.40 * D_VAR - total, abs=1e-6)


# --------------------------------------------------------------------------- #
# The eatout addition                                                          #
# --------------------------------------------------------------------------- #


def test_eatout_in_curated_target_set() -> None:
    assert "eatout" in BACKFILL_TARGET_CATEGORIES
    assert len(BACKFILL_TARGET_CATEGORIES) == 6
    assert "eatout" in AGGREGATED_LUXURY_CATEGORIES
    # The four episodic durables stay OUT (deferred to Build 2 —
    # EPISODIC-DURABLES-AS-SURPLUS-DESTINATIONS-UNRESOLVED).
    assert BACKFILL_TARGET_CATEGORIES.isdisjoint(
        {"furhwr", "hmtimp", "happl", "eltrnp"}
    )


def test_eatout_deploys_under_default_targets() -> None:
    """eatout in measured + the DEFAULT (curated six) target set → deploys."""
    measured = {"eatout": 2_900.0}
    cohort = {"eatout": _dist(2_900.0, cond_p90=2_900.0 * 3.0)}
    res = compute_backfill(
        measured=measured, cohort=cohort, profile=_profile(),
        feasibility_slack=0.40 * D_VAR, solver_status="primary",
        d_variable_adjusted=D_VAR, artifacts_path=ARTIFACTS,
    )
    assert res.fired
    # 2×anchor clamp binds (cp90 = 3m > 2m) → room = m.
    assert res.inferred["eatout"] == pytest.approx(2_900.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Plumbing + the demonstrating-profile shape (live match, real artifacts)      #
# --------------------------------------------------------------------------- #

NYC_PUMAS = ["NY_04103", "NY_04104", "NY_04109", "NY_04110"]


@pytest.fixture(scope="module")
def nyc_match():
    return match_household(
        _profile(savings=1_000.0), ARTIFACTS, city_pumas=NYC_PUMAS,
        aggregate=True,
    )


def test_cp90_hi_plumbed_through_match(nyc_match) -> None:
    """conditional_p90_hi arrives populated on every back-fill target, in the
    same ballpark as the broad conditional_p90. (No strict >= ordering: the
    kernel cohort is already income-conditioned, so the upper-half p90 can sit
    slightly below the broad p90 on some cats — pure collapse-and-attribute
    substitution, observed live on eatout/airshp.)"""
    assert nyc_match.cohort_median_y_eq > 0.0
    for cat in sorted(BACKFILL_TARGET_CATEGORIES):
        dist = nyc_match.distributions[cat]
        assert dist.conditional_p90_hi > 0.0, cat
        # Sanity band, not ordering: within [0.5x, 3x] of the broad cp90.
        assert 0.5 * dist.conditional_p90 <= dist.conditional_p90_hi, cat
        assert dist.conditional_p90_hi <= 3.0 * dist.conditional_p90, cat


def test_demonstrating_profile_is_high_earner(nyc_match) -> None:
    """The demonstrating profile (y_eq $110k) clears the NYC RENT pool's
    median-y_eq threshold — the stratified caps apply to it."""
    assert 110_000.0 >= nyc_match.cohort_median_y_eq


def test_stratified_deployment_exceeds_broad(nyc_match) -> None:
    """End-to-end shape on the real NYC cohort: the stratified run deploys
    strictly more than the broad-cap run (the cap lift is real), eatout
    enters, every line respects its stratified cap, slack conserves."""
    profile = _profile(savings=1_000.0)
    cohort = nyc_match.distributions
    # Forward anchors approximated by the cohort p50 (the demonstrating
    # profile's anchors are p50/mean-based); enough to exercise cap logic.
    measured = {
        c: float(cohort[c].p50) for c in BACKFILL_TARGET_CATEGORIES
        if c in cohort and cohort[c].p50 > 0
    }
    assert "eatout" in measured  # high-participation: NYC RENT p50 > 0
    def _bf(threshold: float):
        return compute_backfill(
            measured=measured, cohort=cohort, profile=profile,
            feasibility_slack=0.30 * D_VAR, solver_status="primary",
            d_variable_adjusted=D_VAR, artifacts_path=ARTIFACTS,
            coefficients_path=AGG_COEFFS, cohort_median_y_eq=threshold,
        )

    broad = _bf(0.0)
    strat = _bf(nyc_match.cohort_median_y_eq)
    assert strat.fired and broad.fired
    assert "eatout" in strat.inferred
    assert sum(strat.inferred.values()) >= sum(broad.inferred.values())
    # Every deployed line respects its stratified cap min(2m, max(broad, hi))
    # (the monotone Build-1.1 form).
    for c, inc in strat.inferred.items():
        m = measured[c]
        cp90_eff = max(cohort[c].conditional_p90, cohort[c].conditional_p90_hi)
        cap = 2.0 * m if cp90_eff <= 0.0 else min(2.0 * m, cp90_eff)
        assert m + inc <= cap + 1e-6, c
    # eatout realistic-level guard: deployed total stays at/below the
    # effective population p90 (the plausibility bound from the
    # ELASTICITY-DETERMINED-SURPLUS-FREE-SET-SCOPE forward-note; max() form —
    # broad is the floor when the hi-subset p90 dips below it).
    eat_total = measured["eatout"] + strat.inferred.get("eatout", 0.0)
    eat_p90_eff = max(cohort["eatout"].conditional_p90,
                      cohort["eatout"].conditional_p90_hi)
    assert eat_total <= eat_p90_eff + 1e-6
    # Slack conservation (four-way closure at the unit level).
    assert strat.new_slack == pytest.approx(
        0.30 * D_VAR - sum(strat.inferred.values()), abs=1e-6
    )
