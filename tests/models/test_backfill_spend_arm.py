"""Low-side spend-arm tests (back-fill benchmark = personalized blended rate).

The spend-arm build (2026-06-09) makes ``compute_backfill``'s trigger + pool
use the personalized (blended) savings rate instead of raw cohort ``s*`` —
the symmetric counterpart of the high-balance ``soft_signal_gate``
suppression found one-sided by ``savings_balance_signal_diagnostic.md``.

Pinned properties:
  - no-balance path byte-identical (benchmark collapses to cohort s*)
  - high-balance path byte-identical (blend up-capped at cohort; g-gate
    suppression unchanged)
  - low balance lowers the benchmark → earlier trigger + larger pool →
    discretionary lifts within cohort ceilings
  - the back-fill benchmark and the savings-label rate are the SAME number
    (shared ``personalized_savings_rate`` helper — the agreement property)
  - constrained profiles immune (primary-only guard)
  - slack conservation: lift + new_slack == old slack; savings + remainder
    == new_slack (four-way closure at the unit level)

Run from repo root (artifacts needed for the QUAIDS elasticity lookup).
"""
from __future__ import annotations

import pytest

from models.optimizer.backfill import (
    assign_residual_savings,
    compute_backfill,
    personalized_savings_rate,
    s_star,
)
from shared.types import HouseholdProfile, SpendingDistribution, Tenure

ARTIFACTS = "pipeline/artifacts"
# Raw (55-cat) target cats present in the default coefficients file, so the
# unit tests don't depend on the aggregated coefficients path.
TARGETS = frozenset({"hotel", "airshp"})
D_VAR = 60_000.0


def _profile(savings: float, age: int = 24, income: float = 110_000.0) -> HouseholdProfile:
    return HouseholdProfile(
        age=age, gross_income=income, puma_code="NY_04103", tenure=Tenure.RENT,
        housing_cost=2_500, household_size=1, savings=savings,
    )


def _dist(measured: float, cond_p90: float) -> SpendingDistribution:
    return SpendingDistribution(
        p10=0.0, p25=0.0, p50=measured, p75=measured * 1.5, p90=measured * 2.5,
        engel_estimate=0.0, feasibility_adjusted=measured, cohort_position=0.5,
        is_structural=False, conditional_p90=cond_p90,
    )


def _run(savings: float, *, slack_rate: float, solver_status: str = "primary"):
    """compute_backfill on a synthetic two-target setup with ample room."""
    measured = {"hotel": 1_200.0, "airshp": 1_500.0}
    cohort = {c: _dist(m, cond_p90=m * 4.0) for c, m in measured.items()}
    return compute_backfill(
        measured=measured,
        cohort=cohort,
        profile=_profile(savings),
        feasibility_slack=slack_rate * D_VAR,
        solver_status=solver_status,
        d_variable_adjusted=D_VAR,
        artifacts_path=ARTIFACTS,
        target_categories=TARGETS,
    )


# --------------------------------------------------------------------------- #
# The shared helper                                                            #
# --------------------------------------------------------------------------- #


def test_helper_collapses_to_cohort_without_signal() -> None:
    s_pers, s_cohort, s_implied, w, up = personalized_savings_rate(_profile(0.0), D_VAR)
    assert s_pers == s_cohort == s_star(110_000.0)
    assert s_implied is None and w == 0.0 and up is False


def test_helper_pulls_down_on_low_balance() -> None:
    s_pers, s_cohort, _, w, _ = personalized_savings_rate(_profile(1_000.0), D_VAR)
    assert w > 0.0
    assert s_pers < s_cohort
    # Structural floor: ½ of cohort since the piece-3 coverage amplifier
    # (W_DOWN_MAX = 1.0, unit weight) — was s_cohort/1.6 at the static w=0.6.
    assert s_pers >= s_cohort / 2.0 - 1e-12


def test_helper_up_capped_on_high_balance() -> None:
    s_pers, s_cohort, s_implied, _, _ = personalized_savings_rate(_profile(50_000.0), D_VAR)
    assert s_implied is not None and s_implied > s_cohort
    assert s_pers == pytest.approx(s_cohort)  # up-cap without confirmation


# --------------------------------------------------------------------------- #
# The low-side arm                                                             #
# --------------------------------------------------------------------------- #


def test_trigger_flip_low_balance_fires_where_no_signal_does_not() -> None:
    """Residual between (pers + buffer) and (cohort + buffer): no-signal user
    does not back-fill; the low-balance user does — the new arm."""
    s_cohort = s_star(110_000.0)
    s_pers = personalized_savings_rate(_profile(1_000.0), D_VAR)[0]
    slack_rate = (s_pers + 0.06 + s_cohort + 0.06) / 2.0  # strictly between
    base = _run(0.0, slack_rate=slack_rate)
    low = _run(1_000.0, slack_rate=slack_rate)
    assert base.fired is False and base.pool == 0.0
    assert low.fired is True and low.pool > 0.0
    assert sum(low.inferred.values()) > 0.0
    assert low.s_star_personalized == pytest.approx(s_pers)
    assert low.s_star == pytest.approx(s_cohort)  # cohort audit value retained


def test_low_balance_grows_pool_above_no_signal_pool() -> None:
    """Both fire (slack above cohort+buffer); the low-balance pool is larger
    by exactly (s_cohort − s_pers) × d_var."""
    s_cohort = s_star(110_000.0)
    s_pers = personalized_savings_rate(_profile(1_000.0), D_VAR)[0]
    slack_rate = s_cohort + 0.10
    base = _run(0.0, slack_rate=slack_rate)
    low = _run(1_000.0, slack_rate=slack_rate)
    assert base.fired and low.fired
    assert low.pool - base.pool == pytest.approx((s_cohort - s_pers) * D_VAR)


def test_lift_respects_cohort_ceilings() -> None:
    res = _run(1_000.0, slack_rate=0.40)  # huge pool — force the caps to bind
    measured = {"hotel": 1_200.0, "airshp": 1_500.0}
    for cat, inc in res.inferred.items():
        cap = min(2.0 * measured[cat], measured[cat] * 4.0)  # = 2×anchor here
        assert measured[cat] + inc <= cap + 1e-6


def test_slack_conservation_and_label_agreement() -> None:
    """lift + new_slack == old slack; savings + remainder == new_slack; and
    the back-fill benchmark equals the savings-label rate (shared helper)."""
    slack = 0.20 * D_VAR
    res = _run(1_000.0, slack_rate=0.20)
    assert sum(res.inferred.values()) + res.new_slack == pytest.approx(slack)
    ra = assign_residual_savings(
        d_variable_adjusted=D_VAR, post_backfill_slack=res.new_slack,
        profile=_profile(1_000.0),
    )
    assert ra.savings_investment + ra.genuine_remainder == pytest.approx(res.new_slack)
    assert ra.realistic_savings_rate == pytest.approx(res.s_star_personalized)


# --------------------------------------------------------------------------- #
# Regression-safety                                                            #
# --------------------------------------------------------------------------- #


def test_no_balance_benchmark_is_cohort() -> None:
    res = _run(0.0, slack_rate=0.30)
    assert res.s_star_personalized == pytest.approx(res.s_star)
    assert res.pool == pytest.approx((0.30 - res.s_star) * D_VAR)  # g = 1.0


def test_high_balance_suppression_preserved() -> None:
    """$50K at this age/income: blend up-caps (benchmark stays cohort) and
    the g-gate fully suppresses — byte-identical to the pre-build high side."""
    res = _run(50_000.0, slack_rate=0.30)
    assert res.s_star_personalized == pytest.approx(res.s_star)  # up-cap
    assert res.g == pytest.approx(0.0)
    assert res.fired is False and res.pool == 0.0


@pytest.mark.parametrize("status", ["soft_constrained", "floor_infeasible", "structural_deficit"])
def test_constrained_profiles_immune(status: str) -> None:
    res = _run(1_000.0, slack_rate=0.30, solver_status=status)
    assert res.fired is False
    assert res.inferred == {}
    assert res.new_slack == pytest.approx(0.30 * D_VAR)
