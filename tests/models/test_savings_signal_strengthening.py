"""Down-direction coverage-amplifier tests (piece-3 mixture-grounded build).

Mechanism (locked SAVINGS-SIGNAL-DOWN-DIRECTION-STRENGTHENING):

    w_down = min(W_DOWN_MAX, 0.6 × w_age × max(1, COVERAGE_REF_MONTHS / cov₀))
    cov₀   = balance / (s_cohort × d_var_adj / 12)

W_DOWN_MAX = 1.0 (unit weight, the ½-of-cohort blend floor) pinned against the
Q1.4 SCF cross-tab split-arm finding (scoping addendum, commit d06c5bb): the
high-income low-liquid population is a ~50/50 saver/non-saver mixture (flow
arm), with the unmodeled savings-elsewhere channel essentially absent below
$200K (stock arm).

Pinned properties: direction-gating (up-side + no-signal byte-identical),
the cap (never below ½ cohort), the no-amplification zone (coverage ≥
COVERAGE_REF_MONTHS byte-identical to the static w=0.6 blend), the legacy
signature (no ``s_cohort`` → no amplifier), the demonstrating-profile pin
($557 → ~$452/mo), and composition with the piece-1 spend-arm.
"""
from __future__ import annotations

import pytest

from models.optimizer.backfill import (
    COVERAGE_REF_MONTHS,
    W_DOWN_MAX,
    assign_residual_savings,
    balance_implied_rate,
    blend_savings_rate,
    personalized_savings_rate,
    s_star,
    w_age,
)
from shared.types import HouseholdProfile, Tenure


def _profile(savings: float, age: int = 24, income: float = 110_000.0) -> HouseholdProfile:
    return HouseholdProfile(
        age=age, gross_income=income, puma_code="NY_04103", tenure=Tenure.RENT,
        housing_cost=2_500, household_size=1, savings=savings,
    )


# --------------------------------------------------------------------------- #
# The amplifier                                                                #
# --------------------------------------------------------------------------- #


def test_amplifier_engages_below_reference_coverage() -> None:
    """cov₀ < COVERAGE_REF_MONTHS → w rises above the static 0.6 × w_age."""
    d_var = 60_000.0
    s_cohort = s_star(110_000.0)
    _, w_legacy = balance_implied_rate(_profile(1_000.0), d_var)
    _, w_amp = balance_implied_rate(_profile(1_000.0), d_var, s_cohort=s_cohort)
    assert w_legacy == pytest.approx(0.6 * w_age(24))
    assert w_amp > w_legacy


def test_cap_binds_at_unit_weight_extreme_coverage() -> None:
    """Balance covering <1 month of cohort flow: w hits W_DOWN_MAX exactly and
    the blend lands at the unit-weight form — never below ½ cohort."""
    d_var = 60_000.0
    s_cohort = s_star(110_000.0)
    bal = 200.0  # cov₀ ≈ 0.25 months — extreme inconsistency
    s_impl, w = balance_implied_rate(_profile(bal), d_var, s_cohort=s_cohort)
    assert s_impl is not None
    assert w == pytest.approx(W_DOWN_MAX)
    blend = blend_savings_rate(s_cohort, s_impl, w, upward_confirmed=False)
    assert blend == pytest.approx((s_impl + s_cohort) / 2.0)
    assert blend >= s_cohort / 2.0 - 1e-12


def test_no_amplification_at_or_above_reference_coverage() -> None:
    """Balance covering ≥ COVERAGE_REF_MONTHS of cohort flow: amp = 1,
    byte-identical to the legacy weight (the moderate-coverage case)."""
    d_var = 60_000.0
    s_cohort = s_star(110_000.0)
    monthly_cohort_flow = s_cohort * d_var / 12.0
    # 2× the reference (6 months coverage) — comfortably in the no-amp zone,
    # while the implied rate still points down.
    bal = 2.0 * COVERAGE_REF_MONTHS * monthly_cohort_flow
    s_impl, w = balance_implied_rate(_profile(bal), d_var, s_cohort=s_cohort)
    assert s_impl is not None
    assert s_impl < s_cohort  # signal still points down
    assert w == pytest.approx(0.6 * w_age(24))  # no amplification


def test_direction_gate_up_side_unamplified() -> None:
    """High balance (implied above cohort): the amplifier never fires; the
    up-cap result is byte-identical to pre-build."""
    d_var = 60_000.0
    s_cohort = s_star(110_000.0)
    s_impl, w = balance_implied_rate(_profile(80_000.0), d_var, s_cohort=s_cohort)
    assert s_impl is not None
    assert s_impl > s_cohort
    assert w == pytest.approx(0.6 * w_age(24))  # base weight, no amp
    s_pers = personalized_savings_rate(_profile(80_000.0), d_var)[0]
    assert s_pers == pytest.approx(s_cohort)  # up-cap unchanged


def test_no_signal_byte_identical() -> None:
    s_pers, s_cohort, s_impl, w, _ = personalized_savings_rate(_profile(0.0), 60_000.0)
    assert s_impl is None and w == 0.0
    assert s_pers == s_cohort


def test_legacy_signature_has_no_amplifier() -> None:
    """Callers not passing s_cohort get the pre-build weight exactly."""
    _, w = balance_implied_rate(_profile(200.0), 60_000.0)
    assert w == pytest.approx(0.6 * w_age(24))


def test_older_profile_cap_still_unit_weight() -> None:
    """w_age scales the unclipped region; the cap is uniform (no new age
    conditioning — w_age owns age, the cap owns the floor)."""
    d_var = 60_000.0
    s_cohort = s_star(110_000.0)
    _, w55 = balance_implied_rate(_profile(200.0, age=55), d_var, s_cohort=s_cohort)
    assert w55 <= W_DOWN_MAX + 1e-12
    # Unclipped region keeps the age gradient: moderate coverage, older user.
    monthly = s_cohort * d_var / 12.0
    _, w55_mod = balance_implied_rate(
        _profile(2.0 * monthly, age=55), d_var, s_cohort=s_cohort
    )
    _, w24_mod = balance_implied_rate(
        _profile(2.0 * monthly, age=24), d_var, s_cohort=s_cohort
    )
    assert w55_mod < w24_mod  # w_age gradient survives below the cap


# --------------------------------------------------------------------------- #
# The demonstrating-profile pin                                                #
# --------------------------------------------------------------------------- #


def test_demonstrating_profile_pin() -> None:
    """24yo / $110K / $1K balance at the live d_var ($66,732/yr): the
    prediction moves $557 → ~$452/mo (coverage 1.8 → ~2.2 months)."""
    d_var = 66_732.0
    ra = assign_residual_savings(
        d_variable_adjusted=d_var,
        post_backfill_slack=11_064.0,  # the live $922/mo slack
        profile=_profile(1_000.0),
    )
    monthly = ra.savings_investment / 12.0
    assert monthly == pytest.approx(452.0, abs=3.0)
    coverage = 1_000.0 / monthly
    assert 2.0 < coverage < 2.4
    assert ra.framing_state == "signal_pulled_down"
    # The savings-elsewhere hedge is named in the source (framing requirement).
    assert "savings" in ra.source and "elsewhere" in ra.source


# --------------------------------------------------------------------------- #
# Composition with the piece-1 spend-arm                                       #
# --------------------------------------------------------------------------- #


def test_strengthening_lowers_backfill_benchmark_further() -> None:
    """The amplified rate < the piece-1-only (static w=0.6) rate < cohort —
    same shared helper feeds the back-fill benchmark, so the pool grows
    beyond piece 1 alone on non-saturated profiles."""
    d_var = 60_000.0
    p = _profile(1_000.0)
    s_pers, s_cohort, s_impl, _, _ = personalized_savings_rate(p, d_var)
    assert s_impl is not None
    # Reconstruct the piece-1-only blend (static weight, no amplifier).
    _, w_legacy = balance_implied_rate(p, d_var)
    s_piece1 = blend_savings_rate(s_cohort, s_impl, w_legacy, upward_confirmed=False)
    assert s_pers < s_piece1 < s_cohort
