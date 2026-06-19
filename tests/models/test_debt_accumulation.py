"""Unit tests for the debt-accumulation annotation (soft-deficit-with-CC-debt).

Pure-logic tests — no artifacts needed. The Q8 test list from
``debt_accumulation_prediction_scoping.md``: signal fires (monthly ==
gap/12), regression-safe non-fire paths (no CC debt / primary /
structural_deficit / floor_infeasible / trivial balance), framing-state
cuts, and the Q7 override pin (out-of-sum, so the pin cannot break
dollar-accounting by construction).
"""
from __future__ import annotations

import pytest

from models.optimizer.debt_accumulation import (
    apply_debt_accumulation_override,
    project_debt_accumulation,
)
from shared.types import HouseholdProfile, Tenure


def _profile(cc_balance: float = 5000.0, gross: float = 42000.0) -> HouseholdProfile:
    return HouseholdProfile(
        age=30,
        gross_income=gross,
        puma_code="WI_02101",
        tenure=Tenure.RENT,
        housing_cost=1000,
        household_size=1,
        cc_carried_balance=cc_balance,
    )


# --------------------------------------------------------------------------- #
# Signal fires                                                                 #
# --------------------------------------------------------------------------- #


def test_signal_fires_with_cc_debt_and_soft_constrained() -> None:
    ann = project_debt_accumulation(
        _profile(cc_balance=5000.0, gross=42000.0),
        solver_status="soft_constrained",
        compression_gap=2400.0,
        d_variable_adjusted=20000.0,
    )
    assert ann.applies is True
    assert ann.annual_potential_growth == pytest.approx(2400.0)
    assert ann.monthly_potential_growth == pytest.approx(200.0)
    assert ann.basis == "cohort_typical_spending_exceeds_take_home"
    assert ann.source == "inferred_from_soft_deficit"
    assert ann.adjustable is True


def test_dollar_figure_is_full_gap_no_haircut() -> None:
    """OVER-PREDICTION direction: the displayed amount is the full honest
    gap; the hedge lives in framing_state, never in a discounted number."""
    gap = 3333.33
    ann = project_debt_accumulation(
        _profile(),
        solver_status="soft_constrained",
        compression_gap=gap,
        d_variable_adjusted=25000.0,
    )
    assert ann.annual_potential_growth == pytest.approx(gap)


# --------------------------------------------------------------------------- #
# Regression-safe non-fire paths (Q6 scope guard)                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "status", ["primary", "structural_deficit", "floor_infeasible"]
)
def test_no_fire_on_other_solver_statuses(status: str) -> None:
    ann = project_debt_accumulation(
        _profile(cc_balance=8000.0),
        solver_status=status,
        compression_gap=2400.0,  # even with a (hypothetical) gap supplied
        d_variable_adjusted=20000.0,
    )
    assert ann.applies is False
    assert ann.annual_potential_growth == 0.0
    assert ann.monthly_potential_growth == 0.0


def test_no_fire_without_cc_debt() -> None:
    ann = project_debt_accumulation(
        _profile(cc_balance=0.0),
        solver_status="soft_constrained",
        compression_gap=2400.0,
        d_variable_adjusted=20000.0,
    )
    assert ann.applies is False


def test_no_fire_on_trivial_balance_below_floor() -> None:
    ann = project_debt_accumulation(
        _profile(cc_balance=200.0),
        solver_status="soft_constrained",
        compression_gap=2400.0,
        d_variable_adjusted=20000.0,
    )
    assert ann.applies is False


def test_no_fire_on_zero_gap() -> None:
    ann = project_debt_accumulation(
        _profile(),
        solver_status="soft_constrained",
        compression_gap=0.0,
        d_variable_adjusted=20000.0,
    )
    assert ann.applies is False


# --------------------------------------------------------------------------- #
# Framing-state cuts                                                           #
# --------------------------------------------------------------------------- #


def test_signal_clear_on_high_balance_to_income() -> None:
    # $5k balance at $42k income = 11.9% >= the 10% cut.
    ann = project_debt_accumulation(
        _profile(cc_balance=5000.0, gross=42000.0),
        solver_status="soft_constrained",
        compression_gap=500.0,
        d_variable_adjusted=20000.0,  # gap ratio 2.5% — below its cut
    )
    assert ann.framing_state == "signal_clear"
    assert ann.cc_balance_to_income == pytest.approx(5000.0 / 42000.0)


def test_signal_clear_on_high_gap_ratio_alone() -> None:
    # Modest balance (0.5% of income) but the gap is 11.7% of the budget.
    ann = project_debt_accumulation(
        _profile(cc_balance=500.0, gross=100000.0),
        solver_status="soft_constrained",
        compression_gap=3500.0,
        d_variable_adjusted=30000.0,
    )
    assert ann.framing_state == "signal_clear"
    assert ann.gap_ratio == pytest.approx(3500.0 / 30000.0)


def test_signal_marginal_when_both_ratios_small() -> None:
    # $1k at $60k (1.7%); $1k gap on $30k budget (3.3%).
    ann = project_debt_accumulation(
        _profile(cc_balance=1000.0, gross=60000.0),
        solver_status="soft_constrained",
        compression_gap=1000.0,
        d_variable_adjusted=30000.0,
    )
    assert ann.framing_state == "signal_marginal"


# --------------------------------------------------------------------------- #
# Q7 override (apply_debt_accumulation_override)                               #
# --------------------------------------------------------------------------- #


def _fired() -> "object":
    return project_debt_accumulation(
        _profile(),
        solver_status="soft_constrained",
        compression_gap=2400.0,
        d_variable_adjusted=20000.0,
    )


def test_override_pins_to_zero() -> None:
    pinned = apply_debt_accumulation_override(_fired(), 0.0)
    assert pinned.applies is True
    assert pinned.framing_state == "user_pinned"
    assert pinned.annual_potential_growth == 0.0
    assert pinned.monthly_potential_growth == 0.0
    assert "user-overridden" in pinned.source


def test_override_allows_negative_paydown() -> None:
    """Pin to a negative value = 'I'm paying the balance down' —
    deliberately not clamped at 0."""
    pinned = apply_debt_accumulation_override(_fired(), -1200.0)
    assert pinned.annual_potential_growth == pytest.approx(-1200.0)
    assert pinned.monthly_potential_growth == pytest.approx(-100.0)
    assert pinned.framing_state == "user_pinned"


def test_override_noop_on_non_applicable_base() -> None:
    base = project_debt_accumulation(
        _profile(cc_balance=0.0),
        solver_status="primary",
        compression_gap=0.0,
        d_variable_adjusted=20000.0,
    )
    pinned = apply_debt_accumulation_override(base, 500.0)
    assert pinned == base
    assert pinned.applies is False
