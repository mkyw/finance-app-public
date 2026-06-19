"""Savings-signal-weighting tests (Build, 2026-05-29).

Pin the personalization of the savings line: the cohort prior s*(y_eq) blended
with the user-reported balance signal, soft-asymmetric (down free, up needs
explicit confirmation), regression-safe for signal-less users.

  - `balance_implied_rate`: balance → implied rate + credibility weight; the
    uninformative cases return (None, 0.0).
  - `blend_savings_rate`: credibility-weighted, w_user=0 → exactly s_cohort;
    down free; up capped without confirmation; up released with confirmation.
  - `assign_residual_savings`: no-signal path byte-identical to the prior
    s*-bounded baseline; framing_state reflects the signal direction.
"""

from __future__ import annotations

from models.optimizer.backfill import (
    SIGNAL_STRENGTH_BALANCE,
    WORKING_START_AGE,
    apply_savings_override,
    assign_residual_savings,
    balance_implied_rate,
    blend_savings_rate,
    s_star,
    w_age,
)
from shared.types.enums import Tenure
from shared.types.household import HouseholdProfile

_DVA = 60_168.0  # representative d_var_adj
_SLACK = 20_000.0  # ample slack so the s* cap binds, not the slack


def _p(age=25, income=90_000, size=1, savings=0.0):
    return HouseholdProfile(
        age=age, gross_income=income, puma_code="IL_03420", tenure=Tenure.RENT,
        housing_cost=1_800.0, household_size=size, savings=savings,
    )


class TestBalanceImpliedRate:
    def test_no_balance_uninformative(self) -> None:
        assert balance_implied_rate(_p(savings=0.0), _DVA) == (None, 0.0)

    def test_compressed_d_var_uninformative(self) -> None:
        assert balance_implied_rate(_p(savings=15_000.0), 0.0) == (None, 0.0)

    def test_age_at_or_below_working_start_uninformative(self) -> None:
        young = _p(age=int(WORKING_START_AGE), savings=15_000.0)
        assert balance_implied_rate(young, _DVA) == (None, 0.0)

    def test_standard_conversion(self) -> None:
        s_user, w_user = balance_implied_rate(_p(age=25, savings=15_000.0), _DVA)
        assert s_user is not None
        # balance / (25-21) / d_var = 15000/4/60168
        assert abs(s_user - (15_000.0 / 4.0 / _DVA)) < 1e-9
        assert abs(w_user - SIGNAL_STRENGTH_BALANCE * w_age(25)) < 1e-9

    def test_weight_decreases_with_age(self) -> None:
        _, w_young = balance_implied_rate(_p(age=25, savings=15_000.0), _DVA)
        _, w_old = balance_implied_rate(_p(age=55, savings=15_000.0), _DVA)
        assert w_old < w_young


class TestBlendSavingsRate:
    def test_no_user_weight_returns_cohort_exactly(self) -> None:
        # Regression-safety: w_user=0 → exactly s_cohort (no float drift).
        assert blend_savings_rate(0.138, 0.99, 0.0, upward_confirmed=False) == 0.138

    def test_down_pull_free(self) -> None:
        out = blend_savings_rate(0.138, 0.05, 0.6, upward_confirmed=False)
        assert out < 0.138

    def test_up_capped_without_confirmation(self) -> None:
        out = blend_savings_rate(0.138, 0.40, 0.6, upward_confirmed=False)
        assert out == 0.138  # capped at cohort

    def test_up_released_with_confirmation(self) -> None:
        out = blend_savings_rate(0.138, 0.40, 0.6, upward_confirmed=True)
        assert out > 0.138
        # equals the symmetric blend
        assert abs(out - (0.6 * 0.40 + 0.138) / 1.6) < 1e-12


class TestAsymmetricDirectionProperty:
    """The locked soft-asymmetry: down free, up needs explicit confirmation."""

    COHORT = 0.14

    def test_below_cohort_never_exceeds(self) -> None:
        for s_user in (0.0, 0.05, 0.13, 0.139):
            assert blend_savings_rate(self.COHORT, s_user, 0.6, upward_confirmed=False) <= self.COHORT

    def test_above_cohort_pinned_to_cohort_without_confirmation(self) -> None:
        for s_user in (0.15, 0.30, 0.99):
            assert blend_savings_rate(self.COHORT, s_user, 0.6, upward_confirmed=False) == self.COHORT

    def test_above_cohort_allowed_with_confirmation(self) -> None:
        for s_user in (0.15, 0.30, 0.99):
            assert blend_savings_rate(self.COHORT, s_user, 0.6, upward_confirmed=True) >= self.COHORT


class TestAssignResidualSavingsPersonalization:
    def test_no_signal_byte_identical_to_baseline(self) -> None:
        """Load-bearing regression: signal-less prediction == prior s*-bounded."""
        prof = _p(savings=0.0)
        s_cohort = s_star(float(prof.equivalized_income))
        expected_savings = min(_SLACK, s_cohort * _DVA)
        ra = assign_residual_savings(
            d_variable_adjusted=_DVA, post_backfill_slack=_SLACK, profile=prof
        )
        assert ra.savings_investment == expected_savings
        assert ra.realistic_savings_rate == s_cohort
        assert ra.framing_state == "signal_confirmed_cohort"

    def test_balance_pulls_down(self) -> None:
        prof = _p(age=25, savings=15_000.0)
        s_cohort = s_star(float(prof.equivalized_income))
        ra = assign_residual_savings(
            d_variable_adjusted=_DVA, post_backfill_slack=_SLACK, profile=prof
        )
        assert ra.realistic_savings_rate < s_cohort
        assert ra.savings_investment < s_cohort * _DVA
        assert ra.framing_state == "signal_pulled_down"

    def test_high_balance_capped_at_cohort(self) -> None:
        prof = _p(age=25, savings=80_000.0)
        s_cohort = s_star(float(prof.equivalized_income))
        ra = assign_residual_savings(
            d_variable_adjusted=_DVA, post_backfill_slack=_SLACK, profile=prof
        )
        assert abs(ra.realistic_savings_rate - s_cohort) < 1e-12  # cap held
        assert ra.framing_state == "signal_would_pull_up_deferred"

    def test_older_messy_balance_near_baseline(self) -> None:
        prof = _p(age=55, savings=500_000.0)
        s_cohort = s_star(float(prof.equivalized_income))
        ra = assign_residual_savings(
            d_variable_adjusted=_DVA, post_backfill_slack=_SLACK, profile=prof
        )
        # implied rate is above cohort → capped at cohort (low w_age + cap)
        assert ra.realistic_savings_rate <= s_cohort + 1e-12

    def test_override_marks_user_pinned(self) -> None:
        prof = _p(age=25, savings=0.0)
        base = assign_residual_savings(
            d_variable_adjusted=_DVA, post_backfill_slack=_SLACK, profile=prof
        )
        pinned = apply_savings_override(base, base.savings_investment + 2_000.0)
        assert pinned.framing_state == "user_pinned"
        # total (savings + remainder) preserved — it's a label on the residual
        assert abs(
            (pinned.savings_investment + pinned.genuine_remainder)
            - (base.savings_investment + base.genuine_remainder)
        ) < 1e-9

    def test_compressed_profile_zero_savings(self) -> None:
        ra = assign_residual_savings(
            d_variable_adjusted=0.0, post_backfill_slack=0.0, profile=_p()
        )
        assert ra.savings_investment == 0.0
        assert ra.framing_state == "signal_confirmed_cohort"
