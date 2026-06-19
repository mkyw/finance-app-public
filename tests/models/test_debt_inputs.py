"""Debt-input collection tests (Build, 2026-05-31).

Pin the four user-reported debt inputs wired through the existing
post-allocation debt-service stage (DEBT-POST-ALLOCATION-OVER-COHORT-SHIFT;
scoping: agent-artifacts/investigations/debt_treatment_architecture_scoping.md):

  - dual-path ``estimate_annual_debt_service``: cohort-balance fallback
    (UNCHANGED) vs user-reported override (carried balance for CC; monthly
    payment for student/auto/other).
  - REGRESSION-SAFETY (load-bearing): with all override kwargs at 0.0, the
    result is byte-identical to the pre-build two-key behavior, and the
    ``total_debt_service`` is unchanged → non-debt profiles see no change.
  - CC carried-balance surfaces interest explicitly (principal + interest).
  - HouseholdProfile carries the four fields, defaulting to 0.
  - Four-way reconciliation closes exactly with debt present.

Run from repo root:
    .venv/bin/python -m pytest tests/models/test_debt_inputs.py -v
"""

from __future__ import annotations

import pytest

from models.optimizer.backfill import reconcile_four_way
from models.optimizer.debt_service import (
    _CC_MIN_MONTHLY_FLOOR,
    _CC_MIN_MONTHLY_FRACTION,
    _CC_REVOLVING_APR,
    _STUDENT_LOAN_ANNUAL_FACTOR,
    estimate_annual_debt_service,
)
from shared.types.enums import Tenure
from shared.types.household import HouseholdProfile


# ---------------------------------------------------------------------------
# Regression-safety — the load-bearing property.
# ---------------------------------------------------------------------------
class TestRegressionSafety:
    """No override kwargs → byte-identical to the pre-build behavior."""

    @pytest.mark.parametrize(
        "cc,sl",
        [(0.0, 0.0), (-500.0, -10_000.0), (100.0, 0.0), (5_000.0, 0.0),
         (0.0, 20_000.0), (3_000.0, 15_000.0)],
    )
    def test_override_defaults_match_two_arg_call(self, cc: float, sl: float) -> None:
        # The new keys are additive; the old keys + total must be identical
        # whether or not the override kwargs are passed at their defaults.
        old = estimate_annual_debt_service(cc, sl)
        new = estimate_annual_debt_service(
            cc, sl,
            cc_carried_balance=0.0, student_loan_payment=0.0,
            auto_loan_payment=0.0, other_debt_payment=0.0,
        )
        assert old == new
        assert new["auto_loan_service"] == 0.0
        assert new["other_debt_service"] == 0.0
        # total is the same as the historical cc+sl sum (auto/other are 0)
        assert new["total_debt_service"] == pytest.approx(
            new["credit_card_service"] + new["student_loan_service"]
        )

    def test_cohort_cc_floor_preserved(self) -> None:
        # The $25 floor on the cohort-balance path is load-bearing (the
        # prompt's pseudocode dropped it; we keep it for regression-safety).
        svc = estimate_annual_debt_service(100.0, 0.0)
        assert svc["credit_card_service"] == pytest.approx(_CC_MIN_MONTHLY_FLOOR * 12)

    def test_cohort_student_loan_amortization_preserved(self) -> None:
        svc = estimate_annual_debt_service(0.0, 20_000.0)
        assert svc["student_loan_service"] == pytest.approx(
            20_000.0 * _STUDENT_LOAN_ANNUAL_FACTOR
        )


# ---------------------------------------------------------------------------
# Credit-card carried-balance override (principal + interest).
# ---------------------------------------------------------------------------
class TestCreditCardOverride:
    def test_carried_balance_includes_interest(self) -> None:
        # $4,000 carried: interest = 4000*0.22/12 ≈ $73.33; principal =
        # max($25, 2%*4000=$80) = $80; monthly ≈ $153.33 → annual ≈ $1,840.
        svc = estimate_annual_debt_service(0.0, 0.0, cc_carried_balance=4_000.0)
        interest = 4_000.0 * (_CC_REVOLVING_APR / 12)
        principal = max(_CC_MIN_MONTHLY_FLOOR, _CC_MIN_MONTHLY_FRACTION * 4_000.0)
        assert svc["credit_card_service"] == pytest.approx((principal + interest) * 12)
        # Sanity: ~$153/mo, materially above the bare 2%-proxy's $80/mo.
        assert 150.0 < svc["credit_card_service"] / 12 < 156.0

    def test_carried_balance_overrides_cohort(self) -> None:
        # When a carried balance is reported, the cohort othdbt is ignored.
        with_cohort = estimate_annual_debt_service(9_999.0, 0.0, cc_carried_balance=4_000.0)
        no_cohort = estimate_annual_debt_service(0.0, 0.0, cc_carried_balance=4_000.0)
        assert with_cohort["credit_card_service"] == no_cohort["credit_card_service"]

    def test_carried_balance_floor_applies(self) -> None:
        # Tiny carried balance: principal hits the $25 floor; interest is added.
        svc = estimate_annual_debt_service(0.0, 0.0, cc_carried_balance=200.0)
        interest = 200.0 * (_CC_REVOLVING_APR / 12)
        assert svc["credit_card_service"] == pytest.approx((25.0 + interest) * 12)


# ---------------------------------------------------------------------------
# Monthly-payment overrides (student / auto / other).
# ---------------------------------------------------------------------------
class TestPaymentOverrides:
    def test_student_loan_payment_annualizes(self) -> None:
        svc = estimate_annual_debt_service(0.0, 0.0, student_loan_payment=400.0)
        assert svc["student_loan_service"] == pytest.approx(400.0 * 12)

    def test_student_loan_payment_overrides_cohort_amortization(self) -> None:
        # Reported payment wins over the cohort-balance amortization.
        svc = estimate_annual_debt_service(0.0, 50_000.0, student_loan_payment=400.0)
        assert svc["student_loan_service"] == pytest.approx(400.0 * 12)

    def test_auto_loan_payment_annualizes(self) -> None:
        svc = estimate_annual_debt_service(0.0, 0.0, auto_loan_payment=350.0)
        assert svc["auto_loan_service"] == pytest.approx(350.0 * 12)

    def test_other_debt_payment_annualizes(self) -> None:
        svc = estimate_annual_debt_service(0.0, 0.0, other_debt_payment=120.0)
        assert svc["other_debt_service"] == pytest.approx(120.0 * 12)

    def test_auto_and_other_have_no_cohort_fallback(self) -> None:
        # No input → $0 (not modeled), regardless of cohort balances.
        svc = estimate_annual_debt_service(5_000.0, 20_000.0)
        assert svc["auto_loan_service"] == 0.0
        assert svc["other_debt_service"] == 0.0

    def test_negative_payments_treated_as_zero(self) -> None:
        svc = estimate_annual_debt_service(
            0.0, 0.0, auto_loan_payment=-100.0, other_debt_payment=-50.0,
            student_loan_payment=-10.0,
        )
        assert svc["auto_loan_service"] == 0.0
        assert svc["other_debt_service"] == 0.0
        # negative SL payment falls through to the (zero) cohort balance
        assert svc["student_loan_service"] == 0.0


# ---------------------------------------------------------------------------
# Mixed scenarios — some user-reported, some cohort.
# ---------------------------------------------------------------------------
class TestMixedComposition:
    def test_mixed_inputs_compose(self) -> None:
        # CC carried $3,000 (override) + SL payment $200 + auto $350 + no other,
        # with a cohort othdbt that should be ignored (carried balance wins).
        svc = estimate_annual_debt_service(
            credit_card_balance=8_000.0,   # cohort — ignored (carried wins)
            student_loan_balance=0.0,
            cc_carried_balance=3_000.0,
            student_loan_payment=200.0,
            auto_loan_payment=350.0,
            other_debt_payment=0.0,
        )
        cc_interest = 3_000.0 * (_CC_REVOLVING_APR / 12)
        cc_principal = max(_CC_MIN_MONTHLY_FLOOR, _CC_MIN_MONTHLY_FRACTION * 3_000.0)
        assert svc["credit_card_service"] == pytest.approx((cc_principal + cc_interest) * 12)
        assert svc["student_loan_service"] == pytest.approx(200.0 * 12)
        assert svc["auto_loan_service"] == pytest.approx(350.0 * 12)
        assert svc["other_debt_service"] == 0.0
        assert svc["total_debt_service"] == pytest.approx(
            svc["credit_card_service"] + svc["student_loan_service"]
            + svc["auto_loan_service"] + svc["other_debt_service"]
        )

    def test_partial_override_keeps_cohort_for_unreported(self) -> None:
        # Only SL payment reported; CC falls back to the cohort othdbt.
        svc = estimate_annual_debt_service(
            credit_card_balance=5_000.0, student_loan_balance=0.0,
            student_loan_payment=300.0,
        )
        assert svc["credit_card_service"] == pytest.approx(100.0 * 12)  # 2% of 5k
        assert svc["student_loan_service"] == pytest.approx(300.0 * 12)


# ---------------------------------------------------------------------------
# HouseholdProfile carries the fields, defaults to 0.
# ---------------------------------------------------------------------------
class TestProfileFields:
    def test_defaults_zero(self) -> None:
        p = HouseholdProfile(
            age=25, gross_income=90_000, puma_code="IL_03420",
            tenure=Tenure.RENT, housing_cost=1_800.0, household_size=1,
        )
        assert p.cc_carried_balance == 0.0
        assert p.student_loan_payment == 0.0
        assert p.auto_loan_payment == 0.0
        assert p.other_debt_payment == 0.0

    def test_fields_settable(self) -> None:
        p = HouseholdProfile(
            age=25, gross_income=90_000, puma_code="IL_03420",
            tenure=Tenure.RENT, housing_cost=1_800.0, household_size=1,
            cc_carried_balance=4_000.0, student_loan_payment=400.0,
            auto_loan_payment=350.0, other_debt_payment=120.0,
        )
        assert p.cc_carried_balance == 4_000.0
        assert p.student_loan_payment == 400.0
        assert p.auto_loan_payment == 350.0
        assert p.other_debt_payment == 120.0


# ---------------------------------------------------------------------------
# Four-way reconciliation closes exactly with debt present.
# ---------------------------------------------------------------------------
class TestFourWayCloses:
    def test_debt_present_reconciles(self) -> None:
        # Take-home = committed + debt_service + spending + savings + remainder.
        svc = estimate_annual_debt_service(
            0.0, 0.0, cc_carried_balance=4_000.0, student_loan_payment=400.0,
        )
        debt = svc["total_debt_service"]
        take_home = 67_690.0
        committed = 7_528.0
        d_var_adj, savings, remainder = reconcile_four_way(
            take_home=take_home,
            committed_total=committed,
            debt_service=debt,
            spending_total=40_000.0,
            s_star_rate=0.138,
        )
        # d_var_adj is post-committed-AND-debt
        assert d_var_adj == pytest.approx(take_home - committed - debt)
        total = committed + debt + 40_000.0 + savings + remainder
        assert total == pytest.approx(take_home)

    def test_more_debt_shrinks_d_var_adj(self) -> None:
        # Monotone: more debt → smaller adjusted budget → smaller savings+remainder.
        low = reconcile_four_way(
            take_home=67_690.0, committed_total=7_528.0,
            debt_service=1_840.0, spending_total=40_000.0, s_star_rate=0.138,
        )
        high = reconcile_four_way(
            take_home=67_690.0, committed_total=7_528.0,
            debt_service=6_640.0, spending_total=40_000.0, s_star_rate=0.138,
        )
        assert high[0] < low[0]  # smaller d_var_adj
        assert (high[1] + high[2]) < (low[1] + low[2])  # less savings+remainder
