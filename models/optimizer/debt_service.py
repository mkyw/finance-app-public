"""Annual debt-service estimation from liability balances / payments.

The allocator needs a "committed outflow" estimate for the household's
debt so it can be subtracted from ``d_variable`` before the flow
feasibility check. Without this step, two households with the same
income but very different debt loads would be told they have the same
discretionary budget — which is wrong.

This is the **post-allocation** treatment of debt locked by
``DEBT-POST-ALLOCATION-OVER-COHORT-SHIFT`` (DECISIONS.md): debt-service
subtracts from take-home *after* the cohort is matched on nominal income,
never adjusts which cohort the user matches against. See
``agent-artifacts/investigations/debt_treatment_architecture_scoping.md``.

Two input sources, **user-reported overrides take precedence over the
cohort-predicted balances** (per-component); absent a user value, each
component falls back to its cohort prior (the regression-safe default —
non-debt users see byte-identical predictions to before this build):

Credit card service
    Cohort-balance fallback (UNCHANGED — regression-safe):
        monthly_min = max($25, 0.02 * balance);  annual = monthly_min * 12
    User-reported carried balance (``cc_carried_balance > 0``):
        monthly = max($25, 0.02 * balance)  +  balance * APR/12
        annual  = monthly * 12
    The carried-balance path surfaces the *interest* cost explicitly
    (principal-ish minimum + revolving interest), which the bare
    2%-of-balance proxy buries — more honest to the real carry cost, and
    "what balance do you carry month to month?" is a more answerable
    question than imputing a balance from othdbt. Example: a $4,000
    carried balance → ~$73/mo interest + ~$80/mo principal ≈ $153/mo,
    vs the bare 2%-proxy's $80/mo. The $25 floor and 2% minimum mirror
    the standard issuer minimum-payment proxy.

Student loan / auto / other service
    User-reported monthly *payment* (the natural quantity for an
    amortizing loan) annualizes directly: ``annual = payment * 12``.
    Student loans fall back to the cohort-balance 10-year amortization
    (UNCHANGED): ``annual = balance * 0.1363`` — the 0.1363 factor is
    12 * (r/12) / (1 - (1 + r/12)^-120) at r = 0.065 (≈ the federal
    Direct Loan undergrad rate; a conservative baseline). Auto and other
    debt have **no cohort balance column** in the donor, so absent a
    user payment they are $0 (not modeled) — not a fabricated prediction.

All formulas assume a constant balance over the year (no paydown / new
charges); MVP heuristics, not amortization simulations.
"""

from __future__ import annotations

_MONTHS_PER_YEAR: float = 12.0

# Credit card constants.
_CC_MIN_MONTHLY_FRACTION: float = 0.02  # 2%-of-balance minimum payment proxy
_CC_MIN_MONTHLY_FLOOR: float = 25.0     # $25 absolute floor issuers round up to
_CC_REVOLVING_APR: float = 0.22         # cohort-typical revolving APR (~22%, 2024-25)
                                        # [CALIBRATE — Fed G.19 assessed-interest APR]

# Student loan constants — 10-year level amortization at 6.5% APR.
# Annual factor = 12 * (r/12) / (1 - (1 + r/12)^-120), r = 0.065.
_STUDENT_LOAN_ANNUAL_FACTOR: float = 0.13627


def estimate_annual_debt_service(
    credit_card_balance: float,
    student_loan_balance: float,
    *,
    cc_carried_balance: float = 0.0,
    student_loan_payment: float = 0.0,
    auto_loan_payment: float = 0.0,
    other_debt_payment: float = 0.0,
) -> dict[str, float]:
    """Estimate annual debt-service, user-reported overrides preferred.

    Args:
        credit_card_balance: Cohort-predicted revolving balance (CEX
            ``othdbt`` p50 — the fallback). Values <= 0 produce 0 CC
            service when no override is given.
        student_loan_balance: Cohort-predicted student-loan balance (CEX
            ``stddbt`` p50 — the fallback).
        cc_carried_balance: USER-reported credit-card balance carried
            month to month. When > 0, OVERRIDES ``credit_card_balance``
            and uses the principal+interest path. <= 0 → fall back.
        student_loan_payment: USER-reported MONTHLY student-loan payment.
            When > 0, OVERRIDES the amortization fallback (annualize ×12).
        auto_loan_payment: USER-reported MONTHLY auto-loan payment. No
            cohort fallback (annualize ×12; absent → $0, not modeled).
        other_debt_payment: USER-reported MONTHLY payment toward other
            debts. No cohort fallback (annualize ×12; absent → $0).

    Returns:
        Dict, all annual dollars:
          - ``credit_card_service``
          - ``student_loan_service``
          - ``auto_loan_service``
          - ``other_debt_service``
          - ``total_debt_service`` (sum of the four)

    Regression-safety: with all four override kwargs at their 0.0
    defaults, the credit-card and student-loan branches reproduce the
    pre-build formulas exactly, ``auto``/``other`` are 0, and
    ``total_debt_service`` is identical to the old two-key result.
    """
    # Credit cards — user-reported carried balance (override) preferred.
    if cc_carried_balance > 0:
        monthly_interest = float(cc_carried_balance) * (_CC_REVOLVING_APR / _MONTHS_PER_YEAR)
        monthly_principal = max(
            _CC_MIN_MONTHLY_FLOOR,
            _CC_MIN_MONTHLY_FRACTION * float(cc_carried_balance),
        )
        cc_annual = (monthly_principal + monthly_interest) * _MONTHS_PER_YEAR
    elif credit_card_balance > 0:
        # Cohort-balance fallback — UNCHANGED from the pre-build formula
        # (the $25 floor is retained; it is load-bearing for regression-safety).
        cc_monthly = max(
            _CC_MIN_MONTHLY_FLOOR,
            _CC_MIN_MONTHLY_FRACTION * float(credit_card_balance),
        )
        cc_annual = cc_monthly * _MONTHS_PER_YEAR
    else:
        cc_annual = 0.0

    # Student loans — user-reported monthly payment (override) preferred,
    # else cohort-balance amortization (UNCHANGED).
    if student_loan_payment > 0:
        sl_annual = float(student_loan_payment) * _MONTHS_PER_YEAR
    else:
        sl_annual = max(0.0, float(student_loan_balance)) * _STUDENT_LOAN_ANNUAL_FACTOR

    # Auto + other — user-reported monthly payment only; no cohort fallback.
    auto_annual = max(0.0, float(auto_loan_payment)) * _MONTHS_PER_YEAR
    other_annual = max(0.0, float(other_debt_payment)) * _MONTHS_PER_YEAR

    return {
        "credit_card_service": cc_annual,
        "student_loan_service": sl_annual,
        "auto_loan_service": auto_annual,
        "other_debt_service": other_annual,
        "total_debt_service": cc_annual + sl_annual + auto_annual + other_annual,
    }
