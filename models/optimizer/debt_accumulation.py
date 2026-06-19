"""Debt-accumulation annotation for the soft-deficit-with-CC-debt regime.

When the soft-constraint optimizer compresses (``solver_status ==
"soft_constrained"``: cohort-typical anchors exceed the debt-adjusted
budget but the composed floors still fit), the gap between cohort-typical
spending and ``d_variable_adjusted`` vanishes into the compressed
allocation — the deficit is *hidden*. For users who report a carried
credit-card balance, that gap has a likely real-world destination: new
revolving balance. This module surfaces it as a **conditional, named,
adjustable annotation** — "if your spending matches typical, ~$X/mo could
add to your card balance" — WITHOUT changing solver behavior and WITHOUT
entering the four-way sum (the four-way keeps closing on the clean
compressed allocation; this line is a counterfactual annotation outside
it, so no dollar is counted twice).

Design locked by ``agent-artifacts/investigations/
debt_accumulation_prediction_scoping.md`` (2026-06-01; Q1–Q8):

  - **Signal** (Q1): ``cc_carried_balance > 0`` AND ``solver_status ==
    "soft_constrained"`` AND ``compression_gap > 0``. CC-only trigger —
    student/auto/other debt is amortizing or deferred and does not
    accumulate under minimum payments; it still tightens ``adjusted_d``
    (enlarging the gap) but never fires the accumulation narrative.
    (The scoping doc predates the Phase-8 rename ``"compressed"`` →
    ``"soft_constrained"``; this is the same regime.)
  - **Form** (Q2/Q3): conditional annotation, NOT a hard four-way flow.
    ``COHORT-AVERAGE-RESPECTS-MUTUAL-EXCLUSION`` forbids asserting a
    trajectory — the regime blends accumulators, compressors, and
    paydowners (Aladangady 2025 shows the consumption-cut margin is
    real). Collapse-and-attribute: predict the cohort-typical outcome
    conditionally; the user attributes their actual trajectory via the
    override (``apply_debt_accumulation_override``).
  - **Honest dollar, hedged words** (Q4/Q8): the figure is the FULL gap
    (no confidence haircut — ``OVER-PREDICTION`` direction); uncertainty
    lives entirely in the framing (``framing_state`` + conditional copy).
  - **Scope guard** (Q6): a strict no-op on ``"primary"`` (no gap),
    ``"structural_deficit"`` and ``"floor_infeasible"`` (those route to
    the deferred [[DEFICIT-BENEFITS-HANDOFF]]), and on no-CC-debt
    profiles — those paths stay byte-identical.

The optional dollar-less ``near_capacity`` framing band (gap just under
the compression boundary) was scoped as a nice-to-have and is NOT built.

Forward note (banked, not built): the CC input is a *balance*
(``cc_carried_balance``), not a *payment* — a user paying more than the
minimum expresses that by pinning this line to 0/negative; a future
"actual monthly CC payment" input is the cleaner long-run fix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from shared.types import HouseholdProfile

_MONTHS_PER_YEAR: float = 12.0

# Framing-strength knobs [CALIBRATE] — these set the *strength of
# language* (signal_clear vs signal_marginal), never the binary trigger.
# signal_clear when EITHER ratio clears its cut: a large balance relative
# to income ($5k at $30k reads very differently from $5k at $200k), or a
# large gap relative to the budget.
_CC_BALANCE_TO_INCOME_CLEAR: float = 0.10
_GAP_RATIO_CLEAR: float = 0.10

# Trivial-balance suppression floor [CALIBRATE] — a sub-$300 carried
# balance is noise relative to any realistic gap; suppress the signal
# rather than annotate a near-transactor.
_MIN_CC_BALANCE_FLOOR: float = 300.0

_BASIS: str = "cohort_typical_spending_exceeds_take_home"
_SOURCE: str = "inferred_from_soft_deficit"


@dataclass(frozen=True)
class DebtAccumulation:
    """The conditional debt-accumulation annotation (out-of-sum).

    ``applies`` is False (with zeroed quantities) whenever the signal
    does not fire — the regression-safe default every non-signal path
    returns. The dollar figures are the full honest gap; ``framing_state``
    carries the hedge:

      "signal_clear"     — meaningful balance-to-income and/or gap ratio
      "signal_marginal"  — small gap and modest balance-to-income
      "user_pinned"      — user overrode the line (Q7 fixity)
    """

    applies: bool
    monthly_potential_growth: float    # annual / 12 (may be ≤ 0 when user-pinned)
    annual_potential_growth: float     # == compression_gap when signal fires
    basis: str = _BASIS
    source: str = _SOURCE
    framing_state: str = "signal_marginal"
    adjustable: bool = True
    # Transparency on what drove the framing strength (0.0 when not applicable).
    cc_balance_to_income: float = 0.0
    gap_ratio: float = 0.0


def _no_signal() -> DebtAccumulation:
    return DebtAccumulation(
        applies=False,
        monthly_potential_growth=0.0,
        annual_potential_growth=0.0,
    )


def project_debt_accumulation(
    profile: HouseholdProfile,
    *,
    solver_status: str,
    compression_gap: float,
    d_variable_adjusted: float,
) -> DebtAccumulation:
    """Project the conditional debt-accumulation annotation.

    Args:
        profile: The querying household — supplies ``cc_carried_balance``
            (the trigger) and ``gross_income`` (the framing denominator).
        solver_status: ``FeasibilityResult.solver_status``. Only
            ``"soft_constrained"`` can fire (Q6 scope guard).
        compression_gap: ``FeasibilityResult.compression_gap`` —
            max(0, anchor_sum − adjusted_d) on the soft-constrained path.
        d_variable_adjusted: post-debt-service budget, the gap-ratio
            denominator.

    Returns:
        ``DebtAccumulation`` — ``applies=False`` (zeroed) unless the
        signal fires. No solver state is read or mutated; calling this is
        side-effect-free on every allocation path.
    """
    if solver_status != "soft_constrained":
        return _no_signal()
    if compression_gap <= 0.0:
        return _no_signal()
    cc_balance = float(profile.cc_carried_balance)
    if cc_balance < _MIN_CC_BALANCE_FLOOR:
        return _no_signal()

    gap_annual = float(compression_gap)
    gross = float(profile.gross_income)
    balance_to_income = cc_balance / gross if gross > 0 else 0.0
    gap_ratio = (
        gap_annual / float(d_variable_adjusted) if d_variable_adjusted > 0 else 0.0
    )
    framing_state = (
        "signal_clear"
        if (
            balance_to_income >= _CC_BALANCE_TO_INCOME_CLEAR
            or gap_ratio >= _GAP_RATIO_CLEAR
        )
        else "signal_marginal"
    )
    return DebtAccumulation(
        applies=True,
        monthly_potential_growth=gap_annual / _MONTHS_PER_YEAR,
        annual_potential_growth=gap_annual,
        framing_state=framing_state,
        cc_balance_to_income=balance_to_income,
        gap_ratio=gap_ratio,
    )


def apply_debt_accumulation_override(
    base: DebtAccumulation,
    user_annual_value: float,
) -> DebtAccumulation:
    """Q7 fixity for the debt-accumulation line.

    The user pins the annotation to their stated reality — 0 ("I compress
    to fit / I don't accumulate") or negative ("I'm paying the balance
    down") are both legitimate, so the value is deliberately NOT clamped
    at 0. Because the line sits *outside* the four-way sum (Q2/Q3), the
    pin is display-only and cannot break dollar-accounting; it simply
    replaces the conditional prediction with the user's attribution
    (collapse-and-attribute, ``USER-ADJUSTMENT-AUTHORITY``).

    A pin on a non-applicable base is a no-op (there is no line to pin).
    """
    if not base.applies:
        return base
    annual = float(user_annual_value)
    return replace(
        base,
        monthly_potential_growth=annual / _MONTHS_PER_YEAR,
        annual_potential_growth=annual,
        source=base.source + " (user-overridden debt_accumulation)",
        framing_state="user_pinned",
    )
