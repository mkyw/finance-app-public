"""Schedule A itemizable deductions derived from the model's own predictions.

Federal tax goes through taxcalc, which compares the standard deduction to the
sum of itemized deductions and applies the larger — plus the OBBBA-2026 SALT cap
(``ID_AllTaxes_c`` = $40,400 single/MFJ/HoH for 2026, with the high-earner
phase-down to a $10k floor), the 7.5%-of-AGI medical floor, and the 60%-of-AGI
cash-charity limit — ALL internally. So this module's only job is to populate the
raw Schedule A line items from what the app already predicts; taxcalc does the
arithmetic. We never compute the cap or the standard-vs-itemized choice here.

LEAN LOW (``ITEMIZABLE-PREDICTION-LEANS-LOW``). Itemizables REDUCE tax, so
over-predicting them lowers tax → raises take-home → over-states slack, the
forbidden direction (the mirror image of the spending side's
``OVER-PREDICTION-IS-THE-SAFE-DIRECTION``). Every choice here is the conservative
(under-stating) one:
  * mortgage interest — the locked-#5 housing pin splits owner housing 70/30
    interest/principal for the *spending* allocation; for the *tax* deduction we
    use a lower current-year interest share of the annual housing payment
    (``_LEAN_LOW_MORTGAGE_INTEREST_FRAC``), ~the lifetime-average interest
    fraction of a 30-yr loan — deliberately below the 70% spending split.
  * property tax / medical — the value-scaled cohort MEDIAN (``p50``), not the
    mean or an upper percentile; both under-state for high-value HCOL homes (no
    home-value input exists — see the forward-note below).
  * charity — the raw cohort ``p50`` (heavy-zero category; ``p50`` is typically
    $0 and the value-layer anchor is zeroed under ``OMIT_BY_DEFAULT``, so the
    retained raw percentile is the only signal).

The inputs are LOOP-INVARIANT: mortgage interest comes from the profile's
housing cost (an input), the cohort medians are fixed at match time
(pre-allocation), and the SALT income component is the committed-baseline
state+local tax (constant across the pre-tax fixed point). So the Schedule A is
computed ONCE and fed unchanged into every federal pass — no circularity with
the allocation/`d_variable` it ultimately feeds.

Forward-notes (Gate-2 prediction accuracy, not Gate-1 arithmetic):
  * ``ITEMIZABLE-MORTGAGE-INHERITS-HOUSING-MVP`` — mortgage interest inherits the
    locked-#5 70/30 housing pin (``flagged to revisit``) and the housing-cost
    semantics (P&I vs PITI is ambiguous in the input); the lean-low fraction is
    the conservative hedge until a real amortization / loan-age model lands.
  * ``ITEMIZABLE-PROPERTY-TAX-NEEDS-HOME-VALUE`` — property tax from the cohort
    median under-states HCOL homes; an accurate figure needs a home-value input
    the app does not collect.
  * ``ITEMIZABLE-CHARITY-ABOVE-THE-LINE`` — OBBBA-2026 grants standard-deduction
    filers an above-the-line cash-charity deduction ($1k single / $2k MFJ), which
    taxcalc applies, so charity reduces tax on the standard path too (not only when
    itemizing). The cohort ``chrty`` p50 is structurally $0 for ~all cohorts, so the
    exposure is negligible today; if charity ever moves off cohort-median, re-confirm
    the lean-low direction — a positive charity would then reduce tax on every path.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.types import Tenure

# Conservative current-year mortgage-interest share of the annual housing
# payment for owners. Below the locked-#5 70/30 spending split (which tends high
# on interest for seasoned loans), ~the lifetime-average interest fraction of a
# 30-yr loan at typical rates; under-stating is the safe direction for a
# tax-reducing quantity (ITEMIZABLE-PREDICTION-LEANS-LOW).
_LEAN_LOW_MORTGAGE_INTEREST_FRAC: float = 0.50


@dataclass(frozen=True)
class ScheduleA:
    """Raw Schedule A itemizable line items (pre-cap, pre-floor) fed to taxcalc.

    Field → taxcalc Records column:
      * ``mortgage_interest`` → ``e19200`` (home mortgage interest)
      * ``property_tax``      → ``e18500`` (real-estate taxes; SALT-capped)
      * ``state_local_income_tax`` → ``e18400`` (state+local income tax;
        SALT-capped). Filled by ``compute_tax`` from the computed state+local
        tax — NOT here (it is not a cohort prediction).
      * ``charity``           → ``e19800`` (cash charitable contributions)
      * ``medical``           → ``e17500`` (medical expenses; taxcalc applies the
        7.5%-of-AGI floor)
    """

    mortgage_interest: float = 0.0
    property_tax: float = 0.0
    state_local_income_tax: float = 0.0
    charity: float = 0.0
    medical: float = 0.0


def derive_schedule_a(profile, distributions) -> ScheduleA:
    """Derive the cohort/profile-based Schedule A itemizables (everything except
    the SALT income component, which ``compute_tax`` fills from the computed
    state+local tax). Leans low on every line.

    Args:
        profile: the ``HouseholdProfile`` (for tenure + housing cost).
        distributions: the match-time ``dict[str, SpendingDistribution]``
            (value-scaled percentiles, fixed pre-allocation).
    """
    if getattr(profile, "tenure", None) == Tenure.OWN:
        annual_housing = max(0.0, float(profile.housing_cost)) * 12.0
        mortgage_interest = _LEAN_LOW_MORTGAGE_INTEREST_FRAC * annual_housing
    else:
        mortgage_interest = 0.0  # renters have no deductible mortgage interest

    def _p50(code: str) -> float:
        rec = distributions.get(code)
        return max(0.0, float(rec.p50)) if rec is not None else 0.0

    return ScheduleA(
        mortgage_interest=mortgage_interest,
        property_tax=_p50("ptaxp"),   # cohort median; RENT cohort → ~0
        charity=_p50("chrty"),        # raw retained percentile (anchor is zeroed)
        medical=_p50("health"),       # taxcalc applies the 7.5%-AGI floor
    )
