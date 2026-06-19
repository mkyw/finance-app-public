"""Passive benefits eligibility screen.

Takes a :class:`HouseholdProfile` and returns a list of programs the
household may qualify for. This is a heads-up — the result always uses
"may qualify" framing, never "you qualify" or "you are eligible".

Covered programs (see ``models.benefits.programs.PROGRAMS``):
    SNAP, EITC, Medicaid, ACA Marketplace subsidies, LIHEAP.

The screen iterates all five in a fixed order and omits programs whose
gross-income threshold is clearly exceeded.
"""

from __future__ import annotations

from shared.constants.programs import (
    FPL_2024_ADDITIONAL_PERSON,
    FPL_2024_CONTIGUOUS,
    MEDICAID_NON_EXPANSION_STATES,
)
from shared.types import HouseholdProfile

from models.benefits.programs import PROGRAMS, BenefitMatch
from models.tax.state import state_from_puma

_MAY_QUALIFY: str = "may qualify"

# 2024 EITC maximum credit (IRS Rev. Proc. 2023-34 Table 4).
_EITC_MAX_CREDIT_2024: float = 7_830.0

# 2024 EITC earned-income caps, single filer (no qualifying children,
# 1 child, 2 children, 3+). MFJ adds ~$6,920 to these caps.
_EITC_CAPS_SINGLE: dict[int, float] = {
    0: 18_591.0,
    1: 49_084.0,
    2: 55_768.0,
    3: 59_899.0,
}
_EITC_MFJ_BONUS: float = 6_920.0


def _fpl(household_size: int) -> float:
    """2024 FPL (contiguous US + DC) for the given household size.

    Sizes beyond 8 add ``FPL_2024_ADDITIONAL_PERSON`` per extra person.
    """
    if household_size <= 0:
        raise ValueError(f"household_size must be >= 1, got {household_size}")
    if household_size in FPL_2024_CONTIGUOUS:
        return float(FPL_2024_CONTIGUOUS[household_size])
    extra = household_size - 8
    return float(FPL_2024_CONTIGUOUS[8]) + extra * FPL_2024_ADDITIONAL_PERSON


def _framing(program_name: str, detail: str | None = None) -> str:
    """Build a 'may qualify' sentence. Appends an optional detail clause."""
    base = f"You may qualify for {program_name}."
    if detail:
        return f"{base} {detail}"
    return base


def _confidence_near(gross: float, threshold: float) -> str:
    """'likely' when well under threshold, 'possible' when within 10%."""
    if threshold <= 0:
        return "possible"
    margin = (threshold - gross) / threshold
    if margin >= 0.10:
        return "likely"
    return "possible"


def screen(
    profile: HouseholdProfile,
    filing_status: str = "single",
    has_utility_costs: bool = True,
) -> list[BenefitMatch]:
    """Run all five program checks; return matches that pass each gate."""
    matches: list[BenefitMatch] = []
    gross = float(profile.gross_income)
    hh_size = int(profile.household_size)
    fpl = _fpl(hh_size)
    state = state_from_puma(profile.puma_code)

    # --- SNAP --------------------------------------------------------------
    snap_threshold = 1.30 * fpl
    if gross <= snap_threshold:
        if hh_size == 1:
            lo, hi = 50.0, 292.0
        else:
            lo, hi = 100.0, 535.0
        matches.append(
            BenefitMatch(
                program_name=PROGRAMS["SNAP"]["name"],
                estimated_monthly_min=lo,
                estimated_monthly_max=hi,
                confidence=_confidence_near(gross, snap_threshold),
                enrollment_url=PROGRAMS["SNAP"]["enrollment_url"],
                framing=_framing(PROGRAMS["SNAP"]["name"]),
            )
        )

    # --- EITC --------------------------------------------------------------
    # MVP: assume 0 qualifying children unless the product extends the
    # profile. Use the cap row that matches; add MFJ bonus when relevant.
    qualifying_children = 0  # TODO: surface as a profile field later
    eitc_cap = _EITC_CAPS_SINGLE[min(qualifying_children, 3)]
    if filing_status == "married_filing_jointly":
        eitc_cap += _EITC_MFJ_BONUS
    if 0 < gross <= eitc_cap:
        matches.append(
            BenefitMatch(
                program_name=PROGRAMS["EITC"]["name"],
                estimated_monthly_min=100.0 / 12.0,
                estimated_monthly_max=_EITC_MAX_CREDIT_2024 / 12.0,
                # Actual amount needs full tax return — surface as check.
                confidence="check",
                enrollment_url=PROGRAMS["EITC"]["enrollment_url"],
                framing=_framing(
                    PROGRAMS["EITC"]["name"],
                    "Actual credit depends on your full tax return.",
                ),
            )
        )

    # --- Medicaid ----------------------------------------------------------
    is_expansion_state = state not in MEDICAID_NON_EXPANSION_STATES
    medicaid_threshold = 1.38 * fpl if is_expansion_state else 1.00 * fpl
    if gross <= medicaid_threshold:
        matches.append(
            BenefitMatch(
                program_name=PROGRAMS["MEDICAID"]["name"],
                estimated_monthly_min=0.0,
                estimated_monthly_max=0.0,  # coverage value, not cash
                confidence=_confidence_near(gross, medicaid_threshold),
                enrollment_url=PROGRAMS["MEDICAID"]["enrollment_url"],
                framing=_framing(
                    PROGRAMS["MEDICAID"]["name"],
                    "Eligibility rules vary by state — check your state's site.",
                ),
            )
        )

    # --- ACA Marketplace subsidies ----------------------------------------
    # 100%-400% FPL: premium tax credit band.
    # <100% in a non-expansion state: coverage gap — flag as "check".
    aca_floor = 1.00 * fpl
    aca_ceiling = 4.00 * fpl
    if aca_floor <= gross <= aca_ceiling:
        matches.append(
            BenefitMatch(
                program_name=PROGRAMS["ACA"]["name"],
                estimated_monthly_min=0.0,
                estimated_monthly_max=500.0,
                confidence="possible",
                enrollment_url=PROGRAMS["ACA"]["enrollment_url"],
                framing=_framing(
                    PROGRAMS["ACA"]["name"],
                    "Actual subsidy depends on the plan you pick.",
                ),
            )
        )
    elif gross < aca_floor and not is_expansion_state:
        # Coverage gap: under 100% FPL in a non-expansion state.
        matches.append(
            BenefitMatch(
                program_name=PROGRAMS["ACA"]["name"],
                estimated_monthly_min=0.0,
                estimated_monthly_max=0.0,
                confidence="check",
                enrollment_url=PROGRAMS["ACA"]["enrollment_url"],
                framing=_framing(
                    PROGRAMS["ACA"]["name"],
                    "Your state has not expanded Medicaid — check "
                    "marketplace options carefully.",
                ),
            )
        )

    # --- LIHEAP ------------------------------------------------------------
    if has_utility_costs:
        liheap_threshold = 1.50 * fpl
        if gross <= liheap_threshold:
            matches.append(
                BenefitMatch(
                    program_name=PROGRAMS["LIHEAP"]["name"],
                    # $200-$500 annually -> ~$17-$42/month
                    estimated_monthly_min=200.0 / 12.0,
                    estimated_monthly_max=500.0 / 12.0,
                    confidence="possible",
                    enrollment_url=PROGRAMS["LIHEAP"]["enrollment_url"],
                    framing=_framing(
                        PROGRAMS["LIHEAP"]["name"],
                        "State agencies administer LIHEAP — amounts vary.",
                    ),
                )
            )

    # Defence-in-depth: every match must carry the 'may qualify' phrase.
    for m in matches:
        assert _MAY_QUALIFY in m.framing, (
            f"framing rule violated for {m.program_name!r}: {m.framing!r}"
        )
    return matches
