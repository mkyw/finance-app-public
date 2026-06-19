"""Convergence-as-a-property gate for the Stage-3b pre-tax fixed point.

The waterfall's traditional-401(k)/HSA top-ups reduce taxable income on the next
pass, so tax ↔ take_home ↔ allocation is a genuine fixed point. The interior
converges trivially; the load-bearing question is whether it is a contraction at
the CLIFFS — bracket edges, the Additional-Medicare / SS-wage-base thresholds,
the EITC/CTC knees — where a discrete tax outcome could flip a contribution and
risk oscillation. This sweep runs the full pipeline across a boundary-dense set
× filing status × household size × {high-savings up-contradiction, no-signal},
and asserts:

  * it CONVERGES (run_profile_analysis hard-errors on non-convergence, so a
    raise here is a finding, surfaced as a test failure — never silently
    accepted);
  * in a bounded number of passes (<= _CONVERGE_BUDGET, well under
    _MAX_PRETAX_ITERS);
  * the four-way closes — run_profile_analysis asserts it EVERY iteration
    internally (an AssertionError surfaces here), and we re-check it from the
    response at the converged state;
  * the low-earner refundable-EITC case converges trivially AND take-home
    exceeds gross (REFUNDABLE-CREDIT-RAISES-TAKE-HOME).

Parquet-dependent (full match per profile); runs locally, deselected in CI until
the population artifact is fetchable there (CI-MODEL-COVERAGE-GAP).
"""
from __future__ import annotations

import json
import os
import sys

import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))
django.setup()

# The full grid is a clean contraction everywhere: no-top-up profiles settle in 1
# pass, most in 3, and the high-feedback corner (low-income MFJ-size-4 with a large
# waterfall top-up — the biggest initial take_home perturbation, ~$573) in 6, with
# monotone geometric deltas (ratio ~0.3, no oscillation). The convergence band is
# relative (max($1, 1e-4 x take_home)) so the criterion is dimensionally sound
# across the 50x income range. Budget 7 = the characterized max (6) + 1 pass of
# margin for these (clear up-contradiction) cases; the hard cap
# (_MAX_PRETAX_ITERS) is the real non-convergence gate. NOTE: this grid uses a
# CLEAR up-contradiction (savings = 4x income); the slowest contraction overall
# is the BORDERLINE saver (~8 passes — the regime-pin makes it converge instead
# of 2-cycling; PRETAX-FRAMING-LOOP-INVARIANT), which this grid does not target.
_CONVERGE_BUDGET = 7

# Boundary-dense income points: 2026 federal bracket edges (TCJA-extended,
# single) ±$1, the Additional-Medicare thresholds (200k/250k), the SS wage base
# (184,500), and the CTC phase-out knees (200k/400k) — the cliffs where a
# discrete tax flip could destabilize the iteration. Plus the Stage-5
# standard-vs-itemized crossovers (217,271 single / 434,543 MFJ): for these CA
# RENT profiles SALT (state income tax, no mortgage) equals the standard
# deduction there, so taxcalc flips standard→itemized — a NEW kink the pre-tax
# loop must still converge across. Plus a few interior points.
_CLIFF_INCOMES = [
    11_925, 11_926, 30_000, 48_474, 48_476, 60_000, 103_349, 103_351,
    150_000, 184_499, 184_501, 197_300, 199_999, 200_001, 217_271, 250_000,
    250_001, 300_000, 400_001, 434_543, 626_350,
]


def _puma_for(cmap: dict, state: str) -> str:
    """First PUMA whose STATE_PUMA prefix matches ``state`` (e.g. 'CA')."""
    for pumas in cmap["place"].values():
        for p in pumas:
            if p.startswith(state + "_"):
                return p
    raise AssertionError(f"no PUMA found for state {state}")


def _check_converges(res: dict, label: str) -> None:
    conv = res["tax_convergence"]
    assert conv["converged"] is True, f"{label}: did not converge"
    assert conv["iterations"] <= _CONVERGE_BUDGET, (
        f"{label}: took {conv['iterations']} passes (> {_CONVERGE_BUDGET}) — "
        f"slow/near-oscillating contraction"
    )
    # Re-check the four-way at the converged state (aggregated primary only —
    # the invariant's scope; run_profile_analysis already asserts it per-pass).
    if res["solver_status"] == "primary":
        spend = sum(
            d["feasibility_adjusted"] + d.get("backfill_inferred", 0.0)
            for d in res["distributions"].values()
        )
        closure = (
            spend
            + res["residual_assignment"]["savings_investment"]["annual"]
            + (res["savings_waterfall"]["total"] if res["savings_waterfall"]["fired"] else 0.0)
        )
        dva = res["d_variable_adjusted"]
        assert abs(closure - dva) < max(2.0, abs(dva) * 1e-6), (
            f"{label}: four-way closure {closure:.2f} != d_var_adj {dva:.2f}"
        )
        assert abs(res["residual_assignment"]["genuine_remainder"]["annual"]) < 2.0, label


@pytest.fixture(scope="module")
def cmap() -> dict:
    with open("pipeline/artifacts/city_puma_map.json") as f:
        return json.load(f)


def test_fixed_point_converges_across_cliffs(cmap) -> None:
    """The iterating case: high-savings (up-contradiction → waterfall fires →
    real 401(k)/HSA top-ups feed back) across the cliffs × filing × size."""
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    ca = _puma_for(cmap, "CA")  # CASDI-uncapped state — the largest payroll wedge
    cases = [
        ("single", 1, 38),
        ("married_filing_jointly", 4, 42),
    ]
    checked = 0
    for income in _CLIFF_INCOMES:
        for filing, size, age in cases:
            profile = HouseholdProfile(
                age=age, gross_income=float(income), puma_code=ca, tenure=Tenure.RENT,
                housing_cost=2_500, household_size=size,
                savings=float(income) * 4.0,  # high balance → up-contradiction
            )
            res = run_profile_analysis(profile, filing_status=filing, city_pumas=[ca])
            _check_converges(res, f"CA {filing} size={size} income={income} high-savings")
            checked += 1
    assert checked == len(_CLIFF_INCOMES) * len(cases)


def test_fixed_point_trivial_when_no_topups(cmap) -> None:
    """No-signal profiles (savings=0 → no up-waterfall → no pre-tax top-ups)
    converge in a single pass — the wedge never changes."""
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    tx = _puma_for(cmap, "TX")  # no state income tax / payroll — isolates federal
    for income in (30_000, 90_000, 200_001, 300_000):
        profile = HouseholdProfile(
            age=40, gross_income=float(income), puma_code=tx, tenure=Tenure.RENT,
            housing_cost=1_800, household_size=2, savings=0.0,
        )
        res = run_profile_analysis(profile, filing_status="married_filing_jointly",
                                   city_pumas=[tx])
        _check_converges(res, f"TX no-signal income={income}")
        assert res["tax_convergence"]["iterations"] == 1, (
            f"no-topup profile should converge in 1 pass, got "
            f"{res['tax_convergence']['iterations']}"
        )


def test_fixed_point_converges_at_borderline_saver(cmap) -> None:
    """Regression for the borderline-saver 2-cycle (PRETAX-FRAMING-LOOP-INVARIANT).

    The up-waterfall fires iff the reported balance implies saving ABOVE cohort
    (``s_user_implied = balance / years / d_var_adj > s_cohort``). Its own
    traditional-401(k)/HSA pre-tax top-up RAISES ``d_var_adj`` on the next pass,
    which LOWERS ``s_user_implied`` — so for a balance landing right at the
    boundary the waterfall toggled on/off between passes: a discontinuous map
    with NO fixed point, the take_home 2-cycled, and run_profile_analysis raised
    after the iteration cap (a spurious 500 for a perfectly ordinary profile).
    The fix pins the savings REGIME to the topup-free committed baseline, making
    the map a clean contraction. This sweep walks a balance band that straddles
    the firing boundary in FINE steps (finer than the old ~$5k failure band, so a
    revert is caught): every point must converge, and the band must actually
    cross the boundary (else a data refresh moved it and the guard silently
    lapsed — assert both regimes appear so that fails loudly instead).

    The unfixed code raised RuntimeError on the boundary points → this test fails
    on a revert; with the fix every point is a bounded contraction.
    """
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    ca = _puma_for(cmap, "CA")
    income = 150_000.0
    iters_seen: list[int] = []
    # $280k–$330k brackets the characterized framing boundary (~$302k–$303k for
    # this profile); $2k steps are finer than the boundary's 2-cycle band.
    for savings in range(280_000, 330_001, 2_000):
        profile = HouseholdProfile(
            age=38, gross_income=income, puma_code=ca, tenure=Tenure.RENT,
            housing_cost=2_500, household_size=1, savings=float(savings),
        )
        # A raise here (non-convergence) is the regression — surfaced as failure.
        res = run_profile_analysis(profile, filing_status="single", city_pumas=[ca])
        _check_converges(res, f"CA single income={income} borderline savings={savings}")
        iters_seen.append(res["tax_convergence"]["iterations"])

    # The band must straddle the regime boundary: at least one sub-boundary point
    # (no top-up → 1 pass) and at least one above-boundary point (waterfall fires
    # → iterates). If both aren't present the boundary drifted out of the band and
    # this regression guard is no longer exercising the 2-cycle — fail to prompt a
    # band update rather than pass vacuously.
    assert min(iters_seen) == 1 and max(iters_seen) > 1, (
        f"borderline band did not straddle the firing boundary (iters "
        f"{min(iters_seen)}..{max(iters_seen)}) — the framing boundary moved; "
        f"recenter the savings band on it"
    )


def test_refundable_low_earner_converges_and_exceeds_gross(cmap) -> None:
    """The refundable-EITC low earner: converges trivially (little/no top-up
    room) and take-home exceeds gross (the refund flows through the loop and the
    four-way still closes)."""
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    tx = _puma_for(cmap, "TX")
    profile = HouseholdProfile(
        age=34, gross_income=26_000, puma_code=tx, tenure=Tenure.RENT,
        housing_cost=1_100, household_size=3, savings=0.0,
    )
    res = run_profile_analysis(profile, filing_status="head_of_household", city_pumas=[tx])
    _check_converges(res, "TX refundable low-earner")
    take_home = res["d_variable_annual"] + res["committed_outflows"]["total_annual"]
    assert take_home > 26_000.0, f"refundable take-home {take_home:.0f} should exceed gross"
