"""Rendered four-way identity test — pins the frontend's sum formula, not just
the backend response closure.

The diagnostic (2026-06-10) found that the existing e2e tests validated the
backend response closes (spend + savings_blend + waterfall_total == d_var_adj)
but left the frontend's rendered formula untested. The rendered formula is:

    take_home == committed_total_annual
               + debt_service_annual
               + Σ(feasibility_adjusted + backfill_inferred)   # spending
               + savings_investment.annual                      # blend
               + savings_waterfall.total                        # routed fills
               + genuine_remainder.annual                       # ≡ 0 for swept/routed

This file pins that identity across all three savings-signal states so any
future field rename or formula drift in either the backend serialization or
the frontend sum breaks a test rather than a rendered output.

Three demonstrating profiles (slow: each requires a full cohort match):
  - High-contradiction  (Santa Clara, $750K savings)  → signal_pulled_up_routed
  - Low-contradiction   (NYC, $1K savings)            → signal_pulled_down
  - No-contradiction    (Chicago, $0 savings)         → signal_confirmed_cohort

Run from repo root:
    .venv/bin/python -m pytest tests/api/test_rendered_four_way.py -v
"""
from __future__ import annotations

import json
import os
import sys

import pytest

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))
django.setup()


def _rendered_four_way(res: dict) -> tuple[float, float]:
    """Construct the four-way sum exactly as the frontend does.

    Returns (take_home, rendered_sum).  The identity is rendered_sum == take_home.
    Uses the exact field names the frontend reads so any rename drifts are caught.
    """
    committed = res["committed_outflows"]["total_annual"]          # un-topped d_var input
    debt = res["debt_service_annual"]
    spend = sum(
        d["feasibility_adjusted"] + d.get("backfill_inferred", 0.0)
        for d in res["distributions"].values()
    )
    savings = res["residual_assignment"]["savings_investment"]["annual"]   # blend, untouched
    waterfall = res["savings_waterfall"]["total"]                          # routed fills
    remainder = res["residual_assignment"]["genuine_remainder"]["annual"]  # ≡ 0 when swept/routed
    take_home = res["d_variable_annual"] + committed
    rendered_sum = committed + debt + spend + savings + waterfall + remainder
    return take_home, rendered_sum


# --------------------------------------------------------------------------- #
# High-contradiction: waterfall fires (signal_pulled_up_routed)                #
# --------------------------------------------------------------------------- #

def test_high_contradiction_rendered_four_way_closes() -> None:
    """Santa Clara 32/$250K/$750K: waterfall routes $1,932/mo → remainder ≡ 0.

    The bug this test guards: the frontend's fourWaySum omitted savings_waterfall.total
    so it was short by exactly the routed amount.  The fix adds waterfall to the sum.
    """
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    with open("pipeline/artifacts/city_puma_map.json") as f:
        cmap = json.load(f)
    pumas = cmap["place"].get("0669084") or cmap["place"]["0673262"]
    profile = HouseholdProfile(
        age=32, gross_income=250_000, puma_code=pumas[0], tenure=Tenure.RENT,
        housing_cost=3_000, household_size=1, savings=750_000,
        auto_loan_payment=700,
    )
    res = run_profile_analysis(profile, city_pumas=pumas)

    # Confirm this is the high-contradiction state (waterfall routed the up-pull).
    assert res["residual_assignment"]["savings_investment"]["framing_state"] == "signal_pulled_up_routed"
    assert res["savings_waterfall"]["fired"] is True
    assert res["savings_waterfall"]["trigger"] == "high_savings_contradiction"
    assert res["residual_sweep"]["fired"] is False

    # remainder ≡ 0 and the waterfall fills carry the dollars.
    assert res["residual_assignment"]["genuine_remainder"]["annual"] == 0.0
    waterfall_total = res["savings_waterfall"]["total"]
    assert waterfall_total > 0

    # Rendered four-way closes to sub-dollar precision.
    take_home, rendered = _rendered_four_way(res)
    assert abs(rendered - take_home) < 1.0, (
        f"Rendered four-way does not close: {rendered:.2f} != {take_home:.2f} "
        f"(Δ={take_home - rendered:.2f}, waterfall={waterfall_total:.2f})"
    )


# --------------------------------------------------------------------------- #
# Low-contradiction: down-sweep fires (signal_pulled_down)                     #
# --------------------------------------------------------------------------- #

def test_low_contradiction_rendered_four_way_closes() -> None:
    """NYC 24/$110K/$1K: down-sweep folds into spending → remainder ≡ 0.

    The down-sweep moves dollars into distributions as backfill_inferred, so
    flatSum includes them.  waterfall.total == 0.  Confirms the frontend's
    flatSum covers sweep dollars (the diagnostic's "no other paths have the gap
    from backend-correct ≠ frontend-renders-correctly" hypothesis).
    """
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    with open("pipeline/artifacts/city_puma_map.json") as f:
        cmap = json.load(f)
    pumas = cmap["place"]["3651000"]
    profile = HouseholdProfile(
        age=24, gross_income=110_000, puma_code=pumas[0], tenure=Tenure.RENT,
        housing_cost=2_500, household_size=1, savings=1_000,
    )
    res = run_profile_analysis(profile, city_pumas=pumas)

    # Confirm this is the low-contradiction state.
    assert res["residual_assignment"]["savings_investment"]["framing_state"] == "signal_pulled_down"
    assert res["residual_sweep"]["fired"] is True
    assert res["residual_sweep"]["trigger"] == "low_savings_contradiction"
    assert res["savings_waterfall"]["fired"] is False

    # remainder ≡ 0 and the waterfall contributes nothing.
    assert res["residual_assignment"]["genuine_remainder"]["annual"] == 0.0
    assert res["savings_waterfall"]["total"] == 0.0

    # Rendered four-way closes — the sweep dollars live in flatSum via backfill_inferred.
    take_home, rendered = _rendered_four_way(res)
    assert abs(rendered - take_home) < 1.0, (
        f"Rendered four-way does not close: {rendered:.2f} != {take_home:.2f} "
        f"(Δ={take_home - rendered:.2f})"
    )


# --------------------------------------------------------------------------- #
# No-contradiction: taxable default (signal_confirmed_cohort)                  #
# --------------------------------------------------------------------------- #

def test_no_contradiction_rendered_four_way_closes() -> None:
    """Chicago 30/$80K/$0 savings: taxable-only default, remainder ≡ 0.

    With no balance signal (savings=0 → collapses to cohort prior), the waterfall
    fires with no_contradiction_default and routes the remainder to taxable savings.
    framing_state stays signal_confirmed_cohort.
    """
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    profile = HouseholdProfile(
        age=30, gross_income=80_000, puma_code="IL_00100", tenure=Tenure.RENT,
        housing_cost=1_500, household_size=1, savings=0,
    )
    res = run_profile_analysis(profile, city_pumas=["IL_00100"])

    framing = res["residual_assignment"]["savings_investment"]["framing_state"]
    # The no-contradiction state keeps signal_confirmed_cohort after the taxable fold.
    assert framing == "signal_confirmed_cohort", (
        f"Expected signal_confirmed_cohort, got {framing!r} — "
        "profile may need savings adjustment to avoid a balance signal"
    )
    assert res["savings_waterfall"]["fired"] is True
    assert res["savings_waterfall"]["trigger"] == "no_contradiction_default"
    assert res["residual_sweep"]["fired"] is False
    assert res["residual_assignment"]["genuine_remainder"]["annual"] == 0.0

    take_home, rendered = _rendered_four_way(res)
    assert abs(rendered - take_home) < 1.0, (
        f"Rendered four-way does not close: {rendered:.2f} != {take_home:.2f} "
        f"(Δ={take_home - rendered:.2f})"
    )


# --------------------------------------------------------------------------- #
# CC-debt profile: paydown fill active, four-way closes                        #
# --------------------------------------------------------------------------- #

def test_cc_debt_rendered_four_way_closes() -> None:
    """Santa Clara 32/$250K/$750K + $8K CC balance: paydown fill appears before
    k401_topup; rendered sum closes to sub-dollar precision; remainder ≡ 0.

    Guards the render-lock discipline (CC-PAYDOWN-PREDICTED-ROUTING): any
    routing change gets a rendered-identity case.  The cc_paydown fill lives
    inside savings_waterfall.total, so the four-way identity is unchanged —
    but the fill must surface in the fills array and the total must include it.
    """
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    with open("pipeline/artifacts/city_puma_map.json") as f:
        cmap = json.load(f)
    pumas = cmap["place"].get("0669084") or cmap["place"]["0673262"]
    profile = HouseholdProfile(
        age=32, gross_income=250_000, puma_code=pumas[0], tenure=Tenure.RENT,
        housing_cost=3_000, household_size=1, savings=750_000,
        auto_loan_payment=700, cc_carried_balance=8_000,
    )
    res = run_profile_analysis(profile, city_pumas=pumas)

    # High-contradiction state (same profile as the baseline test, now with CC debt).
    assert res["residual_assignment"]["savings_investment"]["framing_state"] == "signal_pulled_up_routed"
    assert res["savings_waterfall"]["fired"] is True

    # cc_paydown fill is present and is the FIRST fill (paydown-first ordering).
    fills = res["savings_waterfall"]["fills"]
    assert fills, "Expected at least one fill"
    assert fills[0]["code"] == "cc_paydown", (
        f"Expected cc_paydown first, got {fills[0]['code']!r} — paydown-first ordering violated"
    )

    # paydown amount: min(remainder × 0.40, 8_000)  — the exact value depends on
    # the live remainder, so just assert it's capped correctly.
    paydown_annual = fills[0]["annual"]
    assert paydown_annual > 0.0
    assert paydown_annual <= 8_000.0 + 1e-6

    # k401_topup follows paydown (ordering discipline).
    fill_codes = [f["code"] for f in fills]
    if "k401_topup" in fill_codes:
        assert fill_codes.index("cc_paydown") < fill_codes.index("k401_topup"), (
            "k401_topup appeared before cc_paydown — paydown-first ordering violated"
        )

    # Waterfall total covers the paydown fill.
    waterfall_total = res["savings_waterfall"]["total"]
    assert waterfall_total >= paydown_annual - 1e-6

    # remainder ≡ 0.
    assert res["residual_assignment"]["genuine_remainder"]["annual"] == 0.0

    # Rendered four-way closes.
    take_home, rendered = _rendered_four_way(res)
    assert abs(rendered - take_home) < 1.0, (
        f"Rendered four-way does not close with CC paydown active: "
        f"{rendered:.2f} != {take_home:.2f} (Δ={take_home - rendered:.2f}, "
        f"paydown={paydown_annual:.2f}, waterfall={waterfall_total:.2f})"
    )


# --------------------------------------------------------------------------- #
# Σ fills == waterfall.total (internal waterfall closure, all three states)    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("profile_key", [
    "high_contradiction",
    "low_contradiction",
    "no_contradiction",
])
def test_waterfall_fills_sum_equals_total(profile_key: str) -> None:
    """Waterfall fills sum == waterfall.total for every signal state.

    Catches any future discrepancy between the fills array and the total scalar
    that the four-way sum reads.
    """
    from profiles.services import run_profile_analysis
    from shared.types import HouseholdProfile, Tenure

    with open("pipeline/artifacts/city_puma_map.json") as f:
        cmap = json.load(f)

    if profile_key == "high_contradiction":
        pumas = cmap["place"].get("0669084") or cmap["place"]["0673262"]
        profile = HouseholdProfile(
            age=32, gross_income=250_000, puma_code=pumas[0], tenure=Tenure.RENT,
            housing_cost=3_000, household_size=1, savings=750_000,
            auto_loan_payment=700,
        )
    elif profile_key == "low_contradiction":
        pumas = cmap["place"]["3651000"]
        profile = HouseholdProfile(
            age=24, gross_income=110_000, puma_code=pumas[0], tenure=Tenure.RENT,
            housing_cost=2_500, household_size=1, savings=1_000,
        )
    else:  # no_contradiction
        pumas = ["IL_00100"]
        profile = HouseholdProfile(
            age=30, gross_income=80_000, puma_code="IL_00100", tenure=Tenure.RENT,
            housing_cost=1_500, household_size=1, savings=0,
        )

    res = run_profile_analysis(profile, city_pumas=pumas)
    wf = res["savings_waterfall"]
    fills_sum = sum(f["annual"] for f in wf["fills"])
    assert abs(fills_sum - wf["total"]) < 1e-6, (
        f"{profile_key}: Σ fills {fills_sum:.6f} != waterfall.total {wf['total']:.6f}"
    )
