"""End-to-end pin: the Santa Clara profile under the up-direction waterfall.

The up-direction tax-advantaged waterfall build (2026-06-10, locked
REMAINDER-ZERO-INVARIANT-UP-DIRECTION): the 32yo/$250K/$750K-balance/$3K-rent/
$700-auto Santa Clara profile is the high-savings-contradiction demonstrating
case — the $1,932/mo would-be remainder routes through the statutory waterfall
(401(k) +$478/mo maxed → IRA +$625/mo backdoor → HSA +$287/mo maxed → taxable
+$542/mo), the retirement line displays maxed at ~$2,042/mo via
predicted_topup (post-allocation transfer — the d_var input total is
UNCHANGED, no circularity), remainder ≡ 0, four-way closes exactly. The
savings blend stays at the cohort-capped ~$3,229/mo (net of the Stage-4 real
CA brackets and Stage-5 SALT itemization; route-the-remainder-not-the-blend).

Slow (~10s: full Santa Clara county match). Unit-level coverage in
tests/models/test_savings_waterfall.py.
"""
from __future__ import annotations

import json
import os
import sys

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))
django.setup()


def test_santa_clara_waterfall_demonstrating_profile(capsys) -> None:
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
    ra = res["residual_assignment"]
    wf = res["savings_waterfall"]
    co = res["committed_outflows"]
    fills = {f["code"]: f for f in wf["fills"]}

    with capsys.disabled():
        print()
        print("framing:", ra["savings_investment"]["framing_state"])
        print("waterfall:", wf["trigger"], "| total %.0f/yr | limits %d" % (wf["total"], wf["limits_year"]))
        for f in wf["fills"]:
            print("  %-16s +%7.0f/yr  maxed=%s %s" % (f["code"], f["annual"], f["maxed"], f["mechanism"]))
        for it in co["items"]:
            if it["predicted_topup_annual"] > 0:
                print("  topup %-26s -> display %.0f/mo" % (it["code"], it["display_annual"] / 12))

    # The trigger + the routed framing transition.
    assert ra["savings_investment"]["framing_state"] == "signal_pulled_up_routed"
    assert wf["fired"] is True
    assert wf["trigger"] == "high_savings_contradiction"
    assert wf["limits_year"] == 2026

    # The fill walk: order, max-out, backdoor mechanism.
    assert [f["code"] for f in wf["fills"]] == [
        "k401_topup", "ira", "hsa_topup", "taxable_savings",
    ]
    assert fills["k401_topup"]["maxed"] is True
    assert fills["ira"]["maxed"] is True
    assert fills["ira"]["mechanism"] == "backdoor_roth"
    assert fills["hsa_topup"]["maxed"] is True
    assert fills["ira"]["annual"] == 7_500.0

    # The mutual exclusivity: the down-direction sweep did NOT fire here.
    assert res["residual_sweep"]["fired"] is False

    # remainder ≡ 0 (the invariant) + Σ fills == the routed remainder.
    assert ra["genuine_remainder"]["annual"] == 0.0
    assert abs(sum(f["annual"] for f in wf["fills"]) - wf["total"]) < 1e-6

    # The post-allocation transfer: the retirement line displays maxed
    # (~$2,042/mo) while the d_var-input committed total is UNCHANGED by the
    # top-ups (no circularity).
    ret = next(it for it in co["items"] if it["code"] == "retirement_contribution")
    assert abs(ret["display_annual"] / 12.0 - 2_042.0) < 2.0
    assert ret["display_annual"] == ret["annual"] + ret["predicted_topup_annual"]
    assert co["total_with_topups_annual"] > co["total_annual"]

    # The savings blend sits at the cohort-capped level (route-the-remainder-
    # not-the-blend). ~$3,233/mo, net of three rigorous-tax stages: Stage-4 REAL
    # CA brackets (~7.7% eff at $250K > the old flat 5.5%), Stage-5 SALT itemization
    # (this single filer's ~$19K CA income tax exceeds the $16,100 standard
    # deduction → taxcalc itemizes), and the Stage-7 fix removing the double-counted
    # 0.9% Additional Medicare Tax from federal (it lived in both taxcalc iitax AND
    # our FICA line; ~$450/yr at this income → ~+$4/mo to the capped savings line).
    # Update on rate/law refresh.
    assert abs(ra["savings_investment"]["annual"] / 12.0 - 3_233.0) < 5.0

    # Four-way closes exactly: spending + blend savings + waterfall == d_var_adj.
    spend = sum(d["feasibility_adjusted"] + d.get("backfill_inferred", 0.0)
                for d in res["distributions"].values())
    closure = spend + ra["savings_investment"]["annual"] + wf["total"]
    assert abs(closure - res["d_variable_adjusted"]) < 2.0
