"""End-to-end pin: the demonstrating profile under the stratification build.

The high-income discretionary ceiling stratification + eatout build
(2026-06-09) + the Build-1.1 monotone max() floor: the NYC $110K/24/RENT/
$1K-balance profile stratifies, deploys into all five positive-anchor targets
(incl. eatout), the remainder collapses from ~$471/mo (pre-build) to ~$160/mo,
savings stays at the balance-informed ~$470/mo (pieces 1+3 untouched; net of
the Stage-4 real NY brackets), and the four-way closes exactly.

The landing point (vs the scoping investigation's $66-91 projection) is
honest: the live cp90_hi is the KERNEL-WEIGHTED cohort's upper half — already
income-conditioned around the profile — a tighter ceiling than the
investigation's unweighted raw-pool top-quartile stratum
(KERNEL-COHORT-STRATUM-TIGHTER-THAN-RAW-POOL). The max() floor (Build 1.1)
keeps the stratification monotone: where the hi-subset p90 dips below broad
(eatout/airshp — noise, not signal), the cap holds at broad instead of
tightening (KERNEL-COHORT-P90-BELOW-BROAD-IS-NOISE-NOT-SIGNAL).

Slow (~12s: full 55-PUMA NYC match). Unit-level coverage lives in
tests/models/test_backfill_stratification.py.
"""
from __future__ import annotations

import json
import os
import sys

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))
django.setup()


def test_demonstrating_profile_validation(capsys) -> None:
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
    bf = res["backfill"]
    ra = res["residual_assignment"]
    with capsys.disabled():
        print()
        print("stratified:", bf["stratified"],
              "| threshold y_eq: %.0f" % bf["cohort_median_y_eq"])
        print("fired:", bf["fired"], "| pool: %.0f/yr" % bf["pool"])
        print("inferred:", {c: round(v) for c, v in bf["inferred"].items()})
        print("s_pers: %.4f  s*: %.4f" % (bf["s_star_personalized"], bf["s_star"]))
        print("savings: %.0f/mo  remainder: %.0f/mo" % (
            ra["savings_investment"]["annual"] / 12,
            ra["genuine_remainder"]["annual"] / 12))
        print("remainder label:", ra["genuine_remainder"]["label"])
        print("framing:", ra["savings_investment"]["framing_state"])
        for c in ["entertainment", "shopping", "eatout", "hotel", "airshp", "recrp"]:
            d = res["distributions"].get(c)
            if d:
                meas = d["feasibility_adjusted"]
                inc = d.get("backfill_inferred", 0.0)
                print("%-13s meas %7.0f  inf %6.0f  total %7.0f  cp90 %7.0f  cp90_hi %7.0f"
                      % (c, meas, inc, meas + inc,
                         d["conditional_p90"], d["conditional_p90_hi"]))
        # Four-way closure
        co = res["committed_outflows"]["total_annual"]
        ds = res["debt_service_annual"]
        spend = sum(d["feasibility_adjusted"] + d.get("backfill_inferred", 0.0)
                    for d in res["distributions"].values())
        sav = ra["savings_investment"]["annual"]
        rem = ra["genuine_remainder"]["annual"]
        d_var_adj = res["d_variable_adjusted"]
        print("d_var_adj %.0f | spend %.0f + savings %.0f + remainder %.0f = %.0f"
              % (d_var_adj, spend, sav, rem, spend + sav + rem))
        print("committed %.0f  debt_service %.0f" % (co, ds))

    # The Phase-3 gates, asserted:
    assert bf["stratified"] is True
    assert bf["fired"] is True
    assert "eatout" in bf["inferred"]
    # Remainder collapses from ~$471/mo (pre-build) — lands at ~$160/mo under
    # the monotone max() floor (was ~$224 at pure substitution), NOT the
    # investigation's $66-91 projection: the live cp90_hi is the
    # KERNEL-WEIGHTED cohort's upper half (already income-conditioned around
    # the profile), a tighter — and more defensible — ceiling than the
    # investigation's unweighted raw-pool top-quartile stratum (which included
    # $200K+ households). See KERNEL-COHORT-STRATUM-TIGHTER-THAN-RAW-POOL.
    rem_mo = ra["genuine_remainder"]["annual"] / 12.0
    assert rem_mo < 220.0, rem_mo
    # savings at the balance-informed level (pieces 1+3); ~$470/mo — the Stage-4
    # real NY brackets (~5% effective at $110K) come in UNDER the old flat
    # 5.5%-on-gross fallback, nudging take-home / d_variable (and this capped
    # savings line) up ~$7/mo from the pre-Stage-4 ~$463 (NYC local unchanged).
    assert abs(sav / 12.0 - 470.0) < 5.0
    # four-way closes: spending + savings + remainder == d_var_adjusted
    assert abs((spend + sav + rem) - d_var_adj) < 2.0

    # Down-direction residual sweep (2026-06-10, locked
    # REMAINDER-ZERO-INVARIANT-DOWN-DIRECTION): this profile is the
    # low-savings-contradiction demonstrating case (signal_pulled_down), so
    # the post-stratification ~$158/mo would-be remainder sweeps entirely
    # into the high-participation elastic sinks and the remainder is
    # IDENTICALLY 0 — the build-gate's core check.
    sweep = res["residual_sweep"]
    assert ra["savings_investment"]["framing_state"] == "signal_pulled_down"
    assert sweep["fired"] is True
    assert sweep["trigger"] == "low_savings_contradiction"
    assert rem == 0.0                                   # the invariant
    assert abs(sum(sweep["swept"].values()) - sweep["total"]) < 1e-6
    # the swept dollars surface as predicted spending on the sinks
    for c in ["eathome", "eatout", "shopping", "entertainment", "household_goods"]:
        assert sweep["swept"].get(c, 0.0) > 0.0, c

    # Mutual exclusivity with the up-direction waterfall (2026-06-10): this
    # is the low-contradiction case — the waterfall must NOT fire (and no
    # committed line carries a predicted_topup).
    assert res["savings_waterfall"]["fired"] is False
    assert all(
        it["predicted_topup_annual"] == 0.0
        for it in res["committed_outflows"]["items"]
    )
