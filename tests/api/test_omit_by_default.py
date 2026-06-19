"""Omit-by-default behavioral tests (service layer + response).

The six heavy-zero, cohort-mean-meaningless categories (chrty/educ/ocash +
stdint/othint/finpay) are value-layer-zeroed; the displaced dollars flow to the
four-way remainder; the response flags them omit_from_initial_view. Cohort-
predicted debt lines are omitted from the initial view; only user-reported debt
shows. (HEAVY-ZERO-DISTRIBUTION-NEEDS-ELICITATION, locked this build.)

Run from repo root:
    ARTIFACTS_PATH=pipeline/artifacts python3.11 -m pytest tests/api/test_omit_by_default.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("ARTIFACTS_PATH", "pipeline/artifacts")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

import django  # noqa: E402

django.setup()

from apps.api.profiles.display import (  # noqa: E402
    build_display_rollup,
    rollup_to_dict,
)
from apps.api.profiles.services import (  # noqa: E402
    build_household_profile,
    run_profile_analysis,
)
from shared.constants.categories import OMIT_BY_DEFAULT_CATEGORIES  # noqa: E402

_PUMA = "IL_00100"
_OMIT = sorted(OMIT_BY_DEFAULT_CATEGORIES)


def _profile(age=25, income=90_000, size=1, housing=1500.0, **debt):
    return build_household_profile(
        age=age, gross_income=income, puma_code=_PUMA, tenure="RENT",
        housing_cost=housing, household_size=size, savings=0.0, **debt,
    )


def _analyze(profile, use_aggregated=True):
    a = run_profile_analysis(profile, filing_status="single",
                             city_pumas=[_PUMA], use_aggregated=use_aggregated)
    a["display_rollup"] = rollup_to_dict(build_display_rollup(a, profile.tenure.value))
    return a


def _four_way(a):
    committed = a["committed_outflows"]["total_annual"]
    debt = a["debt_service_annual"]
    spend = sum(
        (d.get("feasibility_adjusted", 0.0) or 0.0)
        + (d.get("backfill_inferred", 0.0) or 0.0)
        for d in a["distributions"].values()
    )
    sav = a["residual_assignment"]["savings_investment"]["annual"]
    rem = a["residual_assignment"]["genuine_remainder"]["annual"]
    # Up-direction waterfall (2026-06-10): routed savings-side fills (taxable
    # terminal for the no-contradiction default; 401(k)/IRA/HSA + taxable on
    # the high contradiction) — part of the savings bucket, remainder ≡ 0.
    waterfall = a["savings_waterfall"]["total"]
    take_home = a["d_variable_annual"] + committed
    return take_home, committed + debt + spend + sav + waterfall + rem


@pytest.mark.parametrize("age,income,size", [(25, 90_000, 1), (35, 90_000, 4), (50, 200_000, 2)])
def test_six_categories_allocate_to_zero(age, income, size) -> None:
    a = _analyze(_profile(age=age, income=income, size=size))
    for cat in _OMIT:
        d = a["distributions"][cat]
        assert d["feasibility_adjusted"] == pytest.approx(0.0), f"{cat} not zeroed"


@pytest.mark.parametrize("age,income,size", [(25, 90_000, 1), (35, 90_000, 4), (50, 200_000, 2)])
def test_six_flagged_omit_others_not(age, income, size) -> None:
    a = _analyze(_profile(age=age, income=income, size=size))
    for cat, d in a["distributions"].items():
        expected = cat in OMIT_BY_DEFAULT_CATEGORIES
        assert d["omit_from_initial_view"] is expected, f"{cat} flag wrong"


def test_percentiles_retained_for_future_ux() -> None:
    # Zeroing the allocation must NOT destroy the cohort percentiles (the future
    # "see typical values" additive UX consumes them). At least one of the six
    # keeps a positive p90.
    a = _analyze(_profile())
    assert any(a["distributions"][c]["p90"] > 0 for c in _OMIT)


def test_display_rollup_members_flagged() -> None:
    a = _analyze(_profile())
    flagged = {
        m["category"]
        for t in a["display_rollup"]["topics"]
        for m in t["members"]
        if m.get("omit_from_initial_view")
    }
    assert flagged == OMIT_BY_DEFAULT_CATEGORIES


@pytest.mark.parametrize("age,income,size", [(25, 90_000, 1), (35, 90_000, 4), (50, 200_000, 2)])
def test_four_way_closes_after_zeroing(age, income, size) -> None:
    a = _analyze(_profile(age=age, income=income, size=size))
    take_home, total = _four_way(a)
    assert total == pytest.approx(take_home, abs=1.0)


def test_debt_liabilities_omit_logic() -> None:
    # No debt input → all liability lines omitted (nothing user-reported).
    a = _analyze(_profile())
    libs = a["balance_sheet"]["liabilities"]
    assert all(libs[k]["omit_from_initial_view"] for k in libs)
    # User-reported CC + SL → those two surface; auto/other still omitted.
    b = _analyze(_profile(cc_carried_balance=4000.0, student_loan_payment=200.0))
    blibs = b["balance_sheet"]["liabilities"]
    assert blibs["othdbt"]["omit_from_initial_view"] is False
    assert blibs["stddbt"]["omit_from_initial_view"] is False
    assert blibs["auto_loan"]["omit_from_initial_view"] is True
    assert blibs["other_debt"]["omit_from_initial_view"] is True
    # Four-way still closes with user-reported debt present.
    th, tot = _four_way(b)
    assert tot == pytest.approx(th, abs=1.0)


def test_zeroing_holds_on_disaggregated_path() -> None:
    # The 55-cat fallback path zeros the six too (rule applies on both paths).
    # No display_rollup here — that builder is aggregated-path only.
    a = run_profile_analysis(_profile(), filing_status="single",
                             city_pumas=[_PUMA], use_aggregated=False)
    for cat in _OMIT:
        assert a["distributions"][cat]["feasibility_adjusted"] == pytest.approx(0.0)
        assert a["distributions"][cat]["omit_from_initial_view"] is True
