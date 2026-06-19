"""Tests for the tax modules.

Run from repo root:
    python3.11 -m pytest tests/models/test_tax.py -v
"""

from __future__ import annotations

import pytest

from models.tax.calculator import (
    TaxDetail,
    _compute_federal_via_taxcalc,
    _compute_fica,
    compute_tax,
)
from models.tax.state import state_from_puma, take_home
from shared.types.household import HouseholdProfile
from shared.types.enums import Tenure


def test_fica_basic() -> None:
    # Live FICA: calculator._compute_fica, the function compute_tax actually
    # uses (2026: SS 6.2% to the $176,100 wage base, Medicare 1.45% on all
    # wages, plus the 0.9% Additional Medicare Tax over $200k single). The
    # dormant models.tax.fica twin (2024 base, no Additional Medicare) was
    # removed; this asserts the live path directly, no taxcalc dependency.
    # 50k: below the SS wage base, full 7.65%, no Additional Medicare.
    assert abs(_compute_fica(50_000, "single") - 50_000 * 0.0765) < 1e-6

    # 250k single: SS caps at the 2026 wage base $184,500; Medicare on the full
    # amount; plus 0.9% Additional Medicare on the 50k above the 200k threshold.
    expected_250k = 184_500 * 0.062 + 250_000 * 0.0145 + (250_000 - 200_000) * 0.009
    assert abs(_compute_fica(250_000, "single") - expected_250k) < 1e-6


def test_federal_via_taxcalc_sane() -> None:
    # Federal income tax is computed by taxcalc (2026 law) via the live
    # _compute_federal_via_taxcalc; the dead standalone-bracket models.tax.federal
    # twin was removed. Exact values are oracle-validated in the Gate-1 harness;
    # here we assert the law-stable properties: zero at zero income, strictly
    # increasing in income, and a broad plausible 2026 band for a 100k single.
    # _compute_federal_via_taxcalc now returns a FederalDetail (Stage 6); the
    # closure-bearing net income tax is ``.federal_tax``.
    assert _compute_federal_via_taxcalc(0, "single", 0).federal_tax == 0.0
    f30 = _compute_federal_via_taxcalc(30_000, "single", 0).federal_tax
    f100 = _compute_federal_via_taxcalc(100_000, "single", 0).federal_tax
    f250 = _compute_federal_via_taxcalc(250_000, "single", 0).federal_tax
    assert 0.0 < f30 < f100 < f250, (f30, f100, f250)
    assert 10_000 < f100 < 20_000, f100


def test_refundable_eitc_above_gross() -> None:
    # Point 5: a low earner with kids — refundable EITC + refundable CTC exceed
    # income-tax liability, so taxcalc iitax is NEGATIVE (a refund). The old
    # max(0, iitax) floor swallowed it; now federal_tax < 0 and take-home
    # EXCEEDS gross (REFUNDABLE-CREDIT-RAISES-TAKE-HOME). No state/local here
    # (TX), so the refund flows straight through.
    detail = TaxDetail(n_children_under_17=2, n_children_under_18=2, eic_children=2)
    br = compute_tax(26_000, "head_of_household", puma_code="TX_01000",
                     num_dependents=2, detail=detail)
    assert br.federal_tax < 0.0, br.federal_tax       # net refund
    assert br.take_home > 26_000.0, br.take_home      # take-home above gross
    # The breakdown still reconciles exactly: take_home == gross - total_tax.
    assert abs(br.take_home - (26_000.0 - br.total_tax)) < 1e-6


def test_pretax_wedge_lowers_tax() -> None:
    # The pre-tax wedge reduces the income-tax base (401k+§125) and the FICA
    # base (§125 only), so both federal income tax and FICA drop vs no wedge.
    base = compute_tax(120_000, "single", puma_code="CA_03761")
    wedged = compute_tax(
        120_000, "single", puma_code="CA_03761",
        detail=TaxDetail(pretax_income_tax_excludable=15_000.0,
                         pretax_fica_excludable=4_000.0),
    )
    assert wedged.federal_tax < base.federal_tax
    assert wedged.fica < base.fica
    assert wedged.take_home > base.take_home


def test_take_home_reasonable() -> None:
    th = take_home(65_000, "CA_03761", "single")
    assert th > 40_000, f"take_home unexpectedly low: {th}"
    assert th < 65_000, f"take_home >= gross: {th}"


def test_state_from_puma() -> None:
    assert state_from_puma("CA_03761") == "CA"
    assert state_from_puma("AK_00101") == "AK"
    assert state_from_puma("NY_03810") == "NY"


def test_local_tax_columbus_place_fips() -> None:
    # Columbus, OH flat rate 2.5% — place FIPS "3918000".
    result = compute_tax(65_000, "single", puma_code="OH_00101", place_fips="3918000")
    assert result.city_tax == pytest.approx(65_000 * 0.025)

    # take_home must be strictly lower than the same call without place_fips.
    no_local = compute_tax(65_000, "single", puma_code="OH_00101")
    assert result.take_home < no_local.take_home


def test_no_local_tax_for_ca() -> None:
    # CA PUMA, no FIPS supplied → no local tax match → city_tax == 0.
    result = compute_tax(80_000, "single", puma_code="CA_03761")
    assert result.city_tax == 0.0


def test_nyc_via_puma_unchanged() -> None:
    # NYC back-compat: puma_code "NY_04103" (no FIPS) must still produce the
    # correct NYC city tax.
    #
    # NYC single brackets (oracle from test_local_tax.py / NYC Dept. of Finance):
    #   0 – 12,000      @ 3.078% → 12,000 × 0.03078 =   369.36
    #   12,000 – 25,000 @ 3.762% → 13,000 × 0.03762 =   489.06
    #   25,000 – 50,000 @ 3.819% → 25,000 × 0.03819 =   954.75
    #   50,000 – 100,000 @ 3.876% → 50,000 × 0.03876 = 1,938.00
    #   Total                                           = 3,751.17
    result = compute_tax(100_000, "single", puma_code="NY_04103")
    assert result.city_tax == pytest.approx(3_751.17)


def test_household_profile_fips_defaults() -> None:
    # Constructing HouseholdProfile without FIPS fields gives empty strings.
    profile = HouseholdProfile(
        age=35,
        gross_income=80_000.0,
        puma_code="CA_03761",
        tenure=Tenure.RENT,
        housing_cost=2_000.0,
        household_size=1,
    )
    assert profile.place_fips == ""
    assert profile.county_fips == ""
