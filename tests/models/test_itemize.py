"""Unit tests for the Stage-5 Schedule A itemizable derivation.

Parquet-free: ``derive_schedule_a`` takes a profile + a plain distributions
mapping; the ``compute_tax`` itemization checks use taxcalc + the state artifact
only (no synthetic population). Run from repo root:
    .venv/bin/python -m pytest tests/models/test_itemize.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

from models.tax.calculator import TaxDetail, compute_tax
from models.tax.itemize import (
    _LEAN_LOW_MORTGAGE_INTEREST_FRAC,
    ScheduleA,
    derive_schedule_a,
)
from shared.types import HouseholdProfile, Tenure


def _dists(**p50s: float) -> dict:
    return {k: SimpleNamespace(p50=v) for k, v in p50s.items()}


def _profile(tenure: Tenure, housing_cost: float = 2_000.0) -> HouseholdProfile:
    return HouseholdProfile(
        age=40, gross_income=200_000, puma_code="CA_00001", tenure=tenure,
        housing_cost=housing_cost, household_size=1,
    )


def test_owner_mortgage_interest_is_lean_low_fraction_of_housing() -> None:
    # mortgage interest = lean-low frac × housing×12; the cohort mrtgip p50 is NOT
    # used (the housing pin is the basis), so it is irrelevant here.
    p = _profile(Tenure.OWN, housing_cost=5_000)
    sa = derive_schedule_a(p, _dists(ptaxp=2_000, chrty=0, health=4_000))
    assert abs(sa.mortgage_interest - _LEAN_LOW_MORTGAGE_INTEREST_FRAC * 60_000) < 1e-6
    assert sa.property_tax == 2_000
    assert sa.medical == 4_000
    # lean-low: well below the locked-#5 70% spending split (0.70 × 60000 = 42000)
    assert sa.mortgage_interest < 0.70 * 60_000


def test_renter_has_no_mortgage_interest() -> None:
    p = _profile(Tenure.RENT, housing_cost=5_000)
    sa = derive_schedule_a(p, _dists(ptaxp=0, chrty=0, health=4_000))
    assert sa.mortgage_interest == 0.0


def test_state_local_income_left_for_compute_tax() -> None:
    # derive_schedule_a never fills the SALT income line — compute_tax does, from
    # the computed state+local tax.
    p = _profile(Tenure.OWN, housing_cost=4_000)
    sa = derive_schedule_a(p, _dists(ptaxp=1_000))
    assert sa.state_local_income_tax == 0.0


def test_missing_codes_and_negatives_clamp_to_zero() -> None:
    p = _profile(Tenure.OWN, housing_cost=3_000)
    assert derive_schedule_a(p, {}).property_tax == 0.0  # missing code → 0
    sa = derive_schedule_a(_profile(Tenure.RENT), _dists(ptaxp=-50, chrty=-1, health=-10))
    assert (sa.property_tax, sa.charity, sa.medical) == (0.0, 0.0, 0.0)


def test_default_schedule_a_is_all_zero() -> None:
    sa = ScheduleA()
    assert (sa.mortgage_interest, sa.property_tax, sa.state_local_income_tax,
            sa.charity, sa.medical) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_compute_tax_detail_none_uses_standard_deduction() -> None:
    # detail=None → no schedule_a → standard deduction (byte-identical guarantee).
    b = compute_tax(80_000, "single", "TX_00001", detail=None)
    assert b.federal_tax > 0


def test_compute_tax_itemizes_when_beneficial() -> None:
    # A $250K CA filer: SALT income alone (~$19K) exceeds the $16,100 standard
    # deduction, and a $30K mortgage deepens it → federal tax strictly below the
    # standard-deduction path. taxcalc makes the max(standard, itemized) choice.
    std = compute_tax(250_000, "single", "CA_00001", detail=None)
    salt_only = compute_tax(250_000, "single", "CA_00001", detail=TaxDetail())
    mortgage = compute_tax(
        250_000, "single", "CA_00001",
        detail=TaxDetail(itemized_mortgage_interest=30_000, itemized_property_tax=1_500),
    )
    assert salt_only.federal_tax < std.federal_tax        # itemizes on SALT alone
    assert mortgage.federal_tax < salt_only.federal_tax    # mortgage deepens it


def test_compute_tax_medical_below_floor_has_no_effect() -> None:
    # Medical below the 7.5%-of-AGI floor contributes nothing, and SALT in a
    # no-tax state is 0, so for a low earner who takes the standard deduction
    # these inputs leave federal tax unchanged. (Charity is deliberately excluded:
    # OBBBA 2026 grants an ABOVE-the-line charitable deduction to standard-
    # deduction filers too, so charity WOULD lower the tax — taxcalc applies it.)
    base = compute_tax(35_000, "single", "TX_00001", detail=TaxDetail())
    with_item = compute_tax(
        35_000, "single", "TX_00001",
        detail=TaxDetail(itemized_medical=800),  # 800 << 7.5% × 35k = 2,625 floor
    )
    assert abs(base.federal_tax - with_item.federal_tax) < 1e-6
