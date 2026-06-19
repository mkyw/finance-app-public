"""Tests for filing-unit imputation (models.tax.imputation).

Run from repo root:
    .venv/bin/python -m pytest tests/models/test_imputation.py -v
"""

from __future__ import annotations

from models.tax.imputation import (
    CTC_AGE_LIMIT,
    DEPENDENT_AGE_LIMIT,
    impute_filing_unit,
)
from shared.types.enums import Tenure
from shared.types.household import HouseholdProfile


def _profile(age: int, household_size: int) -> HouseholdProfile:
    return HouseholdProfile(
        age=age,
        gross_income=100_000.0,
        puma_code="CA_03761",
        tenure=Tenure.RENT,
        housing_cost=2_000.0,
        household_size=household_size,
    )


def test_single_size_one() -> None:
    fu = impute_filing_unit(_profile(35, 1), "single")
    assert fu.filing_status == "single"
    assert fu.num_dependents == 0
    assert fu.n_children_under_17 == 0
    assert fu.spouse_age is None


def test_size_one_mfj_repaired_to_single() -> None:
    # MFJ needs a spouse; size 1 → coerce to single (raises tax = conservative).
    fu = impute_filing_unit(_profile(35, 1), "married_filing_jointly")
    assert fu.filing_status == "single"
    assert fu.imputed is True
    assert any("second household member" in n for n in fu.notes)


def test_size_one_hoh_repaired_to_single() -> None:
    fu = impute_filing_unit(_profile(35, 1), "head_of_household")
    assert fu.filing_status == "single"


def test_mfj_family_of_four() -> None:
    # 2 adults + 2 children; householder 40 → oldest 13, next 10; both < 17.
    fu = impute_filing_unit(_profile(40, 4), "married_filing_jointly")
    assert fu.filing_status == "married_filing_jointly"
    assert fu.num_dependents == 2
    assert fu.n_children_under_17 == 2
    assert fu.dependent_ages == (13, 10)
    assert fu.spouse_age == 40


def test_single_not_upgraded_to_hoh() -> None:
    # Declared single, size 3 (likely a single parent) — we KEEP single
    # (upgrading to HoH would lower tax = forbidden direction). Dependents
    # are still imputed (CTC applies even filing single).
    fu = impute_filing_unit(_profile(37, 3), "single")
    assert fu.filing_status == "single"
    assert fu.num_dependents == 2  # ages 10, 7
    assert fu.n_children_under_17 == 2


def test_older_householder_adult_children_dropped() -> None:
    # Householder 70, size 3: imputed "children" are ~43/40 → adults, dropped.
    fu = impute_filing_unit(_profile(70, 3), "married_filing_jointly")
    assert fu.num_dependents == 0
    assert all(a < DEPENDENT_AGE_LIMIT for a in fu.dependent_ages)


def test_young_family_under_six_bucket() -> None:
    # Householder 30, size 4 MFJ → oldest 3, next 0; both under 6/13/17.
    fu = impute_filing_unit(_profile(30, 4), "married_filing_jointly")
    assert fu.num_dependents == 2
    assert fu.n_children_under_6 == 2
    assert fu.n_children_under_13 == 2
    assert fu.n_children_under_17 == 2


def test_eic_capped_at_three() -> None:
    # Householder 45, size 6 MFJ → 4 children (ages 18,15,12,9); EIC capped 3.
    fu = impute_filing_unit(_profile(45, 6), "married_filing_jointly")
    assert fu.num_dependents == 4
    assert fu.eic_qualifying_children == 3
    # Age 18 is a dependent (<19) but NOT CTC-eligible (>=17).
    assert fu.n_children_under_17 == 3
    assert max(fu.dependent_ages) < DEPENDENT_AGE_LIMIT


def test_ctc_hard_edge_at_17() -> None:
    # A child imputed exactly at the CTC edge is excluded from CTC (hard <17).
    fu = impute_filing_unit(_profile(44, 2), "single")
    # oldest = 44 - 27 = 17 → dependent (<19) but not CTC (<17 is False).
    assert 17 in fu.dependent_ages
    assert fu.n_children_under_17 == sum(1 for a in fu.dependent_ages if a < CTC_AGE_LIMIT)
    assert fu.n_children_under_17 == 0


def test_unrecognized_status_defaults_single() -> None:
    fu = impute_filing_unit(_profile(35, 1), "qualifying_widow")
    assert fu.filing_status == "single"
    assert any("Unrecognized" in n for n in fu.notes)


def test_declared_status_recorded() -> None:
    fu = impute_filing_unit(_profile(40, 4), "married_filing_jointly")
    assert fu.declared_filing_status == "married_filing_jointly"
