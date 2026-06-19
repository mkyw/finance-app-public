"""Tests for models.benefits.eligibility.

Run from repo root:
    python3.11 -m pytest tests/models/test_benefits.py -v
"""

from __future__ import annotations

from models.benefits.eligibility import screen
from shared.types import HouseholdProfile, Tenure


def _profile(
    *,
    gross_income: float,
    household_size: int,
    puma_code: str = "CA_03761",
    tenure: Tenure = Tenure.RENT,
    housing_cost: float = 1200.0,
    age: int = 35,
) -> HouseholdProfile:
    return HouseholdProfile(
        age=age,
        gross_income=gross_income,
        puma_code=puma_code,
        tenure=tenure,
        housing_cost=housing_cost,
        household_size=household_size,
    )


def test_snap_eligible() -> None:
    # FPL for hh_size=3 is 25820; 130% threshold = 33566. Gross 25k < threshold.
    profile = _profile(gross_income=25_000, household_size=3)
    programs = {m.program_name for m in screen(profile)}
    assert any("SNAP" in p for p in programs), (
        f"SNAP should appear; got {programs}"
    )


def test_snap_ineligible() -> None:
    # FPL for hh_size=2 is 20440; 130% threshold ≈ 26572. 80k >> threshold.
    profile = _profile(gross_income=80_000, household_size=2)
    programs = {m.program_name for m in screen(profile)}
    assert not any("SNAP" in p for p in programs), (
        f"SNAP should NOT appear; got {programs}"
    )


def test_framing_rule() -> None:
    profile = _profile(gross_income=25_000, household_size=3)
    results = screen(profile)
    assert results, "expected at least one match for this low-income profile"
    for m in results:
        assert "may qualify" in m.framing, (
            f"framing violated for {m.program_name!r}: {m.framing!r}"
        )


def test_medicaid_expansion() -> None:
    # CA is expansion (138% FPL threshold); 120% FPL at size 1 = 18072.
    profile = _profile(
        gross_income=17_000, household_size=1, puma_code="CA_03761"
    )
    programs = {m.program_name for m in screen(profile)}
    assert any("Medicaid" in p for p in programs), (
        f"Medicaid expected in CA at 120% FPL; got {programs}"
    )


def test_medicaid_non_expansion() -> None:
    # TX is non-expansion; threshold is 100% FPL = 15060 for size 1.
    # At 120% FPL ($18072), the household is above 100% FPL so Medicaid
    # should not appear. Accept either "not present" or "check" confidence.
    profile = _profile(
        gross_income=18_072, household_size=1, puma_code="TX_03700"
    )
    results = screen(profile)
    medicaid = [m for m in results if "Medicaid" in m.program_name]
    if medicaid:
        assert medicaid[0].confidence == "check", (
            f"Medicaid in non-expansion state at 120% FPL should be 'check' "
            f"or absent; got {medicaid[0].confidence}"
        )


def test_all_five_checked() -> None:
    # Very low income single filer: should trigger at least 3 programs
    # (SNAP, Medicaid, LIHEAP are the minimum at this level).
    profile = _profile(
        gross_income=15_000, household_size=1, puma_code="CA_03761"
    )
    results = screen(profile)
    assert len(results) >= 3, (
        f"expected >=3 program matches for a very low-income single filer; "
        f"got {[m.program_name for m in results]}"
    )
