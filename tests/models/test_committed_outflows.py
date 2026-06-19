"""Committed-outflow applicability-conditioning tests (Build A, 2026-05-29).

Pin the Build A fixes to the four small low-participation lines, all instances
of COHORT-AVERAGE-RESPECTS-MUTUAL-EXCLUSION / its conditioning corollary:

  - **HSA + traditional FSA collapse**: one ``pretax_health_savings`` line
    predicted as the UNION (the dominant arm), NOT the sum of two independent
    population-means; no separate ``hsa_contribution`` / ``fsa_health`` codes.
  - **Commuter conditioned on car ownership**: near-zero by default (car-owner
    prior) and for resolved owners; the genuine transit-user set-aside only
    for resolved non-owners (``owns_car=False``).
  - **Supplemental life/disability conditioned on age × dependents**:
    near-zero for young childless singles, rising with age and household size.
  - **Override fixity** (Q7) still pins any single line and re-totals.
"""

from __future__ import annotations

from models.optimizer.committed_outflows import (
    apply_committed_outflow_overrides,
    estimate_commuter_benefit,
    estimate_committed_outflows,
    estimate_pretax_health_savings,
    estimate_supplemental_insurance,
)
from shared.types.enums import Tenure
from shared.types.household import HouseholdProfile


def _profile(age=25, income=90_000, size=1, tenure=Tenure.RENT):
    return HouseholdProfile(
        age=age,
        gross_income=income,
        puma_code="IL_03420",
        tenure=tenure,
        housing_cost=1_800.0,
        household_size=size,
    )


class TestPretaxHealthUnion:
    """HSA + traditional FSA collapse to one mutually-exclusive-respecting line."""

    def test_single_line_no_separate_hsa_fsa_codes(self) -> None:
        codes = {it.code for it in estimate_committed_outflows(_profile()).items}
        assert "pretax_health_savings" in codes
        assert "hsa_contribution" not in codes
        assert "fsa_health" not in codes

    def test_union_is_not_the_sum(self) -> None:
        """The union (dominant arm) must be well below the old HSA+FSA sum.

        Old behavior at $90k summed HSA $360 + traditional-FSA $180 = $540/yr —
        an IRS-illegal pair. The union must equal the dominant arm ($360), never
        the arithmetic sum.
        """
        union = estimate_pretax_health_savings(90_000)
        assert union == 360.0  # dominant HSA arm, not 540 (the summed pair)

    def test_rises_with_income(self) -> None:
        vals = [estimate_pretax_health_savings(i) for i in (25_000, 50_000, 90_000, 130_000, 250_000)]
        assert vals == sorted(vals)
        assert vals[0] < vals[-1]

    def test_dependent_care_fsa_stays_separate(self) -> None:
        """Dependent-care FSA is NOT mutually exclusive with the HSA — own line."""
        codes = {it.code for it in estimate_committed_outflows(_profile(size=4)).items}
        assert "fsa_dependent_care" in codes
        dc = next(
            it for it in estimate_committed_outflows(_profile(size=4)).items
            if it.code == "fsa_dependent_care"
        )
        assert dc.annual > 0.0  # family with kids


class TestCommuterConditioning:
    """Commuter bifurcates on car ownership; near-zero until owns_car elicited."""

    def test_default_near_zero_car_owner_prior(self) -> None:
        assert estimate_commuter_benefit(90_000, owns_car=None) == 0.0

    def test_resolved_owner_zero(self) -> None:
        assert estimate_commuter_benefit(90_000, owns_car=True) == 0.0

    def test_resolved_non_owner_meaningful_and_income_scaled(self) -> None:
        low = estimate_commuter_benefit(40_000, owns_car=False)
        high = estimate_commuter_benefit(150_000, owns_car=False)
        assert low > 0.0
        assert high > low

    def test_block_commuter_zero_for_anchorage_car_default(self) -> None:
        anc = _profile(age=25, income=51_000, size=1)
        commuter = next(
            it for it in estimate_committed_outflows(anc).items
            if it.code == "commuter_benefit"
        )
        assert commuter.annual == 0.0


class TestSupplementalConditioning:
    """Supplemental life/disability rises with age and dependents; ~0 for young singles."""

    def test_young_childless_single_near_zero(self) -> None:
        val = estimate_supplemental_insurance(age=25, household_size=1)
        assert 0.0 < val <= 84.0  # ~$0-7/mo, not the old flat $240/yr

    def test_rises_with_age(self) -> None:
        young = estimate_supplemental_insurance(age=25, household_size=1)
        older = estimate_supplemental_insurance(age=55, household_size=1)
        assert older > young

    def test_rises_with_dependents(self) -> None:
        single = estimate_supplemental_insurance(age=42, household_size=1)
        family = estimate_supplemental_insurance(age=42, household_size=4)
        assert family > single


class TestCommittedBlockTargets:
    """End-state targets from the Build A scoping."""

    def test_chicago_part_b_drops_into_range(self) -> None:
        """Chicago $90k 25yo single: small lines drop from $77/mo to ~$25-35/mo."""
        small_codes = {
            "pretax_health_savings", "fsa_dependent_care",
            "commuter_benefit", "supplemental_insurance",
        }
        items = estimate_committed_outflows(_profile()).items
        small_monthly = sum(it.annual for it in items if it.code in small_codes) / 12.0
        assert 25.0 <= small_monthly <= 35.0

    def test_override_fixity_pins_and_retotals(self) -> None:
        base = estimate_committed_outflows(_profile())
        new = apply_committed_outflow_overrides(base, {"pretax_health_savings": 0.0})
        assert new.by_code()["pretax_health_savings"].annual == 0.0
        assert new.total == base.total - base.by_code()["pretax_health_savings"].annual
