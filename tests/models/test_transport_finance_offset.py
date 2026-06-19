"""Tests for the auto-loan/transportation double-count fix.

When a user reports ``auto_loan_payment``, the debt-service block captures it in
full.  The transportation aggregate's cohort anchor still embeds ``vehint +
vehprn`` (vehicle loan interest + principal) — the double-count.  The fix in
``compute_allocations`` subtracts ``min(transport_finance_mean, auto_annual)``
from the transportation anchor, bounded at the finance sub-component so
operating costs (gas, insurance, maintenance) are never reduced.

These tests run against the live ``compute_allocations()`` with synthetic
distributions (no population artifacts required).  A separate integration
class marks tests that need ``pipeline/artifacts/`` and are slow.

Run from repo root:
    .venv/bin/python -m pytest tests/models/test_transport_finance_offset.py -v
"""

from __future__ import annotations

import pytest

from models.optimizer.allocator import _UX_BIAS_DEFAULT, compute_allocations
from shared.constants.categories import AGGREGATED_CATEGORY_CODES, AGGREGATED_LUXURY_CATEGORIES
from shared.types import HouseholdProfile, SpendingDistribution, Tenure


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _dist(
    *,
    p10: float = 200.0,
    p50: float = 500.0,
    p90: float = 10_000.0,
    weighted_mean: float = 500.0,
    nonzero_rate: float = 1.0,
    conditional_p90: float = 10_000.0,
) -> SpendingDistribution:
    return SpendingDistribution(
        p10=p10,
        p25=p50,
        p50=p50,
        p75=p50,
        p90=p90,
        engel_estimate=p50,
        feasibility_adjusted=0.0,
        cohort_position=0.5,
        is_structural=False,
        nonzero_rate=nonzero_rate,
        conditional_p90=conditional_p90,
        weighted_mean=weighted_mean,
        trimmed_mean=weighted_mean,
    )


_HOUSING_MONTHLY: float = 1_000.0   # $1,000/mo → $12,000/yr pinned
_TRANSPORT_MEAN: float = 2_000.0    # cohort transportation weighted mean (pre-UX)
_FINANCE_MEAN: float = 1_200.0      # vehint + vehprn portion of that mean


def _dists() -> dict[str, SpendingDistribution]:
    """Minimal distributions for all 46 aggregated categories."""
    d = {c: _dist() for c in AGGREGATED_CATEGORY_CODES}
    # Transportation has a known weighted_mean so we can reason about the anchor.
    d["transportation"] = _dist(weighted_mean=_TRANSPORT_MEAN)
    # Housing pin: rntval → lb = ub = housing_cost * 12 (handled by compute_bounds).
    d["rntval"] = _dist(p10=12_000.0, p50=12_000.0, p90=12_000.0)
    # Balance categories: zeroed via tenure/balance constraints.
    for cat in ("check", "retire", "stock", "othfin", "lifval",
                "vehval", "ownval", "othdbt", "stddbt"):
        d[cat] = _dist(p10=0.0, p50=0.0, p90=0.0, weighted_mean=0.0)
    # OWN-only cats zeroed for RENT profile (tenure constraints).
    for cat in ("mrtgip", "mrtgpp", "mrtgps", "ptaxp", "hinsp", "hmtimp", "ohouse"):
        if cat in d:
            d[cat] = _dist(p10=0.0, p50=0.0, p90=0.0, weighted_mean=0.0)
    return d


def _profile(auto_loan_payment: float = 0.0) -> HouseholdProfile:
    return HouseholdProfile(
        age=35,
        gross_income=120_000,
        puma_code="CA_00101",
        tenure=Tenure.RENT,
        housing_cost=_HOUSING_MONTHLY,
        household_size=1,
        auto_loan_payment=auto_loan_payment,
    )


def _alloc(
    auto_loan_payment: float = 0.0,
    transport_finance_mean: float = 0.0,
    d_variable: float = 80_000.0,  # generous — ensures primary path
) -> dict[str, float]:
    """Run compute_allocations with the synthetic setup; return allocations."""
    result = compute_allocations(
        distributions=_dists(),
        profile=_profile(auto_loan_payment=auto_loan_payment),
        d_variable=d_variable,
        cpi_scalars={c: 1.0 for c in AGGREGATED_CATEGORY_CODES},
        lambda_weights={},
        category_codes=AGGREGATED_CATEGORY_CODES,
        luxury_categories=AGGREGATED_LUXURY_CATEGORIES,
        transport_finance_mean=transport_finance_mean,
    )
    return result.allocations


# ---------------------------------------------------------------------------
# Core offset behaviour
# ---------------------------------------------------------------------------

class TestTransportFinanceOffset:
    """Unit tests for the anchor-adjustment path — no population data."""

    def test_no_loan_baseline(self) -> None:
        """Without auto_loan, transport anchor is unchanged (weighted_mean × UX)."""
        alloc = _alloc(auto_loan_payment=0.0, transport_finance_mean=_FINANCE_MEAN)
        expected = _TRANSPORT_MEAN * _UX_BIAS_DEFAULT
        assert alloc["transportation"] == pytest.approx(expected, rel=1e-6)

    def test_zero_finance_mean_is_noop(self) -> None:
        """transport_finance_mean == 0 → no offset even with a reported loan."""
        alloc_no = _alloc(auto_loan_payment=0.0, transport_finance_mean=0.0)
        alloc_with = _alloc(auto_loan_payment=700.0, transport_finance_mean=0.0)
        assert alloc_with["transportation"] == pytest.approx(
            alloc_no["transportation"], rel=1e-6
        )

    def test_offset_applied_when_loan_lt_finance_mean(self) -> None:
        """auto_annual < finance_mean → offset = auto_annual (partial reduction)."""
        auto_monthly = 50.0               # $600/yr < $1,200 finance_mean
        auto_annual = auto_monthly * 12.0
        alloc_base = _alloc(auto_loan_payment=0.0, transport_finance_mean=_FINANCE_MEAN)
        alloc_loan = _alloc(auto_loan_payment=auto_monthly, transport_finance_mean=_FINANCE_MEAN)
        delta = alloc_base["transportation"] - alloc_loan["transportation"]
        assert delta == pytest.approx(auto_annual, rel=1e-4)

    def test_offset_bounded_at_finance_mean(self) -> None:
        """auto_annual > finance_mean → offset = finance_mean (operating floor)."""
        large_monthly = 1_500.0           # $18,000/yr >> $1,200 finance_mean
        alloc_base = _alloc(auto_loan_payment=0.0, transport_finance_mean=_FINANCE_MEAN)
        alloc_loan = _alloc(auto_loan_payment=large_monthly, transport_finance_mean=_FINANCE_MEAN)
        delta = alloc_base["transportation"] - alloc_loan["transportation"]
        assert delta == pytest.approx(_FINANCE_MEAN, rel=1e-4)

    def test_operating_costs_never_zeroed(self) -> None:
        """Even a very large loan cannot drive transportation below p10 floor."""
        alloc = _alloc(auto_loan_payment=100_000.0, transport_finance_mean=_FINANCE_MEAN)
        lb = _dists()["transportation"].p10  # 200.0
        assert alloc["transportation"] >= lb

    def test_other_categories_unaffected(self) -> None:
        """The offset touches only transportation; other allocations are unchanged."""
        alloc_base = _alloc(auto_loan_payment=0.0, transport_finance_mean=_FINANCE_MEAN)
        alloc_loan = _alloc(auto_loan_payment=700.0, transport_finance_mean=_FINANCE_MEAN)
        for cat in AGGREGATED_CATEGORY_CODES:
            if cat == "transportation":
                continue
            assert alloc_loan[cat] == pytest.approx(alloc_base[cat], rel=1e-6), (
                f"{cat} changed with auto_loan: "
                f"base={alloc_base[cat]:.4f}, loan={alloc_loan[cat]:.4f}"
            )

    def test_disaggregated_path_unaffected(self) -> None:
        """On the disaggregated 55-cat path, 'transportation' is not a code.

        The guard ``"transportation" in codes`` ensures no offset is applied —
        the correct behavior because the disaggregated path never aggregates
        vehint/vehprn into a transportation sum.
        """
        from shared.constants.categories import CATEGORY_CODES, LUXURY_CATEGORIES

        # Build minimal 55-cat distributions.
        dists_55 = {
            c: _dist(weighted_mean=500.0, p10=200.0, p90=10_000.0)
            for c in CATEGORY_CODES
        }
        # Zero out balance and RENT-excluded categories.
        for cat in ("check", "retire", "stock", "othfin", "lifval",
                    "vehval", "ownval", "othdbt", "stddbt",
                    "mrtgip", "mrtgpp", "mrtgps", "ptaxp", "hinsp"):
            dists_55[cat] = _dist(p10=0.0, p50=0.0, p90=0.0, weighted_mean=0.0)

        profile = _profile(auto_loan_payment=700.0)
        result_no_finance = compute_allocations(
            distributions=dists_55,
            profile=profile,
            d_variable=120_000.0,
            cpi_scalars={c: 1.0 for c in CATEGORY_CODES},
            lambda_weights={},
            category_codes=CATEGORY_CODES,
            luxury_categories=LUXURY_CATEGORIES,
            transport_finance_mean=1_200.0,  # non-zero but irrelevant on this path
        )
        result_no_loan = compute_allocations(
            distributions=dists_55,
            profile=_profile(auto_loan_payment=0.0),
            d_variable=120_000.0,
            cpi_scalars={c: 1.0 for c in CATEGORY_CODES},
            lambda_weights={},
            category_codes=CATEGORY_CODES,
            luxury_categories=LUXURY_CATEGORIES,
            transport_finance_mean=1_200.0,
        )
        # Debt service differs (auto loan subtracts from d_variable), so
        # d_variable_adjusted differs — but the per-category anchor calculation
        # is unaffected (no transport aggregate to reduce). Verify no per-category
        # allocation changed between the two calls for categories other than
        # ones affected by the reduced d_variable_adjusted.
        # The simplest check: vehint and vehprn individually are UNCHANGED by
        # the transport_finance_mean parameter (the offset only fires on "transportation").
        assert result_no_finance.allocations["vehint"] == pytest.approx(
            result_no_loan.allocations["vehint"], rel=1e-4
        )
        assert result_no_finance.allocations["vehprn"] == pytest.approx(
            result_no_loan.allocations["vehprn"], rel=1e-4
        )

    def test_debt_service_unchanged_by_offset(self) -> None:
        """The auto-loan stays in debt_service unchanged; offset is anchor-only."""
        from models.optimizer.allocator import AllocationResult

        auto_monthly = 700.0
        result: AllocationResult = compute_allocations(
            distributions=_dists(),
            profile=_profile(auto_loan_payment=auto_monthly),
            d_variable=80_000.0,
            cpi_scalars={c: 1.0 for c in AGGREGATED_CATEGORY_CODES},
            lambda_weights={},
            category_codes=AGGREGATED_CATEGORY_CODES,
            luxury_categories=AGGREGATED_LUXURY_CATEGORIES,
            transport_finance_mean=_FINANCE_MEAN,
        )
        assert result.debt_service == pytest.approx(auto_monthly * 12.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Four-way closure with the offset active
# ---------------------------------------------------------------------------

class TestFourWayClosesWithOffset:
    """The four-way identity (committed + debt + spend + savings ≈ take_home)
    must survive the transport anchor reduction."""

    def test_primary_path_anchor_sum_within_budget(self) -> None:
        """With the offset, the anchor sum still fits inside d_variable_adjusted."""
        from models.optimizer.allocator import AllocationResult

        result: AllocationResult = compute_allocations(
            distributions=_dists(),
            profile=_profile(auto_loan_payment=700.0),
            d_variable=80_000.0,
            cpi_scalars={c: 1.0 for c in AGGREGATED_CATEGORY_CODES},
            lambda_weights={},
            category_codes=AGGREGATED_CATEGORY_CODES,
            luxury_categories=AGGREGATED_LUXURY_CATEGORIES,
            transport_finance_mean=_FINANCE_MEAN,
        )
        total = sum(result.allocations.values())
        assert result.solver_status == "primary"
        assert total <= result.d_variable_adjusted + 1e-4

    def test_transport_responds_to_auto_loan(self) -> None:
        """The core bug: transport allocation must DROP when auto_loan is reported."""
        alloc_no_loan = _alloc(auto_loan_payment=0.0, transport_finance_mean=_FINANCE_MEAN)
        alloc_with_loan = _alloc(auto_loan_payment=700.0, transport_finance_mean=_FINANCE_MEAN)
        assert alloc_with_loan["transportation"] < alloc_no_loan["transportation"], (
            "transportation did not respond to auto_loan_payment — "
            f"no-loan={alloc_no_loan['transportation']:.2f}, "
            f"with-loan={alloc_with_loan['transportation']:.2f}"
        )
