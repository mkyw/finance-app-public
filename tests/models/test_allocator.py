"""Tests for models.optimizer.allocator — balance-cat zeroing and
debt-service adjustment of ``d_variable``.

Run from repo root:
    python3.11 -m pytest tests/models/test_allocator.py -v
"""

from __future__ import annotations

import pytest

from models.engel.curves import annotate_distributions
from models.engel.elasticity import lambda_weights_all
from models.matching.algorithm import match_household
from models.matching.cpi_scaler import default_scalars
from models.optimizer.allocator import compute_allocations
from models.pace.calculator import compute_d_variable
from shared.constants.categories import BALANCE_CATEGORIES, FLOW_CATEGORIES
from shared.types import HouseholdProfile, Tenure

ARTIFACTS = "pipeline/artifacts"


@pytest.fixture(scope="module")
def chicago_context() -> dict:
    profile = HouseholdProfile(
        age=35, gross_income=65000, puma_code="IL_00100",
        tenure=Tenure.RENT, housing_cost=1400, household_size=2,
    )
    match = match_household(profile, ARTIFACTS)
    # d_variable is now full take_home (housing_cost is pinned inside
    # the allocator, not pre-subtracted). Use compute_d_variable so
    # the number stays in sync with the tax/pace modules.
    d_variable = compute_d_variable(profile, filing_status="single")
    dists = annotate_distributions(
        match.distributions,
        equivalized_income=profile.equivalized_income,
        disposable_income=d_variable,
        artifacts_path=ARTIFACTS,
    )
    lams = lambda_weights_all(profile.equivalized_income, ARTIFACTS)
    return {
        "profile": profile, "match": match, "dists": dists, "lams": lams,
        "d_variable": d_variable,
    }


def test_balance_cats_zero(chicago_context: dict) -> None:
    c = chicago_context
    result = compute_allocations(
        distributions=c["dists"],
        profile=c["profile"],
        d_variable=c["d_variable"],
        cpi_scalars=default_scalars(),
        lambda_weights=c["lams"],
        cohort_median_income=c["match"].cohort_median_income,
        artifacts_path=ARTIFACTS,
    )
    for cat in BALANCE_CATEGORIES:
        assert result.allocations[cat] == 0.0, (
            f"balance cat {cat} got non-zero allocation {result.allocations[cat]}"
        )


def test_flow_cats_sum_bounded_by_adjusted_d(chicago_context: dict) -> None:
    c = chicago_context
    result = compute_allocations(
        distributions=c["dists"],
        profile=c["profile"],
        d_variable=c["d_variable"],
        cpi_scalars=default_scalars(),
        lambda_weights=c["lams"],
        cohort_median_income=c["match"].cohort_median_income,
        artifacts_path=ARTIFACTS,
    )
    flow_sum = sum(result.allocations[cat] for cat in FLOW_CATEGORIES)
    assert flow_sum <= result.d_variable_adjusted + 1e-4, (
        f"flow sum {flow_sum:.2f} > adjusted d {result.d_variable_adjusted:.2f}"
    )


def test_zero_predicted_debt_preserves_d(chicago_context: dict) -> None:
    c = chicago_context
    result = compute_allocations(
        distributions=c["dists"],
        profile=c["profile"],
        d_variable=c["d_variable"],
        cpi_scalars=default_scalars(),
        lambda_weights=c["lams"],
        cohort_median_income=c["match"].cohort_median_income,
        artifacts_path=ARTIFACTS,
        predicted_othdbt=0.0,
        predicted_stddbt=0.0,
    )
    assert result.debt_service == 0.0
    assert result.d_variable_adjusted == pytest.approx(c["d_variable"])


def test_debt_service_shrinks_adjusted_d(chicago_context: dict) -> None:
    c = chicago_context
    result = compute_allocations(
        distributions=c["dists"],
        profile=c["profile"],
        d_variable=c["d_variable"],
        cpi_scalars=default_scalars(),
        lambda_weights=c["lams"],
        cohort_median_income=c["match"].cohort_median_income,
        artifacts_path=ARTIFACTS,
        predicted_othdbt=3_000.0,  # $60/mo * 12 = $720/yr
        predicted_stddbt=20_000.0,  # ~$2,725/yr
    )
    assert result.debt_service > 0
    assert result.debt_service == pytest.approx(720 + 20_000 * 0.13627, rel=1e-3)
    assert result.d_variable_adjusted == pytest.approx(
        c["d_variable"] - result.debt_service
    )
    flow_sum = sum(result.allocations[cat] for cat in FLOW_CATEGORIES)
    assert flow_sum <= result.d_variable_adjusted + 1e-4
