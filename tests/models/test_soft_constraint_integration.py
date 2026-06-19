"""Integration tests for the soft-constraint optimizer (now the default path).

Exercises the full match -> annotate -> solve_feasibility chain on the disaggregated
path against the real artifacts (relative path, run from repo root). Asserts the
load-bearing properties: conditional_p10 plumbing arrives, high-participation categories
no longer compress to $0, necessities stay protected, and the budget closes exactly.
"""
from __future__ import annotations

import pytest

from models.engel.curves import annotate_distributions
from models.engel.elasticity import lambda_weights_all
from models.matching.algorithm import match_household
from models.optimizer.feasibility import solve_feasibility
from shared.types import HouseholdProfile, Tenure

ARTIFACTS = "pipeline/artifacts"
MADISON = ["WI_02101", "WI_02102", "WI_02103", "WI_02104"]


@pytest.fixture(scope="module")
def state() -> dict:
    profile = HouseholdProfile(
        age=30, gross_income=42000, puma_code=MADISON[0], tenure=Tenure.RENT,
        housing_cost=1000, household_size=1,
    )
    match = match_household(profile, ARTIFACTS, city_pumas=MADISON)
    return {"profile": profile, "match": match}


def _solve(d_variable: float, state: dict):
    profile = state["profile"]
    dists = annotate_distributions(
        state["match"].distributions,
        equivalized_income=profile.equivalized_income,
        disposable_income=d_variable,
        artifacts_path=ARTIFACTS,
    )
    lams = lambda_weights_all(profile.equivalized_income, ARTIFACTS)
    return solve_feasibility(dists, profile, lams, d_variable, artifacts_path=ARTIFACTS)


def test_conditional_p10_plumbed_to_matching(state):
    """The Phase-2 plumbing: conditional_p10 arrives populated for a high-
    participation, window-zero-p10 category (apparel)."""
    cloftw = state["match"].distributions["cloftw"]
    assert cloftw.nonzero_rate >= 0.75
    assert cloftw.p10 == 0.0                 # CEX diary-window zero
    assert cloftw.conditional_p10 > 0.0      # realistic participant floor present


def test_constrained_high_participation_not_zeroed(state):
    """Under compression, high-participation luxuries floor at conditional_p10
    instead of dropping to $0 (the demonstrating bug fix)."""
    res = _solve(20000.0, state)
    assert res.solver_status == "soft_constrained"
    # Apparel and dining-out are high-participation, window-zero-p10 luxuries.
    assert res.allocations["cloftw"] > 0.0
    assert res.allocations["eatout"] > 0.0
    # Budget closes exactly (four-way reconciliation invariant).
    assert abs(sum(res.allocations.values()) - res.d_variable_adjusted) < 1.0


def test_necessities_protected_under_compression(state):
    res = _solve(20000.0, state)
    # Groceries / electricity must keep substantial positive allocations.
    assert res.allocations["eathome"] > 0.0
    assert res.allocations["elec"] > 0.0


def test_primary_returns_clamped_anchors(state):
    """When the budget fits, the optimizer is never entered: allocations are the
    clamped anchors, all non-negative, summing within budget."""
    res = _solve(40000.0, state)
    assert res.solver_status == "primary"
    assert all(v >= 0.0 for v in res.allocations.values())
    assert sum(res.allocations.values()) <= 40000.0 + 1e-4


def test_compression_gap_surfaced_on_soft_constrained(state):
    """The debt-accumulation build (2026-06-09): the soft-deficit gap
    (anchor_sum − adjusted_d) is exposed on the result — positive when the
    optimizer compressed, 0.0 on the primary path. Purely informational;
    allocations are unchanged."""
    res = _solve(20000.0, state)
    assert res.solver_status == "soft_constrained"
    assert res.compression_gap > 0.0
    primary = _solve(40000.0, state)
    assert primary.compression_gap == 0.0


def test_debt_accumulation_annotation_end_to_end(state):
    """Plumbing check: a CC-debt profile + the soft-constrained result fire
    the annotation, with the annual figure equal to the surfaced gap."""
    from models.optimizer.debt_accumulation import project_debt_accumulation

    res = _solve(20000.0, state)
    profile = state["profile"]
    cc_profile = HouseholdProfile(
        age=profile.age, gross_income=profile.gross_income,
        puma_code=profile.puma_code, tenure=profile.tenure,
        housing_cost=profile.housing_cost, household_size=profile.household_size,
        cc_carried_balance=4000.0,
    )
    ann = project_debt_accumulation(
        cc_profile,
        solver_status=res.solver_status,
        compression_gap=res.compression_gap,
        d_variable_adjusted=res.d_variable_adjusted,
    )
    assert ann.applies is True
    assert ann.annual_potential_growth == pytest.approx(res.compression_gap)
    assert ann.monthly_potential_growth == pytest.approx(res.compression_gap / 12.0)
    assert ann.framing_state in {"signal_clear", "signal_marginal"}
    # Out-of-sum invariant: the annotation does not perturb the allocation —
    # the compressed budget still closes exactly.
    assert abs(sum(res.allocations.values()) - res.d_variable_adjusted) < 1.0
