"""Tests for models.optimizer.{bounds,feasibility,decomposition}.

Run from repo root:
    python3.11 -m pytest tests/models/test_optimizer.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from models.engel.curves import annotate_distributions, engel_estimates_all
from models.engel.elasticity import lambda_weights_all
from models.matching.algorithm import match_household
from models.optimizer.bounds import apply_tenure_constraints, compute_bounds
from models.optimizer.decomposition import decompose
from models.optimizer.feasibility import FeasibilityResult, solve_feasibility
from shared.constants.categories import CATEGORY_CODES
from shared.types import FinancialZone, HouseholdProfile, Tenure

ARTIFACTS = "pipeline/artifacts"

# Stubbed take-home used throughout — tax module not implemented yet.
_STUB_TAKE_HOME: float = 48_000.0


def _profile() -> HouseholdProfile:
    return HouseholdProfile(
        age=35,
        gross_income=65000,
        puma_code="AK_00101",
        tenure=Tenure.RENT,
        housing_cost=1200,
        household_size=2,
    )


@pytest.fixture(scope="module")
def pipeline_state() -> dict:
    """Run the full matching + annotate chain once; reused by every test.

    Tests that vary ``d_variable`` re-build the Engel annotations so
    ``engel_estimate`` scales with the specific disposable number used.
    """
    profile = _profile()
    match = match_household(profile, ARTIFACTS)
    return {
        "profile": profile,
        "match": match,
    }


def _annotate_for(d_variable: float, state: dict) -> dict:
    """Return distributions annotated with engel estimates for this d_variable."""
    profile: HouseholdProfile = state["profile"]
    return annotate_distributions(
        state["match"].distributions,
        equivalized_income=profile.equivalized_income,
        disposable_income=d_variable,
        artifacts_path=ARTIFACTS,
    )


# ---------------------------------------------------------------------------
# Shared solver-status bucket. Populated by tests as they run.
# ---------------------------------------------------------------------------

STATUS_COUNTER: dict[str, int] = {}


def _record(result: FeasibilityResult) -> FeasibilityResult:
    STATUS_COUNTER[result.solver_status] = STATUS_COUNTER.get(result.solver_status, 0) + 1
    return result


# ---------------------------------------------------------------------------

def test_bounds_shape(pipeline_state: dict) -> None:
    d_variable = _STUB_TAKE_HOME - 1200 * 12  # 33600
    dists = _annotate_for(d_variable, pipeline_state)
    lb, ub = compute_bounds(dists, pipeline_state["profile"], ARTIFACTS)

    assert lb.shape == (55,)
    assert ub.shape == (55,)
    assert np.all(lb >= 0), f"negative lb: {lb.min()}"
    assert np.all(ub >= 0), f"negative ub: {ub.min()}"
    assert np.all(ub >= lb), f"ub < lb somewhere; min diff = {(ub - lb).min()}"


def test_bounds_zero_floor(pipeline_state: dict) -> None:
    d_variable = _STUB_TAKE_HOME - 1200 * 12
    dists = _annotate_for(d_variable, pipeline_state)
    lb, ub = compute_bounds(dists, pipeline_state["profile"], ARTIFACTS)

    for i, cat in enumerate(CATEGORY_CODES):
        if dists[cat].p10 == 0:
            assert lb[i] == 0, f"{cat}: p10 == 0 but lb = {lb[i]}"
        if dists[cat].p90 == 0:
            # Either falls back to engel_estimate or the symbolic 1.0 floor.
            assert ub[i] >= 0, f"{cat}: ub = {ub[i]}"


def test_feasibility_normal(pipeline_state: dict) -> None:
    d_variable = _STUB_TAKE_HOME - 1200 * 12  # 33600
    dists = _annotate_for(d_variable, pipeline_state)
    lams = lambda_weights_all(pipeline_state["profile"].equivalized_income, ARTIFACTS)

    result = _record(solve_feasibility(dists, pipeline_state["profile"], lams, d_variable))

    assert result.solver_status in (
        "primary", "soft_constrained", "floor_infeasible"
    ), f"unexpected solver_status: {result.solver_status}"
    total = sum(result.allocations.values())
    assert total <= d_variable + 1e-4, (
        f"sum(allocations) = {total:.4f} exceeds d_variable = {d_variable}"
    )
    assert result.financial_zone in FinancialZone
    assert result.structural_deficit == 0.0
    for cat, alloc in result.allocations.items():
        assert alloc >= 0, f"{cat} allocation negative: {alloc}"
    assert set(result.allocations.keys()) == set(CATEGORY_CODES)


def test_feasibility_surplus(pipeline_state: dict) -> None:
    # IMPROVEMENT fires when d_variable substantially exceeds what the
    # cohort-typical engel anchor predicts the household would spend.
    # Production caller (apps/api/profiles/services.py) will derive the
    # engel anchor from a reference spend level, not from d_variable itself.
    d_variable = 80_000.0
    cohort_anchor = 33_600.0
    dists = _annotate_for(cohort_anchor, pipeline_state)
    lams = lambda_weights_all(pipeline_state["profile"].equivalized_income, ARTIFACTS)

    result = _record(solve_feasibility(dists, pipeline_state["profile"], lams, d_variable))

    assert result.financial_zone == FinancialZone.IMPROVEMENT, (
        f"expected IMPROVEMENT at d={d_variable} (engel anchored at "
        f"{cohort_anchor}), got {result.financial_zone} "
        f"(slack={result.feasibility_slack:.2f}, status={result.solver_status})"
    )
    assert result.feasibility_slack > 0


def test_feasibility_tight(pipeline_state: dict) -> None:
    # Tight-budget threshold for AK_00101 RENT shifts when the donor CEI
    # data changes — corrected eathome data raised that category's p10 but
    # other categories' floors moved too. $12k now sits comfortably in
    # STABILITY (slack ~4%); $14k+ flips to IMPROVEMENT.
    d_variable = 12_000.0
    dists = _annotate_for(d_variable, pipeline_state)
    lams = lambda_weights_all(pipeline_state["profile"].equivalized_income, ARTIFACTS)

    result = _record(solve_feasibility(dists, pipeline_state["profile"], lams, d_variable))

    assert result.financial_zone in (FinancialZone.SURVIVAL, FinancialZone.STABILITY), (
        f"expected SURVIVAL or STABILITY at d={d_variable}, got {result.financial_zone}"
    )


def test_structural_deficit(pipeline_state: dict) -> None:
    d_variable = 500.0
    dists = _annotate_for(d_variable, pipeline_state)
    lams = lambda_weights_all(pipeline_state["profile"].equivalized_income, ARTIFACTS)

    result = _record(solve_feasibility(dists, pipeline_state["profile"], lams, d_variable))

    assert result.financial_zone == FinancialZone.SURVIVAL
    assert result.structural_deficit > 0, (
        f"expected structural_deficit > 0, got {result.structural_deficit}"
    )
    assert result.solver_status == "structural_deficit", (
        f"expected 'structural_deficit' status, got {result.solver_status}"
    )


def test_decompose(pipeline_state: dict) -> None:
    d_variable = _STUB_TAKE_HOME - 1200 * 12
    dists = _annotate_for(d_variable, pipeline_state)
    lams = lambda_weights_all(pipeline_state["profile"].equivalized_income, ARTIFACTS)

    result = _record(solve_feasibility(dists, pipeline_state["profile"], lams, d_variable))
    final = decompose(dists, result.allocations)

    # Shape + basic positivity.
    assert set(final.keys()) == set(CATEGORY_CODES)
    for cat, d in final.items():
        assert d.feasibility_adjusted >= 0, (
            f"{cat}: feasibility_adjusted negative: {d.feasibility_adjusted}"
        )
        assert math.isfinite(d.behavioral_gap), (
            f"{cat}: behavioral_gap not finite: {d.behavioral_gap}"
        )

    # cohort_position is in [0, 1] for every category (including rare
    # zero-spread cats, which decompose pins at 0.5).
    for cat, d in final.items():
        assert 0.0 <= d.cohort_position <= 1.0, (
            f"{cat}: cohort_position out of range: {d.cohort_position}"
        )


def test_zone_boundary(pipeline_state: dict) -> None:
    profile: HouseholdProfile = pipeline_state["profile"]

    # Under new semantics, housing is pinned into lb via
    # ``compute_bounds`` (rntval = housing_cost*12 for RENT). Just
    # barely feasible = d_just_under sits right above lb.sum().
    dists_base = apply_tenure_constraints(
        _annotate_for(33_600.0, pipeline_state), profile.tenure
    )
    lb, _ = compute_bounds(dists_base, profile, ARTIFACTS)
    d_just_under = float(lb.sum()) * 1.05

    dists_under = _annotate_for(d_just_under, pipeline_state)
    lams_under = lambda_weights_all(profile.equivalized_income, ARTIFACTS)
    result_under = _record(solve_feasibility(
        dists_under, profile, lams_under, d_just_under
    ))
    assert result_under.financial_zone in (
        FinancialZone.SURVIVAL, FinancialZone.STABILITY
    ), (
        f"at d={d_just_under:.2f} (just above sum(lb)={lb.sum():.2f}) expected "
        f"SURVIVAL/STABILITY, got {result_under.financial_zone}"
    )

    # Very comfortable: d_variable = full take-home well above both
    # the pinned housing line and the flow anchors. $60k gives the
    # AK_00101 RENT profile ~2x headroom over its pinned housing +
    # flow floor, firmly in IMPROVEMENT territory.
    d_comfort = 60_000.0
    dists_comfort = _annotate_for(d_comfort, pipeline_state)
    lams_comfort = lambda_weights_all(profile.equivalized_income, ARTIFACTS)
    result_comfort = _record(solve_feasibility(
        dists_comfort, profile, lams_comfort, d_comfort
    ))
    assert result_comfort.financial_zone == FinancialZone.IMPROVEMENT, (
        f"at d={d_comfort:.2f} expected IMPROVEMENT, "
        f"got {result_comfort.financial_zone} "
        f"(slack={result_comfort.feasibility_slack:.2f}, status={result_comfort.solver_status})"
    )


def test_solver_status_report(pipeline_state: dict) -> None:
    """Reported last so STATUS_COUNTER has results from earlier tests.

    Confirms the two-phase compression mechanism actually runs — not that
    every scenario stays on the primary path. Must run after the other
    feasibility tests; pytest preserves file order by default.
    """
    # Run one more concrete scenario so STATUS_COUNTER is not empty even if
    # this file is executed in isolation.
    d_variable = 40_000.0
    dists = _annotate_for(d_variable, pipeline_state)
    lams = lambda_weights_all(pipeline_state["profile"].equivalized_income, ARTIFACTS)
    _record(solve_feasibility(dists, pipeline_state["profile"], lams, d_variable))

    print("\nSolver status distribution across all feasibility tests:")
    for status, count in sorted(STATUS_COUNTER.items()):
        print(f"  {status:<25} {count}")

    # Every reported status must be a recognized terminal state of the new
    # mechanism (no leftover CVXPY statuses), and at least one scenario must
    # have exercised compression rather than the primary path.
    assert set(STATUS_COUNTER).issubset(
        {"primary", "soft_constrained", "floor_infeasible", "structural_deficit"}
    ), f"unexpected solver statuses: {STATUS_COUNTER}"
    compressed_count = STATUS_COUNTER.get("soft_constrained", 0)
    assert compressed_count > 0, (
        f"compression never ran across any scenario. status counts: {STATUS_COUNTER}"
    )


# ---------------------------------------------------------------------------
# Smooth-category mean anchoring (anchor-statistic switch).
# ---------------------------------------------------------------------------

from models.optimizer.allocator import _primary_anchors  # noqa: E402
from shared.constants.categories import SMOOTH_MEAN_ANCHOR_CATEGORIES  # noqa: E402
from shared.types import SpendingDistribution  # noqa: E402


def _mk_dist(p50: float, weighted_mean: float, *, p10: float = 0.0,
             p90: float = 1e9, trimmed_mean: float = 0.0) -> SpendingDistribution:
    """Minimal distribution with a wide [p10,p90] so anchors aren't clamped."""
    return SpendingDistribution(
        p10=p10, p25=p50, p50=p50, p75=p50, p90=p90,
        engel_estimate=p50, feasibility_adjusted=0.0, cohort_position=0.5,
        is_structural=True, nonzero_rate=1.0, conditional_p90=p90,
        weighted_mean=weighted_mean, trimmed_mean=trimmed_mean,
    )


def _anchor_for(cat: str, dist: SpendingDistribution) -> float:
    """Run _primary_anchors for a single category, engel/UX disabled."""
    anchors = _primary_anchors(
        distributions={cat: dist},
        cpi_scalars={cat: 1.0},
        user_equivalized_income=50_000.0,
        cohort_median_income=None,   # skip Engel income-gap correction
        artifacts_path=None,
        ux_bias=1.0,
        category_codes=[cat],
        coefficients_path=None,
        luxury_categories=frozenset(),
    )
    return float(anchors[0])


def test_smooth_category_anchors_on_weighted_mean() -> None:
    """A smooth cat with mean > median anchors on the mean (high-side)."""
    assert "eathome" in SMOOTH_MEAN_ANCHOR_CATEGORIES
    anchor = _anchor_for("eathome", _mk_dist(p50=100.0, weighted_mean=160.0))
    assert anchor == pytest.approx(160.0)


def test_nonsmooth_category_ignores_weighted_mean() -> None:
    """A non-smooth cat keeps the p50 anchor even when a mean is present.

    (transportation moved into the smooth set after the Build-1/Build-2 anchor
    switch, so use ``health`` — deliberately kept on the median.)"""
    assert "health" not in SMOOTH_MEAN_ANCHOR_CATEGORIES
    anchor = _anchor_for("health", _mk_dist(p50=100.0, weighted_mean=160.0))
    assert anchor == pytest.approx(100.0)


def test_smooth_category_falls_back_to_p50_when_mean_unpopulated() -> None:
    """Smooth cat with weighted_mean == 0 safely falls back to the median."""
    anchor = _anchor_for("eatout", _mk_dist(p50=100.0, weighted_mean=0.0))
    assert anchor == pytest.approx(100.0)


def test_delumped_aggregate_anchors_on_plain_mean() -> None:
    """Build 2: the de-lumped transportation backbone is now smooth and anchors
    on the plain weighted mean (high-side, accurate)."""
    assert "transportation" in SMOOTH_MEAN_ANCHOR_CATEGORIES
    anchor = _anchor_for("transportation", _mk_dist(p50=100.0, weighted_mean=160.0))
    assert anchor == pytest.approx(160.0)


def test_trimmed_anchor_category_uses_trim_not_mean_or_p50() -> None:
    """Build 2: household_goods retains outlier-distortion, so it anchors on the
    trim95 mean — above the median (a lift) but below the distorted plain mean
    (the accuracy correction)."""
    from shared.constants.categories import TRIMMED_MEAN_ANCHOR_CATEGORIES

    assert "household_goods" in TRIMMED_MEAN_ANCHOR_CATEGORIES
    assert "household_goods" not in SMOOTH_MEAN_ANCHOR_CATEGORIES
    anchor = _anchor_for(
        "household_goods",
        _mk_dist(p50=100.0, weighted_mean=200.0, trimmed_mean=130.0),
    )
    assert anchor == pytest.approx(130.0)  # trim, not the 200 mean or 100 median


def test_trimmed_anchor_falls_back_to_p50_when_trim_unpopulated() -> None:
    """A trimmed-anchor cat with no trimmed_mean safely falls back to p50."""
    anchor = _anchor_for(
        "household_goods", _mk_dist(p50=100.0, weighted_mean=0.0, trimmed_mean=0.0)
    )
    assert anchor == pytest.approx(100.0)


def test_weighted_mean_survives_annotate(pipeline_state: dict) -> None:
    """annotate_distributions must carry weighted_mean through (else the
    allocator never sees the mean for smooth cats).

    Restricted to the smooth cats present in the disaggregated match (the
    food/utilities lines); the smooth AGGREGATES (transportation, shopping,
    entertainment) only exist on the aggregated path, validated separately."""
    raw = pipeline_state["match"].distributions
    annotated = _annotate_for(40_000.0, pipeline_state)
    for cat in SMOOTH_MEAN_ANCHOR_CATEGORIES:
        if cat not in raw:
            continue
        assert annotated[cat].weighted_mean == pytest.approx(raw[cat].weighted_mean)


def test_match_populates_weighted_mean_for_smooth(pipeline_state: dict) -> None:
    """Matching populates a positive weighted_mean for the smooth set, and it
    sits at or above the median for these right-skewed-by-participation cats."""
    dists = pipeline_state["match"].distributions
    for cat in SMOOTH_MEAN_ANCHOR_CATEGORIES:
        if cat not in dists:  # smooth aggregates exist only on the aggregated path
            continue
        wm = dists[cat].weighted_mean
        assert wm >= 0.0
        # at least the universally-participated cats carry a positive mean
        if dists[cat].nonzero_rate > 0.5:
            assert wm > 0.0
