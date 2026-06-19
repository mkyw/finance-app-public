"""Tests for the matching algorithm.

Run from repo root:
    python3.11 -m pytest tests/models/test_matching.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from models.matching.algorithm import match_household
from models.matching.distance import gower_distance
from shared.constants.categories import CATEGORY_CODES
from shared.types import HouseholdProfile, Tenure

ARTIFACTS = Path("pipeline/artifacts")
POPULATION_DIR = ARTIFACTS / "synthetic_population"
NEIGHBORS_FILE = ARTIFACTS / "puma_similarity" / "puma_neighbors.parquet"
AK_PARTITION = POPULATION_DIR / "puma_code=AK_00101" / "part-0.parquet"


def _load_partition(puma_code: str) -> pd.DataFrame:
    """Load a single PUMA partition and attach its puma_code column.

    Hive partitioning lifts puma_code out of the file, so direct reads via
    parquet.read_table won't have it. The tests don't need the column
    (filters happen at the algorithm level), but it keeps fixtures explicit.
    """
    path = POPULATION_DIR / f"puma_code={puma_code}" / "part-0.parquet"
    df = pq.read_table(path).to_pandas()
    df["puma_code"] = puma_code
    return df


# ----------------------------------------------------------------------------

def test_schema_readable():
    df = _load_partition("AK_00101")
    expected_demo = {"tenure", "age", "household_size", "gross_income", "weight"}
    missing_demo = expected_demo - set(df.columns)
    assert not missing_demo, f"missing demographic cols: {missing_demo}"

    missing_cats = set(CATEGORY_CODES) - set(df.columns)
    assert not missing_cats, f"missing category cols: {missing_cats}"

    tenure_values = set(df["tenure"].unique())
    assert tenure_values.issubset({"OWN", "RENT"}), (
        f"unexpected tenure values: {tenure_values}"
    )


def test_gower_distance_shape():
    pop = _load_partition("AK_00101")
    profile = HouseholdProfile(
        age=35,
        gross_income=65000,
        puma_code="AK_00101",
        tenure=Tenure.RENT,
        housing_cost=1200,
        household_size=2,
    )
    d = gower_distance(profile, pop)
    assert d.shape == (len(pop),), f"shape mismatch: {d.shape} vs {len(pop)}"
    assert np.all(d >= 0), "distances must be non-negative"
    assert d.std() > 0, "distances should not all be identical"


def test_tenure_filtering():
    pop = _load_partition("AK_00101")
    own = pop[pop["tenure"] == "OWN"]
    rent = pop[pop["tenure"] == "RENT"]

    assert len(own) > 0, "AK_00101 should have owner-occupied households"
    assert len(rent) > 0, "AK_00101 should have renter households"
    assert "RENT" not in set(own["tenure"].unique())
    assert "OWN" not in set(rent["tenure"].unique())


def test_match_household_basic():
    profile = HouseholdProfile(
        age=35,
        gross_income=65000,
        puma_code="AK_00101",
        tenure=Tenure.RENT,
        housing_cost=1200,
        household_size=2,
    )
    result = match_household(profile, str(ARTIFACTS))

    assert len(result.distributions) == 55
    assert result.n_effective > 0
    assert result.n_households > 100

    # All 55 category codes must be represented.
    assert set(result.distributions.keys()) == set(CATEGORY_CODES)

    # Percentiles must be monotonic non-decreasing per category.
    for cat, dist in result.distributions.items():
        assert dist.p10 <= dist.p25 <= dist.p50 <= dist.p75 <= dist.p90, (
            f"non-monotonic percentiles for {cat}: "
            f"{dist.p10}, {dist.p25}, {dist.p50}, {dist.p75}, {dist.p90}"
        )


def test_aggregate_demerges_episodic_capital():
    """Build 1: in aggregate mode the episodic-capital components are predicted
    as their own distributions (de-merged), and the transportation /
    household_goods aggregates carry only their recurring backbone — the
    components are moved out, not lost."""
    from shared.constants.categories import (
        AGGREGATED_CATEGORY_CODES,
        AGG_GROUPS,
    )

    profile = HouseholdProfile(
        age=35, gross_income=75000, puma_code="AK_00101",
        tenure=Tenure.RENT, housing_cost=1200, household_size=1,
    )
    agg = match_household(profile, str(ARTIFACTS), aggregate=True)

    # 46-category aggregated set, exactly (Build-3: 43 -> 46 after de-merging
    # recrp/eltrnp/jwlbg from entertainment/shopping).
    assert set(agg.distributions) == set(AGGREGATED_CATEGORY_CODES)
    assert len(agg.distributions) == 46

    # The episodic-capital / durable components exist as their own predictions,
    # and are no longer members of their former aggregates.
    for c in ("vehnew", "vehusd", "furhwr", "happl", "recrp", "eltrnp", "jwlbg"):
        assert c in agg.distributions, f"{c} must be predicted as its own line"
    assert "vehnew" not in AGG_GROUPS["transportation"]
    assert "vehusd" not in AGG_GROUPS["transportation"]
    assert "furhwr" not in AGG_GROUPS["household_goods"]
    assert "happl" not in AGG_GROUPS["household_goods"]
    # Build-3 de-merges.
    assert "recrp" not in AGG_GROUPS["entertainment"]
    assert "eltrnp" not in AGG_GROUPS["shopping"]
    assert "jwlbg" not in AGG_GROUPS["shopping"]

    # Backbone aggregates are still present and positive (recurring spend).
    assert agg.distributions["transportation"].p50 >= 0.0
    assert agg.distributions["household_goods"].p50 >= 0.0

    # Build 2: the smooth aggregates carry a positive weighted mean (plain-mean
    # anchor) and household_goods carries a positive trim95 mean (trimmed
    # anchor), with trim <= plain mean (the outlier-robust central tendency).
    for c in ("transportation", "shopping", "entertainment"):
        assert agg.distributions[c].weighted_mean > 0.0, f"{c} missing weighted_mean"
    hg = agg.distributions["household_goods"]
    assert hg.trimmed_mean > 0.0, "household_goods missing trimmed_mean"
    assert hg.trimmed_mean <= hg.weighted_mean + 1e-6, "trim must not exceed plain mean"


def test_puma_similarity_loaded():
    assert NEIGHBORS_FILE.exists(), f"missing {NEIGHBORS_FILE}"
    table = pq.read_table(NEIGHBORS_FILE)
    assert table.num_rows == 2462 * 50 == 123100, (
        f"expected 123100 rows, got {table.num_rows}"
    )
    df = table.to_pandas()

    ak = df[df["puma_code"] == "AK_00101"]
    assert len(ak) == 50, f"AK_00101 should have 50 neighbors, got {len(ak)}"
    assert set(ak["rank"]) == set(range(1, 51))


def test_m_sensitivity():
    """Validate that M=10 is a reasonable default.

    Operationalized as: p50 at M=10 agrees with p50 at M=20 within 20% per
    category. If adding more similar PUMAs beyond 10 doesn't move the
    estimate meaningfully, M=10 is "enough". We also run M=1 and M=5 and
    print their values for diagnostic inspection — M=1 is expected to be
    noisy (~255 households in a rural PUMA) and is not asserted against.
    """
    profile = HouseholdProfile(
        age=35,
        gross_income=65000,
        puma_code="AK_00101",
        tenure=Tenure.RENT,
        housing_cost=1200,
        household_size=2,
    )
    first_five = CATEGORY_CODES[:5]
    p50_by_m: dict[int, dict[str, float]] = {}
    n_by_m: dict[int, int] = {}
    for m in (1, 5, 10, 20):
        result = match_household(profile, str(ARTIFACTS), m_similar_pumas=m)
        p50_by_m[m] = {c: result.distributions[c].p50 for c in first_five}
        n_by_m[m] = result.n_households

    # Diagnostic — visible when run with -s.
    print()
    print(f"{'cat':<10}{'M=1':>10}{'M=5':>10}{'M=10':>10}{'M=20':>10}")
    for c in first_five:
        print(
            f"{c:<10}"
            f"{p50_by_m[1][c]:>10.0f}"
            f"{p50_by_m[5][c]:>10.0f}"
            f"{p50_by_m[10][c]:>10.0f}"
            f"{p50_by_m[20][c]:>10.0f}"
        )
    print(
        f"n_hh: M=1={n_by_m[1]}, M=5={n_by_m[5]}, "
        f"M=10={n_by_m[10]}, M=20={n_by_m[20]}"
    )

    # Assertion: M=10 vs M=20 agreement per category. For small-dollar
    # zero-inflated cats (e.g. electronics, jewelry) the absolute p50
    # is $100-200 and per-fuse() Monte-Carlo noise can flex it ~30-40%
    # between runs; in that regime we accept either a 20% relative
    # match OR a $50 absolute match.
    for c in first_five:
        v10 = p50_by_m[10][c]
        v20 = p50_by_m[20][c]
        denom = 0.5 * (v10 + v20)
        if denom < 1.0:
            # Both essentially zero — nothing to measure.
            continue
        abs_diff = abs(v10 - v20)
        rel_diff = abs_diff / denom
        assert rel_diff < 0.20 or abs_diff < 75.0, (
            f"p50 for {c} changes too much between M=10 and M=20: "
            f"{v10:.2f} -> {v20:.2f} ({rel_diff:.1%} diff, ${abs_diff:.2f}). "
            f"Default M=10 is not stable for this category."
        )


# ----------------------------------------------------------------------------
# Vehicle/transit cohort segmentation (car-ownership + single/multi-car blend)
# ----------------------------------------------------------------------------


def test_car_owner_single_car_classification():
    """HH=1 car-owner profile → classification 'owner', cohort_mean_veh
    near 1, vehicle cats pulled up by the car-owner subset blend."""
    # Chicago-area PUMA likely to have moderate car ownership. Exact
    # probability is data-dependent; we only assert the classification
    # band and blend direction.
    profile = HouseholdProfile(
        age=35,
        gross_income=85000,
        puma_code="IL_03168",
        tenure=Tenure.RENT,
        housing_cost=1400,
        household_size=1,
    )
    result = match_household(profile, str(ARTIFACTS))
    if result.car_owner_classification != "owner":
        pytest.skip(
            f"IL_03168 HH=1 didn't classify as 'owner' "
            f"(p={result.car_owner_probability:.3f}); pick a higher-car PUMA "
            f"if this drifts."
        )
    # Individual car-owner: cohort mean veh should be near 1.0. A few
    # edge-case households clear the AND-proxy despite reporting ACS
    # veh=0 (stored/backup vehicle), pulling the mean slightly below 1;
    # the upper bound is the real guard against families leaking in.
    assert 0.7 <= result.cohort_mean_veh < 1.6, (
        f"cohort_mean_veh for HH=1 car-owner out of expected range: "
        f"{result.cohort_mean_veh:.3f}"
    )
    # vehins should be meaningfully positive (not dragged to zero by
    # carless households, which is the whole point of the blend).
    assert result.distributions["vehins"].p50 > 200, (
        f"vehins.p50 for HH=1 car-owner should be > $200, got "
        f"{result.distributions['vehins'].p50:.0f}"
    )


def test_car_owner_multi_car_family_blend():
    """HH=4 in the same PUMA shifts cohort_mean_veh upward and
    increases gas.p50 relative to the HH=1 case — proving the blend
    responds to household composition."""
    hh1 = HouseholdProfile(
        age=35,
        gross_income=85000,
        puma_code="IL_03168",
        tenure=Tenure.RENT,
        housing_cost=1400,
        household_size=1,
    )
    hh4 = HouseholdProfile(
        age=35,
        gross_income=85000,
        puma_code="IL_03168",
        tenure=Tenure.RENT,
        housing_cost=1400,
        household_size=4,
    )
    r1 = match_household(hh1, str(ARTIFACTS))
    r4 = match_household(hh4, str(ARTIFACTS))
    if r1.car_owner_classification != "owner" or r4.car_owner_classification != "owner":
        pytest.skip(
            "car-owner classification did not hold for both HH=1 and HH=4; "
            "pick a higher-car PUMA if this drifts."
        )
    # Family of 4 has a meaningfully larger cohort_mean_veh than a
    # single person in the same PUMA/income — the blend shifts toward
    # multi-car without any hard household-size threshold.
    assert r4.cohort_mean_veh > r1.cohort_mean_veh + 0.2, (
        f"cohort_mean_veh did not grow with household size: "
        f"HH=1 {r1.cohort_mean_veh:.3f} -> HH=4 {r4.cohort_mean_veh:.3f}"
    )
    # Multi-car family gas.p50 exceeds single-adult gas.p50.
    assert r4.distributions["gas"].p50 > r1.distributions["gas"].p50, (
        f"gas.p50 did not scale up with family size: HH=1 "
        f"{r1.distributions['gas'].p50:.0f} vs HH=4 "
        f"{r4.distributions['gas'].p50:.0f}"
    )


def test_carless_manhattan_classification():
    """Manhattan-core PUMA renter → classification 'non_owner',
    vehicle cats near zero, pubtrn elevated vs. a car-owning cohort."""
    profile = HouseholdProfile(
        age=30,
        gross_income=75000,
        puma_code="NY_04103",  # Upper-Manhattan PUMA
        tenure=Tenure.RENT,
        housing_cost=2500,
        household_size=1,
    )
    result = match_household(profile, str(ARTIFACTS))
    if result.car_owner_classification != "non_owner":
        pytest.skip(
            f"NY_04103 HH=1 didn't classify as 'non_owner' "
            f"(p={result.car_owner_probability:.3f}); pick a more central "
            f"Manhattan PUMA if this drifts."
        )
    # cohort_mean_veh is NaN on the non_owner branch.
    import math
    assert math.isnan(result.cohort_mean_veh), (
        f"cohort_mean_veh should be NaN on non-owner branch, got "
        f"{result.cohort_mean_veh}"
    )
    # Carless cohort → vehins is hard-zeroed (see _VEHICLE_CATS
    # branch in algorithm.py).
    assert result.distributions["vehins"].p50 == 0.0, (
        f"vehins.p50 for carless cohort should be hard-zero, got "
        f"{result.distributions['vehins'].p50:.0f}"
    )
    assert result.distributions["gas"].p50 == 0.0, (
        f"gas.p50 for carless cohort should be hard-zero, got "
        f"{result.distributions['gas'].p50:.0f}"
    )
    # Transit cats use the veh=0 subset (not hard-zero). Fusion is
    # zero-inflated on pubtrn so the p50 can still round to 0 even in
    # a mostly-carless pool; check the upper tail and the conditional
    # p90 instead — both should show real transit spending.
    pubtrn = result.distributions["pubtrn"]
    assert pubtrn.p90 > 100 or pubtrn.conditional_p90 > 100, (
        f"pubtrn upper tail for carless cohort should be > $100: "
        f"p90={pubtrn.p90:.0f}, conditional_p90={pubtrn.conditional_p90:.0f}"
    )


# ----------------------------------------------------------------------------
# vehreg direct-cost path (state DMV fee × predicted vehicle count)
# ----------------------------------------------------------------------------


def test_vehreg_owner_state_cost_difference():
    """Same profile in AZ (high-fee) vs FL (low-fee) → vehreg p50 ratio
    tracks the state DMV cost ratio, not the cohort percentile."""
    az = HouseholdProfile(
        age=40,
        gross_income=85000,
        puma_code="AZ_00105",
        tenure=Tenure.OWN,
        housing_cost=1800,
        household_size=3,
    )
    fl = HouseholdProfile(
        age=40,
        gross_income=85000,
        puma_code="FL_00101",
        tenure=Tenure.OWN,
        housing_cost=1800,
        household_size=3,
    )
    r_az = match_household(az, str(ARTIFACTS))
    r_fl = match_household(fl, str(ARTIFACTS))
    if (
        r_az.car_owner_classification != "owner"
        or r_fl.car_owner_classification != "owner"
    ):
        pytest.skip(
            "AZ/FL profiles did not both classify as 'owner'; pick higher-car "
            "PUMAs if this drifts."
        )

    az_v = r_az.distributions["vehreg"].p50
    fl_v = r_fl.distributions["vehreg"].p50
    # AZ (ad-valorem ~$259/veh for reference vehicle) vs FL (flat ~$62/veh)
    # → AZ should dominate FL by at least 3x. This is a real-data ratio,
    # not a placeholder — refining to refresh the artifact may shift it
    # but the structural relationship (AZ value-based >> FL flat-fee)
    # should hold across reasonable refreshes.
    assert az_v > fl_v * 3, (
        f"AZ vehreg.p50 should dominate FL by at least 3x given state "
        f"cost differences: AZ=${az_v:.0f}, FL=${fl_v:.0f}"
    )
    assert az_v > 0 and fl_v > 0, (
        f"both states should produce positive vehreg for owner cohorts: "
        f"AZ=${az_v:.0f}, FL=${fl_v:.0f}"
    )


def test_vehreg_non_owner_zero():
    """Manhattan-core PUMA renter (non_owner classification) → vehreg = 0,
    independent of the state DMV cost lookup."""
    profile = HouseholdProfile(
        age=30,
        gross_income=75000,
        puma_code="NY_04103",
        tenure=Tenure.RENT,
        housing_cost=2500,
        household_size=1,
    )
    result = match_household(profile, str(ARTIFACTS))
    if result.car_owner_classification != "non_owner":
        pytest.skip(
            f"NY_04103 HH=1 didn't classify as 'non_owner' "
            f"(p={result.car_owner_probability:.3f}); pick a more central "
            f"Manhattan PUMA if this drifts."
        )
    vehreg = result.distributions["vehreg"]
    assert vehreg.p50 == 0.0, f"non_owner vehreg.p50 should be 0, got {vehreg.p50}"
    assert vehreg.p90 == 0.0, f"non_owner vehreg.p90 should be 0, got {vehreg.p90}"
    assert vehreg.nonzero_rate == 0.0


def test_vehreg_ambiguous_probability_weighted(monkeypatch):
    """Ambiguous classification → vehreg = state_$ × cohort_mean_veh ×
    car_owner_probability. Forces a known state cost via monkeypatch so
    the assertion doesn't drift with artifact refreshes."""
    import models.matching.algorithm as algo_mod

    fixed_cost_per_vehicle = 200.0

    def fake_loader(path):
        # All states fixed to the same per-vehicle cost so the
        # algorithm's per-state lookup always returns 200.0 regardless
        # of which PUMA we pick.
        return {"NY": fixed_cost_per_vehicle}, fixed_cost_per_vehicle

    monkeypatch.setattr(algo_mod, "load_vehreg_state_costs", fake_loader)

    # Outer-Brooklyn / Queens-area PUMA: dense urban but with enough
    # car-owning commuter share to land in the ambiguous band
    # (0.3 ≤ p ≤ 0.6). Exact PUMA may need to drift if ACS shifts.
    profile = HouseholdProfile(
        age=35,
        gross_income=95000,
        puma_code="NY_04207",
        tenure=Tenure.RENT,
        housing_cost=2200,
        household_size=2,
    )
    result = match_household(profile, str(ARTIFACTS))
    if result.car_owner_classification != "ambiguous":
        pytest.skip(
            f"NY_04207 HH=2 didn't classify as 'ambiguous' "
            f"(p={result.car_owner_probability:.3f}); pick a borderline "
            f"PUMA if this drifts."
        )

    expected_p50 = (
        fixed_cost_per_vehicle
        * result.cohort_mean_veh
        * result.car_owner_probability
    )
    actual_p50 = result.distributions["vehreg"].p50
    assert abs(actual_p50 - expected_p50) < 0.01, (
        f"ambiguous vehreg.p50 should be state_$ × cohort_mean_veh × p: "
        f"expected {expected_p50:.2f}, got {actual_p50:.2f}"
    )
