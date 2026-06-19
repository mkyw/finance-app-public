"""CE-PCE undercapture correction — derivation + runtime tests.

Covers the value-layer factor artifact (decontamination, apparel-only blend,
eatout-anomaly resolution) and its runtime application in the anchor
value-scaling slot (``_category_scalar``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.matching.pce_scaler import load_pce_factors, resolve_pce_factor

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "pipeline" / "artifacts"
PCE_JSON = ARTIFACTS / "pce_correction.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(PCE_JSON.read_text())


# --------------------------------------------------------------------------- #
# Derivation                                                                   #
# --------------------------------------------------------------------------- #
def test_correctable_set_is_eatout_and_shopping(artifact) -> None:
    # Build-3 added the carved personal-care recall (pcare) + entertainment
    # services (oesrv) factors against their final recomposed buckets.
    assert set(artifact["_meta"]["correctable_categories"]) == {
        "eatout", "shopping", "cloftw", "pcare", "oesrv"
    }


def test_entertainment_deferred_factor_is_one(artifact) -> None:
    # The entertainment AGGREGATE code stays 1.0 — the services lift applies at
    # its oesrv member (Build-3), not the aggregate code.
    assert artifact["factors"]["entertainment"] == 1.0
    assert artifact["factors"]["oesrv"] == 1.6
    assert "entertainment" in artifact["_meta"]["deferred"]


def test_shopping_decontamination_excludes_durables(artifact) -> None:
    comp = artifact["details"]["shopping"]["components"]
    # apparel corrected; electronics + jewelry excluded (durable, not recall)
    assert comp["cloftw"]["corrected"] is True
    assert comp["eltrnp"]["corrected"] is False
    assert comp["jwlbg"]["corrected"] is False
    assert artifact["factors"]["eltrnp"] == 1.0
    assert artifact["factors"]["jwlbg"] == 1.0
    # the excluded PCE lines are recorded for provenance
    assert "Jewelry and watches" in comp["jwlbg"]["excluded_lines"]
    assert any("Televisions" in s for s in comp["eltrnp"]["excluded_lines"])


def test_factor_is_inverse_of_decontaminated_ratio(artifact) -> None:
    eo = artifact["details"]["eatout"]
    assert eo["factor"] == pytest.approx(1.0 / eo["decontaminated_ratio"], rel=1e-3)
    cl = artifact["details"]["shopping"]["components"]["cloftw"]
    assert cl["factor"] == pytest.approx(1.0 / cl["decontaminated_ratio"], rel=1e-3)


def test_shopping_aggregate_is_weighted_component_blend(artifact) -> None:
    det = artifact["details"]["shopping"]
    w = det["within_shopping_weights"]
    comp = det["components"]
    blend = (w["cloftw"] * comp["cloftw"]["factor"]
             + w["eltrnp"] * 1.0 + w["jwlbg"] * 1.0)
    assert artifact["factors"]["shopping"] == pytest.approx(blend, rel=1e-3)
    # apparel dominates the bundle
    assert w["cloftw"] > w["eltrnp"] > w["jwlbg"]


def test_eatout_anomaly_resolved_not_the_bad_cell(artifact) -> None:
    # The raw-dollar ratio (~0.61 pre-COVID) gives factor ~1.65 — NOT the
    # erroneous 0.31 ratio cell (which would give ~3.2). Guard the resolution.
    eo = artifact["details"]["eatout"]
    assert 0.55 < eo["decontaminated_ratio"] < 0.66
    assert 1.5 < eo["factor"] < 2.0
    assert "Purchased meals and beverages" in eo["surviving_lines"]
    assert "Food supplied to civilians" in eo["excluded_lines"]  # employee meals


def test_necessities_and_other_discretionary_are_unity(artifact) -> None:
    f = artifact["factors"]
    for code in ("eathome", "elec", "health", "rntval", "transportation",
                 "household_goods", "educ", "finpay"):
        assert f[code] == 1.0


# --------------------------------------------------------------------------- #
# Runtime — loader + scaling slot                                              #
# --------------------------------------------------------------------------- #
def test_loader_and_resolver_defaults() -> None:
    factors = load_pce_factors(str(PCE_JSON))
    assert resolve_pce_factor(factors, "eatout") > 1.0
    assert resolve_pce_factor(factors, "entertainment") == 1.0
    # unknown / non-correctable default to 1.0
    assert resolve_pce_factor(factors, "does_not_exist") == 1.0
    assert resolve_pce_factor({}, "eatout") == 1.0  # missing artifact -> no-op


def test_loader_missing_file_is_noop() -> None:
    assert load_pce_factors(str(ARTIFACTS / "no_such_file.json")) == {}


def test_category_scalar_applies_pce_factor() -> None:
    """The factor multiplies into the anchor scalar; non-correctable unaffected."""
    from models.matching.algorithm import _category_scalar
    from models.matching.cpi_scaler import load_cpi_scalars
    from models.matching.eia_gas import load_eia_gas_scalars
    from models.matching.rpp_scaler import load_rpp_scalars
    from shared.types import HouseholdProfile, Tenure

    prof = HouseholdProfile(age=35, gross_income=65000, puma_code="AK_00101",
                            tenure=Tenure.RENT, housing_cost=1500, household_size=2)
    cpi = load_cpi_scalars(str(ARTIFACTS / "cpi_scalars.json"))
    rpp = load_rpp_scalars(str(ARTIFACTS / "rpp_scalars.json"))
    eia = load_eia_gas_scalars(str(ARTIFACTS / "eia_gas_scalars.json"))
    factors = load_pce_factors(str(PCE_JSON))

    for cat in ("eatout", "cloftw"):
        off = _category_scalar(cat, prof, cpi, rpp, eia, None, {})
        on = _category_scalar(cat, prof, cpi, rpp, eia, None, factors)
        assert on / off == pytest.approx(factors[cat], rel=1e-6)

    # non-correctable categories: scalar identical with/without the factor map
    for cat in ("eathome", "elec", "entertainment", "eltrnp"):
        off = _category_scalar(cat, prof, cpi, rpp, eia, None, {})
        on = _category_scalar(cat, prof, cpi, rpp, eia, None, factors)
        assert on == pytest.approx(off, rel=1e-9)


def test_end_to_end_lift_on_eatout_aggregated_path() -> None:
    """match_household lifts eatout; entertainment (deferred) is untouched."""
    import shutil

    from models.matching.algorithm import match_household
    from shared.types import HouseholdProfile, Tenure

    prof = HouseholdProfile(age=35, gross_income=90000, puma_code="AK_00101",
                            tenure=Tenure.RENT, housing_cost=1800, household_size=1)
    factors = load_pce_factors(str(PCE_JSON))

    on = match_household(prof, str(ARTIFACTS), aggregate=True)
    off_path = PCE_JSON.with_suffix(".json.testoff")
    shutil.move(str(PCE_JSON), str(off_path))
    try:
        off = match_household(prof, str(ARTIFACTS), aggregate=True)
    finally:
        shutil.move(str(off_path), str(PCE_JSON))

    # eatout p50 scales by exactly the eatout factor
    assert (on.distributions["eatout"].p50 / off.distributions["eatout"].p50
            == pytest.approx(factors["eatout"], rel=1e-6))
    # Build-3: entertainment now LIFTS via its oesrv services factor (1.6); the
    # aggregate code stays 1.0 but the member carries the recall factor.
    assert (on.distributions["entertainment"].p50
            > off.distributions["entertainment"].p50)
    # shopping (recurring core = cloftw apparel ×1.83 + carved pcare ×2.27, no
    # durables) lifts to a blend between the two member factors.
    ratio = on.distributions["shopping"].p50 / off.distributions["shopping"].p50
    assert factors["cloftw"] - 0.15 < ratio < factors["pcare"] + 0.05
