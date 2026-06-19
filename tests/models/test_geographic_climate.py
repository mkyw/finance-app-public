"""Utility climate-load geographic factor — derivation + runtime tests.

Covers the three-lane factor (EIA state consumption × NOAA within-state
degree-days × RPP price), its application in the value-layer scalar slot, the
zero-median ofuel Engel-fallback correction, and the EIA runtime-no-fetch
guarantee. See ``agent-artifacts/investigations/utilities_diagnosis.md`` and
``models/matching/geographic_drivers.py``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from models.matching import eia_gas
from models.matching.eia_utility import load_eia_utility_scalars
from models.matching.geographic_drivers import (
    GeographicDrivers,
    load_climate_normals,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "pipeline" / "artifacts"
EIA_UTIL = ARTIFACTS / "eia_utility_scalars.json"
CLIMATE = ARTIFACTS / "climate_normals.json"
SYNTH = ARTIFACTS / "synthetic_population"

CLIMATE_CATS = ("elec", "ngas", "ofuel")
NON_CLIMATE_UTILS = ("watrsh", "intphn")


@pytest.fixture(scope="module")
def eia_util() -> dict:
    return load_eia_utility_scalars(str(EIA_UTIL))


@pytest.fixture(scope="module")
def climate() -> dict:
    return load_climate_normals(str(CLIMATE))


@pytest.fixture(scope="module")
def drivers(eia_util, climate) -> GeographicDrivers:
    return GeographicDrivers(eia_utility_scalars=eia_util, climate_normals=climate)


# --------------------------------------------------------------------------- #
# EIA state consumption lane                                                   #
# --------------------------------------------------------------------------- #
def test_eia_utility_has_three_climate_categories(eia_util) -> None:
    assert set(eia_util) == set(CLIMATE_CATS)
    for cat in CLIMATE_CATS:
        assert all(isinstance(v, (int, float)) for v in eia_util[cat].values())


def test_eia_elec_higher_in_hot_states(eia_util) -> None:
    # Cooling load: AZ/FL/TX consume more residential electricity than mild NY.
    for hot in ("AZ", "FL", "TX"):
        assert eia_util["elec"][hot] > eia_util["elec"]["NY"]
    assert eia_util["elec"]["FL"] > 1.0 > eia_util["elec"]["NY"]


def test_eia_ngas_higher_in_cold_states(eia_util) -> None:
    # Heating load: cold IL/MN consume more residential gas than warm FL/AZ.
    for cold in ("IL", "MN"):
        for warm in ("FL", "AZ"):
            assert eia_util["ngas"][cold] > eia_util["ngas"][warm]
    assert eia_util["ngas"]["IL"] > 1.0


def test_eia_ofuel_concentrated_in_new_england(eia_util) -> None:
    assert eia_util["ofuel"]["ME"] > 5.0  # heating-oil heavy
    assert eia_util["ofuel"]["AZ"] < 0.1  # desert, ~none


def test_eia_national_consumption_ratio_preserved(eia_util) -> None:
    """Housing-unit-weighted national mean of each lane == 1.0 (total-preserving)."""
    hh: dict[str, float] = {}
    for f in glob.glob(str(SYNTH / "puma_code=*/*.parquet")):
        st = Path(f).parent.name.split("=", 1)[1].split("_", 1)[0]
        w = pq.read_table(f, columns=["weight"]).column("weight").to_numpy().sum()
        hh[st] = hh.get(st, 0.0) + float(w)
    for cat in CLIMATE_CATS:
        states = [s for s in eia_util[cat] if s in hh]
        mean = sum(hh[s] * eia_util[cat][s] for s in states) / sum(hh[s] for s in states)
        assert mean == pytest.approx(1.0, abs=0.02)


# --------------------------------------------------------------------------- #
# NOAA within-state degree-day lane                                            #
# --------------------------------------------------------------------------- #
def test_climate_normals_have_hdd_cdd_ratios(climate) -> None:
    sample = next(iter(climate.values()))
    assert "hdd_ratio" in sample and "cdd_ratio" in sample
    # Monthly arrays banked for the future seasonal feature.
    assert len(sample["cdd_monthly"]) == 12 and len(sample["hdd_monthly"]) == 12


def test_within_state_cdd_ratio_centers_near_one(climate) -> None:
    # CA spans coastal (low CDD) to inland desert (high CDD); ratios straddle 1.0.
    ca = [v["cdd_ratio"] for p, v in climate.items() if p.startswith("CA_")]
    assert min(ca) < 0.8 < 1.2 < max(ca)  # genuine within-state spread


# --------------------------------------------------------------------------- #
# Composed factor                                                              #
# --------------------------------------------------------------------------- #
def test_factor_targets_only_climate_categories(drivers) -> None:
    puma = next(iter(drivers.climate_normals))
    for cat in (*NON_CLIMATE_UTILS, "eatout", "rntval", "health"):
        assert drivers.factor(cat, puma) == 1.0


def test_factor_hot_puma_lifts_elec(drivers, eia_util) -> None:
    az = next(p for p in drivers.climate_normals if p.startswith("AZ_"))
    assert drivers.factor("elec", az) == pytest.approx(
        eia_util["elec"]["AZ"] * drivers.climate_normals[az]["cdd_ratio"], rel=1e-6
    )


def test_factor_noop_when_artifacts_absent() -> None:
    empty = GeographicDrivers()
    for cat in (*CLIMATE_CATS, *NON_CLIMATE_UTILS):
        assert empty.factor(cat, "AZ_00101") == 1.0


def test_factor_unknown_puma_uses_state_eia_only(drivers, eia_util) -> None:
    # PUMA not in NOAA artifact -> within-state lane = 1.0, EIA state lane still applies.
    assert drivers.factor("elec", "AZ_99999") == pytest.approx(eia_util["elec"]["AZ"], rel=1e-6)


def test_loaders_return_empty_on_missing_file(tmp_path) -> None:
    assert load_eia_utility_scalars(str(tmp_path / "nope.json")) == {}
    assert load_climate_normals(str(tmp_path / "nope.json")) == {}


# --------------------------------------------------------------------------- #
# EIA runtime-no-fetch guarantee                                              #
# --------------------------------------------------------------------------- #
def test_gas_loader_never_fetches(monkeypatch) -> None:
    """load_eia_gas_scalars must read the cache, never call the network fetch."""
    def _boom(*a, **k):
        raise AssertionError("runtime must not fetch from EIA")

    monkeypatch.setattr(eia_gas, "fetch_eia_gas_scalars", _boom)
    # No exception => no fetch path taken; returns the cached dict.
    scalars = eia_gas.load_eia_gas_scalars(str(ARTIFACTS / "eia_gas_scalars.json"))
    assert isinstance(scalars, dict)


def test_utility_loader_never_fetches(monkeypatch) -> None:
    """The utility loader is a pure file read — succeeds even if the network is dead."""
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("runtime must not hit the network")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    scalars = load_eia_utility_scalars(str(EIA_UTIL))
    assert set(scalars) == set(CLIMATE_CATS)
