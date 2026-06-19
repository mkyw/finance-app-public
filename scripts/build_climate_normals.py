#!/usr/bin/env python3
"""Build the NOAA within-state degree-day artifact for the utility climate factor.

Static reference data (NOAA 1991-2020 Climate Normals) -> per-PUMA within-state
heating/cooling degree-day ratios, normalized so each state's housing-unit-weighted
mean PUMA == 1.0. This is the *within-state* climate lane of the three-lane utility
geographic factor (EIA state baseline x NOAA within-state x RPP price); see
`agent-artifacts/investigations/utilities_diagnosis.md` and `models/matching/
geographic_drivers.py`.

The state-mean normalization is the anti-double-count guarantee: NOAA contributes
exactly 1.0 at the state (hu-weighted) average, so it only *redistributes* climate
within a state (coastal-CA down, inland-CA up) and never re-touches the state level
that EIA's consumption baseline already carries. Because the normalization is
hu-weighted, applying the factor across a state's households preserves the state
total exactly.

Pipeline (all inputs repo-local):
  1. NOAA by-station monthly normals (tar.gz) -> per-station annual HDD/CDD
     (sum of the 12 MLY-HTDD-NORMAL / MLY-CLDD-NORMAL months) + lat/lon. Only the
     ~7,304 temperature stations carry degree-days; precip-only stations are skipped.
  2. County gazetteer centroids -> nearest degree-day station (cKDTree) -> county HDD/CDD.
  3. city_puma_map.json county->PUMA, inverted to PUMA->counties -> PUMA HDD/CDD
     (mean over its counties).
  4. Within each state, normalize PUMA HDD/CDD by the hu-weighted state mean
     (hu = sum of synthetic-population weights per PUMA) -> ratios centered on 1.0.

Monthly HDD/CDD are also banked per PUMA (for a future month-to-month seasonal
feature); the current factor uses only the annual ratio.

Output: pipeline/artifacts/climate_normals.json
Run (static; refresh per-decade when NOAA releases new normals):
  .venv/bin/python scripts/build_climate_normals.py
"""
from __future__ import annotations

import csv
import glob
import io
import json
import math
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

_REPO = Path(__file__).resolve().parent.parent
_NOAA_TAR = (
    _REPO
    / "pipeline/fusionData/geo-raw/noaa-climate-normals"
    / "us-climate-normals_1991-2020_v1.0.1_monthly_multivariate_by-station_c20230404.tar.gz"
)
_GAZ = _REPO / "pipeline/fusionData/geo-raw/gazetteer/2023_Gaz_counties_national.txt"
_CITY_PUMA = _REPO / "pipeline/artifacts/city_puma_map.json"
_SYNTH = _REPO / "pipeline/artifacts/synthetic_population"
_OUT = _REPO / "pipeline/artifacts/climate_normals.json"

_HDD = "MLY-HTDD-NORMAL"
_CDD = "MLY-CLDD-NORMAL"
_K_NEAREST = 3  # average the k nearest stations to a county centroid (robustness)


def _f(x: str) -> float | None:
    """Parse a NOAA normal value; NOAA flags missing as -8888/-9999 / blank."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    # NOAA special/missing sentinels are large-magnitude negatives.
    if v <= -100.0:
        return None
    return v


def parse_stations() -> dict:
    """Return {station_id: {lat, lon, hdd_annual, cdd_annual, hdd_monthly, cdd_monthly}}."""
    out: dict[str, dict] = {}
    with tarfile.open(_NOAA_TAR, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".csv"):
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
            if reader.fieldnames is None or _HDD not in reader.fieldnames:
                continue  # precip-only / non-temperature station
            hdd_m = [None] * 13  # index by month 1..12
            cdd_m = [None] * 13
            lat = lon = None
            sid = None
            for row in reader:
                sid = row["STATION"]
                lat = row["LATITUDE"]
                lon = row["LONGITUDE"]
                try:
                    mo = int(row["month"])
                except (TypeError, ValueError):
                    continue
                if 1 <= mo <= 12:
                    hdd_m[mo] = _f(row.get(_HDD, ""))
                    cdd_m[mo] = _f(row.get(_CDD, ""))
            if sid is None or lat is None:
                continue
            hm = [v for v in hdd_m[1:] if v is not None]
            cm = [v for v in cdd_m[1:] if v is not None]
            if len(hm) < 12 or len(cm) < 12:
                continue  # require complete monthly coverage
            try:
                latf, lonf = float(lat), float(lon)
            except ValueError:
                continue
            out[sid] = {
                "lat": latf,
                "lon": lonf,
                "hdd_annual": float(sum(hm)),
                "cdd_annual": float(sum(cm)),
                "hdd_monthly": [round(v, 1) for v in hdd_m[1:]],
                "cdd_monthly": [round(v, 1) for v in cdd_m[1:]],
            }
    return out


def load_county_centroids() -> dict[str, tuple[float, float]]:
    """{county_fips5: (lat, lon)} from the Census gazetteer."""
    out = {}
    with open(_GAZ, encoding="latin-1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # header has trailing whitespace on INTPTLONG
        cols = {c.strip(): c for c in reader.fieldnames}
        for row in reader:
            fips = row[cols["GEOID"]].strip().zfill(5)
            try:
                out[fips] = (float(row[cols["INTPTLAT"]]), float(row[cols["INTPTLONG"]]))
            except ValueError:
                continue
    return out


def puma_household_weights() -> dict[str, float]:
    """{puma_code: sum of synthetic-population weights} (housing-unit weight)."""
    out = {}
    for f in glob.glob(str(_SYNTH / "puma_code=*/*.parquet")):
        puma = Path(f).parent.name.split("=", 1)[1]
        w = pq.read_table(f, columns=["weight"]).column("weight").to_numpy().sum()
        out[puma] = out.get(puma, 0.0) + float(w)
    return out


def _equirect_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Project lat/lon to a local Cartesian grid (km) good enough for nearest-station."""
    lat0 = np.radians(lat.mean())
    x = np.radians(lon) * math.cos(lat0) * 6371.0
    y = np.radians(lat) * 6371.0
    return np.column_stack([x, y])


def main() -> None:
    print("=== parsing NOAA stations (degree-day stations only) ===")
    stations = parse_stations()
    print(f"  {len(stations)} stations with complete monthly HDD/CDD")

    sids = list(stations)
    s_lat = np.array([stations[s]["lat"] for s in sids])
    s_lon = np.array([stations[s]["lon"] for s in sids])
    s_hdd = np.array([stations[s]["hdd_annual"] for s in sids])
    s_cdd = np.array([stations[s]["cdd_annual"] for s in sids])
    s_hdd_m = np.array([stations[s]["hdd_monthly"] for s in sids])
    s_cdd_m = np.array([stations[s]["cdd_monthly"] for s in sids])

    # Project once on a shared origin so county and station coords are comparable.
    all_lat = np.concatenate([s_lat])
    lat0 = np.radians(all_lat.mean())

    def proj(lat, lon):
        x = np.radians(lon) * math.cos(lat0) * 6371.0
        y = np.radians(lat) * 6371.0
        return np.column_stack([x, y])

    tree = cKDTree(proj(s_lat, s_lon))

    print("=== county centroids -> nearest stations ===")
    counties = load_county_centroids()
    c_fips = list(counties)
    c_lat = np.array([counties[c][0] for c in c_fips])
    c_lon = np.array([counties[c][1] for c in c_fips])
    _, idx = tree.query(proj(c_lat, c_lon), k=_K_NEAREST)
    if idx.ndim == 1:
        idx = idx[:, None]
    county_hdd = s_hdd[idx].mean(axis=1)
    county_cdd = s_cdd[idx].mean(axis=1)
    county_hdd_m = s_hdd_m[idx].mean(axis=1)
    county_cdd_m = s_cdd_m[idx].mean(axis=1)
    county_dd = {
        c_fips[i]: (county_hdd[i], county_cdd[i], county_hdd_m[i], county_cdd_m[i])
        for i in range(len(c_fips))
    }

    print("=== PUMA <- counties (invert city_puma_map) ===")
    city = json.loads(_CITY_PUMA.read_text())
    county_to_pumas: dict[str, list[str]] = city["county"]
    puma_to_counties: dict[str, set[str]] = defaultdict(set)
    for cfips, pumas in county_to_pumas.items():
        cf = cfips.strip().zfill(5)
        for p in pumas:
            puma_to_counties[p].add(cf)

    print("=== PUMA degree-days (mean over counties) ===")
    puma_hdd: dict[str, float] = {}
    puma_cdd: dict[str, float] = {}
    puma_hdd_m: dict[str, list] = {}
    puma_cdd_m: dict[str, list] = {}
    for puma, cset in puma_to_counties.items():
        vals = [county_dd[c] for c in cset if c in county_dd]
        if not vals:
            continue
        puma_hdd[puma] = float(np.mean([v[0] for v in vals]))
        puma_cdd[puma] = float(np.mean([v[1] for v in vals]))
        puma_hdd_m[puma] = list(np.mean([v[2] for v in vals], axis=0))
        puma_cdd_m[puma] = list(np.mean([v[3] for v in vals], axis=0))

    print("=== hu-weighted within-state normalization (state mean -> 1.0) ===")
    hu = puma_household_weights()
    by_state_pumas: dict[str, list[str]] = defaultdict(list)
    for puma in puma_hdd:
        by_state_pumas[puma.split("_", 1)[0]].append(puma)

    by_puma = {}
    state_means = {}
    for state, pumas in by_state_pumas.items():
        w = np.array([hu.get(p, 0.0) for p in pumas])
        if w.sum() <= 0:
            w = np.ones(len(pumas))  # fall back to simple mean if no weights
        hdd = np.array([puma_hdd[p] for p in pumas])
        cdd = np.array([puma_cdd[p] for p in pumas])
        hdd_mean = float(np.average(hdd, weights=w))
        cdd_mean = float(np.average(cdd, weights=w))
        state_means[state] = {"hdd_annual_mean": hdd_mean, "cdd_annual_mean": cdd_mean}
        for p in pumas:
            by_puma[p] = {
                "hdd_ratio": round(puma_hdd[p] / hdd_mean, 4) if hdd_mean > 0 else 1.0,
                "cdd_ratio": round(puma_cdd[p] / cdd_mean, 4) if cdd_mean > 0 else 1.0,
                "hdd_annual": round(puma_hdd[p], 1),
                "cdd_annual": round(puma_cdd[p], 1),
                "hdd_monthly": [round(v, 1) for v in puma_hdd_m[p]],
                "cdd_monthly": [round(v, 1) for v in puma_cdd_m[p]],
            }

    artifact = {
        "metadata": {
            "source": "NOAA 1991-2020 U.S. Climate Normals (monthly multivariate, by-station)",
            "normals_field": {"hdd": _HDD, "cdd": _CDD, "base_f": 65},
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "n_stations_used": len(stations),
            "n_pumas": len(by_puma),
            "join": "PUMA -> counties (city_puma_map) -> nearest %d stations (gazetteer centroid) -> annual DD; within-state hu-weighted mean normalized to 1.0"
            % _K_NEAREST,
            "note": "Within-state ratio centered on the housing-unit-weighted state mean (=1.0) so the lane redistributes climate within a state and preserves the state total. Monthly arrays banked for a future seasonal feature; current factor uses the annual ratio.",
        },
        "by_puma": by_puma,
        "by_state_mean": state_means,
    }
    _OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    print(f"=== wrote {_OUT} ({len(by_puma)} PUMAs, {len(state_means)} states) ===")


if __name__ == "__main__":
    main()
