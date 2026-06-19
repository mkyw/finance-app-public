"""Build pipeline/artifacts/local_tax_rates.json — a comprehensive local /
municipal income-tax rate artifact keyed by Census place FIPS (7-digit,
STATEFP+PLACEFP) and county FIPS (5-digit STATEFP+COUNTYFP).

Stage-1 offline builder for the local-tax layer (the runtime loader, built
separately, never fetches). Network fetches ARE allowed here:

    python3.11 scripts/build_local_tax_rates.py

Sources (live-fetched where available, curated inline fallback otherwise):
  * Ohio  — Dept of Taxation "The Finder" municipal income-tax rate table
            (~650 levying municipalities; JEDD/JEDZ joint districts skipped).
  * Pennsylvania — DCED Official EIT Tax Register (TOTAL RESIDENT EIT rate =
            municipality + school-district combined).
  * Census place->county crosswalk (codes2020) — for the county-tax states
            (MD, IN) only, to wire place_to_county.
  * MD / IN / MI / KY / MO / AL / DE / NY — curated inline dicts (2025/2026
            rates from DOR notices / municipal schedules; ~0.1pp accuracy OK).

Name->FIPS matching reuses the resolver normalizer
(apps.api.profiles.city_resolver._normalize) so source municipality names are
normalized identically to the city_puma_map.json keys. Unmatched names are
unreachable (only city_puma_map places can be user-selected) and are dropped +
sampled in _meta.match_report.

Exit code 0 on success; the script fails LOUDLY (non-zero) on any hard
validation violation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
import logging
import re
import ssl
import sys
import urllib.request
from pathlib import Path

# SSL context — prefer certifi's CA bundle (the repo's eia_gas.py pattern);
# fall back to the system default if certifi is absent.
try:
    import certifi

    _SSL_CONTEXT: ssl.SSLContext = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CONTEXT = ssl.create_default_context()

# ---------------------------------------------------------------------------
# Repo wiring (mirror scripts/refresh_cpi.py).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Resolver normalizer — stdlib-only module, no Django needed.
from apps.api.profiles.city_resolver import (  # noqa: E402
    _COUNTY_SUFFIXES,
    _PLACE_SUFFIXES,
    _normalize,
)

_ARTIFACTS = _REPO_ROOT / "pipeline" / "artifacts"
_CITY_PUMA_MAP = _ARTIFACTS / "city_puma_map.json"
_OUT_PATH = _ARTIFACTS / "local_tax_rates.json"

_TAX_YEAR = 2026
_SOURCE_YEAR = 2026
_AS_OF = "2026-06"  # curated-table as-of tag
_HTTP_TIMEOUT = 90
_UA = "finance-app/build_local_tax_rates (offline artifact builder)"

log = logging.getLogger("build_local_tax_rates")

# ---------------------------------------------------------------------------
# NYC + Yonkers — exact embedded constants (NYC verified against
# models/tax/calculator.py::_NYC_BRACKETS; the inf edge serializes to null).
# ---------------------------------------------------------------------------
_NYC_FIPS = "3651000"
_NYC_BRACKETS = {
    "single": [[12000.0, 0.03078], [25000.0, 0.03762], [50000.0, 0.03819], [None, 0.03876]],
    "married_filing_jointly": [
        [21600.0, 0.03078],
        [45000.0, 0.03762],
        [90000.0, 0.03819],
        [None, 0.03876],
    ],
    "head_of_household": [
        [14400.0, 0.03078],
        [30000.0, 0.03762],
        [60000.0, 0.03819],
        [None, 0.03876],
    ],
}
_YONKERS_FIPS = "3684000"
_YONKERS_RATE = 0.0092  # 0.1675 resident surcharge x 0.055 flat-state NY approx

# ---------------------------------------------------------------------------
# Curated inline rate tables (place/county NAME -> rate), normalized at match
# time. Rates as-of 2026-06 from the cited sources; ~0.1pp accuracy acceptable.
# ---------------------------------------------------------------------------

# Maryland local income tax — all 24 county-equivalents. CY2026 rates per the
# Comptroller of Maryland local-tax schedule (Baltimore City is a
# county-equivalent, county FIPS 24510). Source: marylandtaxes.gov local
# income tax rate table.
_MD_COUNTY_RATES: dict[str, float] = {
    "Allegany": 0.0303,
    "Anne Arundel": 0.0294,
    "Baltimore County": 0.0320,
    "Calvert": 0.0300,
    "Caroline": 0.0320,
    "Carroll": 0.0303,
    "Cecil": 0.0300,
    "Charles": 0.0303,
    "Dorchester": 0.0320,
    "Frederick": 0.0296,
    "Garrett": 0.0265,
    "Harford": 0.0306,
    "Howard": 0.0320,
    "Kent": 0.0320,
    "Montgomery": 0.0320,
    "Prince George's": 0.0320,
    "Queen Anne's": 0.0320,
    "St. Mary's": 0.0310,
    "Somerset": 0.0320,
    "Talbot": 0.0240,
    "Washington": 0.0295,
    "Wicomico": 0.0320,
    "Worcester": 0.0225,
    "Baltimore City": 0.0320,  # county-equivalent, FIPS 24510
}
# Baltimore City must key by 24510 (independent city), not the surrounding
# Baltimore County (24005). The name-match would otherwise collide.
_MD_FORCE_FIPS: dict[str, str] = {"Baltimore City": "24510"}

# Indiana Local Income Tax (LIT) — all 92 counties. Rates per IN DOR
# Departmental Notice #1 (effective 2025/2026). Source: in.gov/dor.
_IN_COUNTY_RATES: dict[str, float] = {
    "Adams": 0.01624,
    "Allen": 0.0159,
    "Bartholomew": 0.0175,
    "Benton": 0.0179,
    "Blackford": 0.015,
    "Boone": 0.017,
    "Brown": 0.025234,
    "Carroll": 0.022733,
    "Cass": 0.0295,
    "Clark": 0.02,
    "Clay": 0.0235,
    "Clinton": 0.0245,
    "Crawford": 0.01,
    "Daviess": 0.015,
    "Dearborn": 0.012,
    "Decatur": 0.025,
    "DeKalb": 0.0213,
    "Delaware": 0.015,
    "Dubois": 0.012,
    "Elkhart": 0.02,
    "Fayette": 0.0257,
    "Floyd": 0.0135,
    "Fountain": 0.021,
    "Franklin": 0.015,
    "Fulton": 0.0288,
    "Gibson": 0.009,
    "Grant": 0.0255,
    "Greene": 0.0195,
    "Hamilton": 0.011,
    "Hancock": 0.0194,
    "Harrison": 0.01,
    "Hendricks": 0.017,
    "Henry": 0.018,
    "Howard": 0.0195,
    "Huntington": 0.0195,
    "Jackson": 0.021,
    "Jasper": 0.02864,
    "Jay": 0.0245,
    "Jefferson": 0.009,
    "Jennings": 0.025,
    "Johnson": 0.014,
    "Knox": 0.017,
    "Kosciusko": 0.01,
    "LaGrange": 0.0165,
    "Lake": 0.015,
    "LaPorte": 0.0145,
    "Lawrence": 0.0175,
    "Madison": 0.0225,
    "Marion": 0.0202,
    "Marshall": 0.0125,
    "Martin": 0.025,
    "Miami": 0.0254,
    "Monroe": 0.02035,
    "Montgomery": 0.0265,
    "Morgan": 0.0272,
    "Newton": 0.01,
    "Noble": 0.0175,
    "Ohio": 0.015,
    "Orange": 0.0175,
    "Owen": 0.016,
    "Parke": 0.0265,
    "Perry": 0.0181,
    "Pike": 0.0075,
    "Porter": 0.005,
    "Posey": 0.0125,
    "Pulaski": 0.0338,
    "Putnam": 0.021,
    "Randolph": 0.03,
    "Ripley": 0.0138,
    "Rush": 0.021,
    "St. Joseph": 0.0175,
    "Scott": 0.0216,
    "Shelby": 0.016,
    "Spencer": 0.008,
    "Starke": 0.0171,
    "Steuben": 0.0179,
    "Sullivan": 0.017,
    "Switzerland": 0.0125,
    "Tippecanoe": 0.0128,
    "Tipton": 0.026,
    "Union": 0.02,
    "Vanderburgh": 0.012,
    "Vermillion": 0.015,
    "Vigo": 0.02,
    "Wabash": 0.029,
    "Warren": 0.0212,
    "Warrick": 0.01,
    "Washington": 0.02,
    "Wayne": 0.015,
    "Wells": 0.021,
    "White": 0.0232,
    "Whitley": 0.0148293,
}

# Michigan city income tax — ~24 cities. Source: Michigan Treasury city
# income-tax resident-rate schedule. Most non-listed cities are 1.0%.
_MI_CITY_RATES: dict[str, float] = {
    "Detroit": 0.024,
    "Highland Park": 0.02,
    "Grand Rapids": 0.015,
    "Saginaw": 0.015,
    "Albion": 0.01,
    "Battle Creek": 0.01,
    "Benton Harbor": 0.01,
    "Big Rapids": 0.01,
    "East Lansing": 0.01,
    "Flint": 0.01,
    "Grayling": 0.01,
    "Hamtramck": 0.01,
    "Hudson": 0.01,
    "Ionia": 0.01,
    "Jackson": 0.01,
    "Lansing": 0.01,
    "Lapeer": 0.01,
    "Muskegon": 0.01,
    "Muskegon Heights": 0.01,
    "Pontiac": 0.01,
    "Port Huron": 0.01,
    "Portland": 0.01,
    "Springfield": 0.01,
    "Walker": 0.01,
}

# Kentucky local occupational/license taxes (resident). The consolidated /
# urban-county governments are in the place map but their compound Census
# names ("Louisville/Jefferson County metro government", "Lexington-Fayette
# urban county") do not survive the place-suffix normalizer cleanly, so they
# are force-mapped to their stable Census place FIPS (the Baltimore City
# precedent). Source: municipal occupational-tax schedules.
_KY_CITY_RATES: dict[str, float] = {
    "Louisville/Jefferson County metro government": 0.022,
    "Lexington-Fayette urban county": 0.0225,
    "Covington": 0.0245,
    "Bowling Green": 0.02,
}
_KY_FORCE_FIPS: dict[str, str] = {
    "Louisville/Jefferson County metro government": "2148006",
    "Lexington-Fayette urban county": "2146027",
}

# Missouri earnings tax (resident). Source: KC / STL earnings-tax ordinances.
# Kansas City is force-mapped (its normalized key collides under " city"
# suffix stripping); St. Louis matches by name.
_MO_CITY_RATES: dict[str, float] = {
    "Kansas City": 0.01,
    "St. Louis": 0.01,
}
_MO_FORCE_FIPS: dict[str, str] = {"Kansas City": "2938000"}

# Alabama occupational tax (resident). Source: Birmingham occupational-tax
# ordinance.
_AL_CITY_RATES: dict[str, float] = {"Birmingham": 0.01}

# Delaware city wage tax. Source: Wilmington earned-income-tax ordinance.
_DE_CITY_RATES: dict[str, float] = {"Wilmington": 0.0125}

# Per-state curated place tables → (state_abbrev_lower, name->rate, force_fips).
_CURATED_PLACE_TABLES: list[tuple[str, dict[str, float], dict[str, str]]] = [
    ("mi", _MI_CITY_RATES, {}),
    ("ky", _KY_CITY_RATES, _KY_FORCE_FIPS),
    ("mo", _MO_CITY_RATES, _MO_FORCE_FIPS),
    ("al", _AL_CITY_RATES, {}),
    ("de", _DE_CITY_RATES, {}),
]

# Curated county tables → (state_abbrev_lower, name->rate).
_CURATED_COUNTY_TABLES: list[tuple[str, dict[str, float]]] = [
    ("md", _MD_COUNTY_RATES),
    ("in", _IN_COUNTY_RATES),
]

# ---------------------------------------------------------------------------
# OHIO — live fetch with curated top-N fallback.
# ---------------------------------------------------------------------------
# The Finder publishes a downloadable municipal rate table. We try a couple of
# known endpoints; on failure we fall back to a curated top-N table (the
# largest levying municipalities by population), and flag the fallback.
# The Finder publishes a headerless CSV at this stable path (verified
# 2026-06): columns = start_date, end_date(99991231 = current), muni_code,
# MUNI_NAME, rate(decimal e.g. .02500). JEDD/JEDZ joint districts are not in
# this file but we still skip any by-name defensively.
_OH_FINDER_URLS = [
    "https://thefinder.tax.ohio.gov/streamlinesalestaxweb/download/BoundaryData/Muni/OHMuniRateTable.csv",
]
# Curated OH fallback — largest levying municipalities, 2026 resident rates.
_OH_FALLBACK_RATES: dict[str, float] = {
    "Columbus": 0.025,
    "Cleveland": 0.025,
    "Cincinnati": 0.018,
    "Toledo": 0.025,
    "Akron": 0.025,
    "Dayton": 0.025,
    "Parma": 0.025,
    "Canton": 0.025,
    "Youngstown": 0.0275,
    "Lorain": 0.025,
    "Hamilton": 0.02,
    "Springfield": 0.02,
    "Kettering": 0.0225,
    "Elyria": 0.0225,
    "Lakewood": 0.015,
    "Cuyahoga Falls": 0.02,
    "Euclid": 0.0285,
    "Middletown": 0.02,
    "Mansfield": 0.02,
    "Newark": 0.0175,
    "Mentor": 0.02,
    "Cleveland Heights": 0.025,
    "Beavercreek": 0.01,
    "Strongsville": 0.02,
    "Fairfield": 0.015,
    "Dublin": 0.02,
    "Warren": 0.025,
    "Findlay": 0.01,
    "Lancaster": 0.0175,
    "Lima": 0.015,
    "Huber Heights": 0.0225,
    "Westerville": 0.02,
    "Marion": 0.02,
    "Grove City": 0.02,
    "Reynoldsburg": 0.025,
    "Delaware": 0.0185,
    "Brunswick": 0.02,
    "Upper Arlington": 0.025,
    "Stow": 0.02,
    "North Olmsted": 0.02,
    "Gahanna": 0.025,
    "Westlake": 0.015,
    "North Royalton": 0.02,
    "Massillon": 0.02,
    "Fairborn": 0.02,
    "Bowling Green": 0.02,
    "Garfield Heights": 0.02,
    "Shaker Heights": 0.0275,
    "Sandusky": 0.0125,
    "Barberton": 0.025,
    "Wooster": 0.015,
}

# ---------------------------------------------------------------------------
# PENNSYLVANIA — live fetch with curated top-N fallback.
# ---------------------------------------------------------------------------
# The DCED Official EIT register. We try the munstats / dced endpoints; on
# failure we fall back to a curated table (Philadelphia MUST be present either
# way — asserted in the matcher).
# The DCED Official EIT register is served behind an ASP.NET viewstate form
# (per-county postback; no clean bulk CSV/XLSX over GET). _parse_pa returns {}
# for the HTML form page, which triggers the curated fallback below — the
# documented, expected outcome for this builder.
_PA_DCED_URLS = [
    "https://apps.dced.pa.gov/munstats-public/ReportInformation2.aspx?report=EitWithCollector_Dyn_Excel&type=O",
]
# Curated PA fallback — resident TOTAL EIT (municipal + school district),
# largest municipalities. Philadelphia's resident wage tax is the city rate
# (no separate school EIT). Source: DCED EIT register.
_PA_FALLBACK_RATES: dict[str, float] = {
    "Philadelphia": 0.0375,
    "Pittsburgh": 0.03,
    "Allentown": 0.0135,
    "Erie": 0.018,
    "Reading": 0.036,
    "Scranton": 0.034,
    "Bethlehem": 0.01,
    "Lancaster": 0.011,
    "Harrisburg": 0.02,
    "Altoona": 0.012,
    "York": 0.0125,
    "Wilkes-Barre": 0.03,
    "Chester": 0.0275,
    "Williamsport": 0.025,
    "Easton": 0.0195,
    "Lebanon": 0.015,
    "Hazleton": 0.0275,
    "New Castle": 0.01,
    "Johnstown": 0.018,
    "McKeesport": 0.015,
    "Norristown": 0.014,
    "State College": 0.022,
    "Pottstown": 0.0175,
}

# ---------------------------------------------------------------------------
# Census place->county crosswalk (codes2020). For MD + IN places only.
# ---------------------------------------------------------------------------
_CENSUS_CROSSWALK_URL = (
    "https://www2.census.gov/geo/docs/reference/codes2020/"
    "national_place_by_county2020.txt"
)


# ===========================================================================
# Fetch helpers
# ===========================================================================
def _http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(  # noqa: S310
        req, timeout=_HTTP_TIMEOUT, context=_SSL_CONTEXT
    ) as resp:
        return resp.read()


def _http_get(url: str) -> str:
    return _http_get_bytes(url).decode("utf-8", errors="replace")


def _try_fetch_ohio() -> tuple[dict[str, float], bool, str]:
    """Return (name->rate, used_live, provenance)."""
    for url in _OH_FINDER_URLS:
        try:
            log.info("OH: trying %s", url)
            raw = _http_get(url)
            rows = _parse_oh_csv(raw)
            if len(rows) >= 100:  # a real register, not an error page
                return rows, True, f"Ohio Dept of Taxation 'The Finder' MuniRateTable, {url} (retrieved {_AS_OF})"
            log.warning("OH: %s parsed only %d rows; trying next", url, len(rows))
        except Exception as exc:  # noqa: BLE001
            log.warning("OH: fetch failed for %s: %s", url, exc)
    log.warning("OH: all live endpoints failed — using curated top-%d fallback", len(_OH_FALLBACK_RATES))
    return (
        dict(_OH_FALLBACK_RATES),
        False,
        f"Ohio Dept of Taxation 'The Finder' (CURATED top-{len(_OH_FALLBACK_RATES)} fallback, live fetch unavailable; as-of {_AS_OF})",
    )


def _parse_oh_csv(raw: str) -> dict[str, float]:
    """Parse The Finder muni rate CSV (headerless): columns
    start_date, end_date, muni_code, MUNI_NAME, rate. Keeps only currently
    effective rows (end_date == 99991231) with a positive rate. JEDD/JEDZ rows
    (if any) are left in place here and filtered by name in the matcher (so the
    skip count lands in match_report). Returns NAME -> rate."""
    out: dict[str, float] = {}
    reader = csv.reader(io.StringIO(raw))
    for r in reader:
        if len(r) < 5:
            continue
        end_date = r[1].strip()
        if end_date != "99991231":  # superseded historical row
            continue
        name = r[3].strip()
        if not name:
            continue
        rate = _parse_rate(r[4])
        if rate is None or rate <= 0:
            continue
        out[name] = rate  # latest current row per name wins
    return out


def _try_fetch_pa() -> tuple[dict[str, float], bool, str]:
    for url in _PA_DCED_URLS:
        try:
            log.info("PA: trying %s", url)
            raw = _http_get_bytes(url)
            rows = _parse_pa(raw, url)
            if len(rows) >= 100:
                return rows, True, f"PA DCED Official EIT Tax Register (total resident EIT), {url} (retrieved {_AS_OF})"
            log.warning("PA: %s parsed only %d rows; trying next", url, len(rows))
        except Exception as exc:  # noqa: BLE001
            log.warning("PA: fetch failed for %s: %s", url, exc)
    log.warning("PA: all live endpoints failed — using curated top-%d fallback", len(_PA_FALLBACK_RATES))
    return (
        dict(_PA_FALLBACK_RATES),
        False,
        f"PA DCED Official EIT Tax Register (CURATED top-{len(_PA_FALLBACK_RATES)} fallback, live fetch unavailable; as-of {_AS_OF})",
    )


def _parse_pa(raw: bytes, url: str) -> dict[str, float]:
    """Best-effort parse of a DCED EIT register (CSV or XLSX inside a download).
    Returns name->total-resident-rate. The register's interactive endpoints do
    not expose a clean bulk CSV; this returns {} for HTML pages, triggering the
    curated fallback."""
    out: dict[str, float] = {}
    # XLSX (zip) — extract sharedStrings/sheet would require openpyxl; skip
    # gracefully if not a CSV.
    text: str
    if raw[:2] == b"PK":  # zip/xlsx
        return out
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return out
    if "<html" in text[:2000].lower():
        return out
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return out
    header = [h.strip().lower() for h in rows[0]]

    def find_col(*cands: str) -> int | None:
        for i, h in enumerate(header):
            if any(c in h for c in cands):
                return i
        return None

    name_i = find_col("municipality", "muni", "name")
    rate_i = find_col("total resident", "resident eit", "total", "rate")
    if name_i is None or rate_i is None:
        return out
    for r in rows[1:]:
        if len(r) <= max(name_i, rate_i):
            continue
        name = r[name_i].strip()
        rate = _parse_rate(r[rate_i])
        if name and rate and rate > 0:
            out[name] = rate
    return out


def _try_fetch_crosswalk() -> tuple[list[list[str]], bool]:
    """Fetch the codes2020 place-by-county crosswalk. Returns (rows, used_live).
    Header: STATE|STATEFP|COUNTYFP|COUNTYNAME|PLACEFP|PLACENS|PLACENAME|TYPE|..."""
    try:
        log.info("Census: fetching %s", _CENSUS_CROSSWALK_URL)
        raw = _http_get(_CENSUS_CROSSWALK_URL)
        rows = [line.split("|") for line in raw.splitlines() if line.strip()]
        if len(rows) > 1000:
            return rows, True
    except Exception as exc:  # noqa: BLE001
        log.warning("Census: crosswalk fetch failed: %s", exc)
    return [], False


def _parse_rate(s: str) -> float | None:
    s = s.strip().replace("%", "").replace(",", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    # Heuristic: values > 1 are percentages (e.g. "2.5" => 0.025); values in
    # (0,1] are already fractions.
    if v > 1.0:
        v = v / 100.0
    return v


# ===========================================================================
# Name -> FIPS matching
# ===========================================================================
def _build_name_indexes(cpm: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Return (place_names, county_names) as already-normalized 'st|name' ->
    FIPS dicts straight from city_puma_map.json."""
    return cpm["place_names"], cpm["county_names"]


def _match_places(
    state: str,
    name_rate: dict[str, float],
    place_names: dict[str, str],
    *,
    skip_jedd: bool = False,
    force_fips: dict[str, str] | None = None,
) -> tuple[dict[str, dict], int, int, list[str]]:
    """Match a state's place name->rate table to 7-digit place FIPS.

    Returns (fips->entry, matched, skipped_jedd, unmatched_sample)."""
    out: dict[str, dict] = {}
    matched = 0
    skipped = 0
    unmatched: list[str] = []
    force_fips = force_fips or {}
    for name, rate in name_rate.items():
        if skip_jedd and re.search(r"\bJED[DZ]\b", name, re.IGNORECASE):
            skipped += 1
            continue
        if name in force_fips:
            fips = force_fips[name]
        else:
            key = f"{state}|{_normalize(name, _PLACE_SUFFIXES)}"
            fips = place_names.get(key)
        if not fips:
            unmatched.append(name)
            continue
        out[fips] = {"name": name, "kind": "flat", "rate": round(rate, 6), "base": "wages"}
        matched += 1
    return out, matched, skipped, unmatched[:10]


def _match_counties(
    state: str,
    name_rate: dict[str, float],
    county_names: dict[str, str],
    *,
    force_fips: dict[str, str] | None = None,
) -> tuple[dict[str, dict], int, list[str]]:
    out: dict[str, dict] = {}
    matched = 0
    unmatched: list[str] = []
    force_fips = force_fips or {}
    for name, rate in name_rate.items():
        if name in force_fips:
            fips = force_fips[name]
        else:
            key = f"{state}|{_normalize(name, _COUNTY_SUFFIXES)}"
            fips = county_names.get(key)
        if not fips:
            unmatched.append(name)
            continue
        out[fips] = {"name": name, "kind": "flat", "rate": round(rate, 6), "base": "wages"}
        matched += 1
    return out, matched, unmatched[:10]


def _build_place_to_county(
    crosswalk: list[list[str]],
    target_states: frozenset[str],
) -> dict[str, str]:
    """For the county-tax states (MD, IN), map each 7-digit place FIPS to its
    primary (first-listed) 5-digit county FIPS — so a place selection can
    inherit the county rate. Emits an entry for every place in those states."""
    # Header: STATE|STATEFP|COUNTYFP|COUNTYNAME|PLACEFP|PLACENS|PLACENAME|...
    out: dict[str, str] = {}
    for row in crosswalk:
        if len(row) < 7:
            continue
        state_ab = row[0].strip().lower()
        if state_ab not in target_states:
            continue
        statefp = row[1].strip()
        countyfp = row[2].strip()
        placefp = row[4].strip()
        if not (statefp.isdigit() and placefp.isdigit() and countyfp.isdigit()):
            continue
        place_fips = f"{statefp}{placefp}"
        county_fips = f"{statefp}{countyfp}"
        # First listing wins (multi-county place -> primary).
        out.setdefault(place_fips, county_fips)
    return out


# ===========================================================================
# Validation
# ===========================================================================
_PLACE_RE = re.compile(r"^\d{7}$")
_COUNTY_RE = re.compile(r"^\d{5}$")


def _validate(artifact: dict) -> None:
    by_place = artifact["by_place_fips"]
    by_county = artifact["by_county_fips"]

    def check_rate(r: float, where: str) -> None:
        if not (0.0 < r <= 0.05):
            raise AssertionError(f"rate out of (0,0.05] at {where}: {r}")

    for fips, e in by_place.items():
        if not _PLACE_RE.match(fips):
            raise AssertionError(f"bad place FIPS: {fips!r}")
        if e["kind"] == "flat":
            check_rate(e["rate"], f"place {fips}")
        elif e["kind"] == "brackets":
            for fs, brs in e["brackets_by_filing_status"].items():
                prev = -1.0
                for i, (ub, rate) in enumerate(brs):
                    check_rate(rate, f"place {fips}/{fs} bracket {i}")
                    last = i == len(brs) - 1
                    if last:
                        if ub is not None:
                            raise AssertionError(f"last bracket ub must be null: {fips}/{fs}")
                    else:
                        if ub is None or ub <= prev:
                            raise AssertionError(
                                f"non-increasing bracket ub at {fips}/{fs} idx {i}: {ub}"
                            )
                        prev = ub
        else:
            raise AssertionError(f"bad kind {e['kind']!r} at place {fips}")

    for fips, e in by_county.items():
        if not _COUNTY_RE.match(fips):
            raise AssertionError(f"bad county FIPS: {fips!r}")
        check_rate(e["rate"], f"county {fips}")

    # No duplicate FIPS across the two maps (place=7, county=5 differ by length
    # so cross-collision is impossible, but guard anyway against malformed keys)
    overlap = set(by_place) & set(by_county)
    if overlap:
        raise AssertionError(f"FIPS appears in both place and county maps: {overlap}")

    # MD/IN county counts.
    md = sum(1 for f in by_county if f.startswith("24"))
    in_ = sum(1 for f in by_county if f.startswith("18"))
    if md != 24:
        raise AssertionError(f"MD county count must be 24, got {md}")
    if in_ != 92:
        raise AssertionError(f"IN county count must be 92, got {in_}")

    # NYC exact-match.
    nyc = by_place.get(_NYC_FIPS)
    if not nyc or nyc.get("kind") != "brackets":
        raise AssertionError("NYC entry missing or not brackets")
    if nyc["brackets_by_filing_status"] != _NYC_BRACKETS:
        raise AssertionError("NYC brackets do not match the embedded constant")
    if nyc.get("base") != "wages":
        raise AssertionError("NYC base must be 'wages'")

    # Philadelphia present.
    if "4260000" not in by_place:
        raise AssertionError("Philadelphia (4260000) missing from by_place_fips")

    # Comprehensive bar.
    if len(by_place) < 400:
        raise AssertionError(f"by_place_fips must have >=400 entries, got {len(by_place)}")


# ===========================================================================
# Build
# ===========================================================================
def build() -> dict:
    cpm = json.loads(_CITY_PUMA_MAP.read_text())
    place_names, county_names = _build_name_indexes(cpm)

    by_place: dict[str, dict] = {}
    by_county: dict[str, dict] = {}
    match_report: dict[str, dict] = {}
    sources: dict[str, str] = {}

    # --- NYC (brackets) + Yonkers (flat) -----------------------------------
    by_place[_NYC_FIPS] = {
        "name": "New York",
        "kind": "brackets",
        "base": "wages",
        "brackets_by_filing_status": _NYC_BRACKETS,
    }
    by_place[_YONKERS_FIPS] = {
        "name": "Yonkers",
        "kind": "flat",
        "rate": _YONKERS_RATE,
        "base": "wages",
    }
    sources["ny"] = (
        "NYC resident bracket schedule migrated verbatim from "
        "models/tax/calculator.py::_NYC_BRACKETS; Yonkers resident flat 0.0092 "
        "= 0.1675 resident surcharge x 0.055 flat-state NY approximation "
        f"(as-of {_AS_OF})"
    )

    # --- Ohio (live or curated) --------------------------------------------
    oh_raw, oh_live, oh_src = _try_fetch_ohio()
    # Count JEDD/JEDZ before matching for the report.
    oh_entries, oh_matched, oh_skipped, oh_unmatched = _match_places(
        "oh", oh_raw, place_names, skip_jedd=True
    )
    by_place.update(oh_entries)
    sources["oh"] = oh_src
    match_report["oh"] = {
        "matched": oh_matched,
        "total": len(oh_raw),
        "skipped_jedd": oh_skipped,
        "live_fetch": oh_live,
        "unmatched_sample": oh_unmatched,
    }

    # --- Pennsylvania (live or curated) ------------------------------------
    pa_raw, pa_live, pa_src = _try_fetch_pa()
    pa_entries, pa_matched, _pa_skip, pa_unmatched = _match_places(
        "pa", pa_raw, place_names
    )
    by_place.update(pa_entries)
    sources["pa"] = pa_src
    match_report["pa"] = {
        "matched": pa_matched,
        "total": len(pa_raw),
        "live_fetch": pa_live,
        "unmatched_sample": pa_unmatched,
    }
    # Philadelphia MUST match.
    if "4260000" not in by_place:
        raise AssertionError("PA: Philadelphia (4260000) failed to match — aborting")

    # --- Curated place tables (MI/KY/MO/AL/DE) -----------------------------
    for st, table, force in _CURATED_PLACE_TABLES:
        entries, matched, _sk, unmatched = _match_places(
            st, table, place_names, force_fips=force
        )
        by_place.update(entries)
        match_report[st] = {
            "matched": matched,
            "total": len(table),
            "live_fetch": False,
            "unmatched_sample": unmatched,
        }
    sources["mi"] = f"Michigan Treasury city income-tax resident-rate schedule (curated, as-of {_AS_OF})"
    sources["ky"] = f"Kentucky municipal occupational/license-tax schedules (curated, as-of {_AS_OF})"
    sources["mo"] = f"Kansas City & St. Louis earnings-tax ordinances (curated, as-of {_AS_OF})"
    sources["al"] = f"Birmingham occupational-tax ordinance (curated, as-of {_AS_OF})"
    sources["de"] = f"Wilmington earned-income-tax ordinance (curated, as-of {_AS_OF})"

    # --- Curated county tables (MD/IN) -------------------------------------
    for st, table in _CURATED_COUNTY_TABLES:
        force = _MD_FORCE_FIPS if st == "md" else None
        entries, matched, unmatched = _match_counties(
            st, table, county_names, force_fips=force
        )
        by_county.update(entries)
        match_report[st] = {
            "matched": matched,
            "total": len(table),
            "live_fetch": False,
            "unmatched_sample": unmatched,
        }
    sources["md"] = (
        "Comptroller of Maryland local income-tax rate schedule, all 24 "
        f"county-equivalents incl. Baltimore City (FIPS 24510) (curated, as-of {_AS_OF})"
    )
    sources["in"] = (
        "Indiana DOR Departmental Notice #1 Local Income Tax (LIT) rates, all "
        f"92 counties (curated, as-of {_AS_OF})"
    )

    # --- place_to_county crosswalk (MD + IN places only) -------------------
    crosswalk, xwalk_live = _try_fetch_crosswalk()
    place_to_county = _build_place_to_county(crosswalk, frozenset({"md", "in"}))
    sources["census_crosswalk"] = (
        f"US Census codes2020 national_place_by_county2020.txt, {_CENSUS_CROSSWALK_URL} "
        f"({'live' if xwalk_live else 'FETCH FAILED — place_to_county empty for MD/IN'}; "
        f"retrieved {_AS_OF})"
    )
    match_report["place_to_county"] = {
        "entries": len(place_to_county),
        "live_fetch": xwalk_live,
        "states": ["MD", "IN"],
    }

    artifact = {
        "_meta": {
            "description": (
                "Local/municipal income-tax rates keyed by Census place FIPS "
                "(7-digit) and county FIPS (5-digit). Supports flat rates and "
                "marginal brackets. Resident rates applied to gross wages."
            ),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "tax_year": _TAX_YEAR,
            "source_year": _SOURCE_YEAR,
            "sources": sources,
            "refresh_procedure": "python3.11 scripts/build_local_tax_rates.py",
            "exclusions": {
                "nj_newark": "employer-side payroll tax, not an employee income tax",
                # Stage 8 — LOCAL-TAX-COVERAGE-GAP closure. The remaining silent-zero
                # localities are STRUCTURALLY outside this FIPS-keyed % -of-wages /
                # brackets model (verified vs official DOR/municipal sources, 2026).
                # Excluding them is the rigorous call — forcing a flat head-fee or a
                # state-tax surtax into a wage-rate model would mis-tax — and is
                # consistent with the Newark precedent above.
                "ia_school_surtax": (
                    "Iowa school-district income surtax (+ county EMS surtax): a % of "
                    "STATE income tax (not wages) AND keyed by SCHOOL DISTRICT, which "
                    "does not map to place/county FIPS — districts cross city + county "
                    "lines (Des Moines spans 5 districts; needs address-level "
                    "resolution). ~283/322 districts levy 0-20% (median 5%); 6 "
                    "counties add a 1% EMS surtax. Wrong base + wrong key."
                ),
                "co_opt": (
                    "Colorado Occupational Privilege Tax = a FLAT $/month head tax, "
                    "not a % of wages: Denver $5.75, Glendale $5, Sheridan $3, "
                    "Greenwood Village $2 employee/mo (Aurora repealed 2025-01-01). "
                    "Place-keyed, but this model has no flat-$ rule kind."
                ),
                "wv_service_fee": (
                    "West Virginia municipal City Service Fee = a FLAT $/week fee, not "
                    "a % of wages: Charleston $3, Huntington $5, Wheeling $2, "
                    "Parkersburg $2.50, Morgantown $3, Weirton $5/wk. Place-keyed, but "
                    "this model has no flat-$ rule kind."
                ),
                "ks_intangibles": (
                    "Kansas has no local WAGE tax; the local intangibles tax is on "
                    "investment income (interest/dividends) only — out of scope for a "
                    "wage calculator."
                ),
            },
            "notes": (
                "resident rates; applied to gross wages (MVP approximation, "
                "consistent with flat-state layer); PA register resident EIT "
                "rates already combine municipality+school district; no stacking"
            ),
            "match_report": match_report,
        },
        "by_place_fips": by_place,
        "by_county_fips": by_county,
        "place_to_county": place_to_county,
    }
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(_OUT_PATH), help="output artifact path"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("Building local tax rates artifact...")
    artifact = build()

    log.info("Validating...")
    _validate(artifact)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    bp = len(artifact["by_place_fips"])
    bc = len(artifact["by_county_fips"])
    p2c = len(artifact["place_to_county"])
    log.info("Wrote %s", out_path)
    log.info("  by_place_fips:  %d", bp)
    log.info("  by_county_fips: %d", bc)
    log.info("  place_to_county:%d", p2c)
    log.info("Match report:")
    for k, v in artifact["_meta"]["match_report"].items():
        log.info("  %s: %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
