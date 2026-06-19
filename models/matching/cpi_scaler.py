"""BLS CPI scalars used to roll the CEX 2024 cohort p50 forward to
current prices.

The matched synthetic population is anchored to the CEX 2024 vintage.
Real-world prices have moved since, so before feeding p50 into the
allocator we multiply by a category-specific CPI scalar:

    scalar_c = CPI_c_current / CPI_c_base_year

Base year is ``2024`` (the year the fusion donor data was pooled).
``current`` is the latest month published by BLS.

BLS series mapping follows the CEX -> CPI item-level crosswalk; most
categories have a direct series, and the remainder fall back to
``CUUR0000SA0`` (All items).

Cache format (``pipeline/artifacts/cpi_scalars.json``):

    {
      "generated_at": "2026-04-18T12:34:56Z",
      "base_year": 2024,
      "base_values": {"CUUR0000SA0": 313.689, ...},
      "current_values": {"CUUR0000SA0": 325.120, ...},
      "scalars": {"eathome": 1.031, "eatout": 1.048, ..., "all_items": 1.036}
    }

Cache refresh policy: if the file is missing or older than
``_CACHE_TTL_DAYS`` days, ``load_cpi_scalars`` attempts a fresh fetch.
On any fetch failure (network, rate limit, BLS outage) the loader
returns all-1.0 scalars and rewrites the cache so subsequent calls
don't keep retrying in a tight loop.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shared.constants.categories import CATEGORY_CODES

# macOS system Python ships without a populated trust store, so certifi's
# bundle is used when available. Falling back to the default context lets
# Linux/CI environments with a system trust store continue to work.
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

_LOG = logging.getLogger(__name__)

_BLS_API_URL: str = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_BASE_YEAR_DEFAULT: int = 2024
_CACHE_TTL_DAYS: int = 30
_FETCH_TIMEOUT_SECONDS: float = 15.0
_FALLBACK_SERIES: str = "CUUR0000SA0"  # All items

# BLS public API (v2, unregistered) caps each request at 25 series.
# Registered keys get 50. We batch to stay under the unregistered cap.
_BLS_SERIES_BATCH_SIZE: int = 25

_DEFAULT_CACHE_PATH: str = "pipeline/artifacts/cpi_scalars.json"

# Pseudo-category key for the all-items (headline CPI) scalar. Stored
# in the scalars dict so callers can fall back to general inflation
# when a category's specific series is unavailable or missing from the
# map. Value = CUUR0000SA0.
DEFAULT_KEY: str = "_default"

# Last-resort fallback when even the all-items fetch fails. Chosen to
# approximate typical US inflation over a year so the allocator
# doesn't silently pretend prices are frozen at base-year levels.
_EMERGENCY_FALLBACK_SCALAR: float = 1.05

# Crosswalk from CEX category code to BLS series ID. Categories not in
# this map fall back to CUUR0000SA0 (All items) in the scalar dict.
BLS_SERIES_MAP: dict[str, str] = {
    "eathome": "CUUR0000SAF11",   # Food at home
    "eatout":  "CUUR0000SEFV",    # Food away from home
    "gas":     "CUUR0000SETB01",  # Gasoline (all types)
    "health":  "CUUR0000SAM",     # Medical care
    "elec":    "CUUR0000SEHF01",  # Electricity
    "ngas":    "CUUR0000SEHF02",  # Utility (piped) gas service
    "vehins":  "CUUR0000SETE",    # Motor vehicle insurance
    "cloftw":  "CUUR0000SAA",     # Apparel
    "rntval":  "CUUR0000SEHA",    # Rent of primary residence (actual for renters, imputed for owners)
    "rntexp":  "CUUR0000SAH3",    # Renter-side materials/appliances/insurance — maps to household furnishings and operations
    "ownval":  "CUUR0000SEHC",    # Owners' equivalent rent
    "hotel":   "CUUR0000SEHB",    # Lodging away from home
    "recrp":   "CUUR0000SAR",     # Recreation
    "oeprd":   "CUUR0000SAR",
    "oesrv":   "CUUR0000SAR",
    "educ":    "CUUR0000SAE",     # Education and communication
    "stdint":  "CUUR0000SAE",
    "pubtrn":  "CUUR0000SETD",    # Public transportation
    "airshp":  "CUUR0000SETG01",  # Airline fares
    "taxis":   "CUUR0000SETD",
    "intphn":  "CUUR0000SEED",    # Information technology services
    "watrsh":  "CUUR0000SEHG",    # Water/sewer/trash
    "ofuel":   "CUUR0000SEHE",    # Fuel oil and other fuels
    "furhwr":  "CUUR0000SAH3",    # Household furnishings and operations
    "happl":   "CUUR0000SAH3",
    "hhpcp":   "CUUR0000SAH3",
    "hhpcs":   "CUUR0000SAS",     # Other services
    "hinsp":   "CUUR0000SAH2",    # Housing (maintenance/insurance proxy)
    "hmtimp":  "CUUR0000SAH2",
    "mrtgip":  "CUUR0000SAH",     # Shelter
    "mrtgpp":  "CUUR0000SAH",
    "mrtgps":  "CUUR0000SAH",
    "ptaxp":   "CUUR0000SAH",
    "ohouse":  "CUUR0000SAH",
    "vehnew":  "CUUR0000SETA01",  # New vehicles
    "vehusd":  "CUUR0000SETA02",  # Used cars and trucks
    "vehprd":  "CUUR0000SETC",    # Motor vehicle parts and equipment
    "vehmlr":  "CUUR0000SETD",
    "vehreg":  "CUUR0000SEGE",    # Motor vehicle fees
    "vehint":  "CUUR0000SA0",
    "vehprn":  "CUUR0000SA0",
    "vehval":  "CUUR0000SETA02",
    "jwlbg":   "CUUR0000SAA",
    "eltrnp":  "CUUR0000SERA01",  # Video and audio products
    "chrty":   "CUUR0000SA0",
    "finpay":  "CUUR0000SA0",
    "ocash":   "CUUR0000SA0",
    "othint":  "CUUR0000SA0",
    "othdbt":  "CUUR0000SA0",
    "othfin":  "CUUR0000SA0",
    "stddbt":  "CUUR0000SA0",
    "check":   "CUUR0000SA0",
    "lifval":  "CUUR0000SA0",
    "retire":  "CUUR0000SA0",
    "stock":   "CUUR0000SA0",
}


# Regional BLS CPI-U series, one per (category, census region). The
# national BLS CPI rolls the cohort forward in time but says nothing
# about regional differences in inflation — West gas can inflate faster
# than Midwest gas in any given year. For categories with large
# region-to-region inflation variance (the five here), the temporal
# factor alone should be region-specific.
#
# Scalar semantics (important): these regional scalars are PURE
# TEMPORAL. An empirical check on 2020 vs 2024 regional/national ratios
# shows the ratios are stable within 1-2% per (region, item) — i.e. both
# series track together under a common cumulative inflator. But the
# absolute regional/national ratios (South eathome = 0.43, West gas =
# 1.66) are physically impossible as spatial premiums — the BLS regional
# subitem indexes were introduced at different base dates than the
# national series and reset to 100 at introduction, so their raw levels
# differ from national by an arbitrary base-period offset, not by a
# real spatial gap. The only information we can extract from a regional
# series is "how much did prices in this region change since year T",
# which is what we compute here: ``reg_current / reg_base``.
#
# Consequently, algorithm.py still multiplies by BEA RPP for the spatial
# component of these five cats — regional CPI does NOT absorb BEA RPP's
# job. This differs from the original prompt's description of the
# approach, but it's the version that produces correct scalars.
#
# Area codes (BLS CPI-U, current "S49X" region series):
#   S49A — Northeast
#   S49B — Midwest
#   S49C — South
#   S49D — West
#
# (The older ``A101/A207/A316/A421`` codes were retired in 2018; they
# still accept API queries but return no data. S49X replaced them.)
#
# Note on elec / ngas: BLS stopped publishing the region-level energy
# subitems (SEHF01 electricity, SEHF02 utility gas) after December
# 2024. The fetch logic still returns Dec 2024 as the "current" obs,
# which effectively freezes the regional temporal factor at Dec 2024
# — a minor lag (≤1 year) on a smoothly-moving series.
REGIONAL_BLS_SERIES: dict[str, dict[str, str]] = {
    "gas": {
        "west":      "CUURS49DSETB01",
        "northeast": "CUURS49ASETB01",
        "midwest":   "CUURS49BSETB01",
        "south":     "CUURS49CSETB01",
    },
    "eathome": {
        "west":      "CUURS49DSAF11",
        "northeast": "CUURS49ASAF11",
        "midwest":   "CUURS49BSAF11",
        "south":     "CUURS49CSAF11",
    },
    "eatout": {
        "west":      "CUURS49DSEFV",
        "northeast": "CUURS49ASEFV",
        "midwest":   "CUURS49BSEFV",
        "south":     "CUURS49CSEFV",
    },
    "elec": {
        "west":      "CUURS49DSEHF01",
        "northeast": "CUURS49ASEHF01",
        "midwest":   "CUURS49BSEHF01",
        "south":     "CUURS49CSEHF01",
    },
    "ngas": {
        "west":      "CUURS49DSEHF02",
        "northeast": "CUURS49ASEHF02",
        "midwest":   "CUURS49BSEHF02",
        "south":     "CUURS49CSEHF02",
    },
}

# Consumers check membership here to decide between "BLS regional scalar
# only" (for these five) vs "national CPI * BEA RPP" (everything else).
REGIONAL_BLS_CATEGORIES: frozenset[str] = frozenset(REGIONAL_BLS_SERIES)

# Valid region keys for ``resolve_regional_scalar``. Lowercase convention.
_VALID_REGIONS: frozenset[str] = frozenset({"west", "northeast", "midwest", "south"})

# Marker key used by the cache-staleness check: if the cache predates the
# regional-series work it won't have this key, and we force a refetch even
# if the file is still within the TTL.
_REGIONAL_MARKER_KEY: str = "gas_west"

# A regional series whose latest published period is more than this many
# months behind its national counterpart is treated as discontinued; the
# national cat scalar is used in its place so current-period inflation
# still gets applied. 6 months is enough to absorb normal publishing
# lag (BLS releases with ~1 month delay) without masking a genuine
# discontinuation.
_STALENESS_WINDOW_MONTHS: int = 6


def _months_between(earlier: tuple[int, str], later: tuple[int, str]) -> int:
    """Absolute month-distance between two (year, ``Mmm``) BLS periods.

    BLS period strings for monthly CPI are ``M01`` .. ``M12``. Annual
    aggregates use ``M13`` and shouldn't appear in latest-period lookups
    for monthly series; if they do, we parse them as month 12 so the
    distance is still a sensible nonnegative integer.
    """
    def _month(p: str) -> int:
        try:
            m = int(p[1:])
        except (ValueError, IndexError):
            return 12
        return min(max(m, 1), 12)

    e_months = earlier[0] * 12 + _month(earlier[1])
    l_months = later[0] * 12 + _month(later[1])
    return abs(l_months - e_months)


def _unique_series() -> list[str]:
    """Unique series IDs referenced by the crosswalk plus the fallback."""
    series = set(BLS_SERIES_MAP.values())
    series.add(_FALLBACK_SERIES)
    for region_map in REGIONAL_BLS_SERIES.values():
        series.update(region_map.values())
    return sorted(series)


def _series_for(category: str) -> str:
    """Resolve a CEX category to its BLS series (falls back to All items)."""
    return BLS_SERIES_MAP.get(category, _FALLBACK_SERIES)


def _post_bls(payload: dict) -> dict:
    """POST to the BLS public API. Returns parsed JSON dict."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _BLS_API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(
        req, timeout=_FETCH_TIMEOUT_SECONDS, context=_SSL_CONTEXT
    ) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_base_and_current(
    bls_response: dict,
    base_year: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[int, str]]]:
    """Return (base_values, current_values, latest_periods) per series.

    base_values: arithmetic mean of the base-year monthly observations.
    current_values: latest available monthly observation.
    latest_periods: (year, period) of the latest observation per series,
        used downstream to detect series that have stopped publishing
        and need to fall back to the national scalar.
    """
    status = bls_response.get("status")
    if status != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS status={status}: {bls_response.get('message')}")

    base_values: dict[str, float] = {}
    current_values: dict[str, float] = {}
    latest_periods: dict[str, tuple[int, str]] = {}
    for series in bls_response.get("Results", {}).get("series", []):
        sid = series.get("seriesID")
        if not sid:
            continue
        base_obs: list[float] = []
        latest_period: tuple[int, str] | None = None
        latest_value: float | None = None
        for obs in series.get("data", []):
            try:
                y = int(obs["year"])
                v = float(obs["value"])
            except (KeyError, TypeError, ValueError):
                continue
            period = obs.get("period", "")
            if y == base_year and period.startswith("M"):
                base_obs.append(v)
            key = (y, period)
            if latest_period is None or key > latest_period:
                latest_period = key
                latest_value = v
        if base_obs:
            base_values[sid] = sum(base_obs) / len(base_obs)
        if latest_value is not None:
            current_values[sid] = latest_value
        if latest_period is not None:
            latest_periods[sid] = latest_period
    return base_values, current_values, latest_periods


def fetch_cpi_scalars(
    base_year: int = _BASE_YEAR_DEFAULT,
    cache_path: str = _DEFAULT_CACHE_PATH,
) -> dict[str, float]:
    """Fetch current CPI scalars per CEX category from BLS and cache.

    Scalar semantics: ``scalar > 1.0`` means category prices are higher
    than the base-year average. ``scalar == 1.0`` is the safe fallback
    when a series is missing or BLS returns no data.

    Args:
        base_year: Vintage of the CEX cohort data. Must match the year
            the synthetic population was fit against.
        cache_path: Where to write the cache JSON.

    Returns:
        Mapping ``{category_code: scalar}`` with one entry per of the 55
        CEX categories. Never raises on BLS failure — falls back to 1.0
        for every category and writes that to the cache.
    """
    series_ids = _unique_series()
    current_year = datetime.now(tz=timezone.utc).year

    base_values: dict[str, float] = {}
    current_values: dict[str, float] = {}
    latest_periods: dict[str, tuple[int, str]] = {}
    for start in range(0, len(series_ids), _BLS_SERIES_BATCH_SIZE):
        batch = series_ids[start:start + _BLS_SERIES_BATCH_SIZE]
        payload = {
            "seriesid": batch,
            "startyear": str(base_year),
            "endyear": str(current_year),
        }
        try:
            response = _post_bls(payload)
            bv, cv, lp = _extract_base_and_current(response, base_year)
            base_values.update(bv)
            current_values.update(cv)
            latest_periods.update(lp)
        except (
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            _LOG.warning(
                "BLS fetch failed for batch %d-%d (%r); those series will fall "
                "back to the _default scalar",
                start, start + len(batch), exc,
            )

    scalars: dict[str, float] = {}
    # Populate the all-items scalar first — it's what every other cat
    # falls back to when its specific series lookup fails.
    sa0_base = base_values.get(_FALLBACK_SERIES)
    sa0_curr = current_values.get(_FALLBACK_SERIES)
    if sa0_base and sa0_curr and sa0_base > 0:
        default_scalar = float(sa0_curr / sa0_base)
    else:
        default_scalar = _EMERGENCY_FALLBACK_SCALAR
    scalars[DEFAULT_KEY] = default_scalar

    for cat in CATEGORY_CODES:
        sid = _series_for(cat)
        base = base_values.get(sid)
        current = current_values.get(sid)
        if base and current and base > 0:
            scalars[cat] = float(current / base)
        else:
            # Series lookup failed for this cat — fall back to the
            # all-items CPI scalar so the cat still gets a general
            # inflation adjustment rather than freezing at 1.0.
            scalars[cat] = default_scalar

    # Regional scalars are region-specific TEMPORAL inflation since the
    # cohort's base year:
    #
    #     scalar_{cat}_{region} = reg_current / reg_base
    #
    # Spatial adjustment (national -> local) stays the BEA RPP's job —
    # see the file-level comment on REGIONAL_BLS_SERIES for why we
    # can't extract spatial from raw regional CPI levels.
    # Missing regional series fall back to the national cat scalar so
    # the category still gets at least national temporal adjustment.
    #
    # Staleness guard: BLS stopped publishing the S49X subitems for
    # elec (SEHF01) and ngas (SEHF02) after 2024-M12. If we used
    # ``reg_latest_2024M12 / reg_base_2024_avg`` as the scalar, those
    # categories would miss all 2025-26 national inflation — a
    # regression relative to the prior (national-CPI-only) behavior.
    # When a regional series' latest period is more than
    # ``_STALENESS_WINDOW_MONTHS`` months behind its national
    # counterpart, fall back to the national cat scalar so the category
    # still gets current temporal adjustment (at the cost of its small
    # region-specific inflation signal, which has no current data).
    national_periods = {
        _series_for(cat): latest_periods.get(_series_for(cat))
        for cat in REGIONAL_BLS_SERIES
    }
    for cat, region_map in REGIONAL_BLS_SERIES.items():
        cat_fallback = scalars.get(cat, default_scalar)
        nat_period = national_periods.get(_series_for(cat))
        for region, regional_sid in region_map.items():
            key = f"{cat}_{region}"
            reg_base = base_values.get(regional_sid)
            reg_current = current_values.get(regional_sid)
            reg_period = latest_periods.get(regional_sid)
            if reg_current is None or not reg_base or reg_base <= 0:
                scalars[key] = cat_fallback
                continue
            if (
                nat_period is not None and reg_period is not None
                and _months_between(reg_period, nat_period) > _STALENESS_WINDOW_MONTHS
            ):
                # Regional series has stopped publishing. The national
                # cat scalar is a better temporal estimate than the
                # frozen regional one.
                scalars[key] = cat_fallback
                continue
            scalars[key] = float(reg_current / reg_base)

    cache = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "base_year": base_year,
        "base_values": base_values,
        "current_values": current_values,
        "scalars": scalars,
    }
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    return scalars


def _cache_is_fresh(path: Path, ttl_days: int = _CACHE_TTL_DAYS) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(tz=timezone.utc) - mtime < timedelta(days=ttl_days)


def load_cpi_scalars(
    cache_path: str = _DEFAULT_CACHE_PATH,
    base_year: int = _BASE_YEAR_DEFAULT,
) -> dict[str, float]:
    """Load CPI scalars, refreshing from BLS when the cache is stale.

    Missing cache or cache older than ``_CACHE_TTL_DAYS`` triggers a
    fetch. Any failure in the fetch path yields a map where every
    category falls back to the all-items CPI scalar (stored under
    ``DEFAULT_KEY = "_default"``). Returns 55 CEX category keys plus
    the ``_default`` key — 56 entries total.
    """
    path = Path(cache_path)
    if _cache_is_fresh(path):
        try:
            with open(path) as f:
                cache = json.load(f)
            scalars = cache.get("scalars", {})
            # Guard against category-list drift and old caches that
            # predate the ``_default`` key or the regional CPI series —
            # any of those forces a refetch rather than silently
            # serving a stale 1.0 (or a missing regional scalar).
            if (
                all(cat in scalars for cat in CATEGORY_CODES)
                and DEFAULT_KEY in scalars
                and _REGIONAL_MARKER_KEY in scalars
            ):
                return {k: float(v) for k, v in scalars.items()}
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("cpi_scalars cache unreadable (%r); refetching", exc)

    return fetch_cpi_scalars(base_year=base_year, cache_path=cache_path)


def default_scalars() -> dict[str, float]:
    """All-1.0 scalar map. Useful as an injectable default in tests."""
    scalars: dict[str, float] = {cat: 1.0 for cat in CATEGORY_CODES}
    scalars[DEFAULT_KEY] = 1.0
    for cat, region_map in REGIONAL_BLS_SERIES.items():
        for region in region_map:
            scalars[f"{cat}_{region}"] = 1.0
    return scalars


def resolve_scalar(cpi_scalars: dict[str, float], cat: str) -> float:
    """Look up a scalar with the standard fallback chain.

    Priority:
      1. ``cpi_scalars[cat]`` — category-specific scalar
      2. ``cpi_scalars[DEFAULT_KEY]`` — all-items CPI scalar
      3. ``_EMERGENCY_FALLBACK_SCALAR`` — hardcoded 1.05

    Callers should use this rather than ``dict.get(cat, 1.0)`` so a
    missing or misconfigured category still receives general
    inflation adjustment instead of freezing at base-year prices.
    """
    if cat in cpi_scalars:
        return float(cpi_scalars[cat])
    if DEFAULT_KEY in cpi_scalars:
        return float(cpi_scalars[DEFAULT_KEY])
    return _EMERGENCY_FALLBACK_SCALAR


def regional_scalar_is_live(
    cpi_scalars: dict[str, float],
    cat: str,
    region: str | None,
) -> bool:
    """True iff the ``{cat}_{region}`` scalar came from a fresh fetch.

    When the regional BLS series is discontinued, stale, or entirely
    missing, ``fetch_cpi_scalars`` writes the national cat scalar to
    the ``{cat}_{region}`` key as a fallback — so equality between the
    regional and national values is the signal that "there is no live
    regional data for this (cat, region)". An exact float comparison
    is sufficient: the fallback path assigns ``cat_fallback`` to
    ``scalars[key]`` by direct reference, so both sides are the same
    float object and compare equal.

    Callers can gate downstream behavior on whether the regional
    signal is real — see ``calibration.gas_underreporting_factor``.
    """
    if region is None:
        return False
    region = region.lower()
    key = f"{cat}_{region}"
    if key not in cpi_scalars:
        return False
    national = cpi_scalars.get(cat, cpi_scalars.get(DEFAULT_KEY))
    if national is None:
        # No national baseline either — treat as not-live, the caller
        # has no way to tell signal from noise.
        return False
    return abs(float(cpi_scalars[key]) - float(national)) > 1e-9


def resolve_regional_scalar(
    cpi_scalars: dict[str, float],
    cat: str,
    region: str | None,
) -> float:
    """Look up the region-specific temporal CPI scalar for ``cat``.

    Returns the ``{cat}_{region}`` scalar from the cache when present —
    a pure temporal multiplier, NOT a combined temporal+spatial one.
    Callers still need to apply a spatial adjustment (typically BEA RPP)
    separately; see REGIONAL_BLS_SERIES's docstring for why.

    If ``region`` is None (e.g. PUMAs outside the four BLS census
    regions) or the cache has no entry for the regional key, falls
    back to the plain national ``resolve_scalar(cat)``.
    """
    if region is not None:
        region = region.lower()
        if region in _VALID_REGIONS:
            key = f"{cat}_{region}"
            if key in cpi_scalars:
                return float(cpi_scalars[key])
    return resolve_scalar(cpi_scalars, cat)


if __name__ == "__main__":
    # Allow `python -m models.matching.cpi_scaler` to prime the cache.
    logging.basicConfig(level=logging.INFO)
    os.chdir(Path(__file__).resolve().parents[2])
    scalars = fetch_cpi_scalars()
    for cat in ("eathome", "eatout", "gas", "health", "elec"):
        print(f"{cat}: {scalars[cat]:.4f}")
