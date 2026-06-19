"""City/county name -> PUMA resolver.

Consumes ``pipeline/artifacts/city_puma_map.json`` (produced by
``pipeline/export/export_city_puma_map.R``). The frontend ships a
static autocomplete built from ``city_list.json`` (see
``apps/web/lib/city-search.ts``); a selected option is POSTed here as
``state_code`` + ``city_name`` (+ optional ``county_name`` fallback).
``addresstype`` is retained on the wire as the Place-vs-County switch
but today always arrives as ``"city"`` from the static autocomplete
flow. This module normalizes names the same way the R artifact does,
looks up the FIPS, and returns the PUMA list with a resolution-path
tag.

Normalization mirrors export_city_puma_map.R::normalize_name and must
stay in lockstep. Changes here must be mirrored there.
"""

from __future__ import annotations

import json
import os
import re
from threading import Lock

# Addresstypes for which we attempt Place lookup first. "suburb" and
# "neighbourhood" cover named neighborhoods inside incorporated places.
# Anything outside this set goes directly to county. The static
# city_list.json autocomplete always supplies "city" today; the other
# values remain accepted for historical compatibility with older API
# callers.
_PLACE_ADDRESSTYPES: frozenset[str] = frozenset({
    "city", "town", "village", "hamlet", "suburb", "neighbourhood",
})

# Suffix lists mirror PLACE_SUFFIXES / COUNTY_SUFFIXES in the R export
# script. Sorted longest-first on load so the longest match wins.
_PLACE_SUFFIXES: tuple[str, ...] = (
    " consolidated government",
    " unified government",
    " metro government",
    " urban county",
    " municipality",
    " corporation",
    " plantation",
    " township",
    " borough",
    " village",
    " city",
    " town",
    " cdp",
)
_COUNTY_SUFFIXES: tuple[str, ...] = (
    " census area",
    " municipality",
    " municipio",
    " borough",
    " parish",
    " county",
    " city",
)

_CITY_PUMA_MAP: dict | None = None
_LOAD_LOCK = Lock()


def _load_map(artifacts_path: str) -> dict:
    """Load and cache the city_puma_map.json artifact."""
    global _CITY_PUMA_MAP
    if _CITY_PUMA_MAP is None:
        with _LOAD_LOCK:
            if _CITY_PUMA_MAP is None:
                path = os.path.join(artifacts_path, "city_puma_map.json")
                with open(path) as f:
                    _CITY_PUMA_MAP = json.load(f)
    return _CITY_PUMA_MAP


def _reset_cache_for_tests() -> None:
    """Clear the module-level cache. Tests only."""
    global _CITY_PUMA_MAP
    with _LOAD_LOCK:
        _CITY_PUMA_MAP = None


def _strip_suffix(name: str, suffixes: tuple[str, ...]) -> str:
    lower = name.lower()
    # Suffixes come in priority order (longest first on module load).
    for s in suffixes:
        if lower.endswith(s):
            return name[: len(name) - len(s)].strip()
    return name.strip()


_ST_DOT_RE = re.compile(r"\bst\.\s+")
_ST_RE = re.compile(r"\bst\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize(name: str, suffixes: tuple[str, ...]) -> str:
    """Normalize a place or county name the same way the R artifact does."""
    x = _strip_suffix(name.strip(), suffixes)
    x = x.lower()
    x = _ST_DOT_RE.sub("saint ", x)
    x = _ST_RE.sub("saint ", x)
    x = _PUNCT_RE.sub("", x)
    x = _WS_RE.sub(" ", x).strip()
    return x


def resolve_to_pumas(
    state_code: str | None,
    county_name: str | None,
    city_name: str | None,
    addresstype: str | None,
    artifacts_path: str,
) -> tuple[list[str], str, str, str]:
    """Resolve a city_list.json-style location to a list of PUMA codes.

    Args:
        state_code: Two-letter USPS postal code (``"CA"``). Case-insensitive.
        county_name: County name as it appears in the city_puma_map
            county index (e.g. ``"Los Angeles County"``). May be
            ``None`` if the caller has no county fallback.
        city_name: Place name from the static city_list autocomplete
            (e.g. ``"Santa Monica"``). May be ``None``.
        addresstype: Place-type classification
            (``"city" | "village" | "hamlet" | ...``). Determines whether
            to attempt the place lookup; values outside
            :data:`_PLACE_ADDRESSTYPES` skip straight to county. Static
            autocomplete callers always pass ``"city"``.
        artifacts_path: Path to ``pipeline/artifacts``.

    Returns:
        ``(pumas, resolved_via, place_fips, county_fips)`` where
        ``resolved_via`` is ``"place"`` or ``"county"``.  On the place
        path ``place_fips`` is the 7-digit FIPS string and
        ``county_fips`` is ``""``.  On the county path ``county_fips``
        is the 5-digit FIPS string and ``place_fips`` is ``""``.

    Raises:
        ValueError: Neither a place nor a county match was found.
    """
    if not state_code:
        raise ValueError("state_code is required")

    mapping = _load_map(artifacts_path)
    state_lower = state_code.strip().lower()

    # Place path — only attempted for place-like addresstypes.
    if (
        city_name
        and addresstype
        and addresstype.lower() in _PLACE_ADDRESSTYPES
    ):
        key = f"{state_lower}|{_normalize(city_name, _PLACE_SUFFIXES)}"
        place_fips = mapping["place_names"].get(key)
        if place_fips:
            pumas = mapping["place"].get(place_fips)
            if pumas:
                return pumas, "place", place_fips, ""
            # Name hit but no PUMA entry (e.g. territory stripped at build
            # time) — fall through to county.

    # County fallback.
    if county_name:
        key = f"{state_lower}|{_normalize(county_name, _COUNTY_SUFFIXES)}"
        county_fips = mapping["county_names"].get(key)
        if county_fips:
            pumas = mapping["county"].get(county_fips)
            if pumas:
                return pumas, "county", "", county_fips

    raise ValueError(
        "No PUMA mapping for "
        f"state={state_code!r} city={city_name!r} "
        f"county={county_name!r} addresstype={addresstype!r}"
    )
