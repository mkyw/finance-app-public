"""Integrity tests for the city autocomplete + resolver artifacts.

Guards the fixes applied by ``pipeline/export/postprocess_city_artifacts.py``
(the HI-CITY-SEARCH-BUG remediation): same-name place dedup/disambiguation,
the Honolulu display-name rename, and the generic resolver-key alias backfill.

Like ``test_city_resolver.py``, this imports ``profiles.city_resolver`` via a
django-free ``sys.path`` insert (the module has no Django dependency, but the
api test suite as a whole configures Django, so we set it up the same way).

Run from repo root:
    .venv/bin/python -m pytest tests/api/test_city_list_integrity.py -v
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from collections import Counter

os.environ.setdefault("ARTIFACTS_PATH", "pipeline/artifacts")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

import django  # noqa: E402

django.setup()

from profiles.city_resolver import (  # noqa: E402
    _PLACE_SUFFIXES,
    _normalize,
    _reset_cache_for_tests,
    resolve_to_pumas,
)

_ARTIFACTS = "pipeline/artifacts"
_CITY_LIST = os.path.join(_ARTIFACTS, "city_list.json")
_CITY_PUMA_MAP = os.path.join(_ARTIFACTS, "city_puma_map.json")
_WEB_PUBLIC = os.path.join("apps", "web", "public", "city_list.json")


def setup_module(module):
    """Clear the module-level map cache before this suite runs."""
    _reset_cache_for_tests()


def _load_city_list() -> list[dict]:
    with open(_CITY_LIST) as f:
        return json.load(f)


def _load_cpm() -> dict:
    with open(_CITY_PUMA_MAP) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# No duplicate (state, name) pairs                                            #
# --------------------------------------------------------------------------- #

def test_no_duplicate_state_name_pairs() -> None:
    cl = _load_city_list()
    counts = Counter((r["state"], r["name"]) for r in cl)
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"Duplicate (state, name) pairs remain: {dupes}"


# --------------------------------------------------------------------------- #
# Honolulu / Kailua list-row expectations                                    #
# --------------------------------------------------------------------------- #

def test_honolulu_present_and_no_urban_honolulu() -> None:
    cl = _load_city_list()
    honolulu = [r for r in cl if r["state"] == "HI" and r["name"] == "Honolulu"]
    assert len(honolulu) == 1, f"Expected exactly one HI Honolulu row, got {honolulu}"
    assert honolulu[0]["population"] > 300_000, (
        f"Honolulu population looks wrong: {honolulu[0]}"
    )
    urban = [r for r in cl if r["state"] == "HI" and r["name"] == "Urban Honolulu"]
    assert not urban, f"'Urban Honolulu' should be renamed away, found: {urban}"


def test_hi_kailua_singular_and_kona_present() -> None:
    cl = _load_city_list()
    kailua = [r for r in cl if r["state"] == "HI" and r["name"] == "Kailua"]
    assert len(kailua) == 1, f"Expected exactly one HI Kailua row, got {kailua}"
    kona = [r for r in cl if r["state"] == "HI" and r["name"] == "Kailua-Kona"]
    assert len(kona) == 1, f"Expected exactly one HI Kailua-Kona row, got {kona}"


# --------------------------------------------------------------------------- #
# Public mirror byte-equality                                                 #
# --------------------------------------------------------------------------- #

def test_public_mirror_byte_equal() -> None:
    with open(_CITY_LIST, "rb") as f:
        artifact = f.read()
    with open(_WEB_PUBLIC, "rb") as f:
        public = f.read()
    assert artifact == public, (
        "apps/web/public/city_list.json is not byte-equal to the artifact copy"
    )


# --------------------------------------------------------------------------- #
# Resolver round-trips                                                         #
# --------------------------------------------------------------------------- #

def test_honolulu_resolves_to_place() -> None:
    pumas, via, place_fips, county_fips = resolve_to_pumas(
        "HI", None, "Honolulu", "city", _ARTIFACTS
    )
    assert via == "place", f"Expected 'place', got {via!r}"
    assert place_fips == "1571550", f"Expected '1571550', got {place_fips!r}"
    assert pumas, "Expected non-empty PUMAs for Honolulu"


def test_kailua_resolves_to_oahu_fips() -> None:
    """Census-2020-verified: Kailua CDP (Honolulu County, Oahu) = FIPS 1523150,
    pop 40,514, PUMAs HI_00301/00302 (Honolulu County set). The R export's
    GEOID-ordered dedup mis-pointed "hi|kailua" at 1523000 (the SMALLER
    Big-Island place) — the postprocess re-points it (verified binding)."""
    pumas, via, place_fips, county_fips = resolve_to_pumas(
        "HI", None, "Kailua", "city", _ARTIFACTS
    )
    assert via == "place"
    assert place_fips == "1523150", f"Expected '1523150' (Oahu), got {place_fips!r}"
    assert pumas == ["HI_00301", "HI_00302"], (
        f"Expected Oahu PUMAs ['HI_00301', 'HI_00302'], got {pumas!r}"
    )


def test_kailua_kona_resolves_to_big_island_fips() -> None:
    """Kailua CDP (Hawaii County, Big Island — 'Kailua-Kona') = FIPS 1523000,
    pop 19,713, PUMA HI_00200 (the Hawaii County PUMA)."""
    pumas, via, place_fips, county_fips = resolve_to_pumas(
        "HI", None, "Kailua-Kona", "city", _ARTIFACTS
    )
    assert via == "place"
    assert place_fips == "1523000", f"Expected '1523000' (Big Island), got {place_fips!r}"
    assert pumas == ["HI_00200"], f"Expected ['HI_00200'], got {pumas!r}"


def test_renamed_namesakes_get_distinct_fips() -> None:
    """The CA/FL/WA disambiguated entries resolve to their own orphan FIPS,
    distinct from their plain-named namesake's FIPS."""
    cases = [
        ("CA", "El Sobrante", "El Sobrante (Riverside County)"),
        ("FL", "University", "University (Orange County)"),
        ("WA", "Fairwood", "Fairwood (Spokane County)"),
    ]
    for state, plain, renamed in cases:
        _, _, plain_fips, _ = resolve_to_pumas(
            state, None, plain, "city", _ARTIFACTS
        )
        _, _, renamed_fips, _ = resolve_to_pumas(
            state, None, renamed, "city", _ARTIFACTS
        )
        assert plain_fips, f"{state} {plain} did not resolve"
        assert renamed_fips, f"{state} {renamed} did not resolve"
        assert plain_fips != renamed_fips, (
            f"{state}: {plain!r} and {renamed!r} share FIPS {plain_fips}"
        )


# --------------------------------------------------------------------------- #
# Full-list resolvability                                                      #
# --------------------------------------------------------------------------- #

def test_every_list_row_has_a_place_names_key() -> None:
    """Every city_list row must be reachable through the resolver's own key
    form ``state|_normalize(name)``. The postprocess alias backfill guarantees
    this (incl. the 'Oklahoma City'-style " City" suffix cases the resolver
    over-strips). No known exceptions are expected post-fix.
    """
    cl = _load_city_list()
    place_names = _load_cpm()["place_names"]
    missing = [
        (r["state"], r["name"])
        for r in cl
        if f"{r['state'].lower()}|{_normalize(r['name'], _PLACE_SUFFIXES)}"
        not in place_names
    ]
    assert not missing, (
        f"{len(missing)} city_list rows have no place_names key; "
        f"sample: {missing[:20]}"
    )
