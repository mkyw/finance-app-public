"""Tests for city_resolver.py — confirms the 4-tuple return shape introduced in
stage 4 of the FIPS plumbing.

``resolve_to_pumas`` now returns ``(pumas, resolved_via, place_fips, county_fips)``.
These tests exercise the three cases called out in the plan:
  - Place path (Columbus, OH → place_fips populated, county_fips "")
  - County path (Los Angeles County, CA → county_fips populated, place_fips "")
  - NYC place path (New York, NY → place_fips == "3651000")

Run from repo root:
    ARTIFACTS_PATH=pipeline/artifacts .venv/bin/python -m pytest tests/api/test_city_resolver.py -v
"""

from __future__ import annotations

import os
import sys

# city_resolver has no Django import, but the test must be importable alongside
# the rest of the api test suite which does configure Django.  Ensure the api
# app is on the path the same way every other api test file does it.
os.environ.setdefault("ARTIFACTS_PATH", "pipeline/artifacts")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

import django  # noqa: E402
django.setup()

from profiles.city_resolver import _reset_cache_for_tests, resolve_to_pumas  # noqa: E402

_ARTIFACTS = "pipeline/artifacts"


def setup_module(module):
    """Clear the module-level map cache before this test suite runs."""
    _reset_cache_for_tests()


# --------------------------------------------------------------------------- #
# Place path — Columbus, OH                                                    #
# --------------------------------------------------------------------------- #

def test_columbus_place_path_returns_four_tuple() -> None:
    """resolve_to_pumas for Columbus, OH returns a 4-tuple, not a 2-tuple."""
    result = resolve_to_pumas("OH", None, "Columbus", "city", _ARTIFACTS)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple: {result!r}"


def test_columbus_place_path_resolved_via() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "OH", None, "Columbus", "city", _ARTIFACTS
    )
    assert resolved_via == "place", f"Expected 'place', got {resolved_via!r}"


def test_columbus_place_fips() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "OH", None, "Columbus", "city", _ARTIFACTS
    )
    assert place_fips == "3918000", (
        f"Expected Columbus place FIPS '3918000', got {place_fips!r}"
    )


def test_columbus_county_fips_empty_on_place_path() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "OH", None, "Columbus", "city", _ARTIFACTS
    )
    assert county_fips == "", (
        f"Expected empty county_fips on place path, got {county_fips!r}"
    )


def test_columbus_pumas_non_empty() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "OH", None, "Columbus", "city", _ARTIFACTS
    )
    assert len(pumas) > 0, "Expected non-empty pumas list for Columbus, OH"


# --------------------------------------------------------------------------- #
# County path — Los Angeles County, CA                                         #
# --------------------------------------------------------------------------- #

def test_la_county_path_returns_four_tuple() -> None:
    result = resolve_to_pumas("CA", "Los Angeles County", None, None, _ARTIFACTS)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple: {result!r}"


def test_la_county_path_resolved_via() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "CA", "Los Angeles County", None, None, _ARTIFACTS
    )
    assert resolved_via == "county", f"Expected 'county', got {resolved_via!r}"


def test_la_county_fips() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "CA", "Los Angeles County", None, None, _ARTIFACTS
    )
    assert county_fips == "06037", (
        f"Expected LA County FIPS '06037', got {county_fips!r}"
    )


def test_la_place_fips_empty_on_county_path() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "CA", "Los Angeles County", None, None, _ARTIFACTS
    )
    assert place_fips == "", (
        f"Expected empty place_fips on county path, got {place_fips!r}"
    )


def test_la_county_pumas_non_empty() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "CA", "Los Angeles County", None, None, _ARTIFACTS
    )
    assert len(pumas) > 0, "Expected non-empty pumas list for LA County"


# --------------------------------------------------------------------------- #
# NYC — place path                                                             #
# --------------------------------------------------------------------------- #

def test_nyc_place_path_returns_four_tuple() -> None:
    result = resolve_to_pumas("NY", None, "New York", "city", _ARTIFACTS)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple: {result!r}"


def test_nyc_place_fips() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "NY", None, "New York", "city", _ARTIFACTS
    )
    assert place_fips == "3651000", (
        f"Expected NYC place FIPS '3651000', got {place_fips!r}"
    )


def test_nyc_resolved_via_place() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "NY", None, "New York", "city", _ARTIFACTS
    )
    assert resolved_via == "place", f"Expected 'place', got {resolved_via!r}"


def test_nyc_county_fips_empty() -> None:
    pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
        "NY", None, "New York", "city", _ARTIFACTS
    )
    assert county_fips == "", (
        f"Expected empty county_fips for NYC place path, got {county_fips!r}"
    )
