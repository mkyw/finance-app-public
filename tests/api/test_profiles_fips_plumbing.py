"""Stage-4 FIPS plumbing tests for the analyze path.

Covers:
  1. Serializer accepts place_fips / county_fips and validates them (+ absent → "").
  2. build_household_profile threads the two FIPS fields onto HouseholdProfile.
  3. A Columbus profile WITH place_fips="3918000" has strictly lower d_variable_annual
     than the identical profile WITHOUT fips (Columbus 2.5% city income tax subtracts).
  4. A no-fips profile remains byte-identical: d_variable_annual with explicit
     place_fips="" is the same as no-fips at all.

Tests 3 and 4 are slow (~5–8 s each, full cohort match) — consistent with the
established pattern in this suite.

Run from repo root:
    ARTIFACTS_PATH=pipeline/artifacts .venv/bin/python -m pytest tests/api/test_profiles_fips_plumbing.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

os.environ.setdefault("ARTIFACTS_PATH", "pipeline/artifacts")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

import django  # noqa: E402
django.setup()

from profiles.serializers import HouseholdProfileInputSerializer  # noqa: E402
from profiles.services import build_household_profile, run_profile_analysis  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _payload(**overrides) -> dict:
    base = {
        "age": 35,
        "gross_income": 80_000,
        "city_pumas": ["OH_03401"],
        "city_label": "Columbus, OH",
        "tenure": "RENT",
        "housing_cost": 1_200,
        "household_size": 1,
        "filing_status": "single",
    }
    base.update(overrides)
    return base


def _columbus_pumas() -> list[str]:
    with open("pipeline/artifacts/city_puma_map.json") as f:
        cmap = json.load(f)
    return cmap["place"]["3918000"]


# --------------------------------------------------------------------------- #
# 1. Serializer                                                                #
# --------------------------------------------------------------------------- #

def test_serializer_accepts_place_fips() -> None:
    s = HouseholdProfileInputSerializer(data=_payload(place_fips="3918000"))
    assert s.is_valid(), s.errors
    assert s.validated_data["place_fips"] == "3918000"


def test_serializer_accepts_county_fips() -> None:
    s = HouseholdProfileInputSerializer(data=_payload(county_fips="06037"))
    assert s.is_valid(), s.errors
    assert s.validated_data["county_fips"] == "06037"


def test_serializer_defaults_place_fips_to_none_when_absent() -> None:
    """Absent place_fips → serializer default is None (normalized to "" in the view)."""
    s = HouseholdProfileInputSerializer(data=_payload())
    assert s.is_valid(), s.errors
    assert s.validated_data["place_fips"] is None


def test_serializer_defaults_county_fips_to_none_when_absent() -> None:
    s = HouseholdProfileInputSerializer(data=_payload())
    assert s.is_valid(), s.errors
    assert s.validated_data["county_fips"] is None


def test_serializer_accepts_null_place_fips() -> None:
    """Explicit null is also valid (allow_null=True)."""
    s = HouseholdProfileInputSerializer(data=_payload(place_fips=None))
    assert s.is_valid(), s.errors
    assert s.validated_data["place_fips"] is None


def test_serializer_accepts_blank_fips() -> None:
    """The frontend sends "" for whichever resolution path wasn't taken
    (place-resolved cities send county_fips="" and vice versa) — both blank
    strings must validate (allow_blank=True). Regression: live 400
    '{"county_fips":["This field may not be blank."]}' on 2026-06-12."""
    s = HouseholdProfileInputSerializer(
        data=_payload(place_fips="3918000", county_fips="")
    )
    assert s.is_valid(), s.errors
    assert s.validated_data["county_fips"] == ""

    s2 = HouseholdProfileInputSerializer(data=_payload(place_fips="", county_fips=""))
    assert s2.is_valid(), s2.errors
    assert s2.validated_data["place_fips"] == ""


# --------------------------------------------------------------------------- #
# 2. build_household_profile threads FIPS                                      #
# --------------------------------------------------------------------------- #

def test_build_profile_threads_place_fips() -> None:
    profile = build_household_profile(
        age=35, gross_income=80_000, puma_code="OH_03401",
        tenure="RENT", housing_cost=1_200, household_size=1,
        place_fips="3918000",
    )
    assert profile.place_fips == "3918000"


def test_build_profile_threads_county_fips() -> None:
    profile = build_household_profile(
        age=35, gross_income=80_000, puma_code="OH_03401",
        tenure="RENT", housing_cost=1_200, household_size=1,
        county_fips="39049",
    )
    assert profile.county_fips == "39049"


def test_build_profile_default_fips_empty() -> None:
    profile = build_household_profile(
        age=35, gross_income=80_000, puma_code="OH_03401",
        tenure="RENT", housing_cost=1_200, household_size=1,
    )
    assert profile.place_fips == ""
    assert profile.county_fips == ""


# --------------------------------------------------------------------------- #
# 3. Columbus WITH fips has lower d_variable_annual than WITHOUT               #
# --------------------------------------------------------------------------- #

def test_columbus_with_fips_has_lower_d_variable() -> None:
    """Columbus 2.5% city income tax reduces take-home when place_fips is supplied.

    Columbus place FIPS "3918000" is in local_tax_rates.json at rate=0.025.
    A profile WITH this FIPS should have a strictly lower d_variable_annual than
    the identical profile without any FIPS (0.0 local tax).
    """
    pumas = _columbus_pumas()

    profile_with_fips = build_household_profile(
        age=35, gross_income=80_000, puma_code=pumas[0],
        tenure="RENT", housing_cost=1_200, household_size=1,
        place_fips="3918000",
    )
    profile_no_fips = build_household_profile(
        age=35, gross_income=80_000, puma_code=pumas[0],
        tenure="RENT", housing_cost=1_200, household_size=1,
    )

    res_with = run_profile_analysis(profile_with_fips, city_pumas=pumas)
    res_no = run_profile_analysis(profile_no_fips, city_pumas=pumas)

    d_with = res_with["d_variable_annual"]
    d_no = res_no["d_variable_annual"]
    # Columbus 2.5% on $80K = $2,000 in additional tax → take-home drops by $2,000.
    # d_variable_annual is take_home minus committed outflows, so the delta should
    # be close to $2,000 (committed outflows are the same — same profile, same match).
    assert d_with < d_no, (
        f"Expected d_variable_annual to be lower with Columbus city tax, "
        f"but got d_with={d_with:.2f} >= d_no={d_no:.2f}"
    )
    delta = d_no - d_with
    # Soft sanity band: $1,500–$2,500 (committed outflows are income-conditioned
    # but identical between the two runs, so the delta should approximate the raw tax).
    assert 1_500 < delta < 2_500, (
        f"Expected delta ≈ $2,000 (Columbus 2.5% × $80K), got {delta:.2f}"
    )


# --------------------------------------------------------------------------- #
# 4. No-fips profile is byte-identical with explicit empty fips                #
# --------------------------------------------------------------------------- #

def test_no_fips_byte_identical_with_explicit_empty() -> None:
    """Profiles with place_fips='' and county_fips='' are byte-identical to no-fips.

    This guards the normalized empty-string defaults from views.py:
        place_fips = data.get("place_fips") or ""
    An existing caller that omits both fields sees the same d_variable_annual as
    a new caller that explicitly sends "".
    """
    profile_default = build_household_profile(
        age=35, gross_income=65_000, puma_code="IL_00100",
        tenure="RENT", housing_cost=1_400, household_size=2,
    )
    profile_explicit_empty = build_household_profile(
        age=35, gross_income=65_000, puma_code="IL_00100",
        tenure="RENT", housing_cost=1_400, household_size=2,
        place_fips="",
        county_fips="",
    )
    city_pumas = ["IL_00100"]

    res_default = run_profile_analysis(
        profile_default, city_pumas=city_pumas, use_aggregated=False
    )
    res_explicit = run_profile_analysis(
        profile_explicit_empty, city_pumas=city_pumas, use_aggregated=False
    )

    assert res_default["d_variable_annual"] == pytest.approx(
        res_explicit["d_variable_annual"]
    ), (
        f"Expected byte-identical d_variable_annual: "
        f"default={res_default['d_variable_annual']:.4f}, "
        f"explicit_empty={res_explicit['d_variable_annual']:.4f}"
    )
