"""Savings round-trip test for the analyze service layer.

Exercises the service-layer contract directly — serializer +
build_household_profile + run_profile_analysis — without spinning
up Django's APIClient, which would require test-database setup
that this mid-refactor repo doesn't have yet.

Run from repo root:
    ARTIFACTS_PATH=pipeline/artifacts python3.11 -m pytest tests/api/ -v
"""

from __future__ import annotations

import os
import sys

import pytest

# Services.py imports django.conf. Configure a minimal Django
# environment so the module can load — _artifacts_path() is env-var
# first, so settings.BASE_DIR/ARTIFACTS_PATH are never read here.
os.environ.setdefault("ARTIFACTS_PATH", "pipeline/artifacts")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api"))

import django  # noqa: E402

django.setup()

from apps.api.profiles.serializers import HouseholdProfileInputSerializer  # noqa: E402
from apps.api.profiles.services import (  # noqa: E402
    build_household_profile,
    run_profile_analysis,
)


def _payload(**overrides) -> dict:
    base = {
        "age": 35,
        "gross_income": 65000,
        "city_pumas": ["IL_00100"],
        "city_label": "Chicago, IL",
        "tenure": "RENT",
        "housing_cost": 1400,
        "household_size": 2,
        "filing_status": "single",
    }
    base.update(overrides)
    return base


def test_serializer_accepts_savings() -> None:
    s = HouseholdProfileInputSerializer(data=_payload(savings=5000))
    assert s.is_valid(), s.errors
    assert s.validated_data["savings"] == 5000


def test_serializer_defaults_savings_to_zero() -> None:
    s = HouseholdProfileInputSerializer(data=_payload())
    assert s.is_valid(), s.errors
    assert s.validated_data["savings"] == 0


def test_serializer_rejects_negative_savings() -> None:
    s = HouseholdProfileInputSerializer(data=_payload(savings=-100))
    assert not s.is_valid()
    assert "savings" in s.errors


def test_build_household_profile_threads_savings() -> None:
    profile = build_household_profile(
        age=35, gross_income=65000, puma_code="IL_00100",
        tenure="RENT", housing_cost=1400, household_size=2,
        savings=7500,
    )
    assert profile.savings == 7500.0


def test_run_profile_analysis_surfaces_user_reported_savings() -> None:
    profile = build_household_profile(
        age=35, gross_income=65000, puma_code="IL_00100",
        tenure="RENT", housing_cost=1400, household_size=2,
        savings=5000,
    )
    # Pin the disaggregated fallback path: the default flipped to
    # use_aggregated=True at the Stage-3 cut-over, but this suite
    # exercises and must keep green the 55-category disaggregated path.
    out = run_profile_analysis(profile, filing_status="single", use_aggregated=False)
    check = out["balance_sheet"]["assets"]["check"]
    assert check["user_reported"] == pytest.approx(5000.0)
    # Other asset cats don't carry user_reported — kept cohort-only.
    for cat in ("retire", "vehval", "ownval"):
        assert "user_reported" not in out["balance_sheet"]["assets"][cat]
