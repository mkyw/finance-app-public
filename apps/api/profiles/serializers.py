"""DRF serializers for the profile-analysis endpoint."""

from __future__ import annotations

from rest_framework import serializers


class HouseholdProfileInputSerializer(serializers.Serializer):
    """Request body for POST /api/profiles/analyze/."""

    age = serializers.IntegerField(min_value=18, max_value=100)
    gross_income = serializers.FloatField(min_value=0)
    # City -> PUMA resolution happens on the frontend via
    # /api/profiles/resolve-city/ and the result is sent here as the
    # seed set for match_household. The first entry is used as the
    # nominal puma_code on HouseholdProfile.
    city_pumas = serializers.ListField(
        child=serializers.CharField(max_length=20),
        min_length=1,
        max_length=100,
    )
    city_label = serializers.CharField(
        max_length=200, required=False, default=""
    )
    # Census FIPS identifiers from resolve-city. The frontend sends "" for
    # whichever path wasn't taken (place vs county) — allow_blank is required
    # or DRF 400s on the empty string.
    place_fips = serializers.CharField(
        max_length=10, required=False, allow_null=True, allow_blank=True, default=None
    )
    county_fips = serializers.CharField(
        max_length=10, required=False, allow_null=True, allow_blank=True, default=None
    )
    tenure = serializers.ChoiceField(choices=["OWN", "RENT"])
    housing_cost = serializers.FloatField(min_value=0)
    household_size = serializers.IntegerField(min_value=1, max_value=20)
    # User-reported liquid savings. Defaults to 0 when the frontend
    # doesn't send it (pre-wiring clients); surfaced back in the
    # response as balance_sheet.assets.check.user_reported.
    savings = serializers.FloatField(min_value=0, required=False, default=0.0)
    # User-reported debt (all optional, default 0 = no signal → cohort
    # prior). CC is a carried balance ("carry month to month", excludes
    # pay-in-full); the other three are monthly payments. Surfaced back
    # in the response as balance_sheet.liabilities.* with per-component
    # source (user_reported / cohort_predicted / not_modeled).
    cc_carried_balance = serializers.FloatField(min_value=0, required=False, default=0.0)
    student_loan_payment = serializers.FloatField(min_value=0, required=False, default=0.0)
    auto_loan_payment = serializers.FloatField(min_value=0, required=False, default=0.0)
    other_debt_payment = serializers.FloatField(min_value=0, required=False, default=0.0)
    filing_status = serializers.ChoiceField(
        choices=["single", "married_filing_jointly", "head_of_household"],
        default="single",
    )


class CityResolveInputSerializer(serializers.Serializer):
    """Request body for POST /api/profiles/resolve-city/."""

    state_code = serializers.CharField(max_length=2)
    county_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    city_name = serializers.CharField(
        max_length=120, required=False, allow_blank=True, default=""
    )
    addresstype = serializers.CharField(
        max_length=40, required=False, allow_blank=True, default=""
    )
    city_label = serializers.CharField(
        max_length=200, required=False, allow_blank=True, default=""
    )


class ProfileAnalysisOutputSerializer(serializers.Serializer):
    """Response shape for POST /api/profiles/analyze/."""

    financial_zone = serializers.CharField()
    structural_deficit = serializers.FloatField()
    feasibility_slack = serializers.FloatField()
    d_variable_annual = serializers.FloatField()
    pace_annual = serializers.FloatField()
    solver_status = serializers.CharField()
    distributions = serializers.DictField()
    benefits = serializers.ListField()
    match_metadata = serializers.DictField()
