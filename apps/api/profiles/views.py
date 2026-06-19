"""DRF views for the profiles app.

POST /api/profiles/analyze/ runs the full model chain on a
household-profile input and returns the analysis payload. MVP:
unauthenticated — the onboarding flow is anonymous.
"""

from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .city_resolver import resolve_to_pumas
from .display import build_display_rollup, rollup_to_dict
from .serializers import (
    CityResolveInputSerializer,
    HouseholdProfileInputSerializer,
)
from .services import (
    _artifacts_path,
    build_household_profile,
    run_profile_analysis,
)


class ProfileAnalysisView(APIView):
    """Anonymous household-profile analysis endpoint."""

    permission_classes: list = []

    def post(self, request):
        serializer = HouseholdProfileInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )

        data = serializer.validated_data
        filing_status = data["filing_status"]
        city_pumas = data["city_pumas"]

        # Serializer defaults place_fips/county_fips to None; normalize to ""
        # for the frozen HouseholdProfile str fields.
        place_fips = data.get("place_fips") or ""
        county_fips = data.get("county_fips") or ""

        # Seed the profile with the first city PUMA. match_household will
        # load the full city_pumas set plus the top-M similar neighbors.
        profile = build_household_profile(
            age=data["age"],
            gross_income=data["gross_income"],
            puma_code=city_pumas[0],
            tenure=data["tenure"],
            housing_cost=data["housing_cost"],
            household_size=data["household_size"],
            savings=data.get("savings", 0.0),
            cc_carried_balance=data.get("cc_carried_balance", 0.0),
            student_loan_payment=data.get("student_loan_payment", 0.0),
            auto_loan_payment=data.get("auto_loan_payment", 0.0),
            other_debt_payment=data.get("other_debt_payment", 0.0),
            place_fips=place_fips,
            county_fips=county_fips,
        )

        try:
            analysis = run_profile_analysis(
                profile,
                filing_status=filing_status,
                city_pumas=city_pumas,
            )
        except FileNotFoundError as exc:
            # Missing artifact — pipeline/export/ hasn't been run.
            return Response(
                {"detail": f"artifact missing: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except ValueError as exc:
            # e.g. unknown puma_code, malformed profile input.
            return Response(
                {"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST
            )

        # Attach the topic-grouped display roll-up (single source of truth for
        # the frontend's grouped view — derived from shared/constants/categories.py
        # via display.py, so the grouping is defined in exactly one place).
        analysis["display_rollup"] = rollup_to_dict(
            build_display_rollup(analysis, profile.tenure.value)
        )

        return Response(analysis, status=status.HTTP_200_OK)


class CityResolveView(APIView):
    """Name-based city -> PUMA resolver.

    POST /api/profiles/resolve-city/

    Request body:
        {
            "state_code": "CA",                 # USPS postal
            "county_name": "Los Angeles County",
            "city_name": "Santa Monica",
            "addresstype": "city",              # Place-vs-County switch; autocomplete always sends "city"
            "city_label": "Santa Monica, CA"    # optional, echoed back
        }

    Response:
        {
            "pumas": ["CA_03728", "CA_03760", "CA_03774"],
            "resolved_via": "place" | "county",
            "city_label": "Santa Monica, CA"
        }
    """

    permission_classes: list = []

    def post(self, request):
        serializer = CityResolveInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors, status=status.HTTP_400_BAD_REQUEST
            )
        data = serializer.validated_data

        try:
            pumas, resolved_via, place_fips, county_fips = resolve_to_pumas(
                state_code=data["state_code"],
                county_name=data.get("county_name") or None,
                city_name=data.get("city_name") or None,
                addresstype=data.get("addresstype") or None,
                artifacts_path=_artifacts_path(),
            )
        except FileNotFoundError as exc:
            return Response(
                {"detail": f"artifact missing: {exc}"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except ValueError as exc:
            return Response(
                {"detail": str(exc)}, status=status.HTTP_404_NOT_FOUND
            )

        return Response(
            {
                "pumas": pumas,
                "resolved_via": resolved_via,
                "city_label": data.get("city_label", ""),
                "place_fips": place_fips,
                "county_fips": county_fips,
            },
            status=status.HTTP_200_OK,
        )
