#!/usr/bin/env python3
"""Before/after validation for the utility climate-load geographic factor.

Runs representative profiles through the aggregated analyze path and prints the
utility predictions (elec/ngas/ofuel + the excluded watrsh/intphn) plus the
national-total preservation check. Run it once with the artifacts present
(``after``) and once with them moved aside (``before``); diff the two. The
$90k Chicago single (the motivating profile) is included.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("ARTIFACTS_PATH", "pipeline/artifacts")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, str(_REPO / "apps" / "api"))
import django  # noqa: E402

django.setup()
from apps.api.profiles.city_resolver import resolve_to_pumas  # noqa: E402
from apps.api.profiles.services import build_household_profile, run_profile_analysis  # noqa: E402

PROFILES = [
    {"label": "Phoenix $90k size-1 RENT", "city": "Phoenix", "state": "AZ", "age": 35, "income": 90000, "size": 1, "tenure": "RENT", "housing": 1400},
    {"label": "Chicago $90k size-1 RENT (motivating)", "city": "Chicago", "state": "IL", "age": 35, "income": 90000, "size": 1, "tenure": "RENT", "housing": 1500},
    {"label": "New York $90k size-1 RENT", "city": "New York", "state": "NY", "age": 35, "income": 90000, "size": 1, "tenure": "RENT", "housing": 2200},
    {"label": "Minneapolis $90k size-1 RENT (cold extreme)", "city": "Minneapolis", "state": "MN", "age": 35, "income": 90000, "size": 1, "tenure": "RENT", "housing": 1400},
    {"label": "Miami $90k size-1 RENT (hot extreme)", "city": "Miami", "state": "FL", "age": 35, "income": 90000, "size": 1, "tenure": "RENT", "housing": 1600},
    {"label": "Portland(ME) $90k size-1 RENT (heating oil)", "city": "Portland", "state": "ME", "age": 35, "income": 90000, "size": 1, "tenure": "RENT", "housing": 1400},
]
UTILS = ["elec", "ngas", "ofuel", "watrsh", "intphn"]


def main() -> int:
    artifacts = os.environ["ARTIFACTS_PATH"]
    out = {}
    for p in PROFILES:
        pumas, *_ = resolve_to_pumas(p["state"], None, p["city"], "city", artifacts)
        profile = build_household_profile(
            age=p["age"], gross_income=p["income"], puma_code=pumas[0],
            tenure=p["tenure"], housing_cost=p["housing"], household_size=p["size"],
        )
        r = run_profile_analysis(profile, city_pumas=pumas, use_aggregated=True)
        d = r["distributions"]
        row = {u: round(float(d[u]["feasibility_adjusted"]), 1) for u in UTILS if u in d}
        row["_alloc_total"] = round(sum(float(v["feasibility_adjusted"]) for v in d.values()), 0)
        out[p["label"]] = row
        print(f"{p['label']:46s} " + "  ".join(f"{u}={row.get(u,0):7.1f}" for u in UTILS)
              + f"  | total={row['_alloc_total']:.0f}")
    Path("/tmp/climate_validation.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
