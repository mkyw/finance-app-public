"""Reference-profile runner for the category-aggregation project (Stage 2).

Runs the six reference profiles through BOTH prediction paths — the live
55-category disaggregated path (``use_aggregated=False``) and the new
aggregated path (``use_aggregated=True``) — and emits a side-by-side
comparison for [HUMAN GATE 2].

Per profile it reports, per aggregate group: the disaggregated allocation
summed over the group's members vs the aggregated group's allocation; plus
totals, model-predicted spend (ex-pinned-housing), feasibility slack, zone,
solver status, and the car-ownership classification. Flags:
  - group divergence > 15% on a well-matched profile (NOTE: for zero-inflated
    bundles this fires by construction — see W2 spec §6; read it as "is the
    aggregate's allocation plausible", not "is there a gap"),
  - implausible aggregate (negative, or > d_variable),
  - material feasibility-slack move (> 10% of d_variable).

Run from repo root:
    ARTIFACTS_PATH=pipeline/artifacts python3.11 scripts/validate_model.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# --- Django bootstrap (services.py imports django.conf) --------------------
os.environ.setdefault("ARTIFACTS_PATH", "pipeline/artifacts")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
sys.path.insert(0, str(_REPO / "apps" / "api"))

import django  # noqa: E402

django.setup()

from apps.api.profiles.city_resolver import resolve_to_pumas  # noqa: E402
from apps.api.profiles.services import (  # noqa: E402
    _artifacts_path,
    build_household_profile,
    run_profile_analysis,
)
from models.matching.algorithm import match_household  # noqa: E402
from shared.constants.categories import (  # noqa: E402
    AGG_GROUPS,
    AGGREGATED_CATEGORY_CODES,
    AGGREGATED_FLOW_CATEGORIES,
)

# --- Reference profiles ----------------------------------------------------
# housing_cost is monthly; pinned and separated from model-predicted spend in
# the comparison, so its exact level only moves feasibility/slack, not the
# model-predicted aggregates. At least two profiles land in the ambiguous car-
# ownership band (NYC, Boston city-resolved) to exercise the W2 mode-split.
PROFILES = [
    {"label": "Chicago $65k size-2 RENT", "city": "Chicago", "state": "IL",
     "age": 35, "income": 65000, "size": 2, "tenure": "RENT", "housing": 1500},
    {"label": "Chicago $75k size-1 RENT", "city": "Chicago", "state": "IL",
     "age": 35, "income": 75000, "size": 1, "tenure": "RENT", "housing": 1500},
    {"label": "NYC $250k size-1 RENT", "city": "New York", "state": "NY",
     "age": 40, "income": 250000, "size": 1, "tenure": "RENT", "housing": 3500},
    {"label": "Atlanta $95k size-4 OWN", "city": "Atlanta", "state": "GA",
     "age": 42, "income": 95000, "size": 4, "tenure": "OWN", "housing": 2200},
    {"label": "Phoenix $38k size-1 RENT", "city": "Phoenix", "state": "AZ",
     "age": 30, "income": 38000, "size": 1, "tenure": "RENT", "housing": 1200},
    {"label": "Boston $500k size-3 OWN", "city": "Boston", "state": "MA",
     "age": 45, "income": 500000, "size": 3, "tenure": "OWN", "housing": 5000},
]

_PINNED = {"RENT": ["rntval"], "OWN": ["mrtgip", "mrtgpp"]}
_DIVERGENCE_FLAG = 0.15
_SLACK_FLAG = 0.10


def _alloc_total(dists: dict) -> float:
    return sum(float(v["feasibility_adjusted"]) for v in dists.values())


def _pinned_total(dists: dict, tenure: str) -> float:
    return sum(float(dists[c]["feasibility_adjusted"]) for c in _PINNED[tenure]
               if c in dists)


def run_one(p: dict) -> dict:
    artifacts = _artifacts_path()
    pumas, via, *_ = resolve_to_pumas(p["state"], None, p["city"], "city", artifacts)
    profile = build_household_profile(
        age=p["age"], gross_income=p["income"], puma_code=pumas[0],
        tenure=p["tenure"], housing_cost=p["housing"], household_size=p["size"],
    )
    dis = run_profile_analysis(profile, city_pumas=pumas, use_aggregated=False)
    agg = run_profile_analysis(profile, city_pumas=pumas, use_aggregated=True)

    dis_d = dis["distributions"]
    agg_d = agg["distributions"]
    tenure = p["tenure"]
    d_var = float(dis["d_variable_annual"])

    groups = {}
    for g, members in AGG_GROUPS.items():
        dis_sum = sum(float(dis_d[m]["feasibility_adjusted"]) for m in members)
        agg_val = float(agg_d[g]["feasibility_adjusted"])
        denom = max(dis_sum, 50.0)
        divergence = (agg_val - dis_sum) / denom
        groups[g] = {
            "disaggregated_sum": round(dis_sum, 2),
            "aggregated": round(agg_val, 2),
            "divergence_pct": round(divergence * 100, 1),
            "diverge_flag": abs(divergence) > _DIVERGENCE_FLAG,
            "implausible": agg_val < 0 or agg_val > d_var,
        }

    slack_move = abs(
        float(agg["feasibility_slack"]) - float(dis["feasibility_slack"])
    ) / max(d_var, 1.0)

    dis_pinned = _pinned_total(dis_d, tenure)
    agg_pinned = _pinned_total(agg_d, tenure)
    return {
        "label": p["label"],
        "resolved_via": via,
        "n_pumas": len(pumas),
        "car_owner_classification": dis["match_metadata"]["car_owner_classification"],
        "car_owner_probability": round(
            float(dis["match_metadata"]["car_owner_probability"]), 3
        ),
        "d_variable_annual": round(d_var, 0),
        "disaggregated": {
            "total_alloc": round(_alloc_total(dis_d), 0),
            "pinned_housing": round(dis_pinned, 0),
            "model_predicted_ex_housing": round(_alloc_total(dis_d) - dis_pinned, 0),
            "slack": round(float(dis["feasibility_slack"]), 0),
            "zone": dis["financial_zone"],
            "solver": dis["solver_status"],
        },
        "aggregated": {
            "total_alloc": round(_alloc_total(agg_d), 0),
            "pinned_housing": round(agg_pinned, 0),
            "model_predicted_ex_housing": round(_alloc_total(agg_d) - agg_pinned, 0),
            "slack": round(float(agg["feasibility_slack"]), 0),
            "zone": agg["financial_zone"],
            "solver": agg["solver_status"],
        },
        "slack_move_pct": round(slack_move * 100, 1),
        "slack_move_flag": slack_move > _SLACK_FLAG,
        "groups": groups,
    }


# ===========================================================================
# Dense profile-grid coverage harness (read-only reconnaissance, 2026-05).
#
# Spans income × age × location × tenure to surface PATTERNS for cause-
# deliberation — it ranks nothing and tunes nothing. household_size is fixed
# (==2) so the four named axes are isolated (size only enters via the sqrt
# equivalence scale; holding it constant keeps the income axis clean). housing
# is a plausible per-(income,location,tenure) amount — it is pinned and only
# moves feasibility/slack, never the matched category predictions (matching
# keys on age/income/size/tenure/puma, NOT housing_cost).
#
# Output: agent-artifacts/investigations/profile_grid_raw.json — the MODEL grid.
# The benchmark/intuition references and the pattern analysis are produced
# separately (by the analyst reading this grid), per the task.
# ===========================================================================

_GRID_INCOMES = [30_000, 50_000, 75_000, 120_000, 250_000]
_GRID_AGES = [25, 30, 35, 40, 45]
# (city, state, cost-tier) — real PUMA sets via resolve_to_pumas.
_GRID_LOCATIONS = [
    ("New York", "NY", "high"),   # high-cost coastal
    ("Chicago", "IL", "mid"),     # mid-cost metro
    ("Phoenix", "AZ", "low"),     # low-cost metro
]
_GRID_TENURES = ["RENT", "OWN"]
_GRID_SIZE = 2
_HOUSING_FRAC = {"high": 0.36, "mid": 0.28, "low": 0.22}  # of monthly gross


def _plausible_housing(income: int, tier: str) -> int:
    """Plausible monthly housing for (income, location). Pinned only — affects
    feasibility/slack, not the matched category predictions."""
    monthly = income / 12.0
    h = monthly * _HOUSING_FRAC[tier]
    return int(round(max(600.0, min(h, 9000.0))))


def _grid_profiles() -> list[dict]:
    """All corners + dense interior, deduped by (income, age, city, tenure).

    Three blocks: (1) the income×location×tenure plane at age 35 (isolates
    income/location/tenure); (2) the full age sweep at the high-cost coastal
    location for three income levels (the age-skew gradient, strongest for young
    high earners); (3) the age×income×location×tenure extreme corners.
    """
    seen: set[tuple] = set()
    out: list[dict] = []

    def add(income, age, loc, tenure):
        city, state, tier = loc
        key = (income, age, city, tenure)
        if key in seen:
            return
        seen.add(key)
        out.append({
            "income": income, "age": age, "city": city, "state": state,
            "tier": tier, "tenure": tenure, "size": _GRID_SIZE,
            "housing": _plausible_housing(income, tier),
            "label": f"{city} ${income//1000}k age{age} {tenure}",
        })

    # (1) income × location × tenure plane at age 35.
    for income in _GRID_INCOMES:
        for loc in _GRID_LOCATIONS:
            for tenure in _GRID_TENURES:
                add(income, 35, loc, tenure)

    # (2) age sweep at high-cost coastal (NYC), RENT, for 3 income levels.
    nyc = _GRID_LOCATIONS[0]
    for income in (50_000, 120_000, 250_000):
        for age in _GRID_AGES:
            add(income, age, nyc, "RENT")

    # (3) extreme corners: income {min,max} × loc {high,low} × tenure × age {min,max}.
    for income in (_GRID_INCOMES[0], _GRID_INCOMES[-1]):
        for loc in (_GRID_LOCATIONS[0], _GRID_LOCATIONS[-1]):
            for tenure in _GRID_TENURES:
                for age in (_GRID_AGES[0], _GRID_AGES[-1]):
                    add(income, age, loc, tenure)

    return out


def run_grid_profile(p: dict, artifacts: str) -> dict:
    """One grid cell: aggregated analyze + cohort_median_income from matching."""
    pumas, via, *_ = resolve_to_pumas(p["state"], None, p["city"], "city", artifacts)
    profile = build_household_profile(
        age=p["age"], gross_income=p["income"], puma_code=pumas[0],
        tenure=p["tenure"], housing_cost=p["housing"], household_size=p["size"],
    )
    r = run_profile_analysis(profile, city_pumas=pumas, use_aggregated=True)
    # cohort_median_income is not surfaced in the analyze output — pull it from
    # the match directly (the age-skew fingerprint: cohort median vs user income).
    match = match_household(profile, artifacts, city_pumas=pumas, aggregate=True)
    d = r["distributions"]
    mm = r["match_metadata"]
    preds = {c: round(float(d[c]["feasibility_adjusted"]), 2)
             for c in AGGREGATED_CATEGORY_CODES}
    return {
        **{k: p[k] for k in ("label", "income", "age", "city", "tier",
                             "tenure", "size", "housing")},
        "n_pumas": len(pumas),
        "d_variable_annual": round(float(r["d_variable_annual"]), 0),
        "feasibility_slack": round(float(r["feasibility_slack"]), 0),
        "financial_zone": r["financial_zone"],
        "solver_status": r["solver_status"],
        "structural_deficit": round(float(r["structural_deficit"]), 0),
        "car_owner_classification": mm["car_owner_classification"],
        "car_owner_probability": round(float(mm["car_owner_probability"]), 3),
        "n_effective": round(float(mm["n_effective"]), 1),
        "cohort_median_income": round(float(match.cohort_median_income), 0),
        "predictions": preds,
    }


def main_grid() -> int:
    artifacts = _artifacts_path()
    profiles = _grid_profiles()
    print(f"Running {len(profiles)}-cell profile grid (aggregated path)...")
    results = []
    for i, p in enumerate(profiles, 1):
        results.append(run_grid_profile(p, artifacts))
        print(f"  [{i:>3}/{len(profiles)}] {p['label']}")
    out_path = (_REPO / "agent-artifacts" / "investigations"
                / "profile_grid_raw.json")
    out_path.write_text(json.dumps(
        {"flow_categories": sorted(AGGREGATED_FLOW_CATEGORIES),
         "all_categories": list(AGGREGATED_CATEGORY_CODES),
         "n_profiles": len(results),
         "profiles": results},
        indent=2,
    ))
    print(f"\nWrote {out_path}  ({len(results)} profiles)")
    return 0


def main() -> int:
    results = [run_one(p) for p in PROFILES]
    out_path = _REPO / "agent-artifacts" / "aggregation" / "gate2_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))

    for r in results:
        print(f"\n{'='*80}\n{r['label']}  "
              f"[{r['car_owner_classification']} P(own)={r['car_owner_probability']}, "
              f"{r['n_pumas']} pumas via {r['resolved_via']}]")
        dd, aa = r["disaggregated"], r["aggregated"]
        print(f"  d_variable=${r['d_variable_annual']:,.0f}  "
              f"zone {dd['zone']}->{aa['zone']}  solver {dd['solver']}->{aa['solver']}")
        print(f"  model-predicted (ex pinned housing): "
              f"disagg=${dd['model_predicted_ex_housing']:,.0f}  "
              f"agg=${aa['model_predicted_ex_housing']:,.0f}   "
              f"pinned=${dd['pinned_housing']:,.0f}")
        print(f"  slack: disagg=${dd['slack']:,.0f}  agg=${aa['slack']:,.0f}  "
              f"move={r['slack_move_pct']}%{'  <-FLAG' if r['slack_move_flag'] else ''}")
        print(f"  {'group':17}{'disagg_sum':>12}{'aggregated':>12}{'diverg%':>9}  flags")
        for g, gd in r["groups"].items():
            flags = []
            if gd["diverge_flag"]:
                flags.append(">15%")
            if gd["implausible"]:
                flags.append("IMPLAUSIBLE")
            print(f"  {g:17}{gd['disaggregated_sum']:>12,.0f}{gd['aggregated']:>12,.0f}"
                  f"{gd['divergence_pct']:>8.1f}%  {' '.join(flags)}")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    # `grid` → dense profile-grid coverage harness; default → Gate-2 runner.
    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        raise SystemExit(main_grid())
    raise SystemExit(main())
