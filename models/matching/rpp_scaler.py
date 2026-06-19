"""BEA Regional Price Parity (RPP) scalars for spatial price adjustment.

Two entry points, serving two different stages of the pipeline:

  1. ``_get_rpp_value`` / ``_get_state_rpp_value`` — raw BEA index
     values (on US=100 base). Used by the household-level RPP
     correction in ``algorithm._apply_household_rpp_correction``: for
     cross-state households in the matched pool, multiply each CEX
     spending value by ``target_rpp / source_state_rpp`` so the
     household's raw spend is translated from its own state's price
     level to the target location's price level. Same-state
     households are untouched (their raw CEX values already reflect
     target-area prices).

  2. ``get_rpp_scalar`` — post-percentile residual spatial scalar.
     After household-level correction, the pool's raw cross-state
     values have been translated to target-area BEA-goods (etc.)
     prices, so for most categories there's nothing more to do at
     display time (returns 1.0). Gas is special: BEA goods RPP
     averages groceries, electronics, apparel, gas, etc., and is
     coarser than the gas-specific spatial premium EIA publishes
     weekly. For gas we route through the EIA state-level scalar
     instead.

Artifact contract (``pipeline/artifacts/rpp_scalars.json``): produced
by ``pipeline/export/export_rpp_scalars.R``. Keys:

    metadata   — source, year, generated_at, index_base
    by_puma    — {"CA_08507": {"all_items": 110.4, "goods": ...}, ...}
    by_state   — {"CA":       {"all_items": 110.7, "goods": ...}, ...}
    national   — {"all_items": 100.0, ...}

Category map: each CEX category is pinned to exactly one of the four
BEA sub-indices (goods / housing services / utilities / other
services). Any category outside the explicit map falls back to
``all_items``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from shared.constants.categories import CATEGORY_CODES

_LOG = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH: str = "pipeline/artifacts/rpp_scalars.json"

# CEX category -> BEA RPP sub-index. Follows the 2024 MARPP / SARPP
# schema (Line 2 = Goods, 3 = Services: Housing, 4 = Services: Utilities,
# 5 = Services: Other). Categories not listed fall back to ``all_items``.
#
# Notes on a few judgment calls:
#   - ``gas`` (motor fuel) is a goods category in the CPI/RPP taxonomy,
#     not a utility. ``elec`` / ``ngas`` / ``watrsh`` / ``ofuel`` are.
#   - ``vehnew`` is a good. (It appears in the services bucket in the
#     prompt's spec as a duplicate — we take the first/goods mapping.)
#   - ``hmtimp`` (home maintenance and improvement) is a services
#     category but not clearly housing vs. other; slotted under
#     ``other_services`` to avoid double-counting with the
#     housing-services adjustment that already hits rntval/ownval.
RPP_CATEGORY_MAP: dict[str, str] = {
    # Goods
    "eathome":  "goods",
    "eatout":   "goods",
    "gas":      "goods",
    "cloftw":   "goods",
    "eltrnp":   "goods",
    "jwlbg":    "goods",
    "furhwr":   "goods",
    "happl":    "goods",
    "hhpcp":    "goods",
    "vehprd":   "goods",
    "vehnew":   "goods",
    "vehusd":   "goods",
    # vehreg uses direct per-state DMV costs (vehreg_state.py),
    # not BEA goods. State registration fees vary ~10x while
    # the goods sub-index varies ~1.05x — far too coarse.
    "vehval":   "goods",
    # Utilities
    "elec":     "utilities",
    "ngas":     "utilities",
    "watrsh":   "utilities",
    "ofuel":    "utilities",
    # Housing services
    "rntval":   "housing_services",
    "ownval":   "housing_services",
    "mrtgip":   "housing_services",
    "mrtgpp":   "housing_services",
    "mrtgps":   "housing_services",
    "ptaxp":    "housing_services",
    "rntexp":   "housing_services",
    # Other services
    "health":   "other_services",
    "hinsp":    "other_services",
    "hhpcs":    "other_services",
    "oesrv":    "other_services",
    "intphn":   "other_services",
    "vehins":   "other_services",
    "vehprn":   "other_services",
    "vehint":   "other_services",
    "vehmlr":   "other_services",
    "educ":     "other_services",
    "taxis":    "other_services",
    "pubtrn":   "other_services",
    "hotel":    "other_services",
    "airshp":   "other_services",
    "oeprd":    "other_services",
    "recrp":    "other_services",
    "chrty":    "other_services",
    "ocash":    "other_services",
    "finpay":   "other_services",
    "othint":   "other_services",
    "stdint":   "other_services",
    "othfin":   "other_services",
    "hmtimp":   "other_services",
    "ohouse":   "other_services",
}

# Guard: every CATEGORY_CODES entry either appears in the map or falls to
# all_items. We don't assert coverage — financial asset cats (check,
# retire, stock, lifval, stddbt, othdbt) legitimately have no RPP signal
# and should use the all-items scalar.

# Categories that must NOT be scaled by the household-level RPP
# correction because they are balance-sheet entries (savings, debt,
# retirement, transfers) rather than consumption spending. Scaling
# these by a spatial price index is meaningless — a dollar of 401(k)
# contribution in Iowa and a dollar in New York are both just dollars.
BALANCE_CATEGORIES: frozenset[str] = frozenset({
    "check",   # liquid savings
    "retire",  # retirement contributions
    "stock",   # taxable brokerage
    "lifval",  # life insurance cash value
    "stddbt",  # student debt principal
    "othdbt",  # other debt principal
    "ocash",   # misc cash transfers
    "chrty",   # charitable giving (not priced by geography)
    "finpay",  # financial-services payments
    "othfin",
    "stdint",  # student-loan interest — debt service, not consumption
    "othint",  # other interest — debt service, not consumption
})

# Categories that represent lumpy, infrequent purchases where the
# household's reported spend may have occurred anywhere (vacation, a
# car bought while visiting, airfare for a trip away from home). For
# these, translating "source state" -> "target location" prices is
# misleading: the purchase wasn't made at source-state prices in the
# first place. Skip the household-level RPP correction and let the
# cohort median carry whatever regional noise it has.
EPISODIC_CATEGORIES: frozenset[str] = frozenset({
    "vehnew",
    "vehusd",
    "vehval",   # cohort-median residual value of owned vehicles
    "happl",    # major appliances
    "jwlbg",    # jewelry, watches, luggage
    "furhwr",   # furniture, home furnishings
    "eltrnp",   # consumer electronics products
    "hotel",    # lodging away from home
    "airshp",   # airfare and ship fares
    "educ",     # tuition and fees
    "hmtimp",   # home maintenance / improvement projects
})

# Categories that bypass the cohort-percentile + RPP + CPI pipeline
# entirely because they have a higher-fidelity direct-cost data source.
# The household-level RPP correction must skip these so the synthetic
# population's raw values don't get scrambled before the dedicated path
# overwrites them in ``algorithm.py``.
DIRECT_COST_CATEGORIES: frozenset[str] = frozenset({
    "vehreg",   # state DMV registration/inspection — see vehreg_state.py
})


def _identity_scalars() -> dict:
    """Minimal RPP dict that yields scalar 1.0 for any lookup. Test fallback."""
    return {
        "metadata": {"source": "identity (tests)"},
        "by_puma":  {},
        "by_state": {},
        "national": {
            "all_items":        100.0,
            "goods":            100.0,
            "housing_services": 100.0,
            "utilities":        100.0,
            "other_services":   100.0,
        },
    }


def load_rpp_scalars(cache_path: str = _DEFAULT_CACHE_PATH) -> dict:
    """Load the RPP lookup JSON. Returns identity (all-1.0) on any failure.

    Unlike CPI, the RPP artifact is produced offline by the R export and
    is expected to be present. If it's missing we log and fall back to
    the identity scalars so the rest of the pipeline still runs (just
    without spatial adjustment).
    """
    path = Path(cache_path)
    if not path.exists():
        _LOG.warning(
            "rpp_scalars.json missing at %s; falling back to identity "
            "(no spatial adjustment). Run export_rpp_scalars.R to build.",
            path,
        )
        return _identity_scalars()
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("rpp_scalars unreadable (%r); using identity", exc)
        return _identity_scalars()


def _state_from_puma(puma_code: str) -> str | None:
    """Extract two-letter state postal from the ``ST_NNNNN`` puma_code."""
    if not puma_code or len(puma_code) < 3 or puma_code[2] != "_":
        return None
    return puma_code[:2]


def _sub_index_for(category: str) -> str:
    """RPP sub-index key for a CEX category (falls back to ``all_items``)."""
    return RPP_CATEGORY_MAP.get(category, "all_items")


def _get_rpp_value(category: str, puma_code: str, rpp_data: dict) -> float:
    """Raw BEA RPP index for ``category`` at ``puma_code`` (on US=100 base).

    Fallback chain: PUMA metro -> state -> national -> 100.0. Used by
    the household-level RPP correction, which divides two raw indexes
    to get a translation ratio. Do NOT divide by 100 here — callers
    that want the 1.0-centered scalar should use ``get_rpp_scalar``.
    """
    sub = _sub_index_for(category)
    by_puma = rpp_data.get("by_puma", {}).get(puma_code)
    if by_puma is not None and sub in by_puma:
        return float(by_puma[sub])
    state = _state_from_puma(puma_code)
    if state is not None:
        by_state = rpp_data.get("by_state", {}).get(state)
        if by_state is not None and sub in by_state:
            return float(by_state[sub])
    national = rpp_data.get("national", {})
    if sub in national:
        return float(national[sub])
    return 100.0


def _get_state_rpp_value(category: str, state_code: str, rpp_data: dict) -> float:
    """Raw BEA RPP index for ``category`` at a two-letter state code.

    Falls back to national (100.0) when the state isn't in ``by_state``.
    Used as the source-location RPP in the household-level correction:
    each pool household's source is a state (not a specific PUMA),
    because the synthetic population is drawn from nationwide CEX and
    we don't know the exact PUMA each household was recorded in.
    """
    sub = _sub_index_for(category)
    by_state = rpp_data.get("by_state", {}).get(state_code)
    if by_state is not None and sub in by_state:
        return float(by_state[sub])
    national = rpp_data.get("national", {})
    if sub in national:
        return float(national[sub])
    return 100.0


def get_rpp_scalar(
    category: str,
    puma_code: str,
    rpp_data: dict,
    eia_gas_scalars: dict[str, float] | None = None,
) -> float:
    """Post-percentile residual spatial scalar for display scaling.

    Architecture note: the heavy lifting of "translate pool spending
    to target location prices" now happens at the *household level*
    via ``_apply_household_rpp_correction``, so by the time this
    function is called the cohort percentiles already reflect
    target-area prices at the BEA sub-index level. For most
    categories there is no further spatial work to do — this returns
    1.0 and the caller's scalar chain is just CPI temporal.

    The one exception is ``gas``: BEA's "goods" sub-index averages
    gasoline with groceries, apparel, electronics, etc., so the
    household-level correction has (correctly) used a coarse
    multiplier for gas. EIA publishes weekly state-level retail
    gasoline prices, which are the gold standard for gas-specific
    spatial scaling. When an EIA scalar is available for the user's
    state we return it; otherwise 1.0 (graceful no-op).
    """
    if category == "gas":
        if eia_gas_scalars:
            state = _state_from_puma(puma_code)
            if state is not None and state in eia_gas_scalars:
                return float(eia_gas_scalars[state])
        return 1.0
    return 1.0
