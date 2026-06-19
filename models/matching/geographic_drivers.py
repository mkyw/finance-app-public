"""Geographic-driver -> category factor slot (value layer).

A small, extensible mechanism for value-layer corrections that vary a category's
anchor by *where the household is*, beyond what the cohort and the RPP price
scalar already carry. Each "driver" owns one geographic dimension and contributes
a multiplicative factor (1.0 = no effect) per category; the factors compose.

**Driver 1 — climate (live).** The fusion model is geographically blind below
census division, so cohort utility dollars are flat across climates (see
``agent-artifacts/investigations/utilities_diagnosis.md``). Two orthogonal lanes
restore the regional signal, both normalized to 1.0 at their baseline so the
national total is preserved (they inject spread, not level):

  * EIA state consumption ratio (``eia_utility.py``) — the measured state-level
    climate-quantity signal (national hu-weighted mean = 1.0).
  * NOAA within-state degree-day ratio (``climate_normals.json``) — redistributes
    within a state (coastal vs inland) without re-touching the state level
    (state hu-weighted mean = 1.0). elec uses cooling degree-days (CDD), ngas and
    ofuel use heating degree-days (HDD). Electricity also responds to heating in
    electric-heat regions, but the EIA state baseline already absorbs that mix, so
    the within-state elec lane uses CDD only and lets EIA carry electric heat.

  These multiply with the existing RPP ``utilities`` price scalar (the price lane,
  applied elsewhere): ``anchor x EIA_state x NOAA_within_state x RPP_price``. No
  double-count — EIA/NOAA are quantity, RPP is price; for a low-price/high-load
  state like AZ they partially offset, which is physically correct.

**Future driver — urban density (pending, A1).** The transportation
income-blindness + classification-band cliff for urban non-car-owners
(``agent-artifacts/investigations/profile_grid_coverage.md`` A1) is the next
geographic driver: an urban-density -> transit-category factor. It plugs into the
same registry below — add a ``_urban_density_driver`` branch to
:meth:`GeographicDrivers.factor`; nothing else changes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

_LOG = logging.getLogger(__name__)


def load_climate_normals(cache_path: str) -> dict[str, dict]:
    """Load the per-PUMA NOAA within-state degree-day ratios (``by_puma``).

    File-read only (NOAA normals are static reference data, rebuilt per-decade by
    ``scripts/build_climate_normals.py``). Missing/unreadable -> ``{}``; the
    within-state lane then no-ops to 1.0.
    """
    try:
        with open(cache_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("climate_normals cache unreadable (%r); no within-state lane.", exc)
        return {}
    by_puma = data.get("by_puma", {})
    return by_puma if isinstance(by_puma, dict) else {}

# CEX category -> (eia_utility key, NOAA degree-day field). Authoritative
# membership of the climate driver: elec is cooling-driven, ngas/ofuel heating.
_CLIMATE_CATEGORIES: dict[str, tuple[str, str]] = {
    "elec": ("elec", "cdd_ratio"),
    "ngas": ("ngas", "hdd_ratio"),
    "ofuel": ("ofuel", "hdd_ratio"),
}


def _climate_driver(
    category: str,
    state: str,
    puma_code: str,
    eia_utility_scalars: dict[str, dict[str, float]],
    climate_normals: dict[str, dict],
) -> float:
    """EIA state consumption ratio x NOAA within-state degree-day ratio.

    Returns 1.0 for non-climate categories and for any missing state/PUMA lookup
    (clean no-op when an artifact is absent).
    """
    spec = _CLIMATE_CATEGORIES.get(category)
    if spec is None:
        return 1.0
    eia_key, dd_field = spec

    eia = eia_utility_scalars.get(eia_key, {}).get(state, 1.0)

    noaa = 1.0
    puma_norm = climate_normals.get(puma_code)
    if puma_norm is not None:
        noaa = float(puma_norm.get(dd_field, 1.0))

    return float(eia) * noaa


@dataclass
class GeographicDrivers:
    """Holds the loaded geographic-driver artifacts and composes their factors.

    Construct once per :func:`match_household` call and pass into the value-layer
    scalar. Built so future drivers (urban density, etc.) add one field + one
    multiply in :meth:`factor` without touching call sites.
    """

    eia_utility_scalars: dict[str, dict[str, float]] = field(default_factory=dict)
    climate_normals: dict[str, dict] = field(default_factory=dict)

    def factor(self, category: str, puma_code: str) -> float:
        """Composite geographic-driver multiplier for ``category`` at ``puma_code``."""
        state = puma_code[:2]
        f = 1.0
        f *= _climate_driver(
            category, state, puma_code, self.eia_utility_scalars, self.climate_normals
        )
        # Future: f *= _urban_density_driver(category, puma_code, self.urban_density)
        return f
