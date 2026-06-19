"""State-level vehicle registration / licensing / inspection costs.

Why a separate module (not folded into ``rpp_scaler`` or
``cpi_scaler``): vehicle registration fees are statutory and
state-specific. They diverge by an order of magnitude between flat-fee
states (FL ~$32, MS ~$14, NM ~$30) and ad-valorem states (AZ ~$350,
CO ~$340, IA ~$200), but BEA's RPP "goods" sub-index averages them
together with groceries, apparel, and electronics — at most a 1.05×
spread vs the ~10× spread in actual fees. CPI's "Motor vehicle fees"
series (CUUR0000SEGE) is national-only with no regional breakout.

The CEX cohort percentile is also a poor estimator: vehreg is small,
discretionary-looking in the CEX diary, and gets noisy reporting.
State DMV-published averages are the authoritative ground truth.

So for ``vehreg`` specifically we bypass the cohort-percentile +
RPP + CPI machinery entirely and compute:

    predicted_vehreg = state_$_per_vehicle * predicted_cars

where ``state_$_per_vehicle`` is the hand-curated average annual
registration cost for one passenger vehicle in the user's state, and
``predicted_cars`` follows the existing car-ownership classification
in ``algorithm.py``:

    owner       (p > 0.6):  cohort_mean_veh
    non_owner   (p < 0.3):  0
    ambiguous   else:       cohort_mean_veh * car_owner_probability

Artifact contract (``pipeline/artifacts/vehreg_state_costs.json``):

    {
      "generated_at": "2026-05-08T...",
      "source_year": 2024,
      "source": "Hand-curated from state DMV websites; per-state notes",
      "national_mean_per_vehicle": 145.0,
      "state_per_vehicle": {"AL": 23.0, "AK": 100.0, "AZ": 350.0, ...},
      "notes": {"AL": "Standard passenger reg + ad-valorem", ...}
    }

Refresh policy: yearly, by hand. There's no API to call; the loader
just reads the on-disk JSON. If the file is missing or malformed the
loader returns ``({}, 0.0)`` and warns; downstream code defaults to
the national mean (which will then be 0.0 — vehreg drops out of the
allocation until the data file is committed). That's intentional: a
stale-but-real number is better than a silent fabrication, and a
missing file should be obvious in the analyzed output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH: str = "pipeline/artifacts/vehreg_state_costs.json"


def load_vehreg_state_costs(
    cache_path: str | Path = _DEFAULT_CACHE_PATH,
) -> tuple[dict[str, float], float]:
    """Load per-state annual vehicle registration costs.

    Returns:
        ``(state_per_vehicle, national_mean_per_vehicle)``. The first
        element is a ``{state_code: $}`` dict. The second is the
        scalar fallback used by ``lookup_state_cost`` when a state is
        absent from the dict (e.g. small territories, missing data).

    Never raises. Missing file or parse error → ``({}, 0.0)`` plus a
    warning. Caller code treats a $0 lookup as "no vehreg signal" and
    the allocator simply emits 0 for that category.
    """
    path = Path(cache_path)
    if not path.exists():
        _LOG.warning(
            "vehreg_state_costs.json not found at %s; "
            "vehreg will fall back to $0 until the data file is committed.",
            path,
        )
        return {}, 0.0
    try:
        with open(path) as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("vehreg_state_costs cache unreadable (%r)", exc)
        return {}, 0.0

    raw = cache.get("state_per_vehicle", {})
    state_costs = {str(k).upper(): float(v) for k, v in raw.items()}
    national_mean = float(cache.get("national_mean_per_vehicle", 0.0))
    return state_costs, national_mean


def lookup_state_cost(
    puma_code: str,
    state_costs: dict[str, float],
    national_mean: float,
) -> float:
    """Look up annual vehreg per vehicle for the user's PUMA.

    PUMAs are encoded ``ST_NNNNN`` so the state prefix is the first two
    characters. Falls back to ``national_mean`` (which may itself be
    0.0 if the artifact was missing) when the state is absent.
    """
    state = puma_code[:2].upper()
    return float(state_costs.get(state, national_mean))
