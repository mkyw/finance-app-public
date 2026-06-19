"""EIA SEDS residential energy *consumption* -> state climate-baseline scalars.

The state-level lane of the utility geographic climate factor (see
``models/matching/geographic_drivers.py`` and
``agent-artifacts/investigations/utilities_diagnosis.md``). The fusion model's
only geographic predictors are region/division, so cohort utility dollars are
flat across climates (Phoenix elec ~= Chicago ~= NYC despite Phoenix's AC load).
This scalar re-injects the measured state-level signal.

**Consumption, not expenditure** — we pull residential *physical consumption*
(billion Btu) per household, normalized state/national. Expenditure would embed
regional price and double-count the RPP ``utilities`` price scalar, which stays
as the separate price lane. Consumption is orthogonal to price.

Normalization: per-household consumption ratio
``(state_consumption / state_hh) / (national_consumption / national_hh)`` where
``hh`` is the housing-unit weight (sum of synthetic-population weights by state).
The national mean is housing-unit-weighted to 1.0, so the factor injects regional
*spread* without shifting the national total.

Categories: ``elec`` <- electricity (ESRCB), ``ngas`` <- natural gas (NGRCB),
``ofuel`` <- distillate + kerosene heating fuel (DFRCB + KSRCB). ``watrsh`` /
``intphn`` are not climate-driven and get no EIA scalar.

Source: EIA SEDS public bulk CSV ``use_all_btu.csv`` (key-free, annual). All EIA
network access lives in :func:`fetch_eia_utility_scalars`, called only by the
offline refresh (``scripts/refresh_cpi.py``). The runtime path
(:func:`load_eia_utility_scalars`) is **file-read only and never fetches** — the
same corrected pattern the gas loader now uses.

Artifact contract (``pipeline/artifacts/eia_utility_scalars.json``):
    generated_at, source, source_period
    national_per_hh_billion_btu : {"elec": .., "ngas": .., "ofuel": ..}
    scalars : {"elec": {"AZ": 1.20, ...}, "ngas": {...}, "ofuel": {...}}
"""
from __future__ import annotations

import json
import logging
import ssl
import urllib.request
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

_LOG = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH: str = "pipeline/artifacts/eia_utility_scalars.json"
_SEDS_BULK_URL: str = "https://www.eia.gov/state/seds/sep_use/total/csv/use_all_btu.csv"

# CEX utility category -> SEDS residential-consumption MSN codes (summed).
_RESIDENTIAL_MSN: dict[str, list[str]] = {
    "elec": ["ESRCB"],            # electricity sales to ultimate customers, residential
    "ngas": ["NGRCB"],            # natural gas delivered to residential
    "ofuel": ["DFRCB", "KSRCB"],  # distillate fuel oil + kerosene, residential
}

_VALID_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
    "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)


def _ssl_context() -> ssl.SSLContext:
    """SSL context that works in environments without a system CA bundle."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - certifi expected present
        return ssl.create_default_context()


def _state_household_weights(synth_pop_path: str | Path) -> dict[str, float]:
    """Sum of synthetic-population weights by state (the housing-unit denominator)."""
    import pyarrow.parquet as pq

    out: dict[str, float] = {}
    for f in glob(str(Path(synth_pop_path) / "puma_code=*/*.parquet")):
        state = Path(f).parent.name.split("=", 1)[1].split("_", 1)[0]
        w = pq.read_table(f, columns=["weight"]).column("weight").to_numpy().sum()
        out[state] = out.get(state, 0.0) + float(w)
    return out


def fetch_eia_utility_scalars(
    synth_pop_path: str | Path,
    cache_path: str = _DEFAULT_CACHE_PATH,
    bulk_url: str = _SEDS_BULK_URL,
) -> dict[str, dict[str, float]]:
    """Download SEDS, compute per-household consumption ratios, write the cache.

    Offline refresh only (the sole EIA-network caller). On any failure, logs and
    returns ``{}`` without overwriting an existing cache (preserves last valid).
    """
    import io

    import pandas as pd

    try:
        with urllib.request.urlopen(bulk_url, timeout=60, context=_ssl_context()) as resp:
            raw = resp.read()
    except Exception as exc:  # network / SSL
        _LOG.warning("EIA SEDS fetch failed (%r); preserving last valid cache.", exc)
        return {}

    df = pd.read_csv(io.BytesIO(raw))
    year_cols = [c for c in df.columns if str(c).isdigit()]
    latest = max(year_cols, key=int)

    hh = _state_household_weights(synth_pop_path)
    states = sorted(s for s in _VALID_STATES if s in hh)

    def consumption(msn_codes: list[str]) -> dict[str, float]:
        sub = df[df["MSN"].isin(msn_codes)]
        agg = sub.groupby("State")[latest].sum()
        return {s: float(agg.get(s, 0.0)) for s in states}

    scalars: dict[str, dict[str, float]] = {}
    national_per_hh: dict[str, float] = {}
    total_hh = sum(hh[s] for s in states)
    for cat, codes in _RESIDENTIAL_MSN.items():
        cons = consumption(codes)
        natl = sum(cons[s] for s in states) / total_hh
        national_per_hh[cat] = natl
        scalars[cat] = {
            s: round((cons[s] / hh[s]) / natl, 4) if natl > 0 and hh[s] > 0 else 1.0
            for s in states
        }

    cache = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source": "EIA SEDS use_all_btu.csv (residential consumption, billion Btu)",
        "source_period": str(latest),
        "national_per_hh_billion_btu": {k: round(v, 4) for k, v in national_per_hh.items()},
        "scalars": scalars,
    }
    path = Path(cache_path)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    return scalars


def load_eia_utility_scalars(
    cache_path: str = _DEFAULT_CACHE_PATH,
) -> dict[str, dict[str, float]]:
    """Runtime loader. **File-read only — never fetches.**

    Returns ``{category: {state: ratio}}`` from the cache (stale or fresh). On a
    missing/unreadable cache returns ``{}``; callers default to factor 1.0 per
    state, so the climate correction silently no-ops rather than erroring. All EIA
    network access is confined to :func:`fetch_eia_utility_scalars` (offline
    refresh), so a profile request never makes a blocking HTTP call.
    """
    path = Path(cache_path)
    try:
        with open(path) as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("eia_utility_scalars cache unreadable (%r); no climate baseline.", exc)
        return {}
    scalars = cache.get("scalars", {})
    return scalars if isinstance(scalars, dict) else {}
