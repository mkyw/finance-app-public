"""EIA state-level retail gasoline prices -> spatial scalars for gas.

Why a separate module (not rolled into ``rpp_scaler``): BEA publishes
RPP sub-indexes at the "goods" level, which averages gasoline with
groceries, apparel, electronics and other goods. Gas has a much larger
state-to-state spatial spread than the "goods" average — e.g. CA gas
prices typically run 30-45% above the US average while CA "goods" RPP
runs 5-10% above. For the ``gas`` CEX category specifically we want the
gas-specific scalar.

EIA's "Weekly U.S. Retail Gasoline Prices" dataset (PET, product EPMR,
motor gasoline all formulations, frequency weekly) covers a national
average (``duoarea=NUS``) and a small set of states (CA, CO, FL, MA,
MN, NY, OH, TX, WA) plus PADD regional aggregates. We only persist
scalars for geographies we actually resolve (the state codes above).
States without a weekly EIA series are simply absent from the scalar
dict; the caller in ``rpp_scaler.get_rpp_scalar`` falls back to 1.0
when a state isn't in the dict, so uncovered PUMAs just get no
gas-specific spatial adjustment (the household-level BEA correction
still applies).

Artifact contract (``pipeline/artifacts/eia_gas_scalars.json``):

    {
      "generated_at": "2026-04-19T...",
      "source_period": "2026-04-14",
      "national_price": 3.41,
      "state_prices":   {"CA": 4.81, "NY": 3.57, ...},
      "scalars":        {"CA": 1.41, "NY": 1.05, ...}
    }

API key: EIA's v2 API requires a registered key passed as
``api_key=...`` query param. Free registration at
https://www.eia.gov/opendata/register.php. Read from env var
``EIA_API_KEY`` at fetch time; if missing we don't raise — we log,
and if there's a previous cache we serve it; otherwise we return an
empty dict and the gas scalar silently becomes 1.0 for every state.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

_LOG = logging.getLogger(__name__)

_EIA_API_BASE: str = "https://api.eia.gov/v2"
_DEFAULT_CACHE_PATH: str = "pipeline/artifacts/eia_gas_scalars.json"
_FETCH_TIMEOUT_SECONDS: float = 15.0

# EIA duoarea codes for the states EIA covers with weekly retail
# gasoline prices. The v2 API uses these codes in the ``duoarea`` facet.
# Keys are two-letter US postal codes; values are EIA's duoarea codes.
_EIA_STATE_DUOAREAS: dict[str, str] = {
    "CA": "SCA",
    "CO": "SCO",
    "FL": "SFL",
    "MA": "SMA",
    "MN": "SMN",
    "NY": "SNY",
    "OH": "SOH",
    "TX": "STX",
    "WA": "SWA",
}
_EIA_NATIONAL_DUOAREA: str = "NUS"

# EIA product code for "total gasoline, all grades, all formulations".
_EIA_PRODUCT_CODE: str = "EPMR"


def _fetch_latest_price(
    api_key: str,
    duoarea: str,
) -> tuple[float | None, str | None]:
    """Fetch the most recent weekly retail gasoline price for ``duoarea``.

    Returns ``(price, period)`` on success, ``(None, None)`` on any
    failure (network, missing data, bad response). We intentionally
    don't raise — the caller collects whatever the API returned and
    computes scalars only for duoareas that actually answered.
    """
    qs = urllib.parse.urlencode(
        [
            ("api_key", api_key),
            ("frequency", "weekly"),
            ("data[0]", "value"),
            ("facets[duoarea][]", duoarea),
            ("facets[product][]", _EIA_PRODUCT_CODE),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "desc"),
            ("length", "1"),
        ]
    )
    url = f"{_EIA_API_BASE}/petroleum/pri/gnd/data/?{qs}"
    try:
        with urllib.request.urlopen(
            url, timeout=_FETCH_TIMEOUT_SECONDS, context=_SSL_CONTEXT
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        _LOG.warning("EIA fetch failed for duoarea=%s (%r)", duoarea, exc)
        return None, None

    rows = body.get("response", {}).get("data", [])
    if not rows:
        return None, None
    try:
        return float(rows[0]["value"]), str(rows[0]["period"])
    except (KeyError, TypeError, ValueError):
        return None, None


def fetch_eia_gas_scalars(
    api_key: str,
    cache_path: str = _DEFAULT_CACHE_PATH,
) -> dict[str, float]:
    """Fetch current weekly retail gasoline price ratios by state.

    Returns a dict ``{state_code: scalar}`` where scalar is
    ``state_price / national_price`` computed from the SAME fetch
    snapshot — no hardcoded constants. Writes the full national +
    per-state price set and the scalars to ``cache_path``.

    If the national fetch fails we can't compute ratios at all and
    we return an empty dict without writing the cache (so the last
    valid cache is preserved).
    """
    national_price, national_period = _fetch_latest_price(api_key, _EIA_NATIONAL_DUOAREA)
    if national_price is None or national_price <= 0:
        _LOG.warning(
            "EIA national gasoline price unavailable; cannot compute "
            "spatial scalars. Preserving last valid cache."
        )
        return {}

    state_prices: dict[str, float] = {}
    for state, duoarea in _EIA_STATE_DUOAREAS.items():
        price, _period = _fetch_latest_price(api_key, duoarea)
        if price is not None and price > 0:
            state_prices[state] = price

    scalars = {
        state: price / national_price
        for state, price in state_prices.items()
    }

    cache = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_period": national_period,
        "national_price": national_price,
        "state_prices": state_prices,
        "scalars": scalars,
    }
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    return scalars


def _read_cache(path: Path) -> dict[str, float]:
    try:
        with open(path) as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("eia_gas_scalars cache unreadable (%r)", exc)
        return {}
    scalars = cache.get("scalars", {})
    return {str(k): float(v) for k, v in scalars.items()}


def load_eia_gas_scalars(
    cache_path: str = _DEFAULT_CACHE_PATH,
    api_key: str | None = None,
) -> dict[str, float]:
    """Load EIA state-level gas scalars. **File-read only — never fetches.**

    Runtime path (called per request inside ``match_household``): reads the cache
    (stale or fresh) and returns it. It never makes a network call, so a profile
    submission can never incur a blocking EIA HTTP request — fetching is confined
    to the offline refresh (``scripts/refresh_cpi.py`` -> :func:`fetch_eia_gas_scalars`),
    which keeps the cache current.

    A missing/unreadable cache returns an empty dict; callers in
    ``rpp_scaler.get_rpp_scalar`` default to 1.0 per state when a state is absent,
    so the gas category silently loses its EIA spatial correction rather than
    erroring. (``api_key`` is accepted for signature stability but unused — fetch
    no longer happens at load time.)
    """
    return _read_cache(Path(cache_path))
