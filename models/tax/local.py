"""Local/municipal income tax lookup from ``pipeline/artifacts/local_tax_rates.json``.

Resident rates applied to gross wages; no stacking across jurisdictions;
missing artifact → 0.0 everywhere (clean no-op).

Lookup precedence for a given call:
  1. ``place_fips`` found directly in ``by_place_fips``
  2. ``place_fips`` found in ``place_to_county`` and that county is in
     ``by_county_fips``
  3. ``county_fips`` found directly in ``by_county_fips``
  4. Back-compat: if no FIPS supplied but ``puma_code`` starts with ``NY_04``,
     inject the NYC place FIPS before the lookup
  5. 0.0 (no match)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Bracket arithmetic lives in the shared helper so the state layer composes
# the same functions (MUNICIPAL-TAX-PRECISION reuse). Re-exported under the
# original private names for back-compat with existing call sites/tests.
from models.tax._brackets import apply_brackets as _apply_brackets
from models.tax._brackets import parse_brackets as _parse_brackets

__all__ = [
    "LocalTaxRule",
    "LocalTaxTable",
    "load_local_tax_table",
    "compute_local_tax",
]

# NYC consolidated city place FIPS (New York City, NY).
_NYC_PLACE_FIPS = "3651000"


@dataclass(frozen=True)
class LocalTaxRule:
    """One jurisdiction's rate rule.

    ``kind`` is ``"flat"`` or ``"brackets"``.  For flat rules ``rate`` holds
    the single marginal rate; for bracket rules ``brackets_by_filing_status``
    maps filing-status strings to sorted ``(upper_bound, rate)`` tuples (the
    final upper bound is ``float("inf")``).  ``base`` is the income concept
    the rate is applied to (``"wages"``).
    """

    name: str
    kind: str
    rate: float = 0.0
    brackets_by_filing_status: Optional[dict[str, tuple[tuple[float, float], ...]]] = None
    base: str = "wages"


@dataclass(frozen=True)
class LocalTaxTable:
    """Parsed ``local_tax_rates.json`` artifact."""

    by_place_fips: dict[str, LocalTaxRule]
    by_county_fips: dict[str, LocalTaxRule]
    place_to_county: dict[str, str]
    tax_year: int


# --------------------------------------------------------------------------- #
# Artifact loader (lru_cached on the resolved path string)                     #
# --------------------------------------------------------------------------- #


def _default_artifacts_path() -> str:
    env = os.environ.get("ARTIFACTS_PATH")
    if env:
        return str(Path(env).resolve())
    return str(
        (Path(__file__).resolve().parents[2] / "pipeline" / "artifacts").resolve()
    )


def _build_rule(raw_rule: dict) -> LocalTaxRule:
    kind = str(raw_rule["kind"])
    name = str(raw_rule["name"])
    base = str(raw_rule.get("base", "wages"))
    if kind == "flat":
        return LocalTaxRule(name=name, kind=kind, rate=float(raw_rule["rate"]), base=base)
    if kind == "brackets":
        brackets = _parse_brackets(raw_rule["brackets_by_filing_status"])
        return LocalTaxRule(name=name, kind=kind, brackets_by_filing_status=brackets, base=base)
    raise ValueError(f"unknown kind: {kind!r}")


@lru_cache(maxsize=None)
def _load_cached(resolved_path: str) -> Optional[LocalTaxTable]:
    """Inner cached loader keyed by the resolved artifact directory path."""
    path = Path(resolved_path) / "local_tax_rates.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    try:
        # tax_year lives in _meta per the artifact schema; accept top-level too.
        tax_year = int(
            data.get("tax_year") or data.get("_meta", {}).get("tax_year", 0)
        )
        by_place: dict[str, LocalTaxRule] = {
            fips: _build_rule(raw)
            for fips, raw in data["by_place_fips"].items()
        }
        by_county: dict[str, LocalTaxRule] = {
            fips: _build_rule(raw)
            for fips, raw in data["by_county_fips"].items()
        }
        place_to_county: dict[str, str] = {
            str(k): str(v) for k, v in data["place_to_county"].items()
        }
        return LocalTaxTable(
            by_place_fips=by_place,
            by_county_fips=by_county,
            place_to_county=place_to_county,
            tax_year=tax_year,
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_local_tax_table(artifacts_path: str | None = None) -> Optional[LocalTaxTable]:
    """Load (and cache) the local-tax lookup table.

    Returns ``None`` on missing file / OSError / JSONDecodeError / malformed
    structure — never raises.  The result is lru_cached on the *resolved*
    artifacts directory path so multiple callers with equivalent paths share
    one parse.
    """
    resolved = str(Path(artifacts_path).resolve()) if artifacts_path else _default_artifacts_path()
    return _load_cached(resolved)


def _reset_cache_for_tests() -> None:
    """Clear the lru_cache; call between artifact swaps in tests."""
    _load_cached.cache_clear()


# --------------------------------------------------------------------------- #
# Rule application (progressive-bracket arithmetic lives in _brackets.py)      #
# --------------------------------------------------------------------------- #


def _apply_rule(gross_income: float, rule: LocalTaxRule, filing_status: str) -> float:
    if rule.kind == "flat":
        return gross_income * rule.rate
    if rule.kind == "brackets" and rule.brackets_by_filing_status is not None:
        return _apply_brackets(gross_income, rule.brackets_by_filing_status, filing_status)
    return 0.0


# --------------------------------------------------------------------------- #
# Public entry point                                                            #
# --------------------------------------------------------------------------- #


def compute_local_tax(
    gross_income: float,
    filing_status: str = "single",
    place_fips: str = "",
    county_fips: str = "",
    puma_code: str = "",
    artifacts_path: str | None = None,
) -> float:
    """Compute annual local/municipal income tax for a household.

    Args:
        gross_income: Annual gross wage income.
        filing_status: ``"single"``, ``"married_filing_jointly"``,
            ``"head_of_household"``, etc.  Unknown values fall back to the
            ``"single"`` bracket schedule.
        place_fips: 7-digit place FIPS string (``"3651000"`` = NYC).
        county_fips: 5-digit county FIPS string.
        puma_code: ``STATE_PUMA`` string.  Back-compat: when neither FIPS is
            provided and ``puma_code`` begins with ``"NY_04"`` the NYC place
            FIPS is injected automatically so existing call sites remain
            byte-identical.
        artifacts_path: Override for the artifacts directory; defaults to the
            ``ARTIFACTS_PATH`` env-var / repo-relative path.

    Returns:
        Annual local tax in dollars, or 0.0 when no matching rule exists or
        the artifact is absent.
    """
    if gross_income <= 0:
        return 0.0

    table = load_local_tax_table(artifacts_path)
    if table is None:
        return 0.0

    # Back-compat: inject NYC FIPS for existing callers that pass puma_code only.
    if not place_fips and puma_code.startswith("NY_04"):
        place_fips = _NYC_PLACE_FIPS

    # Precedence 1: direct place match.
    if place_fips and place_fips in table.by_place_fips:
        return _apply_rule(gross_income, table.by_place_fips[place_fips], filing_status)

    # Precedence 2: place → county bridge.
    if place_fips and place_fips in table.place_to_county:
        county = table.place_to_county[place_fips]
        if county in table.by_county_fips:
            return _apply_rule(gross_income, table.by_county_fips[county], filing_status)

    # Precedence 3: direct county match.
    if county_fips and county_fips in table.by_county_fips:
        return _apply_rule(gross_income, table.by_county_fips[county_fips], filing_status)

    return 0.0
