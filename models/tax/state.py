"""State income tax — per-state brackets / flat + standard deduction + exemptions.

The rigorous schedule lives in ``pipeline/artifacts/state_tax_rates.json``
(brackets-or-flat + standard deduction + personal/dependent exemptions per
state, built by ``scripts/build_state_tax_rates.py`` from official DOR
schedules) and is applied to STATE TAXABLE INCOME — gross net of the conforming
committed-baseline pre-tax wedge, the state standard deduction, and exemptions.
Reuses the shared bracket arithmetic (``models.tax._brackets``), the same
machinery the local/municipal layer uses (``MUNICIPAL-TAX-PRECISION``).

When the artifact is absent (or a state is missing from it), ``state_tax`` falls
back to the flat ``STATE_TAX_RATES`` effective-rate dict below applied to gross —
byte-identical to the pre-Stage-4 behavior. The nine no-wage-income-tax states
(AK, FL, NV, NH, SD, TN, TX, WA, WY) return 0 either way.

State income tax does NOT depend on federal AGI, so it is computed on the
committed-baseline wedge only (NOT the iterating waterfall top-ups) and sits
OUTSIDE the pre-tax fixed-point loop — only the federal-dependent part iterates.
State EITC (a fraction of the federal EITC) is carried on the rule but applied
in the federal-component pass (Stage 6), since it needs the federal EITC.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from models.tax._brackets import apply_brackets as _apply_brackets
from models.tax._brackets import parse_brackets as _parse_brackets

# Flat effective-rate approximation per state — the FALLBACK when the per-state
# schedule artifact is absent. Sources: Tax Foundation 2024 + flat-tax statutes.
STATE_TAX_RATES: dict[str, float] = {
    "AL": 0.040, "AK": 0.000, "AZ": 0.025, "AR": 0.040, "CA": 0.055,
    "CO": 0.044, "CT": 0.050, "DC": 0.060, "DE": 0.055, "FL": 0.000,
    "GA": 0.048, "HI": 0.060, "ID": 0.058, "IL": 0.0495, "IN": 0.0315,
    "IA": 0.044, "KS": 0.052, "KY": 0.045, "LA": 0.035, "ME": 0.058,
    "MD": 0.0475, "MA": 0.050, "MI": 0.0425, "MN": 0.068, "MS": 0.047,
    "MO": 0.0495, "MT": 0.059, "NE": 0.0584, "NV": 0.000, "NH": 0.000,
    "NJ": 0.045, "NM": 0.049, "NY": 0.055, "NC": 0.045, "ND": 0.0195,
    "OH": 0.0285, "OK": 0.0475, "OR": 0.075, "PA": 0.0307, "RI": 0.0475,
    "SC": 0.055, "SD": 0.000, "TN": 0.000, "TX": 0.000, "UT": 0.0465,
    "VT": 0.060, "VA": 0.055, "WA": 0.000, "WV": 0.049, "WI": 0.053,
    "WY": 0.000,
}


@dataclass(frozen=True)
class StateTaxRule:
    """One state's 2026 income-tax rule (from ``state_tax_rates.json``).

    ``kind`` is ``"flat"``, ``"brackets"``, or ``"none"`` (no wage income tax).
    Both flat and brackets apply to STATE TAXABLE INCOME. ``eitc_pct`` (state
    EITC as a fraction of the federal EITC) is carried for the state-credit pass
    (Stage 6); not applied here.
    """

    name: str
    kind: str
    rate: float = 0.0
    brackets_by_filing_status: dict[str, tuple[tuple[float, float], ...]] | None = None
    standard_deduction_by_filing_status: dict[str, float] = field(default_factory=dict)
    personal_exemption: float = 0.0     # per filer (× n_filers)
    dependent_exemption: float = 0.0    # per dependent
    eitc_pct: float = 0.0               # state EITC as a fraction of federal (Stage 6)


@dataclass(frozen=True)
class StateTaxTable:
    """Parsed ``state_tax_rates.json`` — per-state rules + vintage."""

    by_state: dict[str, StateTaxRule]
    tax_year: int


def _default_artifacts_path() -> str:
    env = os.environ.get("ARTIFACTS_PATH")
    if env:
        return str(Path(env).resolve())
    return str((Path(__file__).resolve().parents[2] / "pipeline" / "artifacts").resolve())


def _build_state_rule(code: str, raw: dict) -> StateTaxRule:
    kind = str(raw["kind"])
    brackets = (
        _parse_brackets(raw["brackets_by_filing_status"])
        if kind == "brackets" and raw.get("brackets_by_filing_status")
        else None
    )
    def _f(v: object, default: float = 0.0) -> float:
        # Defensive: a non-numeric field (e.g. a state_eitc_pct that's a
        # structured dict rather than a simple %) coerces to the default rather
        # than dropping the WHOLE table to the flat fallback.
        return float(v) if isinstance(v, (int, float)) else default

    return StateTaxRule(
        name=str(raw.get("name", code)),
        kind=kind,
        rate=_f(raw.get("rate")),
        brackets_by_filing_status=brackets,
        standard_deduction_by_filing_status={
            k: _f(v) for k, v in raw.get("standard_deduction_by_filing_status", {}).items()
        },
        personal_exemption=_f(raw.get("personal_exemption")),
        dependent_exemption=_f(raw.get("dependent_exemption")),
        eitc_pct=_f(raw.get("state_eitc_pct")),
    )


@lru_cache(maxsize=None)
def _load_cached(resolved_path: str) -> StateTaxTable | None:
    path = Path(resolved_path) / "state_tax_rates.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    try:
        tax_year = int(data.get("tax_year") or data.get("_meta", {}).get("tax_year", 0))
        by_state = {
            code.upper(): _build_state_rule(code, raw)
            for code, raw in data["by_state"].items()
        }
        return StateTaxTable(by_state=by_state, tax_year=tax_year)
    except (KeyError, TypeError, ValueError):
        return None


def load_state_tax_table(artifacts_path: str | None = None) -> StateTaxTable | None:
    """Load (and cache) the per-state schedule; ``None`` when the artifact is
    absent/malformed (callers fall back to the flat ``STATE_TAX_RATES`` dict).
    lru_cached on the resolved artifacts directory, mirroring ``models.tax.local``."""
    resolved = (
        str(Path(artifacts_path).resolve()) if artifacts_path else _default_artifacts_path()
    )
    return _load_cached(resolved)


def _reset_cache_for_tests() -> None:
    """Clear the lru_cache; call between artifact swaps in tests."""
    _load_cached.cache_clear()


_N_FILERS: dict[str, int] = {"married_filing_jointly": 2}


def _apply_state_rule(
    gross_income: float,
    rule: StateTaxRule,
    filing_status: str,
    num_dependents: int,
    pretax_excludable: float,
) -> float:
    """State income tax = the rule applied to state taxable income.

    ``state_taxable = max(0, gross − conforming pre-tax exclusions − standard
    deduction(filing) − personal_exemption×n_filers − dependent_exemption×deps)``.
    State EITC / other credits are applied by the caller (Stage 6); not here.
    """
    if rule.kind == "none":
        return 0.0
    std = rule.standard_deduction_by_filing_status.get(
        filing_status, rule.standard_deduction_by_filing_status.get("single", 0.0)
    )
    n_filers = _N_FILERS.get(filing_status, 1)
    exemptions = (
        rule.personal_exemption * n_filers
        + rule.dependent_exemption * max(0, num_dependents)
    )
    state_taxable = max(0.0, gross_income - pretax_excludable - std - exemptions)
    if rule.kind == "brackets" and rule.brackets_by_filing_status:
        return max(
            0.0, _apply_brackets(state_taxable, rule.brackets_by_filing_status, filing_status)
        )
    return max(0.0, state_taxable * rule.rate)


def state_from_puma(puma_code: str) -> str:
    """Extract the 2-letter state abbreviation from a ``STATE_PUMA`` code.

    Example: ``"CA_03761"`` -> ``"CA"``. See ``export_population.R`` for
    how these codes are constructed.

    Raises:
        ValueError: if ``puma_code`` is not in the expected format.
    """
    if "_" not in puma_code:
        raise ValueError(f"malformed puma_code (missing '_'): {puma_code!r}")
    state = puma_code.split("_", 1)[0]
    if len(state) != 2:
        raise ValueError(f"state prefix must be 2 chars: {puma_code!r}")
    return state


def state_tax(
    gross_income: float,
    puma_code: str,
    filing_status: str = "single",
    num_dependents: int = 0,
    pretax_excludable: float = 0.0,
    artifacts_path: str | None = None,
) -> float:
    """State income tax for the state encoded in ``puma_code``.

    Uses the rigorous per-state schedule (``state_tax_rates.json``: brackets/flat
    + standard deduction + exemptions, applied to gross net of the conforming
    committed-baseline pre-tax wedge ``pretax_excludable``). Falls back to the
    flat ``STATE_TAX_RATES`` effective-rate dict (× gross) when the artifact is
    absent or the state is not in it — byte-identical to the pre-Stage-4
    behavior. Returns 0 for non-positive income or a no-income-tax state.

    ``pretax_excludable`` is the COMMITTED-BASELINE income-tax exclusion only
    (not the iterating waterfall top-ups), so state tax is constant across the
    pre-tax fixed-point iterations — it sits outside the loop.
    """
    if gross_income <= 0:
        return 0.0
    state = state_from_puma(puma_code)
    table = load_state_tax_table(artifacts_path)
    if table is not None and state in table.by_state:
        return _apply_state_rule(
            gross_income, table.by_state[state], filing_status, num_dependents, pretax_excludable
        )
    rate = STATE_TAX_RATES.get(state)
    if rate is None:
        raise KeyError(f"no tax rate for state: {state!r}")
    return gross_income * rate


def state_eitc_pct(puma_code: str, artifacts_path: str | None = None) -> float:
    """The state's EITC as a fraction of the federal EITC, for the state-credit
    pass in ``models.tax.calculator.compute_tax`` (Stage 6; it needs the federal
    EITC, so it is applied there, not in ``state_tax``). Returns 0 for a no-tax
    state, a state with no EITC, a structured-EITC state carried as null
    (CA/MN/WI/DE — deferred), or a missing artifact."""
    try:
        state = state_from_puma(puma_code)
    except ValueError:
        return 0.0
    table = load_state_tax_table(artifacts_path)
    if table is None or state not in table.by_state:
        return 0.0
    return max(0.0, table.by_state[state].eitc_pct)


def take_home(
    gross_income: float,
    puma_code: str,
    filing_status: str = "single",
    place_fips: str = "",
    county_fips: str = "",
    num_dependents: int = 0,
    *,
    detail=None,
) -> float:
    """Annual take-home after federal + state + local/municipal + FICA.

    Delegates to :func:`models.tax.calculator.compute_tax`, which uses taxcalc
    for federal income tax (2026 law), the per-state schedule (with flat
    fallback) for state, the ``local_tax_rates.json`` artifact for
    local/municipal tax (NYC included), the state employee-payroll layer, and
    the 2026 FICA schedule including Additional Medicare Tax.

    Args:
        gross_income: Annual gross wage income.
        puma_code: ``STATE_PUMA`` string for state + NYC back-compat.
        filing_status: ``"single"``, ``"married_filing_jointly"``, or
            ``"head_of_household"``.
        place_fips: 7-digit Census place FIPS (passed to ``compute_tax``).
        county_fips: 5-digit Census county FIPS (passed to ``compute_tax``).
        num_dependents: Dependents (CTC/EIC + state exemptions).
        detail: Optional :class:`models.tax.calculator.TaxDetail` (pre-tax
            wedge + dependent buckets).
    """
    from models.tax.calculator import compute_tax

    return compute_tax(
        gross_income=gross_income,
        filing_status=filing_status,
        puma_code=puma_code,
        num_dependents=num_dependents,
        place_fips=place_fips,
        county_fips=county_fips,
        detail=detail,
    ).take_home
