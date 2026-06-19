"""State employee-side payroll/disability/paid-leave tax — the annual registry loader.

The rate VALUES live in ``pipeline/artifacts/state_payroll_tax.json`` (the
hand-curated annual artifact, built by ``scripts/build_state_payroll_tax.py``
from official state agency rate notices; registered as ``state_payroll_tax_rates``
in ``agent-artifacts/registry/data_sources.json``). This module is the typed
access layer; it holds NO rate values of its own (single source of truth is the
JSON), mirroring the ``statutory_limits.py`` precedent.

These are the EMPLOYEE-side mandatory contributions absent from the federal FICA
line: state disability (CASDI/SDI/TDI), paid family & medical leave (PFL/PFML),
WA Cares, and employee-side UI/workforce contributions where they exist.
Employer-side taxes are excluded (the ``nj_newark`` employer-payroll exclusion
precedent in ``local_tax_rates.json``). The structurally important field is
``wage_cap = null`` → UNCAPPED (CA CASDI since SB-951, 2024): a large
high-earner line the flat model entirely missed.

Cadence: ANNUAL — most states publish the coming year's rate in ~Oct–Dec.
Refresh: re-run the builder, bump ``_meta.source_year``. Missing/unreadable
artifact → ``load_state_payroll_tax`` returns ``None`` and the payroll line is
$0 everywhere (clean no-op — same philosophy as the local-tax artifact).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class StatePayrollLine:
    """One employee-side payroll/disability line for a state.

    ``wage_cap``: the wage base the ``rate`` applies to; ``None`` = UNCAPPED
    (applies to every dollar — CA CASDI). ``annual_max``: a hard ceiling on the
    annual dollar contribution (some programs cap the $ amount, e.g. NY SDI's
    ~$0.60/week); ``None`` = no annual ceiling.
    """

    name: str
    rate: float
    wage_cap: float | None = None
    annual_max: float | None = None

    def amount(self, gross_income: float) -> float:
        base = gross_income if self.wage_cap is None else min(gross_income, self.wage_cap)
        tax = self.rate * base
        if self.annual_max is not None:
            tax = min(tax, self.annual_max)
        return max(0.0, tax)


@dataclass(frozen=True)
class StatePayrollTable:
    """Parsed ``state_payroll_tax.json`` — per-state employee-side lines."""

    by_state: dict[str, tuple[StatePayrollLine, ...]]
    tax_year: int

    def amount_for(self, gross_income: float, state: str) -> float:
        """Total employee-side state payroll tax for ``state`` (0.0 if none)."""
        if gross_income <= 0:
            return 0.0
        lines = self.by_state.get(state.upper())
        if not lines:
            return 0.0
        return sum(line.amount(gross_income) for line in lines)


def _default_artifacts_path() -> str:
    env = os.environ.get("ARTIFACTS_PATH")
    if env:
        return str(Path(env).resolve())
    return str(
        (Path(__file__).resolve().parents[2] / "pipeline" / "artifacts").resolve()
    )


def _build_line(raw: dict) -> StatePayrollLine:
    cap = raw.get("wage_cap")
    amax = raw.get("annual_max")
    return StatePayrollLine(
        name=str(raw["name"]),
        rate=float(raw["rate"]),
        wage_cap=None if cap is None else float(cap),
        annual_max=None if amax is None else float(amax),
    )


@lru_cache(maxsize=None)
def _load_cached(resolved_path: str) -> StatePayrollTable | None:
    path = Path(resolved_path) / "state_payroll_tax.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    try:
        tax_year = int(
            data.get("tax_year") or data.get("_meta", {}).get("tax_year", 0)
        )
        by_state: dict[str, tuple[StatePayrollLine, ...]] = {
            state.upper(): tuple(_build_line(ln) for ln in entry.get("lines", []))
            for state, entry in data["by_state"].items()
        }
        return StatePayrollTable(by_state=by_state, tax_year=tax_year)
    except (KeyError, TypeError, ValueError):
        return None


def load_state_payroll_tax(artifacts_path: str | None = None) -> StatePayrollTable | None:
    """Load (and cache) the state employee-payroll table.

    Returns ``None`` on missing/unreadable/malformed artifact — callers treat a
    ``None`` table (or an absent state) as $0 (clean no-op). lru_cached on the
    resolved artifacts directory path, mirroring ``models.tax.local``.
    """
    resolved = (
        str(Path(artifacts_path).resolve()) if artifacts_path else _default_artifacts_path()
    )
    return _load_cached(resolved)


def _reset_cache_for_tests() -> None:
    """Clear the lru_cache; call between artifact swaps in tests."""
    _load_cached.cache_clear()
