#!/usr/bin/env python3.11
"""Build ``pipeline/artifacts/state_payroll_tax.json`` — employee-side state
payroll/disability/paid-leave rates.

The EMPLOYEE share of state disability (CASDI/SDI/TDI), paid family & medical
leave (PFL/PFML), WA Cares, and employee UI/workforce contributions — the
mandatory withholding the federal FICA line does not cover. Employer-side taxes
are excluded (the ``nj_newark`` employer-payroll exclusion precedent).

Curated from official state-agency rate notices (one ``source_url`` per state in
``_SOURCES``); annual cadence. Mirrors ``scripts/build_local_tax_rates.py``'s
curated-table + ``_meta``-provenance pattern. Re-run when a state publishes its
coming-year rate; bump ``_AS_OF``/``_SOURCE_YEAR``.

Line schema: ``{name, rate, wage_cap, annual_max}`` where ``wage_cap = null``
means UNCAPPED (CA CASDI since SB-951) and ``annual_max = null`` means no annual
dollar ceiling. Consumed by ``shared/constants/state_payroll_tax.py``.

Usage:  python3.11 scripts/build_state_payroll_tax.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_TAX_YEAR: int = 2026
_SOURCE_YEAR: int = 2026
_AS_OF: str = "2026-06-13"

# --------------------------------------------------------------------------- #
# Curated employee-side rates, verified against the official sources below.
# wage_cap=None → uncapped; annual_max=None → no annual-$ ceiling.
# POPULATED FROM THE STAGE-2 OFFICIAL-SOURCE RESEARCH SWEEP.
# --------------------------------------------------------------------------- #
# 2026 SS wage base (OASDI max taxable earnings; taxcalc SS_Earnings_c) — the
# wage cap for the PFML programs that cap at the federal SS base. MUST match
# models/tax/calculator.py::_SS_WAGE_BASE_2026 (the live FICA path); on a future
# SS-base bump update BOTH or the PFML wage caps here will lag the FICA cap.
_SS_WAGE_BASE_2026: float = 184_500.0

_CURATED: dict[str, list[dict]] = {
    # CA SDI incl. PFL — UNCAPPED since SB-951 (no wage ceiling, no $ max).
    "CA": [{"name": "casdi", "rate": 0.013, "wage_cap": None, "annual_max": None}],
    # NY: statutory DBL (~$0.60/wk → ~$31.20/yr) + PFL (annual-$ max).
    "NY": [
        {"name": "sdi", "rate": 0.005, "wage_cap": 6_240.0, "annual_max": 31.20},
        {"name": "pfl", "rate": 0.00432, "wage_cap": 95_348.76, "annual_max": 411.91},
    ],
    # NJ: TDI + FLI on the state base $171,100; employee UI/WF/SWF on $44,800.
    "NJ": [
        {"name": "tdi", "rate": 0.0019, "wage_cap": 171_100.0, "annual_max": 325.09},
        {"name": "fli", "rate": 0.0023, "wage_cap": 171_100.0, "annual_max": 393.53},
        {"name": "employee_ui", "rate": 0.00425, "wage_cap": 44_800.0, "annual_max": 190.40},
    ],
    # HI TDI: 0.5% with a $7.50/wk max → ~$390/yr.
    "HI": [{"name": "tdi", "rate": 0.005, "wage_cap": 78_010.92, "annual_max": 390.00}],
    # RI TDI/TCI: 100% employee-funded, base rose to $100k for 2026.
    "RI": [{"name": "tdi", "rate": 0.011, "wage_cap": 100_000.0, "annual_max": 1_100.00}],
    # WA: PFML employee share (71.43% of 1.13%) on SS base + WA Cares (uncapped).
    "WA": [
        {"name": "pfml", "rate": 0.008072, "wage_cap": _SS_WAGE_BASE_2026, "annual_max": None},
        {"name": "wa_cares", "rate": 0.0058, "wage_cap": None, "annual_max": None},
    ],
    # PFML programs capped at the federal SS base, employee share:
    "OR": [{"name": "paid_leave", "rate": 0.006, "wage_cap": _SS_WAGE_BASE_2026, "annual_max": None}],
    "CO": [{"name": "famli", "rate": 0.0044, "wage_cap": _SS_WAGE_BASE_2026, "annual_max": None}],
    "CT": [{"name": "pfml", "rate": 0.005, "wage_cap": _SS_WAGE_BASE_2026, "annual_max": None}],
    "MA": [{"name": "pfml", "rate": 0.0046, "wage_cap": _SS_WAGE_BASE_2026, "annual_max": None}],
    "ME": [{"name": "pfml", "rate": 0.005, "wage_cap": _SS_WAGE_BASE_2026, "annual_max": None}],
    # MN Paid Leave — NEW, contributions start 2026-01-01 (cap stated $185,000).
    "MN": [{"name": "paid_leave", "rate": 0.0044, "wage_cap": 185_000.0, "annual_max": None}],
    "DE": [{"name": "paid_leave", "rate": 0.004, "wage_cap": _SS_WAGE_BASE_2026, "annual_max": None}],
    # PA employee UC withholding — UNCAPPED (the $10k base is employer-only).
    "PA": [{"name": "employee_ui", "rate": 0.0007, "wage_cap": None, "annual_max": None}],
    # AK employee UI (one of only three states with an employee UI share).
    "AK": [{"name": "employee_ui", "rate": 0.005, "wage_cap": 54_200.0, "annual_max": 271.00}],
}

# Official rate-notice source per state (employee-side; verified Stage-2 sweep).
_SOURCES: dict[str, str] = {
    "CA": "https://edd.ca.gov/en/disability/Contribution_Rates_and_Benefit_Amounts/",
    "NY": "https://www.wcb.ny.gov/content/main/PressRe/paid-family-leave-2026.jsp",
    "NJ": "https://www.nj.gov/labor/lwdhome/press/2025/20251229_newbenefitrates2026.shtml",
    "HI": "https://labor.hawaii.gov/dcd/files/2025/12/2026-Maximum-Weekly-Wage-Base.pdf",
    "RI": "https://dlt.ri.gov/press-releases/2026-tax-rates-unemployment-insurance-and-temporary-disability-insurance",
    "WA": "https://esd.wa.gov/about-us/news-release/2025/paid-family-medical-leave-premium-rate-increases-113-2026",
    "OR": "https://paidleave.oregon.gov/employers/contributions-calculator.html",
    "CO": "https://famli.colorado.gov/employers",
    "CT": "https://www.ctpaidleave.org/how-ct-paid-leave-works/contributions",
    "MA": "https://www.mass.gov/info-details/paid-family-and-medical-leave-employer-contribution-rates-and-calculator",
    "ME": "https://www.maine.gov/paidleave/",
    "MN": "https://pl.mn.gov/resources/calculators/premium-rate-and-contributions",
    "DE": "https://labor.delaware.gov/delaware-paid-leave/employers/",
    "PA": "https://www.pa.gov/agencies/dli/resources/for-employers-and-educators/how-to-file/uc-tax/yearly-tax-highlights",
    "AK": "https://labor.alaska.gov/estax/2026-experience-rates.html",
}


def _artifacts_dir() -> Path:
    env = os.environ.get("ARTIFACTS_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "pipeline" / "artifacts"


def build() -> dict:
    if not _CURATED:
        raise SystemExit(
            "build_state_payroll_tax: _CURATED is empty — populate it from the "
            "official-source research sweep before generating the artifact."
        )
    return {
        "_meta": {
            "tax_year": _TAX_YEAR,
            "source_year": _SOURCE_YEAR,
            "as_of": _AS_OF,
            "description": (
                "Employee-side state payroll/disability/paid-leave rates "
                "(CASDI/SDI/TDI, PFL/PFML, WA Cares, employee UI). Employer-side "
                "excluded. wage_cap=null is uncapped; annual_max=null is no $ ceiling."
            ),
            "sources": _SOURCES,
            "notes": (
                "Curated from official state-agency rate notices. Missing artifact "
                "or absent state → $0 (clean no-op). Refresh annually."
            ),
        },
        "tax_year": _TAX_YEAR,
        "by_state": {state: {"lines": lines} for state, lines in _CURATED.items()},
    }


def main() -> None:
    artifact = build()
    out = _artifacts_dir() / "state_payroll_tax.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    n_states = len(artifact["by_state"])
    n_lines = sum(len(v["lines"]) for v in artifact["by_state"].values())
    print(f"Wrote {out} — {n_states} states, {n_lines} employee-side lines ({_TAX_YEAR}).")


if __name__ == "__main__":
    main()
