"""Statutory tax-advantaged contribution limits — the annual registry loader.

The limit VALUES live in ``pipeline/artifacts/statutory_limits.json`` (the
hand-curated annual artifact the freshness checker watches — the
``vehreg_state_costs.json`` precedent; registered as ``irs_statutory_limits``
in ``agent-artifacts/registry/data_sources.json``). This module is the typed
access layer consumed by the up-direction savings waterfall
(``models/optimizer/savings_waterfall.py``) — it holds NO limit values of its
own (single source of truth is the JSON).

Cadence: ANNUAL, forward-vintaged — the limits for year Y are announced
~November of Y−1 (401(k)/IRA: the IRS retirement-plan COLA notice, e.g.
Notice 2025-67 for 2026) and ~May of Y−1 (HSA: the Rev. Proc., e.g.
2025-19). Refresh procedure: add the new year's block to the JSON, bump
``_meta.source_year``. The registry entry flags DUE_SOON each November and
OVERDUE when the registry year falls behind the calendar year
(``expected_lag_years: -1`` — see the manifest notes).

Missing/unreadable artifact → ``load_statutory_limits`` returns ``None`` and
the waterfall degrades to the taxable-terminal-only fill (no statutory-vehicle
assertion without limits — the registry's clean-no-op philosophy: remainder
still closes to 0, just without named vehicles).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class StatutoryLimits:
    """One year's statutory limits, with age/household conditioning helpers."""

    year: int
    k401_elective_deferral: float
    k401_catchup_age50: float
    k401_catchup_age60_63: float
    ira_limit: float
    ira_catchup_age50: float
    roth_phaseout_single: tuple[float, float]
    roth_phaseout_mfj: tuple[float, float]
    hsa_self_only: float
    hsa_family: float
    hsa_catchup_age55: float

    def k401_limit(self, age: float) -> float:
        """Elective-deferral limit incl. the age-conditioned catch-up.

        SECURE 2.0 super catch-up applies at ages 60–63 (replacing, not
        stacking on, the 50+ catch-up); the standard catch-up applies 50+.
        """
        if 60.0 <= age <= 63.0:
            return self.k401_elective_deferral + self.k401_catchup_age60_63
        if age >= 50.0:
            return self.k401_elective_deferral + self.k401_catchup_age50
        return self.k401_elective_deferral

    def ira_limit_for(self, age: float) -> float:
        return self.ira_limit + (self.ira_catchup_age50 if age >= 50.0 else 0.0)

    def hsa_limit(self, age: float, household_size: int) -> float:
        """Self-only vs family coverage tier (household_size >= 2 → family,
        mirroring the KFF premium-tier logic in committed_outflows), plus the
        55+ catch-up."""
        base = self.hsa_family if household_size >= 2 else self.hsa_self_only
        return base + (self.hsa_catchup_age55 if age >= 55.0 else 0.0)

    def roth_mechanism(self, magi: float, filing_status: str) -> str:
        """Likely IRA mechanism at this MAGI: ``direct_roth`` below the
        phase-out, ``partial_phaseout`` inside it, ``backdoor_roth`` above it.
        Same dollar limit either way — the mechanism is a framing/label
        concern (descriptive, never advice)."""
        lo, hi = (
            self.roth_phaseout_mfj
            if filing_status in ("married_joint", "mfj")
            else self.roth_phaseout_single
        )
        if magi < lo:
            return "direct_roth"
        if magi <= hi:
            return "partial_phaseout"
        return "backdoor_roth"


def load_statutory_limits(
    artifacts_path: str, year: int | None = None
) -> StatutoryLimits | None:
    """Load the limits for ``year`` (default: the latest year in the registry).

    Returns ``None`` when the artifact is absent/unreadable or the requested
    year is missing — callers degrade to the taxable-terminal-only fill.
    """
    path = Path(artifacts_path) / "statutory_limits.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    by_year = data.get("limits_by_year", {})
    if not by_year:
        return None
    key = str(year) if year is not None else max(by_year, key=int)
    block = by_year.get(key)
    if not isinstance(block, dict):
        return None
    try:
        return StatutoryLimits(
            year=int(key),
            k401_elective_deferral=float(block["k401_elective_deferral"]),
            k401_catchup_age50=float(block["k401_catchup_age50"]),
            k401_catchup_age60_63=float(block["k401_catchup_age60_63"]),
            ira_limit=float(block["ira_limit"]),
            ira_catchup_age50=float(block["ira_catchup_age50"]),
            roth_phaseout_single=(
                float(block["roth_phaseout_single"][0]),
                float(block["roth_phaseout_single"][1]),
            ),
            roth_phaseout_mfj=(
                float(block["roth_phaseout_mfj"][0]),
                float(block["roth_phaseout_mfj"][1]),
            ),
            hsa_self_only=float(block["hsa_self_only"]),
            hsa_family=float(block["hsa_family"]),
            hsa_catchup_age55=float(block["hsa_catchup_age55"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def limits_stale(limits: StatutoryLimits | None, today: date) -> bool:
    """True when the registry's limit year is behind the calendar year (the
    new year's limits were announced the prior November and should be in).
    The registry-level DUE_SOON/OVERDUE machinery is the primary flag
    (``check_data_freshness.py``); this helper is the in-process check for
    tests and any runtime surfacing."""
    return limits is None or limits.year < today.year
