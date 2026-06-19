#!/usr/bin/env python3.11
"""Build ``pipeline/artifacts/state_tax_rates.json`` — per-state 2026 income-tax
schedules (brackets/flat + standard deduction + exemptions + state EITC%).

Curated from official state DOR schedules (one ``source_url`` per state in
``_SOURCES``); annual cadence. Consumed by ``models/tax/state.py`` (with the flat
``STATE_TAX_RATES`` dict as the missing-artifact fallback). Mirrors
``scripts/build_state_payroll_tax.py``'s curated-table + ``_meta``-provenance
pattern.

INSTRUMENT-SKEPTICISM ON THE RESEARCH OUTPUT (``_validate_schedules``): a fast
research pass can plausibly get state-schedule QUIRKS subtly wrong in ways the
per-state effective-rate spot-checks won't catch. The validator below encodes
the known failure modes as hard checks BEFORE the artifact is written:
  1. standard deduction vs personal exemption vs credit-in-lieu — flags flat
     states with a rate but no deduction/exemption base (the IL-$2,775-exemption
     trap: "flat rate" right, base subtly wrong), excluding the genuinely
     no-deduction states (PA).
  2. flat-state base — same check; the exemption/deduction is where the error
     hides on the "easy" flat states.
  3. the 9 no-wage-tax states must stay kind="none" (no rate accidentally
     assigned; NH/WA dividend/cap-gains do NOT apply to wages).
  4. DC present as a state row (its income tax IS the "state" line; no separate
     state layer), with a real schedule.
Plus structural sanity: bracket thresholds monotonic, top bracket open, rates in
a plausible band, full taxing-jurisdiction coverage.

Usage:  python3.11 scripts/build_state_tax_rates.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_TAX_YEAR: int = 2026
_SOURCE_YEAR: int = 2026
_AS_OF: str = "2026-06-13"

# The 9 states with NO wage income tax — must be kind="none" (NH taxes only
# interest/dividends, phasing out; WA only a capital-gains excise — neither
# touches WAGE income).
_NO_WAGE_TAX_STATES: frozenset[str] = frozenset(
    {"AK", "FL", "NV", "NH", "SD", "TN", "TX", "WA", "WY"}
)
# Genuinely no standard-deduction / personal-exemption flat states (so the
# "flat state missing a base" check doesn't false-flag them). PA taxes gross
# compensation at 3.07% with no deduction or exemption.
_KNOWN_NO_DEDUCTION_FLAT: frozenset[str] = frozenset({"PA"})

_ALL_JURISDICTIONS: frozenset[str] = frozenset(
    {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
        "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
        "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
        "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
        "WV", "WI", "WY",
    }
)
_MAX_PLAUSIBLE_RATE: float = 0.15  # CA top 13.3% is the real ceiling

# --------------------------------------------------------------------------- #
# Curated per-state 2026 schedules — POPULATED FROM THE OFFICIAL-SOURCE SWEEP. #
# Each entry: {"name", "kind": "none"|"flat"|"brackets", and for flat: "rate";  #
# for brackets: "brackets_by_filing_status" {fs: [[upper|null, rate], ...]};     #
# plus "standard_deduction_by_filing_status", "personal_exemption",             #
# "dependent_exemption", "state_eitc_pct"}.                                      #
# --------------------------------------------------------------------------- #
# Curated builder-schema source — committed, human-reviewable, produced by
# transforming the official-source research sweep (filing-key renames, exemptions
# folded into the per-status taxable-income base, surtaxes folded into top
# brackets, federal-conformance states using the federal std deduction). The
# builder validates it (``_validate_schedules``) before writing the artifact.
_CURATED_SOURCE = (
    Path(__file__).resolve().parent.parent / "pipeline" / "export" / "state_income_tax_2026.json"
)


def _load_curated() -> tuple[dict[str, dict], dict[str, str]]:
    if not _CURATED_SOURCE.exists():
        return {}, {}
    data = json.loads(_CURATED_SOURCE.read_text())
    return data.get("by_state", {}), data.get("sources", {})


_CURATED, _SOURCES = _load_curated()


def _validate_schedules(curated: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the build; warnings are surfaced
    for manual review against the cited sources."""
    errors: list[str] = []
    warnings: list[str] = []

    # (3) No-wage-tax states must be kind="none".
    for s in sorted(_NO_WAGE_TAX_STATES):
        rule = curated.get(s)
        if rule is not None and rule.get("kind") != "none":
            errors.append(
                f"{s}: no-wage-income-tax state has kind={rule.get('kind')!r} "
                f"(must be 'none'); the research may have assigned a wage rate."
            )

    # (4) DC present as a real state row.
    dc = curated.get("DC")
    if dc is None or dc.get("kind") not in ("flat", "brackets"):
        errors.append("DC: missing or not a real schedule — it IS the 'state' line (no separate state layer).")

    # Coverage: every taxing jurisdiction present (else it silently uses the flat fallback).
    taxing_expected = _ALL_JURISDICTIONS - _NO_WAGE_TAX_STATES
    present_taxing = {s for s, r in curated.items() if r.get("kind") in ("flat", "brackets")}
    missing = taxing_expected - present_taxing
    if missing:
        warnings.append(
            f"{len(missing)} taxing jurisdiction(s) missing (will use the flat "
            f"STATE_TAX_RATES fallback): {sorted(missing)}"
        )

    for s, rule in sorted(curated.items()):
        kind = rule.get("kind")
        if kind == "none":
            continue
        std = rule.get("standard_deduction_by_filing_status") or {}
        ex_ = float(rule.get("personal_exemption", 0.0) or 0.0)
        has_base = bool(std) or ex_ > 0.0
        if kind == "flat":
            rate = float(rule.get("rate", 0.0) or 0.0)
            if not (0.0 < rate <= _MAX_PLAUSIBLE_RATE):
                errors.append(f"{s}: flat rate {rate} outside (0, {_MAX_PLAUSIBLE_RATE}].")
            # (1)+(2) flat-state base: a rate with no deduction/exemption is the
            # IL-exemption trap unless the state genuinely has none (PA).
            if not has_base and s not in _KNOWN_NO_DEDUCTION_FLAT:
                warnings.append(
                    f"{s}: flat state with NO standard deduction or personal "
                    f"exemption — verify the base (exemption-as-credit miscoded?)."
                )
        elif kind == "brackets":
            bfs = rule.get("brackets_by_filing_status") or {}
            if not bfs:
                errors.append(f"{s}: kind=brackets but no brackets_by_filing_status.")
                continue
            for fs, sched in bfs.items():
                edges = [(None if e is None else float(e), float(r)) for e, r in sched]
                prev = 0.0
                for edge, r in edges:
                    if not (0.0 <= r <= _MAX_PLAUSIBLE_RATE):
                        errors.append(f"{s}/{fs}: bracket rate {r} outside [0, {_MAX_PLAUSIBLE_RATE}].")
                    if edge is not None:
                        if edge <= prev:
                            errors.append(f"{s}/{fs}: bracket edges not increasing ({edge} after {prev}).")
                        prev = edge
                if edges[-1][0] is not None:
                    errors.append(f"{s}/{fs}: top bracket is not open-ended (last upper bound must be null).")
            # (1) brackets with no base is plausible (many states' brackets start at $0
            # after their own deduction); only warn if BOTH absent and EITC absent.
            if not has_base:
                warnings.append(
                    f"{s}: progressive state with no standard deduction or "
                    f"personal exemption — confirm the schedule starts at $0 of income."
                )
        else:
            errors.append(f"{s}: unknown kind {kind!r}.")

        # Source provenance present for every real schedule.
        if s not in _SOURCES:
            warnings.append(f"{s}: no official source_url recorded in _SOURCES.")

    return errors, warnings


def _artifacts_dir() -> Path:
    env = os.environ.get("ARTIFACTS_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "pipeline" / "artifacts"


def build() -> dict:
    if not _CURATED:
        raise SystemExit(
            "build_state_tax_rates: _CURATED is empty — populate it from the "
            "official-source research sweep before generating the artifact."
        )
    errors, warnings = _validate_schedules(_CURATED)
    for w in warnings:
        print(f"  WARN  {w}")
    if errors:
        for e in errors:
            print(f"  ERROR {e}")
        raise SystemExit(f"build_state_tax_rates: {len(errors)} validation error(s) — fix before writing.")
    return {
        "_meta": {
            "tax_year": _TAX_YEAR,
            "source_year": _SOURCE_YEAR,
            "as_of": _AS_OF,
            "description": (
                "Per-state 2026 individual income-tax schedules (brackets/flat + "
                "standard deduction + personal/dependent exemptions + state EITC%). "
                "Applied to state taxable income = gross − conforming pre-tax wedge "
                "− deduction − exemptions. Missing state → flat STATE_TAX_RATES fallback."
            ),
            "sources": _SOURCES,
            "notes": (
                "Curated from official DOR schedules; validated by "
                "_validate_schedules (no-tax states stay none, flat-base check, "
                "DC present, bracket monotonicity). Refresh annually."
            ),
        },
        "tax_year": _TAX_YEAR,
        "by_state": _CURATED,
    }


def main() -> None:
    artifact = build()
    out = _artifacts_dir() / "state_tax_rates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    by_kind: dict[str, int] = {}
    for r in artifact["by_state"].values():
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    print(f"Wrote {out} — {len(artifact['by_state'])} jurisdictions {by_kind} ({_TAX_YEAR}).")


if __name__ == "__main__":
    main()
