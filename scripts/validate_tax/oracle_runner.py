"""PolicyEngine-US oracle runner — RUNS IN THE ISOLATED ORACLE VENV.

Reads a JSON list of scenarios from stdin, maps them into ONE PolicyEngine-US
simulation (each scenario is an independent tax unit — PolicyEngine vectorizes,
so a single ``calculate`` over N units is ~100× faster than N separate
``Simulation`` builds), and writes ``{key: {federal, state, ok}}`` to stdout.
Self-contained: imports only ``policyengine_us`` + stdlib, so it never touches
the repo runtime env (the oracle stays out of ``models/`` — the layering
invariant).

Invoke from ``gate1.py`` (in the repo .venv) via ``$TAX_ORACLE_PYTHON``:
    echo '<scenarios json>' | $TAX_ORACLE_PYTHON scripts/validate_tax/oracle_runner.py
"""

from __future__ import annotations

import json
import sys

_YEAR = 2026
_YR = str(_YEAR)
_FS = {
    "single": "SINGLE",
    "married_filing_jointly": "JOINT",
    "head_of_household": "HEAD_OF_HOUSEHOLD",
}


def _add_scenario(sit: dict, i: int, s: dict) -> str:
    """Add scenario ``i`` as an independent tax unit; return its tax_unit id."""
    filing = s["filing_status"]
    age = int(s.get("primary_age", 40))
    you = f"you_{i}"
    members = [you]
    sit["people"][you] = {
        "age": {_YR: age},
        "employment_income": {_YR: float(s["gross_income"])},
    }
    if s.get("itemize"):
        sit["people"][you]["real_estate_taxes"] = {_YR: float(s.get("property_tax", 0.0))}
        sit["people"][you]["deductible_mortgage_interest"] = {_YR: float(s.get("mortgage_interest", 0.0))}
        sit["people"][you]["charitable_cash_donations"] = {_YR: float(s.get("charity", 0.0))}
    if filing == "married_filing_jointly":
        sp = f"spouse_{i}"
        sit["people"][sp] = {"age": {_YR: age}, "employment_income": {_YR: 0.0}}
        members.append(sp)
        sit["marital_units"][f"m_{i}"] = {"members": [you, sp]}
    for j, dage in enumerate(s.get("dependent_ages") or []):
        dep = f"dep{j}_{i}"
        sit["people"][dep] = {"age": {_YR: int(dage)}}
        members.append(dep)
    tu = f"tu_{i}"
    sit["tax_units"][tu] = {"members": list(members), "filing_status": {_YR: _FS[filing]}}
    sit["households"][f"hh_{i}"] = {"members": list(members), "state_name": {_YR: s["state"] or "TX"}}
    return tu


def main() -> None:
    scenarios = json.load(sys.stdin)
    from policyengine_us import Simulation

    sit: dict = {"people": {}, "tax_units": {}, "households": {}, "marital_units": {}}
    tu_to_key: dict[str, str] = {}
    for i, s in enumerate(scenarios):
        tu = _add_scenario(sit, i, s)
        tu_to_key[tu] = s["key"]
    if not sit["marital_units"]:
        del sit["marital_units"]

    results: dict[str, dict] = {}
    try:
        sim = Simulation(situation=sit)
        fed = sim.calculate("income_tax", _YEAR)
        state = sim.calculate("state_income_tax", _YEAR)
        # The output arrays are in tax_unit insertion order; recover the ids.
        tu_ids = list(sit["tax_units"].keys())
        for idx, tu in enumerate(tu_ids):
            results[tu_to_key[tu]] = {
                "federal": float(fed[idx]), "state": float(state[idx]), "ok": True,
            }
    except Exception as exc:  # noqa: BLE001 — fall back to per-scenario on a batch failure
        results = _run_sequential(Simulation, scenarios, str(exc))

    json.dump(results, sys.stdout)


def _run_sequential(Simulation, scenarios: list[dict], batch_err: str) -> dict:
    """Fallback: one Simulation per scenario (slow but robust) if the batched
    build fails. Each scenario's failure is isolated."""
    out: dict[str, dict] = {}
    for s in scenarios:
        try:
            sit: dict = {"people": {}, "tax_units": {}, "households": {}, "marital_units": {}}
            _add_scenario(sit, 0, s)
            if not sit["marital_units"]:
                del sit["marital_units"]
            sim = Simulation(situation=sit)
            out[s["key"]] = {
                "federal": float(sim.calculate("income_tax", _YEAR)[0]),
                "state": float(sim.calculate("state_income_tax", _YEAR)[0]),
                "ok": True,
            }
        except Exception as exc:  # noqa: BLE001
            out[s["key"]] = {"federal": None, "state": None, "ok": False,
                             "error": f"batch_failed({batch_err[:60]}); seq: {str(exc)[:120]}"}
    return out


if __name__ == "__main__":
    main()
