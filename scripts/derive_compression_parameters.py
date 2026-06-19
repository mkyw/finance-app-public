#!/usr/bin/env python3.11
"""Derive per-category compression parameters for the soft-constraint optimizer.

Productizes the empirical analysis from
``agent-artifacts/investigations/soft_constraint_compression_architecture_scoping.md``
(Q1/Q2) into a reproducible artifact the runtime optimizer consumes.

For every spending atom — on BOTH the aggregated (46-code) and disaggregated
(55-code) paths — this computes:

  * ``elasticity``       — the expenditure (budget) elasticity, the slope of the
                           log-log Engel relation across sample-weighted total-outlay
                           deciles (intensive + extensive margin). >1 ⇒ luxury,
                           compresses harder than proportional under constraint;
                           <1 ⇒ necessity, protected. This is the optimizer's
                           per-category compression rate ``ε_i`` (scale-invariant,
                           hence safe to precompute).
  * ``conditional_p10``  — p10 among nonzero spenders, in RAW 2024 CEX dollars. A
                           REFERENCE value (for validation + a fallback). The LIVE
                           soft floor the optimizer uses comes from the runtime
                           ``SpendingDistribution.conditional_p10``, which carries the
                           same CPI/RPP/PCE value-layer scaling as the anchor; this
                           raw figure is the unscaled cohort reference.
  * ``nonzero_rate``     — sample-weighted participation share (reference; the live
                           gate reads the per-cohort ``dist.nonzero_rate``).

Output: ``pipeline/artifacts/compression_parameters.json`` (JSON, mirroring the
scalar-artifact convention of ``pce_correction.json`` / ``cpi_scalars.json``).
Idempotent — re-running on the same synthetic population yields the same artifact.
Refresh alongside QUAIDS re-derivation when a new CEX vintage lands.

Usage:
    .venv/bin/python scripts/derive_compression_parameters.py [--stride N] [--out PATH]

``--stride`` subsamples the 2,462 PUMA parquets (every Nth file) for runtime; the
default gives good national coverage in well under a minute. Use ``--stride 1`` for
the full population.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Repo root on path so ``shared`` imports resolve when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.constants.categories import (  # noqa: E402
    AGG_CARVED_MEMBER,
    AGG_GROUPS,
    AGGREGATED_FLOW_CATEGORIES,
    CATEGORY_CODES,
    FLOW_CATEGORIES,
)

_DEFAULT_POP_GLOB = "pipeline/artifacts/synthetic_population/**/*.parquet"
_DEFAULT_OUT = "pipeline/artifacts/compression_parameters.json"
_BASE_YEAR = 2024
_MIN_OUTLAY = 1000.0  # drop degenerate near-zero-outlay rows

# Outlay base for the expenditure-elasticity decile regression: DISCRETIONARY
# CONSUMPTION — total flow spending MINUS the pinned/committed lines the optimizer
# does not compress (mortgage interest/principal, property tax, home insurance,
# home maintenance, other-housing, renter durables, and the financial / interest /
# transfer lines). Rent (rntval) is KEPT. This base matches what the optimizer
# actually compresses (the free categories against ``adjusted_d``), so the measured
# elasticity describes the relevant gradient. (Decision 2026-06-02: a broader
# all-flow base flattens elasticities toward 1 because committed housing dominates
# the gradient; this discretionary base is the defensible choice — see the
# soft-constraint architecture scoping doc.)
_OUTLAY_EXCLUDED: frozenset[str] = frozenset({
    "mrtgip", "mrtgpp", "mrtgps", "ptaxp", "hinsp", "hmtimp", "ohouse", "rntexp",
    "finpay", "stdint", "othint", "ocash", "chrty", "educ",
})


def _atom_value(df: pd.DataFrame, atom: str) -> np.ndarray:
    """Per-household dollar value of an aggregated atom from raw CEX columns.

    Mirrors ``algorithm.py``'s aggregation composition (sans the runtime
    value-layer scalar, which is irrelevant to a scale-invariant elasticity and
    to a raw-dollar reference floor):

      * the 4 AGG_GROUPS aggregates fold their members; household_goods nets out
        the carved ``pcare`` (which moved to shopping); shopping adds ``pcare``;
      * ``entertainment`` does NOT add the flat streaming constant here — that is a
        runtime per-household add that does not vary with the cohort, so including
        it would only dampen the measured elasticity. The cohort-varying part
        (oeprd+oesrv) is what the elasticity must describe.
      * every retained atom is just its raw column.
    """
    if atom == "shopping":
        return (df["cloftw"] + df[AGG_CARVED_MEMBER]).to_numpy(float)
    if atom == "household_goods":
        return (df["hhpcp"] - df[AGG_CARVED_MEMBER] + df["hhpcs"]).to_numpy(float)
    if atom in AGG_GROUPS:  # transportation, entertainment
        return df[list(AGG_GROUPS[atom])].sum(axis=1).to_numpy(float)
    return df[atom].to_numpy(float)  # retained single-code atom


def _weighted_pct(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w[order].sum()
    return float(np.interp(q, cw, x[order]))


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float((x * w).sum() / w.sum())


def _elasticity(value: np.ndarray, w: np.ndarray,
                dec: np.ndarray, out_means: np.ndarray) -> float:
    """Unconditional expenditure elasticity via the decile log-log regression.

    ``value`` includes zeros (extensive margin). The slope of
    ln(mean value in decile) on ln(mean outlay in decile) is the budget
    elasticity; deciles with a ~zero mean are dropped (log undefined).
    """
    cm = np.array([_weighted_mean(value[dec == d], w[dec == d]) for d in range(10)])
    msk = cm > 0.5
    if msk.sum() < 3:
        return 1.0  # too sparse to identify — neutral
    return float(np.polyfit(np.log(out_means[msk]), np.log(cm[msk]), 1)[0])


def _conditional_p10(value: np.ndarray, w: np.ndarray) -> float:
    sp = value > 1.0
    if sp.sum() < 2 or w[sp].sum() <= 0:
        return 0.0
    return _weighted_pct(value[sp], w[sp], 0.10)


def _params_for(atoms: list[str], df: pd.DataFrame,
                w: np.ndarray, dec: np.ndarray, out_means: np.ndarray,
                aggregated: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for atom in atoms:
        value = _atom_value(df, atom) if aggregated else df[atom].to_numpy(float)
        nz = float((w * (value > 1.0)).sum() / w.sum())
        out[atom] = {
            "elasticity": round(_elasticity(value, w, dec, out_means), 4),
            "conditional_p10": round(_conditional_p10(value, w), 2),
            "nonzero_rate": round(nz, 4),
        }
    return out


def derive(stride: int, pop_glob: str) -> dict:
    files = sorted(glob.glob(pop_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"no synthetic-population parquets at {pop_glob}")
    sample = files[::stride]
    # Only the raw flow columns + weight are needed (+ pcare, a non-CEX carved col).
    need = ["weight", *FLOW_CATEGORIES, AGG_CARVED_MEMBER]
    df = pd.concat(
        [pd.read_parquet(f, columns=need) for f in sample], ignore_index=True
    )
    base_cols = [c for c in FLOW_CATEGORIES if c not in _OUTLAY_EXCLUDED]
    outlay = df[base_cols].clip(lower=0).sum(axis=1).to_numpy(float)
    keep = outlay > _MIN_OUTLAY
    df = df.loc[keep].reset_index(drop=True)
    outlay = outlay[keep]
    w = df["weight"].to_numpy(float)

    # Weighted outlay deciles (shared identification grid for every atom).
    order = np.argsort(outlay)
    cw = np.cumsum(w[order]) / w[order].sum()
    dec = np.zeros(len(df), dtype=int)
    for i, q in enumerate(np.linspace(0.1, 0.9, 9)):
        dec[outlay > np.interp(q, cw, outlay[order])] = i + 1
    out_means = np.array([_weighted_mean(outlay[dec == d], w[dec == d]) for d in range(10)])

    # Aggregated flow atoms (exclude balance/zeroed cats — no compression role).
    agg_atoms = sorted(AGGREGATED_FLOW_CATEGORIES)
    # Disaggregated flow cats.
    disagg_atoms = [c for c in CATEGORY_CODES if c in FLOW_CATEGORIES]

    return {
        "_meta": {
            "base_year": _BASE_YEAR,
            "methodology": "expenditure elasticity = slope ln(mean atom) ~ ln(mean "
            "outlay) across weighted DISCRETIONARY-consumption-outlay deciles "
            "(total flow spend minus committed/pinned lines: mortgage, prop tax, "
            "home insurance/maint, financial/interest/transfer; rent kept); "
            "conditional_p10 = p10 among nonzero spenders (raw CEX $, reference — "
            "live floor is the value-layer-scaled runtime dist.conditional_p10).",
            "outlay_excluded": sorted(_OUTLAY_EXCLUDED),
            "sample_files": len(sample),
            "rows": int(len(df)),
            "stride": stride,
        },
        "aggregated": _params_for(agg_atoms, df, w, dec, out_means, aggregated=True),
        "disaggregated": _params_for(disagg_atoms, df, w, dec, out_means, aggregated=False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=12,
                    help="subsample every Nth PUMA parquet (default 12; 1 = full)")
    ap.add_argument("--out", default=_DEFAULT_OUT, help="output JSON path")
    ap.add_argument("--pop-glob", default=_DEFAULT_POP_GLOB)
    args = ap.parse_args()

    params = derive(args.stride, args.pop_glob)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
        fh.write("\n")

    meta = params["_meta"]
    print(f"derived compression parameters: {meta['sample_files']} files, "
          f"{meta['rows']} rows -> {args.out}")
    print(f"  aggregated atoms: {len(params['aggregated'])}  "
          f"disaggregated atoms: {len(params['disaggregated'])}")
    # Spot-check the headline atoms against the investigation's Q1 table.
    for atom in ("entertainment", "shopping", "household_goods", "eatout", "eathome"):
        p = params["aggregated"].get(atom)
        if p:
            print(f"  {atom:16s} ε={p['elasticity']:.2f}  "
                  f"cond_p10=${p['conditional_p10']:.0f}  nz={p['nonzero_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
