"""Build/CI-time tax validation harness (Stage 7).

The oracle (PolicyEngine-US, optionally NBER TAXSIM) lives ONLY here in
``scripts/`` — never imported at runtime. ``models/`` stays a pure, oracle-free
serving layer (the layering invariant). This package:

  * ``grid``         — the boundary-dense scenario generator (cliffs enumerated
                       from the engines' own parameters).
  * ``oracle_runner``— runs in the ISOLATED oracle venv; maps each scenario to a
                       PolicyEngine simulation and emits federal/state results.
  * ``gate1``        — Gate 1: identical inputs to our stack AND the oracle,
                       compared within the inter-oracle band; disagreements routed
                       by per-entry confidence (provisional→defer, confident→
                       investigate). Emits the checked-in comparison report.

Gate 2 (consumed-prediction accuracy vs CEX) reuses ``scripts/validate_model.py``.
"""
