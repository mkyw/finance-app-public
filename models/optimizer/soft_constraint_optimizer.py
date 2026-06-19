"""Unified soft-constraint compression — the Stage-2 successor to ``compress``.

Replaces the two-phase (Phase A all-or-nothing step drops + Phase B log-barrier)
mechanism in ``compression.py`` with a SINGLE continuous separable-convex optimizer
driven by per-category empirical expenditure elasticities and conditional-p10 soft
floors. Grounded in
``agent-artifacts/investigations/soft_constraint_compression_architecture_scoping.md``
(Q3: compression is empirically continuous — no regime change, so no step/recurring
split; Q1: per-category elasticities; Q2: conditional-p10 floors; Q4: a single
per-category elasticity reproduces constrained-household spending to MAE 0.019).

**Model (iso-elastic water-filling).** When anchors overshoot the budget, allocate

    x_i(ν) = clip( a_i · ν^(−ε_i),  floor_i,  a_i )

and solve the 1-D budget equation Σ_i x_i(ν) = B (free set) for the dual ν ≥ 1 by
bisection. This is the closed-form realization of the constant-elasticity Engel
relation x_i ∝ (budget)^(ε_i) the investigation validated: a category with a higher
expenditure elasticity ε_i compresses proportionally more as ν rises (budget tightens),
exactly the cohort-typical behavior. ``floor_i`` is the composed soft floor (see
``compose_floors``); the hard floor is 0; the anchor a_i is the ceiling (a category is
never lifted above its cohort-typical anchor by compression — that is the back-fill's
job, not this stage's).

The 1-D dual bisection is O(n log n), deterministic, and microsecond-fast — the same
solver shape as the retired Phase B, so runtime is unchanged. No external convex solver
(respects locked #4: no CVXPY/OSQP QP).

**Status** (consumed as ``solver_status``):

    "soft_constrained" — anchors overshot; categories compressed per their elasticities
                         (the renamed successor to the old ``"compressed"``).
    "floor_infeasible" — even with every category at its composed floor, Σ floors still
                         exceeds the budget. A genuine deficit; held at floors, surfaced
                         for the deficit-handling design (slack = 0). Not a solver error.

Pinned categories (user overrides, housing pin lb==ub) are held fixed and removed from
the free set with their dollars subtracted from the budget — the optimizer solves the
remaining free set against the remaining budget (USER-ADJUSTMENT-AUTHORITY preserved).
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

_LOG = logging.getLogger(__name__)

_DEFAULT_PARAMS_PATH = "pipeline/artifacts/compression_parameters.json"


def load_compression_parameters(path: str = _DEFAULT_PARAMS_PATH) -> dict:
    """Load the per-category elasticity / soft-floor artifact.

    Returns ``{"aggregated": {...}, "disaggregated": {...}}``. On a missing or
    unreadable artifact returns empty maps — ``category_elasticity`` then yields the
    neutral ε = 1.0 everywhere (a clean no-op; the soft floors still come from the
    runtime ``conditional_p10``), so the optimizer degrades gracefully.
    """
    if not os.path.exists(path):
        _LOG.info("compression_parameters artifact absent (%s); ε defaults to 1.0", path)
        return {"aggregated": {}, "disaggregated": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        _LOG.warning("compression_parameters unreadable (%r); ε defaults to 1.0", exc)
        return {"aggregated": {}, "disaggregated": {}}
    return {
        "aggregated": dict(data.get("aggregated", {})),
        "disaggregated": dict(data.get("disaggregated", {})),
    }


def category_elasticity(params: dict, code: str, default: float = 1.0) -> float:
    """Empirical expenditure elasticity for one atom/category code.

    Looks up the aggregated map first (covers the 4 aggregates + retained atoms),
    then the disaggregated map (covers the merged raw members on the 55-cat path);
    retained codes are identical in both. Neutral ε = 1.0 when unknown.
    """
    agg = params.get("aggregated", {})
    if code in agg:
        return float(agg[code].get("elasticity", default))
    dis = params.get("disaggregated", {})
    if code in dis:
        return float(dis[code].get("elasticity", default))
    return default


# Participation gate for the soft floor: a category whose unconditional p10 is a
# diary-recall-window zero but whose participation is high gets the realistic
# conditional-p10 floor; genuinely-rare categories (low participation) keep their
# $0 floor (locked #2 / HEAVY-ZERO guardrail).
_PARTICIPATION_GATE: float = 0.75      # τ — confirmed in the investigation
_WINDOW_ZERO_P10: float = 1.0          # unconditional p10 below this is "≈ 0"

# Necessity protective floor fraction φ(ε) = clip(A − B·ε, MIN, MAX), as a fraction
# of the anchor — preserves the existing necessity protection so the soft-floor
# composition never *lowers* a necessity below it (carried over from the retired
# Phase B; the principle "necessities protected" stays even as the machinery changes).
_FLOOR_FRAC_A: float = 0.75
_FLOOR_FRAC_B: float = 0.50
_FLOOR_FRAC_MIN: float = 0.25
_FLOOR_FRAC_MAX: float = 0.70

# Elasticities are clamped into this band before driving the exponent, so a
# pathological derivation can't produce a degenerate compression rate.
_EPS_CLAMP_LO: float = 0.10
_EPS_CLAMP_HI: float = 2.50

_TOL: float = 1e-6
_BISECT_ITERS: int = 200


def _necessity_floor_fraction(eps: float) -> float:
    frac = _FLOOR_FRAC_A - _FLOOR_FRAC_B * float(eps)
    return float(np.clip(frac, _FLOOR_FRAC_MIN, _FLOOR_FRAC_MAX))


def compose_floors(
    anchors: np.ndarray,
    p10: np.ndarray,
    conditional_p10: np.ndarray,
    nonzero_rate: np.ndarray,
    is_luxury: np.ndarray,
    necessity_eps: np.ndarray,
) -> np.ndarray:
    """Compose the per-category soft floor the optimizer clips to.

    ``floor_i = max(participation_floor_i, necessity_protective_floor_i)`` where:

      * ``participation_floor`` = the realistic minimum for a participating household:
        the value-layer-scaled ``conditional_p10`` when the category is high-
        participation (``nonzero_rate ≥ τ``) AND its unconditional ``p10`` is a
        diary-window zero; otherwise the plain ``max(0, p10)`` (so genuinely-rare,
        low-participation categories keep their $0 floor — locked #2 guardrail).
      * ``necessity_protective_floor`` = ``φ(ε)·anchor`` for necessities (static luxury
        table), 0 for luxuries — preserves existing necessity protection so the
        composition never lowers a necessity (e.g. health: max(φ·anchor, cond_p10)).

    All inputs are arrays in category order; ``necessity_eps`` is the elasticity used
    only for the φ depth (the existing necessity-protection knob).
    """
    a = np.asarray(anchors, dtype=np.float64)
    p10 = np.asarray(p10, dtype=np.float64)
    cp10 = np.asarray(conditional_p10, dtype=np.float64)
    nz = np.asarray(nonzero_rate, dtype=np.float64)
    lux = np.asarray(is_luxury, dtype=bool)

    gated = (nz >= _PARTICIPATION_GATE) & (p10 < _WINDOW_ZERO_P10) & (cp10 > 0.0)
    participation_floor = np.where(gated, cp10, np.maximum(0.0, p10))

    nec_frac = np.array([_necessity_floor_fraction(e) for e in necessity_eps], dtype=np.float64)
    necessity_floor = np.where(lux, 0.0, nec_frac * a)

    floor = np.maximum(participation_floor, necessity_floor)
    # A floor can never exceed the anchor (the ceiling); clip for safety.
    return np.minimum(floor, a)


def soft_constraint_optimize(
    anchors: np.ndarray,
    floors: np.ndarray,
    elasticities: np.ndarray,
    budget: float,
    pinned: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Iso-elastic water-filling. Returns ``(s_vec, status)`` in category order.

    Args:
        anchors: per-category dollar anchor ``a_i`` (post value-layer + Engel + UX
            bias + clamp). Pinned categories carry their pinned value here.
        floors: composed soft floor ``floor_i`` (see ``compose_floors``).
        elasticities: per-category empirical expenditure elasticity ``ε_i`` (>0).
        budget: ``d_variable_adjusted`` — the dollar budget the allocation must meet.
        pinned: bool mask — categories held fixed at ``anchors[i]`` (housing pin /
            user override), removed from the free set.

    Returns ``(s_vec, status)`` with ``s_vec.sum() ≈ budget`` on ``"soft_constrained"``
    (or holding floors on ``"floor_infeasible"``, where the sum is the deficit).
    """
    a_all = np.asarray(anchors, dtype=np.float64)
    floors = np.minimum(np.maximum(np.asarray(floors, dtype=np.float64), 0.0), a_all)
    eps = np.clip(np.asarray(elasticities, dtype=np.float64), _EPS_CLAMP_LO, _EPS_CLAMP_HI)
    pinned = np.asarray(pinned, dtype=bool)

    s = a_all.copy()
    # Free set: not pinned, positive anchor (zero-anchor = tenure/balance-zeroed).
    free = (~pinned) & (a_all > _TOL)
    G = np.where(free)[0]
    if G.size == 0:
        total_now = float(s.sum())
        return s, ("soft_constrained" if total_now <= budget + _TOL else "floor_infeasible")

    a = a_all[G]
    f = floors[G]
    e = eps[G]
    fixed_sum = float(s.sum()) - float(a.sum())   # pinned + zero-anchor dollars
    budget_free = float(budget) - fixed_sum

    # Degenerate: free anchors already fit (caller normally guards this, but be safe).
    if float(a.sum()) <= budget_free + _TOL:
        s[G] = a
        return s, "soft_constrained"

    # Floor breach — genuine deficit. Hold everything at its composed floor.
    if float(f.sum()) > budget_free + _TOL:
        s[G] = f
        return s, "floor_infeasible"

    # Water-filling: x_i(ν) = clip(a_i·ν^(−ε_i), f_i, a_i). total(ν) is monotone
    # decreasing from Σa (ν=1) to Σf (ν→∞); budget_free is bracketed strictly between.
    def total(nu: float) -> float:
        return float(np.clip(a * nu ** (-e), f, a).sum())

    nu_lo = 1.0                       # ν=1 ⇒ x=a ⇒ total=Σa > budget_free
    nu_hi = 2.0
    while total(nu_hi) > budget_free:  # expand until total(ν_hi) ≤ budget_free
        nu_hi *= 2.0
        if nu_hi > 1e9:
            break
    for _ in range(_BISECT_ITERS):
        nu = np.sqrt(nu_lo * nu_hi)
        if total(nu) > budget_free:
            nu_lo = nu
        else:
            nu_hi = nu
        if nu_hi / nu_lo < 1.0 + 1e-12:
            break

    s_g = np.clip(a * nu_hi ** (-e), f, a)
    # Tiny residual from the geometric bracket: shave proportionally from headroom
    # above the floor so Σ matches the budget exactly (four-way closure).
    over = float(s_g.sum()) - budget_free
    if abs(over) > 1e-4:
        if over > 0:
            headroom = s_g - f
            th = float(headroom.sum())
            if th > 0:
                s_g = np.maximum(s_g - (over / th) * headroom, f)
        else:
            headroom = a - s_g
            th = float(headroom.sum())
            if th > 0:
                s_g = np.minimum(s_g + (-over / th) * headroom, a)
    s[G] = s_g
    return s, "soft_constrained"
