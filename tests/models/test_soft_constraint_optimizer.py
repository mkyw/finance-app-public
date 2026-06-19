"""Unit tests for the unified soft-constraint optimizer (Stage 2)."""

import numpy as np

from models.optimizer.soft_constraint_optimizer import (
    compose_floors,
    soft_constraint_optimize,
)


def _opt(anchors, floors, eps, budget, pinned=None):
    anchors = np.array(anchors, float)
    if pinned is None:
        pinned = np.zeros(len(anchors), bool)
    return soft_constraint_optimize(
        anchors, np.array(floors, float), np.array(eps, float), budget, np.array(pinned, bool)
    )


def test_no_compression_returns_anchors():
    # Free anchors already fit the budget — returns anchors untouched.
    s, status = _opt([100, 200, 300], [0, 0, 0], [1.0, 1.0, 1.0], 1000)
    assert np.allclose(s, [100, 200, 300])
    assert status == "soft_constrained"


def test_budget_closes_exactly():
    s, status = _opt([400, 400, 400], [10, 10, 10], [1.1, 1.0, 0.9], 900)
    assert status == "soft_constrained"
    assert abs(s.sum() - 900) < 1e-3


def test_floors_never_violated():
    s, _ = _opt([500, 500, 500], [120, 80, 200], [1.3, 1.2, 0.7], 600)
    assert np.all(s >= np.array([120, 80, 200]) - 1e-6)
    assert np.all(s <= np.array([500, 500, 500]) + 1e-6)


def test_higher_elasticity_compresses_more():
    # Identical anchor and floor; the higher-ε category must end up lower.
    s, _ = _opt([400, 400], [0, 0], [1.6, 0.6], 400)
    assert s[0] < s[1]  # ε=1.6 cut harder than ε=0.6
    assert abs(s.sum() - 400) < 1e-3


def test_floor_infeasible():
    # Composed floors ($300+$300+$300) exceed the budget ($700).
    s, status = _opt([500, 500, 500], [300, 300, 300], [1.0, 1.0, 1.0], 700)
    assert status == "floor_infeasible"
    assert np.allclose(s, [300, 300, 300])


def test_pinned_held_fixed():
    # Category 0 pinned (e.g. housing); the rest compress around it.
    s, status = _opt(
        [1000, 400, 400], [1000, 0, 0], [1.0, 1.1, 0.9], 1500,
        pinned=[True, False, False],
    )
    assert status == "soft_constrained"
    assert abs(s[0] - 1000) < 1e-9        # pin held exactly
    assert abs(s.sum() - 1500) < 1e-3     # budget closes
    assert s[1] < 400 and s[2] < 400      # free cats compressed


def test_zero_anchor_stays_zero():
    s, _ = _opt([0, 500, 500], [0, 50, 50], [1.0, 1.0, 1.0], 600)
    assert s[0] == 0.0


def test_compose_floors_gates_participation():
    anchors = np.array([1000.0, 1000.0, 1000.0])
    p10 = np.array([0.0, 0.0, 200.0])          # window-zero, window-zero, real p10
    cond_p10 = np.array([130.0, 90.0, 250.0])
    nonzero = np.array([0.89, 0.30, 0.95])     # high, LOW (rare), high
    is_luxury = np.array([True, True, True])
    eps = np.array([1.2, 1.2, 1.0])
    floors = compose_floors(anchors, p10, cond_p10, nonzero, is_luxury, eps)
    assert abs(floors[0] - 130.0) < 1e-9   # gated → conditional_p10
    assert floors[1] == 0.0                # rare (nz<τ) → $0 floor (locked #2)
    assert abs(floors[2] - 200.0) < 1e-9   # nonzero p10 → max(0,p10), not gated


def test_compose_floors_necessity_protected():
    # A necessity's floor is at least φ(ε)·anchor even if conditional_p10 is lower.
    anchors = np.array([1000.0])
    p10 = np.array([0.0])
    cond_p10 = np.array([100.0])
    nonzero = np.array([0.90])
    is_luxury = np.array([False])          # necessity
    eps = np.array([0.4])                  # deep necessity → high φ
    floors = compose_floors(anchors, p10, cond_p10, nonzero, is_luxury, eps)
    # φ(0.4) = clip(0.75 − 0.5·0.4, .25, .70) = 0.55 → floor ≥ 550, not the 100 cond_p10.
    assert floors[0] >= 550.0 - 1e-6
