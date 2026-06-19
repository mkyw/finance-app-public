"""Down-direction residual sweep tests (remainder ≡ 0 invariant, spending side).

The 2026-06-10 build (locked REMAINDER-ZERO-INVARIANT-DOWN-DIRECTION): for the
low-savings-contradiction case (``framing_state == "signal_pulled_down"``) the
would-be ``genuine_remainder`` sweeps entirely into the high-participation
elastic sinks (``ELASTIC_SINK_CATEGORIES``), distributed by marginal income
response (measured × ε), with NO ceiling blocking the sweep
(OVER-PREDICTION-IS-THE-SAFE-DIRECTION-FOR-SPENDING).

Pinned properties:
  - trigger discipline: fires ONLY on signal_pulled_down + primary + enabled
    + remainder > 0 (no-contradiction / up-deferred / user_pinned /
    soft_constrained / disaggregated paths are byte-identical no-ops)
  - exact closure: Σ swept == remainder exactly (the invariant's arithmetic);
    apply_remainder_sweep → genuine_remainder == 0.0 identically
  - distribution: all usable sinks participate; weights ∝ measured × ε
  - savings line untouched (the blend set it; the sweep takes only the
    would-be remainder); framing_state preserved
  - sink-set hygiene: high-participation aggregated codes only

Run from repo root (artifacts needed for the QUAIDS elasticity lookup).
"""
from __future__ import annotations

import pytest

from models.engel.elasticity import quaids_elasticity
from models.optimizer.backfill import (
    ResidualAssignment,
    ResidualSweep,
    apply_remainder_sweep,
    sweep_remainder_to_sinks,
)
from shared.constants.categories import (
    AGGREGATED_CATEGORY_CODES,
    ELASTIC_SINK_CATEGORIES,
)

ARTIFACTS = "pipeline/artifacts"
AGG_COEFFS = "agent-artifacts/aggregation/coefficients_aggregated.json"
Y_EQ = 110_000.0

# Cohort-typical measured levels for the five sinks (annual $, NYC-ish scale).
MEASURED = {
    "eathome": 6_000.0,
    "eatout": 4_900.0,
    "shopping": 1_900.0,
    "entertainment": 2_600.0,
    "household_goods": 1_200.0,
}


def _sweep(
    remainder: float = 1_896.0,
    *,
    framing_state: str = "signal_pulled_down",
    solver_status: str = "primary",
    enabled: bool = True,
    measured: dict[str, float] | None = None,
    inferred: dict[str, float] | None = None,
) -> ResidualSweep:
    return sweep_remainder_to_sinks(
        remainder=remainder,
        measured=MEASURED if measured is None else measured,
        inferred=inferred or {},
        y_eq=Y_EQ,
        framing_state=framing_state,
        solver_status=solver_status,
        artifacts_path=ARTIFACTS,
        coefficients_path=AGG_COEFFS,
        enabled=enabled,
    )


# --------------------------------------------------------------------------- #
# Trigger discipline (the direction gate)                                      #
# --------------------------------------------------------------------------- #


def test_fires_on_low_contradiction_primary() -> None:
    res = _sweep()
    assert res.fired is True
    assert res.trigger == "low_savings_contradiction"


@pytest.mark.parametrize(
    "framing",
    ["signal_confirmed_cohort", "signal_would_pull_up_deferred", "user_pinned"],
)
def test_noop_on_every_other_framing_state(framing: str) -> None:
    """High-contradiction (up-deferred) + no-contradiction + pinned: no sweep.

    The high-balance (Santa Clara-style) profile must NOT get its remainder
    swept to spending — that is the wrong direction (the up-direction routes
    to the savings waterfall, the banked follow-up build).
    """
    res = _sweep(framing_state=framing)
    assert res.fired is False and res.swept == {} and res.total == 0.0


@pytest.mark.parametrize(
    "status", ["soft_constrained", "floor_infeasible", "structural_deficit"]
)
def test_noop_on_non_primary_solver(status: str) -> None:
    assert _sweep(solver_status=status).fired is False


def test_noop_when_disabled_or_zero_remainder() -> None:
    assert _sweep(enabled=False).fired is False
    assert _sweep(remainder=0.0).fired is False
    assert _sweep(remainder=-5.0).fired is False


# --------------------------------------------------------------------------- #
# Exact closure (the invariant's arithmetic)                                   #
# --------------------------------------------------------------------------- #


def test_swept_total_equals_remainder_exactly() -> None:
    res = _sweep(remainder=1_896.37)
    assert res.total == 1_896.37
    assert sum(res.swept.values()) == pytest.approx(1_896.37, abs=1e-9)


def test_no_ceiling_blocks_the_sweep() -> None:
    """An absurdly large residual is still fully absorbed (no caps —
    over-prediction of elastic consumption is the safe direction)."""
    res = _sweep(remainder=50_000.0)
    assert res.fired is True
    assert sum(res.swept.values()) == pytest.approx(50_000.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Distribution shape                                                           #
# --------------------------------------------------------------------------- #


def test_all_usable_sinks_participate() -> None:
    res = _sweep()
    assert set(res.swept) == set(MEASURED)
    assert all(v > 0.0 for v in res.swept.values())


def test_weights_proportional_to_measured_times_elasticity() -> None:
    res = _sweep(remainder=10_000.0)
    eps = {
        c: quaids_elasticity(c, Y_EQ, ARTIFACTS, AGG_COEFFS) for c in MEASURED
    }
    w = {c: MEASURED[c] * eps[c] for c in MEASURED}
    wsum = sum(w.values())
    for c in MEASURED:
        assert res.swept[c] == pytest.approx(10_000.0 * w[c] / wsum, rel=1e-6)


def test_backfill_inferred_counts_toward_weight() -> None:
    """The sweep weights on the post-back-fill level (measured + inferred)."""
    base = _sweep(remainder=10_000.0)
    lifted = _sweep(remainder=10_000.0, inferred={"eatout": 3_700.0})
    assert lifted.swept["eatout"] > base.swept["eatout"]


def test_epsilon_only_fallback_when_measured_all_zero() -> None:
    res = _sweep(measured={})
    assert res.fired is True
    assert sum(res.swept.values()) == pytest.approx(1_896.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# apply_remainder_sweep (the assignment fold)                                  #
# --------------------------------------------------------------------------- #


def _assignment(remainder: float = 1_896.0) -> ResidualAssignment:
    return ResidualAssignment(
        savings_investment=5_418.0,
        genuine_remainder=remainder,
        realistic_savings_rate=0.0812,
        realistic_savings_dollars=5_418.0,
        source="test",
        framing_state="signal_pulled_down",
    )


def test_apply_remainder_sweep_zeroes_remainder_only() -> None:
    swept = apply_remainder_sweep(_assignment(), _sweep())
    assert swept.genuine_remainder == 0.0          # the invariant, identically
    assert swept.savings_investment == 5_418.0     # savings line untouched
    assert swept.realistic_savings_rate == 0.0812
    assert swept.framing_state == "signal_pulled_down"


def test_apply_remainder_sweep_noop_when_not_fired() -> None:
    base = _assignment()
    assert apply_remainder_sweep(base, ResidualSweep()) is base


# --------------------------------------------------------------------------- #
# Sink-set hygiene                                                             #
# --------------------------------------------------------------------------- #


def test_sink_set_is_the_locked_five() -> None:
    assert ELASTIC_SINK_CATEGORIES == frozenset(
        {"eathome", "eatout", "shopping", "entertainment", "household_goods"}
    )
    assert ELASTIC_SINK_CATEGORIES <= set(AGGREGATED_CATEGORY_CODES)
