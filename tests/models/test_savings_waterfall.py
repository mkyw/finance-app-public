"""Up-direction tax-advantaged waterfall tests (remainder ≡ 0, savings side).

The 2026-06-10 build (locked REMAINDER-ZERO-INVARIANT-UP-DIRECTION): for the
high-savings-contradiction case (``signal_would_pull_up_deferred``) the
would-be ``genuine_remainder`` routes through the ordered statutory waterfall
(401(k) → IRA/backdoor Roth → HSA → taxable terminal), supply-bounded, exact
zero via the unbounded terminal; no-contradiction profiles default to the
taxable terminal only.

Pinned properties:
  - trigger discipline: full waterfall ONLY on signal_would_pull_up_deferred;
    taxable-only on signal_confirmed_cohort; signal_pulled_down (the
    down-sweep's case) / user_pinned / non-primary / disabled → no-op
    (mutual exclusivity with the down-direction sweep)
  - fill order 401(k) → IRA → HSA → taxable; supply-bounded (fills only from
    the remainder, stops when exhausted; never exceeds statutory headroom —
    the committed over-prediction carve-out)
  - exact closure: Σ fills == remainder; apply_waterfall_fold →
    genuine_remainder == 0.0 identically, savings blend untouched
    (route-the-remainder-not-the-blend)
  - statutory limits read from the annual registry
    (pipeline/artifacts/statutory_limits.json); missing limits degrade to
    taxable-only; limits_stale flags a behind-calendar registry
  - catch-up conditioning (50+/60-63 401(k), 50+ IRA, 55+ HSA, family HSA)

Run from repo root (the limits artifact is read from pipeline/artifacts).
"""
from __future__ import annotations

from datetime import date

import pytest

from models.optimizer.backfill import ResidualAssignment
from models.optimizer.savings_waterfall import (
    PAYDOWN_SHARE,
    WaterfallResult,
    apply_waterfall_fold,
    route_residual_waterfall,
)
from shared.constants.statutory_limits import (
    StatutoryLimits,
    limits_stale,
    load_statutory_limits,
)
from shared.types import HouseholdProfile, Tenure

ARTIFACTS = "pipeline/artifacts"

LIMITS_2026 = StatutoryLimits(
    year=2026,
    k401_elective_deferral=24_500.0,
    k401_catchup_age50=8_000.0,
    k401_catchup_age60_63=11_250.0,
    ira_limit=7_500.0,
    ira_catchup_age50=1_100.0,
    roth_phaseout_single=(153_000.0, 168_000.0),
    roth_phaseout_mfj=(242_000.0, 252_000.0),
    hsa_self_only=4_400.0,
    hsa_family=8_750.0,
    hsa_catchup_age55=1_000.0,
)

# Santa Clara demonstrating-case scale: retirement $18,766/yr, HSA $960/yr.
RET_CURRENT = 18_766.0
HSA_CURRENT = 960.0


def _profile(
    age: int = 32,
    income: float = 250_000.0,
    hh: int = 1,
    cc_balance: float = 0.0,
) -> HouseholdProfile:
    return HouseholdProfile(
        age=age, gross_income=income, puma_code="CA_08507", tenure=Tenure.RENT,
        housing_cost=3_000, household_size=hh, savings=750_000,
        cc_carried_balance=cc_balance,
    )


def _route(
    remainder: float = 23_179.0,
    *,
    framing_state: str = "signal_would_pull_up_deferred",
    solver_status: str = "primary",
    enabled: bool = True,
    age: int = 32,
    hh: int = 1,
    cc_balance: float = 0.0,
    limits: StatutoryLimits | None = LIMITS_2026,
    retirement_current: float = RET_CURRENT,
    hsa_current: float = HSA_CURRENT,
) -> WaterfallResult:
    return route_residual_waterfall(
        remainder=remainder,
        framing_state=framing_state,
        solver_status=solver_status,
        profile=_profile(age=age, hh=hh, cc_balance=cc_balance),
        filing_status="single",
        retirement_current=retirement_current,
        hsa_current=hsa_current,
        artifacts_path=ARTIFACTS,
        enabled=enabled,
        limits=limits,
    )


def _fills_by_code(res: WaterfallResult) -> dict[str, float]:
    return {f.code: f.annual for f in res.fills}


# --------------------------------------------------------------------------- #
# Trigger discipline (mutual exclusivity with the down-direction sweep)        #
# --------------------------------------------------------------------------- #


def test_fires_full_waterfall_on_high_contradiction() -> None:
    res = _route()
    assert res.fired and res.trigger == "high_savings_contradiction"
    assert [f.code for f in res.fills] == ["k401_topup", "ira", "hsa_topup", "taxable_savings"]


def test_no_contradiction_defaults_to_taxable_only() -> None:
    """NO-CONTRADICTION-DEFAULT-TAXABLE-ONLY: no statutory-vehicle assertion
    without balance evidence — the small residual sweeps to the terminal."""
    res = _route(1_000.0, framing_state="signal_confirmed_cohort")
    assert res.fired and res.trigger == "no_contradiction_default"
    assert [f.code for f in res.fills] == ["taxable_savings"]
    assert res.fills[0].annual == 1_000.0
    assert res.topups_by_committed_code == {}


@pytest.mark.parametrize("framing", ["signal_pulled_down", "user_pinned", "signal_pulled_up_routed"])
def test_noop_on_down_direction_and_pinned(framing: str) -> None:
    """signal_pulled_down is the down-sweep's case — the waterfall never
    touches it (mutual exclusivity by the framing gate)."""
    res = _route(framing_state=framing)
    assert res.fired is False and res.fills == ()


@pytest.mark.parametrize("status", ["soft_constrained", "floor_infeasible", "structural_deficit"])
def test_noop_on_non_primary(status: str) -> None:
    assert _route(solver_status=status).fired is False


def test_noop_when_disabled_or_zero_remainder() -> None:
    assert _route(enabled=False).fired is False
    assert _route(0.0).fired is False
    assert _route(-1.0).fired is False


# --------------------------------------------------------------------------- #
# Fill order, supply-bounding, exact closure                                   #
# --------------------------------------------------------------------------- #


def test_santa_clara_fill_walk() -> None:
    """The scoping Q2 fill: 401(k) +5,734 → IRA +7,500 → HSA +3,440 →
    taxable absorbs the rest; all three bounded vehicles maxed."""
    res = _route(23_179.0)
    fills = _fills_by_code(res)
    assert fills["k401_topup"] == pytest.approx(24_500.0 - RET_CURRENT)   # 5,734
    assert fills["ira"] == pytest.approx(7_500.0)
    assert fills["hsa_topup"] == pytest.approx(4_400.0 - HSA_CURRENT)     # 3,440
    assert fills["taxable_savings"] == pytest.approx(
        23_179.0 - 5_734.0 - 7_500.0 - 3_440.0
    )
    assert all(f.maxed for f in res.fills if f.code != "taxable_savings")
    assert sum(fills.values()) == pytest.approx(23_179.0, abs=1e-9)       # exact zero


def test_supply_bounded_small_remainder_fills_first_vehicle_only() -> None:
    """A small remainder partially fills the 401(k) and STOPS — the waterfall
    never manufactures savings beyond the supply (the committed
    over-prediction carve-out)."""
    res = _route(2_000.0)
    fills = _fills_by_code(res)
    assert fills == {"k401_topup": 2_000.0}
    assert res.fills[0].maxed is False
    assert res.topups_by_committed_code == {"retirement_contribution": 2_000.0}


def test_supply_bounded_mid_remainder_stops_mid_waterfall() -> None:
    res = _route(10_000.0)
    fills = _fills_by_code(res)
    assert fills["k401_topup"] == pytest.approx(5_734.0)
    assert fills["ira"] == pytest.approx(4_266.0)   # partial; HSA/taxable never reached
    assert "hsa_topup" not in fills and "taxable_savings" not in fills


def test_excess_beyond_total_headroom_lands_in_taxable() -> None:
    res = _route(100_000.0)
    fills = _fills_by_code(res)
    assert fills["taxable_savings"] == pytest.approx(100_000.0 - 5_734.0 - 7_500.0 - 3_440.0)
    assert sum(fills.values()) == pytest.approx(100_000.0, abs=1e-6)


def test_fill_never_exceeds_statutory_headroom() -> None:
    """The topped retirement line caps at the elective max — over-predicting
    committed beyond the statutory limit is the harm the carve-out guards."""
    res = _route(1_000_000.0)
    for f in res.fills:
        if f.code != "taxable_savings":
            assert f.current + f.annual <= f.limit + 1e-9


def test_already_maxed_current_yields_no_topup() -> None:
    res = _route(5_000.0, retirement_current=24_500.0, hsa_current=4_400.0)
    fills = _fills_by_code(res)
    assert "k401_topup" not in fills and "hsa_topup" not in fills
    assert fills["ira"] == pytest.approx(5_000.0)


# --------------------------------------------------------------------------- #
# Statutory-limit conditioning                                                 #
# --------------------------------------------------------------------------- #


def test_catchup_conditioning() -> None:
    assert LIMITS_2026.k401_limit(32) == 24_500.0
    assert LIMITS_2026.k401_limit(52) == 32_500.0          # 50+ catch-up
    assert LIMITS_2026.k401_limit(61) == 35_750.0          # 60-63 super catch-up
    assert LIMITS_2026.ira_limit_for(52) == 8_600.0
    assert LIMITS_2026.hsa_limit(32, 1) == 4_400.0
    assert LIMITS_2026.hsa_limit(56, 1) == 5_400.0         # 55+ catch-up
    assert LIMITS_2026.hsa_limit(32, 3) == 8_750.0         # family tier


def test_roth_mechanism_classification() -> None:
    assert LIMITS_2026.roth_mechanism(250_000.0, "single") == "backdoor_roth"
    assert LIMITS_2026.roth_mechanism(100_000.0, "single") == "direct_roth"
    assert LIMITS_2026.roth_mechanism(160_000.0, "single") == "partial_phaseout"
    assert LIMITS_2026.roth_mechanism(245_000.0, "married_joint") == "partial_phaseout"


def test_ira_mechanism_labeled_backdoor_for_high_earner() -> None:
    res = _route()
    ira = next(f for f in res.fills if f.code == "ira")
    assert ira.mechanism == "backdoor_roth"


# --------------------------------------------------------------------------- #
# Registry loading + freshness                                                 #
# --------------------------------------------------------------------------- #


def test_limits_load_from_the_annual_registry() -> None:
    limits = load_statutory_limits(ARTIFACTS)
    assert limits is not None and limits.year == 2026
    assert limits.k401_elective_deferral == 24_500.0
    assert limits.ira_limit == 7_500.0
    assert limits.hsa_self_only == 4_400.0 and limits.hsa_family == 8_750.0
    assert limits.roth_phaseout_single == (153_000.0, 168_000.0)


def test_missing_limits_degrade_to_taxable_only(tmp_path) -> None:
    """Registry absent → no statutory-vehicle assertion; the terminal still
    closes to exact zero (the clean-no-op philosophy)."""
    res = route_residual_waterfall(
        remainder=23_179.0,
        framing_state="signal_would_pull_up_deferred",
        solver_status="primary",
        profile=_profile(),
        filing_status="single",
        retirement_current=RET_CURRENT,
        hsa_current=HSA_CURRENT,
        artifacts_path=str(tmp_path),   # empty dir — no statutory_limits.json
        enabled=True,
        limits=None,
    )
    assert res.fired and res.limits_year == 0
    assert [f.code for f in res.fills] == ["taxable_savings"]
    assert res.fills[0].annual == pytest.approx(23_179.0)


def test_stale_limit_flagged() -> None:
    assert limits_stale(LIMITS_2026, date(2026, 6, 10)) is False
    assert limits_stale(LIMITS_2026, date(2027, 1, 15)) is True   # behind calendar
    assert limits_stale(None, date(2026, 6, 10)) is True


# --------------------------------------------------------------------------- #
# apply_waterfall_fold (the assignment fold)                                   #
# --------------------------------------------------------------------------- #


def _assignment(framing: str = "signal_would_pull_up_deferred") -> ResidualAssignment:
    return ResidualAssignment(
        savings_investment=37_754.0,
        genuine_remainder=23_179.0,
        realistic_savings_rate=0.2699,
        realistic_savings_dollars=37_754.0,
        source="test",
        framing_state=framing,
    )


def test_fold_zeroes_remainder_blend_untouched() -> None:
    folded = apply_waterfall_fold(_assignment(), _route())
    assert folded.genuine_remainder == 0.0                  # the invariant
    assert folded.savings_investment == 37_754.0            # blend untouched
    assert folded.realistic_savings_rate == 0.2699          # rate untouched
    assert folded.framing_state == "signal_pulled_up_routed"


def test_fold_keeps_confirmed_cohort_framing_on_default_path() -> None:
    base = _assignment(framing="signal_confirmed_cohort")
    folded = apply_waterfall_fold(base, _route(1_000.0, framing_state="signal_confirmed_cohort"))
    assert folded.genuine_remainder == 0.0
    assert folded.framing_state == "signal_confirmed_cohort"


def test_fold_noop_when_not_fired() -> None:
    base = _assignment()
    assert apply_waterfall_fold(base, WaterfallResult()) is base


# --------------------------------------------------------------------------- #
# CC accelerated paydown (CC-PAYDOWN-PREDICTED-ROUTING)                        #
# --------------------------------------------------------------------------- #


def test_cc_paydown_fires_before_k401_on_high_contradiction() -> None:
    """Paydown-first ordering: cc_paydown is the first fill; k401_topup follows."""
    res = _route(cc_balance=8_000.0)
    codes = [f.code for f in res.fills]
    assert codes[0] == "cc_paydown", f"Expected cc_paydown first, got {codes}"
    assert "k401_topup" in codes
    assert codes.index("cc_paydown") < codes.index("k401_topup")


def test_cc_paydown_amount_is_share_of_surplus() -> None:
    """paydown_fill = min(remaining × PAYDOWN_SHARE, balance)."""
    remainder = 23_179.0
    cc_balance = 8_000.0
    res = _route(remainder=remainder, cc_balance=cc_balance)
    paydown = next(f for f in res.fills if f.code == "cc_paydown")
    expected = min(remainder * PAYDOWN_SHARE, cc_balance)
    assert paydown.annual == pytest.approx(expected, rel=1e-9)


def test_cc_paydown_capped_at_balance() -> None:
    """Large surplus: paydown is capped at the balance, not PAYDOWN_SHARE × remainder."""
    # $250K remainder >> $300 balance → cap at balance
    cc_balance = 500.0
    res = _route(remainder=50_000.0, cc_balance=cc_balance)
    paydown = next(f for f in res.fills if f.code == "cc_paydown")
    assert paydown.annual == pytest.approx(cc_balance, rel=1e-9)
    assert paydown.maxed is True


def test_cc_paydown_partial_below_cap() -> None:
    """Small remainder: paydown = PAYDOWN_SHARE × remaining (not balance-capped)."""
    remainder = 1_000.0
    cc_balance = 8_000.0   # much larger than 40% of remainder
    res = _route(remainder=remainder, cc_balance=cc_balance)
    paydown = next(f for f in res.fills if f.code == "cc_paydown")
    assert paydown.annual == pytest.approx(remainder * PAYDOWN_SHARE, rel=1e-9)
    assert paydown.maxed is False


def test_cc_paydown_gated_off_when_no_balance() -> None:
    """No cc_carried_balance → fills byte-identical to baseline (regression gate)."""
    baseline = _route(cc_balance=0.0)
    assert all(f.code != "cc_paydown" for f in baseline.fills)
    # Fill order unchanged from the pre-paydown build.
    codes = [f.code for f in baseline.fills]
    assert codes == ["k401_topup", "ira", "hsa_topup", "taxable_savings"]


def test_cc_paydown_trivial_balance_suppressed() -> None:
    """Balance ≤ $300 → gate off (mirrors debt_accumulation trivial-balance floor)."""
    res = _route(cc_balance=300.0)
    assert all(f.code != "cc_paydown" for f in res.fills)
    res_just_above = _route(cc_balance=300.01)
    assert any(f.code == "cc_paydown" for f in res_just_above.fills)


def test_cc_paydown_fires_on_no_contradiction_path() -> None:
    """signal_confirmed_cohort + CC balance → paydown before taxable terminal."""
    remainder = 2_000.0
    cc_balance = 4_000.0
    res = _route(
        remainder=remainder,
        framing_state="signal_confirmed_cohort",
        cc_balance=cc_balance,
    )
    assert res.fired
    codes = [f.code for f in res.fills]
    assert "cc_paydown" in codes
    assert codes[0] == "cc_paydown"
    paydown = next(f for f in res.fills if f.code == "cc_paydown")
    assert paydown.annual == pytest.approx(remainder * PAYDOWN_SHARE, rel=1e-9)
    # Taxable absorbs the rest.
    taxable = next(f for f in res.fills if f.code == "taxable_savings")
    assert taxable.annual == pytest.approx(remainder * (1 - PAYDOWN_SHARE), rel=1e-9)


def test_cc_paydown_not_fired_on_down_direction() -> None:
    """signal_pulled_down → waterfall no-ops entirely (co-holding: low-savers spend)."""
    res = _route(framing_state="signal_pulled_down", cc_balance=8_000.0)
    assert res.fired is False
    assert res.fills == ()


def test_cc_paydown_total_closure_with_paydown_active() -> None:
    """Σ fills == remainder exactly when cc_paydown is active (remainder ≡ 0)."""
    res = _route(remainder=23_179.0, cc_balance=8_000.0)
    assert sum(f.annual for f in res.fills) == pytest.approx(23_179.0, abs=1e-9)


def test_cc_paydown_adjustable_flag() -> None:
    """cc_paydown.adjustable == True (the mixture uncertainty is structural)."""
    res = _route(cc_balance=5_000.0)
    paydown = next(f for f in res.fills if f.code == "cc_paydown")
    assert paydown.adjustable is True
