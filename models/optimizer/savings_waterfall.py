"""Up-direction tax-advantaged savings waterfall — stage 7c3 (2026-06-10).

The savings-side counterpart of the down-direction residual sweep (locked
REMAINDER-ZERO-INVARIANT-UP-DIRECTION; scoping:
``up_direction_savings_waterfall_scoping.md``). For the HIGH-savings-
contradiction case (``framing_state == "signal_would_pull_up_deferred"`` —
the reported balance implies saving ABOVE cohort, the heavy-saver), the
post-savings ``genuine_remainder`` is the deferred up-pull sitting
un-applied. It routes through the ordered waterfall:

    CC accelerated paydown (if cc_carried_balance > $300)
    →  401(k) elective-deferral headroom  →  IRA / backdoor Roth headroom
    →  HSA top-up headroom  →  taxable savings (the UNBOUNDED terminal)

CC paydown precedes 401(k) top-up by the paydown-first ordering (locked
CC-PAYDOWN-PREDICTED-ROUTING): the 22% CC APR dominates beyond-match
returns (~7–10%); MPRD evidence (NY Fed, Agarwal 2007) shows discretionary
surplus routed preferentially to debt paydown; the Vanguard sticky-
contribution evidence covers AUTOMATIC PAYROLL DEDUCTIONS (the committed
level already subtracted), not discretionary surplus allocation above it.
``PAYDOWN_SHARE = 0.40`` (calibrated: CFPB 1:1 experiment 50–57%, NY Fed
MPRD ~33%, partial co-holding; adjustable per-user — the mixture uncertainty
is why, locked CC-PAYDOWN-SHARE-IS-JUDGMENT-NOT-MEASURED). Applies to both
savings-side directions (``signal_would_pull_up_deferred`` AND
``signal_confirmed_cohort``); NOT ``signal_pulled_down`` (co-holding: low-
savers spend, not accelerate). Fills only from surplus, capped at the balance.

The statutory vehicles fill ONLY on the high-contradiction trigger and with a
readable limits registry. The taxable terminal guarantees
``genuine_remainder ≡ 0`` exactly; the bounded vehicles in front of it mean
most of the residual lands in correctly-specified destinations (a $250K
earner with a $750K balance almost certainly maxes the 401(k) — the model
previously under-routed retirement and parked the difference).

The load-bearing steers (carried exactly from scoping):

- ROUTE-THE-REMAINDER-NOT-THE-BLEND: the piece-3 up-cap lock (the savings
  blend cannot pull the RATE above cohort without explicit confirmation) is
  UNTOUCHED — the waterfall routes the already-existing remainder dollars to
  destinations; it never reopens the blend.
- POST-ALLOCATION-TRANSFER-NOT-COMMITTED-RE-SUBTRACTION: the 401(k)/HSA
  fills surface as ``predicted_topup`` on the committed lines at the
  serialization layer; they do NOT re-enter ``compute_d_variable`` (committed
  ↑ → d_var ↓ → remainder ↓ → fill ↓ is the circularity this forbids). The
  four-way closes because the top-ups + taxable == the remainder exactly.
- SUPPLY-BOUNDED (the committed over-prediction carve-out, active here):
  fills come only from the actual remainder — the waterfall never
  manufactures savings beyond it, and each vehicle's fill caps at its
  statutory headroom. Over-predicting committed outflows IS a harm
  (OVER-PREDICTION-EXTENDED-TO-COMMITTED-OUTFLOWS); the supply bound + the
  headroom cap are the guards.
- NO-CONTRADICTION-DEFAULT-TAXABLE-ONLY: profiles with no savings
  contradiction (``signal_confirmed_cohort``) and a small cap-bounce
  residual default to the taxable terminal ONLY — no statutory-vehicle
  assertion without balance evidence (predict-not-presume,
  USER-NEUTRAL-FRAMING). Resolves NO-CONTRADICTION-RESIDUAL-DEFAULT-PENDING.

Mutually exclusive with the down-direction sweep by construction: both gate
on ``framing_state``, and a profile carries exactly one of
``signal_pulled_down`` (→ elastic-sink sweep) /
``signal_would_pull_up_deferred`` (→ this waterfall) /
``signal_confirmed_cohort`` (→ taxable default). All three close to
remainder ≡ 0 — the symmetric COMPLETE-DOLLAR-ACCOUNTING invariant.

Missing/stale limits artifact → taxable-terminal-only degradation (the
registry's clean-no-op philosophy: still exact zero, just without named
vehicles).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.optimizer.backfill import ResidualAssignment
from shared.constants.statutory_limits import StatutoryLimits, load_statutory_limits
from shared.types import HouseholdProfile

# Committed-outflow codes the waterfall tops up (the post-allocation transfer
# targets — must match committed_outflows.py item codes).
COMMITTED_CODE_RETIREMENT = "retirement_contribution"
COMMITTED_CODE_HSA = "pretax_health_savings"

# CC paydown routing parameters (CC-PAYDOWN-PREDICTED-ROUTING, locked).
# The share is a population-average over the co-holding mixture — the
# adjustability carries the per-user uncertainty (CC-PAYDOWN-SHARE-IS-
# JUDGMENT-NOT-MEASURED).
PAYDOWN_SHARE: float = 0.40
_CC_TRIVIAL_BALANCE_FLOOR: float = 300.0  # mirrors debt_accumulation.py gate


@dataclass(frozen=True)
class WaterfallFill:
    """One vehicle's fill.

    ``current`` is the model's pre-waterfall prediction for the vehicle
    (the committed line, or 0 where no line exists); ``headroom`` the
    statutory room above it (``inf``-like None for the taxable terminal);
    ``annual`` the dollars routed here; ``maxed`` whether current + fill
    reached the statutory limit. ``mechanism`` carries the IRA route label
    (``direct_roth`` / ``partial_phaseout`` / ``backdoor_roth``) — a
    descriptive framing detail, never advice.
    """

    code: str            # k401_topup | ira | hsa_topup | taxable_savings
    label: str
    annual: float
    current: float = 0.0
    limit: float = 0.0   # 0.0 for the unbounded taxable terminal
    headroom: float = 0.0
    maxed: bool = False
    mechanism: str = ""
    adjustable: bool = True


@dataclass(frozen=True)
class WaterfallResult:
    """Output of the routing stage.

    ``fills`` in waterfall order, zero-fill vehicles omitted. ``total`` ==
    the routed remainder exactly when fired. ``topups_by_committed_code``
    carries the per-committed-line top-up for the serialization layer's
    ``predicted_topup`` surfacing (NEVER fed back into compute_d_variable).
    ``trigger`` ∈ ``high_savings_contradiction`` /
    ``no_contradiction_default``.
    """

    fills: tuple[WaterfallFill, ...] = ()
    total: float = 0.0
    fired: bool = False
    trigger: str = ""
    limits_year: int = 0  # 0 = limits unavailable (taxable-only degradation)
    topups_by_committed_code: dict[str, float] = field(default_factory=dict)


def route_residual_waterfall(
    *,
    remainder: float,
    framing_state: str,
    solver_status: str,
    profile: HouseholdProfile,
    filing_status: str,
    retirement_current: float,
    hsa_current: float,
    artifacts_path: str,
    enabled: bool = True,
    limits: StatutoryLimits | None = None,
) -> WaterfallResult:
    """Route the would-be genuine remainder through the savings waterfall.

    Runs AFTER ``assign_residual_savings`` (the blend set the savings line)
    and after the down-direction sweep stage (which no-ops on this case —
    disjoint framing gates). Supply-bounded ordered fill; the taxable
    terminal absorbs the remainder exactly.

    Args:
        remainder: ``residual_assignment.genuine_remainder`` (annual $).
        framing_state: the blend's signal classification — the trigger.
        retirement_current / hsa_current: the committed lines' annual
            predictions (post any user override — the override value is the
            authoritative "current" for headroom).
        limits: injectable for tests; default loads the annual registry.

    A no-op (``fired=False``) unless: enabled (aggregated path only),
    ``solver_status == "primary"``, ``remainder > 0``, and the framing is
    one of the two routed states. ``signal_pulled_down`` (the down-sweep's
    case) and ``user_pinned`` never route here.
    """
    noop = WaterfallResult()
    routed_states = ("signal_would_pull_up_deferred", "signal_confirmed_cohort")
    if (
        not enabled
        or solver_status != "primary"
        or remainder <= 0.0
        or framing_state not in routed_states
    ):
        return noop

    fills: list[WaterfallFill] = []
    remaining = remainder
    topups: dict[str, float] = {}

    # 0. CC accelerated paydown — paydown-first, before statutory vehicles.
    #    Fires on BOTH savings-side directions (signal_would_pull_up_deferred
    #    + signal_confirmed_cohort); never fires on signal_pulled_down (the
    #    co-holding puzzle: low-savers spend surplus, not accelerate debt).
    #    The minimum payment is already in debt_service (committed); this is
    #    the ACCELERATED portion above it.
    cc_balance = float(profile.cc_carried_balance)
    if cc_balance > _CC_TRIVIAL_BALANCE_FLOOR and remaining > 0.0:
        paydown_fill = min(remaining * PAYDOWN_SHARE, cc_balance)
        if paydown_fill > 0.0:
            fills.append(WaterfallFill(
                code="cc_paydown",
                label="CC accelerated paydown (above minimum — adjustable)",
                annual=paydown_fill,
                current=0.0,
                limit=cc_balance,
                headroom=cc_balance,
                maxed=(paydown_fill >= cc_balance - 1e-9),
            ))
            remaining -= paydown_fill

    if framing_state == "signal_would_pull_up_deferred":
        trigger = "high_savings_contradiction"
        if limits is None:
            limits = load_statutory_limits(artifacts_path)
        # Statutory vehicles fill ONLY on the high-contradiction trigger and
        # ONLY when the limits registry is readable (else degrade to the
        # taxable terminal — still exact zero, no vehicle assertion).
        if limits is not None:
            age = float(profile.age)
            magi = float(profile.gross_income)

            # 1. 401(k) elective deferral — the most tax-advantaged vehicle
            #    fills first. Headroom = limit − the committed-line current.
            k401_limit = limits.k401_limit(age)
            k401_room = max(0.0, k401_limit - max(0.0, float(retirement_current)))
            fill = min(remaining, k401_room)
            if fill > 0.0:
                fills.append(WaterfallFill(
                    code="k401_topup",
                    label="401(k) top-up (toward the elective-deferral max)",
                    annual=fill,
                    current=float(retirement_current),
                    limit=k401_limit,
                    headroom=k401_room,
                    maxed=fill >= k401_room - 1e-9,
                ))
                topups[COMMITTED_CODE_RETIREMENT] = fill
                remaining -= fill

            # 2. IRA / backdoor Roth — no modeled current contribution, so
            #    headroom = the full limit. The mechanism label (backdoor
            #    above the Roth MAGI phase-out) is descriptive framing only.
            ira_limit = limits.ira_limit_for(age)
            fill = min(remaining, ira_limit)
            if fill > 0.0:
                mech = limits.roth_mechanism(magi, filing_status)
                fills.append(WaterfallFill(
                    code="ira",
                    label=(
                        "IRA (backdoor Roth at your income)"
                        if mech == "backdoor_roth"
                        else "IRA / Roth IRA"
                    ),
                    annual=fill,
                    current=0.0,
                    limit=ira_limit,
                    headroom=ira_limit,
                    maxed=fill >= ira_limit - 1e-9,
                    mechanism=mech,
                ))
                remaining -= fill

            # 3. HSA top-up — headroom above the pretax_health_savings union
            #    line. HSA-eligibility (HDHP) is unobservable: presented on
            #    the HSA-arm-dominant assumption (COHORT-AVERAGE-RESPECTS-
            #    MUTUAL-EXCLUSION, the Build-A collapse-and-attribute form),
            #    adjustable to $0 for FSA-only / non-HDHP users.
            hsa_limit = limits.hsa_limit(age, int(profile.household_size))
            hsa_room = max(0.0, hsa_limit - max(0.0, float(hsa_current)))
            fill = min(remaining, hsa_room)
            if fill > 0.0:
                fills.append(WaterfallFill(
                    code="hsa_topup",
                    label="HSA top-up (if HSA-eligible — adjustable)",
                    annual=fill,
                    current=float(hsa_current),
                    limit=hsa_limit,
                    headroom=hsa_room,
                    maxed=fill >= hsa_room - 1e-9,
                ))
                topups[COMMITTED_CODE_HSA] = fill
                remaining -= fill
    else:
        # No-contradiction default: taxable terminal ONLY — no statutory-
        # vehicle assertion without balance evidence (predict-not-presume).
        trigger = "no_contradiction_default"
        limits = None

    # 4. Taxable savings — the unbounded terminal. Absorbs whatever remains,
    #    guaranteeing Σ fills == remainder exactly (remainder ≡ 0).
    if remaining > 0.0:
        fills.append(WaterfallFill(
            code="taxable_savings",
            label="Taxable savings / brokerage",
            annual=remaining,
            current=0.0,
            limit=0.0,
            headroom=0.0,
            maxed=False,
        ))

    return WaterfallResult(
        fills=tuple(fills),
        total=remainder,
        fired=True,
        trigger=trigger,
        limits_year=limits.year if limits is not None else 0,
        topups_by_committed_code=topups,
    )


def apply_waterfall_fold(
    base: ResidualAssignment, waterfall: WaterfallResult
) -> ResidualAssignment:
    """Fold a fired waterfall into the residual assignment: remainder ≡ 0.

    The savings line (the blend) is untouched — ROUTE-THE-REMAINDER-NOT-THE-
    BLEND: the waterfall's fills live in their own response block (+ the
    committed lines' ``predicted_topup``); the blend's
    ``savings_investment`` stays the rate prediction. ``framing_state``
    transitions ``signal_would_pull_up_deferred`` → ``signal_pulled_up_routed``
    (the deferred up-pull is now applied — as routing, not as a rate raise);
    the no-contradiction default keeps ``signal_confirmed_cohort``.
    A non-fired waterfall returns ``base`` unchanged.
    """
    if not waterfall.fired:
        return base
    framing = (
        "signal_pulled_up_routed"
        if waterfall.trigger == "high_savings_contradiction"
        else base.framing_state
    )
    return ResidualAssignment(
        savings_investment=base.savings_investment,
        genuine_remainder=0.0,
        realistic_savings_rate=base.realistic_savings_rate,
        realistic_savings_dollars=base.realistic_savings_dollars,
        source=base.source,
        framing_state=framing,
    )
