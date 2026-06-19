"""Profile-analysis orchestration — the contract boundary between Django
and the pure-Python model-serving layer in `models/`.

One entry point: :func:`run_profile_analysis`. It runs the full chain
(match -> engel annotate -> lambda weights -> feasibility -> decompose
-> benefits screen) and returns a JSON-ready dict.

`save_profile_analysis` persists the result to the database — separate
so the view can call `run_profile_analysis` for anonymous onboarding
without touching the DB.
"""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from django.conf import settings

from models.benefits.eligibility import screen
from models.engel.curves import annotate_distributions
from models.engel.elasticity import lambda_weights_all
from models.matching.algorithm import match_household
from models.matching.cpi_scaler import load_cpi_scalars
from models.matching.eia_utility import load_eia_utility_scalars
from models.matching.geographic_drivers import GeographicDrivers, load_climate_normals
from models.optimizer.backfill import (
    apply_remainder_sweep,
    assign_residual_savings,
    compute_backfill,
    sweep_remainder_to_sinks,
)
from models.optimizer.savings_waterfall import (
    COMMITTED_CODE_HSA,
    COMMITTED_CODE_RETIREMENT,
    apply_waterfall_fold,
    route_residual_waterfall,
)
from models.optimizer.committed_outflows import estimate_committed_outflows, is_pre_tax, pretax_wedge
from models.tax.calculator import TaxDetail
from models.tax.imputation import impute_filing_unit
from models.tax.itemize import derive_schedule_a
from models.optimizer.debt_accumulation import project_debt_accumulation
from models.optimizer.debt_service import estimate_annual_debt_service
from models.optimizer.decomposition import decompose
from models.optimizer.feasibility import solve_feasibility
from models.pace.calculator import compute_d_variable
from shared.constants.categories import (
    AGGREGATED_CATEGORY_CODES,
    AGGREGATED_LUXURY_CATEGORIES,
    ASSET_CATEGORIES,
    CATEGORIES,
    OMIT_BY_DEFAULT_CATEGORIES,
)
from shared.types.enums import Tenure
from shared.types.household import HouseholdProfile

_CATEGORY_LABEL: dict[str, str] = {c["cat"]: c["description"] for c in CATEGORIES}


def _artifacts_path() -> str:
    """Resolve the artifacts directory.

    Precedence: ``ARTIFACTS_PATH`` env var > ``settings.ARTIFACTS_PATH``
    > repo-root fallback. Always returns a string (the models layer
    uses ``Path(str)`` internally).
    """
    env = os.environ.get("ARTIFACTS_PATH")
    if env:
        return env
    configured = getattr(settings, "ARTIFACTS_PATH", None)
    if configured:
        return str(configured)
    return str(Path(settings.BASE_DIR).parent.parent / "pipeline" / "artifacts")


def _aggregated_coefficients_path(artifacts: str) -> str:
    """Path to the aggregated QUAIDS coefficients file.

    Override via ``AGGREGATED_COEFFICIENTS_PATH``; otherwise resolved as
    ``<repo>/agent-artifacts/aggregation/coefficients_aggregated.json``,
    where ``<repo>`` is the parent of the ``pipeline/artifacts`` dir.
    """
    env = os.environ.get("AGGREGATED_COEFFICIENTS_PATH")
    if env:
        return env
    repo_root = Path(artifacts).resolve().parent.parent
    return str(
        repo_root / "agent-artifacts" / "aggregation" / "coefficients_aggregated.json"
    )


def build_household_profile(
    age: int,
    gross_income: float,
    puma_code: str,
    tenure: str,
    housing_cost: float,
    household_size: int,
    savings: float = 0.0,
    cc_carried_balance: float = 0.0,
    student_loan_payment: float = 0.0,
    auto_loan_payment: float = 0.0,
    other_debt_payment: float = 0.0,
    place_fips: str = "",
    county_fips: str = "",
) -> HouseholdProfile:
    """Construct a :class:`HouseholdProfile` from raw API inputs.

    ``tenure`` arrives as the string ``"OWN"`` or ``"RENT"``; convert
    to the :class:`Tenure` enum here. ``equivalized_income`` is filled
    in by ``HouseholdProfile.__post_init__`` (sqrt scale). ``savings``
    and the four debt fields are optional (default 0) so pre-wiring
    clients still work and produce byte-identical predictions.
    ``place_fips`` and ``county_fips`` are Census FIPS identifiers from
    the city resolver; both default to ``""`` so callers that omit them
    remain byte-identical (no local tax applied).
    """
    return HouseholdProfile(
        age=age,
        gross_income=gross_income,
        puma_code=puma_code,
        tenure=Tenure(tenure),
        housing_cost=housing_cost,
        household_size=household_size,
        savings=float(savings),
        cc_carried_balance=float(cc_carried_balance),
        student_loan_payment=float(student_loan_payment),
        auto_loan_payment=float(auto_loan_payment),
        other_debt_payment=float(other_debt_payment),
        place_fips=place_fips,
        county_fips=county_fips,
    )


# --------------------------------------------------------------------------- #
# Pre-tax fixed point (Stage 3b). The waterfall's traditional-401(k)/HSA       #
# top-ups reduce taxable income on the next pass, so tax ↔ take_home ↔         #
# allocation is a genuine fixed point, not a one-pass DAG. We iterate to       #
# convergence on take_home; it is a contraction (top-ups are statutory- and    #
# residual-bounded, and the tax-on-the-delta is a small fraction). The savings #
# REGIME is pinned to the topup-free committed baseline                        #
# (PRETAX-FRAMING-LOOP-INVARIANT — see _allocation_pass / assign_residual_     #
# savings), which is what makes the map continuous: without it a borderline    #
# saver's waterfall toggled on/off between passes (a discontinuity with no     #
# fixed point → take_home 2-cycle → spurious 500). With the regime pinned the  #
# map contracts geometrically; the worst observed corner (low per-capita       #
# income, large household — the high-feedback case) is gain ~0.34 / ~8 passes. #
# The cap carries margin above that for unsampled higher-gain corners; once    #
# the regime is pinned, exceeding it is a genuine FINDING (hard error), never  #
# a silently-accepted last iteration.                                          #
# --------------------------------------------------------------------------- #
_MAX_PRETAX_ITERS: int = 25
_PRETAX_CONVERGE_TOL: float = 1.0   # dollars, absolute floor on the take_home delta
# Relative convergence band on take_home. The fixed point spans a 50x income
# range ($12k–$626k), so an absolute-only criterion is dimensionally wrong — a $1
# delta on a $62k take_home is 1.6e-5, forcing many needless passes for the
# high-feedback corner (low-income MFJ with a large waterfall top-up). 1e-4 is far
# tighter than the model's cohort-typical precision, and the four-way closes
# EXACTLY every pass regardless of this band (it only pins how precisely the
# fixed point is resolved, never whether the dollars reconcile). Effective
# tolerance = max(_PRETAX_CONVERGE_TOL, _PRETAX_CONVERGE_REL × |take_home|).
_PRETAX_CONVERGE_REL: float = 1e-4
_FOUR_WAY_TOL: float = 2.0          # dollars, four-way closure slack
# Waterfall fills that reduce TAXABLE INCOME next pass: traditional-401(k)
# headroom (income-tax base, not FICA) + HSA top-up (above-the-line). The IRA
# fill is routed to Roth (not deductible); cc_paydown / taxable_savings are
# post-tax. This is the ONLY feedback into the wedge — the post-allocation-
# transfer accounting is unchanged (top-ups still surface only at serialization;
# reducing taxable income is a separate concern, per Point 1).
_PRETAX_TOPUP_CODES: frozenset[str] = frozenset({"k401_topup", "hsa_topup"})


def _pretax_topup_income_tax_excludable(savings_waterfall) -> float:
    """Income-tax-excludable waterfall top-ups that feed the next pass's wedge."""
    if not savings_waterfall.fired:
        return 0.0
    return sum(
        float(f.annual)
        for f in savings_waterfall.fills
        if f.code in _PRETAX_TOPUP_CODES
    )


def _assert_four_way_closes(
    final, feasibility, residual_assignment, savings_waterfall, *, context: str,
    use_aggregated: bool,
) -> None:
    """Four-way closure: spending + savings + waterfall == d_variable_adjusted,
    remainder ≡ 0. Asserted every fixed-point iteration AND at convergence
    (Stage 3b gate).

    Scope: the remainder ≡ 0 invariant and the strict closure are properties of
    the AGGREGATED PRIMARY path — the down-sweep / up-waterfall that drive the
    remainder to 0 are gated to ``use_aggregated`` and ``solver_status ==
    'primary'``. The disaggregated A-B baseline never back-fills/sweeps (leftover
    slack is a legitimate non-zero remainder), and compressed/deficit paths have
    no slack to assign — so the assertion is a no-op off the aggregated primary
    path, which is exactly where the pre-tax fixed point's top-ups operate."""
    if not (use_aggregated and feasibility.solver_status == "primary"):
        return
    dva = float(feasibility.d_variable_adjusted)
    tol = max(_FOUR_WAY_TOL, abs(dva) * 1e-6)
    assert abs(float(residual_assignment.genuine_remainder)) <= tol, (
        f"remainder != 0 [{context}]: {residual_assignment.genuine_remainder:.2f}"
    )
    spend = sum(
        float(d.feasibility_adjusted or 0.0)
        + float(getattr(d, "backfill_inferred", 0.0) or 0.0)
        for d in final.values()
    )
    wf_total = float(savings_waterfall.total) if savings_waterfall.fired else 0.0
    closure = spend + float(residual_assignment.savings_investment) + wf_total
    assert abs(closure - dva) <= tol, (
        f"four-way closure broke [{context}]: spend {spend:.2f} + savings "
        f"{residual_assignment.savings_investment:.2f} + waterfall {wf_total:.2f} "
        f"= {closure:.2f} != d_variable_adjusted {dva:.2f}"
    )


def run_profile_analysis(
    profile: HouseholdProfile,
    filing_status: str = "single",
    city_pumas: list[str] | None = None,
    use_aggregated: bool = True,
) -> dict[str, Any]:
    """Run the full matching -> optimizer -> benefits chain.

    Args:
        profile: The querying household.
        filing_status: Tax filing status.
        city_pumas: Optional seed set of PUMAs from city->PUMA
            resolution. When non-empty, match_household uses this as
            the seed pool (plus similarity expansion on the first
            element). When omitted, match_household seeds from
            ``profile.puma_code`` alone.
        use_aggregated: When True (default), run the aggregated
            category set (4 merged groups + 35 retained = 39) using
            ``coefficients_aggregated.json``. When False, run the live
            55-category disaggregated path — the permanent fallback,
            kept byte-for-byte and reachable for debugging / A-B
            baseline. The two paths share all infrastructure; only the
            category set, coefficients file, and luxury table differ.

    Returns a JSON-ready dict. See the module docstring for the
    one-call contract this function provides to the DRF view.
    """
    artifacts = _artifacts_path()

    # Path-specific knobs. Disaggregated (default) passes None for the
    # optimizer overrides, preserving the live behavior exactly.
    if use_aggregated:
        coeff_path: str | None = _aggregated_coefficients_path(artifacts)
        category_codes: list[str] | None = AGGREGATED_CATEGORY_CODES
        luxury_categories: frozenset[str] | None = AGGREGATED_LUXURY_CATEGORIES
    else:
        coeff_path = None
        category_codes = None
        luxury_categories = None

    # 1. Matching on the synthetic population. MatchResult includes
    #    cohort_median_income used by the allocator's income-gap
    #    correction. aggregate=True folds the 24 merged members into 6
    #    aggregate distributions (W2 mode-split preserved).
    match = match_household(
        profile, artifacts, city_pumas=city_pumas, aggregate=use_aggregated
    )

    # 2. Current-price CPI scalars (cached 30 days, refreshed monthly).
    cpi_scalars = load_cpi_scalars(
        cache_path=os.path.join(artifacts, "cpi_scalars.json")
    )

    # 3. D_variable = take-home minus the committed-outflow block.
    #    Committed outflows are external-anchored flows that the donor structurally
    #    cannot see (retirement contributions: CEX 8XXXXX UCC group is pathologically
    #    under-collected per Bee/Mitchell 2018; health-premium employee-share: the
    #    UCC family is absent from the donor). They subtract pre-allocation via
    #    compute_d_variable's additional_committed slot — the same shape as
    #    debt_service, NOT a member of CATEGORY_CODES (fusion contract untouched).
    #    Predicted HIGH-SIDE (assume participation/enrollment): tighter-then-correctable.
    committed_outflows = estimate_committed_outflows(profile)
    # Filing-unit imputation (Stage 1 module, wired here): repair the filing
    # status + impute dependents/ages from householder age + household_size,
    # conservative on tax-reducing structure. The repaired status is used for
    # every downstream tax/benefit call; the unit is surfaced for user override.
    filing_unit = impute_filing_unit(profile, filing_status)
    filing_status = filing_unit.filing_status
    # Pre-tax wedge: the committed 401(k) / §125 / §132 lines reduce the taxable
    # wage bases so federal income tax + FICA are computed on the right base
    # (the committed dollars still subtract from the budget; the wedge only fixes
    # the tax). Committed baseline here; the waterfall-top-up feedback is the
    # Stage-3b fixed point (PRETAX-FEEDBACK-LOOP-FROM-WATERFALL).
    _wedge = pretax_wedge(committed_outflows)
    # Schedule A itemizables (Stage 5): predicted lean-low from the profile's
    # housing cost + the match-time cohort medians — LOOP-INVARIANT (no dependence
    # on the iterating allocation), so derived once here and fed unchanged into
    # every federal pass. The SALT income-tax line is filled inside compute_tax
    # from the computed state+local tax (ITEMIZABLE-PREDICTION-LEANS-LOW).
    _schedule_a = derive_schedule_a(profile, match.distributions)
    # 3b. Pre-tax fixed point: iterate wedge → d_variable → allocation →
    #     waterfall until take_home converges; the waterfall's pre-tax
    #     top-ups (401k/HSA) reduce taxable income on the next pass.
    def _allocation_pass(
        _pretax_topup_it: float, _framing_dva: float | None = None
    ) -> dict:
        tax_detail = TaxDetail(
            n_children_under_17=filing_unit.n_children_under_17,
            n_children_under_18=filing_unit.n_children_under_18,
            n_children_under_6=filing_unit.n_children_under_6,
            eic_children=filing_unit.eic_qualifying_children,
            pretax_income_tax_excludable=_wedge.income_tax_excludable,
            pretax_income_tax_excludable_topup=_pretax_topup_it,
            pretax_fica_excludable=_wedge.fica_excludable,
            itemized_mortgage_interest=_schedule_a.mortgage_interest,
            itemized_property_tax=_schedule_a.property_tax,
            itemized_charity=_schedule_a.charity,
            itemized_medical=_schedule_a.medical,
        )
        d_variable = compute_d_variable(
            profile,
            filing_status=filing_status,
            additional_committed=committed_outflows.total,
            num_dependents=filing_unit.num_dependents,
            detail=tax_detail,
        )

        # 4. Engel annotation — fills engel_estimate and is_structural on
        #    each SpendingDistribution. In the new pipeline this is
        #    metadata (useful for the frontend and for p50==0 fallback),
        #    not the primary allocation anchor.
        dists = annotate_distributions(
            match.distributions,
            equivalized_income=profile.equivalized_income,
            disposable_income=d_variable,
            artifacts_path=artifacts,
            coefficients_path=coeff_path,
        )

        # Geographic climate correction of the Engel fallback anchor. The utility
        # climate factor (EIA state consumption × NOAA within-state degree-days)
        # already rode the percentile/mean anchors through match_household, but
        # ``engel_estimate`` is computed *here* (post-match, from income) and the
        # allocator uses ``engel_estimate × cpi_scalar`` as the anchor for zero-median
        # cats (e.g. ``ofuel``). Re-apply the same value-layer climate factor to
        # ``engel_estimate`` so that fallback anchor is climate-corrected too —
        # mirroring the cpi_scalar the allocator already re-applies there. (Affects
        # only the zero-median climate cats; elec/ngas anchor on the already-scaled
        # cohort mean, so their engel_estimate is unused and unchanged in effect.)
        # NOTE: touches the locked-#4 value-layer/allocation path additively — flagged.
        _geo = GeographicDrivers(
            eia_utility_scalars=load_eia_utility_scalars(
                os.path.join(artifacts, "eia_utility_scalars.json")
            ),
            climate_normals=load_climate_normals(
                os.path.join(artifacts, "climate_normals.json")
            ),
        )
        dists = {
            cat: (
                replace(dist, engel_estimate=float(dist.engel_estimate) * gf)
                if (gf := _geo.factor(cat, profile.puma_code)) != 1.0 and dist.engel_estimate
                else dist
            )
            for cat, dist in dists.items()
        }

        # Omit-by-default value-layer rule (HEAVY-ZERO-DISTRIBUTION-NEEDS-ELICITATION):
        # zero the anchor-determining fields for the six cohort-mean-meaningless
        # categories (chrty/educ/ocash + stdint/othint/finpay) so the allocator
        # anchors them at $0 across every anchor-statistic bucket (weighted_mean /
        # trimmed_mean / p50 / engel_estimate), and the freed budget flows to the
        # four-way remainder. The cohort PERCENTILES (p10/p25/p75/p90/conditional_p90)
        # are intentionally retained for the future "see typical values" additive UX
        # (OMITTED-CATEGORY-ADDITIVE-UX). decompose guards the zero-width range, so the
        # zeroed anchor is numerically safe. These cats are surfaced with
        # ``omit_from_initial_view: True`` in the response (below). Applies on both the
        # aggregated (default) and disaggregated paths — all six are standalone codes
        # in each set, never merged into an aggregate.
        dists = {
            cat: (
                replace(dist, p50=0.0, engel_estimate=0.0,
                        weighted_mean=0.0, trimmed_mean=0.0)
                if cat in OMIT_BY_DEFAULT_CATEGORIES
                else dist
            )
            for cat, dist in dists.items()
        }
        lams = lambda_weights_all(
            profile.equivalized_income, artifacts, coefficients_path=coeff_path
        )

        # 5. Liability prediction — cohort p50 of the balance columns is the
        #    PRIOR; user-reported debt OVERRIDES it component-wise. p50 is $0
        #    for most households (73% of cohorts carry no student debt, etc.),
        #    so a no-debt-input profile falls back to the cohort prior and
        #    sees byte-identical debt-service to before this build. Post-
        #    allocation treatment (DEBT-POST-ALLOCATION-OVER-COHORT-SHIFT):
        #    debt-service subtracts from take-home after cohort matching, never
        #    shifts the cohort. This call MIRRORS the allocator's internal call
        #    (allocator.py) for the response display — both read profile.* so
        #    they agree on the same total.
        predicted_othdbt = float(match.distributions["othdbt"].p50)
        predicted_stddbt = float(match.distributions["stddbt"].p50)
        debt_service = estimate_annual_debt_service(
            credit_card_balance=predicted_othdbt,
            student_loan_balance=predicted_stddbt,
            cc_carried_balance=profile.cc_carried_balance,
            student_loan_payment=profile.student_loan_payment,
            auto_loan_payment=profile.auto_loan_payment,
            other_debt_payment=profile.other_debt_payment,
        )

        # 6. Allocation — primary path is CPI-scaled p50 + Engel gap
        #    correction + 1.05 UX bias; compression QP only runs if
        #    sum(anchors) > (d_variable - debt_service). Housing is
        #    pinned into the allocator via ``profile.housing_cost``
        #    rather than pre-subtracted from d_variable.
        feasibility = solve_feasibility(
            dists,
            profile,
            lams,
            d_variable,
            cpi_scalars=cpi_scalars,
            cohort_median_income=match.cohort_median_income,
            artifacts_path=artifacts,
            predicted_othdbt=predicted_othdbt,
            predicted_stddbt=predicted_stddbt,
            category_codes=category_codes,
            luxury_categories=luxury_categories,
            coefficients_path=coeff_path,
            transport_finance_mean=match.transport_finance_mean,
        )

        # 7. Annotate distributions with optimizer outputs.
        final = decompose(dists, feasibility.allocations)

        # 7b. Residual back-fill — the post-allocation REVERSE stage (Build 4).
        #     Redistributes an implausibly-large feasibility-slack residual into the
        #     slope-ceiling discretionary cats (entertainment/shopping/travel),
        #     conditioned on an income-realistic savings benchmark. Fires only on
        #     solver_status=="primary" (compression/deficit profiles are immune — the
        #     deficit/benefits branch owns those). Distinct layer from the value-layer
        #     factors; runs after composition is fully settled.
        #
        #     Gated to the AGGREGATED path only: the 55-cat disaggregated fallback is
        #     the permanent A-B baseline and stays byte-for-byte (it never back-fills).
        effective_slack = float(feasibility.feasibility_slack)
        backfill = compute_backfill(
            measured=feasibility.allocations,
            cohort=match.distributions,
            profile=profile,
            feasibility_slack=feasibility.feasibility_slack,
            solver_status=feasibility.solver_status,
            d_variable_adjusted=feasibility.d_variable_adjusted,
            artifacts_path=artifacts,
            coefficients_path=coeff_path,
            enabled=use_aggregated,
            # High-earner ceiling stratification (2026-06-09): at/above the
            # cohort-median y_eq the per-category cap reads the high-earner
            # conditional_p90_hi instead of the broad conditional_p90.
            cohort_median_y_eq=match.cohort_median_y_eq,
            # Pre-tax fixed point: pin the savings-rate benchmark to the
            # committed-baseline disposable income (None on pass 0 → live value)
            # so the waterfall's own pre-tax topup can't drift the benchmark
            # mid-loop. See assign_residual_savings below + the Stage-3b loop.
            framing_d_variable_adjusted=_framing_dva,
        )
        # Whether this profile read the high-earner stratified caps (drives the
        # backfill audit block + the remainder's MOE framing below).
        backfill_stratified = bool(
            match.cohort_median_y_eq > 0.0
            and float(profile.equivalized_income) >= match.cohort_median_y_eq
        )
        if backfill.fired:
            final = {
                cat: (
                    replace(
                        dist,
                        backfill_inferred=inc,
                        backfill_confidence="inferred-lifestyle",
                    )
                    if (inc := backfill.inferred.get(cat, 0.0)) > 0.0
                    else dist
                )
                for cat, dist in final.items()
            }
            effective_slack = float(backfill.new_slack)

        # 7c. Residual assignment (Build 6, 2026-05-28): label the post-back-fill
        #     residual as likely-savings (cohort-realistic, bounded by s*) + genuine
        #     remainder. Closes the every-dollar-accounted premise — together with
        #     committed_outflows (committed dollars) and back-fill (discretionary
        #     dollars), this assigns every dollar to a likely destination, all
        #     correctable, none prescriptive (NEUTRAL-FRAMING refinement 2026-05-28:
        #     predict-not-prescribe; the tool's job is dollar-accounting, not
        #     declining-to-predict on knowable-likely amounts).
        residual_assignment = assign_residual_savings(
            d_variable_adjusted=feasibility.d_variable_adjusted,
            post_backfill_slack=effective_slack,
            profile=profile,
            # Loop-invariant framing baseline (PRETAX-FRAMING-LOOP-INVARIANT):
            # the savings-regime decision (down-sweep / up-waterfall / taxable)
            # is taken against the topup-free committed-baseline disposable
            # income, NOT the live value that the up-waterfall's own pre-tax
            # 401(k)/HSA topup inflates each pass. Without this, a borderline
            # saver's regime flips on/off between passes (the waterfall's tax
            # break lowers s_user_implied below s_cohort, killing the waterfall,
            # which restores it next pass) and the fixed point 2-cycles with no
            # solution. None on pass 0 → uses its own (topup-free) live value.
            framing_d_variable_adjusted=_framing_dva,
        )

        # 7c2. Down-direction residual sweep (2026-06-10, locked
        #      REMAINDER-ZERO-INVARIANT-DOWN-DIRECTION): for the low-savings-
        #      contradiction case (framing_state == "signal_pulled_down" — the
        #      balance signal pulled the personalized rate below cohort), the
        #      would-be genuine_remainder is a systematic UNDER-prediction of
        #      spending (the harmful error direction). It sweeps entirely into
        #      the high-participation elastic sinks (ELASTIC_SINK_CATEGORIES),
        #      distributed by marginal income response (measured × ε), no ceiling
        #      blocking — remainder ≡ 0 by construction; the savings line (the
        #      blend) is untouched. No-contradiction / high-contradiction /
        #      non-primary / disaggregated paths are byte-identical no-ops.
        residual_sweep = sweep_remainder_to_sinks(
            remainder=residual_assignment.genuine_remainder,
            measured=feasibility.allocations,
            inferred=backfill.inferred,
            y_eq=float(profile.equivalized_income),
            framing_state=residual_assignment.framing_state,
            solver_status=feasibility.solver_status,
            artifacts_path=artifacts,
            coefficients_path=coeff_path,
            enabled=use_aggregated,
        )
        if residual_sweep.fired:
            final = {
                cat: (
                    replace(
                        dist,
                        backfill_inferred=dist.backfill_inferred + inc,
                        backfill_confidence="inferred-lifestyle",
                    )
                    if (inc := residual_sweep.swept.get(cat, 0.0)) > 0.0
                    else dist
                )
                for cat, dist in final.items()
            }
            # The swept dollars are now predicted spending; the leftover slack is
            # exactly the savings line (remainder ≡ 0).
            effective_slack -= residual_sweep.total
            residual_assignment = apply_remainder_sweep(residual_assignment, residual_sweep)

        # 7c3. Up-direction tax-advantaged savings waterfall (2026-06-10, locked
        #      REMAINDER-ZERO-INVARIANT-UP-DIRECTION): for the high-savings-
        #      contradiction case (signal_would_pull_up_deferred — the balance
        #      implies saving ABOVE cohort), the would-be genuine_remainder routes
        #      through the ordered statutory waterfall (401(k) headroom → IRA/
        #      backdoor Roth → HSA top-up → taxable terminal), supply-bounded,
        #      exact zero via the unbounded terminal. No-contradiction profiles
        #      (signal_confirmed_cohort) default to the taxable terminal ONLY (no
        #      statutory-vehicle assertion without balance evidence). The fills
        #      surface as predicted_topup on the committed lines at SERIALIZATION
        #      ONLY — a post-allocation transfer; they never re-enter
        #      compute_d_variable (no circularity). Mutually exclusive with the
        #      down-direction sweep by the framing gate (a profile is exactly one
        #      of pulled-down / pulled-up / confirmed-cohort). The savings blend
        #      (piece 3, up-capped) is untouched: route-the-remainder-not-the-blend.
        _committed_annual = {it.code: float(it.annual) for it in committed_outflows.items}
        savings_waterfall = route_residual_waterfall(
            remainder=residual_assignment.genuine_remainder,
            framing_state=residual_assignment.framing_state,
            solver_status=feasibility.solver_status,
            profile=profile,
            filing_status=filing_status,
            retirement_current=_committed_annual.get(COMMITTED_CODE_RETIREMENT, 0.0),
            hsa_current=_committed_annual.get(COMMITTED_CODE_HSA, 0.0),
            artifacts_path=artifacts,
            enabled=use_aggregated,
        )
        if savings_waterfall.fired:
            residual_assignment = apply_waterfall_fold(residual_assignment, savings_waterfall)
        return {
            "d_variable": d_variable,
            "tax_detail": tax_detail,
            "final": final,
            "feasibility": feasibility,
            "backfill": backfill,
            "backfill_stratified": backfill_stratified,
            "effective_slack": effective_slack,
            "residual_assignment": residual_assignment,
            "residual_sweep": residual_sweep,
            "savings_waterfall": savings_waterfall,
            "predicted_othdbt": predicted_othdbt,
            "predicted_stddbt": predicted_stddbt,
            "debt_service": debt_service,
            "committed_annual": _committed_annual,
        }

    # Drive the fixed point: the first pass (no top-ups) always runs; iterate
    # only while the waterfall keeps producing pre-tax top-ups, until take_home
    # converges. Four-way closes after EVERY pass (incl. the converged one);
    # non-convergence is a hard error, never a silently-accepted last value.
    def _assert_pass(_pass: dict, ctx: str) -> None:
        _assert_four_way_closes(
            _pass["final"], _pass["feasibility"], _pass["residual_assignment"],
            _pass["savings_waterfall"], context=ctx, use_aggregated=use_aggregated,
        )

    _pass = _allocation_pass(0.0)
    _assert_pass(_pass, "iteration 0")
    # The committed-baseline disposable income (pass 0 carries no waterfall
    # pre-tax topup) is the LOOP-INVARIANT reference for the savings-rate
    # framing. Holding it fixed across the iterations stops the up-waterfall's
    # own 401(k)/HSA pre-tax topup from flipping the savings regime mid-loop —
    # a discontinuity that otherwise admits no fixed point for borderline savers
    # (the take_home 2-cycles to non-convergence). The dollar amounts + tax
    # wedge still iterate on the live d_variable_adjusted; only the regime is
    # pinned. (PRETAX-FRAMING-LOOP-INVARIANT — see assign_residual_savings.)
    _framing_dva = float(_pass["feasibility"].d_variable_adjusted)
    _take_home = _pass["d_variable"] + committed_outflows.total
    _topup = _pretax_topup_income_tax_excludable(_pass["savings_waterfall"])
    _pretax_iterations = 1
    _pretax_converged = _topup <= _PRETAX_CONVERGE_TOL
    while not _pretax_converged and _pretax_iterations < _MAX_PRETAX_ITERS:
        _pass = _allocation_pass(_topup, _framing_dva)
        _assert_pass(_pass, f"iteration {_pretax_iterations}")
        _new_take_home = _pass["d_variable"] + committed_outflows.total
        _pretax_iterations += 1
        _tol = max(_PRETAX_CONVERGE_TOL, _PRETAX_CONVERGE_REL * abs(_new_take_home))
        _pretax_converged = abs(_new_take_home - _take_home) <= _tol
        _take_home = _new_take_home
        _topup = _pretax_topup_income_tax_excludable(_pass["savings_waterfall"])
    if not _pretax_converged:
        raise RuntimeError(
            f"pre-tax fixed point did not converge in {_MAX_PRETAX_ITERS} "
            f"iterations (take_home delta still > ${_PRETAX_CONVERGE_TOL:.0f}); "
            f"gross={profile.gross_income} puma={profile.puma_code!r} "
            f"size={profile.household_size}"
        )

    d_variable = _pass["d_variable"]
    final = _pass["final"]
    feasibility = _pass["feasibility"]
    backfill = _pass["backfill"]
    backfill_stratified = _pass["backfill_stratified"]
    effective_slack = _pass["effective_slack"]
    residual_assignment = _pass["residual_assignment"]
    residual_sweep = _pass["residual_sweep"]
    savings_waterfall = _pass["savings_waterfall"]
    predicted_othdbt = _pass["predicted_othdbt"]
    predicted_stddbt = _pass["predicted_stddbt"]
    debt_service = _pass["debt_service"]

    # 7d. Debt-accumulation annotation (soft-deficit-with-CC-debt regime,
    #     2026-06-09). When the soft-constraint optimizer compressed AND the
    #     user reports a carried CC balance, surface the gap (cohort-typical
    #     anchors − debt-adjusted budget) as a conditional, adjustable,
    #     out-of-sum annotation — "if your spending matches typical, ~$X/mo
    #     could add to your card balance". Solver behavior unchanged; the
    #     four-way keeps closing on the compressed allocation. No-op
    #     (applies=False) on primary / structural_deficit / floor_infeasible
    #     (the latter route to the deferred deficit/benefits handoff) and on
    #     no-CC-debt profiles. See debt_accumulation_prediction_scoping.md.
    debt_accumulation = project_debt_accumulation(
        profile,
        solver_status=feasibility.solver_status,
        compression_gap=feasibility.compression_gap,
        d_variable_adjusted=feasibility.d_variable_adjusted,
    )

    # 8. Benefits screen.
    benefits = screen(profile, filing_status=filing_status)

    # 9. Balance sheet — assets read from raw match distributions
    #    (not the tenure/balance-zeroed ``dists``) so the UI can show
    #    net-worth context even though these cats aren't allocated.
    raw = match.distributions
    balance_sheet = {
        "assets": {
            cat: {
                "cohort_p50": float(raw[cat].p50),
                "cohort_p75": float(raw[cat].p75),
                "cohort_p90": float(raw[cat].p90),
                "label": _CATEGORY_LABEL[cat],
                # ``check`` is the only asset the user reports
                # directly; others are cohort-estimated only. When
                # savings isn't sent (pre-wiring clients), this lands
                # as 0.0 — matches the serializer default.
                **(
                    {"user_reported": float(profile.savings)}
                    if cat == "check"
                    else {}
                ),
            }
            for cat in sorted(ASSET_CATEGORIES)
        },
        # Per-debt-type service lines. ``source`` per component:
        #   user_reported    — the user supplied the input; service derives from it
        #   cohort_predicted — no user input; cohort-balance fallback (mostly $0 p50)
        #   not_modeled      — no user input and no cohort fallback (auto/other → $0)
        # All adjustable: editing a debt input is an INPUT edit that
        # re-predicts downstream (USER-ADJUSTMENT-AUTHORITY: the conditional
        # fact changed), not a category pin. Credit cards carry a balance;
        # the other three carry a monthly payment.
        # ``omit_from_initial_view`` = (source != "user_reported"): the initial
        # view shows ONLY user-reported debt. cohort_predicted lines are omitted
        # (the cohort othdbt/stddbt are heavy-zero at p50=$0 and, when the user
        # DID report debt, redundant with the user-reported line); not_modeled
        # lines are omitted (zero, no input). The future additive UX surfaces them.
        "liabilities": {
            "othdbt": {
                "predicted_balance": predicted_othdbt,
                "reported_balance": float(profile.cc_carried_balance),
                "annual_service": float(debt_service["credit_card_service"]),
                "monthly_service": float(debt_service["credit_card_service"]) / 12.0,
                "cohort_p50": float(raw["othdbt"].p50),
                "cohort_p75": float(raw["othdbt"].p75),
                "cohort_p90": float(raw["othdbt"].p90),
                "label": "Credit card debt",
                "source": (
                    "user_reported" if profile.cc_carried_balance > 0
                    else "cohort_predicted"
                ),
                "adjustable": True,
                "omit_from_initial_view": profile.cc_carried_balance <= 0,
            },
            "stddbt": {
                "predicted_balance": predicted_stddbt,
                "reported_monthly_payment": float(profile.student_loan_payment),
                "annual_service": float(debt_service["student_loan_service"]),
                "monthly_service": float(debt_service["student_loan_service"]) / 12.0,
                "cohort_p50": float(raw["stddbt"].p50),
                "cohort_p75": float(raw["stddbt"].p75),
                "cohort_p90": float(raw["stddbt"].p90),
                "label": "Student loan debt",
                "source": (
                    "user_reported" if profile.student_loan_payment > 0
                    else "cohort_predicted"
                ),
                "adjustable": True,
                "omit_from_initial_view": profile.student_loan_payment <= 0,
            },
            "auto_loan": {
                "reported_monthly_payment": float(profile.auto_loan_payment),
                "annual_service": float(debt_service["auto_loan_service"]),
                "monthly_service": float(debt_service["auto_loan_service"]) / 12.0,
                "label": "Auto loan",
                "source": (
                    "user_reported" if profile.auto_loan_payment > 0
                    else "not_modeled"
                ),
                "adjustable": True,
                "omit_from_initial_view": profile.auto_loan_payment <= 0,
            },
            "other_debt": {
                "reported_monthly_payment": float(profile.other_debt_payment),
                "annual_service": float(debt_service["other_debt_service"]),
                "monthly_service": float(debt_service["other_debt_service"]) / 12.0,
                "label": "Other debt",
                "source": (
                    "user_reported" if profile.other_debt_payment > 0
                    else "not_modeled"
                ),
                "adjustable": True,
                "omit_from_initial_view": profile.other_debt_payment <= 0,
            },
        },
        "total_debt_service_annual": float(debt_service["total_debt_service"]),
    }

    # Tax breakdown (Stage 6): the converged gross->take-home wedge, recomputed
    # from the final pass's tax_detail (committed-baseline wedge + converged
    # waterfall top-up + Schedule A) for display. Additive/display-only — the
    # closure-bearing take_home is unchanged; same per-household authz scope as the
    # financial blocks already surfaced (committed_outflows, balance_sheet, ...).
    from models.tax.calculator import compute_tax as _compute_tax

    _tax_breakdown = _compute_tax(
        gross_income=profile.gross_income,
        filing_status=filing_status,
        puma_code=profile.puma_code,
        num_dependents=filing_unit.num_dependents,
        place_fips=profile.place_fips,
        county_fips=profile.county_fips,
        detail=_pass["tax_detail"],
    )

    # 10. Serialize.
    return {
        "financial_zone": feasibility.financial_zone.value,
        "structural_deficit": float(feasibility.structural_deficit),
        # feasibility_slack / pace_annual are the residual AFTER the back-fill
        # reverse stage redistributes its implausible excess into discretionary
        # categories (== feasibility.feasibility_slack when the back-fill doesn't
        # fire). The spending lines carry the back-fill increment in
        # ``backfill_inferred``; the two reconcile to ``d_variable_adjusted``.
        "feasibility_slack": effective_slack,
        "d_variable_annual": float(d_variable),
        "d_variable_adjusted": float(feasibility.d_variable_adjusted),
        "debt_service_annual": float(feasibility.debt_service),
        # At onboarding we don't have an active paycheck; the annual
        # pace proxy is the feasibility slack — dollars we expect to
        # carry forward across the year if spending follows the
        # feasibility allocation.
        "pace_annual": effective_slack,
        "solver_status": feasibility.solver_status,
        # Pre-tax fixed-point convergence (Stage 3b): how many passes the
        # tax ↔ take_home ↔ allocation loop took to converge (1 = no pre-tax
        # waterfall top-ups; the loop hard-errors rather than reach here without
        # converging, so ``converged`` is always True in a returned response).
        "tax_convergence": {
            "iterations": _pretax_iterations,
            "converged": _pretax_converged,
        },
        # Gross->take-home tax wedge (Stage 6), oracle-verified (Gate 1). Rich
        # display: federal (net of Additional Medicare), AMT/NIIT/EITC/CTC/AGI,
        # state income (net of state EITC), local, FICA, state payroll, itemized-vs-
        # standard, effective rate. Closure-bearing totals unchanged.
        "tax_breakdown": asdict(_tax_breakdown),
        # Imputed filing unit (Stage 1) — surfaced so the user can correct it
        # (USER-ADJUSTMENT-AUTHORITY); every downstream tax call used this unit.
        "filing_unit": asdict(filing_unit),
        # Committed-outflow block (Build 5) — external-anchored flows that subtract
        # from take-home pre-allocation (retirement contribution, health-premium
        # employee share). NOT in CATEGORY_CODES (not fusion targets); NOT in
        # display_rollup (the spending categorization). Surfaced here so the user
        # sees them as distinct committed lines, prominently adjustable via the
        # Q7 fixity helper apply_committed_outflow_overrides. Neutral framing —
        # never prescriptive.
        "committed_outflows": {
            "items": [
                {
                    "code": it.code,
                    "label": it.label,
                    "annual": float(it.annual),
                    "monthly": float(it.annual) / 12.0,
                    "source": it.source,
                    "adjustable": it.adjustable,
                    # Display-partition flag (Stage 6 take-home fix): pre-tax
                    # outflows (reduce AGI) render in the tax wedge between taxes
                    # and take-home; post-tax ones (supplemental) stay in the
                    # four-way. The pre-tax items' display_annual sums to gross−AGI.
                    "pre_tax": is_pre_tax(it.code),
                    # Up-direction waterfall top-up (7c3) — the POST-ALLOCATION
                    # transfer: the displayed line is annual + predicted_topup
                    # (e.g. retirement shown at the maxed elective deferral),
                    # but ``annual`` (the allocation-time prediction) is what
                    # entered compute_d_variable — the top-up never feeds back
                    # (no circularity). 0.0 whenever the waterfall didn't fire
                    # or didn't reach this line.
                    "predicted_topup_annual": float(
                        savings_waterfall.topups_by_committed_code.get(it.code, 0.0)
                    ),
                    "predicted_topup_monthly": float(
                        savings_waterfall.topups_by_committed_code.get(it.code, 0.0)
                    ) / 12.0,
                    "display_annual": float(it.annual)
                    + float(savings_waterfall.topups_by_committed_code.get(it.code, 0.0)),
                }
                for it in committed_outflows.items
            ],
            # ``total_annual`` is the allocation-time committed subtraction
            # (the d_var input — unchanged by the waterfall);
            # ``total_with_topups_annual`` is the displayed committed total.
            "total_annual": float(committed_outflows.total),
            "total_with_topups_annual": float(committed_outflows.total)
            + sum(savings_waterfall.topups_by_committed_code.values()),
        },
        # Residual assignment (Build 6, 2026-05-28) — labels the post-back-fill
        # residual as likely-savings + genuine remainder. Completes the
        # every-dollar-accounted premise: committed + discretionary + savings +
        # remainder = take-home. Neutral-framing refinement: this is PREDICTION
        # (where dollars likely go), not PRESCRIPTION (what the user should do).
        # User-overridable via apply_savings_override (Q7 fixity); remainder
        # absorbs the delta 1:1, no sympathetic shift on discretionary/committed.
        "residual_assignment": {
            "savings_investment": {
                "annual": float(residual_assignment.savings_investment),
                "monthly": float(residual_assignment.savings_investment) / 12.0,
                "label": "Estimated savings / investment (cohort-typical at your income, adjustable)",
                "source": residual_assignment.source,
                "adjustable": True,
                # Drives state-dependent display copy (predict-not-prescribe):
                # signal_confirmed_cohort / signal_pulled_down /
                # signal_would_pull_up_deferred / user_pinned. Set by
                # assign_residual_savings from the balance-signal blend
                # (apply_savings_override sets user_pinned).
                "framing_state": residual_assignment.framing_state,
            },
            "genuine_remainder": {
                "annual": float(residual_assignment.genuine_remainder),
                "monthly": float(residual_assignment.genuine_remainder) / 12.0,
                # MOE framing (stratification build, 2026-06-09): when the
                # back-fill deployed against the high-earner stratified caps,
                # the surviving remainder is the honest cap-bounce — dollars
                # beyond what even the high-earning sub-cohort typically
                # deploys. Described, never prescribed (USER-NEUTRAL-FRAMING);
                # the user can pin any category (USER-ADJUSTMENT-AUTHORITY).
                # Swept/routed case (remainder ≡ 0): there is no remainder
                # line to frame — the dollars are predicted spending on the
                # elastic sinks (down-sweep) or routed savings vehicles
                # (waterfall). USER-NEUTRAL-FRAMING: nothing to apologize for
                # or hand to the user; the money is allocated.
                "label": (
                    "Fully allocated — every dollar accounted for"
                    if residual_sweep.fired or savings_waterfall.fired
                    else "Beyond what high-earning peers at your level typically "
                    "deploy on entertainment, dining, shopping, and travel — "
                    "your call"
                    if backfill.fired and backfill_stratified
                    else "Uncommitted income (could go to extra savings, large purchases, buffer)"
                ),
            },
            "realistic_savings_rate": float(residual_assignment.realistic_savings_rate),
            "realistic_savings_cap": float(residual_assignment.realistic_savings_dollars),
        },
        # Debt-accumulation annotation (7d) — conditional and OUT of the
        # four-way sum (a counterfactual: "cohort-typical spending would run
        # $X above your post-debt budget"; the compressed allocation already
        # sums to d_variable_adjusted, so no dollar is counted twice).
        # framing_state drives the predict-not-prescribe display copy:
        # signal_clear / signal_marginal / user_pinned (set by
        # apply_debt_accumulation_override, the Q7 fixity). Full honest gap,
        # no confidence haircut — the hedge lives in the framing.
        "debt_accumulation": (
            {
                "applies": True,
                "monthly_potential_growth": float(
                    debt_accumulation.monthly_potential_growth
                ),
                "annual_potential_growth": float(
                    debt_accumulation.annual_potential_growth
                ),
                "basis": debt_accumulation.basis,
                "source": debt_accumulation.source,
                "framing_state": debt_accumulation.framing_state,
                "adjustable": debt_accumulation.adjustable,
                "cc_balance_to_income": float(
                    debt_accumulation.cc_balance_to_income
                ),
                "gap_ratio": float(debt_accumulation.gap_ratio),
            }
            if debt_accumulation.applies
            else {"applies": False}
        ),
        # Back-fill audit (dev view) — neutral/internal; the user-facing display
        # surfaces only the measured/inferred split + neutral "estimated lifestyle
        # range" language (Q6). Never a prescription or a savings comparison.
        "backfill": {
            "fired": backfill.fired,
            "s_star": float(backfill.s_star),
            # The personalized (blended) benchmark the trigger/pool actually
            # used (spend-arm build, 2026-06-09): == s_star with no balance
            # signal or up-capped high balances; below it when a low balance
            # pulled the benchmark down (larger pool, earlier trigger).
            "s_star_personalized": float(backfill.s_star_personalized),
            "residual_rate": float(backfill.residual_rate),
            "g": float(backfill.g),
            "pool": float(backfill.pool),
            "inferred": {c: float(v) for c, v in backfill.inferred.items()},
            "feasibility_slack_pre_backfill": float(feasibility.feasibility_slack),
            # High-earner ceiling stratification (2026-06-09): the threshold
            # (the matched pool's median y_eq) + whether this profile read
            # the high-earner conditional_p90_hi caps.
            "cohort_median_y_eq": float(match.cohort_median_y_eq),
            "stratified": backfill_stratified,
        },
        # Down-direction residual-sweep audit (dev view, 2026-06-10). When
        # fired, the per-sink increments are already folded into the
        # distributions' ``backfill_inferred`` (the categories just show
        # higher values — Phase-4 framing: no remainder narration, the
        # money is allocated); this block carries the audit split.
        "residual_sweep": {
            "fired": residual_sweep.fired,
            "trigger": residual_sweep.trigger,
            "total": float(residual_sweep.total),
            "swept": {c: float(v) for c, v in residual_sweep.swept.items()},
        },
        # Up-direction savings-waterfall block (7c3). Descriptive framing
        # (predict-not-presume): "your balance suggests you're likely maxing
        # tax-advantaged retirement and saving the rest" — the model's
        # prediction of where a high-saver's surplus goes; every line stays
        # adjustable (USER-ADJUSTMENT-AUTHORITY — correct it if some is
        # spending or a different vehicle). The committed lines carry the
        # 401(k)/HSA fills as predicted_topup; the IRA + taxable fills live
        # only here (no committed counterpart). Σ fills == the routed
        # remainder exactly (taxable terminal — exact zero).
        "savings_waterfall": {
            "fired": savings_waterfall.fired,
            "trigger": savings_waterfall.trigger,
            "total": float(savings_waterfall.total),
            "limits_year": int(savings_waterfall.limits_year),
            "fills": [
                {
                    "code": f.code,
                    "label": f.label,
                    "annual": float(f.annual),
                    "monthly": float(f.annual) / 12.0,
                    "current": float(f.current),
                    "limit": float(f.limit),
                    "headroom": float(f.headroom),
                    "maxed": f.maxed,
                    "mechanism": f.mechanism,
                    "adjustable": f.adjustable,
                }
                for f in savings_waterfall.fills
            ],
        },
        # ``omit_from_initial_view`` flags the six heavy-zero, cohort-mean-
        # meaningless categories (value-layer-zeroed above). The frontend renders
        # unflagged categories initially and surfaces flagged ones via the future
        # additive UX (OMITTED-CATEGORY-ADDITIVE-UX). Percentiles are retained on
        # the record for that future "see typical values" affordance.
        "distributions": {
            cat: {
                **asdict(dist),
                "omit_from_initial_view": cat in OMIT_BY_DEFAULT_CATEGORIES,
            }
            for cat, dist in final.items()
        },
        "balance_sheet": balance_sheet,
        "benefits": [asdict(b) for b in benefits],
        "match_metadata": {
            "n_effective": float(match.n_effective),
            "kernel_n_effective": float(match.kernel_n_effective),
            "weight_cv": float(match.weight_cv),
            "confidence": match.confidence,
            "pumas_used": list(match.pumas_used),
            "city_pumas_used": list(match.city_pumas_used),
            "n_households": int(match.n_households),
            "car_owner_probability": float(match.car_owner_probability),
            "car_owner_classification": match.car_owner_classification,
            "cohort_mean_veh": (
                float(match.cohort_mean_veh)
                if match.car_owner_classification == "owner"
                else None
            ),
        },
    }


def save_profile_analysis(
    user: Any,
    profile: HouseholdProfile,
    analysis: dict[str, Any],
    filing_status: str = "single",
) -> None:
    """Persist profile + analysis results to the database.

    Creates or updates the household profile row, replaces the user's
    benefit-eligibility records with the fresh screen, and ensures a
    ``BufferBalance`` row exists. ``Paycheck`` rows are created
    separately — this is onboarding-only persistence.

    No-op (returns silently) when ``user`` is ``None`` so anonymous
    analysis can share this module without conditional logic.
    """
    if user is None:
        return

    from benefits.models import BenefitEligibilityRecord
    from profiles.models import HouseholdProfileRecord
    from tracking.models import BufferBalance

    HouseholdProfileRecord.objects.update_or_create(
        user=user,
        defaults=dict(
            age=profile.age,
            gross_income=profile.gross_income,
            puma_code=profile.puma_code,
            tenure=profile.tenure.value,
            housing_cost=profile.housing_cost,
            household_size=profile.household_size,
            filing_status=filing_status,
            financial_zone=analysis.get("financial_zone"),
        ),
    )

    # Replace prior benefit screen rather than accumulating stale
    # matches across re-screens.
    BenefitEligibilityRecord.objects.filter(user=user).delete()
    for b in analysis.get("benefits", []):
        BenefitEligibilityRecord.objects.create(
            user=user,
            program_name=b["program_name"],
            estimated_monthly_min=b["estimated_monthly_min"],
            estimated_monthly_max=b["estimated_monthly_max"],
            confidence=b["confidence"],
            enrollment_url=b["enrollment_url"],
        )

    BufferBalance.objects.get_or_create(user=user, defaults={"balance": 0.0})
