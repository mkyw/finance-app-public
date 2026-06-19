"""Committed-outflow estimation — external-anchored, NOT fusion targets (Build 5).

The CEX fusion donor structurally cannot see two major committed monthly outflows:

  (1) **Retirement contributions** (401k/IRA employee deferral). The CEX 8XXXXX
      "Personal insurance and pensions" UCC group is **not in the 55-cat target**;
      the CEX questionnaire structurally under-collects pre-tax payroll deductions
      (Bee & Mitchell 2018, BLS CARRA WP — only ~39% of over-65 CU's report
      retirement income vs ~73% in IRS 1099-R admin records; contributions worse
      because respondents don't perceive deductions as expenditures).
  (2) **Health-insurance premium employee share** (employer-deducted). The donor
      ``health`` category captures OOP medical + the 2024-added Medicare-premium
      UCCs (retiree-relevant), but **not** the employer-deducted premium share.

Both are added here as **external-anchored flows that subtract from take-home
pre-allocation**, via ``compute_d_variable``'s ``additional_committed`` slot
(see ``models/pace/calculator.py``) — the same pattern as ``debt_service`` and
the established mechanism for an external/derived flow that:

  - subtracts from ``d_variable`` before the allocator runs,
  - is NOT a member of ``CATEGORY_CODES`` (the 55-cat fusion contract is
    untouched — these are *not* fusion targets, not Engel-corrected, not
    matched, not anchored on cohort percentiles),
  - surfaces as a distinct response block (sibling of ``balance_sheet``),
    NOT folded into the spending ``display_rollup``.

This is the **gradient branch** of the locked PCE-ANCHORING-PRINCIPLE (the
saturation/flat branch was streaming): pathological capture (✓ ≫ gap from
Bee/Mitchell + a UCC the donor structurally omits), externally-derivable
age × income gradient (✓ Vanguard HAS admin grids; KFF EHBS coverage tiers),
not-a-repairable-pipeline-bug (✓ unlike DIARY-INTEGRATION-B's repairable
donor-coverage gap — the CEX instrument is the wrong measurement device for
this category, not just under-fused).

OVER-PREDICTION DIRECTION (extends the 1.05 UX bias of the allocator and the
back-fill's measured/inferred over-prediction to committed outflows): predict
cohort-typical **assuming participation/enrollment**, let users adjust down
to their actual. Tighter-then-correctable, not looser-then-painful — adjusting
*down* GROWS the residual (the ego-win mechanic; e.g. a non-contributor sets
retirement to $0 → residual grows by the prediction).

NEUTRAL FRAMING (USER-NEUTRAL-FRAMING, mandatory): present as "estimated
contribution at cohort levels, adjust to your actual" — informational, NEVER
prescriptive. The "room to contribute more" signal stays as opportunity (the
grown-slack-on-correction), never as "you should contribute $X."

BUILD A (2026-05-29) — APPLICABILITY CONDITIONING of the four small
low-participation lines, which previously fired flat cohort-averages that
described a "fake-individual" no real household holds:
  - **HSA + traditional FSA collapsed** into one ``pretax_health_savings``
    line predicted as the UNION (not the sum) — they are mutually exclusive by
    IRS §223, so summing two population-means emitted an illegal pair. The
    user attributes the split via the override path. This is the locking
    instance of [[COHORT-AVERAGE-RESPECTS-MUTUAL-EXCLUSION]] (collapse-and-
    attribute form). Dependent-care FSA stays a separate line (it is NOT
    mutually exclusive with the HSA).
  - **Commuter** conditioned on car ownership (``owns_car``): car-owner → ~$0,
    non-owner → the genuine §132(f) transit-user set-aside. Defaults to
    near-zero (national car-owner prior) until A1/Build B elicits ``owns_car``.
  - **Supplemental life/disability** conditioned on age × household_size:
    near-zero for young childless singles, rising with age and dependents.
The high-participation lines (retirement, health premium) are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.types import HouseholdProfile

# --------------------------------------------------------------------------- #
# Retirement contribution — Vanguard *How America Saves 2025* (2024 plan-year   #
# data; published 2025). PDF at                                                 #
# ``pipeline/export/committed_outflows/Vanguard How America Saves (2025).pdf``  #
# Tables extracted: Figure 28 (participation by income), Figure 37 (deferral    #
# rate by income; deferral rate by age).                                        #
# --------------------------------------------------------------------------- #

# Figure 37, "Employee-elective deferral rates by participant demographics" —
# the 2023 column (most-recent non-estimated). Among PARTICIPANTS only (the
# high-side anchor: predict as-if-participant, let non-contributors adjust down).
# Income bins are Vanguard's own.
_DEFERRAL_BY_INCOME: tuple[tuple[float, float], ...] = (
    (15_000.0,  0.072),
    (30_000.0,  0.060),
    (50_000.0,  0.057),
    (75_000.0,  0.068),
    (100_000.0, 0.080),
    (150_000.0, 0.091),
    (float("inf"), 0.085),
)

# Figure 37, age × deferral rate (2023). Used multiplicatively against the
# income-based base via ``_OVERALL_DEFERRAL``: age multiplier =
# deferral_age / overall. Approximation absent the joint age × income table;
# defensible because age and income effects are largely independent in Vanguard
# admin data.
_DEFERRAL_BY_AGE: tuple[tuple[float, float], ...] = (
    (25.0,  0.056),
    (35.0,  0.068),
    (45.0,  0.075),
    (55.0,  0.085),
    (65.0,  0.093),
    (float("inf"), 0.085),
)
_OVERALL_DEFERRAL: float = 0.077    # Figure 37 "All" row, 2023.

# Figure 28 participation rates kept here for the audit block — NOT applied as
# a blend (the high-side principle is "predict assuming participation"; the
# unblended deferral × salary IS the cohort-typical-among-participants
# prediction).
_PARTICIPATION_BY_INCOME: tuple[tuple[float, float], ...] = (
    (15_000.0,  0.31),
    (30_000.0,  0.49),
    (50_000.0,  0.74),
    (75_000.0,  0.86),
    (100_000.0, 0.88),
    (150_000.0, 0.90),
    (float("inf"), 0.95),
)


def _band_lookup(value: float, bands: tuple[tuple[float, float], ...]) -> float:
    """Pick the rate for the first band whose upper bound covers ``value``."""
    for ub, rate in bands:
        if value <= ub:
            return rate
    return bands[-1][1]  # defensive


def vanguard_participation_rate(gross_income: float) -> float:
    """Vanguard HAS 2025 Figure 28 participation rate for this income band.

    Exposed for the audit / dev view; NOT applied to the contribution estimate
    (the high-side principle is "predict assuming participation").
    """
    return _band_lookup(float(gross_income), _PARTICIPATION_BY_INCOME)


def estimate_retirement_contribution(age: float, gross_income: float) -> float:
    """Annual retirement contribution ($/yr), cohort-typical-assuming-participation.

    Multiplicative joint model from Vanguard HAS 2025 marginal tables:

        deferral_rate(age, income) = deferral_income(income)
                                    * deferral_age(age) / overall_deferral
        annual_contribution        = deferral_rate * gross_income

    High-side direction (per the unified over-prediction principle): predicts
    the level a *participant* contributes at this age × income; non-participants
    adjust to $0 → residual grows by this amount (the ego-win). For low-income
    profiles (<$30k), Vanguard participation is itself ~30–50%, so the unblended
    high-side runs above the population-average expected contribution — this is
    correct (it surfaces a tight profile honestly; the user adjusts down).

    Returns 0 for non-positive income.
    """
    if gross_income <= 0:
        return 0.0
    d_income = _band_lookup(float(gross_income), _DEFERRAL_BY_INCOME)
    d_age = _band_lookup(float(age), _DEFERRAL_BY_AGE)
    rate = d_income * (d_age / _OVERALL_DEFERRAL)
    return rate * float(gross_income)


# --------------------------------------------------------------------------- #
# Health-insurance premium employee share — KFF *Employer Health Benefits      #
# Survey 2024*. Tables at                                                       #
# ``pipeline/export/committed_outflows/KFF Tables-EHBS-2024-Section-06.xlsx``.  #
# Figure 6.4 (Single Coverage, 2024 worker contribution = $1,368) and Figure    #
# 6.5 (Family Coverage, 2024 worker contribution = $6,296). KFF means among     #
# COVERED workers in employer-sponsored insurance — the high-side anchor (predict #
# as-if-enrolled, let uninsured / off-employer-plan users adjust down to $0).   #
# --------------------------------------------------------------------------- #

_KFF_PREMIUM_SINGLE_ANNUAL: float = 1_368.0   # Figure 6.4, 2024
_KFF_PREMIUM_FAMILY_ANNUAL: float = 6_296.0   # Figure 6.5, 2024


def estimate_health_premium_employee_share(household_size: int) -> float:
    """Annual employee-share employer-sponsored health-insurance premium ($/yr).

    Coverage tier driven by household_size:
      - 1   -> single coverage  ($1,368/yr in 2024 = ~$114/mo)
      - 2+  -> family coverage  ($6,296/yr in 2024 = ~$525/mo)

    "household_size" here is the model's household composition input — NOT
    housing tenure. The audit's recommendation said "tenure" but meant the
    coverage tier (single vs family), driven by who's on the plan; the model
    uses ``tenure`` exclusively for housing (RENT/OWN), so this function reads
    household_size.

    High-side direction: predict the full coverage tier average, let
    uncovered / out-of-pocket-only users adjust down. KFF take-up among
    offered is ~75–85%; the unblended average is the high-side anchor — this is
    a HIGH-participation category, so as-if-enrolled ≈ population cohort-mean.

    No coverage gradient for the employee-plus-1 tier (KFF reports it but the
    binary single-vs-family split captures the dominant signal; can refine later
    if a 2-person profile demands the middle tier).

    Returns 0 for non-positive household_size.
    """
    if household_size <= 0:
        return 0.0
    return _KFF_PREMIUM_FAMILY_ANNUAL if household_size >= 2 else _KFF_PREMIUM_SINGLE_ANNUAL


# --------------------------------------------------------------------------- #
# Low-participation committed outflows (HSA / FSA / commuter / supplemental).  #
# Per [[OVER-PREDICTION-EXTENDED-TO-COMMITTED-OUTFLOWS]] operational refinement #
# (Part-B B2, 2026-05-27): for LOW-participation categories, cohort-typical    #
# uses participation-weighted = participation × among-participant contribution  #
# (= the honest population cohort-mean even when small), NOT as-if-enrolled.    #
# As-if-enrolled here would over-predict for the majority who don't have the   #
# benefit and risk tipping tight profiles into false compression. Same         #
# principle (cohort-typical → understate slack → corrections grow residual);   #
# the operational form scales with participation rate.                          #
#                                                                              #
# Sources cited per category. All values are 2024 annual $; income/size        #
# conditioning kept simple (banded lookups) — refine when build experience     #
# argues for it.                                                                #
# --------------------------------------------------------------------------- #

# Tax-advantaged health savings — the UNION of HSA + traditional health FSA,
# which are mutually exclusive by IRS §223 / Pub 969 (an HDHP-gated HSA and a
# general-purpose health FSA cannot both be held by one individual). Predicted
# as the dominant-vehicle population mean, NOT the sum of two independent
# population-means: summing double-counts the same pre-tax-health-saving
# propensity across the disjoint HSA-only and FSA-only sub-populations,
# emitting a combined figure no individual could legally hold. The HSA arm
# dominates the FSA arm in every income band (HSA: KFF EHBS 2024 Fig 8.4 27%
# HDHP enrollment + EBRI/Devenir 2024 ~$2,000/yr among funders; traditional
# FSA: BLS NCS 2024 ~12-15% take-up × ~$1,200/yr), so the union ≈ the HSA-arm
# population mean — at/above the larger arm, well below the two arms summed.
# The user attributes the predicted dollars to whichever vehicle they use (HSA
# / traditional FSA / limited-purpose FSA) via the override path; the model
# carries one combined line. The limited-purpose-FSA-coexists-with-HSA case is
# too niche to model separately (handled by override). This is the locking
# instance of [[COHORT-AVERAGE-RESPECTS-MUTUAL-EXCLUSION]] (collapse-and-attribute).
_PRETAX_HEALTH_BY_INCOME: tuple[tuple[float, float], ...] = (
    (30_000.0,   60.0),    # ~$5/mo
    (60_000.0,  180.0),    # ~$15/mo
    (100_000.0, 360.0),    # ~$30/mo
    (150_000.0, 600.0),    # ~$50/mo
    (float("inf"), 960.0), # ~$80/mo
)

# Dependent-care FSA: IRS limit $5,000/yr. Take-up restricted to households
# with dependent-care expenses (kids <13 or elder care). Among eligible
# families, take-up ~10-15% at avg ~$3,500/yr. Conditioned on household_size
# >= 3 as proxy for "has children." For households without dependent-care:
# $0 (the right answer; the override path handles edge cases like single-parent
# size=2 with kids if the user adjusts). Source: BLS NCS 2024; IRS Pub 503.
_DEPCARE_FSA_ANNUAL_FAMILY: float = 480.0   # ~$40/mo population-blended for families with kids

# Commuter benefits: IRS §132(f), 2024 limit $315/mo each for transit and
# parking. Requires the user to ACTUALLY commute by transit / qualified parking
# AND an employer offering it — so it is conditioned on car ownership (a
# car-owner has ~no pre-tax transit need; a non-owner is the genuine §132(f)
# user). The values below are the RESOLVED-NON-OWNER among-user set-aside
# (rises with income — longer/pricier commutes). When owns_car is unknown
# (Build A, pre-A1-elicitation) the default leans car-owner (national ~91%
# ownership) → near-zero; Build B's elicited owns_car activates this non-owner
# path. Deliberately NOT keyed on the cohort car_owner_probability (the
# unreliable "41%-Chicago" signal A1 fixes). Source: BLS NCS; APTA transit cost.
_COMMUTER_TRANSIT_USER_BY_INCOME: tuple[tuple[float, float], ...] = (
    (30_000.0,  240.0),    # ~$20/mo
    (60_000.0,  360.0),    # ~$30/mo
    (100_000.0, 480.0),    # ~$40/mo
    (float("inf"), 600.0), # ~$50/mo
)

# Supplemental life + disability (employee-share voluntary premiums, add-on to
# employer basic). Basic life/disability is typically employer-paid (~$0
# employee). Voluntary supplemental is bought disproportionately by OLDER users
# (rising life-insurance need) and users with DEPENDENTS (protect the family) —
# so a 25-yo childless single carries ~$0-7/mo, not the old flat $20/mo.
# Conditioned on age (band table, size-1 base) × a household-size multiplier.
# Source: BLS NCS 2024 voluntary-benefit take-up by age; LIMRA life-insurance
# ownership by age / family status.
_SUPPLEMENTAL_BY_AGE_ANNUAL: tuple[tuple[float, float], ...] = (
    (30.0,   48.0),    # <=30  ~$4/mo  young → near-zero
    (40.0,  120.0),    # <=40  ~$10/mo
    (50.0,  240.0),    # <=50  ~$20/mo
    (60.0,  336.0),    # <=60  ~$28/mo
    (float("inf"), 300.0),  # 60+  ~$25/mo (eases past peak earning years)
)
_SUPPLEMENTAL_SIZE_MULT: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),          # single → base
    (2.0, 1.3),          # couple
    (float("inf"), 1.6), # 3+ (dependents) → highest supplemental-life need
)


def estimate_pretax_health_savings(gross_income: float) -> float:
    """Annual tax-advantaged health savings ($/yr) — the HSA ⊎ traditional-FSA union.

    Single combined line for the two mutually-exclusive pre-tax-health vehicles
    (HSA xor traditional health FSA per IRS §223). Predicted as the UNION
    expectation (the dominant-vehicle population mean), NOT the sum of two
    independent population-means — summing double-counts the same set-aside
    propensity across the disjoint sub-populations and emits a combination no
    individual could legally hold. The user attributes the predicted dollars to
    whichever vehicle they actually use via ``apply_committed_outflow_overrides``.
    See ``_PRETAX_HEALTH_BY_INCOME``; [[COHORT-AVERAGE-RESPECTS-MUTUAL-EXCLUSION]]
    (collapse-and-attribute form). Returns 0 for non-positive income.
    """
    if gross_income <= 0:
        return 0.0
    return _band_lookup(float(gross_income), _PRETAX_HEALTH_BY_INCOME)


def estimate_fsa_dependent_care(household_size: int) -> float:
    """Annual Dependent-care FSA employee contribution ($/yr).

    NOT mutually exclusive with the HSA (it is not "other health coverage"), so
    it stays a separate line from the pre-tax-health union. Conditioned on
    household_size >= 3 (proxy for "has children"). For households without
    dependent-care expenses (size 1 or 2), returns 0 — the right
    population-blended answer; edge cases (single-parent size=2 with kids)
    handled via the user-override path.
    """
    if household_size <= 2:
        return 0.0
    return _DEPCARE_FSA_ANNUAL_FAMILY


def estimate_commuter_benefit(
    gross_income: float, owns_car: bool | None = None
) -> float:
    """Annual commuter-benefit pre-tax set-aside ($/yr), conditioned on car ownership.

    IRS §132(f) requires the user to actually commute by transit / qualified
    parking, so the prediction bifurcates on car ownership:

      - ``owns_car is True``  → ~$0 (a car-owner drives; no pre-tax transit).
      - ``owns_car is False`` → the genuine §132(f) transit-user set-aside,
        rising with income (``_COMMUTER_TRANSIT_USER_BY_INCOME``).
      - ``owns_car is None``  → unknown (Build A default, pre-A1-elicitation):
        lean car-owner (national ~91% ownership) → near-zero. Deliberately does
        NOT consult the cohort ``car_owner_probability`` — that is the
        unreliable "41%-Chicago" cohort-average A1/Build B fixes via elicitation
        ([[A1-TRANSPORTATION-COHORT-AVERAGE]]); Build B feeds the resolved
        ``owns_car`` here and this line upgrades from near-zero to the
        conditioned value for free.

    Returns 0 for non-positive income. This is the conditioning corollary of
    [[COHORT-AVERAGE-RESPECTS-MUTUAL-EXCLUSION]] (car-cost xor no-car-cost ⇒
    transit-commuter xor no-transit).
    """
    if gross_income <= 0:
        return 0.0
    if owns_car is False:
        return _band_lookup(float(gross_income), _COMMUTER_TRANSIT_USER_BY_INCOME)
    # owns_car True or unknown → car-owner default → near-zero pre-tax transit.
    return 0.0


def estimate_supplemental_insurance(age: float, household_size: int) -> float:
    """Annual supplemental life + disability premiums (employee share, $/yr).

    Basic employer life/disability is typically employer-paid (~$0 employee);
    *voluntary supplemental* coverage is conditioned on age × dependents:
    near-zero for young childless singles, rising with age (life-insurance need)
    and household size (dependents to protect). Replaces the old flat $20/mo
    that over-applied to 25-yo singles. See ``_SUPPLEMENTAL_BY_AGE_ANNUAL`` ×
    ``_SUPPLEMENTAL_SIZE_MULT``. Returns 0 for non-positive age/size.
    """
    if age <= 0 or household_size <= 0:
        return 0.0
    base = _band_lookup(float(age), _SUPPLEMENTAL_BY_AGE_ANNUAL)
    mult = _band_lookup(float(household_size), _SUPPLEMENTAL_SIZE_MULT)
    return base * mult


# --------------------------------------------------------------------------- #
# Combined result type                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommittedOutflow:
    """One committed-outflow line — what the user sees and can override.

    ``label`` is the user-facing copy (neutral, never prescriptive). ``code``
    is the stable machine identifier (e.g. ``"retirement_contribution"``).
    ``source`` is the external benchmark for transparency (audit / dev view).
    ``adjustable=True`` is always set — every committed outflow is
    Q7-user-overridable.
    """

    code: str
    label: str
    annual: float
    source: str
    adjustable: bool = True


@dataclass(frozen=True)
class CommittedOutflows:
    """The committed-outflow block: per-line items + total + audit fields.

    The total is what subtracts from take-home (passed to
    ``compute_d_variable(additional_committed=...)``). Per-line items are
    surfaced in the response so the user can adjust any of them.
    """

    items: tuple[CommittedOutflow, ...] = field(default_factory=tuple)
    total: float = 0.0

    def by_code(self) -> dict[str, CommittedOutflow]:
        return {it.code: it for it in self.items}


def estimate_committed_outflows(
    profile: HouseholdProfile, *, owns_car: bool | None = None
) -> CommittedOutflows:
    """Build the committed-outflow block from profile inputs.

    Six lines: two high-participation (retirement, health premium — Build 5)
    and four low-participation applicability-conditioned lines (Build A):
    the pre-tax-health union (HSA ⊎ traditional FSA collapsed), dependent-care
    FSA, the car-ownership-conditioned commuter benefit, and the
    age × dependents-conditioned supplemental life/disability.

    ``owns_car`` (None until A1/Build B elicits it) conditions the commuter
    line: None/True → near-zero (car-owner default), False → the transit-user
    set-aside. Future flows plug in as additional ``CommittedOutflow`` entries.
    """
    gi = float(profile.gross_income)
    hs = int(profile.household_size)
    retirement = estimate_retirement_contribution(age=float(profile.age), gross_income=gi)
    premium = estimate_health_premium_employee_share(household_size=hs)
    pretax_health = estimate_pretax_health_savings(gross_income=gi)
    fsa_dc = estimate_fsa_dependent_care(household_size=hs)
    commuter = estimate_commuter_benefit(gi, owns_car=owns_car)
    suppl = estimate_supplemental_insurance(age=float(profile.age), household_size=hs)
    items = (
        # High-participation (as-if-enrolled cohort-typical) — Build 5.
        CommittedOutflow(
            code="retirement_contribution",
            label="Retirement contribution (cohort-typical, adjustable)",
            annual=retirement,
            source="Vanguard How America Saves 2025 (2024 plan-year data), Fig. 37",
        ),
        CommittedOutflow(
            code="health_premium_employee_share",
            label="Health insurance premium — employee share (cohort-typical, adjustable)",
            annual=premium,
            source="KFF Employer Health Benefits Survey 2024, Fig. 6.4 (single) / 6.5 (family)",
        ),
        # Low-participation, applicability-conditioned (Build A) —
        # [[COHORT-AVERAGE-RESPECTS-MUTUAL-EXCLUSION]].
        CommittedOutflow(
            code="pretax_health_savings",
            label="Tax-advantaged health savings — HSA or FSA (pre-tax, adjustable)",
            annual=pretax_health,
            source="KFF EHBS 2024 Fig. 8.4 + EBRI/Devenir 2024 (HSA) ⊎ BLS NCS 2024 (FSA) — union, mutually exclusive per IRS §223",
        ),
        CommittedOutflow(
            code="fsa_dependent_care",
            label="Dependent-care FSA contribution (pre-tax, adjustable)",
            annual=fsa_dc,
            source="BLS NCS 2024 + IRS Pub 503 (population mean for households with children)",
        ),
        CommittedOutflow(
            code="commuter_benefit",
            label="Pre-tax commuter benefit (transit/parking, adjustable)",
            annual=commuter,
            source="BLS NCS 2024 + IRS §132(f) (conditioned on car ownership; near-zero until owns_car elicited — A1/Build B)",
        ),
        CommittedOutflow(
            code="supplemental_insurance",
            label="Supplemental life + disability premiums — employee share (adjustable)",
            annual=suppl,
            source="BLS NCS 2024 + LIMRA (conditioned on age × dependents; basic life is typically employer-paid)",
        ),
    )
    return CommittedOutflows(items=items, total=sum(it.annual for it in items))


# --------------------------------------------------------------------------- #
# Q7 user-adjustment fixity — category-level override helper.                  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Pre-tax wedge — which committed lines reduce taxable income (and which base). #
# Traditional 401(k) reduces income-tax wages but NOT the FICA base; §125       #
# cafeteria (health premium, HSA/FSA, dependent-care FSA) and §132(f) commuter  #
# reduce BOTH bases; voluntary supplemental life/disability is post-tax.        #
# This is the input to the pre-tax feedback (Stage 3): take-home is computed on #
# wages net of these exclusions, so the contributions are taxed correctly       #
# (PRETAX-FEEDBACK-LOOP-FROM-WATERFALL). The committed TOTAL still subtracts     #
# from the discretionary budget (the dollars are committed either way); the     #
# wedge only changes how much TAX take-home reflects.                           #
# --------------------------------------------------------------------------- #

# code -> (reduces income-tax wages?, reduces FICA wage base?)
_PRETAX_TREATMENT: dict[str, tuple[bool, bool]] = {
    "retirement_contribution": (True, False),       # traditional 401(k): income-tax only
    "health_premium_employee_share": (True, True),  # §125 cafeteria POP: both
    "pretax_health_savings": (True, True),          # HSA / health FSA via §125: both
    "fsa_dependent_care": (True, True),             # dependent-care FSA §125: both
    "commuter_benefit": (True, True),               # §132(f) transit/parking: both
    "supplemental_insurance": (False, False),       # voluntary life/disability: post-tax
}


def is_pre_tax(code: str) -> bool:
    """True if this committed-outflow code reduces the income-tax wage base — i.e.
    it is a PRE-TAX contribution that lowers AGI (it contributes to the gross−AGI
    gap). Drives the take-home DISPLAY partition: pre-tax outflows belong in the
    tax wedge (a subtraction between taxes and take-home, since take-home is the
    bank-account number net of them); post-tax outflows (supplemental life/
    disability, debt service) stay in the four-way. Unknown codes → post-tax
    (conservative). Mirrors the income-tax-excludable column of _PRETAX_TREATMENT."""
    return _PRETAX_TREATMENT.get(code, (False, False))[0]


@dataclass(frozen=True)
class PretaxWedge:
    """Pre-tax exclusions applied to the wage bases before tax is computed."""

    income_tax_excludable: float = 0.0  # reduces the federal/state income-tax wage base
    fica_excludable: float = 0.0        # reduces the Social Security/Medicare wage base


def pretax_wedge(committed: CommittedOutflows) -> PretaxWedge:
    """Classify committed-outflow lines into the income-tax / FICA exclusion wedge.

    Unknown codes default to post-tax (no exclusion) — conservative (does not
    reduce tax for a flow whose pre-tax status we can't confirm).
    """
    it_excl = 0.0
    fica_excl = 0.0
    for item in committed.items:
        reduces_it, reduces_fica = _PRETAX_TREATMENT.get(item.code, (False, False))
        if reduces_it:
            it_excl += float(item.annual)
        if reduces_fica:
            fica_excl += float(item.annual)
    return PretaxWedge(income_tax_excludable=it_excl, fica_excludable=fica_excl)


def apply_committed_outflow_overrides(
    base: CommittedOutflows,
    overrides: dict[str, float],
) -> CommittedOutflows:
    """Pin user-reported values for committed-outflow line items; total updates.

    Category-level edit semantics (per Q7): the user's value is authoritative
    for that line, no other line shifts, the *total* updates to reflect the
    new sum (which is what flows into ``compute_d_variable.additional_committed``
    on the next re-prediction). For an input edit (age, income, household_size),
    the caller re-runs ``run_profile_analysis`` instead — that path
    re-computes the committed outflows from scratch.

    The total returned here is the NEW total after overrides. Downstream:

        new_total = apply_committed_outflow_overrides(outflows, {"retirement_contribution": 0}).total
        d_variable = compute_d_variable(profile, additional_committed=new_total)
        # re-run the allocator with the new d_variable; residual absorbs the
        # difference vs the pre-override path (user adjusts retirement to $0
        # -> total drops -> d_variable rises -> residual grows by that amount).
    """
    new_items = tuple(
        (
            CommittedOutflow(
                code=it.code,
                label=it.label,
                annual=float(overrides[it.code]),
                source=it.source + " (user-overridden)",
                adjustable=it.adjustable,
            )
            if it.code in overrides
            else it
        )
        for it in base.items
    )
    return CommittedOutflows(items=new_items, total=sum(it.annual for it in new_items))
