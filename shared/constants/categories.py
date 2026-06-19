"""CEX expenditure categories used as fusion target variables.

Source of truth: fusionData/survey-processed/CEX/cat_assignment.rda,
extracted via Rscript during the initial audit. This list is the 55
unique `cat` codes the 2024 ACS-CEX fusion model (ACS_CEX_2024.fsn) was
trained on, as confirmed by fusion_2024.log ("55 expenditure categories").

Each entry:
  cat         — short code (lowercase), matches fusion variable name
  major       — top-level grouping
  description — human-readable label
  type        — "Goods" | "Services" | "Housing" (PCE classification)
"""

CATEGORIES: list[dict] = [
    {"cat": "cloftw", "major": "Apparel", "description": "Clothing and footwear", "type": "Goods"},
    {"cat": "jwlbg", "major": "Apparel", "description": "Jewelry and handbags", "type": "Goods"},
    {"cat": "educ", "major": "Education", "description": "Education services", "type": "Services"},
    {"cat": "stdint", "major": "Education", "description": "Student loan interest payments", "type": "Services"},
    {"cat": "eltrnp", "major": "Entertainment", "description": "Electronic products", "type": "Goods"},
    {"cat": "hotel", "major": "Entertainment", "description": "Hotels and motels", "type": "Services"},
    {"cat": "oeprd", "major": "Entertainment", "description": "Other entertainment products", "type": "Goods"},
    {"cat": "oesrv", "major": "Entertainment", "description": "Other entertainment services", "type": "Services"},
    {"cat": "recrp", "major": "Entertainment", "description": "Recreational products", "type": "Goods"},
    {"cat": "eathome", "major": "Food and drink", "description": "Eating and drinking at home", "type": "Goods"},
    {"cat": "eatout", "major": "Food and drink", "description": "Eating and drinking out", "type": "Services"},
    {"cat": "health", "major": "Health care", "description": "Health care and insurance premiums", "type": "Services"},
    {"cat": "furhwr", "major": "Household operation", "description": "Furniture, housewares, and tools", "type": "Goods"},
    {"cat": "happl", "major": "Household operation", "description": "Household appliances", "type": "Goods"},
    {"cat": "hhpcp", "major": "Household operation", "description": "Household and personal care products", "type": "Goods"},
    {"cat": "hhpcs", "major": "Household operation", "description": "Household, personal, and child care services", "type": "Services"},
    {"cat": "hinsp", "major": "Housing", "description": "Home insurance, primary", "type": "Housing"},
    {"cat": "hmtimp", "major": "Housing", "description": "Home maintenance and improvement", "type": "Housing"},
    {"cat": "mrtgip", "major": "Housing", "description": "Mortgage interest payments, primary", "type": "Housing"},
    {"cat": "mrtgpp", "major": "Housing", "description": "Mortgage principal payments, primary", "type": "Housing"},
    {"cat": "mrtgps", "major": "Housing", "description": "Mortgage principal payments, secondary", "type": "Housing"},
    {"cat": "ohouse", "major": "Housing", "description": "Other housing expenses", "type": "Housing"},
    {"cat": "ptaxp", "major": "Housing", "description": "Property taxes, primary", "type": "Housing"},
    {"cat": "rntexp", "major": "Housing", "description": "Renter-side expenses (materials, appliances, tenant's insurance)", "type": "Housing"},
    {"cat": "chrty", "major": "Miscellaneous", "description": "Charitable contributions", "type": "Services"},
    {"cat": "finpay", "major": "Miscellaneous", "description": "Insurance, financial services, and other payments", "type": "Services"},
    {"cat": "ocash", "major": "Miscellaneous", "description": "Other cash transfers", "type": "Services"},
    {"cat": "othint", "major": "Miscellaneous", "description": "Interest and finance charges on credit card and other debt", "type": "Services"},
    {"cat": "check", "major": "Other", "description": "Value of checking, savings, money market accounts, and CDs", "type": "Services"},
    {"cat": "lifval", "major": "Other", "description": "Surrender value of life insurance policies", "type": "Services"},
    {"cat": "othdbt", "major": "Other", "description": "Amount owed on credit cards and other debt such as medical or personal loans", "type": "Services"},
    {"cat": "othfin", "major": "Other", "description": "Value of other financial assets, such as annuities, trusts, and royalties", "type": "Services"},
    {"cat": "ownval", "major": "Other", "description": "Owner-occupied home value, zero for rented homes", "type": "Services"},
    {"cat": "retire", "major": "Other", "description": "Value of retirement accounts, such as 401k, IRAs, and TSP", "type": "Services"},
    {"cat": "rntval", "major": "Housing", "description": "Annual rent amount (actual rent for renters, imputed rental equivalence for owners)", "type": "Housing"},
    {"cat": "stddbt", "major": "Other", "description": "Amount owed on student loans", "type": "Services"},
    {"cat": "stock", "major": "Other", "description": "Value of directly-held stocks, bonds, and mutual funds", "type": "Services"},
    {"cat": "vehval", "major": "Other", "description": "Value of owned vehicles", "type": "Services"},
    {"cat": "airshp", "major": "Transportation", "description": "Air and ship travel", "type": "Services"},
    {"cat": "gas", "major": "Transportation", "description": "Gasoline and other motor fuel", "type": "Goods"},
    {"cat": "pubtrn", "major": "Transportation", "description": "Public transportation", "type": "Services"},
    {"cat": "taxis", "major": "Transportation", "description": "Taxi and ride sharing services", "type": "Services"},
    {"cat": "vehins", "major": "Transportation", "description": "Vehicle insurance", "type": "Services"},
    {"cat": "vehint", "major": "Transportation", "description": "Vehicle loan interest payments", "type": "Services"},
    {"cat": "vehmlr", "major": "Transportation", "description": "Vehicle maintenance, leasing, and rental", "type": "Services"},
    {"cat": "vehnew", "major": "Transportation", "description": "Gross value of new vehicle purchases", "type": "Goods"},
    {"cat": "vehprd", "major": "Transportation", "description": "Vehicle parts, accessories, and supplies", "type": "Goods"},
    {"cat": "vehprn", "major": "Transportation", "description": "Vehicle loan principal payments", "type": "Services"},
    {"cat": "vehreg", "major": "Transportation", "description": "Vehicle licensing, registration, and inspection", "type": "Services"},
    {"cat": "vehusd", "major": "Transportation", "description": "Net purchases of used vehicles", "type": "Goods"},
    {"cat": "elec", "major": "Utilities and phone", "description": "Electricity", "type": "Services"},
    {"cat": "intphn", "major": "Utilities and phone", "description": "Internet and phone", "type": "Services"},
    {"cat": "ngas", "major": "Utilities and phone", "description": "Natural gas", "type": "Services"},
    {"cat": "ofuel", "major": "Utilities and phone", "description": "Heating oil, LPG, and other fuels", "type": "Goods"},
    {"cat": "watrsh", "major": "Utilities and phone", "description": "Water, sewer, and trash", "type": "Services"},
]

CATEGORY_CODES: list[str] = [c["cat"] for c in CATEGORIES]

assert len(CATEGORY_CODES) == 55, f"Expected 55 CEX categories, got {len(CATEGORY_CODES)}"
assert len(set(CATEGORY_CODES)) == 55, "Duplicate category codes"

# Balance vs flow partition.
#
# Nine of the 55 CEX "categories" are wealth stocks or liability balances,
# not annual flows. They come from CEX FMLI survey questions on *current
# value* of accounts/assets/debts (e.g. FMLI.irax → retire, FMLI.liquidx
# → check), not from UCC expenditure ledgers. Treating them as annual
# spend inflates the allocator's anchor sum (a $17k vehval p50 alone
# eats half of a typical discretionary budget) and forces unnecessary
# compression.
#
# The allocator zeros balance categories from allocation. They're still
# carried through the matching + API response as display-only context
# (net-worth snapshot, adjustable debt balances).
ASSET_CATEGORIES: frozenset[str] = frozenset({
    "check",    # FMLI.liquidx — checking/savings/MMA/CD value
    "retire",   # FMLI.irax    — 401k/IRA/TSP balance
    "stock",    # FMLI.stockx  — directly-held stocks/bonds/MF value
    "othfin",   # FMLI.othastx — annuities, trusts, royalties value
    "lifval",   # FMLI.wholifx — life insurance surrender value
    "vehval",   # CEI vehicle_value — owned vehicle market value
    "ownval",   # Owner-occupied home value
})

LIABILITY_CATEGORIES: frozenset[str] = frozenset({
    "othdbt",   # FMLI.creditx+othlonx — credit card + personal/medical
    "stddbt",   # FMLI.studntx — student loan balance
})

BALANCE_CATEGORIES: frozenset[str] = ASSET_CATEGORIES | LIABILITY_CATEGORIES

FLOW_CATEGORIES: frozenset[str] = frozenset(CATEGORY_CODES) - BALANCE_CATEGORIES

assert len(BALANCE_CATEGORIES) == 9, (
    f"Expected 9 balance categories, got {len(BALANCE_CATEGORIES)}"
)
assert len(FLOW_CATEGORIES) == 46, (
    f"Expected 46 flow categories, got {len(FLOW_CATEGORIES)}"
)
assert BALANCE_CATEGORIES.isdisjoint(FLOW_CATEGORIES)
assert BALANCE_CATEGORIES <= set(CATEGORY_CODES)


# Necessity vs luxury classification — static audit.
#
# Selects the allocator's correction-factor ceiling (2.0 for necessities,
# 8.0 for luxuries) regardless of what the QUAIDS polynomial's fitted
# elasticity says for a particular income. Runtime ``ε > 1`` selection
# is unstable at extrapolated incomes (e.g. eatout ε=0.66 at $250k but
# behaves like a luxury; vehmlr ε=1.24 at $250k but is a necessity),
# so we use a hand-audited table grounded in standard demand conventions
# rather than the fitted polynomial's discretionary breakpoint.
#
# Balance categories are included in NECESSITY_CATEGORIES for
# completeness; they are zeroed in allocation so the cap never actually
# applies, but their presence here keeps NECESSITY ∪ LUXURY == all 55.
LUXURY_CATEGORIES: frozenset[str] = frozenset({
    "eatout",   # Eating out — ε typically 1.2-1.8 in BLS studies
    "hotel",    # Hotels/motels — discretionary travel
    "airshp",   # Air/ship travel
    "taxis",    # Taxi/ride sharing
    "recrp",    # Recreational products
    "oeprd",    # Other entertainment products
    "oesrv",    # Other entertainment services
    "eltrnp",   # Electronic products (discretionary durable)
    "chrty",    # Charitable contributions
    "ocash",    # Other cash transfers (discretionary)
    "educ",     # Education services — high ε in household spending
    "cloftw",   # Clothing and footwear
    "jwlbg",    # Jewelry and handbags
    "furhwr",   # Furniture, housewares, and tools
    "happl",    # Household appliances (discretionary durable)
    "hmtimp",   # Home maintenance and improvement
    "vehnew",   # New vehicle purchases
    "vehusd",   # Used vehicle purchases
    "vehprd",   # Vehicle parts/accessories (discretionary)
})

NECESSITY_CATEGORIES: frozenset[str] = (
    frozenset(CATEGORY_CODES) - LUXURY_CATEGORIES
)

assert LUXURY_CATEGORIES.isdisjoint(NECESSITY_CATEGORIES)
assert LUXURY_CATEGORIES | NECESSITY_CATEGORIES == set(CATEGORY_CODES)

# Omit-by-default (heavy-zero, cohort-mean-meaningless) categories.
# These six have a heavily right-skewed cohort distribution (p50=$0, low
# nonzero-rate, cohort mean diluted by a high-spending minority) AND their
# across-profile variation is NOT driven by the category's real causal inputs
# (educ's apparent 6.1x variation runs *backwards* on family size via the Engel
# mechanism — see agent-artifacts/investigations/educ_and_heavy_zero_categories_scoping.md).
# So the cohort prediction represents approximately no actual user. Treatment
# (HEAVY-ZERO-DISTRIBUTION-NEEDS-ELICITATION): the value-layer zeros their
# allocation; the displaced dollars flow to the four-way remainder; the response
# flags them ``omit_from_initial_view`` so the initial display omits them, and a
# future additive UX (OMITTED-CATEGORY-ADDITIVE-UX) lets users surface them.
#   chrty / educ / ocash  — value-layer-zero (heavy-zero-meaningless)
#   stdint / othint / finpay — interest/finance lines; when the user reports debt
#       the debt-service derivation captures the interest, and when they don't the
#       cohort $1-15/mo is residual noise either way (interest-displacement),
#       unified here under the same omit-by-default mechanism.
# Scope is deliberately narrow (~3+2): most heavy-zero cats are *correctly*
# heavy-zero via existing structural conditioning (tenure, car-ownership,
# transit-mode, region) and are NOT in this set.
OMIT_BY_DEFAULT_CATEGORIES: frozenset[str] = frozenset({
    "chrty", "educ", "ocash", "stdint", "othint", "finpay",
})

assert OMIT_BY_DEFAULT_CATEGORIES <= frozenset(CATEGORY_CODES)


# ===========================================================================
# AGGREGATED category set (category-aggregation project).
#
# Parallel to the 55-category disaggregated set above. Six substitution-prone
# groups are pre-aggregated; QUAIDS is refit on the merged set
# (agent-artifacts/aggregation/coefficients_aggregated.json). The
# disaggregated set + path remain the permanent fallback (use_aggregated=False).
#
# vehreg is EXCLUDED from the transportation aggregate (W1 Option (a)): it uses
# the direct state-DMV-cost path (locked #8), so it stays its own retained line.
# Group memberships carried forward from pipeline/export/phase1_aggregation_
# feasibility.R, with W1 applied. Member lists MUST match
# agent-artifacts/aggregation/fit_aggregated_coefficients.R byte-for-byte.
# ===========================================================================

# Four aggregates. Gate-2 reverted home_maintenance (hmtimp, ohouse) and
# travel (hotel, airshp) to disaggregated lines: their episodic (p50=0)
# character lost allocation headroom under aggregation (one aggregate
# conditional_p90 ceiling < the sum of the members' ceilings), so they
# behaved erratically vs the disaggregated members. The four kept here are
# the "always-on" bundles whose mode-split / median-of-sum behavior the
# Gate-2 reference profiles confirmed sound.
#
# EPISODIC-CAPITAL REMOVAL (Build 1, 2026-05): the intensive-margin aggregates
# carried episodic capital components that inflated their monthly means and
# don't belong on the monthly/paycheck surface — vehicle purchases (vehnew,
# vehusd) in transportation and durables (furhwr, happl) in household_goods.
# These are large, infrequent draws funded from accumulated capacity, not
# monthly flow, so they were DE-MERGED into their own retained episodic lines
# (tagged episodic ∩ luxury -> step categories, surfaced annual-only, dropped
# all-or-nothing first under compression — exactly like the disaggregated
# path treats them). The aggregate's monthly value is now its recurring
# backbone only. See agent-artifacts/investigations/discretionary_anchor_
# statistic.md (superseded for these aggregates by the Build-2 re-measurement).
# ENTERTAINMENT+SHOPPING RECOMPOSITION (Build 3, 2026-05-27). Settled final
# compositions (composition-before-factor):
#   - entertainment loses recrp (durable rec-goods boats/RVs/instruments) ->
#     de-merged to a retained episodic∩luxury line; gains a flat PCE-anchored
#     `streaming` value-add (handled in algorithm.py, not a member here).
#   - shopping loses eltrnp (electronics) + jwlbg (jewelry) -> de-merged episodic
#     lines; its recurring core is cloftw (apparel) + the carved `pcare`
#     (personal-care products from hhpcp, an Option-B Diary member added in
#     algorithm.py with its own synth-pop column + decontaminated recall factor).
#   - household_goods keeps (hhpcp, hhpcs) BUT the aggregated path nets out pcare
#     (household_goods = hhpcp - pcare + hhpcs), since pcare moved to shopping.
# pcare/streaming are aggregate MEMBERS/value-adds, NOT top-level allocation
# atoms (honors the Gate-2 no-new-atoms lesson) and NOT CEX codes, so they live
# in the algorithm.py aggregation, not in AGG_GROUPS / AGGREGATED_CATEGORY_CODES.
AGG_GROUPS: dict[str, tuple[str, ...]] = {
    "transportation": (
        "gas", "vehins", "vehint", "vehmlr",
        "vehprd", "vehprn", "pubtrn", "taxis",
    ),
    "household_goods": ("hhpcp", "hhpcs"),
    "shopping": ("cloftw",),
    "entertainment": ("oeprd", "oesrv"),
}

# Carved personal-care member (Build 3): a shopping-aggregate member sourced from
# its own synth-pop `pcare` column (the Option-B Diary personal-care substance,
# 640xxx), NOT a CEX category. Netted out of household_goods's hhpcp. Carries the
# decontaminated personal-care recall factor in pce_correction.json.
AGG_CARVED_MEMBER: str = "pcare"
AGG_CARVED_SOURCE: str = "hhpcp"   # pcare is netted out of this aggregate member

# Flat PCE-anchored streaming value added to the entertainment aggregate per
# household (BEA 2.4.5U L223 / households; near-flat — saturates). Refreshable,
# not a per-cohort percentile. Applied in algorithm.py.
ENTERTAINMENT_STREAMING_ANNUAL: float = 476.0

AGG_GROUP_NAMES: list[str] = list(AGG_GROUPS.keys())

# Every merged member -> its aggregate group (reverse index).
AGG_MEMBER_TO_GROUP: dict[str, str] = {
    member: group for group, members in AGG_GROUPS.items() for member in members
}

_MERGED_MEMBERS: frozenset[str] = frozenset(AGG_MEMBER_TO_GROUP)
assert len(_MERGED_MEMBERS) == 13, (
    f"Expected 13 merged members, got {len(_MERGED_MEMBERS)}"
)
assert "vehreg" not in _MERGED_MEMBERS, "vehreg must stay its own line (W1)"
# Build-1 de-merged episodic-capital components + Build-3 de-merged
# entertainment/shopping durables are retained lines, not merged members.
assert {"vehnew", "vehusd", "furhwr", "happl",
        "recrp", "eltrnp", "jwlbg"}.isdisjoint(_MERGED_MEMBERS), (
    "de-merged episodic-capital / durable components must not be merged members"
)
# All merged members are genuine CEX codes (pcare is a non-CEX carved member,
# handled in algorithm.py — deliberately NOT in AGG_GROUPS membership here).
assert _MERGED_MEMBERS <= set(CATEGORY_CODES)

# Categories retained disaggregated (incl. vehreg + the 9 balance cats),
# in canonical CATEGORY_CODES order.
RETAINED_CATEGORIES: list[str] = [
    c for c in CATEGORY_CODES if c not in _MERGED_MEMBERS
]

# Aggregated code order mirrors the R fit's NEW_CODES = c(AGG_NAMES, RETAINED).
AGGREGATED_CATEGORY_CODES: list[str] = AGG_GROUP_NAMES + RETAINED_CATEGORIES

assert len(AGGREGATED_CATEGORY_CODES) == 46, (
    f"Expected 46 aggregated categories, got {len(AGGREGATED_CATEGORY_CODES)}"
)
assert len(set(AGGREGATED_CATEGORY_CODES)) == 46, "Duplicate aggregated codes"

# Which transportation members scale with P(own) vs (1 - P(own)) in the
# ambiguous band (W2). Mirrors algorithm.py _VEHICLE_CATS / _TRANSIT_CATS,
# minus vehreg and (Build 1) minus the de-merged vehnew/vehusd purchases —
# those now flow through the per-category car-ownership loop as their own
# retained lines.
AGG_TRANSPORT_VEHICLE: frozenset[str] = frozenset({
    "gas", "vehins", "vehint", "vehmlr", "vehprd", "vehprn",
})
AGG_TRANSPORT_TRANSIT: frozenset[str] = frozenset({"pubtrn", "taxis"})
assert (AGG_TRANSPORT_VEHICLE | AGG_TRANSPORT_TRANSIT) == set(
    AGG_GROUPS["transportation"]
)

# Vehicle-financing sub-components inside the transportation aggregate:
# vehint (vehicle loan interest) + vehprn (vehicle loan principal).
# When a user reports auto_loan_payment, these cohort-estimated costs
# already live in debt service — the allocator subtracts the cohort
# finance estimate from the transportation anchor to prevent the double-count.
AGG_TRANSPORT_FINANCE: frozenset[str] = frozenset({"vehint", "vehprn"})
assert AGG_TRANSPORT_FINANCE <= set(AGG_GROUPS["transportation"])
assert AGG_TRANSPORT_FINANCE.isdisjoint(AGG_TRANSPORT_TRANSIT)

# Balance cats are all retained (none merged) — unchanged set.
AGGREGATED_BALANCE_CATEGORIES: frozenset[str] = BALANCE_CATEGORIES
assert AGGREGATED_BALANCE_CATEGORIES <= set(AGGREGATED_CATEGORY_CODES)
AGGREGATED_FLOW_CATEGORIES: frozenset[str] = (
    frozenset(AGGREGATED_CATEGORY_CODES) - AGGREGATED_BALANCE_CATEGORIES
)

# Necessity vs luxury for the aggregated set — convention-based audit (same
# philosophy as LUXURY_CATEGORIES above: standard demand conventions, not the
# fitted polynomial's ε). The two kept luxury aggregates (all-luxury members):
#   necessity: transportation (ε≈0.97), household_goods (ε≈0.77, 2/2 split)
#   luxury:    shopping, entertainment
# Retained cats keep their original classification (incl. the reverted
# hmtimp/hotel/airshp luxuries and ohouse necessity).
AGGREGATED_LUXURY_CATEGORIES: frozenset[str] = frozenset(
    {"shopping", "entertainment"}
    | {c for c in RETAINED_CATEGORIES if c in LUXURY_CATEGORIES}
)
AGGREGATED_NECESSITY_CATEGORIES: frozenset[str] = (
    frozenset(AGGREGATED_CATEGORY_CODES) - AGGREGATED_LUXURY_CATEGORIES
)
assert AGGREGATED_LUXURY_CATEGORIES.isdisjoint(AGGREGATED_NECESSITY_CATEGORIES)
assert (
    AGGREGATED_LUXURY_CATEGORIES | AGGREGATED_NECESSITY_CATEGORIES
    == set(AGGREGATED_CATEGORY_CODES)
)


# ===========================================================================
# Recurring vs episodic cadence (category-aggregation project, Stage 4).
#
# A spending-cadence DATA FIELD for the future paycheck-surface filter.
# This is data only — NO paycheck/annual rendering or display logic lives
# here or is implied by it. Strict definition:
#   recurring — a typical household incurs it nearly every month.
#   episodic  — lumpy / seasonal / per-term / annual; not a near-monthly line.
#
# Scope: RECURRING_CATEGORIES / EPISODIC_CATEGORIES tag the 30
# AGGREGATED_FLOW_CATEGORIES only. The 9 balance categories are wealth
# stocks / liability balances (zeroed in allocation, carried display-only),
# not spending flows — a binary recurring/episodic label would misrepresent
# a stock, so they are in NEITHER flow set. They ARE given an explicit
# "balance" value in CATEGORY_CADENCE (rather than being absent) so a future
# consumer can tell "deliberately a stock, no cadence" apart from "tag
# forgotten" — silent absence would make that distinction a guess.
#
# Aggregates mix both cadences; per the Stage-4 spec they are tagged by
# DOMINANT recurring character, with episodic sub-components handled as
# drill-down, not surfaced top-level:
#   transportation -> recurring  (gas + insurance + maintenance + transit
#                                 backbone; vehnew/vehusd purchases are the
#                                 episodic drill-down)
#   household_goods -> recurring  (hhpcp consumables + hhpcs services are
#                                 near-monthly; furhwr/happl durables are
#                                 the episodic drill-down)
#   entertainment  -> recurring  (oesrv subscriptions/recreation services
#                                 near-monthly; oeprd/recrp products are
#                                 the episodic drill-down)
#   shopping       -> episodic   (cloftw/jwlbg/eltrnp are all lumpy; no
#                                 near-monthly backbone)
# ===========================================================================
RECURRING_CATEGORIES: frozenset[str] = frozenset({
    # aggregates (dominant recurring character)
    "transportation", "household_goods", "entertainment",
    # food — groceries and dining
    "eathome", "eatout",
    # utilities & phone — monthly metered/billed
    "elec", "ngas", "intphn", "watrsh",
    # housing carrying costs — rent / mortgage / escrow-smoothed tax+insurance
    "rntval", "mrtgip", "mrtgpp", "mrtgps", "ptaxp", "hinsp",
    # health premiums, debt service, recurring financial payments
    "health", "stdint", "othint", "finpay",
})

EPISODIC_CATEGORIES: frozenset[str] = frozenset({
    # aggregate (apparel + personal-care core; still lumpy at the top level)
    "shopping",
    # travel — trip-driven
    "hotel", "airshp",
    # episodic-capital draws de-merged from the recurring aggregates (Build 1):
    # vehicle purchases (ex-transportation) + furniture/appliance durables
    # (ex-household_goods). Large, infrequent, funded from accumulated capacity.
    "vehnew", "vehusd", "furhwr", "happl",
    # Build-3 de-merged durables: recreation goods (ex-entertainment: boats/RVs/
    # instruments) + electronics + jewelry (ex-shopping). Lumpy, annual-surface.
    "recrp", "eltrnp", "jwlbg",
    # annual / seasonal
    "vehreg", "ofuel",
    # lumpy housing spend & renter durables
    "hmtimp", "ohouse", "rntexp",
    # per-term tuition, irregular transfers/gifts
    "educ", "chrty", "ocash",
})

# The per-category cadence field: every one of the 39 aggregated categories
# maps to exactly one of "recurring" | "episodic" | "balance".
CATEGORY_CADENCE: dict[str, str] = {
    **{c: "recurring" for c in RECURRING_CATEGORIES},
    **{c: "episodic" for c in EPISODIC_CATEGORIES},
    **{c: "balance" for c in AGGREGATED_BALANCE_CATEGORIES},
}

assert RECURRING_CATEGORIES.isdisjoint(EPISODIC_CATEGORIES)
assert (RECURRING_CATEGORIES | EPISODIC_CATEGORIES) == AGGREGATED_FLOW_CATEGORIES, (
    "recurring/episodic tags must cover exactly the 30 aggregated flow categories"
)
# Balance cats are tagged "balance", not recurring/episodic (stocks, not flows).
assert (RECURRING_CATEGORIES | EPISODIC_CATEGORIES).isdisjoint(
    AGGREGATED_BALANCE_CATEGORIES
)
# The three cadence values partition all 39 categories — every category has
# exactly one entry, so a forgotten tag cannot ship silently.
_n_balance = sum(1 for v in CATEGORY_CADENCE.values() if v == "balance")
assert len(RECURRING_CATEGORIES) + len(EPISODIC_CATEGORIES) + _n_balance == 46
assert set(CATEGORY_CADENCE) == set(AGGREGATED_CATEGORY_CODES)
assert len(CATEGORY_CADENCE) == 46


# Episodic sub-components of the recurring aggregates — the "drill-down-handled,
# not top-level" half of the Stage-4 spec. A recurring aggregate surfaces as a
# near-monthly line, but these lumpy members should be drilled into (and kept
# out of the monthly recurring view) by the future paycheck-surface filter,
# not counted as recurring spend. `shopping` is absent: it is episodic at the
# top level, so there is nothing to drill into. Data only — no filter built here.
#
# Build 1 (2026-05): transportation's (vehnew, vehusd) and household_goods'
# (furhwr, happl) were promoted from drill-down sub-components to fully
# DE-MERGED retained episodic lines (they no longer inflate the aggregate's
# monthly value at all, so there is nothing left to drill out of those two).
# Only entertainment retains an episodic sub-component pair — oeprd/recrp were
# assessed as moderate discretionary durables, NOT car/appliance-scale capital,
# so they were conservatively LEFT in the aggregate (their residual lumpiness is
# handled by the Build-2 anchor switch, i.e. a trimmed mean if still distorted,
# not removal).
AGG_EPISODIC_SUBCOMPONENTS: dict[str, tuple[str, ...]] = {
    # Build-3: recrp de-merged to its own retained episodic line, so only oeprd
    # (entertainment goods: gaming/subscriptions/toys + Diary pet food) remains as
    # an episodic sub-component to drill out of the recurring entertainment view.
    "entertainment": ("oeprd",),
}
# Every key is a recurring aggregate; every listed sub-member actually belongs
# to that aggregate.
assert all(g in RECURRING_CATEGORIES for g in AGG_EPISODIC_SUBCOMPONENTS)
assert all(g in AGG_GROUP_NAMES for g in AGG_EPISODIC_SUBCOMPONENTS)
assert all(
    m in AGG_GROUPS[g]
    for g, members in AGG_EPISODIC_SUBCOMPONENTS.items()
    for m in members
)


# ===========================================================================
# RETIRED (2026-06-02, soft-constraint compression Phase 8) — the step (all-or-
# nothing) vs gradual compression classification. The unified soft-constraint
# optimizer (models/optimizer/soft_constraint_optimizer.py) compresses every
# category continuously at its own empirical expenditure elasticity — there is no
# step/recurring distinction and no Phase A all-or-nothing drop. ``STEP_CATEGORIES``,
# ``AGGREGATED_STEP_CATEGORIES`` and ``_EPISODIC_DISAGGREGATED`` are removed; the
# Q3 finding (compression is continuous across the budget gradient, no regime
# change) made the categorical split empirically unfounded. Locked principles:
# DECISIONS.md LOCKED-DROP-ORDER-RETIRES / STEP-VS-RECURRING-DISTINCTION-RETIRES.
# (``EPISODIC_CATEGORIES`` survives above — it is a DISPLAY/cadence field, not a
# compression knob.)
# ===========================================================================


# ===========================================================================
# Residual back-fill target set (the reverse stage, Build 4).
#
# The slope-ceiling discretionary categories — recurring/episodic luxury lines
# where the cohort-anchor income slope is structurally weak (implied elasticity
# ~0.3–0.5 vs a real ~0.6–0.8, the accepted INCOME-SLOPE-CEILING). The reverse
# stage (``models/optimizer/backfill.py``) redistributes implausibly-large
# feasibility-slack residual into THESE categories only — entertainment +
# shopping (aggregates), the retained travel lines (hotel/airshp/recrp), and
# eatout (added 2026-06-09 with the high-income ceiling stratification build:
# a high-participation luxury — nzr ~0.92, ε ~1.12 — the original curated five
# simply missed; see elasticity_determined_backfill_scoping.md Q2).
# This stays a CURATED constant — the four episodic durables
# (furhwr/hmtimp/happl/eltrnp) are deliberately excluded pending Build 2's
# representativeness characterization (lumpy, low-participation lines where
# the cohort mean may not represent the individual —
# EPISODIC-DURABLES-AS-SURPLUS-DESTINATIONS-UNRESOLVED).
# Defined on the AGGREGATED path (the default + the only path the back-fill
# runs on; the 55-cat disaggregated fallback never back-fills and stays
# byte-for-byte). Membership lives here per locked #4.
# ===========================================================================
BACKFILL_TARGET_CATEGORIES: frozenset[str] = frozenset(
    {"entertainment", "shopping", "hotel", "airshp", "recrp", "eatout"}
)
# Every target is an aggregated luxury (never a necessity / never a balance cat).
assert BACKFILL_TARGET_CATEGORIES <= AGGREGATED_LUXURY_CATEGORIES
assert BACKFILL_TARGET_CATEGORIES.isdisjoint(AGGREGATED_BALANCE_CATEGORIES)
assert BACKFILL_TARGET_CATEGORIES <= set(AGGREGATED_CATEGORY_CODES)


# ===========================================================================
# Elastic-sink category set (the down-direction residual sweep, 2026-06-10;
# locked REMAINDER-ZERO-INVARIANT-DOWN-DIRECTION).
#
# The high-participation elastic consumption categories that can always absorb
# additional predicted spending without becoming implausible — the sweep
# destinations for the low-savings-contradiction residual
# (``models/optimizer/backfill.py::sweep_remainder_to_sinks``). Membership
# criteria (verified against ``compression_parameters.json`` aggregated stats):
#
#   - HIGH PARTICIPATION (nonzero_rate >= ~0.92): the deployment is
#     representative of the individual, not a cohort-mean artifact — the
#     opposite of the episodic durables (furhwr/hmtimp/...) and the episodic
#     travel lines (hotel 0.43 / recrp 0.46 / airshp 0.30) that failed the
#     representativeness characterization and are deliberately EXCLUDED.
#   - ELASTIC (positive expenditure elasticity, aggregated QUAIDS coefficient
#     present): eathome 0.72 / entertainment 0.99 / household_goods 1.09 /
#     shopping 1.07 / eatout 1.12.
#   - GENERAL CONSUMPTION (over-prediction is the safe error direction —
#     OVER-PREDICTION-IS-THE-SAFE-DIRECTION-FOR-SPENDING): "a bit high on
#     shopping/dining" is the comfortable adjust-down correction. Excludes
#     needs-driven lines (health, nzr 0.91 but not an elastic sink) and the
#     mode-split/car-conditioned transportation aggregate.
#
# Defined on the AGGREGATED path only (like BACKFILL_TARGET_CATEGORIES; the
# 55-cat disaggregated fallback never sweeps and stays byte-for-byte).
# ===========================================================================
ELASTIC_SINK_CATEGORIES: frozenset[str] = frozenset(
    {"eathome", "eatout", "shopping", "entertainment", "household_goods"}
)
assert ELASTIC_SINK_CATEGORIES <= set(AGGREGATED_CATEGORY_CODES)
assert ELASTIC_SINK_CATEGORIES.isdisjoint(AGGREGATED_BALANCE_CATEGORIES)


# ===========================================================================
# Smooth-category mean anchoring (anchor-statistic switch, 2026-05).
#
# The allocator anchors most categories on the cohort p50 (median). For the
# unambiguously SMOOTH, high-participation, recurring categories below, the
# median under-projects the participation-typical level (right-skew from
# behavioral participation), so the sample×kernel-weighted MEAN is the correct
# high-side anchor — deliberate over-prediction (users correct down; the high
# side is safer for tight households and more encouraging for comfortable ones).
#
# Original clearly-smooth set: food (eathome, eatout) + recurring utilities/phone
# (elec, ngas, intphn, watrsh).
#
# LUMPY ANCHOR SWITCH (Build 2, 2026-05): after Build 1 de-lumped the
# intensive-margin aggregates (vehnew/vehusd out of transportation, furhwr/happl
# out of household_goods), the backbones were re-measured for outlier-distortion
# ((m−t)/m trim95 across the six reference profiles). The de-lumped
# transportation backbone (0.06–0.155) plus shopping (0.10–0.19) and
# entertainment (0.12–0.26, 5/6 ≤0.19) are now SMOOTH — plain weighted mean is
# the accurate high-side anchor (the severe lumpiness was the now-removed
# vehicle purchases). Only household_goods retains genuine outlier-distortion
# (its hhpcs child-care-SERVICES tail; (m−t)/m to 0.27) → trimmed mean (below).
# travel (hotel/airshp) and the de-merged capital lines stay on their episodic
# engel→conditional_p90 anchor (step cats), not the mean. Supersedes the
# per-category trim recommendations in discretionary_anchor_statistic.md, which
# were measured PRE-removal. Necessities below unchanged. EXCLUDES health
# (heavy out-of-pocket tail; kept on median).
#
# ``allocator._primary_anchors`` reads ``SpendingDistribution.weighted_mean``
# for the plain-mean set and ``trimmed_mean`` for the trimmed set, falling back
# to p50 when the statistic is not populated. The two sets are disjoint.
# ===========================================================================
SMOOTH_MEAN_ANCHOR_CATEGORIES: frozenset[str] = frozenset({
    "eathome", "eatout",                  # food — near-universal participation
    "elec", "ngas", "intphn", "watrsh",   # recurring utilities / phone
    "transportation",                     # de-lumped backbone (Build 1) — now smooth
    "shopping", "entertainment",          # smooth discretionary aggregates
})

# Trimmed (trim95) weighted-mean anchor: categories whose plain mean stays
# outlier-distorted post-de-lumping, so the undistorted central tendency is the
# accurate anchor. household_goods: durables (furhwr/happl) were removed in
# Build 1, but the hhpcs child-care-services backbone tail still inflates the
# plain mean for the typical household (trim corrects the ANCHOR accuracy —
# distinct from compression, which corrects the BUDGET).
TRIMMED_MEAN_ANCHOR_CATEGORIES: frozenset[str] = frozenset({
    "household_goods",
})

assert SMOOTH_MEAN_ANCHOR_CATEGORIES <= set(AGGREGATED_CATEGORY_CODES), (
    "smooth-anchor cats must be live aggregated-set categories"
)
assert TRIMMED_MEAN_ANCHOR_CATEGORIES <= set(AGGREGATED_CATEGORY_CODES)
# A category gets one anchor statistic, not two.
assert SMOOTH_MEAN_ANCHOR_CATEGORIES.isdisjoint(TRIMMED_MEAN_ANCHOR_CATEGORIES)
# Anchor statistic is orthogonal to cadence/surface: shopping is an episodic
# (annual-surface) aggregate but is smooth, so it anchors on the plain mean.
# Both sets are flow categories (never balance stocks).
assert (SMOOTH_MEAN_ANCHOR_CATEGORIES | TRIMMED_MEAN_ANCHOR_CATEGORIES).isdisjoint(
    AGGREGATED_BALANCE_CATEGORIES
)


# ===========================================================================
# Topic groups (display-aggregation project, 2026-05).
#
# A DISPLAY grouping of the 39 aggregated categories into ~8 legible topics
# plus an explicit "net_worth" topic for the balance cats. This is a pure
# presentation key — it changes how predictions are grouped for the eye, it
# does NOT re-predict, re-aggregate the model, or alter any value. The
# topic-grouped view is a *projection* over the 39 prediction atoms; a future
# interpretive cut (control-axis Fixed/Essential/Discretionary, or another
# framing) will re-project the same atoms under a different key. So this map
# is deliberately intermediate scaffolding-for-interpretation, not the final
# grouping. See apps/api/profiles/display.py and
# agent-artifacts/aggregation/PLAN.md.
#
# Topic membership is the source of truth here (topic -> members); TOPIC_GROUP
# (category -> topic) is its reverse index, mirroring AGG_GROUPS /
# AGG_MEMBER_TO_GROUP above. The 4 backend aggregates appear as single members
# (they are single predictions, not expanded into components).
#
# net_worth holds exactly the 9 balance categories. They are given an explicit
# topic value (not silent absence) for the same reason CATEGORY_CADENCE tags
# them "balance": a future consumer can tell "deliberately outside spending"
# apart from "topic forgotten". They are never summed into any spending total.
#
# Two placements are provisional (flagged for confirmation, 2026-05-24):
#   stdint -> financial_debt  (student-loan interest = debt service; fits)
#   educ   -> financial_debt  (education services; borderline — adjacent to
#                              student loans, but arguably its own topic)
# Both are a single-line change in _TOPIC_MEMBERS if reclassified.
# ===========================================================================
_TOPIC_MEMBERS: dict[str, tuple[str, ...]] = {
    "food": ("eathome", "eatout"),
    "housing": (
        "rntval", "mrtgip", "mrtgpp", "mrtgps", "ptaxp",
        "hinsp", "hmtimp", "ohouse", "rntexp",
    ),
    "utilities": ("elec", "ngas", "intphn", "watrsh", "ofuel"),
    "transportation_travel": (
        "transportation",  # backend aggregate (vehreg + vehnew/vehusd excluded)
        "vehreg",          # separate direct-cost line (locked #8)
        "vehnew", "vehusd",  # de-merged episodic-capital purchases (Build 1)
        "airshp", "hotel",
    ),
    "health": ("health",),
    "entertainment": (
        "entertainment",   # backend aggregate (oeprd + oesrv + streaming add)
        "recrp",           # Recreational products (gear: boats/RVs/instruments)
    ),
    "shopping": (
        "shopping",        # backend aggregate (cloftw apparel + carved pcare)
        # User-POV grouping (overrides CEX major): buying electronics/jewelry is a
        # "shopping" purchase to a user, even though CEX files eltrnp under
        # Entertainment. recrp (recreation gear) stays with entertainment.
        "eltrnp",          # Electronic products (TVs/computers/audio — a purchase)
        "jwlbg",           # Jewelry and handbags
        # Folded in (2026-06): the former standalone household_goods topic. For
        # most users this is one bucket with shopping ("stuff I buy that isn't
        # food/housing/transport/services"); folding drops a top-level header
        # from the initial view while preserving per-category granularity. PURE
        # display re-grouping — the backend aggregate `household_goods` (backbone
        # hhpcp+hhpcs) and the de-merged durables keep their per-category records,
        # value-layer factors, and cadence; only their topic placement changed.
        "household_goods", # backend aggregate (backbone: hhpcp, hhpcs)
        "furhwr", "happl", # de-merged episodic-capital durables (Build 1)
    ),
    "financial_debt": (
        "finpay", "othint", "chrty", "ocash",
        "stdint",  # ⚑ provisional — debt service
        "educ",    # ⚑ provisional — borderline
    ),
    # Outside spending: the 9 balance categories. Never summed into any
    # spending-group total; carried as net-worth context only.
    "net_worth": (
        "check", "retire", "stock", "othfin", "lifval",
        "vehval", "ownval", "othdbt", "stddbt",
    ),
}

TOPIC_GROUP_NAMES: list[str] = list(_TOPIC_MEMBERS.keys())

# The requested category -> topic mapping (reverse index of _TOPIC_MEMBERS).
TOPIC_GROUP: dict[str, str] = {
    cat: topic for topic, members in _TOPIC_MEMBERS.items() for cat in members
}

# The single spending vs. non-spending split: net_worth is the only
# non-spending topic. A topic group's total is summed only when it is a
# spending topic.
TOPIC_NET_WORTH: str = "net_worth"
SPENDING_TOPIC_NAMES: list[str] = [
    t for t in TOPIC_GROUP_NAMES if t != TOPIC_NET_WORTH
]

# Partition assert: every one of the 39 aggregated categories maps to exactly
# one topic, and no stray code sneaks in. A forgotten category cannot ship.
assert set(TOPIC_GROUP) == set(AGGREGATED_CATEGORY_CODES), (
    "TOPIC_GROUP must cover exactly the 46 aggregated categories"
)
assert len(TOPIC_GROUP) == 46
# No category in two topics (tuple concatenation would have collided in the
# reverse-index build, but assert the count explicitly to be safe).
assert sum(len(m) for m in _TOPIC_MEMBERS.values()) == 46, (
    "a category appears in more than one topic"
)
# net_worth is exactly the balance categories — balance cats are outside
# spending, and only balance cats are.
assert set(_TOPIC_MEMBERS[TOPIC_NET_WORTH]) == set(AGGREGATED_BALANCE_CATEGORIES)
assert all(
    TOPIC_GROUP[c] == TOPIC_NET_WORTH for c in AGGREGATED_BALANCE_CATEGORIES
)
assert all(
    TOPIC_GROUP[c] != TOPIC_NET_WORTH
    for c in AGGREGATED_FLOW_CATEGORIES
), "no flow category may land in net_worth"
