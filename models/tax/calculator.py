"""Unified tax calculator — federal via taxcalc, state + local/municipal + FICA layered on top.

Federal income tax uses the Tax-Calculator library (Urban/TPC), which
encodes current-law 2026 parameters (post-OBBBA / TCJA extensions).
taxcalc is federal-only, so state income tax still reads the flat-rate
dict in ``models.tax.state`` and local/municipal income tax is looked
up from ``pipeline/artifacts/local_tax_rates.json`` via
``models.tax.local.compute_local_tax`` (NYC included; the ``NY_04``
PUMA prefix back-compat fallback lives in local.py). FICA is computed
per 2026 wage base with the Additional Medicare Tax.

taxcalc is expensive to import (numba + bokeh). Heavy imports are
deferred into ``_compute_federal_via_taxcalc`` and the ``Policy``
object is cached across calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from models.tax.itemize import ScheduleA
from models.tax.local import compute_local_tax
from shared.constants.state_payroll_tax import load_state_payroll_tax

__all__ = ["TaxBreakdown", "compute_tax"]

_TAX_YEAR: int = 2026

# taxcalc MARS codes: 1=single, 2=MFJ, 3=MFS, 4=HoH, 5=QW.
_MARS: dict[str, int] = {
    "single": 1,
    "married_filing_jointly": 2,
    "head_of_household": 4,
}

_SS_RATE: float = 0.062
_MEDICARE_RATE: float = 0.0145
_AMT_RATE: float = 0.009
# Durable SS wage base: sourced from taxcalc Policy at runtime (see
# ``_ss_wage_base``); this literal is only the fallback if that read fails.
_SS_WAGE_BASE_FALLBACK: float = 184_500.0  # 2026 OASDI max taxable earnings


@dataclass
class TaxBreakdown:
    # --- closure-bearing (unchanged; take_home = gross - total_tax) ---
    federal_tax: float           # taxcalc iitax net of the Additional Medicare Tax
    state_tax: float             # state income tax net of the state EITC (Stage 6)
    city_tax: float
    fica: float
    state_payroll_tax: float
    total_tax: float
    take_home: float
    tax_year: int = _TAX_YEAR
    # --- additive DISPLAY fields (Stage 6; all already netted INSIDE the
    # closure-bearing lines above — surfaced for the gross->take-home wedge, they
    # do NOT change any total). Default 0 → back-compat for detail=None callers. ---
    federal_amt: float = 0.0           # taxcalc c09600 (alternative minimum tax)
    federal_niit: float = 0.0          # net investment income tax (0 — wage-only model)
    federal_eitc: float = 0.0          # earned income tax credit (already in federal_tax)
    federal_ctc: float = 0.0           # child tax credit (already in federal_tax)
    federal_agi: float = 0.0           # taxcalc c00100
    federal_taxable_income: float = 0.0  # taxcalc c04800
    itemized_deductions: float = 0.0   # taxcalc c04470 (allowed itemized)
    standard_deduction: float = 0.0    # the standard deduction for this filing unit
    itemized: bool = False             # True if taxcalc itemized (c04470 > standard)
    state_eitc: float = 0.0            # state EITC applied (numeric-pct states; reduces state_tax)
    effective_rate: float = 0.0        # total_tax / gross_income


@dataclass(frozen=True)
class TaxDetail:
    """Rich tax-computation inputs beyond gross income + filing status.

    Carries the imputed filing unit's dependent buckets (CTC/EITC) and the
    pre-tax wedge (exclusions that reduce the income-tax / FICA wage bases). All
    fields default to zero, so a default ``TaxDetail`` (or ``detail=None``)
    reproduces the gross-income / no-dependent / no-wedge behavior exactly.
    """

    n_children_under_17: int = 0   # taxcalc n24 (CTC-eligible)
    n_children_under_18: int = 0   # taxcalc nu18
    n_children_under_6: int = 0    # taxcalc nu06
    eic_children: int = 0          # taxcalc EIC (0..3)
    # Committed-baseline income-tax exclusion: reduces BOTH the federal and the
    # state income-tax base (states conform to 401(k)/§125). Stable across the
    # pre-tax fixed-point iterations.
    pretax_income_tax_excludable: float = 0.0
    # Iterating waterfall top-ups (401k/HSA): reduce the FEDERAL income-tax base
    # ONLY — state stays on the baseline, outside the loop.
    pretax_income_tax_excludable_topup: float = 0.0
    pretax_fica_excludable: float = 0.0        # reduces the FICA wage base (baseline §125)
    # Schedule A itemizables (predicted, lean-low; ``models.tax.itemize``). All
    # default 0 → no itemized deductions → taxcalc uses the standard deduction
    # (byte-identical to the pre-Stage-5 behavior). The SALT *income*-tax line
    # (e18400) is NOT here — compute_tax fills it from the computed state+local
    # tax. taxcalc applies the OBBBA SALT cap + 7.5%-AGI medical floor + charity
    # limit + the standard-vs-itemized choice internally.
    itemized_mortgage_interest: float = 0.0    # taxcalc e19200
    itemized_property_tax: float = 0.0         # taxcalc e18500 (SALT-capped)
    itemized_charity: float = 0.0              # taxcalc e19800 (cash)
    itemized_medical: float = 0.0              # taxcalc e17500 (7.5%-AGI floor)


def _compute_fica(wage_base: float, filing_status: str = "single") -> float:
    """Employee-side FICA on ``wage_base`` (wages net of any §125 pre-tax
    exclusions — HSA/FSA/commuter/§125 premium; traditional 401(k) does NOT
    reduce this base). SS to the 2026 OASDI wage base, Medicare on all wages,
    plus the 0.9% Additional Medicare over the filing threshold."""
    if wage_base <= 0:
        return 0.0
    ss = min(wage_base, _ss_wage_base()) * _SS_RATE
    medicare = wage_base * _MEDICARE_RATE
    amt_threshold = 250_000.0 if filing_status == "married_filing_jointly" else 200_000.0
    if wage_base > amt_threshold:
        medicare += (wage_base - amt_threshold) * _AMT_RATE
    return ss + medicare


def _compute_state_payroll_tax(gross_income: float, puma_code: str) -> float:
    """Employee-side state payroll/disability/PFML tax — a sibling of FICA.

    The mandatory employee withholding (CASDI/SDI/TDI, PFL/PFML, WA Cares,
    employee UI) the federal FICA line does not cover. Reads
    ``state_payroll_tax.json`` via the registry loader; a missing artifact or
    a state with no rule → 0.0 (clean no-op, byte-identical to no layer).
    """
    if gross_income <= 0 or not puma_code:
        return 0.0
    from models.tax.state import state_from_puma

    table = load_state_payroll_tax()
    if table is None:
        return 0.0
    try:
        state = state_from_puma(puma_code)
    except ValueError:
        return 0.0
    return table.amount_for(gross_income, state)


@lru_cache(maxsize=1)
def _policy_2026():
    from taxcalc import Policy

    pol = Policy()
    pol.set_year(_TAX_YEAR)
    return pol


@lru_cache(maxsize=1)
def _ss_wage_base() -> float:
    """2026 OASDI max taxable earnings, sourced from taxcalc's authoritative
    Policy parameter (``SS_Earnings_c``) so it tracks the library and never
    re-stales — the durable fix for the formerly-hardcoded constant. Falls back
    to the documented 2026 literal only if the param read fails."""
    try:
        return float(_policy_2026().SS_Earnings_c[0])
    except Exception:
        return _SS_WAGE_BASE_FALLBACK


@dataclass(frozen=True)
class FederalDetail:
    """taxcalc federal outputs. ``federal_tax`` is the closure-bearing net income
    tax (iitax less the Additional Medicare Tax — FICA owns that); the rest are
    DISPLAY components already netted inside ``federal_tax`` (eitc/ctc/amt/niit),
    plus the deduction picture (for the gross->take-home wedge + the state EITC,
    which is a fraction of ``eitc``)."""

    federal_tax: float = 0.0
    eitc: float = 0.0
    ctc: float = 0.0
    amt: float = 0.0
    niit: float = 0.0
    agi: float = 0.0
    taxable_income: float = 0.0
    itemized_deductions: float = 0.0
    standard_deduction: float = 0.0
    itemized: bool = False


def _compute_federal_via_taxcalc(
    taxable_wages: float,
    filing_status: str,
    num_dependents: int,
    *,
    n24: int | None = None,
    nu18: int | None = None,
    nu06: int = 0,
    eic: int | None = None,
    schedule_a: ScheduleA | None = None,
) -> FederalDetail:
    """Federal income tax (2026 law) via taxcalc on ``taxable_wages`` (gross
    wages net of any pre-tax income-tax exclusions).

    Returns the NET income tax: taxcalc's ``iitax`` already nets refundable
    credits (EITC, refundable CTC), so this is NEGATIVE for a low earner due a
    refund — deliberately NOT floored (REFUNDABLE-CREDIT-RAISES-TAKE-HOME; the
    caller lets take-home exceed gross-minus-other-taxes).

    Dependent buckets default to deriving from ``num_dependents`` (back-compat,
    byte-identical); the imputed-filing-unit path passes the granular counts
    (``n24`` CTC-eligible <17, ``nu18`` <18, ``nu06`` <6, ``eic`` EITC children).

    ``schedule_a`` (``models.tax.itemize.ScheduleA``) populates the itemized
    deduction line items; taxcalc applies the OBBBA SALT cap + 7.5%-AGI medical
    floor + charity limit + the standard-vs-itemized choice internally. ``None``
    (or all-zero) → standard deduction, byte-identical to the pre-Stage-5 path.
    """
    if taxable_wages <= 0:
        return FederalDetail()
    if filing_status not in _MARS:
        raise KeyError(
            f"unknown filing_status: {filing_status!r}. "
            f"valid: {sorted(_MARS.keys())}"
        )

    import pandas as pd
    from taxcalc import Calculator, Records

    mars = _MARS[filing_status]
    dep = max(0, int(num_dependents))
    n24_v = dep if n24 is None else max(0, int(n24))
    nu18_v = dep if nu18 is None else max(0, int(nu18))
    nu06_v = max(0, int(nu06))
    eic_v = min(dep, 3) if eic is None else max(0, min(int(eic), 3))
    sa = schedule_a
    row = {
        "RECID": 1,
        "MARS": mars,
        "e00200": float(taxable_wages),
        "e00200p": float(taxable_wages),
        "e00200s": 0.0,
        "XTOT": 1 + (1 if mars == 2 else 0) + dep,
        "n24": n24_v,
        "nu18": nu18_v,
        "nu06": nu06_v,
        "EIC": eic_v,
        # Schedule A line items (0 → taxcalc uses the standard deduction). taxcalc
        # applies the SALT cap, medical floor, charity limit, and max(standard,
        # itemized) internally.
        "e19200": max(0.0, sa.mortgage_interest) if sa else 0.0,        # mortgage interest
        "e18500": max(0.0, sa.property_tax) if sa else 0.0,             # real-estate tax (SALT)
        "e18400": max(0.0, sa.state_local_income_tax) if sa else 0.0,  # state+local income tax (SALT)
        "e19800": max(0.0, sa.charity) if sa else 0.0,                 # cash charity
        "e17500": max(0.0, sa.medical) if sa else 0.0,                 # medical (7.5%-AGI floor)
        "FLPDYR": _TAX_YEAR,
        "s006": 1.0,
    }
    df = pd.DataFrame([row])
    recs = Records(
        data=df,
        start_year=_TAX_YEAR,
        gfactors=None,
        weights=None,
        adjust_ratios=None,
    )
    calc = Calculator(policy=_policy_2026(), records=recs, verbose=False)
    calc.calc_all()
    # taxcalc's ``iitax`` folds the 0.9% Additional Medicare Tax (Form 8959,
    # ``ptax_amc``) into the income-tax line. We compute that SAME tax in
    # ``_compute_fica`` on the correct Medicare base (gross net of §125, 401(k)
    # INCLUDED — vs taxcalc's income-tax base which excludes 401(k)), so leaving it
    # in here would DOUBLE-COUNT it. Subtract it so federal_tax is the pure income
    # tax and FICA owns the Additional Medicare Tax (Gate-1 caught this vs
    # PolicyEngine, whose income_tax == taxcalc c05800, AddlMedicare on the payroll
    # side). NIIT (also in iitax) is left in — it is a genuine income tax, and is
    # $0 for our wage-only model anyway.
    def _f(name: str) -> float:
        return float(calc.array(name)[0])

    iitax = _f("iitax")
    addl_medicare = _f("ptax_amc")
    itemized_ded = _f("c04470")
    standard_ded = _f("standard")
    return FederalDetail(
        federal_tax=iitax - addl_medicare,
        eitc=_f("eitc"),
        ctc=_f("c07220") + _f("c11070"),   # nonrefundable CTC + refundable additional CTC
        amt=_f("c09600"),
        niit=_f("niit"),
        agi=_f("c00100"),
        taxable_income=_f("c04800"),
        itemized_deductions=itemized_ded,
        standard_deduction=standard_ded,
        itemized=itemized_ded > standard_ded,
    )


def compute_tax(
    gross_income: float,
    filing_status: str = "single",
    puma_code: str = "",
    num_dependents: int = 0,
    place_fips: str = "",
    county_fips: str = "",
    *,
    detail: TaxDetail | None = None,
) -> TaxBreakdown:
    """Full tax breakdown for one household using taxcalc + state/local/FICA.

    Args:
        gross_income: Annual gross wage income.
        filing_status: ``"single"``, ``"married_filing_jointly"``, or
            ``"head_of_household"``.
        puma_code: ``STATE_PUMA`` string. State extracted via prefix;
            NYC detected when prefix is ``NY_04`` (back-compat fallback
            in ``models.tax.local``).
        num_dependents: Dependents used for CTC/EIC in taxcalc.
        place_fips: 7-digit Census place FIPS for local/municipal tax
            lookup (e.g. ``"3651000"`` = NYC). Empty = unresolved.
        county_fips: 5-digit Census county FIPS for local/municipal tax
            lookup (e.g. ``"24031"`` = Montgomery County MD). Empty =
            unresolved.

    Returns:
        :class:`TaxBreakdown` for tax year 2026.  ``city_tax`` carries
        the local/municipal income tax (field name unchanged for API
        stability); lookup is via ``local_tax_rates.json`` (NYC
        included).
    """
    from models.tax.state import state_eitc_pct, state_tax

    d = detail
    # Pre-tax wedge: federal income tax on wages net of income-tax exclusions
    # (committed baseline 401(k)+§125 PLUS the iterating waterfall top-ups);
    # FICA on wages net of §125 exclusions only (401(k) is FICA-taxable). The
    # committed dollars still leave the budget downstream; the wedge only makes
    # the TAX correct (PRETAX-FEEDBACK-LOOP-FROM-WATERFALL).
    it_base = d.pretax_income_tax_excludable if d else 0.0
    it_topup = d.pretax_income_tax_excludable_topup if d else 0.0
    it_wage = max(0.0, gross_income - it_base - it_topup)
    fica_wage = max(0.0, gross_income - (d.pretax_fica_excludable if d else 0.0))

    # State + local FIRST: they do NOT depend on federal AGI, and their sum is the
    # SALT income-tax deduction (e18400) fed INTO federal. So the DAG is one-pass
    # (state+local → federal), not a fixed point (SALT-CIRCULARITY risk note).
    # State BRACKETS conform to the committed BASELINE wedge only (pretax_excludable
    # = it_base, never the top-ups), so the bracket term is constant across the
    # pre-tax fixed-point iterations by construction. (The state EITC applied below
    # reads the federal EITC, which is topup-sensitive; it is loop-invariant only
    # because the EITC income regime [low income] and the waterfall-top-up regime
    # [above-cohort savers] are economically disjoint — see the state-EITC note.
    # Forward: STATE-EITC-BASELINE-FEDERAL — compute the state EITC off the
    # it_base-only federal EITC to restore the structural guarantee for the rare
    # low-income-high-saver overlap.)
    state = (
        state_tax(
            gross_income, puma_code, filing_status=filing_status,
            num_dependents=num_dependents, pretax_excludable=it_base,
        )
        if puma_code
        else 0.0
    )
    city = compute_local_tax(
        gross_income,
        filing_status,
        place_fips=place_fips,
        county_fips=county_fips,
        puma_code=puma_code,
    )
    # Schedule A: the predicted itemizables from ``detail`` (lean-low) plus the
    # SALT income component (state+local income tax) computed just above. taxcalc
    # caps SALT, floors medical, limits charity, and picks max(standard,
    # itemized). Absent ``detail`` → None → standard deduction (byte-identical).
    schedule_a = (
        ScheduleA(
            mortgage_interest=d.itemized_mortgage_interest,
            property_tax=d.itemized_property_tax,
            state_local_income_tax=state + city,
            charity=d.itemized_charity,
            medical=d.itemized_medical,
        )
        if d
        else None
    )
    fed = _compute_federal_via_taxcalc(
        it_wage,
        filing_status,
        num_dependents,
        n24=(d.n_children_under_17 if d else None),
        nu18=(d.n_children_under_18 if d else None),
        nu06=(d.n_children_under_6 if d else 0),
        eic=(d.eic_children if d else None),
        schedule_a=schedule_a,
    )
    # State EITC (Stage 6): a fraction of the federal EITC, applied here because it
    # needs the federal EITC. Numeric-pct states only (structured CA/MN/WI/DE carry
    # pct=0 → deferred). Clamped at 0 — non-refundable in our model, the SAFE
    # direction: a refundable state EITC in PolicyEngine only shrinks our over-tax
    # gap toward the oracle, never overshoots into under-tax. EITC (low income) and
    # the pre-tax waterfall top-ups (high savers) are disjoint income regimes, so
    # this stays loop-invariant despite reading the per-call federal EITC.
    state_eitc = state_eitc_pct(puma_code) * fed.eitc if puma_code else 0.0
    state_after_eitc = max(0.0, state - state_eitc)
    payroll = _compute_fica(fica_wage, filing_status)
    state_payroll = _compute_state_payroll_tax(gross_income, puma_code) if puma_code else 0.0
    total = fed.federal_tax + state_after_eitc + city + payroll + state_payroll
    # take_home = gross minus taxes only (federal_tax may be negative when
    # refundable credits exceed liability → take_home can exceed gross minus the
    # other taxes; do NOT cap at gross). Floored at 0 only as a numeric guard.
    take_home_amt = max(0.0, gross_income - total)
    return TaxBreakdown(
        federal_tax=fed.federal_tax,
        state_tax=state_after_eitc,
        city_tax=city,
        fica=payroll,
        state_payroll_tax=state_payroll,
        total_tax=total,
        take_home=take_home_amt,
        tax_year=_TAX_YEAR,
        federal_amt=fed.amt,
        federal_niit=fed.niit,
        federal_eitc=fed.eitc,
        federal_ctc=fed.ctc,
        federal_agi=fed.agi,
        federal_taxable_income=fed.taxable_income,
        itemized_deductions=fed.itemized_deductions,
        standard_deduction=fed.standard_deduction,
        itemized=fed.itemized,
        state_eitc=state_eitc,
        effective_rate=(total / gross_income if gross_income > 0 else 0.0),
    )
