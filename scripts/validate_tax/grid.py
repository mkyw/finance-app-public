"""Boundary-dense scenario grid for Gate 1 — cliffs enumerated from the engines'
own parameters, NOT random sampling.

A tax function is piecewise-linear; the only places our arithmetic can diverge
from an oracle are the KINKS. So we enumerate them directly:

  * Federal (from taxcalc ``Policy`` 2026): income-tax bracket edges, the
    standard-deduction amounts (the standard-vs-itemized crossover), the SS wage
    base, the Additional-Medicare thresholds, the AMT exemption phase-out start,
    the SALT cap + its high-earner phase-down start, the CTC phase-out start, and
    the EITC plateau/phase-out knees (the structured-EITC low-earner region).
  * State (from our ``state_tax_rates.json``): every bracket edge per filing
    status, and the folded surtax thresholds (CA MHST $1M, MA $1.08M, ME $1M/1.5M).

Each cliff income is emitted at ``edge-1, edge, edge+1`` so a misplaced or
off-by-one threshold shows up as a one-sided disagreement. Scenarios cross
{single, MFJ, HoH} × household sizes × {standard, itemizing}, and carry the
state's research-sweep ``confidence`` (for Gate-1 routing) and a ``cliff_tag``
(what boundary they probe).

Pure + oracle-free: imports taxcalc (our federal engine's parameter source) and
our state schedule loader only. Runs in the repo ``.venv``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TAX_YEAR = 2026

# taxcalc MARS index by filing status (0=single, 1=MFJ, 3=HoH).
_MARS_IDX = {"single": 0, "married_filing_jointly": 1, "head_of_household": 3}
_FILINGS = ("single", "married_filing_jointly", "head_of_household")
# Realistic income ceiling: a few states' top bracket edges sit at $5M–$25M, which
# no app user reaches and which only exercises high-earner recapture (an
# unmodeled state feature). Cap the grid so the gate probes the realistic range;
# the folded surtax thresholds (CA/MA $1.0–1.1M) stay below it.
_MAX_GRID_INCOME = 2_000_000.0


@dataclass(frozen=True)
class TaxScenario:
    """One identical-input tax scenario fed to BOTH our stack and the oracle.

    The itemizables are fed to both sides as if reported, so the only possible
    Gate-1 delta is tax-law arithmetic (not prediction). ``confidence`` is the
    state schedule's research-sweep tag; ``cliff_tag`` records which boundary the
    income probes.
    """

    gross_income: float
    filing_status: str
    state: str                       # 2-letter (e.g. "CA"); "" = federal-only
    num_dependents: int = 0
    dependent_ages: tuple[int, ...] = ()
    primary_age: int = 40
    itemize: bool = False
    # Itemizables (reported identically to both engines when ``itemize``).
    mortgage_interest: float = 0.0
    property_tax: float = 0.0
    charity: float = 0.0
    medical: float = 0.0
    cliff_tag: str = ""
    confidence: str = "high"

    def key(self) -> str:
        return (
            f"{self.state or 'US'}|{self.filing_status}|{self.gross_income:.0f}"
            f"|dep{self.num_dependents}|{'item' if self.itemize else 'std'}"
        )


def _policy():
    import warnings

    import taxcalc

    warnings.filterwarnings("ignore")
    pol = taxcalc.Policy()
    pol.set_year(_TAX_YEAR)
    return pol


def _federal_cliff_incomes(filing: str) -> dict[float, str]:
    """Federal kink incomes for ``filing`` → {income: cliff_tag}, from taxcalc."""
    pol = _policy()
    i = _MARS_IDX[filing]

    def col(name: str) -> float:
        v = getattr(pol, name)
        row = v[0]
        return float(row[i]) if hasattr(row, "__len__") else float(row)

    def scalar(name: str) -> float:
        return float(getattr(pol, name)[0])

    cliffs: dict[float, str] = {}
    for brk in ("II_brk1", "II_brk2", "II_brk3", "II_brk4", "II_brk5", "II_brk6"):
        cliffs[col(brk)] = f"fed_bracket_{brk}"
    cliffs[col("STD")] = "fed_std_deduction"          # std-vs-itemized crossover region
    cliffs[scalar("SS_Earnings_c")] = "ss_wage_base"
    cliffs[col("AMEDT_ec")] = "addl_medicare_threshold"
    cliffs[col("AMT_em_ps")] = "amt_exemption_phaseout"
    cliffs[col("ID_AllTaxes_c_ps")] = "salt_cap_phasedown_start"
    cliffs[col("CTC_ps")] = "ctc_phaseout_start"
    # EITC plateau end (2-child column) — the structured-EITC low-earner region.
    # EITC_ps is 2-D (indexed by number of qualifying children 0..3), not scalar.
    eitc_ps = getattr(pol, "EITC_ps")[0]
    cliffs[float(eitc_ps[2])] = "eitc_plateau_end"
    return cliffs


def _state_cliff_incomes(state: str, filing: str, table) -> dict[float, str]:
    """State kink incomes (bracket edges + folded surtax thresholds) for one
    (state, filing) from the loaded ``StateTaxTable``."""
    cliffs: dict[float, str] = {}
    rule = table.by_state.get(state)
    if rule is None or rule.kind == "none":
        return cliffs
    if rule.kind == "brackets" and rule.brackets_by_filing_status:
        sched = rule.brackets_by_filing_status.get(
            filing, rule.brackets_by_filing_status.get("single")
        )
        if sched:
            for upper, _rate in sched:
                # ``upper`` is a taxable-income edge; add back this filing's std +
                # exemptions so the GROSS income lands the household on the edge.
                # The open top bracket is stored as None OR inf (the _brackets
                # null→inf idiom) — skip both (no finite edge to probe).
                if upper is None or not math.isfinite(float(upper)):
                    continue
                std = rule.standard_deduction_by_filing_status.get(
                    filing, rule.standard_deduction_by_filing_status.get("single", 0.0)
                )
                n_filers = 2 if filing == "married_filing_jointly" else 1
                gross_at_edge = float(upper) + std + rule.personal_exemption * n_filers
                if gross_at_edge > _MAX_GRID_INCOME:
                    continue
                cliffs[gross_at_edge] = f"{state}_bracket_edge"
    return cliffs


def _load_state_table():
    import os

    os.environ.setdefault("ARTIFACTS_PATH", str(_REPO / "pipeline" / "artifacts"))
    from models.tax.state import load_state_tax_table

    return load_state_tax_table(str(_REPO / "pipeline" / "artifacts"))


def _confidence_map() -> dict[str, str]:
    src = _REPO / "pipeline" / "export" / "state_income_tax_2026.json"
    try:
        return json.loads(src.read_text()).get("confidence", {})
    except (OSError, json.JSONDecodeError):
        return {}


# Representative itemizable bundle for the {itemizing} dimension — a mid mortgage +
# property tax that, with SALT, clears the standard deduction for most filers.
_ITEMIZE_BUNDLE = dict(mortgage_interest=18_000.0, property_tax=6_000.0, charity=2_000.0, medical=0.0)

# Household compositions probed: (num_dependents, dependent_ages).
_HOUSEHOLDS = {
    0: (),
    1: (8,),       # one CTC-eligible child
    2: (8, 14),    # two children (one <6 boundary nearby via age, one teen)
    3: (3, 8, 14),
}


def _emit(income: float, filing: str, state: str, deps: int, ages: tuple[int, ...],
          itemize: bool, tag: str, conf: str) -> TaxScenario:
    extra = _ITEMIZE_BUNDLE if itemize else {}
    return TaxScenario(
        gross_income=max(0.0, round(income, 2)), filing_status=filing, state=state,
        num_dependents=deps, dependent_ages=ages, itemize=itemize,
        cliff_tag=tag, confidence=conf, **extra,
    )


def generate_grid(scope: str = "full") -> list[TaxScenario]:
    """Generate the boundary-dense scenario list.

    ``scope``:
      * ``"band"`` — a compact federal-focused sample (TX, no state tax) for
        measuring the inter-oracle band fast.
      * ``"state"`` — every taxing jurisdiction × {single, MFJ} at its own bracket
        edges + a few generic incomes (the 4b adjudication set).
      * ``"full"``  — band ∪ state ∪ the itemizing + multi-dependent crosses.
    """
    table = _load_state_table()
    if table is None:
        raise SystemExit("state_tax_rates.json missing — cannot build the state grid.")
    conf = _confidence_map()
    out: list[TaxScenario] = []
    seen: set[str] = set()

    def add(s: TaxScenario) -> None:
        if s.key() not in seen:
            seen.add(s.key())
            out.append(s)

    # --- Federal arithmetic band: TX (no state tax) isolates federal. ---
    for filing in _FILINGS:
        for inc, tag in _federal_cliff_incomes(filing).items():
            for d in (-1.0, 0.0, 1.0):
                add(_emit(inc + d, filing, "TX", 0, (), False, tag, "high"))
    if scope == "band":
        return out

    # --- State adjudication: each taxing jurisdiction at its own edges. ---
    generic = [30_000.0, 75_000.0, 150_000.0, 300_000.0]
    surtax_pts = {  # folded surtax thresholds (probe just above)
        "CA": [1_000_000.0, 1_000_001.0], "MA": [1_083_150.0, 1_083_151.0],
        "ME": [1_000_000.0, 1_500_000.0],
    }
    for state in sorted(table.by_state):
        rule = table.by_state[state]
        if rule.kind == "none":
            add(_emit(150_000.0, "single", state, 0, (), False, "no_tax_state", conf.get(state, "high")))
            continue
        c = conf.get(state, "high")
        for filing in ("single", "married_filing_jointly"):
            edges = _state_cliff_incomes(state, filing, table)
            for inc, tag in edges.items():
                for dd in (-1.0, 0.0, 1.0):
                    add(_emit(inc + dd, filing, state, 0, (), False, tag, c))
            for inc in generic:
                add(_emit(inc, filing, state, 0, (), False, f"{state}_generic", c))
        for inc in surtax_pts.get(state, []):
            add(_emit(inc, "single", state, 0, (), False, f"{state}_surtax", c))
        # Low-earner-with-dependents for EVERY taxing state — exercises the state
        # EITC (numeric-pct states) + low-income credits/exemptions so the oracle
        # adjudicates the Stage-6 state-EITC pass (structured CA/MN/WI/DE stay
        # deferred → still model-scope here).
        for inc in (14_000.0, 22_000.0, 30_000.0):
            add(_emit(inc, "head_of_household", state, 2, (4, 8), False, f"{state}_eitc_lowearner", c))
    if scope == "state":
        return out

    # --- Itemizing + multi-dependent crosses (high-SALT HCOL owners). ---
    for state in ("CA", "NY", "NJ", "TX"):
        c = conf.get(state, "high")
        for filing in ("single", "married_filing_jointly"):
            for inc in (150_000.0, 300_000.0, 600_000.0):
                add(_emit(inc, filing, state, 0, (), True, f"{state}_itemizing", c))
        for deps, ages in _HOUSEHOLDS.items():
            if deps:
                add(_emit(120_000.0, "married_filing_jointly", state, deps, ages, False,
                          f"{state}_dependents", c))
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(_REPO))
    scope = sys.argv[1] if len(sys.argv) > 1 else "full"
    grid = generate_grid(scope)
    print(f"scope={scope}: {len(grid)} scenarios")
    from collections import Counter

    tags = Counter(s.cliff_tag.split("_")[0] if s.state else "fed" for s in grid)
    print("by area:", dict(tags))
    print("itemizing:", sum(s.itemize for s in grid), "| with deps:", sum(s.num_dependents > 0 for s in grid))
