"""Filing-unit imputation from the app's inputs (income, location, household
size, householder age) — the "best tax calc given what we know" principle.

The app does not run a tax-prep questionnaire: it collects householder ``age``
and ``household_size`` and a (defaulted) ``filing_status``, but never a dependent
count or dependent ages. Federal CTC/EITC and the state credits that mirror them
need those, so we impute them from typical demographic standards. The imputed
unit is surfaced in the response (USER-ADJUSTMENT-AUTHORITY) so the user can
correct it — correction, not a future pipeline change, is the accuracy path; the
imputation is just the sensible default until then.

Conservative-direction discipline (Point 4 / the locked over-prediction asymmetry):
dependents and a more favorable filing status are *tax-reducing* → they *raise*
take-home → over-imputing them would over-state slack, the forbidden direction
(`DEBT-POST-ALLOCATION-OVER-COHORT-SHIFT`, `OVER-PREDICTION-EXTENDED-TO-COMMITTED-OUTFLOWS`).
So at every ambiguous margin we lean toward the *less* tax-reducing assumption:
  - filing status is only ever *repaired downward* (an impossible size-1 MFJ/HoH
    becomes single); we never *upgrade* a declared ``single`` to ``head_of_household``
    even when household_size suggests a single parent — the user upgrades it.
  - dependents are counted only while their imputed age is **< 19** (no student
    extension to 24 — fewer dependents), and CTC only while **< 17** (a hard edge).

A future cohort-empirical upgrade (drawing the filing structure from the matched
ACS tax units) is possible but NOT required — it needs the ACS export to retain
HHT/NOC/child-age, which it currently drops. `[[FILING-UNIT-COHORT-SOURCE]]` is an
optional forward-note, not a blocking deferral.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.types import HouseholdProfile

# Valid filing statuses — must match ``models.tax.calculator._MARS`` keys.
_VALID_STATUSES: frozenset[str] = frozenset(
    {"single", "married_filing_jointly", "head_of_household"}
)

# Typical-standards constants (knowledge anchors, documented; not law).
# US mean age at first birth is ~27–30; 27 is a defensible central value.
TYPICAL_FIRST_BIRTH_AGE: int = 27
CHILD_SPACING_YEARS: int = 3

# Age limits (conservative — bias toward fewer tax-reducing claims).
DEPENDENT_AGE_LIMIT: int = 19   # dependent only while < 19 (no student extension to 24)
CTC_AGE_LIMIT: int = 17         # child tax credit: child must be < 17
CDCC_AGE_LIMIT: int = 13        # child & dependent care credit: < 13
YOUNG_CHILD_AGE_LIMIT: int = 6  # the higher young-child bucket: < 6
_MAX_EIC_CHILDREN: int = 3      # taxcalc EIC input is capped at 3

_DEFAULT_HOUSEHOLDER_AGE: int = 40


@dataclass(frozen=True)
class FilingUnit:
    """The imputed tax filing unit consumed by the federal Records build.

    All counts are conservative (lean against tax-reducing structure). The
    ``imputed`` flag and ``notes`` drive the surfaced "we assumed X — correct
    it" explanation; ``declared_filing_status`` records what the user gave so
    the UI can show the repair.
    """

    filing_status: str            # repaired status (one of _VALID_STATUSES)
    declared_filing_status: str   # what the user provided (pre-repair)
    num_dependents: int           # dependents (imputed age < 19)
    n_children_under_18: int      # taxcalc nu18
    n_children_under_17: int      # CTC-eligible (taxcalc n24)
    n_children_under_13: int      # CDCC-eligible (taxcalc nu13 / f2441)
    n_children_under_6: int       # young-child bucket (taxcalc nu06)
    eic_qualifying_children: int  # taxcalc EIC (0..3)
    dependent_ages: tuple[int, ...]
    spouse_age: int | None        # MFJ only; ~householder age (matters near 65)
    imputed: bool                 # True if any field was imputed/repaired
    notes: tuple[str, ...]        # human-readable, for the surfaced explanation


def _normalize_status(declared: str) -> str:
    """Coerce an unrecognized status to ``single`` (the safe default)."""
    s = (declared or "").strip().lower()
    return s if s in _VALID_STATUSES else "single"


def _repair_filing_status(
    declared: str, household_size: int
) -> tuple[str, list[str]]:
    """Repair only *impossible* statuses; never upgrade to a favorable one.

    A size-1 household cannot be MFJ (no spouse) or HoH (no qualifying
    dependent) → coerce to single (which *raises* tax → conservative-safe).
    Everything else is kept as declared: a declared ``single`` with
    household_size >= 2 stays single (we do NOT auto-upgrade to HoH — that
    would lower tax; the user upgrades it).
    """
    notes: list[str] = []
    status = _normalize_status(declared)
    if status != (declared or "").strip().lower():
        notes.append(
            f"Unrecognized filing status {declared!r}; defaulted to 'single'."
        )
    if household_size <= 1 and status != "single":
        notes.append(
            f"Filing status {status!r} needs a second household member but "
            f"household_size is {household_size}; using 'single'."
        )
        status = "single"
    return status, notes


def _impute_dependent_ages(
    householder_age: int, num_candidate_children: int
) -> list[int]:
    """Impute descending child ages from typical first-birth age + spacing.

    Oldest child ≈ householder_age − TYPICAL_FIRST_BIRTH_AGE; younger children
    each CHILD_SPACING_YEARS apart; floored at 0 (newborn). Candidate children
    whose imputed age is >= DEPENDENT_AGE_LIMIT are adult household members, not
    tax dependents, and are dropped here (conservative).
    """
    if num_candidate_children <= 0:
        return []
    oldest = householder_age - TYPICAL_FIRST_BIRTH_AGE
    ages = [max(0, oldest - CHILD_SPACING_YEARS * i) for i in range(num_candidate_children)]
    # Keep only plausibly-dependent children (age < 19, no student extension).
    return [a for a in ages if a < DEPENDENT_AGE_LIMIT]


def impute_filing_unit(
    profile: HouseholdProfile, declared_filing_status: str = "single"
) -> FilingUnit:
    """Impute the tax filing unit from the profile's inputs.

    Uses householder ``age`` and ``household_size`` + the declared filing
    status. Conservative at every tax-reducing margin (see module docstring).
    """
    household_size = max(1, int(profile.household_size))
    householder_age = int(profile.age) if profile.age and profile.age > 0 else _DEFAULT_HOUSEHOLDER_AGE

    status, notes = _repair_filing_status(declared_filing_status, household_size)

    adults = 2 if status == "married_filing_jointly" else 1
    num_candidate_children = max(0, household_size - adults)

    dependent_ages = _impute_dependent_ages(householder_age, num_candidate_children)
    num_dependents = len(dependent_ages)
    n_under_18 = sum(1 for a in dependent_ages if a < 18)
    n_under_17 = sum(1 for a in dependent_ages if a < CTC_AGE_LIMIT)
    n_under_13 = sum(1 for a in dependent_ages if a < CDCC_AGE_LIMIT)
    n_under_6 = sum(1 for a in dependent_ages if a < YOUNG_CHILD_AGE_LIMIT)
    eic_children = min(num_dependents, _MAX_EIC_CHILDREN)

    spouse_age = householder_age if status == "married_filing_jointly" else None

    if num_candidate_children > 0:
        notes.append(
            f"Imputed {num_dependents} dependent(s) (ages {dependent_ages}) from "
            f"household_size {household_size} and householder age {householder_age}; "
            f"{n_under_17} under 17 (CTC). Provide your real filing status / "
            f"dependents to refine."
        )
    elif household_size > adults:
        notes.append(
            f"Household_size {household_size} exceeds {adults} adult(s) but the "
            f"extra member(s) imputed as non-dependent adults (age >= "
            f"{DEPENDENT_AGE_LIMIT}); no dependents claimed."
        )

    imputed = bool(notes)

    return FilingUnit(
        filing_status=status,
        declared_filing_status=_normalize_status(declared_filing_status),
        num_dependents=num_dependents,
        n_children_under_18=n_under_18,
        n_children_under_17=n_under_17,
        n_children_under_13=n_under_13,
        n_children_under_6=n_under_6,
        eic_qualifying_children=eic_children,
        dependent_ages=tuple(dependent_ages),
        spouse_age=spouse_age,
        imputed=imputed,
        notes=tuple(notes),
    )
