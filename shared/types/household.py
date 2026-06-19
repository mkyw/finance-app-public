"""Household profile type: the input to matching, Engel curves, and benefits eligibility."""

from dataclasses import dataclass

from shared.types.enums import Tenure


@dataclass(frozen=True)
class HouseholdProfile:
    age: int
    gross_income: float
    puma_code: str
    tenure: Tenure
    housing_cost: float
    household_size: int
    equivalized_income: float = 0.0
    # User-reported liquid savings. Surfaced as
    # ``balance_sheet.assets.check.user_reported`` in the analyze
    # response so the UI can show the user's own number alongside
    # cohort percentiles. Defaults to 0 so existing call sites that
    # don't collect savings don't need to change.
    savings: float = 0.0
    # User-reported debt. All four default to 0 = "no signal, use the
    # cohort prior" (same <=0 convention as ``savings``), so existing
    # call sites and non-debt users see byte-identical predictions: the
    # debt-service stage falls back to the cohort-predicted othdbt/stddbt
    # balances (mostly $0 at p50). When present, these OVERRIDE the cohort
    # prediction component-wise (DEBT-POST-ALLOCATION-OVER-COHORT-SHIFT):
    # debt-service subtracts post-cohort-matching from d_variable, never
    # adjusts the cohort match. Credit cards are collected as a carried
    # *balance* ("carry month to month", excludes pay-in-full); the other
    # three as monthly *payments* (their natural, answerable quantity).
    cc_carried_balance: float = 0.0   # CC balance carried month to month
    student_loan_payment: float = 0.0  # monthly student-loan payment
    auto_loan_payment: float = 0.0     # monthly auto-loan payment (sum if multiple)
    other_debt_payment: float = 0.0    # monthly payment toward other debts
    # Census FIPS identifiers from the city resolver (resolve-city endpoint).
    # Drive the local/municipal income tax lookup (models/tax/local.py).
    # Empty = unresolved; NYC remains detected via the NY_04 PUMA prefix.
    place_fips: str = ""    # 7-digit place FIPS (e.g. "3651000" = NYC)
    county_fips: str = ""   # 5-digit county FIPS (e.g. "24031" = Montgomery MD)

    @staticmethod
    def _equivalence_scale(household_size: int) -> float:
        """Square-root equivalence scale.
        MUST match pipeline/export/export_coefficients.R.
        Returns divisor d where y_eq = gross_income / d.
        """
        import math
        return math.sqrt(household_size)

    def __post_init__(self) -> None:
        # Default-fill equivalized_income when the caller didn't supply it.
        # Frozen dataclasses require object.__setattr__ to mutate fields.
        if self.equivalized_income == 0.0 and self.household_size > 0:
            scale = self._equivalence_scale(self.household_size)
            object.__setattr__(
                self, "equivalized_income", self.gross_income / scale
            )
