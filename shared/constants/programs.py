"""Benefits program definitions, FPL thresholds, and enrollment URLs.

FPL = Federal Poverty Level. Base table is for the 48 contiguous US states
plus DC. Alaska and Hawaii have higher thresholds (populate later).
"""

# 2024 HHS Federal Poverty Guidelines, 48 contiguous states + DC.
# Annual income threshold keyed by household size (1..8).
# For each additional person beyond 8, add $5,380.
FPL_2024_CONTIGUOUS: dict[int, int] = {
    1: 15060,
    2: 20440,
    3: 25820,
    4: 31200,
    5: 36580,
    6: 41960,
    7: 47340,
    8: 52720,
}
FPL_2024_ADDITIONAL_PERSON: int = 5380


class FederalProgram:
    SNAP = "snap"
    EITC = "eitc"
    MEDICAID = "medicaid"
    ACA_PTC = "aca_ptc"  # Premium Tax Credit on ACA Marketplace
    LIHEAP = "liheap"


PROGRAMS: dict[str, dict] = {
    FederalProgram.SNAP: {
        "name": "Supplemental Nutrition Assistance Program",
        "threshold_pct_fpl": 130,
        "enrollment_url": "https://www.fns.usda.gov/snap/state-directory",
    },
    FederalProgram.EITC: {
        "name": "Earned Income Tax Credit",
        "inputs_required": ("income", "filing_status", "qualifying_children"),
        "enrollment_url": "https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit-eitc",
    },
    FederalProgram.MEDICAID: {
        "name": "Medicaid",
        "threshold_pct_fpl_expansion_states": 138,
        "inputs_required": ("state", "income", "household_size"),
        "enrollment_url": "https://www.healthcare.gov/medicaid-chip/",
    },
    FederalProgram.ACA_PTC: {
        "name": "ACA Marketplace Premium Tax Credit",
        "threshold_pct_fpl_min": 100,
        "threshold_pct_fpl_max": 400,
        "enrollment_url": "https://www.healthcare.gov/",
    },
    FederalProgram.LIHEAP: {
        "name": "Low Income Home Energy Assistance Program",
        "threshold_pct_fpl": 150,
        "inputs_required": ("income", "tenure", "state"),
        "enrollment_url": "https://www.acf.hhs.gov/ocs/programs/liheap",
    },
}

# States that did NOT expand Medicaid under the ACA (as of 2024).
MEDICAID_NON_EXPANSION_STATES: frozenset[str] = frozenset(
    {"AL", "FL", "GA", "KS", "MS", "SC", "TN", "TX", "WI", "WY"}
)
