"""Benefits program registry and the ``BenefitMatch`` return type.

Consumed by ``models.benefits.eligibility.screen``. The ``framing``
field on every match is required to contain the phrase "may qualify"
— the eligibility module enforces this at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenefitMatch:
    program_name: str
    estimated_monthly_min: float
    estimated_monthly_max: float
    confidence: str  # "likely" | "possible" | "check"
    enrollment_url: str
    framing: str  # always "may qualify" language


PROGRAMS: dict[str, dict[str, str]] = {
    "SNAP": {
        "name": "SNAP (food assistance)",
        "enrollment_url": "https://www.benefits.gov/benefit/361",
        "description": "Monthly food assistance benefits",
    },
    "EITC": {
        "name": "Earned Income Tax Credit",
        "enrollment_url": "https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit",
        "description": "Annual tax credit, paid as refund",
    },
    "MEDICAID": {
        "name": "Medicaid",
        "enrollment_url": "https://www.healthcare.gov/medicaid-chip/getting-medicaid-chip/",
        "description": "Free or low-cost health coverage",
    },
    "ACA": {
        "name": "ACA Marketplace subsidies",
        "enrollment_url": "https://www.healthcare.gov",
        "description": "Health insurance premium subsidies",
    },
    "LIHEAP": {
        "name": "LIHEAP (energy assistance)",
        "enrollment_url": "https://www.acf.hhs.gov/ocs/programs/liheap",
        "description": "Help with heating and cooling costs",
    },
}

__all__ = ["BenefitMatch", "PROGRAMS"]
