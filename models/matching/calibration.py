"""Cohort-level calibration for CEX diary underreporting.

CEX is a two-week diary survey. Infrequent purchases (the obvious case
being gasoline, which a car-owning household might refuel two or three
times in a diary window) can be undercaptured relative to benchmarks
built from external data — for gas, ``FHWA VMT * EIA retail price``
aggregates run ~25% above the CEX annualized mean.

Whether we need the correction at all depends on which of the
downstream scalars we ALREADY apply bring the anchor up to real
regional prices:

  * If the EIA state retail-gas scalar is available, the EIA price is
    the current pump price; its ratio to national already reflects
    the true spatial level. Applying an additional diary correction
    would inflate the anchor past pump prices.
    -> factor = 1.0.

  * If EIA is unavailable but the BLS regional gas CPI fetched
    successfully (the ``gas_{region}`` scalar in the cache is NOT
    the national fallback), the regional temporal series moves the
    anchor to current regional prices. The diary undercapture that
    the 1.25 factor was designed to correct lives primarily in the
    national-average baseline that CEX is known to lag; once the
    regional CPI re-prices the cohort median, the national-level gap
    doesn't compound on top.
    -> factor = 1.0.

  * Neither EIA nor live BLS regional — we have only the national
    CPI temporal scalar, which moves with headline inflation but
    stays anchored to the (low) CEX national mean. Apply the full
    diary-undercapture correction so the displayed p50 catches up to
    the benchmark.
    -> factor = 1.25.

Non-gas categories: no underreporting correction. The CEX diary
captures housing, utilities, rent, and services adequately; only
discretionary infrequent purchases suffer, and of those only gas has
a strong-enough independent benchmark to justify a multiplicative
correction.
"""

from __future__ import annotations


# Tier multipliers. Exposed as module-level constants so the ratios are
# visible at a glance and can be tuned without touching call sites.
# The two "regional data available" tiers are both 1.0 so that
# regional signals don't compound with an additional undercapture
# correction — see the module docstring for the rationale.
GAS_UNDERREPORTING_EIA_AVAILABLE: float = 1.0
GAS_UNDERREPORTING_BLS_REGIONAL: float = 1.0
GAS_UNDERREPORTING_NATIONAL_ONLY: float = 1.25


def gas_underreporting_factor(
    eia_available: bool,
    bls_regional_available: bool,
) -> float:
    """Return the multiplicative gas underreporting correction.

    See module docstring for the tier semantics.
    """
    if eia_available:
        return GAS_UNDERREPORTING_EIA_AVAILABLE
    if bls_regional_available:
        return GAS_UNDERREPORTING_BLS_REGIONAL
    return GAS_UNDERREPORTING_NATIONAL_ONLY
