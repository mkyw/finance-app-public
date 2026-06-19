"""Per-category lower/upper bounds for the feasibility optimizer.

Both bounds are returned as ``np.ndarray`` indexed by
``shared.constants.categories.CATEGORY_CODES`` — the canonical column
order used throughout ``models/optimizer/``.

Lower bound (per Decision 2 in CLAUDE.md):

    lb_c = distribution[c].p10

Rare CEX categories (electronics, jewelry, student-loan interest, …)
are legitimately zero-median; when matching returns ``p10 == 0``, we
set ``lb_c = 0`` explicitly so the feasibility solver can allocate
zero without tripping the Phase 1 structural-deficit check.

Upper bound:

    ub_c = distribution[c].p90

Two fallbacks:
  * ``p90 == 0`` but ``engel_estimate > 0``: use ``engel_estimate`` as
    a soft ceiling. Matching had no signal for this category
    (zero-spread cohort), so the Engel curve is the only anchor.
  * both ``p90 == 0`` and ``engel_estimate == 0``: ``ub_c = 1.0``. The
    optimizer will naturally allocate 0 for this cat given
    ``lb_c == 0`` — the symbolic ceiling keeps CVXPY from seeing
    ``lb == ub`` (which can make OSQP unhappy).

Housing is NOT a CEX category here. Rent/mortgage costs are handled
upstream as committed fixed expenses that reduce ``d_variable``
before the solve.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from shared.constants.categories import BALANCE_CATEGORIES, CATEGORY_CODES
from shared.types import HouseholdProfile, SpendingDistribution, Tenure

_MIN_SYMBOLIC_CEILING: float = 1.0

# Mirror of ``allocator._ZERO_INFLATED_THRESHOLD``. Kept as a module-
# level constant here so ``compute_bounds`` can return the same ub
# the allocator uses for its clamp — otherwise the post-clamp in
# ``compute_allocations`` would clip anchors back to unconditional
# p90 and defeat the conditional-p90 step-D clamp.
_ZERO_INFLATED_THRESHOLD: float = 0.7

# Housing categories mutually exclusive by tenure. ``rntval`` carries
# the authoritative rent amount (FMLI ``renteqvx * 12`` — actual rent
# for renters, imputed rental equivalence for owners) and is *pinned*
# to the user's reported ``housing_cost`` inside ``compute_bounds``
# for RENT, so it shows up in the spending plan as the rent line item.
# For owners, ``rntval`` is zeroed (imputed rental-equivalence isn't a
# cash flow) and housing_cost instead pins ``mrtgip`` + ``mrtgpp`` 70/30.
#
# ``rntexp`` (renter-side expenses: appliance repairs, materials,
# tenant's insurance) is NOT tenure-mutexed: it is a genuinely
# renter-incurred bucket (26/28 UCCs are BLS "(renter)"-tagged or
# tenant's insurance) and renters report it at a *higher* nonzero rate
# than owners (~26% vs ~18%), so it must not be zeroed for RENT. It is
# zero-inflated for both tenures, which is handled symmetrically by the
# ``conditional_p90`` ceiling in ``compute_bounds`` below — no hard zero
# is needed for either tenure. (UCC 210110, "Rent", was stripped from
# this bucket in the 2026-04-22 rename so it no longer double-counts the
# rent amount already carried by ``rntval``.)
_RENTER_ZERO: frozenset[str] = frozenset(
    {"ownval", "mrtgip", "mrtgpp", "mrtgps", "ptaxp"}
)
_OWNER_ZERO: frozenset[str] = frozenset({"rntval"})

# Owner housing_cost split between mortgage interest and principal.
# Rough MVP assumption; should be refined once we collect separate
# principal/interest figures on the form.
_OWNER_MORTGAGE_INTEREST_SHARE: float = 0.70
_OWNER_MORTGAGE_PRINCIPAL_SHARE: float = 0.30


def apply_tenure_constraints(
    distributions: dict[str, SpendingDistribution],
    tenure: Tenure,
) -> dict[str, SpendingDistribution]:
    """Zero categories that should not participate in flow allocation.

    Two reasons for zeroing:

    * **Tenure mutex.** RENT households zero the five owner-only
      housing cats (``ownval``, ``mrtgip``, ``mrtgpp``, ``mrtgps``,
      ``ptaxp``) plus ``rntval``, which is already counted upstream
      as ``housing_cost``. OWN zeroes ``rntval`` only — owners may
      still carry mortgage interest/principal and property-tax line
      items beyond what ``housing_cost`` captures.

    * **Balance categories** (``BALANCE_CATEGORIES``: retire, check,
      stock, othfin, lifval, vehval, ownval, othdbt, stddbt). These
      are wealth stocks and liability balances, not annual flows.
      They're carried through matching for display but zeroed here
      so they can't anchor the allocator.

    Zeroing replaces ``p10..p90`` and ``engel_estimate`` with 0.0;
    ``compute_bounds`` then yields ``lb=0`` with the symbolic upper
    ceiling, and the allocator's primary-path anchor collapses to 0
    for those cats.

    ``SpendingDistribution`` is frozen, so excluded categories get
    fresh instances; everything else passes through unchanged.
    """
    tenure_excluded = _RENTER_ZERO if tenure is Tenure.RENT else _OWNER_ZERO
    excluded = tenure_excluded | BALANCE_CATEGORIES
    out: dict[str, SpendingDistribution] = {}
    for cat, dist in distributions.items():
        if cat in excluded:
            out[cat] = replace(
                dist,
                p10=0.0,
                p25=0.0,
                p50=0.0,
                p75=0.0,
                p90=0.0,
                engel_estimate=0.0,
                conditional_p90=0.0,
                conditional_p10=0.0,
            )
        else:
            out[cat] = dist
    return out


def compute_bounds(
    distributions: dict[str, SpendingDistribution],
    profile: HouseholdProfile,
    artifacts_path: str | None = None,  # noqa: ARG001 - reserved for future use
    category_codes: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lb, ub)`` arrays in ``category_codes`` order.

    ``category_codes`` defaults to the 55 disaggregated codes; the
    aggregated path passes ``AGGREGATED_CATEGORY_CODES`` (37). The housing
    line items (``rntval``/``mrtgip``/``mrtgpp``) are retained disaggregated
    in both sets, so the pin logic is identical — only the array indices
    differ, which is why they're resolved from ``category_codes`` here.

    After computing per-category lb/ub from the cohort distribution,
    the user's housing_cost is pinned as a fixed allocation:

      - RENT: ``rntval`` is pinned to ``housing_cost * 12``
        (``lb == ub`` so the allocator cannot move it).
      - OWN:  ``mrtgip`` is pinned to ``housing_cost * 0.70 * 12``
              ``mrtgpp`` is pinned to ``housing_cost * 0.30 * 12``

    Housing therefore appears in the spending plan as the user's own
    number instead of being silently subtracted from d_variable.

    Args:
        distributions: Output of ``engel.annotate_distributions`` — each
            entry must already have ``engel_estimate`` populated.
        profile: Household profile — used for ``housing_cost`` and
            ``tenure`` to pin the housing line item.
        artifacts_path: Reserved for future category-level overrides from
            an external table. Currently unused.

    Returns:
        ``(lb, ub)`` with ``lb.shape == ub.shape == (55,)``. Always
        ``ub[i] >= lb[i]`` elementwise; both non-negative.

    Raises:
        KeyError: if any of the 55 canonical codes is missing from
            ``distributions``.
    """
    codes = category_codes if category_codes is not None else CATEGORY_CODES
    lb = np.zeros(len(codes), dtype=np.float64)
    ub = np.zeros(len(codes), dtype=np.float64)

    for i, cat in enumerate(codes):
        dist = distributions[cat]

        # Lower bound — p10 from matching, zero-clipped. p10 can in
        # principle be slightly negative due to weighted-percentile
        # interpolation at the edge of a sparse pool; treat any
        # non-positive p10 as a zero floor.
        lb[i] = max(0.0, float(dist.p10))

        # Upper bound — conditional p90 for zero-inflated cats, else
        # unconditional p90; fall back to engel_estimate, then to a
        # symbolic 1.0 when both are zero.
        if (
            dist.nonzero_rate < _ZERO_INFLATED_THRESHOLD
            and dist.conditional_p90 > 0.0
        ):
            p90 = float(dist.conditional_p90)
        else:
            p90 = float(dist.p90)
        engel = float(dist.engel_estimate)
        if p90 > 0.0:
            ub[i] = p90
        elif engel > 0.0:
            ub[i] = engel
        else:
            ub[i] = _MIN_SYMBOLIC_CEILING

        # Safety: guarantee ub >= lb even if p90 < p10 in rare edge
        # cases. When that happens we push ub up to lb so the feasible
        # region isn't empty for this category.
        if ub[i] < lb[i]:
            ub[i] = lb[i]

    # Pin housing to the user-reported number. For RENT we pin
    # rntval; for OWN we pin mortgage interest + principal. These
    # overrides come last so they always win over the cohort p10/p90.
    annual_housing = float(profile.housing_cost) * 12.0
    if profile.tenure is Tenure.RENT:
        rntval_idx = codes.index("rntval")
        lb[rntval_idx] = ub[rntval_idx] = annual_housing
    else:  # Tenure.OWN
        mrtgip_idx = codes.index("mrtgip")
        mrtgpp_idx = codes.index("mrtgpp")
        lb[mrtgip_idx] = ub[mrtgip_idx] = (
            annual_housing * _OWNER_MORTGAGE_INTEREST_SHARE
        )
        lb[mrtgpp_idx] = ub[mrtgpp_idx] = (
            annual_housing * _OWNER_MORTGAGE_PRINCIPAL_SHARE
        )

    return lb, ub
