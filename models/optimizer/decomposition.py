"""Post-solve annotation: push optimizer outputs back into SpendingDistribution.

Reads the per-category allocation from :func:`solve_feasibility` and
fills three fields on each ``SpendingDistribution``:

  * ``feasibility_adjusted`` — the optimizer's allocation for the category.
  * ``behavioral_gap`` — ``allocations[c] - engel_estimate_c``.
      Positive means the cohort spends more than income predicts;
      negative means less. Used downstream to describe how the
      recommendation differs from the pure Engel anchor.
  * ``cohort_position`` — percentile rank of the allocation within the
      cohort's [p10, p90] range, clamped to [0, 1]. When the range has
      zero width (rare cat with p10 == p90), returns 0.5.

``p10..p90``, ``engel_estimate``, and ``is_structural`` are preserved
from the incoming distributions. Returns a new dict of fresh
``SpendingDistribution`` instances — the dataclass is frozen.
"""

from __future__ import annotations

import math

from shared.constants.categories import BALANCE_CATEGORIES
from shared.types import SpendingDistribution


def decompose(
    distributions: dict[str, SpendingDistribution],
    allocations: dict[str, float],
) -> dict[str, SpendingDistribution]:
    """Annotate ``distributions`` with optimizer outputs.

    Args:
        distributions: Output of ``engel.annotate_distributions``
            (p10..p90, engel_estimate, is_structural populated).
        allocations: ``FeasibilityResult.allocations`` — one dollar
            amount per category code.

    Returns:
        New dict with the same keys, values updated with
        ``feasibility_adjusted``, ``behavioral_gap``, and
        ``cohort_position``.

    Balance categories (retire, check, stock, othfin, lifval, vehval,
    ownval, othdbt, stddbt) are wealth stocks/liability balances, not
    flows. Their allocation is pinned to 0 upstream. Here we explicitly
    set ``cohort_position = 0.5`` and ``behavioral_gap = 0.0`` for
    them so the UI doesn't misread a zero allocation as "cohort-bottom
    spender" — these are context-only fields for balance cats.

    Raises:
        KeyError: if any category in ``distributions`` is missing from
            ``allocations`` or vice versa.
    """
    out: dict[str, SpendingDistribution] = {}
    for cat, dist in distributions.items():
        alloc = float(allocations[cat])

        if cat in BALANCE_CATEGORIES:
            gap = 0.0
            position = 0.5
        else:
            gap = alloc - float(dist.engel_estimate)
            spread = float(dist.p90) - float(dist.p10)
            if spread <= 0.0 or not math.isfinite(spread):
                position = 0.5
            else:
                raw = (alloc - float(dist.p10)) / spread
                position = max(0.0, min(1.0, raw))

        out[cat] = SpendingDistribution(
            p10=dist.p10,
            p25=dist.p25,
            p50=dist.p50,
            p75=dist.p75,
            p90=dist.p90,
            engel_estimate=dist.engel_estimate,
            feasibility_adjusted=alloc,
            cohort_position=position,
            is_structural=dist.is_structural,
            behavioral_gap=gap,
            nonzero_rate=dist.nonzero_rate,
            conditional_p90=dist.conditional_p90,
            conditional_p10=dist.conditional_p10,
            conditional_p90_hi=dist.conditional_p90_hi,
            weighted_mean=dist.weighted_mean,
        )
    return out
