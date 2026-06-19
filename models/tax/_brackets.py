"""Shared progressive-bracket arithmetic for the tax layer.

Lifted verbatim from ``models.tax.local`` so both the local/municipal
layer and the state income-tax layer (``models.tax.state.StateTaxRule``)
compose the SAME bracket helpers rather than each carrying a copy — the
``MUNICIPAL-TAX-PRECISION`` "reuse the machinery" steer made literal.

Pure stdlib; no artifact, no I/O. ``local.py`` re-exports both names for
back-compat with existing call sites and tests.
"""

from __future__ import annotations


def parse_brackets(
    raw: dict[str, list[list[float | None]]],
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Convert JSON bracket lists to nested tuples; JSON ``null`` → ``float("inf")``.

    Each schedule is a list of ``[upper_bound, rate]`` pairs; a ``null``
    upper bound (the top, open-ended bracket) becomes ``float("inf")``.
    """
    result: dict[str, tuple[tuple[float, float], ...]] = {}
    for fs, pairs in raw.items():
        result[fs] = tuple(
            (
                float("inf") if pair[0] is None else float(pair[0]),
                float(pair[1] if pair[1] is not None else 0.0),
            )
            for pair in pairs
        )
    return result


def apply_brackets(
    taxable: float,
    brackets: dict[str, tuple[tuple[float, float], ...]],
    filing_status: str,
) -> float:
    """Progressive-bracket arithmetic.

    Uses ``.get(filing_status, brackets["single"])`` so any unrecognised
    status falls back to the single schedule — matching the original
    ``local.py`` / NYC behaviour exactly.

    ``taxable`` is whatever income concept the caller applies the schedule
    to (gross wages for the local layer; income after the standard
    deduction + exemption for the state layer).
    """
    schedule = brackets.get(filing_status, brackets["single"])
    tax = 0.0
    prev = 0.0
    for edge, rate in schedule:
        if taxable <= edge:
            tax += (taxable - prev) * rate
            break
        tax += (edge - prev) * rate
        prev = edge
    return tax
