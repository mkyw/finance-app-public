"""Omit-by-default category set — pure constant tests (no Django).

The behavioral (zeroing / four-way / response-flag) tests live in
tests/api/test_omit_by_default.py (they need the services layer). These pin
the OMIT_BY_DEFAULT_CATEGORIES contract itself.

Run from repo root:
    .venv/bin/python -m pytest tests/models/test_omit_by_default.py -v
"""

from __future__ import annotations

from shared.constants.categories import (
    CATEGORY_CODES,
    OMIT_BY_DEFAULT_CATEGORIES,
)


def test_exactly_the_six_categories() -> None:
    assert OMIT_BY_DEFAULT_CATEGORIES == frozenset(
        {"chrty", "educ", "ocash", "stdint", "othint", "finpay"}
    )


def test_all_are_real_fusion_targets() -> None:
    # Must be a subset of the 55 fusion-target codes — never invents a code.
    assert OMIT_BY_DEFAULT_CATEGORIES <= frozenset(CATEGORY_CODES)


def test_scope_is_narrow() -> None:
    # The investigation found heavy-zero is pervasive (~22/46) but mostly
    # already-conditioned; the genuinely-meaningless set is ~3+2. Guard against
    # scope creep (a future edit that dumps many cats here without re-deriving).
    assert len(OMIT_BY_DEFAULT_CATEGORIES) == 6
