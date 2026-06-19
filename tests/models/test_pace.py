"""Tests for the pace module (project_pace, pace_display, buffer, compute_d_variable).

Run from repo root:
    python3.11 -m pytest tests/models/test_pace.py -v
"""

from __future__ import annotations

from datetime import date

from models.pace.buffer import BufferState, carry_forward
from models.pace.calculator import compute_d_variable, pace_display, project_pace
from models.tax.state import take_home
from shared.types import HouseholdProfile, Tenure


def test_pace_normal() -> None:
    # Spent 400 in 7 days; 7 remaining at same rate -> spend another 400,
    # total 800; starting 1000 - 800 = 200. Wait — spec says "== 600".
    # Re-reading: "project_pace(1000, 400, 7, 7) == 600.0". That implies
    # pace = available - (spend/elapsed)*remaining = 1000 - (400/7)*7
    #      = 1000 - 400 = 600. Yes, the formula tests ONLY the remaining
    # period's projected spend, not the total.
    assert project_pace(1000.0, 400.0, 7, 7) == 600.0


def test_pace_overspending() -> None:
    # project_pace(1000, 800, 7, 7) == 1000 - (800/7)*7 == 200.0.
    assert project_pace(1000.0, 800.0, 7, 7) == 200.0


def test_pace_zero_elapsed() -> None:
    assert project_pace(1000.0, 0.0, 0, 14) == 1000.0


def test_pace_negative() -> None:
    result = project_pace(500.0, 600.0, 7, 7)
    # 500 - (600/7)*7 = -100
    assert result < 0


def test_pace_display_positive() -> None:
    s = pace_display(340.60)
    assert "carry forward" in s.lower()
    # Rounded to nearest dollar -> 341.
    assert "341" in s


def test_pace_display_negative() -> None:
    s = pace_display(-180.0)
    assert "short" in s.lower()
    assert "180" in s
    assert "-" not in s, f"display must not contain a negative sign: {s!r}"


def test_compute_d_variable() -> None:
    profile = HouseholdProfile(
        age=35,
        gross_income=65_000,
        puma_code="CA_03761",
        tenure=Tenure.RENT,
        housing_cost=1500,
        household_size=2,
    )
    d = compute_d_variable(profile, "single")
    th = take_home(65_000, "CA_03761", "single")
    # Housing is no longer deducted from d_variable — it is pinned as a
    # fixed allocation inside the allocator instead. With no
    # additional_committed passed, d_variable equals take_home.
    assert d == th
    assert d > 0
    assert d < 65_000


def test_buffer_carry_forward() -> None:
    state = BufferState(balance=200.0, last_updated=date.today(), history=[])
    new = carry_forward(1000.0, 750.0, state)
    assert new.balance == 450.0
    assert len(new.history) == 1


def test_buffer_no_negative() -> None:
    state = BufferState(balance=200.0, last_updated=date.today(), history=[])
    new = carry_forward(500.0, 700.0, state)
    assert new.balance == 200.0, (
        f"buffer should not shrink on shortfall; got {new.balance}"
    )
