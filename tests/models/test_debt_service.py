"""Tests for models.optimizer.debt_service.

Run from repo root:
    python3.11 -m pytest tests/models/test_debt_service.py -v
"""

from __future__ import annotations

import pytest

from models.optimizer.debt_service import estimate_annual_debt_service


def test_zero_balances() -> None:
    service = estimate_annual_debt_service(0.0, 0.0)
    assert service["credit_card_service"] == 0.0
    assert service["student_loan_service"] == 0.0
    assert service["total_debt_service"] == 0.0


def test_negative_balances_treated_as_zero() -> None:
    service = estimate_annual_debt_service(-500.0, -10_000.0)
    assert service["credit_card_service"] == 0.0
    assert service["student_loan_service"] == 0.0
    assert service["total_debt_service"] == 0.0


def test_cc_tiny_balance_hits_minimum_floor() -> None:
    # $100 balance * 2% = $2/mo, but $25 floor applies.
    service = estimate_annual_debt_service(100.0, 0.0)
    assert service["credit_card_service"] == pytest.approx(25.0 * 12)


def test_cc_large_balance_uses_2pct() -> None:
    # $5000 * 2% = $100/mo > $25 floor.
    service = estimate_annual_debt_service(5_000.0, 0.0)
    assert service["credit_card_service"] == pytest.approx(100.0 * 12)


def test_student_loan_typical_balance() -> None:
    # $20k at 6.5% APR over 10yr ≈ $227.10/mo → $2,725/yr.
    service = estimate_annual_debt_service(0.0, 20_000.0)
    assert service["student_loan_service"] == pytest.approx(20_000 * 0.13627, rel=1e-3)
    # Sanity: monthly ~= $227.
    assert 2_600 < service["student_loan_service"] < 2_800


def test_combined_total() -> None:
    service = estimate_annual_debt_service(3_000.0, 15_000.0)
    expected_cc = 3_000.0 * 0.02 * 12  # $60/mo > $25 floor
    expected_sl = 15_000.0 * 0.13627
    assert service["credit_card_service"] == pytest.approx(expected_cc)
    assert service["student_loan_service"] == pytest.approx(expected_sl)
    assert service["total_debt_service"] == pytest.approx(expected_cc + expected_sl)
