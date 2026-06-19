"""Tests for the state employee-payroll/disability tax layer.

Loader/compute tested against a tmp fixture artifact so they're robust to
the live artifact's (refreshable) rates. Run from repo root:
    .venv/bin/python -m pytest tests/models/test_state_payroll_tax.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

from models.tax.calculator import compute_tax
from shared.constants.state_payroll_tax import (
    StatePayrollLine,
    _reset_cache_for_tests,
    load_state_payroll_tax,
)


def _write_fixture(d: Path) -> None:
    (d / "state_payroll_tax.json").write_text(
        json.dumps(
            {
                "_meta": {"tax_year": 2026},
                "by_state": {
                    # CA CASDI: uncapped since SB-951 (the headline high-earner line).
                    "CA": {"lines": [{"name": "casdi", "rate": 0.012, "wage_cap": None, "annual_max": None}]},
                    # NY: SDI with a tiny annual-$ cap + PFL capped at a wage base.
                    "NY": {"lines": [
                        {"name": "sdi", "rate": 0.005, "wage_cap": None, "annual_max": 31.20},
                        {"name": "pfl", "rate": 0.00388, "wage_cap": 91_373.88, "annual_max": None},
                    ]},
                },
            }
        )
    )


def test_line_uncapped() -> None:
    line = StatePayrollLine(name="casdi", rate=0.012, wage_cap=None, annual_max=None)
    assert line.amount(250_000) == 250_000 * 0.012  # uncapped: applies to full wage


def test_line_wage_cap() -> None:
    line = StatePayrollLine(name="pfl", rate=0.01, wage_cap=100_000, annual_max=None)
    assert line.amount(250_000) == 100_000 * 0.01  # capped at the wage base


def test_line_annual_max() -> None:
    line = StatePayrollLine(name="sdi", rate=0.005, wage_cap=None, annual_max=31.20)
    assert line.amount(250_000) == 31.20  # rate*wage exceeds the annual $ cap


def test_table_ca_uncapped(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    table = load_state_payroll_tax(str(tmp_path))
    assert table is not None
    # CASDI uncapped: a $250k earner pays 1.2% of the FULL wage (the miss the
    # flat model had — $0 — that this layer closes).
    assert table.amount_for(250_000, "CA") == 250_000 * 0.012


def test_table_ny_multi_line(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    table = load_state_payroll_tax(str(tmp_path))
    assert table is not None
    expected = 31.20 + 0.00388 * 91_373.88  # sdi annual-cap + pfl wage-capped
    assert abs(table.amount_for(250_000, "NY") - expected) < 1e-6


def test_table_absent_state_zero(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    table = load_state_payroll_tax(str(tmp_path))
    assert table is not None
    assert table.amount_for(100_000, "TX") == 0.0  # no rule → 0 (no-income-tax state)


def test_missing_artifact_returns_none(tmp_path) -> None:
    _reset_cache_for_tests()
    assert load_state_payroll_tax(str(tmp_path)) is None  # empty dir → clean no-op


def test_live_artifact_ca_casdi_uncapped() -> None:
    # Regression against the live 2026 artifact (update on annual rate refresh):
    # CA CASDI is UNCAPPED at 1.3% — a $250k earner pays the full 1.3%, the
    # high-earner line the flat model missed entirely ($0).
    _reset_cache_for_tests()
    table = load_state_payroll_tax("pipeline/artifacts")
    assert table is not None, "build_state_payroll_tax.py must have generated the artifact"
    assert table.tax_year == 2026
    assert table.amount_for(250_000, "CA") == 250_000 * 0.013
    assert table.amount_for(40_000, "CA") == 40_000 * 0.013  # uncapped: scales at every income
    assert table.amount_for(250_000, "TX") == 0.0  # no-payroll state


def test_compute_tax_includes_payroll_in_total() -> None:
    # Structural invariant (value depends on the live artifact): the field is
    # non-negative and summed into total_tax; take_home reconciles.
    br = compute_tax(150_000, "single", puma_code="CA_03761")
    assert br.state_payroll_tax >= 0.0
    expected_total = (
        br.federal_tax + br.state_tax + br.city_tax + br.fica + br.state_payroll_tax
    )
    assert abs(br.total_tax - expected_total) < 1e-6
    assert abs(br.take_home - (150_000 - br.total_tax)) < 1e-6
