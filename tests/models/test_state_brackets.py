"""Tests for the rigorous per-state income-tax layer (models.tax.state).

Loader/bracket/wedge-conformance tested against a tmp fixture artifact so they
are robust to the live schedule's (refreshable) values. Run from repo root:
    .venv/bin/python -m pytest tests/models/test_state_brackets.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

from models.tax.state import STATE_TAX_RATES, _reset_cache_for_tests, state_tax


def _write_fixture(d: Path) -> None:
    (d / "state_tax_rates.json").write_text(
        json.dumps(
            {
                "_meta": {"tax_year": 2026},
                "tax_year": 2026,
                "by_state": {
                    # Progressive: std deduction + per-filer + per-dependent exemption.
                    "ZA": {
                        "name": "Test-Progressive",
                        "kind": "brackets",
                        "brackets_by_filing_status": {
                            "single": [[10000, 0.02], [50000, 0.05], [None, 0.09]],
                            "married_filing_jointly": [[20000, 0.02], [100000, 0.05], [None, 0.09]],
                            "head_of_household": [[15000, 0.02], [75000, 0.05], [None, 0.09]],
                        },
                        "standard_deduction_by_filing_status": {
                            "single": 5000, "married_filing_jointly": 10000, "head_of_household": 7500,
                        },
                        "personal_exemption": 1000,
                        "dependent_exemption": 500,
                    },
                    # Flat with a per-filer exemption, no standard deduction.
                    "ZF": {
                        "name": "Test-Flat",
                        "kind": "flat",
                        "rate": 0.05,
                        "standard_deduction_by_filing_status": {"single": 0},
                        "personal_exemption": 2000,
                    },
                    # No wage income tax.
                    "ZN": {"name": "Test-None", "kind": "none"},
                },
            }
        )
    )


def test_flat_rule_applies_to_taxable(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    # ZF flat 5%, std 0, exemption 2000 → taxable 98,000 → 4,900.
    t = state_tax(100_000, "ZF_00001", filing_status="single", artifacts_path=str(tmp_path))
    assert abs(t - 98_000 * 0.05) < 1e-6


def test_progressive_brackets(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    # ZA single, gross 60k, std 5k + exemption 1k → taxable 54k.
    # 10k@2% + 40k@5% + 4k@9% = 200 + 2000 + 360 = 2560.
    t = state_tax(60_000, "ZA_00001", filing_status="single", artifacts_path=str(tmp_path))
    assert abs(t - (200 + 2000 + 360)) < 1e-6


def test_wedge_reduces_state_taxable(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    # Committed-baseline pre-tax wedge $10k conforms → taxable 44k.
    # 10k@2% + 34k@5% = 200 + 1700 = 1900.
    t = state_tax(60_000, "ZA_00001", filing_status="single",
                  pretax_excludable=10_000, artifacts_path=str(tmp_path))
    assert abs(t - (200 + 1700)) < 1e-6


def test_mfj_doubles_personal_exemption(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    # ZF mfj: std (no mfj entry → falls back to single=0), exemption 2000 × 2
    # filers = 4000 → taxable 96k → 4800.
    t = state_tax(100_000, "ZF_00001", filing_status="married_filing_jointly",
                  artifacts_path=str(tmp_path))
    assert abs(t - 96_000 * 0.05) < 1e-6


def test_dependent_exemption(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    # ZA single, 2 deps: exemption 1000 + 500×2 = 2000; std 5000 → taxable 53k.
    # 10k@2% + 40k@5% + 3k@9% = 200 + 2000 + 270 = 2470.
    t = state_tax(60_000, "ZA_00001", filing_status="single", num_dependents=2,
                  artifacts_path=str(tmp_path))
    assert abs(t - (200 + 2000 + 270)) < 1e-6


def test_none_state_zero(tmp_path) -> None:
    _reset_cache_for_tests()
    _write_fixture(tmp_path)
    assert state_tax(100_000, "ZN_00001", artifacts_path=str(tmp_path)) == 0.0


def test_flat_fallback_without_artifact(tmp_path) -> None:
    # No artifact in the dir → fall back to the flat STATE_TAX_RATES dict applied
    # to gross (byte-identical to the pre-Stage-4 behavior).
    _reset_cache_for_tests()
    t = state_tax(100_000, "CA_00001", artifacts_path=str(tmp_path))
    assert abs(t - 100_000 * STATE_TAX_RATES["CA"]) < 1e-6


def test_zero_income() -> None:
    assert state_tax(0, "CA_00001") == 0.0
