"""Local/municipal tax lookup tests (models/tax/local.py).

Covers:
  - NYC bracket byte-identical regression vs the frozen oracle from calculator.py
  - PUMA-code NYC back-compat fallback
  - Flat-rate place lookup
  - County FIPS lookup
  - place_to_county bridge fallback
  - Precedence (place beats county when both supplied)
  - Missing / malformed artifact → clean 0.0 no-op
  - Zero income, unknown FIPS edge cases
  - Committed-artifact sanity class (auto-skipped if artifact absent)

Run from repo root:
    .venv/bin/python -m pytest tests/models/test_local_tax.py -v
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from models.tax.local import (
    _NYC_PLACE_FIPS,
    _reset_cache_for_tests,
    compute_local_tax,
    load_local_tax_table,
)

# --------------------------------------------------------------------------- #
# Frozen NYC oracle (verbatim copy from calculator.py — do not edit)           #
# --------------------------------------------------------------------------- #

_NYC_BRACKETS_ORACLE: dict[str, list[tuple[float, float]]] = {
    "single": [
        (12_000.0, 0.03078),
        (25_000.0, 0.03762),
        (50_000.0, 0.03819),
        (float("inf"), 0.03876),
    ],
    "married_filing_jointly": [
        (21_600.0, 0.03078),
        (45_000.0, 0.03762),
        (90_000.0, 0.03819),
        (float("inf"), 0.03876),
    ],
    "head_of_household": [
        (14_400.0, 0.03078),
        (30_000.0, 0.03762),
        (60_000.0, 0.03819),
        (float("inf"), 0.03876),
    ],
}


def _oracle_nyc_tax(gross_income: float, filing_status: str) -> float:
    """Frozen verbatim copy of _compute_nyc_city_tax from calculator.py."""
    if gross_income <= 0:
        return 0.0
    brackets = _NYC_BRACKETS_ORACLE.get(filing_status, _NYC_BRACKETS_ORACLE["single"])
    tax = 0.0
    prev = 0.0
    for edge, rate in brackets:
        if gross_income <= edge:
            tax += (gross_income - prev) * rate
            break
        tax += (edge - prev) * rate
        prev = edge
    return tax


# --------------------------------------------------------------------------- #
# Synthetic artifact fixture                                                    #
# --------------------------------------------------------------------------- #

# NYC brackets expressed as JSON-serialisable lists (upper_bound None = inf).
_NYC_BRACKETS_JSON = {
    "single": [
        [12000.0, 0.03078],
        [25000.0, 0.03762],
        [50000.0, 0.03819],
        [None, 0.03876],
    ],
    "married_filing_jointly": [
        [21600.0, 0.03078],
        [45000.0, 0.03762],
        [90000.0, 0.03819],
        [None, 0.03876],
    ],
    "head_of_household": [
        [14400.0, 0.03078],
        [30000.0, 0.03762],
        [60000.0, 0.03819],
        [None, 0.03876],
    ],
}

_SYNTHETIC_ARTIFACT: dict = {
    "tax_year": 2026,
    "by_place_fips": {
        # NYC — bracket rule
        "3651000": {
            "name": "New York City",
            "kind": "brackets",
            "brackets_by_filing_status": _NYC_BRACKETS_JSON,
            "base": "wages",
        },
        # Columbus, OH — flat rule
        "3918000": {
            "name": "Columbus",
            "kind": "flat",
            "rate": 0.025,
            "base": "wages",
        },
    },
    "by_county_fips": {
        # Montgomery County, MD
        "24031": {
            "name": "Montgomery County",
            "kind": "flat",
            "rate": 0.032,
            "base": "wages",
        },
    },
    # Gaithersburg city (place 2407125) bridges to Montgomery County
    "place_to_county": {
        "2407125": "24031",
    },
}


@pytest.fixture()
def synthetic_artifact_path(tmp_path: Path) -> Iterator[str]:
    """Write the synthetic local_tax_rates.json and return its directory."""
    artifact_file = tmp_path / "local_tax_rates.json"
    artifact_file.write_text(json.dumps(_SYNTHETIC_ARTIFACT))
    _reset_cache_for_tests()
    yield str(tmp_path)
    _reset_cache_for_tests()


# --------------------------------------------------------------------------- #
# 1. NYC byte-identical regression                                              #
# --------------------------------------------------------------------------- #

_INCOMES = [0, 5_000, 12_000, 24_999.99, 25_000, 50_000, 80_000, 250_000]
_FILING_STATUSES_NYC = ["single", "married_filing_jointly", "head_of_household"]


@pytest.mark.parametrize("income", _INCOMES)
@pytest.mark.parametrize("fs", _FILING_STATUSES_NYC)
def test_nyc_byte_identical_regression(
    income: float, fs: str, synthetic_artifact_path: str
) -> None:
    """compute_local_tax must equal the frozen oracle EXACTLY (same floats)."""
    result = compute_local_tax(
        income, fs, place_fips="3651000", artifacts_path=synthetic_artifact_path
    )
    expected = _oracle_nyc_tax(income, fs)
    assert result == expected, (
        f"income={income}, filing_status={fs}: got {result!r}, expected {expected!r}"
    )


def test_nyc_qualifying_widow_fallback(synthetic_artifact_path: str) -> None:
    """Unknown filing_status 'qualifying_widow' must fall back to the single schedule."""
    income = 60_000.0
    result = compute_local_tax(
        income, "qualifying_widow", place_fips="3651000", artifacts_path=synthetic_artifact_path
    )
    expected = _oracle_nyc_tax(income, "qualifying_widow")  # oracle also falls back to single
    assert result == expected


# --------------------------------------------------------------------------- #
# 2. PUMA-code NYC back-compat fallback                                        #
# --------------------------------------------------------------------------- #


def test_puma_nyc_fallback_equals_explicit_fips(synthetic_artifact_path: str) -> None:
    """puma_code NY_04103 without any FIPS must equal the explicit-FIPS result."""
    income = 75_000.0
    via_puma = compute_local_tax(
        income, "single", puma_code="NY_04103", artifacts_path=synthetic_artifact_path
    )
    via_fips = compute_local_tax(
        income, "single", place_fips="3651000", artifacts_path=synthetic_artifact_path
    )
    assert via_puma == via_fips


def test_puma_non_nyc_returns_zero(synthetic_artifact_path: str) -> None:
    """Non-NYC NY PUMA (NY_03xxx) with no FIPS → 0.0."""
    result = compute_local_tax(
        75_000.0, "single", puma_code="NY_03112", artifacts_path=synthetic_artifact_path
    )
    assert result == 0.0


def test_puma_non_ny_returns_zero(synthetic_artifact_path: str) -> None:
    """Non-NY PUMA with no FIPS → 0.0."""
    result = compute_local_tax(
        75_000.0, "single", puma_code="CA_00101", artifacts_path=synthetic_artifact_path
    )
    assert result == 0.0


# --------------------------------------------------------------------------- #
# 3. Flat place lookup                                                          #
# --------------------------------------------------------------------------- #


def test_flat_place_lookup(synthetic_artifact_path: str) -> None:
    """Columbus (3918000) flat rate 0.025."""
    income = 50_000.0
    result = compute_local_tax(
        income, "single", place_fips="3918000", artifacts_path=synthetic_artifact_path
    )
    assert result == income * 0.025


# --------------------------------------------------------------------------- #
# 4. County FIPS lookup                                                         #
# --------------------------------------------------------------------------- #


def test_county_lookup(synthetic_artifact_path: str) -> None:
    """Montgomery County (24031) flat rate 0.032."""
    income = 80_000.0
    result = compute_local_tax(
        income, "single", county_fips="24031", artifacts_path=synthetic_artifact_path
    )
    assert result == income * 0.032


# --------------------------------------------------------------------------- #
# 5. place_to_county bridge fallback                                            #
# --------------------------------------------------------------------------- #


def test_place_to_county_fallback(synthetic_artifact_path: str) -> None:
    """Gaithersburg place FIPS (2407125) has no direct entry; bridges to Montgomery County."""
    income = 60_000.0
    result = compute_local_tax(
        income, "single", place_fips="2407125", artifacts_path=synthetic_artifact_path
    )
    assert result == income * 0.032


# --------------------------------------------------------------------------- #
# 6. Precedence: place beats county                                             #
# --------------------------------------------------------------------------- #


def test_precedence_place_over_county(synthetic_artifact_path: str) -> None:
    """Supplying both place_fips (Columbus 0.025) and county_fips (Montgomery 0.032)
    must yield the place rate — no stacking."""
    income = 100_000.0
    result = compute_local_tax(
        income,
        "single",
        place_fips="3918000",
        county_fips="24031",
        artifacts_path=synthetic_artifact_path,
    )
    assert result == income * 0.025


# --------------------------------------------------------------------------- #
# 7. Missing artifact → clean no-op                                             #
# --------------------------------------------------------------------------- #


def test_missing_artifact_no_op(tmp_path: Path) -> None:
    """Empty tmp dir: load returns None, compute returns 0.0 even for NYC PUMA."""
    _reset_cache_for_tests()
    table = load_local_tax_table(str(tmp_path))
    assert table is None

    result = compute_local_tax(
        80_000.0, "single", puma_code="NY_04103", artifacts_path=str(tmp_path)
    )
    assert result == 0.0
    _reset_cache_for_tests()


# --------------------------------------------------------------------------- #
# 8. Malformed artifact → clean no-op                                           #
# --------------------------------------------------------------------------- #


def test_malformed_artifact_no_op(tmp_path: Path) -> None:
    """Invalid JSON → None → 0.0."""
    _reset_cache_for_tests()
    (tmp_path / "local_tax_rates.json").write_text("not valid json {{")
    table = load_local_tax_table(str(tmp_path))
    assert table is None
    result = compute_local_tax(
        50_000.0, "single", place_fips="3651000", artifacts_path=str(tmp_path)
    )
    assert result == 0.0
    _reset_cache_for_tests()


# --------------------------------------------------------------------------- #
# 9. Zero income, unknown FIPS                                                  #
# --------------------------------------------------------------------------- #


def test_zero_income_zero(synthetic_artifact_path: str) -> None:
    """gross_income <= 0 → 0.0 regardless of FIPS."""
    assert compute_local_tax(0.0, "single", place_fips="3651000", artifacts_path=synthetic_artifact_path) == 0.0
    assert compute_local_tax(-100.0, "single", place_fips="3651000", artifacts_path=synthetic_artifact_path) == 0.0


def test_unknown_fips_zero(synthetic_artifact_path: str) -> None:
    """Unrecognised FIPS codes return 0.0."""
    result = compute_local_tax(
        50_000.0, "single", place_fips="9999999", county_fips="99999",
        artifacts_path=synthetic_artifact_path
    )
    assert result == 0.0


# --------------------------------------------------------------------------- #
# 10. Committed-artifact sanity (auto-skipped if artifact absent)              #
# --------------------------------------------------------------------------- #

_ARTIFACT_PATH = Path("pipeline/artifacts/local_tax_rates.json")

import re


@pytest.mark.skipif(
    not _ARTIFACT_PATH.exists(),
    reason="artifact pipeline/artifacts/local_tax_rates.json not yet built",
)
class TestCommittedArtifactSanity:
    """Structural sanity checks against the real produced artifact."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        _reset_cache_for_tests()
        self.table = load_local_tax_table("pipeline/artifacts")
        assert self.table is not None, "artifact exists but failed to parse"

    def test_all_rates_in_range(self) -> None:
        """All effective rates must be in (0, 0.05]."""
        for fips, rule in {**self.table.by_place_fips, **self.table.by_county_fips}.items():
            if rule.kind == "flat":
                assert 0.0 < rule.rate <= 0.05, f"{fips}: flat rate {rule.rate} out of range"
            elif rule.kind == "brackets" and rule.brackets_by_filing_status:
                for fs, schedule in rule.brackets_by_filing_status.items():
                    for _edge, rate in schedule:
                        assert 0.0 < rate <= 0.05, f"{fips}/{fs}: bracket rate {rate} out of range"

    def test_place_fips_format(self) -> None:
        """Place FIPS must match 7-digit pattern."""
        pattern = re.compile(r"^\d{7}$")
        for fips in self.table.by_place_fips:
            assert pattern.match(fips), f"place FIPS {fips!r} not 7 digits"

    def test_county_fips_format(self) -> None:
        """County FIPS must match 5-digit pattern."""
        pattern = re.compile(r"^\d{5}$")
        for fips in self.table.by_county_fips:
            assert pattern.match(fips), f"county FIPS {fips!r} not 5 digits"

    def test_brackets_monotone_with_final_inf(self) -> None:
        """Bracket upper bounds must be strictly increasing; final must be inf."""
        for fips, rule in self.table.by_place_fips.items():
            if rule.kind != "brackets" or not rule.brackets_by_filing_status:
                continue
            for fs, schedule in rule.brackets_by_filing_status.items():
                edges = [e for e, _ in schedule]
                for i in range(len(edges) - 1):
                    assert edges[i] < edges[i + 1], (
                        f"{fips}/{fs}: brackets not monotone at index {i}"
                    )
                assert edges[-1] == float("inf"), (
                    f"{fips}/{fs}: final bracket upper bound is not inf"
                )

    def test_oh_place_count(self) -> None:
        """OH (39-prefix) place FIPS count must be >= 400."""
        oh_places = [f for f in self.table.by_place_fips if f.startswith("39")]
        assert len(oh_places) >= 400, f"OH place count {len(oh_places)} < 400"

    def test_md_county_count(self) -> None:
        """MD (24-prefix) county FIPS count must be exactly 24."""
        md_counties = [f for f in self.table.by_county_fips if f.startswith("24")]
        assert len(md_counties) == 24, f"MD county count {len(md_counties)} != 24"

    def test_in_county_count(self) -> None:
        """IN (18-prefix) county FIPS count must be exactly 92."""
        in_counties = [f for f in self.table.by_county_fips if f.startswith("18")]
        assert len(in_counties) == 92, f"IN county count {len(in_counties)} != 92"

    def test_nyc_present_with_correct_brackets(self) -> None:
        """NYC place FIPS 3651000 must be present with brackets equal to the oracle table."""
        assert "3651000" in self.table.by_place_fips, "NYC (3651000) missing from by_place_fips"
        rule = self.table.by_place_fips["3651000"]
        assert rule.kind == "brackets"
        assert rule.brackets_by_filing_status is not None
        for fs, oracle_brackets in _NYC_BRACKETS_ORACLE.items():
            assert fs in rule.brackets_by_filing_status, f"missing filing_status {fs!r} in NYC rule"
            actual = rule.brackets_by_filing_status[fs]
            assert len(actual) == len(oracle_brackets), f"NYC {fs}: bracket count mismatch"
            for i, ((actual_edge, actual_rate), (oracle_edge, oracle_rate)) in enumerate(
                zip(actual, oracle_brackets)
            ):
                assert actual_edge == oracle_edge, (
                    f"NYC {fs} bracket[{i}] edge: {actual_edge} != {oracle_edge}"
                )
                assert actual_rate == oracle_rate, (
                    f"NYC {fs} bracket[{i}] rate: {actual_rate} != {oracle_rate}"
                )

    def test_yonkers_present(self) -> None:
        """Yonkers place FIPS 3684000 must be present."""
        assert "3684000" in self.table.by_place_fips, "Yonkers (3684000) missing"
