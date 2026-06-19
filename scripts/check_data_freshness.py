"""Report the freshness of every external data source in the registry.

Reads the manifest at ``agent-artifacts/registry/data_sources.json`` (the
single source of truth), then for each source locates its on-disk artifact,
reads the metadata field that carries its vintage, and classifies it against
the declared release cadence. Computes the next expected government release
relative to the run date.

    python3.11 scripts/check_data_freshness.py            # status table
    python3.11 scripts/check_data_freshness.py --markdown # table for DATA_SOURCES.md
    python3.11 scripts/check_data_freshness.py --json      # machine-readable
    python3.11 scripts/check_data_freshness.py --strict    # non-zero exit if any OVERDUE

Pure stdlib, read-only — never fetches. The companion refresh tooling lives in
``scripts/refresh_cpi.py`` (CPI + EIA gas + EIA SEDS), ``scripts/build_climate_normals.py``
(NOAA), and the R re-exports in ``pipeline/export/``. See
``agent-artifacts/registry/DATA_SOURCES.md`` for the human-readable registry.

Exit code 0 by default; with --strict, non-zero when any source is OVERDUE.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO_ROOT / "agent-artifacts" / "registry" / "data_sources.json"

# Status order for sorting (worst first) and exit-code logic.
_STATUS_RANK = {"MISSING": 0, "OVERDUE": 1, "DUE_SOON": 2, "MANUAL": 3, "OK": 4}

# ANSI colors (suppressed when stdout is not a TTY).
_COLORS = {
    "MISSING": "\033[91m",   # red
    "OVERDUE": "\033[91m",   # red
    "DUE_SOON": "\033[93m",  # yellow
    "MANUAL": "\033[90m",    # grey
    "OK": "\033[92m",        # green
}
_RESET = "\033[0m"

# Approximate days per cadence period, used to flag staleness (OVERDUE) for
# timestamp sources.
_CADENCE_DAYS = {
    "weekly": 7,
    "monthly": 31,
    "annual": 366,
    "per_decade": 3653,
    "adhoc": None,
    "manual": None,
}

# How close to the next scheduled government release a source must be before it
# is flagged DUE_SOON (a refresh is about to be warranted). Forward-looking and
# release-driven — NOT a fraction of elapsed age. annual: <1 month, monthly:
# <1 week, weekly: <1 day (the comparison is strict, in _evaluate).
_DUE_SOON_DAYS = {
    "weekly": 1,
    "monthly": 7,
    "annual": 30,
    "per_decade": 30,
}


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp (tolerating a trailing Z)."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _next_release(pattern: dict, now: datetime) -> datetime | None:
    """Next expected government release date after ``now`` from a release pattern."""
    kind = pattern.get("type")
    if kind == "weekly":
        # release_weekday: 0=Mon .. 6=Sun
        target = pattern.get("release_weekday", 1)
        days = (target - now.weekday()) % 7
        days = days or 7  # always strictly in the future
        return datetime.fromordinal(now.toordinal() + days).replace(tzinfo=timezone.utc)
    if kind == "monthly":
        day = pattern.get("release_day", 12)
        year, month = now.year, now.month
        if now.day >= day:
            month += 1
            if month > 12:
                month, year = 1, year + 1
        return datetime(year, month, day, tzinfo=timezone.utc)
    if kind == "annual":
        month = pattern.get("release_month", 1)
        year = now.year if (now.month, now.day) < (month, 28) else now.year + 1
        return datetime(year, month, 1, tzinfo=timezone.utc)
    if kind == "per_decade":
        year = pattern.get("next_period_release_year")
        return datetime(year, 1, 1, tzinfo=timezone.utc) if year else None
    return None  # adhoc / manual — no scheduled date


def _load_artifact_meta(path: Path) -> dict | None:
    """Return the metadata block of a JSON artifact, or {} for a dir, or None if absent."""
    if not path.exists():
        return None
    if path.is_dir():
        return {}  # directory artifact — present but carries no JSON metadata
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}  # binary (e.g. .fst/.fsn) or unreadable — present but no JSON metadata
    if not isinstance(data, dict):
        return {}
    # Metadata may sit at the top level or under _meta / metadata.
    meta = dict(data)
    for nest in ("_meta", "metadata"):
        if isinstance(data.get(nest), dict):
            meta = {**data[nest], **meta}
    return meta


def _evaluate(source_id: str, src: dict, now: datetime) -> dict:
    """Classify one source: locate artifact, read vintage, compute status + next release."""
    artifacts = [_REPO_ROOT / a for a in src.get("artifacts", [])]
    primary = artifacts[0] if artifacts else None
    meta = _load_artifact_meta(primary) if primary else None

    cadence = src.get("cadence", "manual")
    kind = src.get("freshness", {}).get("kind", "none")
    field = src.get("freshness", {}).get("field")
    pattern = src.get("release_pattern", {})
    next_rel = _next_release(pattern, now)

    vintage = "—"
    age_days: int | None = None
    status = "MANUAL"

    if meta is None:
        status, vintage = "MISSING", "absent"
    elif kind == "timestamp":
        ts = _parse_iso(meta.get(field, ""))
        if ts is None:
            # File present but no readable timestamp — fall back to mtime.
            ts = datetime.fromtimestamp(primary.stat().st_mtime, tz=timezone.utc)
            vintage = f"mtime {ts:%Y-%m-%d}"
        else:
            vintage = f"{ts:%Y-%m-%d}"
        age_days = (now - ts).days
        status = _classify_timestamp(age_days, cadence)
    elif kind == "period_string":
        raw = meta.get(field)
        vintage = str(raw)
        status = _classify_period(raw, cadence, now, src.get("expected_lag_years", 1))
    else:  # kind == "none"
        seen = src.get("vintage_field", {}).get("value_seen", "?")
        vintage = f"{seen} (unstamped)"
        status = "MANUAL"

    # Forward-looking DUE_SOON: a scheduled release is imminent, so a refresh is
    # about to be warranted. Only an otherwise-OK source is upgraded — a stale
    # (OVERDUE), absent (MISSING), or unstamped/adhoc (MANUAL) source keeps its
    # status. Window is cadence-specific and the comparison is strict.
    if status == "OK" and next_rel is not None:
        window = _DUE_SOON_DAYS.get(cadence)
        if window is not None and 0 <= (next_rel - now).days < window:
            status = "DUE_SOON"

    return {
        "id": source_id,
        "name": src.get("name", source_id),
        "artifact": src["artifacts"][0] if src.get("artifacts") else "—",
        "cadence": cadence,
        "vintage": vintage,
        "age_days": age_days,
        "next_release": f"{next_rel:%Y-%m-%d}" if next_rel else "—",
        "refresh_command": src.get("refresh_command", "—"),
        "status": status,
    }


def _classify_timestamp(age_days: int, cadence: str) -> str:
    """OVERDUE when the artifact is older than one full cadence period, else OK.

    DUE_SOON is assigned separately in ``_evaluate`` from proximity to the next
    scheduled release — not from elapsed age.
    """
    period = _CADENCE_DAYS.get(cadence)
    if period is None:
        return "MANUAL"
    return "OVERDUE" if age_days > period else "OK"


def _classify_period(raw, cadence: str, now: datetime, lag_years: int = 1) -> str:
    """Classify a period_string vintage (year int, 'YYYY', or 'YYYY-MM-DD').

    ``lag_years`` is the source's expected publication lag (manifest
    ``expected_lag_years``); e.g. EIA SEDS publishes ~2 years behind, so its
    latest-available vintage is now.year - 2, not now.year - 1.
    """
    if cadence in ("adhoc", "manual"):
        return "MANUAL"
    # Date-valued weekly periods (EIA gas survey week).
    if isinstance(raw, str) and len(raw) == 10 and raw[4] == "-":
        dt = _parse_iso(raw)
        if dt:
            return _classify_timestamp((now - dt).days, cadence)
    # Year-valued periods (RPP year, SEDS source_period, source_year, ratio_basis_year).
    try:
        year = int(str(raw)[:4])
    except (TypeError, ValueError):
        return "MANUAL"
    # Expected latest available vintage given typical government release lag.
    if cadence in ("monthly", "weekly"):
        lag_years = 0
    expected = now.year - lag_years
    # OVERDUE only when 2+ years behind the expected-latest vintage; one year
    # behind is treated as OK (DUE_SOON is decided in _evaluate from the next
    # scheduled release date, not from the vintage year).
    return "OVERDUE" if year < expected - 1 else "OK"


def _print_table(rows: list[dict], use_color: bool) -> None:
    header = f"  Data-source freshness  ({datetime.now(tz=timezone.utc):%Y-%m-%d})"
    print(header)
    print("  " + "-" * (len(header) - 2))
    fmt = "  {st:<9} {id:<14} {vint:<22} {cad:<10} {nxt:<12} {name}"
    print(fmt.format(st="STATUS", id="ID", vint="VINTAGE", cad="CADENCE",
                     nxt="NEXT REL.", name="SOURCE"))
    for r in rows:
        st = r["status"]
        st_disp = f"{_COLORS[st]}{st:<9}{_RESET}" if use_color else f"{st:<9}"
        line = fmt.format(st=st_disp, id=r["id"], vint=r["vintage"][:22],
                          cad=r["cadence"], nxt=r["next_release"], name=r["name"])
        print(line)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items(),
                                                       key=lambda x: _STATUS_RANK[x[0]]))
    print("\n  " + summary)


def _print_markdown(rows: list[dict]) -> None:
    print(f"<!-- generated by scripts/check_data_freshness.py on "
          f"{datetime.now(tz=timezone.utc):%Y-%m-%d} -->")
    print()
    print("| Status | Source | Artifact | Vintage | Cadence | Next release |")
    print("|--------|--------|----------|---------|---------|--------------|")
    for r in rows:
        print(f"| {r['status']} | {r['name']} | `{r['artifact']}` | {r['vintage']} "
              f"| {r['cadence']} | {r['next_release']} |")


def main() -> int:
    ap = argparse.ArgumentParser(description="Report external data-source freshness.")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the Current-status table for DATA_SOURCES.md")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any source is OVERDUE")
    args = ap.parse_args()

    try:
        manifest = json.loads(_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read manifest {_MANIFEST}: {exc}", file=sys.stderr)
        return 2

    now = datetime.now(tz=timezone.utc)
    rows = [_evaluate(sid, src, now) for sid, src in manifest.items()
            if not sid.startswith("_")]
    rows.sort(key=lambda r: (_STATUS_RANK[r["status"]], r["id"]))

    if args.json:
        print(json.dumps(rows, indent=2))
    elif args.markdown:
        _print_markdown(rows)
    else:
        _print_table(rows, use_color=sys.stdout.isatty())

    if args.strict and any(r["status"] == "OVERDUE" for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
