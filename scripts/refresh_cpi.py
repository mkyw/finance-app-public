"""Refresh pipeline/artifacts/cpi_scalars.json (BLS) and
pipeline/artifacts/eia_gas_scalars.json (EIA).

Active counterpart to the passive 30-day refresh in
``models.matching.cpi_scaler.load_cpi_scalars`` and
``models.matching.eia_gas.load_eia_gas_scalars``. Intended to run
monthly after the BLS CPI release (~10th-12th of each month):

    python3.11 scripts/refresh_cpi.py

Reports BLS category scalars and EIA state gasoline scalars.

Environment:
    EIA_API_KEY — required for EIA refresh; register free at
        https://www.eia.gov/opendata/register.php. If unset, the
        EIA step is skipped and a warning printed.

Exit code 0 on success (including graceful fallback); non-zero only on
unexpected exceptions.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow ``python3.11 scripts/refresh_cpi.py`` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load repo-root .env so EIA_API_KEY is available to the EIA fetch.
# ``override=False`` — a shell-exported key wins over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    # python-dotenv not installed — rely on shell-exported env only.
    pass

from models.matching.cpi_scaler import DEFAULT_KEY, fetch_cpi_scalars  # noqa: E402
from models.matching.eia_gas import fetch_eia_gas_scalars  # noqa: E402
from models.matching.eia_utility import fetch_eia_utility_scalars  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    os.chdir(_REPO_ROOT)  # writers use repo-relative paths by default
    cpi_path = str(_REPO_ROOT / "pipeline" / "artifacts" / "cpi_scalars.json")
    eia_path = str(_REPO_ROOT / "pipeline" / "artifacts" / "eia_gas_scalars.json")

    print("Fetching current BLS CPI values (base year 2024)...")
    scalars = fetch_cpi_scalars(base_year=2024, cache_path=cpi_path)

    cat_count = len(scalars) - (1 if DEFAULT_KEY in scalars else 0)
    print(f"Updated {cat_count} category scalars (+ _default)")
    print(f"  _default (CUUR0000SA0 all-items): {scalars[DEFAULT_KEY]:.4f}")
    print(f"  cache path: {cpi_path}")

    notable = {
        k: v for k, v in scalars.items() if k != DEFAULT_KEY and abs(v - 1.0) > 0.05
    }
    if notable:
        print()
        print(f"Categories with >5% change from 2024 ({len(notable)}):")
        for cat, scalar in sorted(notable.items(), key=lambda x: -abs(x[1] - 1.0)):
            direction = "up" if scalar > 1.0 else "dn"
            print(f"  {cat:<18} {scalar:.4f}  {direction}")
    else:
        print("No categories moved >5% from 2024 base.")

    print()
    print("Fetching current EIA state gasoline prices...")
    eia_key = os.environ.get("EIA_API_KEY", "")
    if not eia_key:
        print("  EIA_API_KEY not set; skipping EIA gasoline refresh.")
    else:
        eia_scalars = fetch_eia_gas_scalars(eia_key, cache_path=eia_path)
        if not eia_scalars:
            print("  EIA gasoline fetch returned no data; cache preserved.")
        else:
            print(f"Updated {len(eia_scalars)} state gasoline scalars")
            print(f"  cache path: {eia_path}")
            print()
            print("State gasoline scalars vs national:")
            for state, scalar in sorted(eia_scalars.items(), key=lambda x: -x[1]):
                direction = "up" if scalar > 1.0 else "dn" if scalar < 1.0 else "  "
                print(f"  {state:<4} {scalar:.4f}  {direction}")

    # EIA SEDS residential consumption -> utility climate baseline scalars.
    # Key-free (public bulk CSV); the only network caller for the utility lane.
    print()
    print("Fetching EIA SEDS residential consumption (utility climate baseline)...")
    util_path = str(_REPO_ROOT / "pipeline" / "artifacts" / "eia_utility_scalars.json")
    synth_path = str(_REPO_ROOT / "pipeline" / "artifacts" / "synthetic_population")
    util = fetch_eia_utility_scalars(synth_path, cache_path=util_path)
    if not util:
        print("  EIA SEDS fetch returned no data; cache preserved.")
    else:
        print(
            f"  Updated utility scalars for {len(util)} categories: {', '.join(util)}"
        )
        print(f"  cache path: {util_path}")
        for cat in util:
            top = sorted(util[cat].items(), key=lambda x: -x[1])[:3]
            print(f"  {cat:6s} highest: " + ", ".join(f"{s}={v:.2f}" for s, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
