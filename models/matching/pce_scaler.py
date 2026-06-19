"""CE-PCE undercapture correction — value-layer factor loader.

Reads ``pipeline/artifacts/pce_correction.json`` (produced by
``pipeline/export/derive_pce_correction.py``) and exposes the per-category
multiplicative factor used in the anchor value-scaling slot, alongside CPI and
RPP (see ``algorithm._category_scalar``).

The factor lifts CEX recall-underreported discretionary anchors toward their
PCE level. It is applied **by category/member code**, so on the aggregated path
the ``shopping`` lift is realized at the ``cloftw`` member during aggregation
(per-household composition) — ``factors["shopping"]`` is a documented
cohort-mean-equivalent reference and is *not* separately applied (doing so would
double-count). Factor ``1.0`` = no correction (the default for every category
not in the correctable set, and for the deferred ``entertainment``).

Missing/unreadable artifact degrades to "no correction everywhere" (every
``resolve_pce_factor`` returns 1.0), so the matching path stays functional when
the correction has not been built.
"""
from __future__ import annotations

import json
import logging
import os

_LOG = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = "pipeline/artifacts/pce_correction.json"


def load_pce_factors(cache_path: str = _DEFAULT_CACHE_PATH) -> dict[str, float]:
    """Load the per-category CE-PCE factor map.

    Returns ``{category_code: factor}``. On a missing or unreadable artifact,
    returns ``{}`` — ``resolve_pce_factor`` then yields 1.0 for every category,
    a clean no-op.
    """
    if not os.path.exists(cache_path):
        _LOG.info("pce_correction artifact absent (%s); CE-PCE correction off", cache_path)
        return {}
    try:
        with open(cache_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        _LOG.warning("pce_correction artifact unreadable (%r); CE-PCE correction off", exc)
        return {}
    factors = data.get("factors", {})
    return {str(k): float(v) for k, v in factors.items()}


def resolve_pce_factor(pce_factors: dict[str, float], cat: str) -> float:
    """CE-PCE factor for one category code; 1.0 (no correction) by default."""
    return float(pce_factors.get(cat, 1.0))
