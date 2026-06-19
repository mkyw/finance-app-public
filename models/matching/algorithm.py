"""Match a HouseholdProfile against the synthetic population.

Design principle: always load the user's own PUMA plus the top-M similar
PUMAs from pipeline/artifacts/puma_similarity/puma_neighbors.parquet
(default M=10). This is the default behavior — there is no fallback chain
and no N<100 trigger. The Epanechnikov kernel does all the weighting at
the household level; there is no PUMA-level weight multiplier.

Pipeline per query:
  1. Look up top-M similar PUMAs for profile.puma_code, combine with the
     user's own PUMA (total M+1 PUMAs loaded).
  2. Read the parquet partitions for those PUMAs from
     pipeline/artifacts/synthetic_population/.
  3. Hard-filter on tenure (OWN never mixes with RENT).
  4. Compute Gower distance (see distance.py — log(income), household
     size, age with weights 0.5/0.3/0.2 normalized by population range).
  5. Adaptive Epanechnikov bandwidth at the 150th-nearest neighbor.
  6. Weighted percentiles per CEX category using ACS sample weight times
     the kernel weight.
  7. Return MatchResult with 55 SpendingDistributions plus diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from models.matching.calibration import gas_underreporting_factor
from models.matching.confidence import confidence_label, effective_sample_size
from models.matching.cpi_scaler import (
    REGIONAL_BLS_CATEGORIES,
    load_cpi_scalars,
    regional_scalar_is_live,
    resolve_regional_scalar,
    resolve_scalar,
)
from models.matching.distance import gower_distance
from models.matching.eia_gas import load_eia_gas_scalars
from models.matching.eia_utility import load_eia_utility_scalars
from models.matching.geographic_drivers import GeographicDrivers, load_climate_normals
from models.matching.pce_scaler import load_pce_factors, resolve_pce_factor
from models.matching.rpp_scaler import (
    BALANCE_CATEGORIES,
    DIRECT_COST_CATEGORIES,
    EPISODIC_CATEGORIES,
    _get_rpp_value,
    _get_state_rpp_value,
    get_rpp_scalar,
    load_rpp_scalars,
)
from models.matching.vehreg_state import (
    load_vehreg_state_costs,
    lookup_state_cost,
)
from shared.constants.categories import (
    AGG_CARVED_MEMBER,
    AGG_CARVED_SOURCE,
    AGG_GROUPS,
    AGG_MEMBER_TO_GROUP,
    AGG_TRANSPORT_FINANCE,
    AGG_TRANSPORT_TRANSIT,
    CATEGORY_CODES,
    ENTERTAINMENT_STREAMING_ANNUAL,
)
from shared.constants.geography import STATE_TO_DIVISION, STATE_TO_REGION
from shared.types import HouseholdProfile, SpendingDistribution

# Bandwidth is set at the distance of the k-th nearest neighbor, with k=150.
# The Epanechnikov kernel zeroes out weights outside that radius.
_BANDWIDTH_K: int = 150

# Percentile targets returned in every SpendingDistribution.
_PERCENTILES: tuple[float, float, float, float, float] = (0.10, 0.25, 0.50, 0.75, 0.90)

# Geographic-tier preference for similar-PUMA expansion. Demographic
# similarity alone can pull, say, a New Hampshire PUMA into an Idaho
# match if their income/age/housing profiles line up — but the cohort's
# spending levels are wrong for Idaho prices. We bias the expansion
# toward geographically proximate PUMAs so the matched cohort shares
# not just demographics but also regional price structure.
#
# Semantics: fill at most ``_SAME_STATE_QUOTA`` slots from the same
# state first, then fall through to division, region, and finally
# unrestricted (any-state) neighbors — all in demographic-rank order
# within each tier. If a tier doesn't exist for a PUMA (small states,
# DC's division-of-one for this purpose), it contributes nothing and
# the next tier is consulted.
_SAME_STATE_QUOTA: int = 6

# Per-household RPP ratio (target / source) is clipped to this band
# before being applied. Keeps a runaway source-state lookup (e.g. a
# missing state in the JSON defaulting to national index = 100 while
# the target is a >200-index housing sub-index) from multiplying
# single-household values by absurd factors.
_RPP_RATIO_CLIP: tuple[float, float] = (0.5, 2.5)

# Thresholds for the car-ownership segmentation branch. The matched
# pool's weighted share of ACS ``vehicles >= 1`` households drives a
# three-way branch:
#
#   > HIGH: classified ``owner``. Percentile on a weighted blend of
#           the single-car (``vehicles==1``) and multi-car
#           (``vehicles>=2``) active-driver subsets, mix controlled
#           by the cohort's empirical ``cohort_mean_veh``.
#   < LOW:  classified ``non_owner``. Percentile on ``vehicles==0``
#           subset — vehicle cats collapse to near-zero naturally,
#           transit cats reflect higher non-car-owner usage.
#   mid:    classified ``ambiguous``. Full pool (no segmentation) —
#           the unconditional percentile already mixes the two
#           subpopulations reasonably.
_CAR_OWNER_THRESHOLD: float = 0.6
_CARLESS_THRESHOLD: float = 0.3

# Per-household active-driver refinement thresholds in raw CEX
# dollars. Applied to the single- and multi-car subsets so stored /
# backup cars (insured minimally, rarely driven) don't drag the
# active-driver spending percentiles down:
#   vehins > $400/year (~$33/month minimum) — pay for active coverage.
#   gas > $200/year (~one tank/month) — actually drive.
_CAR_VEHINS_THRESHOLD: float = 400.0
_CAR_GAS_THRESHOLD: float = 200.0

# Categories whose cohort percentiles are segmented on car-ownership.
# Vehicle categories correlate positively with ownership; transit
# categories correlate negatively. The branch logic handles both, but
# the non-owner branch distinguishes them:
#   Vehicle cats on non-owner: hard-zeroed. Fusion over-predicts
#     CEX vehicle spending for ACS veh=0 households because the model
#     doesn't take veh as an input — predictions come purely from
#     demographics. Categories like ``vehins`` and ``vehreg`` literally
#     can't exist without a car, so we emit zeros instead of carrying
#     the fusion prediction into the allocator.
#   Transit cats on non-owner: percentile on the veh=0 subset. Fusion
#     isn't perfect here either, but under-prediction is less
#     catastrophic than carrying phantom vehicle costs.
# Excluded from conditioning entirely: ``airshp`` (long-distance
# travel, not a daily-use car substitute), ``vehval`` (BALANCE
# category), ``finpay``/``othint`` (generic).
_VEHICLE_CATS: frozenset[str] = frozenset({
    "gas", "vehins", "vehint", "vehmlr", "vehnew",
    "vehprd", "vehprn", "vehreg", "vehusd",
})
_TRANSIT_CATS: frozenset[str] = frozenset({"pubtrn", "taxis"})
_CAR_OWNER_CATS: frozenset[str] = _VEHICLE_CATS | _TRANSIT_CATS


@dataclass(frozen=True)
class MatchResult:
    distributions: dict[str, SpendingDistribution]
    n_effective: float
    confidence: str
    pumas_used: list[str]
    city_pumas_used: list[str]
    n_households: int
    m_similar_used: int
    cohort_median_income: float
    # Kish n_eff computed on Epanechnikov kernel weights *only* (before
    # multiplying by ACS sample weights). A direct measure of match
    # quality: how many households does the kernel pull in, and how
    # evenly distributed is the kernel weight across them? Typically
    # near ``_BANDWIDTH_K`` (~110-120) when the kernel reaches a dense
    # neighborhood. Much lower = genuinely sparse neighborhood.
    kernel_n_effective: float
    # Coefficient of variation of ``kernel_w * sample_w`` across
    # kernel-nonzero households. Captures ACS replicate-weight
    # concentration independent of match quality. Values near 0 mean
    # uniform weighting; values > 1 mean one or two ACS respondents
    # dominate the weighted average.
    weight_cv: float
    # Weighted share of the matched cohort with ACS ``vehicles >= 1``
    # (i.e. the household actually owns at least one vehicle).
    # Drives the three-way branch (car-owner / non-car-owner /
    # ambiguous) that selects which subset of the cohort is used to
    # percentile the 11 vehicle + transit categories. Uses ACS ground
    # truth instead of a fused-CEX AND proxy because the fusion model
    # doesn't take ACS veh as a predictor, so carless urban households
    # still receive non-trivial predicted vehicle spending purely from
    # their demographics, which would mis-classify them.
    car_owner_probability: float
    # Discrete classification derived from ``car_owner_probability``
    # plus the _CAR_OWNER_THRESHOLD / _CARLESS_THRESHOLD bands:
    # ``"owner"`` (> 0.6) | ``"non_owner"`` (< 0.3) | ``"ambiguous"``.
    # Surfaced to the API so the dev view can explain which subset
    # drove the vehicle/transit percentile numbers.
    car_owner_classification: str
    # Empirical E[veh | car-owner, demographics] — the weighted mean of
    # ACS ``vehicles`` over the car-owner subset of the matched cohort.
    # Drives the continuous blend between the single-car (veh==1) and
    # multi-car (veh>=2) subsets when ``car_owner_classification ==
    # "owner"``, and feeds the vehreg state-cost path when classification
    # is ``"owner"`` or ``"ambiguous"``. Populated for both of those
    # branches; set to NaN on the ``"non_owner"`` branch where it would
    # not be meaningful (no car-owner data in the cohort). Typical
    # values: 1.0-1.2 for fam_size=1 car-owners, 1.8-2.2 for fam_size=2+
    # car-owners.
    cohort_mean_veh: float
    # Weighted median of per-household equivalized income (gross_income /
    # sqrt(household_size), locked #3) in the matched pool. The back-fill's
    # high-earner stratification threshold: profiles with y_eq at/above this
    # read the high-earner-subset ``conditional_p90_hi`` cap instead of the
    # broad ``conditional_p90`` (HIGH-INCOME-DISCRETIONARY-CEILING-
    # STRATIFICATION, 2026-06-09). Defaults to 0.0 (stratification off) so
    # pre-existing constructions stay valid.
    cohort_median_y_eq: float = 0.0
    # Weighted mean of the vehicle-financing sub-component (vehint + vehprn)
    # inside the transportation aggregate, on the same mode-split weights used
    # to build the aggregate distribution. Populated only on the aggregated
    # path (aggregate=True); 0.0 otherwise. Consumed by compute_allocations()
    # to subtract the cohort finance estimate when the user reports an
    # auto_loan_payment (the reported loan lives in debt service; the cohort
    # finance embed in the transportation anchor would double-count it).
    transport_finance_mean: float = 0.0


def _state_of(puma_code: str) -> str:
    """Two-letter state postal prefix of a ``ST_NNNNN`` puma_code."""
    return puma_code[:2]


def _select_similar_pumas_tiered(
    ranked_neighbors: list[str],
    user_puma: str,
    m_similar: int,
) -> list[str]:
    """Pick up to m PUMAs from ranked_neighbors, preferring nearer geography.

    ``ranked_neighbors`` is the similarity table's output for this PUMA,
    in demographic-rank order (rank 1 = most similar). Filling strategy:

      1. Up to ``_SAME_STATE_QUOTA`` neighbors from the same state.
      2. Fill the remaining slots from the same census division.
      3. Then same region, then finally any remaining unused neighbors.

    Within each tier we preserve demographic rank order, so the best
    same-state match outranks every same-division match even if the
    division match has a smaller Euclidean distance. This is
    intentional: geography is a stronger prior on local prices than the
    5-feature demographic signature alone.
    """
    user_state = _state_of(user_puma)
    user_division = STATE_TO_DIVISION.get(user_state)
    user_region = STATE_TO_REGION.get(user_state)

    same_state: list[str] = []
    same_division: list[str] = []
    same_region: list[str] = []
    national: list[str] = []
    for p in ranked_neighbors:
        st = _state_of(p)
        if st == user_state:
            same_state.append(p)
            continue
        div = STATE_TO_DIVISION.get(st)
        if user_division is not None and div == user_division:
            same_division.append(p)
            continue
        reg = STATE_TO_REGION.get(st)
        if user_region is not None and reg == user_region:
            same_region.append(p)
            continue
        national.append(p)

    picked: list[str] = list(same_state[:_SAME_STATE_QUOTA])
    # Remaining slots fall through the tiers in order.
    remaining = m_similar - len(picked)
    for tier in (same_division, same_region, national):
        if remaining <= 0:
            break
        take = tier[:remaining]
        picked.extend(take)
        remaining -= len(take)
    # If demographic neighbors in earlier tiers ran short AND later
    # tiers also didn't fill the quota, the unused same-state slots
    # (beyond the quota) become eligible again as a last resort.
    if remaining > 0 and len(same_state) > _SAME_STATE_QUOTA:
        extra = same_state[_SAME_STATE_QUOTA:_SAME_STATE_QUOTA + remaining]
        picked.extend(extra)
    return picked


def _load_similar_pumas(
    artifacts_path: Path, user_puma: str, m_similar: int
) -> list[str]:
    """Top-m similar PUMAs for ``user_puma``, with geographic preference.

    Reads all 50 demographic neighbors from the similarity parquet, then
    selects m of them applying the same-state / same-division /
    same-region / national tier preference (see
    ``_select_similar_pumas_tiered``). Returns the user's own PUMA
    first, followed by the selected neighbors.
    """
    neighbors_path = artifacts_path / "puma_similarity" / "puma_neighbors.parquet"
    if not neighbors_path.exists():
        raise FileNotFoundError(f"Missing {neighbors_path}. Run export_puma_similarity.R first.")

    neighbors = ds.dataset(str(neighbors_path), format="parquet").to_table(
        filter=(ds.field("puma_code") == user_puma)
    ).to_pandas()
    if neighbors.empty:
        raise ValueError(f"Unknown puma_code: {user_puma!r}")

    ranked = neighbors.sort_values("rank")["neighbor_puma"].tolist()
    top = _select_similar_pumas_tiered(ranked, user_puma, m_similar)
    # Keep the user's PUMA first for deterministic ordering.
    return [user_puma, *top]


def _load_population(artifacts_path: Path, pumas: list[str]) -> pd.DataFrame:
    dataset = ds.dataset(
        str(artifacts_path / "synthetic_population"),
        format="parquet",
        partitioning="hive",
    )
    table = dataset.to_table(filter=ds.field("puma_code").isin(pumas))
    return table.to_pandas()


def _weighted_percentiles(
    values: np.ndarray, weights: np.ndarray, percentiles: tuple[float, ...]
) -> np.ndarray:
    """Sample-weighted percentiles via cumulative-weight interpolation.

    numpy.percentile does not support weights, so we sort by value, build a
    normalized CDF over the weights, and linearly interpolate at the
    requested quantile targets. Rows with zero weight have no effect.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Drop zero-weight rows (e.g., Epanechnikov cutoff).
    mask = weights > 0
    if not np.any(mask):
        return np.full(len(percentiles), np.nan)

    v = values[mask]
    w = weights[mask]
    order = np.argsort(v, kind="stable")
    v_sorted = v[order]
    w_sorted = w[order]

    # Cumulative share to the *midpoint* of each observation's weight —
    # standard "Type 4 / Weibull" plotting position generalized with weights.
    cum = np.cumsum(w_sorted)
    total = cum[-1]
    if total <= 0:
        return np.full(len(percentiles), np.nan)
    positions = (cum - 0.5 * w_sorted) / total

    # np.interp extrapolates as the endpoint value outside [positions[0],
    # positions[-1]] which matches the expected quantile behavior.
    return np.interp(np.asarray(percentiles), positions, v_sorted)


def _weighted_mean_and_trim(
    values: np.ndarray, weights: np.ndarray, trim_q: float = 0.95
) -> tuple[float, float]:
    """Sample-weighted mean and weighted trim-mean (mean over values at/below
    the weighted ``trim_q`` quantile).

    The trim-mean drops the upper ``1 - trim_q`` weight tail — the
    outlier-robust central tendency the lumpy anchor switch uses. Where a few
    within-cohort big-spenders inflate the plain mean, the trim-mean sits below
    it; where the category is smooth the two coincide. Returns
    ``(mean, trim_mean)``; falls back to ``mean`` when the trim leaves no mass.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = weights > 0
    if not np.any(mask):
        return 0.0, 0.0
    v = values[mask]
    w = weights[mask]
    mean = float(np.average(v, weights=w))
    cutoff = float(_weighted_percentiles(v, w, (trim_q,))[0])
    keep = v <= cutoff
    if not np.any(keep) or float(w[keep].sum()) <= 0:
        return mean, mean
    trim = float(np.average(v[keep], weights=w[keep]))
    return mean, trim


def _apply_household_rpp_correction(
    df: pd.DataFrame,
    target_puma: str,
    rpp_data: dict,
    categories: list[str],
) -> pd.DataFrame:
    """Translate each cross-state household's spending to target-location prices.

    For every non-balance, non-episodic category:
      - Same-state households: no change (their raw CEX spending is
        already measured at roughly target-area prices).
      - Cross-state households: multiply by ``target_rpp / source_rpp``
        clipped to ``_RPP_RATIO_CLIP``, where ``target_rpp`` is the
        BEA index for ``target_puma`` (MSA-level when the PUMA maps
        to an MSA, otherwise state-level), and ``source_rpp`` is the
        household's state-level BEA index.

    This is the primary spatial correction. After it runs, the pool's
    CEX values reflect target-location prices at the BEA sub-index
    level, so subsequent percentile computation doesn't need an
    additional distribution-level RPP multiplier for non-gas cats.
    """
    df = df.copy()
    target_state = target_puma.split("_")[0]
    source_states = df["puma_code"].str[:2]
    is_same_state = (source_states == target_state).to_numpy()
    cross_mask = ~is_same_state
    if not cross_mask.any():
        # Pool is entirely same-state (tier-1 quota filled on a
        # large-state PUMA with 10+ similar neighbors in the same
        # state). Nothing to translate.
        return df

    lo, hi = _RPP_RATIO_CLIP
    for cat in categories:
        if (
            cat in BALANCE_CATEGORIES
            or cat in EPISODIC_CATEGORIES
            or cat in DIRECT_COST_CATEGORIES
        ):
            continue
        if cat not in df.columns:
            continue
        target_rpp = _get_rpp_value(cat, target_puma, rpp_data)
        # State -> source BEA index, cached so repeated states don't
        # re-walk the fallback chain. Only states that actually appear
        # in the cross-state subset need a lookup.
        state_to_rpp: dict[str, float] = {}
        for s in source_states.unique():
            if s == target_state:
                continue
            state_to_rpp[s] = _get_state_rpp_value(cat, s, rpp_data)
        source_rpps = source_states.map(state_to_rpp).to_numpy(dtype=np.float64)
        # Same-state entries are NaN in source_rpps (not mapped), but
        # we only use ratios on cross_mask so they don't matter.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                (source_rpps > 0),
                target_rpp / source_rpps,
                1.0,
            )
        ratios = np.clip(ratios, lo, hi)
        # CEX category columns land in the parquet as int32. Cast to
        # float before the cross-state multiplication so pandas
        # doesn't warn about the implicit int->float assignment.
        col = df[cat].to_numpy(dtype=np.float64)
        col[cross_mask] = col[cross_mask] * ratios[cross_mask]
        df[cat] = col
    return df


def _percentiles_with_nonzero_stats(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute (p10, p25, p50, p75, p90, conditional_p10, conditional_p90, nonzero_rate).

    Returns all zeros if weights sum to zero or the values are empty.
    ``conditional_p10``/``conditional_p90`` are the p10/p90 of the strictly-positive
    subset with its kernel weights preserved — the realistic min/max among actual
    spenders (used downstream: conditional_p90 as the allocator's zero-inflated
    ceiling, conditional_p10 as the soft-constraint optimizer's participation floor).
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weight_total = float(weights.sum())
    if weight_total <= 0 or values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    p10, p25, p50, p75, p90 = _weighted_percentiles(values, weights, _PERCENTILES)
    if not np.all(np.isfinite([p10, p25, p50, p75, p90])):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    nonzero_mask = values > 0
    nonzero_weight = weights * nonzero_mask
    nonzero_w_sum = float(nonzero_weight.sum())
    nz_rate = nonzero_w_sum / weight_total if weight_total > 0 else 0.0
    if int(nonzero_mask.sum()) >= 2 and nonzero_w_sum > 0:
        c_p10, c_p90 = _weighted_percentiles(values, nonzero_weight, (0.10, 0.90))
        c_p10 = float(c_p10) if np.isfinite(c_p10) else 0.0
        c_p90 = float(c_p90) if np.isfinite(c_p90) else 0.0
    else:
        c_p10 = c_p90 = 0.0
    return (float(p10), float(p25), float(p50), float(p75), float(p90),
            c_p10, c_p90, float(nz_rate))


def _conditional_p90_subset(
    values: np.ndarray, weights: np.ndarray, subset_mask: np.ndarray
) -> float:
    """Weighted p90 of the strictly-positive values within ``subset_mask``.

    The high-earner-stratified counterpart of ``conditional_p90`` (the same
    nonzero-only weighted p90, restricted to the ``y_eq >= cohort median``
    sub-population). Returns 0.0 when the masked nonzero subset is too thin
    (< 2 households) or carries no weight — downstream treats 0.0 as
    "unpopulated, fall back to the broad conditional_p90".
    """
    mask = (values > 0) & subset_mask
    w = weights * mask
    w_sum = float(w.sum())
    if int(mask.sum()) < 2 or w_sum <= 0:
        return 0.0
    (c_p90_hi,) = _weighted_percentiles(values, w, (0.90,))
    return float(c_p90_hi) if np.isfinite(c_p90_hi) else 0.0


def _category_scalar(
    cat: str,
    profile: HouseholdProfile,
    cpi_scalars: dict,
    rpp_data: dict,
    eia_gas_scalars: dict,
    user_region: str | None,
    pce_factors: dict | None = None,
    geo_drivers: "GeographicDrivers | None" = None,
) -> float:
    """CPI × spatial × gas-underreporting × CE-PCE × geographic-driver scalar.

    Mirrors the post-percentile scaling applied in the per-category matching
    loop. Extracted so the aggregated (W2) path can bake it into per-household
    values *before* summing into an aggregate — members carry different
    scalars (``gas`` gets a regional CPI + EIA premium + diary-underreporting
    factor; other members get national CPI), so a single post-sum scalar would
    be wrong.

    The CE-PCE undercapture factor (``pce_scaler``) lifts recall-underreported
    discretionary anchors toward their PCE level. It is keyed by ``cat``, so on
    the aggregated path a merged aggregate is lifted at its *member* codes
    (e.g. ``shopping`` via its ``cloftw`` member) — the right per-household
    composition — and the post-sum aggregate must not be re-scaled. Factor 1.0
    (the default) for every non-correctable category.
    """
    if cat in REGIONAL_BLS_CATEGORIES:
        cpi_s = resolve_regional_scalar(cpi_scalars, cat, user_region)
    else:
        cpi_s = resolve_scalar(cpi_scalars, cat)
    spatial_s = get_rpp_scalar(cat, profile.puma_code, rpp_data, eia_gas_scalars)
    if cat == "gas":
        user_state = profile.puma_code[:2]
        eia_available = user_state in eia_gas_scalars
        bls_live = regional_scalar_is_live(cpi_scalars, "gas", user_region)
        underreport_s = gas_underreporting_factor(
            eia_available=eia_available,
            bls_regional_available=bls_live,
        )
    else:
        underreport_s = 1.0
    pce_s = resolve_pce_factor(pce_factors or {}, cat)
    # Geographic-driver factor (value layer, alongside CPI/RPP). Currently the
    # utility climate lane (EIA state consumption × NOAA within-state degree-days)
    # for elec/ngas/ofuel; 1.0 for every other category and when artifacts are
    # absent. Orthogonal to spatial_s (RPP = price; this = climate quantity).
    geo_s = geo_drivers.factor(cat, profile.puma_code) if geo_drivers is not None else 1.0
    return cpi_s * spatial_s * underreport_s * pce_s * geo_s


def match_household(
    profile: HouseholdProfile,
    artifacts_path: str | Path,
    city_pumas: list[str] | None = None,
    m_similar_pumas: int = 10,
    aggregate: bool = False,
) -> MatchResult:
    """Match a household profile and return weighted percentile distributions.

    Args:
        profile: The querying household.
        artifacts_path: Directory holding both ``puma_similarity/`` and
            ``synthetic_population/`` subdirectories (i.e. the repo's
            ``pipeline/artifacts`` directory).
        city_pumas: Optional seed pool from city->PUMA resolution.
            When provided and non-empty, these PUMAs are the seed set
            and similarity expansion runs off ``city_pumas[0]``.
            When omitted or empty, seeding uses ``profile.puma_code``
            alone (legacy single-PUMA behavior).
        m_similar_pumas: Number of similar PUMAs to pool alongside the
            seed set. Default 10 gives pools of ~1k–30k households
            depending on geography.

    Returns:
        MatchResult with 55 SpendingDistributions keyed by category code.

    Raises:
        ValueError: if the seed PUMA is unknown to the similarity
            table, or if no households remain after tenure filtering.
        FileNotFoundError: if required artifacts are missing.
    """
    root = Path(artifacts_path)

    # Step a. Resolve seed pool and expand by similarity around the
    # representative PUMA (either city_pumas[0] or profile.puma_code).
    if city_pumas:
        seed_pumas = list(city_pumas)
        representative = seed_pumas[0]
    else:
        seed_pumas = [profile.puma_code]
        representative = profile.puma_code

    expanded = _load_similar_pumas(root, representative, m_similar_pumas)
    # Dedupe while preserving order: seeds first (city precedence), then
    # the representative + similarity neighbors, skipping anything already
    # in the seed set.
    seen: set[str] = set()
    pumas_used: list[str] = []
    for p in (*seed_pumas, *expanded):
        if p not in seen:
            seen.add(p)
            pumas_used.append(p)

    # Step b. Load household partitions.
    pop = _load_population(root, pumas_used)
    if pop.empty:
        raise ValueError(f"No households found for pumas: {pumas_used}")

    # Step c. Hard tenure filter.
    pop = pop.loc[pop["tenure"] == profile.tenure.value].reset_index(drop=True)
    if pop.empty:
        raise ValueError(
            f"No households with tenure={profile.tenure.value} in pool of {len(pumas_used)} PUMAs"
        )

    # Step d. Gower distance.
    distances = gower_distance(profile, pop)

    # Step e. Adaptive bandwidth at the k-th nearest neighbor.
    sorted_d = np.sort(distances)
    idx = min(_BANDWIDTH_K - 1, len(sorted_d) - 1)
    h = float(sorted_d[idx])
    # If the k-th neighbor is at distance 0 (near-duplicates), fall back to
    # max distance so every row gets a positive kernel weight.
    if h <= 0.0:
        h = float(sorted_d[-1]) if sorted_d[-1] > 0 else 1.0

    # Step f. Epanechnikov kernel weights.
    u = distances / h
    kernel_w = np.maximum(0.0, 1.0 - u * u)

    # Combine ACS sampling weight with kernel weight.
    sample_w = pop["weight"].to_numpy(dtype=np.float64)
    final_w = sample_w * kernel_w

    # Step g. Spatial & temporal scaling pipeline.
    #
    # The cohort median of a category-column in the pool is an
    # approximation of "typical spend for a household like this at
    # target-area prices". Translating it to that reference involves:
    #
    #  (1) Household-level RPP correction: re-prices cross-state
    #      pool households from their own state to target location at
    #      the BEA sub-index level. Same-state households unchanged.
    #  (2) Conditional gas percentiles: filter to car owners if the
    #      cohort is predominantly car-owning; zero out if
    #      predominantly carless.
    #  (3) Weighted percentiles over the corrected pool (or the
    #      car-owner subset for gas).
    #  (4) CEX underreporting correction (gas × 1.25 — see
    #      calibration.py).
    #  (5) CPI temporal: national CPI, or census-region BLS CPI for
    #      the five cats in REGIONAL_BLS_CATEGORIES.
    #  (6) EIA state gasoline premium for gas only; 1.0 for all other
    #      cats (spatial already handled at the household level).
    cpi_scalars = load_cpi_scalars(str(root / "cpi_scalars.json"))
    rpp_data = load_rpp_scalars(str(root / "rpp_scalars.json"))
    eia_gas_scalars = load_eia_gas_scalars(str(root / "eia_gas_scalars.json"))
    # CE-PCE undercapture factors (value-layer, alongside CPI/RPP). Absent
    # artifact -> empty dict -> factor 1.0 everywhere (clean no-op).
    pce_factors = load_pce_factors(str(root / "pce_correction.json"))
    # Geographic-driver artifacts (utility climate lane). Both file-read only —
    # never fetch at runtime (EIA network access is confined to the offline
    # refresh). Absent artifacts -> empty dict -> factor 1.0 everywhere.
    geo_drivers = GeographicDrivers(
        eia_utility_scalars=load_eia_utility_scalars(str(root / "eia_utility_scalars.json")),
        climate_normals=load_climate_normals(str(root / "climate_normals.json")),
    )
    vehreg_state_costs, vehreg_national_mean = load_vehreg_state_costs(
        str(root / "vehreg_state_costs.json")
    )
    # Lowercase census region for the user's PUMA. None for PUMAs with
    # no census-region mapping (PR and similar non-regional inputs),
    # in which case resolve_regional_scalar falls back to national CPI.
    user_region = STATE_TO_REGION.get(profile.puma_code[:2])
    user_region = user_region.lower() if user_region else None

    # Pool's weighted share of households that actually own a vehicle,
    # read straight from the ACS recipient side (``vehicles >= 1``).
    # Earlier revisions of this code tried to infer car ownership from
    # fused CEX ``vehins``/``gas`` (AND proxy at $400 + $200). That
    # approach over-classified urban households as car-owners because
    # the fusion model doesn't take ACS ``veh`` as an input, so
    # carless Manhattan households still received non-trivial CEX
    # vehicle spending purely from their demographics. ACS ``veh`` is
    # ground truth per household — using it directly matches the
    # carless share ACS actually reports (80-92% for Manhattan-core
    # PUMA renters, ~5-15% for suburban). The CEX AND proxy is still
    # used below to refine the single/multi-car subset to
    # active-driver households (stored / backup cars with minimal
    # insurance don't contaminate the blend).
    vehicles_raw = pop["vehicles"].to_numpy(dtype=np.int64)
    has_veh = (vehicles_raw >= 1).astype(np.float64)
    weight_total = float(final_w.sum())
    if weight_total > 0:
        car_owner_probability = float(np.average(has_veh, weights=final_w))
    else:
        car_owner_probability = 0.0

    # Household-level spatial correction. Operates on a copy so the
    # raw pool stays available for any future diagnostics (car
    # indicator above was already extracted).
    corrected = _apply_household_rpp_correction(
        pop, profile.puma_code, rpp_data, CATEGORY_CODES
    )

    # Pre-compute the car-ownership masks once.
    # Classification is ACS-ground-truth: vehicles >= 1 owns a car.
    # The single/multi-car subsets layer in the CEX AND proxy so that
    # stored / backup vehicles (insured minimally, rarely driven) don't
    # drag the spending percentiles down — we want the blend to reflect
    # active-driver households. The thresholds are in raw CEX units;
    # cross-state households may see their ``vehins`` scaled by the
    # household-level RPP correction but that preserves the
    # positive/nonzero relationship being tested.
    vehicles_arr = corrected["vehicles"].to_numpy(dtype=np.int64)
    has_car_mask_arr = vehicles_arr >= 1
    non_car_mask_arr = vehicles_arr == 0

    active_driver_mask_arr = (
        (corrected["vehins"] > _CAR_VEHINS_THRESHOLD)
        & (corrected["gas"] > _CAR_GAS_THRESHOLD)
    ).to_numpy()
    single_car_mask_arr = (vehicles_arr == 1) & active_driver_mask_arr
    multi_car_mask_arr = (vehicles_arr >= 2) & active_driver_mask_arr

    # Empirical E[veh | owns-car, demographics]. Computed *within* the
    # has-car subset so it's conditional on "this cohort owns cars".
    # Used below to blend the single-car and multi-car subsets when
    # the user is classified as a car-owner.
    car_owner_weight_total = float(final_w[has_car_mask_arr].sum())
    if car_owner_weight_total > 0:
        cohort_mean_veh = float(np.average(
            vehicles_arr[has_car_mask_arr],
            weights=final_w[has_car_mask_arr],
        ))
    else:
        cohort_mean_veh = 1.0  # defensive; unused unless branch taken

    # Blend weight w_multi in [0, 1] driven by cohort_mean_veh:
    #   cohort_mean_veh = 1.0 -> 0.0 (all single-car)
    #   cohort_mean_veh = 1.5 -> 0.5 (50/50)
    #   cohort_mean_veh >= 2  -> 1.0 (all multi-car)
    blend_w_multi = float(np.clip(cohort_mean_veh - 1.0, 0.0, 1.0))

    # Pre-built blended weight vector for the car-owner branch. Reused
    # across every category in _CAR_OWNER_CATS so we don't rebuild it
    # 11 times per match.
    w_blend = np.zeros_like(final_w)
    w_blend[single_car_mask_arr] = (
        final_w[single_car_mask_arr] * (1.0 - blend_w_multi)
    )
    w_blend[multi_car_mask_arr] = (
        final_w[multi_car_mask_arr] * blend_w_multi
    )

    # Classification band — used for API surfacing and for branch
    # selection below.
    if car_owner_probability > _CAR_OWNER_THRESHOLD:
        car_owner_classification = "owner"
    elif car_owner_probability < _CARLESS_THRESHOLD:
        car_owner_classification = "non_owner"
    else:
        car_owner_classification = "ambiguous"

    # High-earner stratification mask (back-fill discretionary-ceiling
    # stratification, 2026-06-09). Per-household equivalized income on the
    # same sqrt scale as ``HouseholdProfile._equivalence_scale`` (locked #3);
    # the threshold is the weighted cohort MEDIAN y_eq
    # (STRATIFY_THRESHOLD_QUANTILE = 0.50 — the stratification investigation's
    # calibration: the high-earner cp90 gap holds robustly down to the
    # top-tercile boundary and attenuates below, so median captures the
    # population where the broad-cohort cap under-calibration is real).
    hh_y_eq = pop["gross_income"].to_numpy(dtype=np.float64) / np.sqrt(
        np.maximum(pop["household_size"].to_numpy(dtype=np.float64), 1.0)
    )
    cohort_median_y_eq = float(
        _weighted_percentiles(hh_y_eq, final_w, (0.50,))[0]
    )
    if not np.isfinite(cohort_median_y_eq) or cohort_median_y_eq <= 0:
        cohort_median_y_eq = float(profile.equivalized_income)
    hi_earner_mask = hh_y_eq >= cohort_median_y_eq

    distributions: dict[str, SpendingDistribution] = {}
    for cat in CATEGORY_CODES:
        # Aggregated path: merged members are folded into their aggregate
        # below (W2). vehreg is NOT a merged member, so it still flows
        # through its direct-cost branch here.
        if aggregate and cat in AGG_MEMBER_TO_GROUP:
            continue
        if cat not in corrected.columns:
            raise ValueError(f"Category column missing from parquet: {cat}")
        values = corrected[cat].to_numpy(dtype=np.float64)
        # Sample×kernel-weighted mean + trim95 mean, populated only in the
        # plain-percentile (non-car-owner) branch below — that is where the
        # smooth-category set lives. Stay 0.0 (unused) for car-owner cats and
        # vehreg.
        wmean_raw = 0.0
        wtrim_raw = 0.0
        # High-earner-stratified conditional p90 — populated only in the
        # plain-percentile branch (0.0 = unpopulated elsewhere; downstream
        # falls back to the broad conditional_p90).
        c_p90_hi_raw = 0.0

        # Direct-cost categories bypass the cohort percentile + RPP +
        # CPI machinery. Currently only vehreg: state DMV registration
        # fees vary ~10x and are statutory, so we use the hand-curated
        # per-state cost multiplied by predicted vehicle count.
        if cat == "vehreg":
            state_cost = lookup_state_cost(
                profile.puma_code, vehreg_state_costs, vehreg_national_mean
            )
            if car_owner_classification == "owner":
                predicted_cars = cohort_mean_veh
            elif car_owner_classification == "non_owner":
                predicted_cars = 0.0
            else:
                # Ambiguous: E[vehreg] = P(owns car) * E[vehreg | owns car].
                # cohort_mean_veh is the conditional mean over the
                # has-car subset (already computed above), so multiplying
                # by car_owner_probability gives the unconditional
                # expectation across the cohort.
                predicted_cars = cohort_mean_veh * car_owner_probability
            p50_v = state_cost * predicted_cars
            distributions[cat] = SpendingDistribution(
                p10=p50_v * 0.85,
                p25=p50_v * 0.92,
                p50=p50_v,
                p75=p50_v * 1.08,
                p90=p50_v * 1.15,
                engel_estimate=0.0,
                feasibility_adjusted=0.0,
                cohort_position=0.0,
                is_structural=False,
                behavioral_gap=0.0,
                nonzero_rate=1.0 if predicted_cars > 0 else 0.0,
                conditional_p10=p50_v * 0.85,
                conditional_p90=p50_v * 1.15,
            )
            continue

        if cat in _CAR_OWNER_CATS:
            if car_owner_classification == "owner":
                # Blend single-car and multi-car subsets of the
                # car-owner cohort. Percentile on the reweighted
                # mixture distribution — statistically coherent
                # (not a naive blend of two separate percentile
                # numbers). Falls back to the full car-owner subset
                # if w_blend is all-zero (shouldn't happen given the
                # classification was "owner", but guard for empty
                # subsets).
                if w_blend.sum() > 0:
                    (p10, p25, p50, p75, p90,
                     c_p10, c_p90, nz_rate) = _percentiles_with_nonzero_stats(
                        values, w_blend,
                    )
                else:
                    fallback_w = final_w * has_car_mask_arr
                    if fallback_w.sum() > 0:
                        (p10, p25, p50, p75, p90,
                         c_p10, c_p90, nz_rate) = _percentiles_with_nonzero_stats(
                            values, fallback_w,
                        )
                    else:
                        (p10, p25, p50, p75, p90,
                         c_p10, c_p90, nz_rate) = _percentiles_with_nonzero_stats(
                            values, final_w,
                        )
            elif car_owner_classification == "non_owner":
                if cat in _VEHICLE_CATS:
                    # Vehicle cats cannot exist without a car. Fusion
                    # predicts non-zero values anyway because it only
                    # sees demographics, so we hard-zero these rather
                    # than let phantom insurance/registration costs
                    # enter the allocator.
                    p10 = p25 = p50 = p75 = p90 = 0.0
                    c_p10 = c_p90 = 0.0
                    nz_rate = 0.0
                else:
                    # Transit cats — percentile on the ACS veh=0 subset
                    # (cohort of households that really don't own cars).
                    non_car_w = final_w * non_car_mask_arr
                    if non_car_w.sum() > 0:
                        (p10, p25, p50, p75, p90,
                         c_p10, c_p90, nz_rate) = _percentiles_with_nonzero_stats(
                            values, non_car_w,
                        )
                    else:
                        (p10, p25, p50, p75, p90,
                         c_p10, c_p90, nz_rate) = _percentiles_with_nonzero_stats(
                            values, final_w,
                        )
            else:
                # Ambiguous — percentile on the full pool, then scale by
                # ownership probability so the bundle is internally
                # coherent with the probability-weighted vehreg path
                # (see locked decision VEH-AMBIG-WEIGHT). The fusion
                # model never sees ACS ``veh``, so the unconditional pool
                # assigns full vehicle/transit spend to carless
                # households; scaling re-expresses one consistent
                # ownership expectation across the bundle.
                #   Vehicle cats: × car_owner_probability (here all 8 of
                #     the vehicle cats minus vehreg, which was already
                #     special-cased out above).
                #   Transit cats: × (1 − car_owner_probability) — full-pool
                #     scaling rather than the carless-subset percentile,
                #     to avoid spurious zeros for part-car households.
                # Scaling a percentile value is linear, so the multiplier
                # applies to the dollar-valued percentiles (and the
                # dollar-valued conditional_p90). ``nz_rate`` is a
                # probability and is left unscaled, matching how the
                # owner/non_owner branches treat it.
                (p10, p25, p50, p75, p90,
                 c_p10, c_p90, nz_rate) = _percentiles_with_nonzero_stats(
                    values, final_w,
                )
                if cat in _TRANSIT_CATS:
                    ambig_scale = 1.0 - car_owner_probability
                else:
                    ambig_scale = car_owner_probability
                p10 *= ambig_scale
                p25 *= ambig_scale
                p50 *= ambig_scale
                p75 *= ambig_scale
                p90 *= ambig_scale
                c_p10 *= ambig_scale
                c_p90 *= ambig_scale
        else:
            (p10, p25, p50, p75, p90,
             c_p10, c_p90, nz_rate) = _percentiles_with_nonzero_stats(
                values, final_w,
            )
            # High-earner-stratified conditional p90 (back-fill cap
            # stratification). Plain-percentile branch only — every back-fill
            # target retained cat (eatout/hotel/airshp/recrp) flows through
            # here; car-owner cats and vehreg are never back-fill targets and
            # keep the 0.0 default (broad-cp90 fallback downstream).
            c_p90_hi_raw = _conditional_p90_subset(values, final_w, hi_earner_mask)
            # Weighted mean + trim95 mean on the same (values, final_w) the
            # percentiles use. The smooth-category set the allocator anchors on
            # the mean all flow through this branch (non-car-owner, non-merged
            # retained); the trim-mean serves the lumpy anchor switch (Build 2).
            if float(final_w.sum()) > 0.0:
                wmean_raw, wtrim_raw = _weighted_mean_and_trim(values, final_w)
            else:
                wmean_raw = wtrim_raw = 0.0

        # CPI temporal + spatial residual + gas diary-undercapture, all
        # folded into one positive multiplier (see _category_scalar).
        scalar = _category_scalar(
            cat, profile, cpi_scalars, rpp_data, eia_gas_scalars, user_region,
            pce_factors, geo_drivers,
        )
        distributions[cat] = SpendingDistribution(
            p10=p10 * scalar,
            p25=p25 * scalar,
            p50=p50 * scalar,
            p75=p75 * scalar,
            p90=p90 * scalar,
            engel_estimate=0.0,
            feasibility_adjusted=0.0,
            cohort_position=0.0,
            is_structural=False,
            behavioral_gap=0.0,
            nonzero_rate=nz_rate,
            conditional_p10=c_p10 * scalar,
            conditional_p90=c_p90 * scalar,
            conditional_p90_hi=c_p90_hi_raw * scalar,
            weighted_mean=wmean_raw * scalar,
            trimmed_mean=wtrim_raw * scalar,
        )

    # Aggregated path (W2): fold the merged members into 6 aggregate
    # distributions. The mode-split survives by working at the
    # per-household scaled-dollar level, then percentiling the household
    # transportation total once — members carry different price scalars
    # and the bands scale vehicle vs transit in opposite directions, so a
    # single aggregate-level scalar is impossible. See
    # agent-artifacts/aggregation/W2_modesplit_spec.md.
    # Finance sub-component of the transportation aggregate (vehint + vehprn)
    # on the aggregate mode-split weights. Populated inside the aggregated
    # block below; 0.0 on the disaggregated path. Surfaced in MatchResult so
    # compute_allocations() can subtract it when auto_loan_payment is reported
    # (the reported loan lives in debt service — the cohort embed double-counts).
    _transport_finance_mean: float = 0.0

    if aggregate:
        def _scaled(member: str) -> np.ndarray:
            vals = corrected[member].to_numpy(dtype=np.float64)
            return vals * _category_scalar(
                member, profile, cpi_scalars, rpp_data, eia_gas_scalars,
                user_region, pce_factors, geo_drivers,
            )

        non_car_w = final_w * non_car_mask_arr
        for group, members in AGG_GROUPS.items():
            if group == "transportation":
                if car_owner_classification == "owner":
                    # All members percentiled over the active-driver
                    # mixture weights (mirror the disaggregated owner
                    # branch + its empty-subset fallbacks).
                    if w_blend.sum() > 0:
                        weights_v = w_blend
                    elif float((final_w * has_car_mask_arr).sum()) > 0:
                        weights_v = final_w * has_car_mask_arr
                    else:
                        weights_v = final_w
                    v_agg = np.zeros_like(final_w)
                    v_finance = np.zeros_like(final_w)
                    for m in members:
                        sv = _scaled(m)
                        v_agg += sv
                        if m in AGG_TRANSPORT_FINANCE:
                            v_finance += sv
                    w_sum = float(weights_v.sum())
                    _transport_finance_mean = (
                        float(np.average(v_finance, weights=weights_v))
                        if w_sum > 0 else 0.0
                    )
                elif car_owner_classification == "non_owner":
                    # Vehicle bill structurally zero; transit on the
                    # veh==0 subset -> aggregate = transit-only.
                    weights_v = (
                        non_car_w if float(non_car_w.sum()) > 0 else final_w
                    )
                    v_agg = np.zeros_like(final_w)
                    for m in members:
                        if m in AGG_TRANSPORT_TRANSIT:
                            v_agg += _scaled(m)
                        # vehicle members (incl. finance) contribute 0
                    _transport_finance_mean = 0.0
                else:  # ambiguous — opposite-direction P(own) blend per HH
                    weights_v = final_w
                    v_agg = np.zeros_like(final_w)
                    v_finance = np.zeros_like(final_w)
                    for m in members:
                        sv = _scaled(m)
                        if m in AGG_TRANSPORT_TRANSIT:
                            v_agg += sv * (1.0 - car_owner_probability)
                        else:
                            v_agg += sv * car_owner_probability
                            if m in AGG_TRANSPORT_FINANCE:
                                v_finance += sv * car_owner_probability
                    w_sum = float(final_w.sum())
                    _transport_finance_mean = (
                        float(np.average(v_finance, weights=final_w))
                        if w_sum > 0 else 0.0
                    )
            else:
                weights_v = final_w
                v_agg = np.zeros_like(final_w)
                for m in members:
                    v_agg += _scaled(m)
                # Build-3 recomposition: carved personal-care + flat streaming.
                if group == "shopping":
                    # recurring core = cloftw (apparel) + carved pcare member,
                    # the latter at its own decontaminated recall factor.
                    pcare_raw = corrected[AGG_CARVED_MEMBER].to_numpy(dtype=np.float64)
                    v_agg = v_agg + pcare_raw * _category_scalar(
                        AGG_CARVED_MEMBER, profile, cpi_scalars, rpp_data,
                        eia_gas_scalars, user_region, pce_factors, geo_drivers,
                    )
                elif group == "household_goods":
                    # net the carved pcare out of hhpcp (raw x hhpcp's own scalar,
                    # i.e. exactly the pcare slice embedded in the scaled hhpcp).
                    pcare_raw = corrected[AGG_CARVED_MEMBER].to_numpy(dtype=np.float64)
                    v_agg = v_agg - pcare_raw * _category_scalar(
                        AGG_CARVED_SOURCE, profile, cpi_scalars, rpp_data,
                        eia_gas_scalars, user_region, pce_factors, geo_drivers,
                    )
                    np.clip(v_agg, 0.0, None, out=v_agg)  # exact carve => already >=0
                elif group == "entertainment":
                    # flat PCE-anchored streaming value add (saturates; near-flat
                    # in income). Per-household constant => shifts all percentiles.
                    v_agg = v_agg + ENTERTAINMENT_STREAMING_ANNUAL

            (p10, p25, p50, p75, p90, c_p10, c_p90, nz_rate) = (
                _percentiles_with_nonzero_stats(v_agg, weights_v)
            )
            # High-earner-stratified conditional p90 for the aggregate, on
            # the same mode-split weights (scalar already baked into v_agg).
            c_p90_hi_agg = _conditional_p90_subset(v_agg, weights_v, hi_earner_mask)
            # Weighted mean + trim95 mean of the per-household aggregate sum, on
            # the same mode-split weights the percentiles use — drives the
            # Build-2 lumpy anchor switch (plain mean for the now-smooth
            # de-lumped backbones, trim-mean for any residual outlier-distortion).
            wmean_agg, wtrim_agg = _weighted_mean_and_trim(v_agg, weights_v)
            # Scalar already baked into v_agg — do NOT re-apply it here.
            distributions[group] = SpendingDistribution(
                p10=p10,
                p25=p25,
                p50=p50,
                p75=p75,
                p90=p90,
                engel_estimate=0.0,
                feasibility_adjusted=0.0,
                cohort_position=0.0,
                is_structural=False,
                behavioral_gap=0.0,
                nonzero_rate=nz_rate,
                conditional_p10=c_p10,
                conditional_p90=c_p90,
                conditional_p90_hi=c_p90_hi_agg,
                weighted_mean=wmean_agg,
                trimmed_mean=wtrim_agg,
            )

    # Weighted median of gross_income in the matched pool. Used
    # downstream by the allocator's Engel income-gap correction: if
    # the user's income deviates from cohort median, elasticities
    # shift the CPI-scaled p50 up or down.
    incomes = pop["gross_income"].to_numpy(dtype=np.float64)
    cohort_median = float(_weighted_percentiles(incomes, final_w, (0.50,))[0])
    if not np.isfinite(cohort_median) or cohort_median <= 0:
        cohort_median = float(profile.gross_income)

    n_eff = effective_sample_size(final_w)

    # Match-quality diagnostics, reported separately from n_eff so the
    # UI can distinguish "the kernel found a thin neighborhood" from
    # "ACS replicate-weights are concentrated on one respondent."
    kernel_n_eff = effective_sample_size(kernel_w)
    nonzero_mask = kernel_w > 0
    if np.any(nonzero_mask):
        nz = final_w[nonzero_mask]
        mean_nz = float(nz.mean())
        weight_cv = float(nz.std() / mean_nz) if mean_nz > 0 else 0.0
    else:
        weight_cv = 0.0

    return MatchResult(
        distributions=distributions,
        n_effective=n_eff,
        confidence=confidence_label(n_eff),
        pumas_used=pumas_used,
        city_pumas_used=seed_pumas,
        n_households=int(len(pop)),
        m_similar_used=m_similar_pumas,
        cohort_median_income=cohort_median,
        kernel_n_effective=kernel_n_eff,
        weight_cv=weight_cv,
        car_owner_probability=car_owner_probability,
        car_owner_classification=car_owner_classification,
        cohort_mean_veh=(
            cohort_mean_veh
            if car_owner_classification != "non_owner"
            else float("nan")
        ),
        cohort_median_y_eq=cohort_median_y_eq,
        transport_finance_mean=_transport_finance_mean if aggregate else 0.0,
    )
