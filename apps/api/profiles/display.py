"""Display-aggregation data layer (display-aggregation project, 2026-05).

A PURE PRESENTATION layer over already-validated analyze output. It groups the
39 aggregated prediction categories into ~8 legible topic groups (food,
housing, utilities, …) plus an explicit ``net_worth`` topic for the balance
cats, each spending topic showing a *summed* total over its members with the
individual predictions preserved as addressable children.

What this layer is NOT:
  * It does not re-predict, re-aggregate the model, or touch matching / Engel /
    optimizer / coefficients. It reads ``run_profile_analysis`` output and sums
    already-computed ``feasibility_adjusted`` values for display only.
  * It is not the final framing. The topic grouping is deliberately
    intermediate scaffolding: a future interpretive cut (control-axis
    Fixed/Essential/Discretionary, or another framing — TBD) will re-project
    these same atoms under a different key. To keep that re-projection cheap,
    the output is built as a re-groupable ATOM SET (each of the 39 categories
    addressable, with its value + cadence + topic + flags) plus a TOPIC
    PROJECTION derived from it — never a flattening that loses the atoms.
  * It renders nothing. No UI, no labels-for-humans — that is deferred.

Correctness story (the whole of it): the roll-up is lossless. Grouping changes
presentation, never totals — a topic total is exactly the sum of its member
predicted values, and Σ(spending-topic predicted totals) + pinned housing
equals the flat total model-predicted spend, with balance atoms excluded.

Inputs: the dict returned by
``apps.api.profiles.services.run_profile_analysis`` on the default aggregated
path (``use_aggregated=True``, so ``distributions`` is keyed by the 39
``AGGREGATED_CATEGORY_CODES``), plus the household ``tenure`` (needed to know
which housing line is the user-pinned input vs. a prediction — see locked #5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from shared.constants.categories import (
    AGG_EPISODIC_SUBCOMPONENTS,
    AGGREGATED_BALANCE_CATEGORIES,
    AGGREGATED_CATEGORY_CODES,
    CATEGORY_CADENCE,
    OMIT_BY_DEFAULT_CATEGORIES,
    SPENDING_TOPIC_NAMES,
    TOPIC_GROUP,
    TOPIC_GROUP_NAMES,
    TOPIC_NET_WORTH,
    _TOPIC_MEMBERS,
)

Surface = Literal["paycheck", "annual"]

# Which housing line is the user-reported INPUT (pinned lb==ub by
# compute_bounds), by tenure — mirrors locked decision #5. mrtgps (secondary
# mortgage) is NOT pinned: it is predicted (and zeroed under RENT). These atoms
# are flagged so the display can keep the reported fact distinct from the
# predicted housing-adjacent lines and never blur the pin into the predicted
# total.
PINNED_HOUSING_BY_TENURE: dict[str, frozenset[str]] = {
    "RENT": frozenset({"rntval"}),
    "OWN": frozenset({"mrtgip", "mrtgpp"}),
}


@dataclass(frozen=True)
class DisplayAtom:
    """One of the 39 prediction categories, kept individually addressable.

    The re-groupable unit. ``value`` is the predicted flow allocation
    (``feasibility_adjusted``); it is 0 for balance atoms (they are zeroed in
    allocation and carried as net-worth context, never summed into spend).
    """

    category: str
    value: float                  # displayed total == measured_value + inferred_value
    cadence: str                  # "recurring" | "episodic" | "balance"
    topic: str
    is_pinned: bool               # user-reported housing input, not a prediction
    is_balance: bool              # wealth stock / liability balance (non-spending)
    # Residual back-fill split (Build 4, Q6): ``measured_value`` is the
    # cohort-typical baseline (the forward ``feasibility_adjusted``);
    # ``inferred_value`` is the back-fill increment (≥ 0, nonzero only on the
    # slope-ceiling discretionary targets that fired). Invariant:
    # ``value == measured_value + inferred_value``. ``confidence`` is
    # "inferred-lifestyle" when ``inferred_value > 0`` else "measured" — the
    # display shows the split neutrally ("Cohort-typical: $X · estimated lifestyle
    # range: +$Y"), never as a prescription.
    measured_value: float = 0.0
    inferred_value: float = 0.0
    confidence: str = "measured"
    # Heavy-zero, cohort-mean-meaningless cats (chrty/educ/ocash/stdint/othint/
    # finpay) — value-layer-zeroed; the initial display omits them, a future
    # additive UX surfaces them (HEAVY-ZERO-DISTRIBUTION-NEEDS-ELICITATION).
    omit_from_initial_view: bool = False
    # Episodic drill-down members folded inside a recurring aggregate (e.g.
    # transportation -> vehnew, vehusd). Carried so the future paycheck-surface
    # filter can drill in; empty for everything but the 3 recurring aggregates.
    # These are NOT separately predicted, so they are metadata, not atoms.
    episodic_subcomponents: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicGroupView:
    """A topic projected over its member atoms.

    ``predicted_total`` sums member values that are predictions (not pinned,
    not balance). ``pinned_total`` sums the user-reported pinned housing lines
    (only ever nonzero for the housing topic). For ``net_worth`` both totals
    are 0 — balance atoms are carried as ``members`` for context but never
    summed into any spending figure.
    """

    topic: str
    is_spending: bool
    predicted_total: float
    pinned_total: float
    members: tuple[DisplayAtom, ...]

    @property
    def total(self) -> float:
        """Full displayed total for the group (predicted + pinned)."""
        return self.predicted_total + self.pinned_total


@dataclass(frozen=True)
class DisplayRollup:
    """The full re-groupable structure: atoms + the topic projection.

    ``atoms`` is layer (a) — every one of the 39 categories addressable by
    code. ``topics`` is layer (b) — the topic-grouped view *derived from* the
    atoms. A future interpretive layer re-projects ``atoms`` under a new key;
    nothing here blocks it.
    """

    atoms: dict[str, DisplayAtom]
    topics: tuple[TopicGroupView, ...] = field(default_factory=tuple)

    # --- reconciliation-relevant aggregates (display sums, never re-predicts) -

    @property
    def spending_predicted_total(self) -> float:
        """Σ predicted (non-pinned, non-balance) over the spending topics."""
        return sum(
            t.predicted_total for t in self.topics if t.is_spending
        )

    @property
    def pinned_housing_total(self) -> float:
        """The user-reported pinned housing total (lives in the housing topic)."""
        return sum(t.pinned_total for t in self.topics if t.is_spending)

    @property
    def flat_total(self) -> float:
        """Flat total model-predicted spend = spending predicted + pinned.

        Equals the sum of ``value`` over every flow atom — the losslessness
        identity this layer must preserve.
        """
        return self.spending_predicted_total + self.pinned_housing_total

    def topic(self, name: str) -> TopicGroupView:
        for t in self.topics:
            if t.topic == name:
                return t
        raise KeyError(name)


def build_display_rollup(
    analyze_output: dict,
    tenure: str,
) -> DisplayRollup:
    """Build the atom set + topic projection from analyze output.

    Args:
        analyze_output: the dict from ``run_profile_analysis`` on the
            aggregated path. Its ``distributions`` must be keyed by the 39
            ``AGGREGATED_CATEGORY_CODES`` (this layer is only defined over the
            aggregated path; the 55-cat disaggregated output is rejected).
        tenure: "RENT" | "OWN" — selects the pinned housing line(s).

    Returns:
        A ``DisplayRollup`` whose topic totals sum losslessly to the flat
        model-predicted spend.
    """
    dists = analyze_output["distributions"]
    missing = set(AGGREGATED_CATEGORY_CODES) - set(dists)
    if missing:
        raise ValueError(
            "build_display_rollup expects aggregated analyze output keyed by "
            f"the 39 aggregated categories; missing {sorted(missing)}. "
            "(Was this run with use_aggregated=False?)"
        )

    pinned = PINNED_HOUSING_BY_TENURE.get(tenure.upper(), frozenset())

    atoms: dict[str, DisplayAtom] = {}
    for cat in AGGREGATED_CATEGORY_CODES:
        is_balance = cat in AGGREGATED_BALANCE_CATEGORIES
        # Measured = the forward cohort-typical allocation; inferred = the
        # back-fill increment (0 unless this discretionary target fired). The
        # displayed value is their sum (Q6 invariant). ``.get`` keeps older
        # synthetic analyze dicts (no back-fill fields) working — they read as
        # all-measured.
        measured = float(dists[cat]["feasibility_adjusted"])
        inferred = float(dists[cat].get("backfill_inferred", 0.0))
        atoms[cat] = DisplayAtom(
            category=cat,
            value=measured + inferred,
            cadence=CATEGORY_CADENCE[cat],
            topic=TOPIC_GROUP[cat],
            # balance atoms are never pinned-housing inputs
            is_pinned=(cat in pinned) and not is_balance,
            is_balance=is_balance,
            measured_value=measured,
            inferred_value=inferred,
            confidence="inferred-lifestyle" if inferred > 0.0 else "measured",
            episodic_subcomponents=AGG_EPISODIC_SUBCOMPONENTS.get(cat, ()),
            omit_from_initial_view=cat in OMIT_BY_DEFAULT_CATEGORIES,
        )

    topics: list[TopicGroupView] = []
    for name in TOPIC_GROUP_NAMES:
        members = tuple(atoms[c] for c in _TOPIC_MEMBERS[name])
        is_spending = name != TOPIC_NET_WORTH
        if is_spending:
            predicted = sum(
                a.value for a in members if not a.is_pinned and not a.is_balance
            )
            pinned_total = sum(a.value for a in members if a.is_pinned)
        else:
            # net_worth: balance atoms are context, never summed into spend.
            predicted = 0.0
            pinned_total = 0.0
        topics.append(
            TopicGroupView(
                topic=name,
                is_spending=is_spending,
                predicted_total=predicted,
                pinned_total=pinned_total,
                members=members,
            )
        )

    return DisplayRollup(atoms=atoms, topics=tuple(topics))


def rollup_to_dict(rollup: DisplayRollup) -> dict:
    """JSON-serializable form of the roll-up for the analyze API response.

    This is the single source of truth the frontend renders — the topic
    grouping, per-category values, and cadence all flow from here (which
    derives from shared/constants/categories.py), so there is no second
    grouping definition to keep in sync.
    """
    return {
        "topics": [
            {
                "topic": t.topic,
                "is_spending": t.is_spending,
                "predicted_total": t.predicted_total,
                "pinned_total": t.pinned_total,
                "members": [
                    {
                        "category": a.category,
                        "value": a.value,
                        "cadence": a.cadence,
                        "is_pinned": a.is_pinned,
                        "is_balance": a.is_balance,
                        # measured/inferred split (Build 4, Q6) — neutral framing.
                        "measured_value": a.measured_value,
                        "inferred_value": a.inferred_value,
                        "confidence": a.confidence,
                        "episodic_subcomponents": list(a.episodic_subcomponents),
                        "omit_from_initial_view": a.omit_from_initial_view,
                    }
                    for a in t.members
                ],
            }
            for t in rollup.topics
        ],
    }


# ---------------------------------------------------------------------------
# Cadence-driven surface derivation (paycheck vs. annual).
#
# A filter OVER the rolled-up structure, using the existing cadence tags — it
# selects which atoms appear on a surface and re-sums each topic over only
# those members. It computes nothing new.
#
#   paycheck — recurring atoms only. The recurring aggregates appear at their
#              full value: their episodic drill-down sub-members
#              (AGG_EPISODIC_SUBCOMPONENTS) are not separately predicted, so
#              they cannot be subtracted out here; they are carried as atom
#              metadata for a future filter to drill into. The aggregate's
#              membership of the surface is decided by its dominant (recurring)
#              cadence tag.
#   annual   — all flow atoms (recurring + episodic).
#   balance  — appears on neither spending surface.
#
# A topic's total on a surface is the sum of only its members on that surface,
# so a group's paycheck total can differ from its annual total when some
# members are episodic (e.g. shopping is episodic and absent from paycheck).
# ---------------------------------------------------------------------------


def _on_surface(atom: DisplayAtom, surface: Surface) -> bool:
    if atom.is_balance:
        return False
    if surface == "annual":
        return True  # all flow atoms
    # paycheck: recurring flow atoms only
    return atom.cadence == "recurring"


def surface_atom_codes(rollup: DisplayRollup, surface: Surface) -> set[str]:
    """The set of category codes appearing on ``surface``.

    paycheck ⊂ annual (strict) — the losslessness check for the surface split.
    """
    return {
        a.category for a in rollup.atoms.values() if _on_surface(a, surface)
    }


def project_surface(
    rollup: DisplayRollup,
    surface: Surface,
) -> tuple[TopicGroupView, ...]:
    """Re-project the spending topics, summing only members on ``surface``.

    Returns one ``TopicGroupView`` per spending topic (net_worth is omitted —
    balance atoms appear on no spending surface). Member lists are filtered to
    the surface; a topic with no members on the surface yields zero totals.
    """
    out: list[TopicGroupView] = []
    for name in SPENDING_TOPIC_NAMES:
        members = tuple(
            a for a in rollup.topic(name).members if _on_surface(a, surface)
        )
        predicted = sum(
            a.value for a in members if not a.is_pinned and not a.is_balance
        )
        pinned_total = sum(a.value for a in members if a.is_pinned)
        out.append(
            TopicGroupView(
                topic=name,
                is_spending=True,
                predicted_total=predicted,
                pinned_total=pinned_total,
                members=members,
            )
        )
    return tuple(out)
