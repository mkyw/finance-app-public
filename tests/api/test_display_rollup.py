"""Losslessness + partition tests for the display-aggregation layer.

The display roll-up (apps/api/profiles/display.py) is a pure presentation
projection of analyze output. Its entire correctness story is arithmetic
losslessness: grouping changes presentation, never totals. These tests pin
that, plus the TOPIC_GROUP partition, using a synthetic analyze output (no
artifacts / pipeline needed — the layer never re-predicts).

Run from repo root:
    python3.11 -m pytest tests/api/test_display_rollup.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

# The display module imports only shared.constants (no Django), but it lives
# under apps/api so make that importable.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "apps", "api")
)

from apps.api.profiles.display import (  # noqa: E402
    PINNED_HOUSING_BY_TENURE,
    build_display_rollup,
    project_surface,
    surface_atom_codes,
)
from shared.constants.categories import (  # noqa: E402
    AGGREGATED_BALANCE_CATEGORIES,
    AGGREGATED_CATEGORY_CODES,
    AGGREGATED_FLOW_CATEGORIES,
    RECURRING_CATEGORIES,
    SPENDING_TOPIC_NAMES,
    TOPIC_GROUP,
    TOPIC_GROUP_NAMES,
    TOPIC_NET_WORTH,
    _TOPIC_MEMBERS,
)


def _synthetic_analyze(values: dict[str, float] | None = None) -> dict:
    """Minimal analyze output: distinct, nonzero values per flow category.

    Balance cats get 0 (zeroed in allocation, as on the real path). Distinct
    values per category make a dropped/duplicated/misrouted category change a
    total, so the losslessness asserts actually bite.
    """
    dists: dict[str, dict] = {}
    for i, cat in enumerate(AGGREGATED_CATEGORY_CODES):
        if cat in AGGREGATED_BALANCE_CATEGORIES:
            v = 0.0
        elif values and cat in values:
            v = values[cat]
        else:
            v = 100.0 + i  # distinct, nonzero
        dists[cat] = {"feasibility_adjusted": v}
    return {"distributions": dists}


# --------------------------------------------------------------------------
# TOPIC_GROUP partition
# --------------------------------------------------------------------------

def test_topic_group_partitions_all_43():
    assert set(TOPIC_GROUP) == set(AGGREGATED_CATEGORY_CODES)
    assert len(TOPIC_GROUP) == 46  # 43 -> 46 after Build-3 entertainment/shopping de-merge
    # Every category resolves to a declared topic, exactly once.
    seen = [c for members in _TOPIC_MEMBERS.values() for c in members]
    assert len(seen) == 46
    assert len(set(seen)) == 46


def test_rollup_to_dict_is_json_serializable_and_groups_demerged_lines():
    """The API serialization the frontend consumes (single source of truth):
    JSON-safe, carries cadence for italics, and groups the Build-3 de-merged
    lines per the backend TOPIC_GROUP (recrp->entertainment; eltrnp/jwlbg->shopping)."""
    import json

    from profiles.display import rollup_to_dict

    payload = rollup_to_dict(build_display_rollup(_synthetic_analyze(), "RENT"))
    json.dumps(payload)  # must be JSON-serializable

    topics = {t["topic"]: [m["category"] for m in t["members"]] for t in payload["topics"]}
    assert "recrp" in topics["entertainment"]
    assert "eltrnp" in topics["shopping"] and "jwlbg" in topics["shopping"]
    # every member carries the fields the frontend renders
    for t in payload["topics"]:
        for m in t["members"]:
            assert {"category", "value", "cadence", "is_pinned", "is_balance"} <= set(m)


def test_net_worth_topic_is_exactly_balance_cats():
    assert set(_TOPIC_MEMBERS[TOPIC_NET_WORTH]) == set(AGGREGATED_BALANCE_CATEGORIES)
    assert all(
        TOPIC_GROUP[c] == TOPIC_NET_WORTH for c in AGGREGATED_BALANCE_CATEGORIES
    )
    # No flow category lands in net_worth.
    assert all(TOPIC_GROUP[c] != TOPIC_NET_WORTH for c in AGGREGATED_FLOW_CATEGORIES)


def test_about_eight_spending_topics():
    # 8 spending topics + net_worth. (Was 9 until the 2026-06 household_goods
    # fold into shopping — see test_household_goods_folded_into_shopping.)
    assert len(SPENDING_TOPIC_NAMES) == 8
    assert TOPIC_NET_WORTH not in SPENDING_TOPIC_NAMES
    assert TOPIC_NET_WORTH in TOPIC_GROUP_NAMES


def test_household_goods_folded_into_shopping():
    """2026-06 display fold: household_goods is no longer a standalone topic;
    its three categories live under the consolidated shopping topic. Pure
    display re-grouping — the category codes still exist as addressable atoms,
    each in exactly one topic. The fusion contract / per-category records are
    untouched (asserted elsewhere)."""
    from shared.constants.categories import TOPIC_GROUP

    # The standalone topic is gone.
    assert "household_goods" not in TOPIC_GROUP_NAMES
    assert "household_goods" not in SPENDING_TOPIC_NAMES
    # The three folded categories now map to the shopping topic.
    for cat in ("household_goods", "furhwr", "happl"):
        assert TOPIC_GROUP[cat] == "shopping", f"{cat} should be under shopping"
    # Shopping retains its existing members alongside the folded ones.
    for cat in ("shopping", "eltrnp", "jwlbg"):
        assert TOPIC_GROUP[cat] == "shopping"
    # The rollup surfaces the folded cats under shopping and no household_goods
    # topic appears; the shopping total includes the folded categories.
    rollup = build_display_rollup(_synthetic_analyze(), "RENT")
    shopping_members = {a.category for a in rollup.topic("shopping").members}
    assert {"household_goods", "furhwr", "happl"} <= shopping_members
    with pytest.raises(KeyError):
        rollup.topic("household_goods")


# --------------------------------------------------------------------------
# Roll-up losslessness
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tenure", ["RENT", "OWN"])
def test_atom_set_covers_all_43(tenure):
    rollup = build_display_rollup(_synthetic_analyze(), tenure)
    assert set(rollup.atoms) == set(AGGREGATED_CATEGORY_CODES)
    assert len(rollup.atoms) == 46


@pytest.mark.parametrize("tenure", ["RENT", "OWN"])
def test_topic_total_equals_sum_of_children(tenure):
    rollup = build_display_rollup(_synthetic_analyze(), tenure)
    for t in rollup.topics:
        members_sum = sum(a.value for a in t.members)
        if t.is_spending:
            # predicted + pinned == sum of all flow members (no balance in a
            # spending topic, so members_sum is purely flow).
            assert t.predicted_total + t.pinned_total == pytest.approx(members_sum)
        else:
            # net_worth: nothing summed into spend.
            assert t.predicted_total == 0.0
            assert t.pinned_total == 0.0


@pytest.mark.parametrize("tenure", ["RENT", "OWN"])
def test_flat_total_reconciles(tenure):
    """Σ spending predicted + pinned housing == flat model-predicted spend."""
    analyze = _synthetic_analyze()
    rollup = build_display_rollup(analyze, tenure)

    flat = sum(
        analyze["distributions"][c]["feasibility_adjusted"]
        for c in AGGREGATED_FLOW_CATEGORIES
    )
    assert rollup.flat_total == pytest.approx(flat)
    assert (
        rollup.spending_predicted_total + rollup.pinned_housing_total
        == pytest.approx(flat)
    )


@pytest.mark.parametrize("tenure", ["RENT", "OWN"])
def test_balance_atoms_excluded_from_spending(tenure):
    rollup = build_display_rollup(_synthetic_analyze(), tenure)
    # No balance atom contributes to any spending total.
    for c in AGGREGATED_BALANCE_CATEGORIES:
        assert rollup.atoms[c].is_balance is True
    nw = rollup.topic(TOPIC_NET_WORTH)
    assert nw.is_spending is False
    assert {a.category for a in nw.members} == set(AGGREGATED_BALANCE_CATEGORIES)


def test_pinned_housing_flagged_and_separated():
    # RENT: rntval is the pinned input; mortgage lines are not.
    rollup = build_display_rollup(
        _synthetic_analyze({"rntval": 24000.0}), "RENT"
    )
    assert rollup.atoms["rntval"].is_pinned is True
    assert rollup.atoms["mrtgip"].is_pinned is False
    housing = rollup.topic("housing")
    assert housing.pinned_total == pytest.approx(24000.0)
    # The pin is NOT blurred into the predicted total.
    assert rollup.atoms["rntval"].value not in {
        a.value for a in housing.members if not a.is_pinned
    } or housing.predicted_total != pytest.approx(housing.total)

    # OWN: mrtgip + mrtgpp are the pinned input; rntval is not.
    rollup_own = build_display_rollup(
        _synthetic_analyze({"mrtgip": 14000.0, "mrtgpp": 6000.0}), "OWN"
    )
    assert rollup_own.atoms["mrtgip"].is_pinned is True
    assert rollup_own.atoms["mrtgpp"].is_pinned is True
    assert rollup_own.atoms["rntval"].is_pinned is False
    assert rollup_own.topic("housing").pinned_total == pytest.approx(20000.0)


def test_pinned_sets_match_locked_decision_5():
    assert PINNED_HOUSING_BY_TENURE["RENT"] == frozenset({"rntval"})
    assert PINNED_HOUSING_BY_TENURE["OWN"] == frozenset({"mrtgip", "mrtgpp"})


def test_recurring_aggregate_carries_episodic_subcomponents():
    rollup = build_display_rollup(_synthetic_analyze(), "RENT")
    # Build 1 de-merged transportation's vehnew/vehusd (and household_goods'
    # furhwr/happl) into their own episodic lines, so those aggregates no longer
    # carry them as drill-down subcomponents.
    assert rollup.atoms["transportation"].episodic_subcomponents == ()
    assert rollup.atoms["household_goods"].episodic_subcomponents == ()
    # Build-3 de-merged recrp to its own episodic line, so entertainment carries
    # only oeprd as a drill-down subcomponent.
    assert rollup.atoms["entertainment"].episodic_subcomponents == (
        "oeprd",
    )
    # The de-merged components are now their own episodic atoms.
    assert rollup.atoms["vehnew"].cadence == "episodic"
    assert rollup.atoms["furhwr"].cadence == "episodic"
    # Build-3 de-merged recreation/electronics/jewelry durables.
    assert rollup.atoms["recrp"].cadence == "episodic"
    assert rollup.atoms["eltrnp"].cadence == "episodic"
    assert rollup.atoms["jwlbg"].cadence == "episodic"
    # A plain retained category has none.
    assert rollup.atoms["eathome"].episodic_subcomponents == ()


def test_build1_episodic_capital_demerged_and_annual_only():
    """Build 1: vehnew/vehusd (ex-transportation) and furhwr/happl
    (ex-household_goods) are de-merged retained episodic lines, surfaced
    annual-only — never on the monthly/paycheck surface, and no longer summed
    into their former aggregates."""
    from shared.constants.categories import AGG_GROUPS, CATEGORY_CADENCE

    for c in ("vehnew", "vehusd"):
        assert c not in AGG_GROUPS["transportation"]
    for c in ("furhwr", "happl"):
        assert c not in AGG_GROUPS["household_goods"]
    for c in ("vehnew", "vehusd", "furhwr", "happl"):
        assert c in AGGREGATED_CATEGORY_CODES
        assert CATEGORY_CADENCE[c] == "episodic"

    rollup = build_display_rollup(_synthetic_analyze(), "RENT")
    pay = surface_atom_codes(rollup, "paycheck")
    ann = surface_atom_codes(rollup, "annual")
    for c in ("vehnew", "vehusd", "furhwr", "happl"):
        assert c not in pay, f"{c} (episodic capital) must not be on paycheck"
        assert c in ann, f"{c} must be on the annual surface"
    # The recurring backbones stay on the monthly/paycheck surface.
    assert "transportation" in pay
    assert "household_goods" in pay


def test_rejects_disaggregated_output():
    # 55-cat output is missing the aggregate keys -> rejected.
    bad = {"distributions": {"eathome": {"feasibility_adjusted": 1.0}}}
    with pytest.raises(ValueError, match="aggregated"):
        build_display_rollup(bad, "RENT")


# --------------------------------------------------------------------------
# Cadence surface filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tenure", ["RENT", "OWN"])
def test_paycheck_is_strict_subset_of_annual(tenure):
    rollup = build_display_rollup(_synthetic_analyze(), tenure)
    paycheck = surface_atom_codes(rollup, "paycheck")
    annual = surface_atom_codes(rollup, "annual")
    assert paycheck < annual  # strict subset
    # annual == all flow cats; paycheck == recurring flow cats.
    assert annual == set(AGGREGATED_FLOW_CATEGORIES)
    assert paycheck == set(RECURRING_CATEGORIES)


@pytest.mark.parametrize("tenure", ["RENT", "OWN"])
def test_balance_on_neither_surface(tenure):
    rollup = build_display_rollup(_synthetic_analyze(), tenure)
    for surface in ("paycheck", "annual"):
        codes = surface_atom_codes(rollup, surface)
        assert codes.isdisjoint(AGGREGATED_BALANCE_CATEGORIES)


def test_surface_projection_sums_only_surface_members():
    rollup = build_display_rollup(_synthetic_analyze(), "RENT")
    paycheck = {t.topic: t for t in project_surface(rollup, "paycheck")}
    annual = {t.topic: t for t in project_surface(rollup, "annual")}

    # Post 2026-06 fold: shopping mixes the recurring household_goods backbone
    # (on the paycheck surface) with its episodic members (shopping/eltrnp/jwlbg
    # apparel+electronics+jewelry, furhwr/happl durables — annual-only). So the
    # paycheck total is the recurring backbone only and the annual total exceeds
    # it by the episodic members. (Before the fold shopping was purely episodic
    # and absent from the paycheck surface.)
    assert paycheck["shopping"].predicted_total > 0.0
    assert annual["shopping"].predicted_total > paycheck["shopping"].predicted_total

    # entertainment is a recurring aggregate -> present on both. Build-3 added the
    # de-merged recrp (episodic recreation gear) to the entertainment TOPIC, which
    # is annual-only, so the annual topic total now exceeds the paycheck total.
    assert paycheck["entertainment"].predicted_total > 0.0
    assert (annual["entertainment"].predicted_total
            >= paycheck["entertainment"].predicted_total)

    # net_worth is never a spending surface topic.
    assert TOPIC_NET_WORTH not in paycheck
    assert TOPIC_NET_WORTH not in annual


def test_annual_surface_total_equals_flat_total():
    """Summing every spending topic on the annual surface == flat total."""
    rollup = build_display_rollup(_synthetic_analyze(), "RENT")
    annual = project_surface(rollup, "annual")
    total = sum(t.predicted_total + t.pinned_total for t in annual)
    assert total == pytest.approx(rollup.flat_total)
