"""PUMA -> state crosswalk, census divisions, and matching fallback hierarchy.

Stub only — populate PUMA_TO_STATE from the ACS 2022 PUMS dictionary
(or fusionData/geo-processed/) once the pipeline produces the crosswalk.
"""

PUMA_TO_STATE: dict[str, str] = {}

CENSUS_DIVISIONS: dict[str, list[str]] = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
}

# Census Bureau's four super-regions (divisions grouped).
CENSUS_REGIONS: dict[str, list[str]] = {
    "Northeast": ["New England", "Middle Atlantic"],
    "Midwest":   ["East North Central", "West North Central"],
    "South":     ["South Atlantic", "East South Central", "West South Central"],
    "West":      ["Mountain", "Pacific"],
}

# Derived reverse lookups. Used by models.matching.algorithm.select_similar_pumas
# to apply same-state / same-division / same-region tier preferences.
STATE_TO_DIVISION: dict[str, str] = {
    state: division
    for division, states in CENSUS_DIVISIONS.items()
    for state in states
}

STATE_TO_REGION: dict[str, str] = {
    state: region
    for region, divisions in CENSUS_REGIONS.items()
    for division in divisions
    for state in CENSUS_DIVISIONS[division]
}

MATCHING_MIN_POOL: int = 100
MATCHING_BANDWIDTH_NEIGHBOR: int = 150
