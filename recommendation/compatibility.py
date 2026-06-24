from __future__ import annotations

from typing import Any


DIETARY_TYPES = ["halal", "vegan", "kosher", "vegetarian", "gluten_free"]

CONFIDENCE_FULL = {1, 2}
CONFIDENCE_PARTIAL = {3, 4}
CONFIDENCE_UNKNOWN_OR_WORSE = {5, 6}


def compute_dietary_compatibility(
    *,
    dietary_requirements: list[str],
    dietary_preferences: list[str],
    restaurant_dietary: list[dict[str, Any]],
) -> float:
    """Compute dietary compatibility score (0.0-1.0) between a user profile and a restaurant.

    restaurant_dietary: list of dicts with keys dietary_type, value, confidence_tier.
    """
    attrs_by_type: dict[str, dict[str, Any]] = {}
    for attr in restaurant_dietary:
        dtype = attr["dietary_type"]
        if dtype not in attrs_by_type:
            attrs_by_type[dtype] = attr
        else:
            existing_tier = attrs_by_type[dtype].get("confidence_tier") or 6
            new_tier = attr.get("confidence_tier") or 6
            if new_tier < existing_tier:
                attrs_by_type[dtype] = attr

    if not dietary_requirements and not dietary_preferences:
        return 1.0

    score = 0.0
    total_weight = 0.0

    for req in dietary_requirements:
        normalized = req.lower().replace(" ", "_").replace("-", "_")
        total_weight += 1.0
        attr = attrs_by_type.get(normalized)

        if attr is None:
            return 0.0

        value = (attr.get("value") or "").lower()
        tier = attr.get("confidence_tier") or 6

        if value != "true" or tier in CONFIDENCE_UNKNOWN_OR_WORSE:
            return 0.0

        if tier in CONFIDENCE_FULL:
            score += 1.0
        elif tier in CONFIDENCE_PARTIAL:
            score += 0.6

    for pref in dietary_preferences:
        normalized = pref.lower().replace(" ", "_").replace("-", "_")
        weight = 0.5
        total_weight += weight
        attr = attrs_by_type.get(normalized)

        if attr is None:
            continue

        value = (attr.get("value") or "").lower()
        tier = attr.get("confidence_tier") or 6

        if value == "true":
            if tier in CONFIDENCE_FULL:
                score += weight * 1.0
            elif tier in CONFIDENCE_PARTIAL:
                score += weight * 0.6

    if total_weight == 0.0:
        return 1.0

    return min(1.0, max(0.0, score / total_weight))
