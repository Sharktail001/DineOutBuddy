from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from db.supabase_client import create_pool
from recommendation.compatibility import compute_dietary_compatibility
from recommendation.knn_recommender import (
    KNN_WEIGHT,
    COMPATIBILITY_WEIGHT,
    Recommendation,
    fetch_restaurant_dietary,
    recommend,
)

load_dotenv()

GROUP_KNN_OVERSAMPLE = 3


@dataclass(slots=True)
class GroupRecommendation:
    restaurant_id: str
    name: str
    group_score: float
    per_person_scores: dict[int, dict[str, float]]


async def group_recommend(
    *,
    profiles: list[dict[str, Any]],
    limit: int = 10,
    pool=None,
) -> list[GroupRecommendation]:
    owns_pool = pool is None
    if owns_pool:
        pool = await create_pool()
    try:
        per_person_limit = limit * GROUP_KNN_OVERSAMPLE
        all_recommendations: list[list[Recommendation]] = []

        for profile in profiles:
            recs = await recommend(
                profile=profile,
                limit=per_person_limit,
                pool=pool,
            )
            all_recommendations.append(recs)

        restaurant_scores: dict[str, list[dict[str, float]]] = {}
        restaurant_names: dict[str, str] = {}

        for person_idx, recs in enumerate(all_recommendations):
            for r in recs:
                if r.restaurant_id not in restaurant_scores:
                    restaurant_scores[r.restaurant_id] = [{} for _ in profiles]
                    restaurant_names[r.restaurant_id] = r.name
                restaurant_scores[r.restaurant_id][person_idx] = {
                    "final_score": r.final_score,
                    "compatibility_score": r.compatibility_score,
                    "similarity_score": r.similarity_score,
                }

        all_rids = list(restaurant_scores.keys())
        async with pool.acquire() as connection:
            dietary_map = await fetch_restaurant_dietary(
                connection, [uuid.UUID(rid) for rid in all_rids],
            )

        for rid, person_scores_list in restaurant_scores.items():
            for person_idx, scores in enumerate(person_scores_list):
                if scores:
                    continue
                profile = profiles[person_idx]
                dietary_reqs = profile.get("dietary_requirements") or []
                dietary_prefs = profile.get("dietary_preferences") or []
                restaurant_dietary = dietary_map.get(rid, [])
                compat = compute_dietary_compatibility(
                    dietary_requirements=dietary_reqs,
                    dietary_preferences=dietary_prefs,
                    restaurant_dietary=restaurant_dietary,
                )
                person_scores_list[person_idx] = {
                    "final_score": COMPATIBILITY_WEIGHT * compat,
                    "compatibility_score": compat,
                    "similarity_score": 0.0,
                }

        viable: list[GroupRecommendation] = []
        for rid, person_scores_list in restaurant_scores.items():
            skip = False
            for person_idx, scores in enumerate(person_scores_list):
                # recommend() already filters 0.0-compat restaurants from its
                # results, but we recheck here for restaurants that were
                # back-filled via direct compatibility computation above.
                if scores.get("compatibility_score", 0.0) == 0.0:
                    dietary_reqs = (profiles[person_idx].get("dietary_requirements") or [])
                    if dietary_reqs:
                        skip = True
                        break
            if skip:
                continue

            total = sum(s["final_score"] for s in person_scores_list)
            group_score = total / len(profiles)

            per_person = {
                i: s for i, s in enumerate(person_scores_list)
            }

            viable.append(GroupRecommendation(
                restaurant_id=rid,
                name=restaurant_names[rid],
                group_score=round(group_score, 4),
                per_person_scores=per_person,
            ))

        viable.sort(key=lambda g: g.group_score, reverse=True)
        return viable[:limit]
    finally:
        if owns_pool:
            await pool.close()


def format_group_results(
    results: list[GroupRecommendation],
    num_people: int,
) -> None:
    if not results:
        print("No group recommendations found.")
        return
    print(f"\nTop {len(results)} Group Recommendations ({num_people} people):")
    print("-" * 80)
    for i, g in enumerate(results, 1):
        print(f"{i:>2}. {g.name}")
        print(f"    Group Score: {g.group_score:.4f}")
        for person_idx in range(num_people):
            scores = g.per_person_scores.get(person_idx, {})
            if scores:
                print(
                    f"    Person {person_idx + 1}: "
                    f"final={scores['final_score']:.4f}, "
                    f"compat={scores['compatibility_score']:.4f}, "
                    f"sim={scores['similarity_score']:.4f}"
                )
            else:
                print(f"    Person {person_idx + 1}: not scored")
    print("-" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Group restaurant recommender")
    parser.add_argument(
        "--profiles", type=str, required=True,
        help="JSON array of taste profile objects",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of recommendations")
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    profiles = json.loads(args.profiles)
    if not isinstance(profiles, list) or len(profiles) < 2:
        parser.error("--profiles must be a JSON array with at least 2 profiles")

    results = await group_recommend(profiles=profiles, limit=args.limit)
    format_group_results(results, len(profiles))


if __name__ == "__main__":
    asyncio.run(main())
