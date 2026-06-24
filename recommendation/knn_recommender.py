from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from db.supabase_client import create_pool
from recommendation.compatibility import compute_dietary_compatibility
from recommendation.vector_generator import (
    DIETARY_TYPES,
    VECTOR_DIM,
    VECTOR_VERSION,
    encode_restaurant_vector,
)

load_dotenv()

KNN_WEIGHT = 0.70
COMPATIBILITY_WEIGHT = 0.30
KNN_OVERSAMPLE = 3


@dataclass(slots=True)
class Recommendation:
    restaurant_id: str
    name: str
    address: str | None
    cuisine_tags: list[str]
    final_score: float
    compatibility_score: float
    similarity_score: float


def build_query_vector(profile: dict[str, Any]) -> list[float]:
    cuisine_prefs = profile.get("cuisine_preferences") or []
    spice_tolerance = profile.get("spice_tolerance")
    price_preference = profile.get("price_preference")
    dietary_reqs = profile.get("dietary_requirements") or []
    dietary_prefs = profile.get("dietary_preferences") or []

    dietary_flags = {}
    for dtype in DIETARY_TYPES:
        if dtype in dietary_reqs or dtype in dietary_prefs:
            dietary_flags[dtype] = True

    return encode_restaurant_vector(
        cuisine_tags=cuisine_prefs,
        price_level=price_preference,
        rating_score=4.0,
        spice_level=float(spice_tolerance) if spice_tolerance is not None else None,
        dietary_flags=dietary_flags,
        data_quality_score=0.5,
    )


async def fetch_taste_profile(connection, user_id: str) -> dict[str, Any]:
    row = await connection.fetchrow(
        """
        SELECT spice_tolerance, price_preference,
               cuisine_preferences, cuisine_dislikes,
               dietary_requirements, dietary_preferences,
               adventure_score
        FROM taste_profiles
        WHERE user_id = $1
        ORDER BY last_updated DESC NULLS LAST
        LIMIT 1
        """,
        user_id,
    )
    if row is None:
        raise ValueError(f"No taste profile found for user {user_id}")
    return {
        "spice_tolerance": row["spice_tolerance"],
        "price_preference": row["price_preference"],
        "cuisine_preferences": list(row["cuisine_preferences"] or []),
        "cuisine_dislikes": list(row["cuisine_dislikes"] or []),
        "dietary_requirements": list(row["dietary_requirements"] or []),
        "dietary_preferences": list(row["dietary_preferences"] or []),
        "adventure_score": row["adventure_score"],
    }


async def knn_search(
    connection,
    query_vector: list[float],
    k: int,
) -> list[dict[str, Any]]:
    vector_str = str(query_vector)
    rows = await connection.fetch(
        """
        SELECT fv.restaurant_id,
               r.name,
               r.address,
               1 - (fv.vector <=> $1::vector) / 2.0 AS similarity,
               COALESCE(
                   (SELECT array_agg(ct.tag) FROM cuisine_tags ct WHERE ct.restaurant_id = fv.restaurant_id),
                   ARRAY[]::text[]
               ) AS cuisine_tags
        FROM feature_vectors fv
        JOIN restaurants r ON r.id = fv.restaurant_id
        WHERE fv.vector_version = $2
        ORDER BY fv.vector <=> $1::vector
        LIMIT $3
        """,
        vector_str,
        VECTOR_VERSION,
        k,
    )
    return [dict(row) for row in rows]


async def fetch_restaurant_dietary(
    connection,
    restaurant_ids: list,
) -> dict[str, list[dict[str, Any]]]:
    if not restaurant_ids:
        return {}
    rows = await connection.fetch(
        """
        SELECT restaurant_id, dietary_type, value, confidence_tier
        FROM dietary_attributes
        WHERE restaurant_id = ANY($1::uuid[])
        """,
        restaurant_ids,
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rid = str(row["restaurant_id"])
        if rid not in result:
            result[rid] = []
        result[rid].append({
            "dietary_type": row["dietary_type"],
            "value": row["value"],
            "confidence_tier": row["confidence_tier"],
        })
    return result


async def recommend(
    *,
    profile: dict[str, Any],
    limit: int = 10,
    pool=None,
) -> list[Recommendation]:
    owns_pool = pool is None
    if owns_pool:
        pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            query_vector = build_query_vector(profile)
            k = limit * KNN_OVERSAMPLE
            candidates = await knn_search(connection, query_vector, k)

            if not candidates:
                return []

            restaurant_ids = [c["restaurant_id"] for c in candidates]
            dietary_map = await fetch_restaurant_dietary(connection, restaurant_ids)

            dietary_reqs = profile.get("dietary_requirements") or []
            dietary_prefs = profile.get("dietary_preferences") or []

            scored: list[Recommendation] = []
            for c in candidates:
                rid = str(c["restaurant_id"])
                restaurant_dietary = dietary_map.get(rid, [])

                compat = compute_dietary_compatibility(
                    dietary_requirements=dietary_reqs,
                    dietary_preferences=dietary_prefs,
                    restaurant_dietary=restaurant_dietary,
                )

                if compat == 0.0 and dietary_reqs:
                    continue

                similarity = c["similarity"]
                final = (KNN_WEIGHT * similarity) + (COMPATIBILITY_WEIGHT * compat)

                scored.append(Recommendation(
                    restaurant_id=rid,
                    name=c["name"],
                    address=c["address"],
                    cuisine_tags=list(c["cuisine_tags"] or []),
                    final_score=round(final, 4),
                    compatibility_score=round(compat, 4),
                    similarity_score=round(similarity, 4),
                ))

            scored.sort(key=lambda r: r.final_score, reverse=True)
            return scored[:limit]
    finally:
        if owns_pool:
            await pool.close()


def format_results(results: list[Recommendation]) -> None:
    if not results:
        print("No recommendations found.")
        return
    print(f"\nTop {len(results)} Recommendations:")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        cuisines = ", ".join(r.cuisine_tags[:5]) if r.cuisine_tags else "N/A"
        print(
            f"{i:>2}. {r.name}"
            f"\n    Address: {r.address or 'N/A'}"
            f"\n    Cuisines: {cuisines}"
            f"\n    Final: {r.final_score:.4f}  "
            f"(similarity={r.similarity_score:.4f}, compatibility={r.compatibility_score:.4f})"
        )
    print("-" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KNN restaurant recommender")
    parser.add_argument("--user-id", type=str, help="User UUID to load taste profile from DB")
    parser.add_argument("--profile", type=str, help="JSON taste profile string")
    parser.add_argument("--limit", type=int, default=10, help="Number of recommendations")
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.user_id and not args.profile:
        parser.error("Either --user-id or --profile is required")

    pool = await create_pool()
    try:
        if args.user_id:
            async with pool.acquire() as connection:
                profile = await fetch_taste_profile(connection, args.user_id)
        else:
            profile = json.loads(args.profile)

        results = await recommend(profile=profile, limit=args.limit, pool=pool)
        format_results(results)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
