from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Sequence

from dotenv import load_dotenv

from db.supabase_client import create_pool

load_dotenv()

VECTOR_VERSION = "v1"

CUISINE_VOCABULARY: list[str] = [
    "african",
    "american",
    "asian",
    "bakery",
    "barbecue",
    "brazilian",
    "breakfast",
    "british",
    "bubble_tea",
    "burgers",
    "cajun",
    "caribbean",
    "chicken",
    "chinese",
    "coffee",
    "deli",
    "dessert",
    "ethiopian",
    "fast_food",
    "french",
    "greek",
    "halal",
    "hawaiian",
    "healthy",
    "ice_cream",
    "indian",
    "italian",
    "japanese",
    "korean",
    "latin_american",
    "lebanese",
    "mediterranean",
    "mexican",
    "middle_eastern",
    "moroccan",
    "noodles",
    "pakistani",
    "persian",
    "peruvian",
    "pizza",
    "ramen",
    "seafood",
    "smoothies",
    "soul_food",
    "southern",
    "spanish",
    "steak",
    "sushi",
    "tacos",
    "thai",
    "turkish",
    "vegan",
    "vegetarian",
    "vietnamese",
    "wings",
]

CUISINE_INDEX: dict[str, int] = {tag: i for i, tag in enumerate(CUISINE_VOCABULARY)}

NUM_CUISINE_DIMS = len(CUISINE_VOCABULARY)
NUM_SCALAR_DIMS = 9  # price, rating, spice, 5 dietary flags, data_quality
VECTOR_DIM = NUM_CUISINE_DIMS + NUM_SCALAR_DIMS

DIETARY_TYPES = ["halal", "vegan", "kosher", "vegetarian", "gluten_free"]

BATCH_SIZE = 500


def encode_restaurant_vector(
    *,
    cuisine_tags: list[str],
    price_level: int | None,
    rating_score: float | None,
    spice_level: float | None,
    dietary_flags: dict[str, bool],
    data_quality_score: float | None,
) -> list[float]:
    vector = [0.0] * VECTOR_DIM

    for tag in cuisine_tags:
        normalized = tag.lower().replace(" ", "_").replace("-", "_")
        idx = CUISINE_INDEX.get(normalized)
        if idx is not None:
            vector[idx] = 1.0

    offset = NUM_CUISINE_DIMS
    vector[offset] = (price_level / 4.0) if price_level is not None else 0.0
    vector[offset + 1] = (rating_score / 5.0) if rating_score is not None else 0.0
    vector[offset + 2] = (spice_level / 5.0) if spice_level is not None else 0.0

    for i, dtype in enumerate(DIETARY_TYPES):
        vector[offset + 3 + i] = 1.0 if dietary_flags.get(dtype, False) else 0.0

    vector[offset + 8] = data_quality_score if data_quality_score is not None else 0.0

    return vector


async def fetch_restaurant_batch(
    connection,
    *,
    limit: int,
    offset: int,
) -> list[dict]:
    rows = await connection.fetch(
        """
        SELECT r.id, r.price_level, r.rating_score, r.data_quality_score,
               COALESCE(
                   (SELECT array_agg(ct.tag) FROM cuisine_tags ct WHERE ct.restaurant_id = r.id),
                   ARRAY[]::text[]
               ) AS tags,
               COALESCE(
                   (SELECT json_object_agg(da.dietary_type, da.value)
                    FROM dietary_attributes da
                    WHERE da.restaurant_id = r.id
                      AND da.value = 'true'
                      AND da.dietary_type IN ('halal', 'vegan', 'kosher', 'vegetarian', 'gluten_free')
                   ),
                   '{}'::json
               ) AS dietary_json
        FROM restaurants r
        ORDER BY r.id
        LIMIT $1 OFFSET $2
        """,
        limit,
        offset,
    )
    results = []
    for row in rows:
        dietary_flags = {}
        if row["dietary_json"]:
            raw = row["dietary_json"]
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            for dtype in DIETARY_TYPES:
                dietary_flags[dtype] = dtype in parsed
        results.append({
            "id": row["id"],
            "price_level": row["price_level"],
            "rating_score": row["rating_score"],
            "data_quality_score": row["data_quality_score"],
            "tags": list(row["tags"]) if row["tags"] else [],
            "dietary_flags": dietary_flags,
        })
    return results


async def upsert_feature_vectors(
    connection,
    rows: list[tuple[uuid.UUID, list[float]]],
) -> int:
    if not rows:
        return 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    params = [
        (rid, str(vec), VECTOR_VERSION, now)
        for rid, vec in rows
    ]
    await connection.executemany(
        """
        INSERT INTO feature_vectors (restaurant_id, vector, vector_version, computed_at)
        VALUES ($1, $2::vector, $3, $4)
        ON CONFLICT (restaurant_id) DO UPDATE SET
            vector = EXCLUDED.vector,
            vector_version = EXCLUDED.vector_version,
            computed_at = EXCLUDED.computed_at
        """,
        params,
    )
    return len(params)


async def generate_vectors(*, limit: int | None = None) -> int:
    pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            if limit is None:
                row = await connection.fetchrow("SELECT count(*) AS cnt FROM restaurants")
                total = row["cnt"]
            else:
                total = limit

            print(f"Generating feature vectors for up to {total} restaurants (dim={VECTOR_DIM})...")

            processed = 0
            offset = 0

            while processed < total:
                batch_limit = min(BATCH_SIZE, total - processed)
                restaurants = await fetch_restaurant_batch(
                    connection, limit=batch_limit, offset=offset,
                )
                if not restaurants:
                    break

                vector_rows: list[tuple[uuid.UUID, list[float]]] = []
                for r in restaurants:
                    vec = encode_restaurant_vector(
                        cuisine_tags=r["tags"],
                        price_level=r["price_level"],
                        rating_score=r["rating_score"],
                        spice_level=None,
                        dietary_flags=r["dietary_flags"],
                        data_quality_score=r["data_quality_score"],
                    )
                    vector_rows.append((r["id"], vec))

                upserted = await upsert_feature_vectors(connection, vector_rows)
                processed += len(restaurants)
                offset += len(restaurants)
                print(f"  Progress: {processed}/{total} restaurants vectorized ({upserted} upserted)")

            print(f"Done. {processed} feature vectors generated (version={VECTOR_VERSION}).")
            return processed
    finally:
        await pool.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate feature vectors for restaurants")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--limit", type=int, help="Number of restaurants to process")
    group.add_argument("--all", action="store_true", help="Process all restaurants")
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    limit = None if args.all else args.limit
    await generate_vectors(limit=limit)


if __name__ == "__main__":
    asyncio.run(main())
