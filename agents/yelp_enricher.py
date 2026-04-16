from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv

from agents.csv_importer import chunked, compute_data_quality_score, normalize_tag
from db.rate_limiter import RateLimitResult, check_rate_limit
from db.supabase_client import create_pool

load_dotenv()

YELP_API_BASE_URL = "https://api.yelp.com/v3"
SEARCH_ENDPOINT = f"{YELP_API_BASE_URL}/businesses/search"
DETAILS_ENDPOINT_TEMPLATE = f"{YELP_API_BASE_URL}/businesses/{{yelp_id}}"
DIETARY_TYPES = {
    "halal": "halal",
    "vegan": "vegan",
    "vegetarian": "vegetarian",
    "kosher": "kosher",
    "gluten_free": "gluten_free",
}


class RateLimitPaused(Exception):
    pass


@dataclass(slots=True)
class YelpEnrichmentStats:
    restaurants_updated: int = 0
    cuisine_tags_added: int = 0
    dietary_attributes_added: int = 0
    api_calls: int = 0
    restaurants_processed: int = 0
    low_confidence_rejections: int = 0


def get_yelp_api_key() -> str:
    api_key = os.getenv("YELP_API_KEY")
    if not api_key:
        raise RuntimeError("YELP_API_KEY is required to run the Yelp enricher.")
    return api_key


def map_yelp_price_level(price_text: str | None) -> int | None:
    if not price_text:
        return None
    mapping = {
        "$": 1,
        "$$": 2,
        "$$$": 3,
        "$$$$": 4,
    }
    return mapping.get(price_text.strip())


def name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, (left or "").lower(), (right or "").lower()).ratio()


def extract_category_tags(categories: Sequence[dict[str, Any]] | None) -> list[str]:
    if not categories:
        return []
    tags: list[str] = []
    for category in categories:
        alias = category.get("alias") or ""
        title = category.get("title") or ""
        normalized = normalize_tag(alias or title)
        if normalized:
            tags.append(normalized)
    return list(dict.fromkeys(tags))


def extract_dietary_types(categories: Sequence[dict[str, Any]] | None) -> list[str]:
    if not categories:
        return []
    found: list[str] = []
    for category in categories:
        alias = (category.get("alias") or "").lower()
        for needle, dietary_type in DIETARY_TYPES.items():
            if needle in alias and dietary_type not in found:
                found.append(dietary_type)
    return found


class YelpEnricher:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        pool=None,
        client: httpx.AsyncClient | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        self.api_key = api_key or get_yelp_api_key()
        self.pool = pool
        self.client = client
        self.sleep = sleep
        self._owns_pool = pool is None
        self._owns_client = client is None

    async def __aenter__(self) -> "YelpEnricher":
        if self.pool is None:
            self.pool = await create_pool()
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()
        if self._owns_pool and self.pool is not None:
            await self.pool.close()

    async def log_api_call(
        self,
        connection,
        *,
        query: str,
        status: str,
        records_added: int = 0,
        records_updated: int = 0,
    ) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await connection.execute(
            """
            INSERT INTO data_fetch_log (
                id, source, query, status, fetched_at, records_added, records_updated
            ) VALUES ($1, 'yelp', $2, $3, $4, $5, $6)
            """,
            uuid.uuid4(),
            query,
            status,
            now,
            records_added,
            records_updated,
        )

    async def check_yelp_rate_limit(self, connection) -> RateLimitResult:
        result = await check_rate_limit(connection, "yelp")
        if result.status == "warn":
            print(f"Warning: yelp usage is at {result.calls_today}/{result.daily_limit} today.")
        if result.status == "pause":
            await self.log_api_call(connection, query="rate_limit_pause", status="failed")
            raise RateLimitPaused("yelp rate limit has reached the pause threshold.")
        return result

    async def request_json(
        self,
        connection,
        *,
        method: str,
        url: str,
        query_label: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(1, 4):
            await self.check_yelp_rate_limit(connection)
            try:
                response = await self.client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                )
            except httpx.HTTPError:
                await self.log_api_call(connection, query=f"{query_label}|attempt={attempt}", status="failed")
                if attempt == 3:
                    raise
                await self.sleep(2 ** (attempt - 1))
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                await self.log_api_call(
                    connection,
                    query=f"{query_label}|attempt={attempt}|status={response.status_code}",
                    status="failed",
                )
                if attempt == 3:
                    response.raise_for_status()
                await self.sleep(2 ** (attempt - 1))
                continue

            response.raise_for_status()
            payload = response.json()
            added = len(payload.get("businesses", [])) if isinstance(payload, dict) else 0
            await self.log_api_call(
                connection,
                query=f"{query_label}|attempt={attempt}",
                status="success",
                records_added=added,
            )
            return payload

        raise RuntimeError("Yelp request failed after maximum retries.")

    async def fetch_targets(self, connection, *, city: str | None, limit: int) -> list[dict[str, Any]]:
        if city:
            city_name = city.split(",")[0].strip()
            rows = await connection.fetch(
                """
                SELECT id, name, address, city, price_level, rating_score, rating_count,
                       phone, website, yelp_id, yelp_url, google_place_id
                FROM restaurants
                WHERE yelp_id IS NULL
                  AND name IS NOT NULL
                  AND address IS NOT NULL
                  AND (city ILIKE $1 OR address ILIKE $2)
                ORDER BY updated_at NULLS FIRST, created_at NULLS FIRST
                LIMIT $3
                """,
                city_name,
                f"%{city_name}%",
                limit,
            )
        else:
            rows = await connection.fetch(
                """
                SELECT id, name, address, city, price_level, rating_score, rating_count,
                       phone, website, yelp_id, yelp_url, google_place_id
                FROM restaurants
                WHERE yelp_id IS NULL
                  AND name IS NOT NULL
                  AND address IS NOT NULL
                ORDER BY updated_at NULLS FIRST, created_at NULLS FIRST
                LIMIT $1
                """,
                limit,
            )
        return [dict(row) for row in rows]

    async def fetch_menu_presence(self, connection, restaurant_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        if not restaurant_ids:
            return set()
        rows = await connection.fetch(
            """
            SELECT DISTINCT restaurant_id
            FROM menu_items
            WHERE restaurant_id = ANY($1::uuid[])
            """,
            list(restaurant_ids),
        )
        return {row["restaurant_id"] for row in rows}

    async def fetch_existing_dietary(self, connection, restaurant_ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, set[str]]:
        if not restaurant_ids:
            return {}
        rows = await connection.fetch(
            """
            SELECT restaurant_id, dietary_type
            FROM dietary_attributes
            WHERE restaurant_id = ANY($1::uuid[])
            """,
            list(restaurant_ids),
        )
        result: dict[uuid.UUID, set[str]] = {}
        for row in rows:
            result.setdefault(row["restaurant_id"], set()).add(row["dietary_type"])
        return result

    async def fetch_dietary_presence(
        self,
        connection,
        restaurant_ids: Sequence[uuid.UUID],
    ) -> dict[uuid.UUID, int]:
        if not restaurant_ids:
            return {}
        rows = await connection.fetch(
            """
            SELECT restaurant_id, MIN(confidence_tier) AS best_tier
            FROM dietary_attributes
            WHERE restaurant_id = ANY($1::uuid[])
            GROUP BY restaurant_id
            """,
            list(restaurant_ids),
        )
        return {row["restaurant_id"]: row["best_tier"] for row in rows}

    async def update_restaurants(self, connection, restaurant_rows: list[tuple], stats: YelpEnrichmentStats) -> None:
        if not restaurant_rows:
            return
        query = """
            UPDATE restaurants
            SET yelp_id = COALESCE(restaurants.yelp_id, $2),
                rating_score = COALESCE(restaurants.rating_score, $3),
                rating_count = COALESCE(restaurants.rating_count, $4),
                price_level = COALESCE(restaurants.price_level, $5),
                phone = COALESCE(restaurants.phone, $6),
                website = COALESCE(restaurants.website, $7),
                yelp_url = COALESCE(restaurants.yelp_url, $8),
                data_quality_score = GREATEST(restaurants.data_quality_score, $9),
                last_verified_at = $10,
                updated_at = $11
            WHERE id = $1
        """
        for batch in chunked(restaurant_rows):
            await connection.executemany(query, batch)
            stats.restaurants_updated += len(batch)

    async def upsert_cuisine_tags(self, connection, cuisine_tag_rows: list[tuple], stats: YelpEnrichmentStats) -> None:
        if not cuisine_tag_rows:
            return
        existing_rows = {
            (row["restaurant_id"], row["tag"], row["source"])
            for row in await connection.fetch(
                """
                SELECT restaurant_id, tag, source
                FROM cuisine_tags
                WHERE restaurant_id = ANY($1::uuid[])
                """,
                sorted({row[0] for row in cuisine_tag_rows}),
            )
        }
        stats.cuisine_tags_added += sum(
            1 for row in cuisine_tag_rows if (row[0], row[1], row[2]) not in existing_rows
        )
        query = """
            INSERT INTO cuisine_tags (restaurant_id, tag, source, derived_from_menu)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
        """
        for batch in chunked(cuisine_tag_rows):
            await connection.executemany(query, batch)

    async def insert_dietary_attributes(
        self,
        connection,
        dietary_rows: list[tuple],
        stats: YelpEnrichmentStats,
    ) -> None:
        if not dietary_rows:
            return
        query = """
            INSERT INTO dietary_attributes (
                id, restaurant_id, dietary_type, value, confidence_tier,
                source, source_url, notes, fetched_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        for batch in chunked(dietary_rows):
            await connection.executemany(query, batch)
            stats.dietary_attributes_added += len(batch)

    async def persist_enrichment(
        self,
        connection,
        *,
        restaurant: dict[str, Any],
        business_details: dict[str, Any],
        stats: YelpEnrichmentStats,
    ) -> None:
        restaurant_id = restaurant["id"]
        categories = business_details.get("categories") or []
        category_tags = extract_category_tags(categories)
        dietary_types = extract_dietary_types(categories)
        menu_presence = await self.fetch_menu_presence(connection, [restaurant_id])
        existing_dietary = await self.fetch_existing_dietary(connection, [restaurant_id])
        dietary_presence = await self.fetch_dietary_presence(connection, [restaurant_id])

        phone = business_details.get("phone") or restaurant.get("phone")
        yelp_url = business_details.get("url")
        price_level = restaurant.get("price_level")
        if price_level is None:
            price_level = map_yelp_price_level(business_details.get("price"))

        best_existing_tier = dietary_presence.get(restaurant_id)
        incoming_has_dietary = bool(dietary_types) or best_existing_tier is not None
        high_confidence = best_existing_tier is not None and best_existing_tier <= 2
        score = compute_data_quality_score(
            has_menu_items=restaurant_id in menu_presence,
            has_google_place_id=bool(restaurant.get("google_place_id")),
            has_contact_or_website=bool(phone or restaurant.get("website")),
            has_dietary_attribute=incoming_has_dietary,
            has_high_confidence_dietary=high_confidence,
            verified_recently=True,
        )

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.update_restaurants(
            connection,
            [
                (
                    restaurant_id,
                    business_details.get("id"),
                    business_details.get("rating"),
                    business_details.get("review_count"),
                    price_level,
                    phone,
                    None,     # website — not sourced from Yelp
                    yelp_url,
                    score,
                    now,
                    now,
                )
            ],
            stats,
        )

        cuisine_rows = [(restaurant_id, tag, "yelp", False) for tag in category_tags]
        await self.upsert_cuisine_tags(connection, cuisine_rows, stats)

        dietary_rows: list[tuple] = []
        existing_types = existing_dietary.get(restaurant_id, set())
        for dietary_type in dietary_types:
            if dietary_type in existing_types:
                continue
            dietary_rows.append(
                (
                    uuid.uuid4(),
                    restaurant_id,
                    dietary_type,
                    "true",
                    3,
                    "yelp",
                    business_details.get("url"),
                    "Derived from Yelp categories",
                    now,
                )
            )
        await self.insert_dietary_attributes(connection, dietary_rows, stats)

    async def enrich(self, *, city: str | None = None, limit: int = 50) -> YelpEnrichmentStats:
        stats = YelpEnrichmentStats()
        async with self.pool.acquire() as connection:
            targets = await self.fetch_targets(connection, city=city, limit=limit)
            for index, restaurant in enumerate(targets, start=1):
                search_payload = await self.request_json(
                    connection,
                    method="GET",
                    url=SEARCH_ENDPOINT,
                    query_label=f"search|restaurant_id={restaurant['id']}",
                    params={
                        "term": restaurant["name"],
                        "location": restaurant["address"],
                        "limit": 1,
                    },
                )
                stats.api_calls += 1
                businesses = search_payload.get("businesses", [])
                if not businesses:
                    stats.restaurants_processed += 1
                    if index % 10 == 0:
                        print(f"Processed {index} restaurants...")
                    continue

                match = businesses[0]
                similarity = name_similarity(restaurant["name"], match.get("name", ""))
                if similarity < 0.80:
                    stats.low_confidence_rejections += 1
                    await self.log_api_call(
                        connection,
                        query=(
                            f"low_confidence_reject|restaurant_id={restaurant['id']}|"
                            f"candidate={match.get('id')}|similarity={similarity:.3f}"
                        ),
                        status="partial",
                    )
                    stats.restaurants_processed += 1
                    if index % 10 == 0:
                        print(f"Processed {index} restaurants...")
                    continue

                details = await self.request_json(
                    connection,
                    method="GET",
                    url=DETAILS_ENDPOINT_TEMPLATE.format(yelp_id=match["id"]),
                    query_label=f"details|restaurant_id={restaurant['id']}|yelp_id={match['id']}",
                )
                stats.api_calls += 1
                await self.persist_enrichment(
                    connection,
                    restaurant=restaurant,
                    business_details=details,
                    stats=stats,
                )
                stats.restaurants_processed += 1
                if index % 10 == 0:
                    print(f"Processed {index} restaurants...")

        return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yelp Fusion enricher")
    parser.add_argument("--city")
    parser.add_argument("--limit", type=int, default=50)
    return parser


async def run_cli(args: argparse.Namespace) -> YelpEnrichmentStats:
    async with YelpEnricher() as enricher:
        return await enricher.enrich(city=args.city, limit=args.limit)


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        stats = await run_cli(args)
    except RateLimitPaused as exc:
        print(str(exc))
        return

    print(
        "Yelp enrichment complete:",
        {
            "restaurants_updated": stats.restaurants_updated,
            "cuisine_tags_added": stats.cuisine_tags_added,
            "dietary_attributes_added": stats.dietary_attributes_added,
            "api_calls": stats.api_calls,
            "restaurants_processed": stats.restaurants_processed,
            "low_confidence_rejections": stats.low_confidence_rejections,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
