from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv

from agents.csv_importer import chunked, compute_data_quality_score, normalize_tag
from db.rate_limiter import RateLimitResult, check_rate_limit
from db.supabase_client import create_pool

load_dotenv()

DISCOVERY_URL = "https://places.googleapis.com/v1/places:searchNearby"
DETAILS_URL_TEMPLATE = "https://places.googleapis.com/v1/places/{place_id}"
GOOGLE_PLACE_NAMESPACE = uuid.UUID("e1ddf0a2-52e0-4497-9d1d-d1fd8f2e33a7")
ZIP_CODE_PATTERN = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
DISCOVERY_FIELD_MASK = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.priceLevel",
        "places.rating",
        "places.userRatingCount",
        "places.nationalPhoneNumber",
        "places.websiteUri",
        "places.regularOpeningHours",
        "places.types",
    ]
)
DETAILS_FIELD_MASK = ",".join(
    [
        "id",
        "displayName",
        "formattedAddress",
        "location",
        "priceLevel",
        "rating",
        "userRatingCount",
        "nationalPhoneNumber",
        "websiteUri",
        "regularOpeningHours",
        "types",
    ]
)


class RateLimitPaused(Exception):
    pass


@dataclass(slots=True)
class GooglePlacesStats:
    restaurants_added: int = 0
    restaurants_updated: int = 0
    cuisine_tags_added: int = 0
    api_calls: int = 0
    places_processed: int = 0


def get_google_places_api_key() -> str:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_PLACES_API_KEY is required to run the Google Places agent.")
    return api_key


def map_price_level(price_level: str | None) -> int | None:
    if not price_level:
        return None
    mapping = {
        "PRICE_LEVEL_FREE": 0,
        "PRICE_LEVEL_1": 1,
        "PRICE_LEVEL_2": 2,
        "PRICE_LEVEL_3": 3,
        "PRICE_LEVEL_4": 4,
    }
    return mapping.get(price_level)


def extract_city(address: str | None) -> str | None:
    if not address:
        return None
    parts = [part.strip() for part in address.split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[-2]
    return None


def extract_zip(address: str | None) -> str | None:
    if not address:
        return None
    match = ZIP_CODE_PATTERN.search(address)
    return match.group(1) if match else None


def normalize_types(types: Sequence[str] | None) -> list[str]:
    if not types:
        return []
    tags: list[str] = []
    for value in types:
        normalized = normalize_tag(value)
        if normalized:
            tags.append(normalized)
    return list(dict.fromkeys(tags))


def serialize_hours(hours_payload: dict[str, Any] | None) -> str | None:
    if not hours_payload:
        return None
    return json.dumps(hours_payload, sort_keys=True)


def build_restaurant_id(google_place_id: str) -> uuid.UUID:
    return uuid.uuid5(GOOGLE_PLACE_NAMESPACE, google_place_id)


class GooglePlacesAgent:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        pool=None,
        client: httpx.AsyncClient | None = None,
        sleep=asyncio.sleep,
    ) -> None:
        self.api_key = api_key or get_google_places_api_key()
        self.pool = pool
        self.client = client
        self.sleep = sleep
        self._owns_pool = pool is None
        self._owns_client = client is None

    async def __aenter__(self) -> "GooglePlacesAgent":
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

    async def ensure_restaurant_columns(self, connection) -> None:
        await connection.execute(
            """
            ALTER TABLE restaurants
                ADD COLUMN IF NOT EXISTS website TEXT,
                ADD COLUMN IF NOT EXISTS phone TEXT,
                ADD COLUMN IF NOT EXISTS hours TEXT,
                ADD COLUMN IF NOT EXISTS rating_score DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS rating_count INT
            """
        )

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
            ) VALUES ($1, 'google_places', $2, $3, $4, $5, $6)
            """,
            uuid.uuid4(),
            query,
            status,
            now,
            records_added,
            records_updated,
        )

    async def check_google_rate_limit(self, connection) -> RateLimitResult:
        result = await check_rate_limit(connection, "google_places")
        if result.status == "warn":
            print(
                f"Warning: google_places usage is at {result.calls_today}/{result.daily_limit} today."
            )
        if result.status == "pause":
            await self.log_api_call(
                connection,
                query="rate_limit_pause",
                status="failed",
            )
            raise RateLimitPaused("google_places rate limit has reached the pause threshold.")
        return result

    async def request_json(
        self,
        connection,
        *,
        method: str,
        url: str,
        query_label: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        field_mask: str,
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

        for attempt in range(1, 4):
            await self.check_google_rate_limit(connection)
            try:
                response = await self.client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
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
            places_count = len(payload.get("places", [])) if isinstance(payload, dict) else 0
            await self.log_api_call(
                connection,
                query=f"{query_label}|attempt={attempt}",
                status="success",
                records_added=places_count,
            )
            return payload

        raise RuntimeError("Google Places request failed after maximum retries.")

    async def resolve_city_center(self, connection, city: str) -> tuple[float, float]:
        city_name = city.split(",")[0].strip()
        row = await connection.fetchrow(
            """
            SELECT AVG(lat)::float AS lat, AVG(lng)::float AS lng
            FROM restaurants
            WHERE city ILIKE $1
              AND lat IS NOT NULL
              AND lng IS NOT NULL
            """,
            city_name,
        )
        if row and row["lat"] is not None and row["lng"] is not None:
            return row["lat"], row["lng"]

        row = await connection.fetchrow(
            """
            SELECT AVG(lat)::float AS lat, AVG(lng)::float AS lng
            FROM restaurants
            WHERE address ILIKE $1
              AND lat IS NOT NULL
              AND lng IS NOT NULL
            """,
            f"%{city_name}%",
        )
        if row and row["lat"] is not None and row["lng"] is not None:
            return row["lat"], row["lng"]

        raise ValueError(
            f"Could not resolve a city center for '{city}'. Provide --lat and --lng instead."
        )

    def place_to_record(self, place: dict[str, Any], *, existing_id: uuid.UUID | None = None) -> dict[str, Any]:
        place_id = place["id"]
        address = place.get("formattedAddress")
        location = place.get("location") or {}
        website = place.get("websiteUri")
        phone = place.get("nationalPhoneNumber")
        return {
            "id": existing_id or build_restaurant_id(place_id),
            "name": (place.get("displayName") or {}).get("text") or place_id,
            "address": address,
            "lat": location.get("latitude"),
            "lng": location.get("longitude"),
            "city": extract_city(address),
            "zip": extract_zip(address),
            "price_level": map_price_level(place.get("priceLevel")),
            "google_place_id": place_id,
            "yelp_id": None,
            "ubereats_id": None,
            "website": website,
            "phone": phone,
            "hours": serialize_hours(place.get("regularOpeningHours")),
            "rating_score": place.get("rating"),
            "rating_count": place.get("userRatingCount"),
            "last_verified_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "types": normalize_types(place.get("types")),
        }

    async def fetch_existing_restaurants(self, connection, google_place_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not google_place_ids:
            return {}
        rows = await connection.fetch(
            """
            SELECT id, google_place_id, website, phone, hours
            FROM restaurants
            WHERE google_place_id = ANY($1::text[])
            """,
            list(google_place_ids),
        )
        return {row["google_place_id"]: dict(row) for row in rows}

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

    async def fetch_dietary_presence(
        self,
        connection,
        restaurant_ids: Sequence[uuid.UUID],
    ) -> dict[uuid.UUID, int]:
        """Return {restaurant_id: best_confidence_tier} for restaurants that have
        at least one dietary_attributes row. Absent means no dietary data."""
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

    async def upsert_restaurants(self, connection, restaurant_rows: list[tuple], stats: GooglePlacesStats) -> None:
        if not restaurant_rows:
            return

        existing_ids = {
            row["id"]
            for row in await connection.fetch(
                "SELECT id FROM restaurants WHERE id = ANY($1::uuid[])",
                [row[0] for row in restaurant_rows],
            )
        }
        stats.restaurants_added += sum(1 for row in restaurant_rows if row[0] not in existing_ids)
        stats.restaurants_updated += sum(1 for row in restaurant_rows if row[0] in existing_ids)

        query = """
            INSERT INTO restaurants (
                id, name, address, lat, lng, city, zip, price_level,
                google_place_id, yelp_id, ubereats_id, website, phone, hours,
                rating_score, rating_count, data_quality_score, last_verified_at,
                needs_refresh, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8,
                $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19, $20, $21
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                address = EXCLUDED.address,
                lat = EXCLUDED.lat,
                lng = EXCLUDED.lng,
                city = EXCLUDED.city,
                zip = EXCLUDED.zip,
                price_level = EXCLUDED.price_level,
                google_place_id = EXCLUDED.google_place_id,
                website = COALESCE(EXCLUDED.website, restaurants.website),
                phone = COALESCE(EXCLUDED.phone, restaurants.phone),
                hours = COALESCE(EXCLUDED.hours, restaurants.hours),
                rating_score = COALESCE(EXCLUDED.rating_score, restaurants.rating_score),
                rating_count = COALESCE(EXCLUDED.rating_count, restaurants.rating_count),
                data_quality_score = GREATEST(EXCLUDED.data_quality_score, restaurants.data_quality_score),
                last_verified_at = EXCLUDED.last_verified_at,
                needs_refresh = EXCLUDED.needs_refresh,
                updated_at = EXCLUDED.updated_at
        """
        for batch in chunked(restaurant_rows):
            await connection.executemany(query, batch)

    async def upsert_cuisine_tags(self, connection, cuisine_tag_rows: list[tuple], stats: GooglePlacesStats) -> None:
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

    async def persist_places(self, connection, places: Sequence[dict[str, Any]], stats: GooglePlacesStats) -> None:
        if not places:
            return

        existing = await self.fetch_existing_restaurants(connection, [place["id"] for place in places])
        restaurant_models = [
            self.place_to_record(place, existing_id=existing.get(place["id"], {}).get("id"))
            for place in places
            if place.get("id")
        ]
        restaurant_ids = [restaurant["id"] for restaurant in restaurant_models]
        menu_presence = await self.fetch_menu_presence(connection, restaurant_ids)
        dietary_presence = await self.fetch_dietary_presence(connection, restaurant_ids)

        restaurant_rows: list[tuple] = []
        cuisine_tag_rows: list[tuple] = []
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        for restaurant in restaurant_models:
            has_menu_items = restaurant["id"] in menu_presence
            has_contact_or_website = bool(restaurant["website"] or restaurant["phone"])
            best_dietary_tier = dietary_presence.get(restaurant["id"])
            has_dietary_attribute = best_dietary_tier is not None
            has_high_confidence_dietary = best_dietary_tier is not None and best_dietary_tier <= 2
            restaurant_rows.append(
                (
                    restaurant["id"],
                    restaurant["name"],
                    restaurant["address"],
                    restaurant["lat"],
                    restaurant["lng"],
                    restaurant["city"],
                    restaurant["zip"],
                    restaurant["price_level"],
                    restaurant["google_place_id"],
                    restaurant["yelp_id"],
                    restaurant["ubereats_id"],
                    restaurant["website"],
                    restaurant["phone"],
                    restaurant["hours"],
                    restaurant["rating_score"],
                    restaurant["rating_count"],
                    compute_data_quality_score(
                        has_menu_items=has_menu_items,
                        has_google_place_id=True,
                        has_contact_or_website=has_contact_or_website,
                        has_dietary_attribute=has_dietary_attribute,
                        has_high_confidence_dietary=has_high_confidence_dietary,
                        verified_recently=True,
                    ),
                    now,
                    not has_menu_items,
                    restaurant["created_at"],
                    now,
                )
            )
            for tag in restaurant["types"]:
                cuisine_tag_rows.append((restaurant["id"], tag, "google_places", False))

        await self.upsert_restaurants(connection, restaurant_rows, stats)
        await self.upsert_cuisine_tags(connection, cuisine_tag_rows, stats)

    async def discovery(
        self,
        *,
        city: str | None = None,
        lat: float | None = None,
        lng: float | None = None,
        radius: int = 10000,
        max_pages: int = 1,
    ) -> GooglePlacesStats:
        stats = GooglePlacesStats()
        async with self.pool.acquire() as connection:
            await self.ensure_restaurant_columns(connection)

            if city and (lat is None or lng is None):
                lat, lng = await self.resolve_city_center(connection, city)
            if lat is None or lng is None:
                raise ValueError("Discovery requires either --city or both --lat and --lng.")

            page_token: str | None = None
            page_number = 0
            while page_number < max_pages:
                page_number += 1
                body: dict[str, Any] = {
                    "includedTypes": ["restaurant"],
                    "locationRestriction": {
                        "circle": {
                            "center": {"latitude": lat, "longitude": lng},
                            "radius": float(radius),
                        }
                    },
                    "maxResultCount": 20,
                }
                if page_token:
                    body["pageToken"] = page_token

                payload = await self.request_json(
                    connection,
                    method="POST",
                    url=DISCOVERY_URL,
                    query_label=f"discovery|city={city}|lat={lat}|lng={lng}|radius={radius}|page={page_number}",
                    json_body=body,
                    field_mask=DISCOVERY_FIELD_MASK,
                )
                stats.api_calls += 1

                places = payload.get("places", [])
                await self.persist_places(connection, places, stats)
                for _ in places:
                    stats.places_processed += 1
                    if stats.places_processed % 10 == 0:
                        print(f"Processed {stats.places_processed} places...")

                page_token = payload.get("nextPageToken")
                if not page_token:
                    break

        return stats

    async def fetch_enrichment_targets(
        self,
        connection,
        *,
        restaurant_ids: Sequence[str] | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if restaurant_ids:
            rows = await connection.fetch(
                """
                SELECT id, google_place_id
                FROM restaurants
                WHERE id = ANY($1::uuid[])
                  AND google_place_id IS NOT NULL
                """,
                list(restaurant_ids),
            )
            return [dict(row) for row in rows]

        rows = await connection.fetch(
            """
            SELECT id, google_place_id
            FROM restaurants
            WHERE google_place_id IS NOT NULL
              AND (
                    hours IS NULL
                 OR website IS NULL
                 OR phone IS NULL
                 OR price_level IS NULL
                 OR rating_score IS NULL
                 OR rating_count IS NULL
              )
            ORDER BY updated_at NULLS FIRST, created_at NULLS FIRST
            LIMIT $1
            """,
            limit,
        )
        return [dict(row) for row in rows]

    async def enrichment(
        self,
        *,
        restaurant_ids: Sequence[str] | None = None,
        limit: int = 100,
    ) -> GooglePlacesStats:
        """Fetch full place details for restaurants that have a google_place_id
        but are missing one or more fields (hours, website, phone, etc.).

        WARNING — BILLING: Every details request is billed at the Enterprise
        SKU tier because the field mask includes Enterprise-tier fields
        (regularOpeningHours, websiteUri, nationalPhoneNumber, rating,
        userRatingCount, priceLevel). With 62,000+ Tier B restaurants in the
        DB, a bulk run would cost hundreds of dollars. Do NOT run enrichment
        as a batch job against all Tier B restaurants. Only trigger enrichment
        for specific restaurants on demand (e.g. when a user views a restaurant
        page) or for small targeted batches with an explicit --restaurant-id
        list.
        """
        stats = GooglePlacesStats()
        async with self.pool.acquire() as connection:
            await self.ensure_restaurant_columns(connection)
            targets = await self.fetch_enrichment_targets(
                connection,
                restaurant_ids=restaurant_ids,
                limit=limit,
            )
            for index, target in enumerate(targets, start=1):
                payload = await self.request_json(
                    connection,
                    method="GET",
                    url=DETAILS_URL_TEMPLATE.format(place_id=target["google_place_id"]),
                    query_label=f"details|restaurant_id={target['id']}|place_id={target['google_place_id']}",
                    field_mask=DETAILS_FIELD_MASK,
                )
                stats.api_calls += 1
                await self.persist_places(connection, [payload], stats)
                stats.places_processed += 1
                if index % 10 == 0:
                    print(f"Processed {index} places...")
        return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Google Places discovery and enrichment agent")
    parser.add_argument("--mode", choices=["discovery", "enrichment"], required=True)
    parser.add_argument("--city")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lng", type=float)
    parser.add_argument("--radius", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--restaurant-id", action="append", dest="restaurant_ids")
    return parser


async def run_cli(args: argparse.Namespace) -> GooglePlacesStats:
    async with GooglePlacesAgent() as agent:
        if args.mode == "discovery":
            return await agent.discovery(
                city=args.city,
                lat=args.lat,
                lng=args.lng,
                radius=args.radius,
                max_pages=args.max_pages,
            )
        return await agent.enrichment(
            restaurant_ids=args.restaurant_ids,
            limit=args.limit,
        )


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        stats = await run_cli(args)
    except RateLimitPaused as exc:
        print(str(exc))
        return

    print(
        "Google Places run complete:",
        {
            "restaurants_added": stats.restaurants_added,
            "restaurants_updated": stats.restaurants_updated,
            "cuisine_tags_added": stats.cuisine_tags_added,
            "api_calls": stats.api_calls,
            "places_processed": stats.places_processed,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
