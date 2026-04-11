from __future__ import annotations

import asyncio
import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from db.supabase_client import create_pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESTAURANT_CSV_CANDIDATES = (
    PROJECT_ROOT / "data" / "restaurants.csv",
    PROJECT_ROOT / "data" / "restuarant.csv",
)
DEFAULT_MENU_CSV_PATH = PROJECT_ROOT / "data" / "menu.csv"
RESTAURANT_NAMESPACE = uuid.UUID("14f4b4b2-bd80-4be5-a1bd-455e25f85bf4")
MENU_ITEM_NAMESPACE = uuid.UUID("ac5447b9-c37e-49a8-9df0-f03945433834")
FETCH_LOG_NAMESPACE = uuid.UUID("78a3230f-7499-46e5-b196-3881da53a755")
BATCH_SIZE = 1000


@dataclass(slots=True)
class ImportStats:
    restaurants_added: int = 0
    restaurants_updated: int = 0
    menu_items_added: int = 0
    menu_items_updated: int = 0
    cuisine_tags_added: int = 0
    tier_a_count: int = 0
    tier_b_count: int = 0
    tier_c_count: int = 0
    unmatched_menu_restaurants: int = 0


def resolve_restaurant_csv_path(path: str | Path | None = None) -> Path:
    if path is not None:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Restaurant CSV not found: {candidate}")

    for candidate in DEFAULT_RESTAURANT_CSV_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No restaurant CSV found. Expected one of: "
        + ", ".join(str(path) for path in DEFAULT_RESTAURANT_CSV_CANDIDATES)
    )


def resolve_menu_csv_path(path: str | Path | None = None) -> Path:
    candidate = Path(path) if path is not None else DEFAULT_MENU_CSV_PATH
    if not candidate.exists():
        raise FileNotFoundError(f"Menu CSV not found: {candidate}")
    return candidate


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else None


def parse_price(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace("USD", "").replace("$", "").strip()
    return parse_float(cleaned)


def parse_id_array(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def extract_city(full_address: str | None) -> str | None:
    if not full_address:
        return None
    parts = [part.strip() for part in full_address.split(",") if part.strip()]
    if len(parts) >= 2:
        return parts[-2]
    return None


def normalize_tag(tag: str) -> str:
    normalized = tag.strip().lower().replace("&", "and")
    for character in ("/", "-", " "):
        normalized = normalized.replace(character, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def derive_cuisine_tags(category_value: str | None) -> list[str]:
    if not category_value:
        return []
    tags = []
    for raw_tag in category_value.split(","):
        normalized = normalize_tag(raw_tag)
        if normalized:
            tags.append(normalized)
    return list(dict.fromkeys(tags))


def build_restaurant_uuid(external_id: str) -> uuid.UUID:
    return uuid.uuid5(RESTAURANT_NAMESPACE, external_id)


def build_menu_item_uuid(
    restaurant_id: uuid.UUID,
    category: str | None,
    name: str,
    description: str | None,
    price: float | None,
) -> uuid.UUID:
    payload = "|".join(
        [
            str(restaurant_id),
            category or "",
            name,
            description or "",
            "" if price is None else f"{price:.2f}",
        ]
    )
    return uuid.uuid5(MENU_ITEM_NAMESPACE, payload)


def compute_data_quality_score(
    *,
    has_menu_items: bool,
    has_google_place_id: bool,
    has_dietary_attribute: bool = False,
    has_high_confidence_dietary: bool = False,
    has_contact_or_website: bool = False,
    verified_recently: bool = True,
) -> float:
    score = 0.0
    if has_menu_items:
        score += 0.30
    if has_google_place_id:
        score += 0.15
    if has_dietary_attribute:
        score += 0.20
    if has_high_confidence_dietary:
        score += 0.15
    if has_contact_or_website:
        score += 0.10
    if verified_recently:
        score += 0.10
    return min(score, 1.0)


def classify_tier(*, has_menu_items: bool, has_supporting_metadata: bool) -> str:
    if has_menu_items:
        return "A"
    if has_supporting_metadata:
        return "B"
    return "C"


def chunked(items: Sequence[tuple], size: int = BATCH_SIZE) -> Iterator[list[tuple]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


async def apply_schema(connection) -> None:
    schema_path = PROJECT_ROOT / "db" / "schema.sql"
    await connection.execute(schema_path.read_text(encoding="utf-8"))


async def truncate_import_tables(connection) -> None:
    await connection.execute(
        """
        TRUNCATE TABLE
            group_recommendations,
            group_sessions,
            user_interactions,
            taste_profiles,
            users,
            feature_vectors,
            cuisine_tags,
            dietary_attributes,
            menu_items,
            restaurants,
            data_fetch_log
        RESTART IDENTITY CASCADE
        """
    )


def load_restaurant_rows(
    restaurant_csv_path: Path,
    *,
    now: datetime,
) -> tuple[dict[uuid.UUID, dict], dict[str, uuid.UUID]]:
    restaurant_rows: dict[uuid.UUID, dict] = {}
    restaurant_lookup: dict[str, uuid.UUID] = {}

    with restaurant_csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = (row.get("id") or "").strip()
            if not raw_id:
                continue

            restaurant_id = build_restaurant_uuid(raw_id)
            candidate_ids = {raw_id}
            candidate_ids.update(parse_id_array(row.get("id_array")))
            if row.get("places_id"):
                candidate_ids.add(row["places_id"].strip())

            for candidate_id in candidate_ids:
                if candidate_id:
                    restaurant_lookup[candidate_id] = restaurant_id

            restaurant_rows[restaurant_id] = {
                "id": restaurant_id,
                "name": (row.get("label") or "").strip() or f"Restaurant {raw_id}",
                "address": (row.get("full_address") or "").strip() or None,
                "lat": parse_float(row.get("lat")),
                "lng": parse_float(row.get("lng")),
                "city": extract_city(row.get("full_address")),
                "zip": (row.get("zip_code") or "").strip() or None,
                "price_level": parse_int(row.get("price_level")),
                "google_place_id": (row.get("places_id") or "").strip() or None,
                "yelp_id": None,
                "ubereats_id": None,
                "website": (row.get("website") or "").strip() or None,
                "phone": (row.get("phone_number") or "").strip() or None,
                "hours": None,
                "rating_score": parse_float(row.get("score")),
                "rating_count": parse_int(row.get("ratings")),
                "data_quality_score": 0.0,
                "last_verified_at": now,
                "needs_refresh": False,
                "created_at": now,
                "updated_at": now,
                "source_has_website_or_phone": bool(
                    (row.get("website") or "").strip() or (row.get("phone_number") or "").strip()
                ),
                "tier": "C",
                "external_source_id": raw_id,
                "category": (row.get("category") or "").strip() or None,
                "matched_menu_external_ids": set(),
            }

    return restaurant_rows, restaurant_lookup


def scan_menu_relationships(
    menu_csv_path: Path,
    *,
    restaurant_rows: dict[uuid.UUID, dict],
    restaurant_lookup: dict[str, uuid.UUID],
    now: datetime,
) -> dict[uuid.UUID, dict]:
    placeholder_restaurants: dict[uuid.UUID, dict] = {}

    with menu_csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            external_restaurant_id = (row.get("restaurant_id") or "").strip()
            if not external_restaurant_id:
                continue

            restaurant_id = restaurant_lookup.get(external_restaurant_id)
            if restaurant_id is None:
                restaurant_id = build_restaurant_uuid(f"menu-only:{external_restaurant_id}")
                if restaurant_id not in placeholder_restaurants:
                    placeholder_restaurants[restaurant_id] = {
                        "id": restaurant_id,
                        "name": f"Menu source {external_restaurant_id}",
                        "address": None,
                        "lat": None,
                        "lng": None,
                        "city": None,
                        "zip": None,
                        "price_level": None,
                        "google_place_id": None,
                        "yelp_id": None,
                        "ubereats_id": external_restaurant_id,
                        "website": None,
                        "phone": None,
                        "hours": None,
                        "data_quality_score": 0.0,
                        "last_verified_at": now,
                        "needs_refresh": False,
                        "created_at": now,
                        "updated_at": now,
                        "source_has_website_or_phone": False,
                        "tier": "A",
                        "external_source_id": external_restaurant_id,
                        "category": None,
                    }
            else:
                restaurant_rows[restaurant_id]["matched_menu_external_ids"].add(external_restaurant_id)

    return placeholder_restaurants


def restaurant_tuple(restaurant: dict) -> tuple:
    return (
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
        restaurant.get("website"),
        restaurant.get("phone"),
        restaurant.get("hours"),
        restaurant.get("rating_score"),
        restaurant.get("rating_count"),
        restaurant["data_quality_score"],
        restaurant["last_verified_at"],
        restaurant["needs_refresh"],
        restaurant["created_at"],
        restaurant["updated_at"],
    )


def build_cuisine_tag_rows(restaurant: dict) -> list[tuple]:
    return [
        (restaurant["id"], tag, "restaurant_csv", False)
        for tag in derive_cuisine_tags(restaurant.get("category"))
    ]


def update_tier_stats(tier: str, stats: ImportStats) -> None:
    if tier == "A":
        stats.tier_a_count += 1
    elif tier == "B":
        stats.tier_b_count += 1
    else:
        stats.tier_c_count += 1


def finalize_restaurants(
    *,
    restaurant_rows: dict[uuid.UUID, dict],
    placeholder_restaurants: dict[uuid.UUID, dict],
    stats: ImportStats,
) -> tuple[list[tuple], list[tuple]]:
    finalized_rows: list[tuple] = []
    cuisine_tag_rows: list[tuple] = []

    for restaurant in restaurant_rows.values():
        has_menu_items = bool(restaurant["matched_menu_external_ids"])
        has_supporting_metadata = bool(
            restaurant["google_place_id"]
            or restaurant["address"]
            or restaurant["source_has_website_or_phone"]
            or restaurant["category"]
        )
        restaurant["tier"] = classify_tier(
            has_menu_items=has_menu_items,
            has_supporting_metadata=has_supporting_metadata,
        )
        if has_menu_items and not restaurant["ubereats_id"]:
            restaurant["ubereats_id"] = sorted(restaurant["matched_menu_external_ids"])[0]
        restaurant["needs_refresh"] = restaurant["tier"] != "A"
        restaurant["data_quality_score"] = compute_data_quality_score(
            has_menu_items=has_menu_items,
            has_google_place_id=bool(restaurant["google_place_id"]),
            has_contact_or_website=restaurant["source_has_website_or_phone"],
            verified_recently=True,
        )
        update_tier_stats(restaurant["tier"], stats)
        finalized_rows.append(restaurant_tuple(restaurant))
        cuisine_tag_rows.extend(build_cuisine_tag_rows(restaurant))

    for restaurant in placeholder_restaurants.values():
        restaurant["data_quality_score"] = compute_data_quality_score(
            has_menu_items=True,
            has_google_place_id=False,
            has_contact_or_website=False,
            verified_recently=True,
        )
        stats.unmatched_menu_restaurants += 1
        update_tier_stats("A", stats)
        finalized_rows.append(restaurant_tuple(restaurant))

    return finalized_rows, cuisine_tag_rows


async def upsert_restaurants(connection, restaurant_rows: list[tuple], stats: ImportStats) -> None:
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
            yelp_id = EXCLUDED.yelp_id,
            ubereats_id = EXCLUDED.ubereats_id,
            website = EXCLUDED.website,
            phone = EXCLUDED.phone,
            hours = EXCLUDED.hours,
            rating_score = EXCLUDED.rating_score,
            rating_count = EXCLUDED.rating_count,
            data_quality_score = EXCLUDED.data_quality_score,
            last_verified_at = EXCLUDED.last_verified_at,
            needs_refresh = EXCLUDED.needs_refresh,
            updated_at = EXCLUDED.updated_at
    """
    for batch in chunked(restaurant_rows):
        await connection.executemany(query, batch)
        print(f"  Restaurants: {stats.restaurants_added} added, {stats.restaurants_updated} updated")


async def upsert_menu_items(connection, menu_item_rows: list[tuple], stats: ImportStats) -> None:
    if not menu_item_rows:
        return

    existing_ids = {
        row["id"]
        for row in await connection.fetch(
            "SELECT id FROM menu_items WHERE id = ANY($1::uuid[])",
            [row[0] for row in menu_item_rows],
        )
    }
    stats.menu_items_added += sum(1 for row in menu_item_rows if row[0] not in existing_ids)
    stats.menu_items_updated += sum(1 for row in menu_item_rows if row[0] in existing_ids)

    query = """
        INSERT INTO menu_items (
            id, restaurant_id, category, name, description, price, source, fetched_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8
        )
        ON CONFLICT (id) DO UPDATE SET
            restaurant_id = EXCLUDED.restaurant_id,
            category = EXCLUDED.category,
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            price = EXCLUDED.price,
            source = EXCLUDED.source,
            fetched_at = EXCLUDED.fetched_at
    """
    for batch in chunked(menu_item_rows):
        await connection.executemany(query, batch)


async def stream_menu_items(
    connection,
    *,
    menu_csv_path: Path,
    restaurant_lookup: dict[str, uuid.UUID],
    placeholder_restaurants: dict[uuid.UUID, dict],
    stats: ImportStats,
    now: datetime,
) -> None:
    batch: list[tuple] = []
    placeholder_lookup = {
        restaurant["external_source_id"]: restaurant["id"]
        for restaurant in placeholder_restaurants.values()
    }

    with menu_csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            external_restaurant_id = (row.get("restaurant_id") or "").strip()
            menu_name = (row.get("name") or "").strip()
            if not external_restaurant_id or not menu_name:
                continue

            restaurant_id = restaurant_lookup.get(external_restaurant_id) or placeholder_lookup.get(
                external_restaurant_id
            )
            if restaurant_id is None:
                continue

            category = (row.get("category") or "").strip() or None
            description = (row.get("description") or "").strip() or None
            price = parse_price(row.get("price"))
            item_id = build_menu_item_uuid(restaurant_id, category, menu_name, description, price)
            batch.append(
                (
                    item_id,
                    restaurant_id,
                    category,
                    menu_name,
                    description,
                    price,
                    "menu_csv",
                    now,
                )
            )

            if len(batch) >= BATCH_SIZE:
                await upsert_menu_items(connection, batch, stats)
                print(f"  Menu items: {stats.menu_items_added} added...")
                batch = []

    if batch:
        await upsert_menu_items(connection, batch, stats)


async def upsert_cuisine_tags(connection, cuisine_tag_rows: list[tuple], stats: ImportStats) -> None:
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


async def log_import_run(
    connection,
    *,
    restaurant_csv_path: Path,
    menu_csv_path: Path,
    stats: ImportStats,
    now: datetime,
) -> None:
    query_text = (
        f"restaurants={restaurant_csv_path.name}; menu={menu_csv_path.name}; "
        f"tiers=A:{stats.tier_a_count},B:{stats.tier_b_count},C:{stats.tier_c_count}; "
        f"unmatched_menu_restaurants={stats.unmatched_menu_restaurants}"
    )
    await connection.execute(
        """
        INSERT INTO data_fetch_log (
            id, source, query, status, fetched_at, records_added, records_updated
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7
        )
        """,
        uuid.uuid5(FETCH_LOG_NAMESPACE, f"{query_text}|{now.isoformat()}"),
        "csv_importer",
        query_text,
        "partial" if stats.unmatched_menu_restaurants else "success",
        now,
        stats.restaurants_added + stats.menu_items_added + stats.cuisine_tags_added,
        stats.restaurants_updated + stats.menu_items_updated,
    )


async def import_csv_data(
    *,
    restaurant_csv_path: str | Path | None = None,
    menu_csv_path: str | Path | None = None,
    apply_schema_first: bool = False,
    truncate_existing: bool = False,
) -> ImportStats:
    resolved_restaurant_csv = resolve_restaurant_csv_path(restaurant_csv_path)
    resolved_menu_csv = resolve_menu_csv_path(menu_csv_path)
    stats = ImportStats()
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                if apply_schema_first:
                    await apply_schema(connection)
                if truncate_existing:
                    await truncate_import_tables(connection)

                restaurant_rows, restaurant_lookup = load_restaurant_rows(
                    resolved_restaurant_csv,
                    now=now,
                )
                placeholder_restaurants = scan_menu_relationships(
                    resolved_menu_csv,
                    restaurant_rows=restaurant_rows,
                    restaurant_lookup=restaurant_lookup,
                    now=now,
                )
                finalized_restaurants, cuisine_tag_rows = finalize_restaurants(
                    restaurant_rows=restaurant_rows,
                    placeholder_restaurants=placeholder_restaurants,
                    stats=stats,
                )

                await upsert_restaurants(connection, finalized_restaurants, stats)
                await stream_menu_items(
                    connection,
                    menu_csv_path=resolved_menu_csv,
                    restaurant_lookup=restaurant_lookup,
                    placeholder_restaurants=placeholder_restaurants,
                    stats=stats,
                    now=now,
                )
                await upsert_cuisine_tags(connection, cuisine_tag_rows, stats)
                await log_import_run(
                    connection,
                    restaurant_csv_path=resolved_restaurant_csv,
                    menu_csv_path=resolved_menu_csv,
                    stats=stats,
                    now=now,
                )
        return stats
    finally:
        await pool.close()


async def main() -> None:
    print("Starting CSV import...")
    stats = await import_csv_data()
    print(stats)
    print("Import complete.")


if __name__ == "__main__":
    asyncio.run(main())
