from __future__ import annotations

import asyncio
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.csv_importer import (
    RESTAURANT_NAMESPACE,
    build_restaurant_uuid,
    compute_data_quality_score,
    derive_cuisine_tags,
    extract_city,
    parse_float,
    parse_int,
)
from db.supabase_client import create_pool

HALAL_CSV_PATH = PROJECT_ROOT / "data" / "halal2.csv"
DIETARY_SOURCE = "zabihah_csv"
DIETARY_NOTES = "Community verified via Zabihah.com manual export"


def strip_halal_from_tags(tags: list[str]) -> list[str]:
    return [t for t in tags if t != "halal"]


async def import_halal_csv() -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    rows: list[dict] = []
    with HALAL_CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {HALAL_CSV_PATH.name}")

    pool = await create_pool()
    try:
        async with pool.acquire() as conn:
            matched_updated = 0
            inserted_new = 0
            dietary_added = 0
            cuisine_added = 0
            skipped = 0

            for i, row in enumerate(rows):
                places_id = (row.get("places_id") or "").strip()
                raw_id = (row.get("id") or "").strip()
                name = (row.get("label") or "").strip()

                if not name:
                    skipped += 1
                    continue

                seed = places_id or raw_id
                if not seed:
                    skipped += 1
                    continue

                restaurant_id = build_restaurant_uuid(seed)

                existing = None
                if places_id:
                    existing = await conn.fetchrow(
                        "SELECT id FROM restaurants WHERE google_place_id = $1",
                        places_id,
                    )

                address = (row.get("full_address") or "").strip() or None
                lat = parse_float(row.get("lat"))
                lng = parse_float(row.get("lng"))
                city = extract_city(row.get("full_address"))
                zip_code = (row.get("zip_code") or "").strip() or None
                price_level = parse_int(row.get("price_level"))
                rating_score = parse_float(row.get("score"))
                rating_count = parse_int(row.get("ratings"))
                website = (row.get("website") or "").strip() or None
                phone = (row.get("phone_number") or "").strip() or None

                has_menu = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM menu_items WHERE restaurant_id = $1)",
                    existing["id"] if existing else restaurant_id,
                )
                has_contact = bool(website or phone)
                if existing:
                    existing_contact = await conn.fetchrow(
                        "SELECT website, phone FROM restaurants WHERE id = $1",
                        existing["id"],
                    )
                    has_contact = has_contact or bool(
                        (existing_contact["website"] if existing_contact else None)
                        or (existing_contact["phone"] if existing_contact else None)
                    )

                dq_score = compute_data_quality_score(
                    has_menu_items=has_menu,
                    has_google_place_id=bool(places_id),
                    has_dietary_attribute=True,
                    has_high_confidence_dietary=True,
                    has_contact_or_website=has_contact,
                    verified_recently=True,
                )

                if existing:
                    rid = existing["id"]
                    await conn.execute(
                        """
                        UPDATE restaurants SET
                            address = COALESCE(restaurants.address, $2),
                            lat = COALESCE(restaurants.lat, $3),
                            lng = COALESCE(restaurants.lng, $4),
                            city = COALESCE(restaurants.city, $5),
                            zip = COALESCE(restaurants.zip, $6),
                            price_level = COALESCE(restaurants.price_level, $7),
                            rating_score = COALESCE(restaurants.rating_score, $8),
                            rating_count = COALESCE(restaurants.rating_count, $9),
                            website = COALESCE(restaurants.website, $10),
                            phone = COALESCE(restaurants.phone, $11),
                            data_quality_score = GREATEST(restaurants.data_quality_score, $12),
                            last_verified_at = $13,
                            updated_at = $13
                        WHERE id = $1
                        """,
                        rid, address, lat, lng, city, zip_code,
                        price_level, rating_score, rating_count,
                        website, phone, dq_score, now,
                    )
                    matched_updated += 1
                else:
                    rid = restaurant_id
                    await conn.execute(
                        """
                        INSERT INTO restaurants (
                            id, name, address, lat, lng, city, zip, price_level,
                            google_place_id, rating_score, rating_count,
                            website, phone,
                            data_quality_score, last_verified_at, needs_refresh,
                            created_at, updated_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8,
                            $9, $10, $11,
                            $12, $13,
                            $14, $15, $16,
                            $15, $15
                        )
                        ON CONFLICT (id) DO NOTHING
                        """,
                        rid, name, address, lat, lng, city, zip_code, price_level,
                        places_id or None, rating_score, rating_count,
                        website, phone,
                        dq_score, now, False,
                    )
                    inserted_new += 1

                result = await conn.execute(
                    """
                    INSERT INTO dietary_attributes (
                        id, restaurant_id, dietary_type, value,
                        confidence_tier, source, notes, fetched_at
                    ) VALUES (
                        gen_random_uuid(), $1, 'halal', 'true',
                        2, $2, $3, $4
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    rid, DIETARY_SOURCE, DIETARY_NOTES, now,
                )
                if "INSERT 0 0" not in result:
                    dietary_added += 1

                category = (row.get("category") or "").strip()
                tags = strip_halal_from_tags(derive_cuisine_tags(category))
                for tag in tags:
                    await conn.execute(
                        """
                        INSERT INTO cuisine_tags (restaurant_id, tag, source, derived_from_menu)
                        VALUES ($1, $2, $3, false)
                        ON CONFLICT DO NOTHING
                        """,
                        rid, tag, DIETARY_SOURCE,
                    )
                    cuisine_added += 1

                if (i + 1) % 50 == 0:
                    print(
                        f"  Progress: {i + 1}/{len(rows)} — "
                        f"matched={matched_updated}, inserted={inserted_new}, "
                        f"dietary={dietary_added}, skipped={skipped}"
                    )

            print(f"\n--- Final Stats ---")
            print(f"  matched_updated: {matched_updated}")
            print(f"  inserted_new:    {inserted_new}")
            print(f"  dietary_added:   {dietary_added}")
            print(f"  cuisine_added:   {cuisine_added}")
            print(f"  skipped:         {skipped}")
            print(f"  total processed: {matched_updated + inserted_new + skipped}")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(import_halal_csv())
