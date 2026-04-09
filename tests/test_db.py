from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.csv_importer import import_csv_data
from db.supabase_client import create_pool, test_connection as run_test_connection

pytestmark = pytest.mark.asyncio


def require_database_url() -> None:
    if not (os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")):
        pytest.skip("SUPABASE_DB_URL or DATABASE_URL is required for integration tests.")


@pytest.fixture
def sample_csvs(tmp_path: Path) -> tuple[Path, Path]:
    restaurants_path = tmp_path / "restaurants.csv"
    menu_path = tmp_path / "menu.csv"

    restaurants_path.write_text(
        "\n".join(
            [
                "id,label,score,ratings,category,price_level,full_address,zip_code,lat,lng,places_id,hours_operational,hours_delivery,hours_pickup,hours_dine_in,website,phone_number,id_array",
                'rest-1,Sunrise Grill,4.5,120,"Breakfast, American",2,"123 Main St, Dallas, TX",75001,32.7767,-96.7970,place-1,,,,,https://sunrise.example,5551234567,"[""menu-1""]"',
                "rest-2,Hidden Gem,,,,,,,,,,,,,,,,",
            ]
        ),
        encoding="utf-8",
    )
    menu_path.write_text(
        "\n".join(
            [
                "restaurant_id,category,name,description,price",
                "menu-1,Entrees,Chicken and Waffles,Crispy chicken with waffles,14.99 USD",
                "menu-only-9,Dessert,Chocolate Cake,Rich chocolate layer cake,6.50 USD",
            ]
        ),
        encoding="utf-8",
    )
    return restaurants_path, menu_path


async def test_connection_works() -> None:
    require_database_url()
    details = await run_test_connection()
    assert details["database_name"]
    assert details["current_user"]
    assert "PostgreSQL" in details["version"]


async def test_schema_exists() -> None:
    require_database_url()
    pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            await connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
            tables = {
                row["table_name"]
                for row in await connection.fetch(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name = ANY($1::text[])
                    """,
                    [
                        "restaurants",
                        "menu_items",
                        "dietary_attributes",
                        "cuisine_tags",
                        "feature_vectors",
                        "users",
                        "taste_profiles",
                        "user_interactions",
                        "group_sessions",
                        "group_recommendations",
                        "data_fetch_log",
                    ],
                )
            }
    finally:
        await pool.close()

    assert tables == {
        "restaurants",
        "menu_items",
        "dietary_attributes",
        "cuisine_tags",
        "feature_vectors",
        "users",
        "taste_profiles",
        "user_interactions",
        "group_sessions",
        "group_recommendations",
        "data_fetch_log",
    }


async def test_sample_import_runs_correctly(sample_csvs: tuple[Path, Path]) -> None:
    require_database_url()
    restaurants_path, menu_path = sample_csvs

    stats = await import_csv_data(
        restaurant_csv_path=restaurants_path,
        menu_csv_path=menu_path,
        apply_schema_first=True,
        truncate_existing=True,
    )

    assert stats.restaurants_added == 3
    assert stats.menu_items_added == 2
    assert stats.tier_a_count == 2
    assert stats.tier_b_count == 0
    assert stats.tier_c_count == 1
    assert stats.unmatched_menu_restaurants == 1

    pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            restaurant_count = await connection.fetchval("SELECT COUNT(*) FROM restaurants")
            menu_count = await connection.fetchval("SELECT COUNT(*) FROM menu_items")
            import_log = await connection.fetchrow(
                """
                SELECT source, status, records_added
                FROM data_fetch_log
                WHERE source = 'csv_importer'
                ORDER BY fetched_at DESC
                LIMIT 1
                """
            )
    finally:
        await pool.close()

    assert restaurant_count == 3
    assert menu_count == 2
    assert import_log["status"] == "partial"
    assert import_log["records_added"] >= 5
