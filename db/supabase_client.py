from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg
from dotenv import load_dotenv

load_dotenv()


def get_database_url() -> str:
    database_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    host = os.getenv("PGHOST")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE")
    user = os.getenv("PGUSER")
    password = os.getenv("PGPASSWORD")

    if all([host, database, user, password]):
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    raise RuntimeError(
        "Database connection details are missing. Set SUPABASE_DB_URL or the "
        "PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD variables."
    )


async def create_pool(
    *,
    min_size: int = 1,
    max_size: int = 5,
    command_timeout: float = 60.0,
) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=get_database_url(),
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        statement_cache_size=0,
    )


@asynccontextmanager
async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    pool = await create_pool(min_size=1, max_size=1)
    try:
        async with pool.acquire() as connection:
            yield connection
    finally:
        await pool.close()


async def test_connection() -> dict[str, Any]:
    async with get_connection() as connection:
        row = await connection.fetchrow(
            "SELECT current_database() AS database_name, current_user AS current_user, version() AS version"
        )
    return dict(row)
