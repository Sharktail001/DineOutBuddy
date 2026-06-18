from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping

import asyncpg
from dotenv import load_dotenv

load_dotenv()

# AGENTS.md lists the sources to track, but not the numeric daily quotas.
# These defaults are conservative fallbacks and can be overridden with env vars
# like GOOGLE_PLACES_DAILY_LIMIT or YELP_DAILY_LIMIT.
DEFAULT_DAILY_LIMITS: dict[str, int] = {
    "google_places": 1000,
    "yelp": 500,
    "anthropic": 1000,
    "openai": 500,
    "zabihah": 200,
    "web_research": 150,
    "llm_normalizer": 1000,
}


@dataclass(slots=True)
class RateLimitResult:
    source: str
    status: str
    calls_today: int
    daily_limit: int
    remaining_calls: int
    usage_ratio: float
    warning_threshold: float
    pause_threshold: float


def load_daily_limits() -> dict[str, int]:
    limits = dict(DEFAULT_DAILY_LIMITS)
    for source, default in DEFAULT_DAILY_LIMITS.items():
        raw_value = os.getenv(f"{source.upper()}_DAILY_LIMIT")
        limits[source] = int(raw_value) if raw_value else default
    return limits


async def count_calls_today(
    connection: asyncpg.Connection,
    source: str,
    *,
    on_date: date | None = None,
) -> int:
    return await connection.fetchval(
        """
        SELECT COUNT(*)
        FROM data_fetch_log
        WHERE source = $1
          AND fetched_at::date = $2
          AND status IN ('success', 'partial')
        """,
        source,
        on_date or datetime.now(timezone.utc).date(),
    )


async def check_rate_limit(
    connection: asyncpg.Connection,
    source: str,
    *,
    daily_limits: Mapping[str, int] | None = None,
    warning_threshold: float = 0.8,
    pause_threshold: float = 0.95,
    on_date: date | None = None,
) -> RateLimitResult:
    limits = dict(daily_limits or load_daily_limits())
    daily_limit = limits.get(source)
    if daily_limit is None:
        raise ValueError(
            f"No daily rate limit configured for source '{source}'. "
            "Add it to DEFAULT_DAILY_LIMITS or pass daily_limits explicitly."
        )

    calls_today = await count_calls_today(connection, source, on_date=on_date)
    usage_ratio = calls_today / daily_limit if daily_limit else 1.0
    remaining_calls = max(daily_limit - calls_today, 0)

    if calls_today >= daily_limit or usage_ratio >= pause_threshold:
        status = "pause"
    elif usage_ratio >= warning_threshold:
        status = "warn"
    else:
        status = "proceed"

    return RateLimitResult(
        source=source,
        status=status,
        calls_today=calls_today,
        daily_limit=daily_limit,
        remaining_calls=remaining_calls,
        usage_ratio=usage_ratio,
        warning_threshold=warning_threshold,
        pause_threshold=pause_threshold,
    )
