from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from dotenv import load_dotenv

from agents.csv_importer import chunked, compute_data_quality_score, normalize_tag
from db.rate_limiter import RateLimitResult, check_rate_limit
from db.supabase_client import create_pool

load_dotenv()

ANTHROPIC_MODEL = "claude-3-5-haiku-latest"
BATCH_SIZE = 10
MAX_MENU_ITEMS = 20
SYSTEM_PROMPT = (
    "You normalize restaurant data for a recommendation engine. "
    "Return valid JSON only. Do not wrap the JSON in markdown."
)


class RateLimitPaused(Exception):
    pass


@dataclass(slots=True)
class NormalizerStats:
    restaurants_processed: int = 0
    restaurants_updated: int = 0
    cuisine_tags_added: int = 0
    dietary_attributes_added: int = 0
    api_calls: int = 0
    malformed_responses: int = 0
    batches_processed: int = 0


def get_anthropic_api_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required to run the LLM normalizer.")
    return api_key


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")
    return parsed


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None) or []
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def normalize_tags(tags: Sequence[str] | None) -> list[str]:
    if not tags:
        return []
    normalized: list[str] = []
    for tag in tags:
        value = normalize_tag(tag)
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def clamp_spice_level(value: Any) -> int | None:
    try:
        spice_level = int(value)
    except (TypeError, ValueError):
        return None
    return min(5, max(1, spice_level))


class LLMNormalizer:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        pool=None,
        client=None,
        sleep=asyncio.sleep,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.pool = pool
        self.client = client
        self.sleep = sleep
        self._owns_pool = pool is None
        self._owns_client = client is None

    async def __aenter__(self) -> "LLMNormalizer":
        if self.pool is None:
            self.pool = await create_pool()
        if self.client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Anthropic SDK is required for the LLM normalizer. Install it with `pip install anthropic`."
                ) from exc
            self.client = AsyncAnthropic(api_key=self.api_key or get_anthropic_api_key())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_pool and self.pool is not None:
            await self.pool.close()

    async def ensure_restaurant_columns(self, connection) -> None:
        await connection.execute(
            """
            ALTER TABLE restaurants
                ADD COLUMN IF NOT EXISTS spice_level_estimate INT,
                ADD COLUMN IF NOT EXISTS cuisine_detail TEXT
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
            ) VALUES ($1, 'anthropic', $2, $3, $4, $5, $6)
            """,
            uuid.uuid4(),
            query,
            status,
            now,
            records_added,
            records_updated,
        )

    async def check_anthropic_rate_limit(self, connection) -> RateLimitResult:
        result = await check_rate_limit(connection, "anthropic")
        if result.status == "warn":
            print(f"Warning: anthropic usage is at {result.calls_today}/{result.daily_limit} today.")
        if result.status == "pause":
            await self.log_api_call(connection, query="rate_limit_pause", status="failed")
            raise RateLimitPaused("anthropic rate limit has reached the pause threshold.")
        return result

    async def fetch_priority_targets(self, connection, *, tier: str | None, limit: int) -> list[dict[str, Any]]:
        targets: list[dict[str, Any]] = []
        seen: set[uuid.UUID] = set()

        async def add_rows(query: str, *args) -> None:
            rows = await connection.fetch(query, *args)
            for row in rows:
                row_dict = dict(row)
                restaurant_id = row_dict["id"]
                if restaurant_id in seen:
                    continue
                seen.add(restaurant_id)
                targets.append(row_dict)
                if len(targets) >= limit:
                    return

        if tier in (None, "A"):
            await add_rows(
                """
                SELECT r.id, r.name, r.address, r.google_place_id, r.needs_refresh
                FROM restaurants r
                WHERE EXISTS (
                    SELECT 1 FROM menu_items mi WHERE mi.restaurant_id = r.id
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM cuisine_tags ct
                    WHERE ct.restaurant_id = r.id
                      AND ct.source = 'llm_normalized'
                )
                ORDER BY r.updated_at NULLS FIRST, r.created_at NULLS FIRST
                LIMIT $1
                """,
                limit,
            )

        if tier is None and len(targets) < limit:
            await add_rows(
                """
                SELECT r.id, r.name, r.address, r.google_place_id, r.needs_refresh
                FROM restaurants r
                WHERE EXISTS (
                    SELECT 1 FROM dietary_attributes da WHERE da.restaurant_id = r.id
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM cuisine_tags ct WHERE ct.restaurant_id = r.id
                )
                ORDER BY r.updated_at NULLS FIRST, r.created_at NULLS FIRST
                LIMIT $1
                """,
                limit,
            )

        if tier is None and len(targets) < limit:
            await add_rows(
                """
                SELECT r.id, r.name, r.address, r.google_place_id, r.needs_refresh
                FROM restaurants r
                WHERE r.needs_refresh = TRUE
                ORDER BY r.updated_at NULLS FIRST, r.created_at NULLS FIRST
                LIMIT $1
                """,
                limit,
            )

        return targets[:limit]

    async def fetch_restaurant_context(self, connection, restaurant_id: uuid.UUID) -> dict[str, Any]:
        restaurant = await connection.fetchrow(
            """
            SELECT id, name, address, google_place_id, website, phone,
                   spice_level_estimate, cuisine_detail, needs_refresh
            FROM restaurants
            WHERE id = $1
            """,
            restaurant_id,
        )
        if restaurant is None:
            raise ValueError(f"Restaurant {restaurant_id} not found.")

        cuisine_rows = await connection.fetch(
            """
            SELECT tag, source, derived_from_menu
            FROM cuisine_tags
            WHERE restaurant_id = $1
            ORDER BY source, tag
            """,
            restaurant_id,
        )
        menu_rows = await connection.fetch(
            """
            SELECT name, description
            FROM menu_items
            WHERE restaurant_id = $1
            ORDER BY name
            LIMIT $2
            """,
            restaurant_id,
            MAX_MENU_ITEMS,
        )
        dietary_rows = await connection.fetch(
            """
            SELECT dietary_type, value, confidence_tier, source, source_url, notes
            FROM dietary_attributes
            WHERE restaurant_id = $1
            ORDER BY confidence_tier NULLS LAST, source
            """,
            restaurant_id,
        )

        return {
            "restaurant": dict(restaurant),
            "cuisine_tags": [dict(row) for row in cuisine_rows],
            "menu_items": [dict(row) for row in menu_rows],
            "dietary_attributes": [dict(row) for row in dietary_rows],
        }

    def build_prompt(self, context: dict[str, Any]) -> str:
        restaurant = context["restaurant"]
        payload = {
            "restaurant_name": restaurant["name"],
            "address": restaurant["address"],
            "existing_cuisine_tags": context["cuisine_tags"],
            "menu_items": context["menu_items"],
            "existing_dietary_attributes": context["dietary_attributes"],
        }
        return (
            "Normalize and enrich this restaurant record.\n"
            "Return JSON only with this exact top-level shape:\n"
            '{'
            '"normalized_tags": ["tag1", "tag2"], '
            '"inferred_dietary": ['
            '{"type": "halal", "value": "unknown", "confidence_tier": 5, "reasoning": "..."}'
            '], '
            '"spice_level": 3, '
            '"cuisine_detail": "lebanese", '
            '"data_issues": ["issue"]'
            '}\n'
            "Rules:\n"
            "- Deduplicate and normalize cuisine tags to short snake_case tags.\n"
            "- Infer dietary attributes only from the provided evidence.\n"
            "- Use confidence tiers 4-5 for inferred dietary values.\n"
            "- spice_level must be an integer 1-5.\n"
            "- cuisine_detail should be a specific cuisine or null.\n"
            "- data_issues should contain only obvious issues, otherwise return [].\n"
            f"Restaurant data:\n{json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)}"
        )

    async def request_normalization(self, connection, *, restaurant_id: uuid.UUID, prompt: str) -> dict[str, Any]:
        for attempt in range(1, 4):
            await self.check_anthropic_rate_limit(connection)
            try:
                message = await self.client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=1200,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                await self.log_api_call(
                    connection,
                    query=f"normalize|restaurant_id={restaurant_id}|attempt={attempt}|error={type(exc).__name__}",
                    status="failed",
                )
                if attempt == 3:
                    raise
                await self.sleep(2 ** (attempt - 1))
                continue

            await self.log_api_call(
                connection,
                query=f"normalize|restaurant_id={restaurant_id}|attempt={attempt}",
                status="success",
                records_updated=1,
            )
            return extract_json_object(extract_message_text(message))

        raise RuntimeError("Anthropic request failed after maximum retries.")

    async def fetch_existing_cuisine_rows(
        self,
        connection,
        restaurant_ids: Sequence[uuid.UUID],
    ) -> set[tuple[uuid.UUID, str, str]]:
        if not restaurant_ids:
            return set()
        rows = await connection.fetch(
            """
            SELECT restaurant_id, tag, source
            FROM cuisine_tags
            WHERE restaurant_id = ANY($1::uuid[])
            """,
            list(restaurant_ids),
        )
        return {(row["restaurant_id"], row["tag"], row["source"]) for row in rows}

    async def upsert_cuisine_tags(self, connection, rows: list[tuple], stats: NormalizerStats) -> None:
        if not rows:
            return
        existing_rows = await self.fetch_existing_cuisine_rows(connection, sorted({row[0] for row in rows}))
        stats.cuisine_tags_added += sum(1 for row in rows if (row[0], row[1], row[2]) not in existing_rows)
        query = """
            INSERT INTO cuisine_tags (restaurant_id, tag, source, derived_from_menu)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT DO NOTHING
        """
        for batch in chunked(rows):
            await connection.executemany(query, batch)

    async def fetch_existing_dietary_by_type(
        self,
        connection,
        restaurant_id: uuid.UUID,
    ) -> dict[str, list[dict[str, Any]]]:
        rows = await connection.fetch(
            """
            SELECT dietary_type, value, confidence_tier, source, source_url, notes
            FROM dietary_attributes
            WHERE restaurant_id = $1
            """,
            restaurant_id,
        )
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            result.setdefault(row["dietary_type"], []).append(dict(row))
        return result

    async def insert_dietary_attributes(self, connection, rows: list[tuple], stats: NormalizerStats) -> None:
        if not rows:
            return
        query = """
            INSERT INTO dietary_attributes (
                id, restaurant_id, dietary_type, value, confidence_tier,
                source, source_url, notes, fetched_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        for batch in chunked(rows):
            await connection.executemany(query, batch)
            stats.dietary_attributes_added += len(batch)

    async def update_restaurant(
        self,
        connection,
        *,
        restaurant: dict[str, Any],
        spice_level: int | None,
        cuisine_detail: str | None,
        has_menu_items: bool,
        has_dietary_attribute: bool,
        has_high_confidence_dietary: bool,
        stats: NormalizerStats,
    ) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        score = compute_data_quality_score(
            has_menu_items=has_menu_items,
            has_google_place_id=bool(restaurant.get("google_place_id")),
            has_contact_or_website=bool(restaurant.get("website") or restaurant.get("phone")),
            has_dietary_attribute=has_dietary_attribute,
            has_high_confidence_dietary=has_high_confidence_dietary,
            verified_recently=True,
        )
        await connection.execute(
            """
            UPDATE restaurants
            SET spice_level_estimate = COALESCE($2, spice_level_estimate),
                cuisine_detail = COALESCE($3, cuisine_detail),
                data_quality_score = GREATEST(data_quality_score, $4),
                last_verified_at = $5,
                updated_at = $6,
                needs_refresh = FALSE
            WHERE id = $1
            """,
            restaurant["id"],
            spice_level,
            cuisine_detail,
            score,
            now,
            now,
        )
        stats.restaurants_updated += 1

    async def log_data_issues(
        self,
        connection,
        *,
        restaurant_id: uuid.UUID,
        issues: Sequence[str],
    ) -> None:
        for issue in issues:
            await self.log_api_call(
                connection,
                query=f"data_issue|restaurant_id={restaurant_id}|issue={issue}",
                status="partial",
            )

    async def persist_normalization(
        self,
        connection,
        *,
        context: dict[str, Any],
        payload: dict[str, Any],
        stats: NormalizerStats,
    ) -> None:
        restaurant = context["restaurant"]
        restaurant_id = restaurant["id"]
        has_menu_items = bool(context["menu_items"])
        normalized_tags = normalize_tags(payload.get("normalized_tags"))
        cuisine_rows = [
            (restaurant_id, tag, "llm_normalized", has_menu_items)
            for tag in normalized_tags
        ]
        await self.upsert_cuisine_tags(connection, cuisine_rows, stats)

        existing_dietary = await self.fetch_existing_dietary_by_type(connection, restaurant_id)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        dietary_rows: list[tuple] = []
        for row in payload.get("inferred_dietary") or []:
            dietary_type = normalize_tag(row.get("type") or "")
            if not dietary_type:
                continue
            try:
                confidence_tier = int(row.get("confidence_tier"))
            except (TypeError, ValueError):
                stats.malformed_responses += 1
                await self.log_api_call(
                    connection,
                    query=f"malformed_confidence_tier|restaurant_id={restaurant_id}|type={row.get('type')}",
                    status="failed",
                )
                continue
            existing_rows = existing_dietary.get(dietary_type, [])
            if any(
                existing.get("confidence_tier") is not None
                and int(existing["confidence_tier"]) < confidence_tier
                for existing in existing_rows
            ):
                continue
            source_url = f"anthropic://{ANTHROPIC_MODEL}"
            value = str(row.get("value") or "unknown")
            notes = str(row.get("reasoning") or "") or None
            if any(
                existing.get("source") == "llm_inferred"
                and existing.get("value") == value
                and existing.get("confidence_tier") == confidence_tier
                for existing in existing_rows
            ):
                continue
            dietary_rows.append(
                (
                    uuid.uuid4(),
                    restaurant_id,
                    dietary_type,
                    value,
                    confidence_tier,
                    "llm_inferred",
                    source_url,
                    notes,
                    now,
                )
            )
            existing_dietary.setdefault(dietary_type, []).append(
                {
                    "dietary_type": dietary_type,
                    "value": value,
                    "confidence_tier": confidence_tier,
                    "source": "llm_inferred",
                }
            )
        await self.insert_dietary_attributes(connection, dietary_rows, stats)

        all_dietary = context["dietary_attributes"] + [
            {"confidence_tier": row[4]} for row in dietary_rows
        ]
        has_any_dietary = bool(all_dietary)
        has_high_confidence_dietary = any(
            entry.get("confidence_tier") is not None and int(entry["confidence_tier"]) <= 2
            for entry in all_dietary
        )
        cuisine_detail = payload.get("cuisine_detail")
        if cuisine_detail is not None:
            cuisine_detail = normalize_tag(str(cuisine_detail))
        await self.update_restaurant(
            connection,
            restaurant=restaurant,
            spice_level=clamp_spice_level(payload.get("spice_level")),
            cuisine_detail=cuisine_detail,
            has_menu_items=has_menu_items,
            has_dietary_attribute=has_any_dietary,
            has_high_confidence_dietary=has_high_confidence_dietary,
            stats=stats,
        )
        data_issues = [str(issue) for issue in (payload.get("data_issues") or []) if str(issue).strip()]
        await self.log_data_issues(connection, restaurant_id=restaurant_id, issues=data_issues)

    async def normalize(self, *, tier: str | None = None, limit: int = 50) -> NormalizerStats:
        stats = NormalizerStats()
        async with self.pool.acquire() as connection:
            await self.ensure_restaurant_columns(connection)
            targets = await self.fetch_priority_targets(connection, tier=tier, limit=limit)

        for batch_number, batch in enumerate(chunked(targets, BATCH_SIZE), start=1):
            async with self.pool.acquire() as connection:
                await self.ensure_restaurant_columns(connection)
                for restaurant in batch:
                    context = await self.fetch_restaurant_context(connection, restaurant["id"])
                    prompt = self.build_prompt(context)
                    try:
                        payload = await self.request_normalization(
                            connection,
                            restaurant_id=restaurant["id"],
                            prompt=prompt,
                        )
                    except json.JSONDecodeError:
                        stats.malformed_responses += 1
                        await self.log_api_call(
                            connection,
                            query=f"malformed_json|restaurant_id={restaurant['id']}",
                            status="failed",
                        )
                        continue
                    except ValueError:
                        stats.malformed_responses += 1
                        await self.log_api_call(
                            connection,
                            query=f"malformed_json|restaurant_id={restaurant['id']}",
                            status="failed",
                        )
                        continue

                    stats.api_calls += 1
                    async with connection.transaction():
                        await self.persist_normalization(
                            connection,
                            context=context,
                            payload=payload,
                            stats=stats,
                        )
                    stats.restaurants_processed += 1

            stats.batches_processed += 1
            print(
                f"Processed batch {batch_number}: "
                f"{stats.restaurants_processed} restaurants total."
            )

        return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anthropic-backed restaurant data normalizer")
    parser.add_argument("--tier", choices=["A"])
    parser.add_argument("--limit", type=int, default=50)
    return parser


async def run_cli(args: argparse.Namespace) -> NormalizerStats:
    async with LLMNormalizer() as normalizer:
        return await normalizer.normalize(tier=args.tier, limit=args.limit)


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        stats = await run_cli(args)
    except RateLimitPaused as exc:
        print(str(exc))
        return

    print(
        "LLM normalization complete:",
        {
            "restaurants_processed": stats.restaurants_processed,
            "restaurants_updated": stats.restaurants_updated,
            "cuisine_tags_added": stats.cuisine_tags_added,
            "dietary_attributes_added": stats.dietary_attributes_added,
            "api_calls": stats.api_calls,
            "malformed_responses": stats.malformed_responses,
            "batches_processed": stats.batches_processed,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
