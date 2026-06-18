from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.llm_normalizer import LLMNormalizer


pytestmark = pytest.mark.asyncio


class FakeMessageText:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [FakeMessageText(text)]


class FakeMessagesAPI:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def create(self, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return FakeMessage(response)


class FakeAnthropicClient:
    def __init__(self, responses: list[str]) -> None:
        self.messages = FakeMessagesAPI(responses)


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.restaurants: dict[uuid.UUID, dict] = {}
        self.cuisine_tags: set[tuple[uuid.UUID, str, str, bool]] = set()
        self.menu_items: dict[uuid.UUID, list[dict]] = {}
        self.dietary_attributes: list[dict] = []
        self.logs: list[dict] = []

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split())
        if "ALTER TABLE restaurants" in normalized:
            return "ALTER TABLE"
        if "INSERT INTO data_fetch_log" in normalized:
            self.logs.append(
                {
                    "source": "anthropic",
                    "query": args[1],
                    "status": args[2],
                    "records_added": args[4],
                    "records_updated": args[5],
                }
            )
            return "INSERT 0 1"
        if "UPDATE restaurants SET" in normalized:
            restaurant = self.restaurants[args[0]]
            if args[1] is not None:
                restaurant["spice_level_estimate"] = args[1]
            if args[2] is not None:
                restaurant["cuisine_detail"] = args[2]
            restaurant["data_quality_score"] = max(restaurant.get("data_quality_score", 0.0), args[3])
            restaurant["last_verified_at"] = args[4]
            restaurant["updated_at"] = args[5]
            restaurant["needs_refresh"] = False
            return "UPDATE 1"
        return "OK"

    async def executemany(self, query: str, rows):
        normalized = " ".join(query.split())
        if "INSERT INTO cuisine_tags" in normalized:
            for row in rows:
                self.cuisine_tags.add((row[0], row[1], row[2], row[3]))
            return
        if "INSERT INTO dietary_attributes" in normalized:
            for row in rows:
                self.dietary_attributes.append(
                    {
                        "id": row[0],
                        "restaurant_id": row[1],
                        "dietary_type": row[2],
                        "value": row[3],
                        "confidence_tier": row[4],
                        "source": row[5],
                        "source_url": row[6],
                        "notes": row[7],
                    }
                )

    async def fetchrow(self, query: str, *args):
        normalized = " ".join(query.split())
        if "FROM restaurants WHERE id = $1" in normalized:
            restaurant = self.restaurants.get(args[0])
            return dict(restaurant) if restaurant else None
        return None

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.split())
        if "ct.source = 'llm_normalized'" in normalized:
            limit = args[0]
            rows = []
            for restaurant in self.restaurants.values():
                restaurant_id = restaurant["id"]
                has_menu = bool(self.menu_items.get(restaurant_id))
                has_llm_tag = any(
                    tag_row[0] == restaurant_id and tag_row[2] == "llm_normalized"
                    for tag_row in self.cuisine_tags
                )
                if has_menu and not has_llm_tag:
                    rows.append(dict(restaurant))
            return rows[:limit]
        if "SELECT r.id, r.name, r.address, r.google_place_id, r.needs_refresh FROM restaurants r" in normalized and "dietary_attributes da" in normalized:
            limit = args[0]
            rows = []
            for restaurant in self.restaurants.values():
                restaurant_id = restaurant["id"]
                has_dietary = any(row["restaurant_id"] == restaurant_id for row in self.dietary_attributes)
                has_cuisine = any(tag_row[0] == restaurant_id for tag_row in self.cuisine_tags)
                if has_dietary and not has_cuisine:
                    rows.append(dict(restaurant))
            return rows[:limit]
        if "WHERE r.needs_refresh = TRUE" in normalized:
            limit = args[0]
            return [dict(row) for row in self.restaurants.values() if row.get("needs_refresh")][:limit]
        if "SELECT tag, source, derived_from_menu FROM cuisine_tags" in normalized:
            restaurant_id = args[0]
            return [
                {"tag": tag, "source": source, "derived_from_menu": derived}
                for rid, tag, source, derived in self.cuisine_tags
                if rid == restaurant_id
            ]
        if "SELECT name, description FROM menu_items" in normalized:
            restaurant_id = args[0]
            limit = args[1]
            return self.menu_items.get(restaurant_id, [])[:limit]
        if "SELECT dietary_type, value, confidence_tier, source, source_url, notes FROM dietary_attributes WHERE restaurant_id = $1 ORDER BY confidence_tier NULLS LAST, source" in normalized:
            restaurant_id = args[0]
            return [dict(row) for row in self.dietary_attributes if row["restaurant_id"] == restaurant_id]
        if "SELECT restaurant_id, tag, source FROM cuisine_tags" in normalized:
            restaurant_ids = set(args[0])
            return [
                {"restaurant_id": rid, "tag": tag, "source": source}
                for rid, tag, source, _derived in self.cuisine_tags
                if rid in restaurant_ids
            ]
        if "SELECT dietary_type, value, confidence_tier, source, source_url, notes FROM dietary_attributes WHERE restaurant_id = $1" in normalized:
            restaurant_id = args[0]
            return [dict(row) for row in self.dietary_attributes if row["restaurant_id"] == restaurant_id]
        return []

    async def fetchval(self, query: str, *args):
        normalized = " ".join(query.split())
        if "SELECT COUNT(*) FROM data_fetch_log" in normalized:
            source = args[0]
            return sum(
                1 for row in self.logs if row["source"] == source and row["status"] in {"success", "partial"}
            )
        return None


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)

    async def close(self) -> None:
        return None


def seed_restaurant(fake_connection: FakeConnection, *, name: str = "Cedars Kitchen") -> uuid.UUID:
    restaurant_id = uuid.uuid4()
    fake_connection.restaurants[restaurant_id] = {
        "id": restaurant_id,
        "name": name,
        "address": "123 Main St, Dallas, TX 75201",
        "google_place_id": "google-1",
        "website": None,
        "phone": None,
        "spice_level_estimate": None,
        "cuisine_detail": None,
        "needs_refresh": True,
        "data_quality_score": 0.2,
    }
    fake_connection.menu_items[restaurant_id] = [
        {"name": "Chicken Shawarma", "description": "Garlic sauce and pickles"},
        {"name": "Spicy Hummus", "description": "Chili oil and paprika"},
    ]
    fake_connection.cuisine_tags.add((restaurant_id, "mediterranean", "yelp", False))
    return restaurant_id


async def test_successful_normalization_and_tag_upsert(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_connection = FakeConnection()
    restaurant_id = seed_restaurant(fake_connection)
    response = """
    {
      "normalized_tags": ["Mediterranean", "Lebanese"],
      "inferred_dietary": [],
      "spice_level": 3,
      "cuisine_detail": "Lebanese",
      "data_issues": []
    }
    """

    async with LLMNormalizer(
        pool=FakePool(fake_connection),
        client=FakeAnthropicClient([response]),
    ) as normalizer:
        stats = await normalizer.normalize(limit=1)

    assert stats.restaurants_processed == 1
    assert stats.cuisine_tags_added == 2
    assert (restaurant_id, "mediterranean", "llm_normalized", True) in fake_connection.cuisine_tags
    assert (restaurant_id, "lebanese", "llm_normalized", True) in fake_connection.cuisine_tags


async def test_dietary_inference_respects_higher_confidence_sources(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_connection = FakeConnection()
    restaurant_id = seed_restaurant(fake_connection)
    fake_connection.dietary_attributes.append(
        {
            "id": uuid.uuid4(),
            "restaurant_id": restaurant_id,
            "dietary_type": "halal",
            "value": "true",
            "confidence_tier": 1,
            "source": "zabihah",
            "source_url": "https://zabihah.example/cedars",
            "notes": "Verified",
        }
    )
    response = """
    {
      "normalized_tags": ["Mediterranean"],
      "inferred_dietary": [
        {"type": "halal", "value": "unknown", "confidence_tier": 5, "reasoning": "Menu does not confirm meat sourcing."},
        {"type": "vegan_friendly", "value": "partial", "confidence_tier": 4, "reasoning": "Several mezze items appear vegan."}
      ],
      "spice_level": 2,
      "cuisine_detail": "Lebanese",
      "data_issues": []
    }
    """

    async with LLMNormalizer(
        pool=FakePool(fake_connection),
        client=FakeAnthropicClient([response]),
    ) as normalizer:
        stats = await normalizer.normalize(limit=1)

    inserted_types = [row["dietary_type"] for row in fake_connection.dietary_attributes if row["source"] == "llm_inferred"]
    assert stats.dietary_attributes_added == 1
    assert inserted_types == ["vegan_friendly"]
    assert sum(1 for row in fake_connection.dietary_attributes if row["dietary_type"] == "halal") == 1


async def test_spice_level_storage(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_connection = FakeConnection()
    restaurant_id = seed_restaurant(fake_connection, name="Fire Bowl")
    response = """
    {
      "normalized_tags": ["Sichuan"],
      "inferred_dietary": [],
      "spice_level": 5,
      "cuisine_detail": "sichuan",
      "data_issues": ["menu items appear much spicier than broad tags suggest"]
    }
    """

    async with LLMNormalizer(
        pool=FakePool(fake_connection),
        client=FakeAnthropicClient([response]),
    ) as normalizer:
        await normalizer.normalize(limit=1)

    restaurant = fake_connection.restaurants[restaurant_id]
    assert restaurant["spice_level_estimate"] == 5
    assert restaurant["cuisine_detail"] == "sichuan"
    assert any("data_issue|restaurant_id=" in log["query"] for log in fake_connection.logs)


async def test_malformed_json_response_handling(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake_connection = FakeConnection()
    restaurant_id = seed_restaurant(fake_connection)
    response = "not valid json"

    async with LLMNormalizer(
        pool=FakePool(fake_connection),
        client=FakeAnthropicClient([response]),
    ) as normalizer:
        stats = await normalizer.normalize(limit=1)

    assert stats.malformed_responses == 1
    assert stats.restaurants_processed == 0
    assert fake_connection.restaurants[restaurant_id]["spice_level_estimate"] is None
    assert any("malformed_json" in log["query"] for log in fake_connection.logs)
