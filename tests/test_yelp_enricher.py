from __future__ import annotations

import sys
import uuid
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.yelp_enricher import RateLimitPaused, YelpEnricher


pytestmark = pytest.mark.asyncio


class FakeConnection:
    def __init__(self) -> None:
        self.restaurants: dict[uuid.UUID, dict] = {}
        self.menu_presence: set[uuid.UUID] = set()
        self.cuisine_tags: set[tuple[uuid.UUID, str, str]] = set()
        self.dietary_attributes: list[dict] = []
        self.logs: list[dict] = []

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split())
        if "INSERT INTO data_fetch_log" in normalized:
            self.logs.append(
                {
                    "id": args[0],
                    "source": "yelp",
                    "query": args[1],
                    "status": args[2],
                    "records_added": args[4],
                    "records_updated": args[5],
                }
            )
            return "INSERT 0 1"
        return "OK"

    async def executemany(self, query: str, rows):
        normalized = " ".join(query.split())
        if "UPDATE restaurants SET" in normalized:
            for row in rows:
                restaurant = self.restaurants[row[0]]
                restaurant["yelp_id"] = restaurant.get("yelp_id") or row[1]
                if restaurant.get("rating_score") is None:
                    restaurant["rating_score"] = row[2]
                if restaurant.get("rating_count") is None:
                    restaurant["rating_count"] = row[3]
                if restaurant.get("price_level") is None:
                    restaurant["price_level"] = row[4]
                if restaurant.get("phone") is None:
                    restaurant["phone"] = row[5]
                if restaurant.get("website") is None:
                    restaurant["website"] = row[6]
                restaurant["yelp_url"] = restaurant.get("yelp_url") or row[7]
                restaurant["data_quality_score"] = max(restaurant.get("data_quality_score", 0.0), row[8])
                restaurant["last_verified_at"] = row[9]
                restaurant["updated_at"] = row[10]
            return
        if "INSERT INTO cuisine_tags" in normalized:
            for row in rows:
                self.cuisine_tags.add((row[0], row[1], row[2]))
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
                    }
                )

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.split())
        if "FROM restaurants" in normalized and "WHERE yelp_id IS NULL" in normalized:
            limit = args[-1]
            city = args[0] if len(args) == 3 else None
            rows = []
            for restaurant in self.restaurants.values():
                if restaurant.get("yelp_id") is not None:
                    continue
                if not restaurant.get("name") or not restaurant.get("address"):
                    continue
                if city and city.lower() not in (restaurant.get("city") or "").lower() and city.lower() not in (restaurant.get("address") or "").lower():
                    continue
                rows.append(dict(restaurant))
            return rows[:limit]
        if "SELECT DISTINCT restaurant_id FROM menu_items" in normalized:
            restaurant_ids = set(args[0])
            return [{"restaurant_id": rid} for rid in self.menu_presence if rid in restaurant_ids]
        if "SELECT restaurant_id, dietary_type FROM dietary_attributes" in normalized:
            restaurant_ids = set(args[0])
            return [
                {"restaurant_id": row["restaurant_id"], "dietary_type": row["dietary_type"]}
                for row in self.dietary_attributes
                if row["restaurant_id"] in restaurant_ids
            ]
        if "SELECT restaurant_id, MIN(confidence_tier) AS best_tier" in normalized:
            restaurant_ids = set(args[0])
            grouped: dict[uuid.UUID, int] = {}
            for row in self.dietary_attributes:
                if row["restaurant_id"] not in restaurant_ids:
                    continue
                tier = row["confidence_tier"]
                current = grouped.get(row["restaurant_id"])
                grouped[row["restaurant_id"]] = tier if current is None else min(current, tier)
            return [
                {"restaurant_id": restaurant_id, "best_tier": best_tier}
                for restaurant_id, best_tier in grouped.items()
            ]
        if "SELECT restaurant_id, tag, source FROM cuisine_tags" in normalized:
            restaurant_ids = set(args[0])
            return [
                {"restaurant_id": rid, "tag": tag, "source": source}
                for rid, tag, source in self.cuisine_tags
                if rid in restaurant_ids
            ]
        return []

    async def fetchval(self, query: str, *args):
        normalized = " ".join(query.split())
        if "SELECT COUNT(*) FROM data_fetch_log" in normalized:
            source = args[0]
            return sum(
                1
                for row in self.logs
                if row["source"] == source and row["status"] in {"success", "partial"}
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


@pytest.fixture
def fake_connection() -> FakeConnection:
    return FakeConnection()


def build_transport(handler):
    return httpx.MockTransport(handler)


def seed_restaurant(fake_connection: FakeConnection, *, name: str = "Test Tacos", city: str = "Dallas") -> uuid.UUID:
    restaurant_id = uuid.uuid4()
    fake_connection.restaurants[restaurant_id] = {
        "id": restaurant_id,
        "name": name,
        "address": f"123 Main St, {city}, TX 75201",
        "city": city,
        "google_place_id": None,
        "yelp_id": None,
        "price_level": None,
        "rating_score": None,
        "rating_count": None,
        "phone": None,
        "website": None,
        "yelp_url": None,
        "data_quality_score": 0.0,
    }
    return restaurant_id


async def test_successful_match_and_enrichment(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("YELP_API_KEY", "test-key")
    restaurant_id = seed_restaurant(fake_connection)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/businesses/search":
            return httpx.Response(200, json={"businesses": [{"id": "yelp-123", "name": "Test Tacos"}]})
        if request.url.path == "/v3/businesses/yelp-123":
            return httpx.Response(
                200,
                json={
                    "id": "yelp-123",
                    "rating": 4.5,
                    "review_count": 120,
                    "price": "$$",
                    "phone": "+12145551111",
                    "url": "https://yelp.example/test-tacos",
                    "categories": [{"alias": "mexican", "title": "Mexican"}],
                },
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    async with YelpEnricher(
        pool=FakePool(fake_connection),
        client=httpx.AsyncClient(transport=build_transport(handler)),
    ) as enricher:
        stats = await enricher.enrich(limit=10)

    assert stats.restaurants_updated == 1
    assert fake_connection.restaurants[restaurant_id]["yelp_id"] == "yelp-123"
    assert fake_connection.restaurants[restaurant_id]["rating_score"] == 4.5
    assert fake_connection.restaurants[restaurant_id]["rating_count"] == 120
    assert fake_connection.restaurants[restaurant_id]["price_level"] == 2
    assert fake_connection.restaurants[restaurant_id]["phone"] == "+12145551111"
    assert fake_connection.restaurants[restaurant_id]["website"] is None
    assert fake_connection.restaurants[restaurant_id]["yelp_url"] == "https://yelp.example/test-tacos"
    assert (restaurant_id, "mexican", "yelp") in fake_connection.cuisine_tags


async def test_low_confidence_match_is_rejected(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("YELP_API_KEY", "test-key")
    restaurant_id = seed_restaurant(fake_connection, name="Test Tacos")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/businesses/search":
            return httpx.Response(200, json={"businesses": [{"id": "yelp-999", "name": "Completely Different Bistro"}]})
        raise AssertionError("Details endpoint should not be called for low-confidence matches.")

    async with YelpEnricher(
        pool=FakePool(fake_connection),
        client=httpx.AsyncClient(transport=build_transport(handler)),
    ) as enricher:
        stats = await enricher.enrich(limit=10)

    assert stats.low_confidence_rejections == 1
    assert fake_connection.restaurants[restaurant_id]["yelp_id"] is None
    assert any(log["status"] == "partial" and "low_confidence_reject" in log["query"] for log in fake_connection.logs)


async def test_dietary_attribute_extraction(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("YELP_API_KEY", "test-key")
    restaurant_id = seed_restaurant(fake_connection, name="Green Garden")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v3/businesses/search":
            return httpx.Response(200, json={"businesses": [{"id": "yelp-green", "name": "Green Garden"}]})
        if request.url.path == "/v3/businesses/yelp-green":
            return httpx.Response(
                200,
                json={
                    "id": "yelp-green",
                    "categories": [
                        {"alias": "vegan", "title": "Vegan"},
                        {"alias": "halal", "title": "Halal"},
                        {"alias": "vegetarian", "title": "Vegetarian"},
                        {"alias": "kosher", "title": "Kosher"},
                    ],
                    "url": "https://yelp.example/green-garden",
                },
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    async with YelpEnricher(
        pool=FakePool(fake_connection),
        client=httpx.AsyncClient(transport=build_transport(handler)),
    ) as enricher:
        stats = await enricher.enrich(limit=10)

    found_types = {row["dietary_type"] for row in fake_connection.dietary_attributes if row["restaurant_id"] == restaurant_id}
    assert stats.dietary_attributes_added == 4
    assert found_types == {"vegan", "halal", "vegetarian", "kosher"}
    assert all(row["confidence_tier"] == 3 for row in fake_connection.dietary_attributes)


async def test_rate_limit_pause(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("YELP_API_KEY", "test-key")
    seed_restaurant(fake_connection)

    async def pause_rate_limit(connection, source):
        class Result:
            status = "pause"
            calls_today = 480
            daily_limit = 500
        return Result()

    import agents.yelp_enricher as yelp_module

    monkeypatch.setattr(yelp_module, "check_rate_limit", pause_rate_limit)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP should not be called when rate limit is paused.")

    async with YelpEnricher(
        pool=FakePool(fake_connection),
        client=httpx.AsyncClient(transport=build_transport(handler)),
    ) as enricher:
        with pytest.raises(RateLimitPaused):
            await enricher.enrich(limit=10)

    assert any(log["query"] == "rate_limit_pause" for log in fake_connection.logs)
