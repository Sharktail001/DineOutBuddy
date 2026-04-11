from __future__ import annotations

import sys
import uuid
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.google_places import GooglePlacesAgent, RateLimitPaused, build_restaurant_id


pytestmark = pytest.mark.asyncio


class FakeConnection:
    def __init__(self) -> None:
        self.restaurants: dict[uuid.UUID, dict] = {}
        self.menu_presence: set[uuid.UUID] = set()
        self.cuisine_tags: set[tuple[uuid.UUID, str, str]] = set()
        self.logs: list[dict] = []
        self.city_centers: dict[str, tuple[float, float]] = {}

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split())
        if "INSERT INTO data_fetch_log" in normalized:
            self.logs.append(
                {
                    "id": args[0],
                    "source": "google_places",
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
        if "INSERT INTO restaurants" in normalized:
            for row in rows:
                restaurant_id = row[0]
                self.restaurants[restaurant_id] = {
                    "id": restaurant_id,
                    "name": row[1],
                    "address": row[2],
                    "lat": row[3],
                    "lng": row[4],
                    "city": row[5],
                    "zip": row[6],
                    "price_level": row[7],
                    "google_place_id": row[8],
                    "website": row[11],
                    "phone": row[12],
                    "hours": row[13],
                    "rating_score": row[14],
                    "rating_count": row[15],
                    "data_quality_score": row[16],
                    "needs_refresh": row[18],
                }
            return
        if "INSERT INTO cuisine_tags" in normalized:
            for row in rows:
                self.cuisine_tags.add((row[0], row[1], row[2]))

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.split())
        if "SELECT id, google_place_id, website, phone, hours FROM restaurants" in normalized:
            google_place_ids = set(args[0])
            return [
                row
                for row in self.restaurants.values()
                if row.get("google_place_id") in google_place_ids
            ]
        if "SELECT DISTINCT restaurant_id FROM menu_items" in normalized:
            restaurant_ids = set(args[0])
            return [
                {"restaurant_id": restaurant_id}
                for restaurant_id in self.menu_presence
                if restaurant_id in restaurant_ids
            ]
        if "SELECT id FROM restaurants WHERE id = ANY" in normalized:
            restaurant_ids = set(args[0])
            return [{"id": restaurant_id} for restaurant_id in self.restaurants if restaurant_id in restaurant_ids]
        if "SELECT restaurant_id, tag, source FROM cuisine_tags" in normalized:
            restaurant_ids = set(args[0])
            return [
                {"restaurant_id": restaurant_id, "tag": tag, "source": source}
                for restaurant_id, tag, source in self.cuisine_tags
                if restaurant_id in restaurant_ids
            ]
        if "SELECT id, google_place_id FROM restaurants" in normalized and "WHERE id = ANY" in normalized:
            restaurant_ids = set(args[0])
            return [
                {"id": restaurant_id, "google_place_id": row["google_place_id"]}
                for restaurant_id, row in self.restaurants.items()
                if restaurant_id in restaurant_ids and row.get("google_place_id")
            ]
        if "SELECT id, google_place_id FROM restaurants" in normalized and "google_place_id IS NOT NULL" in normalized:
            limit = args[0]
            rows = []
            for restaurant_id, row in self.restaurants.items():
                if row.get("google_place_id") and (
                    row.get("hours") is None
                    or row.get("website") is None
                    or row.get("phone") is None
                    or row.get("price_level") is None
                    or row.get("rating_score") is None
                    or row.get("rating_count") is None
                ):
                    rows.append({"id": restaurant_id, "google_place_id": row["google_place_id"]})
            return rows[:limit]
        return []

    async def fetchrow(self, query: str, *args):
        normalized = " ".join(query.split())
        if "SELECT AVG(lat)::float AS lat, AVG(lng)::float AS lng FROM restaurants WHERE city ILIKE" in normalized:
            city = args[0]
            if city in self.city_centers:
                lat, lng = self.city_centers[city]
                return {"lat": lat, "lng": lng}
            return {"lat": None, "lng": None}
        if "SELECT AVG(lat)::float AS lat, AVG(lng)::float AS lng FROM restaurants WHERE address ILIKE" in normalized:
            return {"lat": None, "lng": None}
        return None

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


async def test_discovery_inserts_new_restaurants_and_tags(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/places:searchNearby"
        return httpx.Response(
            200,
            json={
                "places": [
                    {
                        "id": "place-123",
                        "displayName": {"text": "Test Tacos"},
                        "formattedAddress": "123 Main St, Dallas, TX 75201",
                        "location": {"latitude": 32.77, "longitude": -96.79},
                        "priceLevel": "PRICE_LEVEL_2",
                        "rating": 4.6,
                        "userRatingCount": 220,
                        "nationalPhoneNumber": "+1 214-555-1111",
                        "websiteUri": "https://testtacos.example",
                        "regularOpeningHours": {"weekdayDescriptions": ["Mon: 9:00 AM - 9:00 PM"]},
                        "types": ["mexican_restaurant", "restaurant"],
                    }
                ]
            },
        )

    async with GooglePlacesAgent(
        pool=FakePool(fake_connection),
        client=httpx.AsyncClient(transport=build_transport(handler)),
    ) as agent:
        stats = await agent.discovery(lat=32.7767, lng=-96.7970, radius=10000)

    restaurant_id = build_restaurant_id("place-123")
    assert stats.restaurants_added == 1
    assert restaurant_id in fake_connection.restaurants
    assert fake_connection.restaurants[restaurant_id]["name"] == "Test Tacos"
    assert fake_connection.restaurants[restaurant_id]["website"] == "https://testtacos.example"
    assert any(tag == "mexican_restaurant" for _, tag, _ in fake_connection.cuisine_tags)
    assert any(log["source"] == "google_places" for log in fake_connection.logs)


async def test_enrichment_updates_incomplete_restaurants(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    restaurant_id = uuid.uuid4()
    fake_connection.restaurants[restaurant_id] = {
        "id": restaurant_id,
        "name": "Needs Details",
        "address": "Old Address",
        "lat": 32.70,
        "lng": -96.80,
        "city": "Dallas",
        "zip": "75201",
        "price_level": None,
        "google_place_id": "place-999",
        "website": None,
        "phone": None,
        "hours": None,
        "rating_score": None,
        "rating_count": None,
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/places/place-999"
        return httpx.Response(
            200,
            json={
                "id": "place-999",
                "displayName": {"text": "Needs Details"},
                "formattedAddress": "500 Elm St, Dallas, TX 75202",
                "location": {"latitude": 32.781, "longitude": -96.801},
                "priceLevel": "PRICE_LEVEL_3",
                "rating": 4.8,
                "userRatingCount": 400,
                "nationalPhoneNumber": "+1 214-555-9999",
                "websiteUri": "https://details.example",
                "regularOpeningHours": {"weekdayDescriptions": ["Tue: 10:00 AM - 10:00 PM"]},
                "types": ["steak_house", "restaurant"],
            },
        )

    async with GooglePlacesAgent(
        pool=FakePool(fake_connection),
        client=httpx.AsyncClient(transport=build_transport(handler)),
    ) as agent:
        stats = await agent.enrichment(limit=10)

    assert stats.restaurants_updated == 1
    assert fake_connection.restaurants[restaurant_id]["website"] == "https://details.example"
    assert fake_connection.restaurants[restaurant_id]["phone"] == "+1 214-555-9999"
    assert fake_connection.restaurants[restaurant_id]["hours"] is not None
    assert fake_connection.restaurants[restaurant_id]["price_level"] == 3
    assert fake_connection.restaurants[restaurant_id]["rating_score"] == 4.8
    assert any(tag == "steak_house" for _, tag, _ in fake_connection.cuisine_tags)


async def test_rate_limit_pause_raises_and_logs(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")
    # Set daily limit to 1 so a single prior successful call saturates the quota.
    monkeypatch.setenv("GOOGLE_PLACES_DAILY_LIMIT", "1")

    fake_connection.logs.append({
        "id": uuid.uuid4(),
        "source": "google_places",
        "query": "prior_call",
        "status": "success",
        "records_added": 0,
        "records_updated": 0,
    })

    with pytest.raises(RateLimitPaused):
        async with GooglePlacesAgent(
            pool=FakePool(fake_connection),
            client=httpx.AsyncClient(transport=build_transport(lambda r: httpx.Response(200, json={}))),
        ) as agent:
            await agent.discovery(lat=32.7767, lng=-96.7970)

    failed_logs = [log for log in fake_connection.logs if log["status"] == "failed"]
    assert failed_logs, "Expected a failed log entry for the rate limit pause"
    assert all(log["source"] == "google_places" for log in failed_logs)


async def test_retry_on_429_then_succeeds(fake_connection: FakeConnection, monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", "test-key")

    attempt = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return httpx.Response(429)
        return httpx.Response(
            200,
            json={
                "places": [
                    {
                        "id": "place-retry",
                        "displayName": {"text": "Retry Bistro"},
                        "formattedAddress": "1 Retry Lane, Dallas, TX 75201",
                        "location": {"latitude": 32.77, "longitude": -96.79},
                        "priceLevel": "PRICE_LEVEL_1",
                        "rating": 4.2,
                        "userRatingCount": 80,
                        "nationalPhoneNumber": "+1 214-555-0000",
                        "websiteUri": "https://retrybistro.example",
                        "regularOpeningHours": None,
                        "types": ["american_restaurant"],
                    }
                ]
            },
        )

    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async with GooglePlacesAgent(
        pool=FakePool(fake_connection),
        client=httpx.AsyncClient(transport=build_transport(handler)),
        sleep=fake_sleep,
    ) as agent:
        stats = await agent.discovery(lat=32.7767, lng=-96.7970)

    assert attempt == 2, "Expected exactly 2 HTTP attempts (1 failure + 1 success)"
    assert len(sleep_calls) == 1, "Expected exactly one sleep between retry attempts"
    assert sleep_calls[0] == 1, "Expected 2^(attempt-1) = 1 second sleep after first failure"
    assert stats.restaurants_added == 1
    restaurant_id = build_restaurant_id("place-retry")
    assert fake_connection.restaurants[restaurant_id]["name"] == "Retry Bistro"
    failed_logs = [log for log in fake_connection.logs if log["status"] == "failed"]
    assert failed_logs, "Expected a failed log entry for the 429 attempt"
