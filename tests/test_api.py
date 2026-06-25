from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.knn_recommender import Recommendation
from recommendation.group_recommender import GroupRecommendation


def _make_mock_pool():
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_pool.close = AsyncMock()
    return mock_pool, mock_conn


@pytest.fixture()
def client():
    mock_pool, mock_conn = _make_mock_pool()

    with patch("api.main.create_pool", new_callable=AsyncMock, return_value=mock_pool):
        from api.main import app
        import api.main as api_module

        api_module.pool = mock_pool

        with TestClient(app, raise_server_exceptions=False) as c:
            c._mock_pool = mock_pool
            c._mock_conn = mock_conn
            yield c

        api_module.pool = None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_counts(self, client):
        client._mock_conn.fetchrow = AsyncMock(
            return_value={"restaurants": 500, "vectors": 480}
        )
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["restaurants"] == 500
        assert data["vectors"] == 480


# ---------------------------------------------------------------------------
# POST /recommend
# ---------------------------------------------------------------------------

class TestRecommend:

    def test_recommend_returns_list(self, client):
        rid = str(uuid.uuid4())
        mock_results = [
            Recommendation(
                restaurant_id=rid,
                name="Test Place",
                address="123 Main St",
                cuisine_tags=["italian"],
                final_score=0.95,
                compatibility_score=1.0,
                similarity_score=0.93,
            )
        ]

        with patch(
            "api.main.recommend",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            resp = client.post("/recommend", json={
                "cuisine_preferences": ["italian"],
                "dietary_requirements": ["halal"],
                "limit": 5,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["restaurant_id"] == rid
        assert data[0]["name"] == "Test Place"
        assert data[0]["final_score"] == 0.95
        assert data[0]["similarity_score"] == 0.93
        assert data[0]["compatibility_score"] == 1.0
        assert data[0]["cuisine_tags"] == ["italian"]

    def test_recommend_empty_results(self, client):
        with patch(
            "api.main.recommend",
            new_callable=AsyncMock,
            return_value=[],
        ):
            resp = client.post("/recommend", json={
                "cuisine_preferences": ["thai"],
            })

        assert resp.status_code == 200
        assert resp.json() == []

    def test_recommend_validation_error(self, client):
        resp = client.post("/recommend", json={
            "limit": 0,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /recommend/group
# ---------------------------------------------------------------------------

class TestGroupRecommend:

    def test_group_recommend_returns_list(self, client):
        rid = str(uuid.uuid4())
        mock_results = [
            GroupRecommendation(
                restaurant_id=rid,
                name="Group Place",
                group_score=0.88,
                per_person_scores={
                    0: {"final_score": 0.9, "compatibility_score": 1.0, "similarity_score": 0.85},
                    1: {"final_score": 0.86, "compatibility_score": 0.8, "similarity_score": 0.9},
                },
            )
        ]

        with patch(
            "api.main.group_recommend",
            new_callable=AsyncMock,
            return_value=mock_results,
        ):
            resp = client.post("/recommend/group", json={
                "profiles": [
                    {"dietary_requirements": ["halal"], "cuisine_preferences": []},
                    {"dietary_requirements": ["vegan"], "cuisine_preferences": []},
                ],
                "limit": 10,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["restaurant_id"] == rid
        assert data[0]["group_score"] == 0.88
        assert "0" in data[0]["per_person_scores"]
        assert "1" in data[0]["per_person_scores"]

    def test_group_recommend_requires_two_profiles(self, client):
        resp = client.post("/recommend/group", json={
            "profiles": [
                {"dietary_requirements": ["halal"], "cuisine_preferences": []},
            ],
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /restaurants/{restaurant_id}
# ---------------------------------------------------------------------------

class TestGetRestaurant:

    def test_get_restaurant_found(self, client):
        rid = uuid.uuid4()

        restaurant_row = {
            "id": rid, "name": "Deccan Grill", "address": "123 St",
            "lat": 32.97, "lng": -96.72, "city": "Dallas", "zip": "75080",
            "price_level": 2, "rating_score": 4.5, "rating_count": 120,
            "data_quality_score": 0.8,
        }
        tag_rows = [{"tag": "indian"}, {"tag": "halal"}]
        dietary_rows = [
            {"dietary_type": "halal", "value": "true",
             "confidence_tier": 2, "source": "zabihah_csv"},
        ]
        menu_rows = [
            {"id": uuid.uuid4(), "category": "Entrees", "name": "Biryani",
             "description": "Spiced rice", "price": 14.99},
        ]

        async def mock_fetchrow(query, *args):
            if "restaurants" in query:
                return restaurant_row
            return None

        async def mock_fetch(query, *args):
            if "cuisine_tags" in query:
                return tag_rows
            if "dietary_attributes" in query:
                return dietary_rows
            if "menu_items" in query:
                return menu_rows
            return []

        client._mock_conn.fetchrow = mock_fetchrow
        client._mock_conn.fetch = mock_fetch

        resp = client.get(f"/restaurants/{rid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Deccan Grill"
        assert data["cuisine_tags"] == ["indian", "halal"]
        assert len(data["dietary_attributes"]) == 1
        assert data["dietary_attributes"][0]["dietary_type"] == "halal"
        assert len(data["menu_items"]) == 1
        assert data["menu_items"][0]["name"] == "Biryani"

    def test_get_restaurant_not_found(self, client):
        client._mock_conn.fetchrow = AsyncMock(return_value=None)

        resp = client.get(f"/restaurants/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_get_restaurant_invalid_uuid(self, client):
        resp = client.get("/restaurants/not-a-uuid")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /restaurants/{restaurant_id}/menu
# ---------------------------------------------------------------------------

class TestGetMenu:

    def test_menu_paginated(self, client):
        rid = uuid.uuid4()

        async def mock_fetchval(query, *args):
            if "EXISTS" in query:
                return True
            if "count" in query:
                return 50
            return None

        menu_rows = [
            {"id": uuid.uuid4(), "category": "Appetizers", "name": "Hummus",
             "description": "Classic", "price": 7.99},
            {"id": uuid.uuid4(), "category": "Appetizers", "name": "Falafel",
             "description": None, "price": 8.99},
        ]

        client._mock_conn.fetchval = mock_fetchval
        client._mock_conn.fetch = AsyncMock(return_value=menu_rows)

        resp = client.get(f"/restaurants/{rid}/menu?page=1&page_size=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 50
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["items"]) == 2

    def test_menu_restaurant_not_found(self, client):
        client._mock_conn.fetchval = AsyncMock(return_value=False)

        resp = client.get(f"/restaurants/{uuid.uuid4()}/menu")
        assert resp.status_code == 404

    def test_menu_page_size_capped(self, client):
        resp = client.get(f"/restaurants/{uuid.uuid4()}/menu?page_size=200")
        assert resp.status_code == 422
