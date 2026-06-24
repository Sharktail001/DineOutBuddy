from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommendation.vector_generator import (
    CUISINE_INDEX,
    DIETARY_TYPES,
    NUM_CUISINE_DIMS,
    NUM_SCALAR_DIMS,
    VECTOR_DIM,
    encode_restaurant_vector,
)
from recommendation.compatibility import (
    compute_dietary_compatibility,
)
from recommendation.knn_recommender import (
    KNN_WEIGHT,
    COMPATIBILITY_WEIGHT,
    Recommendation,
    build_query_vector,
    recommend,
)
from recommendation.group_recommender import (
    GroupRecommendation,
    group_recommend,
)


# ---------------------------------------------------------------------------
# vector_generator tests
# ---------------------------------------------------------------------------

class TestVectorGenerator:

    def test_vector_dimension(self):
        vec = encode_restaurant_vector(
            cuisine_tags=[],
            price_level=None,
            rating_score=None,
            spice_level=None,
            dietary_flags={},
            data_quality_score=None,
        )
        assert len(vec) == VECTOR_DIM
        assert VECTOR_DIM == NUM_CUISINE_DIMS + NUM_SCALAR_DIMS

    def test_cuisine_multihot_encoding(self):
        vec = encode_restaurant_vector(
            cuisine_tags=["italian", "pizza"],
            price_level=None,
            rating_score=None,
            spice_level=None,
            dietary_flags={},
            data_quality_score=None,
        )
        assert vec[CUISINE_INDEX["italian"]] == 1.0
        assert vec[CUISINE_INDEX["pizza"]] == 1.0
        assert vec[CUISINE_INDEX["chinese"]] == 0.0

    def test_unknown_cuisine_tag_ignored(self):
        vec = encode_restaurant_vector(
            cuisine_tags=["nonexistent_cuisine_xyz"],
            price_level=None,
            rating_score=None,
            spice_level=None,
            dietary_flags={},
            data_quality_score=None,
        )
        assert all(v == 0.0 for v in vec[:NUM_CUISINE_DIMS])

    def test_price_normalization(self):
        vec = encode_restaurant_vector(
            cuisine_tags=[],
            price_level=2,
            rating_score=None,
            spice_level=None,
            dietary_flags={},
            data_quality_score=None,
        )
        assert vec[NUM_CUISINE_DIMS] == pytest.approx(0.5)

    def test_rating_normalization(self):
        vec = encode_restaurant_vector(
            cuisine_tags=[],
            price_level=None,
            rating_score=3.5,
            spice_level=None,
            dietary_flags={},
            data_quality_score=None,
        )
        assert vec[NUM_CUISINE_DIMS + 1] == pytest.approx(0.7)

    def test_spice_normalization(self):
        vec = encode_restaurant_vector(
            cuisine_tags=[],
            price_level=None,
            rating_score=None,
            spice_level=5.0,
            dietary_flags={},
            data_quality_score=None,
        )
        assert vec[NUM_CUISINE_DIMS + 2] == pytest.approx(1.0)

    def test_dietary_flags_encoding(self):
        vec = encode_restaurant_vector(
            cuisine_tags=[],
            price_level=None,
            rating_score=None,
            spice_level=None,
            dietary_flags={"halal": True, "vegan": False, "kosher": True},
            data_quality_score=None,
        )
        offset = NUM_CUISINE_DIMS + 3
        assert vec[offset + 0] == 1.0  # halal
        assert vec[offset + 1] == 0.0  # vegan
        assert vec[offset + 2] == 1.0  # kosher
        assert vec[offset + 3] == 0.0  # vegetarian
        assert vec[offset + 4] == 0.0  # gluten_free

    def test_data_quality_passthrough(self):
        vec = encode_restaurant_vector(
            cuisine_tags=[],
            price_level=None,
            rating_score=None,
            spice_level=None,
            dietary_flags={},
            data_quality_score=0.85,
        )
        assert vec[NUM_CUISINE_DIMS + 8] == pytest.approx(0.85)

    def test_all_values_in_range(self):
        vec = encode_restaurant_vector(
            cuisine_tags=["mexican", "tacos", "seafood"],
            price_level=4,
            rating_score=5.0,
            spice_level=5.0,
            dietary_flags={d: True for d in DIETARY_TYPES},
            data_quality_score=1.0,
        )
        for v in vec:
            assert 0.0 <= v <= 1.0

    def test_none_values_default_to_zero(self):
        vec = encode_restaurant_vector(
            cuisine_tags=[],
            price_level=None,
            rating_score=None,
            spice_level=None,
            dietary_flags={},
            data_quality_score=None,
        )
        assert all(v == 0.0 for v in vec)


# ---------------------------------------------------------------------------
# compatibility tests
# ---------------------------------------------------------------------------

class TestCompatibility:

    def test_no_requirements_returns_full_score(self):
        score = compute_dietary_compatibility(
            dietary_requirements=[],
            dietary_preferences=[],
            restaurant_dietary=[],
        )
        assert score == 1.0

    def test_hard_requirement_missing_returns_zero(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[],
        )
        assert score == 0.0

    def test_hard_requirement_tier_5_returns_zero(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 5},
            ],
        )
        assert score == 0.0

    def test_hard_requirement_tier_6_returns_zero(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 6},
            ],
        )
        assert score == 0.0

    def test_hard_requirement_value_false_returns_zero(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "false", "confidence_tier": 1},
            ],
        )
        assert score == 0.0

    def test_hard_requirement_tier_1_full_score(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 1},
            ],
        )
        assert score == 1.0

    def test_hard_requirement_tier_2_full_score(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 2},
            ],
        )
        assert score == 1.0

    def test_hard_requirement_tier_3_partial_score(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 3},
            ],
        )
        assert score == pytest.approx(0.6)

    def test_hard_requirement_tier_4_partial_score(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 4},
            ],
        )
        assert score == pytest.approx(0.6)

    def test_soft_preference_contributes_positively(self):
        with_pref = compute_dietary_compatibility(
            dietary_requirements=[],
            dietary_preferences=["vegan"],
            restaurant_dietary=[
                {"dietary_type": "vegan", "value": "true", "confidence_tier": 1},
            ],
        )
        without_pref = compute_dietary_compatibility(
            dietary_requirements=[],
            dietary_preferences=["vegan"],
            restaurant_dietary=[],
        )
        assert with_pref > without_pref

    def test_mixed_requirements_and_preferences(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=["vegan"],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 1},
                {"dietary_type": "vegan", "value": "true", "confidence_tier": 2},
            ],
        )
        assert 0.0 < score <= 1.0

    def test_best_confidence_tier_used_when_duplicates(self):
        score = compute_dietary_compatibility(
            dietary_requirements=["halal"],
            dietary_preferences=[],
            restaurant_dietary=[
                {"dietary_type": "halal", "value": "true", "confidence_tier": 5},
                {"dietary_type": "halal", "value": "true", "confidence_tier": 2},
            ],
        )
        assert score == 1.0


# ---------------------------------------------------------------------------
# knn_recommender tests
# ---------------------------------------------------------------------------

class TestKnnRecommender:

    def test_build_query_vector_dimensions(self):
        profile = {
            "cuisine_preferences": ["italian", "mediterranean"],
            "spice_tolerance": 3,
            "price_preference": 2,
            "dietary_requirements": ["halal"],
            "dietary_preferences": ["vegan"],
        }
        vec = build_query_vector(profile)
        assert len(vec) == VECTOR_DIM

    def test_build_query_vector_encodes_cuisine(self):
        profile = {
            "cuisine_preferences": ["thai"],
            "spice_tolerance": None,
            "dietary_requirements": [],
        }
        vec = build_query_vector(profile)
        assert vec[CUISINE_INDEX["thai"]] == 1.0

    @pytest.mark.asyncio
    async def test_recommend_returns_sorted_by_final_score(self):
        rid1 = str(uuid.uuid4())
        rid2 = str(uuid.uuid4())
        rid3 = str(uuid.uuid4())

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        knn_rows = [
            {"restaurant_id": uuid.UUID(rid1), "name": "A", "address": "1 St",
             "similarity": 0.95, "cuisine_tags": ["italian"]},
            {"restaurant_id": uuid.UUID(rid2), "name": "B", "address": "2 St",
             "similarity": 0.80, "cuisine_tags": ["thai"]},
            {"restaurant_id": uuid.UUID(rid3), "name": "C", "address": "3 St",
             "similarity": 0.70, "cuisine_tags": ["mexican"]},
        ]

        dietary_rows = [
            {"restaurant_id": uuid.UUID(rid1), "dietary_type": "halal",
             "value": "true", "confidence_tier": 1},
            {"restaurant_id": uuid.UUID(rid2), "dietary_type": "halal",
             "value": "true", "confidence_tier": 3},
            {"restaurant_id": uuid.UUID(rid3), "dietary_type": "halal",
             "value": "true", "confidence_tier": 2},
        ]

        async def mock_fetch(query, *args):
            if "feature_vectors" in query:
                return knn_rows
            if "dietary_attributes" in query:
                return dietary_rows
            return []

        mock_conn.fetch = mock_fetch

        profile = {
            "cuisine_preferences": ["italian"],
            "dietary_requirements": ["halal"],
        }
        results = await recommend(profile=profile, limit=10, pool=mock_pool)

        assert len(results) == 3
        for i in range(len(results) - 1):
            assert results[i].final_score >= results[i + 1].final_score

    @pytest.mark.asyncio
    async def test_recommend_filters_zero_compatibility(self):
        rid1 = str(uuid.uuid4())
        rid2 = str(uuid.uuid4())

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        knn_rows = [
            {"restaurant_id": uuid.UUID(rid1), "name": "Halal Place", "address": "1 St",
             "similarity": 0.9, "cuisine_tags": ["mediterranean"]},
            {"restaurant_id": uuid.UUID(rid2), "name": "No Halal", "address": "2 St",
             "similarity": 0.95, "cuisine_tags": ["american"]},
        ]

        dietary_rows = [
            {"restaurant_id": uuid.UUID(rid1), "dietary_type": "halal",
             "value": "true", "confidence_tier": 1},
        ]

        async def mock_fetch(query, *args):
            if "feature_vectors" in query:
                return knn_rows
            if "dietary_attributes" in query:
                return dietary_rows
            return []

        mock_conn.fetch = mock_fetch

        profile = {
            "cuisine_preferences": ["mediterranean"],
            "dietary_requirements": ["halal"],
        }
        results = await recommend(profile=profile, limit=10, pool=mock_pool)

        assert len(results) == 1
        assert results[0].restaurant_id == rid1

    @pytest.mark.asyncio
    async def test_recommend_no_requirements_keeps_all(self):
        rid1 = str(uuid.uuid4())
        rid2 = str(uuid.uuid4())

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        knn_rows = [
            {"restaurant_id": uuid.UUID(rid1), "name": "A", "address": "1 St",
             "similarity": 0.9, "cuisine_tags": []},
            {"restaurant_id": uuid.UUID(rid2), "name": "B", "address": "2 St",
             "similarity": 0.8, "cuisine_tags": []},
        ]

        async def mock_fetch(query, *args):
            if "feature_vectors" in query:
                return knn_rows
            if "dietary_attributes" in query:
                return []
            return []

        mock_conn.fetch = mock_fetch

        profile = {"cuisine_preferences": ["italian"]}
        results = await recommend(profile=profile, limit=10, pool=mock_pool)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_final_score_uses_70_30_weighting(self):
        rid = str(uuid.uuid4())

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        similarity = 0.85
        knn_rows = [
            {"restaurant_id": uuid.UUID(rid), "name": "Test Place", "address": "1 St",
             "similarity": similarity, "cuisine_tags": ["italian"]},
        ]

        dietary_rows = [
            {"restaurant_id": uuid.UUID(rid), "dietary_type": "halal",
             "value": "true", "confidence_tier": 1},
        ]

        async def mock_fetch(query, *args):
            if "feature_vectors" in query:
                return knn_rows
            if "dietary_attributes" in query:
                return dietary_rows
            return []

        mock_conn.fetch = mock_fetch

        profile = {
            "cuisine_preferences": ["italian"],
            "dietary_requirements": ["halal"],
        }
        results = await recommend(profile=profile, limit=10, pool=mock_pool)

        assert len(results) == 1
        compat = results[0].compatibility_score
        expected = round(KNN_WEIGHT * similarity + COMPATIBILITY_WEIGHT * compat, 4)
        assert results[0].final_score == expected


# ---------------------------------------------------------------------------
# group_recommender tests
# ---------------------------------------------------------------------------

class TestGroupRecommender:

    @pytest.mark.asyncio
    async def test_group_eliminates_incompatible_for_any_member(self):
        rid_good = str(uuid.uuid4())
        rid_bad = str(uuid.uuid4())

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_pool.close = AsyncMock()

        knn_rows_all = [
            {"restaurant_id": uuid.UUID(rid_good), "name": "Good Place",
             "address": "1 St", "similarity": 0.85, "cuisine_tags": ["mediterranean"]},
            {"restaurant_id": uuid.UUID(rid_bad), "name": "Bad Place",
             "address": "2 St", "similarity": 0.90, "cuisine_tags": ["american"]},
        ]

        dietary_rows = [
            {"restaurant_id": uuid.UUID(rid_good), "dietary_type": "halal",
             "value": "true", "confidence_tier": 1},
            {"restaurant_id": uuid.UUID(rid_good), "dietary_type": "vegan",
             "value": "true", "confidence_tier": 2},
            {"restaurant_id": uuid.UUID(rid_bad), "dietary_type": "halal",
             "value": "true", "confidence_tier": 1},
        ]

        async def mock_fetch(query, *args):
            if "feature_vectors" in query:
                return knn_rows_all
            if "dietary_attributes" in query:
                return dietary_rows
            return []

        mock_conn.fetch = mock_fetch

        profiles = [
            {"dietary_requirements": ["halal"], "cuisine_preferences": ["mediterranean"]},
            {"dietary_requirements": ["vegan"], "cuisine_preferences": ["mediterranean"]},
        ]
        results = await group_recommend(profiles=profiles, limit=10, pool=mock_pool)

        assert len(results) == 1
        assert results[0].restaurant_id == rid_good

    @pytest.mark.asyncio
    async def test_group_scores_averaged(self):
        rid = str(uuid.uuid4())

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_pool.close = AsyncMock()

        knn_rows = [
            {"restaurant_id": uuid.UUID(rid), "name": "Shared Place",
             "address": "1 St", "similarity": 0.80, "cuisine_tags": ["italian"]},
        ]

        dietary_rows = [
            {"restaurant_id": uuid.UUID(rid), "dietary_type": "halal",
             "value": "true", "confidence_tier": 1},
        ]

        async def mock_fetch(query, *args):
            if "feature_vectors" in query:
                return knn_rows
            if "dietary_attributes" in query:
                return dietary_rows
            return []

        mock_conn.fetch = mock_fetch

        profiles = [
            {"dietary_requirements": ["halal"], "cuisine_preferences": ["italian"]},
            {"dietary_requirements": [], "cuisine_preferences": ["italian"]},
        ]
        results = await group_recommend(profiles=profiles, limit=10, pool=mock_pool)

        assert len(results) == 1
        assert results[0].per_person_scores is not None
        assert 0 in results[0].per_person_scores
        assert 1 in results[0].per_person_scores

    @pytest.mark.asyncio
    async def test_group_sorted_by_group_score(self):
        rid1 = str(uuid.uuid4())
        rid2 = str(uuid.uuid4())

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_pool.close = AsyncMock()

        knn_rows = [
            {"restaurant_id": uuid.UUID(rid1), "name": "Higher",
             "address": "1 St", "similarity": 0.95, "cuisine_tags": []},
            {"restaurant_id": uuid.UUID(rid2), "name": "Lower",
             "address": "2 St", "similarity": 0.60, "cuisine_tags": []},
        ]

        async def mock_fetch(query, *args):
            if "feature_vectors" in query:
                return knn_rows
            if "dietary_attributes" in query:
                return []
            return []

        mock_conn.fetch = mock_fetch

        profiles = [
            {"cuisine_preferences": ["italian"]},
            {"cuisine_preferences": ["italian"]},
        ]
        results = await group_recommend(profiles=profiles, limit=10, pool=mock_pool)

        assert len(results) == 2
        assert results[0].group_score >= results[1].group_score
