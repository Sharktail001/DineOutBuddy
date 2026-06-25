from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from db.supabase_client import create_pool
from recommendation.group_recommender import group_recommend
from recommendation.knn_recommender import recommend


pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = await create_pool()
    yield
    await pool.close()
    pool = None


app = FastAPI(title="DineOutBuddy API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TasteProfile(BaseModel):
    cuisine_preferences: list[str] = Field(default_factory=list)
    spice_tolerance: int | None = None
    price_preference: int | None = None
    dietary_requirements: list[str] = Field(default_factory=list)
    dietary_preferences: list[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    cuisine_preferences: list[str] = Field(default_factory=list)
    spice_tolerance: int | None = None
    price_preference: int | None = None
    dietary_requirements: list[str] = Field(default_factory=list)
    dietary_preferences: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)


class RecommendationResponse(BaseModel):
    restaurant_id: str
    name: str
    address: str | None
    cuisine_tags: list[str]
    final_score: float
    similarity_score: float
    compatibility_score: float


class GroupRecommendRequest(BaseModel):
    profiles: list[TasteProfile] = Field(..., min_length=2)
    limit: int = Field(default=10, ge=1, le=100)


class PersonScores(BaseModel):
    final_score: float
    compatibility_score: float
    similarity_score: float


class GroupRecommendationResponse(BaseModel):
    restaurant_id: str
    name: str
    group_score: float
    per_person_scores: dict[int, PersonScores]


class DietaryAttributeResponse(BaseModel):
    dietary_type: str
    value: str
    confidence_tier: int
    source: str | None


class MenuItemResponse(BaseModel):
    id: str
    category: str | None
    name: str
    description: str | None
    price: float | None


class RestaurantDetailResponse(BaseModel):
    id: str
    name: str
    address: str | None
    lat: float | None
    lng: float | None
    city: str | None
    zip: str | None
    price_level: int | None
    rating_score: float | None
    rating_count: int | None
    data_quality_score: float | None
    cuisine_tags: list[str]
    dietary_attributes: list[DietaryAttributeResponse]
    menu_items: list[MenuItemResponse]


class PaginatedMenuResponse(BaseModel):
    items: list[MenuItemResponse]
    page: int
    page_size: int
    total: int


class HealthResponse(BaseModel):
    status: str
    restaurants: int
    vectors: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT "
            "(SELECT count(*) FROM restaurants) AS restaurants, "
            "(SELECT count(*) FROM feature_vectors) AS vectors"
        )
    return HealthResponse(
        status="ok",
        restaurants=row["restaurants"],
        vectors=row["vectors"],
    )


@app.post("/recommend", response_model=list[RecommendationResponse])
async def post_recommend(body: RecommendRequest):
    profile: dict[str, Any] = {
        "cuisine_preferences": body.cuisine_preferences,
        "spice_tolerance": body.spice_tolerance,
        "price_preference": body.price_preference,
        "dietary_requirements": body.dietary_requirements,
        "dietary_preferences": body.dietary_preferences,
    }
    results = await recommend(profile=profile, limit=body.limit, pool=pool)
    return [
        RecommendationResponse(
            restaurant_id=r.restaurant_id,
            name=r.name,
            address=r.address,
            cuisine_tags=r.cuisine_tags,
            final_score=r.final_score,
            similarity_score=r.similarity_score,
            compatibility_score=r.compatibility_score,
        )
        for r in results
    ]


@app.post("/recommend/group", response_model=list[GroupRecommendationResponse])
async def post_group_recommend(body: GroupRecommendRequest):
    profiles = [p.model_dump() for p in body.profiles]
    results = await group_recommend(profiles=profiles, limit=body.limit, pool=pool)
    return [
        GroupRecommendationResponse(
            restaurant_id=g.restaurant_id,
            name=g.name,
            group_score=g.group_score,
            per_person_scores={
                idx: PersonScores(**scores)
                for idx, scores in g.per_person_scores.items()
            },
        )
        for g in results
    ]


@app.get("/restaurants/{restaurant_id}", response_model=RestaurantDetailResponse)
async def get_restaurant(restaurant_id: UUID):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, address, lat, lng, city, zip, price_level, "
            "rating_score, rating_count, data_quality_score "
            "FROM restaurants WHERE id = $1",
            restaurant_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Restaurant not found")

        tag_rows = await conn.fetch(
            "SELECT tag FROM cuisine_tags WHERE restaurant_id = $1",
            restaurant_id,
        )
        dietary_rows = await conn.fetch(
            "SELECT dietary_type, value, confidence_tier, source "
            "FROM dietary_attributes WHERE restaurant_id = $1",
            restaurant_id,
        )
        menu_rows = await conn.fetch(
            "SELECT id, category, name, description, price "
            "FROM menu_items WHERE restaurant_id = $1 LIMIT 20",
            restaurant_id,
        )

    return RestaurantDetailResponse(
        id=str(row["id"]),
        name=row["name"],
        address=row["address"],
        lat=row["lat"],
        lng=row["lng"],
        city=row["city"],
        zip=row["zip"],
        price_level=row["price_level"],
        rating_score=row["rating_score"],
        rating_count=row["rating_count"],
        data_quality_score=row["data_quality_score"],
        cuisine_tags=[r["tag"] for r in tag_rows],
        dietary_attributes=[
            DietaryAttributeResponse(
                dietary_type=r["dietary_type"],
                value=r["value"],
                confidence_tier=r["confidence_tier"],
                source=r["source"],
            )
            for r in dietary_rows
        ],
        menu_items=[
            MenuItemResponse(
                id=str(r["id"]),
                category=r["category"],
                name=r["name"],
                description=r["description"],
                price=float(r["price"]) if r["price"] is not None else None,
            )
            for r in menu_rows
        ],
    )


@app.get("/restaurants/{restaurant_id}/menu", response_model=PaginatedMenuResponse)
async def get_restaurant_menu(
    restaurant_id: UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    async with pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM restaurants WHERE id = $1)",
            restaurant_id,
        )
        if not exists:
            raise HTTPException(status_code=404, detail="Restaurant not found")

        total = await conn.fetchval(
            "SELECT count(*) FROM menu_items WHERE restaurant_id = $1",
            restaurant_id,
        )
        offset = (page - 1) * page_size
        rows = await conn.fetch(
            "SELECT id, category, name, description, price "
            "FROM menu_items WHERE restaurant_id = $1 "
            "ORDER BY category, name LIMIT $2 OFFSET $3",
            restaurant_id,
            page_size,
            offset,
        )

    return PaginatedMenuResponse(
        items=[
            MenuItemResponse(
                id=str(r["id"]),
                category=r["category"],
                name=r["name"],
                description=r["description"],
                price=float(r["price"]) if r["price"] is not None else None,
            )
            for r in rows
        ],
        page=page,
        page_size=page_size,
        total=total,
    )
