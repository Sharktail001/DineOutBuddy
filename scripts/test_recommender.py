import asyncio
from dotenv import load_dotenv
load_dotenv()

from recommendation.knn_recommender import recommend
from db.supabase_client import create_pool

async def main():
    pool = await create_pool()
    try:
        # Test 1: No dietary requirements
        print("--- Test 1: No dietary requirements ---")
        profile = {
            "cuisine_preferences": ["mediterranean", "american"],
            "spice_tolerance": 3,
            "price_preference": 2,
            "dietary_requirements": [],
            "dietary_preferences": []
        }
        results = await recommend(profile=profile, limit=5, pool=pool)
        if not results:
            print("No results returned.")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.name} | score: {r.final_score:.3f} | compat: {r.compatibility_score:.3f}")

        print()

        # Test 2: Halal requirement
        print("--- Test 2: Halal requirement ---")
        profile2 = {
            "cuisine_preferences": ["american", "mediterranean", "indian"],
            "spice_tolerance": 3,
            "price_preference": 2,
            "dietary_requirements": ["halal"],
            "dietary_preferences": []
        }
        results2 = await recommend(profile=profile2, limit=5, pool=pool)
        if not results2:
            print("No results — no halal restaurants found.")
        for i, r in enumerate(results2, 1):
            print(f"{i}. {r.name} | score: {r.final_score:.3f} | compat: {r.compatibility_score:.3f}")

    finally:
        await pool.close()

asyncio.run(main())
