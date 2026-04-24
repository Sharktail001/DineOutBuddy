from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.zabihah_scraper import (
    MatchCandidate,
    ZabihahListing,
    ZabihahScraper,
    ZabihahStats,
    parse_detail_page,
    parse_search_results,
    resolve_match,
)


class FakeConnection:
    def __init__(self) -> None:
        self.restaurants: dict[uuid.UUID, dict] = {}
        self.menu_presence: set[uuid.UUID] = set()
        self.dietary_attributes: list[dict] = []
        self.logs: list[dict] = []

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split())
        if "INSERT INTO data_fetch_log" in normalized:
            self.logs.append(
                {
                    "id": args[0],
                    "source": "zabihah",
                    "query": args[1],
                    "status": args[2],
                    "records_added": args[4],
                    "records_updated": args[5],
                }
            )
        return "OK"

    async def executemany(self, query: str, rows):
        normalized = " ".join(query.split())
        if "INSERT INTO restaurants" in normalized:
            for row in rows:
                current = self.restaurants.get(row[0], {})
                self.restaurants[row[0]] = {
                    "id": row[0],
                    "name": row[1],
                    "address": row[2],
                    "city": row[5],
                    "zip": row[6],
                    "google_place_id": row[8],
                    "data_quality_score": max(current.get("data_quality_score", 0.0), row[14]),
                    "needs_refresh": row[16],
                    "website": current.get("website"),
                    "phone": current.get("phone"),
                }
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

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.split())
        if "SELECT DISTINCT restaurant_id FROM menu_items" in normalized:
            restaurant_ids = set(args[0])
            return [{"restaurant_id": rid} for rid in self.menu_presence if rid in restaurant_ids]
        if "SELECT restaurant_id, dietary_type, source_url" in normalized:
            restaurant_ids = set(args[0])
            return [
                {
                    "restaurant_id": row["restaurant_id"],
                    "dietary_type": row["dietary_type"],
                    "source_url": row["source_url"],
                }
                for row in self.dietary_attributes
                if row["restaurant_id"] in restaurant_ids
            ]
        if "SELECT id FROM restaurants WHERE id = ANY" in normalized:
            requested = set(args[0])
            return [{"id": rid} for rid in self.restaurants if rid in requested]
        if "has_contact" in normalized:
            row = self.restaurants.get(args[0])
            if row is None:
                return []
            has_contact = bool(row.get("website")) or bool(row.get("phone"))
            return [{"has_contact": has_contact}]
        return []


def seed_restaurant(
    fake_connection: FakeConnection,
    *,
    name: str = "Zabiha Grill",
    address: str = "123 Main St, Dallas, TX 75201",
    city: str = "Dallas",
    zip_code: str = "75201",
    google_place_id: str | None = "google-1",
    website: str | None = None,
    phone: str | None = None,
) -> uuid.UUID:
    restaurant_id = uuid.uuid4()
    fake_connection.restaurants[restaurant_id] = {
        "id": restaurant_id,
        "name": name,
        "address": address,
        "city": city,
        "zip": zip_code,
        "google_place_id": google_place_id,
        "data_quality_score": 0.2,
        "needs_refresh": False,
        "website": website,
        "phone": phone,
    }
    return restaurant_id


def build_listing(
    *,
    name: str = "Zabiha Grill",
    address: str = "123 Main St, Dallas, TX 75201",
    is_zabiha: bool = True,
    halal_status: str = "Fully halal",
) -> ZabihahListing:
    notes = (
        "Community says all meat is hand slaughtered. Zabiha certified."
        if is_zabiha
        else "Community says all meat is hand slaughtered."
    )
    return ZabihahListing(
        name=name,
        address=address,
        street="123 Main St",
        city="Dallas",
        state="TX",
        zip_code="75201",
        is_zabiha=is_zabiha,
        halal_status=halal_status,
        user_notes=notes,
        zabihah_url="https://www.zabihah.com/restaurants/abc123/zabiha-grill-dallas-tx",
    )


@pytest.mark.asyncio
async def test_successful_match_and_dietary_insert() -> None:
    """Matched zabiha restaurant: two dietary rows (halal + zabiha), both tier 1."""
    fake_connection = FakeConnection()
    restaurant_id = seed_restaurant(fake_connection)
    listing = build_listing()  # is_zabiha=True, halal_status="Fully halal"
    candidate = MatchCandidate(
        restaurant_id=restaurant_id,
        name="Zabiha Grill",
        address="123 Main St, Dallas, TX 75201",
        city="Dallas",
        zip_code="75201",
        google_place_id="google-1",
        data_quality_score=0.2,
    )
    match_result = resolve_match(listing, [candidate])

    async with ZabihahScraper(pool=object()) as scraper:
        stats = ZabihahStats()
        await scraper.persist_listing(
            fake_connection,
            listing=listing,
            match_result=match_result,
            existing_candidates={restaurant_id: candidate},
            stats=stats,
        )

    stored_rows = [row for row in fake_connection.dietary_attributes if row["restaurant_id"] == restaurant_id]
    assert match_result.matched is True
    assert stats.restaurants_updated == 1
    assert stats.dietary_attributes_added == 2
    assert {row["dietary_type"] for row in stored_rows} == {"halal", "zabiha"}
    assert fake_connection.restaurants[restaurant_id]["needs_refresh"] is True
    halal_row = next(r for r in stored_rows if r["dietary_type"] == "halal")
    zabiha_row = next(r for r in stored_rows if r["dietary_type"] == "zabiha")
    assert halal_row["confidence_tier"] == 1
    assert halal_row["value"] == "true"
    assert zabiha_row["confidence_tier"] == 1
    assert zabiha_row["value"] == "true"


def test_parse_search_and_detail_pages() -> None:
    """Search-page container selector and detail-page extraction across all badge variants."""

    # --- Search results page ---
    search_html = """
    <html><body>
      <nav>
        <a href="/about">Our story</a>
        <a href="/faq">FAQ</a>
        <a href="/careers">Careers</a>
        <a href="/search?type=restaurants&sort=distance">Halal places near me</a>
      </nav>
      <main>
        <div class="grid">
          <a class="block" href="/restaurants/uuid1/restaurant-one-dallas-tx">
            <div><h3>Restaurant One</h3><p>123 Main St, Dallas, TX</p></div>
          </a>
          <a class="block" href="/restaurants/uuid2/restaurant-two-dallas-tx">
            <div><h3>Restaurant Two</h3><p>456 Oak Ave, Dallas, TX</p></div>
          </a>
        </div>
      </main>
      <footer>
        <a href="/privacy">Privacy policy</a>
        <a href="/terms">Terms of service</a>
      </footer>
    </body></html>
    """
    results = parse_search_results(search_html)
    assert len(results) == 2, f"Expected 2 listings, got {len(results)}: {results}"
    names = [r[0] for r in results]
    assert "Restaurant One" in names
    assert "Restaurant Two" in names
    # address extracted from the sole <p> inside each card
    one = next(r for r in results if r[0] == "Restaurant One")
    assert one[1] == "123 Main St, Dallas, TX"
    # URL prefixed with ZABIHAH_BASE
    assert one[2] == "https://www.zabihah.com/restaurants/uuid1/restaurant-one-dallas-tx"

    # --- Detail page: zabiha (fully halal + "zabiha" in FAQ notes) ---
    detail_zabiha_html = """
    <html><head>
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":["Restaurant","LocalBusiness"],
         "name":"Restaurant One",
         "address":{"@type":"PostalAddress","streetAddress":"123 Main St",
                    "addressLocality":"Dallas","addressRegion":"TX","postalCode":"75201"}}
      </script>
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"FAQPage",
         "mainEntity":[{"@type":"Question","name":"Is Restaurant One halal?",
           "acceptedAnswer":{"@type":"Answer","text":"All food is zabiha certified."}}]}
      </script>
    </head><body><main>
      <span class="text-mint-800">$$</span>
      <span class="text-mint-800">Fully halal</span>
    </main></body></html>
    """
    listing = parse_detail_page(
        detail_zabiha_html,
        "Restaurant One",
        "123 Main St, Dallas, TX",
        "https://www.zabihah.com/restaurants/uuid1/restaurant-one-dallas-tx",
    )
    assert listing is not None
    assert listing.is_zabiha is True
    assert listing.halal_status == "Fully halal"
    assert listing.zip_code == "75201"          # extracted from JSON-LD address
    assert listing.user_notes is not None
    assert "zabiha" in listing.user_notes.lower()

    # --- Detail page: fully halal, no zabiha in notes ---
    detail_halal_html = """
    <html><head>
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"FAQPage",
         "mainEntity":[{"@type":"Question","name":"Is Restaurant Two halal?",
           "acceptedAnswer":{"@type":"Answer","text":"Halal certified. No alcohol served."}}]}
      </script>
    </head><body><main>
      <span class="text-mint-800">$</span>
      <span class="text-mint-800">Fully halal</span>
    </main></body></html>
    """
    listing2 = parse_detail_page(
        detail_halal_html,
        "Restaurant Two",
        "456 Oak Ave, Dallas, TX",
        "https://www.zabihah.com/restaurants/uuid2/restaurant-two-dallas-tx",
    )
    assert listing2 is not None
    assert listing2.is_zabiha is False
    assert listing2.halal_status == "Fully halal"

    # --- Detail page: partially halal ---
    detail_partial_html = """
    <html><body><main>
      <span class="text-mint-800">Partially halal</span>
    </main></body></html>
    """
    listing3 = parse_detail_page(
        detail_partial_html,
        "Restaurant Three",
        None,
        "https://www.zabihah.com/restaurants/uuid3/restaurant-three-dallas-tx",
    )
    assert listing3 is not None
    assert listing3.halal_status == "Partially halal"
    assert listing3.is_zabiha is False

    # --- Detail page: no halal badge → None ---
    assert parse_detail_page(
        "<html><body><p>Page not found</p></body></html>",
        "X",
        None,
        "https://www.zabihah.com/restaurants/uuid4/x",
    ) is None


@pytest.mark.asyncio
async def test_unmatched_restaurant_insertion() -> None:
    """Unmatched fully-halal restaurant: new record with needs_refresh=True, tier-2 halal row."""
    fake_connection = FakeConnection()
    listing = build_listing(
        name="New Halal Spot",
        address="789 Oak Ave, Dallas, TX 75202",
        is_zabiha=False,
        halal_status="Fully halal",
    )
    match_result = resolve_match(listing, [])

    async with ZabihahScraper(pool=object()) as scraper:
        stats = ZabihahStats()
        await scraper.persist_listing(
            fake_connection,
            listing=listing,
            match_result=match_result,
            existing_candidates={},
            stats=stats,
        )

    assert match_result.matched is False
    assert stats.restaurants_added == 1
    assert stats.dietary_attributes_added == 1
    created = next(iter(fake_connection.restaurants.values()))
    assert created["name"] == "New Halal Spot"
    assert created["needs_refresh"] is True
    row = fake_connection.dietary_attributes[0]
    assert row["dietary_type"] == "halal"
    assert row["confidence_tier"] == 2
    assert row["value"] == "true"


def test_fuzzy_match_rejection_below_threshold() -> None:
    """Name similarity below 0.75 must never produce a match."""
    restaurant_id = uuid.uuid4()
    listing = build_listing(name="Alpha Grill", address="123 Main St, Dallas, TX 75201")
    candidate = MatchCandidate(
        restaurant_id=restaurant_id,
        name="Beta Seafood House",
        address="999 Different Rd, Houston, TX 77002",
        city="Houston",
        zip_code="77002",
        google_place_id=None,
        data_quality_score=0.1,
    )
    match_result = resolve_match(listing, [candidate])

    assert match_result.matched is False
    assert match_result.restaurant_id is None
    assert match_result.reason in {"name_below_threshold", "address_mismatch"}
