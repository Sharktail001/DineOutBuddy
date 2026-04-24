from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Sequence
from urllib.parse import quote

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from agents.csv_importer import chunked, compute_data_quality_score
from db.rate_limiter import RateLimitResult, check_rate_limit
from db.supabase_client import create_pool

load_dotenv()

NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
ZIP_CODE_PATTERN = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
STATE_ZIP_PATTERN = re.compile(r"\b([A-Z]{2})\s+(\d{5})(?:-\d{4})?\b")
STREET_HINT_PATTERN = re.compile(r"\d+\s+[A-Za-z0-9]")
SEARCH_URL_TEMPLATE = "https://www.zabihah.com/search?q={city}%2C+{state}&type=restaurants"
ZABIHAH_BASE = "https://www.zabihah.com"


class RateLimitPaused(Exception):
    pass


@dataclass(slots=True)
class ZabihahListing:
    name: str
    address: str | None
    street: str | None
    city: str | None
    state: str | None
    zip_code: str | None
    is_zabiha: bool
    halal_status: str       # "Fully halal" | "Partially halal" | ...
    user_notes: str | None
    zabihah_url: str


@dataclass(slots=True)
class MatchCandidate:
    restaurant_id: uuid.UUID
    name: str
    address: str | None
    city: str | None
    zip_code: str | None
    google_place_id: str | None
    data_quality_score: float | None = None


@dataclass(slots=True)
class ListingMatchResult:
    matched: bool
    restaurant_id: uuid.UUID | None
    reason: str
    name_similarity: float
    address_similarity: float


@dataclass(slots=True)
class ZabihahStats:
    restaurants_added: int = 0
    restaurants_updated: int = 0
    dietary_attributes_added: int = 0
    pages_scraped: int = 0
    listings_processed: int = 0
    matches_found: int = 0
    matches_rejected: int = 0


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split()).strip()


def normalize_for_match(value: str | None) -> str:
    return NON_ALNUM_PATTERN.sub("", (value or "").lower())


def sequence_similarity(left: str | None, right: str | None) -> float:
    return SequenceMatcher(None, normalize_for_match(left), normalize_for_match(right)).ratio()


def normalize_address_for_match(address: str | None) -> str:
    value = (address or "").lower()
    for source, target in {
        "street": "st",
        "st.": "st",
        "avenue": "ave",
        "ave.": "ave",
        "road": "rd",
        "rd.": "rd",
        "drive": "dr",
        "dr.": "dr",
        "boulevard": "blvd",
        "blvd.": "blvd",
        "lane": "ln",
        "ln.": "ln",
        "suite": "ste",
        "ste.": "ste",
    }.items():
        value = value.replace(source, target)
    return NON_ALNUM_PATTERN.sub("", value)


def address_similarity(left: str | None, right: str | None) -> float:
    return SequenceMatcher(
        None,
        normalize_address_for_match(left),
        normalize_address_for_match(right),
    ).ratio()


def split_address_parts(address: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    cleaned = normalize_text(address)
    if not cleaned:
        return None, None, None, None
    parts = [part.strip(" ,") for part in re.split(r"\s*\|\s*|\s{2,}|\s*\n\s*", cleaned) if part.strip(" ,")]

    street = None
    city = None
    state = None
    zip_code = None
    for part in parts:
        if street is None and STREET_HINT_PATTERN.search(part):
            street = part
        if city is None and street and part != street and not STATE_ZIP_PATTERN.search(part):
            city = part
        state_zip_match = STATE_ZIP_PATTERN.search(part)
        if state_zip_match:
            state = state_zip_match.group(1)
            zip_code = state_zip_match.group(2)

    if zip_code is None:
        zip_match = ZIP_CODE_PATTERN.search(cleaned)
        if zip_match:
            zip_code = zip_match.group(1)
    if street is None and parts:
        street = parts[0]
    if city is None and len(parts) >= 2:
        city = parts[1]
    return street, city, state, zip_code


def parse_search_results(html: str) -> list[tuple[str, str | None, str]]:
    """Return (name, address, detail_url) for each restaurant card on a search results page.

    Container selector ``a[href^="/restaurants/"]`` is URL-anchored: nav and footer
    links all use paths like /about, /faq, /login, /playlists/..., or external https://
    URLs — none can match this prefix.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str | None, str]] = []
    seen_hrefs: set[str] = set()
    for anchor in soup.select('a[href^="/restaurants/"]'):
        href = anchor.get("href", "")
        if not href or href in seen_hrefs:
            continue
        name_node = anchor.select_one("h3")
        if not name_node:
            continue
        name = normalize_text(name_node.get_text(" ", strip=True))
        if not name:
            continue
        addr_node = anchor.select_one("p")
        address = normalize_text(addr_node.get_text(" ", strip=True)) if addr_node else None
        results.append((name, address, f"{ZABIHAH_BASE}{href}"))
        seen_hrefs.add(href)
    return results


def parse_detail_page(
    html: str,
    name: str,
    search_address: str | None,
    zabihah_url: str,
) -> ZabihahListing | None:
    """Parse a restaurant detail page.

    Returns None when no halal status badge is found (page unavailable or
    non-halal restaurant erroneously linked).

    Halal status comes from the first ``span.text-mint-800`` that does not
    start with ``$`` (price badges share the same class and always start with
    a dollar sign).

    is_zabiha is True only when the badge reads "Fully halal" AND the word
    "zabiha" appears (case-insensitive) in the user notes.  The new site has
    no dedicated zabiha badge; zabiha status is expressed in free-text notes.

    Address with zip code is pulled from the Restaurant JSON-LD block.  The
    halal notes are pulled from the FAQPage JSON-LD block (the first question
    whose text contains "halal"), with a DOM fallback to ``p.leading-relaxed``.
    """
    soup = BeautifulSoup(html, "html.parser")

    full_address = search_address
    user_notes: str | None = None

    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        schema_type = data.get("@type", "")
        types = schema_type if isinstance(schema_type, list) else [schema_type]

        if any(t in ("Restaurant", "LocalBusiness") for t in types):
            addr = data.get("address", {})
            if isinstance(addr, dict):
                street_addr = addr.get("streetAddress", "")
                locality = addr.get("addressLocality", "")
                region = addr.get("addressRegion", "")
                postal = addr.get("postalCode", "")
                # Join as "Street, City, ST 12345" so STATE_ZIP_PATTERN matches.
                state_zip = f"{region} {postal}".strip() if (region or postal) else ""
                parts = [p for p in [street_addr, locality, state_zip] if p]
                if parts:
                    full_address = ", ".join(parts)

        if data.get("@type") == "FAQPage" and user_notes is None:
            for item in data.get("mainEntity", []):
                if "halal" in (item.get("name") or "").lower():
                    text = normalize_text((item.get("acceptedAnswer") or {}).get("text") or "")
                    if text:
                        user_notes = text
                        break

    if user_notes is None:
        notes_node = soup.select_one("p.leading-relaxed")
        if notes_node:
            user_notes = normalize_text(notes_node.get_text(" ", strip=True)) or None

    # First mint-800 span that is not a price badge.
    halal_status: str | None = None
    for span in soup.select("span.text-mint-800"):
        text = normalize_text(span.get_text(" ", strip=True))
        if text and not text.startswith("$"):
            halal_status = text
            break

    if halal_status is None:
        return None

    is_zabiha = (
        halal_status.lower() == "fully halal"
        and "zabiha" in (user_notes or "").lower()
    )

    street, city, state, zip_code = split_address_parts(full_address)

    return ZabihahListing(
        name=name,
        address=full_address,
        street=street,
        city=city,
        state=state,
        zip_code=zip_code,
        is_zabiha=is_zabiha,
        halal_status=halal_status,
        user_notes=user_notes,
        zabihah_url=zabihah_url,
    )


def resolve_match(
    listing: ZabihahListing,
    candidates: Sequence[MatchCandidate],
    *,
    name_threshold: float = 0.75,
    address_threshold: float = 0.55,
) -> ListingMatchResult:
    best = ListingMatchResult(False, None, "no_candidates", 0.0, 0.0)
    for candidate in candidates:
        name_score = sequence_similarity(listing.name, candidate.name)
        if name_score < name_threshold:
            if name_score > best.name_similarity:
                best = ListingMatchResult(False, None, "name_below_threshold", name_score, 0.0)
            continue
        addr_score = address_similarity(listing.address, candidate.address)
        if listing.zip_code and candidate.zip_code and listing.zip_code != candidate.zip_code:
            addr_score = 0.0
        if listing.city and candidate.city and listing.city.lower() != candidate.city.lower():
            addr_score = 0.0
        if listing.address and candidate.address and addr_score < address_threshold:
            if name_score > best.name_similarity or (
                abs(name_score - best.name_similarity) < 1e-9 and addr_score > best.address_similarity
            ):
                best = ListingMatchResult(False, None, "address_mismatch", name_score, addr_score)
            continue
        if name_score > best.name_similarity or (
            abs(name_score - best.name_similarity) < 1e-9 and addr_score > best.address_similarity
        ):
            best = ListingMatchResult(True, candidate.restaurant_id, "matched", name_score, addr_score)
    return best


class ZabihahScraper:
    def __init__(
        self,
        *,
        pool=None,
        sleep=asyncio.sleep,
        random_uniform=random.uniform,
    ) -> None:
        self.pool = pool
        self.sleep = sleep
        self.random_uniform = random_uniform
        self._owns_pool = pool is None

    async def __aenter__(self) -> "ZabihahScraper":
        if self.pool is None:
            self.pool = await create_pool()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_pool and self.pool is not None:
            await self.pool.close()

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
            ) VALUES ($1, 'zabihah', $2, $3, $4, $5, $6)
            """,
            uuid.uuid4(),
            query,
            status,
            now,
            records_added,
            records_updated,
        )

    async def check_zabihah_rate_limit(self, connection) -> RateLimitResult:
        result = await check_rate_limit(connection, "zabihah")
        if result.status == "warn":
            print(f"Warning: zabihah usage is at {result.calls_today}/{result.daily_limit} today.")
        if result.status == "pause":
            await self.log_api_call(connection, query="rate_limit_pause", status="failed")
            raise RateLimitPaused("zabihah rate limit has reached the pause threshold.")
        return result

    async def fetch_page_html(self, connection, url: str) -> str:
        await self.check_zabihah_rate_limit(connection)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for the Zabihah scraper. "
                "Install it with `pip install playwright` and `playwright install chromium`."
            ) from exc

        await self.sleep(self.random_uniform(2.0, 5.0))
        async with async_playwright() as playwright:
            for attempt in range(1, 4):
                try:
                    browser = await playwright.chromium.launch(headless=True)
                    try:
                        page = await browser.new_page()
                        await page.goto(url, wait_until="networkidle", timeout=45000)
                        html = await page.content()
                    finally:
                        await browser.close()
                    await self.log_api_call(
                        connection,
                        query=f"page_scrape|url={url}|attempt={attempt}",
                        status="success",
                        records_added=0,
                    )
                    return html
                except Exception:
                    await self.log_api_call(
                        connection,
                        query=f"page_scrape|url={url}|attempt={attempt}",
                        status="failed",
                    )
                    if attempt == 3:
                        raise
                    await self.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"Failed to scrape {url}")

    def build_search_url(self, city: str, state: str) -> str:
        return SEARCH_URL_TEMPLATE.format(
            city=quote(normalize_text(city), safe=""),
            state=quote(normalize_text(state), safe=""),
        )

    async def fetch_city_restaurants(self, connection, city: str, state: str) -> list[MatchCandidate]:
        rows = await connection.fetch(
            """
            SELECT id, name, address, city, zip, google_place_id, data_quality_score
            FROM restaurants
            WHERE name IS NOT NULL
              AND (city ILIKE $1 OR address ILIKE $2)
            """,
            city,
            f"%{city}%",
        )
        candidates: list[MatchCandidate] = []
        for row in rows:
            address = row["address"]
            row_state = None
            if address:
                match = STATE_ZIP_PATTERN.search(address)
                row_state = match.group(1) if match else None
            if row_state and row_state.lower() != state.lower():
                continue
            candidates.append(
                MatchCandidate(
                    restaurant_id=row["id"],
                    name=row["name"],
                    address=address,
                    city=row["city"],
                    zip_code=row["zip"],
                    google_place_id=row["google_place_id"],
                    data_quality_score=row["data_quality_score"],
                )
            )
        return candidates

    async def fetch_menu_presence(self, connection, restaurant_ids: Sequence[uuid.UUID]) -> set[uuid.UUID]:
        if not restaurant_ids:
            return set()
        rows = await connection.fetch(
            """
            SELECT DISTINCT restaurant_id
            FROM menu_items
            WHERE restaurant_id = ANY($1::uuid[])
            """,
            list(restaurant_ids),
        )
        return {row["restaurant_id"] for row in rows}

    async def fetch_existing_dietary(
        self, connection, restaurant_ids: Sequence[uuid.UUID]
    ) -> set[tuple[uuid.UUID, str, str]]:
        if not restaurant_ids:
            return set()
        rows = await connection.fetch(
            """
            SELECT restaurant_id, dietary_type, source_url
            FROM dietary_attributes
            WHERE restaurant_id = ANY($1::uuid[])
              AND source = 'zabihah'
            """,
            list(restaurant_ids),
        )
        return {(row["restaurant_id"], row["dietary_type"], row["source_url"] or "") for row in rows}

    async def fetch_has_contact(self, connection, restaurant_id: uuid.UUID) -> bool:
        rows = await connection.fetch(
            """
            SELECT (website IS NOT NULL OR phone IS NOT NULL) AS has_contact
            FROM restaurants
            WHERE id = $1
            """,
            restaurant_id,
        )
        return bool(rows[0]["has_contact"]) if rows else False

    async def upsert_restaurants(self, connection, restaurant_rows: list[tuple], stats: ZabihahStats) -> None:
        if not restaurant_rows:
            return
        existing_ids = {
            row["id"]
            for row in await connection.fetch(
                "SELECT id FROM restaurants WHERE id = ANY($1::uuid[])",
                [row[0] for row in restaurant_rows],
            )
        }
        stats.restaurants_added += sum(1 for row in restaurant_rows if row[0] not in existing_ids)
        stats.restaurants_updated += sum(1 for row in restaurant_rows if row[0] in existing_ids)
        query = """
            INSERT INTO restaurants (
                id, name, address, lat, lng, city, zip, price_level,
                google_place_id, yelp_id, yelp_url, ubereats_id,
                rating_score, rating_count, data_quality_score,
                last_verified_at, needs_refresh, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8,
                $9, $10, $11, $12,
                $13, $14, $15,
                $16, $17, $18, $19
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                address = COALESCE(EXCLUDED.address, restaurants.address),
                city = COALESCE(EXCLUDED.city, restaurants.city),
                zip = COALESCE(EXCLUDED.zip, restaurants.zip),
                data_quality_score = GREATEST(EXCLUDED.data_quality_score, restaurants.data_quality_score),
                last_verified_at = EXCLUDED.last_verified_at,
                needs_refresh = EXCLUDED.needs_refresh,
                updated_at = EXCLUDED.updated_at
        """
        for batch in chunked(restaurant_rows):
            await connection.executemany(query, batch)

    async def insert_dietary_attributes(self, connection, rows: list[tuple], stats: ZabihahStats) -> None:
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

    async def persist_listing(
        self,
        connection,
        *,
        listing: ZabihahListing,
        match_result: ListingMatchResult,
        existing_candidates: dict[uuid.UUID, MatchCandidate],
        stats: ZabihahStats,
    ) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        restaurant_id = match_result.restaurant_id or uuid.uuid4()

        # Confidence tier and halal value are derived from the detail page status.
        # Tier 1: zabiha verified (fully halal + "zabiha" in notes)
        # Tier 2: halal verified (fully halal, no zabiha signal)
        # Tier 3: halal self-reported (partially halal)
        if listing.is_zabiha:
            confidence_tier = 1
            halal_value = "true"
        elif listing.halal_status.lower() == "fully halal":
            confidence_tier = 2
            halal_value = "true"
        else:
            confidence_tier = 3
            halal_value = "partial"

        menu_presence: set[uuid.UUID] = set()
        has_google_place_id = False

        if match_result.restaurant_id is None:
            score = compute_data_quality_score(
                has_menu_items=False,
                has_google_place_id=False,
                has_dietary_attribute=True,
                has_high_confidence_dietary=confidence_tier <= 2,
                has_contact_or_website=False,
                verified_recently=True,
            )
            await self.upsert_restaurants(
                connection,
                [
                    (
                        restaurant_id,
                        listing.name,
                        listing.address,
                        None,
                        None,
                        listing.city,
                        listing.zip_code,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        score,
                        now,
                        True,
                        now,
                        now,
                    )
                ],
                stats,
            )
            existing_candidates[restaurant_id] = MatchCandidate(
                restaurant_id=restaurant_id,
                name=listing.name,
                address=listing.address,
                city=listing.city,
                zip_code=listing.zip_code,
                google_place_id=None,
                data_quality_score=score,
            )
        else:
            candidate = existing_candidates[restaurant_id]
            menu_presence = await self.fetch_menu_presence(connection, [restaurant_id])
            has_google_place_id = bool(candidate.google_place_id)
            score = compute_data_quality_score(
                has_menu_items=restaurant_id in menu_presence,
                has_google_place_id=has_google_place_id,
                has_dietary_attribute=True,
                has_high_confidence_dietary=confidence_tier <= 2,
                has_contact_or_website=await self.fetch_has_contact(connection, restaurant_id),
                verified_recently=True,
            )
            await self.upsert_restaurants(
                connection,
                [
                    (
                        restaurant_id,
                        candidate.name,
                        candidate.address or listing.address,
                        None,
                        None,
                        candidate.city or listing.city,
                        candidate.zip_code or listing.zip_code,
                        None,
                        candidate.google_place_id,
                        None,
                        None,
                        None,
                        None,
                        None,
                        score,
                        now,
                        restaurant_id not in menu_presence,
                        now,
                        now,
                    )
                ],
                stats,
            )

        existing_rows = await self.fetch_existing_dietary(connection, [restaurant_id])
        dietary_rows: list[tuple] = []

        halal_key = (restaurant_id, "halal", listing.zabihah_url)
        if halal_key not in existing_rows:
            dietary_rows.append(
                (
                    uuid.uuid4(),
                    restaurant_id,
                    "halal",
                    halal_value,
                    confidence_tier,
                    "zabihah",
                    listing.zabihah_url,
                    listing.user_notes,
                    now,
                )
            )
        if listing.is_zabiha:
            zabiha_key = (restaurant_id, "zabiha", listing.zabihah_url)
            if zabiha_key not in existing_rows:
                dietary_rows.append(
                    (
                        uuid.uuid4(),
                        restaurant_id,
                        "zabiha",
                        "true",
                        1,
                        "zabihah",
                        listing.zabihah_url,
                        listing.user_notes,
                        now,
                    )
                )
        await self.insert_dietary_attributes(connection, dietary_rows, stats)

    async def scrape(self, *, city: str, state: str) -> ZabihahStats:
        stats = ZabihahStats()
        async with self.pool.acquire() as connection:
            existing_candidates = {
                row.restaurant_id: row
                for row in await self.fetch_city_restaurants(connection, city, state)
            }

            # Pass 1 — search page: collect (name, address, detail_url) tuples.
            search_url = self.build_search_url(city, state)
            search_html = await self.fetch_page_html(connection, search_url)
            stats.pages_scraped += 1
            search_results = parse_search_results(search_html)

            # Pass 2 — detail pages: build one ZabihahListing per result.
            all_listings: dict[str, ZabihahListing] = {}
            for name, search_address, detail_url in search_results:
                if detail_url in all_listings:
                    continue
                detail_html = await self.fetch_page_html(connection, detail_url)
                stats.pages_scraped += 1
                listing = parse_detail_page(detail_html, name, search_address, detail_url)
                if listing is None:
                    await self.log_api_call(
                        connection,
                        query=f"detail_skip|url={detail_url}|reason=no_halal_status",
                        status="partial",
                    )
                    continue
                all_listings[detail_url] = listing

            for index, listing in enumerate(all_listings.values(), start=1):
                match_result = resolve_match(listing, list(existing_candidates.values()))
                if match_result.matched:
                    stats.matches_found += 1
                    await self.log_api_call(
                        connection,
                        query=(
                            f"match|name={listing.name}|restaurant_id={match_result.restaurant_id}|"
                            f"name_similarity={match_result.name_similarity:.3f}|"
                            f"address_similarity={match_result.address_similarity:.3f}"
                        ),
                        status="success",
                        records_updated=1,
                    )
                else:
                    stats.matches_rejected += 1
                    await self.log_api_call(
                        connection,
                        query=(
                            f"rejection|name={listing.name}|reason={match_result.reason}|"
                            f"name_similarity={match_result.name_similarity:.3f}|"
                            f"address_similarity={match_result.address_similarity:.3f}"
                        ),
                        status="partial",
                        records_added=1,
                    )
                await self.persist_listing(
                    connection,
                    listing=listing,
                    match_result=match_result,
                    existing_candidates=existing_candidates,
                    stats=stats,
                )
                stats.listings_processed += 1
                if index % 10 == 0:
                    print(f"Processed {index} listings...")
        return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zabihah.com halal verification scraper")
    parser.add_argument("--city", required=True)
    parser.add_argument("--state", required=True)
    return parser


async def run_cli(args: argparse.Namespace) -> ZabihahStats:
    async with ZabihahScraper() as scraper:
        return await scraper.scrape(city=args.city, state=args.state)


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        stats = await run_cli(args)
    except RateLimitPaused as exc:
        print(str(exc))
        return
    print(
        "Zabihah scrape complete:",
        {
            "restaurants_added": stats.restaurants_added,
            "restaurants_updated": stats.restaurants_updated,
            "dietary_attributes_added": stats.dietary_attributes_added,
            "pages_scraped": stats.pages_scraped,
            "listings_processed": stats.listings_processed,
            "matches_found": stats.matches_found,
            "matches_rejected": stats.matches_rejected,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
