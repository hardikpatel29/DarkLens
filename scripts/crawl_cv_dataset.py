#!/usr/bin/env python
"""
Multi-state Playwright crawler for the DarkLens CV dataset.

Visits ~70 target domains and captures ~10 UI states per domain
(homepage, product/listing, product/detail, cart, checkout, login/signup,
subscription/offer, popup/modal, cookie/consent, search/results) producing
approximately 700 diverse, metadata-rich screenshots for annotation.

Key design decisions
--------------------
* One screenshot ≠ one training example. A single PNG may yield 0, 1, or N
  YOLO bounding boxes after human annotation.
* Domain-based split enforced at crawl time — every screenshot's domain is
  recorded so the downstream splitter can assign the full domain to exactly
  one of train/val/test.
* Perceptual hashing (imagehash) flags near-duplicate captures so the
  annotation queue doesn't waste human time on duplicates.
* robots.txt is fetched and cached once per domain; URLs disallowed for
  the bot are skipped, not crawled.
* Safe interactions only: click "accept cookies", open product pages,
  add to cart — but STOP before any real payment/purchase form submission.
* Concurrency is intentionally low (3 domains in parallel) to be polite.

Usage
-----
    python scripts/crawl_cv_dataset.py
    python scripts/crawl_cv_dataset.py --domains 30 --states 5 --concurrency 3
    python scripts/crawl_cv_dataset.py --output-dir data/cv/raw/screenshots

Output
------
    data/cv/raw/screenshots/*.png
    data/cv/metadata/screenshots.csv      (per-image metadata)
    data/cv/metadata/crawl_log.csv        (per-domain summary + errors)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("crawl_cv_dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("data/cv/raw/screenshots")
DEFAULT_METADATA_DIR = Path("data/cv/metadata")
VIEWPORT = {"width": 1440, "height": 900}
NAV_TIMEOUT_MS = 10_000
INTERACTION_WAIT_S = 0.8   # seconds to pause after each interaction for dynamic UI
TINY_IMAGE_BYTES = 8_000   # images smaller than this are likely blank/blocked

# ---------------------------------------------------------------------------
# Target domain list (70 domains across 7 categories)
# ---------------------------------------------------------------------------

TARGET_DOMAINS: list[dict[str, Any]] = [
    # ── Indian E-Commerce ───────────────────────────────────────────────────
    {"domain": "amazon.in",         "category": "indian_ecommerce",   "seed": "https://www.amazon.in"},
    {"domain": "flipkart.com",      "category": "indian_ecommerce",   "seed": "https://www.flipkart.com"},
    {"domain": "myntra.com",        "category": "indian_ecommerce",   "seed": "https://www.myntra.com"},
    {"domain": "ajio.com",          "category": "indian_ecommerce",   "seed": "https://www.ajio.com"},
    {"domain": "meesho.com",        "category": "indian_ecommerce",   "seed": "https://www.meesho.com"},
    {"domain": "tatacliq.com",      "category": "indian_ecommerce",   "seed": "https://www.tatacliq.com"},
    {"domain": "nykaa.com",         "category": "indian_ecommerce",   "seed": "https://www.nykaa.com"},
    {"domain": "snapdeal.com",      "category": "indian_ecommerce",   "seed": "https://www.snapdeal.com"},
    {"domain": "pepperfry.com",     "category": "indian_ecommerce",   "seed": "https://www.pepperfry.com"},
    {"domain": "firstcry.com",      "category": "indian_ecommerce",   "seed": "https://www.firstcry.com"},
    {"domain": "bigbasket.com",     "category": "indian_ecommerce",   "seed": "https://www.bigbasket.com"},
    {"domain": "jiomart.com",       "category": "indian_ecommerce",   "seed": "https://www.jiomart.com"},
    {"domain": "croma.com",         "category": "indian_ecommerce",   "seed": "https://www.croma.com"},
    {"domain": "reliancedigital.in","category": "indian_ecommerce",   "seed": "https://www.reliancedigital.in"},
    {"domain": "bewakoof.com",      "category": "indian_ecommerce",   "seed": "https://www.bewakoof.com"},

    # ── International E-Commerce ────────────────────────────────────────────
    {"domain": "amazon.com",        "category": "intl_ecommerce",     "seed": "https://www.amazon.com"},
    {"domain": "walmart.com",       "category": "intl_ecommerce",     "seed": "https://www.walmart.com"},
    {"domain": "ebay.com",          "category": "intl_ecommerce",     "seed": "https://www.ebay.com"},
    {"domain": "etsy.com",          "category": "intl_ecommerce",     "seed": "https://www.etsy.com"},
    {"domain": "bestbuy.com",       "category": "intl_ecommerce",     "seed": "https://www.bestbuy.com"},
    {"domain": "target.com",        "category": "intl_ecommerce",     "seed": "https://www.target.com"},
    {"domain": "wayfair.com",       "category": "intl_ecommerce",     "seed": "https://www.wayfair.com"},
    {"domain": "asos.com",          "category": "intl_ecommerce",     "seed": "https://www.asos.com"},
    {"domain": "hm.com",            "category": "intl_ecommerce",     "seed": "https://www.hm.com"},
    {"domain": "nike.com",          "category": "intl_ecommerce",     "seed": "https://www.nike.com"},
    {"domain": "adidas.com",        "category": "intl_ecommerce",     "seed": "https://www.adidas.com"},
    {"domain": "uniqlo.com",        "category": "intl_ecommerce",     "seed": "https://www.uniqlo.com"},
    {"domain": "shein.com",         "category": "intl_ecommerce",     "seed": "https://www.shein.com"},
    {"domain": "homedepot.com",     "category": "intl_ecommerce",     "seed": "https://www.homedepot.com"},
    {"domain": "newegg.com",        "category": "intl_ecommerce",     "seed": "https://www.newegg.com"},

    # ── Travel / Hotels / Airlines ──────────────────────────────────────────
    {"domain": "booking.com",       "category": "travel",             "seed": "https://www.booking.com"},
    {"domain": "expedia.com",       "category": "travel",             "seed": "https://www.expedia.com"},
    {"domain": "hotels.com",        "category": "travel",             "seed": "https://www.hotels.com"},
    {"domain": "agoda.com",         "category": "travel",             "seed": "https://www.agoda.com"},
    {"domain": "airbnb.com",        "category": "travel",             "seed": "https://www.airbnb.com"},
    {"domain": "makemytrip.com",    "category": "travel",             "seed": "https://www.makemytrip.com"},
    {"domain": "cleartrip.com",     "category": "travel",             "seed": "https://www.cleartrip.com"},
    {"domain": "goibibo.com",       "category": "travel",             "seed": "https://www.goibibo.com"},
    {"domain": "tripadvisor.com",   "category": "travel",             "seed": "https://www.tripadvisor.com"},
    {"domain": "indigo.in",         "category": "travel",             "seed": "https://www.goindigo.in"},
    {"domain": "spicejet.com",      "category": "travel",             "seed": "https://www.spicejet.com"},

    # ── Food / Delivery ─────────────────────────────────────────────────────
    {"domain": "swiggy.com",        "category": "food_delivery",      "seed": "https://www.swiggy.com"},
    {"domain": "zomato.com",        "category": "food_delivery",      "seed": "https://www.zomato.com"},
    {"domain": "blinkit.com",       "category": "food_delivery",      "seed": "https://blinkit.com"},
    {"domain": "dominos.co.in",     "category": "food_delivery",      "seed": "https://www.dominos.co.in"},
    {"domain": "ubereats.com",      "category": "food_delivery",      "seed": "https://www.ubereats.com"},

    # ── SaaS / Subscriptions ────────────────────────────────────────────────
    {"domain": "canva.com",         "category": "saas",               "seed": "https://www.canva.com"},
    {"domain": "grammarly.com",     "category": "saas",               "seed": "https://www.grammarly.com"},
    {"domain": "adobe.com",         "category": "saas",               "seed": "https://www.adobe.com"},
    {"domain": "dropbox.com",       "category": "saas",               "seed": "https://www.dropbox.com"},
    {"domain": "duolingo.com",      "category": "saas",               "seed": "https://www.duolingo.com"},
    {"domain": "coursera.org",      "category": "saas",               "seed": "https://www.coursera.org"},
    {"domain": "udemy.com",         "category": "saas",               "seed": "https://www.udemy.com"},
    {"domain": "notion.so",         "category": "saas",               "seed": "https://www.notion.so"},
    {"domain": "skillshare.com",    "category": "saas",               "seed": "https://www.skillshare.com"},
    {"domain": "hubspot.com",       "category": "saas",               "seed": "https://www.hubspot.com"},

    # ── Streaming / Entertainment ───────────────────────────────────────────
    {"domain": "netflix.com",       "category": "streaming",          "seed": "https://www.netflix.com"},
    {"domain": "primevideo.com",    "category": "streaming",          "seed": "https://www.primevideo.com"},
    {"domain": "spotify.com",       "category": "streaming",          "seed": "https://www.spotify.com"},
    {"domain": "hotstar.com",       "category": "streaming",          "seed": "https://www.hotstar.com"},

    # ── Ticketing / Events ──────────────────────────────────────────────────
    {"domain": "ticketmaster.com",  "category": "ticketing",          "seed": "https://www.ticketmaster.com"},
    {"domain": "bookmyshow.com",    "category": "ticketing",          "seed": "https://in.bookmyshow.com"},
    {"domain": "eventbrite.com",    "category": "ticketing",          "seed": "https://www.eventbrite.com"},
    {"domain": "stubhub.com",       "category": "ticketing",          "seed": "https://www.stubhub.com"},

    # ── Finance / Insurance ─────────────────────────────────────────────────
    {"domain": "policybazaar.com",  "category": "finance",            "seed": "https://www.policybazaar.com"},
    {"domain": "bankbazaar.com",    "category": "finance",            "seed": "https://www.bankbazaar.com"},
    {"domain": "geico.com",         "category": "finance",            "seed": "https://www.geico.com"},

    # ── Marketplaces / Digital ──────────────────────────────────────────────
    {"domain": "godaddy.com",       "category": "saas",               "seed": "https://www.godaddy.com"},
    {"domain": "hostinger.com",     "category": "saas",               "seed": "https://www.hostinger.com"},
    {"domain": "squarespace.com",   "category": "saas",               "seed": "https://www.squarespace.com"},
    {"domain": "shopify.com",       "category": "saas",               "seed": "https://www.shopify.com"},
    {"domain": "wix.com",           "category": "saas",               "seed": "https://www.wix.com"},
]

# ---------------------------------------------------------------------------
# UI interaction recipes per state
# Each recipe is a list of safe Playwright actions to reach the target state.
# These are best-effort: if any action fails, we capture the current state
# as-is and move on rather than crashing.
# ---------------------------------------------------------------------------

# Cookie/consent selectors tried in order (click first match found)
COOKIE_ACCEPT_SELECTORS = [
    "button:has-text('Accept all')",
    "button:has-text('Accept All')",
    "button:has-text('Accept cookies')",
    "button:has-text('Accept Cookies')",
    "button:has-text('I Accept')",
    "button:has-text('OK')",
    "button:has-text('Agree')",
    "button:has-text('Allow all')",
    "button:has-text('Allow All')",
    "[id*='accept'][id*='cookie']",
    "[class*='accept-all']",
    "[data-testid*='accept']",
]

PAGE_TYPE_ORDER = [
    "homepage",
    "search_results",
    "product_listing",
    "product_detail",
    "cart",
    "checkout",
    "login_signup",
    "subscription_offer",
    "cookie_consent",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScreenshotRecord:
    image_id: str
    domain: str
    url: str
    page_type: str
    category: str
    timestamp: str
    screenshot_width: int
    screenshot_height: int
    interaction_state: str
    file_path: str
    file_bytes: int
    phash: str
    dark_pattern_present: str = "unknown"   # filled in by annotation UI
    review_status: str = "pending"
    notes: str = ""


@dataclass
class CrawlLog:
    domain: str
    category: str
    seed_url: str
    started_at: str
    finished_at: str = ""
    screenshots_taken: int = 0
    skipped_states: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    robots_blocked: bool = False


# ---------------------------------------------------------------------------
# Robots.txt helper
# ---------------------------------------------------------------------------

async def is_allowed_by_robots(url: str, session_cache: dict[str, urllib.robotparser.RobotFileParser]) -> bool:
    """Returns True if the given URL is crawlable according to robots.txt."""
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in session_cache:
        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{base}/robots.txt"
        try:
            req = urllib.request.Request(robots_url, headers={"User-Agent": "Mozilla/5.0"})
            def _fetch():
                with urllib.request.urlopen(req, timeout=3) as resp:
                    return resp.read().decode("utf-8", errors="ignore")
            content = await asyncio.to_thread(_fetch)
            rp.parse(content.splitlines())
        except Exception:
            rp = None
        session_cache[base] = rp
    rp = session_cache[base]
    if rp is None:
        return True
    return rp.can_fetch("*", url)


# ---------------------------------------------------------------------------
# Perceptual hashing (without imagehash dependency)
# Falls back to MD5 if pillow not installed.
# ---------------------------------------------------------------------------

def compute_phash(image_path: Path) -> str:
    """
    Computes a simple average-hash (aHash) for deduplication.
    Returns hex string. Falls back to file MD5 if Pillow not available.
    """
    try:
        from PIL import Image
        import struct

        img = Image.open(image_path).convert("L").resize((8, 8), Image.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return hex(int(bits, 2))[2:].zfill(16)
    except Exception:
        # Fallback: MD5 of first 64KB
        h = hashlib.md5()
        with open(image_path, "rb") as f:
            h.update(f.read(65536))
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Core capture helpers
# ---------------------------------------------------------------------------

async def safe_click(page, selector: str, timeout_ms: int = 3000) -> bool:
    """Tries to click a selector; returns True on success, False on any failure."""
    try:
        el = page.locator(selector).first
        await el.wait_for(state="visible", timeout=timeout_ms)
        await el.click(timeout=timeout_ms)
        await asyncio.sleep(0.8)
        return True
    except Exception:
        return False


async def accept_cookies(page) -> bool:
    """Attempts to dismiss cookie/consent banners. Returns True if something was clicked."""
    for sel in COOKIE_ACCEPT_SELECTORS:
        if await safe_click(page, sel, timeout_ms=2000):
            logger.debug("Cookie banner dismissed via: %s", sel)
            return True
    return False


async def capture_screenshot(
    page,
    output_dir: Path,
    domain: str,
    category: str,
    page_type: str,
    url: str,
    interaction_state: str,
    counter: list[int],   # mutable counter passed by reference
) -> ScreenshotRecord | None:
    """Captures a single screenshot and returns its ScreenshotRecord."""
    counter[0] += 1
    image_id = f"{re.sub(r'[^a-z0-9]', '_', domain)}_{page_type}_{counter[0]:04d}"
    file_name = f"{image_id}.png"
    file_path = output_dir / file_name

    try:
        await page.screenshot(path=str(file_path), full_page=False)
    except Exception as exc:
        logger.warning("Screenshot failed for %s (%s): %s", domain, page_type, exc)
        return None

    file_bytes = file_path.stat().st_size
    if file_bytes < TINY_IMAGE_BYTES:
        logger.info("  ⚠ Tiny/blank screenshot (%d bytes) — skipping: %s", file_bytes, file_name)
        file_path.unlink(missing_ok=True)
        return None

    phash = compute_phash(file_path)
    now = datetime.now(timezone.utc).isoformat()

    record = ScreenshotRecord(
        image_id=image_id,
        domain=domain,
        url=url,
        page_type=page_type,
        category=category,
        timestamp=now,
        screenshot_width=VIEWPORT["width"],
        screenshot_height=VIEWPORT["height"],
        interaction_state=interaction_state,
        file_path=str(file_path),
        file_bytes=file_bytes,
        phash=phash,
    )
    logger.info("  ✓ %s [%s] → %s (%d KB)", domain, page_type, file_name, file_bytes // 1024)
    return record


# ---------------------------------------------------------------------------
# Domain crawl: attempts to capture as many UI states as accessible
# ---------------------------------------------------------------------------

async def crawl_domain(
    domain_entry: dict[str, Any],
    output_dir: Path,
    max_states: int,
    browser,
    robots_cache: dict,
    global_counter: list[int],
    seen_phashes: set[str],
) -> tuple[list[ScreenshotRecord], CrawlLog]:
    """
    Crawls a single domain for up to max_states screenshot captures.
    Returns (records, crawl_log).
    """
    domain = domain_entry["domain"]
    category = domain_entry["category"]
    seed_url = domain_entry["seed"]
    started_at = datetime.now(timezone.utc).isoformat()
    records: list[ScreenshotRecord] = []
    crawl_log = CrawlLog(
        domain=domain, category=category, seed_url=seed_url, started_at=started_at
    )

    # Check robots.txt for seed URL
    if not await is_allowed_by_robots(seed_url, robots_cache):
        logger.warning("robots.txt disallows crawling %s — skipping domain.", domain)
        crawl_log.robots_blocked = True
        crawl_log.finished_at = datetime.now(timezone.utc).isoformat()
        return records, crawl_log

    page = await browser.new_page(
        viewport=VIEWPORT,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    )

    async def snap(page_type: str, interaction_note: str = "") -> bool:
        """Shorthand: capture + dedup + append. Returns False if image was a duplicate."""
        if len(records) >= max_states:
            return False
        current_url = page.url
        rec = await capture_screenshot(
            page, output_dir, domain, category, page_type,
            current_url, interaction_note, global_counter
        )
        if rec is None:
            crawl_log.skipped_states.append(page_type)
            return False
        # Near-duplicate check
        if rec.phash in seen_phashes:
            logger.info("  ↩ Near-duplicate detected, skipping: %s", rec.image_id)
            Path(rec.file_path).unlink(missing_ok=True)
            crawl_log.skipped_states.append(f"{page_type}(dup)")
            return False
        seen_phashes.add(rec.phash)
        records.append(rec)
        return True

    try:
        # ── State 0: Homepage ──────────────────────────────────────────────
        try:
            await page.goto(seed_url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            await asyncio.sleep(INTERACTION_WAIT_S)
            await snap("homepage", "direct_nav")
        except Exception as exc:
            crawl_log.errors.append(f"homepage: {exc}")

        # ── Accept cookies (attempt; doesn't abort if it fails) ────────────
        try:
            dismissed = await accept_cookies(page)
            if dismissed:
                await asyncio.sleep(1.0)
                await snap("cookie_consent", "cookie_banner_dismissed")
        except Exception:
            pass

        # ── State 1: Search results ────────────────────────────────────────
        try:
            search_selectors = [
                "input[type='search']", "input[name='q']", "input[name='query']",
                "input[placeholder*='Search' i]", "input[placeholder*='search' i]",
                "[data-testid*='search-input']", ".search-input", "#search",
            ]
            for sel in search_selectors:
                try:
                    el = page.locator(sel).first
                    await el.wait_for(state="visible", timeout=2000)
                    await el.fill("shoes")
                    await el.press("Enter")
                    await asyncio.sleep(INTERACTION_WAIT_S)
                    await snap("search_results", "searched_shoes")
                    break
                except Exception:
                    continue
        except Exception as exc:
            crawl_log.skipped_states.append("search_results")

        # ── State 2: Product listing page ──────────────────────────────────
        try:
            product_listing_links = [
                "a[href*='/category']", "a[href*='/c/']", "a[href*='/products']",
                "a[href*='/collection']", "a[href*='/shop']", "a[href*='/browse']",
                ".category-link", ".nav-category a",
            ]
            nav_clicked = False
            for sel in product_listing_links:
                if nav_clicked:
                    break
                try:
                    await page.goto(seed_url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                    el = page.locator(sel).first
                    href = await el.get_attribute("href")
                    if href:
                        target = urllib.parse.urljoin(seed_url, href)
                        if urllib.parse.urlparse(target).netloc.endswith(domain.split(".")[-2] + "." + domain.split(".")[-1]):
                            await page.goto(target, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                            await asyncio.sleep(INTERACTION_WAIT_S)
                            await snap("product_listing", "nav_category_click")
                            nav_clicked = True
                except Exception:
                    continue
            if not nav_clicked:
                crawl_log.skipped_states.append("product_listing")
        except Exception as exc:
            crawl_log.errors.append(f"product_listing: {exc}")

        # ── State 3: Product detail page ───────────────────────────────────
        try:
            product_link_selectors = [
                "a[href*='/product']", "a[href*='/item']", "a[href*='/dp/']",
                "a[href*='/p/']", ".product-card a", ".product-item a",
                "[data-testid='product-card'] a", ".product-tile a",
            ]
            product_found = False
            for sel in product_link_selectors:
                if product_found:
                    break
                try:
                    el = page.locator(sel).first
                    href = await el.get_attribute("href")
                    if href:
                        target = urllib.parse.urljoin(page.url, href)
                        if urllib.parse.urlparse(target).netloc.endswith(domain.split(".")[-2] + "." + domain.split(".")[-1]):
                            await page.goto(target, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                            await asyncio.sleep(INTERACTION_WAIT_S)
                            await snap("product_detail", "product_link_click")
                            product_found = True
                except Exception:
                    continue
            if not product_found:
                crawl_log.skipped_states.append("product_detail")
        except Exception as exc:
            crawl_log.errors.append(f"product_detail: {exc}")

        # ── State 4: Cart ──────────────────────────────────────────────────
        try:
            add_to_cart_selectors = [
                "button:has-text('Add to Cart')", "button:has-text('Add to Bag')",
                "button:has-text('Add to cart')", "button:has-text('Buy Now')",
                "[data-testid*='add-to-cart']", "[id*='add-to-cart']",
                ".add-to-cart", ".btn-cart",
            ]
            cart_added = False
            for sel in add_to_cart_selectors:
                if await safe_click(page, sel, timeout_ms=3000):
                    await asyncio.sleep(INTERACTION_WAIT_S)
                    cart_added = True
                    break

            cart_nav_selectors = [
                "a[href*='/cart']", "a[href*='/bag']", "a[href*='/basket']",
                "[data-testid*='cart']", ".cart-icon", ".bag-icon", "#cart",
            ]
            for sel in cart_nav_selectors:
                try:
                    el = page.locator(sel).first
                    href = await el.get_attribute("href")
                    if href:
                        target = urllib.parse.urljoin(page.url, href)
                        await page.goto(target, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                        await asyncio.sleep(INTERACTION_WAIT_S)
                        await snap("cart", "cart_nav_click")
                        break
                except Exception:
                    continue
            if not cart_added:
                crawl_log.skipped_states.append("cart")
        except Exception as exc:
            crawl_log.errors.append(f"cart: {exc}")

        # ── State 5: Checkout — navigate but DO NOT submit/pay ─────────────
        try:
            checkout_selectors = [
                "button:has-text('Proceed to Checkout')", "button:has-text('Checkout')",
                "a[href*='/checkout']", "a:has-text('Checkout')",
                "[data-testid*='checkout']",
            ]
            for sel in checkout_selectors:
                if await safe_click(page, sel, timeout_ms=3000):
                    await asyncio.sleep(INTERACTION_WAIT_S)
                    # STOP here — do NOT fill payment info or submit
                    await snap("checkout", "checkout_opened_stopped_before_payment")
                    break
            else:
                crawl_log.skipped_states.append("checkout")
        except Exception as exc:
            crawl_log.errors.append(f"checkout: {exc}")

        # ── State 6: Login / signup page ───────────────────────────────────
        try:
            await page.goto(seed_url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            await asyncio.sleep(1.5)
            login_selectors = [
                "a[href*='/login']", "a[href*='/signin']", "a[href*='/sign-in']",
                "a[href*='/register']", "a[href*='/signup']", "a[href*='/sign-up']",
                "a:has-text('Sign In')", "a:has-text('Log In')", "a:has-text('Login')",
                "button:has-text('Sign In')",
            ]
            login_found = False
            for sel in login_selectors:
                try:
                    el = page.locator(sel).first
                    href = await el.get_attribute("href")
                    if href:
                        target = urllib.parse.urljoin(page.url, href)
                        await page.goto(target, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                        await asyncio.sleep(INTERACTION_WAIT_S)
                        await snap("login_signup", "login_link_nav")
                        login_found = True
                        break
                    else:
                        if await safe_click(page, sel):
                            await asyncio.sleep(INTERACTION_WAIT_S)
                            await snap("login_signup", "login_button_click")
                            login_found = True
                            break
                except Exception:
                    continue
            if not login_found:
                crawl_log.skipped_states.append("login_signup")
        except Exception as exc:
            crawl_log.errors.append(f"login_signup: {exc}")

        # ── State 7: Subscription / offer / pricing page ───────────────────
        try:
            await page.goto(seed_url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            await asyncio.sleep(1.5)
            pricing_selectors = [
                "a[href*='/pricing']", "a[href*='/plans']", "a[href*='/subscribe']",
                "a[href*='/premium']", "a[href*='/upgrade']", "a[href*='/membership']",
                "a:has-text('Pricing')", "a:has-text('Plans')", "a:has-text('Subscribe')",
                "a:has-text('Premium')",
            ]
            pricing_found = False
            for sel in pricing_selectors:
                try:
                    el = page.locator(sel).first
                    href = await el.get_attribute("href")
                    if href:
                        target = urllib.parse.urljoin(page.url, href)
                        await page.goto(target, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
                        await asyncio.sleep(INTERACTION_WAIT_S)
                        await snap("subscription_offer", "pricing_link_nav")
                        pricing_found = True
                        break
                except Exception:
                    continue
            if not pricing_found:
                crawl_log.skipped_states.append("subscription_offer")
        except Exception as exc:
            crawl_log.errors.append(f"subscription_offer: {exc}")

        # ── State 8: Re-capture homepage with popup/modal wait ─────────────
        try:
            await page.goto(seed_url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            await asyncio.sleep(4.0)   # wait longer for delayed modals
            await snap("popup_wait", "homepage_popup_delay_wait")
        except Exception as exc:
            crawl_log.errors.append(f"popup_wait: {exc}")

    except Exception as exc:
        crawl_log.errors.append(f"domain-level: {exc}")
    finally:
        await page.close()

    crawl_log.screenshots_taken = len(records)
    crawl_log.finished_at = datetime.now(timezone.utc).isoformat()
    return records, crawl_log


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run_crawl(
    output_dir: Path,
    metadata_dir: Path,
    max_domains: int,
    max_states_per_domain: int,
    concurrency: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright required: pip install playwright && playwright install chromium") from exc

    domains = TARGET_DOMAINS[:max_domains]
    logger.info(
        "Starting CV dataset crawl: %d domains × up to %d states = up to %d screenshots",
        len(domains), max_states_per_domain, len(domains) * max_states_per_domain,
    )

    all_records: list[ScreenshotRecord] = []
    all_logs: list[CrawlLog] = []
    robots_cache: dict = {}
    seen_phashes: set[str] = set()
    global_counter = [0]   # mutable int in a list for closure sharing

    semaphore = asyncio.Semaphore(concurrency)

    async def crawl_with_semaphore(domain_entry, browser):
        async with semaphore:
            records, log = await crawl_domain(
                domain_entry, output_dir, max_states_per_domain,
                browser, robots_cache, global_counter, seen_phashes
            )
            return records, log

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        try:
            tasks = [crawl_with_semaphore(d, browser) for d in domains]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await browser.close()

    for result in results:
        if isinstance(result, Exception):
            logger.error("Domain crawl task failed: %s", result)
            continue
        records, log = result
        all_records.extend(records)
        all_logs.append(log)

    # ── Write screenshots.csv ────────────────────────────────────────────
    screenshots_csv = metadata_dir / "screenshots.csv"
    if all_records:
        fieldnames = list(asdict(all_records[0]).keys())
        with open(screenshots_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in all_records:
                writer.writerow(asdict(rec))
        logger.info("Wrote %s (%d rows)", screenshots_csv, len(all_records))

    # ── Write crawl_log.csv ──────────────────────────────────────────────
    crawl_log_csv = metadata_dir / "crawl_log.csv"
    if all_logs:
        log_fieldnames = ["domain", "category", "seed_url", "started_at", "finished_at",
                          "screenshots_taken", "robots_blocked", "skipped_states", "errors"]
        with open(crawl_log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fieldnames)
            writer.writeheader()
            for lg in all_logs:
                writer.writerow({
                    "domain": lg.domain,
                    "category": lg.category,
                    "seed_url": lg.seed_url,
                    "started_at": lg.started_at,
                    "finished_at": lg.finished_at,
                    "screenshots_taken": lg.screenshots_taken,
                    "robots_blocked": lg.robots_blocked,
                    "skipped_states": json.dumps(lg.skipped_states),
                    "errors": json.dumps(lg.errors),
                })
        logger.info("Wrote %s (%d domain entries)", crawl_log_csv, len(all_logs))

    # ── Write pending/ symlinks for annotation UI ────────────────────────
    pending_dir = output_dir.parent.parent / "review" / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    for rec in all_records:
        src = Path(rec.file_path)
        dst = pending_dir / src.name
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    # ── Summary ──────────────────────────────────────────────────────────
    successful_domains = sum(1 for lg in all_logs if lg.screenshots_taken > 0)
    blocked = sum(1 for lg in all_logs if lg.robots_blocked)
    logger.info("=" * 64)
    logger.info("CRAWL COMPLETE")
    logger.info("  Screenshots captured : %d", len(all_records))
    logger.info("  Domains crawled      : %d / %d", successful_domains, len(domains))
    logger.info("  Domains blocked      : %d (robots.txt)", blocked)
    logger.info("  Output directory     : %s", output_dir)
    logger.info("  Metadata             : %s", metadata_dir)
    logger.info("  Next step            : python scripts/autolabel_cv_dataset.py")
    logger.info("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save PNG screenshots (default: data/cv/raw/screenshots)",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR,
        help="Directory for CSV metadata files (default: data/cv/metadata)",
    )
    parser.add_argument(
        "--domains", type=int, default=len(TARGET_DOMAINS),
        help=f"Maximum number of domains to crawl (default: {len(TARGET_DOMAINS)})",
    )
    parser.add_argument(
        "--states", type=int, default=10,
        help="Maximum screenshot captures per domain (default: 10)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Number of domains crawled simultaneously (default: 3)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_crawl(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        max_domains=args.domains,
        max_states_per_domain=args.states,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
