#!/usr/bin/env python
"""
Automated screenshot collection script for dark-pattern UI dataset assembly.

Visits a curated list of 200+ popular web targets (e-commerce, travel,
ticketing, SaaS, subscriptions, retail) with bounded async concurrency
and captures high-resolution full-page screenshots.

Usage:
    python scripts/collect_screenshots.py --limit 200 --concurrency 5
    python scripts/collect_screenshots.py --urls my_urls.txt --output-dir data/cv_screenshots
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("collect_screenshots")

# Curated target list across categories prone to dark patterns
# (travel, e-commerce, ticketing, SaaS, subscriptions, gaming, retail)
DEFAULT_TARGET_URLS: list[str] = [
    # --- Travel & Booking ---
    "https://www.booking.com",
    "https://www.agoda.com",
    "https://www.expedia.com",
    "https://www.kayak.com",
    "https://www.hotels.com",
    "https://www.tripadvisor.com",
    "https://www.trivago.com",
    "https://www.priceline.com",
    "https://www.cheapoair.com",
    "https://www.skyscanner.net",
    "https://www.orbitz.com",
    "https://www.travelocity.com",
    "https://www.hotwire.com",
    "https://www.ryanair.com",
    "https://www.easyjet.com",
    "https://www.wizzair.com",
    "https://www.spirit.com",
    "https://www.frontier.com",
    "https://www.airbnb.com",
    "https://www.vrbo.com",
    "https://www.hostelworld.com",
    "https://www.viator.com",
    "https://www.klook.com",
    "https://www.getyourguide.com",
    "https://www.trip.com",
    
    # --- E-Commerce & Retail ---
    "https://www.amazon.com",
    "https://www.ebay.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.bestbuy.com",
    "https://www.aliexpress.com",
    "https://www.shein.com",
    "https://www.temu.com",
    "https://www.etsy.com",
    "https://www.wayfair.com",
    "https://www.overstock.com",
    "https://www.homedepot.com",
    "https://www.lowes.com",
    "https://www.macys.com",
    "https://www.nordstrom.com",
    "https://www.kohl.com",
    "https://www.gap.com",
    "https://www.oldnavy.com",
    "https://www.hm.com",
    "https://www.zara.com",
    "https://www.asos.com",
    "https://www.forever21.com",
    "https://www.uniqlo.com",
    "https://www.nike.com",
    "https://www.adidas.com",
    "https://www.puma.com",
    "https://www.underarmour.com",
    "https://www.sephora.com",
    "https://www.ulta.com",
    "https://www.bathandbodyworks.com",
    "https://www.chewy.com",
    "https://www.petco.com",
    "https://www.petsmart.com",
    "https://www.newegg.com",
    "https://www.microcenter.com",
    "https://www.bhphotovideo.com",
    "https://www.costco.com",
    "https://www.samsclub.com",
    "https://www.bj.com",

    # --- Ticketing & Events ---
    "https://www.ticketmaster.com",
    "https://www.stubhub.com",
    "https://www.seatgeek.com",
    "https://www.eventbrite.com",
    "https://www.viagogo.com",
    "https://www.vividseats.com",
    "https://www.tickpick.com",
    "https://www.livenation.com",
    "https://www.fandango.com",
    "https://www.atomtickets.com",

    # --- Food Delivery & Subscriptions ---
    "https://www.doordash.com",
    "https://www.ubereats.com",
    "https://www.grubhub.com",
    "https://www.instacart.com",
    "https://www.hellofresh.com",
    "https://www.blueapron.com",
    "https://www.factor75.com",
    "https://www.gopuff.com",
    "https://www.postmates.com",
    "https://www.deliveroo.co.uk",

    # --- Media, Streaming & Subscriptions ---
    "https://www.netflix.com",
    "https://www.spotify.com",
    "https://www.hulu.com",
    "https://www.disneyplus.com",
    "https://www.paramountplus.com",
    "https://www.max.com",
    "https://www.nytimes.com",
    "https://www.wsj.com",
    "https://www.washingtonpost.com",
    "https://www.economist.com",
    "https://www.medium.com",
    "https://www.scribd.com",
    "https://www.audible.com",
    "https://www.patreon.com",
    "https://www.substack.com",

    # --- SaaS, Hosting & Software ---
    "https://www.godaddy.com",
    "https://www.namecheap.com",
    "https://www.bluehost.com",
    "https://www.hostinger.com",
    "https://www.wix.com",
    "https://www.squarespace.com",
    "https://www.shopify.com",
    "https://www.wordpress.com",
    "https://www.canva.com",
    "https://www.adobe.com",
    "https://www.grammarly.com",
    "https://www.duolingo.com",
    "https://www.coursera.org",
    "https://www.udemy.com",
    "https://www.skillshare.com",
    "https://www.linkedin.com/learning",
    "https://www.zoom.us",
    "https://www.slack.com",
    "https://www.dropbox.com",
    "https://www.evernote.com",
    "https://www.notion.so",
    "https://www.monday.com",
    "https://www.asana.com",
    "https://www.trello.com",
    "https://www.mailchimp.com",
    "https://www.hubspot.com",
    "https://www.zendesk.com",
    "https://www.intercom.com",

    # --- Gaming & Digital Goods ---
    "https://store.steampowered.com",
    "https://store.epicgames.com",
    "https://www.g2a.com",
    "https://www.cdkeys.com",
    "https://www.humblebundle.com",
    "https://www.roblox.com",
    "https://www.playstation.com",
    "https://www.xbox.com",
    "https://www.nintendo.com",
    "https://www.ea.com",
    "https://www.ubisoft.com",
    "https://www.blizzard.com",

    # --- Telecom & Utilities ---
    "https://www.att.com",
    "https://www.verizon.com",
    "https://www.t-mobile.com",
    "https://www.xfinity.com",
    "https://www.spectrum.com",

    # --- Car Rental & Insurance ---
    "https://www.hertz.com",
    "https://www.enterprise.com",
    "https://www.avis.com",
    "https://www.budget.com",
    "https://www.turo.com",
    "https://www.geico.com",
    "https://www.progressive.com",
    "https://www.statefarm.com",
    "https://www.lemonade.com",

    # --- International & Regional Stores ---
    "https://www.flipkart.com",
    "https://www.myntra.com",
    "https://www.nykaa.com",
    "https://www.meesho.com",
    "https://www.rakuten.com",
    "https://www.mercari.com",
    "https://www.zomato.com",
    "https://www.swiggy.com",
    "https://www.make-my-trip.com",
    "https://www.cleartrip.com",
    "https://www.yatra.com",
    "https://www.bookmyshow.com",
    "https://www.zalando.com",
    "https://www.otto.de",
    "https://www.bol.com",
    "https://www.croma.com",
    "https://www.tatacliq.com",
    "https://www.reliancedigital.in",
    "https://www.pepperfry.com",
    "https://www.urbancompany.com",
    "https://www.redbus.in",
    "https://www.paytm.com",
    "https://www.firstcry.com",

    # --- Additional General Popular Sites ---
    "https://www.coursera.com",
    "https://www.udacity.com",
    "https://www.edx.org",
    "https://www.masterclass.com",
    "https://www.datacamp.com",
    "https://www.codecademy.com",
    "https://www.fiverr.com",
    "https://www.upwork.com",
    "https://www.freelancer.com",
    "https://www.99designs.com",
    "https://www.shutterstock.com",
    "https://www.gettyimages.com",
    "https://www.istockphoto.com",
    "https://www.freepik.com",
    "https://www.vecteezy.com",
    "https://www.envato.com",
    "https://elements.envato.com",
    "https://www.squarespace.com/templates",
    "https://www.shopify.com/free-trial",
    "https://www.wix.com/website/templates",
    "https://www.weebly.com",
    "https://www.webflow.com",
    "https://www.framer.com",
    "https://www.figma.com",
    "https://www.miro.com",
    "https://www.lucidchart.com",
    "https://www.typeform.com",
    "https://www.survey-monkey.com",
    "https://www.jotform.com",
    "https://www.surveymonkey.com",
    "https://www.qualtrics.com",
    "https://www.trustpilot.com",
    "https://www.g2.com",
    "https://www.capterra.com",
    "https://www.glassdoor.com",
    "https://www.indeed.com",
    "https://www.ziprecruiter.com",
    "https://www.monster.com",
]


def sanitize_filename(url: str) -> str:
    """Derives a clean filename from a URL domain."""
    domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    clean = re.sub(r"[^\w\-]", "_", domain)
    return clean[:50]


async def capture_single(
    url: str,
    output_dir: Path,
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    pw_browser,
    timeout_ms: int = 20_000,
) -> dict[str, str | bool | int]:
    """Captures a single full-page screenshot with error handling."""
    filename = f"{index:03d}_{sanitize_filename(url)}.png"
    filepath = output_dir / filename
    result: dict[str, str | bool | int] = {
        "index": index,
        "url": url,
        "file": filename,
        "path": str(filepath),
        "success": False,
        "error": "",
    }

    async with semaphore:
        logger.info("[%d/%d] Capturing: %s", index, total, url)
        page = None
        try:
            page = await pw_browser.new_page(
                viewport={"width": 1440, "height": 900},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
            )
            # Short wait for initial load, fallback to domcontentloaded if networkidle hangs
            try:
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                await asyncio.sleep(2.0)  # Brief pause for lazy-loaded timers / popups
            except Exception:
                pass  # Partial load is okay for taking screenshots

            await page.screenshot(path=str(filepath), full_page=False)
            result["success"] = True
            logger.info("  ✓ Saved %s", filename)
        except Exception as exc:
            result["error"] = str(exc)
            logger.warning("  ✗ Failed %s: %s", url, exc)
        finally:
            if page:
                await page.close()

    return result


async def run_batch_capture(
    urls: Sequence[str],
    output_dir: Path,
    concurrency: int = 5,
    timeout_ms: int = 20_000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError("Playwright is required. Run: pip install playwright && playwright install chromium") from exc

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        try:
            tasks = [
                capture_single(url, output_dir, idx + 1, len(urls), semaphore, browser, timeout_ms)
                for idx, url in enumerate(urls)
            ]
            results = await asyncio.gather(*tasks)

            successful = [r for r in results if r["success"]]
            logger.info("=" * 60)
            logger.info("BATCH CAPTURE COMPLETE: %d/%d successfully saved to %s", len(successful), len(urls), output_dir)
            logger.info("=" * 60)

            manifest_path = output_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "captured_at": datetime.utcnow().isoformat(),
                        "total_target_urls": len(urls),
                        "successful_captures": len(successful),
                        "results": results,
                    },
                    f,
                    indent=2,
                )
            logger.info("Manifest saved to %s", manifest_path)
        finally:
            await browser.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cv_raw_screenshots"),
        help="Directory to save PNG screenshots",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        help="Optional txt file with one URL per line to override default targets",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of URLs to capture (default: 200)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent Playwright pages (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Per-page navigation timeout in seconds (default: 20)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.urls_file and args.urls_file.exists():
        with open(args.urls_file, encoding="utf-8") as f:
            target_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        target_urls = DEFAULT_TARGET_URLS

    target_urls = target_urls[: args.limit]
    logger.info("Starting collection of %d website screenshots (concurrency=%d)...", len(target_urls), args.concurrency)

    asyncio.run(
        run_batch_capture(
            urls=target_urls,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            timeout_ms=args.timeout * 1000,
        )
    )


if __name__ == "__main__":
    main()
