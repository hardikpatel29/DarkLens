"""
Playwright implementation of the BrowserGateway port, plus
PlaywrightPdfRenderer (PdfRendererPort) below it.

This is the ONLY file in the project allowed to `import playwright`.
That's not a style preference — it's what makes "swap the rendering
engine" a one-file change instead of a grep-and-replace across the
codebase. `PlaywrightPdfRenderer` lives here rather than in its own
file in `infrastructure/reporting/` for the same reason: two classes
that both happen to use Playwright is a smaller footprint than two
separate files each importing it.

The JS injected into the page (`_EXTRACTION_SCRIPT`) does the DOM
flattening in-browser rather than round-tripping every node individually
over the CDP protocol, which is the difference between a scan taking
~1-2s of extraction time versus tens of seconds on a DOM-heavy page.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from playwright.async_api import async_playwright

from darklens.application.ports.detector import BrowserGateway
from darklens.application.ports.pdf_renderer import PdfRendererPort
from darklens.domain.entities.page_snapshot import DomNode, PageSnapshot

logger = logging.getLogger(__name__)

# Runs in-page. Flattens visible/interactive elements into a JSON-serializable
# list, capturing the specific computed-style properties our rules need.
# Keeping this list narrow (not "capture every CSS property") keeps snapshot
# size and extraction time bounded on large pages.
_EXTRACTION_SCRIPT = """
() => {
  const STYLE_PROPS = [
    "visibility", "display", "opacity", "position",
    "z-index", "font-size", "color", "background-color"
  ];
  // Query only interactive/semantic leaf-level elements — deliberately
  // excludes generic <div> and <span> wrappers. Those almost never carry
  // standalone dark-pattern microcopy; they exist as layout containers,
  // and including them causes every nested string to appear once per
  // ancestor in the query set (div > span > a all share the same text).
  const els = Array.from(document.querySelectorAll(
    "a, button, input, label, p, h1, h2, h3, h4, h5, h6"
  )).slice(0, 5000); // hard cap: protects against pathological pages

  // Build a Set of all selected elements so we can quickly test
  // child membership without a full querySelectorAll per node.
  const elSet = new Set(els);

  function selectorFor(el, index) {
    if (el.id) return `#${el.id}`;
    const cls = el.className && typeof el.className === "string"
      ? "." + el.className.trim().split(/\\s+/).join(".")
      : "";
    return `${el.tagName.toLowerCase()}${cls}:nth-of-type(${index})`;
  }

  // Returns true if any *direct or indirect* child of el is also in
  // our query set and carries the same trimmed text. When that is the
  // case, el is a container whose text is fully represented by the
  // child — we skip it to avoid duplicate evidence.
  function textCoveredByChild(el) {
    const ownText = (el.innerText || el.textContent || "").trim();
    if (!ownText) return false;
    for (const child of el.querySelectorAll("*")) {
      if (elSet.has(child)) {
        const childText = (child.innerText || child.textContent || "").trim();
        if (childText && ownText.includes(childText) && childText.length >= ownText.length * 0.85) {
          return true;
        }
      }
    }
    return false;
  }

  return els
    .filter(el => !textCoveredByChild(el))
    .map((el, i) => {
      const rect = el.getBoundingClientRect();
      const cs = window.getComputedStyle(el);
      const style = {};
      for (const prop of STYLE_PROPS) style[prop] = cs.getPropertyValue(prop);

      const attrs = {};
      for (const attr of el.attributes) attrs[attr.name] = attr.value;

      let labelText = "";
      if (el.labels && el.labels.length > 0) {
        labelText = Array.from(el.labels).map(l => l.innerText || l.textContent || "").join(" ").trim();
      }

      return {
        selector: selectorFor(el, i),
        tag: el.tagName.toLowerCase(),
        text: (el.innerText || el.textContent || "").trim().slice(0, 500),
        associated_label_text: labelText.slice(0, 500),
        attributes: attrs,
        computed_style: style,
        bounding_box: [
          Math.round(rect.x), Math.round(rect.y),
          Math.round(rect.width), Math.round(rect.height)
        ],
        is_visible: rect.width > 0 && rect.height > 0 && cs.visibility !== "hidden" && cs.display !== "none",
      };
    });
}
"""


class PlaywrightBrowserGateway(BrowserGateway):
    def __init__(self, screenshot_dir: Path, headless: bool = True, timeout_ms: int = 45_000):
        self._screenshot_dir = screenshot_dir
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    async def capture(self, url: str) -> PageSnapshot:
        start = time.monotonic()
        console_errors: list[str] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self._headless)
            try:
                page = await browser.new_page(viewport={"width": 1440, "height": 900})
                page.on(
                    "console",
                    lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
                )

                # Use domcontentloaded (not networkidle) so SPAs with
                # persistent background analytics/telemetry requests don't
                # cause an indefinite hang. We add a short fixed pause after
                # navigation to let JS-rendered content settle before we
                # extract the DOM — this is cheaper and more reliable than
                # waiting for the network to fully go quiet.
                response = await page.goto(
                    url, timeout=self._timeout_ms, wait_until="domcontentloaded"
                )
                await page.wait_for_timeout(2500)
                final_url = page.url
                html = await page.content()

                raw_nodes = await page.evaluate(_EXTRACTION_SCRIPT)
                dom_nodes = [DomNode(**n) for n in raw_nodes]

                screenshot_path = self._screenshot_dir / f"{uuid.uuid4().hex}.png"
                await page.screenshot(path=str(screenshot_path), full_page=True)

                load_time_ms = int((time.monotonic() - start) * 1000)

                logger.info(
                    "Captured %s -> %s (%d nodes, %dms, http_status=%s)",
                    url,
                    final_url,
                    len(dom_nodes),
                    load_time_ms,
                    response.status if response else "unknown",
                )

                return PageSnapshot(
                    url=url,
                    final_url=final_url,
                    html=html,
                    dom_nodes=dom_nodes,
                    screenshot_path=str(screenshot_path),
                    viewport_width=1440,
                    viewport_height=900,
                    console_errors=console_errors,
                    load_time_ms=load_time_ms,
                )
            finally:
                await browser.close()


class PlaywrightPdfRenderer(PdfRendererPort):
    """
    Renders report HTML to PDF via headless Chromium's native
    print-to-PDF — not a separate PDF templating library. This is the
    reason `report_generator.to_html()` exists as the single source of
    truth for report layout (see that module's docstring): the PDF is
    always exactly what the HTML report looks like when printed, so a
    layout change in one place never drifts out of sync with the other.
    """

    async def render_html_to_pdf(self, html: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content(html, wait_until="networkidle")
                await page.pdf(path=str(output_path), format="A4", print_background=True)
                return output_path
            finally:
                await browser.close()
