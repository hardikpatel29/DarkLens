"""
Port for turning report HTML into a PDF file. Deliberately a separate
interface from `BrowserGateway`, even though today's implementation
happens to also use Playwright — `BrowserGateway.capture()` navigates to
and analyzes a live URL; this renders a static HTML string we already
generated ourselves. Conflating them would make a future swap (e.g. a
lighter-weight HTML-to-PDF library that doesn't need a full browser)
require touching the scan pipeline instead of just this one seam.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class PdfRendererPort(ABC):
    @abstractmethod
    async def render_html_to_pdf(self, html: str, output_path: Path) -> Path:
        """Render `html` to a PDF file at `output_path` and return that path."""
