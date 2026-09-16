"""
Port for text extraction from an image region.

The project brief explicitly asks for OCR to stay swappable ("keep OCR
modular so another engine can easily replace it"). This interface is
that seam. `infrastructure/ocr/easyocr_engine.py` is today's
implementation — see that module's docstring for why EasyOCR was chosen
over Tesseract — but nothing outside `infrastructure/ocr/` and
`container.py` knows that.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class OcrPort(ABC):
    @abstractmethod
    async def extract_text(
        self, screenshot_path: str, bounding_box: tuple[int, int, int, int]
    ) -> str:
        """
        Extract text from the region of the image at `screenshot_path`
        defined by `bounding_box` = (x, y, width, height) in image pixels.
        Returns an empty string if no text is found — never raises for
        "no text detected", only for actual I/O or engine failures.
        """
