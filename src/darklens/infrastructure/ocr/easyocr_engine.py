"""
EasyOcrEngine: the ONLY file in this codebase that imports easyocr.

OCR engine choice, and why it's not a coin flip:

  Candidates considered: Tesseract (pytesseract), EasyOCR, PaddleOCR.

  Tesseract is the default "obvious" choice (mature, zero GPU
  requirement, tiny footprint) but its accuracy on short, stylized UI
  text — small badge text, low-contrast banner text, non-standard fonts
  on colored backgrounds, which is exactly what dark-pattern UI elements
  tend to use — is noticeably worse than modern deep-learning OCR
  engines in published comparisons, because Tesseract's traditional
  page-segmentation pipeline assumes fairly clean document-style text.

  PaddleOCR is generally the strongest of the three on benchmarks, but
  adds a second deep-learning framework (PaddlePaddle) to a project
  already standardized on PyTorch (torch backs both the NLP classifier
  and YOLO detector) — one more framework is one more thing to install,
  version-pin, and explain, for a benchmarking-time win that hasn't
  been measured on THIS project's actual screenshots yet.

  EasyOCR is the pragmatic middle: PyTorch-based (keeps the project on
  one deep-learning framework end to end), noticeably more robust than
  Tesseract on stylized short text, and has a stable, simple Python API
  well-suited to "crop a region, extract text" rather than full-page
  document OCR.

  This is a reasoned default, not a final benchmarked answer — see the
  [USER TASK] below, per the project's OCR module requirement that
  engine selection depending on benchmarking be marked explicitly rather
  than asserted.

[USER TASK]
Before relying on this in a real report: run all three candidates
(Tesseract, EasyOCR, PaddleOCR) against a labeled sample of real
dark-pattern screenshot crops (countdown timers, discount badges,
cookie banners) and measure character/word-level accuracy and latency
per crop. Record the results in docs/ocr_benchmark.md and either
confirm EasyOCR or swap this file for the better-measured engine — the
`OcrPort` interface makes that a contained change.
"""
from __future__ import annotations

import asyncio
import logging

from darklens.application.ports.ocr import OcrPort
from darklens.domain.exceptions import ModelNotAvailableError

logger = logging.getLogger(__name__)


class EasyOcrEngine(OcrPort):
    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False) -> None:
        try:
            # Imported lazily inside __init__ — see container.py's
            # try/except around this class for why.
            import easyocr
            from PIL import Image
        except ImportError as exc:
            raise ModelNotAvailableError(
                detector_name="ocr",
                model_path="<easyocr model cache>",
                reason="easyocr/Pillow not installed — pip install 'darklens[cv]'",
            ) from exc

        self._reader = easyocr.Reader(list(languages), gpu=gpu)
        self._Image = Image
        logger.info("Loaded EasyOCR reader (languages=%s, gpu=%s)", languages, gpu)

    async def extract_text(
        self, screenshot_path: str, bounding_box: tuple[int, int, int, int]
    ) -> str:
        return await asyncio.to_thread(self._extract_text_sync, screenshot_path, bounding_box)

    def _extract_text_sync(
        self, screenshot_path: str, bounding_box: tuple[int, int, int, int]
    ) -> str:
        x, y, width, height = bounding_box
        with self._Image.open(screenshot_path) as image:
            crop = image.crop((x, y, x + width, y + height))
            results = self._reader.readtext(
                # easyocr accepts a numpy array or file path; converting
                # the crop directly avoids writing a temp file per region.
                self._to_array(crop),
                detail=0,  # text only, not per-word bounding boxes/confidences
            )
        return " ".join(results).strip()

    @staticmethod
    def _to_array(pil_image):
        import numpy as np

        return np.array(pil_image.convert("RGB"))
