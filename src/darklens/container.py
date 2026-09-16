"""
Composition root: the ONE place in the codebase where concrete
infrastructure classes are constructed and wired into application-layer
abstractions. Every other module depends on interfaces; this module is
where those interfaces get real implementations plugged in.

Phase 1 said adding the NLP classifier would mean "implement Detector,
add one line, nothing else changes." Two things about that turned out
to need more care, worth being explicit about rather than glossing over:

  1. `pattern_catalog`/`fusion_engine` needed a small, deliberate
     generalization — see fusion_engine.py's module docstring.
  2. Unlike the rule engine, learned-model detectors depend on a trained
     model that may not exist yet (this codebase ships the code; training
     the models is `[USER TASK]` work in `scripts/train_nlp_classifier.py`
     and `scripts/train_cv_detector.py`). "Add one line" would mean the
     app refuses to start on a fresh checkout with no trained checkpoint.
     Instead, `build_detectors()` tries to construct each learned-model
     detector and degrades gracefully, loudly, if its model isn't there —
     the CV detector and its OCR sub-dependency degrade independently of
     each other and of the NLP detector, so a partial checkpoint set
     still produces the best scan the available models support.
"""
from __future__ import annotations

import logging

from darklens.application.ports.detector import BrowserGateway, Detector
from darklens.application.ports.pdf_renderer import PdfRendererPort
from darklens.application.use_cases.scan_website import ScanWebsiteUseCase
from darklens.config import Settings, settings
from darklens.domain.exceptions import ModelNotAvailableError
from darklens.infrastructure.browser.playwright_gateway import (
    PlaywrightBrowserGateway,
    PlaywrightPdfRenderer,
)
from darklens.infrastructure.rules.rule_engine import RuleEngineDetector

logger = logging.getLogger(__name__)


def build_browser_gateway(cfg: Settings = settings) -> BrowserGateway:
    return PlaywrightBrowserGateway(
        screenshot_dir=cfg.screenshot_dir,
        headless=cfg.headless_browser,
        timeout_ms=cfg.browser_timeout_ms,
    )


def build_pdf_renderer() -> PdfRendererPort:
    return PlaywrightPdfRenderer()


def build_detectors(cfg: Settings = settings) -> list[Detector]:
    detectors: list[Detector] = [RuleEngineDetector()]

    nlp_detector = _try_build_nlp_detector(cfg)
    if nlp_detector is not None:
        detectors.append(nlp_detector)

    cv_detector = _try_build_cv_detector(cfg)
    if cv_detector is not None:
        detectors.append(cv_detector)

    return detectors


def _try_build_nlp_detector(cfg: Settings) -> Detector | None:
    try:
        # Imported here, not at module top level, so that a base install
        # (no `[nlp]` extra) can still import `container` — and every
        # other detector in it — without torch/transformers present.
        from darklens.infrastructure.nlp.nlp_classifier_detector import NlpClassifierDetector
        from darklens.infrastructure.nlp.transformer_classifier import TransformerTextClassifier

        classifier = TransformerTextClassifier(
            model_path=cfg.nlp_model_path,
            device=cfg.nlp_device,
            max_sequence_length=cfg.nlp_max_sequence_length,
            batch_size=cfg.nlp_batch_size,
        )
        return NlpClassifierDetector(
            classifier=classifier, confidence_threshold=cfg.nlp_confidence_threshold
        )
    except ModelNotAvailableError as exc:
        logger.warning(
            "NLP classifier detector disabled for this run: %s. "
            "Scans will proceed without it.",
            exc,
        )
        return None


def _try_build_cv_detector(cfg: Settings) -> Detector | None:
    try:
        from darklens.infrastructure.cv.cv_detector import CvDetector
        from darklens.infrastructure.cv.yolo_cv_model import YoloCvModel

        cv_model = YoloCvModel(
            model_path=cfg.cv_model_path, confidence_threshold=cfg.cv_confidence_threshold
        )
    except ModelNotAvailableError as exc:
        logger.warning(
            "CV detector disabled for this run: %s. Scans will proceed without it.", exc
        )
        return None

    ocr_engine = None
    if cfg.ocr_enabled:
        try:
            from darklens.infrastructure.ocr.easyocr_engine import EasyOcrEngine

            ocr_engine = EasyOcrEngine(languages=cfg.ocr_languages, gpu=cfg.ocr_use_gpu)
        except ModelNotAvailableError as exc:
            # OCR failing to load does NOT disable the CV detector as a
            # whole — bounding-box-only visual evidence (no extracted
            # text) is still real, useful evidence. See CvDetector's
            # constructor: ocr_engine=None is an explicitly supported mode.
            logger.warning(
                "OCR engine disabled for this run: %s. "
                "CV detector will still run, without text extraction.",
                exc,
            )

    return CvDetector(
        cv_model=cv_model, ocr_engine=ocr_engine, confidence_threshold=cfg.cv_confidence_threshold
    )


def build_scan_use_case(cfg: Settings = settings) -> ScanWebsiteUseCase:
    return ScanWebsiteUseCase(
        browser_gateway=build_browser_gateway(cfg),
        detectors=build_detectors(cfg),
    )
