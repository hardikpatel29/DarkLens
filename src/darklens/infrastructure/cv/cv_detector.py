"""
CvDetector: wires CvModelPort + OcrPort to the Detector port.

Why CV exists as a separate detector at all, instead of folding into the
rule engine or NLP classifier: some UI patterns leave no reliable DOM
signature. A countdown timer rendered via canvas/WebGL, or a fake
purchase toast built from the same generic <div> soup as legitimate UI,
looks identical to normal content in the DOM — the only place the
pattern is actually visible is the rendered pixels. That is the
entire justification for this module's existence; if a pattern CAN be
read reliably from the DOM, it belongs in the rule engine instead
(cheaper, deterministic, no model needed) — see domain/cv/labels.py.
"""
from __future__ import annotations

import logging

from darklens.application.ports.cv_model import CvModelPort
from darklens.application.ports.detector import Detector
from darklens.application.ports.ocr import OcrPort
from darklens.domain.cv.labels import TEXT_BEARING_LABELS
from darklens.domain.entities.dark_pattern import Evidence, EvidenceSourceType
from darklens.domain.entities.page_snapshot import PageSnapshot

logger = logging.getLogger(__name__)


class CvDetector(Detector):
    def __init__(
        self,
        cv_model: CvModelPort,
        ocr_engine: OcrPort | None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self._cv_model = cv_model
        # OCR is optional even when the CV model is available: a CV-only
        # deployment (bounding boxes, no text extraction) is still a
        # legitimate, useful configuration, since OCR is a separate
        # optional dependency chain (see infrastructure/ocr).
        self._ocr_engine = ocr_engine
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "cv_detector"

    async def detect(self, snapshot: PageSnapshot) -> list[Evidence]:
        detections = await self._cv_model.detect(snapshot.screenshot_path)

        evidence: list[Evidence] = []
        for detection in detections:
            if detection.confidence < self._confidence_threshold:
                continue

            extracted_text: str | None = None
            if self._ocr_engine is not None and detection.label in TEXT_BEARING_LABELS:
                extracted_text = await self._ocr_engine.extract_text(
                    snapshot.screenshot_path, detection.bounding_box
                )
                extracted_text = extracted_text or None  # empty string -> None, not ""

            description = f"Visual pattern '{detection.label.value}' detected"
            if extracted_text:
                description += f' (OCR text: "{extracted_text}")'
            description += f" (confidence {detection.confidence:.2f})"

            evidence.append(
                Evidence(
                    source=EvidenceSourceType.CV_DETECTOR,
                    description=description,
                    cv_label=detection.label,
                    extracted_text=extracted_text,
                    bounding_box=detection.bounding_box,
                    raw_confidence=detection.confidence,
                )
            )

        logger.info(
            "CV detector produced %d evidence item(s) for %s from %d raw detection(s)",
            len(evidence),
            snapshot.url,
            len(detections),
        )
        return evidence
