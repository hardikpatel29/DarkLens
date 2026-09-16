from __future__ import annotations

from darklens.application.ports.cv_model import CvModelPort, Detection
from darklens.application.ports.ocr import OcrPort
from darklens.domain.cv.labels import VisualPatternLabel
from darklens.domain.entities.dark_pattern import EvidenceSourceType
from darklens.infrastructure.cv.cv_detector import CvDetector
from tests.fixtures.snapshot_builder import make_snapshot


class FakeCvModel(CvModelPort):
    def __init__(self, detections: list[Detection]):
        self._detections = detections

    async def detect(self, screenshot_path: str) -> list[Detection]:
        return self._detections


class FakeOcrEngine(OcrPort):
    def __init__(self, text: str = ""):
        self._text = text
        self.calls: list[tuple[str, tuple[int, int, int, int]]] = []

    async def extract_text(self, screenshot_path, bounding_box) -> str:
        self.calls.append((screenshot_path, bounding_box))
        return self._text


async def test_filters_out_low_confidence_detections():
    snapshot = make_snapshot([])
    cv_model = FakeCvModel(
        [Detection(label=VisualPatternLabel.COUNTDOWN_TIMER, confidence=0.3, bounding_box=(0, 0, 10, 10))]
    )
    detector = CvDetector(cv_model=cv_model, ocr_engine=None, confidence_threshold=0.5)

    evidence = await detector.detect(snapshot)
    assert evidence == []


async def test_produces_evidence_for_confident_detection():
    snapshot = make_snapshot([])
    cv_model = FakeCvModel(
        [
            Detection(
                label=VisualPatternLabel.TINY_CLOSE_BUTTON,
                confidence=0.87,
                bounding_box=(100, 200, 12, 12),
            )
        ]
    )
    detector = CvDetector(cv_model=cv_model, ocr_engine=None, confidence_threshold=0.5)

    evidence = await detector.detect(snapshot)
    assert len(evidence) == 1
    item = evidence[0]
    assert item.source == EvidenceSourceType.CV_DETECTOR
    assert item.cv_label == VisualPatternLabel.TINY_CLOSE_BUTTON
    assert item.raw_confidence == 0.87
    assert item.bounding_box == (100, 200, 12, 12)


async def test_calls_ocr_only_for_text_bearing_labels():
    snapshot = make_snapshot([])
    cv_model = FakeCvModel(
        [
            Detection(
                label=VisualPatternLabel.COUNTDOWN_TIMER, confidence=0.9, bounding_box=(0, 0, 50, 20)
            ),
            Detection(
                label=VisualPatternLabel.TINY_CLOSE_BUTTON, confidence=0.9, bounding_box=(0, 0, 12, 12)
            ),
        ]
    )
    ocr = FakeOcrEngine(text="00:04:59")
    detector = CvDetector(cv_model=cv_model, ocr_engine=ocr, confidence_threshold=0.5)

    evidence = await detector.detect(snapshot)
    assert len(ocr.calls) == 1  # only the countdown timer, not the close button
    countdown = next(e for e in evidence if e.cv_label == VisualPatternLabel.COUNTDOWN_TIMER)
    close_button = next(e for e in evidence if e.cv_label == VisualPatternLabel.TINY_CLOSE_BUTTON)
    assert countdown.extracted_text == "00:04:59"
    assert close_button.extracted_text is None


async def test_runs_without_ocr_engine_configured():
    snapshot = make_snapshot([])
    cv_model = FakeCvModel(
        [Detection(label=VisualPatternLabel.COUNTDOWN_TIMER, confidence=0.9, bounding_box=(0, 0, 50, 20))]
    )
    detector = CvDetector(cv_model=cv_model, ocr_engine=None, confidence_threshold=0.5)

    evidence = await detector.detect(snapshot)
    assert len(evidence) == 1
    assert evidence[0].extracted_text is None
