from __future__ import annotations

from darklens.application.ports.text_classifier import ClassificationResult, TextClassifierPort
from darklens.domain.entities.dark_pattern import EvidenceSourceType
from darklens.domain.entities.page_snapshot import DomNode
from darklens.domain.nlp.labels import NlpLabel
from darklens.infrastructure.nlp.nlp_classifier_detector import NlpClassifierDetector
from tests.fixtures.snapshot_builder import make_snapshot


class FakeTextClassifier(TextClassifierPort):
    """Returns a fixed, ordered sequence of predictions, one per call to
    predict_batch, so tests never need torch/transformers installed."""

    def __init__(self, predictions: list[ClassificationResult]):
        self._predictions = predictions
        self.received_batches: list[list[str]] = []

    async def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        self.received_batches.append(texts)
        assert len(texts) == len(self._predictions)
        return self._predictions


async def test_filters_out_normal_predictions():
    snapshot = make_snapshot([DomNode(selector="p.copy", tag="span", text="Free shipping today")])
    classifier = FakeTextClassifier([ClassificationResult(label=NlpLabel.NORMAL, confidence=0.98)])
    detector = NlpClassifierDetector(classifier=classifier)

    evidence = await detector.detect(snapshot)
    assert evidence == []


async def test_filters_out_low_confidence_predictions():
    snapshot = make_snapshot([DomNode(selector="span.badge", tag="span", text="Only 2 left in stock")])
    classifier = FakeTextClassifier(
        [ClassificationResult(label=NlpLabel.SCARCITY, confidence=0.4)]
    )
    detector = NlpClassifierDetector(classifier=classifier, confidence_threshold=0.6)

    evidence = await detector.detect(snapshot)
    assert evidence == []


async def test_produces_evidence_for_confident_dark_pattern_prediction():
    snapshot = make_snapshot([DomNode(selector="span.badge", tag="span", text="Only 2 left in stock")])
    classifier = FakeTextClassifier(
        [ClassificationResult(label=NlpLabel.SCARCITY, confidence=0.91)]
    )
    detector = NlpClassifierDetector(classifier=classifier, confidence_threshold=0.6)

    evidence = await detector.detect(snapshot)
    assert len(evidence) == 1
    item = evidence[0]
    assert item.source == EvidenceSourceType.NLP_CLASSIFIER
    assert item.nlp_label == NlpLabel.SCARCITY
    assert item.raw_confidence == 0.91
    assert item.dom_selector == "span.badge"
    assert item.extracted_text == "Only 2 left in stock"


async def test_skips_classifier_call_entirely_when_no_candidates():
    snapshot = make_snapshot([])  # no DOM nodes at all
    classifier = FakeTextClassifier([])
    detector = NlpClassifierDetector(classifier=classifier)

    evidence = await detector.detect(snapshot)
    assert evidence == []
    assert classifier.received_batches == []
