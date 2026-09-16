"""
NlpClassifierDetector: wires domain-layer text extraction to a
TextClassifierPort and exposes the result as Evidence, same shape as
RuleEngineDetector wires domain Rules to the rule execution loop.

Contract this holds to, same as every Detector: it never decides a
finding's severity or writes report copy — it emits raw Evidence with
`nlp_label` + `raw_confidence` set, and `pattern_catalog` /
`FusionEngine` decide what that means. See detector.py's port docstring.
"""
from __future__ import annotations

import logging

from darklens.application.ports.detector import Detector
from darklens.application.ports.text_classifier import TextClassifierPort
from darklens.domain.entities.dark_pattern import Evidence, EvidenceSourceType
from darklens.domain.entities.page_snapshot import PageSnapshot
from darklens.domain.nlp.labels import NlpLabel
from darklens.domain.nlp.text_extraction import extract_candidate_texts

logger = logging.getLogger(__name__)


class NlpClassifierDetector(Detector):
    def __init__(self, classifier: TextClassifierPort, confidence_threshold: float = 0.6) -> None:
        self._classifier = classifier
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "nlp_classifier"

    async def detect(self, snapshot: PageSnapshot) -> list[Evidence]:
        candidates = extract_candidate_texts(snapshot)
        if not candidates:
            return []

        predictions = await self._classifier.predict_batch([c.text for c in candidates])

        evidence: list[Evidence] = []
        for candidate, prediction in zip(candidates, predictions, strict=True):
            # NORMAL is not evidence of anything — it's the classifier
            # saying "this is ordinary copy". Filtering it here, rather
            # than giving it a no-op catalog entry, keeps the fusion
            # engine's "every group in `by_pattern` is an actual
            # candidate finding" invariant true for every source, not
            # just the rule engine.
            if prediction.label == NlpLabel.NORMAL:
                continue
            if prediction.confidence < self._confidence_threshold:
                continue

            evidence.append(
                Evidence(
                    source=EvidenceSourceType.NLP_CLASSIFIER,
                    description=(
                        f"Text classified as '{prediction.label.value}' "
                        f"(confidence {prediction.confidence:.2f}): "
                        f'"{candidate.text}"'
                    ),
                    dom_selector=candidate.selector,
                    extracted_text=candidate.text,
                    nlp_label=prediction.label,
                    raw_confidence=prediction.confidence,
                )
            )

        logger.info(
            "NLP classifier produced %d evidence item(s) for %s from %d candidate string(s)",
            len(evidence),
            snapshot.url,
            len(candidates),
        )
        return evidence
