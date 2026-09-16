"""
Port for whatever turns a batch of strings into label predictions.

Same reasoning as `BrowserGateway` in `detector.py`: `NlpClassifierDetector`
(application-adjacent orchestration) should depend on THIS interface, not
on `transformers.AutoModelForSequenceClassification` directly. That's what
lets `tests/unit/test_nlp_classifier_detector.py` run in CI without torch
installed at all — inject a fake that returns canned predictions — and
it's what makes "swap the backbone model, or swap HF for ONNX Runtime for
faster inference" a change confined to `infrastructure/nlp/`, not a change
that ripples into detector logic or tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from darklens.domain.nlp.labels import NlpLabel


class ClassificationResult(BaseModel):
    label: NlpLabel
    confidence: float


class TextClassifierPort(ABC):
    """A batch text classifier. One call, many strings — see docstring
    on `predict_batch` for why batching is part of the contract, not an
    optimization bolted on later."""

    @abstractmethod
    async def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """
        Classify each string in `texts` independently, returning results
        in the same order and of the same length as `texts`.

        Batching is in the interface itself, not left to the caller to
        loop N times, because for a transformer the difference between
        one forward pass over a batch of 32 and 32 separate forward
        passes is roughly an order of magnitude of wall-clock time on
        CPU. A page can easily produce 30-80 text candidates
        (`domain.nlp.text_extraction`); classifying them one at a time
        would dominate total scan latency for no benefit.
        """
