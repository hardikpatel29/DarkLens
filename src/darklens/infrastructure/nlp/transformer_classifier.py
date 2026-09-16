"""
TransformerTextClassifier: the ONLY file in this codebase that imports
torch/transformers, same discipline as `PlaywrightBrowserGateway` being
the only file that imports playwright. Everything else depends on
`TextClassifierPort`.

Architecture choice, and why it's not "just use BERT" or "just prompt
an LLM":

  Backbone: DistilBERT-base-uncased (Sanh et al., 2019), fine-tuned here
  with a linear classification head over 10 classes — not used
  zero-shot, per the project's explicit "do not use zero-shot
  prompting" constraint, and for a concrete reason beyond following
  instructions: zero-shot label-description matching against an LLM has
  no calibrated confidence score tied to a fixed label set, and this
  system's fusion/report layer needs a real, comparable `raw_confidence`
  float per prediction — that requires a model actually trained against
  this label set's decision boundary, not one guessing from a prompt.

  Why DistilBERT over full BERT-base or a larger model: the inputs here
  are short (`text_extraction._MAX_CHARS = 200`, usually a phrase or one
  sentence — button labels, banners, checkbox copy), the labeled dataset
  this fine-tunes on (Mathur et al. 2019 / Yamana Lab, ~2-3k labeled
  sentences) is small enough that a larger backbone mostly adds
  overfitting risk and inference latency, not accuracy. DistilBERT
  retains ~97% of BERT-base's GLUE performance at 60% of the inference
  time and 40% of the parameters (Sanh et al.) — and this classifier
  needs to run synchronously inside a per-scan latency budget on
  commodity CPU-only infra, which a resume/demo deployment realistically
  is. If a later benchmarking pass (`[USER TASK]` in the training
  script) shows DistilBERT's accuracy is the bottleneck rather than
  latency, swapping to a larger backbone is a one-line change to
  `--base-model` in that script — not an architecture change here.

Loading contract: `model_path` must be a local directory produced by
`model.save_pretrained()` / `tokenizer.save_pretrained()` (see
`scripts/train_nlp_classifier.py`). This class does not download a
model from the Hub at request time — a production service silently
pulling network-hosted weights on first request is a reliability and
supply-chain problem, not a convenience worth having.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from darklens.application.ports.text_classifier import ClassificationResult, TextClassifierPort
from darklens.domain.exceptions import ModelNotAvailableError
from darklens.domain.nlp.labels import NlpLabel

logger = logging.getLogger(__name__)


class TransformerTextClassifier(TextClassifierPort):
    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        max_sequence_length: int = 64,
        batch_size: int = 16,
    ) -> None:
        if not model_path.exists():
            raise ModelNotAvailableError(
                detector_name="nlp_classifier",
                model_path=str(model_path),
                reason="path does not exist — run scripts/train_nlp_classifier.py first",
            )

        try:
            # Imported lazily, inside __init__, not at module top level.
            # This module is only imported by container.py inside a
            # try/except (see container.py) precisely so that a base
            # `pip install darklens` — no `[nlp]` extra — never fails
            # to import the package as a whole just because this one
            # optional adapter exists on disk.
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelNotAvailableError(
                detector_name="nlp_classifier",
                model_path=str(model_path),
                reason="torch/transformers not installed — pip install 'darklens[nlp]'",
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as exc:
            raise ModelNotAvailableError(
                detector_name="nlp_classifier", model_path=str(model_path), reason=str(exc)
            ) from exc

        self._model.eval()
        self._device = device
        self._model.to(device)
        self._max_sequence_length = max_sequence_length
        self._batch_size = batch_size
        self._torch = torch

        # The training script writes id2label using NlpLabel.value strings
        # (see scripts/train_nlp_classifier.py), so this mapping is a
        # direct lookup, not a guess at index order.
        self._id2label: dict[int, NlpLabel] = {
            int(i): NlpLabel(v) for i, v in self._model.config.id2label.items()
        }
        logger.info(
            "Loaded NLP classifier from %s (%d labels, device=%s)",
            model_path,
            len(self._id2label),
            device,
        )

    async def predict_batch(self, texts: list[str]) -> list[ClassificationResult]:
        if not texts:
            return []
        # Inference is a blocking CPU/GPU call; offload it so it doesn't
        # block the event loop that's also serving other concurrent scans.
        return await asyncio.to_thread(self._predict_batch_sync, texts)

    def _predict_batch_sync(self, texts: list[str]) -> list[ClassificationResult]:
        torch = self._torch
        results: list[ClassificationResult] = []

        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self._max_sequence_length,
                return_tensors="pt",
            ).to(self._device)

            encoded_inputs = {
                k: v for k, v in encoded.items() if k in ("input_ids", "attention_mask")
            }

            with torch.no_grad():
                logits = self._model(**encoded_inputs).logits
                probs = torch.softmax(logits, dim=-1)

            top_confidence, top_index = probs.max(dim=-1)
            for confidence, index in zip(top_confidence.tolist(), top_index.tolist(), strict=True):
                results.append(
                    ClassificationResult(label=self._id2label[index], confidence=confidence)
                )

        return results
