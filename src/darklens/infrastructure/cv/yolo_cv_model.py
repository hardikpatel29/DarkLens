"""
YoloCvModel: the ONLY file in this codebase that imports ultralytics.
Same discipline as PlaywrightBrowserGateway (playwright) and
TransformerTextClassifier (torch/transformers) — one adapter file per
third-party ML dependency, everything else depends on the port.

Why YOLO and not a generic pretrained detector: per the project brief,
"DO NOT assume a generic pretrained detector solves the problem." A
pretrained COCO model has no concept of "countdown timer" or "fake
purchase notification" — those are UI-specific classes that don't exist
in any general-purpose detection dataset. This class loads a
CUSTOM-TRAINED YOLO checkpoint (see scripts/train_cv_detector.py), not
a stock `yolov8n.pt`. If `model_path` points at an off-the-shelf
pretrained checkpoint instead of a fine-tuned one, the resulting
predictions are meaningless for this task — that's a training-data
problem this class cannot detect or protect you from, only a real
mAP-evaluated eval run can.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from darklens.application.ports.cv_model import CvModelPort, Detection
from darklens.domain.cv.labels import VisualPatternLabel
from darklens.domain.exceptions import ModelNotAvailableError

logger = logging.getLogger(__name__)


class YoloCvModel(CvModelPort):
    def __init__(self, model_path: Path, confidence_threshold: float = 0.5) -> None:
        if not model_path.exists():
            raise ModelNotAvailableError(
                detector_name="cv_detector",
                model_path=str(model_path),
                reason="path does not exist — run scripts/train_cv_detector.py first",
            )
        try:
            # Imported lazily inside __init__ — see container.py's
            # try/except around this class for why.
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelNotAvailableError(
                detector_name="cv_detector",
                model_path=str(model_path),
                reason="ultralytics not installed — pip install 'darklens[cv]'",
            ) from exc

        try:
            self._model = YOLO(str(model_path))
        except Exception as exc:
            raise ModelNotAvailableError(
                detector_name="cv_detector", model_path=str(model_path), reason=str(exc)
            ) from exc

        self._confidence_threshold = confidence_threshold

        # Validate that the loaded model is actually a fine-tuned dark-pattern
        # checkpoint, not a stock COCO model.  A COCO model's class names
        # ("person", "car", "chair" …) are not valid VisualPatternLabel values,
        # so the mapping below would raise ValueError for every class — and
        # even if it somehow didn't, the resulting detections would be
        # meaningless.  We detect this up-front and raise ModelNotAvailableError
        # so container.py's try/except silently disables the CV detector
        # (returning partial results without it) rather than letting it run
        # and pollute reports with garbage.
        valid_label_values = {lbl.value for lbl in VisualPatternLabel}
        unknown_classes = [
            name for name in self._model.names.values()
            if name not in valid_label_values
        ]
        if unknown_classes:
            raise ModelNotAvailableError(
                detector_name="cv_detector",
                model_path=str(model_path),
                reason=(
                    f"Model class names {unknown_classes[:5]} are not valid "
                    "VisualPatternLabel values — this looks like a stock "
                    "pretrained checkpoint (e.g. COCO), not a fine-tuned "
                    "dark-pattern model. Run scripts/train_cv_detector.py "
                    "to produce a valid checkpoint."
                ),
            )

        # The training script writes class names as VisualPatternLabel.value
        # strings (see scripts/train_cv_detector.py), so this is a direct
        # lookup, not an assumption about index order.
        self._names: dict[int, VisualPatternLabel] = {
            int(i): VisualPatternLabel(name) for i, name in self._model.names.items()
        }
        logger.info("Loaded CV detector from %s (%d classes)", model_path, len(self._names))

    async def detect(self, screenshot_path: str) -> list[Detection]:
        return await asyncio.to_thread(self._detect_sync, screenshot_path)

    def _detect_sync(self, screenshot_path: str) -> list[Detection]:
        results = self._model.predict(
            source=screenshot_path,
            conf=self._confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=self._names[class_id],
                        confidence=float(box.conf.item()),
                        bounding_box=(x1, y1, x2 - x1, y2 - y1),
                    )
                )
        return detections
