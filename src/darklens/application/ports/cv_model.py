"""
Port for whatever turns a screenshot into a set of visual pattern
detections. Same reasoning as `TextClassifierPort`: application-layer
code depends on this interface, not on ultralytics/YOLO directly, so
`CvDetector` is unit-testable with a fake and swapping the backbone
(YOLOv8 -> YOLOv11, or to a totally different detector architecture) is
confined to `infrastructure/cv/`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from darklens.domain.cv.labels import VisualPatternLabel


class Detection(BaseModel):
    label: VisualPatternLabel
    confidence: float
    bounding_box: tuple[int, int, int, int]  # (x, y, width, height) in image pixels


class CvModelPort(ABC):
    @abstractmethod
    async def detect(self, screenshot_path: str) -> list[Detection]:
        """Run object detection on the full-page screenshot at `screenshot_path`."""
