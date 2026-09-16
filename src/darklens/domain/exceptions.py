"""
Specific exception types. The API layer maps each of these to a
specific HTTP status and error body — 'something went wrong, 500' is not
an acceptable error contract for a tool people will point at arbitrary
untrusted URLs.
"""
from __future__ import annotations


class DarkLensError(Exception):
    """Base class for all DarkLens domain errors."""


class PageCaptureError(DarkLensError):
    """Raised when a URL cannot be rendered/captured (timeout, DNS failure, etc.)."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to capture {url}: {reason}")


class InvalidUrlError(DarkLensError):
    """Raised when the input is not a scannable http(s) URL."""


class ModelNotAvailableError(DarkLensError):
    """
    Raised when a learned-model detector (NLP classifier, CV detector)
    is asked to load a model that hasn't been trained/exported yet.

    This is deliberately NOT a fatal startup error. The composition root
    catches it, logs a clear message, and runs without that detector —
    see container.py. A resume project that hard-crashes on `uvicorn
    ...` because nobody has run the Phase 2 training script yet is worse
    than one that degrades to "rule engine only" and says so in the logs.
    """

    def __init__(self, detector_name: str, model_path: str, reason: str):
        self.detector_name = detector_name
        self.model_path = model_path
        self.reason = reason
        super().__init__(f"{detector_name}: model not available at {model_path} ({reason})")
