"""
Centralized, typed configuration. One rule: nothing else in the codebase
reads `os.environ` directly. If a new setting is needed, it gets added
here with a type and a default, so a misconfigured deployment fails at
startup with a clear pydantic validation error — not three requests into
a scan with an obscure KeyError.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DARKLENS_", env_file=".env", extra="ignore")

    # Browser capture
    headless_browser: bool = True
    browser_timeout_ms: int = 45_000
    screenshot_dir: Path = Path("./data/screenshots")

    # Storage
    database_url: str = "sqlite+aiosqlite:///./data/darklens.db"

    # NLP classifier (Phase 2). `nlp_model_path` points at a directory
    # produced by `scripts/train_nlp_classifier.py` (a HF
    # `save_pretrained()` checkpoint: config.json, model weights,
    # tokenizer files). It intentionally has no default that points
    # anywhere real — see container.py for what happens when the path
    # doesn't exist: the detector is skipped, not a crash.
    nlp_model_path: Path = Path("./models/nlp_classifier")
    nlp_device: str = "cpu"
    nlp_confidence_threshold: float = 0.6
    nlp_max_sequence_length: int = 64
    nlp_batch_size: int = 16

    # CV detector (Phase 4). `cv_model_path` points at a `.pt` checkpoint
    # produced by `scripts/train_cv_detector.py` (a fine-tuned YOLO
    # export). Same "no default that points anywhere real" reasoning as
    # nlp_model_path — see container.py for the graceful-skip behavior.
    cv_model_path: Path = Path("./models/cv_detector/best.pt")
    cv_confidence_threshold: float = 0.5
    ocr_enabled: bool = True
    ocr_languages: tuple[str, ...] = ("en",)
    ocr_use_gpu: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"


settings = Settings()
