"""
The ScanWebsite use case: the single orchestration point that ties
capture -> detection -> fusion -> scoring together.

This class deliberately knows nothing about FastAPI, the CLI, Playwright,
or SQLite. It depends only on the BrowserGateway and Detector ports plus
the FusionEngine. That's what lets the exact same use case be called from
a REST endpoint, a CLI command, or a test — with a fake BrowserGateway
substituted in tests, no network or browser required.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from darklens.application.ports.detector import BrowserGateway, Detector
from darklens.application.services.fusion_engine import (
    FusionEngine,
    compute_trust_and_risk_scores,
)
from darklens.domain.entities.dark_pattern import Evidence, ScanResult
from darklens.domain.exceptions import PageCaptureError

logger = logging.getLogger(__name__)


class ScanWebsiteUseCase:
    def __init__(
        self,
        browser_gateway: BrowserGateway,
        detectors: list[Detector],
        fusion_engine: FusionEngine | None = None,
    ):
        self._browser_gateway = browser_gateway
        self._detectors = detectors
        self._fusion_engine = fusion_engine or FusionEngine()

    async def execute(self, url: str) -> ScanResult:
        start = time.monotonic()

        try:
            snapshot = await self._browser_gateway.capture(url)
        except Exception as exc:
            raise PageCaptureError(url=url, reason=str(exc)) from exc

        all_evidence: list[Evidence] = []
        for detector in self._detectors:
            try:
                evidence = await detector.detect(snapshot)
                all_evidence.extend(evidence)
            except Exception:
                # One detector failing shouldn't fail the whole scan; we'd
                # rather return a partial report than nothing at all, but
                # we log loudly so it's visible in monitoring.
                logger.exception("Detector '%s' failed for %s", detector.name, url)

        findings = self._fusion_engine.fuse(all_evidence)
        trust_score, risk_score = compute_trust_and_risk_scores(findings)

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Scan complete: url=%s findings=%d trust=%.1f risk=%.1f duration_ms=%d",
            url,
            len(findings),
            trust_score,
            risk_score,
            duration_ms,
        )

        return ScanResult(
            url=url,
            scanned_at=datetime.now(UTC),
            trust_score=trust_score,
            risk_score=risk_score,
            findings=findings,
            screenshot_path=snapshot.screenshot_path,
            scan_duration_ms=duration_ms,
        )
