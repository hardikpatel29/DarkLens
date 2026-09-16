"""
End-to-end test: launches a real Chromium instance via the real
PlaywrightBrowserGateway, captures a local fixture page with known dark
patterns baked in, and asserts the full pipeline (capture -> rule engine
-> fusion -> scoring) produces the expected findings.

This is skipped automatically if Chromium isn't installed locally
(`playwright install chromium`), rather than failing CI with a confusing
browser-launch error. That's a deliberate choice: a skipped test with a
clear reason is honest; a red CI run that's "expected to fail in this
sandbox" trains people to ignore CI, which is worse than not having the
test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from darklens.application.services.fusion_engine import FusionEngine
from darklens.application.use_cases.scan_website import ScanWebsiteUseCase
from darklens.infrastructure.browser.playwright_gateway import PlaywrightBrowserGateway
from darklens.infrastructure.rules.rule_engine import RuleEngineDetector

FIXTURE = Path(__file__).parent.parent / "fixtures" / "dark_pattern_sample.html"


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


_SKIP_REASON = "Chromium not installed (run: playwright install chromium)"


@pytest.mark.skipif(not _chromium_available(), reason=_SKIP_REASON)
@pytest.mark.asyncio
async def test_full_pipeline_detects_known_patterns(tmp_path):
    gateway = PlaywrightBrowserGateway(screenshot_dir=tmp_path / "shots", headless=True)
    use_case = ScanWebsiteUseCase(
        browser_gateway=gateway,
        detectors=[RuleEngineDetector()],
        fusion_engine=FusionEngine(),
    )

    result = await use_case.execute(f"file://{FIXTURE.resolve()}")

    fired_rules = {e.matched_rule_id for f in result.findings for e in f.evidence}
    assert "PRECHECKED_OPTOUT_CHECKBOX" in fired_rules
    assert "HIDDEN_UNSUBSCRIBE_LINK" in fired_rules
    assert "INTERFACE_INTERFERENCE_CONSENT" in fired_rules
    assert result.trust_score < 100.0
    assert Path(result.screenshot_path).exists()
