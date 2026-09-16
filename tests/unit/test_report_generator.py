from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from pathlib import Path

from darklens.application.ports.pdf_renderer import PdfRendererPort
from darklens.domain.entities.dark_pattern import (
    DarkPatternFinding,
    Evidence,
    EvidenceSourceType,
    PatternCategory,
    ScanResult,
    Severity,
)
from darklens.infrastructure.reporting.report_generator import (
    to_csv,
    to_html,
    to_json,
    write_pdf_report,
    write_report,
)


class FakePdfRenderer(PdfRendererPort):
    def __init__(self):
        self.received_html: str | None = None

    async def render_html_to_pdf(self, html: str, output_path: Path) -> Path:
        self.received_html = html
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("fake pdf bytes")
        return output_path


def _make_result(findings: list[DarkPatternFinding] | None = None) -> ScanResult:
    return ScanResult(
        url="https://example.com",
        scanned_at=datetime(2026, 1, 1, tzinfo=UTC),
        trust_score=72.0,
        risk_score=28.0,
        findings=findings or [],
        scan_duration_ms=1500,
    )


def _finding() -> DarkPatternFinding:
    return DarkPatternFinding(
        id="f1",
        category=PatternCategory.SNEAKING,
        title="Pre-checked opt-out consent checkbox",
        severity=Severity.HIGH,
        confidence=0.9,
        evidence=[
            Evidence(
                source=EvidenceSourceType.RULE_ENGINE,
                description="checkbox #newsletter is prechecked",
                matched_rule_id="PRECHECKED_OPTOUT_CHECKBOX",
                raw_confidence=1.0,
            )
        ],
        reasoning="Flagged as prechecked based on 1 instance.",
        suggested_redesign="Default the checkbox to unchecked.",
    )


def test_to_json_round_trips_through_scanresult():
    result = _make_result([_finding()])
    parsed = ScanResult.model_validate_json(to_json(result))
    assert parsed.url == result.url
    assert len(parsed.findings) == 1


def test_to_html_escapes_url_and_includes_scores():
    result = _make_result()
    output = to_html(result)
    assert "72.0" in output
    assert "28.0" in output
    assert "example.com" in output


def test_to_html_with_no_findings_says_so():
    output = to_html(_make_result())
    assert "No dark patterns detected" in output


def test_to_csv_has_one_row_per_finding():
    result = _make_result([_finding()])
    rows = list(csv.reader(io.StringIO(to_csv(result))))
    assert len(rows) == 2  # header + one finding
    header, row = rows
    assert header[0] == "url"
    assert row[header.index("title")] == "Pre-checked opt-out consent checkbox"
    assert row[header.index("severity")] == "high"


def test_to_csv_with_no_findings_has_only_header():
    rows = list(csv.reader(io.StringIO(to_csv(_make_result()))))
    assert len(rows) == 1


def test_write_report_writes_only_requested_formats(tmp_path):
    result = _make_result([_finding()])
    paths = write_report(result, tmp_path, formats=("json", "csv"))
    assert set(paths) == {"json", "csv"}
    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert not (tmp_path / "report.html").exists()


async def test_write_pdf_report_delegates_to_renderer(tmp_path):
    result = _make_result([_finding()])
    renderer = FakePdfRenderer()

    pdf_path = await write_pdf_report(result, tmp_path, renderer)

    assert pdf_path == tmp_path / "report.pdf"
    assert pdf_path.exists()
    # The PDF is exactly the HTML report, printed — single source of
    # layout truth, see report_generator.py's module docstring.
    assert renderer.received_html == to_html(result)
