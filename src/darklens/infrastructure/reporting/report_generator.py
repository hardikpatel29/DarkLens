"""
Report generation. JSON is the API's native output (it's just the
ScanResult's own schema — no separate DTO needed, pydantic gives us
`.model_dump_json()` for free). HTML is a minimal, dependency-free
template for a human-readable report. CSV is a flat, evidence-level
export for spreadsheet analysis (one row per finding). PDF renders this
same HTML via a headless-browser print-to-PDF (`PlaywrightPdfRenderer`)
rather than a second templating path, so there's exactly one source of
truth for report layout.
"""
from __future__ import annotations

import csv
import html
import io
from pathlib import Path

from darklens.application.ports.pdf_renderer import PdfRendererPort
from darklens.domain.entities.dark_pattern import ScanResult

_SEVERITY_COLOR = {
    "low": "#38bdf8",
    "medium": "#f59e0b",
    "high": "#f97316",
    "critical": "#ef4444",
}


def to_json(result: ScanResult) -> str:
    return result.model_dump_json(indent=2)


def to_html(result: ScanResult) -> str:
    findings_html = "\n".join(_render_finding(f, i + 1) for i, f in enumerate(result.findings)) or (
        "<div class='none-card'><h3>✓ Clean Website</h3><p>No dark patterns detected on this page.</p></div>"
    )

    trust_color = "#10b981" if result.trust_score >= 70 else ("#f59e0b" if result.trust_score >= 40 else "#ef4444")
    risk_color = "#ef4444" if result.risk_score >= 60 else ("#f59e0b" if result.risk_score >= 30 else "#10b981")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DarkLens Security & Compliance Report — {html.escape(result.url)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0b0d14;
    --surface: #141724;
    --card: #1d2136;
    --border: #2e3452;
    --primary: #6c63ff;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif;
    line-height: 1.6; padding: 32px 16px;
  }}
  .container {{ max-width: 920px; margin: 0 auto; }}
  
  /* Header */
  .header {{
    background: linear-gradient(135deg, #181b2e 0%, #101221 100%);
    border: 1px solid var(--border); border-radius: 16px; padding: 28px;
    margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  }}
  .brand {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }}
  .brand-logo {{ display: flex; align-items: center; gap: 10px; font-size: 1.25rem; font-weight: 800; color: var(--primary); }}
  .brand-badge {{ background: rgba(108, 99, 255, 0.15); color: #818cf8; border: 1px solid rgba(108, 99, 255, 0.3); font-size: 0.75rem; padding: 4px 10px; border-radius: 20px; font-weight: 600; }}
  .url-title {{ font-size: 1.35rem; font-weight: 700; word-break: break-all; margin-bottom: 6px; color: #fff; }}
  .meta-text {{ font-size: 0.82rem; color: var(--text-muted); display: flex; gap: 16px; flex-wrap: wrap; }}

  /* Score Dashboard */
  .scores-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .score-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px;
    display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
    position: relative; overflow: hidden;
  }}
  .score-card .value {{ font-size: 2.5rem; font-weight: 800; line-height: 1; margin-bottom: 6px; }}
  .score-card .label {{ font-size: 0.82rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .progress-bar-bg {{ width: 100%; background: var(--card); height: 6px; border-radius: 3px; margin-top: 12px; overflow: hidden; }}
  .progress-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.6s ease; }}

  /* Section Title */
  .section-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  .section-title {{ font-size: 1.1rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }}

  /* Findings Cards */
  .finding-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 24px; margin-bottom: 20px; transition: transform 0.2s, border-color 0.2s;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  }}
  .finding-card:hover {{ border-color: var(--primary); transform: translateY(-2px); }}
  .finding-header {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }}
  .finding-title {{ font-size: 1.1rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }}
  .severity-tag {{ font-size: 0.72rem; font-weight: 800; text-transform: uppercase; padding: 4px 10px; border-radius: 6px; color: #000; letter-spacing: 0.04em; }}
  .confidence-tag {{ font-size: 0.78rem; font-weight: 600; color: var(--text-muted); background: var(--card); padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); }}

  .category-badge {{ font-size: 0.75rem; color: #a5b4fc; background: rgba(99, 102, 241, 0.1); padding: 2px 8px; border-radius: 4px; font-weight: 500; margin-bottom: 12px; display: inline-block; }}
  
  .reasoning-box {{ font-size: 0.92rem; color: #cbd5e1; margin-bottom: 16px; line-height: 1.6; background: rgba(0,0,0,0.15); padding: 12px 16px; border-radius: 8px; border-left: 3px solid var(--primary); }}

  /* Evidence List */
  .evidence-section {{ margin-top: 16px; }}
  .evidence-title {{ font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px; letter-spacing: 0.05em; }}
  .evidence-item {{
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px; margin-bottom: 8px; font-size: 0.85rem; display: flex; flex-direction: column; gap: 4px;
  }}
  .evidence-source {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--primary); width: fit-content; background: rgba(108, 99, 255, 0.12); padding: 2px 8px; border-radius: 4px; }}
  .evidence-selector {{ font-family: monospace; font-size: 0.8rem; background: #0f111a; padding: 4px 8px; border-radius: 4px; color: #38bdf8; word-break: break-all; margin-top: 4px; }}

  /* Suggested Redesign */
  .redesign-box {{
    margin-top: 16px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 8px; padding: 12px 16px; font-size: 0.88rem; color: #6ee7b7;
    display: flex; gap: 10px; align-items: flex-start;
  }}
  .redesign-icon {{ font-size: 1.1rem; flex-shrink: 0; }}

  .none-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 40px; text-align: center; color: var(--text-muted); }}
  .none-card h3 {{ color: #10b981; font-size: 1.2rem; margin-bottom: 8px; }}

  @media (max-width: 640px) {{
    body {{ padding: 16px 8px; }}
    .finding-card {{ padding: 16px; }}
  }}
</style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <div class="header">
      <div class="brand">
        <div class="brand-logo">🛡️ DarkLens</div>
        <div class="brand-badge">Multi-Modal AI Audit</div>
      </div>
      <div class="url-title">{html.escape(result.url)}</div>
      <div class="meta-text">
        <span>🕒 Scanned: {result.scanned_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
        <span>⚡ Latency: {result.scan_duration_ms}ms</span>
        <span>🔍 Detectors Active: DOM Rules, DistilBERT NLP, YOLOv8 CV</span>
      </div>
    </div>

    <!-- Executive Score Summary -->
    <div class="scores-grid">
      <div class="score-card">
        <div class="value" style="color: {trust_color}">{result.trust_score}%</div>
        <div class="label">Trust Index</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: {result.trust_score}%; background: {trust_color}"></div>
        </div>
      </div>
      <div class="score-card">
        <div class="value" style="color: {risk_color}">{result.risk_score}%</div>
        <div class="label">Risk Index</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: {result.risk_score}%; background: {risk_color}"></div>
        </div>
      </div>
      <div class="score-card">
        <div class="value" style="color: #6c63ff">{len(result.findings)}</div>
        <div class="label">Dark Pattern Findings</div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: {min(100, len(result.findings) * 20)}%; background: #6c63ff"></div>
        </div>
      </div>
    </div>

    <!-- Findings Section -->
    <div class="section-header">
      <div class="section-title">
        <span>📋</span> Detailed Deceptive Pattern Findings ({len(result.findings)})
      </div>
    </div>

    {findings_html}
  </div>
</body>
</html>"""


def _render_finding(f, index: int) -> str:
    color = _SEVERITY_COLOR.get(f.severity.value, "#888")
    evidence_html = "".join(
        f"""<div class="evidence-item">
          <div class="evidence-source">Source: {e.source.value}</div>
          <div>{html.escape(e.description)}</div>
          {f'<div class="evidence-selector">Selector: {html.escape(e.dom_selector)}</div>' if e.dom_selector else ''}
        </div>"""
        for e in f.evidence
    )

    redesign_html = (
        f"""<div class="redesign-box">
          <div class="redesign-icon">💡</div>
          <div><strong>Recommended Redesign:</strong> {html.escape(f.suggested_redesign)}</div>
        </div>"""
        if f.suggested_redesign
        else ""
    )

    return f"""<div class="finding-card">
  <div class="finding-header">
    <div class="finding-title">
      <span>#{index}</span> {html.escape(f.title)}
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
      <span class="severity-tag" style="background:{color}">{f.severity.value}</span>
      <span class="confidence-tag">Confidence {f.confidence:.0%}</span>
    </div>
  </div>
  <div class="category-badge">Category: {html.escape(f.category.value.replace('_', ' ').title())}</div>
  <div class="reasoning-box">{html.escape(f.reasoning)}</div>
  <div class="evidence-section">
    <div class="evidence-title">Verifiable Evidence ({len(f.evidence)})</div>
    {evidence_html}
  </div>
  {redesign_html}
</div>"""


def to_csv(result: ScanResult) -> str:
    """
    One row per finding, not per evidence item — a finding is the unit a
    spreadsheet user actually wants to filter/sort by (severity,
    confidence, category). Evidence descriptions for a multi-evidence
    finding are joined into one cell rather than exploded into extra
    rows, so row count matches `len(result.findings)` predictably.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "url",
            "scanned_at",
            "category",
            "title",
            "severity",
            "confidence",
            "evidence_count",
            "evidence_summary",
            "suggested_redesign",
        ]
    )
    for f in result.findings:
        evidence_summary = " | ".join(e.description for e in f.evidence)
        writer.writerow(
            [
                result.url,
                result.scanned_at.isoformat(),
                f.category.value,
                f.title,
                f.severity.value,
                f"{f.confidence:.4f}",
                len(f.evidence),
                evidence_summary,
                f.suggested_redesign or "",
            ]
        )
    return buffer.getvalue()


def write_report(
    result: ScanResult,
    output_dir: Path,
    formats: tuple[str, ...] = ("json", "html"),
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if "json" in formats:
        p = output_dir / "report.json"
        p.write_text(to_json(result), encoding="utf-8")
        paths["json"] = p
    if "html" in formats:
        p = output_dir / "report.html"
        p.write_text(to_html(result), encoding="utf-8")
        paths["html"] = p
    if "csv" in formats:
        p = output_dir / "report.csv"
        p.write_text(to_csv(result), encoding="utf-8")
        paths["csv"] = p
    return paths


async def write_pdf_report(
    result: ScanResult, output_dir: Path, renderer: PdfRendererPort
) -> Path:
    """
    Separate from `write_report` (which is sync) because PDF rendering
    needs an async browser call — see PlaywrightPdfRenderer. Takes the
    renderer as a parameter rather than constructing one here, same
    dependency-injection discipline as everything else: this module
    stays free of any playwright import.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return await renderer.render_html_to_pdf(to_html(result), output_dir / "report.pdf")
