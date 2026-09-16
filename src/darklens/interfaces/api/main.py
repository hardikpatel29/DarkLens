"""
FastAPI interface layer. This module's only job is HTTP concerns:
request validation, status codes, and delegating to the use case. No
business logic belongs here — if you find yourself writing an `if`
statement about dark patterns in this file, it's misplaced.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, HttpUrl

from darklens.config import settings
from darklens.container import build_pdf_renderer, build_scan_use_case
from darklens.domain.entities.dark_pattern import ScanResult
from darklens.domain.exceptions import PageCaptureError
from darklens.infrastructure.reporting.report_generator import to_html, write_pdf_report
from darklens.logging_config import configure_logging

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DarkLens API",
    description="Explainable AI framework for detecting dark patterns on websites.",
    version="0.1.0",
)


class ScanRequest(BaseModel):
    url: HttpUrl


@app.post("/api/v1/scan", response_model=ScanResult)
async def scan(request: ScanRequest) -> ScanResult:
    use_case = build_scan_use_case()
    try:
        return await use_case.execute(str(request.url))
    except PageCaptureError as exc:
        logger.warning("Capture failed for %s: %s", exc.url, exc.reason)
        raise HTTPException(status_code=422, detail=f"Could not render URL: {exc.reason}") from exc


@app.post("/api/v1/scan/report", response_class=HTMLResponse)
async def scan_html_report(request: ScanRequest) -> str:
    use_case = build_scan_use_case()
    try:
        result = await use_case.execute(str(request.url))
    except PageCaptureError as exc:
        raise HTTPException(status_code=422, detail=f"Could not render URL: {exc.reason}") from exc
    return to_html(result)


@app.post("/api/v1/scan/report.pdf", response_class=FileResponse)
async def scan_pdf_report(request: ScanRequest) -> FileResponse:
    use_case = build_scan_use_case()
    try:
        result = await use_case.execute(str(request.url))
    except PageCaptureError as exc:
        raise HTTPException(status_code=422, detail=f"Could not render URL: {exc.reason}") from exc

    # A per-request temp directory, not `settings.screenshot_dir` or any
    # shared path — concurrent requests must never write report.pdf to
    # the same location and race each other.
    output_dir = Path(tempfile.mkdtemp(prefix="darklens_pdf_"))
    pdf_path = await write_pdf_report(result, output_dir, build_pdf_renderer())
    return FileResponse(pdf_path, media_type="application/pdf", filename="darklens_report.pdf")


WEB_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DarkLens — Automated Dark Pattern Web Scanner</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0b0d14; --surface: #141724; --card: #1d2136; --border: #2e3452;
    --primary: #6c63ff; --primary-hover: #5b52e0; --text: #f1f5f9; --text-muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; min-height: 100vh; padding: 40px 16px; }
  .container { max-width: 960px; margin: 0 auto; }
  .header { text-align: center; margin-bottom: 36px; }
  .logo { font-size: 2.2rem; font-weight: 800; color: var(--primary); display: inline-flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .subtitle { font-size: 1rem; color: var(--text-muted); }
  
  /* Input box */
  .search-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3); margin-bottom: 32px;
  }
  .input-group { display: flex; gap: 12px; }
  .url-input {
    flex: 1; background: #0f111a; border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 18px; font-size: 1rem; color: #fff; outline: none; transition: border-color 0.2s;
  }
  .url-input:focus { border-color: var(--primary); }
  .scan-btn {
    background: var(--primary); color: #fff; border: none; border-radius: 10px;
    padding: 14px 28px; font-size: 1rem; font-weight: 700; cursor: pointer;
    transition: background 0.2s, transform 0.1s; display: flex; align-items: center; gap: 8px;
  }
  .scan-btn:hover { background: var(--primary-hover); transform: translateY(-1px); }
  .scan-btn:disabled { opacity: 0.6; cursor: not-allowed; }
  
  /* Loader */
  #loader { display: none; text-align: center; padding: 40px 20px; }
  .spinner { width: 44px; height: 44px; border: 4px solid var(--border); border-top-color: var(--primary); border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 16px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text { font-size: 0.95rem; color: var(--text-muted); font-weight: 500; }
  
  /* Quick samples */
  .sample-row { margin-top: 14px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 0.8rem; color: var(--text-muted); }
  .sample-tag { background: var(--card); border: 1px solid var(--border); padding: 4px 10px; border-radius: 6px; cursor: pointer; color: #cbd5e1; transition: all 0.15s; }
  .sample-tag:hover { border-color: var(--primary); color: #fff; }

  /* Output frame */
  #report-frame { width: 100%; height: 900px; border: 1px solid var(--border); border-radius: 16px; background: var(--surface); display: none; margin-top: 24px; }
  #error-box { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #fca5a5; padding: 16px 20px; border-radius: 12px; display: none; margin-top: 24px; }
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="logo">🛡️ DarkLens</div>
      <div class="subtitle">AI-Powered Deceptive Web Pattern Audit Framework (DOM Rules + NLP DistilBERT + CV YOLOv8)</div>
    </div>

    <div class="search-card">
      <form id="scan-form">
        <div class="input-group">
          <input type="url" id="url-input" class="url-input" placeholder="Enter website URL (e.g. https://www.booking.com)" required>
          <button type="submit" id="scan-btn" class="scan-btn">
            <span>⚡ Scan Website</span>
          </button>
        </div>
      </form>
      <div class="sample-row">
        <span>Try sample:</span>
        <span class="sample-tag" onclick="setSample('https://www.booking.com')">Booking.com</span>
        <span class="sample-tag" onclick="setSample('https://www.agoda.com')">Agoda.com</span>
        <span class="sample-tag" onclick="setSample('https://www.expedia.com')">Expedia.com</span>
        <span class="sample-tag" onclick="setSample('https://www.ticketmaster.com')">Ticketmaster.com</span>
      </div>
    </div>

    <div id="loader">
      <div class="spinner"></div>
      <div class="loading-text" id="loading-msg">Launching Playwright Headless Chromium & Analyzing Page…</div>
    </div>

    <div id="error-box"></div>
    <iframe id="report-frame"></iframe>
  </div>

  <script>
    function setSample(url) {
      document.getElementById('url-input').value = url;
    }

    document.getElementById('scan-form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const url = document.getElementById('url-input').value.trim();
      if (!url) return;

      const btn = document.getElementById('scan-btn');
      const loader = document.getElementById('loader');
      const frame = document.getElementById('report-frame');
      const errBox = document.getElementById('error-box');

      btn.disabled = true;
      loader.style.display = 'block';
      frame.style.display = 'none';
      errBox.style.display = 'none';

      try {
        const resp = await fetch('/api/v1/scan/report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        });

        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || 'Scan failed');
        }

        const html = await resp.text();
        frame.style.display = 'block';
        const doc = frame.contentWindow.document;
        doc.open();
        doc.write(html);
        doc.close();
      } catch (err) {
        errBox.textContent = '❌ Error: ' + err.message;
        errBox.style.display = 'block';
      } finally {
        btn.disabled = false;
        loader.style.display = 'none';
      }
    });
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
async def web_scanner_ui():
    return WEB_DASHBOARD_HTML


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

