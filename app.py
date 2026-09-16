"""
DarkLens — Streamlit Web App
Run with: streamlit run app.py
"""
from __future__ import annotations

import asyncio
import base64
import io
import sys
import time
from pathlib import Path

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DarkLens — Dark Pattern Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

  /* Hero header */
  .hero {
    background: linear-gradient(135deg, #0f1129 0%, #181b35 100%);
    border: 1px solid #2e3452;
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    text-align: center;
  }
  .hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #6c63ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }
  .hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    line-height: 1.6;
  }

  /* Score cards */
  .metric-card {
    background: #141724;
    border: 1px solid #2e3452;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
  }
  .metric-value { font-size: 2.8rem; font-weight: 800; line-height: 1; margin-bottom: 4px; }
  .metric-label { font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }

  /* Finding cards */
  .finding-card {
    background: #141724;
    border: 1px solid #2e3452;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
  }
  .finding-card:hover { border-color: #6c63ff; }
  .severity-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 6px;
    color: #000;
    margin-right: 8px;
  }
  .finding-title { font-size: 1.05rem; font-weight: 700; margin: 8px 0 6px; color: #f1f5f9; }
  .finding-category {
    font-size: 0.75rem; color: #a5b4fc;
    background: rgba(99,102,241,0.12); border-radius: 4px;
    padding: 2px 8px; display: inline-block; margin-bottom: 12px;
  }
  .reasoning-box {
    background: rgba(0,0,0,0.2);
    border-left: 3px solid #6c63ff;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.88rem;
    color: #cbd5e1;
    margin-bottom: 12px;
  }
  .evidence-item {
    background: #1d2136;
    border: 1px solid #2e3452;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.82rem;
    margin-bottom: 6px;
    color: #cbd5e1;
  }
  .evidence-source {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; color: #818cf8;
    background: rgba(108,99,255,0.12);
    padding: 1px 7px; border-radius: 4px;
    margin-right: 6px;
  }
  .redesign-box {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #6ee7b7;
    margin-top: 10px;
  }

  /* Sample buttons */
  div[data-testid="column"] .stButton > button {
    background: #1d2136;
    border: 1px solid #2e3452;
    color: #cbd5e1;
    border-radius: 8px;
    font-size: 0.82rem;
    padding: 6px 14px;
    width: 100%;
    transition: all 0.15s;
  }
  div[data-testid="column"] .stButton > button:hover {
    border-color: #6c63ff;
    color: #fff;
    background: #252a42;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
_SEVERITY_COLORS = {
    "low": "#38bdf8",
    "medium": "#f59e0b",
    "high": "#f97316",
    "critical": "#ef4444",
}

SAMPLE_SITES = [
    "https://www.booking.com",
    "https://www.agoda.com",
    "https://www.snapdeal.com",
    "https://www.expedia.com",
    "https://www.ticketmaster.com",
    "https://www.ryanair.com",
]

def run_scan(url: str):
    """Run a DarkLens scan in a dedicated thread with its own event loop.

    Streamlit runs inside an asyncio event loop on Windows that uses the
    Selector policy, which cannot spawn subprocesses (Playwright needs that).
    Spinning up a clean thread with a fresh ProactorEventLoop (Windows default
    for new threads) sidesteps the conflict entirely.
    """
    import concurrent.futures

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    def _worker():
        import asyncio
        # On Windows, Playwright needs ProactorEventLoop to spawn subprocesses.
        # Streamlit sets SelectorEventLoop as the global policy which blocks this.
        # We override inside this thread before creating the loop.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from darklens.container import build_scan_use_case
            use_case = build_scan_use_case()
            return loop.run_until_complete(use_case.execute(url))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_worker)
        return future.result(timeout=120)


def image_to_b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode()


def get_annotated_screenshot(result):
    """Draws red bounding box outlines on the screenshot for any visual evidence."""
    if not result.screenshot_path or not Path(result.screenshot_path).exists():
        return None
    try:
        from PIL import Image, ImageDraw
        img = Image.open(result.screenshot_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        has_boxes = False
        for finding in result.findings:
            for ev in finding.evidence:
                if ev.bounding_box:
                    x, y, w, h = ev.bounding_box
                    draw.rectangle([x, y, x + w, y + h], outline="#ef4444", width=5)
                    has_boxes = True
        if has_boxes:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        pass
    return result.screenshot_path


def render_score_card(label: str, value, color: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-value" style="color:{color}">{value}</div>
      <div class="metric-label">{label}</div>
    </div>"""


def render_finding_card(f, index: int) -> str:
    color = _SEVERITY_COLORS.get(f.severity.value, "#888")
    evidence_html = "".join(
        f'<div class="evidence-item">'
        f'<span class="evidence-source">{e.source.value}</span>{e.description}'
        f'</div>'
        for e in f.evidence
    )
    redesign_html = (
        f'<div class="redesign-box">💡 <strong>Recommended Redesign:</strong> {f.suggested_redesign}</div>'
        if f.suggested_redesign else ""
    )
    category = f.category.value.replace("_", " ").title()
    return f"""
    <div class="finding-card">
      <span class="severity-badge" style="background:{color}">{f.severity.value}</span>
      <span style="font-size:0.78rem;color:#94a3b8">Confidence {f.confidence:.0%}</span>
      <div class="finding-title">#{index} {f.title}</div>
      <div class="finding-category">Category: {category}</div>
      <div class="reasoning-box">{f.reasoning}</div>
      <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;color:#64748b;margin-bottom:8px;">Evidence ({len(f.evidence)})</div>
      {evidence_html}
      {redesign_html}
    </div>"""


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🛡️ DarkLens</div>
  <div class="hero-sub">
    AI-Powered Dark Pattern Detection &amp; Web Compliance Audit<br>
    <strong style="color:#818cf8">DOM Rule Engine · DistilBERT NLP · YOLOv8 Computer Vision</strong>
  </div>
</div>
""", unsafe_allow_html=True)


# ── URL Input ─────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    url_input = st.text_input(
        label="Website URL",
        placeholder="https://www.booking.com",
        label_visibility="collapsed",
    )
with col_btn:
    scan_clicked = st.button("⚡ Scan", use_container_width=True, type="primary")

# Sample buttons
st.markdown("<div style='margin-top:-8px;font-size:0.78rem;color:#64748b;margin-bottom:4px;'>Try a sample site:</div>", unsafe_allow_html=True)
sample_cols = st.columns(len(SAMPLE_SITES))
for i, site in enumerate(SAMPLE_SITES):
    with sample_cols[i]:
        label = site.replace("https://www.", "").replace("https://", "").split("/")[0]
        if st.button(label, key=f"sample_{i}"):
            st.session_state["url_override"] = site
            st.rerun()

# Handle sample click
if "url_override" in st.session_state:
    url_input = st.session_state.pop("url_override")


# ── Scan Logic ────────────────────────────────────────────────────────────────
if scan_clicked and url_input:
    if not url_input.startswith("http"):
        url_input = "https://" + url_input

    with st.spinner(""):
        placeholder = st.empty()
        steps = [
            "🌐  Launching Playwright Headless Chromium…",
            "🔎  Running DOM & CSS Rule Engine…",
            "🧠  Running DistilBERT NLP Classifier…",
            "👁️   Running YOLOv8 CV Detector…",
            "🔗  Fusing Evidence with Noisy-OR…",
        ]
        for step in steps:
            placeholder.markdown(
                f"<div style='padding:12px 0;font-size:0.95rem;color:#94a3b8'>{step}</div>",
                unsafe_allow_html=True,
            )
            time.sleep(0.3)

        try:
            result = run_scan(url_input)
            placeholder.empty()
        except Exception as exc:
            placeholder.empty()
            st.error(f"❌ Scan failed: {exc}")
            st.stop()

    st.success(f"✅ Scan Complete — {result.scan_duration_ms}ms")

    # ── Score Dashboard ───────────────────────────────────────────────────────
    trust_color = "#10b981" if result.trust_score >= 70 else ("#f59e0b" if result.trust_score >= 40 else "#ef4444")
    risk_color = "#ef4444" if result.risk_score >= 60 else ("#f59e0b" if result.risk_score >= 30 else "#10b981")

    sc1, sc2, sc3 = st.columns(3)
    sc1.markdown(render_score_card("Trust Index", f"{result.trust_score}%", trust_color), unsafe_allow_html=True)
    sc2.markdown(render_score_card("Risk Index", f"{result.risk_score}%", risk_color), unsafe_allow_html=True)
    sc3.markdown(render_score_card("Dark Pattern Findings", len(result.findings), "#6c63ff"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Screenshot + Findings side by side ───────────────────────────────────
    if result.findings:
        left_col, right_col = st.columns([1.2, 1.8])

        with left_col:
            st.markdown("#### 📸 Page Screenshot")
            if result.screenshot_path and Path(result.screenshot_path).exists():
                annotated_img = get_annotated_screenshot(result)
                st.image(annotated_img, width="stretch", caption=result.url)
            else:
                st.info("Screenshot capture active.")

        with right_col:
            st.markdown(f"#### 📋 Findings ({len(result.findings)})")
            for i, finding in enumerate(result.findings):
                st.markdown(render_finding_card(finding, i + 1), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#141724;border:1px solid #2e3452;border-radius:14px;padding:40px;text-align:center;'>
          <div style='font-size:2rem;margin-bottom:12px;'>✅</div>
          <div style='font-size:1.2rem;font-weight:700;color:#10b981;margin-bottom:8px;'>Clean Website</div>
          <div style='color:#94a3b8'>No manipulative dark patterns detected on this page.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Download Reports ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📥 Download Report")

    from darklens.infrastructure.reporting.report_generator import to_html, to_json, to_csv

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "📄 Download HTML Report",
            data=to_html(result).encode("utf-8"),
            file_name="darklens_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📊 Download JSON Report",
            data=to_json(result).encode("utf-8"),
            file_name="darklens_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "📋 Download CSV Report",
            data=to_csv(result).encode("utf-8"),
            file_name="darklens_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

elif not url_input and scan_clicked:
    st.warning("⚠️ Please enter a website URL to scan.")
