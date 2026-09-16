#!/usr/bin/env python
"""
DarkLens Annotation Review UI

Lightweight browser-based interface for human verification of Florence-2
annotation proposals. Run locally; no authentication required.

Usage
-----
    python tools/annotation_ui/app.py
    python tools/annotation_ui/app.py --proposals data/cv/raw/florence_predictions
    python tools/annotation_ui/app.py --port 7860

Then open http://localhost:7860 in your browser.

Features
--------
- Shows each screenshot with Florence candidate bounding boxes overlaid.
- Draw, resize, and delete bounding boxes on canvas.
- Accept / Edit / Reject / Mark Uncertain / Mark Negative controls.
- Keyboard shortcuts: a=accept, r=reject, u=uncertain, n=negative, →=next, ←=prev
- Saves verified annotations as YOLO .txt files in data/cv/review/accepted/.
- Writes human-corrected metadata to data/cv/metadata/annotations.csv.
- Progress bar and filter by class / domain / review status.

Architecture
------------
Single-file FastAPI app. HTML/JS/CSS are served inline so there are no
static file dependencies. The JS canvas annotation editor is vanilla JS
with no framework dependencies.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("annotation_ui")

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_PROPOSALS_DIR = Path("data/cv/raw/florence_predictions")
DEFAULT_SCREENSHOTS_DIR = Path("data/cv/raw/screenshots")
DEFAULT_ACCEPTED_DIR = Path("data/cv/review/accepted")
DEFAULT_REJECTED_DIR = Path("data/cv/review/rejected")
DEFAULT_UNCERTAIN_DIR = Path("data/cv/review/uncertain")
DEFAULT_ANNOTATIONS_CSV = Path("data/cv/metadata/annotations.csv")

VISUAL_LABELS = [
    "fake_purchase_notification",
    "countdown_timer",
    "discount_badge",
    "floating_overlay",
    "subscription_popup",
    "cookie_popup",
    "tiny_close_button",
    "urgency_banner",
    "scarcity_label",
]
CLASS_INDEX = {label: i for i, label in enumerate(VISUAL_LABELS)}

# ── HTML Template ──────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DarkLens — Annotation UI</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d2e; --card: #242740;
    --border: #363a5a; --accent: #6c63ff; --accent2: #ff6584;
    --text: #e8eaf6; --text-muted: #9499c4; --success: #4ade80;
    --warn: #fbbf24; --danger: #f87171;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif;
         display: flex; height: 100vh; overflow: hidden; }
  /* Sidebar */
  #sidebar { width: 300px; min-width: 260px; background: var(--surface); border-right: 1px solid var(--border);
             display: flex; flex-direction: column; overflow: hidden; }
  #sidebar-header { padding: 16px; border-bottom: 1px solid var(--border); }
  #sidebar-header h1 { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
  #progress-bar-wrap { margin-top: 8px; background: var(--card); border-radius: 4px; height: 6px; }
  #progress-bar { height: 6px; background: var(--accent); border-radius: 4px; transition: width 0.3s; }
  #progress-label { font-size: 0.7rem; color: var(--text-muted); margin-top: 4px; }
  #filter-row { padding: 8px 16px; border-bottom: 1px solid var(--border); display: flex; gap: 6px; flex-wrap: wrap; }
  #filter-row select, #filter-row input { background: var(--card); color: var(--text); border: 1px solid var(--border);
      border-radius: 4px; padding: 4px 8px; font-size: 0.75rem; width: 100%; }
  #image-list { flex: 1; overflow-y: auto; }
  .image-item { padding: 10px 14px; cursor: pointer; border-bottom: 1px solid var(--border);
                display: flex; gap: 8px; align-items: center; transition: background 0.15s; font-size: 0.78rem; }
  .image-item:hover { background: var(--card); }
  .image-item.active { background: var(--card); border-left: 3px solid var(--accent); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .status-dot.pending { background: var(--warn); }
  .status-dot.accepted { background: var(--success); }
  .status-dot.rejected { background: var(--text-muted); }
  .status-dot.uncertain { background: var(--accent2); }
  .item-meta { color: var(--text-muted); font-size: 0.68rem; }
  /* Main area */
  #main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #toolbar { padding: 10px 16px; background: var(--surface); border-bottom: 1px solid var(--border);
             display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .btn { padding: 7px 14px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.82rem;
         font-weight: 600; transition: all 0.15s; }
  .btn-accept { background: var(--success); color: #000; }
  .btn-reject { background: var(--text-muted); color: #fff; }
  .btn-uncertain { background: var(--accent2); color: #fff; }
  .btn-negative { background: var(--card); color: var(--text); border: 1px solid var(--border); }
  .btn-save { background: var(--accent); color: #fff; }
  .btn-nav { background: var(--card); color: var(--text); border: 1px solid var(--border); }
  .btn:hover { opacity: 0.85; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  #class-select { background: var(--card); color: var(--text); border: 1px solid var(--border);
                  border-radius: 6px; padding: 7px 10px; font-size: 0.82rem; }
  #shortcut-hint { margin-left: auto; font-size: 0.7rem; color: var(--text-muted); }
  /* Canvas area */
  #canvas-wrap { flex: 1; overflow: auto; background: #080a12; display: flex;
                 justify-content: center; align-items: flex-start; padding: 20px; }
  #canvas-container { position: relative; display: inline-block; }
  canvas { display: block; cursor: crosshair; border: 2px solid var(--border); border-radius: 4px; }
  /* Right panel */
  #right-panel { width: 280px; background: var(--surface); border-left: 1px solid var(--border);
                 display: flex; flex-direction: column; overflow-y: auto; }
  #annotation-list { flex: 1; padding: 12px; }
  .ann-item { background: var(--card); border-radius: 6px; margin-bottom: 8px; padding: 10px;
              border: 1px solid var(--border); font-size: 0.78rem; position: relative; }
  .ann-item.selected-ann { border-color: var(--accent); }
  .ann-label { font-weight: 700; margin-bottom: 4px; }
  .ann-meta { color: var(--text-muted); font-size: 0.7rem; }
  .ann-delete { position: absolute; top: 6px; right: 8px; background: none; border: none;
                color: var(--danger); cursor: pointer; font-size: 1rem; }
  #notes-area { padding: 12px; border-top: 1px solid var(--border); }
  #notes-area label { font-size: 0.75rem; color: var(--text-muted); display: block; margin-bottom: 4px; }
  #notes { width: 100%; background: var(--card); color: var(--text); border: 1px solid var(--border);
           border-radius: 6px; padding: 8px; font-size: 0.78rem; resize: vertical; min-height: 60px; }
  #meta-panel { padding: 12px; border-top: 1px solid var(--border); font-size: 0.72rem; color: var(--text-muted); line-height: 1.6; }
  /* Toast */
  #toast { position: fixed; bottom: 24px; right: 24px; background: var(--accent);
           color: #fff; padding: 10px 18px; border-radius: 8px; font-size: 0.82rem;
           display: none; z-index: 9999; box-shadow: 0 4px 20px rgba(108,99,255,0.4); }
</style>
</head>
<body>

<!-- Sidebar: image list -->
<div id="sidebar">
  <div id="sidebar-header">
    <h1>🔍 DarkLens Annotator</h1>
    <div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
    <div id="progress-label">0 / 0 reviewed</div>
  </div>
  <div id="filter-row">
    <select id="filter-status">
      <option value="">All statuses</option>
      <option value="pending">Pending</option>
      <option value="accepted">Accepted</option>
      <option value="rejected">Rejected (negative)</option>
      <option value="uncertain">Uncertain</option>
    </select>
    <select id="filter-class">
      <option value="">All classes</option>
      <option value="fake_purchase_notification">fake_purchase_notification</option>
      <option value="countdown_timer">countdown_timer</option>
      <option value="discount_badge">discount_badge</option>
      <option value="floating_overlay">floating_overlay</option>
      <option value="subscription_popup">subscription_popup</option>
      <option value="cookie_popup">cookie_popup</option>
      <option value="tiny_close_button">tiny_close_button</option>
      <option value="urgency_banner">urgency_banner</option>
      <option value="scarcity_label">scarcity_label</option>
    </select>
    <input type="text" id="filter-domain" placeholder="Filter by domain…">
  </div>
  <div id="image-list"></div>
</div>

<!-- Main: toolbar + canvas -->
<div id="main">
  <div id="toolbar">
    <button class="btn btn-nav" onclick="navigate(-1)">← Prev</button>
    <button class="btn btn-nav" onclick="navigate(1)">Next →</button>
    <div style="width:1px;height:24px;background:var(--border)"></div>
    <select id="class-select">
      <option value="fake_purchase_notification">fake_purchase_notification</option>
      <option value="countdown_timer">countdown_timer</option>
      <option value="discount_badge">discount_badge</option>
      <option value="floating_overlay">floating_overlay</option>
      <option value="subscription_popup">subscription_popup</option>
      <option value="cookie_popup">cookie_popup</option>
      <option value="tiny_close_button">tiny_close_button</option>
      <option value="urgency_banner">urgency_banner</option>
      <option value="scarcity_label">scarcity_label</option>
    </select>
    <button class="btn btn-accept" onclick="acceptAll()" title="Accept all Florence candidates (a)">✓ Accept All (a)</button>
    <button class="btn btn-negative" onclick="markNegative()" title="No dark patterns here (n)">✕ Negative (n)</button>
    <button class="btn btn-uncertain" onclick="markUncertain()" title="Mark as uncertain (u)">? Uncertain (u)</button>
    <button class="btn btn-reject" onclick="markRejected()" title="Reject all / clear (r)">✗ Reject (r)</button>
    <button class="btn btn-save" onclick="saveAnnotation()" title="Save verified annotation">💾 Save</button>
    <span id="shortcut-hint">Draw box on canvas → assign class → Save</span>
  </div>
  <div id="canvas-wrap">
    <div id="canvas-container">
      <canvas id="canvas" width="1440" height="900"></canvas>
    </div>
  </div>
</div>

<!-- Right panel: annotation list + notes -->
<div id="right-panel">
  <div style="padding:12px;border-bottom:1px solid var(--border);font-size:0.8rem;font-weight:700;">
    Annotations <span id="ann-count" style="color:var(--text-muted)">0</span>
  </div>
  <div id="annotation-list"></div>
  <div id="notes-area">
    <label>Notes / rationale</label>
    <textarea id="notes" placeholder="Optional annotation note…"></textarea>
  </div>
  <div id="meta-panel" id="meta-panel"></div>
</div>

<div id="toast"></div>

<script>
// ── State ────────────────────────────────────────────────────────────────────
const VISUAL_LABELS = %LABELS_JSON%;
const CLASS_INDEX = Object.fromEntries(VISUAL_LABELS.map((l, i) => [l, i]));
const BOX_COLORS = [
  '#6c63ff','#ff6584','#4ade80','#fbbf24','#38bdf8',
  '#f97316','#a78bfa','#34d399','#fb7185'
];

let allImages = [];
let filteredImages = [];
let currentIdx = -1;
let currentProposal = null;
let annotations = [];   // [{label, bbox_pixels:[x1,y1,x2,y2], score, source, verified}]
let selectedAnn = -1;
let drawing = false;
let drawStart = {x:0, y:0};
let drawCurrent = {x:0, y:0};
let imgEl = null;
let scale = 1.0;

// ── Canvas setup ─────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const pos = canvasPos(e);
  drawing = true;
  drawStart = pos;
  drawCurrent = pos;
});
canvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  drawCurrent = canvasPos(e);
  redraw();
});
canvas.addEventListener('mouseup', e => {
  if (!drawing) return;
  drawing = false;
  const pos = canvasPos(e);
  const x1 = Math.round(Math.min(drawStart.x, pos.x) / scale);
  const y1 = Math.round(Math.min(drawStart.y, pos.y) / scale);
  const x2 = Math.round(Math.max(drawStart.x, pos.x) / scale);
  const y2 = Math.round(Math.max(drawStart.y, pos.y) / scale);
  if ((x2-x1)*(y2-y1) < 100) { redraw(); return; }  // too small
  const label = document.getElementById('class-select').value;
  annotations.push({label, class_id: CLASS_INDEX[label], bbox_pixels:[x1,y1,x2,y2],
                    score: 1.0, source:'human', verified: true});
  selectedAnn = annotations.length - 1;
  renderAnnotationList();
  redraw();
});

function canvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return {x: e.clientX - r.left, y: e.clientY - r.top};
}

// ── Draw ─────────────────────────────────────────────────────────────────────
function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (imgEl) ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);

  annotations.forEach((ann, i) => {
    const [x1,y1,x2,y2] = ann.bbox_pixels.map(v => v * scale);
    const color = BOX_COLORS[CLASS_INDEX[ann.label] % BOX_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = i === selectedAnn ? 3 : 2;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    if (i === selectedAnn) {
      ctx.fillStyle = color + '22';
      ctx.fillRect(x1, y1, x2-x1, y2-y1);
    }
    // Label tag
    ctx.fillStyle = color;
    const tag = ann.label.replace(/_/g,' ') + (ann.score < 1.0 ? ' ' + (ann.score*100).toFixed(0)+'%%' : '');
    const tw = ctx.measureText(tag).width;
    ctx.fillRect(x1, y1 - 18, tw + 10, 18);
    ctx.fillStyle = '#000';
    ctx.font = '11px monospace';
    ctx.fillText(tag, x1 + 5, y1 - 5);
  });

  // Current draw rectangle
  if (drawing) {
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.setLineDash([4,4]);
    const dx = drawCurrent.x - drawStart.x, dy = drawCurrent.y - drawStart.y;
    ctx.strokeRect(drawStart.x, drawStart.y, dx, dy);
    ctx.setLineDash([]);
  }
}

// ── Load image ───────────────────────────────────────────────────────────────
async function loadImage(idx) {
  if (idx < 0 || idx >= filteredImages.length) return;
  currentIdx = idx;
  currentProposal = filteredImages[idx];
  annotations = [];
  selectedAnn = -1;

  // Highlight sidebar
  document.querySelectorAll('.image-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
    if (i === idx) el.scrollIntoView({block:'nearest'});
  });

  // Load proposal from server
  const resp = await fetch('/api/proposal/' + encodeURIComponent(currentProposal.image_id));
  const proposal = await resp.json();
  currentProposal = proposal;

  // Populate annotations from Florence candidates
  annotations = (proposal.candidates || []).map(c => ({
    label: c.label, class_id: c.class_id, bbox_pixels: c.bbox_pixels,
    score: c.score, source: c.annotation_source || 'florence', verified: c.verified || false,
  }));

  // Load image
  const imgResp = await fetch('/api/image/' + encodeURIComponent(currentProposal.image_id));
  const blob = await imgResp.blob();
  const url = URL.createObjectURL(blob);
  imgEl = new Image();
  imgEl.onload = () => {
    // Scale to fit canvas (max 1440 wide)
    scale = Math.min(1440 / imgEl.naturalWidth, 900 / imgEl.naturalHeight, 1.0);
    canvas.width = imgEl.naturalWidth * scale;
    canvas.height = imgEl.naturalHeight * scale;
    redraw();
  };
  imgEl.src = url;

  document.getElementById('notes').value = proposal.notes || '';
  renderAnnotationList();
  renderMeta(proposal);
}

// ── Annotation list (right panel) ─────────────────────────────────────────────
function renderAnnotationList() {
  const el = document.getElementById('annotation-list');
  document.getElementById('ann-count').textContent = annotations.length;
  el.innerHTML = annotations.map((ann, i) => `
    <div class="ann-item ${i===selectedAnn?'selected-ann':''}" onclick="selectAnn(${i})">
      <div class="ann-label" style="color:${BOX_COLORS[CLASS_INDEX[ann.label]%BOX_COLORS.length]}">
        ${ann.label.replace(/_/g,' ')}
      </div>
      <div class="ann-meta">
        Box: [${ann.bbox_pixels.map(Math.round).join(', ')}]<br>
        Score: ${(ann.score*100).toFixed(0)}%%  Source: ${ann.source}
      </div>
      <button class="ann-delete" onclick="deleteAnn(event,${i})">✕</button>
    </div>
  `).join('');
}

function selectAnn(i) {
  selectedAnn = i;
  if (i >= 0 && i < VISUAL_LABELS.length) {
    document.getElementById('class-select').value = annotations[i].label;
  }
  renderAnnotationList();
  redraw();
}

function deleteAnn(e, i) {
  e.stopPropagation();
  annotations.splice(i, 1);
  if (selectedAnn >= annotations.length) selectedAnn = annotations.length - 1;
  renderAnnotationList();
  redraw();
}

// ── Toolbar actions ──────────────────────────────────────────────────────────
function acceptAll() {
  annotations.forEach(ann => { ann.verified = true; ann.source = 'florence+human'; });
  renderAnnotationList();
  redraw();
  saveAnnotation();
}
function markNegative() {
  annotations = [];
  saveAnnotation('rejected');
}
function markUncertain() { saveAnnotation('uncertain'); }
function markRejected()  { saveAnnotation('rejected'); }

async function saveAnnotation(forceStatus) {
  if (!currentProposal) return;
  const status = forceStatus || (annotations.length > 0 ? 'accepted' : 'rejected');
  const payload = {
    image_id: currentProposal.image_id,
    annotations: annotations.map(ann => ({...ann, verified: true})),
    notes: document.getElementById('notes').value,
    status,
  };
  const resp = await fetch('/api/save', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const result = await resp.json();
  if (result.ok) {
    toast('Saved as ' + status);
    currentProposal.review_status = status;
    rebuildSidebarItem(currentIdx, status);
    updateProgress();
    navigate(1);
  } else {
    toast('Error: ' + result.error, true);
  }
}

function navigate(dir) {
  const next = currentIdx + dir;
  if (next >= 0 && next < filteredImages.length) loadImage(next);
}

// ── Sidebar ──────────────────────────────────────────────────────────────────
function renderSidebar() {
  const el = document.getElementById('image-list');
  el.innerHTML = filteredImages.map((img, i) => `
    <div class="image-item ${i===currentIdx?'active':''}" onclick="loadImage(${i})">
      <div class="status-dot ${img.review_status || 'pending'}"></div>
      <div>
        <div>${img.image_id}</div>
        <div class="item-meta">${img.domain || ''} · ${img.page_type || ''}</div>
      </div>
    </div>
  `).join('');
  updateProgress();
}

function rebuildSidebarItem(idx, status) {
  const items = document.querySelectorAll('.image-item');
  if (items[idx]) {
    const dot = items[idx].querySelector('.status-dot');
    dot.className = 'status-dot ' + status;
  }
}

function updateProgress() {
  const total = filteredImages.length;
  const done = filteredImages.filter(i => i.review_status && i.review_status !== 'pending').length;
  document.getElementById('progress-bar').style.width = (total ? done/total*100 : 0) + '%%';
  document.getElementById('progress-label').textContent = done + ' / ' + total + ' reviewed';
}

function renderMeta(p) {
  const el = document.getElementById('meta-panel');
  el.innerHTML = `
    <b>Image ID:</b> ${p.image_id}<br>
    <b>Domain:</b> ${p.domain || '—'}<br>
    <b>URL:</b> <span title="${p.image_path || ''}">${(p.image_path || '').split('/').pop()}</span><br>
    <b>Size:</b> ${p.width}×${p.height}<br>
    <b>Status:</b> ${p.review_status || 'pending'}
  `;
}

// ── Filter ───────────────────────────────────────────────────────────────────
function applyFilters() {
  const status = document.getElementById('filter-status').value;
  const cls = document.getElementById('filter-class').value;
  const domain = document.getElementById('filter-domain').value.toLowerCase();
  filteredImages = allImages.filter(img => {
    if (status && (img.review_status || 'pending') !== status) return false;
    if (domain && !(img.domain || '').toLowerCase().includes(domain)) return false;
    // class filter: check if image has a candidate with that label
    if (cls && !(img.classes || []).includes(cls)) return false;
    return true;
  });
  renderSidebar();
  if (filteredImages.length > 0) loadImage(0);
}
document.getElementById('filter-status').addEventListener('change', applyFilters);
document.getElementById('filter-class').addEventListener('change', applyFilters);
document.getElementById('filter-domain').addEventListener('input', applyFilters);

// ── Keyboard ─────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'a') acceptAll();
  else if (e.key === 'r') markRejected();
  else if (e.key === 'u') markUncertain();
  else if (e.key === 'n') markNegative();
  else if (e.key === 'ArrowRight') navigate(1);
  else if (e.key === 'ArrowLeft') navigate(-1);
  else if (e.key === 'Delete') { if (selectedAnn >= 0) deleteAnn({stopPropagation:()=>{}}, selectedAnn); }
});

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, error=false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = error ? 'var(--danger)' : 'var(--accent)';
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 2500);
}

// ── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  const resp = await fetch('/api/images');
  allImages = await resp.json();
  filteredImages = [...allImages];
  renderSidebar();
  if (filteredImages.length > 0) loadImage(0);
}
init();
</script>
</body>
</html>"""

# ── FastAPI server ─────────────────────────────────────────────────────────────

def build_app(
    proposals_dir: Path,
    screenshots_dir: Path,
    accepted_dir: Path,
    rejected_dir: Path,
    uncertain_dir: Path,
    annotations_csv: Path,
):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, Response
    except ImportError as exc:
        raise RuntimeError("FastAPI is required: pip install fastapi uvicorn") from exc

    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    uncertain_dir.mkdir(parents=True, exist_ok=True)
    annotations_csv.parent.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="DarkLens Annotation UI")

    # ── Load all proposal files from proposals_dir ─────────────────────────
    def load_proposal_index() -> list[dict]:
        index = []
        if not proposals_dir.exists():
            return index
        for p in sorted(proposals_dir.glob("*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                # Determine review status from output dirs
                status = data.get("review_status", "pending")
                if (accepted_dir / f"{p.stem}.txt").exists():
                    status = "accepted"
                elif (rejected_dir / f"{p.stem}.json").exists():
                    status = "rejected"
                elif (uncertain_dir / f"{p.stem}.json").exists():
                    status = "uncertain"
                # Collect unique classes for filter
                classes = list({c["label"] for c in data.get("candidates", [])})
                index.append({
                    "image_id": p.stem,
                    "domain": data.get("domain", ""),
                    "page_type": data.get("page_type", ""),
                    "review_status": status,
                    "classes": classes,
                    "image_path": data.get("image_path", ""),
                })
            except Exception as exc:
                logger.warning("Could not load proposal %s: %s", p.name, exc)
        return index

    @app.get("/", response_class=HTMLResponse)
    def index():
        labels_json = json.dumps(VISUAL_LABELS)
        return HTML_TEMPLATE.replace("%LABELS_JSON%", labels_json)

    @app.get("/api/images")
    def list_images():
        return JSONResponse(load_proposal_index())

    @app.get("/api/proposal/{image_id}")
    def get_proposal(image_id: str):
        p = proposals_dir / f"{image_id}.json"
        if not p.exists():
            raise HTTPException(404, "Proposal not found")
        with open(p, encoding="utf-8") as f:
            return JSONResponse(json.load(f))

    @app.get("/api/image/{image_id}")
    def get_image(image_id: str):
        # Try proposals dir to get image_path, then search screenshots_dir
        prop_path = proposals_dir / f"{image_id}.json"
        img_path = None
        if prop_path.exists():
            with open(prop_path, encoding="utf-8") as f:
                data = json.load(f)
            candidate_path = Path(data.get("image_path", ""))
            if candidate_path.exists():
                img_path = candidate_path
        if img_path is None:
            for ext in [".png", ".jpg", ".jpeg"]:
                p = screenshots_dir / f"{image_id}{ext}"
                if p.exists():
                    img_path = p
                    break
        if img_path is None:
            raise HTTPException(404, "Image file not found")
        with open(img_path, "rb") as f:
            data = f.read()
        return Response(content=data, media_type="image/png")

    @app.post("/api/save")
    async def save_annotation(payload: dict):
        try:
            image_id = payload["image_id"]
            anns = payload.get("annotations", [])
            notes = payload.get("notes", "")
            status = payload.get("status", "accepted")
            now = datetime.now(timezone.utc).isoformat()

            # Update proposal JSON with human decisions
            prop_path = proposals_dir / f"{image_id}.json"
            if prop_path.exists():
                with open(prop_path, encoding="utf-8") as f:
                    proposal = json.load(f)
            else:
                proposal = {"image_id": image_id, "candidates": []}

            proposal["review_status"] = status
            proposal["notes"] = notes
            proposal["annotator"] = "human"
            proposal["annotation_timestamp"] = now

            # Replace candidates with human-verified annotations
            proposal["candidates"] = [
                {
                    "label": a["label"],
                    "class_id": CLASS_INDEX.get(a["label"], 0),
                    "bbox_pixels": a["bbox_pixels"],
                    "score": a.get("score", 1.0),
                    "annotation_source": a.get("source", "human"),
                    "verified": True,
                }
                for a in anns
            ]

            with open(prop_path, "w", encoding="utf-8") as f:
                json.dump(proposal, f, indent=2)

            # Write YOLO label file (only for accepted with annotations)
            img_w = proposal.get("width", 1440)
            img_h = proposal.get("height", 900)

            if status == "accepted" and anns:
                yolo_lines = []
                for a in anns:
                    x1, y1, x2, y2 = a["bbox_pixels"]
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    # Clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))
                    cls_id = CLASS_INDEX.get(a["label"], 0)
                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                label_file = accepted_dir / f"{image_id}.txt"
                label_file.write_text("\n".join(yolo_lines), encoding="utf-8")
            elif status == "rejected":
                # Negative: empty label file (or store JSON marker)
                marker = rejected_dir / f"{image_id}.json"
                with open(marker, "w") as f:
                    json.dump({"image_id": image_id, "status": "rejected", "timestamp": now}, f)
            elif status == "uncertain":
                marker = uncertain_dir / f"{image_id}.json"
                with open(marker, "w") as f:
                    json.dump({"image_id": image_id, "status": "uncertain",
                               "notes": notes, "timestamp": now}, f)

            # Append to annotations.csv
            write_header = not annotations_csv.exists()
            with open(annotations_csv, "a", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                if write_header:
                    writer.writerow([
                        "image_id", "class_name", "x_center", "y_center", "width", "height",
                        "annotator", "annotation_source", "verification_status", "notes", "timestamp"
                    ])
                if anns:
                    for a in anns:
                        x1, y1, x2, y2 = a["bbox_pixels"]
                        cx = ((x1 + x2) / 2) / img_w
                        cy = ((y1 + y2) / 2) / img_h
                        bw = (x2 - x1) / img_w
                        bh = (y2 - y1) / img_h
                        writer.writerow([
                            image_id, a["label"],
                            f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}",
                            "human", a.get("source", "human"), status, notes, now,
                        ])
                else:
                    writer.writerow([image_id, "", "", "", "", "", "human", "human", status, notes, now])

            logger.info("Saved %s → %s (%d annotations)", image_id, status, len(anns))
            return JSONResponse({"ok": True})
        except Exception as exc:
            logger.error("Save failed: %s", exc)
            return JSONResponse({"ok": False, "error": str(exc)})

    return app


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS_DIR)
    parser.add_argument("--screenshots", type=Path, default=DEFAULT_SCREENSHOTS_DIR)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED_DIR)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED_DIR)
    parser.add_argument("--uncertain", type=Path, default=DEFAULT_UNCERTAIN_DIR)
    parser.add_argument("--annotations-csv", type=Path, default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required: pip install uvicorn") from exc

    app = build_app(
        proposals_dir=args.proposals,
        screenshots_dir=args.screenshots,
        accepted_dir=args.accepted,
        rejected_dir=args.rejected,
        uncertain_dir=args.uncertain,
        annotations_csv=args.annotations_csv,
    )
    logger.info("Starting DarkLens Annotation UI at http://%s:%d", args.host, args.port)
    logger.info("Proposals: %s", args.proposals)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
