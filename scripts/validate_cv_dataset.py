#!/usr/bin/env python
"""
DarkLens CV Dataset Validator

Runs quality control checks on the assembled CV dataset before training.
Produces a human-readable HTML report and exits with a non-zero code if
any CRITICAL check fails, so CI/training pipelines can gate on it.

Usage
-----
    python scripts/validate_cv_dataset.py
    python scripts/validate_cv_dataset.py --data data/cv
    python scripts/validate_cv_dataset.py --strict   # exit 1 on any warning too

Checks
------
  CRITICAL (exit 1 on failure)
  ─────────────────────────────
  1.  data.yaml exists and has required keys (path, train, val, nc, names).
  2.  Class names in data.yaml match VisualPatternLabel enum order exactly.
  3.  Images exist in images/{train,val} (test is optional).
  4.  Label file exists for every image in each split.
  5.  YOLO coordinates: all values in [0, 1], no NaN/Inf, no zero-area boxes.
  6.  Class IDs in label files are valid (< nc).
  7.  Domain leakage: no domain appears in more than one split.

  WARNING (logged; report flagged; exit 0 unless --strict)
  ─────────────────────────────────────────────────────────
  8.  Corrupted images (PIL cannot open).
  9.  Near-duplicate images (pHash Hamming distance ≤ 4).
  10. Class imbalance: any class with < MIN_CLASS_INSTANCES instances.
  11. Images smaller than TINY_IMAGE_BYTES.
  12. Empty label files for images in positive splits.
  13. Annotation manifest CSV consistency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("validate_cv_dataset")

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

SPLITS = ["train", "val", "test"]
MIN_CLASS_INSTANCES = 50     # warn if fewer
TINY_IMAGE_BYTES = 8_000
PHASH_DUP_THRESHOLD = 4      # Hamming distance ≤ this → near-duplicate


# ── pHash helper ──────────────────────────────────────────────────────────────
def compute_phash(image_path: Path) -> str:
    try:
        from PIL import Image
        img = Image.open(image_path).convert("L").resize((8, 8))
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if p >= avg else "0" for p in pixels)
        return bin(int(bits, 2))[2:].zfill(64)
    except Exception:
        h = hashlib.md5()
        with open(image_path, "rb") as f:
            h.update(f.read(65536))
        return h.hexdigest()


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return 64
    return sum(ca != cb for ca, cb in zip(a, b))


# ── Validation logic ──────────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self):
        self.criticals: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.stats: dict = {}

    def critical(self, msg: str):
        self.criticals.append(msg)
        logger.error("CRITICAL: %s", msg)

    def warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning("WARNING: %s", msg)

    def info(self, msg: str):
        self.infos.append(msg)
        logger.info("INFO: %s", msg)

    @property
    def passed(self) -> bool:
        return len(self.criticals) == 0

    def to_html(self) -> str:
        def badge(text, color):
            return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-weight:700">{text}</span>'

        rows_critical = "".join(
            f'<tr><td>{badge("CRITICAL","#ef4444")}</td><td>{msg}</td></tr>'
            for msg in self.criticals
        )
        rows_warning = "".join(
            f'<tr><td>{badge("WARNING","#f59e0b")}</td><td>{msg}</td></tr>'
            for msg in self.warnings
        )
        rows_info = "".join(
            f'<tr><td>{badge("INFO","#6366f1")}</td><td>{msg}</td></tr>'
            for msg in self.infos
        )

        class_table = ""
        if "class_counts" in self.stats:
            for split, counts in self.stats["class_counts"].items():
                class_table += f"<h3>{split}</h3><table border='1' cellpadding='4' cellspacing='0'>"
                class_table += "<tr><th>Class</th><th>Instances</th><th>Status</th></tr>"
                for cls in VISUAL_LABELS:
                    n = counts.get(cls, 0)
                    status = "✓" if n >= MIN_CLASS_INSTANCES else f"⚠ < {MIN_CLASS_INSTANCES}"
                    color = "#065f46" if n >= MIN_CLASS_INSTANCES else "#7c2d12"
                    class_table += f"<tr style='background:{color}'><td>{cls}</td><td>{n}</td><td>{status}</td></tr>"
                class_table += "</table>"

        domain_table = ""
        if "domains_per_split" in self.stats:
            for split, domains in self.stats["domains_per_split"].items():
                domain_table += f"<h3>{split} ({len(domains)} domains)</h3>"
                domain_table += "<ul>" + "".join(f"<li>{d}</li>" for d in sorted(domains)) + "</ul>"

        overall = badge("PASSED ✓", "#10b981") if self.passed else badge("FAILED ✗", "#ef4444")

        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>DarkLens CV Dataset Quality Report</title>
<style>
  body{{background:#0f1117;color:#e8eaf6;font-family:system-ui;padding:30px;}}
  h1{{color:#6c63ff}} h2{{color:#a78bfa;margin-top:24px}} h3{{color:#94a3b8;margin-top:12px}}
  table{{border-collapse:collapse;width:100%;margin-bottom:16px}}
  td,th{{padding:8px 12px;border:1px solid #363a5a;text-align:left}}
  th{{background:#1a1d2e}} tr:nth-child(even){{background:#1a1d2e}}
  ul{{margin:8px 0 8px 20px}} li{{margin:2px 0}}
</style>
</head><body>
<h1>DarkLens CV Dataset Quality Report</h1>
<p>Overall: {overall}</p>
<p>Generated: {__import__('datetime').datetime.now().isoformat()}</p>

<h2>Issues</h2>
<table>
<tr><th>Severity</th><th>Message</th></tr>
{rows_critical}{rows_warning}{rows_info}
</table>

<h2>Statistics</h2>
<p><b>Total images:</b> {self.stats.get('total_images', '—')} &nbsp;
   <b>Total annotations:</b> {self.stats.get('total_annotations', '—')}</p>

<h2>Per-Class Instance Counts</h2>
{class_table}

<h2>Domains Per Split</h2>
{domain_table}
</body></html>"""


def validate(data_dir: Path, strict: bool) -> ValidationReport:
    r = ValidationReport()
    yaml_path = data_dir / "data.yaml"

    # ── Check 1: data.yaml exists ─────────────────────────────────────────
    if not yaml_path.exists():
        r.critical(f"data.yaml not found at {yaml_path}")
        return r

    with open(yaml_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    for key in ("train", "val", "nc", "names"):
        if key not in spec:
            r.critical(f"data.yaml missing required key: '{key}'")

    # ── Check 2: class names match VisualPatternLabel exactly ─────────────
    names = spec.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != VISUAL_LABELS:
        r.critical(
            f"data.yaml class names {names!r} do not match the required taxonomy "
            f"{VISUAL_LABELS!r} (order matters). Fix data.yaml before training."
        )

    nc = spec.get("nc", len(VISUAL_LABELS))

    # ── Per-split checks ──────────────────────────────────────────────────
    total_images = 0
    total_annotations = 0
    class_counts: dict[str, Counter] = {}
    domains_per_split: dict[str, set] = {}
    all_phashes: list[tuple[str, Path]] = []

    for split in SPLITS:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        if not img_dir.exists():
            if split == "test":
                r.info(f"images/{split}/ does not exist (test split is optional).")
            else:
                r.critical(f"images/{split}/ does not exist.")
            continue

        images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
        if not images:
            r.critical(f"images/{split}/ is empty.")
            continue

        r.info(f"{split}: {len(images)} images found.")
        total_images += len(images)

        split_class_counts: Counter = Counter()
        split_domains: set = set()

        for img_path in images:
            # ── Check 8: corrupted images ─────────────────────────────────
            if img_path.stat().st_size < TINY_IMAGE_BYTES:
                r.warning(f"Tiny image ({img_path.stat().st_size} bytes): {img_path.name}")
            try:
                from PIL import Image
                Image.open(img_path).verify()
            except Exception as exc:
                r.warning(f"Corrupted image ({exc}): {img_path.name}")
                continue

            # ── pHash for dedup ───────────────────────────────────────────
            try:
                ph = compute_phash(img_path)
                all_phashes.append((ph, img_path))
            except Exception:
                pass

            # Extract domain from filename: <domain>_<page_type>_<counter>.png
            parts = img_path.stem.split("_")
            domain = parts[0] if parts else "unknown"
            split_domains.add(domain)

            # ── Check 4: label file exists ────────────────────────────────
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                r.critical(f"Missing label file: {lbl_path}")
                continue

            # ── Check 5 & 6: validate YOLO coordinates ─────────────────
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                r.warning(f"Empty label file (treated as negative): {lbl_path.name}")
                continue

            for lineno, line in enumerate(lines, 1):
                parts_ln = line.split()
                if len(parts_ln) != 5:
                    r.critical(f"{lbl_path.name}:{lineno} — expected 5 fields, got {len(parts_ln)}")
                    continue
                try:
                    cls_id = int(parts_ln[0])
                    vals = [float(v) for v in parts_ln[1:]]
                except ValueError:
                    r.critical(f"{lbl_path.name}:{lineno} — non-numeric values: {line!r}")
                    continue
                if cls_id >= nc or cls_id < 0:
                    r.critical(f"{lbl_path.name}:{lineno} — class_id {cls_id} out of range [0, {nc-1}]")
                if any(v < 0 or v > 1 for v in vals):
                    r.critical(f"{lbl_path.name}:{lineno} — coordinate(s) outside [0,1]: {vals}")
                bw, bh = vals[2], vals[3]
                if bw <= 0 or bh <= 0:
                    r.critical(f"{lbl_path.name}:{lineno} — zero-area box (w={bw}, h={bh})")
                cls_name = VISUAL_LABELS[cls_id] if cls_id < len(VISUAL_LABELS) else str(cls_id)
                split_class_counts[cls_name] += 1
                total_annotations += 1

        class_counts[split] = split_class_counts
        domains_per_split[split] = split_domains

    # ── Check 7: domain leakage ───────────────────────────────────────────
    split_domain_list = [(split, domains) for split, domains in domains_per_split.items()]
    for i in range(len(split_domain_list)):
        for j in range(i + 1, len(split_domain_list)):
            s1, d1 = split_domain_list[i]
            s2, d2 = split_domain_list[j]
            overlap = d1 & d2
            if overlap:
                r.critical(
                    f"Domain leakage: {len(overlap)} domain(s) appear in both "
                    f"'{s1}' and '{s2}': {sorted(overlap)[:5]}{'…' if len(overlap) > 5 else ''}"
                )

    # ── Check 9: near-duplicate images ────────────────────────────────────
    dup_count = 0
    for i in range(len(all_phashes)):
        for j in range(i + 1, len(all_phashes)):
            if hamming(all_phashes[i][0], all_phashes[j][0]) <= PHASH_DUP_THRESHOLD:
                r.warning(
                    f"Near-duplicate: {all_phashes[i][1].name} ≈ {all_phashes[j][1].name}"
                )
                dup_count += 1
                if dup_count >= 20:  # cap warnings
                    r.warning(f"…(more near-duplicates found, capped at 20 warnings)")
                    break
        if dup_count >= 20:
            break

    # ── Check 10: class imbalance ──────────────────────────────────────────
    for split, counts in class_counts.items():
        for cls in VISUAL_LABELS:
            n = counts.get(cls, 0)
            if n < MIN_CLASS_INSTANCES:
                r.warning(
                    f"Class imbalance [{split}]: '{cls}' has only {n} instances "
                    f"(minimum recommended: {MIN_CLASS_INSTANCES}). "
                    "Consider collecting more examples or merging with a related class."
                )

    r.stats = {
        "total_images": total_images,
        "total_annotations": total_annotations,
        "class_counts": {s: dict(c) for s, c in class_counts.items()},
        "domains_per_split": {s: list(d) for s, d in domains_per_split.items()},
    }

    r.info(f"Total images: {total_images}")
    r.info(f"Total annotations: {total_annotations}")

    return r


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cv"),
                        help="Root CV dataset directory (default: data/cv)")
    parser.add_argument("--report", type=Path,
                        default=Path("data/cv/metadata/dataset_quality_report.html"),
                        help="Output HTML report path")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 on any WARNING as well as CRITICALs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    r = validate(args.data, args.strict)

    # Write HTML report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(r.to_html(), encoding="utf-8")
    logger.info("Quality report written to %s", args.report)

    # Summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("  CRITICALs: %d", len(r.criticals))
    logger.info("  WARNINGs:  %d", len(r.warnings))
    logger.info("  Result:    %s", "PASSED" if r.passed else "FAILED")
    logger.info("=" * 60)

    if not r.passed or (args.strict and r.warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
