#!/usr/bin/env python
"""
DarkLens CV Dataset Statistics Reporter

Reads the assembled CV dataset and annotation metadata and prints a
comprehensive report to stdout and saves to data/cv/metadata/dataset_stats.txt.

Usage
-----
    python scripts/dataset_report.py
    python scripts/dataset_report.py --data data/cv --metadata data/cv/metadata
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

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
MIN_CLASS_INSTANCES = 50


def count_images_and_annotations(data_dir: Path) -> dict:
    """Reads images/ and labels/ directories, returns counts per split and class."""
    result = {}
    for split in SPLITS:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split
        if not img_dir.exists():
            continue
        images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        class_counts: Counter = Counter()
        images_with_annotations = 0
        for img in images:
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                lines = [l for l in lbl.read_text().splitlines() if l.strip()]
                if lines:
                    images_with_annotations += 1
                for line in lines:
                    parts = line.split()
                    if parts:
                        try:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < len(VISUAL_LABELS):
                                class_counts[VISUAL_LABELS[cls_id]] += 1
                        except ValueError:
                            pass
        result[split] = {
            "total_images": len(images),
            "images_with_annotations": images_with_annotations,
            "images_negative": len(images) - images_with_annotations,
            "class_counts": dict(class_counts),
            "total_annotations": sum(class_counts.values()),
        }
    return result


def read_screenshots_csv(metadata_dir: Path) -> list[dict]:
    p = metadata_dir / "screenshots.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_annotations_csv(metadata_dir: Path) -> list[dict]:
    p = metadata_dir / "annotations.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_review_dirs(data_dir: Path) -> dict[str, int]:
    counts = {}
    for status in ("pending", "accepted", "rejected", "uncertain"):
        review_dir = data_dir / "review" / status
        if review_dir.exists():
            counts[status] = len(list(review_dir.iterdir()))
        else:
            counts[status] = 0
    return counts


def bar(value: int, max_val: int, width: int = 30) -> str:
    if max_val == 0:
        return "[" + " " * width + "]"
    filled = int(value / max_val * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def generate_report(data_dir: Path, metadata_dir: Path) -> str:
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("  DARKLENS CV DATASET STATISTICS REPORT")
    lines.append(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # ── Review queue status ────────────────────────────────────────────────
    review_counts = read_review_dirs(data_dir)
    lines.append("")
    lines.append("REVIEW QUEUE STATUS")
    lines.append("-" * 40)
    total_in_review = sum(review_counts.values())
    for status, count in review_counts.items():
        lines.append(f"  {status:<12} : {count:>5}")
    lines.append(f"  {'Total':<12} : {total_in_review:>5}")

    # ── Per-split counts ──────────────────────────────────────────────────
    split_data = count_images_and_annotations(data_dir)
    total_images_final = sum(v["total_images"] for v in split_data.values())
    total_annotations = sum(v["total_annotations"] for v in split_data.values())
    total_positives = sum(v["images_with_annotations"] for v in split_data.values())
    total_negatives = sum(v["images_negative"] for v in split_data.values())

    lines.append("")
    lines.append("DATASET SPLIT SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  {'Split':<10} {'Images':>8} {'Positive':>10} {'Negative':>10} {'Annotations':>13}")
    lines.append(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*13}")
    for split in SPLITS:
        if split not in split_data:
            continue
        d = split_data[split]
        lines.append(
            f"  {split:<10} {d['total_images']:>8} {d['images_with_annotations']:>10} "
            f"{d['images_negative']:>10} {d['total_annotations']:>13}"
        )
    lines.append(f"  {'TOTAL':<10} {total_images_final:>8} {total_positives:>10} {total_negatives:>10} {total_annotations:>13}")

    # ── Per-class instance counts ──────────────────────────────────────────
    lines.append("")
    lines.append("PER-CLASS INSTANCE COUNTS (all splits combined)")
    lines.append("-" * 70)
    combined_class_counts: Counter = Counter()
    for split, d in split_data.items():
        for cls, n in d["class_counts"].items():
            combined_class_counts[cls] += n
    max_count = max(combined_class_counts.values()) if combined_class_counts else 1
    lines.append(f"  {'Class':<35} {'Count':>6}  Distribution")
    lines.append(f"  {'-'*35} {'-'*6}  {'-'*32}")
    for cls in VISUAL_LABELS:
        n = combined_class_counts.get(cls, 0)
        flag = " ⚠ LOW" if n < MIN_CLASS_INSTANCES else ""
        lines.append(f"  {cls:<35} {n:>6}  {bar(n, max_count, 28)}{flag}")

    # ── Per-class per-split breakdown ──────────────────────────────────────
    lines.append("")
    lines.append("PER-CLASS COUNT BY SPLIT")
    lines.append("-" * 70)
    header = f"  {'Class':<35}"
    for split in SPLITS:
        if split in split_data:
            header += f" {split:>8}"
    lines.append(header)
    lines.append(f"  {'-'*35}" + "".join(f" {'-'*8}" for s in SPLITS if s in split_data))
    for cls in VISUAL_LABELS:
        row = f"  {cls:<35}"
        for split in SPLITS:
            if split not in split_data:
                continue
            n = split_data[split]["class_counts"].get(cls, 0)
            row += f" {n:>8}"
        lines.append(row)

    # ── Screenshots metadata analysis ─────────────────────────────────────
    screenshots = read_screenshots_csv(metadata_dir)
    if screenshots:
        lines.append("")
        lines.append("SCREENSHOT METADATA ANALYSIS")
        lines.append("-" * 40)

        domain_counts: Counter = Counter(s["domain"] for s in screenshots)
        page_type_counts: Counter = Counter(s.get("page_type", "unknown") for s in screenshots)
        category_counts: Counter = Counter(s.get("category", "unknown") for s in screenshots)

        lines.append(f"  Total screenshots crawled  : {len(screenshots)}")
        lines.append("")
        lines.append("  Screenshots per category:")
        for cat, cnt in category_counts.most_common():
            lines.append(f"    {cat:<30} : {cnt:>4}")
        lines.append("")
        lines.append("  Screenshots per page type:")
        for pt, cnt in page_type_counts.most_common():
            lines.append(f"    {pt:<30} : {cnt:>4}")
        lines.append("")
        lines.append(f"  Unique domains crawled     : {len(domain_counts)}")
        lines.append("  Top 10 domains by screenshot count:")
        for domain, cnt in domain_counts.most_common(10):
            lines.append(f"    {domain:<35} : {cnt:>4}")

    # ── Annotation metadata analysis ───────────────────────────────────────
    annotations = read_annotations_csv(metadata_dir)
    if annotations:
        lines.append("")
        lines.append("ANNOTATION METADATA")
        lines.append("-" * 40)
        source_counts: Counter = Counter(a.get("annotation_source", "unknown") for a in annotations)
        status_counts: Counter = Counter(a.get("verification_status", "unknown") for a in annotations)
        lines.append("  Annotation sources:")
        for src, cnt in source_counts.most_common():
            lines.append(f"    {src:<25} : {cnt:>5}")
        lines.append("  Verification status:")
        for st, cnt in status_counts.most_common():
            lines.append(f"    {st:<25} : {cnt:>5}")

    # ── Flags ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("FLAGS")
    lines.append("-" * 40)
    flags = []
    for cls in VISUAL_LABELS:
        n = combined_class_counts.get(cls, 0)
        if n < MIN_CLASS_INSTANCES:
            flags.append(
                f"  ⚠  '{cls}' has only {n} instances (< {MIN_CLASS_INSTANCES}). "
                "Collect more examples before training."
            )
    if not flags:
        lines.append("  ✓ No critical flags. All classes meet minimum instance threshold.")
    else:
        lines.extend(flags)

    lines.append("")
    lines.append(sep)
    lines.append("  Run scripts/validate_cv_dataset.py for full quality checks.")
    lines.append("  Run scripts/train_cv_detector.py --data data/cv/data.yaml after validation.")
    lines.append(sep)

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cv"),
                        help="Root CV dataset directory (default: data/cv)")
    parser.add_argument("--metadata", type=Path, default=Path("data/cv/metadata"),
                        help="Metadata directory (default: data/cv/metadata)")
    parser.add_argument("--output", type=Path, default=Path("data/cv/metadata/dataset_stats.txt"),
                        help="Output stats text file path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    report = generate_report(args.data, args.metadata)
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
