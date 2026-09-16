#!/usr/bin/env python
"""
Prepares domain-based train/val splits for YOLOv8 training.

Reads proposal/accepted annotation files from data/cv/raw/florence_predictions
or data/cv/review/accepted, derives YOLO normalized bounding box format,
and assigns entire domains to either train (80%) or val (20%) split
to strictly prevent domain leakage.

Output structure:
    data/cv/images/train/*.png
    data/cv/images/val/*.png
    data/cv/labels/train/*.txt
    data/cv/labels/val/*.txt
    data/cv/data.yaml
"""
from __future__ import annotations

import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prepare_cv_splits")

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


def prepare_splits(
    proposals_dir: Path,
    screenshots_dir: Path,
    output_dir: Path,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> None:
    accepted_dir = output_dir / "review" / "accepted"

    # Gather proposal JSON files
    prop_files = sorted(proposals_dir.glob("*.json"))
    if not prop_files:
        raise FileNotFoundError(f"No proposal JSON files found in {proposals_dir}")

    logger.info("Found %d proposal files in %s", len(prop_files), proposals_dir)

    # Group images by domain to prevent domain leakage across splits
    domain_groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for pf in prop_files:
        # Check if corresponding PNG exists
        img_name = pf.stem + ".png"
        img_path = screenshots_dir / img_name
        if not img_path.exists():
            continue

        # Extract domain from filename (e.g. 001_booking_com.png -> booking_com)
        parts = pf.stem.split("_")
        domain = parts[1] if len(parts) > 1 else parts[0]
        domain_groups[domain].append((pf, img_path))

    domains = sorted(domain_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(domains)

    val_count = max(1, int(len(domains) * val_fraction))
    val_domains = set(domains[:val_count])
    train_domains = set(domains[val_count:])

    logger.info("Domain split: %d train domains, %d val domains", len(train_domains), len(val_domains))

    # Clean destination directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_labels = 0

    for domain, items in domain_groups.items():
        split = "val" if domain in val_domains else "train"
        img_dst_dir = output_dir / "images" / split
        lbl_dst_dir = output_dir / "labels" / split

        for pf, img_path in items:
            with open(pf, encoding="utf-8") as f:
                prop = json.load(f)

            # Check if human accepted label exists first
            accepted_txt = accepted_dir / (pf.stem + ".txt")
            if accepted_txt.exists():
                yolo_lines = accepted_txt.read_text(encoding="utf-8").strip().splitlines()
            else:
                # Convert candidates to YOLO format
                img_w = prop.get("width", 1440)
                img_h = prop.get("height", 900)
                yolo_lines = []
                for c in prop.get("candidates", []):
                    x1, y1, x2, y2 = c["bbox_pixels"]
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    bw = (x2 - x1) / img_w
                    bh = (y2 - y1) / img_h
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))
                    cls_id = CLASS_INDEX.get(c["label"], 0)
                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Copy image
            shutil.copy2(img_path, img_dst_dir / img_path.name)

            # Write label file
            label_file = lbl_dst_dir / (img_path.stem + ".txt")
            label_file.write_text("\n".join(yolo_lines), encoding="utf-8")

            total_images += 1
            total_labels += len(yolo_lines)

    # Write data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(VISUAL_LABELS),
        "names": VISUAL_LABELS,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info("=" * 60)
    logger.info("YOLO DATASET SPLIT COMPLETE")
    logger.info("  Images written    : %d", total_images)
    logger.info("  Bounding boxes    : %d", total_labels)
    logger.info("  data.yaml written : %s", yaml_path)
    logger.info("=" * 60)


def main() -> None:
    prepare_splits(
        proposals_dir=Path("data/cv/raw/florence_predictions"),
        screenshots_dir=Path("data/cv/raw/screenshots"),
        output_dir=Path("data/cv"),
    )


if __name__ == "__main__":
    main()
