#!/usr/bin/env python
"""
Fine-tunes a YOLO detector on UI dark-pattern visual elements, producing
a checkpoint that `YoloCvModel` (infrastructure/cv/yolo_cv_model.py)
can load.

This script is real and runnable end-to-end against a correctly
formatted YOLO dataset — it is not pseudocode. What it does NOT do is
invent the annotated dataset for you. Object detection training data
(bounding boxes on real screenshots of countdown timers, fake purchase
popups, tiny close buttons, etc.) does not exist yet and cannot be
synthesized responsibly; see the [USER TASK] block below.

Expected input: a standard Ultralytics YOLO dataset directory —
    <data_root>/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt           # YOLO-format: class x_center y_center w h (normalized)
      data.yaml                          # names: VisualPatternLabel.value strings, in class-index order

`data.yaml`'s `names` list MUST exactly match the value strings in
`domain.cv.labels.VisualPatternLabel`, in the same order used when you
annotate — see the [USER TASK] block for why this is a step you do
once, carefully, rather than something this script can infer.

Usage:
    python scripts/train_cv_detector.py \\
        --data data/cv/data.yaml \\
        --output-dir models/cv_detector \\
        --base-model yolov8n.pt \\
        --epochs 100

Requires the `cv` extra: pip install -e ".[cv]"
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("train_cv_detector")

# ============================================================
# [USER TASK]
#
# Collect dataset
# Choose public datasets
# Scrape additional webpages if needed
# Define object classes
# Annotate images
# Split dataset
# Fine-tune YOLO
# Evaluate mAP
# Run error analysis
# Export optimized model
# Document experiments
# Replace placeholders
# ------------------------------------------------------------
# Concretely, for this module:
#
# Collect dataset / choose public datasets
#   -> No public dataset is annotated specifically for dark-pattern UI
#      elements (countdown timers, fake purchase toasts, tiny close
#      buttons) — unlike Phase 2's text data, this one likely needs to
#      be assembled largely from scratch. Reasonable seed sources: your
#      own scan screenshots from Phase 1's Playwright capture pipeline
#      (screenshot_dir in config.py already produces full-page PNGs),
#      plus manually sourced/scraped e-commerce screenshots. Respect
#      robots.txt / ToS of any site you scrape from.
# Define object classes
#   -> Already fixed: domain/cv/labels.py's VisualPatternLabel — 9
#      classes. Don't add a 10th class here without adding it there
#      first (and adding its pattern_catalog.py entry).
# Annotate images
#   -> Use a bounding-box annotation tool (e.g. CVAT, Label Studio,
#      Roboflow) exporting to YOLO format. Budget real time for this —
#      it's the actual bottleneck of this phase, not the training run.
# Split dataset
#   -> Stratify by class where possible; countdown timers and cookie
#      popups will be far more common in a raw crawl than e.g. tiny
#      close buttons — an unstratified split can leave a class with
#      zero validation examples.
# Fine-tune YOLO
#   -> Run this script with --data pointing at your data.yaml.
# Evaluate mAP
#   -> This script writes Ultralytics' own validation output (mAP50,
#      mAP50-95, per-class breakdown) to <output_dir>/val_results —
#      read it, don't estimate it.
# Run error analysis
#   -> Inspect false positives/negatives per class with
#      `model.val(...)`'s saved prediction images, or
#      `yolo predict save=True` on held-out images. Document patterns
#      (e.g. "confuses discount_badge with urgency_banner when both are
#      red") in docs/cv_experiments.md.
# Export optimized model
#   -> Handled by `model.export()` / the best.pt copy below. Consider
#      `format="onnx"` for faster CPU inference once accuracy is
#      acceptable — that's a follow-up optimization, not this step.
# Document experiments
#   -> Fill in docs/cv_experiments.md (base model, dataset size/split,
#      hyperparameters, final mAP, per-class weak points, exported
#      format) so report numbers are traceable to a real run.
# Replace placeholders
#   -> `cv_confidence_threshold` in config.py (default 0.5) is
#      unvalidated until you've run a precision/recall sweep against
#      real validation predictions — update it with a comment citing
#      the number you picked and why, same as nlp_confidence_threshold.
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("models/cv_detector"))
    parser.add_argument(
        "--base-model",
        default="yolov8n.pt",
        help="Ultralytics base checkpoint to fine-tune from (nano by default: "
        "smallest/fastest, appropriate starting point before a real dataset "
        "size justifies a larger variant — see docs/cv_experiments.md once run).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_class_names(data_yaml: Path) -> None:
    """
    Fail fast, before spending a training run, if data.yaml's class
    names don't match the domain taxonomy — this is the single most
    likely annotation-pipeline mistake (wrong order, typo, extra class)
    and it's much cheaper to catch here than after 100 epochs.
    """
    import yaml

    from darklens.domain.cv.labels import ALL_VISUAL_LABELS

    with open(data_yaml) as f:
        spec = yaml.safe_load(f)

    names = spec.get("names")
    if isinstance(names, dict):  # ultralytics allows {0: "name", ...} too
        names = [names[i] for i in range(len(names))]

    expected = [label.value for label in ALL_VISUAL_LABELS]
    if names != expected:
        raise ValueError(
            f"{data_yaml} names {names!r} do not match "
            f"domain.cv.labels.VisualPatternLabel {expected!r} (order matters). "
            f"Fix data.yaml before training."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    # Imported inside main(), not at module top level, so `--help` works
    # without the `cv` extra installed.
    from ultralytics import YOLO

    validate_class_names(args.data)

    model = YOLO(args.base_model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        seed=args.seed,
        project=str(args.output_dir),
        name="run",
    )
    logger.info("Training complete: %s", results.save_dir)

    metrics = model.val(data=str(args.data))
    logger.info(
        "Validation mAP50=%.4f mAP50-95=%.4f — full breakdown in %s",
        metrics.box.map50,
        metrics.box.map,
        args.output_dir / "run" / "val",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy(best_weights, args.output_dir / "best.pt")
    logger.info(
        "Exported best checkpoint to %s — YoloCvModel can now load it.",
        args.output_dir / "best.pt",
    )


if __name__ == "__main__":
    main()
