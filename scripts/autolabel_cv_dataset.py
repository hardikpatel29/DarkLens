#!/usr/bin/env python
"""
Florence-2 annotation assistant for the DarkLens CV dataset.

This script uses Microsoft Florence-2 (open-weights, runs locally,
zero API cost) as a PROPOSAL GENERATOR ONLY.

CRITICAL: This script does NOT produce ground-truth YOLO labels.
It outputs candidate bounding boxes and class suggestions as JSON,
which are then fed into the human annotation review UI
(tools/annotation_ui/app.py) for verification, correction, and
final approval.

The pipeline is:

    Screenshot
        ↓
    Florence-2 (this script)
        ↓
    data/cv/raw/florence_predictions/<image_id>.json
        ↓
    Human review via annotation UI
        ↓
    data/cv/review/accepted/<image_id>.txt (YOLO format)
        ↓
    train_cv_detector.py

Never skip human review and use Florence output directly as training
labels. Florence is a proposal generator, not an oracle.

Florence-2 model size
---------------------
Default: microsoft/Florence-2-base (~900 MB, first run downloads and
caches). Set --model to microsoft/Florence-2-large for better recall on
small UI elements (requires 6+ GB VRAM or significantly more CPU RAM).

Usage
-----
    # Label screenshots from the new crawler:
    python scripts/autolabel_cv_dataset.py

    # Explicit paths / GPU inference:
    python scripts/autolabel_cv_dataset.py \\
        --input-dir  data/cv/raw/screenshots \\
        --output-dir data/cv/raw/florence_predictions \\
        --confidence 0.20 \\
        --device     cuda \\
        --model      microsoft/Florence-2-large

Requirements
------------
    transformers>=4.45  torch>=2.0  pillow  pyyaml  tqdm
    (already installed by `pip install -e ".[nlp]"` except pillow/pyyaml
     which are pulled in by `pip install -e ".[cv]"`)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("autolabel_cv_dataset")

# ── Class taxonomy ─────────────────────────────────────────────────────────────
# Must match VisualPatternLabel in domain/cv/labels.py — value strings in order.
# The ORDER here determines the class index written to every .txt label file.
VISUAL_LABELS: list[str] = [
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

# Florence-2 text prompts per class.
# Written as short, visually descriptive noun phrases — Florence-2's
# open-vocabulary detector responds well to concrete visual descriptions
# rather than abstract category names.
LABEL_PROMPTS: dict[str, str] = {
    "fake_purchase_notification": (
        "toast notification showing someone purchased or is viewing a product"
    ),
    "countdown_timer": (
        "countdown timer with digits showing hours minutes seconds"
    ),
    "discount_badge": (
        "discount percentage badge or sale label on product"
    ),
    "floating_overlay": (
        "floating modal dialog or pop-up overlay covering the page"
    ),
    "subscription_popup": (
        "email subscription sign-up pop-up or newsletter form"
    ),
    "cookie_popup": (
        "cookie consent banner or GDPR privacy notice bar"
    ),
    "tiny_close_button": (
        "small close button X icon on a pop-up or banner"
    ),
    "urgency_banner": (
        "urgency message banner with limited time offer or hurry text"
    ),
    "scarcity_label": (
        "scarcity label showing only a few items left in stock"
    ),
}

CLASS_INDEX: dict[str, int] = {label: i for i, label in enumerate(VISUAL_LABELS)}

# Minimum pixel area (width × height) for a detection to be kept.
# Removes single-pixel noise and sub-pixel artefacts.
MIN_BOX_AREA = 100   # pixels²


# ── Model loading ──────────────────────────────────────────────────────────────

def load_florence2(model_name: str, device: str):
    """Downloads and returns (model, processor) for Florence-2."""
    try:
        import transformers.dynamic_module_utils
        orig_check = transformers.dynamic_module_utils.check_imports
        def safe_check(filename, *args, **kwargs):
            try:
                return orig_check(filename, *args, **kwargs)
            except ImportError as e:
                if "flash_attn" in str(e):
                    return []
                raise
        transformers.dynamic_module_utils.check_imports = safe_check
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Run: pip install 'transformers>=4.45' torch pillow"
        ) from exc

    logger.info("Loading %s (first run downloads model, cached after that)…", model_name)

    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    logger.info("Florence-2 loaded on %s.", device)
    return model, processor


# ── Detection ──────────────────────────────────────────────────────────────────

def detect_label(
    image,
    label: str,
    model,
    processor,
    device: str,
    confidence_threshold: float,
) -> list[dict]:
    """
    Runs Florence-2 OPEN_VOCABULARY_DETECTION for one label prompt.
    Returns list of candidate proposals: {box, score, label, prompt}.
    These are CANDIDATES, not ground truth.
    """
    import torch

    prompt_text = LABEL_PROMPTS[label]
    task_token = "<OPEN_VOCABULARY_DETECTION>"
    full_prompt = f"{task_token}{prompt_text}"

    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_token,
        image_size=(image.width, image.height),
    )

    result = parsed.get(task_token, {})
    bboxes = result.get("bboxes", [])
    phrase_scores = result.get("scores", None)

    candidates = []
    for idx, box in enumerate(bboxes):
        score = 1.0
        if phrase_scores is not None:
            score = float(phrase_scores[idx]) if idx < len(phrase_scores) else 1.0
        if score < confidence_threshold:
            continue

        x1, y1, x2, y2 = (int(v) for v in box)
        # Clamp to image bounds
        x1 = max(0, min(x1, image.width - 1))
        y1 = max(0, min(y1, image.height - 1))
        x2 = max(x1 + 1, min(x2, image.width))
        y2 = max(y1 + 1, min(y2, image.height))

        # Filter sub-pixel boxes
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
            continue

        candidates.append({
            "label": label,
            "class_id": CLASS_INDEX[label],
            "prompt": prompt_text,
            "bbox_pixels": [x1, y1, x2, y2],
            "score": round(score, 4),
            "annotation_source": "florence",
            "verified": False,       # set to True by annotation UI after human review
            "human_label": None,     # filled by annotation UI
            "notes": "",
        })

    return candidates


# ── Per-image processing ───────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    model,
    processor,
    device: str,
    confidence_threshold: float,
) -> dict:
    """
    Returns a proposal dict for one screenshot:
    {
        image_id: str,
        image_path: str,
        width: int,
        height: int,
        candidates: [ { label, class_id, bbox_pixels, score, ... }, ... ]
    }
    """
    from PIL import Image

    image = Image.open(img_path).convert("RGB")
    all_candidates = []

    for label in VISUAL_LABELS:
        candidates = detect_label(image, label, model, processor, device, confidence_threshold)
        all_candidates.extend(candidates)

    return {
        "image_id": img_path.stem,
        "image_path": str(img_path),
        "width": image.width,
        "height": image.height,
        "florence_model": None,      # filled in by build_proposals
        "candidates": all_candidates,
        "review_status": "pending",  # pending | accepted | rejected | uncertain
        "is_negative": False,        # set to True by annotation UI for confirmed no-pattern images
        "annotator": None,
        "annotation_timestamp": None,
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_proposals(
    input_dir: Path,
    output_dir: Path,
    confidence_threshold: float,
    device: str,
    model_name: str,
    skip_tiny: int,
) -> None:
    """
    Processes every PNG in input_dir, writes one JSON proposal file per
    image to output_dir. Does NOT write any YOLO .txt files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(input_dir.glob("*.png"))
    usable = [p for p in all_images if p.stat().st_size >= skip_tiny]
    skipped = len(all_images) - len(usable)
    if skipped:
        logger.info("Skipped %d tiny/blank images (< %d bytes).", skipped, skip_tiny)

    # Skip images that already have a proposal file (resume support)
    pending = [p for p in usable if not (output_dir / f"{p.stem}.json").exists()]
    logger.info(
        "Processing %d images with %s (skipping %d already done)…",
        len(pending), model_name, len(usable) - len(pending),
    )

    if not pending:
        logger.info("Nothing to do — all images already have proposal files.")
        return

    model, processor = load_florence2(model_name, device)

    total_candidates = 0
    for i, img_path in enumerate(pending, 1):
        logger.info("[%d/%d] %s", i, len(pending), img_path.name)
        try:
            proposal = process_image(img_path, model, processor, device, confidence_threshold)
            proposal["florence_model"] = model_name
        except Exception as exc:
            logger.warning("  Failed to process %s: %s", img_path.name, exc)
            continue

        # Save JSON proposal — NOT a YOLO label file
        out_path = output_dir / f"{img_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2)

        n = len(proposal["candidates"])
        total_candidates += n
        if n:
            by_class = {}
            for c in proposal["candidates"]:
                by_class[c["label"]] = by_class.get(c["label"], 0) + 1
            logger.info("  → %d candidate(s): %s", n, by_class)
        else:
            logger.info("  → no candidates (likely a negative/clean page)")

    logger.info("=" * 64)
    logger.info("FLORENCE PROPOSAL GENERATION COMPLETE")
    logger.info("  Proposal files     : %s", output_dir)
    logger.info("  Total candidates   : %d", total_candidates)
    logger.info("  IMPORTANT: These are PROPOSALS, not ground truth.")
    logger.info("  Next step          : python tools/annotation_ui/app.py")
    logger.info("                       then review and verify each image.")
    logger.info("=" * 64)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/cv/raw/screenshots"),
        help="Directory of raw PNG screenshots (default: data/cv/raw/screenshots)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/cv/raw/florence_predictions"),
        help="Directory for JSON proposal files (default: data/cv/raw/florence_predictions)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.20,
        help="Minimum detection confidence from Florence-2 (default: 0.20)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Compute device for Florence-2 (default: cpu; use cuda if NVIDIA GPU available)",
    )
    parser.add_argument(
        "--model", default="microsoft/Florence-2-base",
        help=(
            "Florence-2 checkpoint (default: microsoft/Florence-2-base ~900 MB). "
            "Use microsoft/Florence-2-large for better recall if 6+ GB VRAM available."
        ),
    )
    parser.add_argument(
        "--skip-tiny", type=int, default=8_000,
        help="Skip images smaller than this many bytes (default: 8000)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input dir not found: {args.input_dir}. "
            "Run scripts/crawl_cv_dataset.py first."
        )

    build_proposals(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        device=args.device,
        model_name=args.model,
        skip_tiny=args.skip_tiny,
    )


if __name__ == "__main__":
    main()
