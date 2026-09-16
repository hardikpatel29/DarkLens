# CV Detector — Experiment Log

Fill this in after running `scripts/train_cv_detector.py` against a real
annotated dataset. Same rule as `docs/nlp_experiments.md`: an empty
section here is more honest than invented mAP numbers.

## Run 1

- **Date:**
- **Base model:** (default: `yolov8n.pt`)
- **Dataset:** source/collection method, image count, per-class instance
  count (call out any severely underrepresented class — e.g. if
  tiny_close_button has 40 instances vs. countdown_timer's 800, that's
  worth flagging before trusting its precision/recall)
- **Hyperparameters:** epochs, image size, batch size
- **Results:**
  - mAP50:
  - mAP50-95:
  - Per-class AP (from Ultralytics' own `val/` output):
- **Confidence threshold chosen:** and the precision/recall tradeoff
  behind it — update `cv_confidence_threshold` in `config.py` to match
- **Error analysis:** a few concrete false positives/negatives with
  images, and what they suggest (lighting, false detection on unrelated
  UI, class confusion between visually similar patterns)
- **Exported to:** `models/cv_detector/best.pt` (not committed — see `.gitignore`)

## OCR benchmark (see infrastructure/ocr/easyocr_engine.py's [USER TASK])

- **Engines compared:**
- **Accuracy (char/word-level) per engine:**
- **Latency per crop, per engine:**
- **Chosen engine and why:**
