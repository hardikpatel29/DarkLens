# DarkLens CV Dataset Construction & Annotation Specification

## 1. Objective & Scope

DarkLens uses a hybrid multi-modal architecture combining:
1. **DOM & CSS Rule Engine** (deterministic structural inspection)
2. **DistilBERT NLP Classifier** (97.8% F1 sequence classification on manipulative copy)
3. **YOLOv8 CV Detector + EasyOCR** (visual element localization and text extraction)
4. **Probabilistic Noisy-OR Evidence Fusion**

The CV module exists specifically to locate **visual evidence** for patterns that leave no clean DOM signature (e.g. countdown timers rendered on `<canvas>`, fake purchase toasts built from generic `div` soups, tiny close buttons made deliberately non-standard).

---

## 2. Core Taxonomy & Design Principles

### 1 Screenshot ≠ 1 Example
A single screenshot may yield 0, 1, or N bounding boxes across any of the 9 visual dark-pattern classes. We track both **total screenshots** and **per-class instance counts**.

### Taxonomy (9 Visual Classes)

| Class Index | Label | Visual Pattern Description | OCR Enabled |
|---|---|---|---|
| 0 | `fake_purchase_notification` | Toast/popup claiming someone recently bought/viewed a product | Yes |
| 1 | `countdown_timer` | Timer widget with active digits counting down | Yes |
| 2 | `discount_badge` | Prominent discount percentage / sale callout badge | Yes |
| 3 | `floating_overlay` | Modal dialog or backdrop overlay obscuring page content | No |
| 4 | `subscription_popup` | Email sign-up or newsletter pop-up overlay | Yes |
| 5 | `cookie_popup` | GDPR/cookie consent banner or modal bar | Yes |
| 6 | `tiny_close_button` | Sub-standard or visually obscured close `X` icon | No |
| 7 | `urgency_banner` | "Hurry!", "Offer ends today" urgency message bar | Yes |
| 8 | `scarcity_label` | "Only 2 left in stock!" scarcity indicator | Yes |

> **Note**: OCR is enabled for text-bearing labels to feed extracted text into the NLP classifier for multi-modal validation. Purely geometric labels (`floating_overlay`, `tiny_close_button`) do not run OCR.

---

## 3. Dataset Collection & Crawling Methodology

- **Crawler**: `scripts/crawl_cv_dataset.py`
- **Target Budget**: ~70 domains × ~10 UI states = ~700 high-resolution screenshots (1440×900).
- **Target Categories**: Indian e-commerce, international e-commerce, travel/booking, food delivery, SaaS, streaming, ticketing, finance.
- **UI States Captured**:
  1. `homepage`
  2. `search_results`
  3. `product_listing`
  4. `product_detail`
  5. `cart`
  6. `checkout` (safe interaction: stops BEFORE payment submit)
  7. `login_signup`
  8. `subscription_offer`
  9. `cookie_consent`
  10. `popup_wait`
- **Ethical & Safety Controls**:
  - Checks and respects `robots.txt` per domain.
  - Low concurrency (3 domains) to avoid overloading servers.
  - No real payment or financial submission.
  - Perceptual hash (`pHash`) deduplication to reject identical/near-identical captures.

---

## 4. Annotation Workflow & Florence-2 Proposal Model

```text
               PUBLIC WEBSITES
                      ↓
          scripts/crawl_cv_dataset.py
                      ↓
           Raw PNG Screenshots (data/cv/raw/screenshots/)
                      ↓
         scripts/autolabel_cv_dataset.py (Florence-2)
                      ↓
       JSON Proposals (data/cv/raw/florence_predictions/)
                      ↓
     Human Review UI (tools/annotation_ui/app.py)
                      ↓
   Verified Ground Truth YOLO Labels (data/cv/review/accepted/)
                      ↓
         scripts/validate_cv_dataset.py
                      ↓
     Domain-Based Split (data/cv/images/ & data/cv/labels/)
                      ↓
       scripts/train_cv_detector.py (YOLOv8)
```

> **CRITICAL RULE**: Florence-2 is used purely as an **annotation assistant / candidate proposal generator**. No Florence output is ever fed directly into YOLO training without human review.

---

## 5. Bounding-Box Annotation Rules

Annotators must evaluate every screenshot using 3 core questions:
1. **What is the user being influenced to do?**
2. **What visual element communicates that influence?**
3. **Can I point to that element with a bounding box?**

If the answer to Question 3 is **No** (e.g. multi-step cancellation flow, hidden fee revealed only at checkout), the item is handled by interaction/DOM analysis, **not YOLO**.

- **Positive Examples**: Box tightly around the specific visual element (`"Only 2 left in stock"`), NOT the whole page.
- **Negative Examples**: Normal product pages, search bars, standard UI without dark patterns $\rightarrow$ mark as `Negative` (produces empty label file).
- **Uncertain Examples**: Ambiguous claims $\rightarrow$ mark as `Uncertain` (saved to `data/cv/review/uncertain/`, excluded from training).

---

## 6. Domain-Based Dataset Splitting

To ensure real-world generalization test:
- Screenshots are split strictly **by domain** (e.g., all Amazon screenshots in `train/`, all Booking.com in `val/`).
- **No domain appears in more than one split**.
- Prevents data leakage and ensures mAP reflects performance on unseen websites.

---

## 7. Quality Control & Automated Validation

Run before training:
```bash
python scripts/validate_cv_dataset.py
```
Checks performed:
1. `data.yaml` class name order matches `VisualPatternLabel` enum.
2. Every image has a corresponding label file.
3. YOLO coordinates normalized in $[0, 1]$, no NaN/Inf, no zero-area boxes.
4. Valid class IDs ($0 \le \text{cls} < 9$).
5. Zero domain leakage across splits.
6. Flags near-duplicates (pHash Hamming distance $\le 4$).
7. Warns on class imbalance ($< 50$ instances per class).

---

## 8. Training & Evaluation Pipeline

Once validation passes:
```bash
python scripts/train_cv_detector.py \
    --data data/cv/data.yaml \
    --output-dir models/cv_detector \
    --epochs 100
```
Outputs `models/cv_detector/best.pt`, which is loaded automatically by `darklens.container.build_detectors()`.
