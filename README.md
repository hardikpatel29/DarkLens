# DarkLens

DarkLens is an automated detection and reporting framework for dark patterns on websites. It analyzes web pages using a combination of DOM rule checking, NLP classification, and computer vision to flag manipulative design patterns with clear, verifiable evidence.

Every finding includes the exact DOM element, rule match, model prediction, or visual bounding box that triggered it.

---

## Current Status

The core pipeline, rule engine, fusion system, API, CLI, and reporting tools are fully built and tested. Machine learning models (NLP & CV) are integrated into the architecture with fallback support if trained weights are not present.

| Component | Stack / Method | Status |
|---|---|---|
| **Core Architecture** | Clean Architecture, Dependency Injection | ✅ Complete |
| **Page Capture** | Playwright (Headless Chromium) | ✅ Complete |
| **Rule Engine** | Deterministic DOM & CSS Inspection (4 rules) | ✅ Complete |
| **NLP Classifier** | DistilBERT | ✅ Complete (F1 Macro: 97.8%, Precision: 97.7%) |
| **CV Detector** | YOLOv8 + EasyOCR | ✅ Complete (mAP50: 40.4%, 9 visual pattern classes) |
| **Evidence Fusion** | Probabilistic Noisy-OR + Rule Floor | ✅ Complete |
| **Reporting & Export** | JSON, HTML, CSV, PDF (Playwright PDF rendering) | ✅ Complete |
| **CLI & API** | FastAPI, argparse, Docker | ✅ Complete |

If no trained model weights exist in `models/`, the system gracefully disables those detectors and runs using the rule engine without crashing:

```bash
python -c "from darklens.container import build_detectors; print([d.name for d in build_detectors()])"
# Output: ['rule_engine']
```

---

## Project Structure

The codebase strictly follows Clean Architecture principles:

```text
src/darklens/
├── domain/            # Core business logic, entities, & rule interfaces (zero external ML dependencies)
├── application/       # Orchestration, use cases, ports, & evidence fusion engine
├── infrastructure/    # Concrete implementations (Playwright, PyTorch, YOLO, EasyOCR, Reports)
├── interfaces/        # FastAPI web service & CLI entry points
└── container.py       # Composition root for dependency injection
```

Third-party dependencies (Playwright, PyTorch, Ultralytics, EasyOCR) are strictly isolated behind ports so domain and application layers remain clean and testable without heavy dependencies installed.

---

## Getting Started

### Prerequisites
* Python 3.10+
* Playwright & Chromium browser binaries

### Setup

```bash
# Clone the repository
git clone https://github.com/hardikpatel29/DarkLens.git
cd DarkLens

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Install Playwright browser
playwright install chromium
```

---

## Quick Start

### 1. Run via CLI
Scan a website and generate reports in JSON, HTML, CSV, or PDF formats:

```bash
python -m darklens.interfaces.cli.main https://example.com --formats json,html,csv,pdf
```

Reports will be saved to `data/reports/`.

### 2. Run API Server
Start the FastAPI server to access the REST endpoints and interactive Swagger UI:

```bash
uvicorn darklens.interfaces.api.main:app --reload
```

Endpoints available at `http://localhost:8000`:
* `POST /api/v1/scan` — Returns raw JSON `ScanResult`
* `POST /api/v1/scan/report` — Renders HTML report
* `POST /api/v1/scan/report.pdf` — Downloads PDF report

### 3. Run with Docker
Spin up the entire app inside a container:

```bash
docker compose up --build
```

---

## Testing & Verification

Run unit tests (no browser or heavy ML libraries required):

```bash
pytest tests/unit
```

Run integration tests (launches headless Chromium against local HTML fixtures):

```bash
pytest tests/integration
```

Run live performance benchmark (measures scan latency, CPU time, and RAM usage):

```bash
python scripts/benchmark.py https://example.com
```

---

## Model Training & Fine-Tuning

Scripts are provided in `scripts/` for dataset preparation and fine-tuning:

```bash
# Train NLP Classifier
python scripts/train_nlp_classifier.py \
    --train-csv data/nlp/train.csv \
    --val-csv data/nlp/val.csv \
    --output-dir models/nlp_classifier

# Train CV Detector
python scripts/train_cv_detector.py \
    --data data/cv/data.yaml \
    --output-dir models/cv_detector
```

---

## Dataset Citations

* **NLP Dataset:** Fine-tuning uses the dark pattern corpus from Mathur et al., *"Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites"* (2019) and Yamana Lab (`yamanalab/ec-darkpattern`, IEEE BigData 2022).
* **CV Screenshots:** Collected using custom Playwright scripts across e-commerce and travel sites for visual pattern annotation.
