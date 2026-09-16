FROM python:3.12-slim

WORKDIR /app

# System deps Playwright's Chromium needs to actually launch (fonts, libs).
# Using `playwright install-deps` instead of hand-rolling an apt list keeps
# this in sync with whatever Playwright version pyproject.toml pins.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY models/ ./models/

RUN pip install --no-cache-dir -e ".[nlp,cv]" \
    && playwright install --with-deps chromium

ENV DARKLENS_HEADLESS_BROWSER=true \
    DARKLENS_SCREENSHOT_DIR=/app/data/screenshots \
    DARKLENS_NLP_MODEL_PATH=/app/models/nlp_classifier \
    DARKLENS_CV_MODEL_PATH=/app/models/cv_detector/best.pt \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "darklens.interfaces.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
