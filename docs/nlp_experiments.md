# NLP Classifier — Experiment Log

Fill this in after running `scripts/train_nlp_classifier.py` against a real
labeled dataset. Don't publish placeholder numbers — an empty section here
is more honest than an invented one, and more useful to a reviewer than a
generic "achieved high accuracy" claim.

## Run 1

- **Date:**
- **Base model:** (default: `distilbert-base-uncased`)
- **Dataset:** source, size (train/val/test row counts), label distribution
  (especially the NORMAL-vs-dark-pattern ratio — see `domain/nlp/labels.py`
  on why an imbalanced or NORMAL-free dataset produces a classifier that
  can't tell ordinary copy from manipulative copy)
- **Hyperparameters:** epochs, batch size, learning rate, max sequence length
- **Results** (from `<output_dir>/eval_metrics.json`):
  - Precision (macro):
  - Recall (macro):
  - F1 (macro):
  - Per-class breakdown (from `confusion_matrix.json`) — call out which
    labels the model confuses most, and why that's plausible given the
    label definitions (e.g. Fear-based Copy vs. Emotional Manipulation are
    close enough that some confusion is expected, not necessarily a bug)
- **Confidence threshold chosen:** and the precision/recall tradeoff at
  that threshold vs. neighboring thresholds (0.5, 0.6, 0.7, 0.8) — update
  `nlp_confidence_threshold` in `config.py` to match, with a comment
  linking back to this entry
- **Error analysis:** a few concrete false positives / false negatives,
  and what they suggest about the dataset or label boundaries (not just
  the aggregate metric)
- **Exported to:** `models/nlp_classifier` (not committed — see `.gitignore`)

## Baseline comparison

Per the project's benchmarking philosophy: report the transformer's
numbers against a simple baseline (e.g. TF-IDF + linear SVM) trained on
the same split, not just in isolation. If the transformer doesn't clearly
beat a much cheaper baseline on this dataset size, that's worth reporting
honestly rather than omitting.

- **Baseline model:**
- **Baseline F1 (macro):**
- **Transformer F1 (macro):**
- **Delta and interpretation:**
