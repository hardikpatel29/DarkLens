#!/usr/bin/env python
"""
Fine-tunes a DistilBERT-base-uncased sequence classifier on manipulative
UX-copy text, producing a checkpoint that `TransformerTextClassifier`
(infrastructure/nlp/transformer_classifier.py) can load.

This script is real and runnable end-to-end against a correctly
formatted CSV — it is not pseudocode. What it does NOT do, on purpose,
is invent or download the labeled dataset for you, or claim a trained
model exists when it doesn't. That's [USER TASK] work below, not
something this script can safely fake.

Expected input format: a CSV with columns `text,label`, where `label`
is one of the values in `NlpLabel` (see domain/nlp/labels.py) — i.e.
"normal", "false_urgency", "confirmshaming", etc. If your source dataset
uses different label names (the Yamana Lab release does), map them to
this taxonomy during the [USER TASK] step below — see the mapping
rationale already documented in
application/services/pattern_catalog.py, which this script's labels
must stay consistent with.

Usage:
    python scripts/train_nlp_classifier.py \\
        --train-csv data/nlp/train.csv \\
        --val-csv data/nlp/val.csv \\
        --output-dir models/nlp_classifier \\
        --base-model distilbert-base-uncased \\
        --epochs 4

Requires the `nlp` extra: pip install -e ".[nlp]"
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("train_nlp_classifier")

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
# (The block above is the project's standard template and includes
# steps that belong to the CV module, not this one — kept verbatim
# per the project convention rather than silently edited. The NLP-
# specific version of the same checklist:)
#
# Collect text dataset
#   -> Start from Mathur et al. 2019 / Yamana Lab `ec-darkpattern`
#      (cited in README.md's "Dataset citation" section). Do not
#      invent a dataset for this project.
# Label samples
#   -> Map source labels onto NlpLabel (domain/nlp/labels.py) using
#      the rationale already written in
#      application/services/pattern_catalog.py. Make sure NORMAL is
#      well-represented (see labels.py docstring on why) — a raw
#      dark-pattern crawl is mostly NORMAL text; sample negatives from
#      the same pages, don't skip them.
# Fine-tune transformer
#   -> Run this script with --train-csv/--val-csv pointing at your
#      labeled split.
# Evaluate Precision / Recall / F1 / Confusion Matrix
#   -> This script writes these to <output_dir>/eval_metrics.json and
#      <output_dir>/confusion_matrix.json automatically (see
#      `evaluate_and_save` below) — run it and read the numbers,
#      don't assume them.
# Threshold tuning
#   -> `nlp_confidence_threshold` in config.py defaults to 0.6. Once
#      you have eval_metrics.json, sweep threshold vs. precision/recall
#      on the validation set and update the default with a comment
#      citing the number you picked and why.
# Export model
#   -> Handled by `trainer.save_model()` / `tokenizer.save_pretrained()`
#      below — produces the directory `TransformerTextClassifier` loads.
# Document experiments
#   -> Fill in docs/nlp_experiments.md (base model, hyperparameters,
#      dataset size/split, final metrics, threshold chosen) so the
#      numbers in a report/interview are traceable to a real run.
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True, help="CSV with text,label columns")
    parser.add_argument("--val-csv", type=Path, required=True, help="CSV with text,label columns")
    parser.add_argument("--output-dir", type=Path, default=Path("models/nlp_classifier"))
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-sequence-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    # Imported here (not at module top level) so `--help` works without
    # the `nlp` extra installed.
    from darklens.domain.nlp.labels import ALL_LABELS

    label2id = {label.value: i for i, label in enumerate(ALL_LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def load_dataset(train_csv: Path, val_csv: Path, label2id: dict[str, int]):
    import pandas as pd
    from datasets import Dataset

    def _load(path: Path) -> Dataset:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. This script does not synthesize training "
                f"data — see the [USER TASK] block above for how to obtain it."
            )
        df = pd.read_csv(path)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {missing}")
        unknown_labels = set(df["label"]) - set(label2id)
        if unknown_labels:
            raise ValueError(
                f"{path} contains label(s) not in the NlpLabel taxonomy: {unknown_labels}. "
                f"Map source labels onto domain/nlp/labels.py before training."
            )
        df["label_id"] = df["label"].map(label2id)
        return Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "label"}))

    return _load(train_csv), _load(val_csv)


def compute_metrics(eval_pred) -> dict:
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        # Full confusion matrix is saved separately post-training (see
        # evaluate_and_save) — Trainer's compute_metrics return value
        # must be flat scalars for logging, not a matrix.
        "_confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def evaluate_and_save(trainer, val_dataset, id2label: dict[int, str], output_dir: Path) -> None:
    metrics = trainer.evaluate(val_dataset)
    confusion = metrics.pop("eval__confusion_matrix", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if confusion is not None:
        with open(output_dir / "confusion_matrix.json", "w") as f:
            labels_in_order = [id2label[i] for i in range(len(id2label))]
            json.dump({"labels": labels_in_order, "matrix": confusion}, f, indent=2)

    logger.info("Eval metrics: %s", metrics)
    logger.info("Wrote %s/eval_metrics.json and confusion_matrix.json", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()

    # Deliberately imported inside main(), not at module top level, so
    # `python scripts/train_nlp_classifier.py --help` doesn't require
    # torch/transformers/datasets to be installed just to print usage.
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    label2id, id2label = load_label_maps()
    train_dataset, val_dataset = load_dataset(args.train_csv, args.val_csv, label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=args.max_sequence_length
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=args.seed,
        report_to=[],  # no wandb/tensorboard dependency forced on the user
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    evaluate_and_save(trainer, val_dataset, id2label, args.output_dir)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("Model exported to %s — TransformerTextClassifier can now load it.", args.output_dir)


if __name__ == "__main__":
    main()
