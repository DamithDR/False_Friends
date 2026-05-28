"""
Sentence-level false friend classification using XLM-RoBERTa base.

For each language pair (EN-ES, EN-FR) the model takes an English sentence and
the target-language sentence as a sentence pair and predicts:
  - label 1: the target sentence contains a false friend
  - label 0: the target sentence uses the correct translation

Usage:
    python sentence_classification.py --lang es   # English-Spanish
    python sentence_classification.py --lang fr   # English-French
    python sentence_classification.py             # Both (sequential)
"""

import argparse
import os
import random
import numpy as np
import torch
import csv
from dataclasses import dataclass
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "xlm-roberta-base"
MAX_LENGTH   = 512          # sentence pairs can be long; 256 covers >95 % of data
BATCH_SIZE   = 2
EPOCHS       = 5
LR           = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED         = 42
OUTPUT_DIR   = "outputs/sentence_classification"

DATA = {
    "es": {
        "train": "data/sentence_classification/EN-ES_train.csv",
        "test":  "data/sentence_classification/EN-ES_test.csv",
        "tgt_col": "spanish_sentence",
        "lang_name": "English-Spanish",
    },
    "fr": {
        "train": "data/sentence_classification/EN-FR_train.csv",
        "test":  "data/sentence_classification/EN-FR_test.csv",
        "tgt_col": "french_sentence",
        "lang_name": "English-French",
    },
}

# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

@dataclass
class Example:
    text_a: str   # English sentence
    text_b: str   # Target-language sentence
    label:  int   # 0 or 1


def load_csv(path: str, tgt_col: str) -> List[Example]:
    examples = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(Example(
                text_a=row["english_sentence"].strip(),
                text_b=row[tgt_col].strip(),
                label=int(row["label"]),
            ))
    return examples


class FalseFriendDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, max_length: int):
        self.encodings = tokenizer(
            [e.text_a for e in examples],
            [e.text_b for e in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor([e.label for e in examples], dtype=torch.long)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return {
        "loss":      total_loss / len(loader),
        "accuracy":  accuracy_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "preds":     all_preds,
        "labels":    all_labels,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def run(lang: str):
    cfg = DATA[lang]
    print(f"\n{'='*65}")
    print(f"  Task : Sentence-level False Friend Classification")
    print(f"  Pair : {cfg['lang_name']}")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*65}\n")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Load data ────────────────────────────────────────────────────────────
    train_examples = load_csv(cfg["train"], cfg["tgt_col"])
    test_examples  = load_csv(cfg["test"],  cfg["tgt_col"])
    print(f"\nTrain examples : {len(train_examples)}"
          f"  (pos={sum(e.label for e in train_examples)}, "
          f"neg={sum(1-e.label for e in train_examples)})")
    print(f"Test  examples : {len(test_examples)}"
          f"  (pos={sum(e.label for e in test_examples)}, "
          f"neg={sum(1-e.label for e in test_examples)})")

    # ── Tokenizer & datasets ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = FalseFriendDataset(train_examples, tokenizer, MAX_LENGTH)
    test_ds   = FalseFriendDataset(test_examples,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    no_decay      = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if     any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(params, lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_f1, best_epoch = 0.0, 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_save_path = os.path.join(OUTPUT_DIR, f"best_model_{lang}")

    print(f"\n{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<12}"
          f"{'Accuracy':<12}{'F1 (macro)':<14}{'Precision':<12}{'Recall'}")
    print("-" * 82)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        metrics    = evaluate(model, test_loader, device)

        print(f"{epoch:<8}{train_loss:<14.4f}{metrics['loss']:<12.4f}"
              f"{metrics['accuracy']:<12.4f}{metrics['f1']:<14.4f}"
              f"{metrics['precision']:<12.4f}{metrics['recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_epoch = epoch
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

    # ── Final evaluation on best checkpoint ─────────────────────────────────
    print(f"\nBest epoch: {best_epoch}  |  Best macro-F1: {best_f1:.4f}")
    print(f"\nLoading best checkpoint for final evaluation...")

    best_model = AutoModelForSequenceClassification.from_pretrained(
        model_save_path
    ).to(device)
    final = evaluate(best_model, test_loader, device)

    print(f"\n{'─'*65}")
    print(f"  Final Test Results  [{cfg['lang_name']}]")
    print(f"{'─'*65}")
    print(f"  Accuracy  : {final['accuracy']:.4f}")
    print(f"  F1 (macro): {final['f1']:.4f}")
    print(f"  Precision : {final['precision']:.4f}")
    print(f"  Recall    : {final['recall']:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(
        final["labels"], final["preds"],
        target_names=["No False Friend (0)", "False Friend (1)"],
        digits=4
    ))
    print(f"  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(final["labels"], final["preds"])
    print(f"  {cm}")

    # ── Save results ─────────────────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, f"results_{lang}.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Language pair : {cfg['lang_name']}\n")
        f.write(f"Model         : {MODEL_NAME}\n")
        f.write(f"Best epoch    : {best_epoch} / {EPOCHS}\n")
        f.write(f"Best macro-F1 : {best_f1:.4f}\n\n")
        f.write(f"Accuracy  : {final['accuracy']:.4f}\n")
        f.write(f"F1 (macro): {final['f1']:.4f}\n")
        f.write(f"Precision : {final['precision']:.4f}\n")
        f.write(f"Recall    : {final['recall']:.4f}\n\n")
        f.write(classification_report(
            final["labels"], final["preds"],
            target_names=["No False Friend (0)", "False Friend (1)"],
            digits=4
        ))
        f.write(f"\nConfusion matrix (rows=true, cols=pred):\n{cm}\n")
    print(f"\n  Results saved → {results_path}")
    print(f"  Model  saved  → {model_save_path}/\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentence-level false friend classification with XLM-RoBERTa"
    )
    parser.add_argument(
        "--lang", choices=["es", "fr"], default=None,
        help="Language pair to train: 'es' (EN-ES) or 'fr' (EN-FR). "
             "Omit to run both sequentially."
    )
    args = parser.parse_args()

    langs = [args.lang] if args.lang else ["es", "fr"]
    for lang in langs:
        run(lang)
