"""
Token Classification for False Friend Detection using HuggingFace Transformers.

Source and target sentences are concatenated into a single sequence:
  [CLS] source_tokens [SEP] target_tokens [SEP]

Data can come from EITHER:
  * a local JSONL dir produced by create_token_classification_dataset.py,
    selected with --data_dir + --lang  (recommended for this project), OR
  * a HuggingFace Hub dataset id via --dataset_name.

Notebook usage:

    from token_classification import train_model, predict, evaluate_model

    # Train on local data
    trainer, tokenizer = train_model(
        model_name="xlm-roberta-base",
        data_dir="data/token_classification",
        lang="es",
        output_dir="./ff_xlmr_es",
        epochs=10, batch_size=16, lr=5e-5,
    )

    # Predict
    results = predict(
        model_path="./ff_xlmr_es",
        source="This is a sensible solution",
        target="Esta es una solución sensible",
    )

    # Evaluate
    evaluate_model(
        model_path="./ff_xlmr_es",
        data_dir="data/token_classification",
        lang="es",
    )

CLI usage:

    python token_classification.py train \\
        --model_name xlm-roberta-base \\
        --data_dir data/token_classification --lang es \\
        --output_dir ./ff_xlmr_es

    python token_classification.py predict --model_path ./ff_xlmr_es \\
        --source "..." --target "..."

    python token_classification.py evaluate --model_path ./ff_xlmr_es \\
        --data_dir data/token_classification --lang es
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_DATASET = "false-friends/en_es_token"
LABEL_LIST = ["O", "B-FF"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — concatenated source + target
# ──────────────────────────────────────────────────────────────────────────────
class FalseFriendPairDataset(TorchDataset):
    """Each example concatenates source and target words into one sequence:
        [CLS] src_w1 src_w2 ... [SEP] tgt_w1 tgt_w2 ... [SEP]
    Labels are aligned for both halves; special tokens get -100.
    """

    def __init__(self, src_words, src_labels, tgt_words, tgt_labels, tokenizer, max_length=512):
        assert len(src_words) == len(tgt_words)
        self.src_words = src_words
        self.src_labels = src_labels
        self.tgt_words = tgt_words
        self.tgt_labels = tgt_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_words)

    def __getitem__(self, idx):
        s_words  = self.src_words[idx]
        s_labels = self.src_labels[idx]
        t_words  = self.tgt_words[idx]
        t_labels = self.tgt_labels[idx]

        encoding = self.tokenizer(
            s_words,
            t_words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        # NOTE: when tokenising a sentence pair with is_split_into_words=True,
        # word_ids() restarts at 0 for the target sequence. We therefore use
        # sequence_ids() to know which side a token belongs to and look up the
        # word index in the matching label list.
        word_ids     = encoding.word_ids()
        sequence_ids = encoding.sequence_ids()

        label_ids = []
        prev_key  = (None, None)  # (seq_id, word_id) of the previously-emitted token
        for word_id, seq_id in zip(word_ids, sequence_ids):
            if word_id is None or seq_id is None:
                label_ids.append(-100)
                prev_key = (None, None)
                continue

            labels_for_seq = s_labels if seq_id == 0 else t_labels
            key = (seq_id, word_id)
            if key != prev_key and word_id < len(labels_for_seq):
                lab = labels_for_seq[word_id]
                label_ids.append(lab if isinstance(lab, int) else LABEL2ID[lab])
            else:
                label_ids.append(-100)
            prev_key = key

        encoding["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in encoding.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_hf_split(dataset_name, split, token=None):
    """Load a split from HuggingFace Hub and extract parallel lists.
    Returns: src_words, src_labels, tgt_words, tgt_labels"""
    ds = load_dataset(dataset_name, split=split, token=token)
    src_words = [row["source_words"] for row in ds]
    src_labels = [row["source_labels"] for row in ds]
    tgt_words = [row["target_words"] for row in ds]
    tgt_labels = [row["target_labels"] for row in ds]
    return src_words, src_labels, tgt_words, tgt_labels


def load_local_split(jsonl_path):
    """Load a JSONL file produced by create_token_classification_dataset.py.
    Each line must have keys: source_words, source_labels, target_words,
    target_labels (labels as strings 'O'/'B-FF' or ints)."""
    src_words, src_labels, tgt_words, tgt_labels = [], [], [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src_words.append(r["source_words"])
            src_labels.append(r["source_labels"])
            tgt_words.append(r["target_words"])
            tgt_labels.append(r["target_labels"])
    return src_words, src_labels, tgt_words, tgt_labels


def resolve_splits(dataset_name=None, data_dir=None, lang=None, hf_token=None):
    """Pick HF-Hub or local-JSONL loader and return (train, test) tuples.

    Local mode is used when both ``data_dir`` and ``lang`` are given; files are
    looked up at ``{data_dir}/EN-{LANG}_{train|test}.jsonl`` (lang upper-cased).
    Otherwise falls back to the HuggingFace Hub via ``dataset_name``.
    """
    if data_dir and lang:
        prefix = f"EN-{lang.upper()}"
        train_path = os.path.join(data_dir, f"{prefix}_train.jsonl")
        test_path  = os.path.join(data_dir, f"{prefix}_test.jsonl")
        print(f"Loading local JSONL data:")
        print(f"  train: {train_path}")
        print(f"  test : {test_path}")
        return load_local_split(train_path), load_local_split(test_path)

    if not dataset_name:
        raise ValueError(
            "Provide either --data_dir and --lang for local JSONL, "
            "or --dataset_name for a HuggingFace Hub dataset."
        )
    print(f"Loading dataset from HF Hub: {dataset_name}")
    return (load_hf_split(dataset_name, "train", token=hf_token),
            load_hf_split(dataset_name, "test",  token=hf_token))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, label_ids):
        seq_preds, seq_labels = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(ID2LABEL[l])
            seq_preds.append(ID2LABEL[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


def detailed_report(predictions, label_ids, src_words_list, tgt_words_list):
    """Print full per-label metrics (P/R/F1), macro/micro averages, and per-side breakdown."""
    preds = np.argmax(predictions, axis=-1)

    all_true, all_pred = [], []
    src_true, src_pred = [], []
    tgt_true, tgt_pred = [], []

    truncated = 0
    for i, (pred_seq, label_seq) in enumerate(zip(preds, label_ids)):
        seq_preds, seq_labels = [], []
        n_src = len(src_words_list[i]) if i < len(src_words_list) else 0
        n_tgt = len(tgt_words_list[i]) if i < len(tgt_words_list) else 0

        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            seq_labels.append(ID2LABEL[l])
            seq_preds.append(ID2LABEL[p])

        all_true.append(seq_labels)
        all_pred.append(seq_preds)

        # Per-side split is only safe when every word survived tokenisation.
        # If max_length truncated some words, we can't reliably attribute the
        # surviving labels to src vs tgt by index — skip this row for the
        # per-side breakdown (it's still counted in overall metrics).
        if len(seq_labels) == n_src + n_tgt:
            src_true.append(seq_labels[:n_src])
            src_pred.append(seq_preds[:n_src])
            tgt_true.append(seq_labels[n_src:])
            tgt_pred.append(seq_preds[n_src:])
        else:
            truncated += 1

    # Flatten for sklearn per-label metrics
    from sklearn.metrics import classification_report as sklearn_report
    flat_true = [l for seq in all_true for l in seq]
    flat_pred = [l for seq in all_pred for l in seq]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Per-label token-level metrics (sklearn — includes O and B-FF separately)
    print("\n--- Token-level metrics (per label) ---")
    print(sklearn_report(flat_true, flat_pred, digits=4, zero_division=0))

    # Entity-level metrics (seqeval — evaluates B-FF spans)
    print("--- Entity-level metrics (seqeval) ---")
    print(classification_report(all_true, all_pred, digits=4, zero_division=0))

    # Summary
    print("--- Summary ---")
    print(f"  Macro F1 (token-level):  {_macro_f1_token(flat_true, flat_pred):.4f}")
    print(f"  Entity F1 (seqeval):     {f1_score(all_true, all_pred):.4f}")
    print(f"  Entity Precision:        {precision_score(all_true, all_pred):.4f}")
    print(f"  Entity Recall:           {recall_score(all_true, all_pred):.4f}")

    # Per-side breakdown
    if truncated:
        print(f"\n  (Per-side metrics computed on {len(src_true)} / "
              f"{len(src_true) + truncated} examples; "
              f"{truncated} skipped due to max_length truncation.)")

    if src_true:
        print(f"\n--- Source (English) side ---")
        print(f"  Entity F1:        {f1_score(src_true, src_pred):.4f}")
        print(f"  Entity Precision: {precision_score(src_true, src_pred):.4f}")
        print(f"  Entity Recall:    {recall_score(src_true, src_pred):.4f}")

    if tgt_true:
        print(f"\n--- Target side ---")
        print(f"  Entity F1:        {f1_score(tgt_true, tgt_pred):.4f}")
        print(f"  Entity Precision: {precision_score(tgt_true, tgt_pred):.4f}")
        print(f"  Entity Recall:    {recall_score(tgt_true, tgt_pred):.4f}")

    print("=" * 60)

    return {
        "macro_f1_token": _macro_f1_token(flat_true, flat_pred),
        "entity_f1": f1_score(all_true, all_pred),
        "entity_precision": precision_score(all_true, all_pred),
        "entity_recall": recall_score(all_true, all_pred),
        "src_entity_f1": f1_score(src_true, src_pred) if src_true else None,
        "tgt_entity_f1": f1_score(tgt_true, tgt_pred) if tgt_true else None,
    }


def _macro_f1_token(flat_true, flat_pred):
    """Compute macro F1 at the token level across all labels."""
    from sklearn.metrics import f1_score as sklearn_f1
    labels = sorted(set(flat_true) | set(flat_pred))
    return sklearn_f1(flat_true, flat_pred, labels=labels, average="macro", zero_division=0)


# ──────────────────────────────────────────────────────────────────────────────
# Weighted Trainer — upweight rare B-FF class
# ──────────────────────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the cross-entropy loss,
    giving more importance to the rare B-FF label."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _compute_class_weights(label_lists, ff_weight=None):
    """Compute class weights from label data.
    If ff_weight is given, use [1.0, ff_weight].
    Otherwise, compute inverse frequency weights automatically."""
    if ff_weight is not None:
        weights = [1.0, ff_weight]
        print(f"  Class weights: O={weights[0]:.1f}, B-FF={weights[1]:.1f} (manual)")
        return weights

    # Auto-compute from label distribution
    counts = {0: 0, 1: 0}
    for labels in label_lists:
        for l in labels:
            lid = l if isinstance(l, int) else LABEL2ID.get(l, 0)
            counts[lid] += 1

    total = counts[0] + counts[1]
    if counts[1] == 0:
        weights = [1.0, 1.0]
    else:
        # Inverse frequency, normalised so O=1.0
        weights = [1.0, counts[0] / counts[1]]

    print(f"  Label distribution: O={counts[0]}, B-FF={counts[1]}")
    print(f"  Class weights: O={weights[0]:.1f}, B-FF={weights[1]:.1f} (auto)")
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────
def train_model(
    model_name="xlm-roberta-base",
    dataset_name=DEFAULT_DATASET,
    data_dir=None,
    lang=None,
    output_dir="./ff_model",
    epochs=10,
    batch_size=16,
    lr=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_length=512,
    seed=42,
    early_stopping_patience=3,
    ff_weight=None,
    hf_token=None,
):
    """
    Train a token classification model for false friend detection.

    Args:
        model_name:    HuggingFace model name (e.g. xlm-roberta-base, bert-base-multilingual-cased)
        dataset_name:  HuggingFace dataset name (default: false-friends/en_es_token)
        output_dir:    Directory to save the trained model
        epochs:        Number of training epochs
        batch_size:    Training batch size
        lr:            Learning rate
        weight_decay:  Weight decay for AdamW
        warmup_ratio:  Warmup ratio for scheduler
        max_length:    Max sequence length (source + target combined)
        seed:          Random seed
        early_stopping_patience: Early stopping patience (0 to disable)
        ff_weight:     Weight for B-FF class in loss (default: None = auto-compute
                       from inverse frequency). Set e.g. 10.0, 20.0, 50.0 to manually
                       control how much the model focuses on false friends.
        hf_token:      HuggingFace token for private datasets

    Returns:
        (trainer, tokenizer) tuple
    """
    (tr_sw, tr_sl, tr_tw, tr_tl), (va_sw, va_sl, va_tw, va_tl) = resolve_splits(
        dataset_name=dataset_name, data_dir=data_dir, lang=lang, hf_token=hf_token,
    )
    print(f"  Train: {len(tr_sw)} pairs, Test: {len(va_sw)} pairs")

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = FalseFriendPairDataset(tr_sw, tr_sl, tr_tw, tr_tl, tokenizer, max_length)
    val_dataset = FalseFriendPairDataset(va_sw, va_sl, va_tw, va_tl, tokenizer, max_length)
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    # Compute class weights for imbalanced B-FF label
    all_train_labels = list(tr_sl) + list(tr_tl)
    class_weights = _compute_class_weights(all_train_labels, ff_weight=ff_weight)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "label_list": LABEL_LIST,
        "max_length": max_length,
    }
    with open(os.path.join(output_dir, "ff_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Final evaluation on test set
    print("\n\n" + "#" * 60)
    print(f"# FINAL TEST SET EVALUATION")
    print(f"# Model: {model_name}")
    print(f"# Dataset: {dataset_name}")
    print("#" * 60)

    preds_output = trainer.predict(val_dataset)
    test_results = detailed_report(preds_output.predictions, preds_output.label_ids, va_sw, va_tw)

    # Save results to file
    results_path = os.path.join(output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return trainer, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Predict
# ──────────────────────────────────────────────────────────────────────────────
def predict(model_path, source, target):
    """
    Predict false friends in a source-target sentence pair.

    Args:
        model_path: Path to trained model directory
        source:     Source (English) sentence string
        target:     Target (Spanish) sentence string

    Returns:
        dict with keys: source_tokens, target_tokens, source_ff, target_ff
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config_path = os.path.join(model_path, "ff_config.json")
    max_length = 512
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_length = config.get("max_length", 512)

    src_words = source.split()
    tgt_words = target.split()

    encoding = tokenizer(
        src_words, tgt_words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    encoding_for_ids = tokenizer(
        src_words, tgt_words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    input_tensors = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**input_tensors)
    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # word_ids restart at 0 for the target sequence — use sequence_ids() to
    # split predictions into separate src/tgt word→label dicts.
    word_ids     = encoding_for_ids.word_ids()
    sequence_ids = encoding_for_ids.sequence_ids()

    src_word_preds, tgt_word_preds = {}, {}
    for token_idx, (word_id, seq_id) in enumerate(zip(word_ids, sequence_ids)):
        if word_id is None or seq_id is None:
            continue
        side = src_word_preds if seq_id == 0 else tgt_word_preds
        if word_id not in side:
            side[word_id] = ID2LABEL[preds[token_idx]]

    src_ff, tgt_ff = [], []

    print(f"\n{'='*60}")
    print(f"Source:  {source}")
    print(f"Target:  {target}")
    print(f"{'='*60}\n")
    print(f"  {'Token':<25} {'Side':<10} {'Prediction':<10}")
    print(f"  {'-'*45}")

    for i, word in enumerate(src_words):
        label = src_word_preds.get(i, "O")
        marker = " <<<" if label == "B-FF" else ""
        print(f"  {word:<25} {'source':<10} {label:<10}{marker}")
        if label == "B-FF":
            src_ff.append(word)
    for i, word in enumerate(tgt_words):
        label = tgt_word_preds.get(i, "O")
        marker = " <<<" if label == "B-FF" else ""
        print(f"  {word:<25} {'target':<10} {label:<10}{marker}")
        if label == "B-FF":
            tgt_ff.append(word)

    print()
    if src_ff or tgt_ff:
        if src_ff:
            print(f"  Source false friends: {src_ff}")
        if tgt_ff:
            print(f"  Target false friends: {tgt_ff}")
    else:
        print("  No false friends detected.")

    return {
        "source_tokens": src_words,
        "target_tokens": tgt_words,
        "source_ff": src_ff,
        "target_ff": tgt_ff,
        "source_predictions": {i: src_word_preds.get(i, "O") for i in range(len(src_words))},
        "target_predictions": {i: tgt_word_preds.get(i, "O") for i in range(len(tgt_words))},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_path, dataset_name=DEFAULT_DATASET, split="test",
                   data_dir=None, lang=None, hf_token=None):
    """
    Evaluate a trained model on a dataset split.

    Args:
        model_path:    Path to trained model directory
        dataset_name:  HuggingFace dataset name (used if data_dir/lang not given)
        split:         Dataset split to evaluate on (default: test)
        data_dir:      Local directory containing JSONL files
                       (e.g. data/token_classification). Pair with --lang.
        lang:          'es' or 'fr' — picks EN-ES or EN-FR files in data_dir.
        hf_token:      HuggingFace token for private datasets
    """
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    config_path = os.path.join(model_path, "ff_config.json")
    max_length = 512
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        max_length = config.get("max_length", 512)

    if data_dir and lang:
        path = os.path.join(data_dir, f"EN-{lang.upper()}_{split}.jsonl")
        print(f"Loading local JSONL: {path}")
        src_w, src_l, tgt_w, tgt_l = load_local_split(path)
    else:
        print(f"Loading dataset: {dataset_name} [{split}]")
        src_w, src_l, tgt_w, tgt_l = load_hf_split(dataset_name, split, token=hf_token)
    print(f"  Evaluating on {len(src_w)} sentence pairs")

    dataset = FalseFriendPairDataset(src_w, src_l, tgt_w, tgt_l, tokenizer, max_length)
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    preds_output = trainer.predict(dataset)
    detailed_report(preds_output.predictions, preds_output.label_ids, src_w, tgt_w)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="False Friend Token Classification (Paired)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train ---
    tp = subparsers.add_parser("train", help="Train a token classification model")
    tp.add_argument("--model_name", type=str, default="xlm-roberta-base")
    tp.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET,
                    help="HuggingFace Hub dataset id (used if --data_dir/--lang not given).")
    tp.add_argument("--data_dir", type=str, default=None,
                    help="Local directory with EN-{ES|FR}_{train,test}.jsonl files "
                         "(produced by create_token_classification_dataset.py).")
    tp.add_argument("--lang", choices=["es", "fr"], default=None,
                    help="Language pair to load from --data_dir.")
    tp.add_argument("--output_dir", type=str, default="./ff_model")
    tp.add_argument("--epochs", type=int, default=10)
    tp.add_argument("--batch_size", type=int, default=16)
    tp.add_argument("--lr", type=float, default=5e-5)
    tp.add_argument("--weight_decay", type=float, default=0.01)
    tp.add_argument("--warmup_ratio", type=float, default=0.1)
    tp.add_argument("--max_length", type=int, default=512)
    tp.add_argument("--seed", type=int, default=42)
    tp.add_argument("--early_stopping_patience", type=int, default=3)
    tp.add_argument("--ff_weight", type=float, default=None,
                    help="Weight for B-FF class in loss. Default: auto-compute from inverse frequency. "
                         "Try 10, 20, or 50 to upweight false friends.")
    tp.add_argument("--hf_token", type=str, default=None)

    # --- Predict ---
    pp = subparsers.add_parser("predict", help="Predict false friends in a sentence pair")
    pp.add_argument("--model_path", type=str, required=True)
    pp.add_argument("--source", type=str, required=True)
    pp.add_argument("--target", type=str, required=True)

    # --- Evaluate ---
    ep = subparsers.add_parser("evaluate", help="Evaluate model on a dataset")
    ep.add_argument("--model_path", type=str, required=True)
    ep.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET)
    ep.add_argument("--data_dir", type=str, default=None,
                    help="Local directory with JSONL files (overrides --dataset_name).")
    ep.add_argument("--lang", choices=["es", "fr"], default=None)
    ep.add_argument("--split", type=str, default="test")
    ep.add_argument("--hf_token", type=str, default=None)

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            lang=args.lang,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_length=args.max_length,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            ff_weight=args.ff_weight,
            hf_token=args.hf_token,
        )
    elif args.command == "predict":
        predict(args.model_path, args.source, args.target)
    elif args.command == "evaluate":
        evaluate_model(
            args.model_path,
            dataset_name=args.dataset_name,
            split=args.split,
            data_dir=args.data_dir,
            lang=args.lang,
            hf_token=args.hf_token,
        )


if __name__ == "__main__":
    main()