"""
train_distilbert.py

Fine-tunes distilbert-base-uncased as a multi-class intent classifier over
the intents defined in data/intents.json. The trained model + tokenizer are
saved to ./models/distilbert-intent so chatbot.py can load them at inference
time.

Usage:
    python train_distilbert.py --epochs 8 --batch-size 16

This script is fully self-contained: it builds the training set directly
from data/intents.json's "patterns" field (used as-is, plus light
augmentation), so no external dataset file is required to get a working
baseline model.
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from preprocessing import clean_text

SEED = 42
DATA_DIR = "data"
MODEL_OUT_DIR = os.path.join("models", "distilbert-intent")
BASE_MODEL_NAME = "distilbert-base-uncased"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Lightweight augmentation so a small hand-written pattern set produces a
# more robust training distribution. This is a simple synonym/paraphrase
# style swap intended for demonstration/FYP scope, not a production-grade
# augmentation pipeline.
# ---------------------------------------------------------------------------

_PREFIXES = ["", "honestly, ", "lately ", "i just wanted to say ", "so, "]
_SUFFIXES = ["", " lately", " right now", " today", " these days"]


def augment_pattern(pattern: str, n: int = 3) -> List[str]:
    variants = {pattern}
    for _ in range(n):
        prefix = random.choice(_PREFIXES)
        suffix = random.choice(_SUFFIXES)
        variant = f"{prefix}{pattern}{suffix}".strip()
        variants.add(variant)
    return list(variants)


def load_dataset(intents_path: str, label2id_path: str, augment_factor: int = 3) -> Tuple[List[str], List[int]]:
    with open(intents_path, "r") as f:
        intents_data = json.load(f)
    with open(label2id_path, "r") as f:
        label2id = json.load(f)

    texts: List[str] = []
    labels: List[int] = []

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        label_id = label2id[tag]
        for pattern in intent["patterns"]:
            for variant in augment_pattern(pattern, n=augment_factor):
                texts.append(clean_text(variant))
                labels.append(label_id)

    return texts, labels


class IntentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 5e-5
    max_length: int = 64
    warmup_ratio: float = 0.1
    val_split: float = 0.15


def train(config: TrainConfig) -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_distilbert] Using device: {device}")

    label2id_path = os.path.join(DATA_DIR, "label2id.json")
    id2label_path = os.path.join(DATA_DIR, "id2label.json")
    intents_path = os.path.join(DATA_DIR, "intents.json")

    with open(label2id_path) as f:
        label2id = json.load(f)
    with open(id2label_path) as f:
        id2label_raw = json.load(f)
    id2label = {int(k): v for k, v in id2label_raw.items()}
    num_labels = len(label2id)

    print(f"[train_distilbert] Building dataset from {intents_path} ({num_labels} labels)")
    texts, labels = load_dataset(intents_path, label2id_path)
    print(f"[train_distilbert] Total examples after augmentation: {len(texts)}")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=config.val_split, random_state=SEED, stratify=labels
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    train_ds = IntentDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_ds = IntentDataset(val_texts, val_labels, tokenizer, config.max_length)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)

        val_acc, val_f1, val_report = evaluate(model, val_loader, id2label, device)
        print(
            f"[epoch {epoch}/{config.epochs}] train_loss={avg_train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            print(f"  -> new best macro-F1 ({val_f1:.4f}), saving checkpoint to {MODEL_OUT_DIR}")
            model.save_pretrained(MODEL_OUT_DIR)
            tokenizer.save_pretrained(MODEL_OUT_DIR)

    print("\n[train_distilbert] Final validation report (best checkpoint):")
    print(val_report)
    print(f"[train_distilbert] Training complete. Best macro-F1: {best_val_f1:.4f}")
    print(f"[train_distilbert] Model artifacts saved to: {MODEL_OUT_DIR}")


@torch.no_grad()
def evaluate(model, data_loader, id2label, device) -> Tuple[float, float, str]:
    model.eval()
    all_preds, all_labels = [], []

    for batch in data_loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0
    )
    return acc, f1, report


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the MindCare DistilBERT intent classifier")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
