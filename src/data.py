"""SST-2 (GLUE) loader for RoBERTa-style models.

Returns torch DataLoaders yielding dicts with `input_ids`, `attention_mask`,
`labels` — exactly what HuggingFace classification models consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@dataclass
class SST2Loaders:
    train: DataLoader
    val: DataLoader
    num_labels: int
    tokenizer: AutoTokenizer


def get_sst2_loaders(
    model_name: str = "roberta-base",
    batch_size: int = 16,
    eval_batch_size: int = 64,
    max_length: int = 128,
    train_subset: Optional[int] = None,
    seed: int = 0,
) -> SST2Loaders:
    """Load GLUE/SST-2 train + validation as DataLoaders ready for `model(**batch)`.

    `train_subset`: if set, take only the first N training examples (after shuffle).
    Useful for fast iteration / few-shot regimes that match MeZO paper setup.
    """
    raw = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_ds = tokenized["train"].shuffle(seed=seed)
    if train_subset is not None:
        train_ds = train_ds.select(range(min(train_subset, len(train_ds))))

    val_ds = tokenized["validation"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False)

    return SST2Loaders(train=train_loader, val=val_loader, num_labels=2, tokenizer=tokenizer)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
