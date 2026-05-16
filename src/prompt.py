"""Prompt-based fine-tuning utilities for MeZO (following Malladi et al., 2023).

Instead of a randomly initialized classification head, we keep the pretrained
MLM head and predict label words at a [MASK] position.

SST-2 template:  "{sentence} It was <mask>."
Label mapping:    0 (negative) -> " terrible"   (leading space = single BPE token for RoBERTa)
                  1 (positive) -> " great"
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader


LABEL_WORDS = {0: " terrible", 1: " great"}


def build_prompt_dataset(raw_sentences: list[str], labels: list[int], tokenizer, max_length: int = 128):
    """Tokenize prompt-wrapped sentences. Returns dict of tensors."""
    prompts = [f'{s.rstrip()} It was {tokenizer.mask_token}.' for s in raw_sentences]
    enc = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc


def get_label_token_ids(tokenizer) -> dict[int, int]:
    """Return {class_idx: single token_id} for each label word.

    Uses leading-space variants so that RoBERTa's BPE treats each word as
    a single token (e.g., ' terrible' -> id 6587, ' great' -> id 372).
    """
    result = {}
    for cls_idx, word in LABEL_WORDS.items():
        ids = tokenizer.encode(word, add_special_tokens=False)
        assert len(ids) == 1, f"'{word}' tokenizes to {len(ids)} tokens — choose a single-token word"
        result[cls_idx] = ids[0]
    return result


def prompt_loss_fn(
    model,
    batch: dict,
    label_token_ids: dict[int, int],
    mask_token_id: int,
) -> torch.Tensor:
    """Cross-entropy loss at the [MASK] position over the two label-word logits.

    Args:
        model:           RobertaForMaskedLM
        batch:           dict with 'input_ids', 'attention_mask', 'labels' (class indices 0/1)
        label_token_ids: {0: token_id_of_negative_word, 1: token_id_of_positive_word}
        mask_token_id:   tokenizer.mask_token_id  (NOT model.config — config may be None for RoBERTa)
    """
    input_ids = batch["input_ids"]
    mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)  # (B, 2)
    batch_idx = mask_positions[:, 0]
    pos_idx   = mask_positions[:, 1]

    out = model(input_ids=input_ids, attention_mask=batch["attention_mask"])
    mask_logits = out.logits[batch_idx, pos_idx, :]  # (B, vocab_size)

    token_ids = torch.tensor(
        [label_token_ids[0], label_token_ids[1]], device=input_ids.device
    )
    label_logits = mask_logits[:, token_ids]  # (B, 2)
    return torch.nn.functional.cross_entropy(label_logits, batch["labels"])


@torch.no_grad()
def prompt_evaluate(
    model,
    loader: DataLoader,
    label_token_ids: dict[int, int],
    mask_token_id: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean CE loss, accuracy) for prompt-based MLM eval."""
    model.eval()
    tok_ids = torch.tensor([label_token_ids[0], label_token_ids[1]], device=device)
    correct = total = 0
    total_loss = 0.0

    for batch in loader:
        # batch is a TensorDataset tuple: (input_ids, attention_mask, labels)
        ids, attn, labs = (t.to(device) for t in batch)
        b = {"input_ids": ids, "attention_mask": attn, "labels": labs}

        loss = prompt_loss_fn(model, b, label_token_ids, mask_token_id)

        mask_pos = (ids == mask_token_id).nonzero(as_tuple=False)
        logits = model(input_ids=ids, attention_mask=attn).logits
        mask_logits = logits[mask_pos[:, 0], mask_pos[:, 1], :][:, tok_ids]
        preds = mask_logits.argmax(dim=-1)

        correct += (preds == labs).sum().item()
        total += labs.size(0)
        total_loss += loss.item() * labs.size(0)

    return total_loss / total, correct / total
