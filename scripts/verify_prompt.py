"""Quick zero-shot verification of prompt-based label tokens."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

# `datasets` MUST be imported before `torch` on Windows — pyarrow's native
# libs collide with torch's otherwise and the process segfaults silently.
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.prompt import build_prompt_dataset, get_label_token_ids

MODEL_NAME = "roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_token_ids = get_label_token_ids(tokenizer)
print("label_token_ids:", label_token_ids)
print("decoded:", {k: repr(tokenizer.decode([v])) for k, v in label_token_ids.items()})

raw = load_dataset("glue", "sst2")
ds = raw["validation"]
enc = build_prompt_dataset(ds["sentence"][:128], ds["label"][:128], tokenizer)
loader = DataLoader(
    TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"]),
    batch_size=64,
)

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

mask_id = model.config.mask_token_id
tok_ids = torch.tensor([label_token_ids[0], label_token_ids[1]], device=DEVICE)
correct = total = 0

with torch.no_grad():
    for ids, attn, labs in loader:
        ids, attn, labs = ids.to(DEVICE), attn.to(DEVICE), labs.to(DEVICE)
        mask_pos = (ids == mask_id).nonzero(as_tuple=False)
        logits = model(input_ids=ids, attention_mask=attn).logits
        mask_logits = logits[mask_pos[:, 0], mask_pos[:, 1], :][:, tok_ids]
        preds = mask_logits.argmax(dim=-1)
        correct += (preds == labs).sum().item()
        total += labs.size(0)

print(f"zero-shot accuracy (128 val examples): {correct}/{total} = {correct/total:.4f}")
