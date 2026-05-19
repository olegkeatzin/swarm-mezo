"""Smoke test: 10 steps × N=2 of the reputation pipeline.

Catches: vmap-incompat in the reputation consensus path, shape mismatches
in the probe-batch broadcast, missing FedHistory fields. Should finish in <60s.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.federated import train_fedavg_mezo
from src.prompt import build_prompt_dataset, get_label_token_ids
from src.reputation import ReputationConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
label_token_ids = get_label_token_ids(tokenizer)
MASK_TOKEN_ID = tokenizer.mask_token_id


def loader(sentences, labels, batch_size, shuffle):
    enc = build_prompt_dataset(sentences, labels, tokenizer, max_length=128)
    ds  = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


train = raw["train"].shuffle(seed=0).select(range(64))
agent_loaders = [
    loader(train.select(range(32))["sentence"], train.select(range(32))["label"], 16, True),
    loader(train.select(range(32, 64))["sentence"], train.select(range(32, 64))["label"], 16, True),
]
val_loader = loader(
    raw["validation"]["sentence"][:32], raw["validation"]["label"][:32], 16, False,
)

probe = train.select(range(48, 64))
enc = build_prompt_dataset(probe["sentence"], probe["label"], tokenizer, max_length=128)
probe_batch = (
    enc["input_ids"].to(DEVICE),
    enc["attention_mask"].to(DEVICE),
    enc["labels"].to(DEVICE),
)


def make_model():
    m = AutoModelForMaskedLM.from_pretrained(
        "roberta-base", attn_implementation="eager",
        dtype=torch.bfloat16,
    ).to(DEVICE)
    m.eval()
    return m


print("Loading model...")
_ = make_model(); del _
print("Running 10-step reputation smoke test (N=2, K=5, β=10)...")

hist = train_fedavg_mezo(
    model_factory=make_model,
    train_loaders=agent_loaders,
    val_loader=val_loader,
    label_token_ids=label_token_ids,
    mask_token_id=MASK_TOKEN_ID,
    device=DEVICE,
    n_agents=2,
    total_steps=10,
    local_steps=5,
    lr=1e-6, eps=1e-3,
    eval_every=10,
    log_every=2,
    seed=0,
    reputation_config=ReputationConfig(eval_batch=probe_batch, beta=10.0, gamma_r=1.0),
)

print(f"\nfinal val acc: {hist.eval_acc[-1]:.4f}")
print(f"reputation rounds: {len(hist.reputations)}")
print(f"reputations per round:")
for i, r in enumerate(hist.reputations):
    s = sum(r)
    print(f"  round {i+1}: {[f'{x:.3f}' for x in r]}  shares={[f'{x/s:.3f}' for x in r]}")
print(f"eval losses per round:")
for i, l in enumerate(hist.consensus_eval_losses):
    print(f"  round {i+1}: {[f'{x:.4f}' for x in l]}")
print("\nOK")
