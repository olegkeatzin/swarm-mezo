"""Reputation-modulated consensus on non-IID RoBERTa+SST-2.

Tests whether the working window β ∈ [1, 10] found in the synthetic E3
(теория/swarm_mezo/) survives the move to a real LLM loss landscape.

Same non-IID Dirichlet(α=0.5) partition, same N=8, same K=100 used in
Day 3, so results are directly comparable to the {full, ring, star} runs
already in outputs/day3_consensus.json.

Sweep (β values span 4 orders of magnitude — the synthetic E3 working
window depends on the typical inter-agent loss gap on the probe batch,
which is unknown a priori on RoBERTa+SST-2, so we bracket aggressively):
  β = 0.0    baseline ≡ FedAvg-MeZO (reputations stay equal -> exact FedAvg)
  β = 0.1    likely lower edge of the working window
  β = 0.5    middle of the suspected working window
  β = 1.0    likely upper edge / onset of slowdown
  β = 10.0   expected cascade regime — one agent monopolises reputation

Eval batch for fitness scoring: 32 samples from train[1000:1032] — disjoint
from training data, no leakage into val.

Results saved incrementally to outputs/day5_reputation.json.

    uv run python scripts/run_reputation.py
"""
# datasets import BEFORE torch to dodge pyarrow/torch DLL clash on Windows
from datasets import load_dataset

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.federated import train_fedavg_mezo
from src.prompt import build_prompt_dataset, get_label_token_ids
from src.reputation import ReputationConfig


# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "roberta-base"
SEED            = 0
TRAIN_SUBSET    = 1000
PROBE_SIZE      = 32
BATCH_SIZE      = 16
MAX_LENGTH      = 128
MEZO_LR         = 1e-6
MEZO_EPS        = 1e-3
TOTAL_STEPS     = 5_000
EVAL_EVERY      = 500
N_AGENTS        = 8
LOCAL_STEPS     = 100
DIRICHLET_ALPHA = 0.5
GAMMA_R         = 1.0

REPUTATION_CONFIGS = [
    {"beta": 0.0},      # baseline ≡ FedAvg-MeZO (control that the code path is correct)
    {"beta": 0.1},      # likely lower edge of the working window per E3
    {"beta": 0.5},      # middle of the suspected working window
    {"beta": 1.0},      # likely upper edge / onset of slowdown
    {"beta": 10.0},     # expected cascade regime
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name(0)}")

OUT_PATH = ROOT / "outputs" / "day5_reputation.json"
OUT_PATH.parent.mkdir(exist_ok=True)


# ── data ──────────────────────────────────────────────────────────────────────
raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_token_ids = get_label_token_ids(tokenizer)
MASK_TOKEN_ID = tokenizer.mask_token_id

train_full = raw["train"].shuffle(seed=SEED)
train_ds   = train_full.select(range(TRAIN_SUBSET))
probe_ds   = train_full.select(range(TRAIN_SUBSET, TRAIN_SUBSET + PROBE_SIZE))


def make_prompt_loader(sentences, labels, batch_size, shuffle):
    enc = build_prompt_dataset(sentences, labels, tokenizer, max_length=MAX_LENGTH)
    ds  = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


val_loader = make_prompt_loader(
    raw["validation"]["sentence"], raw["validation"]["label"], 64, shuffle=False,
)


def _dirichlet_partition(n_agents, alpha, seed):
    """Standard Dirichlet partition (Hsu et al. 2019). Mirrors run_consensus.py
    exactly so the comparison is head-to-head."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(train_ds["label"])
    classes = np.unique(labels)

    agent_indices: list[list[int]] = [[] for _ in range(n_agents)]
    for c in classes:
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet([alpha] * n_agents)
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        chunks = np.split(class_idx, split_points)
        for i, chunk in enumerate(chunks):
            agent_indices[i].extend(chunk.tolist())

    min_shard = min(len(idx) for idx in agent_indices)
    if min_shard < BATCH_SIZE:
        sizes = [len(idx) for idx in agent_indices]
        raise RuntimeError(
            f"smallest agent shard ({min_shard}) is below BATCH_SIZE ({BATCH_SIZE}); "
            f"shard sizes: {sizes}."
        )

    rng2 = np.random.default_rng(seed)
    for idx in agent_indices:
        rng2.shuffle(idx)
    return agent_indices


_AGENT_INDICES = _dirichlet_partition(N_AGENTS, DIRICHLET_ALPHA, SEED)


def make_agent_loaders(n_agents):
    assert n_agents == N_AGENTS, "Dirichlet split is pre-computed for N_AGENTS"
    return [
        make_prompt_loader(
            train_ds.select(_AGENT_INDICES[i])["sentence"],
            train_ds.select(_AGENT_INDICES[i])["label"],
            BATCH_SIZE, shuffle=True,
        )
        for i in range(n_agents)
    ]


def _agent_class_balance(n_agents):
    out = []
    for i in range(n_agents):
        labels = train_ds.select(_AGENT_INDICES[i])["label"]
        n_pos = sum(labels); n_neg = len(labels) - n_pos
        out.append((n_neg, n_pos))
    return out


def make_probe_batch():
    enc = build_prompt_dataset(
        probe_ds["sentence"], probe_ds["label"], tokenizer, max_length=MAX_LENGTH,
    )
    ids  = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)
    labs = enc["labels"].to(DEVICE)
    pos = int(sum(probe_ds["label"]))
    neg = PROBE_SIZE - pos
    print(f"reputation probe batch: {PROBE_SIZE} samples (neg={neg}, pos={pos})")
    return ids, attn, labs


def make_model():
    m = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager",
        dtype=torch.bfloat16,
    ).to(DEVICE)
    m.eval()
    return m


# ── load/save with idempotency ────────────────────────────────────────────────
def load_results():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {
        "config": {
            "model": MODEL_NAME, "train_subset": TRAIN_SUBSET,
            "probe_size": PROBE_SIZE,
            "total_steps": TOTAL_STEPS, "mezo_lr": MEZO_LR, "mezo_eps": MEZO_EPS,
            "eval_every": EVAL_EVERY, "n_agents": N_AGENTS,
            "local_steps": LOCAL_STEPS,
            "sharding": "dirichlet",
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "gamma_r": GAMMA_R,
            "agent_class_balance": _agent_class_balance(N_AGENTS),
        },
        "runs": {},
    }


def save_results(results):
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"  -> saved to {OUT_PATH}")


def hist_to_dict(h):
    return {
        "step":              h.step,
        "train_loss":        h.train_loss,
        "eval_step":         h.eval_step,
        "eval_loss":         h.eval_loss,
        "eval_acc":          h.eval_acc,
        "reputations":           h.reputations,
        "consensus_eval_losses": h.consensus_eval_losses,
    }


# ── warm up model cache ──────────────────────────────────────────────────────
print("\nNon-IID per-agent class balance (neg, pos):")
for i, (neg, pos) in enumerate(_agent_class_balance(N_AGENTS)):
    print(f"  agent {i}: neg={neg:3d}, pos={pos:3d}")

probe_batch = make_probe_batch()

print("\nLoading model weights...")
_tmp = make_model(); del _tmp
print("Weights cached.\n")

results = load_results()

# ── run each beta ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Day-5 reputation sweep (N={N_AGENTS}, K={LOCAL_STEPS}, steps={TOTAL_STEPS})")
print("=" * 60)

for cfg in REPUTATION_CONFIGS:
    key = f"beta{cfg['beta']}"
    if key in results["runs"]:
        prev_acc = results["runs"][key]["eval_acc"][-1]
        print(f"{key}: already done (acc={prev_acc:.4f}), skipping.")
        continue

    print(f"\n--- {key}  (β={cfg['beta']}, γ_r={GAMMA_R}) ---")

    rep_cfg = ReputationConfig(
        eval_batch=probe_batch, beta=cfg["beta"], gamma_r=GAMMA_R,
    )

    torch.manual_seed(SEED)
    hist = train_fedavg_mezo(
        model_factory=make_model,
        train_loaders=make_agent_loaders(N_AGENTS),
        val_loader=val_loader,
        label_token_ids=label_token_ids,
        mask_token_id=MASK_TOKEN_ID,
        device=DEVICE,
        n_agents=N_AGENTS,
        total_steps=TOTAL_STEPS,
        local_steps=LOCAL_STEPS,
        lr=MEZO_LR, eps=MEZO_EPS,
        eval_every=EVAL_EVERY,
        log_every=max(1, EVAL_EVERY // 10),
        seed=SEED,
        reputation_config=rep_cfg,
    )

    results["runs"][key] = {"beta": cfg["beta"], **hist_to_dict(hist)}
    print(f"{key} final val acc: {hist.eval_acc[-1]:.4f}")

    if hist.reputations:
        last_reps = hist.reputations[-1]
        max_share = max(last_reps) / sum(last_reps)
        print(f"{key} final reputation concentration: "
              f"max share = {max_share:.3f} (uniform = {1/N_AGENTS:.3f})")

    save_results(results)

print("\nDone.")
