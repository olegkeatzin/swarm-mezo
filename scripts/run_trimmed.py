"""Trimmed-mean consensus on RoBERTa+SST-2 — robust-aggregation control branch.

The stable form of "steer away from bad agents". True repulsion (W_ij < 0)
breaks the row-stochasticity the Day-3 spectral contraction relies on and
diverges; instead the worst `trim_k` agents (by probe loss) are dropped from
the centroid each consensus round. W stays non-negative and row-stochastic.
Trimmed-mean / Krum family of Byzantine-robust FL aggregators (Yin et al. 2018).

The trim_k modifier composes with the §4 reputation law (mode="loss"):
  β = 0  + trim_k>0  → plain trimmed mean (uniform over survivors)
  β > 0  + trim_k>0  → reputation-weighting among the survivors

IID only. Day 4 established the non-IID regime is the deliberate "bad case" —
there the worst probe-loss agent may merely be on a hard shard, so dropping it
discards data rather than a bad model; the theory marks non-IID out of scope.
This run uses the IID split where the per-agent probe loss is a clean fitness
signal, the premise the whole reputation/robust-aggregation argument rests on.

Sweep — for each β the trim_k ∈ {2, 4} grid (N=8 agents):
  β ∈ {0, 0.1, 0.5, 1, 10}  ×  trim_k ∈ {2, 4}   = 10 runs

The trim_k=0 column (pure β-sweep, no trimming) is the existing IID reputation
run in outputs/day5_reputation_iid.json — use it as the baseline.

Results saved incrementally (idempotent — finished runs are skipped).

    uv run python scripts/run_trimmed.py
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


# ── config (identical to run_reputation.py so the comparison is head-to-head) ──
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
GAMMA_R         = 1.0

# β × trim_k grid. trim_k=0 (no trimming) lives in day5_reputation_iid.json.
BETA_VALUES   = [0.0, 0.1, 0.5, 1.0, 10.0]
TRIM_K_VALUES = [2, 4]
REPUTATION_CONFIGS = [
    {"beta": b, "trim_k": k} for b in BETA_VALUES for k in TRIM_K_VALUES
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name(0)}")

OUT_PATH = ROOT / "outputs" / "day5_trimmed.json"
OUT_PATH.parent.mkdir(exist_ok=True)
print(f"sharding: iid  ->  {OUT_PATH.name}")


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


def _iid_partition(n_agents, seed):
    """IID partition: uniform random split of train_ds. Every agent's shard is
    an IID sample of the global SST-2 distribution, so the per-agent probe loss
    L_i is a clean fitness signal — the premise the robust-aggregation argument
    rests on. Mirrors run_reputation.py's SHARDING=iid path."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_ds))
    rng.shuffle(idx)
    return [chunk.tolist() for chunk in np.array_split(idx, n_agents)]


_AGENT_INDICES = _iid_partition(N_AGENTS, SEED)


def make_agent_loaders(n_agents):
    assert n_agents == N_AGENTS, "IID split is pre-computed for N_AGENTS"
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
    print(f"trimmed-mean probe batch: {PROBE_SIZE} samples (neg={neg}, pos={pos})")
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
            "sharding": "iid",
            "gamma_r": GAMMA_R,
            "mode": "loss+trim",
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
        "reputations":           h.reputations,            # carried reputation state
        "consensus_eval_losses": h.consensus_eval_losses,
    }


# ── warm up ───────────────────────────────────────────────────────────────────
print("\nPer-agent class balance (neg, pos) — iid sharding:")
for i, (neg, pos) in enumerate(_agent_class_balance(N_AGENTS)):
    print(f"  agent {i}: neg={neg:3d}, pos={pos:3d}")

probe_batch = make_probe_batch()

print("\nLoading model weights...")
_tmp = make_model(); del _tmp
print("Weights cached.\n")

results = load_results()

# ── run each (β, trim_k) ──────────────────────────────────────────────────────
print("=" * 60)
print(f"Day-5 trimmed-mean sweep (N={N_AGENTS}, K={LOCAL_STEPS}, steps={TOTAL_STEPS})")
print(f"grid: β ∈ {BETA_VALUES} × trim_k ∈ {TRIM_K_VALUES}  ({len(REPUTATION_CONFIGS)} runs)")
print("=" * 60)

for cfg in REPUTATION_CONFIGS:
    beta, trim_k = cfg["beta"], cfg["trim_k"]
    key = f"beta{beta}_trim{trim_k}"
    if key in results["runs"]:
        prev_acc = results["runs"][key]["eval_acc"][-1]
        print(f"{key}: already done (acc={prev_acc:.4f}), skipping.")
        continue

    print(f"\n--- {key}  (β={beta}, γ_r={GAMMA_R}, drop worst {trim_k}/{N_AGENTS}) ---")

    rep_cfg = ReputationConfig(
        eval_batch=probe_batch, beta=beta, gamma_r=GAMMA_R,
        mode="loss", trim_k=trim_k,
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

    results["runs"][key] = {
        "beta": beta, "trim_k": trim_k, "mode": "loss+trim", **hist_to_dict(hist),
    }
    print(f"{key} final val acc: {hist.eval_acc[-1]:.4f}")
    save_results(results)

print("\nDone.")
