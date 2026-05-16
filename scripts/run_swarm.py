"""Day-4 experiment: Swarm-MeZO (PSO-style consensus) on non-IID data.

Uses the SAME Dirichlet(α=0.5) partition as Day 3, the SAME N=8 / K=100 /
5000 steps, so swarm results are directly comparable to the {full, ring, star}
runs already in outputs/day3_consensus.json.

Two hyperparameters (see src/swarm.py for the math):
  - β (selectivity / inverse temperature in softmax)
  - α (social coefficient / inertia)

Sweep (3 configs, ~2.5h each on RTX 4060 Ti):

  (α=0.5, β=1.0)   moderate swarm — expected sweet spot
  (α=0.5, β=5.0)   sharp selection — does aggressive leader-following help
                   on non-IID, or does noise in 16-sample loss estimates make
                   it pick random winners?
  (α=1.0, β=2.0)   no inertia, just selection-weighted averaging — bridges
                   smoothly to FedAvg (β→0) and to pure ES (β→∞)

Baselines (already in outputs/day3_consensus.json):
  full = (α=1.0, β=0.0)        exact FedAvg — no selection
  Day 2 single-MeZO            no agents, no consensus

Eval batch for fitness scoring is sampled from TRAIN (indices 1000:1032,
disjoint from the 0:1000 used for actual training) — class-balanced by virtue
of being a uniform random slice, and no leakage into the val metric.

Results saved incrementally to outputs/day4_swarm.json.

    uv run python scripts/run_swarm.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.federated import train_fedavg_mezo
from src.prompt import build_prompt_dataset, get_label_token_ids
from src.swarm import SwarmConfig


# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "roberta-base"
SEED            = 0
TRAIN_SUBSET    = 1000
PROBE_SIZE      = 32                       # eval-batch for swarm fitness scoring
BATCH_SIZE      = 16
MAX_LENGTH      = 128
MEZO_LR         = 1e-6
MEZO_EPS        = 1e-3
TOTAL_STEPS     = 5_000
EVAL_EVERY      = 500
N_AGENTS        = 8
LOCAL_STEPS     = 100
DIRICHLET_ALPHA = 0.5                      # same as Day 3 → results comparable

# (alpha, beta) configurations to run.
SWARM_CONFIGS = [
    {"alpha": 0.5, "beta": 1.0},
    {"alpha": 0.5, "beta": 5.0},
    {"alpha": 1.0, "beta": 2.0},
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name(0)}")

OUT_PATH = ROOT / "outputs" / "day4_swarm.json"
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
    """Standard Dirichlet partition for FL non-IID (Hsu et al. 2019).

    Same code as scripts/run_consensus.py — kept inline for self-containment
    and so the comparison is exactly head-to-head.
    """
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
    """Build the fixed (B=PROBE_SIZE, L=MAX_LENGTH) batch used to score agents."""
    enc = build_prompt_dataset(
        probe_ds["sentence"], probe_ds["label"], tokenizer, max_length=MAX_LENGTH,
    )
    ids  = enc["input_ids"].to(DEVICE)
    attn = enc["attention_mask"].to(DEVICE)
    labs = enc["labels"].to(DEVICE)
    pos = int(sum(probe_ds["label"]))
    neg = PROBE_SIZE - pos
    print(f"swarm probe batch: {PROBE_SIZE} samples (neg={neg}, pos={pos})")
    return ids, attn, labs


def make_model():
    m = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager",
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
        "swarm_weights":     h.swarm_weights,
        "swarm_eval_losses": h.swarm_eval_losses,
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

# ── run each (alpha, beta) configuration ──────────────────────────────────────
print("=" * 60)
print(f"Day-4 swarm sweep  (N={N_AGENTS}, K={LOCAL_STEPS}, steps={TOTAL_STEPS})")
print("=" * 60)

for cfg in SWARM_CONFIGS:
    key = f"alpha{cfg['alpha']}_beta{cfg['beta']}"
    if key in results["runs"]:
        prev_acc = results["runs"][key]["eval_acc"][-1]
        print(f"{key}: already done (acc={prev_acc:.4f}), skipping.")
        continue

    print(f"\n--- {key}  (α={cfg['alpha']}, β={cfg['beta']}) ---")

    swarm_cfg = SwarmConfig(
        eval_batch=probe_batch, alpha=cfg["alpha"], beta=cfg["beta"],
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
        swarm_config=swarm_cfg,
    )

    results["runs"][key] = {"alpha": cfg["alpha"], "beta": cfg["beta"], **hist_to_dict(hist)}
    print(f"{key} final val acc: {hist.eval_acc[-1]:.4f}")

    # Snapshot how concentrated the swarm weights were on average.
    import statistics
    if hist.swarm_weights:
        max_weights = [max(w) for w in hist.swarm_weights]
        print(f"{key} swarm-weight concentration: "
              f"mean max-w={statistics.mean(max_weights):.3f} "
              f"(uniform={1/N_AGENTS:.3f}, fully concentrated=1.000)")

    save_results(results)

print("\nDone.")
