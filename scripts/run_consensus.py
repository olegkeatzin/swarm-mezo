"""Day-3 experiment: consensus topology sweep on non-IID data.

For each topology in {full, ring, star} we run N agents through the vmap
federated MeZO loop. The training data is sharded **non-IID via Dirichlet
partition** (Hsu et al. 2019, standard FL non-IID recipe): for each class,
the share going to each agent is drawn from Dir(α, ..., α). α=0.5 gives
moderate skew — every agent sees both classes but with strong bias.
Validation stays balanced.

The defense story is two-layered:

  (a) Theory of consensus: regardless of data, the residual ‖θ − θ̄‖ contracts
      by ≈|λ₂(W)| every consensus round. We record this and overlay the
      theoretical line — works on non-IID just as on IID.

  (b) Effect on accuracy: on non-IID, full graph (gap=1) averages aggressively
      and destroys per-agent specialization → expected to underperform. Ring
      and star (smaller gap) preserve diversity → can match or beat full.

This contrasts directly with Day 2 (FedAvg-MeZO on IID SST-2, all topologies
would behave the same), making "topology matters" a real claim.

Results are saved incrementally to outputs/day3_consensus.json.

    uv run python scripts/run_consensus.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# `datasets` MUST be imported before `torch` on Windows.
from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.consensus import (
    apply_consensus,
    build_topology,
    second_eigenvalue,
    spectral_gap,
)
from src.federated import train_fedavg_mezo
from src.prompt import build_prompt_dataset, get_label_token_ids


# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "roberta-base"
SEED         = 0
TRAIN_SUBSET = 1000
BATCH_SIZE   = 16
MAX_LENGTH   = 128
MEZO_LR      = 1e-6
MEZO_EPS     = 1e-3
TOTAL_STEPS  = 5_000
EVAL_EVERY   = 500
N_AGENTS     = 8
LOCAL_STEPS  = 100             # 50 consensus rounds — enough resolution on the residual curve
DIRICHLET_ALPHA = 0.5          # non-IID strength: 0 → extreme (each agent one class),
                               # ∞ → IID. α=0.5 is the standard moderate-skew choice
                               # used in Hsu et al. 2019 and most FL benchmarks.
TOPOLOGIES   = ["full", "ring", "star"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"gpu:    {torch.cuda.get_device_name(0)}")

OUT_PATH = ROOT / "outputs" / "day3_consensus.json"
OUT_PATH.parent.mkdir(exist_ok=True)


# ── data ──────────────────────────────────────────────────────────────────────
raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_token_ids = get_label_token_ids(tokenizer)
MASK_TOKEN_ID = tokenizer.mask_token_id

train_ds = raw["train"].shuffle(seed=SEED).select(range(TRAIN_SUBSET))


def make_prompt_loader(sentences, labels, batch_size, shuffle):
    enc = build_prompt_dataset(sentences, labels, tokenizer, max_length=MAX_LENGTH)
    ds  = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


val_loader = make_prompt_loader(
    raw["validation"]["sentence"], raw["validation"]["label"], 64, shuffle=False,
)


def _dirichlet_partition(n_agents, alpha, seed):
    """Standard Dirichlet partition for FL non-IID (Hsu et al. 2019).

    For each class c, the share going to agent i is proportional to a sample
    from Dir(α, ..., α). Returns list-of-index-lists for each agent.
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

    # DataLoader needs at least BATCH_SIZE samples per agent (drop_last=True
    # with shuffle=True), otherwise an agent loader is empty and `cycle()`
    # raises StopIteration mid-step. With α=0.5, N=8, 1000 samples this is
    # safe for seed=0; we check explicitly so a bad seed fails fast.
    min_shard = min(len(idx) for idx in agent_indices)
    if min_shard < BATCH_SIZE:
        sizes = [len(idx) for idx in agent_indices]
        raise RuntimeError(
            f"smallest agent shard ({min_shard}) is below BATCH_SIZE ({BATCH_SIZE}); "
            f"shard sizes: {sizes}. Increase α, reduce BATCH_SIZE, or change SEED."
        )

    rng2 = np.random.default_rng(seed)   # re-shuffle within each agent for batching variety
    for idx in agent_indices:
        rng2.shuffle(idx)
    return agent_indices


_AGENT_INDICES = _dirichlet_partition(N_AGENTS, DIRICHLET_ALPHA, SEED)


def make_agent_loaders(n_agents):
    """Non-IID class-skewed sharding via Dirichlet(α=DIRICHLET_ALPHA) partition.

    Each agent gets a different proportion of positives/negatives. With α=0.5
    every agent sees both classes (no degenerate single-class agents) but with
    strong skew, so consensus mixing has real work to do.
    """
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
    """Per-agent (neg, pos) counts after Dirichlet split."""
    out = []
    for i in range(n_agents):
        labels = train_ds.select(_AGENT_INDICES[i])["label"]
        n_pos = sum(labels); n_neg = len(labels) - n_pos
        out.append((n_neg, n_pos))
    return out


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
            "total_steps": TOTAL_STEPS, "mezo_lr": MEZO_LR, "mezo_eps": MEZO_EPS,
            "eval_every": EVAL_EVERY, "n_agents": N_AGENTS,
            "local_steps": LOCAL_STEPS,
            "sharding": "dirichlet",
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "agent_class_balance": _agent_class_balance(N_AGENTS),
        },
        "topologies": {},
    }


def save_results(results):
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"  -> saved to {OUT_PATH}")


def hist_to_dict(h):
    return {
        "step":                  h.step,
        "train_loss":            h.train_loss,
        "eval_step":             h.eval_step,
        "eval_loss":             h.eval_loss,
        "eval_acc":              h.eval_acc,
        "consensus_round":       h.consensus_round,
        "consensus_dist_before": h.consensus_dist_before,
        "consensus_dist_after":  h.consensus_dist_after,
    }


# ── topology metadata (for the visualizer) ────────────────────────────────────
print("\nNon-IID per-agent class balance (neg, pos):")
for i, (neg, pos) in enumerate(_agent_class_balance(N_AGENTS)):
    print(f"  agent {i}: neg={neg:3d}, pos={pos:3d}")

print("\nTopologies and their spectral gaps (N={}):".format(N_AGENTS))
topology_meta = {}
for name in TOPOLOGIES:
    W = build_topology(name, N_AGENTS)
    gap = spectral_gap(W)
    lam2 = second_eigenvalue(W)
    topology_meta[name] = {
        "spectral_gap":     gap,
        "second_eigenvalue": lam2,
    }
    print(f"  {name:>5}: gap={gap:.4f}, |λ₂|={lam2:.4f}")


# ── warm up model cache ──────────────────────────────────────────────────────
print("\nLoading model weights...")
_tmp = make_model(); del _tmp
print("Weights cached.\n")

results = load_results()
results.setdefault("topology_meta", {}).update(topology_meta)

# ── run each topology ─────────────────────────────────────────────────────────
print("=" * 60)
print(f"Day-3 consensus sweep  (N={N_AGENTS}, K={LOCAL_STEPS}, steps={TOTAL_STEPS})")
print("=" * 60)

for name in TOPOLOGIES:
    if name in results["topologies"]:
        prev_acc = results["topologies"][name]["eval_acc"][-1]
        print(f"{name}: already done (acc={prev_acc:.4f}), skipping.")
        continue

    print(f"\n--- topology={name} ---")
    W = build_topology(name, N_AGENTS).to(DEVICE)

    def consensus_fn(params):
        apply_consensus(params, W)

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
        consensus_fn=consensus_fn,
        track_consensus=True,
    )

    results["topologies"][name] = hist_to_dict(hist)
    print(f"{name} final val acc: {hist.eval_acc[-1]:.4f}")
    save_results(results)   # save first — empirical-rate math must not risk erasing the run

    # Empirical contraction rate over all consensus rounds.
    # For `full`, d_after is exactly 0 (all agents identical post-mixing) → log(0).
    # Filter positive ratios only; an empty list means "full-mixing topology, rate=0".
    ratios = [
        a / b for a, b in zip(hist.consensus_dist_after, hist.consensus_dist_before)
        if b > 1e-12 and a > 1e-12
    ]
    if ratios:
        import math
        log_mean = sum(math.log(r) for r in ratios) / len(ratios)
        emp_rate = math.exp(log_mean)
        print(f"{name} empirical contraction rate: {emp_rate:.4f}  "
              f"(theory |λ₂|={topology_meta[name]['second_eigenvalue']:.4f})")
    else:
        print(f"{name} empirical contraction rate: 0  (full-mixing, theory |λ₂|=0)")

print("\nDone.")
