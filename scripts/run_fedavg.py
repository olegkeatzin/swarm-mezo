"""Run FedAvg-MeZO experiments and save results incrementally to JSON.

Usage:
    uv run python scripts/run_fedavg.py

Results are saved to outputs/day2_fedavg.json after each configuration,
so the run can be interrupted and results inspected at any point.
"""
import json
import sys
from pathlib import Path

# Force UTF-8 stdout so non-ASCII chars in print()s don't kill the process
# under Windows' default cp1251 console encoding.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# `datasets` MUST be imported before `torch` on Windows — pyarrow's native
# libs collide with torch's otherwise and the process segfaults silently.
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.federated import train_fedavg_mezo
from src.prompt import build_prompt_dataset, get_label_token_ids

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = 'roberta-base'
SEED         = 0
TRAIN_SUBSET = 1000
BATCH_SIZE   = 16
MAX_LENGTH   = 128
MEZO_LR      = 1e-6
MEZO_EPS     = 1e-3
TOTAL_STEPS  = 5_000   # enough to show convergence trends
EVAL_EVERY   = 500

N_VALUES          = [1, 2, 4, 8]
LOCAL_STEPS_FIXED = 10

N_FIXED            = 4
LOCAL_STEPS_VALUES = [1, 10, 100, 1000]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'gpu:    {torch.cuda.get_device_name(0)}')

OUT_PATH = ROOT / 'outputs' / 'day2_fedavg.json'
OUT_PATH.parent.mkdir(exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
raw = load_dataset('glue', 'sst2')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_token_ids = get_label_token_ids(tokenizer)
MASK_TOKEN_ID   = tokenizer.mask_token_id

train_ds = raw['train'].shuffle(seed=SEED).select(range(TRAIN_SUBSET))

def make_prompt_loader(sentences, labels, batch_size, shuffle):
    enc = build_prompt_dataset(sentences, labels, tokenizer, max_length=MAX_LENGTH)
    ds  = TensorDataset(enc['input_ids'], enc['attention_mask'], enc['labels'])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)

val_loader = make_prompt_loader(
    raw['validation']['sentence'], raw['validation']['label'], 64, shuffle=False
)

def make_agent_loaders(n_agents):
    shard_size = len(train_ds) // n_agents
    return [
        make_prompt_loader(
            train_ds.select(range(i * shard_size, (i + 1) * shard_size))['sentence'],
            train_ds.select(range(i * shard_size, (i + 1) * shard_size))['label'],
            BATCH_SIZE, shuffle=True,
        )
        for i in range(n_agents)
    ]

def make_model():
    # eager attention bypasses transformers' SDPA fast-path whose
    # `_ignore_bidirectional_mask_sdpa` does `padding_mask.all()` in a Python
    # `if` — that breaks vmap (data-dependent control flow).
    m = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager",
    ).to(DEVICE)
    m.eval()
    return m

def hist_to_dict(h):
    return {
        'step': h.step, 'train_loss': h.train_loss,
        'eval_step': h.eval_step, 'eval_loss': h.eval_loss, 'eval_acc': h.eval_acc,
    }

def load_results():
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {
        'config': {
            'model': MODEL_NAME, 'train_subset': TRAIN_SUBSET,
            'total_steps': TOTAL_STEPS, 'mezo_lr': MEZO_LR,
            'mezo_eps': MEZO_EPS, 'eval_every': EVAL_EVERY,
        },
        'n_sweep': {},
        'k_sweep': {},
    }

def save_results(results):
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f'  -> saved to {OUT_PATH}')

# ── warm up model cache ───────────────────────────────────────────────────────
print('Loading model weights...')
_tmp = make_model(); del _tmp
print('Weights cached.\n')

results = load_results()

# ── Experiment 1: N sweep ─────────────────────────────────────────────────────
print('=' * 60)
print(f'Experiment 1: N sweep  (K={LOCAL_STEPS_FIXED}, steps={TOTAL_STEPS})')
print('=' * 60)

for N in N_VALUES:
    key = str(N)
    if key in results['n_sweep']:
        print(f'N={N}: already done (acc={results["n_sweep"][key]["eval_acc"][-1]:.4f}), skipping.')
        continue

    print(f'\n--- N={N}, local_steps={LOCAL_STEPS_FIXED} ---')
    torch.manual_seed(SEED)
    hist = train_fedavg_mezo(
        model_factory=make_model,
        train_loaders=make_agent_loaders(N),
        val_loader=val_loader,
        label_token_ids=label_token_ids,
        mask_token_id=MASK_TOKEN_ID,
        device=DEVICE,
        n_agents=N,
        total_steps=TOTAL_STEPS,
        local_steps=LOCAL_STEPS_FIXED,
        lr=MEZO_LR, eps=MEZO_EPS,
        eval_every=EVAL_EVERY,
        log_every=max(1, EVAL_EVERY // 10),
        seed=SEED,
    )
    results['n_sweep'][key] = hist_to_dict(hist)
    print(f'N={N} final val acc: {hist.eval_acc[-1]:.4f}')
    save_results(results)

# ── Experiment 2: K sweep ─────────────────────────────────────────────────────
print('\n' + '=' * 60)
print(f'Experiment 2: K sweep  (N={N_FIXED}, steps={TOTAL_STEPS})')
print('=' * 60)

for K in LOCAL_STEPS_VALUES:
    key = str(K)
    if key in results['k_sweep']:
        print(f'K={K}: already done (acc={results["k_sweep"][key]["eval_acc"][-1]:.4f}), skipping.')
        continue

    print(f'\n--- N={N_FIXED}, local_steps={K} ---')
    torch.manual_seed(SEED)
    hist = train_fedavg_mezo(
        model_factory=make_model,
        train_loaders=make_agent_loaders(N_FIXED),
        val_loader=val_loader,
        label_token_ids=label_token_ids,
        mask_token_id=MASK_TOKEN_ID,
        device=DEVICE,
        n_agents=N_FIXED,
        total_steps=TOTAL_STEPS,
        local_steps=K,
        lr=MEZO_LR, eps=MEZO_EPS,
        eval_every=EVAL_EVERY,
        log_every=max(1, EVAL_EVERY // 10),
        seed=SEED,
    )
    results['k_sweep'][key] = hist_to_dict(hist)
    print(f'K={K} final val acc: {hist.eval_acc[-1]:.4f}')
    save_results(results)

print('\nAll done.')
