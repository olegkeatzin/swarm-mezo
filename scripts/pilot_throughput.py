"""Throughput pilot for vmap FedAvg-MeZO at production batch/seqlen.

Runs 50 MeZO steps for each N in {1, 2, 4, 8} using the same config as
scripts/run_fedavg.py (BATCH_SIZE=16, MAX_LENGTH=128) and reports:
  - seconds / step
  - peak GPU memory
  - extrapolated ETA for the 5000-step production sweep

Does NOT touch outputs/day2_fedavg.json. Pilot results go to
outputs/pilot_throughput.json + log to outputs/pilot_throughput.log.

    uv run python scripts/pilot_throughput.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_PATH    = ROOT / "outputs" / "pilot_throughput.log"
RESULT_PATH = ROOT / "outputs" / "pilot_throughput.json"
LOG_PATH.parent.mkdir(exist_ok=True)
LOG_PATH.write_text("")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# datasets MUST come before torch on Windows (pyarrow/torch DLL conflict).
log("import datasets ...")
from datasets import load_dataset

log("import torch ...")
import torch

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.federated import train_fedavg_mezo
from src.prompt import build_prompt_dataset, get_label_token_ids


# Production-matched config.
MODEL_NAME    = "roberta-base"
BATCH_SIZE    = 16
MAX_LENGTH    = 128
MEZO_LR       = 1e-6
MEZO_EPS      = 1e-3
PILOT_STEPS   = 50
LOCAL_STEPS   = 10
TRAIN_SUBSET  = 1000
SEED          = 0
N_VALUES      = [1, 2, 4, 8]
TARGET_STEPS  = 5_000   # full sweep target, used for ETA extrapolation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"device={DEVICE}")
if DEVICE.type == "cuda":
    log(f"gpu={torch.cuda.get_device_name(0)}")

log("loading dataset + tokenizer ...")
raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_token_ids = get_label_token_ids(tokenizer)
MASK_ID = tokenizer.mask_token_id
train_ds = raw["train"].shuffle(seed=SEED).select(range(TRAIN_SUBSET))


def make_loader(sentences, labels, batch_size, shuffle):
    enc = build_prompt_dataset(sentences, labels, tokenizer, max_length=MAX_LENGTH)
    ds  = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# Tiny val loader — eval cost would distort the per-step measurement otherwise.
val_loader = make_loader(
    raw["validation"]["sentence"][:64],
    raw["validation"]["label"][:64],
    BATCH_SIZE, shuffle=False,
)


def make_agent_loaders(n_agents):
    shard = len(train_ds) // n_agents
    return [
        make_loader(
            train_ds.select(range(i * shard, (i + 1) * shard))["sentence"],
            train_ds.select(range(i * shard, (i + 1) * shard))["label"],
            BATCH_SIZE, shuffle=True,
        )
        for i in range(n_agents)
    ]


def make_model():
    m = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager",
    ).to(DEVICE)
    m.eval()
    return m


results: dict[str, dict] = {}

try:
    for n in N_VALUES:
        log(f"")
        log(f"=== N={n}, {PILOT_STEPS} steps ===")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        loaders = make_agent_loaders(n)
        t0 = time.time()
        hist = train_fedavg_mezo(
            model_factory=make_model,
            train_loaders=loaders,
            val_loader=val_loader,
            label_token_ids=label_token_ids,
            mask_token_id=MASK_ID,
            device=DEVICE,
            n_agents=n,
            total_steps=PILOT_STEPS,
            local_steps=LOCAL_STEPS,
            lr=MEZO_LR,
            eps=MEZO_EPS,
            eval_every=PILOT_STEPS,   # only one eval at the end
            log_every=PILOT_STEPS,    # minimal logging overhead
            seed=SEED,
        )
        elapsed = time.time() - t0
        sec_per_step = elapsed / PILOT_STEPS
        eta_full_h = TARGET_STEPS * sec_per_step / 3600.0
        peak_mem_gb = (
            torch.cuda.max_memory_allocated() / 1024**3 if DEVICE.type == "cuda" else 0.0
        )

        log(f"  elapsed:      {elapsed:.1f}s")
        log(f"  sec/step:     {sec_per_step:.3f}")
        log(f"  peak GPU mem: {peak_mem_gb:.2f} GB")
        log(f"  ETA @ {TARGET_STEPS} steps: {eta_full_h:.2f} h")

        results[str(n)] = {
            "elapsed_s":       round(elapsed, 2),
            "sec_per_step":    round(sec_per_step, 4),
            "peak_mem_gb":     round(peak_mem_gb, 3),
            "eta_full_hours":  round(eta_full_h, 2),
        }

        # release the stacked weights between configs
        del hist, loaders
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    total_eta_h = sum(r["eta_full_hours"] for r in results.values())
    log(f"")
    log(f"=== summary ===")
    for n in N_VALUES:
        r = results[str(n)]
        log(f"  N={n}: {r['sec_per_step']:.3f}s/step, peak {r['peak_mem_gb']:.2f} GB, "
            f"ETA {r['eta_full_hours']:.2f} h")
    log(f"  N-sweep total ETA: {total_eta_h:.2f} h")
    log(f"  + K-sweep (3 extra runs at N=4): "
        f"{3 * results['4']['eta_full_hours']:.2f} h")
    log(f"  GRAND TOTAL for run_fedavg.py: "
        f"{total_eta_h + 3 * results['4']['eta_full_hours']:.2f} h")

    RESULT_PATH.write_text(json.dumps({
        "config": {
            "batch_size":   BATCH_SIZE,
            "max_length":   MAX_LENGTH,
            "pilot_steps":  PILOT_STEPS,
            "target_steps": TARGET_STEPS,
            "local_steps":  LOCAL_STEPS,
        },
        "by_n": results,
    }, indent=2))
    log(f"saved -> {RESULT_PATH}")

except BaseException as e:
    log(f"!!! crashed with {type(e).__name__}: {e}")
    log("traceback:\n" + traceback.format_exc())
    raise
