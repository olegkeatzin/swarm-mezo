"""End-to-end smoke test for vmap-based FedAvg-MeZO with real RoBERTa.

Runs 5 MeZO steps with N=2 agents on a tiny SST-2 subset to verify that
torch.func.vmap composes with HuggingFace RobertaForMaskedLM (the riskiest
part of the new src/federated.py — unit tests cover the helpers but can't
exercise the full vmap+functional_call+HF stack).

If this script completes without error and reports finite per-agent losses,
the production training loop is good to go.

    uv run python scripts/smoke_test_vmap.py

Diagnostic note: every progress message is printed with flush=True AND
appended to outputs/smoke_test_vmap.log so that if the process dies silently
(WSL/CUDA DLL conflict, HF Hub hang, etc.) we can still see the last
checkpoint reached.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "outputs" / "smoke_test_vmap.log"
LOG_PATH.parent.mkdir(exist_ok=True)
LOG_PATH.write_text("")  # truncate previous run


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


log(f"booting; python={sys.executable}")
log(f"cwd={Path.cwd()}, ROOT={ROOT}")

try:
    # NOTE: `datasets` MUST be imported before `torch` — otherwise pyarrow's
    # native libs collide with torch's on Windows and the process segfaults
    # with no Python-level traceback. Do not reorder.
    log("import datasets ...")
    from datasets import load_dataset
    log("  datasets ok")

    log("import torch ...")
    import torch
    log(f"  torch {torch.__version__}, cuda_available={torch.cuda.is_available()}")

    log("import torch.utils.data ...")
    from torch.utils.data import DataLoader, TensorDataset
    log("  ok")

    log("import transformers ...")
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    log("  transformers ok")

    log("import src.federated, src.prompt ...")
    from src.federated import train_fedavg_mezo
    from src.prompt import build_prompt_dataset, get_label_token_ids
    log("  ok")

    MODEL_NAME  = "roberta-base"
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_AGENTS    = 2
    TOTAL_STEPS = 5
    BATCH       = 4
    MAX_LEN     = 64
    N_TRAIN     = 32
    N_VAL       = 16

    log(f"device={DEVICE}")
    if DEVICE.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    label_token_ids = get_label_token_ids(tokenizer)
    MASK_ID = tokenizer.mask_token_id
    log("  tokenizer ok")

    log("loading SST-2 ...")
    raw = load_dataset("glue", "sst2")
    log(f"  dataset loaded; train={len(raw['train'])}, val={len(raw['validation'])}")


    def make_loader(sentences, labels, batch_size, shuffle):
        enc = build_prompt_dataset(sentences, labels, tokenizer, max_length=MAX_LEN)
        ds  = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["labels"])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


    log("building agent loaders ...")
    train_subset = raw["train"].shuffle(seed=0).select(range(N_TRAIN))
    shard = N_TRAIN // N_AGENTS
    train_loaders = [
        make_loader(
            train_subset.select(range(i * shard, (i + 1) * shard))["sentence"],
            train_subset.select(range(i * shard, (i + 1) * shard))["label"],
            BATCH, shuffle=True,
        )
        for i in range(N_AGENTS)
    ]
    val_loader = make_loader(
        raw["validation"]["sentence"][:N_VAL],
        raw["validation"]["label"][:N_VAL],
        BATCH, shuffle=False,
    )
    log("  loaders ok")


    def make_model():
        # `attn_implementation="eager"` bypasses transformers' SDPA fast-path
        # whose `_ignore_bidirectional_mask_sdpa` does `padding_mask.all()` in
        # a Python `if` — that breaks vmap (data-dependent control flow).
        m = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME, attn_implementation="eager",
        ).to(DEVICE)
        m.eval()
        return m


    log(f"running {TOTAL_STEPS} vmap'd MeZO steps with N={N_AGENTS} agents ...")
    t0 = time.time()
    hist = train_fedavg_mezo(
        model_factory=make_model,
        train_loaders=train_loaders,
        val_loader=val_loader,
        label_token_ids=label_token_ids,
        mask_token_id=MASK_ID,
        device=DEVICE,
        n_agents=N_AGENTS,
        total_steps=TOTAL_STEPS,
        local_steps=2,
        lr=1e-6,
        eps=1e-3,
        eval_every=TOTAL_STEPS,
        log_every=1,
        seed=0,
    )
    elapsed = time.time() - t0

    log(f"--- results ---")
    log(f"elapsed:                 {elapsed:.1f}s   ({elapsed / TOTAL_STEPS:.2f}s/step)")
    log(f"per-agent train loss:    {hist.per_agent_train_loss}")
    log(f"final eval loss/acc:     {hist.eval_loss[-1]:.4f} / {hist.eval_acc[-1]:.4f}")

    assert all(
        all(0 < x < 100 for x in losses) for losses in hist.per_agent_train_loss
    ), "non-finite or absurd train loss — vmap pipeline likely broken"
    assert 0 <= hist.eval_acc[-1] <= 1, "eval accuracy out of range"

    log("SMOKE TEST PASSED — vmap + functional_call + RoBERTa works.")

except BaseException as e:
    log(f"!!! crashed with {type(e).__name__}: {e}")
    log("traceback:\n" + traceback.format_exc())
    raise
