"""Training loops for MeZO and standard backprop fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import move_batch
from src.mezo import MeZOOptimizer


@dataclass
class TrainHistory:
    step: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    eval_step: list[int] = field(default_factory=list)
    eval_loss: list[float] = field(default_factory=list)
    eval_acc: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (mean CE loss, accuracy) over `loader`."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(**batch)
        n = batch["labels"].size(0)
        total_loss += out.loss.item() * n
        preds = out.logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_count += n
    return total_loss / total_count, total_correct / total_count


def train_mezo(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    steps: int = 20_000,
    lr: float = 1e-6,
    eps: float = 1e-3,
    eval_every: int = 500,
    log_every: int = 50,
    progress: bool = True,
) -> TrainHistory:
    """Fine-tune `model` with MeZO. Model is kept in eval() mode throughout.

    Returns a TrainHistory with step-indexed train loss (loss at θ+εz, smoothed
    only by the natural noise of MeZO) and periodic eval (loss + accuracy).
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)  # MeZO still gates on requires_grad for "which params to perturb"

    optimizer = MeZOOptimizer(model, lr=lr, eps=eps)
    history = TrainHistory()
    iterator = cycle(train_loader)
    bar = tqdm(range(steps), disable=not progress, desc="MeZO")

    for step in bar:
        batch = move_batch(next(iterator), device)

        def loss_fn() -> torch.Tensor:
            return model(**batch).loss

        loss_plus = optimizer.step(loss_fn)

        if step % log_every == 0:
            history.step.append(step)
            history.train_loss.append(loss_plus)
            bar.set_postfix(loss=f"{loss_plus:.4f}")

        if (step + 1) % eval_every == 0 or step == steps - 1:
            eval_loss, eval_acc = evaluate(model, val_loader, device)
            history.eval_step.append(step + 1)
            history.eval_loss.append(eval_loss)
            history.eval_acc.append(eval_acc)
            bar.set_postfix(loss=f"{loss_plus:.4f}", val_acc=f"{eval_acc:.4f}")

    return history


def train_adamw(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    eval_every: int = 500,
    log_every: int = 50,
    progress: bool = True,
) -> TrainHistory:
    """Standard fine-tuning baseline with AdamW + backprop."""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    history = TrainHistory()
    global_step = 0
    total_steps = epochs * len(train_loader)
    bar = tqdm(total=total_steps, disable=not progress, desc="AdamW")

    for _ in range(epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)
            model.train()
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % log_every == 0:
                history.step.append(global_step)
                history.train_loss.append(loss.item())
                bar.set_postfix(loss=f"{loss.item():.4f}")

            if (global_step + 1) % eval_every == 0:
                eval_loss, eval_acc = evaluate(model, val_loader, device)
                history.eval_step.append(global_step + 1)
                history.eval_loss.append(eval_loss)
                history.eval_acc.append(eval_acc)
                bar.set_postfix(loss=f"{loss.item():.4f}", val_acc=f"{eval_acc:.4f}")

            global_step += 1
            bar.update(1)

    # Final eval
    eval_loss, eval_acc = evaluate(model, val_loader, device)
    history.eval_step.append(global_step)
    history.eval_loss.append(eval_loss)
    history.eval_acc.append(eval_acc)
    bar.close()

    return history
