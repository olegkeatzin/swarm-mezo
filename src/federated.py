"""Federated MeZO with true GPU parallelism via torch.func.vmap.

N agents share one base model architecture; their weights are stacked along a
leading agent dim and run through `vmap(functional_call)` so all N forward
passes execute as a single batched GPU op (real parallelism, not Python
threading).

MeZO perturbations apply directly to the stacked weight tensors. A single
`empty(N, *shape).normal_(generator=rng)` call places independent N(0, I)
samples into each agent slice (different chunks of the rng stream), and
re-seeding `rng` with the same per-step seed reproduces them exactly — that
is what preserves MeZO's three-pass perturb pattern that keeps memory at
inference level.

HuggingFace MLM models tie the input embedding tensor with the LM head
decoder. `stack_module_state` deduplicates by data_ptr, which would leave
the decoder out of the stacked params dict; functional_call would then keep
its pre-trained values fixed and MeZO would only ever perturb the embedding.
We break those ties before stacking with `_untie_weights_inplace`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Callable

import torch
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _patch_transformers_for_vmap() -> None:
    """Disable transformers' vmap-incompatible mask fast-path.

    `transformers.masking_utils._ignore_bidirectional_mask_sdpa` ends in
    `padding_mask.all()` consumed by a Python `if` — that's data-dependent
    control flow on a BatchedTensor, which vmap rejects. Forcing the helper
    to return False skips the optimization and falls through to the regular
    tensor-only mask construction (no perf cost worth caring about for ZO).
    Both `eager` and `sdpa` attn implementations route through this helper.
    """
    try:
        import transformers.masking_utils as mu
    except ImportError:
        return
    if not getattr(mu, "_mezo_vmap_patched", False):
        mu._ignore_bidirectional_mask_sdpa = lambda *a, **k: False
        mu._mezo_vmap_patched = True


_patch_transformers_for_vmap()


@dataclass
class FedHistory:
    """Per-agent train loss + shared eval (evaluated on agent 0's weights).

    consensus_round / consensus_dist_before / consensus_dist_after are populated
    only when train_fedavg_mezo is called with track_consensus=True (Day-3 setup).
    They record ‖θ − θ̄‖ across all stacked params immediately before and after
    each consensus mixing round, so that log(d_after / d_before) ≈ log|λ₂(W)|.

    swarm_weights is populated only when swarm_config is set: one (N,)
    softmax-weight vector per consensus round, useful to show which agents
    "led" the swarm over time.
    """
    eval_step:            list[int]        = field(default_factory=list)
    eval_loss:            list[float]      = field(default_factory=list)
    eval_acc:             list[float]      = field(default_factory=list)
    step:                 list[int]        = field(default_factory=list)
    train_loss:           list[float]      = field(default_factory=list)  # mean across agents
    per_agent_train_loss: list[list[float]] = field(default_factory=list)
    consensus_round:       list[int]   = field(default_factory=list)
    consensus_dist_before: list[float] = field(default_factory=list)
    consensus_dist_after:  list[float] = field(default_factory=list)
    swarm_weights:         list[list[float]] = field(default_factory=list)
    swarm_eval_losses:     list[list[float]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unit-tested in tests/test_federated_vmap.py)
# ──────────────────────────────────────────────────────────────────────────────

def _untie_weights_inplace(model: torch.nn.Module) -> None:
    """Break parameter tying so each tied parameter becomes independent.

    For each duplicate (same data_ptr as something already seen),
    rebind the attribute to a fresh `Parameter` cloning the data.
    """
    seen: dict[int, str] = {}
    duplicates: list[tuple[str, torch.nn.Parameter]] = []
    for name, p in model.named_parameters(remove_duplicate=False):
        if p.data_ptr() in seen:
            duplicates.append((name, p))
        else:
            seen[p.data_ptr()] = name

    for name, p in duplicates:
        if "." in name:
            module_path, attr = name.rsplit(".", 1)
            module = model.get_submodule(module_path)
        else:
            module, attr = model, name
        new_p = torch.nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad)
        setattr(module, attr, new_p)


def _stack_models(model_factory: Callable[[], torch.nn.Module], n_agents: int):
    """Build N model copies (untying tied weights first) and stack their state.

    Returns:
        base:    one model instance whose params will be substituted by
                 functional_call during each forward.
        params:  dict[str, Tensor of shape (N, *param_shape)]
        buffers: dict[str, Tensor of shape (N, *buffer_shape)]
    """
    models = []
    for _ in range(n_agents):
        m = model_factory()
        _untie_weights_inplace(m)
        models.append(m)
    base = models[0]
    params, buffers = stack_module_state(models)
    # Drop references to the redundant copies; their tensors live on inside `params`/`buffers`.
    del models
    return base, params, buffers


def _perturb_stacked(
    params: dict[str, torch.Tensor],
    scaling: float,
    seed: int,
    rng: torch.Generator,
) -> None:
    """In-place: each per-agent slice gets its own independent N(0, I) draw.

    Iteration order over `params` is fixed (insertion-ordered dict), so the
    rng stream is consumed identically across re-seeded calls — reseeding
    with the same `seed` regenerates the exact same z values.
    """
    rng.manual_seed(seed)
    for p in params.values():
        z = torch.empty_like(p).normal_(generator=rng)
        p.add_(z, alpha=scaling)


def _apply_mezo_update(
    params: dict[str, torch.Tensor],
    projected_grad: torch.Tensor,
    lr: float,
    seed: int,
    rng: torch.Generator,
) -> None:
    """params[name][i] ← params[name][i] - lr · projected_grad[i] · z[name][i]."""
    rng.manual_seed(seed)
    for p in params.values():
        z = torch.empty_like(p).normal_(generator=rng)
        bcast = projected_grad.view(-1, *([1] * (p.dim() - 1)))   # (N, 1, ..., 1)
        p.sub_(z * bcast, alpha=lr)


def fedavg_consensus(params: dict[str, torch.Tensor]) -> None:
    """In-place: replace each agent's slice with the mean across agents (W = 1/N)."""
    for p in params.values():
        mean = p.mean(dim=0, keepdim=True)
        p.copy_(mean.expand_as(p))


def _make_prompt_loss(base_model, label_tok_ids: torch.Tensor, mask_token_id: int):
    """vmap-safe prompt-based MLM loss closure.

    Avoids `nonzero` (data-dependent shape, breaks vmap). Locates the mask
    via argmax of the equality tensor — assumes exactly one mask per sample.
    """
    def per_agent_loss(params_i, buffers_i, input_ids, attention_mask, labels):
        out = functional_call(
            base_model, (params_i, buffers_i),
            args=(input_ids,),
            kwargs={"attention_mask": attention_mask},
        )
        mask_pos     = (input_ids == mask_token_id).long().argmax(dim=1)        # (B,)
        b_range      = torch.arange(input_ids.size(0), device=input_ids.device)
        mask_logits  = out.logits[b_range, mask_pos]                             # (B, V)
        label_logits = mask_logits[:, label_tok_ids]                             # (B, 2)
        return F.cross_entropy(label_logits, labels)
    return per_agent_loss


def _stack_agent_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """N batches of shape (B, ...) -> single tensor of shape (N, B, ...)."""
    ids  = torch.stack([b[0] for b in batches]).to(device, non_blocking=True)
    attn = torch.stack([b[1] for b in batches]).to(device, non_blocking=True)
    labs = torch.stack([b[2] for b in batches]).to(device, non_blocking=True)
    return ids, attn, labs


@torch.no_grad()
def _eval_one_agent(
    base_model,
    params_i: dict[str, torch.Tensor],
    buffers_i: dict[str, torch.Tensor],
    val_loader: DataLoader,
    label_tok_ids: torch.Tensor,
    mask_token_id: int,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one agent's weights on `val_loader`. Returns (mean CE loss, accuracy)."""
    correct = total = 0
    total_loss = 0.0
    for batch in val_loader:
        ids, attn, labs = (t.to(device) for t in batch)
        out = functional_call(
            base_model, (params_i, buffers_i),
            args=(ids,), kwargs={"attention_mask": attn},
        )
        mask_pos     = (ids == mask_token_id).long().argmax(dim=1)
        b_range      = torch.arange(ids.size(0), device=device)
        mask_logits  = out.logits[b_range, mask_pos]
        label_logits = mask_logits[:, label_tok_ids]
        loss = F.cross_entropy(label_logits, labs)
        preds = label_logits.argmax(dim=-1)
        correct += (preds == labs).sum().item()
        total   += labs.size(0)
        total_loss += loss.item() * labs.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def train_fedavg_mezo(
    model_factory: Callable[[], torch.nn.Module],
    train_loaders: list[DataLoader],
    val_loader: DataLoader,
    label_token_ids: dict[int, int],
    mask_token_id: int,
    device: torch.device,
    n_agents: int,
    total_steps: int,
    local_steps: int,
    lr: float = 1e-6,
    eps: float = 1e-3,
    eval_every: int = 500,
    log_every: int = 50,
    seed: int = 0,
    consensus_fn: Callable[[dict[str, torch.Tensor]], None] | None = None,
    track_consensus: bool = False,
    swarm_config: "SwarmConfig | None" = None,
) -> FedHistory:
    """vmap-based federated MeZO: N agents run as one batched forward on a single GPU.

    Args:
        model_factory:   () -> module on `device`, in eval mode. Tied weights
                         are broken automatically before stacking.
                         For HF transformers models, pass
                         `attn_implementation="eager"` to `from_pretrained` —
                         the default SDPA path branches on `padding_mask.all()`,
                         which breaks vmap (data-dependent control flow).
        train_loaders:   one DataLoader per agent. Each batch must be a
                         (input_ids, attention_mask, labels) tuple of equal
                         shapes across agents.
        val_loader:      shared validation set; evaluated on agent 0's weights.
        total_steps:     MeZO steps per agent.
        local_steps:     steps between consensus rounds.
        consensus_fn:    in-place mixer over stacked params. Defaults to
                         `fedavg_consensus` (= full graph, W = 1/N · 1·1ᵀ).
                         Pass a closure over `apply_consensus(params, W)` from
                         src.consensus for ring / star / arbitrary W (Day 3).
        track_consensus: when True, records ‖θ − θ̄‖ before and after each
                         consensus round into FedHistory. Adds two distance
                         computations per round — cheap relative to a MeZO step.
        swarm_config:    when set, replaces `consensus_fn` with PSO-style
                         swarm mixing (θ_i ← (1−α)·θ_i + α·Σ_j softmax(−β·L)_j·θ_j).
                         L is computed on a fixed eval batch via the same vmapped
                         loss used for training. See src/swarm.py.
    """
    if len(train_loaders) != n_agents:
        raise ValueError(
            f"need {n_agents} train_loaders, got {len(train_loaders)}"
        )

    if swarm_config is not None and consensus_fn is not None:
        raise ValueError("pass either consensus_fn or swarm_config, not both")
    if consensus_fn is None and swarm_config is None:
        consensus_fn = fedavg_consensus

    torch.manual_seed(seed)

    base_model, params, buffers = _stack_models(model_factory, n_agents)

    label_tok_ids = torch.tensor(
        [label_token_ids[0], label_token_ids[1]], device=device,
    )
    per_agent_loss = _make_prompt_loss(base_model, label_tok_ids, mask_token_id)
    vmapped_loss = vmap(per_agent_loss, in_dims=(0, 0, 0, 0, 0))

    rng   = torch.Generator(device=device)
    iters = [cycle(dl) for dl in train_loaders]

    hist  = FedHistory()
    bar   = tqdm(range(total_steps), desc=f"FedAvg-vmap N={n_agents} K={local_steps}")

    for step in bar:
        batches = [next(it) for it in iters]
        ids, attn, labs = _stack_agent_batches(batches, device)

        seed_step = int(torch.randint(0, 2**31 - 1, (1,)).item())

        with torch.no_grad():
            # θ → θ + ε·z
            _perturb_stacked(params, eps, seed_step, rng)
            loss_plus = vmapped_loss(params, buffers, ids, attn, labs)         # (N,)

            # θ → θ − ε·z
            _perturb_stacked(params, -2.0 * eps, seed_step, rng)
            loss_minus = vmapped_loss(params, buffers, ids, attn, labs)        # (N,)

            # θ → θ (restore)
            _perturb_stacked(params, eps, seed_step, rng)

            projected_grad = (loss_plus - loss_minus) / (2.0 * eps)            # (N,)
            _apply_mezo_update(params, projected_grad, lr, seed_step, rng)

            if (step + 1) % local_steps == 0:
                if swarm_config is not None:
                    from src.swarm import swarm_consensus_step
                    sw_ids, sw_attn, sw_labs = swarm_config.eval_batch
                    n = ids.shape[0]
                    sw_ids_n  = sw_ids.unsqueeze(0).expand(n, *sw_ids.shape)
                    sw_attn_n = sw_attn.unsqueeze(0).expand(n, *sw_attn.shape)
                    sw_labs_n = sw_labs.unsqueeze(0).expand(n, *sw_labs.shape)
                    eval_losses = vmapped_loss(
                        params, buffers, sw_ids_n, sw_attn_n, sw_labs_n,
                    )                                                         # (N,)
                    if track_consensus:
                        from src.consensus import consensus_distance
                        d_before = consensus_distance(params)
                        w = swarm_consensus_step(
                            params, eval_losses,
                            swarm_config.alpha, swarm_config.beta,
                        )
                        d_after = consensus_distance(params)
                        hist.consensus_round.append(step + 1)
                        hist.consensus_dist_before.append(d_before)
                        hist.consensus_dist_after.append(d_after)
                    else:
                        w = swarm_consensus_step(
                            params, eval_losses,
                            swarm_config.alpha, swarm_config.beta,
                        )
                    hist.swarm_weights.append(w.detach().cpu().tolist())
                    hist.swarm_eval_losses.append(eval_losses.detach().cpu().tolist())
                elif track_consensus:
                    from src.consensus import consensus_distance
                    d_before = consensus_distance(params)
                    consensus_fn(params)
                    d_after = consensus_distance(params)
                    hist.consensus_round.append(step + 1)
                    hist.consensus_dist_before.append(d_before)
                    hist.consensus_dist_after.append(d_after)
                else:
                    consensus_fn(params)

        if step % log_every == 0:
            losses_cpu = loss_plus.detach().cpu().tolist()
            mean_loss  = sum(losses_cpu) / n_agents
            hist.step.append(step)
            hist.train_loss.append(mean_loss)
            hist.per_agent_train_loss.append(losses_cpu)
            bar.set_postfix(loss=f"{mean_loss:.4f}")

        if (step + 1) % eval_every == 0 or step == total_steps - 1:
            agent0_params  = {k: v[0] for k, v in params.items()}
            agent0_buffers = {k: v[0] for k, v in buffers.items()}
            ev_loss, ev_acc = _eval_one_agent(
                base_model, agent0_params, agent0_buffers, val_loader,
                label_tok_ids, mask_token_id, device,
            )
            hist.eval_step.append(step + 1)
            hist.eval_loss.append(ev_loss)
            hist.eval_acc.append(ev_acc)
            bar.set_postfix(
                loss=f"{loss_plus.mean().item():.4f}",
                val_acc=f"{ev_acc:.4f}",
            )

    return hist
