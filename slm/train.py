"""
Two-phase training (paper §5 recipe), stage 1 / exact regime.

Phase A — guided pre-specialization: each expert is trained, in isolation (forced single-step
path), to implement its primitive decoder. This is the paper's "pré-spécialisation par domaine".

Phase B — joint composition training: random compositions from the TRAIN set, padded to n_loop
with identity, are run with the FORCED correct path under per-step deep supervision (each loop
step must produce the partial decode). Simultaneously the router is SUPERVISED to predict the
forced expert from the current state — turning it into a learned proposer for inference — and
the bounded MoE aux losses (L_div / L_bal) keep the experts distinct and balanced.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import (EXPERT_ID, N_EXPERTS, apply_skill, forced_path, pad_chain,
                   partials, rand_payload)
from .model import BlocRoutedMoESLM

ID2NAME = {v: k for k, v in EXPERT_ID.items()}


def pre_specialize(model: BlocRoutedMoESLM, P, S=24, bs=64, steps=1500, lr=2e-3):
    cfg = model.cfg
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # context-dependent experts (transpose) need the shared attention to learn a neighbour
    # gather, which the pointwise experts pull toward identity — so oversample them.
    schedule = []
    for e in range(N_EXPERTS):
        schedule += [e] * (3 if ID2NAME[e] == "shift" else 1)
    for step in range(steps):
        e = schedule[step % len(schedule)]
        name = ID2NAME[e]
        X = rand_payload(bs, S)
        target = apply_skill(name, X, P)
        trace = model(X, n_loop=1, forced_experts=[e])
        loss = F.cross_entropy(trace.step_logits[0].reshape(-1, 256), target.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    @torch.no_grad()
    def acc(name):
        X = rand_payload(256, S)
        out = model(X, n_loop=1, forced_experts=[EXPERT_ID[name]]).step_logits[0].argmax(-1)
        return (out == apply_skill(name, X, P)).float().mean().item()
    return {n: round(acc(n), 3) for n in EXPERT_ID}


def joint_train(model: BlocRoutedMoESLM, P, train_tasks, S=24, bs=64, steps=2500, lr=1e-3,
                router_weight=0.5):
    cfg = model.cfg
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        task = train_tasks[np.random.randint(len(train_tasks))]
        padded = pad_chain(task, cfg.n_loop)
        path = forced_path(padded)
        X = rand_payload(bs, S)
        tg = partials(padded, X, P)

        trace = model(X, forced_experts=path)
        ce = sum(F.cross_entropy(trace.step_logits[i].reshape(-1, 256), tg[i].reshape(-1))
                 for i in range(cfg.n_loop)) / cfg.n_loop

        # supervise the router (proposer): each bloc's probs should predict its step's expert
        rce = trace.probs[0].new_zeros(())
        for li in range(cfg.n_loop):
            tgt = torch.full((bs,), path[li], device=X.device, dtype=torch.long)
            for b in range(cfg.n_blocs):
                # probs are already a distribution -> NLL on log-probs (not cross_entropy,
                # which would apply a second log_softmax)
                rce = rce + F.nll_loss(trace.probs[li * cfg.n_blocs + b].clamp_min(1e-9).log(),
                                       tgt)
        rce = rce / (cfg.n_loop * cfg.n_blocs)

        l_bal, l_div = model.aux_losses(trace, model.embed(X) + model.pos[:, :S, :])
        loss = ce + router_weight * rce + l_bal + l_div
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model
