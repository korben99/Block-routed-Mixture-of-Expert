"""
Autonomous routing at inference — the paper's propose-and-verify (§6), at SLM scale with a
LEARNED proposer instead of exhaustive search.

The task is given only by demos (obfuscated -> revealed) pairs. The model:
  * PROPOSES the next skill with its trained router (a learned next-skill predictor),
  * APPLIES candidate chains (beam, shortest-first) with its own experts,
  * a sufficiency CRITIC verifies each chain on the demos (decode == revealed), and the
    shortest verified chain is returned — and applied to unknown traffic.

`exhaustive_route` is the brute-force baseline; comparing `explored` counts shows the router
prunes the search.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import torch

from .data import ATOMS, EXPERT_ID
from .model import BlocRoutedMoESLM

ID2NAME = {v: k for k, v in EXPERT_ID.items()}


@torch.no_grad()
def apply_chain(model: BlocRoutedMoESLM, X: torch.Tensor, chain: List[str]) -> torch.Tensor:
    """Decode X by forcing the model through `chain` (one skill per loop step)."""
    if not chain:
        return X
    path = [EXPERT_ID[a] for a in chain]
    trace = model(X, n_loop=len(path), forced_experts=path)
    return trace.step_logits[-1].argmax(-1)


@torch.no_grad()
def _verifies(model, Xd, Yd, chain, thr=0.999) -> bool:
    return (apply_chain(model, Xd, chain) == Yd).float().mean().item() >= thr


@torch.no_grad()
def autonomous_route(model, Xd, Yd, beam=2, max_depth=3, thr=0.999
                     ) -> Tuple[Optional[List[str]], int]:
    """Router-guided, shortest-first beam search; sufficiency critic halts on the demos."""
    if _verifies(model, Xd, Yd, [], thr):
        return [], 0
    frontier = [([], Xd)]
    explored = 0
    for _ in range(max_depth):
        nxt = []
        for chain, state in frontier:
            probs = model.propose(state)                       # (N,)
            order = probs.argsort(descending=True).tolist()
            picks = [e for e in order if ID2NAME[e] != "identity"][:beam]
            for e in picks:
                new_chain = chain + [ID2NAME[e]]
                new_state = apply_chain(model, Xd, new_chain)
                explored += 1
                if (new_state == Yd).float().mean().item() >= thr:
                    return new_chain, explored
                nxt.append((new_chain, new_state))
        frontier = nxt
    return None, explored


@torch.no_grad()
def exhaustive_route(model, Xd, Yd, max_depth=3, thr=0.999
                     ) -> Tuple[Optional[List[str]], int]:
    """Brute-force baseline: every chain shortest-first, verified on the demos."""
    explored = 0
    for length in range(1, max_depth + 1):
        for combo in itertools.product(ATOMS, repeat=length):
            explored += 1
            if _verifies(model, Xd, Yd, list(combo), thr):
                return list(combo), explored
    return None, explored
