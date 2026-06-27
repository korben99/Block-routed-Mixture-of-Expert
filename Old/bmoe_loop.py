"""
B-MoE — the loop architecture (toward a POC)
============================================
Journey so far (see README):
  vanilla     -> compositions not composable (no path works).
  Lever 1     -> per-bloc supervised routing; experts co-adapt to predecessor, still fails.
  Lever 2     -> deep supervision (canonical interface); the PAIR composes (0.64) but the
                 TRIPLE fails, because experts are SEPARATE per bloc and E_geo was never
                 trained at bloc 2 (a depth-coverage gap).

Insight (from the depth-abstraction discussion): tying experts to fixed layer depths is the
problem. Instead, separate the two roles:

    input -> [CORE: a few transformer layers]  ->  h   (a STATIONARY working representation)
          -> LOOP k times:   h = norm(h + E_sigma(h))   (SHARED, reusable expert skills)
          -> [RENDER: head]  -> output
    + deep supervision: head(h) after each loop step decodes the partial composition.

The core absorbs the low->high abstraction lift ONCE; the loop then applies skills at a
single, stationary abstraction level, so one shared expert is the SAME function at every
step (no depth-abstraction conflict). This is Universal-Transformer recurrence + B-MoE
routing. Experts are reusable by construction, so the depth-coverage gap disappears and the
TRIPLE should finally compose.

This is a toy proof; the POC will add learned routing/halting and real tokens.
Run: `python bmoe_loop.py`.
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toyBMoE import (
    Expert, build_dataset, label, make_domain_rules, make_pairs, sample_batch,
)
from bmoe_lever2 import EXPERT, N_EXPERTS, expert_path, partial_targets, task_atoms


class CoreLayer(nn.Module):
    """A standard causal transformer layer used in the (shared-free) core encoder."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = Expert(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h):
        S = h.size(1)
        mask = torch.triu(torch.ones(S, S, device=h.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h


class LoopBMoE(nn.Module):
    """Core encoder + a loop of SHARED expert skills + render head."""

    def __init__(self, vocab, d_model=64, n_heads=4, n_experts=N_EXPERTS,
                 core_layers=2, max_len=64):
        super().__init__()
        self.vocab = vocab
        self.token_embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.core = nn.ModuleList([CoreLayer(d_model, n_heads) for _ in range(core_layers)])
        # ONE shared expert pool, reused at every loop step (iteration-agnostic skills)
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])
        self.step_norm = nn.LayerNorm(d_model)   # shared across iterations on purpose
        self.head = nn.Linear(d_model, vocab)

    def encode(self, x):
        h = self.token_embed(x) + self.pos_embed[:, : x.size(1), :]
        for layer in self.core:
            h = layer(h)
        return h

    def forward(self, x, path, return_step_logits=False, reground=False):
        """path: sequence of expert indices, one per loop step (forced routing).

        reground=True re-grounds the stream to TOKEN-EMBEDDING space after each step
        (soft re-embedding of the decoded distribution). This makes every expert receive
        a clean, predecessor-independent token embedding -> a true canonical interface.
        """
        h = self.encode(x)
        step_logits = []
        last = None
        for i, e in enumerate(path):
            h = self.step_norm(h + self.experts[e](h))   # apply skill
            logit = self.head(h)                          # (B, S, vocab)
            last = logit
            if return_step_logits:
                step_logits.append(logit.reshape(-1, self.vocab))
            if reground and i < len(path) - 1:
                p = F.softmax(logit, dim=-1)              # (B, S, vocab)
                h = p @ self.token_embed.weight           # soft re-embedding -> (B, S, d)
        logits = last.reshape(-1, self.vocab)
        if return_step_logits:
            return logits, step_logits
        return logits


@torch.no_grad()
def path_acc(model, X, Y, path, reground=False):
    model.eval()
    logits = model(X, path, reground=reground)
    return (logits.argmax(-1) == Y.reshape(-1)).float().mean().item()


def run(reground, rules, train, test, train_tasks, heldout, seen, cfg):
    """Train a loop model (continuous or re-grounded interface) and report composition."""
    VOCAB, D_MODEL, N_HEADS, CORE_LAYERS, LOOP_K, SEQ_LEN, BATCH, N_STEPS, LR = cfg

    torch.manual_seed(42)
    np.random.seed(42)
    model = LoopBMoE(VOCAB, D_MODEL, N_HEADS, N_EXPERTS, CORE_LAYERS, max_len=SEQ_LEN)
    atoms = {label(t): task_atoms(t, LOOP_K) for t in train_tasks}
    paths = {k: expert_path(a) for k, a in atoms.items()}
    train_labels = list(paths)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(N_STEPS):
        name = train_labels[torch.randint(0, len(train_labels), (1,)).item()]
        Xb, Yb = sample_batch(*train[name], BATCH)
        model.train()
        _, step_logits = model(Xb, paths[name], return_step_logits=True, reground=reground)
        tgts = partial_targets(Xb, atoms[name], VOCAB, rules)
        loss = sum(F.cross_entropy(step_logits[j], tgts[j]) for j in range(LOOP_K))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    def acc(task):
        return path_acc(model, *test[label(task)],
                        expert_path(task_atoms(task, LOOP_K)), reground=reground)

    tag = "RE-GROUNDED (token interface)" if reground else "CONTINUOUS (deep-sup only)"
    print(f"\n  {tag}")
    print(f"    train tasks (forced path) = "
          f"{np.mean([acc(t) for t in train_tasks]):.3f}")
    for t in heldout:
        split = "ZERO-SHOT pair" if len(t) == 2 else "ZERO-SHOT triple"
        print(f"    [{label(t):<17}] {split:<16} forced-path acc = {acc(t):.3f}")

    # Auto-discovery: does a likelihood-free path search recover the correct chain?
    for t in heldout:
        X, Y = test[label(t)]
        scored = sorted(((path_acc(model, X, Y, p, reground=reground), p)
                         for p in itertools.product(range(N_EXPERTS), repeat=LOOP_K)),
                        reverse=True)
        correct = expert_path(task_atoms(t, LOOP_K))
        rank = [p for _, p in scored].index(correct) + 1
        best_a, best_p = scored[0]
        print(f"      search [{label(t):<15}] best {' > '.join('E'+str(e) for e in best_p)}"
              f"={best_a:.2f}  correct-path rank {rank}/{len(scored)}")
    return acc(heldout[-1])  # triple acc


def main():
    PURE = [[a] for a in ["math", "history", "geography"]]
    SEEN = [["math", "history"], ["history", "geography"]]
    HELDOUT = [["math", "geography"], ["math", "history", "geography"]]
    TRAIN_TASKS = PURE + SEEN
    ALL_TASKS = TRAIN_TASKS + HELDOUT

    VOCAB, D_MODEL, N_HEADS = 50, 64, 4
    CORE_LAYERS, LOOP_K = 2, 3
    SEQ_LEN, BATCH = 30, 32
    N_TRAIN, N_TEST = 300, 150
    N_STEPS, LR = 1500, 1e-3
    cfg = (VOCAB, D_MODEL, N_HEADS, CORE_LAYERS, LOOP_K, SEQ_LEN, BATCH, N_STEPS, LR)

    print("=" * 72)
    print("  B-MoE loop — shared-expert loop: continuous vs re-grounded interface")
    print("=" * 72)
    print(f"  Experts (shared, reused every step): "
          f"{', '.join(f'E{EXPERT[a]}={a}' for a in ['math','history','geography'])}, "
          f"E{EXPERT['identity']}=identity")
    print(f"  Core layers = {CORE_LAYERS} | loop steps = {LOOP_K}")
    print(f"  Trained: pures + {', '.join('[' + label(t) + ']' for t in SEEN)}")
    print(f"  Held out (zero-shot): {', '.join('[' + label(t) + ']' for t in HELDOUT)}")

    rules = make_domain_rules(VOCAB, seed=0)
    train, test = {}, {}
    for t in TRAIN_TASKS:
        train[label(t)] = make_pairs(build_dataset(N_TRAIN, SEQ_LEN, t, VOCAB, rules))
    for t in ALL_TASKS:
        test[label(t)] = make_pairs(build_dataset(N_TEST, SEQ_LEN, t, VOCAB, rules))

    print("\n" + "=" * 72)
    print("  RESULT — forced correct path, continuous vs re-grounded")
    print("=" * 72)
    cont_triple = run(False, rules, train, test, TRAIN_TASKS, HELDOUT, SEEN, cfg)
    reg_triple = run(True, rules, train, test, TRAIN_TASKS, HELDOUT, SEEN, cfg)

    print("\n" + "-" * 72)
    if reg_triple > 0.7 and reg_triple > cont_triple + 0.3:
        print(f"  RE-GROUNDING CRACKS COMPOSITION: triple {cont_triple:.2f} -> {reg_triple:.2f}.")
        print("  A token-grounded interface makes shared experts reusable token->token maps,")
        print("  so an unseen chain (incl. the triple) composes. This is the POC architecture:")
        print("  core (context) + loop of reusable skills with re-grounding + render head.")
    else:
        print(f"  Re-grounded triple = {reg_triple:.2f} (continuous {cont_triple:.2f}). Not solved;")
        print("  next: stronger token bottleneck / hard (argmax) re-grounding with STE.")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
