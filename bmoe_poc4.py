"""
B-MoE POC — step 2.3: critic-guided routing (sufficiency judge + few-shot demos) on MPS
=======================================================================================
Step 2.2 showed a parametric "next-expert" head can't infer a multi-step chain, because a
demonstration shows the net input->output MAP, not its DECOMPOSITION (many chains give the
same map). The fix the user proposed: a CRITIC that judges "is the goal reached yet?".

This script realizes it. The model owns modular skills (phase A) and, at inference, ROUTES
ITSELF with no router and no expert-id token:

  task = few-shot demos (X_demo -> Y_demo)              # content only
  the model PROPOSES chains of its own primitives, shortest first;
  a SUFFICIENCY CRITIC checks each on the demos: does chain(X_demo) reproduce Y_demo?
  the shortest chain the critic deems SUFFICIENT is applied to the query.

The critic = "goal reached?" judged against the demonstrations. Here it is exact token
match on the demos (robust in a toy); at LLM scale it becomes a learned/semantic judge of
"is this answer complete/correct" — which the user notes likely needs LLM-scale experts.
This loop resolves the map<->decomposition ambiguity by *composing primitives and verifying
against the goal*. Runs on the M4 GPU (MPS).
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bmoe_poc import DEVICE, Skill, apply_skill, compose, rand_batch, rules

ATOMS = ["math", "history", "geography"]
N = len(ATOMS)
K_MAX = 3
D_DEMOS = 4


def label(t):
    return "+".join(a[:4] for a in t)


def partials(chain_atoms, X, V, R):
    outs, cur = [], X
    for a in chain_atoms:
        cur = apply_skill(a, cur, V, R)
        outs.append(cur.reshape(-1))
    return outs


class LoopMoE(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, max_len=64):
        super().__init__()
        self.vocab, self.d = vocab, d
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_ctx = nn.LayerNorm(d)
        self.experts = nn.ModuleList([Skill(d) for _ in range(N)])
        self.norm_step = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def _ctx(self, w):
        S = w.size(1)
        m = torch.triu(torch.ones(S, S, device=w.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(w, w, w, attn_mask=m)
        return self.norm_ctx(w + a)

    def step(self, w, e):
        S = w.size(1)
        w = self.norm_step(w + self.experts[e](self._ctx(w)))
        logit = self.head(w)
        w = F.softmax(logit, -1) @ self.embed.weight + self.pos[:, :S, :]
        return w, logit

    def forced(self, X, chain):
        S = X.size(1)
        w = self.embed(X) + self.pos[:, :S, :]
        outs = []
        for e in chain:
            w, logit = self.step(w, e)
            outs.append(logit.reshape(-1, self.vocab))
        return outs

    @torch.no_grad()
    def run_chain(self, X, chain):
        """Apply a chain; return the final-step token predictions (B, S)."""
        S = X.size(1)
        w = self.embed(X) + self.pos[:, :S, :]
        logit = self.head(w)
        for e in chain:
            w, logit = self.step(w, e)
        return logit.argmax(-1)


@torch.no_grad()
def critic_route(model, Xd, Yd):
    """Propose chains shortest-first; the sufficiency critic accepts the first that
    reproduces the demos (chain(Xd) == Yd). Returns that chain (or None)."""
    for length in range(1, K_MAX + 1):
        for chain in itertools.product(range(N), repeat=length):
            if (model.run_chain(Xd, list(chain)) == Yd).float().mean().item() > 0.999:
                return list(chain)
    return None


def main():
    V, Dm, S, BS = 50, 128, 16, 64
    R = rules(V)
    PA, LR = 3000, 1e-3

    comps = ([[a] for a in ATOMS]
             + [list(p) for p in itertools.permutations(ATOMS, 2)]
             + [list(p) for p in itertools.permutations(ATOMS, 3)])
    HELD = [["math", "geography"], ["math", "history", "geography"]]
    TRAIN = [c for c in comps if c not in HELD]
    a2i = {a: i for i, a in enumerate(ATOMS)}

    print("=" * 72)
    print(f"  POC step 2.3 — critic-guided routing (sufficiency judge) on {str(DEVICE).upper()}")
    print("=" * 72)
    print(f"  {len(TRAIN)} train compositions, hold out: "
          f"{', '.join('['+label(t)+']' for t in HELD)}")
    print("-" * 72)

    torch.manual_seed(0)
    model = LoopMoE(V, Dm).to(DEVICE)

    # ── Phase A: modular skills via forced chains + deep supervision ────────────
    print("\nPhase A — modular skills (forced chains)...")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(PA):
        t = TRAIN[np.random.randint(len(TRAIN))]
        X = rand_batch(BS, S, V)
        outs = model.forced(X, [a2i[a] for a in t])
        tg = partials(t, X, V, R)
        loss = sum(F.cross_entropy(outs[j], tg[j]) for j in range(len(t)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    @torch.no_grad()
    def skill_acc(a):
        X = rand_batch(256, S, V)
        return (model.run_chain(X, [a2i[a]]) == apply_skill(a, X, V, R)).float().mean().item()
    print("  skills:", {a: round(skill_acc(a), 3) for a in ATOMS})

    # ── Inference: the model routes itself via the sufficiency critic on demos ──
    @torch.no_grad()
    def evaluate(t):
        Xd = rand_batch(D_DEMOS, S, V)
        Yd = compose(t, Xd, V, R)
        chain = critic_route(model, Xd, Yd)                  # autonomous, from demos only
        Xq = rand_batch(256, S, V)
        Yq = compose(t, Xq, V, R)
        if chain is None:
            return 0.0, "(none)"
        acc = (model.run_chain(Xq, chain) == Yq).float().mean().item()
        return acc, "+".join(ATOMS[c][:4] for c in chain)

    print("\n" + "=" * 72)
    print("  RESULT — the model proposes chains; the critic verifies them on the demos")
    print("=" * 72)
    print(f"\n  {'Task':<22} {'split':<12} {'acc':>6}  routed chain (self-chosen)")
    print("  " + "-" * 58)
    for t in (TRAIN[:4] + HELD):
        a, em = evaluate(t)
        split = "ZERO-SHOT" if t in HELD else "seen"
        ok = "✓" if em == label(t) else f"want {label(t)}"
        print(f"  [{label(t):<19}] {split:<12} {a:>6.3f}  {em}  {ok}")

    zs_acc = np.mean([evaluate(t)[0] for t in HELD])
    zs_chain_ok = all(evaluate(t)[1] == label(t) for t in HELD)
    print("\n" + "-" * 72)
    print(f"  zero-shot accuracy = {zs_acc:.3f}   chains correct = {zs_chain_ok}")
    if zs_acc > 0.9 and zs_chain_ok:
        print("  CONCLUSIVE: with a sufficiency critic (goal-reached judge), the model routes")
        print("  ITSELF from few-shot demos — no router, no expert token — and generalizes to")
        print("  unseen compositions. The map<->decomposition ambiguity is resolved by")
        print("  composing primitives and verifying against the goal. Next: scale to an LLM,")
        print("  where the sufficiency judge becomes learned/semantic.")
    else:
        print("  Not fully conclusive — inspect skills / critic threshold.")
    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
