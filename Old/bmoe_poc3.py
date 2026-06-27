"""
B-MoE POC — step 2.2: decentralized expert-emitted routing from few-shot demos (MPS)
====================================================================================
No central router, and crucially NO predefined expert-id token. The task is communicated
purely by CONTENT: a few demonstrations (X->Y) of the wanted transformation. Each expert
owns a "next" head that, from its state + the encoded demos + descriptors of the OTHER
experts, decides which expert runs next (or STOP). The experts hand off to each other.

  task signal : few-shot demos [(X1->Y1), ..., (Xd->Yd)]  (content, no expert ids)
  routing     : expert-emitted -> attend over learned expert descriptors -> {experts, STOP}
  loop        : core context + shared skill + token re-grounding (validated in step 1)

Curriculum:
  Phase A  forced chains + deep supervision  -> modular reusable skills.
  Phase B  freeze skills; learn the next/STOP heads. The correct chain is SUPERVISED at
           train time, but at inference it is EMITTED by the experts from the demos only.

Decisive test: few-shot demos of a composition NEVER trained -> do the experts emit the
right chain and solve the query? Runs on the M4 GPU (MPS).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bmoe_poc import DEVICE, Skill, apply_skill, compose, rand_batch, rules

ATOMS = ["math", "history", "geography"]
N = len(ATOMS)          # experts (no identity; STOP ends the chain)
STOP = N
K_MAX = 4               # safety cap on chain length
D_DEMOS = 4


def label(t):
    return "+".join(a[:4] for a in t)


def partials(chain_atoms, X, V, R):
    outs, cur = [], X
    for a in chain_atoms:
        cur = apply_skill(a, cur, V, R)
        outs.append(cur.reshape(-1))
    return outs


class FewShotMoE(nn.Module):
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
        # ── decentralized routing (learned in phase B) ──
        self.demo_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.expert_desc = nn.Parameter(torch.randn(N, d) * 0.02)   # knowledge of others
        self.start_desc = nn.Parameter(torch.randn(d) * 0.02)
        self.stop_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.nq = nn.Linear(3 * d, d)

    def ctx(self, w):
        S = w.size(1)
        m = torch.triu(torch.ones(S, S, device=w.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(w, w, w, attn_mask=m)
        return self.norm_ctx(w + a)

    def step(self, w, e):
        S = w.size(1)
        w = self.norm_step(w + self.experts[e](self.ctx(w)))
        logit = self.head(w)
        w = F.softmax(logit, -1) @ self.embed.weight + self.pos[:, :S, :]
        return w, logit

    def encode_demos(self, Xd, Yd):
        ex, ey = self.embed(Xd), self.embed(Yd)                 # (Dd,S,d)
        return self.demo_mlp(torch.cat([ex, ey], -1)).mean((0, 1))  # (d,)

    def next_logits(self, w, tau, desc_cur):
        q = self.nq(torch.cat([w.mean(1), tau, desc_cur], -1))  # (B,d)
        keys = torch.cat([self.expert_desc, self.stop_emb[None]], 0)  # (N+1,d)
        return q @ keys.t()                                      # (B,N+1)

    def forced(self, Xq, chain):
        """Apply a known chain (phase A) -> per-step logits for deep supervision."""
        S = Xq.size(1)
        w = self.embed(Xq) + self.pos[:, :S, :]
        outs = []
        for e in chain:
            w, logit = self.step(w, e)
            outs.append(logit.reshape(-1, self.vocab))
        return outs

    def route_train(self, Xq, tau, chain):
        """Teacher-forced chain; returns routing logits at each decision (incl. STOP)."""
        B, S = Xq.shape
        w = self.embed(Xq) + self.pos[:, :S, :]
        desc = self.start_desc.expand(B, self.d)
        rlogits, targets = [], []
        for j in range(len(chain) + 1):
            rlogits.append(self.next_logits(w, tau, desc))
            tgt = chain[j] if j < len(chain) else STOP
            targets.append(tgt)
            if tgt == STOP:
                break
            w, _ = self.step(w, chain[j])
            desc = self.expert_desc[chain[j]].expand(B, self.d)
        return rlogits, targets

    @torch.no_grad()
    def emit(self, Xq, tau):
        """Inference: experts emit the chain autonomously (argmax), then we read the output."""
        B, S = Xq.shape
        w = self.embed(Xq) + self.pos[:, :S, :]
        desc = self.start_desc.expand(B, self.d)
        chain, logit = [], self.head(w)
        for _ in range(K_MAX):
            nxt = int(self.next_logits(w, tau, desc).mean(0).argmax())  # batch shares task
            if nxt == STOP:
                break
            chain.append(nxt)
            w, logit = self.step(w, nxt)
            desc = self.expert_desc[nxt].expand(B, self.d)
        return chain, logit


def main():
    V, Dm, S, BS = 50, 128, 16, 64
    R = rules(V)
    PA, PB, LR = 3000, 3000, 1e-3

    import itertools
    comps = ([[a] for a in ATOMS]
             + [list(p) for p in itertools.permutations(ATOMS, 2)]
             + [list(p) for p in itertools.permutations(ATOMS, 3)])
    HELD = [["math", "geography"], ["math", "history", "geography"]]
    TRAIN = [c for c in comps if c not in HELD]
    a2i = {a: i for i, a in enumerate(ATOMS)}

    def chain_of(t):
        return [a2i[a] for a in t]

    def demos_for(t, n=D_DEMOS):
        Xd = rand_batch(n, S, V)
        return Xd, compose(t, Xd, V, R)

    print("=" * 72)
    print(f"  POC step 2.2 — decentralized expert-emitted routing, few-shot, on {str(DEVICE).upper()}")
    print("=" * 72)
    print(f"  {len(TRAIN)} train compositions, hold out: "
          f"{', '.join('['+label(t)+']' for t in HELD)}")
    print("-" * 72)

    torch.manual_seed(0)
    model = FewShotMoE(V, Dm).to(DEVICE)

    # ── Phase A: modular skills via forced chains + deep supervision ────────────
    print("\nPhase A — modular skills (forced chains)...")
    optA = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(PA):
        t = TRAIN[np.random.randint(len(TRAIN))]
        X = rand_batch(BS, S, V)
        outs = model.forced(X, chain_of(t))
        tg = partials(t, X, V, R)
        loss = sum(F.cross_entropy(outs[j], tg[j]) for j in range(len(t)))
        optA.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optA.step()

    @torch.no_grad()
    def skill_acc(a):
        X = rand_batch(256, S, V)
        out = model.forced(X, [a2i[a]])[-1]
        return (out.argmax(-1) == apply_skill(a, X, V, R).reshape(-1)).float().mean().item()
    print("  skills:", {a: round(skill_acc(a), 3) for a in ATOMS})

    # ── Phase B: freeze skills, learn the next/STOP heads ───────────────────────
    print("\nPhase B — learn decentralized next/STOP heads (chain supervised)...")
    route_params = [model.demo_mlp, model.nq]
    for p in model.parameters():
        p.requires_grad_(False)
    for m in route_params:
        for p in m.parameters():
            p.requires_grad_(True)
    for p in (model.expert_desc, model.start_desc, model.stop_emb):
        p.requires_grad_(True)
    optB = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for _ in range(PB):
        t = TRAIN[np.random.randint(len(TRAIN))]
        X = rand_batch(BS, S, V)
        Xd, Yd = demos_for(t)
        tau = model.encode_demos(Xd, Yd).expand(BS, -1)
        rlogits, targets = model.route_train(X, tau, chain_of(t))
        tgt = torch.tensor(targets, device=DEVICE)
        loss = sum(F.cross_entropy(rlogits[j], tgt[j].expand(BS)) for j in range(len(targets)))
        optB.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optB.step()

    # ── Evaluation: experts emit the chain from demos (autonomous) ──────────────
    @torch.no_grad()
    def evaluate(t):
        X = rand_batch(256, S, V)
        Y = compose(t, X, V, R)
        Xd, Yd = demos_for(t)
        tau = model.encode_demos(Xd, Yd).expand(256, -1)
        chain, logit = model.emit(X, tau)
        acc = (logit.argmax(-1) == Y).float().mean().item()
        emitted = "+".join(ATOMS[c][:4] for c in chain) or "(none)"
        return acc, emitted

    print("\n" + "=" * 72)
    print("  RESULT — experts emit the chain from few-shot demos")
    print("=" * 72)
    print(f"\n  {'Task':<22} {'split':<12} {'acc':>6}  emitted chain")
    print("  " + "-" * 56)
    for t in (TRAIN[:4] + HELD):
        a, em = evaluate(t)
        split = "ZERO-SHOT" if t in HELD else "seen"
        want = label(t)
        ok = "✓" if em == want else f"want {want}"
        print(f"  [{label(t):<19}] {split:<12} {a:>6.3f}  {em}  {ok}")

    zs = np.mean([evaluate(t)[0] for t in HELD])
    print("\n" + "-" * 72)
    print(f"  zero-shot accuracy = {zs:.3f}")
    if zs > 0.7:
        print("  Experts route THEMSELVES from content (few-shot demos), no router, no expert")
        print("  token — and it generalizes to unseen compositions.")
    else:
        print("  Partial: the emitted chain doesn't yet generalize — inspect demo encoding.")
    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
