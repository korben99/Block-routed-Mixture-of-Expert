"""
B-MoE POC — step 2.1: learned routing + halting (PonderNet) on MPS
==================================================================
Step 1 validated the dual-stream loop with FORCED routing. Step 2.1 makes the routing and
the loop length LEARNED:

  - Routing: an instruction (the list of skills to apply, e.g. [math, delta]) is given; a
    learned router reads it and, at each loop step, selects which expert to apply. Trained
    with a straight-through estimator; experts are frozen (curriculum) so this isolates
    "can a learned router discover and execute the right skill chain".
  - Halting: a PonderNet head emits a per-step halting probability; the loss is the expected
    task loss over the halting distribution + a KL to a geometric prior. The model learns to
    STOP once the instruction is consumed.

Decisive test (generalization): the model is trained only on compositions of length 1 and 2,
then evaluated zero-shot on an UNSEEN length-2 pair AND the length-3 triple — so routing must
generalize to an unseen skill combination and halting to an unseen loop length.

Curriculum:
  Phase A  forced routing + deep supervision  -> modular, reusable skills (as in step 1).
  Phase B  freeze skills; learn router + halting (PonderNet) with token re-grounding.

Runs on the M4 GPU (MPS).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bmoe_poc import (
    ATOMS, DEVICE, EXPERT, N_EXPERTS, Skill, apply_skill, compose, label,
    partials, rand_batch, rules, task_atoms,
)

K_MAX = 3
IDENT = EXPERT["identity"]


def instr_ids(task):
    """Instruction = expert ids of the skills, padded with identity (acts as PAD/halt)."""
    return [EXPERT[a] for a in task] + [IDENT] * (K_MAX - len(task))


class RoutedLoop(nn.Module):
    def __init__(self, vocab, d=128, n_heads=4, max_len=64):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_ctx = nn.LayerNorm(d)
        self.experts = nn.ModuleList([Skill(d) for _ in range(N_EXPERTS)])
        self.norm_step = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        # routing + halting (learned in phase B)
        self.instr_embed = nn.Embedding(N_EXPERTS, d)
        self.instr_pos = nn.Parameter(torch.randn(1, K_MAX, d) * 0.02)
        self.step_embed = nn.Embedding(K_MAX, d)
        self.router = nn.Linear(d, N_EXPERTS)
        self.halt = nn.Linear(d, 1)
        self.d = d

    def _ctx(self, w):
        S = w.size(1)
        mask = torch.triu(torch.ones(S, S, device=w.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(w, w, w, attn_mask=mask)
        return self.norm_ctx(w + a)

    def _apply_experts(self, w, ctx, weights):
        # weights: (B, N_EXPERTS) soft/straight-through selection
        eo = torch.stack([e(ctx) for e in self.experts], dim=-1)  # (B,S,d,N)
        z = (eo * weights[:, None, None, :]).sum(-1)
        return self.norm_step(w + z)

    def forward_forced(self, X, path, return_steps=True):
        S = X.size(1)
        w = self.embed(X) + self.pos[:, :S, :]
        steps = []
        for i, e in enumerate(path):
            ctx = self._ctx(w)
            oneh = F.one_hot(torch.tensor(e, device=X.device), N_EXPERTS).float()
            w = self._apply_experts(w, ctx, oneh.expand(X.size(0), -1))
            logit = self.head(w)
            steps.append(logit.reshape(-1, self.vocab))
            if i < len(path) - 1:
                w = F.softmax(logit, -1) @ self.embed.weight + self.pos[:, :S, :]
        return steps

    def forward_routed(self, X, instr):
        """instr: (B, K_MAX) instruction tokens. A learned router maps each instruction
        token to an expert; a learned halt head decides when to stop. Returns per-step
        logits, halting probs, and executed expert indices."""
        B, S = X.shape
        w = self.embed(X) + self.pos[:, :S, :]
        logits_per_step, lambdas, idxs = [], [], []
        for n in range(K_MAX):
            ctx = self._ctx(w)
            emb = self.instr_embed(instr[:, n])                        # (B,d) step-n instruction
            rp = self.router(emb).softmax(-1)                          # (B,N) expert distribution
            st = F.one_hot(rp.argmax(-1), N_EXPERTS).float() + rp - rp.detach()
            idxs.append(rp.argmax(-1))
            w = self._apply_experts(w, ctx, st)
            logit = self.head(w)
            logits_per_step.append(logit)
            lambdas.append(torch.sigmoid(self.halt(emb)).squeeze(-1))  # halt from instruction
            w = F.softmax(logit, -1) @ self.embed.weight + self.pos[:, :S, :]
        return logits_per_step, lambdas, idxs


def halting_probs(lambdas):
    """PonderNet: p_n = lambda_n * prod_{j<n}(1-lambda_j); last step takes the remainder."""
    B = lambdas[0].shape[0]
    remain = torch.ones(B, device=lambdas[0].device)
    ps = []
    for n, lam in enumerate(lambdas):
        lam = lam if n < len(lambdas) - 1 else torch.ones_like(lam)
        ps.append(remain * lam)
        remain = remain * (1 - lam)
    return torch.stack(ps, dim=1)  # (B, K)


def seq_ce(logit, Y):
    ce = F.cross_entropy(logit.reshape(-1, logit.size(-1)), Y.reshape(-1), reduction="none")
    return ce.view(Y.shape).mean(dim=1)  # (B,)


def main():
    V, D, S, BS = 50, 128, 24, 64
    R = rules(V)
    PA_STEPS, PB_STEPS, LR = 2500, 2500, 1e-3
    BETA, PRIOR = 0.01, 0.5  # PonderNet KL weight + geometric prior

    PURE = [[a] for a in ATOMS]
    SEEN2 = [["math", "history"], ["geography", "delta"], ["delta", "math"],
             ["history", "geography"]]
    SEEN3 = [["geography", "history", "delta"]]   # one triple, to cover routing at step 3
    HELD = [["math", "delta"], ["math", "history", "delta"]]   # unseen pair + unseen triple
    TRAIN = PURE + SEEN2 + SEEN3

    print("=" * 72)
    print(f"  B-MoE POC step 2.1 — learned routing + PonderNet halting on {str(DEVICE).upper()}")
    print("=" * 72)
    print(f"  Train (len 1-2): pures + {', '.join('['+label(t)+']' for t in SEEN2)}")
    print(f"  Zero-shot: [{label(HELD[0])}] (unseen pair), [{label(HELD[1])}] (unseen LENGTH 3)")
    print("-" * 72)

    torch.manual_seed(0)
    model = RoutedLoop(V, D).to(DEVICE)

    # ── Phase A: forced routing + deep supervision -> modular skills ────────────
    print("\nPhase A — forced routing (learn modular skills)...")
    optA = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(PA_STEPS):
        t = TRAIN[np.random.randint(len(TRAIN))]
        atoms = task_atoms(t, K_MAX)
        path = [EXPERT[a] for a in atoms]
        X = rand_batch(BS, S, V)
        steps = model.forward_forced(X, path)
        tg = partials(atoms, X, V, R)
        loss = sum(F.cross_entropy(steps[j], tg[j]) for j in range(K_MAX))
        optA.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optA.step()

    @torch.no_grad()
    def forced_acc(task):
        Xte = rand_batch(256, S, V)
        Y = compose(task, Xte, V, R)
        out = model.forward_forced(Xte, [EXPERT[a] for a in task_atoms(task, K_MAX)])[-1]
        return (out.argmax(-1) == Y.reshape(-1)).float().mean().item()
    print("  skills:", {a: round(forced_acc([a]), 3) for a in ATOMS})

    # ── Phase B: freeze skills, learn router + halting (PonderNet) ──────────────
    print("\nPhase B — freeze skills, learn router + halting...")
    for p in model.parameters():
        p.requires_grad_(False)
    for m in (model.router, model.halt, model.instr_embed, model.step_embed):
        for p in m.parameters():
            p.requires_grad_(True)
    model.instr_pos.requires_grad_(True)
    optB = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    prior = torch.tensor([(1 - PRIOR) ** n * PRIOR for n in range(K_MAX)], device=DEVICE)
    prior = prior / prior.sum()

    for _ in range(PB_STEPS):
        t = TRAIN[np.random.randint(len(TRAIN))]
        X = rand_batch(BS, S, V)
        Y = compose(t, X, V, R)
        instr = torch.tensor(instr_ids(t), device=DEVICE).unsqueeze(0).expand(BS, -1)
        logits, lambdas, _ = model.forward_routed(X, instr)
        p = halting_probs(lambdas)                                   # (B,K)
        rec = sum(p[:, n] * seq_ce(logits[n], Y) for n in range(K_MAX)).mean()
        kl = (p * (p.clamp_min(1e-9).log() - prior.log())).sum(1).mean()
        loss = rec + BETA * kl
        optB.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optB.step()

    # ── Evaluation: learned routing + halting, incl. zero-shot ──────────────────
    @torch.no_grad()
    def routed_eval(task):
        Xte = rand_batch(256, S, V)
        Y = compose(task, Xte, V, R)
        instr = torch.tensor(instr_ids(task), device=DEVICE).unsqueeze(0).expand(256, -1)
        logits, lambdas, idxs = model.forward_routed(Xte, instr)
        p = halting_probs(lambdas)                                   # (B,K)
        halt_step = p.argmax(1)                                      # (B,)
        stacked = torch.stack(logits, dim=1)                         # (B,K,S,V)
        chosen = stacked[torch.arange(256), halt_step]               # (B,S,V)
        acc = (chosen.argmax(-1) == Y).float().mean().item()
        path = [int(idxs[n][0]) for n in range(K_MAX)]               # executed path (seq 0)
        return acc, float(halt_step.float().mean() + 1), path

    print("\n" + "=" * 72)
    print("  RESULT — learned routing + halting")
    print("=" * 72)
    print(f"\n  {'Task':<20} {'split':<14} {'acc':>6} {'avg halt step':>14} {'routed path'}")
    print("  " + "-" * 66)
    for t in TRAIN[:3] + HELD:
        a, hs, path = routed_eval(t)
        split = ("seen len%d" % len(t)) if t in TRAIN else (
            "ZS pair" if len(t) == 2 else "ZS triple(len3)")
        want = instr_ids(t)
        ok = "✓" if path == want else f"want {want}"
        print(f"  [{label(t):<17}] {split:<14} {a:>6.3f} {hs:>14.2f}   "
              f"{path} {ok}")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
