"""
B-MoE POC — step 1: validate the dual-stream loop architecture (with context)
=============================================================================
The toy `bmoe_loop.py` cracked zero-shot composition with a loop of shared experts +
token re-grounding, but it discarded context after step 0 (fine for token-wise maps,
not for language). Step 1 validates the architecture that CARRIES CONTEXT:

    w = embed(X)                                  # working (token) stream
    loop k times:
        ctx = causal_attention(w)                 # context from the current stream
        z   = E_sigma(ctx)                         # context-aware skill
        w   = norm(w + z)                          # update working stream
        logit = head(w)                           # decode (deep supervision target)
        w   = soft_embed(logit) + pos             # RE-GROUND to token space
    output = last logit

Two streams: a context view (shared attention, recomputed each step) and a token-grounded
working stream (re-grounded each step so experts stay reusable). To prove the context
stream is load-bearing we add a CONTEXT-DEPENDENT skill `delta: Y[t]=(X[t]-X[t-1]) mod V`,
which is impossible token-wise. We then check:
  (1) skills (incl. delta) are learnable WITH context and delta collapses WITHOUT it;
  (2) held-out compositions (incl. delta) compose zero-shot.

Task framing is seq->seq transduction: input X (random tokens), target Y = compose(skills)(X).
Runs on the M4 GPU via MPS.
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ATOMS = ["math", "history", "geography", "delta"]
EXPERT = {**{a: i for i, a in enumerate(ATOMS)}, "identity": len(ATOMS)}
N_EXPERTS = len(ATOMS) + 1


def label(task):
    return "+".join(a[:4] for a in task)


def rules(vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"perm": torch.randperm(vocab, generator=g).to(DEVICE), "a": 3, "b": 7}


def apply_skill(name, X, V, R):
    """Vectorized sequence->sequence skill on a (B,S) LongTensor."""
    if name == "math":
        return (X + 5) % V
    if name == "history":
        return (R["a"] * X + R["b"]) % V
    if name == "geography":
        return R["perm"][X]
    if name == "delta":  # context-dependent: needs the previous token
        Y = X.clone()
        Y[:, 1:] = (X[:, 1:] - X[:, :-1]) % V
        return Y
    if name == "identity":
        return X
    raise ValueError(name)


def compose(task, X, V, R):
    for name in task:
        X = apply_skill(name, X, V, R)
    return X


def partials(task_atoms, X, V, R):
    """Deep-supervision targets: sequence after applying the first j skills."""
    outs, cur = [], X
    for name in task_atoms:
        cur = apply_skill(name, cur, V, R)
        outs.append(cur.reshape(-1))
    return outs


def task_atoms(task, k):
    return list(task) + ["identity"] * (k - len(task))


def expert_path(atoms):
    return tuple(EXPERT[a] for a in atoms)


class Skill(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class PoCLoop(nn.Module):
    """embed -> loop( shared attention(context) + shared skill + re-ground ) -> head."""

    def __init__(self, vocab, d=128, n_heads=4, n_experts=N_EXPERTS, max_len=64,
                 use_context=True):
        super().__init__()
        self.vocab, self.use_context = vocab, use_context
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)  # shared context view
        self.norm_ctx = nn.LayerNorm(d)
        self.experts = nn.ModuleList([Skill(d) for _ in range(n_experts)])  # shared skills
        self.norm_step = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, X, path, reground=True, return_steps=False):
        S = X.size(1)
        w = self.embed(X) + self.pos[:, :S, :]
        mask = torch.triu(torch.ones(S, S, device=X.device, dtype=torch.bool), diagonal=1)
        steps, last = [], None
        for i, e in enumerate(path):
            if self.use_context:
                a, _ = self.attn(w, w, w, attn_mask=mask)
                ctx = self.norm_ctx(w + a)
            else:
                ctx = w
            w = self.norm_step(w + self.experts[e](ctx))
            logit = self.head(w)
            last = logit
            if return_steps:
                steps.append(logit.reshape(-1, self.vocab))
            if reground and i < len(path) - 1:
                p = F.softmax(logit, dim=-1)
                w = p @ self.embed.weight + self.pos[:, :S, :]
        out = last.reshape(-1, self.vocab)
        return (out, steps) if return_steps else out


def rand_batch(bs, S, V):
    return torch.randint(0, V, (bs, S), device=DEVICE)


@torch.no_grad()
def acc(model, Xte, task, V, R, k, reground=True):
    Y = compose(task, Xte, V, R)
    out = model(Xte, expert_path(task_atoms(task, k)), reground=reground)
    return (out.argmax(-1) == Y.reshape(-1)).float().mean().item()


def train_model(use_context, train_tasks, V, R, k, d, n_heads, S, bs, n_steps, lr, max_len):
    torch.manual_seed(0)
    model = PoCLoop(V, d, n_heads, N_EXPERTS, max_len, use_context=use_context).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    atoms = {label(t): task_atoms(t, k) for t in train_tasks}
    paths = {kk: expert_path(a) for kk, a in atoms.items()}
    labels = list(paths)
    for _ in range(n_steps):
        name = labels[np.random.randint(len(labels))]
        X = rand_batch(bs, S, V)
        _, steps = model(X, paths[name], reground=True, return_steps=True)
        tg = partials(atoms[name], X, V, R)
        loss = sum(F.cross_entropy(steps[j], tg[j]) for j in range(k))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def main():
    V, D, N_HEADS = 50, 128, 4
    K, S, BS = 3, 24, 64
    MAXLEN, N_STEPS, LR = 24, 2500, 1e-3
    R = rules(V)

    PURE = [[a] for a in ATOMS]
    SEEN = [["math", "history"], ["geography", "delta"], ["delta", "history"]]
    HELDOUT = [["math", "delta"], ["math", "history", "delta"]]
    TRAIN = PURE + SEEN

    print("=" * 72)
    print(f"  B-MoE POC step 1 — dual-stream loop (context) on {str(DEVICE).upper()}")
    print("=" * 72)
    print(f"  Skills: {', '.join(ATOMS)}  (delta is CONTEXT-DEPENDENT)")
    print(f"  Trained: pures + {', '.join('[' + label(t) + ']' for t in SEEN)}")
    print(f"  Held out (zero-shot): {', '.join('[' + label(t) + ']' for t in HELDOUT)}")
    print(f"  vocab={V} d={D} loop_k={K} seq={S} | {N_STEPS} steps")
    print("-" * 72)

    Xte = rand_batch(512, S, V)

    # ── Full architecture (with context) ───────────────────────────────────────
    print("\nTraining WITH context (full architecture)...")
    model = train_model(True, TRAIN, V, R, K, D, N_HEADS, S, BS, N_STEPS, LR, MAXLEN)
    print("\n  per-skill accuracy (forced path, held-out random inputs):")
    for a in ATOMS:
        tag = " <- context-dependent" if a == "delta" else ""
        print(f"    {a:<11}: {acc(model, Xte, [a], V, R, K):.3f}{tag}")
    print("\n  zero-shot composition (forced correct path):")
    for t in HELDOUT:
        print(f"    [{label(t):<17}]: {acc(model, Xte, t, V, R, K):.3f}")
    pair = acc(model, Xte, HELDOUT[0], V, R, K)
    triple = acc(model, Xte, HELDOUT[1], V, R, K)

    # path search (auto-discovery) on the triple
    Y = compose(HELDOUT[1], Xte, V, R)
    scored = sorted(
        (((model(Xte, p).argmax(-1) == Y.reshape(-1)).float().mean().item(), p)
         for p in itertools.product(range(N_EXPERTS), repeat=K)), reverse=True)
    correct = expert_path(task_atoms(HELDOUT[1], K))
    rank = [p for _, p in scored].index(correct) + 1
    print(f"\n  path search on triple: best {scored[0][1]}={scored[0][0]:.2f}, "
          f"correct {correct} rank {rank}/{len(scored)}")

    # ── Ablation: without context, delta must collapse ─────────────────────────
    print("\nTraining WITHOUT context (ablation)...")
    nomodel = train_model(False, TRAIN, V, R, K, D, N_HEADS, S, BS, N_STEPS, LR, MAXLEN)
    d_ctx = acc(model, Xte, ["delta"], V, R, K)
    d_noctx = acc(nomodel, Xte, ["delta"], V, R, K)
    tw = np.mean([acc(nomodel, Xte, [a], V, R, K) for a in ["math", "history", "geography"]])

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)
    print(f"  delta accuracy:  with context = {d_ctx:.3f}   without context = {d_noctx:.3f}")
    print(f"  token-wise skills without context = {tw:.3f} (still fine — they need no context)")
    print(f"  zero-shot composition (with context):  pair = {pair:.3f}  triple = {triple:.3f}")
    ok_ctx = d_ctx > 0.7 and d_noctx < 0.4
    ok_comp = pair > 0.7 and triple > 0.7
    if ok_ctx and ok_comp:
        print("\n  STEP 1 VALIDATED: the context stream is load-bearing (delta works only with")
        print("  it), AND the dual-stream loop composes unseen chains zero-shot. Architecture")
        print("  ready for real tokens (step 2): learned routing + halting next.")
    else:
        print(f"\n  Partial (ctx_ok={ok_ctx}, comp_ok={ok_comp}) — inspect before step 2.")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
