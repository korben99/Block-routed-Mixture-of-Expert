"""
B-MoE POC — cybersecurity scope: autonomous deobfuscation-pipeline discovery
============================================================================
Same architecture as the autonomous-routing result (bmoe_poc4.py): a loop of SHARED
skills with token re-grounding + a SUFFICIENCY CRITIC that lets the model route itself
from a few demonstrations — but instantiated on a concrete security task instead of an
abstract toy.

THE PROBLEM (real SOC / malware-analysis pain).  Attackers hide a payload behind a STACK
of cheap, reversible obfuscation layers — a single-byte XOR key, a custom substitution
alphabet (S-box / non-standard base64), a rolling/CBC stream where each byte is masked by
the previous one. Stacks evade static signatures because the *combination* is unseen even
when each layer is well known. A defender must recover the right INVERSE DECODING PIPELINE
(which decoders, in which order) to reveal the underlying signature — for arbitrary,
never-before-seen layerings.

THE SETUP.  The model owns three DECODER skills (one per obfuscation primitive). It is
trained on individual decoders and a few decoding stacks. At inference it is handed an
*incident*: a handful of (obfuscated -> revealed) example pairs an analyst extracted (the
few-shot demos), plus fresh captured traffic obfuscated the same way. With NO router and NO
"which-decoder" tag, the model:

    proposes decoding chains of its own primitives, shortest first;
    a SUFFICIENCY CRITIC checks each chain on the demos (does it reproduce the revealed
        bytes?) and accepts the shortest that does;
    that recovered pipeline is applied to the fresh traffic.

The two HELD-OUT incidents use obfuscation stacks the model never trained on. The win: it
recovers the pipeline itself and decodes the unseen traffic, zero-shot.

  caesar : single-byte XOR / additive key       Y[t] = (X[t]-k) mod V      (token-wise)
  sbox   : custom substitution alphabet (S-box)  Y[t] = inv_perm[X[t]]      (token-wise)
  stream : rolling / CBC keystream               Y[t] = (X[t]-X[t-1]) mod V (CONTEXT-dep.)

`stream` is invertible only with the per-step context view (each byte depends on the
previous one) — it collapses without it, proving the context stream is load-bearing.
Runs on the M4 GPU (MPS) when available. ~20 s.
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Security decoder primitives (the model's own skills) ─────────────────────
SKILLS = ["caesar", "sbox", "stream"]
N = len(SKILLS)
K_MAX = 3          # longest decoding pipeline the critic will propose
D_DEMOS = 4        # (obfuscated -> revealed) example pairs the analyst supplies
KEY = 5            # single-byte XOR/additive key


def label(chain):
    return " > ".join(SKILLS[c] if isinstance(c, int) else c for c in chain)


def boxes(vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"inv_perm": torch.randperm(vocab, generator=g).to(DEVICE)}


def decode(name, X, V, B):
    """One decoder applied to a (batch, seq) byte tensor — undoes one obfuscation layer."""
    if name == "caesar":                       # undo single-byte XOR / additive key
        return (X - KEY) % V
    if name == "sbox":                         # undo custom substitution alphabet
        return B["inv_perm"][X]
    if name == "stream":                       # undo rolling/CBC keystream (needs context)
        Y = X.clone()
        Y[:, 1:] = (X[:, 1:] - X[:, :-1]) % V
        return Y
    raise ValueError(name)


def run_pipeline(chain, X, V, B):
    for c in chain:
        X = decode(SKILLS[c] if isinstance(c, int) else c, X, V, B)
    return X


def stage_targets(chain, X, V, B):
    """Deep-supervision targets: bytes after applying the first j decoders."""
    outs, cur = [], X
    for c in chain:
        cur = decode(SKILLS[c] if isinstance(c, int) else c, cur, V, B)
        outs.append(cur.reshape(-1))
    return outs


def rand_traffic(bs, S, V):
    """Captured payload bytes (the obfuscated stream the model receives)."""
    return torch.randint(0, V, (bs, S), device=DEVICE)


# ── The architecture: a loop of shared decoder skills with token re-grounding ─
class Skill(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class DeobfuscatorLoop(nn.Module):
    """embed -> loop( shared context attn + shared decoder + re-ground ) -> byte head."""

    def __init__(self, vocab, d=128, n_heads=4, max_len=64, use_context=True):
        super().__init__()
        self.vocab, self.use_context = vocab, use_context
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm_ctx = nn.LayerNorm(d)
        self.experts = nn.ModuleList([Skill(d) for _ in range(N)])  # shared decoders
        self.norm_step = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def _ctx(self, w):
        if not self.use_context:
            return w
        S = w.size(1)
        m = torch.triu(torch.ones(S, S, device=w.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(w, w, w, attn_mask=m)
        return self.norm_ctx(w + a)

    def step(self, w, e):
        S = w.size(1)
        w = self.norm_step(w + self.experts[e](self._ctx(w)))
        logit = self.head(w)
        w = F.softmax(logit, -1) @ self.embed.weight + self.pos[:, :S, :]  # re-ground
        return w, logit

    def forced(self, X, chain):
        """Run a known chain, return per-stage logits (for deep-supervised training)."""
        S = X.size(1)
        w = self.embed(X) + self.pos[:, :S, :]
        outs = []
        for e in chain:
            w, logit = self.step(w, e)
            outs.append(logit.reshape(-1, self.vocab))
        return outs

    @torch.no_grad()
    def decode_chain(self, X, chain):
        """Apply a decoding chain; return recovered byte predictions (B, S)."""
        S = X.size(1)
        w = self.embed(X) + self.pos[:, :S, :]
        logit = self.head(w)
        for e in chain:
            w, logit = self.step(w, e)
        return logit.argmax(-1)


CRITIC_TOL = 0.95   # a wrong chain scores ~1/V per byte (chance), so this cleanly
                    # separates the right pipeline while absorbing model decode noise


@torch.no_grad()
def critic_route(model, Xobf, Xrev):
    """Propose decoding chains shortest-first; the sufficiency critic accepts the first
    chain that reproduces the analyst's revealed bytes on the demos. No router, no tag."""
    for length in range(1, K_MAX + 1):
        for chain in itertools.product(range(N), repeat=length):
            if (model.decode_chain(Xobf, list(chain)) == Xrev).float().mean().item() > CRITIC_TOL:
                return list(chain)
    return None


def obfuscate(stack, X, V, B):
    """Apply an attacker's obfuscation stack = the decoder pipeline run in REVERSE order
    with each layer's inverse, so that the model's forward decoders recover X."""
    inv = {"caesar": lambda x: (x + KEY) % V,
           "sbox":   lambda x: torch.argsort(B["inv_perm"])[x],
           "stream": lambda x: torch.cumsum(x, dim=1) % V}
    for name in reversed(stack):
        X = inv[name](X)
    return X


def main():
    V, Dm, S, BS = 64, 128, 16, 64          # V=64: byte/base64-ish alphabet
    B = boxes(V)
    PHASE_A, LR = 3000, 1e-3

    # decoding pipelines: each single decoder + a few short stacks; two stacks held out
    stacks = ([[s] for s in SKILLS]
              + [list(p) for p in itertools.permutations(SKILLS, 2)]
              + [list(p) for p in itertools.permutations(SKILLS, 3)])
    HELD = [["sbox", "caesar"], ["sbox", "caesar", "stream"]]   # unseen attacker stacks
    TRAIN = [s for s in stacks if s not in HELD]
    s2i = {s: i for i, s in enumerate(SKILLS)}

    print("=" * 74)
    print(f"  B-MoE CYBER POC — autonomous deobfuscation-pipeline discovery on {str(DEVICE).upper()}")
    print("=" * 74)
    print(f"  decoder skills : {', '.join(SKILLS)}   (stream is CONTEXT-dependent)")
    print(f"  trained on     : each decoder + {len(TRAIN) - N} known stacks")
    print(f"  held-out stacks: {', '.join('[' + label(h) + ']' for h in HELD)}  (never trained)")
    print("-" * 74)

    torch.manual_seed(0)
    model = DeobfuscatorLoop(V, Dm).to(DEVICE)

    # ── Phase A: learn the decoder skills via forced chains + deep supervision ──
    print("\nPhase A — learning decoder skills (forced pipelines)...")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(PHASE_A):
        stack = TRAIN[np.random.randint(len(TRAIN))]
        clean = rand_traffic(BS, S, V)
        Xobf = obfuscate(stack, clean, V, B)
        outs = model.forced(Xobf, [s2i[s] for s in stack])
        tg = stage_targets(stack, Xobf, V, B)
        loss = sum(F.cross_entropy(outs[j], tg[j]) for j in range(len(stack)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    @torch.no_grad()
    def skill_acc(s):
        clean = rand_traffic(256, S, V)
        Xobf = obfuscate([s], clean, V, B)
        return (model.decode_chain(Xobf, [s2i[s]]) == clean).float().mean().item()
    print("  per-decoder accuracy:", {s: round(skill_acc(s), 3) for s in SKILLS})

    # ── Incident response: recover the pipeline from few-shot demos, then decode ─
    @torch.no_grad()
    def respond(stack):
        clean_d = rand_traffic(D_DEMOS, S, V)              # analyst's example payloads
        Xobf_d = obfuscate(stack, clean_d, V, B)           # ...as the attacker hid them
        chain = critic_route(model, Xobf_d, clean_d)       # MODEL recovers the pipeline
        clean_q = rand_traffic(256, S, V)                  # fresh captured traffic
        Xobf_q = obfuscate(stack, clean_q, V, B)
        if chain is None:
            return 0.0, "(no pipeline found)"
        acc = (model.decode_chain(Xobf_q, chain) == clean_q).float().mean().item()
        return acc, label(chain)

    print("\n" + "=" * 74)
    print("  INCIDENT RESPONSE — model recovers the decoding pipeline from demos, alone")
    print("=" * 74)
    print(f"\n  {'attacker obfuscation stack':<34} {'split':<11} {'decoded':>8}  recovered pipeline")
    print("  " + "-" * 70)
    results = {label(stack): respond(stack) for stack in (TRAIN[:3] + HELD)}
    for stack in (TRAIN[:3] + HELD):
        acc, got = results[label(stack)]
        split = "ZERO-SHOT" if stack in HELD else "seen"
        ok = "✓" if got == label(stack) else f"!= {label(stack)}"
        print(f"  [{label(stack):<32}] {split:<11} {acc:>8.3f}  {got}  {ok}")

    zs_acc = np.mean([results[label(h)][0] for h in HELD])
    zs_ok = all(results[label(h)][1] == label(h) for h in HELD)
    print("\n" + "-" * 74)
    print(f"  zero-shot decode accuracy = {zs_acc:.3f}   pipelines recovered correctly = {zs_ok}")
    if zs_acc > 0.9 and zs_ok:
        print("\n  CONCLUSIVE: handed only a few (obfuscated -> revealed) examples, the model")
        print("  reconstructs the inverse decoding pipeline ITSELF — no router, no decoder tag —")
        print("  and decodes fresh traffic hidden behind an obfuscation stack it never trained")
        print("  on. Composing decoders + verifying against the demo resolves which layers were")
        print("  applied. At scale the exact-match critic becomes a learned 'is-this-revealed?'")
        print("  judge and the search becomes a guided proposer (agentic decode loop).")
    else:
        print("\n  Not fully conclusive — inspect decoder accuracy / critic threshold.")

    # ── Ablation: the context stream is load-bearing for the stream cipher ───────
    print("\nAblation — retraining WITHOUT the context view...")
    torch.manual_seed(0)
    nomodel = DeobfuscatorLoop(V, Dm, use_context=False).to(DEVICE)
    opt = torch.optim.Adam(nomodel.parameters(), lr=LR)
    for _ in range(PHASE_A):
        stack = TRAIN[np.random.randint(len(TRAIN))]
        clean = rand_traffic(BS, S, V)
        Xobf = obfuscate(stack, clean, V, B)
        outs = nomodel.forced(Xobf, [s2i[s] for s in stack])
        tg = stage_targets(stack, Xobf, V, B)
        loss = sum(F.cross_entropy(outs[j], tg[j]) for j in range(len(stack)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(nomodel.parameters(), 1.0); opt.step()

    @torch.no_grad()
    def stream_acc(m):
        clean = rand_traffic(256, S, V)
        Xobf = obfuscate(["stream"], clean, V, B)
        return (m.decode_chain(Xobf, [s2i["stream"]]) == clean).float().mean().item()
    print(f"  stream-cipher decode:  with context = {stream_acc(model):.3f}   "
          f"without context = {stream_acc(nomodel):.3f}")

    print("\n" + "=" * 74)
    print("  DONE")
    print("=" * 74)


if __name__ == "__main__":
    main()
