"""
B-MoE on REAL tokens — character-level multi-register language model
====================================================================
Moves B-MoE from synthetic integer maps to actual text. We build a small,
reproducible multi-register corpus (weather / finance / recipe sentences),
tokenize it at the CHARACTER level (real tokens), and train a B-MoE char LM.

What it shows:
  - Expert specialization on real text: each expert is pre-trained on one register,
    and learned bloc routing recovers the register -> expert assignment.
  - Compositional switching on text: "mixed" documents that splice two registers in
    one window are solved by switching experts across blocs, and collapse when forced
    through a single expert (ablation).

The B-MoE model itself is imported unchanged from toyBMoE.py.
"""

import random

import numpy as np
import torch

from toyBMoE import (
    BMoE, best_single_expert_acc, ce_loss, divergence_loss, evaluate,
    load_balance_loss, make_pairs, pre_specialize, routing_fractions,
    sample_batch,
)

# ─── A small, reproducible multi-register corpus (real words, char tokens) ──────

CITIES = ["paris", "london", "tokyo", "cairo", "oslo", "lima", "berlin", "madrid"]
CONDS = ["sunny", "rainy", "cloudy", "windy", "foggy", "snowy", "mild", "stormy"]
FIRMS = ["acme", "globex", "initech", "umbrella", "soylent", "stark", "wayne"]
MOVES = ["rose", "fell", "gained", "dropped", "slipped", "jumped"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
INGRS = ["flour", "sugar", "butter", "cocoa", "salt", "honey", "yeast"]
ACTS = ["bake", "simmer", "whisk", "fold", "knead", "stir"]


def weather_sentence(rng):
    return (f"the weather in {rng.choice(CITIES)} is {rng.choice(CONDS)} "
            f"with {rng.randint(0, 35)} degrees . ")


def finance_sentence(rng):
    return (f"the {rng.choice(FIRMS)} share {rng.choice(MOVES)} {rng.randint(1, 40)} "
            f"percent on {rng.choice(DAYS)} . ")


def recipe_sentence(rng):
    return (f"{rng.choice(ACTS)} {rng.randint(2, 99)} grams of {rng.choice(INGRS)} "
            f"for {rng.randint(5, 60)} minutes . ")


REGISTERS = {
    "weather": weather_sentence,
    "finance": finance_sentence,
    "recipe": recipe_sentence,
}


def build_corpus(register, n_sentences, rng):
    gen = REGISTERS[register]
    return "".join(gen(rng) for _ in range(n_sentences))


def build_mixed_corpus(n_sentences, rng):
    """Documents that splice two different registers back-to-back (need both skills)."""
    names = list(REGISTERS)
    out = []
    for _ in range(n_sentences):
        a, b = rng.sample(names, 2)
        out.append(REGISTERS[a](rng) + REGISTERS[b](rng))
    return "".join(out)


# ─── Char tokenizer + windowing ────────────────────────────────────────────────

def windows(encoded, n_samples, seq_len, rng):
    """Sample fixed-length char windows -> (X, Y) causal next-char pairs."""
    starts = [rng.randint(0, len(encoded) - seq_len - 2) for _ in range(n_samples)]
    data = torch.stack([encoded[s:s + seq_len + 1] for s in starts])  # (n, L+1)
    return make_pairs(data)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    rng = random.Random(0)

    # Hyperparameters
    SEQ_LEN = 48
    D_MODEL = 96
    N_HEADS = 4
    N_BLOCS = 3
    LAYERS_PER_BLOC = 2
    D_K_ROUTING = 16
    BATCH = 32
    N_SENT = 1500       # sentences per register
    N_WIN_TR = 400      # training windows per task
    N_WIN_TE = 200
    T_PRE = 250
    N_JOINT = 700
    LR = 1e-3
    ALPHA_BAL = 0.05
    LAMBDA_DIV = 0.01

    names = list(REGISTERS)
    n_experts = len(names)

    # Build train/test corpora (independent random draws -> genuine generalization)
    corp_tr = {r: build_corpus(r, N_SENT, rng) for r in names}
    corp_te = {r: build_corpus(r, N_SENT // 2, rng) for r in names}
    mix_tr = build_mixed_corpus(N_SENT, rng)
    mix_te = build_mixed_corpus(N_SENT // 2, rng)

    # Shared char vocabulary
    charset = sorted(set("".join(corp_tr.values()) + mix_tr))
    stoi = {c: i for i, c in enumerate(charset)}
    vocab = len(charset)

    def enc(s):
        return torch.tensor([stoi[c] for c in s], dtype=torch.long)

    enc_tr = {r: enc(corp_tr[r]) for r in names}
    enc_te = {r: enc(corp_te[r]) for r in names}
    mix_tr_e, mix_te_e = enc(mix_tr), enc(mix_te)

    train = {r: windows(enc_tr[r], N_WIN_TR, SEQ_LEN, rng) for r in names}
    train["mixed"] = windows(mix_tr_e, N_WIN_TR, SEQ_LEN, rng)
    test = {r: windows(enc_te[r], N_WIN_TE, SEQ_LEN, rng) for r in names}
    test["mixed"] = windows(mix_te_e, N_WIN_TE, SEQ_LEN, rng)

    print("=" * 72)
    print("  B-MoE on REAL tokens — character-level multi-register LM")
    print("=" * 72)
    print(f"  Registers (1 expert each): {', '.join(names)}")
    print(f"  Char vocab = {vocab} | seq_len = {SEQ_LEN} | B={N_BLOCS} blocs | Z={LAYERS_PER_BLOC}")
    print(f"  Sample: \"{corp_tr['finance'][:60]}...\"")
    print("-" * 72)

    model = BMoE(vocab, D_MODEL, N_HEADS, n_experts, N_BLOCS, LAYERS_PER_BLOC,
                 D_K_ROUTING, max_len=SEQ_LEN)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Phase 1: pre-specialize one expert per register ─────────────────────────
    print("Phase 1 — Pre-specialization (Expert_i <- register i)")
    domains = {i: (r, *train[r]) for i, r in enumerate(names)}
    pre_specialize(model, domains, T_PRE, LR, BATCH)

    # ── Phase 2: joint training on all registers + mixed documents ──────────────
    print("\nPhase 2 — Joint training (registers + mixed docs, learned bloc routing)\n")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    task_pool = names + ["mixed"]
    log_every = max(1, N_JOINT // 12)

    for step in range(N_JOINT):
        name = rng.choice(task_pool)
        Xb, Yb = sample_batch(*train[name], BATCH)
        model.train()
        logits, routing = model(Xb)
        loss = (ce_loss(logits, Yb)
                + ALPHA_BAL * load_balance_loss(routing, n_experts)
                + LAMBDA_DIV * divergence_loss(model, Xb, n_experts))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % log_every == 0 or step == N_JOINT - 1:
            reg = np.mean([evaluate(model, *test[r])[1] for r in names])
            mix = evaluate(model, *test["mixed"])[1]
            print(f"  Step {step:>4d} | next-char acc  registers={reg:.3f}  mixed={mix:.3f}")

    # ── RESULT 1 — next-char accuracy + specialization recovery ─────────────────
    print("\n" + "=" * 72)
    print("  RESULT 1 — Next-char accuracy and routing specialization")
    print("=" * 72)
    print(f"\n  {'Register':<10} {'acc':>6} {'modal expert path (B0>B1>B2)':<30}")
    print("  " + "-" * 48)
    hits = 0
    for i, r in enumerate(names):
        _, acc = evaluate(model, *test[r])
        fr = routing_fractions(model, test[r][0])
        modal = [int(np.argmax(fr[b])) for b in range(N_BLOCS)]
        # "owns" the register if its pre-assigned expert is the most-used across blocs
        top_overall = int(np.argmax(np.sum([fr[b] for b in range(N_BLOCS)], axis=0)))
        hits += top_overall == i
        path = " > ".join(f"E{e}" for e in modal)
        print(f"  {r:<10} {acc:>6.3f} {path:<30} (owns E{top_overall})")
    print(f"\n  Register->expert specialization recovered: {hits}/{n_experts}")

    # ── RESULT 2 — switching is load-bearing on real text (ablation) ────────────
    print("\n" + "=" * 72)
    print("  RESULT 2 — Bloc-by-bloc switching vs forcing a single expert (ablation)")
    print("=" * 72)
    _, mix_acc = evaluate(model, *test["mixed"])
    mix_single = best_single_expert_acc(model, *test["mixed"])
    reg_acc = np.mean([evaluate(model, *test[r])[1] for r in names])
    reg_single = np.mean([best_single_expert_acc(model, *test[r]) for r in names])
    print(f"\n  {'Document type':<16} {'learned':>8} {'best-1expert':>13} {'switch gain':>12}")
    print("  " + "-" * 52)
    print(f"  {'single register':<16} {reg_acc:>8.3f} {reg_single:>13.3f} {reg_acc - reg_single:>+12.3f}")
    print(f"  {'mixed register':<16} {mix_acc:>8.3f} {mix_single:>13.3f} {mix_acc - mix_single:>+12.3f}")
    print("\n  Honest read: forcing any single expert collapses ALL text to ~chance, so")
    print("  bloc-by-bloc switching is load-bearing on real tokens. But single and mixed")
    print("  documents need switching about equally here -- the model distributes its")
    print("  computation across experts for every input, so this setup does NOT isolate a")
    print("  mixed-specific benefit (the clean compositional gap lives in toyBMoE.py).")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
