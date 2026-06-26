"""
B-MoE — Lever 2: deep supervision (canonical token-grounded inter-bloc interface)
=================================================================================
Lever 1 (bmoe_lever1.py) made each expert act at a single bloc, yet zero-shot
composition still failed (~0.05) even with the correct forced path. Reason: the experts
CO-ADAPT to their training-time predecessor. E_geo at bloc 1 only ever consumed E_history's
output; fed E_math's output (unseen) it breaks. The inter-bloc representation is
expert-specific, not canonical.

Lever 2 fixes the interface. With supervised atom->bloc routing AND an identity expert, we
add DEEP SUPERVISION: after bloc j, the shared head must decode the *partially composed*
token (apply the first j atoms). This pins every bloc's output to canonical token space, so
any expert downstream receives an in-distribution, token-grounded input.

    sequence transition x_{t+1} = atom_k(...atom_1(x_t))
    target after bloc j (deep supervision) = atom_j(...atom_1(x_t))   [identity = no-op]

If a canonical interface is the missing ingredient, the held-out composition math+geography
(forced path E_math>E_geo>E_id) should now compose.

Run: `python bmoe_lever2.py`.
"""

import itertools

import numpy as np
import torch
import torch.nn.functional as F

from toyBMoE import (
    ATOMS, BMoE, build_dataset, evaluate, label, make_domain_rules, make_pairs,
    sample_batch,
)

EXPERT = {**{a: i for i, a in enumerate(ATOMS)}, "identity": len(ATOMS)}
N_EXPERTS = len(ATOMS) + 1


def task_atoms(task, n_blocs):
    return list(task) + ["identity"] * (n_blocs - len(task))


def expert_path(atoms):
    return tuple(EXPERT[a] for a in atoms)


def apply_atom(name, x, vocab, rules):
    """Vectorized single-atom map on a LongTensor of token ids."""
    if name == "math":
        return (x + 5) % vocab
    if name == "history":
        return (rules["a"] * x + rules["b"]) % vocab
    if name == "geography":
        return rules["perm"][x]
    if name == "identity":
        return x
    raise ValueError(name)


def partial_targets(X, atoms, vocab, rules):
    """Per-bloc deep-supervision targets: token after applying the first j atoms."""
    targets, cur = [], X
    for name in atoms:
        cur = apply_atom(name, cur, vocab, rules)
        targets.append(cur.reshape(-1))
    return targets  # list length n_blocs, each (B*S,)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    PURE = [[a] for a in ATOMS]
    SEEN = [["math", "history"], ["history", "geography"]]
    HELDOUT = [["math", "geography"], ["math", "history", "geography"]]
    TRAIN_TASKS = PURE + SEEN
    ALL_TASKS = TRAIN_TASKS + HELDOUT

    VOCAB, D_MODEL, N_HEADS = 50, 64, 4
    N_BLOCS, Z, D_K = 3, 2, 16
    SEQ_LEN, BATCH = 30, 32
    N_TRAIN, N_TEST = 300, 150
    N_STEPS, LR = 1500, 1e-3

    print("=" * 72)
    print("  B-MoE Lever 2 — deep supervision (canonical inter-bloc interface)")
    print("=" * 72)
    print(f"  Experts: {', '.join(f'E{EXPERT[a]}={a}' for a in ATOMS)}, E{EXPERT['identity']}=identity")
    print(f"  Trained: pures + {', '.join('[' + label(t) + ']' for t in SEEN)}")
    print(f"  Held out (zero-shot): {', '.join('[' + label(t) + ']' for t in HELDOUT)}")
    print("  Deep supervision: head after bloc j must decode the partially-composed token")
    print("-" * 72)

    rules = make_domain_rules(VOCAB, seed=0)
    train, test = {}, {}
    for t in TRAIN_TASKS:
        train[label(t)] = make_pairs(build_dataset(N_TRAIN, SEQ_LEN, t, VOCAB, rules))
    for t in ALL_TASKS:
        test[label(t)] = make_pairs(build_dataset(N_TEST, SEQ_LEN, t, VOCAB, rules))

    model = BMoE(VOCAB, D_MODEL, N_HEADS, N_EXPERTS, N_BLOCS, Z, D_K, max_len=SEQ_LEN)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    atoms = {label(t): task_atoms(t, N_BLOCS) for t in TRAIN_TASKS}
    paths = {k: expert_path(a) for k, a in atoms.items()}
    train_labels = list(paths)

    print("Training (forced atom-routing + deep supervision per bloc)\n")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(N_STEPS):
        name = train_labels[torch.randint(0, len(train_labels), (1,)).item()]
        Xb, Yb = sample_batch(*train[name], BATCH)
        model.train()
        _, _, bloc_logits = model(Xb, force_expert=paths[name], return_bloc_logits=True)
        tgts = partial_targets(Xb, atoms[name], VOCAB, rules)
        loss = sum(F.cross_entropy(bloc_logits[j], tgts[j]) for j in range(N_BLOCS))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % (N_STEPS // 12) == 0 or step == N_STEPS - 1:
            tr = np.mean([evaluate(model, *test[k], force_expert=paths[k])[1]
                          for k in train_labels])
            print(f"  Step {step:>4d} | train-task acc (forced path) = {tr:.3f}")

    # ── Zero-shot: force the correct atom path on held-out compositions ─────────
    print("\n" + "=" * 72)
    print("  RESULT — forcing each task's correct atom path")
    print("=" * 72)
    print(f"\n  {'Task':<22} {'split':<10} {'forced-path acc':>16}")
    print("  " + "-" * 50)
    for t in ALL_TASKS:
        acc = evaluate(model, *test[label(t)], force_expert=expert_path(task_atoms(t, N_BLOCS)))[1]
        split = "pure" if len(t) == 1 else ("seen" if t in SEEN else "ZERO-SHOT")
        print(f"  [{label(t):<19}] {split:<10} {acc:>16.3f}")
    zero = np.mean([evaluate(model, *test[label(t)],
                             force_expert=expert_path(task_atoms(t, N_BLOCS)))[1]
                    for t in HELDOUT])

    # ── Does path search now recover the right chain? (sets up Lever 3) ─────────
    print("\n" + "=" * 72)
    print("  PATH SEARCH on held-out compositions")
    print("=" * 72)
    for t in HELDOUT:
        name = label(t)
        X, Y = test[name]
        scored = sorted(((evaluate(model, X, Y, force_expert=p)[1], p)
                         for p in itertools.product(range(N_EXPERTS), repeat=N_BLOCS)),
                        reverse=True)
        correct = expert_path(task_atoms(t, N_BLOCS))
        rank = [p for _, p in scored].index(correct) + 1
        best_acc, best_path = scored[0]
        print(f"\n  [{name}]   correct path {' > '.join('E'+str(e) for e in correct)} "
              f"acc={evaluate(model, X, Y, force_expert=correct)[1]:.3f} (rank {rank}/{len(scored)})")
        print(f"    best searched {' > '.join('E'+str(e) for e in best_path)} acc={best_acc:.3f}")

    print("\n" + "-" * 72)
    if zero > 0.7:
        print(f"  LEVER 2 WORKS: zero-shot composition acc = {zero:.3f} (Lever 1 was ~0.04).")
        print("  A canonical token-grounded interface makes experts compose on unseen chains.")
        print("  Next: (Lever 3) pick the chain automatically by scoring paths via likelihood;")
        print("  and make the interface emerge WITHOUT atom labels (re-embedding / train-alone).")
    else:
        print(f"  Partial: zero-shot acc = {zero:.3f}. Deep supervision alone did not fully")
        print("  canonicalize the interface (head-projection is shared, but h is still 64-d).")
        print("  Next experiment: hard re-grounding (re-embed each bloc's decoded token) =")
        print("  the 'train experts alone, then compose' design.")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
