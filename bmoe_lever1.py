"""
B-MoE — Lever 1: supervised atom->bloc routing (explicit modularity)
====================================================================
Diagnosis (bmoe_diagnose.py) showed zero-shot composition fails because the experts are
NOT composable: no fixed expert path solves an unseen composition (best oracle ~0.06).
Reason: pre-specialization trains each expert across ALL blocs, so "1 bloc of expert-i"
is not "one application of skill i".

Lever 1 fixes this with an explicit inductive bias. We add an IDENTITY (no-op) expert and,
during training, FORCE the routing of a composition to follow its atom sequence, padded
with identity:

    [math]            -> path  E_math  > E_id   > E_id
    [math, geography] -> path  E_math  > E_geo  > E_id
    [math,hist,geo]   -> path  E_math  > E_hist > E_geo

Because each expert is supervised to act at a single bloc, it must learn ONE reusable
application of its skill. We then test the HELD-OUT composition `math+geography` (never
trained) by forcing its correct atom path E_math>E_geo>E_id. If accuracy jumps from ~0.06
(diagnosed best path) to high, modularity unlocks zero-shot composition.

Run: `python bmoe_lever1.py`.
"""

import itertools

import numpy as np
import torch

from toyBMoE import (
    ATOMS, BMoE, build_dataset, ce_loss, evaluate, label, make_domain_rules,
    make_pairs, sample_batch,
)

EXPERT = {a: i for i, a in enumerate(ATOMS)}   # math=0, history=1, geography=2
IDENTITY = len(ATOMS)                          # 3 -> the no-op expert
N_EXPERTS = len(ATOMS) + 1                      # 4


def task_path(task, n_blocs):
    """Atom sequence -> forced expert path, padded with the identity expert."""
    p = [EXPERT[a] for a in task] + [IDENTITY] * (n_blocs - len(task))
    return tuple(p)


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
    print("  B-MoE Lever 1 — supervised atom->bloc routing (+ identity expert)")
    print("=" * 72)
    print(f"  Experts: {', '.join(f'E{EXPERT[a]}={a}' for a in ATOMS)}, E{IDENTITY}=identity")
    print(f"  Trained: pures + {', '.join('[' + label(t) + ']' for t in SEEN)}")
    print(f"  Held out (zero-shot): {', '.join('[' + label(t) + ']' for t in HELDOUT)}")
    for t in ALL_TASKS:
        print(f"    [{label(t):<19}] -> path {' > '.join('E'+str(e) for e in task_path(t, N_BLOCS))}")
    print("-" * 72)

    rules = make_domain_rules(VOCAB, seed=0)
    train, test = {}, {}
    for t in TRAIN_TASKS:
        train[label(t)] = make_pairs(build_dataset(N_TRAIN, SEQ_LEN, t, VOCAB, rules))
    for t in ALL_TASKS:
        test[label(t)] = make_pairs(build_dataset(N_TEST, SEQ_LEN, t, VOCAB, rules))

    model = BMoE(VOCAB, D_MODEL, N_HEADS, N_EXPERTS, N_BLOCS, Z, D_K, max_len=SEQ_LEN)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Training with supervised (forced) atom->bloc routing ────────────────────
    print("Training (forced atom-routing, experts learn one reusable step each)\n")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    paths = {label(t): task_path(t, N_BLOCS) for t in TRAIN_TASKS}
    train_labels = list(paths)

    for step in range(N_STEPS):
        name = train_labels[torch.randint(0, len(train_labels), (1,)).item()]
        model.train()
        Xb, Yb = sample_batch(*train[name], BATCH)
        logits, _ = model(Xb, force_expert=paths[name])   # forced atom path
        loss = ce_loss(logits, Yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % (N_STEPS // 12) == 0 or step == N_STEPS - 1:
            tr = np.mean([evaluate(model, *test[label(t)], force_expert=paths[label(t)])[1]
                          for t in TRAIN_TASKS])
            print(f"  Step {step:>4d} | train-task acc (forced path) = {tr:.3f}")

    # ── Evaluation: force the CORRECT atom path on every task ───────────────────
    print("\n" + "=" * 72)
    print("  RESULT — accuracy when forcing each task's correct atom path")
    print("=" * 72)
    print(f"\n  {'Task':<22} {'split':<10} {'forced-path acc':>16}")
    print("  " + "-" * 50)
    for t in ALL_TASKS:
        name = label(t)
        acc = evaluate(model, *test[name], force_expert=task_path(t, N_BLOCS))[1]
        split = "pure" if len(t) == 1 else ("seen" if t in SEEN else "ZERO-SHOT")
        print(f"  [{name:<19}] {split:<10} {acc:>16.3f}")

    zero = np.mean([evaluate(model, *test[label(t)], force_expert=task_path(t, N_BLOCS))[1]
                    for t in HELDOUT])

    # ── Does a path SEARCH now find the right chain? (sets up Lever 3) ──────────
    print("\n" + "=" * 72)
    print("  PATH SEARCH on held-out compositions (is the correct atom path the winner?)")
    print("=" * 72)
    for t in HELDOUT:
        name = label(t)
        X, Y = test[name]
        scored = sorted(
            ((evaluate(model, X, Y, force_expert=p)[1], p)
             for p in itertools.product(range(N_EXPERTS), repeat=N_BLOCS)),
            reverse=True,
        )
        correct = task_path(t, N_BLOCS)
        rank = [p for _, p in scored].index(correct) + 1
        best_acc, best_path = scored[0]
        print(f"\n  [{name}]   (searched {N_EXPERTS**N_BLOCS} paths)")
        print(f"    correct atom path {' > '.join('E'+str(e) for e in correct)} : "
              f"acc={evaluate(model, X, Y, force_expert=correct)[1]:.3f}  (rank {rank} of search)")
        print(f"    best searched path {' > '.join('E'+str(e) for e in best_path)} : acc={best_acc:.3f}")

    print("\n" + "-" * 72)
    if zero > 0.7:
        print(f"  LEVER 1 WORKS: zero-shot composition acc = {zero:.3f} (was ~0.01 / oracle 0.06).")
        print("  Modular experts make an unseen composition solvable by chaining experts.")
        print("  Next: Lever 2 (make modularity emerge WITHOUT atom labels) + path search at")
        print("  inference (your idea) to pick the chain automatically.")
    else:
        print(f"  Inconclusive: zero-shot acc = {zero:.3f}. Modularity did not fully transfer;")
        print("  inspect per-position expert reuse / identity behavior before Lever 2.")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
