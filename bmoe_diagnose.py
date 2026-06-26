"""
B-MoE — diagnosing WHY zero-shot composition fails
==================================================
For a composition the model never trained on (e.g. math+geography), we enumerate ALL
expert paths (one expert per bloc, N^B of them), force each, and measure accuracy.

This separates two failure modes:
  (a) ROUTING failure       — a good path EXISTS but the greedy router doesn't pick it.
                              => an oracle/search over paths would fix it.
  (b) COMPOSABILITY failure — even the BEST forced path is poor: experts don't implement
                              reusable transforms, so no fixed chain composes.
                              => path search alone can't help; experts must be made reusable.

Run: `python bmoe_diagnose.py`.
"""

import itertools

import numpy as np
import torch

from toyBMoE import (
    ATOMS, evaluate, label, routing_paths, run_bmoe,
)


def diagnose(model, test, task, n_blocs, n_experts):
    name = label(task)
    X, Y = test[name]

    # learned (greedy) routing
    learned_acc = evaluate(model, X, Y)[1]
    paths = routing_paths(model, X)
    modal = tuple(int(torch.mode(paths[:, b]).values) for b in range(n_blocs))

    # enumerate every forced expert path
    scored = []
    for path in itertools.product(range(n_experts), repeat=n_blocs):
        acc = evaluate(model, X, Y, force_expert=path)[1]
        scored.append((acc, path))
    scored.sort(reverse=True)
    best_acc, best_path = scored[0]

    print(f"\n  Held-out task: [{name}]   ({n_experts}^{n_blocs} = "
          f"{n_experts ** n_blocs} possible paths)")
    print(f"    learned greedy routing : acc={learned_acc:.3f}   modal path "
          f"{' > '.join('E'+str(e) for e in modal)}")
    print(f"    BEST path (oracle)     : acc={best_acc:.3f}   path "
          f"{' > '.join('E'+str(e) for e in best_path)}")
    print("    top-5 paths:  " + "   ".join(
        f"{'>'.join('E'+str(e) for e in p)}={a:.2f}" for a, p in scored[:5]))
    return learned_acc, best_acc


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    PURE = [[a] for a in ATOMS]
    SEEN = [["math", "history"], ["history", "geography"]]
    HELDOUT = [["math", "geography"], ["math", "history", "geography"]]
    N_BLOCS = 3

    print("=" * 72)
    print("  B-MoE: diagnosing zero-shot composition failure (path enumeration)")
    print("=" * 72)
    print(f"  Trained on: pures + {', '.join('[' + label(t) + ']' for t in SEEN)}")
    print(f"  Held out  : {', '.join('[' + label(t) + ']' for t in HELDOUT)}")
    print("-" * 72)

    model, test, meta = run_bmoe(train_tasks=PURE + SEEN, eval_tasks=HELDOUT,
                                 n_blocs=N_BLOCS)
    n_experts = meta["n_experts"]

    print("\n" + "=" * 72)
    print("  PATH ENUMERATION on held-out compositions")
    print("=" * 72)
    results = [diagnose(model, test, t, N_BLOCS, n_experts) for t in HELDOUT]

    best_over_all = max(b for _, b in results)
    print("\n" + "-" * 72)
    if best_over_all > 0.7:
        print("  VERDICT (a) ROUTING failure: a good expert path EXISTS for the unseen")
        print("  composition, but the greedy router never selects it. A search/scoring over")
        print("  expert-path combinations (your idea) should recover zero-shot composition.")
    else:
        print("  VERDICT (b) COMPOSABILITY failure: even the BEST forced path is poor, so no")
        print("  fixed chain of the current experts composes the unseen skill. Path search")
        print("  alone cannot help — the experts must first be made reusable (supervised")
        print("  atom-routing, or a per-bloc skill bottleneck) so single blocs implement one")
        print(f"  reusable transform. (best oracle path acc over held-out = {best_over_all:.3f})")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
