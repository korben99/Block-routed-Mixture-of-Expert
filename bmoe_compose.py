"""
B-MoE — zero-shot compositional generalization (the open problem)
=================================================================
We hold out whole compositions from training and test them zero-shot. The model is
trained on the atomic skills and TWO of the three pairwise compositions; it is then
evaluated on a pair it never saw (`math+geography`) and on the full triple.

This probes *systematic* composition: did the model learn reusable expert skills (so an
unseen chain just works), or did it memorize each trained composition as a holistic map?

Run: `python bmoe_compose.py`. Reuses the model + training loop from toyBMoE.py.
"""

import numpy as np
import torch

from toyBMoE import ATOMS, best_single_expert_acc, evaluate, label, run_bmoe


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    PURE = [[a] for a in ATOMS]
    SEEN = [["math", "history"], ["history", "geography"]]          # trained compositions
    HELDOUT = [["math", "geography"], ["math", "history", "geography"]]  # zero-shot
    TRAIN_TASKS = PURE + SEEN

    print("=" * 72)
    print("  B-MoE: zero-shot compositional generalization")
    print("=" * 72)
    print(f"  Atomic skills: {', '.join(ATOMS)}")
    print(f"  Compositions SEEN in training:     {', '.join('[' + label(t) + ']' for t in SEEN)}")
    print(f"  Compositions HELD OUT (zero-shot): {', '.join('[' + label(t) + ']' for t in HELDOUT)}")
    print("-" * 72)

    model, test, meta = run_bmoe(train_tasks=TRAIN_TASKS, eval_tasks=HELDOUT, n_blocs=3)

    # Evaluate everything (trained + held-out)
    print("\n" + "=" * 72)
    print("  RESULT — Zero-shot compositional generalization")
    print("=" * 72)
    print(f"\n  {'Task':<22} {'split':<10} {'acc':>7} {'best-1expert':>13}")
    print("  " + "-" * 56)

    def show(task, split):
        name = label(task)
        acc = evaluate(model, *test[name])[1]
        single = best_single_expert_acc(model, *test[name])
        print(f"  [{name:<19}] {split:<10} {acc:>7.3f} {single:>13.3f}")
        return acc

    for t in PURE:
        show(t, "pure")
    seen_acc = np.mean([show(t, "seen") for t in SEEN])
    zero_acc = np.mean([show(t, "zero-shot") for t in HELDOUT])

    print(f"\n  seen-composition acc = {seen_acc:.3f}   |   zero-shot acc = {zero_acc:.3f}")
    print("  " + "-" * 70)
    if zero_acc > 0.5:
        print("  -> B-MoE GENERALIZES compositionally: unseen chains work (reusable skills).")
    else:
        print("  HONEST NEGATIVE: vanilla B-MoE does NOT generalize to unseen compositions.")
        print("  It solves *trained* compositions by bloc-switching, but learns them as")
        print("  holistic maps, not reusable skills. Systematic zero-shot composition needs a")
        print("  stronger inductive bias (e.g. routing supervised by the atom sequence, or a")
        print("  bloc-level skill bottleneck). This is the next research target.")

    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
