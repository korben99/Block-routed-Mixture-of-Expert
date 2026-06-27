"""
Stage-1 experiment: the bloc-routed MoE SLM, trained (pre-spec + joint), routing ITSELF on
held-out compositions via a LEARNED proposer + sufficiency critic — the paper end-to-end on a
real (small) neural model. Compares router-guided search to brute force.

Run:  python -m slm.experiment
"""

import itertools
import time

import numpy as np
import torch

from .config import DEVICE, SLMConfig
from .data import (ATOMS, REALISTIC, apply_skill, compose, make_params, rand_payload,
                   realistic_batch)
from .model import BlocRoutedMoESLM
from .route import apply_chain, autonomous_route, exhaustive_route
from .train import joint_train, pre_specialize

S, BS = 24, 64
D_DEMOS = 4
THR = 0.95          # critic tolerance on demos (absorbs the context-skill's ~0.99 decode)


def label(t):
    return "+".join(a[:4] for a in t)


def main():
    torch.manual_seed(0)
    cfg = SLMConfig(d_model=128, n_experts=5, n_blocs=3, layers_per_bloc=2, n_loop=3,
                    max_len=32, lambda_div=0.02, alpha_bal=0.02)
    P = make_params(seed=0)
    model = BlocRoutedMoESLM(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    comps = ([[a] for a in ATOMS]
             + [list(p) for p in itertools.permutations(ATOMS, 2)]
             + [list(p) for p in itertools.permutations(ATOMS, 3)])
    HELD = [["sbox", "caesar"], ["sbox", "caesar", "stream"]]
    TRAIN = [c for c in comps if c not in HELD]

    print("=" * 78)
    print(f"  B-MoE SLM — stage 1 (exact regime) on {str(DEVICE).upper()}   {n_params/1e6:.2f}M params")
    print("=" * 78)
    print(f"  L={cfg.n_layers} (B={cfg.n_blocs}×Z={cfg.layers_per_bloc})  N={cfg.n_experts} "
          f"experts  loop={cfg.n_loop}  | atoms: {', '.join(ATOMS)} (stream=context-dep.)")
    print(f"  train {len(TRAIN)} compositions, hold out: "
          f"{', '.join('['+label(h)+']' for h in HELD)}")
    print("-" * 78)

    t0 = time.time()
    print("\nPhase A — guided pre-specialization (one expert per primitive)...")
    accs = pre_specialize(model, P, S=S, bs=BS, steps=1500, lr=2e-3)
    print("  per-expert decode acc:", accs)

    print("\nPhase B — joint composition training (forced path + supervised router + aux)...")
    joint_train(model, P, TRAIN, S=S, bs=BS, steps=2500, lr=1e-3)

    @torch.no_grad()
    def skill_acc(name):
        from .data import EXPERT_ID
        X = rand_payload(256, S)
        out = apply_chain(model, X, [name])
        return (out == apply_skill(name, X, P)).float().mean().item()
    print(f"  done ({time.time()-t0:.0f}s total)")
    print("  per-expert decode acc (post-B):", {a: round(skill_acc(a), 3) for a in ATOMS})

    # ── autonomous routing: the model discovers the chain from demos, alone ──────
    @torch.no_grad()
    def evaluate(task):
        Xd = rand_payload(D_DEMOS, S)
        Yd = compose(task, Xd, P)
        chain, expl_r = autonomous_route(model, Xd, Yd, beam=2, max_depth=3, thr=THR)
        _, expl_e = exhaustive_route(model, Xd, Yd, max_depth=3, thr=THR)
        if chain is None:
            return 0.0, "(none)", expl_r, expl_e
        Xq = rand_payload(256, S)
        Yq = compose(task, Xq, P)
        acc = (apply_chain(model, Xq, chain) == Yq).float().mean().item()
        return acc, label(chain), expl_r, expl_e

    print("\n" + "=" * 78)
    print("  AUTONOMOUS ROUTING — learned proposer + sufficiency critic (no expert tag)")
    print("=" * 78)
    print(f"\n  {'task':<20}{'split':<11}{'acc':>6}  {'chain (self-chosen)':<22}"
          f"{'explored':>10}")
    print(f"  {'':<20}{'':<11}{'':>6}  {'':<22}{'router/brute':>10}")
    print("  " + "-" * 72)
    for task in (TRAIN[:3] + HELD):
        acc, chain, er, ee = evaluate(task)
        split = "ZERO-SHOT" if task in HELD else "seen"
        ok = "✓" if chain == label(task) else f"want {label(task)}"
        print(f"  [{label(task):<17}]{split:<11}{acc:>6.3f}  {chain:<22}{f'{er}/{ee}':>10}  {ok}")

    zs = [evaluate(h) for h in HELD]
    zs_acc = float(np.mean([z[0] for z in zs]))
    zs_ok = all(zs[i][1] == label(HELD[i]) for i in range(len(HELD)))

    # ── a concrete realistic decode (held-out triple) ───────────────────────────
    task = HELD[1]
    Xd = rand_payload(D_DEMOS, S)
    chain, _ = autonomous_route(model, Xd, compose(task, Xd, P), beam=2, max_depth=3, thr=THR)
    Xr = realistic_batch(S)
    obf = compose(task, Xr, P)
    if chain:
        rec = apply_chain(model, Xr, chain)
        sample = bytes(rec[0].tolist()).split(b"\x00")[0].decode("latin-1").rstrip()
    else:
        sample = "(no chain)"

    print("\n" + "-" * 78)
    print(f"  zero-shot decode acc = {zs_acc:.3f}   chains correct = {zs_ok}")
    print(f"  realistic decode (held-out [{label(task)}], pipeline {label(chain) if chain else '—'}):")
    print(f"    revealed: {sample!r}")
    if zs_acc > 0.9 and zs_ok:
        print("\n  STAGE 1 VALIDATED: the bloc-routed MoE SLM pre-specializes its experts, composes")
        print("  unseen chains zero-shot, and ROUTES ITSELF from demos with a learned proposer +")
        print("  sufficiency critic — exploring far fewer chains than brute force. Next: stage 2,")
        print("  the fuzzy regime (real obfuscated scripts, semantic critic).")
    else:
        print("\n  Partial — inspect per-expert acc / router / critic threshold before stage 2.")
    print("\n" + "=" * 78)
    print("  DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
