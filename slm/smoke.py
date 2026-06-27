"""
Smoke test for the bloc-routed MoE SLM: shapes, gradient flow, and that the LOOP does its job
— learn a 2-skill COMPOSITION with per-step deep supervision (the real regime, as in
bmoe_cyber), where each loop step is one skill and the intermediate targets differ.

  skill A: caesar   Y = (X + 7) mod 256
  skill B: subst    Y = perm[X]   (fixed byte permutation)
  task:    B∘A      loop step 1 -> A(X) ; loop step 2 -> B(A(X))

This is the right test for the loop (distinct per-step targets), unlike a single repeated map
which pathologically forces every step to emit the same target. Aux losses are off to isolate
the core. Run:  python -m slm.smoke
"""

import time

import torch
import torch.nn.functional as F

from .config import DEVICE, SLMConfig
from .model import BlocRoutedMoESLM


def main():
    torch.manual_seed(0)
    cfg = SLMConfig(d_model=128, n_experts=6, n_blocs=3, layers_per_bloc=2, n_loop=2,
                    max_len=32, lambda_div=0.0, alpha_bal=0.0)
    model = BlocRoutedMoESLM(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={DEVICE}  params={n_params/1e6:.2f}M  "
          f"L={cfg.n_layers} (B={cfg.n_blocs}×Z={cfg.layers_per_bloc})  "
          f"N={cfg.n_experts} experts  loop={cfg.n_loop}")

    S, BS = 24, 64
    perm = torch.randperm(256, generator=torch.Generator().manual_seed(1)).to(DEVICE)

    def targets(X):
        a = (X + 7) % 256
        b = perm[a]
        return [a, b]                       # per-loop-step deep-supervision targets

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    X = torch.randint(0, 256, (BS, S), device=DEVICE)
    trace = model(X)
    assert trace.step_logits[-1].shape == (BS, S, 256)
    assert len(trace.step_logits) == cfg.n_loop
    print(f"forward OK: {len(trace.step_logits)} step-logits (= n_loop), "
          f"{len(trace.probs)} routing decisions per forward")

    print("\ntraining on subst∘caesar  (step1=caesar, step2=subst) ...")
    t0 = time.time()
    for step in range(500):
        X = torch.randint(0, 256, (BS, S), device=DEVICE)
        tg = targets(X)
        trace = model(X)
        ce = sum(F.cross_entropy(lg.reshape(-1, 256), tg[i].reshape(-1))
                 for i, lg in enumerate(trace.step_logits)) / cfg.n_loop
        opt.zero_grad()
        ce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 499:
            with torch.no_grad():
                acc = (trace.step_logits[-1].argmax(-1) == tg[1]).float().mean().item()
            print(f"  step {step:>3}  ce={ce.item():.3f}  final_acc={acc:.3f}")
    dt = time.time() - t0

    with torch.no_grad():
        X = torch.randint(0, 256, (256, S), device=DEVICE)
        tg = targets(X)
        trace = model(X)
        acc1 = (trace.step_logits[0].argmax(-1) == tg[0]).float().mean().item()
        acc2 = (trace.step_logits[1].argmax(-1) == tg[1]).float().mean().item()
        # which expert each loop step routes to (majority over blocs/batch)
        ch = torch.stack(trace.choices).view(cfg.n_loop, cfg.n_blocs, -1).cpu()
        routed = [int(torch.bincount(ch[s].flatten(), minlength=cfg.n_experts).argmax())
                  for s in range(cfg.n_loop)]
    print(f"\nstep1 (caesar) acc={acc1:.3f}   step2 (subst∘caesar) acc={acc2:.3f}"
          f"   ({dt:.1f}s)   routed experts per step={routed}")
    print("SMOKE OK — loop composes" if acc2 > 0.9 else "SMOKE WEAK — check training")


if __name__ == "__main__":
    main()
