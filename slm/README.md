# slm — the bloc-routed MoE SLM (the paper, as a real neural model)

> Stage 1 of pre-production: the architecture of [`hierarchical_bmoe_v3.tex`](../hierarchical_bmoe_v3.tex)
> as a small (~1.2M-param) byte-level neural model that **routes itself** to deobfuscate
> traffic. Not a toy script — a trainable MoE SLM. Runs on Apple MPS / CPU.

## What it is

A byte-level (vocab 256) sequence transducer faithful to the paper, end to end:

| Paper | Code |
|---|---|
| §3 blocs of Z layers, one expert per bloc | `B` blocs × `Z` layers; the selected expert **is** the FFN across the bloc → routing decisions L→B |
| §4 inter-expert attention routing | `InterExpertRouter` attends over the experts' partial-forward summaries `O_b` |
| §5 guided pre-specialization + L_div/L_bal | `pre_specialize` + bounded divergence / load-balance aux losses |
| §6 (v3) shared-expert loop + token re-grounding | the loop iterates the bloc-stack `n_loop`×, `softmax(head)·W_emb` between passes |
| §6 sufficiency critic / propose-and-verify | `critic_route` over the model's own experts |

Experts are **shared** across blocs and loop steps (a reusable skill library); routing decides
which skill applies where. Applied depth is up to `n_loop × B×Z`, but the discrete decisions are
only `B` per loop step — the bloc structure bounds the traversal.

## Stage 1 result (`python -m slm.experiment`, ~6 min on MPS)

Skills are **length-preserving** byte decoders (the regime a fixed-length transducer owns;
length-changing codecs like base64/gzip stay in the deterministic [`deobf/`](../deobf/README.md)
engine — the two are complementary):

```
caesar  Y=(X-k)%256      xor  Y=X^k       sbox  Y=inv_perm[X]      shift  Y[t]=X[t-1] (context-dep.)
```

Trained on individual skills + most compositions; **two compositions held out**. The model
pre-specializes its experts, composes the unseen chains zero-shot, and routes itself from a few
`(obfuscated→revealed)` demos via the sufficiency critic — no router tag, no expert id:

```
[caes]            seen        1.000   caes
[xor]             seen        1.000   xor
[sbox]            seen        1.000   sbox
[sbox+caes]       ZERO-SHOT   1.000   sbox+caes
[sbox+caes+shif]  ZERO-SHOT   0.998   sbox+caes+shif
zero-shot decode acc = 0.998   chains correct = True

realistic reveal (held-out [sbox+caes]):
  captured: '!sÄ\x00±O3\x02\x03\x010O\x05\x882±O...'
  revealed: 'IEX(New-Object Net.WebCl...'
```

## Honest findings (these shape stage 2)

- **Routing is critic-driven, not proposer-driven, in this regime.** The learned router
  (proposer) is weak on content-ambiguous random bytes (top-1 ≈ 0.1–0.4): a decoded state
  doesn't reveal which skill produced it (the map↔decomposition ambiguity). The critic resolves
  it by verifying against the demos. The proposer's payoff is **stage 2** — real obfuscated
  scripts whose *content* predicts the decoder.
- **Skill learnability is a real constraint.** A modular/CBC keystream (`Y[t]=(X[t]-X[t-1])%256`)
  is grokking-hard over a 256-byte alphabet (stuck at chance), so the context-dependent skill is
  a positional `shift` (attention-learnable, acc 1.0). Modular experts are deferred to stage 2 /
  a smaller alphabet.

## Files

`config.py` (SLMConfig) · `model.py` (`BlocRoutedMoESLM`, router, bloc, loop, aux losses) ·
`data.py` (byte decoders + corpus) · `train.py` (pre-spec + joint) · `route.py` (critic routing
+ proposer metric) · `experiment.py` (the run above) · `smoke.py` (`python -m slm.smoke`).

## Stage 2 (next)

Real obfuscated **scripts** (text, public datasets) for the **fuzzy regime**: semantic
normalizers as experts (not byte-invertible), a learned *"is-this-revealed?"* critic, and the
learned proposer earning its keep where content predicts the decoder.
