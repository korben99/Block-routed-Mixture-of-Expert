# B-MoE — Bloc-routed Mixture of Experts

**Routing experts by *blocs of layers* instead of layer-by-layer, with inter-expert
attention and guided pre-specialization — so a model can chain specialized experts
across depth to solve compositional queries.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Status](https://img.shields.io/badge/status-research%20toy-orange)

---

## TL;DR

Standard Mixture-of-Experts routes **independently at every layer**, with no memory of
past routing decisions and no guarantee that experts specialize. **B-MoE** changes three
things:

1. **Bloc routing** — one expert is chosen per *bloc* of `Z` layers (not per layer),
   cutting the number of discrete decisions from `L` to `B = L/Z` and stabilizing the
   gradient.
2. **Inter-expert attention routing** — the next bloc's router *attends over the experts'
   representations* (`softmax(QKᵀ/√d_k)·V` over `O_b ∈ ℝ^{N×d}`), instead of a scalar gate.
3. **Guided pre-specialization** — each expert is pre-trained on its own domain *before*
   joint training, then kept distinct with a divergence loss.

This repo contains a **minimal, fully-runnable toy** (`toyBMoE.py`, ~450 lines, CPU, <90 s)
that implements all three mechanisms and **empirically validates the central claim**:

> 🧩 **A model can switch experts bloc-by-bloc to compose specialized skills it could not
> solve with any single expert.**

On composed tasks (e.g. `math → history`), learned bloc routing reaches **~95–99 %**
accuracy, while forcing **any single expert across all blocs collapses to chance**
(2–25 %). The ability to *chain* experts is what unlocks the composition.

---

## Headline result — bloc-by-bloc composition

We build three **atomic skills** as deterministic maps over the same vocabulary
(`math: x+5`, `history: a·x+b`, `geography: a fixed permutation π`), pre-specialize one
expert per skill, then train on **compositions** of these skills. A complex query like
*"GDP growth over 200 years"* is modeled as `geography(history(math(x)))` — it needs
several skills chained.

**Ablation (the decisive test):** force a single expert on every bloc vs. let the router
switch.

| Task | Learned routing | Best **single** expert | Switch gain |
|------|:---:|:---:|:---:|
| `math` (pure) | 0.989 | 0.974 | **+0.01** |
| `geography` (pure) | 0.992 | 0.981 | **+0.01** |
| `math + history` | 0.990 | 0.033 | **+0.96** |
| `math + geography` | 0.960 | 0.023 | **+0.94** |
| `math + history + geography` | 0.973 | 0.255 | **+0.72** |

➡️ **Pure tasks** need one expert (switching adds nothing). **Complex tasks** are
*unsolvable* by any single expert and only work when the model routes through different
experts across its blocs. That is the B-MoE thesis, demonstrated end-to-end.

> All scores are on a **held-out test split** — every domain is a deterministic rule, so
> structured tasks genuinely generalize (this is real generalization, not memorization).

---

## The three mechanisms (mapped to the code)

| Paper (`hierarchical_bmoe_v2.tex`) | Implementation in `toyBMoE.py` |
|---|---|
| **§3.2 Bloc routing** — one `σ_b` per bloc, shared over `Z` layers | `Bloc` passes a single per-sequence routing decision to all its `BMoELayer`s |
| **§4.2 Inter-expert attention** — `s_{b+1}=softmax(QKᵀ/√d_k)V` over `O_b ∈ ℝ^{N×d}` | `InterExpertRouter`; `O_b` built by partial forward of all experts (§4.3, strategy 2) |
| **§4 Prop. (differentiable argmax)** | straight-through estimator in `BMoE._route` |
| **§5 Pre-specialization + `L_div` + `L_bal`** | `pre_specialize()`, `divergence_loss()`, `load_balance_loss()` |

The full formal write-up is in [`hierarchical_bmoe_v2.tex`](hierarchical_bmoe_v2.tex).

---

## Quickstart

```bash
pip install -r requirements.txt   # torch, numpy

# core demos
python toyBMoE.py      # composition + decisive single-expert ablation     (~70 s CPU)
python bmoe_text.py    # REAL char-level text: multi-register LM            (~100 s CPU)

# research journey toward zero-shot composition (see "Research journey" below)
python bmoe_compose.py  # zero-shot probe: vanilla model fails              (~70 s CPU)
python bmoe_diagnose.py # path enumeration: it's a composability failure    (~50 s CPU)
python bmoe_lever1.py   # supervised atom→bloc routing — still fails        (~45 s CPU)
python bmoe_lever2.py   # + deep supervision — the PAIR composes            (~45 s CPU)
python bmoe_loop.py     # loop + token re-grounding — PAIR & TRIPLE compose (~35 s CPU)
```

- **`toyBMoE.py`** — pre-specializes one expert per atomic skill, trains on pure + all
  compositions, prints the ablation table above and the per-bloc expert paths. Exposes the
  shared `BMoE` model + `run_bmoe()` reused by the journey scripts.
- **`bmoe_text.py`** — a character-level B-MoE on a small multi-register corpus (weather /
  finance / recipe): per-register accuracy, routing specialization, single-expert ablation.

---

## Real tokens

A character-level B-MoE learns genuine multi-register text (~0.88 next-char accuracy), and
forcing any single expert collapses every input to ~chance — bloc-by-bloc switching is
load-bearing on real tokens too (`bmoe_text.py`).

---

## Research journey: cracking zero-shot composition

We kept every step in the repo — including the dead ends — because the *path* is the
contribution. The question: can the model solve a composition it was **never trained on**
(train on `math+history` & `history+geography`, test on the never-seen `math+geography` and
the full triple `math+history+geography`)?

| Step | Idea | Triple (zero-shot) | What we learned |
|---|---|:---:|---|
| `bmoe_compose.py` | vanilla B-MoE | ~0.01 | compositions are learned as **holistic maps**, not reusable skills |
| `bmoe_diagnose.py` | enumerate all expert paths | best ~0.06 | it's a **composability** failure, not a routing one |
| `bmoe_lever1.py` | supervised atom→bloc routing | ~0.02 | experts **co-adapt to their predecessor** (non-canonical interface) |
| `bmoe_lever2.py` | + deep supervision | ~0.02 (pair ✅ 0.64) | a *soft* canonical interface fixes pairs, not the triple |
| `bmoe_loop.py` (continuous) | shared experts in a loop | ~0.08 | the residual **vector** still isn't canonical |
| **`bmoe_loop.py` (re-grounded)** | **+ token re-grounding** | **1.00** | **a hard token interface makes skills reusable — it composes** |

**The architecture that works** (and the target for the POC):

```
input → [CORE: a few transformer layers]        establish a working representation
      → LOOP k times:
            pick an expert (skill), apply it,
            RE-GROUND the stream to token space  ← the key: every expert sees a clean token
      → [RENDER head] → output
```

A **core** absorbs the low→high abstraction lift once; a **loop of shared experts** then
applies skills at a single, stationary abstraction level, **re-grounding to token space**
between steps so each expert is a reusable `token→token` map. This is Universal-Transformer
recurrence + B-MoE routing — faithful to the paper's goal ("use the best of all experts for
complex tasks"), while resolving the fact that transformer layers specialize by depth.

Result on the toy: the held-out **pair and triple both compose at 1.00** (vs 0.07 for the
continuous interface), and a **path search auto-discovers the correct chain** (triple
`math→history→geography`, rank 1/64) — i.e. the model finds the skill sequence on its own,
scored by likelihood (label-free) at inference.

> Honest scope: proven on the **synthetic** atomic-skill toy; routing is forced for the
> composability proof (search is shown to recover it). The POC extends this to real tokens,
> a context-carrying re-grounding, and learned/halting routing (see roadmap).

---

## What this is (and isn't)

- ✅ A faithful, readable **proof-of-concept** that the three B-MoE mechanisms work and that
  bloc-by-bloc switching composes specialized skills.
- ✅ A controlled, **honest** benchmark: held-out test splits, single-expert ablations, and
  a transparent negative result on zero-shot composition.
- ❌ Not a pretrained language model, not tuned for SOTA, not yet sparse/efficient at scale.

---

## Roadmap to a real PoC

- [x] Compositional **solving** via bloc switching (trained compositions) — `toyBMoE.py`.
- [x] Real tokens / a small natural-language corpus — `bmoe_text.py`.
- [x] **Zero-shot compositional generalization** — solved on the toy by the loop +
      token-re-grounding architecture (`bmoe_loop.py`).
- [ ] **POC**: loop architecture on real tokens, carrying context alongside the re-grounded
      token (dual stream), with **learned routing + halting** (ACT/PonderNet style).
- [ ] Likelihood-scored **path search** at inference (label-free chain discovery).
- [ ] Sparse experts (true top-1 compute) and a scaling study over loop depth.
- [ ] Interactive demo (HF Space) visualizing the discovered skill chain of a query.

---

## Citation

If this is useful, please cite the technical report (`BMOE paper v1.pdf`):

```bibtex
@techreport{bmoe2026,
  title  = {B-MoE: Bloc-routed Mixture of Experts and Guided Expert Specialization},
  year   = {2026},
  note   = {Technical report}
}
```

## License

Released under the [MIT License](LICENSE).
