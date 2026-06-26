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
python toyBMoE.py
```

You'll see three phases: **pre-specialization** (one expert per skill), **joint training**
(pure + complex tasks), and an **evaluation** printing the ablation table above plus the
per-bloc expert paths. Runs on CPU in ~60–90 s.

---

## What this is (and isn't)

- ✅ A faithful, readable **proof-of-concept** that the B-MoE mechanisms work and that
  bloc-by-bloc switching enables composition.
- ✅ A controlled, **honest** benchmark: held-out test splits and single-expert ablations.
- ❌ Not a pretrained language model, not tuned for SOTA, not yet sparse/efficient at scale.

---

## Roadmap to a real PoC

- [ ] **Compositional generalization**: train on pairs, test zero-shot on the unseen triple.
- [ ] Real tokens / a small natural-language corpus instead of synthetic maps.
- [ ] Sparse experts (true top-1 compute) with the §4.3 *learned* inactive-expert estimator.
- [ ] Scaling study over `Z` and `B` (the gradient-stability proposition, §6).
- [ ] Interactive demo (HF Space) visualizing the per-bloc expert path of a query.

---

## Citation

If this is useful, please cite the technical report (`hierarchical_bmoe_v2.tex`):

```bibtex
@techreport{bmoe2026,
  title  = {B-MoE: Bloc-routed Mixture of Experts and Guided Expert Specialization},
  year   = {2026},
  note   = {Technical report}
}
```

## License

Choose a license before publishing (MIT recommended for a research toy) and add a `LICENSE`
file.
