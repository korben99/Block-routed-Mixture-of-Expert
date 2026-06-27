# Autonomous Compositional Routing for Mixture-of-Experts

**A model that, given a few examples of a task, picks *which of its specialized skills to
apply and in what order* — on its own, with no router and no task tokens — and composes
them to solve compositions it was never trained on.**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Device](https://img.shields.io/badge/runs%20on-CPU%20%2F%20Apple%20MPS-success)

> This is the **research-frontier** README: the loop architecture and the autonomous
> "route-yourself" result. The original **B-MoE bloc-routing** validation (paper §3–5,
> ablations, real-token LM) lives in [`README.old.md`](README.old.md).

---

## TL;DR

We started from B-MoE (route experts by *blocs of layers*) and asked a harder question:
can a model **compose specialized skills to solve a task it has never seen**, and **decide
the composition itself**? After a chain of experiments (kept in the repo — the dead ends are
the contribution), the answer is yes, with one architecture:

```
input → [CORE: a few transformer layers]      build a stationary working representation
      → LOOP:
            propose & apply a skill (expert),
            RE-GROUND the stream to token space     ← every expert sees a clean token
            a CRITIC judges "is the goal reached?"   ← stop when sufficient
      → [RENDER head] → output
```

Two ingredients make it work:

1. **A loop of *shared* experts with token re-grounding.** The core does the low→high
   abstraction lift once; the loop then applies skills at one stationary level, re-grounding
   to token space between steps so each expert is a reusable `token→token` map. This makes
   skills **composable** (Universal-Transformer recurrence + MoE routing).
2. **A sufficiency critic + few-shot demos for routing.** The task is given by examples
   (`X→Y`), never by an expert-id token. The model **proposes chains of its own primitives**
   and a **critic judges whether the goal is reached** by verifying each chain on the demos.
   This resolves the core ambiguity (a demonstration shows the net *map*, not its
   *decomposition*) by *composing primitives and verifying against the goal*.

**Result (synthetic toy):** held-out compositions — an unseen **pair** and an unseen
**triple** — are solved **zero-shot at 1.00 accuracy**, with the model choosing the chain
itself. No router, no task tokens.

---

## The experience (how we got there)

Every step is a runnable script. Task: solve a composition **never trained on** (skills seen
individually / in other combinations; the target combination held out).

| Step | Idea | Held-out triple | Lesson |
|---|---|:---:|---|
| `bmoe_compose.py` | vanilla bloc-MoE | ~0.01 | compositions learned as **holistic maps**, not skills |
| `bmoe_diagnose.py` | enumerate all expert paths | ~0.06 | a **composability** failure, not a routing one |
| `bmoe_lever1.py` | supervised atom→bloc routing | ~0.02 | experts **co-adapt to their predecessor** |
| `bmoe_lever2.py` | + deep supervision | pair ✅ 0.64 | a *soft* canonical interface fixes pairs only |
| `bmoe_loop.py` | shared experts in a loop + **token re-grounding** | **1.00** | a **hard token interface** makes skills reusable → composes |
| `bmoe_poc.py` | dual-stream loop (carry **context**) | — | a context-dependent skill works only with the context stream (0.97 vs 0.06) |
| `bmoe_poc2/3.py` | parametric / decentralized router from demos | ~0.03 | a head **can't infer the chain** from a net map (ambiguous) |
| **`bmoe_poc4.py`** | **sufficiency critic + few-shot demos** | **1.00** | **the model routes itself**, zero-shot, no token |

Full narrative and the bloc-routing ablations: [`README.old.md`](README.old.md).

---

## Quickstart

```bash
pip install -r requirements.txt          # torch, numpy

python bmoe_loop.py    # composability: loop + re-grounding cracks the triple   (~35 s)
python bmoe_poc.py     # dual-stream loop with a context-dependent skill (MPS)  (~40 s)
python bmoe_poc4.py    # THE result: autonomous critic-guided routing (MPS)     (~20 s)
```

`bmoe_poc4.py` prints, for each held-out task, the **chain the model chose by itself** and
its zero-shot accuracy. The journey scripts (`bmoe_compose/diagnose/lever1/lever2`) and the
original bloc-routing demos (`toyBMoE.py`, `bmoe_text.py`) are documented in `README.old.md`.
Apple-Silicon GPU (MPS) is used automatically when available.

---

## Scaling it up (toy → LLM)

The toy proves the **mechanism**. Two components are deliberately simple and must be swapped
to scale; **the rest of the architecture transfers unchanged.**

**Architecture mapping**

| Toy component | LLM-scale counterpart |
|---|---|
| Core (a few transformer layers) | the pretrained backbone (lower layers) producing contextual states |
| Shared experts (small MLP skills) | specialized modules — MoE experts, **LoRA adapters**, or skill-tuned heads — reused across loop steps |
| Token **re-grounding** between steps | decode intermediate results to tokens that re-enter the context — i.e. a **chain-of-thought / scratchpad** loop (re-grounding is literally CoT) |
| Render head | the LM head |
| Few-shot demos as task signal | the **prompt** (instructions + few-shot examples) — already how LLMs receive tasks |

**The two swaps**

1. **Exact-match critic → learned/semantic verifier.** Replace `chain(X_demo)==Y_demo` with
   a model that judges *"is this answer complete/correct for the task?"* — a trained
   **verifier/reward model**, or the LLM self-critiquing. This is where the experts' own
   intelligence is required (a tiny toy MLP cannot self-judge; an LLM partly can). Train it
   on `(task, candidate) → sufficient?` labels (preference data / RL).
2. **Exhaustive `N^K` search → guided proposer.** Replace enumeration with the model
   **proposing the next skill/tool** given (state, goal), explored with **beam / best-of-N**
   and re-ranked by the critic. This is exactly an **agent loop**: propose → apply → verify →
   continue or stop. Halting comes from the critic (adaptive computation, no fixed depth).

**Training recipe at scale**

1. **Pre-specialize** each expert/adapter on its domain and keep them distinct (the B-MoE
   pre-specialization + divergence idea — see `README.old.md`).
2. Train the **verifier** (sufficiency critic).
3. Bootstrap the **proposer**: imitate good chains found by search/self-play, then improve
   with RL using the verifier as reward.
4. **Evaluate** on held-out compositional tasks: skills seen individually, **combinations
   unseen** — the zero-shot compositional-generalization test, now on real tasks.

**Why this should hold:** composability comes from re-grounding (✓ proven), and re-grounding
*is* chain-of-thought at scale; autonomy comes from propose-and-verify (✓ proven with an
exact critic), and propose-and-verify *is* the agent/verifier paradigm at scale. The toy
removes every confound and shows the loop is sound; scaling supplies the missing
*intelligence of the judge*.

---

## Honest scope

- ✅ Proven: the loop + re-grounding makes skills compose zero-shot; a sufficiency critic +
  few-shot demos lets the model route itself, zero-shot, with no router and no task token.
- ⚠️ The critic is exact token-match on the demos and routing is exhaustive search — robust
  for a handful of skills, **not** the scalable form (see swaps above).
- ⚠️ The *semantic* "is-this-complete" judgment is **not** demonstrated by tiny toy experts;
  it is expected to be an LLM-scale property. We separate the **mechanism** (toy-proven)
  from the **judge's intelligence** (scale-dependent) and do not over-claim.

## License

Released under the [MIT License](LICENSE). Origin & formal write-up:
[`hierarchical_bmoe_v2.tex`](hierarchical_bmoe_v2.tex), [`README.old.md`](README.old.md).
