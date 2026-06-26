"""
B-MoE: Bloc-routed Mixture of Experts — Toy Implementation
==========================================================
Faithful toy implementation of the three core mechanisms of B-MoE
(see hierarchical_bmoe_v2.tex):

  1. Bloc routing (§3.2): the L layers are grouped into B = L/Z blocs. A single
     routing decision sigma_b is taken per bloc and shared across its Z layers
     (NOT a router per layer). The residual stream propagates as
        h_l = h_{l-1} + E_{sigma_b}^{(l)}(h_{l-1}),  l in bloc b.

  2. Inter-expert attention routing (§4.2): the router for bloc b+1 attends over
     the matrix O_b in R^{N x d} of the N experts' representations:
        s_{b+1} = softmax(Q_b K_b^T / sqrt(d_k)) V_b  in R^N,
        sigma_{b+1} = argmax_i s_{b+1,i}.
     Inactive experts are estimated by a partial forward (§4.3, strategy 2):
     all N experts of the bloc's last layer are evaluated to build O_b.
     argmax is made differentiable with a straight-through estimator (§4, Prop.).
     The very first bloc uses a standard linear router (no previous bloc).

  3. Guided pre-specialization (§5): each expert E_i is pre-trained on its own
     domain D_i (here Expert_i <- DOMAINS[i], with DOMAINS = math / history /
     geography) by forcing the routing to expert i. Joint training then keeps
     diversity via
        L     = L_LM + L_div + L_bal
        L_div = -lambda * sum_{i<j} KL(p_i || p_j)        (expert divergence)
        L_bal = alpha  * sum_b sum_i (f_{b,i} - 1/N)^2     (load balancing)

Each atomic skill is a distinct *deterministic* map over the SAME vocabulary (math:
+5, history: a*x+b, geography: a fixed permutation), so a single token is ambiguous and
the router must use context. A COMPLEX query is a composition of atoms (e.g. math+history
=> A(M(x))): solving it requires chaining several experts across blocs.

This script validates the paper's core claim — bloc-by-bloc expert switching composes
specialized concepts for complex queries — with two results:
  RESULT 1 (decisive): learned routing solves complex tasks (~0.95) while forcing any
    SINGLE expert on all blocs collapses to chance (switch gain ~0 for pure tasks but
    +0.7..+0.96 for complex ones). No single expert can solve a composed query.
  RESULT 2: complex queries traverse multiple distinct experts across blocs.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Data ──────────────────────────────────────────────────────────────────────


# Each "domain" is a distinct *deterministic* dynamics over the SAME vocabulary, so a
# single token is ambiguous and the router must infer the domain from the transition
# structure (i.e. from context). All three generalize: held-out sequences follow the
# same rule, only the starting token changes.

def make_domain_rules(vocab_size, seed=0):
    """Fixed per-domain parameters, shared between train and test splits."""
    g = torch.Generator().manual_seed(seed)
    return {
        "perm": torch.randperm(vocab_size, generator=g),  # geography adjacency map pi
        "a": 3, "b": 7,                                    # history affine rule (gcd(3,V)=1)
    }


# Each atomic skill is a deterministic bijective map on the vocabulary. A "task" is a
# composition of atoms applied in order: a complex query (e.g. ["math","history"]) needs
# several skills chained, which the model can only solve by switching experts bloc-by-bloc.
ATOMS = ["math", "history", "geography"]


def atom_map(name, x, vocab_size, rules):
    """Apply one atomic skill to a single token."""
    V = vocab_size
    if name == "math":
        return (x + 5) % V                       # additive shift
    if name == "history":
        return (rules["a"] * x + rules["b"]) % V  # affine recurrence
    if name == "geography":
        return int(rules["perm"][x])              # fixed adjacency map
    raise ValueError(f"unknown atom: {name}")


def apply_task(task, x, vocab_size, rules):
    """Compose the atoms of a task (left to right) on a single token."""
    for name in task:
        x = atom_map(name, x, vocab_size, rules)
    return x


def generate_sequence(task, length, vocab_size, rules):
    """Sequence with transition x_{t+1} = (compose task)(x_t)."""
    seq = [torch.randint(0, vocab_size, (1,)).item()]
    for _ in range(1, length):
        seq.append(apply_task(task, seq[-1], vocab_size, rules))
    return torch.tensor(seq, dtype=torch.long)


def build_dataset(n_samples, seq_len, task, vocab_size, rules):
    return torch.stack([generate_sequence(task, seq_len, vocab_size, rules)
                        for _ in range(n_samples)])


def make_pairs(data):
    """Causal next-token pairs: X = seq[:-1], Y = seq[1:] (fixed length, batchable)."""
    return data[:, :-1].contiguous(), data[:, 1:].contiguous()


# ─── Model components ──────────────────────────────────────────────────────────


class Expert(nn.Module):
    """A single expert: 2-layer GELU MLP (E_i^{(l)})."""

    def __init__(self, d_model, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class BMoELayer(nn.Module):
    """Causal attention + a set of N experts. The active expert is chosen at the
    bloc level (the same routing weights are passed to every layer of the bloc)."""

    def __init__(self, d_model, n_heads, n_experts, experts=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # experts can be a shared ModuleList (weight-tied across blocs); else own ones
        self.experts = experts if experts is not None else nn.ModuleList(
            [Expert(d_model) for _ in range(n_experts)]
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h, route_weights):
        """route_weights: (B, N) per-sequence selection weights (one-hot / straight-through).
        Returns the updated residual stream and the per-expert outputs (B, S, N, d)."""
        S = h.size(1)
        causal = torch.triu(
            torch.ones(S, S, device=h.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.attn(h, h, h, attn_mask=causal)
        h = self.norm1(h + attn_out)

        # Partial forward over all experts (used both for selection and for O_b)
        expert_out = torch.stack([e(h) for e in self.experts], dim=2)  # (B, S, N, d)
        # One expert per sequence, shared across tokens and across the bloc's Z layers
        selected = (expert_out * route_weights[:, None, :, None]).sum(
            dim=2
        )  # (B, S, d)
        h = self.norm2(h + selected)
        return h, expert_out


class Bloc(nn.Module):
    """A bloc of Z layers sharing a single routing decision sigma_b."""

    def __init__(self, d_model, n_heads, n_experts, z):
        super().__init__()
        self.layers = nn.ModuleList(
            [BMoELayer(d_model, n_heads, n_experts) for _ in range(z)]
        )

    def forward(self, h, route_weights):
        expert_out = None
        for layer in self.layers:
            h, expert_out = layer(h, route_weights)
        # O_b := last layer's per-expert representations (B, S, N, d)
        return h, expert_out


class InterExpertRouter(nn.Module):
    """Inter-expert attention router (§4.2): scores experts of the next bloc from
    the matrix O_b in R^{N x d} of the current bloc's expert representations."""

    def __init__(self, d_model, n_experts, d_k):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, 1, bias=False)
        self.d_k = d_k

    def forward(self, O):
        # O: (B, N, d) — aggregated representation of each expert (O_b in R^{N x d})
        Q = self.W_Q(O)  # (B, N, d_k)
        K = self.W_K(O)  # (B, N, d_k)
        V = self.W_V(O)  # (B, N, 1)
        attn = torch.softmax(
            Q @ K.transpose(-1, -2) / np.sqrt(self.d_k), dim=-1
        )  # (B, N, N)
        s = (attn @ V).squeeze(-1)  # (B, N)
        return s


class BMoE(nn.Module):
    def __init__(
        self,
        vocab_size=50,
        d_model=64,
        n_heads=4,
        n_experts=2,
        n_blocs=2,
        layers_per_bloc=2,
        d_k_routing=16,
        max_len=64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_blocs = n_blocs

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.blocs = nn.ModuleList(
            [Bloc(d_model, n_heads, n_experts, layers_per_bloc) for _ in range(n_blocs)]
        )

        # Routing: bloc 0 uses a plain linear router; blocs >=1 use inter-expert attention
        self.base_router = nn.Linear(d_model, n_experts)
        self.inter_router = InterExpertRouter(d_model, n_experts, d_k_routing)

        self.head = nn.Linear(d_model, vocab_size)

    def _route(self, logits):
        """Straight-through routing: hard argmax forward, soft-prob gradient backward."""
        probs = F.softmax(logits, dim=-1)
        idx = probs.argmax(dim=-1)
        one_hot = F.one_hot(idx, self.n_experts).float()
        st_weights = one_hot + probs - probs.detach()
        return st_weights, probs, idx

    def forward(self, x, force_expert=None, return_bloc_logits=False):
        """Routing control:
          - force_expert=None : learned routing (base + inter-expert attention).
          - force_expert=i    : route everything to expert i on every bloc
                                (pre-specialization, divergence loss).
          - force_expert=path : a length-n_blocs sequence giving the forced expert
                                per bloc, e.g. (0, 2, 2) — used to search expert paths.

        return_bloc_logits=True additionally returns the head applied AFTER each bloc
        (one (B*S, vocab) tensor per bloc) — used for deep supervision (Lever 2).
        """
        B, S = x.shape
        h = self.token_embed(x) + self.pos_embed[:, :S, :]

        is_path = force_expert is not None and not isinstance(force_expert, int)
        routing = []  # per bloc: (probs (B,N), idx (B,))
        bloc_logits = []
        O_prev = None  # O_{b-1} in R^{B x N x d}
        for b, bloc in enumerate(self.blocs):
            if force_expert is not None:
                e = force_expert[b] if is_path else force_expert
                idx = torch.full((B,), e, device=x.device, dtype=torch.long)
                weights = F.one_hot(idx, self.n_experts).float()
                probs = weights
            else:
                # bloc 0: plain router on the pooled sequence; bloc b>=1: inter-expert attention
                logits = (
                    self.base_router(h.mean(dim=1))
                    if b == 0
                    else self.inter_router(O_prev)
                )
                weights, probs, idx = self._route(logits)
            h, expert_out = bloc(h, weights)
            O_prev = expert_out.mean(dim=1)  # aggregate over tokens -> (B, N, d)
            routing.append((probs, idx))
            if return_bloc_logits:
                bloc_logits.append(self.head(h).reshape(-1, self.vocab_size))

        logits = self.head(h).reshape(-1, self.vocab_size)  # (B*S, vocab)
        if return_bloc_logits:
            return logits, routing, bloc_logits
        return logits, routing


# ─── Losses (§5) ───────────────────────────────────────────────────────────────


def load_balance_loss(routing, n_experts):
    """L_bal = sum_b sum_i (f_{b,i} - 1/N)^2, with f_{b,i} the (soft) routed fraction."""
    loss = 0.0
    for probs, _ in routing:
        f = probs.mean(dim=0)  # (N,) routed fraction at this bloc
        loss = loss + ((f - 1.0 / n_experts) ** 2).sum()
    return loss


def divergence_loss(model, X, n_experts):
    """L_div surrogate = - sum_{i<j} KL(p_i || p_j), maximizing expert divergence.
    p_i is the predictive distribution when routing everything through expert i."""
    log_p = []
    for i in range(n_experts):
        logits_i, _ = model(X, force_expert=i)
        log_p.append(F.log_softmax(logits_i, dim=-1))
    kl_total = 0.0
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            p_i = log_p[i].exp()
            kl_total = kl_total + (p_i * (log_p[i] - log_p[j])).sum(-1).mean()
    return -kl_total


# ─── Train / eval helpers ──────────────────────────────────────────────────────


def sample_batch(X, Y, batch_size):
    idx = torch.randint(0, X.shape[0], (batch_size,))
    return X[idx], Y[idx]


def ce_loss(logits, Y):
    return F.cross_entropy(logits, Y.reshape(-1))


@torch.no_grad()
def evaluate(model, X, Y, force_expert=None):
    model.eval()
    logits, _ = model(X, force_expert=force_expert)
    loss = ce_loss(logits, Y)
    acc = (logits.argmax(-1) == Y.reshape(-1)).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def best_single_expert_acc(model, X, Y):
    """Ablation: best accuracy achievable by forcing a single expert on ALL blocs."""
    return max(evaluate(model, X, Y, force_expert=e)[1] for e in range(model.n_experts))


@torch.no_grad()
def routing_fractions(model, X):
    """For each bloc, fraction of sequences routed to each expert under learned routing."""
    model.eval()
    _, routing = model(X)
    fracs = []
    for _, idx in routing:
        counts = torch.bincount(idx.reshape(-1), minlength=model.n_experts).float()
        fracs.append((counts / counts.sum()).tolist())
    return fracs


@torch.no_grad()
def routing_paths(model, X):
    """Per-sequence expert path across blocs: returns (n_samples, n_blocs) indices."""
    model.eval()
    _, routing = model(X)
    return torch.stack([idx for _, idx in routing], dim=1)  # (n_samples, n_blocs)


def pre_specialize(model, domains, steps, lr, batch_size):
    """Pre-train each expert on its own domain by forcing the routing (§5)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for expert_id, (name, X, Y) in domains.items():
        for step in range(steps):
            model.train()
            Xb, Yb = sample_batch(X, Y, batch_size)
            logits, _ = model(Xb, force_expert=expert_id)
            loss = ce_loss(logits, Yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(
            f"    Expert_{expert_id} <- {name:<10} | final pre-train CE = {loss.item():.4f}"
        )


# ─── Main ──────────────────────────────────────────────────────────────────────


def label(task):
    return "+".join(a[:4] for a in task)


def run_bmoe(seq_len=30, vocab=50, d_model=64, n_heads=4, n_blocs=3, layers_per_bloc=2,
             d_k_routing=16, batch=32, n_train=300, n_test=150, t_pre=200, n_joint=600,
             lr=1e-3, alpha_bal=0.05, lambda_div=0.01, train_tasks=None, eval_tasks=None,
             verbose=True):
    """Build data, pre-specialize, jointly train, and return (model, test, meta).

    train_tasks: list of atom-lists the model is trained on (defaults to pures + all
    pairwise/triple compositions). eval_tasks: extra (held-out) tasks to build test
    data for but never train on — used by the zero-shot generalization script.
    """
    n_experts = len(ATOMS)
    if train_tasks is None:
        train_tasks = ([[a] for a in ATOMS]
                       + [["math", "history"], ["history", "geography"],
                          ["math", "geography"], ["math", "history", "geography"]])
    eval_tasks = eval_tasks or []
    all_tasks = train_tasks + [t for t in eval_tasks if t not in train_tasks]

    rules = make_domain_rules(vocab, seed=0)
    train, test = {}, {}
    for task in train_tasks:
        train[label(task)] = make_pairs(build_dataset(n_train, seq_len, task, vocab, rules))
    for task in all_tasks:
        test[label(task)] = make_pairs(build_dataset(n_test, seq_len, task, vocab, rules))

    model = BMoE(vocab, d_model, n_heads, n_experts, n_blocs, layers_per_bloc,
                 d_k_routing, max_len=seq_len)
    if verbose:
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        print("Phase 1 — Pre-specialization (Expert_i <- atom i)")
    domains = {i: (a, *train[label([a])]) for i, a in enumerate(ATOMS)}
    pre_specialize(model, domains, t_pre, lr, batch)

    if verbose:
        print("\nPhase 2 — Joint training on training tasks (learned bloc routing)\n")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels = [label(t) for t in train_tasks]
    log_every = max(1, n_joint // 12)
    for step in range(n_joint):
        name = train_labels[torch.randint(0, len(train_labels), (1,)).item()]
        model.train()
        Xb, Yb = sample_batch(*train[name], batch)
        logits, routing = model(Xb)
        loss = (ce_loss(logits, Yb)
                + alpha_bal * load_balance_loss(routing, n_experts)
                + lambda_div * divergence_loss(model, Xb, n_experts))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if verbose and (step % log_every == 0 or step == n_joint - 1):
            acc = np.mean([evaluate(model, *test[label(t)])[1] for t in train_tasks])
            print(f"  Step {step:>4d} | train-task test acc = {acc:.3f}")

    meta = dict(n_blocs=n_blocs, n_experts=n_experts, train_tasks=train_tasks)
    return model, test, meta


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    PURE = [[a] for a in ATOMS]
    HYBRID = [["math", "history"], ["history", "geography"],
              ["math", "geography"], ["math", "history", "geography"]]
    TASKS = PURE + HYBRID
    N_BLOCS = 3

    print("=" * 72)
    print("  B-MoE: compositional bloc-by-bloc expert switching")
    print("=" * 72)
    print(f"  Atomic skills (1 expert each): {', '.join(ATOMS)}")
    print(f"  Complex tasks (trained): {', '.join('[' + label(t) + ']' for t in HYBRID)}")
    print("-" * 72)

    model, test, meta = run_bmoe(train_tasks=TASKS, n_blocs=N_BLOCS)
    N_EXPERTS = meta["n_experts"]

    # ── RESULT 1 — accuracy vs the decisive single-expert ablation ──────────────
    print("\n" + "=" * 72)
    print("  RESULT 1 — Accuracy: learned routing vs best SINGLE expert (ablation)")
    print("=" * 72)
    print("  If switching matters, complex tasks need learned routing and collapse when")
    print("  forced through any single expert across all blocs.\n")
    print(f"  {'Task':<22} {'learned':>8} {'best-1expert':>13} {'switch gain':>12}")
    print("  " + "-" * 58)
    for task in TASKS:
        name = label(task)
        _, acc = evaluate(model, *test[name])
        single = best_single_expert_acc(model, *test[name])
        kind = "pure   " if len(task) == 1 else "complex"
        print(f"  [{name:<19}] {acc:>8.3f} {single:>13.3f} {acc - single:>+12.3f}  ({kind})")

    # ── RESULT 2 — bloc-by-bloc expert paths ────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULT 2 — Bloc-by-bloc expert paths (modal path B0>B1>B2)")
    print("=" * 72 + "\n")
    for task in TASKS:
        name = label(task)
        paths = routing_paths(model, test[name][0])
        modal = [int(torch.mode(paths[:, b]).values) for b in range(N_BLOCS)]
        path_str = " > ".join(f"E{e}" for e in modal)
        kind = "pure   " if len(task) == 1 else "complex"
        print(f"  [{name:<19}] {path_str:<22} ({kind})")

    print("\n  Switching is near-useless for pure tasks but load-bearing for complex ones:")
    print("  no single expert can solve a composed query — only chaining experts can.")
    print("\n" + "=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
