"""
Bloc-routed MoE SLM — the paper's architecture as a real (small) neural model.

Forward, for one input byte sequence:

    h = embed(X) + pos
    repeat n_loop times (the LOOP, paper §6):
        for each of B blocs (paper §3):
            router attends over the N experts' partial-forward reps  (paper §4)
            -> pick one expert (top-1, straight-through) for this bloc
            apply Z transformer layers whose FFN IS that expert       (expert switches per bloc)
        logit = head(h)
        h = softmax(logit) @ W_emb + pos                              (token RE-GROUNDING, §6)
    output = last logit

Experts are SHARED across blocs and loop steps (a reusable skill library); routing decides
which skill applies where. Two auxiliary losses keep the experts a useful MoE: L_div pushes
their token distributions apart (paper §5 divergence) and L_bal balances their usage.

The applied depth is up to n_loop * (B*Z) layers, but the discrete routing decisions are only
B per loop step — the bloc structure is what keeps the traversal bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SLMConfig


class Expert(nn.Module):
    """A pre-specializable FFN skill (token->token map in representation space)."""

    def __init__(self, d: int, hidden_mult: int, dropout: float):
        super().__init__()
        h = hidden_mult * d
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InterExpertRouter(nn.Module):
    """Paper §4: the router for a bloc attends over the experts' partial-forward reps O_b.

    O_b[i] = mean_s E_i(ctx)   (one summary vector per expert) — the "partial forward" of §4.3.
    The query comes from the current stream; attention scores over O_b give p(expert).
    """

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, h: torch.Tensor, expert_summaries: torch.Tensor) -> torch.Tensor:
        # h: (B,S,d) ; expert_summaries O_b: (B,N,d) -> probs (B,N)
        q = self.q(h.mean(dim=1))                       # (B,d)
        k = self.k(expert_summaries)                    # (B,N,d)
        scores = torch.einsum("bd,bnd->bn", q, k) * self.scale
        return F.softmax(scores, dim=-1)


class BlocLayers(nn.Module):
    """Z pre-norm transformer layers sharing one (externally selected) expert as their FFN."""

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.Z = cfg.layers_per_bloc
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True,
                                   dropout=cfg.dropout) for _ in range(self.Z)]
        )
        self.n1 = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(self.Z)])
        self.n2 = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(self.Z)])

    def forward(self, h, expert_fn, attn_mask):
        for z in range(self.Z):
            x = self.n1[z](h)
            a, _ = self.attn[z](x, x, x, attn_mask=attn_mask, need_weights=False)
            h = h + a
            h = h + expert_fn(self.n2[z](h))
        return h


@dataclass
class RoutingTrace:
    probs: List[torch.Tensor]          # per (loop,bloc) router distributions (B, N)
    choices: List[torch.Tensor]        # per (loop,bloc) argmax expert id (B,)
    step_logits: List[torch.Tensor]    # per loop-step output logits (B,S,V) for deep supervision


class BlocRoutedMoESLM(nn.Module):
    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_len, cfg.d_model) * 0.02)
        self.experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.ffn_hidden, cfg.dropout) for _ in range(cfg.n_experts)]
        )
        self.blocs = nn.ModuleList([BlocLayers(cfg) for _ in range(cfg.n_blocs)])
        self.router = InterExpertRouter(cfg.d_model)
        self.ctx_norm = nn.LayerNorm(cfg.d_model)
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab)
        self.halt = nn.Linear(cfg.d_model, 1)   # halting / sufficiency signal (PonderNet-style)

    # ---- expert application with straight-through top-1 selection ----
    def _select(self, probs: torch.Tensor):
        """Return (expert_fn, choice). expert_fn applies the top-1 expert per batch row with
        straight-through gradients so the router stays trainable."""
        idx = probs.argmax(dim=-1)                              # (B,)
        onehot = F.one_hot(idx, self.cfg.n_experts).float()     # (B,N)
        weights = onehot + probs - probs.detach()               # straight-through (B,N)

        def expert_fn(x):                                       # x: (B,S,d)
            outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B,N,S,d)
            return torch.einsum("bn,bnsd->bsd", weights, outs)

        return expert_fn, idx

    def _expert_summaries(self, ctx: torch.Tensor) -> torch.Tensor:
        """O_b: each expert's partial-forward summary on the current context (B,N,d)."""
        pooled = ctx.mean(dim=1)                                # (B,d)
        return torch.stack([e(pooled) for e in self.experts], dim=1)

    def forward(self, X: torch.Tensor, n_loop: Optional[int] = None,
                reground: Optional[bool] = None,
                forced_experts: Optional[Sequence[int]] = None) -> RoutingTrace:
        """If `forced_experts` is given (one expert id per loop step), that expert is used for
        all blocs of the step (skill = one bloc-pass), bypassing the router's choice — but the
        router probs are still computed so routing can be supervised against the forced path."""
        cfg = self.cfg
        n_loop = cfg.n_loop if n_loop is None else n_loop
        reground = cfg.reground if reground is None else reground
        B, S = X.shape
        mask = torch.triu(torch.ones(S, S, device=X.device, dtype=torch.bool), diagonal=1)

        h = self.embed(X) + self.pos[:, :S, :]
        probs_all, choices_all, step_logits = [], [], []

        for li in range(n_loop):
            forced_e = None if forced_experts is None else int(forced_experts[li])
            for b in range(cfg.n_blocs):
                ctx = self.ctx_norm(h)
                probs = self.router(h, self._expert_summaries(ctx))   # (B,N)
                if forced_e is None:
                    expert_fn, choice = self._select(probs)
                else:
                    expert_fn = (lambda e: (lambda x: self.experts[e](x)))(forced_e)
                    choice = torch.full((B,), forced_e, device=X.device, dtype=torch.long)
                h = self.blocs[b](h, expert_fn, mask)
                probs_all.append(probs)
                choices_all.append(choice)

            logit = self.head(self.head_norm(h))                      # (B,S,V)
            step_logits.append(logit)
            if reground:
                p = F.softmax(logit, dim=-1)
                h = p @ self.embed.weight + self.pos[:, :S, :]

        return RoutingTrace(probs_all, choices_all, step_logits)

    @torch.no_grad()
    def propose(self, X: torch.Tensor) -> torch.Tensor:
        """Router proposal for the NEXT skill given a (re-grounded) byte state X. Returns the
        expert distribution averaged over the batch (the demos) — the learned proposer."""
        S = X.size(1)
        h = self.embed(X) + self.pos[:, :S, :]
        probs = self.router(h, self._expert_summaries(self.ctx_norm(h)))   # (B,N)
        return probs.mean(0)

    # ---- auxiliary MoE losses (paper §5) ----
    def aux_losses(self, trace: RoutingTrace, ctx_sample: torch.Tensor):
        """Returns (L_bal, L_div). L_div uses a BOUNDED divergence (mean pairwise cosine
        similarity of expert outputs, in [-1,1]) instead of raw -KL, which is unbounded and
        destabilizes joint training by pushing the task expert's distribution to extremes."""
        cfg = self.cfg
        # L_bal: usage frequency per expert should approach 1/N
        usage = torch.stack(trace.probs, 0).mean(dim=(0, 1))          # (N,)
        l_bal = ((usage - 1.0 / cfg.n_experts) ** 2).sum()

        # L_div: minimize pairwise similarity of expert transforms on shared context (bounded)
        pooled = self.ctx_norm(ctx_sample).mean(dim=1)                # (B,d)
        outs = torch.stack([e(pooled) for e in self.experts], dim=1)  # (B,N,d)
        outs = F.normalize(outs, dim=-1)
        sim = torch.einsum("bnd,bmd->bnm", outs, outs)                # (B,N,N) cosine sims
        N = cfg.n_experts
        off = sim.sum(dim=(1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(-1)
        l_div = (off / (N * (N - 1))).mean()                          # mean off-diagonal sim
        return cfg.alpha_bal * l_bal, cfg.lambda_div * l_div
