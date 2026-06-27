"""Configuration for the bloc-routed MoE SLM (faithful to the paper, byte-level)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class SLMConfig:
    # vocabulary: byte-level so any payload (binary / network / script) is representable
    vocab: int = 256
    d_model: int = 256
    n_heads: int = 4

    # bloc structure (paper §3): B blocs of Z layers => L = B*Z layers, B discrete routing
    # decisions instead of L. One expert is selected per bloc and used as the FFN across its
    # Z layers, so an expert is a reusable skill applied over a bounded depth.
    n_blocs: int = 3          # B
    layers_per_bloc: int = 2  # Z   (L = 6 layers per loop pass)

    # mixture of experts (paper §4–5): N pre-specializable experts, shared across blocs AND
    # across loop steps (v3 refinement).
    n_experts: int = 6        # N

    # the loop (paper §6): iterate the bloc-stack up to n_loop times with token re-grounding
    # between passes. Total applied depth <= n_loop * L, bounded by the bloc structure.
    n_loop: int = 4
    reground: bool = True

    # sequence
    max_len: int = 128

    # routing / training
    ffn_hidden: int = 4         # expert hidden = ffn_hidden * d_model
    dropout: float = 0.0
    lambda_div: float = 0.01    # expert-divergence weight (L_div)
    alpha_bal: float = 0.01     # load-balance weight (L_bal)
    halt_target: float = 0.0    # ponder/halting regularization weight (0 = off in v1)

    @property
    def n_layers(self) -> int:
        return self.n_blocs * self.layers_per_bloc
