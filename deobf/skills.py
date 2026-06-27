"""
Skills — the decoder primitives the engine composes.

Each skill is a PURE byte->byte function (stateless except where noted). A skill either
applies to raw data with no parameter (`base64`, `gunzip`, ...) or carries a parameter
(`xor_byte` needs a key). Reversibility + statelessness is what makes skills composable: a
chain is just function composition, and the engine can reason about chains freely.

A skill exposes, for the search:
  apply(data, param) -> bytes | None   # None signals "this decoder does not apply" (prune)
  candidates(state)  -> Iterable       # parameter values to try mid-chain (parametric only)
  solve(state, target) -> param | None # O(1) parameter recovery when this is the LAST step
  family                               # parametric skills of one family don't stack usefully
  stateful                             # depends on neighbouring bytes (ordering matters)

`solve` is the key speed trick: when a parametric step is terminal we know its output must
equal the revealed payload, so the parameter is determined directly from (state, target)
instead of grid search (e.g. a XOR key is state[0] ^ target[0], verified constant).
"""

from __future__ import annotations

import base64
import binascii
import codecs
import gzip
import urllib.parse
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

MAX_OUTPUT = 8 * 1024 * 1024  # guard against decompression bombs


@dataclass
class Skill:
    name: str
    _fn: Callable[[bytes, Any], Optional[bytes]]
    parametric: bool = False
    stateful: bool = False
    family: str = ""
    _candidates: Optional[Callable[[bytes], Iterable[Any]]] = None
    _solve: Optional[Callable[[bytes, bytes], Any]] = None
    structural: bool = False  # validates input structure (base64/gzip/...) — strong prune signal
    fmt: Callable[[Any], str] = field(default=lambda p: "")

    def __post_init__(self):
        if not self.family:
            self.family = self.name

    def apply(self, data: bytes, param: Any = None) -> Optional[bytes]:
        try:
            out = self._fn(data, param)
        except (binascii.Error, ValueError, zlib.error, OSError, EOFError, KeyError):
            return None
        if out is None or len(out) > MAX_OUTPUT:
            return None
        return out

    def candidates(self, state: bytes) -> Iterable[Any]:
        return self._candidates(state) if self._candidates else ()

    def solve(self, state: bytes, target: bytes) -> Any:
        return self._solve(state, target) if self._solve else None

    def label(self, param: Any = None) -> str:
        return f"{self.name}({self.fmt(param)})" if self.parametric else self.name


# ── structural decoders (no parameter; either applies cleanly or prunes) ──────

def _b64(data, _):
    s = data.rstrip(b"\n\r ")
    if len(s) < 4 or len(s) % 4 != 0:
        return None
    out = base64.b64decode(s, validate=True)
    return out if out != data else None


def _b64url(data, _):
    s = data.rstrip(b"\n\r ")
    if len(s) < 4 or len(s) % 4 != 0:
        return None
    out = base64.urlsafe_b64decode(s)
    return out if out != data else None


def _b32(data, _):
    s = data.rstrip(b"\n\r ")
    if len(s) < 8 or len(s) % 8 != 0:
        return None
    return base64.b32decode(s)


def _hex(data, _):
    s = data.strip()
    if len(s) < 2 or len(s) % 2 != 0:
        return None
    return binascii.unhexlify(s)


def _url(data, _):
    if b"%" not in data:
        return None
    out = urllib.parse.unquote_to_bytes(bytes(data))
    return out if out != data else None


def _gunzip(data, _):
    if not data.startswith(b"\x1f\x8b"):
        return None
    return gzip.decompress(data)


def _inflate(data, _):
    if len(data) < 2 or data[0] != 0x78:
        return None
    return zlib.decompress(data)


# ── reversible transforms (no parameter) ─────────────────────────────────────

def _reverse(data, _):
    return data[::-1] if data else None


def _rot13(data, _):
    return codecs.encode(data.decode("latin-1"), "rot_13").encode("latin-1")


def _swap_nibbles(data, _):
    return bytes(((b << 4) | (b >> 4)) & 0xFF for b in data)


def _stream_cbc(data, _):
    # undo a rolling/CBC keystream: out[0]=in[0]; out[i] = in[i] XOR in[i-1]
    if len(data) < 2:
        return None
    out = bytearray(data)
    for i in range(len(data) - 1, 0, -1):
        out[i] = data[i] ^ data[i - 1]
    return bytes(out)


# ── parametric transforms ────────────────────────────────────────────────────

def _xor(data, key):
    return bytes(b ^ key for b in data)


def _xor_solve(state, target):
    if len(state) != len(target) or not state:
        return None
    k = state[0] ^ target[0]
    return k if k and all(s ^ k == t for s, t in zip(state, target)) else None


def _add(data, k):
    return bytes((b + k) & 0xFF for b in data)


def _add_solve(state, target):
    if len(state) != len(target) or not state:
        return None
    k = (target[0] - state[0]) & 0xFF
    return k if k and all((s + k) & 0xFF == t for s, t in zip(state, target)) else None


def _sub_apply(data, table):
    try:
        return bytes(table[b] for b in data)
    except (KeyError, IndexError):
        return None


def _sub_solve(state, target):
    """Recover a custom 1-byte substitution alphabet from a demo pair, if consistent."""
    if len(state) != len(target) or not state:
        return None
    mapping: dict = {}
    for s, t in zip(state, target):
        if mapping.setdefault(s, t) != t:
            return None
    if all(k == v for k, v in mapping.items()):  # identity → useless
        return None
    return mapping


def get_skills() -> list[Skill]:
    """The default decoder set (N=14, within the <=15 search budget)."""
    return [
        Skill("base64", _b64, structural=True),
        Skill("base64url", _b64url, structural=True),
        Skill("base32", _b32, structural=True),
        Skill("hex", _hex, structural=True),
        Skill("url", _url, structural=True),
        Skill("gunzip", _gunzip, structural=True),
        Skill("inflate", _inflate, structural=True),
        Skill("reverse", _reverse),
        Skill("rot13", _rot13),
        Skill("swap_nibbles", _swap_nibbles),
        Skill("stream_cbc", _stream_cbc, stateful=True),
        Skill("xor", _xor, parametric=True, family="xor",
              _candidates=lambda s: range(1, 256), _solve=_xor_solve,
              fmt=lambda k: f"0x{k:02x}"),
        Skill("add", _add, parametric=True, family="add",
              _candidates=lambda s: range(1, 256), _solve=_add_solve,
              fmt=lambda k: f"+{k}"),
        Skill("substitution", _sub_apply, parametric=True, family="sub",
              _solve=_sub_solve, fmt=lambda t: "learned"),
    ]


DEFAULT_SKILLS = get_skills()
