"""Unit tests for the deobf engine. Run: python -m pytest tests/ -q"""

import base64
import gzip

import pytest

from deobf import DEFAULT_SKILLS, Critic, discover, execute_pipeline, format_pipeline
from deobf.skills import get_skills


def _skill(name):
    return next(s for s in get_skills() if s.name == name)


# ── skills ────────────────────────────────────────────────────────────────────

def test_structural_decoders_roundtrip():
    payload = b"the quick brown fox" * 3
    assert _skill("base64").apply(base64.b64encode(payload)) == payload
    assert _skill("gunzip").apply(gzip.compress(payload)) == payload


def test_structural_decoders_prune_on_garbage():
    assert _skill("base64").apply(b"\x00\x01\x02not-base64!") is None
    assert _skill("gunzip").apply(b"not gzip") is None


def test_xor_solve_recovers_key():
    state, key = b"hello world", 0x5A
    target = bytes(b ^ key for b in state)
    assert _skill("xor").solve(state, target) == key
    assert _skill("xor").solve(state, state) is None  # key 0 = identity → rejected


def test_stream_cbc_is_inverse_of_running_xor():
    plain = b"ABCDEFGH"
    c = bytearray(plain)
    for i in range(1, len(plain)):
        c[i] = plain[i] ^ c[i - 1]
    assert _skill("stream_cbc").apply(bytes(c)) == plain


# ── critic ────────────────────────────────────────────────────────────────────

def test_critic_modes():
    assert Critic("exact").accepts(b"abc", b"abc")
    assert not Critic("exact").accepts(b"abd", b"abc")
    assert Critic("token", 0.6).accepts(b"abd", b"abc")        # 2/3 bytes match
    assert Critic("levenshtein", 0.8).accepts(b"abcd", b"abxcd")  # one insertion


# ── engine ────────────────────────────────────────────────────────────────────

def _demos(decode_pipeline, payloads):
    enc = {
        "base64": lambda d, _: base64.b64encode(d),
        "gunzip": lambda d, _: gzip.compress(d),
        "xor": lambda d, k: bytes(b ^ k for b in d),
    }
    out = []
    for p in payloads:
        data = p
        for name, param in reversed(decode_pipeline):
            data = enc[name](data, param)
        out.append((data, p))
    return out


def test_discover_simple_chain():
    truth = [("base64", None), ("gunzip", None)]
    demos = _demos(truth, [b"alpha payload", b"bravo payload", b"charlie load"])
    found = discover(demos, DEFAULT_SKILLS, Critic("exact"))
    assert found == truth


def test_discover_with_terminal_param():
    truth = [("base64", None), ("xor", 0x42)]
    demos = _demos(truth, [b"secret one here", b"secret two there", b"third secret!"])
    found = discover(demos, DEFAULT_SKILLS, Critic("exact"))
    assert found == truth
    # generalises to unseen traffic
    new = b"a brand new payload"
    obf = base64.b64encode(bytes(b ^ 0x42 for b in new))
    assert execute_pipeline(found, obf) == new


def test_discover_returns_none_when_unsolvable():
    # revealed unrelated to obfuscated by any chain of our skills
    demos = [(b"\x01\x02\x03", b"completely different"),
             (b"\x04\x05\x06", b"nothing in common")]
    assert discover(demos, DEFAULT_SKILLS, Critic("exact"), max_depth=3) is None


def test_empty_pipeline_when_already_revealed():
    demos = [(b"plain text", b"plain text")]
    assert discover(demos, DEFAULT_SKILLS, Critic("exact")) == []
    assert "identity" in format_pipeline([])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
