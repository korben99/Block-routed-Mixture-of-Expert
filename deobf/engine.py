"""
Engine — the DEMOS -> PROPOSER -> EXECUTOR -> CRITIC -> OUTPUT loop.

`discover(demos, ...)` searches for the shortest decoder chain (with parameters) that turns
every demo's obfuscated bytes into its revealed bytes, then you `execute_pipeline(...)` that
chain on unknown traffic.

The proposer is breadth-first (shortest chains first) over the skill set, kept tractable by:
  * solve() fast-path     — a terminal parametric step is solved O(1) from (state, target);
  * promising-gate        — a mid-chain parametric candidate is only expanded if some
                            structural decoder then validates (XOR/ADD mask structured data);
  * caps                  — at most MAX_PARAMETRIC parametric steps, no same-family stacking;
  * seen-state dedup      — identical demo states reached twice are pruned (kills cycles like
                            reverse∘reverse and xor∘xor);
  * demo[0]-first verify  — cheap check on the first demo, full check on all only when it hits.

These are the practical answers to the spec's Problem 1 (unknown params) and keep depth-4,
N≤15-skill discovery well under the 30 s budget.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, List, Optional, Sequence, Tuple

from .critic import Critic
from .skills import DEFAULT_SKILLS, Skill

Pipeline = List[Tuple[str, Any]]  # ordered [(skill_name, param), ...]

MAX_PARAMETRIC = 2


def execute_pipeline(pipeline: Pipeline, data: bytes,
                     skills: Sequence[Skill] = DEFAULT_SKILLS) -> Optional[bytes]:
    """Apply a discovered pipeline to raw bytes. Returns None if any step fails to apply."""
    by_name = {s.name: s for s in skills}
    cur = data
    for name, param in pipeline:
        cur = by_name[name].apply(cur, param)
        if cur is None:
            return None
    return cur


def format_pipeline(pipeline: Optional[Pipeline],
                    skills: Sequence[Skill] = DEFAULT_SKILLS) -> str:
    if pipeline is None:
        return "(no pipeline found)"
    if not pipeline:
        return "(identity — already revealed)"
    by_name = {s.name: s for s in skills}
    return " › ".join(by_name[name].label(param) for name, param in pipeline)


def _structural_applies(state: bytes, skills: Sequence[Skill]) -> bool:
    return any(s.structural and s.apply(state) is not None for s in skills)


def discover(demos: Sequence[Tuple[bytes, bytes]],
             skills: Sequence[Skill] = DEFAULT_SKILLS,
             critic: Optional[Critic] = None,
             max_depth: int = 4,
             time_budget: float = 30.0) -> Optional[Pipeline]:
    """Discover the shortest pipeline mapping every obfuscated demo to its revealed payload."""
    critic = critic or Critic()
    if not demos:
        return None
    obf = [bytes(o) for o, _ in demos]
    rev = [bytes(r) for _, r in demos]

    def passes(states: Sequence[bytes]) -> bool:
        # demo[0] first (cheap), then the rest
        if not critic.accepts(states[0], rev[0]):
            return False
        return all(critic.accepts(s, r) for s, r in zip(states[1:], rev[1:]))

    if passes(obf):
        return []  # already revealed

    start = (tuple(), tuple(obf), 0, "")          # (pipeline, states, n_param, last_family)
    frontier: deque = deque([start])
    seen = {tuple(obf)}
    t0 = time.time()

    while frontier:
        if time.time() - t0 > time_budget:
            break
        pipeline, states, n_param, last_family = frontier.popleft()
        if len(pipeline) >= max_depth:
            continue

        for skill in skills:
            if skill.parametric:
                # ---- terminal fast-path: solve the parameter from (state, target) ----
                if skill.family != last_family:
                    param = skill.solve(states[0], rev[0])
                    if param is not None:
                        outs = tuple(skill.apply(s, param) for s in states)
                        if all(o is not None for o in outs) and passes(outs):
                            return list(pipeline) + [(skill.name, param)]

                # ---- mid-chain expansion (gated, capped) ----
                if n_param >= MAX_PARAMETRIC or skill.family == last_family:
                    continue
                for param in skill.candidates(states[0]):
                    out0 = skill.apply(states[0], param)
                    if out0 is None:
                        continue
                    # only keep parametric branches that unlock a structural decoder
                    if not _structural_applies(out0, skills):
                        continue
                    outs = tuple(skill.apply(s, param) for s in states)
                    if any(o is None for o in outs) or outs in seen:
                        continue
                    seen.add(outs)
                    new = list(pipeline) + [(skill.name, param)]
                    if passes(outs):
                        return new
                    frontier.append((tuple(new), outs, n_param + 1, skill.family))
            else:
                outs = tuple(skill.apply(s) for s in states)
                if any(o is None for o in outs) or outs in seen:
                    continue
                seen.add(outs)
                new = list(pipeline) + [(skill.name, None)]
                if passes(outs):
                    return new
                frontier.append((tuple(new), outs, n_param, last_family))

    return None
