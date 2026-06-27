"""
deobf — autonomous attack-pipeline discovery from analyst examples.

Given a few (obfuscated_traffic -> revealed_payload) pairs captured on an incident, the
engine discovers the chain of decoders that maps one to the other (the DEMOS -> PROPOSER ->
EXECUTOR -> CRITIC loop), then applies that pipeline to unknown traffic.

Public API:
    from deobf import discover, execute_pipeline, DEFAULT_SKILLS, Critic, format_pipeline
"""

from .skills import DEFAULT_SKILLS, Skill, get_skills
from .critic import Critic
from .engine import discover, execute_pipeline, format_pipeline, Pipeline

__all__ = [
    "discover",
    "execute_pipeline",
    "format_pipeline",
    "Pipeline",
    "Critic",
    "Skill",
    "DEFAULT_SKILLS",
    "get_skills",
]
