"""
Command-line interface:  python -m deobf discover <incident.json>

Incident JSON:
{
  "demos":   [ {"obfuscated": "<base64>", "revealed": "<base64>"}, ... ],
  "unknown": [ "<base64>", ... ],          # optional: traffic to decode with the recipe
  "critic":  {"mode": "exact", "threshold": 0.999},   # optional
  "max_depth": 4                           # optional
}
"""

from __future__ import annotations

import argparse
import base64
import json
import sys

from .critic import Critic
from .engine import discover, execute_pipeline, format_pipeline
from .skills import DEFAULT_SKILLS


def _b64(s: str) -> bytes:
    return base64.b64decode(s)


def _show(data: bytes) -> str:
    """Render decoded bytes: utf-8 if printable, else hex."""
    try:
        text = data.decode("utf-8")
        if text.isprintable() or all(c in "\r\n\t" or c.isprintable() for c in text):
            return text
    except UnicodeDecodeError:
        pass
    return data.hex()


def cmd_discover(args) -> int:
    incident = json.load(args.incident)
    demos = [(_b64(d["obfuscated"]), _b64(d["revealed"])) for d in incident["demos"]]
    c = incident.get("critic", {})
    critic = Critic(c.get("mode", "exact"), c.get("threshold", 0.999))
    max_depth = incident.get("max_depth", 4)

    pipeline = discover(demos, DEFAULT_SKILLS, critic, max_depth=max_depth)
    print(f"recipe: {format_pipeline(pipeline)}")
    if pipeline is None:
        print("no decoding pipeline reproduces the demonstrations "
              "(try a larger max_depth, more skills, or a looser critic).")
        return 1

    for i, blob in enumerate(incident.get("unknown", [])):
        decoded = execute_pipeline(pipeline, _b64(blob))
        if decoded is None:
            print(f"unknown[{i}]: <pipeline did not apply>")
        else:
            print(f"unknown[{i}]: {_show(decoded)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="deobf",
                                     description="autonomous attack-pipeline discovery")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover", help="discover the decoder pipeline from an incident file")
    d.add_argument("incident", type=argparse.FileType("r"),
                   help="incident JSON (use '-' for stdin)")
    d.set_defaults(func=cmd_discover)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
