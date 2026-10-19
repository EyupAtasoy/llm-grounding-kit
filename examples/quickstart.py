"""Resolve a handful of messy strings against the bundled port list.

    python examples/quickstart.py
"""

from pathlib import Path

from grounding import Reference, Resolver

ports = Reference.from_json(Path(__file__).resolve().parent.parent / "data" / "ports.json")
resolver = Resolver.build(ports)  # alias + fuzzy; pass encode=... to add embeddings

# Imagine these came out of an LLM extraction step.
extracted = ["Port of Hamburg", "Constanța", "Istambul", "Houston, TX", "Santos", ""]

for match in resolver.resolve_many(extracted):
    if match.is_known:
        print(f"{match.query!r:20} -> {match.canonical} ({match.method}, {match.score:.2f})")
    else:
        hint = f", review suggestion: {match.suggestion}" if match.suggestion else ""
        print(f"{match.query!r:20} -> Unknown{hint}")
