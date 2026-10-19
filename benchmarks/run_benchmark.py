"""Accuracy per layer configuration on the synthetic eval set.

    python benchmarks/run_benchmark.py              # includes embeddings if installed
    python benchmarks/run_benchmark.py --no-embeddings
    python benchmarks/run_benchmark.py --errors     # also list every miss

Prints markdown tables so the numbers can be pasted into the README as-is.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from grounding import AliasLayer, EmbeddingLayer, FuzzyLayer, Match, Reference, Resolver
from grounding.layers import Encoder

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Case:
    ref: str
    query: str
    expected: str | None
    tag: str


@dataclass(frozen=True)
class Score:
    correct: int
    in_ref_found: int
    wrong_accepts: int
    abstained: int


ResolverFactory = Callable[[Reference], Resolver]


def load_cases(path: Path) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def score(cases: list[Case], matches: list[Match]) -> Score:
    pairs = list(zip(cases, matches))
    correct = sum(m.key == c.expected for c, m in pairs)
    in_ref_found = sum(c.expected is not None and m.key == c.expected for c, m in pairs)
    wrong = sum(m.key is not None and m.key != c.expected for c, m in pairs)
    abstained = sum(m.key is None for m in matches)
    return Score(correct, in_ref_found, wrong, abstained)


def run(cases: list[Case], refs: dict[str, Reference], factory: ResolverFactory) -> list[Match]:
    resolvers = {name: factory(ref) for name, ref in refs.items()}
    return [resolvers[c.ref].resolve(c.query) for c in cases]


def configs(encode: Encoder | None) -> dict[str, ResolverFactory]:
    out: dict[str, ResolverFactory] = {
        "alias only": lambda r: Resolver(r, (AliasLayer(r),)),
        "alias + fuzzy, never abstain": lambda r: Resolver(
            r, (AliasLayer(r), FuzzyLayer(r, threshold=0.0, min_margin=0.0, min_length=0))
        ),
        "alias + fuzzy": lambda r: Resolver.build(r),
    }
    if encode is not None:
        out["alias + fuzzy + embeddings"] = lambda r: Resolver(
            r, (AliasLayer(r), FuzzyLayer(r), EmbeddingLayer(r, encode))
        )
    return out


def print_summary(cases: list[Case], results: dict[str, list[Match]]) -> None:
    n = len(cases)
    n_in = sum(c.expected is not None for c in cases)
    print(f"{n} queries: {n_in} have a correct answer in the reference, {n - n_in} do not.\n")
    print("| configuration | accuracy | in-reference found | wrong accepts | abstained |")
    print("|---|---|---|---|---|")
    for name, matches in results.items():
        s = score(cases, matches)
        print(
            f"| {name} | {s.correct}/{n} ({s.correct / n:.0%}) | {s.in_ref_found}/{n_in} "
            f"| {s.wrong_accepts} | {s.abstained} |"
        )
    print()


def print_by_tag(cases: list[Case], results: dict[str, list[Match]]) -> None:
    tags = list(dict.fromkeys(c.tag for c in cases))
    totals = Counter(c.tag for c in cases)
    names = list(results)
    print("Correct answers per query type:\n")
    print("| tag | n | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for tag in tags:
        cells = [
            str(sum(c.tag == tag and m.key == c.expected for c, m in zip(cases, results[name])))
            for name in names
        ]
        print(f"| {tag} | {totals[tag]} | " + " | ".join(cells) + " |")
    print()


def print_threshold_sweep(cases: list[Case], refs: dict[str, Reference]) -> None:
    print("Fuzzy threshold sweep (alias + fuzzy, margin 3):\n")
    print("| threshold | accuracy | in-reference found | wrong accepts |")
    print("|---|---|---|---|")
    for t in (70.0, 75.0, 80.0, 85.0, 90.0, 95.0):
        matches = run(
            cases, refs, lambda r, t=t: Resolver(r, (AliasLayer(r), FuzzyLayer(r, threshold=t)))
        )
        s = score(cases, matches)
        print(f"| {t:.0f} | {s.correct}/{len(cases)} | {s.in_ref_found} | {s.wrong_accepts} |")
    print()


def print_errors(cases: list[Case], results: dict[str, list[Match]]) -> None:
    for name, matches in results.items():
        misses = [(c, m) for c, m in zip(cases, matches) if m.key != c.expected]
        print(f"### {name}: {len(misses)} misses")
        for c, m in misses:
            got = f"{m.key} via {m.method}" if m.key else f"unknown (suggested {m.suggestion})"
            print(f"- {c.query!r}: expected {c.expected}, got {got}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()
    # Queries contain non-ASCII text; Windows consoles default to a legacy codepage.
    sys.stdout.reconfigure(encoding="utf-8")

    refs = {name: Reference.from_json(DATA / f"{name}.json") for name in ("ports", "cargoes")}
    cases = load_cases(DATA / "eval_set.jsonl")

    encode = None
    if not args.no_embeddings:
        try:
            from grounding import sentence_transformer_encoder

            encode = sentence_transformer_encoder()
        except ImportError as exc:
            print(f"(skipping embeddings: {exc})\n")

    results = {name: run(cases, refs, f) for name, f in configs(encode).items()}
    print_summary(cases, results)
    print_by_tag(cases, results)
    print_threshold_sweep(cases, refs)
    if args.errors:
        print_errors(cases, results)


if __name__ == "__main__":
    main()
