"""Run the layers in order and return the first accepted candidate, or Unknown."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .layers import AliasLayer, Candidate, EmbeddingLayer, Encoder, FuzzyLayer, Layer
from .reference import Reference

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Match:
    query: str
    key: str | None
    canonical: str | None
    score: float
    method: str
    # What each layer proposed but did not accept. Kept for human review;
    # never used to fill in `key`.
    rejected: tuple[Candidate, ...] = ()

    @property
    def is_known(self) -> bool:
        return self.key is not None

    @property
    def suggestion(self) -> str | None:
        """Guess from the cheapest layer that proposed anything (see `rejected` for all)."""
        return self.rejected[0].key if self.rejected else None


@dataclass(frozen=True)
class Resolver:
    reference: Reference
    layers: tuple[Layer, ...]

    @classmethod
    def build(
        cls,
        reference: Reference,
        *,
        fuzzy: bool = True,
        encode: Encoder | None = None,
    ) -> Resolver:
        layers: list[Layer] = [AliasLayer(reference)]
        if fuzzy:
            layers.append(FuzzyLayer(reference))
        if encode is not None:
            layers.append(EmbeddingLayer(reference, encode))
        return cls(reference=reference, layers=tuple(layers))

    def resolve(self, raw: str | None) -> Match:
        text = raw or ""
        query = self.reference.normalizer(text)
        if not query:
            return Match(query=text, key=None, canonical=None, score=0.0, method=UNKNOWN)

        rejected: list[Candidate] = []
        for layer in self.layers:
            candidate = layer.propose(query)
            if candidate is None:
                continue
            if candidate.accepted:
                return Match(
                    query=text,
                    key=candidate.key,
                    canonical=self.reference.get(candidate.key).canonical,
                    score=round(candidate.score, 3),
                    method=candidate.method,
                    rejected=tuple(rejected),
                )
            rejected.append(candidate)

        return Match(
            query=text, key=None, canonical=None, score=0.0, method=UNKNOWN,
            rejected=tuple(rejected),
        )

    def resolve_many(self, raws: Iterable[str | None]) -> list[Match]:
        return [self.resolve(r) for r in raws]
