"""Matching layers, cheapest first.

Each layer looks at an already-normalised query and either proposes nothing
or proposes a Candidate. A candidate carries its own accept/reject verdict, so
the resolver does not need to know how each layer scores things.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rapidfuzz import fuzz, process

from .reference import Reference


@dataclass(frozen=True)
class Candidate:
    key: str
    score: float  # 0..1, but only comparable within the same layer
    accepted: bool
    method: str


class Layer(Protocol):
    name: str

    def propose(self, query: str) -> Candidate | None: ...


@dataclass(frozen=True)
class AliasLayer:
    """Exact hit in the alias index. Free, and never wrong if the data is right."""

    reference: Reference
    name: str = "alias"

    def propose(self, query: str) -> Candidate | None:
        key = self.reference.lookup(query)
        if key is None:
            return None
        return Candidate(key=key, score=1.0, accepted=True, method=self.name)


@dataclass(frozen=True)
class FuzzyLayer:
    """Edit-distance match against every surface form.

    Accepts only when the best entry clears `threshold` AND beats the best
    *different* entry by `min_margin` points, so a query that sits halfway
    between two entries is left for a human instead of being coin-flipped.
    Very short queries are skipped: at 3 characters one typo is a 33% change
    and the scores stop meaning anything.
    """

    reference: Reference
    threshold: float = 85.0
    min_margin: float = 3.0
    min_length: int = 4
    scorer: Callable[..., float] = fuzz.token_sort_ratio
    name: str = "fuzzy"
    _forms: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _keys: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = self.reference.surface_index()
        object.__setattr__(self, "_forms", tuple(form for form, _ in index))
        object.__setattr__(self, "_keys", tuple(key for _, key in index))

    def propose(self, query: str) -> Candidate | None:
        if len(query.replace(" ", "")) < self.min_length or not self._forms:
            return None
        hits = process.extract(
            query, self._forms, scorer=self.scorer, processor=None, limit=len(self._forms)
        )
        # hits are sorted best-first, so the first time we see a key is its best score.
        best: dict[str, float] = {}
        for _form, score, idx in hits:
            best.setdefault(self._keys[idx], float(score))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        top_key, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        accepted = top >= self.threshold and (top - runner_up) >= self.min_margin
        return Candidate(key=top_key, score=top / 100.0, accepted=accepted, method=self.name)


Encoder = Callable[[Sequence[str]], Any]  # returns an (n, d) array-like


@dataclass(frozen=True)
class EmbeddingLayer:
    """Cosine similarity between the query and every surface form.

    This is the only layer that can know "maize" means corn. It is also the
    layer most likely to be confidently wrong ("wood pellets" sits close to
    "timber"), so it runs last and with the same threshold + margin rule.
    The encoder is injected, which keeps the layer testable without a model.
    """

    reference: Reference
    encode: Encoder
    threshold: float = 0.60
    min_margin: float = 0.05
    name: str = "embedding"
    _keys: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _matrix: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        import numpy as np

        index = self.reference.surface_index()
        vectors = np.asarray(self.encode([form for form, _ in index]), dtype=float)
        object.__setattr__(self, "_keys", tuple(key for _, key in index))
        object.__setattr__(self, "_matrix", _unit_rows(vectors))

    def propose(self, query: str) -> Candidate | None:
        import numpy as np

        if not self._keys:
            return None
        q = _unit_rows(np.asarray(self.encode([query]), dtype=float))[0]
        sims = self._matrix @ q
        best: dict[str, float] = {}
        for i in np.argsort(-sims):
            best.setdefault(self._keys[int(i)], float(sims[int(i)]))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        top_key, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        accepted = top >= self.threshold and (top - runner_up) >= self.min_margin
        return Candidate(key=top_key, score=top, accepted=accepted, method=self.name)


def _unit_rows(m: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


def sentence_transformer_encoder(model_name: str = "all-MiniLM-L6-v2") -> Encoder:
    """Build an encoder from sentence-transformers (pip install .[embeddings])."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "EmbeddingLayer needs sentence-transformers: pip install '.[embeddings]'"
        ) from exc
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), convert_to_numpy=True)
