"""Ground free-text entity strings to a canonical reference, or return Unknown."""

from .layers import (
    AliasLayer,
    Candidate,
    EmbeddingLayer,
    FuzzyLayer,
    Layer,
    sentence_transformer_encoder,
)
from .normalize import Normalizer, fold
from .reference import Entry, Reference
from .resolver import UNKNOWN, Match, Resolver

__all__ = [
    "AliasLayer",
    "Candidate",
    "EmbeddingLayer",
    "Entry",
    "FuzzyLayer",
    "Layer",
    "Match",
    "Normalizer",
    "Reference",
    "Resolver",
    "UNKNOWN",
    "fold",
    "sentence_transformer_encoder",
]
