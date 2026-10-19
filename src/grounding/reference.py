"""Canonical reference list: the closed set of values a field is allowed to take."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import Normalizer


@dataclass(frozen=True)
class Entry:
    key: str
    canonical: str
    aliases: tuple[str, ...] = ()

    def surface_forms(self) -> tuple[str, ...]:
        """Every string that should resolve to this entry exactly."""
        return (self.key, self.canonical, *self.aliases)


@dataclass(frozen=True)
class Reference:
    """Immutable reference with a prebuilt alias index.

    Building the index fails loudly on duplicate keys or on one alias pointing
    at two different entries. Those are data bugs, and it is much cheaper to
    catch them at load time than to debug a wrong match later.
    """

    entries: tuple[Entry, ...]
    normalizer: Normalizer = field(default_factory=Normalizer)
    _by_key: dict[str, Entry] = field(init=False, repr=False, compare=False)
    _alias_index: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_key: dict[str, Entry] = {}
        index: dict[str, str] = {}
        for entry in self.entries:
            if entry.key in by_key:
                raise ValueError(f"duplicate key: {entry.key!r}")
            by_key[entry.key] = entry
            for form in entry.surface_forms():
                norm = self.normalizer(form)
                if not norm:
                    continue
                owner = index.setdefault(norm, entry.key)
                if owner != entry.key:
                    raise ValueError(
                        f"alias {form!r} maps to both {owner!r} and {entry.key!r}"
                    )
        # Frozen dataclass: derived fields are set once here and never again.
        object.__setattr__(self, "_by_key", by_key)
        object.__setattr__(self, "_alias_index", index)

    @classmethod
    def from_json(cls, path: str | Path) -> Reference:
        """Load {"prefixes": [...], "suffixes": [...], "entries": [...]}."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = tuple(
            Entry(
                key=item["key"],
                canonical=item["canonical"],
                aliases=tuple(item.get("aliases", ())),
            )
            for item in raw["entries"]
        )
        normalizer = Normalizer(
            prefixes=tuple(raw.get("prefixes", ())),
            suffixes=tuple(raw.get("suffixes", ())),
        )
        return cls(entries=entries, normalizer=normalizer)

    @classmethod
    def from_entries(
        cls, entries: Iterable[Entry], normalizer: Normalizer | None = None
    ) -> Reference:
        return cls(entries=tuple(entries), normalizer=normalizer or Normalizer())

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, key: str) -> Entry:
        return self._by_key[key]

    def lookup(self, normalized: str) -> str | None:
        """Exact lookup of an already-normalised string."""
        return self._alias_index.get(normalized)

    def surface_index(self) -> tuple[tuple[str, str], ...]:
        """(normalised form, key) pairs, used by the fuzzy and embedding layers."""
        return tuple(self._alias_index.items())
