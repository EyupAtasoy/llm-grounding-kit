"""Text normalisation shared by the reference index and incoming queries.

The same function is applied to both sides. That symmetry matters more than
the exact rules: if "Constanța" is folded to "constanta" in the index, the
query has to be folded the same way or the exact layer will never fire.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_NON_WORD = re.compile(r"[^a-z0-9]+")

# Latin letters that NFKD does not split into base letter + accent.
_NO_DECOMPOSITION = str.maketrans(
    {"ø": "o", "Ø": "o", "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
     "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ı": "i"}
)


def fold(text: str) -> str:
    """Lowercase, strip accents, turn punctuation/underscores into spaces.

    Output is ASCII only: non-Latin scripts fold to "" and resolve to Unknown.
    """
    decomposed = unicodedata.normalize("NFKD", text.translate(_NO_DECOMPOSITION))
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_WORD.sub(" ", no_marks.lower()).strip()


@dataclass(frozen=True)
class Normalizer:
    """Folds text and removes filler phrases at either end.

    Fillers are whole phrases ("port of", "in bulk"), not single words, so
    "port klang" survives while "port of hamburg" becomes "hamburg".
    """

    prefixes: tuple[str, ...] = ()
    suffixes: tuple[str, ...] = ()

    def __call__(self, text: str) -> str:
        out = fold(text)
        out = _strip_once(out, tuple(fold(p) for p in self.prefixes), at_start=True)
        out = _strip_once(out, tuple(fold(s) for s in self.suffixes), at_start=False)
        return out


def _strip_once(text: str, phrases: tuple[str, ...], *, at_start: bool) -> str:
    # Longest phrase first so "in bulk" wins over "bulk". Requiring the
    # trailing/leading space means a filler is never stripped down to "".
    for phrase in sorted(phrases, key=len, reverse=True):
        if at_start and text.startswith(phrase + " "):
            return text[len(phrase) + 1 :]
        if not at_start and text.endswith(" " + phrase):
            return text[: -(len(phrase) + 1)]
    return text
