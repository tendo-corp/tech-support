"""Tiny text helpers shared by the offline parts (few-shot ranking, contract matching)."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
# Words that say nothing about the subject of a message.
_STOP = {"это", "как", "что", "для", "при", "уже", "или", "так", "все", "всё", "нет", "еще", "ещё", "они", "его", "она",
         "наш", "наши", "мне", "нам", "вас", "вам", "можно", "нужно", "где", "почему", "когда", "тоже", "только"}


def stems(text: str) -> set[str]:
    # Crude stemming: Russian inflection mostly lives in the last letters.
    return {w[:5] for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def similarity(a: set[str], b: set[str]) -> float:
    """Overlap normalised by the smaller set, so a short phrase can match a long message."""
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0
