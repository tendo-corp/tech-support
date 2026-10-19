"""Picks labelled examples for the prompt: the most similar cases plus one of each type."""

from __future__ import annotations

from collections.abc import Sequence

from .dataset import Case
from .domain import Message
from .text import stems as _stems


def _similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class FewShotSelector:
    def __init__(self, cases: Sequence[Case], k: int = 8):
        self.cases = list(cases)
        self.k = k
        self._stems = {c.id: _stems(c.message.full_text) for c in self.cases}

    def select(self, message: Message) -> list[Case]:
        query = _stems(message.full_text)
        # Team messages teach the model to recognise our own chatter; keep one similar example of it.
        if message.author_side == "team":
            pool = [c for c in self.cases if c.message.author_side == "team" and c.message.text != message.text]
            pool.sort(key=lambda c: _similarity(query, self._stems[c.id]), reverse=True)
            return pool[: self.k]
        # Never show the model the message it is classifying (keeps eval honest).
        pool = [c for c in self.cases if c.message.author_side == "client" and c.message.text != message.text]
        ranked = sorted(pool, key=lambda c: _similarity(query, self._stems[c.id]), reverse=True)

        chosen: list[Case] = []
        for kind in ("A", "B", "C", "D", "E"):
            best = next((c for c in ranked if c.expected["type"] == kind), None)
            if best:
                chosen.append(best)
        for case in ranked:
            if len(chosen) >= self.k:
                break
            if case not in chosen:
                chosen.append(case)
        return chosen[: self.k]
