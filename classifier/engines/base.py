from __future__ import annotations

from typing import Protocol

from ..domain import Context, EngineResult, Message


class EngineError(RuntimeError):
    """Engine could not produce answers (network, quota, unparsable output)."""


class Engine(Protocol):
    name: str

    def answer(self, message: Message, context: Context | None = None) -> EngineResult: ...
