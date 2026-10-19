"""Entry point used by the bot: cheap rules first, LLM for everything that matters, rules as fallback."""

from __future__ import annotations

import logging

from .aggregate import to_verdict
from .domain import Context, Message, Verdict
from .engines.base import Engine, EngineError
from .engines.rules import RulesEngine

log = logging.getLogger(__name__)


class Classifier:
    def __init__(self, smart: Engine | None = None, rules: RulesEngine | None = None,
                 skip_threshold: float = 0.9):
        self.smart = smart
        self.rules = rules or RulesEngine()
        # Rules answers at or above this confidence (team chatter, "спасибо") skip the LLM.
        self.skip_threshold = skip_threshold

    def classify(self, message: Message, context: Context | None = None) -> Verdict:
        quick = self.rules.answer(message, context)
        if self.smart is None or quick.confidence >= self.skip_threshold:
            return to_verdict(quick)
        try:
            return to_verdict(self.smart.answer(message, context))
        except EngineError as exc:
            log.warning("smart engine failed, using rules: %s", exc)
            return to_verdict(quick)
