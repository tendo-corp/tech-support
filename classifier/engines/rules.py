"""Dumb keyword engine: free, instant, offline.

Jobs: drop noise and team chatter before paying for the LLM, and stand in when the LLM is down.
It knows nothing about the contract, so its coverage/complexity answers are rough defaults.
"""

from __future__ import annotations

import re

from ..domain import Answers, Context, EngineResult, Message
from ..questionnaire import normalize_answers

_F = re.IGNORECASE | re.UNICODE


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, _F) for p in patterns]


BUG = _compile([
    r"\bбаг\w*", r"\bbug\w*", r"\bошибк\w*", r"\berror\w*", r"\bexception\b",
    r"не\s+работа\w*", r"перестал\w*", r"не\s*корректн\w*", r"некорректн\w*",
    r"не\s+(?:грузит\w*|загружа\w*|открыва\w*|отправля\w*|сохраня\w*|приход\w*|отобража\w*|показыва\w*|считает)",
    r"не\s+\w+\s+(?:распознан\w*|корректн\w*)", r"распознан\w*\s+как",
    r"некликабельн\w*", r"не\s*нажима\w*", r"заблокирован\w*", r"\bзавис\w*", r"\bпада\w*", r"\bупал\w*",
    r"\bвылета\w*", r"\bсломал\w*", r"\bглюч\w*", r"свалил\w*", r"пропал\w*", r"\bотличаются\b",
    r"\b(?:500|502|503|504)\b", r"\bcrash\w*", r"\bbroken\b",
])

FEATURE = _compile([
    r"нужно\s+(?:доработать|добавить|сделать|изменить|убрать)", r"\bдобав(?:ить|ьте)\b", r"\bдоработ\w*",
    r"\bпереименова\w*", r"\bубрать\b", r"\bуберите\b", r"\bскры(?:ть|йте)\b", r"\bпредлагаю\b",
    r"очень\s+прошу", r"\bхотим\b", r"хотелось\s+бы", r"\bсделайте\b", r"можно\s+ли\s+(?:сделать|добавить)",
    r"должн\w*\s+(?:открываться|быть|сопровождаться)", r"\bизмените\b", r"\bпоменяйте\b",
    r"нужн\w*\s+(?:возможност\w*|функци\w*|кнопк\w*|редактируем\w*|форм\w*|шаблон\w*)",
])

QUESTION = _compile([
    r"^\s*(?:как|где|когда|каким\s+образом|подскажите)\b", r"\bнапомнит\w*", r"\bкак\s+(?:их|его|её|это)?\s*\w+(?:ть|ить)\b",
    r"не\s*понятно\s+как", r"\bподскажите\b",
])

DISCUSSION = _compile([
    r"\bновост\w*", r"\bждем\b|\bждём\b", r"\bутверд\w*", r"\bотложим\b", r"дайте\s+доступ", r"\bв\s+работе\b",
    r"\bвзяли\b", r"\bпочинен\w*", r"какой\s+ответ", r"другой\s+чат", r"следующие\s+шаги",
])

NOISE = _compile([
    r"^\s*(?:ок|окей|ok|спасибо|спс|благодарю|понял\w*|принято|ясно|супер|отлично|да|нет|\+1?|"
    r"привет|всем\s+привет|здравствуйте|добрый\s+(?:день|вечер)|доброе\s+утро)"
    r"(?:[\s,!.)]+(?:ок|спасибо|большое))*[\s!.,)]*$",
])

URGENT = _compile([
    r"\bсрочно\b", r"\basap\b", r"не\s+мо(?:жет|гут|гу)\s+(?:подать|отправить|войти|зайти)",
    r"пытаюсь\s+отправить", r"некликабельн\w*", r"ничего\s+не\s+работает",
])

_NO_WORDS = re.compile(r"^[\W_\d]*$", _F)
_MENTIONS = re.compile(r"@\w+")


def _count(patterns: list[re.Pattern[str]], text: str) -> int:
    return sum(1 for p in patterns if p.search(text))


class RulesEngine:
    name = "rules"

    def answer(self, message: Message, context: Context | None = None) -> EngineResult:
        answers, confidence, reason = self._classify(message)
        return EngineResult(answers=normalize_answers(answers), engine=self.name, reasoning=reason, confidence=confidence)

    def _classify(self, message: Message) -> tuple[Answers, float, str]:
        if message.author_side == "team":
            return _non_task("D"), 0.95, "сообщение команды"

        own = _MENTIONS.sub("", message.text).strip()
        text = message.full_text
        if not message.attachments and (_NO_WORDS.match(own) or any(p.match(own) for p in NOISE)):
            return _non_task("E"), 0.9, "шум"

        scores = {
            "A": _count(BUG, text) * 1.3,
            "B": _count(FEATURE, text),
            "C": _count(QUESTION, text) * 0.8,
            "D": _count(DISCUSSION, text) * 0.9,
        }
        kind, score = max(scores.items(), key=lambda kv: kv[1])
        if score == 0:
            return _non_task("D"), 0.3, "нет совпадений"

        confidence = round(min(0.8, 0.35 + 0.15 * score + 0.2 * (score / sum(scores.values()))), 2)
        reason = f"ключевые слова: {kind} ({', '.join(f'{k}={v:g}' for k, v in scores.items() if v)})"
        if kind in ("C", "D"):
            return _non_task(kind, bug_probability=40 if scores["A"] else 0), confidence, reason

        urgent = _count(URGENT, text) > 0
        short = len(own) < 60
        if kind == "A":
            answers = {
                "is_task": True, "type": "A", "bug_probability": 80, "coverage": "A",
                "complexity": "B", "urgency": "A" if urgent else "B", "needs_clarification": short,
            }
        else:
            small = len(text) < 220 and not re.search(r"редактируем\w*|должн\w*|все\s+сообщения|интеграц\w*", text, _F)
            answers = {
                "is_task": True, "type": "B", "bug_probability": 10, "coverage": "B" if small else "C",
                "complexity": "A" if small else "C", "urgency": "C", "needs_clarification": False,
            }
        return answers, confidence, reason


def _non_task(kind: str, bug_probability: int = 0) -> Answers:
    return {"is_task": False, "type": kind, "bug_probability": bug_probability, "coverage": "D"}
